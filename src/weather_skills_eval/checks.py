"""The result a task expects: the files the assistant presents, with their provenance.

A reply presents a file by a successful ``display_image`` call of an image
or an .html page (its ``path``), a successful ``send_email`` call (its
``attachments``), a relative markdown image or link target in its text, or
text in single backticks in its text, outside fenced code blocks, that is
exactly the path of a listed entry other than a folder; backtick text that
is not such a path presents nothing. Text inside tool-call and reasoning
blocks presents nothing. A path named only in backticks never blocks the
judge: its problems are notes, and the judge is told it was named in passing
and not presented, unless it is of a kind the task needs and no file of that
kind was presented outright. The simulated user is shown the files of the
latest reply that presents any; the judge rules on every file presented
during the play, one per path. Each one's kind and provenance come from its
entry in the chat's file listing: the app reads the
``weather_skills_history`` chain of PNG images, .html pages and zarr stores
into ``provenance_detail``, so only those need provenance. Only presented
images are downloaded; other files of the chat are never fetched.
"""

import html
import json
import mimetypes
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from pathlib import PurePosixPath
from urllib.parse import unquote

from .app_client import AppClient, ChatError

_DETAILS = re.compile(r"<details\b([^>]*)>.*?</details>", re.IGNORECASE | re.DOTALL)
_ATTR = re.compile(r'(\w+)="([^"]*)"')
# A markdown image or link target, without its optional title.
_LINK = re.compile(r"""!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+(?:"[^"]*"|'[^']*'))?\s*\)""")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
# Text in single backticks, not part of a longer run of backticks.
_CODE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
# A fenced code block, to its closing fence or the end of the text. A
# backtick fence's info string has no backtick, so a line starting with an
# inline span such as ```x``` opens no block.
_FENCE = re.compile(
    r"^[ \t]*(?:(`{3,})[^`\n]*$.*?(?:^[ \t]*\1|\Z)|(~{3,}).*?(?:^[ \t]*\2|\Z))",
    re.MULTILINE | re.DOTALL,
)
# The app's display_image result for an .html page it shows the user.
_PAGE_SHOWN = re.compile(r"Showing `[^`]*` to the user as an interactive page\.")


def _decode(raw: str | None):
    """A tool-call attribute: HTML-escaped JSON, holding JSON text for arguments
    and for list or object results."""
    value = html.unescape(raw or "")
    for _ in range(2):
        if not isinstance(value, str):
            break
        try:
            value = json.loads(value)
        except ValueError:
            break
    return value


def _path(raw) -> str | None:
    if not isinstance(raw, str):
        return None
    text = raw.strip().replace("\\", "/").lstrip("/")
    if not text:
        return None
    path = PurePosixPath(text).as_posix()
    return None if path == "." else path


def _tool_paths(attrs: dict[str, str]) -> list[str]:
    name = attrs.get("name")
    arguments = _decode(attrs.get("arguments"))
    result = _decode(attrs.get("result"))
    if not isinstance(arguments, dict):
        return []
    # display_image returns [note, image data URL] for an image, and the app
    # stores the note list; for an .html page it returns one text.
    page_shown = isinstance(result, str) and _PAGE_SHOWN.match(result)
    if name == "display_image" and (isinstance(result, list) or page_shown):
        return [p for p in [_path(arguments.get("path"))] if p]
    if name == "send_email" and isinstance(result, str) and result.startswith("Email sent to"):
        attachments = arguments.get("attachments")
        if isinstance(attachments, str):
            attachments = [attachments]
        if not isinstance(attachments, list):
            return []
        return [p for p in map(_path, attachments) if p]
    return []


def _link_path(target: str) -> str | None:
    if _SCHEME.match(target) or target.startswith(("/", "#")):
        return None
    return _path(unquote(re.split(r"[?#]", target, maxsplit=1)[0]))


def _text_paths(text: str) -> list[tuple[str, bool]]:
    """Link targets, and backtick candidates (True), in the order they appear,
    outside fenced code blocks."""
    fences = [m.span() for m in _FENCE.finditer(text)]

    def fenced(match) -> bool:
        return any(start <= match.start() < end for start, end in fences)

    found = [
        (m.start(), _link_path(m.group(1)), False) for m in _LINK.finditer(text) if not fenced(m)
    ]
    found += [(m.start(), _path(m.group(1)), True) for m in _CODE.finditer(text) if not fenced(m)]
    return [(path, candidate) for _, path, candidate in sorted(found) if path]


@dataclass(frozen=True)
class ReplyFiles:
    """The chat paths one stored assistant reply names, in order, each once.
    A path in ``candidates`` was named only in backticks: it is presented
    only when the chat's file listing has an entry with exactly that path."""

    turn: int
    paths: list[str]
    candidates: frozenset[str] = frozenset()

    def presented(self, listed) -> list[str]:
        return [p for p in self.paths if p not in self.candidates or p in listed]


def reply_files(content: str, turn: int) -> ReplyFiles:
    """What one stored assistant reply, the reply to ``turn``, names as files."""
    text = content or ""
    named: list[tuple[str, bool]] = []
    position = 0
    for match in _DETAILS.finditer(text):
        named += _text_paths(text[position : match.start()])
        named += [(p, False) for p in _tool_paths(dict(_ATTR.findall(match.group(1))))]
        position = match.end()
    named += _text_paths(text[position:])
    definite = {path for path, candidate in named if not candidate}
    paths = list(dict.fromkeys(path for path, _ in named))
    return ReplyFiles(turn, paths, frozenset(p for p in paths if p not in definite))


def _is_folder(entry: dict) -> bool:
    # The app lists a zarr store, a folder too, with the kind "zarr".
    return entry.get("kind") == "directory" or (
        entry.get("is_dir") is True and entry.get("kind") != "zarr"
    )


def _definite(replies: list[ReplyFiles]) -> set[str]:
    """The paths some reply names other than in backticks."""
    return {p for reply in replies for p in reply.paths if p not in reply.candidates}


def _resolve(
    replies: list[ReplyFiles], entries: dict[str, dict], every: bool
) -> tuple[list[str], dict[str, int], set[str]]:
    """The presented paths in order, the turn that last presented each, and
    the blocking ones: the files of the latest reply that presents any,
    except paths no reply names other than in backticks. With ``every``,
    the paths are every presented path, each at its latest presentation;
    without, only that reply's. A path some reply names other than in
    backticks keeps the turn of its latest such reply: a later backtick
    mention does not move it."""
    listed = {path for path, entry in entries.items() if not _is_folder(entry)}
    definite = _definite(replies)
    latest: list[str] = []
    turns: dict[str, int] = {}
    for reply in replies:
        found = reply.presented(listed)
        if not found:
            continue
        latest = found
        for path in found:
            if path in definite and path in reply.candidates and path in turns:
                continue
            # Moved to the end: the latest presentation of a path wins.
            turns.pop(path, None)
            turns[path] = reply.turn
    blocks = set(latest) & definite
    if every:
        return list(turns), turns, blocks
    return latest, {p: turns[p] for p in latest}, blocks


# The image types the harness models are shown as images, by file suffix;
# strict mimetypes on Python 3.12 does not know .webp.
IMAGE_MEDIA_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class PresentedFile:
    path: str
    # "unknown" for a path that is not in the chat's file listing.
    kind: str
    # None when the listing entry has none.
    provenance: dict | None
    image: bytes | None = None
    # The turn whose reply last presented the file.
    turn: int | None = None
    # A problem with a file that does not block the judge, given to it instead.
    note: str | None = None
    # A listed image whose download failed.
    download_failed: bool = False
    # No reply names the path other than in backticks, and a file of its kind
    # was presented outright or the task does not need its kind: the judge is
    # told it was named in passing and not presented.
    named_in_passing: bool = False

    @property
    def media_type(self) -> str:
        suffix = PurePosixPath(self.path).suffix.lower()
        return (
            IMAGE_MEDIA_TYPES.get(suffix)
            or mimetypes.guess_type(self.path)[0]
            or "application/octet-stream"
        )

    @property
    def shown_as_image(self) -> bool:
        """An image file whose media type is in ``IMAGE_MEDIA_TYPES``; any
        other is neither downloaded nor shown as an image."""
        return self.kind == "image" and self.media_type in IMAGE_MEDIA_TYPES.values()


@dataclass(frozen=True)
class Evidence:
    """The presented files that could be read, and why the result cannot be
    judged, when it cannot. The judge rules only when ``problem`` is None."""

    files: list[PresentedFile] = field(default_factory=list)
    problem: str | None = None
    # Blocking presented paths not found in the chat's file listing, or every
    # presented path not named only in backticks when the listing failed.
    unlisted: list[str] = field(default_factory=list)
    # The listing or a download failed: a later read may succeed.
    read_failed: bool = False
    # Backtick paths in the latest reply naming files that the listing lacks;
    # a later listing may have them.
    unlisted_candidates: list[str] = field(default_factory=list)


def _has_provenance(entry: dict) -> bool:
    """A ``provenance_detail`` with at least one branch of steps."""
    detail = entry.get("provenance_detail")
    branches = detail.get("branches") if isinstance(detail, dict) else None
    return isinstance(branches, list) and len(branches) > 0


def _needs_provenance(entry: dict, path: str) -> bool:
    """The app reads provenance only from PNG images, .html pages and zarr stores."""
    kind = entry.get("kind")
    png = kind == "image" and PurePosixPath(path).suffix.lower() == ".png"
    return png or kind in ("html", "zarr")


async def _no_wait() -> None:
    return None


def _article(kind: str) -> str:
    return "an" if kind[:1] in "aeiou" else "a"


async def _with_retries(read, retries: int, wait: Callable[[], Awaitable[None]]):
    for attempt in range(retries + 1):
        try:
            return await read()
        except ChatError:
            if attempt == retries:
                raise
            await wait()


NOTHING_PRESENTED = "No result presented: no reply presented a file"


async def collect_evidence(
    app: AppClient,
    chat_id: str | None,
    replies: list[ReplyFiles],
    deliverables: list[str],
    *,
    every: bool = False,
    retries: int = 0,
    known_images: dict[tuple[str, int], bytes] | None = None,
    wait: Callable[[], Awaitable[None]] | None = None,
) -> Evidence:
    """The files of the latest reply that presents any or, with ``every``,
    every presented path at the turn that last presented it. Backtick
    candidates are resolved against the listing read here. A file of the
    latest presenting reply that some reply names other than only in
    backticks, and that is not listed, has no provenance although its kind
    carries it, or cannot be downloaded, makes the result one that cannot be
    judged; any other file keeps the problem as its ``note``. The deliverable kinds are checked
    against every presented file. The listing is read up to ``retries`` more
    times when it fails, lacks a path of the latest presenting reply, or
    lacks a backtick path of the latest reply that names any file (the
    listing may lag), and each download is tried ``retries`` more times when
    it fails, after ``wait``. When an extra listing fails, the last listing
    that succeeded is used. An image in ``known_images``, by path and the
    turn that presented it, is not downloaded."""
    if chat_id is None or not any(reply.paths for reply in replies):
        return Evidence(problem=NOTHING_PRESENTED)
    wait = wait or _no_wait
    naming = next(reply for reply in reversed(replies) if reply.paths)
    attempts = 0
    entries: dict[str, dict] | None = None
    # The error of an extra listing that failed after one succeeded.
    relisting_error: str | None = None
    while True:
        try:
            listed = await app.artifacts(chat_id)
        except ChatError as exc:
            if attempts < retries:
                attempts += 1
                await wait()
                continue
            if entries is not None:
                relisting_error = str(exc)
                break
            # Without a listing no candidate counts as presented.
            return Evidence(
                problem=f"The chat's files could not be listed: {exc}",
                unlisted=_resolve(replies, {}, every)[0],
                read_failed=True,
            )
        entries = {
            e["path"]: e for e in listed if isinstance(e, dict) and isinstance(e.get("path"), str)
        }
        _, _, blocks = _resolve(replies, entries, every)
        # A backtick path of the latest naming reply may not be listed yet;
        # without it, the result would fall back to an earlier reply's files.
        lagging = [p for p in blocks if p not in entries] + [
            p for p in naming.candidates if p not in entries
        ]
        if lagging and attempts < retries:
            attempts += 1
            await wait()
            continue
        break
    unlisted_candidates = [p for p in naming.paths if p in naming.candidates and p not in entries]
    presented, turns, blocks = _resolve(replies, entries, every)
    if not presented:
        return Evidence(problem=NOTHING_PRESENTED, unlisted_candidates=unlisted_candidates)
    kinds = {p: entries[p].get("kind") for p in presented if p in entries}
    definite = _definite(replies)
    files, missing, unproven = [], [], []
    unreadable, read_failed = None, False
    for path in presented:
        turn = turns[path]
        entry = entries.get(path)
        if entry is None:
            # Unlisted, so some reply names it other than in backticks.
            if path in blocks:
                missing.append(path)
            else:
                files.append(
                    PresentedFile(
                        path=path,
                        kind="unknown",
                        provenance=None,
                        turn=turn,
                        note="it is not in the chat's files",
                    )
                )
            continue
        notes = []
        image, download_failed = None, False
        listed_file = PresentedFile(path=path, kind=kinds[path], provenance=None)
        if listed_file.shown_as_image and (path, turn) in (known_images or {}):
            image = known_images[(path, turn)]
        elif listed_file.shown_as_image:
            try:
                image = await _with_retries(
                    lambda path=path: app.artifact_content(chat_id, path), retries, wait
                )
            except ChatError as exc:
                read_failed = download_failed = True
                if path in blocks:
                    unreadable = unreadable or f"{path} could not be read: {exc}"
                else:
                    notes.append(f"it could not be read: {exc}")
        if "provenance_detail" not in entry and _needs_provenance(entry, path):
            # The listing carries only crumbs; the app serves each file's branches apart.
            try:
                entry["provenance_detail"] = await app.artifact_provenance(chat_id, path)
            except ChatError:
                pass
        provenance = entry["provenance_detail"] if _has_provenance(entry) else None
        if provenance is None:
            if path in blocks and _needs_provenance(entry, path):
                unproven.append(path)
            else:
                notes.append("it has no provenance")
        files.append(
            PresentedFile(
                path=path,
                kind=kinds[path],
                provenance=provenance,
                image=image,
                turn=turn,
                note="; ".join(notes) or None,
                download_failed=download_failed,
                named_in_passing=path not in definite,
            )
        )
    # Files named in passing count as presented when they are the only files
    # of a kind the task needs: no file of that kind was presented outright.
    outright = {file.kind for file in files if not file.named_in_passing}
    files = [
        replace(file, named_in_passing=False)
        if file.named_in_passing and file.kind in deliverables and file.kind not in outright
        else file
        for file in files
    ]

    if missing:
        problem = f"Presented files are not in the chat's files: {missing}"
        if relisting_error:
            problem += f"; the listing read again failed: {relisting_error}"
        return Evidence(
            files=files,
            problem=problem,
            unlisted=missing,
            read_failed=read_failed,
            unlisted_candidates=unlisted_candidates,
        )
    shown = ", ".join(f"{p} ({kinds.get(p, 'unknown')})" for p in presented)
    for kind in dict.fromkeys(deliverables):
        if kind not in kinds.values():
            return Evidence(
                files=files,
                problem=f"No presented file is {_article(kind)} {kind}; presented: {shown}",
                read_failed=read_failed,
                unlisted_candidates=unlisted_candidates,
            )
    if unproven:
        return Evidence(
            files=files,
            problem=f"Presented files have no provenance: {unproven}",
            read_failed=read_failed,
            unlisted_candidates=unlisted_candidates,
        )
    return Evidence(
        files=files,
        problem=unreadable,
        read_failed=read_failed,
        unlisted_candidates=unlisted_candidates,
    )
