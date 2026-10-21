"""The result a task expects: the files the assistant presents, with their provenance.

A reply presents a file by a successful ``display_image`` call (its
``path``), a successful ``send_email`` call (its ``attachments``), or a
relative markdown image or link target in its text. The simulated user is
shown the files of the latest reply that presents any; the judge rules on
every file presented during the play, one per path. Each one's kind and
provenance come from its entry in the chat's file listing: the app reads the
``weather_skills_history`` chain of PNG images and zarr stores into
``provenance_detail``. Only presented images are downloaded; other files of
the chat are never fetched.
"""

import html
import json
import mimetypes
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from urllib.parse import unquote

from .app_client import AppClient, ChatError

_DETAILS = re.compile(r"<details\b([^>]*)>.*?</details>", re.IGNORECASE | re.DOTALL)
_ATTR = re.compile(r'(\w+)="([^"]*)"')
# A markdown image or link target, without its optional title.
_LINK = re.compile(r"""!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+(?:"[^"]*"|'[^']*'))?\s*\)""")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")


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
    # display_image returns [note, image data URL]; the app stores the note list.
    if name == "display_image" and isinstance(result, list):
        return [p for p in [_path(arguments.get("path"))] if p]
    if name == "send_email" and isinstance(result, str) and result.startswith("Email sent to"):
        attachments = arguments.get("attachments")
        if isinstance(attachments, str):
            attachments = [attachments]
        if not isinstance(attachments, list):
            return []
        return [p for p in map(_path, attachments) if p]
    return []


def _link_paths(text: str) -> list[str]:
    paths = []
    for target in _LINK.findall(text):
        if _SCHEME.match(target) or target.startswith(("/", "#")):
            continue
        path = _path(unquote(re.split(r"[?#]", target, maxsplit=1)[0]))
        if path:
            paths.append(path)
    return paths


def presented_paths(content: str) -> list[str]:
    """The chat paths one stored assistant reply presents, in order, each once."""
    found: list[str] = []
    position = 0
    for match in _DETAILS.finditer(content or ""):
        found += _link_paths(content[position : match.start()])
        found += _tool_paths(dict(_ATTR.findall(match.group(1))))
        position = match.end()
    found += _link_paths((content or "")[position:])
    return list(dict.fromkeys(found))


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
    # The turn whose reply presented the file, when the caller gives it.
    turn: int | None = None
    # A problem with a file that does not block the judge, given to it instead.
    note: str | None = None
    # A listed image whose download failed.
    download_failed: bool = False

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
    # presented path when the listing failed.
    unlisted: list[str] = field(default_factory=list)
    # The listing or a download failed: a later read may succeed.
    read_failed: bool = False


def _has_provenance(entry: dict) -> bool:
    """A ``provenance_detail`` with at least one branch of steps."""
    detail = entry.get("provenance_detail")
    branches = detail.get("branches") if isinstance(detail, dict) else None
    return isinstance(branches, list) and len(branches) > 0


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


async def collect_evidence(
    app: AppClient,
    chat_id: str | None,
    presented: list[str],
    deliverables: list[str],
    turns: dict[str, int] | None = None,
    blocking: list[str] | None = None,
    retries: int = 0,
    known_images: dict[str, bytes] | None = None,
    wait: Callable[[], Awaitable[None]] | None = None,
) -> Evidence:
    """``turns`` gives the turn that presented each path. A file that is not
    listed, has no provenance or cannot be downloaded makes the result one
    that cannot be judged when its path is in ``blocking`` (every presented
    path when None); any other file keeps the problem as its ``note``. The
    deliverable kinds are checked against every presented file. The listing
    is read up to ``retries`` more times when it fails or lacks a blocking
    path (the listing may lag), and each download is tried ``retries`` more
    times when it fails, after ``wait``. When an extra listing fails, the
    last listing that succeeded is used. An image in ``known_images`` is not
    downloaded."""
    if chat_id is None or not presented:
        return Evidence(problem="No result presented: no reply presented a file")
    wait = wait or _no_wait
    blocks = set(presented if blocking is None else blocking)
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
            return Evidence(
                problem=f"The chat's files could not be listed: {exc}",
                unlisted=list(presented),
                read_failed=True,
            )
        entries = {
            e["path"]: e for e in listed if isinstance(e, dict) and isinstance(e.get("path"), str)
        }
        lagging = [p for p in presented if p in blocks and p not in entries]
        if lagging and attempts < retries:
            attempts += 1
            await wait()
            continue
        break
    kinds = {p: entries[p].get("kind") for p in presented if p in entries}
    files, missing, unproven = [], [], []
    unreadable, read_failed = None, False
    for path in presented:
        turn = (turns or {}).get(path)
        entry = entries.get(path)
        if entry is None:
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
        if listed_file.shown_as_image and path in (known_images or {}):
            image = known_images[path]
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
        provenance = entry["provenance_detail"] if _has_provenance(entry) else None
        if provenance is None:
            if path in blocks:
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
            )
        )

    if missing:
        problem = f"Presented files are not in the chat's files: {missing}"
        if relisting_error:
            problem += f"; the listing read again failed: {relisting_error}"
        return Evidence(
            files=files,
            problem=problem,
            unlisted=missing,
            read_failed=read_failed,
        )
    shown = ", ".join(f"{p} ({kinds.get(p, 'unknown')})" for p in presented)
    for kind in dict.fromkeys(deliverables):
        if kind not in kinds.values():
            return Evidence(
                files=files,
                problem=f"No presented file is {_article(kind)} {kind}; presented: {shown}",
                read_failed=read_failed,
            )
    if unproven:
        return Evidence(
            files=files,
            problem=f"Presented files have no provenance: {unproven}",
            read_failed=read_failed,
        )
    return Evidence(files=files, problem=unreadable, read_failed=read_failed)
