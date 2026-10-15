"""The result a task expects: the files the assistant presents, with their provenance.

A reply presents a file by a successful ``display_image`` call (its
``path``), a successful ``send_email`` call (its ``attachments``), or a
relative markdown image or link target in its text. The result is the files
of the latest reply that presents any. Each one's kind and provenance come
from its entry in the chat's file listing: the app reads the
``weather_skills_history`` chain of PNG images and zarr stores into
``provenance_detail``. Only presented images are downloaded; other files of
the chat are never fetched.
"""

import html
import json
import mimetypes
import re
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


@dataclass(frozen=True)
class PresentedFile:
    path: str
    kind: str
    provenance: dict
    image: bytes | None = None

    @property
    def media_type(self) -> str:
        return mimetypes.guess_type(self.path)[0] or "application/octet-stream"


@dataclass(frozen=True)
class Evidence:
    """The presented files with their provenance, or why the result cannot be judged."""

    files: list[PresentedFile] = field(default_factory=list)
    problem: str | None = None


def _has_provenance(entry: dict) -> bool:
    """A ``provenance_detail`` with at least one branch of steps."""
    detail = entry.get("provenance_detail")
    branches = detail.get("branches") if isinstance(detail, dict) else None
    return isinstance(branches, list) and len(branches) > 0


def _article(kind: str) -> str:
    return "an" if kind[:1] in "aeiou" else "a"


async def collect_evidence(
    app: AppClient, chat_id: str | None, presented: list[str], deliverables: list[str]
) -> Evidence:
    if chat_id is None or not presented:
        return Evidence(problem="No result presented: no reply presented a file")
    try:
        listed = await app.artifacts(chat_id)
    except ChatError as exc:
        return Evidence(problem=f"The chat's files could not be listed: {exc}")
    entries = {
        e["path"]: e for e in listed if isinstance(e, dict) and isinstance(e.get("path"), str)
    }
    missing = [p for p in presented if p not in entries]
    if missing:
        return Evidence(problem=f"Presented files are not in the chat's files: {missing}")
    kinds = {p: entries[p].get("kind") for p in presented}
    shown = ", ".join(f"{p} ({kinds[p]})" for p in presented)
    for kind in dict.fromkeys(deliverables):
        if kind not in kinds.values():
            return Evidence(
                problem=f"No presented file is {_article(kind)} {kind}; presented: {shown}"
            )
    unproven = [p for p in presented if not _has_provenance(entries[p])]
    if unproven:
        return Evidence(problem=f"Presented files have no provenance: {unproven}")

    files = []
    for path in presented:
        image = None
        if kinds[path] == "image":
            try:
                image = await app.artifact_content(chat_id, path)
            except ChatError as exc:
                return Evidence(problem=f"{path} could not be read: {exc}")
        files.append(
            PresentedFile(
                path=path,
                kind=kinds[path],
                provenance=entries[path]["provenance_detail"],
                image=image,
            )
        )
    return Evidence(files=files)
