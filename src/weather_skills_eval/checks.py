"""Automatic goal checks on the files a chat produced, read through the app's artifact API."""

from dataclasses import dataclass
from fnmatch import fnmatchcase

from .app_client import AppClient, ChatError
from .tasks import ArtifactCheck


@dataclass(frozen=True)
class CheckResult:
    check: ArtifactCheck
    passed: bool
    detail: str


async def _meets(app: AppClient, chat_id: str, check: ArtifactCheck, entry: dict) -> str | None:
    """None when the entry meets every condition, else why not."""
    path = entry.get("path", "")
    if check.kind and entry.get("kind") != check.kind:
        return f"{path} is {entry.get('kind')}, not {check.kind}"
    if check.variables or check.dims:
        try:
            meta = await app.zarr_meta(chat_id, path)
        except ChatError as exc:
            return f"{path} could not be read as zarr: {exc}"
        missing = [v for v in check.variables if v not in (meta.get("variables") or {})]
        if missing:
            return f"{path} lacks variables {missing}"
        missing = [d for d in check.dims if d not in (meta.get("dims") or {})]
        if missing:
            return f"{path} lacks dimensions {missing}"
    if check.contains is not None:
        try:
            content = await app.artifact_content(chat_id, path)
        except ChatError as exc:
            return f"{path} could not be read: {exc}"
        if check.contains not in content.decode("utf-8", errors="replace"):
            return f"{path} does not contain {check.contains!r}"
    return None


async def run_checks(
    app: AppClient, chat_id: str | None, checks: list[ArtifactCheck]
) -> list[CheckResult]:
    if not checks:
        return []
    if chat_id is None:
        return [CheckResult(check, False, "no chat yet") for check in checks]
    entries = await app.artifacts(chat_id)
    results = []
    for check in checks:
        matches = [e for e in entries if fnmatchcase(e.get("path", ""), check.path)]
        if not matches:
            results.append(CheckResult(check, False, f"no file matches {check.path}"))
            continue
        reasons = []
        for entry in matches:
            reason = await _meets(app, chat_id, check, entry)
            if reason is None:
                results.append(CheckResult(check, True, f"{entry.get('path')} matches"))
                break
            reasons.append(reason)
        else:
            results.append(CheckResult(check, False, "; ".join(reasons)))
    return results
