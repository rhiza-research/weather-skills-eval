"""The app's chat API, used the way the app's own automation runner uses it.

The first turn pre-creates the chat with a user and an empty assistant
message, then posts the completion with that chat id, the assistant id and a
session id, so the app runs the tool loop as a background task and returns at
once. Later turns send the new messages in the ``turn`` field and the app
rebuilds the transcript from storage. An API-key client cannot join the
app's socket events, so completion is detected by polling the chat until
the assistant message is ``done``.
"""

import asyncio
import re
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import httpx

_DETAILS = re.compile(r"<details\b[^>]*>.*?</details>", re.IGNORECASE | re.DOTALL)


class ChatError(Exception):
    """The app returned an error for a turn."""


def visible_text(content: str) -> str:
    """The reply as a user reads it: tool-call and reasoning blocks removed."""
    return _DETAILS.sub("", content or "").strip()


def _version_key(version: str | None) -> tuple:
    text = (version or "0").lstrip("vV")
    return tuple(
        (0, int(part), "") if part.isdigit() else (1, 0, part.lower())
        for part in re.findall(r"\d+|[A-Za-z]+", text)
    )


def highest_skill_versions(tools: list[dict]) -> list[str]:
    """Tool ids for chat: every tool, with each skill name kept at its highest version."""
    chosen: list[dict] = []
    by_skill: dict[str, int] = {}
    for tool in tools:
        manifest = (tool.get("meta") or {}).get("manifest") or {}
        if manifest.get("kind") != "skill":
            chosen.append(tool)
            continue
        key = (manifest.get("skill_name") or tool.get("name") or "").strip().lower()
        if key not in by_skill:
            by_skill[key] = len(chosen)
            chosen.append(tool)
            continue
        current = (chosen[by_skill[key]].get("meta") or {}).get("manifest") or {}
        if _version_key(manifest.get("version")) > _version_key(current.get("version")):
            chosen[by_skill[key]] = tool
    return [tool["id"] for tool in chosen]


class AppClient:
    def __init__(
        self, base_url: str, api_key: str, transport: httpx.AsyncBaseTransport | None = None
    ):
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(60),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _json(self, method: str, path: str, **kwargs):
        response = await self._http.request(method, path, **kwargs)
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail")
            except ValueError:
                detail = response.text
            raise ChatError(f"{method} {path} returned {response.status_code}: {detail}")
        return response.json()

    async def tool_summary(self) -> list[dict]:
        return await self._json("GET", "/api/v1/tool_summary")

    async def skill_packs(self) -> list[dict]:
        return await self._json("GET", "/api/v1/skills/")

    async def create_chat(self, chat: dict) -> dict:
        return await self._json("POST", "/api/v1/chats/new", json={"chat": chat})

    async def complete(self, body: dict) -> dict:
        return await self._json("POST", "/api/chat/completions", json=body)

    async def get_chat(self, chat_id: str) -> dict:
        return await self._json("GET", f"/api/v1/chats/{chat_id}")

    async def artifacts(self, chat_id: str) -> list[dict]:
        return await self._json("GET", f"/api/v1/chats/{chat_id}/artifacts")

    async def zarr_meta(self, chat_id: str, path: str) -> dict:
        return await self._json(
            "GET", f"/api/v1/chats/{chat_id}/artifacts/zarr/meta", params={"path": path}
        )

    async def artifact_content(self, chat_id: str, path: str) -> bytes:
        response = await self._http.get(
            f"/api/v1/chats/{chat_id}/artifacts/content", params={"path": path}
        )
        if response.status_code >= 400:
            raise ChatError(f"artifact {path} returned {response.status_code}")
        return response.content


@dataclass
class Answer:
    content: str
    seconds: float


def _message(role: str, content: str, model: str, parent_id: str | None) -> dict:
    message = {
        "id": str(uuid.uuid4()),
        "parentId": parent_id,
        "childrenIds": [],
        "role": role,
        "content": content,
        "timestamp": int(time.time()),
    }
    if role == "assistant":
        message.update({"model": model, "done": False})
    else:
        message["models"] = [model]
    return message


class Conversation:
    """One chat with the app. Each ``send`` is one turn and waits for the reply."""

    def __init__(
        self,
        app: AppClient,
        *,
        model: str,
        tool_ids: list[str],
        turn_timeout: float,
        poll_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        self._app = app
        self._model = model
        self._tool_ids = tool_ids
        self._turn_timeout = turn_timeout
        self._poll_interval = poll_interval
        self._clock = clock
        self._sleep = sleep
        self._session_id = f"weather-skills-eval:{uuid.uuid4()}"
        self._last_assistant_id: str | None = None
        self.chat_id: str | None = None
        self.turns = 0
        self.answer_seconds: list[float] = []
        self.error: str | None = None

    def _body(self, assistant_id: str) -> dict:
        return {
            "model": self._model,
            "chat_id": self.chat_id,
            "id": assistant_id,
            "session_id": self._session_id,
            "tool_ids": self._tool_ids,
            "features": {},
            "params": {"function_calling": "native"},
            "background_tasks": {"title_generation": False},
        }

    async def _start(self, text: str) -> str:
        user = _message("user", text, self._model, None)
        assistant = _message("assistant", "", self._model, user["id"])
        user["childrenIds"] = [assistant["id"]]
        chat = await self._app.create_chat(
            {
                "title": "weather-skills-eval",
                "models": [self._model],
                "history": {
                    "messages": {user["id"]: user, assistant["id"]: assistant},
                    "currentId": assistant["id"],
                },
                "messages": [user, assistant],
            }
        )
        self.chat_id = chat["id"]
        body = self._body(assistant["id"])
        body["messages"] = [{"role": "user", "content": text}]
        await self._app.complete(body)
        return assistant["id"]

    async def _continue(self, text: str) -> str:
        parent = self._last_assistant_id
        user = _message("user", text, self._model, parent)
        assistant = _message("assistant", "", self._model, user["id"])
        body = self._body(assistant["id"])
        body["turn"] = {
            "parent_id": parent,
            "expected_revision": None,
            "user_message": user,
            "assistant_message": assistant,
        }
        await self._app.complete(body)
        return assistant["id"]

    async def _wait_done(self, assistant_id: str, started: float) -> str:
        while True:
            chat = await self._app.get_chat(self.chat_id)
            messages = ((chat.get("chat") or {}).get("history") or {}).get("messages") or {}
            message = messages.get(assistant_id) or {}
            if message.get("done"):
                error = message.get("error")
                if error:
                    content = error.get("content") if isinstance(error, dict) else error
                    raise ChatError(f"The app reported an error: {content}")
                return message.get("content") or ""
            if self._clock() - started >= self._turn_timeout:
                raise ChatError(f"No reply within {self._turn_timeout:g} s")
            await self._sleep(self._poll_interval)

    async def send(self, text: str) -> Answer:
        started = self._clock()
        self.turns += 1
        try:
            if self.chat_id is None:
                assistant_id = await self._start(text)
            else:
                assistant_id = await self._continue(text)
            content = await self._wait_done(assistant_id, started)
        except ChatError as exc:
            self.error = str(exc)
            raise
        except httpx.HTTPError as exc:
            self.error = f"HTTP error talking to the app: {exc}"
            raise ChatError(self.error) from exc
        self._last_assistant_id = assistant_id
        seconds = self._clock() - started
        self.answer_seconds.append(seconds)
        return Answer(content=content, seconds=seconds)
