"""Fakes for the app, Docker, Langfuse and the harness models. No test reaches a live service."""

import asyncio
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from types import SimpleNamespace

import docker.errors
import httpx
import litellm

from weather_skills_eval.langfuse_gateway import play_scores


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def sleep_sync(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@dataclass
class Reply:
    content: str = ""
    error: str | None = None
    artifacts: list[dict] = field(default_factory=list)
    http_status: int | None = None


# reply(opening_message, turn_number, user_text) -> Reply
ReplyFn = Callable[[str, int, str], Reply]


class FakeApp:
    """The app's HTTP API, in memory. Each completion finishes at once and takes ``answer_seconds``."""

    def __init__(self, reply: ReplyFn, clock: FakeClock, answer_seconds: float = 5.0):
        self.reply = reply
        self.clock = clock
        self.answer_seconds = answer_seconds
        self.chats: dict[str, dict] = {}
        self.completions: list[dict] = []
        self.tools = [
            {
                "id": "plot-1",
                "name": "plot",
                "meta": {"manifest": {"kind": "skill", "skill_name": "plot", "version": "1.2.0"}},
            },
            {
                "id": "plot-2",
                "name": "plot",
                "meta": {"manifest": {"kind": "skill", "skill_name": "plot", "version": "1.10.0"}},
            },
            {
                "id": "fetch",
                "name": "fetch",
                "meta": {"manifest": {"kind": "skill", "skill_name": "fetch", "version": "2"}},
            },
        ]
        self.transport = httpx.MockTransport(self._handle)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        assert request.headers["authorization"] == "Bearer sk-test-key"
        if path == "/api/v1/tool_summary":
            return httpx.Response(200, json=self.tools)
        if path == "/api/v1/skills/":
            return httpx.Response(
                200,
                json=[
                    {
                        "git_url": "https://example.invalid/pack.git",
                        "git_ref": "main",
                        "commit_sha": "1234567",
                    }
                ],
            )
        if path == "/api/v1/chats/new":
            chat = json.loads(request.content)["chat"]
            chat_id = f"chat-{len(self.chats) + 1}"
            self.chats[chat_id] = {"chat": chat, "opening": None, "turns": 0, "artifacts": []}
            return httpx.Response(200, json={"id": chat_id, "chat": chat})
        if path == "/api/chat/completions":
            return self._complete(json.loads(request.content))
        match = re.fullmatch(r"/api/v1/chats/([^/]+)(/.*)?", path)
        if match:
            state = self.chats[match.group(1)]
            rest = match.group(2) or ""
            if rest == "":
                return httpx.Response(200, json={"id": match.group(1), "chat": state["chat"]})
            if rest == "/artifacts":
                return httpx.Response(200, json=state["artifacts"])
            if rest == "/artifacts/zarr/meta":
                return httpx.Response(200, json={"variables": {"precip": {}}, "dims": {"time": 7}})
            if rest == "/artifacts/content":
                return httpx.Response(200, content=b"rain total")
        return httpx.Response(404, json={"detail": f"no route {path}"})

    def _complete(self, body: dict) -> httpx.Response:
        self.completions.append(body)
        state = self.chats[body["chat_id"]]
        history = state["chat"]["history"]["messages"]
        if "turn" in body:
            user = body["turn"]["user_message"]
            assistant = body["turn"]["assistant_message"]
            history[user["id"]] = user
            history[assistant["id"]] = assistant
            text = user["content"]
        else:
            text = body["messages"][-1]["content"]
            state["opening"] = text
        state["turns"] += 1
        reply = self.reply(state["opening"], state["turns"], text)
        if reply.http_status:
            return httpx.Response(reply.http_status, json={"detail": reply.error})
        self.clock.now += self.answer_seconds
        message = history[body["id"]]
        message["done"] = True
        message["content"] = reply.content
        if reply.error:
            message["error"] = {"content": reply.error}
        state["artifacts"].extend(reply.artifacts)
        return httpx.Response(200, json={"status": True, "task_id": body["chat_id"]})


class FakeContainer:
    def __init__(self, statuses: list[str], logs: bytes = b""):
        self._statuses = statuses
        self.status = "created"
        self._logs = logs
        self.removed = False

    def reload(self) -> None:
        if self._statuses:
            self.status = self._statuses.pop(0)

    def logs(self, tail: int = 200) -> bytes:
        return self._logs

    def remove(self, force: bool = False, v: bool = False) -> None:
        self.removed = True


class FakeDocker:
    def __init__(self, container: FakeContainer, local_image: bool = True):
        self.container = container
        self.local_image = local_image
        self.run_calls: list[dict] = []
        self.pulled: list[str] = []
        image = SimpleNamespace(
            id="sha256:image", attrs={"RepoDigests": ["ghcr.io/x/app@sha256:abc"]}
        )
        outer = self

        class Images:
            def get(self, ref):
                if not outer.local_image:
                    raise docker.errors.ImageNotFound("missing")
                return image

            def pull(self, ref):
                outer.pulled.append(ref)
                return image

        class Containers:
            def run(self, image_id, **kwargs):
                outer.run_calls.append({"image": image_id, **kwargs})
                return outer.container

        self.images = Images()
        self.containers = Containers()


def _response(content: str | None = None, tool_call: dict | None = None):
    message = {"role": "assistant", "content": content}
    if tool_call:
        message["tool_calls"] = [{"id": "call-1", "type": "function", "function": tool_call}]
    return litellm.ModelResponse(
        choices=[{"message": message}],
        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    )


class FakeModels:
    """The simulated user and judge models.

    ``user_replies`` maps a word in the opening message to the replies the
    simulated user gives in order; the judge passes every criterion when the
    conversation contains ``DONE``.
    """

    def __init__(self, user_replies: dict[str, list[dict]] | None = None):
        self.user_replies = {k: list(v) for k, v in (user_replies or {}).items()}
        self.calls: list[tuple[str, str | None, dict]] = []

    def __call__(self, role: str, session_id: str | None, **kwargs):
        self.calls.append((role, session_id, kwargs))
        messages = kwargs["messages"]
        if role == "simulated-user":
            system = messages[0]["content"]
            for word, replies in self.user_replies.items():
                if word in system and replies:
                    return _response(json.dumps(replies.pop(0)))
            return _response(
                json.dumps(
                    {
                        "classification": "progress",
                        "revealed_hidden_step": False,
                        "give_up": False,
                        "give_up_reason": "",
                        "message": "please continue",
                    }
                )
            )
        assert role == "judge"
        tools = kwargs.get("tools")
        if tools:
            finish = next(t for t in tools if t["function"]["name"] == "finish_test")
            keys = finish["function"]["parameters"]["properties"]["criteria"]["properties"]
            conversation = json.dumps(messages)
            status = "passed" if "DONE" in conversation else "failed"
            answers = {
                key: {"requirement": key, "reasoning": "checked", "status": status} for key in keys
            }
            return _response(
                tool_call={
                    "name": "finish_test",
                    "arguments": json.dumps(
                        {"criteria": answers, "reasoning": f"verdict {status}"}
                    ),
                }
            )
        return _response(json.dumps({"score": 4, "reasoning": "clear answer"}))


def app_observations(chat_id: str, turns: int) -> list[dict]:
    """One app trace per turn: a cached call, a cache-missing call and a tool run."""
    observations = []
    for turn in range(turns):
        trace = f"{chat_id}-trace-{turn}"
        observations += [
            {
                "id": f"{trace}-root",
                "trace_id": trace,
                "type": "SPAN",
                "name": "weather-skills-chat",
                "level": "DEFAULT",
                "usage_details": {},
                "cost_details": {},
                "total_cost": None,
            },
            {
                "id": f"{trace}-g1",
                "trace_id": trace,
                "type": "GENERATION",
                "name": "llm",
                "level": "DEFAULT",
                "usage_details": {
                    "input": 80000,
                    "output": 300,
                    "total": 80300,
                    "cache_read_input_tokens": 79000,
                    "cache_creation_input_tokens": 500,
                },
                "cost_details": {"total": 0.02},
                "total_cost": 0.02,
            },
            {
                "id": f"{trace}-g2",
                "trace_id": trace,
                "type": "GENERATION",
                "name": "llm",
                "level": "DEFAULT",
                "usage_details": {"input": 77000, "output": 100, "total": 77100},
                "cost_details": {"total": 0.19},
                "total_cost": 0.19,
            },
            {
                "id": f"{trace}-t1",
                "trace_id": trace,
                "type": "TOOL",
                "name": "tool:plot",
                "level": "ERROR" if turn == 0 else "DEFAULT",
                "usage_details": {},
                "cost_details": {},
                "total_cost": None,
            },
        ]
    return observations


class FakeGateway:
    def __init__(self, items: list[dict], models: FakeModels):
        self.items = [
            SimpleNamespace(
                id=item["id"],
                input=item["input"],
                expected_output=item["expected_output"],
                metadata=item["metadata"],
            )
            for item in items
        ]
        self.models = models
        self.experiments: list[dict] = []
        self.linked: list[str] = []
        self.flushed = False

    def dataset(self, name: str):
        return SimpleNamespace(name=name, items=self.items)

    def run_experiment(self, *, dataset, name, task, metadata, concurrency):
        results = []
        for item in dataset.items:
            output = asyncio.run(task(item=item))
            results.append(
                SimpleNamespace(item=item, output=output, evaluations=play_scores(output=output))
            )
        experiment = {"name": name, "metadata": metadata, "items": results}
        self.experiments.append(experiment)
        return SimpleNamespace(run_name=f"{name} - run", experiment_url=None, item_results=results)

    def traced_completion(self, role, session_id, **kwargs):
        return self.models(role, session_id, **kwargs)

    async def chat_observations(self, chat_id: str, turns: int):
        return app_observations(chat_id, turns), True

    def link_item_to_chat(self, chat_id: str) -> None:
        self.linked.append(chat_id)

    def flush(self) -> None:
        self.flushed = True
