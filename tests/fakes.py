"""Fakes for the app, Docker, Langfuse and the harness models, and the files and
replies the app stores. No test reaches a live service."""

import asyncio
import hashlib
import html
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
    """``artifacts`` are added to the chat's file listing; ``files`` maps a chat
    path to the bytes the app serves for it."""

    content: str = ""
    error: str | None = None
    artifacts: list[dict] = field(default_factory=list)
    files: dict[str, bytes] = field(default_factory=dict)
    http_status: int | None = None


def chain(fetcher: str = "arco-era5-fetch") -> list[dict]:
    """A provenance chain: a fetch for western Kenya, then a weekly sum."""
    return [
        {
            "skill": fetcher,
            "version": "0.0.2",
            "args": {"bbox": "33.9,-4.7,35.9,1.2", "start_time": "2026-10-05"},
            "input": None,
        },
        {
            "skill": "aggregate-temporal",
            "version": "0.0.2",
            "args": {"period": "weekly", "reducer": "sum"},
            "input": {
                "basename": "rain.zarr",
                "hash": hashlib.sha256(b"rain.zarr").hexdigest(),
            },
        },
    ]


def provenance_detail(history: list) -> dict:
    """The listing's ``provenance_detail`` for one unlabeled chain."""
    crumbs = [step["skill"] for step in history]
    return {"crumbs": crumbs, "branches": [{"label": None, "steps": history, "crumbs": crumbs}]}


def image_bytes(path: str) -> bytes:
    """What the app serves for an image; the harness only passes it on."""
    return f"image {path}".encode()


def _entry(path: str, kind: str, history: list | None) -> dict:
    entry = {"path": path, "name": path.rsplit("/", 1)[-1], "kind": kind, "is_dir": kind == "zarr"}
    if history is not None:
        detail = provenance_detail(history)
        entry.update(provenance=detail["crumbs"], provenance_detail=detail)
    return entry


def tool_call(name: str, arguments: dict, result) -> str:
    """A tool call as the app stores it in the assistant message: arguments and
    a list or object result are JSON text, each attribute HTML-escaped JSON."""
    if isinstance(result, (list, dict)):
        result = json.dumps(result, indent=2)
    return (
        f'<details type="tool_calls" done="true" id="call-{name}" name="{name}" '
        f'arguments="{html.escape(json.dumps(json.dumps(arguments)))}" '
        f'result="{html.escape(json.dumps(result))}" files="">\n'
        "<summary>Tool Executed</summary>\n</details>\n"
    )


def display_image(path: str) -> str:
    """A successful display_image call; the app keeps only the note."""
    return tool_call("display_image", {"path": path}, [f"Showing `{path}`."])


def image_entry(path: str, fetcher: str = "arco-era5-fetch", provenance: bool = True) -> dict:
    """A PNG listing entry, with the provenance of ``chain(fetcher)`` unless told not."""
    return _entry(path, "image", chain(fetcher) if provenance else None)


def zarr_entry(path: str, fetcher: str = "arco-era5-fetch", provenance: bool = True) -> dict:
    """A zarr store listing entry, with the provenance of ``chain(fetcher)`` unless told not."""
    return _entry(path, "zarr", chain(fetcher) if provenance else None)


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
        # What the app reports as the catalog model's base model; None for none.
        self.base_model_id: str | None = "open-router-.anthropic/claude-haiku-5.5"
        self.catalog_model_requests: list[str] = []
        # (endpoint, path) of each file request: listings and downloads.
        self.file_requests: list[tuple[str, str | None]] = []
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
        if path == "/api/v1/models/model":
            model_id = request.url.params["id"]
            self.catalog_model_requests.append(model_id)
            return httpx.Response(200, json={"id": model_id, "base_model_id": self.base_model_id})
        if path == "/api/v1/chats/new":
            chat = json.loads(request.content)["chat"]
            chat_id = f"chat-{len(self.chats) + 1}"
            self.chats[chat_id] = {
                "chat": chat,
                "opening": None,
                "turns": 0,
                "artifacts": [],
                "files": {},
            }
            return httpx.Response(200, json={"id": chat_id, "chat": chat})
        if path == "/api/chat/completions":
            return self._complete(json.loads(request.content))
        match = re.fullmatch(r"/api/v1/chats/([^/]+)(/.*)?", path)
        if match:
            state = self.chats[match.group(1)]
            rest = match.group(2) or ""
            if rest == "":
                return httpx.Response(200, json={"id": match.group(1), "chat": state["chat"]})
            if rest.startswith("/artifacts"):
                return self._files(state, rest, request.url.params)
        return httpx.Response(404, json={"detail": f"no route {path}"})

    def _files(self, state: dict, rest: str, params) -> httpx.Response:
        wanted = params.get("path")
        self.file_requests.append((rest, wanted))
        files = state["files"]
        if rest == "/artifacts":
            return httpx.Response(200, json=state["artifacts"])
        if rest == "/artifacts/content" and wanted in files:
            return httpx.Response(200, content=files[wanted])
        return httpx.Response(404, json={"detail": "Not found"})

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
        state["files"].update(reply.files)
        return httpx.Response(200, json={"status": True, "task_id": body["chat_id"]})


class FakeContainer:
    """``statuses`` are returned by successive reloads; the status ``gone``
    makes reload raise NotFound, as for a container removed from outside."""

    def __init__(
        self,
        statuses: list[str],
        logs: bytes = b"",
        name: str = "app",
        remove_error: Exception | None = None,
    ):
        self._statuses = statuses
        self.status = "created"
        self._logs = logs
        self.name = name
        self.remove_error = remove_error
        self.removed = False

    def reload(self) -> None:
        if self._statuses:
            self.status = self._statuses.pop(0)
        if self.status == "gone":
            raise docker.errors.NotFound("container gone")

    def logs(self, tail: int = 200) -> bytes:
        return self._logs

    def remove(self, force: bool = False, v: bool = False) -> None:
        if self.remove_error:
            raise self.remove_error
        self.removed = True


APP_REPOSITORY = "ghcr.io/rhiza-research/weather-skills-chat"


def local_image(repo_digests: list[str] | None = None) -> SimpleNamespace:
    """An image built on the Docker host; it has no repo digest unless given one."""
    return SimpleNamespace(id="sha256:local", attrs={"RepoDigests": repo_digests or []})


NO_LOOKUP = object()


class FakeDocker:
    """Docker with one image to pull. ``pull_error`` fails the pull; ``local``
    is the image already on the host that ``images.get`` returns, ``None``
    when the host does not have it (``get`` raises ImageNotFound), and by
    default no lookup is expected (``get`` fails the test); ``get_error``
    fails ``images.get``; ``run_error`` fails ``containers.run`` after the
    container was created; ``leftovers`` are labeled containers from an
    earlier run."""

    def __init__(
        self,
        container: FakeContainer,
        *,
        pull_error: Exception | None = None,
        local: SimpleNamespace | None | object = NO_LOOKUP,
        get_error: Exception | None = None,
        run_error: Exception | None = None,
        leftovers: list[FakeContainer] | None = None,
    ):
        self.container = container
        self.run_calls: list[dict] = []
        self.pulled: list[str] = []
        self.looked_up: list[str] = []
        self.list_filters: list[dict] = []
        self.created: dict[str, FakeContainer] = {}
        self.leftovers = leftovers or []
        image = SimpleNamespace(
            id="sha256:image",
            attrs={
                "RepoDigests": [
                    "ghcr.io/other/mirror@sha256:111",
                    f"{APP_REPOSITORY}@sha256:abc",
                ]
            },
        )
        outer = self

        class Images:
            def get(self, ref):
                outer.looked_up.append(ref)
                if get_error:
                    raise get_error
                if local is NO_LOOKUP:
                    raise AssertionError("the harness must pull, not use a local tag")
                if local is None:
                    raise docker.errors.ImageNotFound(f"No such image: {ref}")
                return local

            def pull(self, ref):
                outer.pulled.append(ref)
                if pull_error:
                    raise pull_error
                return image

        class Containers:
            def run(self, image_id, **kwargs):
                outer.run_calls.append({"image": image_id, **kwargs})
                outer.created[kwargs["name"]] = outer.container
                if run_error:
                    raise run_error
                return outer.container

            def get(self, name):
                if name not in outer.created:
                    raise docker.errors.NotFound(name)
                return outer.created[name]

            def list(self, all=False, filters=None):
                outer.list_filters.append(filters)
                return outer.leftovers

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
    messages it gets contain ``DONE`` and not ``judge_fails_on``.
    """

    def __init__(
        self,
        user_replies: dict[str, list[dict]] | None = None,
        judge_fails_on: str | None = None,
    ):
        self.user_replies = {k: list(v) for k, v in (user_replies or {}).items()}
        self.judge_fails_on = judge_fails_on
        self.calls: list[tuple[str, str | None, dict]] = []

    def verdict_calls(self) -> list[dict]:
        """The kwargs of each judge call that delivered a verdict."""
        return [c[2] for c in self.calls if c[0] == "judge" and c[2].get("tools")]

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
            passed = "DONE" in conversation and not (
                self.judge_fails_on and self.judge_fails_on in conversation
            )
            status = "passed" if passed else "failed"
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
                "model": "provider/model-b",
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
                "model": "provider/model-a",
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


def broadcast_observations(chat_id: str) -> list[dict]:
    """One app trace with two OpenRouter Broadcast generations, shaped like the
    real ones: usage_details holds only input, output and total (input being
    the whole prompt), and the cache counts are top-level metadata keys with
    dotted names."""
    trace = f"{chat_id}-trace-0"

    def generation(obs_id, cached, cache_write, cost_details, total_cost):
        return {
            "id": obs_id,
            "trace_id": trace,
            "type": "GENERATION",
            "name": "llm",
            "level": "DEFAULT",
            "usage_details": {
                "input": cached + cache_write,
                "output": 50,
                "total": cached + cache_write + 50,
            },
            "cost_details": cost_details,
            "total_cost": total_cost,
            "metadata": {
                "attributes.gen_ai.usage.input_tokens.cached": str(cached),
                "attributes.gen_ai.usage.input_tokens.cache_write": str(cache_write),
            },
        }

    return [
        {
            "id": f"{trace}-root",
            "trace_id": trace,
            "type": "SPAN",
            "name": "weather-skills-chat",
            "level": "DEFAULT",
            "usage_details": {},
            "cost_details": {},
            "total_cost": None,
            "metadata": {},
        },
        generation("b1", 80000, 0, {"input": 0.016}, 0.016),
        generation("b2", 0, 60000, {"total": 0.2}, 0.2),
    ]


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
        # (cost_usd, app_trace_ids) of each app cost rollup.
        self.app_costs: list[tuple[float, list[str]]] = []
        self.flushed = False
        self.observation_requests: list[tuple[str, int]] = []
        # Replace to change what the app's traces hold, or to raise.
        self.observations = lambda chat_id, turns: (app_observations(chat_id, turns), True)
        self.dataset_error: Exception | None = None
        self.dataset_metadata: object = None

    def dataset(self, name: str):
        if self.dataset_error:
            raise self.dataset_error
        return SimpleNamespace(
            name=name, items=self.items, version=None, metadata=self.dataset_metadata
        )

    def run_experiment(self, *, dataset, name, description, task, metadata, concurrency):
        results = []
        for item in dataset.items:
            output = asyncio.run(task(item=item))
            results.append(
                SimpleNamespace(item=item, output=output, evaluations=play_scores(output=output))
            )
        experiment = {
            "name": name,
            "description": description,
            "metadata": metadata,
            "items": results,
        }
        self.experiments.append(experiment)
        return SimpleNamespace(run_name=f"{name} - run", experiment_url=None, item_results=results)

    def traced_completion(self, role, session_id, **kwargs):
        return self.models(role, session_id, **kwargs)

    async def chat_observations(self, chat_id: str, turns: int):
        self.observation_requests.append((chat_id, turns))
        return self.observations(chat_id, turns)

    def link_item_to_chat(self, chat_id: str) -> None:
        self.linked.append(chat_id)

    def record_app_cost(self, cost_usd: float, app_trace_ids: list[str]) -> None:
        self.app_costs.append((cost_usd, app_trace_ids))

    def flush(self) -> None:
        self.flushed = True
