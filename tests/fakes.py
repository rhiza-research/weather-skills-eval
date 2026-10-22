"""Fakes for the app, Docker, Langfuse and the harness models, and the files and
replies the app stores. No test reaches a live service."""

import asyncio
import hashlib
import html
import io
import json
import re
import tarfile
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace

import docker.errors
import httpx
import litellm

from weather_skills_eval.langfuse_gateway import play_scores
from weather_skills_eval.scoring import TraceRead


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


def chain(fetcher: str = "arco-era5-fetch", period: str = "weekly") -> list[dict]:
    """A provenance chain: a fetch for western Kenya, then a sum over ``period``."""
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
            "args": {"period": period, "reducer": "sum"},
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


def display_page(path: str) -> str:
    """A successful display_image call of an .html page; the app returns one text."""
    return tool_call(
        "display_image",
        {"path": path},
        f"Showing `{path}` to the user as an interactive page. You cannot see it; to look "
        "at a figure yourself, also save it as a PNG and use read_artifact.",
    )


def html_entry(path: str) -> dict:
    """An .html page listing entry, with the provenance of ``chain()``."""
    return _entry(path, "html", chain())


def image_entry(
    path: str, fetcher: str = "arco-era5-fetch", provenance: bool = True, period: str = "weekly"
) -> dict:
    """A PNG listing entry, with the provenance of ``chain(fetcher, period)`` unless told not."""
    return _entry(path, "image", chain(fetcher, period) if provenance else None)


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
        # How many more downloads of a path, or listings, fail with a 500.
        self.failing_downloads: dict[str, int] = {}
        self.failing_listings = 0
        # Only file requests after this many fail.
        self.failures_after = 0
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
        can_fail = len(self.file_requests) > self.failures_after
        if rest == "/artifacts" and can_fail and self.failing_listings:
            self.failing_listings -= 1
            return httpx.Response(500, json={"detail": "storage busy"})
        if rest == "/artifacts":
            return httpx.Response(200, json=state["artifacts"])
        if rest == "/artifacts/content" and can_fail and self.failing_downloads.get(wanted):
            self.failing_downloads[wanted] -= 1
            return httpx.Response(500, json={"detail": "storage busy"})
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
    makes reload raise NotFound, as for a container removed from outside,
    and ``api-error`` makes it raise APIError. An entry may also be a
    (status, restart count, exit code) tuple, which sets ``attrs``.
    ``archives`` holds each (path, tar bytes) given to ``put_archive``;
    ``put_archive_error`` and ``start_error`` fail those calls, and
    ``put_archive_result`` is what ``put_archive`` returns."""

    def __init__(
        self,
        statuses: list[str],
        logs: bytes = b"",
        name: str = "app",
        remove_error: Exception | None = None,
        put_archive_error: Exception | None = None,
        start_error: Exception | None = None,
        put_archive_result: bool = True,
    ):
        self._statuses = statuses
        self.status = "created"
        self._logs = logs
        self.name = name
        self.remove_error = remove_error
        self.put_archive_error = put_archive_error
        self.start_error = start_error
        self.put_archive_result = put_archive_result
        self.removed = False
        self.started = False
        self.reloads = 0
        # A crashed container that Docker has not restarted, unless a status entry says otherwise.
        self.attrs: dict = {"RestartCount": 0, "State": {"ExitCode": 1}}
        self.archives: list[tuple[str, bytes]] = []

    def put_archive(self, path: str, data: bytes) -> bool:
        if self.put_archive_error:
            raise self.put_archive_error
        self.archives.append((path, data))
        return self.put_archive_result

    def start(self) -> None:
        if self.start_error:
            raise self.start_error
        self.started = True

    def reload(self) -> None:
        self.reloads += 1
        if self._statuses:
            entry = self._statuses.pop(0)
            if isinstance(entry, tuple):
                # (status, restart count, exit code), as ``docker inspect`` reports them.
                self.status, restarts, exit_code = entry
                self.attrs = {"RestartCount": restarts, "State": {"ExitCode": exit_code}}
            else:
                self.status = entry
        if self.status == "gone":
            raise docker.errors.NotFound("container gone")
        if self.status == "api-error":
            raise docker.errors.APIError("daemon unavailable")

    def logs(self, tail: int = 200) -> bytes:
        return self._logs

    def remove(self, force: bool = False, v: bool = False) -> None:
        if self.remove_error:
            raise self.remove_error
        self.removed = True


APP_REPOSITORY = "ghcr.io/rhiza-research/weather-skills-chat"
RUNNER_REPOSITORY = "ghcr.io/rhiza-research/weather-skills-sandbox-runner"
RUNNER_IMAGE = f"{RUNNER_REPOSITORY}:src-171514bbb4e1"
APP_EMULATOR_PATHS = ("/app/sandbox/emulator/emulator.py", "/app/sandbox/emulator/local_runtime.py")


def local_image(
    repo_digests: list[str] | None = None, image_id: str = "sha256:local"
) -> SimpleNamespace:
    """An image built on the Docker host; it has no repo digest unless given one."""
    return SimpleNamespace(id=image_id, attrs={"RepoDigests": repo_digests or []})


def emulator_file(path: str) -> bytes:
    """What the app image holds at ``path``."""
    return f"# {path}\n".encode()


def app_image_files() -> dict[str, bytes]:
    """The files of an app image that ships the emulator."""
    return {path: emulator_file(path) for path in APP_EMULATOR_PATHS}


class FakeSourceContainer:
    """A container created from the app image and never started; ``files``
    maps an absolute path to the bytes ``get_archive`` serves for it."""

    def __init__(
        self,
        image_id: str,
        files: dict[str, bytes],
        archive_error: Exception | None,
        archive_stream: Callable | None = None,
    ):
        self.image_id = image_id
        self.files = files
        self.archive_error = archive_error
        self.archive_stream = archive_stream
        self.read: list[str] = []
        self.started = False
        self.removed = False

    def get_archive(self, path: str):
        self.read.append(path)
        if self.archive_error:
            raise self.archive_error
        if self.archive_stream:
            return self.archive_stream(), {"name": path.rsplit("/", 1)[-1]}
        if path not in self.files:
            raise docker.errors.NotFound(f"Could not find the file {path} in container")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w") as tar:
            data = self.files[path]
            info = tarfile.TarInfo(path.rsplit("/", 1)[-1])
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        raw = buffer.getvalue()
        # The SDK streams the archive in chunks.
        return iter([raw[:100], raw[100:]]), {"name": path.rsplit("/", 1)[-1]}

    def start(self) -> None:
        self.started = True

    def remove(self, force: bool = False, v: bool = False) -> None:
        self.removed = True


class FakeVolume:
    def __init__(self, name: str, labels: dict, remove_error: Exception | None = None):
        self.name = name
        self.labels = labels
        self.remove_error = remove_error
        self.removed = False

    def remove(self, force: bool = False) -> None:
        if self.remove_error:
            raise self.remove_error
        self.removed = True


NO_LOOKUP = object()


class FakeDocker:
    """Docker with an app image and a runner image to pull. ``pull_error``
    fails the pull of every image, ``runner_pull_error`` only the runner's;
    ``local`` is the app image already on the host that ``images.get``
    returns, ``None`` when the host does not have it (``get`` raises
    ImageNotFound), and by default no lookup is expected (``get`` fails the
    test); the runner image is never looked up; ``get_error`` fails
    ``images.get``; ``run_error`` fails ``containers.run`` after the
    container was created; ``create_error`` fails ``containers.create`` of
    the runner; ``app_files`` are the files of the app image the emulator
    is read from (by default both emulator files), ``archive_error``
    fails ``get_archive`` on it and ``archive_stream`` makes the stream it
    returns; ``volume_error`` fails ``volumes.create``;
    ``leftovers`` are labeled containers from an earlier run.
    ``image_digests`` and ``runner_digests`` replace the ``RepoDigests`` of
    the pulled app and runner images. ``events``
    records each container and volume created or removed, in order;
    ``source`` is the container created from the app image to read the
    emulator."""

    def __init__(
        self,
        container: FakeContainer,
        *,
        runner: FakeContainer | None = None,
        pull_error: Exception | None = None,
        runner_pull_error: Exception | None = None,
        local: SimpleNamespace | None | object = NO_LOOKUP,
        get_error: Exception | None = None,
        run_error: Exception | None = None,
        create_error: Exception | None = None,
        app_files: dict[str, bytes] | None = None,
        archive_error: Exception | None = None,
        archive_stream: Callable | None = None,
        volume_error: Exception | None = None,
        volume_remove_error: Exception | None = None,
        leftovers: list[FakeContainer] | None = None,
        image_digests: list[str] | None = None,
        runner_digests: list[str] | None = None,
    ):
        self.container = container
        self.runner = runner or FakeContainer(["running"], name="runner")
        self.source: FakeSourceContainer | None = None
        self.run_calls: list[dict] = []
        self.create_calls: list[dict] = []
        self.pulled: list[str] = []
        self.looked_up: list[str] = []
        self.list_filters: list[dict] = []
        self.created: dict[str, FakeContainer] = {}
        self.volumes_created: dict[str, FakeVolume] = {}
        self.events: list[str] = []
        self.leftovers = leftovers or []
        image = SimpleNamespace(
            id="sha256:image",
            attrs={
                "RepoDigests": image_digests
                if image_digests is not None
                else [
                    "ghcr.io/other/mirror@sha256:111",
                    f"{APP_REPOSITORY}@sha256:abc",
                ]
            },
        )
        runner_image = SimpleNamespace(
            id="sha256:runner",
            attrs={
                "RepoDigests": runner_digests
                if runner_digests is not None
                else [f"{RUNNER_REPOSITORY}@sha256:run"]
            },
        )
        outer = self

        def is_runner(ref: str) -> bool:
            return ref.startswith(RUNNER_REPOSITORY)

        class Images:
            def get(self, ref):
                outer.looked_up.append(ref)
                if is_runner(ref):
                    raise AssertionError("the runner image must be pulled, whatever pull says")
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
                if is_runner(ref):
                    if runner_pull_error:
                        raise runner_pull_error
                    return runner_image
                return image

        class Containers:
            def run(self, image_id, **kwargs):
                outer.run_calls.append({"image": image_id, **kwargs})
                outer.created[kwargs["name"]] = outer.container
                outer.events.append(f"run {kwargs['name']}")
                if run_error:
                    raise run_error
                return outer.container

            def create(self, image_id, **kwargs):
                if "-emulator-" in kwargs["name"]:
                    outer.source = FakeSourceContainer(
                        image_id,
                        app_image_files() if app_files is None else app_files,
                        archive_error,
                        archive_stream,
                    )
                    outer.source.labels = kwargs["labels"]
                    outer.created[kwargs["name"]] = outer.source
                    outer.events.append(f"create {kwargs['name']}")
                    return outer.source
                outer.create_calls.append({"image": image_id, **kwargs})
                if create_error:
                    raise create_error
                outer.created[kwargs["name"]] = outer.runner
                outer.events.append(f"create {kwargs['name']}")
                return outer.runner

            def get(self, name):
                # A removed container is gone, as on a real daemon.
                if name not in outer.created or outer.created[name].removed:
                    raise docker.errors.NotFound(name)
                outer.events.append(f"remove {name}")
                return outer.created[name]

            def list(self, all=False, filters=None):
                outer.list_filters.append(filters)
                return outer.leftovers

        class Volumes:
            def create(self, name, labels):
                if volume_error:
                    raise volume_error
                volume = FakeVolume(name, labels, remove_error=volume_remove_error)
                outer.volumes_created[name] = volume
                outer.events.append(f"volume {name}")
                return volume

            def get(self, name):
                if name not in outer.volumes_created:
                    raise docker.errors.NotFound(name)
                outer.events.append(f"remove {name}")
                return outer.volumes_created[name]

        self.images = Images()
        self.containers = Containers()
        self.volumes = Volumes()

    def volume(self) -> FakeVolume:
        """The one volume the run created."""
        assert len(self.volumes_created) == 1
        return next(iter(self.volumes_created.values()))


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
    simulated user gives in order, each a JSON object, a text sent as is,
    None for a reply without text, or an exception the call raises. Past
    those, the simulated user is done when the assistant's last message
    contains ``DONE``, and asks it to go on otherwise. The judge passes every
    criterion when the messages it gets contain ``DONE`` and not
    ``judge_fails_on``; with ``judge_without_verdict`` its verdict call
    answers in text, without the finish_test call; the answer-quality call
    raises ``quality_error`` and the verdict call ``verdict_error`` when given.
    """

    def __init__(
        self,
        user_replies: dict[str, list[dict]] | None = None,
        judge_fails_on: str | None = None,
        judge_without_verdict: bool = False,
        quality_error: Exception | None = None,
        verdict_error: Exception | None = None,
    ):
        self.user_replies = {k: list(v) for k, v in (user_replies or {}).items()}
        self.judge_fails_on = judge_fails_on
        self.judge_without_verdict = judge_without_verdict
        self.quality_error = quality_error
        self.verdict_error = verdict_error
        self.calls: list[tuple[str, str | None, dict]] = []

    def verdict_calls(self) -> list[dict]:
        """The kwargs of each judge call that delivered a verdict."""
        return [c[2] for c in self.calls if c[0] == "judge" and c[2].get("tools")]

    def user_calls(self) -> list[dict]:
        """The kwargs of each simulated-user call."""
        return [c[2] for c in self.calls if c[0] == "simulated-user"]

    def __call__(self, role: str, session_id: str | None, **kwargs):
        self.calls.append((role, session_id, kwargs))
        messages = kwargs["messages"]
        if role == "simulated-user":
            system = messages[0]["content"]
            for word, replies in self.user_replies.items():
                if word in system and replies:
                    reply = replies.pop(0)
                    if isinstance(reply, Exception):
                        raise reply
                    if reply is None:
                        return _response(None)
                    return _response(reply if isinstance(reply, str) else json.dumps(reply))
            last_reply = messages[1]["content"].rsplit("Assistant:", 1)[-1]
            done = "DONE" in last_reply
            return _response(
                json.dumps(
                    {
                        "classification": "answer" if done else "progress",
                        "revealed_hidden_step": False,
                        "done": done,
                        "give_up": False,
                        "give_up_reason": "",
                        "message": "" if done else "please continue",
                    }
                )
            )
        assert role == "judge"
        tools = kwargs.get("tools")
        if tools and self.verdict_error:
            raise self.verdict_error
        if tools and self.judge_without_verdict:
            return _response("I think it looks fine.")
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
        if self.quality_error:
            raise self.quality_error
        return _response(json.dumps({"score": 4, "reasoning": "clear answer"}))


def app_observations(chat_id: str, turns: int) -> list[dict]:
    """One app trace per turn: a cached call, a cache-missing call and a tool run,
    which fails at level ERROR on turn 0 and with ``ok`` false on turn 1."""
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
                # The app records a failed skill run at DEFAULT with "ok": false.
                "ok": False if turn == 1 else None,
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
        # Replace to change what the app's traces hold, or to raise. It returns
        # (observations, complete), or with the last read error as a third item.
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
        observations, complete, *read_error = self.observations(chat_id, turns)
        return TraceRead(observations, complete, read_error=read_error[0] if read_error else None)

    def link_item_to_chat(self, chat_id: str) -> None:
        self.linked.append(chat_id)

    def record_app_cost(self, cost_usd: float, app_trace_ids: list[str]) -> None:
        self.app_costs.append((cost_usd, app_trace_ids))

    def flush(self) -> None:
        self.flushed = True


def _page(data: list, size: int, cursor: str | None) -> SimpleNamespace:
    """One page of ``data``; the cursor is the next page's start index."""
    start = int(cursor or 0)
    end = start + size
    next_cursor = str(end) if end < len(data) else None
    return SimpleNamespace(data=data[start:end], meta=SimpleNamespace(cursor=next_cursor))


class FakeRunApi:
    """Langfuse's experiments, observations and trace APIs, with only the four
    methods a delete may call; any other attribute does not exist.

    ``experiments`` maps an experiment id to its name; ``items`` maps it to
    its items as (trace_id, output); ``sessions`` maps a chat id to the trace
    ids of its observations; ``traces`` maps a trace id to the session ids of
    its observations (None for one without). Pages hold ``page_size``
    entries, so a delete must follow the cursor. ``error`` is raised by every
    call; ``delete_errors`` maps the number of a ``delete_multiple`` call,
    from 1, to the error it raises, deleting nothing. A deleted trace leaves
    the items and sessions, and an experiment whose item traces are all
    deleted is no longer found. ``calls`` records (method, kwargs) of each
    call.
    """

    START = datetime(2026, 10, 20, 9, 0, tzinfo=UTC)

    def __init__(
        self,
        experiments: dict[str, str] | None = None,
        items: dict[str, list[tuple[str, object]]] | None = None,
        sessions: dict[str, list[str]] | None = None,
        traces: dict[str, list[str | None]] | None = None,
        page_size: int = 2,
        error: Exception | None = None,
        delete_errors: dict[int, Exception] | None = None,
    ):
        self._experiments = experiments or {}
        self._items = items or {}
        self._sessions = sessions or {}
        self._traces = traces or {}
        self._page_size = page_size
        self._error = error
        self._delete_errors = delete_errors or {}
        self.calls: list[tuple[str, dict]] = []
        self.deleted: list[str] = []
        fake = self

        def record(method: str, kwargs: dict) -> None:
            fake.calls.append((method, kwargs))
            if fake._error:
                raise fake._error

        def list_experiments(**kwargs):
            record("experiments.list", kwargs)
            assert kwargs["from_start_time"] == datetime(1970, 1, 1, tzinfo=UTC)
            ids = kwargs["id"].split(",")
            data = [
                SimpleNamespace(id=i, name=name, start_time=fake.START)
                for i, name in fake._experiments.items()
                if i in ids
            ]
            return SimpleNamespace(data=data, meta=SimpleNamespace(cursor=None))

        def list_items(**kwargs):
            record("experiments.list_items", kwargs)
            assert kwargs["from_start_time"] == fake.START
            assert "io" in kwargs["fields"].split(",")
            items = [
                SimpleNamespace(trace_id=trace_id, output=output)
                for trace_id, output in fake._items.get(kwargs["experiment_id"], [])
            ]
            return _page(items, fake._page_size, kwargs.get("cursor"))

        def get_many(**kwargs):
            record("observations.get_many", kwargs)
            if "trace_id" in kwargs:
                assert "basic" in kwargs["fields"].split(",")
                trace = [
                    SimpleNamespace(id=f"t{n}", trace_id=kwargs["trace_id"], session_id=session_id)
                    for n, session_id in enumerate(fake._traces.get(kwargs["trace_id"], []))
                ]
                return _page(trace, fake._page_size, kwargs.get("cursor"))
            session = [
                SimpleNamespace(id=f"o{n}", trace_id=trace_id)
                for n, trace_id in enumerate(fake._sessions.get(kwargs["session_id"], []))
            ]
            return _page(session, fake._page_size, kwargs.get("cursor"))

        def delete_multiple(**kwargs):
            record("trace.delete_multiple", kwargs)
            number = sum(1 for method, _ in fake.calls if method == "trace.delete_multiple")
            if number in fake._delete_errors:
                raise fake._delete_errors[number]
            fake.deleted.extend(kwargs["trace_ids"])
            gone = set(kwargs["trace_ids"])
            for experiment_id, items in list(fake._items.items()):
                left = [item for item in items if item[0] not in gone]
                fake._items[experiment_id] = left
                if items and not left:
                    fake._experiments.pop(experiment_id, None)
            for chat_id, trace_ids in fake._sessions.items():
                fake._sessions[chat_id] = [t for t in trace_ids if t not in gone]
            return SimpleNamespace(message="deleted")

        self.api = SimpleNamespace(
            experiments=SimpleNamespace(list=list_experiments, list_items=list_items),
            observations=SimpleNamespace(get_many=get_many),
            trace=SimpleNamespace(delete_multiple=delete_multiple),
        )

    def methods(self) -> set[str]:
        return {method for method, _ in self.calls}
