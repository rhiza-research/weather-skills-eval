import asyncio
import io
import json
import logging
import types
import uuid
import warnings
from pathlib import Path

import docker.errors
import httpx
import litellm
import pytest
import scenario
from fakes import (
    FakeContainer,
    FakeModels,
    FakeRunApi,
    FakeSourceContainer,
    FakeVolume,
    app_image_files,
)
from langfuse import Langfuse
from langfuse.api.core.api_error import ApiError as LangfuseApiError
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from scenario import AgentAdapter, AgentInput, AgentRole, ScenarioResult

from weather_skills_eval import cli
from weather_skills_eval import instance as instance_module
from weather_skills_eval.app_client import AppClient, ChatError
from weather_skills_eval.checks import Evidence, ReplyFiles, collect_evidence, reply_files
from weather_skills_eval.config import ConfigError, load_config, parse_config
from weather_skills_eval.instance import (
    AppInstance,
    AppStartError,
    RunnerStartError,
    repository_of,
)
from weather_skills_eval.judge import GoalJudge, rate_answer
from weather_skills_eval.langfuse_gateway import LangfuseGateway, LangfuseRuns
from weather_skills_eval.play import neutral_transcript
from weather_skills_eval.runner import DatasetError, RunSummary, base_model
from weather_skills_eval.settings import SettingsError, load_settings
from weather_skills_eval.simulated_user import SimulatedUser
from weather_skills_eval.tasks import TaskError, parse_task

REPO = Path(__file__).resolve().parent.parent

BASE_ENV = {
    "WEBUI_SECRET_KEY": "${S}",
    "BOOTSTRAP_ADMIN_EMAIL": "admin",
    "BOOTSTRAP_ADMIN_PASSWORD": "${P}",
    "BOOTSTRAP_ADMIN_API_KEY": "${K}",
}
ENVIRON = {"S": "secret", "P": "password", "K": "sk-key", "EMPTY": "  "}


def parse(env_extra=None, environ=ENVIRON):
    raw = {
        "image": "img:tag",
        "runner_image": "runner:tag",
        "model": "m",
        "dataset": "d",
        "env": {**BASE_ENV, **(env_extra or {})},
    }
    return parse_config(raw, default_name="x", environ=environ)


# Configuration


def test_references_in_structured_values_are_substituted_into_the_structure():
    result = parse(
        {"CATALOG_MODELS": [{"id": "m", "params": {"api_key": "${K}", "note": 'a "${S}"'}}]}
    )
    models = json.loads(result.env["CATALOG_MODELS"])
    assert models[0]["params"]["api_key"] == "sk-key"
    assert models[0]["params"]["note"] == 'a "secret"'


def test_literal_secret_inside_a_mapping_stops_the_run():
    with pytest.raises(ConfigError) as error:
        parse({"CATALOG_MODELS": [{"id": "m", "params": {"api_key": "sk-literal"}}]})
    assert error.value.field == "env.CATALOG_MODELS[0].params.api_key"


def test_token_counts_and_tokenizer_names_are_not_secrets():
    result = parse(
        {
            "CATALOG_MODELS": [{"id": "m", "params": {"max_tokens": 4000}}],
            "TIKTOKEN_ENCODING_NAME": "cl100k_base",
        }
    )
    assert json.loads(result.env["CATALOG_MODELS"])[0]["params"]["max_tokens"] == 4000


def test_secret_reference_with_whitespace_resolves_to_the_value_alone():
    result = parse({"OPENAI_API_KEYS": "  ${K}  "})
    assert result.env["OPENAI_API_KEYS"] == "sk-key"


def test_empty_referenced_variable_stops_the_run():
    with pytest.raises(ConfigError, match="empty") as error:
        parse({"OPENAI_API_KEYS": "${EMPTY}"})
    assert error.value.field == "env.OPENAI_API_KEYS"


@pytest.mark.parametrize("port", ["0", "70000", "http", "80a"])
def test_invalid_port_stops_the_run(port):
    with pytest.raises(ConfigError) as error:
        parse({"PORT": port})
    assert error.value.field == "env.PORT"


def test_valid_port_is_used():
    assert parse({"PORT": "3000"}).port == "3000"


@pytest.mark.parametrize(
    "configs, field, message",
    [
        ("not json", "env.OPENAI_API_CONFIGS", "is not JSON"),
        ('["a"]', "env.OPENAI_API_CONFIGS", "must map each connection"),
        ('{"0": "a"}', "env.OPENAI_API_CONFIGS", "must map each connection"),
        ({"0": {"prefix_id": 1}}, "env.OPENAI_API_CONFIGS.0.prefix_id", "string"),
        ({"0": {"enable": "no"}}, "env.OPENAI_API_CONFIGS.0.enable", "true or false"),
        ({"0": {"model_ids": "m"}}, "env.OPENAI_API_CONFIGS.0.model_ids", "list"),
    ],
)
def test_invalid_connection_configs_stop_the_run(configs, field, message):
    with pytest.raises(ConfigError, match=message) as error:
        parse({"OPENAI_API_CONFIGS": configs})
    assert error.value.field == field


ROUTER = {"prefix_id": "open-router-"}


@pytest.mark.parametrize(
    "base, connections, expected",
    [
        ("open-router-.anthropic/claude-haiku-5.5", [ROUTER], "anthropic/claude-haiku-5.5"),
        (" open-router-.anthropic/claude-haiku-5.5 ", [ROUTER], "anthropic/claude-haiku-5.5"),
        ("anthropic/claude-haiku-5.5", [], "anthropic/claude-haiku-5.5"),
        ("~anthropic/claude-opus-latest", [ROUTER], "~anthropic/claude-opus-latest"),
        # The last connection that may serve the model wins, as in the app.
        ("a.b.model", [{"prefix_id": "a.b"}, {"prefix_id": "a"}], "b.model"),
        ("a.b.model", [{"prefix_id": "a"}, {"prefix_id": "a.b"}], "model"),
        # A disabled connection, or one whose model_ids leave the model out, does not serve it.
        ("a.model", [{"prefix_id": "a"}, {"prefix_id": "b", "enable": False}], "model"),
        ("a.model", [{"prefix_id": "a"}, {"model_ids": ["other"]}], "model"),
        ("a.model", [{}, {"prefix_id": "a", "model_ids": ["model"]}], "model"),
        # A connection without a prefix serving the model leaves it as is.
        ("a.model", [{"prefix_id": "a"}, {}], "a.model"),
    ],
)
def test_base_model_drops_the_serving_connections_prefix(base, connections, expected):
    assert base_model({"base_model_id": base}, "catalog", connections) == expected


def test_connections_are_read_in_the_order_written():
    configs = {"0": {"prefix_id": "a"}, "1": {"enable": False}}
    assert parse({"OPENAI_API_CONFIGS": configs}).connections == list(configs.values())
    assert parse().connections == []


def test_non_utf8_config_file_is_a_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"image: \xff\xfe\n")
    with pytest.raises(ConfigError, match="not UTF-8"):
        load_config(path, ENVIRON)


def test_example_config_is_valid():
    environ = {
        name: "value"
        for name in (
            "APP_IMAGE",
            "RUNNER_IMAGE",
            "APP_WEBUI_SECRET_KEY",
            "APP_ADMIN_PASSWORD",
            "APP_ADMIN_API_KEY",
            "APP_OPENROUTER_API_KEY",
            "LANGFUSE_BASE_URL",
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
        )
    }
    config = load_config(REPO / "configs" / "example.yaml", environ)
    assert config.name == "example"
    assert config.runner_image == "value"
    assert json.loads(config.env["OPENAI_API_CONFIGS"]) == {
        "0": {"model_call_tracing": "openrouter"}
    }


def env_example_names() -> list[str]:
    """The variables ``.env.example`` sets (not the commented-out ones)."""
    lines = (REPO / ".env.example").read_text().splitlines()
    return [line.split("=", 1)[0] for line in lines if line and not line.startswith("#")]


@pytest.mark.parametrize(
    "path", sorted((REPO / "configs").glob("*.yaml")), ids=lambda path: path.name
)
def test_every_config_loads_with_the_env_example_variables(path):
    assert "RUNNER_IMAGE" in env_example_names()
    environ = {name: "value" for name in env_example_names()}
    config = load_config(path, environ)
    assert config.image == "value"
    assert config.runner_image == "value"


# Settings


def settings_env(**overrides):
    env = {
        "LANGFUSE_PUBLIC_KEY": "pk",
        "LANGFUSE_SECRET_KEY": "sk",
        "OPENROUTER_API_KEY": "or",
        "LANGFUSE_TRACING_ENVIRONMENT": "eval",
        "EVAL_SIMULATED_USER_MODEL": "u",
        "EVAL_JUDGE_MODEL": "j",
        "EVAL_DOCKER_NETWORK": "n",
    }
    env.update(overrides)
    return env


@pytest.mark.parametrize("value", ["Production", " PRODUCTION ", "production\n"])
def test_production_guard_ignores_case_and_whitespace(value):
    with pytest.raises(SettingsError, match="LANGFUSE_TRACING_ENVIRONMENT"):
        load_settings(settings_env(LANGFUSE_TRACING_ENVIRONMENT=value))


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_non_finite_numbers_are_rejected(value):
    with pytest.raises(SettingsError, match="EVAL_TURN_TIMEOUT_SECONDS"):
        load_settings(settings_env(EVAL_TURN_TIMEOUT_SECONDS=value))


def test_settle_must_be_less_than_trace_timeout():
    with pytest.raises(SettingsError, match="EVAL_TRACE_SETTLE_SECONDS"):
        load_settings(settings_env(EVAL_TRACE_SETTLE_SECONDS="60", EVAL_TRACE_TIMEOUT_SECONDS="60"))


def test_retries_and_harness_commit_are_read():
    settings = load_settings(settings_env(EVAL_MODEL_RETRIES="0", EVAL_HARNESS_COMMIT="abc123"))
    assert settings.model_retries == 0
    assert settings.harness_commit == "abc123"


# Tasks


@pytest.mark.parametrize("opening", ["", "   ", None])
def test_empty_or_blank_input_is_a_task_error(opening):
    item = {
        "id": "i1",
        "input": opening,
        "expected_output": {"judge_criteria": ["c"]},
        "metadata": {"facts": "f", "turn_limit": 2},
    }
    with pytest.raises(TaskError, match="i1"):
        parse_task(item)


# App client and checks


def client(handler) -> AppClient:
    return AppClient("http://app", "sk-key", transport=httpx.MockTransport(handler))


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>proxy error</html>"),
        httpx.Response(200, json=["not", "an", "object"]),
        httpx.Response(502, text="Bad Gateway"),
    ],
)
def test_unreadable_app_responses_are_chat_errors(response):
    app = client(lambda request: response)
    with pytest.raises(ChatError):
        asyncio.run(app.get_chat("c1"))


def test_create_chat_without_id_is_a_chat_error():
    app = client(lambda request: httpx.Response(200, json={"chat": {}}))
    with pytest.raises(ChatError, match="no chat id"):
        asyncio.run(app.create_chat({}))


def test_artifact_listing_error_is_reported():
    app = client(lambda request: httpx.Response(500, json={"detail": "disk"}))
    evidence = asyncio.run(collect_evidence(app, "c1", presenting("maps/rain.png"), ["image"]))
    assert "could not be listed" in evidence.problem
    assert evidence.files == []


def presenting(*paths: str, candidates: frozenset[str] = frozenset()) -> list[ReplyFiles]:
    """One reply, the reply to turn 1, that names ``paths``."""
    return [ReplyFiles(1, list(paths), candidates)]


def test_a_failed_listing_never_reports_a_candidate_as_unlisted():
    app = client(lambda request: httpx.Response(500, json={"detail": "disk"}))
    replies = presenting("maps/rain.png", "kenya.zarr", candidates=frozenset({"kenya.zarr"}))
    evidence = asyncio.run(collect_evidence(app, "c1", replies, ["zarr"]))
    assert evidence.unlisted == ["maps/rain.png"]
    assert evidence.read_failed


def test_an_unlisted_candidate_is_neither_presented_nor_missing():
    entry = {
        "path": "maps/rain.png",
        "kind": "image",
        "provenance_detail": {"branches": [{"label": None, "steps": [], "crumbs": []}]},
    }

    def handler(request):
        if request.url.path.endswith("/artifacts"):
            return httpx.Response(200, json=[entry])
        return httpx.Response(200, content=b"png")

    waits = []

    async def wait():
        waits.append(1)

    replies = [
        ReplyFiles(1, ["maps/rain.png"]),
        # A later reply that names only paths the listing does not hold presents nothing.
        ReplyFiles(2, ["missing.zarr", "precip"], frozenset({"missing.zarr", "precip"})),
    ]
    for every in (False, True):
        evidence = asyncio.run(
            collect_evidence(
                client(handler), "c1", replies, ["image"], every=every, retries=1, wait=wait
            )
        )
        assert evidence.problem is None
        assert evidence.unlisted == []
        assert [(f.path, f.turn) for f in evidence.files] == [("maps/rain.png", 1)]
    # The latest naming reply's candidates are not listed: the listing is read
    # once more in each call, in case it lags.
    assert waits == [1, 1]


def test_the_final_read_waits_for_a_backtick_file_the_listing_lags():
    rain = {"path": "maps/rain.png", "kind": "image", "provenance_detail": PROVENANCE}
    store = {"path": "k.zarr", "kind": "zarr", "is_dir": True, "provenance_detail": PROVENANCE}
    listings = []

    def handler(request):
        if request.url.path.endswith("/artifacts"):
            listings.append(request)
            # The store reaches the listing on its second read.
            return httpx.Response(200, json=[rain] if len(listings) == 1 else [rain, store])
        return httpx.Response(200, content=b"png")

    waits = []

    async def wait():
        waits.append(1)

    replies = [
        ReplyFiles(1, ["maps/rain.png"]),
        ReplyFiles(2, ["k.zarr"], frozenset({"k.zarr"})),
    ]
    evidence = asyncio.run(
        collect_evidence(client(handler), "c1", replies, ["zarr"], retries=1, wait=wait)
    )

    # The latest reply's store, not the earlier map.
    assert evidence.problem is None
    assert [(f.path, f.turn) for f in evidence.files] == [("k.zarr", 2)]
    assert len(listings) == 2
    assert waits == [1]


@pytest.mark.parametrize(
    "content, paths",
    [
        ("```foo``` is a span; the data is in `out.zarr`.", ["out.zarr"]),
        ("```py\n[data](out.zarr) and `k.zarr`\n```\nNothing else.", []),
        ("~~~\n![map](maps/rain.png)\n~~~\nSee [data](out.zarr).", ["out.zarr"]),
    ],
    ids=["inline-triple-backticks", "link-in-backtick-fence", "link-in-tilde-fence"],
)
def test_fenced_code_presents_nothing_and_an_inline_span_opens_no_fence(content, paths):
    assert reply_files(content, 1).paths == paths


PROVENANCE = {"branches": [{"label": None, "steps": [], "crumbs": []}]}


def test_a_path_named_only_in_backticks_never_blocks_the_judge():
    entries = [
        {"path": "out.zarr", "kind": "zarr", "is_dir": True, "provenance_detail": PROVENANCE},
        # Named in passing: an input store without provenance, and a map that cannot be read.
        {"path": "era5_kenya.zarr", "kind": "zarr", "is_dir": True},
        {"path": "maps/input.png", "kind": "image", "provenance_detail": PROVENANCE},
    ]

    def handler(request):
        if request.url.path.endswith("/artifacts"):
            return httpx.Response(200, json=entries)
        return httpx.Response(404, json={"detail": "Not found"})

    replies = presenting(
        "out.zarr",
        "era5_kenya.zarr",
        "maps/input.png",
        candidates=frozenset({"era5_kenya.zarr", "maps/input.png"}),
    )
    evidence = asyncio.run(collect_evidence(client(handler), "c1", replies, ["zarr"]))
    assert evidence.problem is None
    notes = {f.path: f.note for f in evidence.files}
    assert notes["out.zarr"] is None
    assert notes["era5_kenya.zarr"] == "it has no provenance"
    assert notes["maps/input.png"].startswith("it could not be read:")


def test_the_same_path_named_by_a_link_in_another_reply_still_blocks():
    entries = [{"path": "out.zarr", "kind": "zarr", "is_dir": True}]
    app = client(lambda request: httpx.Response(200, json=entries))
    replies = [
        ReplyFiles(1, ["out.zarr"]),
        ReplyFiles(2, ["out.zarr"], frozenset({"out.zarr"})),
    ]
    evidence = asyncio.run(collect_evidence(app, "c1", replies, ["zarr"], every=True))
    assert evidence.problem == "Presented files have no provenance: ['out.zarr']"


@pytest.mark.parametrize(
    "entry",
    [
        {"path": "maps", "kind": "directory", "is_dir": True},
        {"path": "maps", "kind": "file", "is_dir": True},
    ],
    ids=["directory-kind", "is-dir"],
)
def test_a_listed_folder_named_in_backticks_presents_nothing(entry):
    app = client(lambda request: httpx.Response(200, json=[entry]))
    evidence = asyncio.run(
        collect_evidence(app, "c1", presenting("maps", candidates=frozenset({"maps"})), [])
    )
    assert evidence.problem == "No result presented: no reply presented a file"


def test_only_kinds_the_app_reads_provenance_from_need_it():
    entries = [
        {"path": "maps/rain.jpg", "kind": "image"},
        {"path": "notes.txt", "kind": "file"},
    ]

    def handler(request):
        if request.url.path.endswith("/artifacts"):
            return httpx.Response(200, json=entries)
        return httpx.Response(200, content=b"jpg")

    evidence = asyncio.run(
        collect_evidence(client(handler), "c1", presenting("maps/rain.jpg", "notes.txt"), [])
    )
    assert evidence.problem is None
    assert [f.note for f in evidence.files] == ["it has no provenance"] * 2
    for path, kind in (("maps/rain.png", "image"), ("map.html", "html"), ("o.zarr", "zarr")):
        listing = [{"path": path, "kind": kind}]
        app = client(lambda request, listing=listing: httpx.Response(200, json=listing))
        evidence = asyncio.run(collect_evidence(app, "c1", presenting(path), []))
        assert evidence.problem == f"Presented files have no provenance: ['{path}']"


def test_only_unlisted_candidates_is_no_result_presented():
    app = client(lambda request: httpx.Response(200, json=[{"path": "out/rain.zarr"}]))
    replies = presenting("rain.zarr", candidates=frozenset({"rain.zarr"}))
    evidence = asyncio.run(collect_evidence(app, "c1", replies, ["zarr"], every=True))
    assert evidence.problem == "No result presented: no reply presented a file"
    assert evidence.unlisted == []
    # A later listing may have it, so the evidence is read again.
    assert evidence.unlisted_candidates == ["rain.zarr"]


def test_a_failed_extra_listing_falls_back_to_the_earlier_listing():
    listings = []
    entry = {
        "path": "maps/rain.png",
        "kind": "image",
        "provenance_detail": {"branches": [{"label": None, "steps": [], "crumbs": []}]},
    }

    def handler(request):
        if request.url.path.endswith("/artifacts"):
            listings.append(request)
            if len(listings) == 1:
                return httpx.Response(200, json=[entry])
            return httpx.Response(500, json={"detail": "busy"})
        return httpx.Response(200, content=b"png")

    waits = []

    async def wait():
        waits.append(1)

    evidence = asyncio.run(
        collect_evidence(
            client(handler),
            "c1",
            presenting("maps/rain.png", "maps/gone.png"),
            ["image"],
            retries=1,
            wait=wait,
        )
    )

    # The lagging path is reported from the first listing; the other file keeps
    # its provenance and image.
    assert evidence.problem.startswith(
        "Presented files are not in the chat's files: ['maps/gone.png']; "
        "the listing read again failed: "
    )
    assert "500" in evidence.problem
    assert evidence.files[0].provenance == entry["provenance_detail"]
    assert evidence.files[0].image == b"png"
    assert len(listings) == 2
    assert waits == [1]


def test_presented_file_missing_from_the_listing_is_named():
    entries = [{"path": None, "kind": "image"}, {"kind": "image"}, "junk"]
    app = client(lambda request: httpx.Response(200, json=entries))
    evidence = asyncio.run(collect_evidence(app, "c1", presenting("maps/rain.png"), ["image"]))
    assert evidence.problem == "Presented files are not in the chat's files: ['maps/rain.png']"


@pytest.mark.parametrize("detail", [None, ["arco-era5-fetch"], "{}", {}, {"branches": []}])
def test_provenance_detail_must_be_an_object(detail):
    entry = {"path": "out.zarr", "kind": "zarr"}
    if detail is not None:
        entry["provenance_detail"] = detail
    app = client(lambda request: httpx.Response(200, json=[entry]))
    evidence = asyncio.run(collect_evidence(app, "c1", presenting("out.zarr"), ["zarr"]))
    assert evidence.problem == "Presented files have no provenance: ['out.zarr']"


@pytest.mark.parametrize(
    "response, healthy",
    [
        (httpx.Response(200, json={"status": True}), True),
        (httpx.Response(200, json=[True]), False),
        (httpx.Response(200, text="ok"), False),
        (httpx.Response(503, json={"status": True}), False),
    ],
)
def test_health_check_needs_a_status_object(monkeypatch, response, healthy):
    monkeypatch.setattr(instance_module.httpx, "get", lambda url, timeout: response)
    assert instance_module._health_ok("http://app/health") is healthy


def test_repository_of_image_references():
    assert repository_of("ghcr.io/org/app:1.2") == "ghcr.io/org/app"
    assert repository_of("ghcr.io/org/app@sha256:abc") == "ghcr.io/org/app"
    assert repository_of("localhost:5000/app") == "localhost:5000/app"
    assert repository_of("app") == "app"


# Judge


def quality_reply(score):
    return litellm.ModelResponse(
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"score": score, "reasoning": "r"}),
                }
            }
        ]
    )


@pytest.mark.parametrize("score", [True, False, 0, 6, "4"])
def test_answer_quality_rejects_non_integer_scores(score):
    with pytest.raises(ValueError):
        rate_answer(
            complete=lambda role, session, **kwargs: quality_reply(score),
            model="m",
            session_id="c1",
            opening="o",
            criteria=[],
            transcript="t",
        )


def test_quality_transcript_uses_neutral_labels():
    prompts = []

    def complete(role, session, **kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        return quality_reply(5)

    rate_answer(
        complete=complete,
        model="m",
        session_id="c1",
        opening="o",
        criteria=[],
        transcript=neutral_transcript([("rain?", "20 mm")]),
    )
    assert "User: rain?\n\nAssistant: 20 mm" in prompts[0]
    assert "You:" not in prompts[0]


class DoneAssistant(AgentAdapter):
    role = AgentRole.AGENT

    async def call(self, input):
        return "DONE: 20 mm next week"


class OneQuestionUser(AgentAdapter):
    role = AgentRole.USER

    def __init__(self):
        self.calls = 0

    async def call(self, input):
        self.calls += 1
        if self.calls == 1:
            return "rain next week?"
        return ScenarioResult(success=True, messages=list(input.messages), reasoning="done")


def test_final_judge_and_quality_rating_send_8192_output_tokens(monkeypatch):
    sent = []
    models = FakeModels()

    def completion(**kwargs):
        sent.append(kwargs)
        return models("judge", None, **kwargs)

    monkeypatch.setattr(litellm, "completion", completion)
    gateway = LangfuseGateway(
        environment="eval",
        trace_timeout=2,
        trace_settle=1,
        poll_interval=1,
        model_max_tokens=load_settings(settings_env()).model_max_tokens,
        client=Langfuse(
            public_key=f"pk-test-{uuid.uuid4()}",
            secret_key="sk-test",
            base_url="http://langfuse.invalid",
            span_exporter=InMemorySpanExporter(),
            tracer_provider=TracerProvider(),
        ),
    )
    judge = GoalJudge(
        criteria=["The assistant gives the rain total"],
        deliverables=[],
        model="m",
        complete=gateway.traced_completion,
        session_id=lambda: "c1",
        collect_evidence=lambda: None,
    )
    states = []

    async def play_then_judge():
        await scenario.arun(
            name="t",
            description="d",
            agents=[DoneAssistant(), OneQuestionUser()],
            script=[scenario.proceed(on_turn=states.append)],
            max_turns=3,
            verbose=False,
        )
        return await judge.rule(states[-1], at_turn_cap=False)

    assert asyncio.run(play_then_judge()) is True
    rate_answer(
        complete=gateway.traced_completion,
        model="m",
        session_id="c1",
        opening="o",
        criteria=[],
        transcript="t",
    )
    assert [call["max_tokens"] for call in sent] == [8192, 8192]
    assert sent[0]["tools"] and "tools" not in sent[1]


def test_simulated_user_call_sends_8192_output_tokens(monkeypatch):
    sent = []

    def completion(**kwargs):
        sent.append(kwargs)
        reply = {
            "classification": "progress",
            "revealed_hidden_step": False,
            "done": False,
            "give_up": False,
            "give_up_reason": "",
            "message": "go on",
        }
        return litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": json.dumps(reply)}}]
        )

    monkeypatch.setattr(litellm, "completion", completion)
    gateway = LangfuseGateway(
        environment="eval",
        trace_timeout=2,
        trace_settle=1,
        poll_interval=1,
        model_max_tokens=load_settings(settings_env()).model_max_tokens,
        client=Langfuse(
            public_key=f"pk-test-{uuid.uuid4()}",
            secret_key="sk-test",
            base_url="http://langfuse.invalid",
            span_exporter=InMemorySpanExporter(),
            tracer_provider=TracerProvider(),
        ),
    )

    async def no_files():
        return Evidence()

    user = SimulatedUser(
        task=parse_task(
            {
                "id": "i1",
                "input": "rain next week?",
                "expected_output": {"judge_criteria": ["c"]},
                "metadata": {"facts": "Kenya"},
            }
        ),
        model="m",
        complete=gateway.traced_completion,
        session_id=lambda: "c1",
        collect_evidence=no_files,
    )
    messages = [
        {"role": "user", "content": "rain next week?"},
        {"role": "assistant", "content": "Which region?"},
    ]
    agent_input = AgentInput(thread_id="t", messages=messages, new_messages=[], scenario_state=None)

    assert asyncio.run(user.call(agent_input)) == "rain next week?"
    assert asyncio.run(user.call(agent_input)) == "go on"
    assert [call["max_tokens"] for call in sent] == [8192]


def test_final_judge_reads_the_run_that_arun_leaves_current():
    judge = GoalJudge(
        criteria=["The assistant gives the rain total"],
        deliverables=[],
        model="m",
        complete=FakeModels(),
        session_id=lambda: "c1",
        collect_evidence=lambda: None,
    )
    states = []

    async def play_then_judge():
        await scenario.arun(
            name="t",
            description="d",
            agents=[DoneAssistant(), OneQuestionUser()],
            script=[scenario.proceed(on_turn=states.append)],
            max_turns=3,
            verbose=False,
        )
        return await judge.rule(states[-1], at_turn_cap=False)

    assert asyncio.run(play_then_judge()) is True
    assert judge.verdict == "verdict passed"
    # Outside the context arun ran in, Scenario's judge finds no run.
    with pytest.raises(LookupError):
        asyncio.run(judge.rule(states[-1], at_turn_cap=False))


# CLI


@pytest.mark.parametrize(
    "error, code, message",
    [
        (SettingsError("EVAL_JUDGE_MODEL is not set"), 2, "Run not started"),
        (ConfigError("image", "is missing"), 2, "Run not started"),
        (TaskError("dataset item t1: bad"), 2, "Run not started"),
        (DatasetError("Dataset d has no items"), 2, "Run not started"),
        (AppStartError("container exited"), 3, "the app did not start"),
        (RunnerStartError("runner exited"), 3, "the sandbox runner did not start"),
        (docker.errors.APIError("daemon down"), 3, "Docker failed"),
        (ChatError("GET /api/v1/tool_summary returned 500"), 4, "the app failed"),
        (httpx.ConnectError("refused"), 4, "the app failed"),
        (LangfuseApiError(status_code=500, body="boom"), 5, "Langfuse returned an error"),
        (RuntimeError("unexpected"), 1, "Run failed"),
    ],
)
def test_cli_exit_codes_and_messages(monkeypatch, capsys, error, code, message):
    def failing_run(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "run", failing_run)
    assert cli.main(["run", "configs/example.yaml"]) == code
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "runner_lost, lost_line",
    [
        ([], None),
        (["t2"], "1 task not scored because the sandbox runner restarted during it: t2"),
        (
            ["t1", "t3"],
            "2 tasks not scored because the sandbox runner restarted during them: t1, t3",
        ),
    ],
    ids=["none-lost", "one-lost", "two-lost"],
)
def test_cli_summary_names_tasks_lost_to_a_runner_restart(
    monkeypatch, capsys, runner_lost, lost_line
):
    summary = RunSummary(
        experiment_name="cfg",
        run_name="cfg - run",
        experiment_url=None,
        items=3,
        goals_reached=1,
        metadata={},
        runner_lost=runner_lost,
    )
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: summary)
    assert cli.main(["run", "configs/example.yaml"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[:2] == ["Experiment cfg, run cfg - run", "Goal reached on 1 of 3 tasks"]
    assert lines[2:] == ([lost_line] if lost_line else [])


@pytest.mark.parametrize(
    "name, value", [("EVAL_MAX_TURNS", "0"), ("EVAL_MODEL_MAX_TOKENS", "lots")]
)
def test_cli_with_a_bad_turn_cap_or_token_limit_exits_2(monkeypatch, capsys, name, value):
    for key, setting in settings_env(**{name: value}).items():
        monkeypatch.setenv(key, setting)
    assert cli.main(["run", str(REPO / "configs" / "example.yaml")]) == 2
    assert name in capsys.readouterr().err


def test_cli_without_settings_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    assert cli.main(["run", str(REPO / "configs" / "example.yaml")]) == 2
    assert "LANGFUSE_PUBLIC_KEY is not set" in capsys.readouterr().err


def test_cli_with_an_unset_image_reference_exits_2(monkeypatch, capsys):
    for key, setting in settings_env().items():
        monkeypatch.setenv(key, setting)
    for name in (
        "APP_WEBUI_SECRET_KEY",
        "APP_ADMIN_PASSWORD",
        "APP_ADMIN_API_KEY",
        "APP_OPENROUTER_API_KEY",
        "LANGFUSE_BASE_URL",
    ):
        monkeypatch.setenv(name, "value")
    monkeypatch.delenv("APP_IMAGE", raising=False)
    # configs/example.yaml sets image: ${APP_IMAGE}.
    assert cli.main(["run", str(REPO / "configs" / "example.yaml")]) == 2
    assert "image: references ${APP_IMAGE}, which is not set" in capsys.readouterr().err


def test_only_the_langwatch_notices_are_silenced(caplog):
    no_trace = (
        "No trace in context when calling langwatch.get_current_trace(), "
        "perhaps you forgot to use @langwatch.trace()?"
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cli.silence_langwatch()
        warnings.warn_explicit(
            no_trace, UserWarning, "context.py", 47, "langwatch.telemetry.context"
        )
        warnings.warn_explicit(
            "Failed to set LangWatch trace context: x",
            UserWarning,
            "context.py",
            107,
            "langwatch.telemetry.context",
        )
        warnings.warn_explicit(no_trace, UserWarning, "other.py", 1, "weather_skills_eval.other")
    assert [(str(w.message), w.filename) for w in caught] == [
        ("Failed to set LangWatch trace context: x", "context.py"),
        (no_trace, "other.py"),
    ]

    reporter = logging.getLogger("scenario._events.event_reporter")
    cli.silence_langwatch()
    # Calling it again adds no second filter.
    assert len(reporter.filters) == 1
    with caplog.at_level(logging.WARNING, logger="scenario._events.event_reporter"):
        reporter.warning(
            "No LangWatch api_key configured (set LANGWATCH_API_KEY env var or call "
            "langwatch.setup(api_key=...) before scenarios run); scenario events will not be reported."
        )
        reporter.warning("No LANGWATCH_ENDPOINT configured, skipping event posting")
    assert [r.getMessage() for r in caplog.records] == [
        "No LANGWATCH_ENDPOINT configured, skipping event posting"
    ]
    for f in list(reporter.filters):
        reporter.removeFilter(f)


def test_the_langwatch_banner_is_not_printed(monkeypatch, capsys):
    from scenario._events.event_alert_message_logger import EventAlertMessageLogger

    monkeypatch.delenv("SCENARIO_DISABLE_SIMULATION_REPORT_INFO", raising=False)
    monkeypatch.delenv("LANGWATCH_API_KEY", raising=False)
    monkeypatch.setattr(EventAlertMessageLogger, "_shown_batch_ids", set())
    cli.silence_langwatch()
    EventAlertMessageLogger().handle_greeting()
    assert "LangWatch API key not configured" not in capsys.readouterr().out


def test_sigterm_handler_unwinds_with_exit_code_143():
    with pytest.raises(SystemExit) as exit_info:
        cli._terminate(15, None)
    assert exit_info.value.code == 143


def test_cli_passes_dash_through_to_read_stdin(monkeypatch):
    seen = []

    def recording_run(path, environ, **kwargs):
        seen.append(path)
        raise SettingsError("stop here")

    monkeypatch.setattr(cli, "run", recording_run)
    cli.main(["run", "-"])
    assert str(seen[0]) == "-"


def run_api(**options) -> FakeRunApi:
    """Experiment e1 (cfg): one item with a chat, one without."""
    defaults = {
        "experiments": {"e1": "cfg", "e2": "empty"},
        "items": {"e1": [("item-1", {"chat_id": "chat-1"}), ("item-2", {"chat_id": None})]},
        "sessions": {"chat-1": ["item-1", "app-a", "user-1", "judge-1"]},
    }
    defaults.update(options)
    return FakeRunApi(**defaults)


PROMPT = "Type yes to delete these traces: "


def delete_with(api: FakeRunApi, *args: str) -> int:
    return cli.main(["delete", *args], runs_factory=lambda: LangfuseRuns(client=api))


def answer(monkeypatch, text: str) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(text))


def test_cli_delete_deletes_every_trace_of_the_run_on_yes(monkeypatch, capsys):
    # Only the Langfuse settings are needed, not the run's.
    for name in ("OPENROUTER_API_KEY", "EVAL_SIMULATED_USER_MODEL", "EVAL_JUDGE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    answer(monkeypatch, "yes\n")
    api = run_api()

    assert delete_with(api, "e1") == 0

    # The item traces go last.
    assert api.deleted == ["app-a", "judge-1", "user-1", "item-1", "item-2"]
    assert capsys.readouterr().out == (
        f"Experiment cfg: 2 items, 5 traces\n{PROMPT}Deleted experiment cfg: 2 items, 5 traces\n"
    )


def test_cli_delete_with_yes_flag_deletes_without_asking(monkeypatch, capsys):
    # Nothing is read from stdin.
    monkeypatch.setattr("sys.stdin", None)
    api = run_api()

    assert delete_with(api, "--yes", "e1") == 0

    # The item traces go last.
    assert api.deleted == ["app-a", "judge-1", "user-1", "item-1", "item-2"]
    assert capsys.readouterr().out == (
        "Experiment cfg: 2 items, 5 traces\nDeleted experiment cfg: 2 items, 5 traces\n"
    )


@pytest.mark.parametrize("text", ["no\n", "y\n", "YES\n", "\n", ""], ids=repr)
def test_cli_delete_without_yes_deletes_nothing(monkeypatch, capsys, text):
    answer(monkeypatch, text)
    api = run_api()

    assert delete_with(api, "e1") == 0

    assert "trace.delete_multiple" not in api.methods()
    assert capsys.readouterr().out == (
        f"Experiment cfg: 2 items, 5 traces\n{PROMPT}Nothing deleted\n"
    )


def test_cli_delete_of_an_unknown_id_exits_2(capsys):
    api = run_api()

    assert delete_with(api, "missing") == 2

    assert "trace.delete_multiple" not in api.methods()
    assert "no experiment has id missing" in capsys.readouterr().err


def test_cli_delete_of_an_experiment_without_items_exits_0(capsys):
    api = run_api()

    assert delete_with(api, "e2") == 0

    assert "trace.delete_multiple" not in api.methods()
    assert capsys.readouterr().out == "Experiment empty has no items; nothing to delete\n"


def test_cli_delete_with_a_langfuse_error_exits_5(capsys):
    api = run_api(error=LangfuseApiError(status_code=503, body="unavailable"))

    assert delete_with(api, "--yes", "e1") == 5

    assert api.deleted == []
    assert "Delete stopped, Langfuse returned an error" in capsys.readouterr().err


def test_cli_delete_with_a_langfuse_error_on_a_later_batch_exits_5(capsys):
    chat = [f"app-{n:04d}" for n in range(1500)]
    api = run_api(
        sessions={"chat-1": chat},
        delete_errors={2: LangfuseApiError(status_code=503, body="unavailable")},
    )

    assert delete_with(api, "--yes", "e1") == 5

    # The first batch stays deleted; the item traces, sent last, are left.
    assert api.deleted == chat[:1000]
    error = capsys.readouterr().err
    assert "Delete stopped, Langfuse returned an error" in error
    assert "run delete again to delete the rest" in error
    # So a second delete on the same project still finds the run and deletes the rest.
    assert delete_with(api, "--yes", "e1") == 0
    assert api.deleted == [*chat, "item-1", "item-2"]
    assert capsys.readouterr().out.endswith("Deleted experiment cfg: 2 items, 502 traces\n")
    assert delete_with(api, "--yes", "e1") == 2


def test_cli_delete_with_another_error_while_deleting_says_to_run_it_again(capsys):
    # The second request, of the item traces, fails after the first deleted the rest.
    api = run_api(delete_errors={2: httpx.ConnectError("refused")})

    assert delete_with(api, "--yes", "e1") == 1

    assert api.deleted == ["app-a", "judge-1", "user-1"]
    assert capsys.readouterr().err == (
        "Delete failed: ConnectError: refused; run delete again to delete the rest\n"
    )


def test_cli_delete_with_another_error_before_deleting_gives_no_hint(capsys):
    api = run_api(error=httpx.ConnectError("refused"))

    assert delete_with(api, "--yes", "e1") == 1

    assert api.methods() == {"experiments.list"}
    assert capsys.readouterr().err == "Delete failed: ConnectError: refused\n"


def test_cli_delete_with_any_other_error_exits_1(capsys):
    def broken():
        raise RuntimeError("unexpected")

    assert cli.main(["delete", "e1"], runs_factory=broken) == 1
    assert "Delete failed: RuntimeError: unexpected" in capsys.readouterr().err


def test_load_config_from_stdin_and_from_a_path(tmp_path):
    text = (
        "name: from-stdin\nimage: img:tag\nrunner_image: runner:tag\n"
        "model: m\ndataset: d\nenv:\n"
        "  WEBUI_SECRET_KEY: ${S}\n  BOOTSTRAP_ADMIN_EMAIL: admin\n"
        "  BOOTSTRAP_ADMIN_PASSWORD: ${P}\n  BOOTSTRAP_ADMIN_API_KEY: ${K}\n"
    )
    assert load_config(Path("-"), ENVIRON, stdin=io.BytesIO(text.encode())).name == "from-stdin"
    path = tmp_path / "file.yaml"
    path.write_text(text.replace("name: from-stdin\n", ""))
    assert load_config(path, ENVIRON).name == "file"
    with pytest.raises(ConfigError, match="not UTF-8"):
        load_config(Path("-"), ENVIRON, stdin=io.BytesIO(b"\xff\xfe"))


class SharedDaemon:
    """One Docker daemon shared by several runs. Listing containers or volumes
    is not allowed: a run must only touch the objects it created."""

    def __init__(self):
        self.containers_by_name: dict[str, FakeContainer] = {}
        self.volumes_by_name: dict[str, FakeVolume] = {}
        self.labels: dict[str, dict] = {}
        image = types.SimpleNamespace(
            id="sha256:image",
            attrs={"RepoDigests": ["ghcr.io/org/app@sha256:abc", "ghcr.io/org/runner@sha256:def"]},
        )
        daemon = self

        class Images:
            def pull(self, ref):
                return image

        class Containers:
            def run(self, image_id, *, name, labels, **kwargs):
                container = FakeContainer(["running"], name=name)
                daemon.containers_by_name[name] = container
                daemon.labels[name] = labels
                return container

            def create(self, image_id, *, name, labels, **kwargs):
                if "-emulator-" in name:
                    source = FakeSourceContainer(image_id, app_image_files(), None)
                    daemon.containers_by_name[name] = source
                    daemon.labels[name] = labels
                    return source
                return self.run(image_id, name=name, labels=labels)

            def get(self, name):
                if name not in daemon.containers_by_name:
                    raise docker.errors.NotFound(name)
                return daemon.containers_by_name[name]

            def list(self, **kwargs):
                raise AssertionError("a run must not list other runs' containers")

        class Volumes:
            def create(self, name, labels):
                daemon.volumes_by_name[name] = FakeVolume(name, labels)
                daemon.labels[name] = labels
                return daemon.volumes_by_name[name]

            def get(self, name):
                if name not in daemon.volumes_by_name:
                    raise docker.errors.NotFound(name)
                return daemon.volumes_by_name[name]

            def list(self, **kwargs):
                raise AssertionError("a run must not list other runs' volumes")

        self.images = Images()
        self.containers = Containers()
        self.volumes = Volumes()

    def removed(self, instance: AppInstance) -> list[bool]:
        """Whether the run's emulator source, app, runner and volume were removed."""
        return [
            self.containers_by_name[instance.emulator_source_name].removed,
            self.containers_by_name[instance.name].removed,
            self.containers_by_name[instance.runner_name].removed,
            self.volumes_by_name[instance.volume_name].removed,
        ]


def instance_on(daemon) -> AppInstance:
    return AppInstance(
        docker_client=daemon,
        image="ghcr.io/org/app:1",
        runner_image="ghcr.io/org/runner:1",
        env={},
        network="n",
        port="8080",
        start_timeout=10,
        poll_interval=1,
        health_check=lambda url: True,
    )


def test_concurrent_runs_on_one_daemon_do_not_interfere():
    daemon = SharedDaemon()
    first, second = instance_on(daemon), instance_on(daemon)

    with first:
        with second:
            run_ids = {labels["weather-skills-eval.run"] for labels in daemon.labels.values()}
            assert run_ids == {first.run_id, second.run_id}
            # Each run's emulator source, app, runner and volume.
            assert len(daemon.labels) == 8
            # Each run's app is configured with its own runner and token.
            first_env, second_env = first.app_env, second.app_env
            assert first.runner_name in first_env["SKILL_K8S_APISERVER"]
            assert second.runner_name in second_env["SKILL_K8S_APISERVER"]
            assert first_env["SKILL_K8S_BIND_TOKEN"] != second_env["SKILL_K8S_BIND_TOKEN"]
        assert daemon.removed(second) == [True, True, True, True]
        # The emulator source goes once the emulator is read.
        assert daemon.removed(first) == [True, False, False, False]
    assert daemon.removed(first) == [True, True, True, True]


def test_sigterm_during_a_run_removes_its_containers_and_volume():
    daemon = SharedDaemon()
    instance = instance_on(daemon)

    with pytest.raises(SystemExit):
        with instance:
            cli._terminate(15, None)

    assert daemon.removed(instance) == [True, True, True, True]
