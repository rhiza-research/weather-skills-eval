import asyncio
import io
import json
import types
from pathlib import Path

import docker.errors
import httpx
import litellm
import pytest
from fakes import FakeContainer
from langfuse.api.core.api_error import ApiError as LangfuseApiError

from weather_skills_eval import cli
from weather_skills_eval import instance as instance_module
from weather_skills_eval.app_client import AppClient, ChatError
from weather_skills_eval.checks import collect_evidence
from weather_skills_eval.config import ConfigError, load_config, parse_config
from weather_skills_eval.instance import AppInstance, AppStartError, repository_of
from weather_skills_eval.judge import rate_answer
from weather_skills_eval.play import neutral_transcript
from weather_skills_eval.runner import DatasetError
from weather_skills_eval.settings import SettingsError, load_settings
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


def test_non_utf8_config_file_is_a_config_error(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_bytes(b"image: \xff\xfe\n")
    with pytest.raises(ConfigError, match="not UTF-8"):
        load_config(path, ENVIRON)


def test_example_config_is_valid():
    environ = {
        name: "value"
        for name in (
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
    assert json.loads(config.env["OPENAI_API_CONFIGS"]) == {
        "0": {"model_call_tracing": "openrouter"}
    }


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
    evidence = asyncio.run(collect_evidence(app, "c1", ["maps/rain.png"], ["image"]))
    assert "could not be listed" in evidence.problem
    assert evidence.files == []


def test_presented_file_missing_from_the_listing_is_named():
    entries = [{"path": None, "kind": "image"}, {"kind": "image"}, "junk"]
    app = client(lambda request: httpx.Response(200, json=entries))
    evidence = asyncio.run(collect_evidence(app, "c1", ["maps/rain.png"], ["image"]))
    assert evidence.problem == "Presented files are not in the chat's files: ['maps/rain.png']"


@pytest.mark.parametrize("detail", [None, ["arco-era5-fetch"], "{}", {}, {"branches": []}])
def test_provenance_detail_must_be_an_object(detail):
    entry = {"path": "out.zarr", "kind": "zarr"}
    if detail is not None:
        entry["provenance_detail"] = detail
    app = client(lambda request: httpx.Response(200, json=[entry]))
    evidence = asyncio.run(collect_evidence(app, "c1", ["out.zarr"], ["zarr"]))
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


# CLI


@pytest.mark.parametrize(
    "error, code, message",
    [
        (SettingsError("EVAL_JUDGE_MODEL is not set"), 2, "Run not started"),
        (ConfigError("image", "is missing"), 2, "Run not started"),
        (TaskError("dataset item t1: bad"), 2, "Run not started"),
        (DatasetError("Dataset d has no items"), 2, "Run not started"),
        (AppStartError("container exited"), 3, "the app did not start"),
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


def test_cli_without_settings_exits_2(monkeypatch, capsys):
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    assert cli.main(["run", str(REPO / "configs" / "example.yaml")]) == 2
    assert "LANGFUSE_PUBLIC_KEY is not set" in capsys.readouterr().err


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


def test_load_config_from_stdin_and_from_a_path(tmp_path):
    text = (
        "name: from-stdin\nimage: img:tag\nmodel: m\ndataset: d\nenv:\n"
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
    """One Docker daemon shared by several runs. Listing containers is not
    allowed: a run must only touch the container it started."""

    def __init__(self):
        self.containers_by_name: dict[str, FakeContainer] = {}
        self.labels: dict[str, dict] = {}
        image = types.SimpleNamespace(
            id="sha256:image", attrs={"RepoDigests": ["ghcr.io/org/app@sha256:abc"]}
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

            def get(self, name):
                if name not in daemon.containers_by_name:
                    raise docker.errors.NotFound(name)
                return daemon.containers_by_name[name]

            def list(self, **kwargs):
                raise AssertionError("a run must not list other runs' containers")

        self.images = Images()
        self.containers = Containers()


def instance_on(daemon) -> AppInstance:
    return AppInstance(
        docker_client=daemon,
        image="ghcr.io/org/app:1",
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
            assert len(run_ids) == 2
        assert daemon.containers_by_name[second.name].removed
        assert not daemon.containers_by_name[first.name].removed
    assert daemon.containers_by_name[first.name].removed


def test_sigterm_during_a_run_removes_its_container():
    daemon = SharedDaemon()
    instance = instance_on(daemon)

    with pytest.raises(SystemExit):
        with instance:
            cli._terminate(15, None)

    assert daemon.containers_by_name[instance.name].removed
