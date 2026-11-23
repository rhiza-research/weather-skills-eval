"""The I/O and edge-case matrix, end to end through ``runner.run`` with a fake app,
fake Docker, fake Langfuse and fake harness models."""

import asyncio
import base64
import io
import json
import random
import tarfile
import textwrap
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import docker.errors
import httpx
import pytest
from fakes import (
    APP_EMULATOR_PATHS,
    APP_REPOSITORY,
    RUNNER_IMAGE,
    RUNNER_REPOSITORY,
    FakeApp,
    FakeClock,
    FakeContainer,
    FakeDocker,
    FakeGateway,
    FakeModels,
    KeepOrder,
    Reply,
    app_observations,
    broadcast_observations,
    chain,
    display_image,
    display_page,
    emulator_file,
    html_entry,
    image_bytes,
    image_entry,
    local_image,
    provenance_detail,
    tool_call,
    tool_choice_refusal,
    zarr_entry,
)

from weather_skills_eval import cli
from weather_skills_eval import play as play_module
from weather_skills_eval import runner as runner_module
from weather_skills_eval import simulated_user as simulated_user_module
from weather_skills_eval.app_client import ChatError
from weather_skills_eval.checks import NOTHING_PRESENTED, Evidence, PresentedFile
from weather_skills_eval.config import RUNNER_ENV, ConfigError
from weather_skills_eval.configs_repo import (
    CatalogMismatchError,
    ConfigsRepoError,
    SkillStateError,
)
from weather_skills_eval.instance import (
    APP_DATA_DIR,
    ARTIFACTS_SUBPATH,
    BATCH_LABEL,
    EMULATOR_API_PORT,
    EMULATOR_BIND_PORT,
    EMULATOR_DIR,
    EMULATOR_LABEL_VALUE,
    LABEL,
    RUN_LABEL,
    RUNNER_ARTIFACTS_DIR,
    RUNNER_LABEL_VALUE,
    START_DELAY,
    TOKEN_BYTES,
    VOLUME_LABEL_VALUE,
    VOLUME_SETUP_LABEL_VALUE,
    AppInstance,
    AppStartError,
    RunnerStartError,
)
from weather_skills_eval.judge import (
    EVIDENCE_HEADER,
    PROVENANCE_CRITERION,
    evidence_message,
    started_line,
)
from weather_skills_eval.langfuse_gateway import TraceReadError
from weather_skills_eval.llm import ModelReplyError
from weather_skills_eval.runner import DatasetError, run
from weather_skills_eval.settings import (
    DEFAULT_APP_START_TIMEOUT_SECONDS,
    DEFAULT_MAX_TURNS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    SettingsError,
)
from weather_skills_eval.simulated_user import REPLY_REQUEST, files_message
from weather_skills_eval.tasks import TaskError

ENVIRON = {
    "LANGFUSE_PUBLIC_KEY": "pk-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
    "OPENROUTER_API_KEY": "or-test",
    "LANGFUSE_TRACING_ENVIRONMENT": "eval",
    "EVAL_SIMULATED_USER_MODEL": "openrouter/test/user",
    "EVAL_JUDGE_MODELS": "openrouter/test/judge",
    "EVAL_DOCKER_NETWORK": "weather-skills-eval",
    "EVAL_TRACE_SETTLE_SECONDS": "1",
    "APP_SECRET": "webui-secret",
    "APP_ADMIN_PASSWORD": "admin-password",
    "APP_API_KEY": "sk-test-key",
    "APP_OPENROUTER_KEY": "or-app",
    "EVAL_CONFIGS_REPO": "https://github.com/rhiza-research/weather-skills-configs.git",
    "EVAL_CONFIGS_COMMIT": "4d174dae" + "0" * 32,
    "EVAL_CONNECTION_KEYS": "https://openrouter.ai/api/v1=APP_OPENROUTER_KEY",
}

CONFIG = textwrap.dedent(
    """\
    image: ghcr.io/rhiza-research/weather-skills-chat:test
    runner_image: ghcr.io/rhiza-research/weather-skills-sandbox-runner:src-171514bbb4e1
    dataset: weather-tasks
    env:
      WEBUI_SECRET_KEY: ${APP_SECRET}
      BOOTSTRAP_ADMIN_EMAIL: eval@example.invalid
      BOOTSTRAP_ADMIN_PASSWORD: ${APP_ADMIN_PASSWORD}
      BOOTSTRAP_ADMIN_API_KEY: ${APP_API_KEY}
    """
)

MODEL = "weather-agent"
PACK_URL = "https://example.invalid/pack.git"
PACK_REF = "a" * 40
CHC_URL = "https://github.com/rhiza-research/chc-skills.git"
OPENROUTER = "https://openrouter.ai/api/v1"
SCHEMAS = ("model", "skill-packs", "connections", "settings")


def pack(git_url=PACK_URL, *, enabled_by_default=True, is_active=True, skills=()):
    """A skill-packs.yaml entry."""
    return {
        "git_url": git_url,
        "ref": PACK_REF,
        "enabled_by_default": enabled_by_default,
        "is_active": is_active,
        "skills": list(skills),
    }


def write_data(path: Path, key: str, value, version=1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": version, key: value}))


class ConfigsRepo:
    """A copy of the configs repo at one commit: one model by default, the
    pack https://example.invalid/pack.git, the OpenRouter connection."""

    def __init__(self, root: Path):
        self.root = root
        for name in SCHEMAS:
            path = root / "schemas" / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"x-schema-version": 1}))
        self.model(MODEL)
        self.packs([pack()])
        write_data(
            root / "connections.yaml", "connections", {OPENROUTER: {"prefix_id": "open-router-"}}
        )
        write_data(
            root / "settings.yaml", "settings", {"TASK_MODEL": "", "TASK_MODEL_EXTERNAL": ""}
        )

    def model(self, model_id, variant=None) -> None:
        entry = {
            "id": model_id,
            "base_model_id": "open-router-.anthropic/claude-haiku-5.5",
            "name": model_id,
            "meta": {},
            "params": {},
        }
        path = self.root / "models" / f"{model_id}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"schema_version": 1, "model": entry, "eval": {"variant": variant}})
        )

    def packs(self, packs) -> None:
        write_data(self.root / "skill-packs.yaml", "skill_packs", packs)


def task(
    item_id, opening, *, turn_limit=None, criteria=None, deliverables=None, hidden=None, pauses=None
):
    """A dataset item; without ``turn_limit``, the play runs to ``EVAL_MAX_TURNS``."""
    item = {
        "id": item_id,
        "input": opening,
        "expected_output": {
            "judge_criteria": criteria
            if criteria is not None
            else ["The assistant gives the rain forecast"],
            "deliverables": deliverables or [],
        },
        "metadata": {
            "facts": "Region: Kenya. Period: next week.",
            "hidden_steps": hidden or [],
            "pauses": pauses or [],
            "tags": ["test"],
        },
    }
    if turn_limit is not None:
        item["metadata"]["turn_limit"] = turn_limit
    return item


PLAY_START = datetime(2026, 10, 22, 9, 30, 15, tzinfo=UTC)


def progress(message="go on", classification="progress", hint=False):
    return {
        "classification": classification,
        "revealed_hidden_step": hint,
        "done": False,
        "give_up": False,
        "give_up_reason": "",
        "message": message,
    }


class Harness:
    def __init__(
        self,
        tmp_path,
        items,
        reply,
        user_replies=None,
        container=None,
        judge_fails_on=None,
        judge_without_verdict=False,
        quality_error=None,
        verdict_error=None,
        verdict_error_for=None,
        failing_judges=None,
        quality_scores=None,
        refuses_forced_tool=None,
        **docker_options,
    ):
        self.clock = FakeClock()
        self.app = FakeApp(reply, self.clock)
        self.models = FakeModels(
            user_replies,
            judge_fails_on=judge_fails_on,
            judge_without_verdict=judge_without_verdict,
            quality_error=quality_error,
            verdict_error=verdict_error,
            verdict_error_for=verdict_error_for,
            failing_judges=failing_judges,
            quality_scores=quality_scores,
            refuses_forced_tool=refuses_forced_tool,
        )
        self.gateway = FakeGateway(items, self.models)
        self.container = container or FakeContainer(["running"])
        self.docker = FakeDocker(self.container, **docker_options)
        self.config_path = tmp_path / "production-agent.yaml"
        self.config_path.write_text(CONFIG)
        self.repo = ConfigsRepo(tmp_path / "configs-repo")
        self.model = MODEL
        # The app serves its catalog and packs from the env it was started with.
        self.app.app_env = lambda: self.docker.run_calls[-1]["environment"]
        self.health_checks = 0

    def health(self, url):
        self.health_checks += 1
        return True

    def run(self, health=None, config_path=None, stdin=None, rng=None, keep_order=True, **settings):
        """``settings`` are harness settings to add to ENVIRON. The tasks are
        played in the dataset's order unless ``rng`` is given, or with
        ``keep_order`` false, in the order of run's own random source."""
        return run(
            config_path or self.config_path,
            {
                **ENVIRON,
                "EVAL_CONFIGS_DIR": str(self.repo.root),
                "EVAL_MODEL": self.model,
                **settings,
            },
            stdin=stdin,
            docker_client_factory=lambda: self.docker,
            gateway_factory=lambda settings: self.gateway,
            app_transport=self.app.transport,
            instance_options={
                "health_check": health or self.health,
                "clock": self.clock,
                "sleep": self.clock.sleep_sync,
            },
            play_options={
                "clock": self.clock,
                "sleep": self.clock.sleep,
                "now": lambda: PLAY_START,
            },
            clock=self.clock,
            rng=rng or (KeepOrder() if keep_order else None),
        )

    def scores(self, index=0):
        item = self.gateway.experiments[0]["items"][index]
        return {e.name: e.value for e in item.evaluations}

    def output(self, index=0):
        return self.gateway.experiments[0]["items"][index].output


def reply_done_on(turn):
    def reply(opening, number, text):
        if number >= turn:
            return Reply("DONE: rain totals for Kenya next week are 20-40 mm.")
        return Reply("Which region do you need?")

    return reply


def test_run_plays_every_task_into_one_experiment_and_removes_the_container(tmp_path):
    items = [
        task("t1", "rain forecast kenya"),
        task("t2", "rain map kenya"),
        task("t3", "weekly rain"),
    ]
    harness = Harness(tmp_path, items, reply_done_on(1))

    summary = harness.run()

    assert len(harness.docker.run_calls) == 1
    call = harness.docker.run_calls[0]
    assert call["network"] == "weather-skills-eval"
    assert call["environment"]["BOOTSTRAP_ADMIN_API_KEY"] == "sk-test-key"
    assert json.loads(call["environment"]["SKILL_PACKS"]) == [
        {"git_url": PACK_URL, "ref": PACK_REF, "enabled_by_default": True}
    ]
    assert harness.container.removed
    assert harness.docker.runner.removed
    assert harness.docker.volume().removed
    assert len(harness.gateway.experiments) == 1
    experiment = harness.gateway.experiments[0]
    # Named after the model OpenRouter answers the base model with.
    assert harness.gateway.served_requests == ["anthropic/claude-haiku-5.5"]
    assert experiment["name"] == "anthropic/claude-haiku-5.5"
    assert experiment["metadata"]["served_model"] == "anthropic/claude-haiku-5.5"
    # A run's config name is its model id.
    assert experiment["metadata"]["config_name"] == MODEL
    assert summary.experiment_name == "anthropic/claude-haiku-5.5"
    assert len(experiment["items"]) == 3
    for index in range(3):
        assert harness.scores(index)["goal_reached"] == 1.0
    metadata = experiment["metadata"]
    # The configured images are pulled, and the digest of each one's repository recorded.
    assert harness.docker.pulled == [
        "ghcr.io/rhiza-research/weather-skills-chat:test",
        RUNNER_IMAGE,
    ]
    assert harness.docker.looked_up == []
    assert call["image"] == "sha256:image"
    assert metadata["image_digest"] == f"{APP_REPOSITORY}@sha256:abc"
    assert metadata["image_id"] == "sha256:image"
    assert metadata["image_pulled"] == "true"
    assert metadata["runner_image"] == RUNNER_IMAGE
    assert metadata["runner_image_digest"] == f"{RUNNER_REPOSITORY}@sha256:run"
    assert metadata["runner_image_id"] == "sha256:runner"
    assert "OPENAI_API_KEYS" in metadata["env_var_names"].split(",")
    assert "1234567" in metadata["skill_packs"]
    assert metadata["tool_ids"] == "plot-2,fetch"
    # The catalog model's base model, without the connection prefix.
    assert harness.app.catalog_model_requests == ["weather-agent"]
    assert metadata["model"] == "anthropic/claude-haiku-5.5"
    assert metadata["catalog_model"] == "weather-agent"
    assert experiment["description"] == "anthropic/claude-haiku-5.5"
    assert harness.output()["provider_models"] == ["provider/model-a", "provider/model-b"]
    assert metadata["harness_commit"] == runner_module.UNRECORDED
    # Without EVAL_BATCH the run belongs to no batch.
    assert metadata["batch"] == ""
    assert metadata["dataset"] == "weather-tasks"
    assert metadata["dataset_version"] == runner_module.LATEST
    assert metadata["dataset_items"] == "3"
    assert metadata["dataset_skill_packs"] == ""
    # The models that graded the run, and the hash of what grades it.
    assert metadata["simulated_user_model"] == "openrouter/test/user"
    assert metadata["judge_models"] == "openrouter/test/judge"
    assert "judge_model" not in metadata
    assert len(metadata["comparability_hash"]) == runner_module.COMPARABILITY_HASH_LENGTH
    for index in range(3):
        assert harness.output(index)["judges"] == [
            {
                "model": "openrouter/test/judge",
                "state": "passed",
                "passed": True,
                "reasoning": "verdict passed",
            }
        ]
        # One judge cannot disagree with itself.
        assert "judges_agree" not in harness.scores(index)
        assert harness.scores(index)["answer_quality:test/judge"] == 4
    assert all(isinstance(v, str) for v in metadata.values())
    assert summary.items == 3 and summary.goals_reached == 3
    assert summary.runner_lost == []
    # The items' cost_to_goal_usd summed: one turn of app_observations each, 0.02 + 0.19.
    assert summary.agent_cost_usd == pytest.approx(3 * 0.21)
    assert harness.gateway.linked == ["chat-1", "chat-2", "chat-3"]
    # Each item's rollup carries its own chat's app traces.
    assert [ids for _, ids in harness.gateway.app_costs] == [
        ["chat-1-trace-0"],
        ["chat-2-trace-0"],
        ["chat-3-trace-0"],
    ]
    assert harness.gateway.flushed
    for index in range(3):
        assert harness.output(index)["trace_scores"] == "complete"
    # Highest version per skill name, other tools kept.
    assert harness.app.completions[0]["tool_ids"] == ["plot-2", "fetch"]
    # The container carries its run id, and no other container is listed or touched.
    assert call["labels"][RUN_LABEL] in call["name"]
    assert harness.docker.list_filters == []


def test_the_batch_is_recorded_in_the_experiment_metadata(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))

    harness.run(
        EVAL_BATCH=" 2026-10-22T20:07:57Z ",
        EVAL_HARNESS_COMMIT="7f454f0",
        EVAL_APP_IMAGE_REVISION="8dd46ed4",
        EVAL_APP_IMAGE_PR_NUMBER="42",
    )

    metadata = harness.gateway.experiments[0]["metadata"]
    assert metadata["batch"] == "2026-10-22T20:07:57Z"
    assert metadata["harness_commit"] == "7f454f0"
    assert metadata["app_image_revision"] == "8dd46ed4"
    assert metadata["app_image_pr_number"] == "42"
    # Every container and the volume the run creates carry the batch id.
    docker_ = harness.docker
    labels = [docker_.source.labels, docker_.volume().labels]
    labels += [call["labels"] for call in docker_.setup_calls + docker_.create_calls]
    labels += [call["labels"] for call in docker_.run_calls]
    assert [label[BATCH_LABEL] for label in labels] == ["2026-10-22T20:07:57Z"] * 5


def test_an_image_id_is_never_pulled(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    image_id = "sha256:" + "a" * 64
    local = local_image([f"{APP_REPOSITORY}@sha256:abc"], image_id=image_id)
    harness = Harness(tmp_path, items, reply_done_on(1), local=local)
    harness.config_path.write_text(
        CONFIG.replace(APP_IMAGE_LINE, "image: ${APP_IMAGE}").replace(
            RUNNER_IMAGE, "${RUNNER_IMAGE}"
        )
    )

    # As a batch passes them; pull is true by default.
    harness.run(APP_IMAGE=image_id, RUNNER_IMAGE=image_id)

    assert harness.docker.pulled == []
    assert harness.docker.looked_up == [image_id, image_id]
    metadata = harness.gateway.experiments[0]["metadata"]
    assert metadata["image_id"] == metadata["runner_image_id"] == image_id
    assert metadata["image_digest"] == f"{APP_REPOSITORY}@sha256:abc"


def test_config_read_from_stdin(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))

    harness.run(config_path=Path("-"), stdin=io.BytesIO(CONFIG.encode()))

    assert harness.gateway.experiments[0]["metadata"]["config_name"] == MODEL
    assert harness.scores()["goal_reached"] == 1.0


def test_an_alias_names_the_experiment_after_the_model_it_points_to(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.app.base_model_id = "open-router-.~anthropic/claude-opus-latest"
    harness.gateway.served = "anthropic/claude-opus-5.5"

    summary = harness.run()

    assert harness.gateway.served_requests == ["~anthropic/claude-opus-latest"]
    experiment = harness.gateway.experiments[0]
    assert experiment["name"] == "anthropic/claude-opus-5.5"
    assert experiment["description"] == "~anthropic/claude-opus-latest"
    assert experiment["metadata"]["model"] == "~anthropic/claude-opus-latest"
    assert experiment["metadata"]["served_model"] == "anthropic/claude-opus-5.5"
    assert summary.experiment_name == "anthropic/claude-opus-5.5"


@pytest.mark.parametrize(
    "variant, name",
    [("reasoning_high", "z-ai/glm-5.3 (reasoning_high)"), (None, "z-ai/glm-5.3")],
    ids=["set", "null"],
)
def test_the_models_variant_is_appended_to_the_experiment_name(tmp_path, variant, name):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.served = "z-ai/glm-5.3"
    harness.repo.model(MODEL, variant=variant)

    summary = harness.run()

    experiment = harness.gateway.experiments[0]
    assert experiment["name"] == name
    assert summary.experiment_name == name
    # The metadata and the description keep the bare models, and add no key.
    assert experiment["metadata"]["served_model"] == "z-ai/glm-5.3"
    assert experiment["description"] == experiment["metadata"]["model"]
    assert "variant" not in experiment["metadata"]


def test_the_cli_prints_a_cost_line_with_the_bare_model(tmp_path, monkeypatch, capsys):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.served = "z-ai/glm-5.3"
    harness.repo.model(MODEL, variant="reasoning_high")
    monkeypatch.setattr(cli, "run", lambda *args, **kwargs: harness.run())

    assert cli.main(["run", str(harness.config_path)]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[-1].startswith(cli.COST_LINE)
    assert json.loads(lines[-1].removeprefix(cli.COST_LINE))["served_model"] == "z-ai/glm-5.3"


def test_a_failed_served_model_request_stops_the_run_before_the_experiment(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.served_error = ModelReplyError("OpenRouter named no model in its reply")

    with pytest.raises(ModelReplyError, match="named no model"):
        harness.run()

    assert harness.gateway.experiments == []
    assert harness.app.chats == {}
    assert harness.container.removed
    assert harness.docker.runner.removed
    assert harness.docker.volume().removed


def test_image_reference_is_pulled_and_recorded_resolved(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    literal = "image: ghcr.io/rhiza-research/weather-skills-chat:test"
    assert literal in CONFIG
    harness.config_path.write_text(CONFIG.replace(literal, "image: ${APP_IMAGE}"))
    image = f"{APP_REPOSITORY}:40be9ee"

    harness.run(APP_IMAGE=image)

    assert harness.docker.pulled == [image, RUNNER_IMAGE]
    assert harness.docker.run_calls[0]["image"] == "sha256:image"
    metadata = harness.gateway.experiments[0]["metadata"]
    assert metadata["image"] == image
    assert metadata["image_digest"] == f"{APP_REPOSITORY}@sha256:abc"


# The configs repo


def test_the_app_runs_as_the_configs_commit_defines_it(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.repo.model("other-agent")

    harness.run()

    env = harness.docker.run_calls[0]["environment"]
    # Every model at the commit, so references between them resolve in the app.
    assert sorted(m["id"] for m in json.loads(env["CATALOG_MODELS"])) == ["other-agent", MODEL]
    assert env["OPENAI_API_BASE_URLS"] == OPENROUTER
    assert json.loads(env["OPENAI_API_CONFIGS"]) == {"0": {"prefix_id": "open-router-"}}
    # The key of the variable EVAL_CONNECTION_KEYS names for the connection.
    assert env["OPENAI_API_KEYS"] == ENVIRON["APP_OPENROUTER_KEY"]
    assert env["TASK_MODEL"] == env["TASK_MODEL_EXTERNAL"] == ""
    metadata = harness.gateway.experiments[0]["metadata"]
    # HTTPS without .git, the SHA, and the model id.
    assert metadata["configs_repo"] == "https://github.com/rhiza-research/weather-skills-configs"
    assert metadata["configs_commit"] == ENVIRON["EVAL_CONFIGS_COMMIT"]
    assert metadata["catalog_model"] == MODEL


def test_configs_of_another_schema_version_do_not_start_the_run(tmp_path):
    for change in (
        lambda root: write_data(root / "settings.yaml", "settings", {}, version=2),
        lambda root: (root / "schemas" / "model.json").write_text('{"x-schema-version": 2}'),
    ):
        harness = Harness(tmp_path, [task("t1", "rain forecast kenya")], reply_done_on(1))
        change(harness.repo.root)

        with pytest.raises(ConfigsRepoError, match="version 2"):
            harness.run()

        assert harness.docker.pulled == []


def test_a_connection_without_a_key_variable_does_not_start_the_run(tmp_path):
    harness = Harness(tmp_path, [task("t1", "rain forecast kenya")], reply_done_on(1))

    with pytest.raises(ConfigsRepoError, match=OPENROUTER):
        harness.run(EVAL_CONNECTION_KEYS="https://other.invalid/v1=APP_OPENROUTER_KEY")

    assert harness.docker.pulled == []


@pytest.mark.parametrize(
    "line, field",
    [
        ("model: weather-agent\n", "model"),
        ("variant: high\n", "variant"),
        ("  SKILL_PACKS: []\n", "env.SKILL_PACKS"),
        ("  TASK_MODEL: x\n", "env.TASK_MODEL"),
    ],
)
def test_a_config_setting_what_the_configs_repo_defines_does_not_start_the_run(
    tmp_path, line, field
):
    harness = Harness(tmp_path, [task("t1", "rain forecast kenya")], reply_done_on(1))
    text = line + CONFIG if not line.startswith(" ") else CONFIG + line
    harness.config_path.write_text(text)

    with pytest.raises(ConfigError) as error:
        harness.run()

    assert error.value.field == field
    assert harness.docker.pulled == []


def test_an_app_that_ignores_its_catalog_stops_the_run(tmp_path):
    harness = Harness(tmp_path, [task("t1", "rain forecast kenya")], reply_done_on(1))
    harness.app.catalog_ids = ["prod-model"]

    with pytest.raises(CatalogMismatchError):
        harness.run()

    assert harness.gateway.experiments == []
    assert harness.container.removed


def test_skill_state_is_set_only_where_it_differs_from_the_commit(tmp_path):
    harness = Harness(tmp_path, [task("t1", "rain forecast kenya")], reply_done_on(1))
    plot = {"name": "plot", "is_active": False, "enabled_by_default": False}
    harness.repo.packs([pack(is_active=False, skills=[plot]), pack(CHC_URL)])

    harness.run()

    # The first pack and its skill were toggled and set; the second matched.
    assert harness.app.skill_changes == [
        ("toggle", "pack-0", None, None),
        ("toggle", "pack-0", "skill_plot", None),
        ("enabled-by-default", "pack-0", "skill_plot", {"enabled": False}),
    ]
    assert harness.output()["outcome"] == "goal_reached"
    # The chats get no tool of the skill switched off.
    assert harness.app.completions[0]["tool_ids"] == ["fetch"]


@pytest.mark.parametrize("missing", ["pack", "skill"])
def test_a_pack_or_skill_missing_from_the_app_stops_the_run(tmp_path, missing):
    harness = Harness(tmp_path, [task("t1", "rain forecast kenya")], reply_done_on(1))
    absent = {"name": "absent", "is_active": True, "enabled_by_default": True}
    harness.repo.packs([pack(skills=[absent])])
    if missing == "pack":
        # The app lists no pack.
        harness.app.app_env = lambda: {"CATALOG_MODELS": json.dumps([{"id": MODEL}])}

    with pytest.raises(SkillStateError):
        harness.run()

    assert harness.app.skill_changes == []
    assert harness.gateway.experiments == []


def test_goal_met_records_turns_cost_tokens_and_time(tmp_path):
    items = [task("t1", "rain forecast kenya", deliverables=["image"])]

    def reply(opening, number, text):
        if number == 1:
            return Reply("Which region do you need?")
        return Reply(
            "DONE: here is the map.\n" + display_image("rain.png"),
            artifacts=[image_entry("rain.png")],
            files={"rain.png": image_bytes("rain.png")},
        )

    harness = Harness(
        tmp_path,
        items,
        reply,
        user_replies={"rain forecast kenya": [progress("kenya", "clarifying_question")]},
    )
    harness.run()

    scores = harness.scores()
    assert scores["goal_reached"] == 1.0
    assert scores["turns_to_goal"] == 2
    assert scores["clarifying_questions"] == 1
    assert scores["hints_used"] == 0
    assert scores["time_to_answer_seconds"] == 10.0
    assert scores["answer_quality"] == 4
    assert scores["model_calls_to_goal"] == 4
    assert scores["cost_to_goal_usd"] == pytest.approx(0.42)
    # Per turn: cached call 80000 in = 79000 read + 500 write + 500 uncached; missing call 77000.
    assert scores["tokens_uncached_input"] == 2 * (500 + 77000)
    assert scores["tokens_cache_read"] == 2 * 79000
    assert scores["tokens_cache_write"] == 2 * 500
    assert scores["tokens_output"] == 2 * 400
    assert scores["tool_calls"] == 2
    # One tool at level ERROR, one at DEFAULT with "ok": false.
    assert scores["tool_failures"] == 2
    assert scores["cache_miss_calls"] == 2
    output = harness.output()
    assert output["outcome"] == "goal_reached"
    assert output["chat_id"] == "chat-1"
    assert output["app_trace_ids"] == ["chat-1-trace-0", "chat-1-trace-1"]
    # The item trace carries the app's cost, the same number as the score.
    assert harness.gateway.app_costs == [
        (scores["cost_to_goal_usd"], ["chat-1-trace-0", "chat-1-trace-1"])
    ]
    roles = {role for role, _, _ in harness.models.calls}
    assert roles == {"simulated-user", "judge"}
    assert all(session == "chat-1" for _, session, _ in harness.models.calls)


def test_every_turn_requests_a_streaming_completion(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(2))
    harness.run()

    completions = harness.app.completions
    assert len(completions) == 2
    assert all(body["stream"] is True for body in completions)
    assert "turn" not in completions[0]
    assert "turn" in completions[1]


def test_the_simulated_user_is_sent_its_prompt_filled_with_the_task(tmp_path, monkeypatch):
    template = "MOCK PROMPT\nopening: {opening}\nfacts: {facts}\nhidden:\n{hidden_steps}"
    monkeypatch.setattr(simulated_user_module, "SYSTEM_PROMPT", template)
    items = [task("t1", "rain forecast kenya", hidden=["Use the weekly total", "Plot it"])]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.run()

    [call] = harness.models.user_calls()
    assert call["messages"][0] == {
        "role": "system",
        "content": (
            "MOCK PROMPT\nopening: rain forecast kenya\nfacts: Region: Kenya. Period: next week.\n"
            "hidden:\n- Use the weekly total\n- Plot it"
        ),
    }


def test_the_judge_and_the_rater_are_told_when_the_play_started(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.run()

    started = started_line(PLAY_START)
    # In the verdict's system prompt.
    [verdict] = harness.models.verdict_calls()
    assert started in verdict["messages"][0]["content"]
    # Not as something the user said: the transcript does not hold it.
    assert started not in verdict["messages"][1]["content"]
    [quality] = [c[2] for c in harness.models.calls if c[0] == "judge" and not c[2].get("tools")]
    assert started in quality["messages"][0]["content"]
    # The simulated user is not told.
    assert started not in json.dumps(harness.models.user_calls())


# The presented result and its provenance


def judge_parts(harness) -> list[dict]:
    """The harness-evidence parts of the last verdict call, which hold image parts."""
    messages = harness.models.verdict_calls()[-1]["messages"]
    evidence = [m for m in messages if isinstance(m.get("content"), list)]
    assert len(evidence) == 1
    return evidence[0]["content"]


def map_reply(*, presenting: str, files=None, artifacts=None):
    """Turn 1 asks; later turns present. The chat also holds an intermediate
    zarr and a plot, never presented."""

    def reply(opening, number, text):
        if number == 1:
            return Reply("Which data source should I use?")
        return Reply(
            f"DONE: weekly rain for western Kenya from ERA5.\n{presenting}",
            artifacts=[
                zarr_entry("intermediate_results/era5.zarr"),
                image_entry("intermediate_results/daily.png"),
                *(artifacts or [image_entry("maps/rain.png")]),
            ],
            files={
                "intermediate_results/daily.png": image_bytes("intermediate_results/daily.png"),
                **(files if files is not None else {"maps/rain.png": image_bytes("maps/rain.png")}),
            },
        )

    return reply


def test_presented_image_reaches_the_judge_with_its_provenance(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(tmp_path, items, map_reply(presenting=display_image("maps/rain.png")))
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    verdict = harness.models.verdict_calls()[-1]
    # The fixed criterion is judged with the task's.
    assert "The assistant gives the rain forecast" in json.dumps(verdict["tools"])
    assert PROVENANCE_CRITERION in json.dumps(verdict["tools"])
    transcript = verdict["messages"][1]["content"]
    assert "DONE: weekly rain for western Kenya from ERA5." in transcript
    assert "<details" not in transcript
    # The map with its provenance and image; files no reply presents never reach the judge.
    rain = PresentedFile(
        "maps/rain.png",
        "image",
        provenance_detail(chain()),
        image=image_bytes("maps/rain.png"),
        turn=2,
    )
    assert judge_parts(harness) == evidence_message(Evidence(files=[rain]))["content"]
    # The simulated user saw what a user sees: the path, the kind and the image, before
    # the instruction; no provenance and no harness problem note.
    user_messages = harness.models.user_calls()[-1]["messages"]
    assert user_messages[-2] == files_message(Evidence(files=[rain]))
    assert user_messages[-1] == {"role": "user", "content": REPLY_REQUEST}
    # Files no reply presents are never fetched; the user and the judge each read the map.
    downloads = [r for r in harness.app.file_requests if r[0] != "/artifacts"]
    assert downloads == [("/artifacts/content", "maps/rain.png")] * 2


def test_linked_zarr_is_judged_from_the_listing_alone(tmp_path):
    items = [task("t1", "rain data kenya", deliverables=["zarr"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(
            presenting="Download it: [data](out.zarr)",
            files={},
            artifacts=[zarr_entry("out.zarr")],
        ),
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    parts = judge_parts(harness)
    assert [p["type"] for p in parts] == ["text", "text"]
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        ("out.zarr", "zarr", 2)
    )
    assert '"skill": "aggregate-temporal"' in parts[1]["text"]
    # A zarr deliverable is never downloaded.
    assert {r[0] for r in harness.app.file_requests} == {"/artifacts"}


def test_emailed_attachment_is_presented(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    email = tool_call(
        "send_email",
        {
            "to": ["user@example.invalid"],
            "subject": "Rain",
            "body": "Map attached",
            "attachments": ["maps/rain.png"],
        },
        "Email sent to user@example.invalid with reply-to `user@example.invalid`.",
    )
    harness = Harness(tmp_path, items, map_reply(presenting=email))
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        ("maps/rain.png", "image", 2)
    )


def test_a_zarr_named_only_in_backticks_reaches_the_goal(tmp_path):
    items = [task("t1", "rain data kenya", deliverables=["zarr"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(
            presenting="The data is saved as `kenya_rain.zarr`.",
            files={},
            artifacts=[zarr_entry("kenya_rain.zarr")],
        ),
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    # The only file of a kind the task needs counts as presented.
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        ("kenya_rain.zarr", "zarr", 2)
    )
    assert evidence_files(harness.models.user_calls()[-1]["messages"]) == shown(
        ("kenya_rain.zarr", "zarr")
    )


def test_backtick_files_of_a_needed_kind_none_presented_outright_are_all_presented(tmp_path):
    items = [task("t1", "rain data kenya", deliverables=["zarr"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(
            presenting="Saved `weekly.zarr` and `daily.zarr`.",
            files={},
            artifacts=[zarr_entry("weekly.zarr"), zarr_entry("daily.zarr")],
        ),
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        ("weekly.zarr", "zarr", 2), ("daily.zarr", "zarr", 2)
    )


def test_a_backtick_file_beside_another_of_its_needed_kind_stays_named_in_passing(tmp_path):
    items = [task("t1", "rain data kenya", deliverables=["zarr"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(
            presenting="Download it: [data](out.zarr). It was made from `era5_kenya.zarr`.",
            files={},
            artifacts=[zarr_entry("out.zarr"), zarr_entry("era5_kenya.zarr")],
        ),
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        ("out.zarr", "zarr", 2), ("era5_kenya.zarr", "zarr", 2, True)
    )


def test_a_later_backtick_mention_keeps_the_turn_that_presented_the_file(tmp_path):
    items = [task("t1", "rain data kenya", deliverables=["zarr"])]

    def reply(opening, number, text):
        if number == 1:
            return Reply("Which data source should I use?")
        if number == 2:
            return Reply("Download it: [data](out.zarr)", artifacts=[zarr_entry("out.zarr")])
        return Reply("DONE: `out.zarr` holds the weekly totals.")

    harness = Harness(
        tmp_path, items, reply, user_replies={"rain data kenya": [progress("era5"), progress("ok")]}
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        ("out.zarr", "zarr", 2)
    )


def test_a_backtick_file_is_labeled_and_left_out_of_the_provenance_criterion(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(
            presenting=display_image("maps/rain.png") + "The data is in `kenya_rain.zarr`.",
            artifacts=[image_entry("maps/rain.png"), zarr_entry("kenya_rain.zarr")],
        ),
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    verdict = harness.models.verdict_calls()[0]
    # Labeled as named in passing; its provenance still reaches the judge, as data.
    provenance = provenance_detail(chain())
    files = [
        PresentedFile(
            "maps/rain.png", "image", provenance, image=image_bytes("maps/rain.png"), turn=2
        ),
        PresentedFile("kenya_rain.zarr", "zarr", provenance, turn=2, named_in_passing=True),
    ]
    assert judge_parts(harness) == evidence_message(Evidence(files=files))["content"]
    assert judge_parts(harness)[0]["text"] == EVIDENCE_HEADER
    # The fixed criterion covers only the presented files.
    assert PROVENANCE_CRITERION in json.dumps(verdict["tools"])


def test_backtick_text_that_is_not_a_listed_path_presents_nothing(tmp_path):
    items = [task("t1", "rain data kenya", deliverables=["zarr"])]

    def reply(opening, number, text):
        if number == 1:
            return Reply("Which data source should I use?")
        if number == 2:
            return Reply(
                "Download it: [data](out.zarr). The variable is `precip`; `missing.zarr` failed.",
                artifacts=[zarr_entry("out.zarr")],
            )
        return Reply("DONE: the variable in it is `precip`.")

    harness = Harness(
        tmp_path, items, reply, user_replies={"rain data kenya": [progress("era5"), progress("ok")]}
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    # A later reply naming only unlisted text does not replace the presented files.
    assert [evidence_files(c["messages"]) for c in harness.models.user_calls()] == [
        [],
        shown(("out.zarr", "zarr")),
        shown(("out.zarr", "zarr")),
    ]
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        ("out.zarr", "zarr", 2)
    )


@pytest.mark.parametrize(
    "presenting",
    [
        "DONE: saved `rain.zarr`.",
        "DONE: saved it.\n" + tool_call("list_artifacts", {}, "- `out/rain.zarr/` (zarr)"),
    ],
    ids=["basename-only", "inside-a-tool-block"],
)
def test_backticks_that_do_not_name_a_listed_path_in_the_reply_text_skip_the_judge(
    tmp_path, presenting
):
    items = [task("t1", "rain data kenya", turn_limit=2, deliverables=["zarr"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(presenting=presenting, files={}, artifacts=[zarr_entry("out/rain.zarr")]),
    )
    harness.run()

    output = harness.output()
    assert output["outcome"] == "user_done"
    assert NOTHING_PRESENTED in output["reasoning"]
    assert harness.models.verdict_calls() == []
    assert evidence_files(harness.models.user_calls()[-1]["messages"]) == []


def test_a_displayed_html_page_is_presented_but_is_not_an_image(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=2, deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(
            presenting=display_page("map.html"), files={}, artifacts=[html_entry("map.html")]
        ),
    )
    harness.run()

    assert harness.models.verdict_calls() == []
    assert evidence_files(harness.models.user_calls()[-1]["messages"]) == shown(
        ("map.html", "html")
    )
    # A page is never downloaded.
    assert {r[0] for r in harness.app.file_requests} == {"/artifacts"}


def test_a_displayed_html_page_and_a_zarr_reach_the_judge(tmp_path):
    items = [task("t1", "rain data kenya", deliverables=["zarr"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(
            presenting=display_page("map.html") + "Data: [data](out.zarr)",
            files={},
            artifacts=[html_entry("map.html"), zarr_entry("out.zarr")],
        ),
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        ("map.html", "html", 2), ("out.zarr", "zarr", 2)
    )


def evidence_files(messages: list[dict]) -> list[str]:
    """The first line of each file's text in the file list of a model call's messages."""
    evidence = [m for m in messages if isinstance(m.get("content"), list)]
    if not evidence:
        return []
    texts = [p["text"] for p in evidence[0]["content"] if p["type"] == "text"]
    return [t.split("\n")[0] for t in texts[1:]]


def judged(*files) -> list[str]:
    """``evidence_files`` of the judge's evidence for ``files``, each (path, kind,
    turn) or (path, kind, turn, named in passing)."""
    presented = [
        PresentedFile(path, kind, None, turn=turn, named_in_passing=any(passing))
        for path, kind, turn, *passing in files
    ]
    return evidence_files([evidence_message(Evidence(files=presented))])


def shown(*files, unopened: tuple[str, ...] = ()) -> list[str]:
    """``evidence_files`` of what the simulated user is shown for ``files``, each
    (path, kind), and for ``unopened`` paths, which could not be opened."""
    presented = [PresentedFile(path, kind, None) for path, kind in files]
    evidence = Evidence(files=presented, unlisted=list(unopened))
    return evidence_files([files_message(evidence)])


def test_judge_gets_every_presented_file_labeled_by_turn(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=5, deliverables=["image"])]
    presenting = {
        2: display_image("maps/ecmwf.png"),
        3: "",
        4: display_image("maps/gefs.png"),
    }

    def reply(opening, number, text):
        if number == 1:
            return Reply("Which forecast should I use?")
        return Reply(
            f"DONE: weekly rain for western Kenya.\n{presenting[number]}",
            artifacts=[
                image_entry("intermediate_results/ecmwf_daily.png"),
                image_entry("maps/ecmwf.png", fetcher="ecmwf-fetch"),
                image_entry("maps/gefs.png", fetcher="gefs-fetch"),
            ],
            files={
                "intermediate_results/ecmwf_daily.png": image_bytes("x"),
                "maps/ecmwf.png": image_bytes("maps/ecmwf.png"),
                "maps/gefs.png": image_bytes("maps/gefs.png"),
            },
        )

    harness = Harness(
        tmp_path,
        items,
        reply,
        user_replies={"rain map kenya": [progress("ecmwf"), progress("now gefs"), progress("ok")]},
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert harness.scores()["turns_to_goal"] == 4
    # The user sees the latest presenting reply's files.
    assert [evidence_files(c["messages"]) for c in harness.models.user_calls()] == [
        [],
        shown(("maps/ecmwf.png", "image")),
        shown(("maps/ecmwf.png", "image")),
        shown(("maps/gefs.png", "image")),
    ]
    # The judge gets both, in order, labeled by turn, and no intermediate file.
    assert [evidence_files(c["messages"]) for c in harness.models.verdict_calls()] == [
        judged(("maps/ecmwf.png", "image", 2), ("maps/gefs.png", "image", 4))
    ]
    assert "intermediate_results" not in json.dumps(harness.models.verdict_calls())
    # Each presenting reply's files are read once for the user; the judge reads them all.
    assert harness.app.file_requests == [
        ("/artifacts", None),
        ("/artifacts/content", "maps/ecmwf.png"),
        ("/artifacts", None),
        ("/artifacts/content", "maps/gefs.png"),
        ("/artifacts", None),
        ("/artifacts/content", "maps/ecmwf.png"),
        ("/artifacts/content", "maps/gefs.png"),
    ]


def test_a_path_presented_twice_reaches_the_judge_once_with_its_latest_image(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]

    def reply(opening, number, text):
        # Turn 2 presents the map in inches, turn 3 the same path replotted in mm.
        unit = "inches" if number == 2 else "mm"
        return Reply(
            f"DONE: weekly rain in {unit}.\n{display_image('maps/rain.png')}",
            artifacts=[image_entry("maps/rain.png")],
            files={"maps/rain.png": f"map in {unit}".encode()},
        )

    harness = Harness(
        tmp_path,
        items,
        lambda opening, number, text: (
            Reply("Which region?") if number == 1 else reply(opening, number, text)
        ),
        user_replies={"rain map kenya": [progress("kenya"), progress("replot that in mm")]},
    )
    harness.run()

    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        ("maps/rain.png", "image", 3)
    )
    in_mm = f"data:image/png;base64,{base64.b64encode(b'map in mm').decode()}"
    images = [p["image_url"]["url"] for p in judge_parts(harness) if p["type"] == "image_url"]
    assert images == [in_mm]
    # The user saw the replotted map too: downloaded for each presentation, then by the judge.
    user_parts = harness.models.user_calls()[-1]["messages"][-2]["content"]
    assert [p["image_url"]["url"] for p in user_parts if p["type"] == "image_url"] == [in_mm]
    downloads = [r for r in harness.app.file_requests if r[0] != "/artifacts"]
    assert downloads == [("/artifacts/content", "maps/rain.png")] * 3


def daily_then_weekly(opening, number, text):
    """Turn 1 asks, turn 2 presents a daily map, later turns a weekly one."""
    if number == 1:
        return Reply("Which data source should I use?")
    path, period = ("maps/daily.png", "daily") if number == 2 else ("maps/weekly.png", "weekly")
    return Reply(
        f"DONE: rain for western Kenya.\n{display_image(path)}",
        artifacts=[image_entry(path, period=period)],
        files={path: image_bytes(path)},
    )


def test_wrong_result_is_corrected_and_the_play_continues(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=5, deliverables=["image"])]
    correction = progress("I asked for weekly totals, not daily. Please make it weekly.")
    harness = Harness(
        tmp_path,
        items,
        daily_then_weekly,
        user_replies={"rain map kenya": [progress("era5"), correction]},
    )
    harness.run()

    calls = harness.models.user_calls()
    # The user saw the daily map's path and image, and asked for the fix.
    assert evidence_files(calls[1]["messages"]) == shown(("maps/daily.png", "image"))
    assert [p["type"] for p in calls[1]["messages"][-2]["content"]] == ["text", "text", "image_url"]
    sent = [body["turn"]["user_message"]["content"] for body in harness.app.completions[1:]]
    assert sent == ["era5", correction["message"]]
    assert harness.output()["outcome"] == "goal_reached"
    assert harness.scores()["turns_to_goal"] == 3
    # The judge gets both maps, labeled by turn; the criteria decide which counts.
    assert [evidence_files(c["messages"]) for c in harness.models.verdict_calls()] == [
        judged(("maps/daily.png", "image", 2), ("maps/weekly.png", "image", 3))
    ]


def test_no_files_yet_shows_the_user_text_only(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=3, deliverables=["image"])]
    harness = Harness(tmp_path, items, map_reply(presenting=display_image("maps/rain.png")))
    harness.run()

    first = harness.models.user_calls()[0]["messages"]
    assert [m["role"] for m in first] == ["system", "user", "user"]
    assert all(isinstance(m["content"], str) for m in first)
    assert "Which data source should I use?" in first[1]["content"]
    # Nothing was presented yet, so nothing was listed or downloaded for it; then the
    # user and the judge each read the map.
    read_map = [("/artifacts", None), ("/artifacts/content", "maps/rain.png")]
    assert harness.app.file_requests == read_map * 2


def test_user_done_and_judge_passed_reaches_the_goal(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=4, deliverables=["image"])]
    harness = Harness(tmp_path, items, map_reply(presenting=display_image("maps/rain.png")))
    harness.run()

    output = harness.output()
    assert output["outcome"] == "goal_reached"
    assert output["goal_reached"] is True
    assert output["reasoning"] == "verdict passed"
    assert harness.scores()["turns_to_goal"] == 2
    assert len(harness.models.verdict_calls()) == 1


def test_user_done_and_judge_failed_ends_as_user_done(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=4, deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(
            presenting=display_image("maps/rain.png"),
            artifacts=[image_entry("maps/rain.png", fetcher="chirps-fetch")],
        ),
        judge_fails_on="chirps-fetch",
    )
    harness.run()

    output = harness.output()
    assert output["outcome"] == "user_done"
    assert harness.scores()["goal_reached"] == 0.0
    assert harness.scores()["turns_to_goal"] == 2
    assert len(harness.models.verdict_calls()) == 1


# The judge panel

PANEL = ["openrouter/test/judge-a", "openrouter/test/judge-b", "openrouter/test/judge-c"]


def evidence_messages(call: dict) -> list[dict]:
    """The harness-evidence messages of a verdict call."""
    return [m for m in call["messages"] if isinstance(m.get("content"), list)]


def test_a_judge_panel_rules_on_the_same_evidence_and_the_majority_decides(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=3, deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(presenting=display_image("maps/rain.png")),
        failing_judges={PANEL[1]},
        quality_scores={PANEL[0]: 5, PANEL[1]: 2, PANEL[2]: 4},
    )
    harness.run(EVAL_JUDGE_MODELS=",".join(PANEL))

    output = harness.output()
    assert output["outcome"] == "goal_reached"
    # Every judge ruled once, with its own model, on the same evidence.
    verdicts = harness.models.verdict_calls()
    assert sorted(call["model"] for call in verdicts) == PANEL
    assert evidence_messages(verdicts[0])
    assert all(evidence_messages(call) == evidence_messages(verdicts[0]) for call in verdicts)
    # The evidence was read once for the whole panel, after the user's read.
    read_map = [("/artifacts", None), ("/artifacts/content", "maps/rain.png")]
    assert harness.app.file_requests == read_map * 2
    # Each judge's verdict and reasoning is on the item, in the order of EVAL_JUDGE_MODELS.
    assert output["judges"] == [
        {"model": PANEL[0], "state": "passed", "passed": True, "reasoning": "verdict passed"},
        {"model": PANEL[1], "state": "failed", "passed": False, "reasoning": "verdict failed"},
        {"model": PANEL[2], "state": "passed", "passed": True, "reasoning": "verdict passed"},
    ]
    scores = harness.scores()
    assert scores["goal_reached"] == 1.0
    assert scores["judges_agree"] == 0.0
    # Every judge rates the answer; answer_quality is their mean.
    ratings = [c[2] for c in harness.models.calls if c[0] == "judge" and not c[2].get("tools")]
    assert sorted(call["model"] for call in ratings) == PANEL
    assert scores["answer_quality:test/judge-a"] == 5
    assert scores["answer_quality:test/judge-b"] == 2
    assert scores["answer_quality:test/judge-c"] == 4
    assert scores["answer_quality"] == pytest.approx(11 / 3)


def test_a_panel_that_agrees_records_judges_agree(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.run(EVAL_JUDGE_MODELS=",".join(PANEL))

    output = harness.output()
    assert output["outcome"] == "goal_reached"
    # The same reasoning from every judge is given once.
    assert output["reasoning"] == "verdict passed"
    assert [j["passed"] for j in output["judges"]] == [True, True, True]
    assert harness.scores()["judges_agree"] == 1.0


def test_a_tied_panel_fails(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), failing_judges={PANEL[0]})
    harness.run(EVAL_JUDGE_MODELS=",".join(PANEL[:2]))

    output = harness.output()
    assert output["outcome"] == "user_done"
    assert output["goal_reached"] is False
    assert harness.scores()["goal_reached"] == 0.0
    assert harness.scores()["judges_agree"] == 0.0


def test_a_panel_without_a_result_to_judge_calls_no_judge(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=2, deliverables=["zarr"])]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.run(EVAL_JUDGE_MODELS=",".join(PANEL))

    output = harness.output()
    assert output["outcome"] == "user_done"
    assert NOTHING_PRESENTED in output["reasoning"]
    assert output["judges"] == [
        {"model": model, "state": "failed", "passed": False, "reasoning": NOTHING_PRESENTED}
        for model in PANEL
    ]
    assert harness.models.verdict_calls() == []
    # No judge model was called, so the judges did not agree on anything.
    assert "judges_agree" not in harness.scores()


CUT_OFF = "The judge reply was cut off at max_tokens 8192"


def test_an_errored_judge_is_left_out_and_the_others_decide(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(1),
        verdict_error=ModelReplyError(CUT_OFF),
        verdict_error_for=PANEL[1],
    )
    harness.run(EVAL_JUDGE_MODELS=",".join(PANEL))

    output = harness.output()
    assert output["outcome"] == "goal_reached"
    assert output["error"] is None
    # The errored judge is recorded with its error; the other judges ruled.
    assert [(j["state"], j["passed"]) for j in output["judges"]] == [
        ("passed", True),
        ("error", None),
        ("passed", True),
    ]
    assert CUT_OFF in output["judges"][1]["reasoning"]
    scores = harness.scores()
    assert scores["goal_reached"] == 1.0
    # The two judges that ruled agree.
    assert scores["judges_agree"] == 1.0
    # Every judge still rated the answer.
    assert scores["answer_quality"] == 4


def test_an_errored_judge_beside_a_failing_one_leaves_a_failing_majority(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(1),
        failing_judges={PANEL[2]},
        verdict_error=ModelReplyError(CUT_OFF),
        verdict_error_for=PANEL[1],
    )
    harness.run(EVAL_JUDGE_MODELS=",".join(PANEL))

    output = harness.output()
    # One pass and one fail among the judges that ruled is a tie, which fails.
    assert output["outcome"] == "user_done"
    assert harness.scores()["judges_agree"] == 0.0


def test_one_judge_left_after_an_error_decides_without_judges_agree(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(1),
        verdict_error=ModelReplyError(CUT_OFF),
        verdict_error_for=PANEL[1],
    )
    harness.run(EVAL_JUDGE_MODELS=",".join(PANEL[:2]))

    assert harness.output()["outcome"] == "goal_reached"
    assert "judges_agree" not in harness.scores()


def test_a_panel_where_no_judge_ruled_ends_the_play_as_error(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), verdict_error=ModelReplyError(CUT_OFF))
    harness.run(EVAL_JUDGE_MODELS=",".join(PANEL[:2]))

    output = harness.output()
    assert output["outcome"] == "error"
    assert [j["state"] for j in output["judges"]] == ["error", "error"]
    scores = harness.scores()
    assert scores["goal_reached"] == 0.0
    assert "judges_agree" not in scores


def verdict_calls_of(harness, model: str) -> list[dict]:
    return [call for call in harness.models.verdict_calls() if call["model"] == model]


def test_a_judge_whose_provider_refuses_a_forced_tool_choice_rules_with_auto(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=3, deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(presenting=display_image("maps/rain.png")),
        refuses_forced_tool={PANEL[1]},
    )
    harness.run(EVAL_JUDGE_MODELS=",".join(PANEL))

    output = harness.output()
    assert output["outcome"] == "goal_reached"
    # Every judge ruled; the refused one is not errored.
    assert [(j["state"], j["passed"]) for j in output["judges"]] == [("passed", True)] * 3
    refused, retried = verdict_calls_of(harness, PANEL[1])
    assert refused["tool_choice"] == {"type": "function", "function": {"name": "finish_test"}}
    assert retried["tool_choice"] == "auto"
    # The evidence stays at index 2, and the instruction comes last.
    assert retried["messages"][2] == evidence_messages(refused)[0]
    assert retried["messages"][:-1] == refused["messages"]
    assert retried["messages"][-1]["role"] == "user"
    # A judge whose provider does not refuse is called once, as Scenario sends it.
    for model in (PANEL[0], PANEL[2]):
        [call] = verdict_calls_of(harness, model)
        assert call["tool_choice"] == refused["tool_choice"]
        assert call["messages"] == refused["messages"]
    assert harness.scores()["judges_agree"] == 1.0


def test_a_judge_refused_again_after_auto_is_errored_without_a_third_call(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(1),
        verdict_error=tool_choice_refusal(PANEL[1]),
        verdict_error_for=PANEL[1],
    )
    harness.run(EVAL_JUDGE_MODELS=",".join(PANEL))

    output = harness.output()
    assert [j["state"] for j in output["judges"]] == ["passed", "error", "passed"]
    assert (
        'tool_choice: type "tool" and "any" are not supported' in output["judges"][1]["reasoning"]
    )
    assert [call["tool_choice"] for call in verdict_calls_of(harness, PANEL[1])] == [
        {"type": "function", "function": {"name": "finish_test"}},
        "auto",
    ]
    assert output["outcome"] == "goal_reached"


def test_an_error_that_is_not_a_refusal_is_not_sent_again(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(1),
        verdict_error=ModelReplyError(CUT_OFF),
        verdict_error_for=PANEL[1],
    )
    harness.run(EVAL_JUDGE_MODELS=",".join(PANEL[:2]))

    assert [j["state"] for j in harness.output()["judges"]] == ["passed", "error"]
    assert len(verdict_calls_of(harness, PANEL[1])) == 1


def test_a_judge_whose_rating_fails_is_left_out_of_the_mean(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    # A score of 0 is not a rating, so judge-b's rating fails.
    harness = Harness(tmp_path, items, reply_done_on(1), quality_scores={PANEL[0]: 5, PANEL[1]: 0})
    harness.run(EVAL_JUDGE_MODELS=",".join(PANEL[:2]))

    scores = harness.scores()
    assert scores["answer_quality:test/judge-a"] == 5
    assert "answer_quality:test/judge-b" not in scores
    assert scores["answer_quality"] == 5


# The comparability hash


def run_metadata(tmp_path, items=None, change=None, **settings) -> dict:
    """The experiment metadata of one run; ``change`` edits the harness first."""
    harness = Harness(tmp_path, items or [task("t1", "rain forecast kenya")], reply_done_on(1))
    if change:
        change(harness)
    harness.run(**settings)
    return harness.gateway.experiments[0]["metadata"]


def test_runs_with_the_same_judges_in_another_order_share_the_comparability_hash(tmp_path):
    first = run_metadata(tmp_path, EVAL_JUDGE_MODELS=",".join(PANEL))
    reordered = run_metadata(tmp_path, EVAL_JUDGE_MODELS=",".join(reversed(PANEL)))

    assert first["judge_models"] == reordered["judge_models"] == ",".join(sorted(PANEL))
    assert first["comparability_hash"] == reordered["comparability_hash"]


def test_what_is_under_test_does_not_change_the_comparability_hash(tmp_path):
    first = run_metadata(tmp_path)

    def other_agent(harness):
        harness.app.base_model_id = "open-router-.anthropic/claude-opus-5.5"
        harness.config_path.write_text(
            CONFIG.replace(APP_IMAGE_LINE, f"image: {APP_REPOSITORY}:other").replace(
                RUNNER_IMAGE, f"{RUNNER_REPOSITORY}:other"
            )
        )
        harness.repo.model("other-agent")
        harness.model = "other-agent"

    second = run_metadata(tmp_path, change=other_agent)

    assert second["model"] != first["model"]
    assert second["image"] != first["image"]
    assert second["runner_image"] != first["runner_image"]
    assert second["catalog_model"] != first["catalog_model"]
    assert second["comparability_hash"] == first["comparability_hash"]


def test_runs_graded_differently_have_different_comparability_hashes(tmp_path):
    first = run_metadata(tmp_path)
    second = run_metadata(tmp_path, EVAL_JUDGE_MODELS="openrouter/test/other-judge")

    assert second["comparability_hash"] != first["comparability_hash"]


# The play order

SIX_TASKS = [task(f"t{n}", f"rain forecast kenya {n}") for n in range(1, 7)]


def test_each_run_plays_the_tasks_in_an_order_of_its_own_with_the_same_hash(tmp_path):
    orders, hashes = [], []
    for seed in (1, 2):
        harness = Harness(tmp_path, SIX_TASKS, reply_done_on(1))
        harness.run(rng=random.Random(seed))
        experiment = harness.gateway.experiments[0]
        orders.append([r.item.id for r in experiment["items"]])
        hashes.append(experiment["metadata"]["comparability_hash"])

    # Each run's order is its own random source's shuffle of the dataset's order.
    expected = []
    for seed in (1, 2):
        ids = [item["id"] for item in SIX_TASKS]
        random.Random(seed).shuffle(ids)
        expected.append(ids)
    assert orders == expected
    assert orders[0] != orders[1]
    # The hash reads the items in order of id, so the play order leaves it alone.
    assert hashes[0] == hashes[1]


def test_without_a_random_source_a_run_shuffles_with_an_unseeded_one(tmp_path, monkeypatch):
    built = []

    class Recording(random.Random):
        def __init__(self, *args):
            built.append(self)
            self.args = args
            self.shuffled = []
            super().__init__(*args)

        def shuffle(self, items):
            self.shuffled.append([item.id for item in items])
            super().shuffle(items)

    monkeypatch.setattr(runner_module, "random", SimpleNamespace(Random=Recording))
    harness = Harness(tmp_path, SIX_TASKS, reply_done_on(1))

    harness.run(keep_order=False)

    [source] = built
    assert source.args == ()
    # It shuffled the dataset's items, as read.
    assert source.shuffled == [[item["id"] for item in SIX_TASKS]]
    played = [r.item.id for r in harness.gateway.experiments[0]["items"]]
    assert sorted(played) == sorted(source.shuffled[0])


def test_play_continues_while_the_user_asks_for_more(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]

    def reply(opening, number, text):
        # Every reply would pass the judge; the second adds what the user asked for.
        extra = " with country borders" if "borders" in text else ""
        return Reply(
            f"DONE: weekly rain map{extra}.\n{display_image('maps/rain.png')}",
            artifacts=[image_entry("maps/rain.png")],
            files={"maps/rain.png": image_bytes("maps/rain.png")},
        )

    harness = Harness(
        tmp_path,
        items,
        reply,
        user_replies={"rain map kenya": [progress("also add the country borders")]},
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert harness.scores()["turns_to_goal"] == 2
    roles = [role for role, _, _ in harness.models.calls]
    # Two user replies, then the one verdict and the quality rating.
    assert roles == ["simulated-user", "simulated-user", "judge", "judge"]
    transcript = harness.models.verdict_calls()[0]["messages"][1]["content"]
    assert "DONE: weekly rain map with country borders." in transcript


def test_nothing_presented_skips_the_judge(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=2, deliverables=["zarr"])]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.run()

    output = harness.output()
    assert harness.scores()["goal_reached"] == 0.0
    assert output["outcome"] == "user_done"
    assert NOTHING_PRESENTED in output["reasoning"]
    assert harness.models.verdict_calls() == []


def test_file_without_provenance_skips_the_judge_and_is_shown_to_the_user(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(
            presenting=display_image("maps/rain.png"),
            artifacts=[image_entry("maps/rain.png", provenance=False)],
        ),
    )
    harness.run()

    assert harness.models.verdict_calls() == []
    # The user sees the map as a user would.
    user_messages = harness.models.user_calls()[-1]["messages"]
    assert [p["type"] for p in user_messages[-2]["content"]] == ["text", "text", "image_url"]
    assert evidence_files(user_messages) == shown(("maps/rain.png", "image"))


def test_download_that_fails_once_is_tried_again_for_the_judge(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(tmp_path, items, map_reply(presenting=display_image("maps/rain.png")))
    harness.app.failing_downloads["maps/rain.png"] = 1
    harness.run()

    # The user got the map as one it could not open; the judge read it again and saw it.
    user_messages = harness.models.user_calls()[-1]["messages"]
    assert evidence_files(user_messages) == shown(unopened=("maps/rain.png",))
    assert harness.output()["outcome"] == "goal_reached"
    assert [p["type"] for p in judge_parts(harness)] == ["text", "text", "image_url"]
    downloads = [r for r in harness.app.file_requests if r[0] != "/artifacts"]
    assert downloads == [("/artifacts/content", "maps/rain.png")] * 2


def test_user_sees_only_the_file_list_for_a_missing_kind(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(presenting="[data](out.zarr)", artifacts=[zarr_entry("out.zarr")]),
    )
    harness.run()

    # From the file list alone the user can see it got data and no picture.
    user_messages = harness.models.user_calls()[-1]["messages"]
    assert [p["type"] for p in user_messages[-2]["content"]] == ["text", "text"]
    assert evidence_files(user_messages) == shown(("out.zarr", "zarr"))
    assert harness.output()["outcome"] == "user_done"


def test_user_sees_only_the_listed_files_for_an_unlisted_file(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(presenting=display_image("maps/rain.png") + display_image("maps/gone.png")),
    )
    harness.run()

    # The path that is not in the chat's files is shown as one that could not be opened.
    user_messages = harness.models.user_calls()[-1]["messages"]
    parts = user_messages[-2]["content"]
    assert [p["type"] for p in parts] == ["text", "text", "image_url", "text"]
    assert evidence_files(user_messages) == shown(
        ("maps/rain.png", "image"), unopened=("maps/gone.png",)
    )
    assert harness.output()["outcome"] == "user_done"


def test_evidence_with_a_problem_is_not_read_again_until_a_new_reply_presents_files(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]

    def reply(opening, number, text):
        if number == 2:
            return Reply(
                f"Here is a draft.\n{display_image('maps/rain.png')}",
                artifacts=[image_entry("maps/rain.png", provenance=False)],
                files={"maps/rain.png": image_bytes("maps/rain.png")},
            )
        return Reply("DONE: that is all." if number == 3 else "Which region?")

    harness = Harness(
        tmp_path,
        items,
        reply,
        user_replies={"rain map kenya": [progress("kenya"), progress("ok")]},
    )
    harness.run()

    # Read for the user after reply 2, reused after reply 3, and read for the judge.
    read_map = [("/artifacts", None), ("/artifacts/content", "maps/rain.png")]
    assert harness.app.file_requests == read_map * 2
    assert len(harness.models.user_calls()) == 3


def early_then_final(early_provenance: bool, final_provenance: bool):
    def reply(opening, number, text):
        if number == 1:
            return Reply("Which region?")
        path = "maps/draft.png" if number == 2 else "maps/final.png"
        return Reply(
            f"DONE: weekly rain map.\n{display_image(path)}",
            artifacts=[
                image_entry("maps/draft.png", provenance=early_provenance),
                image_entry("maps/final.png", provenance=final_provenance),
            ],
            files={p: image_bytes(p) for p in ("maps/draft.png", "maps/final.png")},
        )

    return reply


def test_an_early_file_problem_is_a_note_for_the_judge(tmp_path, monkeypatch):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        early_then_final(early_provenance=False, final_provenance=True),
        user_replies={"rain map kenya": [progress("kenya"), progress("make it final")]},
    )
    judge_evidence = []
    collect = play_module.collect_evidence

    async def recording(*args, **kwargs):
        evidence = await collect(*args, **kwargs)
        if kwargs.get("every"):
            judge_evidence.append(evidence)
        return evidence

    monkeypatch.setattr(play_module, "collect_evidence", recording)
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        ("maps/draft.png", "image", 2), ("maps/final.png", "image", 3)
    )
    # The draft's note reaches the judge with it.
    [evidence] = judge_evidence
    [note] = [f.note for f in evidence.files if f.path == "maps/draft.png"]
    assert note is not None
    texts = [p["text"] for p in judge_parts(harness) if p["type"] == "text"]
    assert any(text.startswith(judged(("maps/draft.png", "image", 2))[0]) for text in texts)
    assert any(note in text for text in texts)
    # A file named in passing reads otherwise than a presented one.
    assert judged(("maps/draft.png", "image", 2, True)) != judged(("maps/draft.png", "image", 2))
    # The user is shown neither the note nor the final map's provenance.
    for call in harness.models.user_calls():
        shown_text = json.dumps(call["messages"])
        assert note not in shown_text
        assert "branches" not in shown_text


def test_a_final_file_without_provenance_still_blocks_the_judge(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        early_then_final(early_provenance=True, final_provenance=False),
        user_replies={"rain map kenya": [progress("kenya"), progress("make it final")]},
    )
    harness.run()

    output = harness.output()
    assert output["outcome"] == "user_done"
    assert harness.models.verdict_calls() == []


def test_the_final_read_tries_a_failed_listing_and_download_once_more(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(tmp_path, items, map_reply(presenting=display_image("maps/rain.png")))
    # The user's read (a listing and a download) succeeds; the judge's first
    # listing and first download fail once each.
    harness.app.failures_after = 2
    harness.app.failing_listings = 1
    harness.app.failing_downloads["maps/rain.png"] = 1
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert [p["type"] for p in judge_parts(harness)] == ["text", "text", "image_url"]
    read_map = [("/artifacts", None), ("/artifacts/content", "maps/rain.png")]
    assert harness.app.file_requests == read_map + [
        ("/artifacts", None),
        ("/artifacts", None),
        ("/artifacts/content", "maps/rain.png"),
        ("/artifacts/content", "maps/rain.png"),
    ]


def test_the_final_read_lists_again_when_a_last_reply_path_is_missing(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(tmp_path, items, map_reply(presenting=display_image("maps/rain.png")))
    listings = []
    handle = harness.app._files

    def lagging(state, rest, params):
        # The judge's first listing does not hold the map yet.
        if rest == "/artifacts":
            listings.append(rest)
            if len(listings) == 2:
                harness.app.file_requests.append((rest, None))
                return httpx.Response(200, json=[])
        return handle(state, rest, params)

    harness.app._files = lagging
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert len(listings) == 3
    assert [p["type"] for p in judge_parts(harness)] == ["text", "text", "image_url"]


def test_images_already_downloaded_are_reused_when_a_path_is_unlisted(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]

    def reply(opening, number, text):
        if number == 1:
            return Reply("Which region?")
        if number == 2:
            return Reply(
                f"Maps.\n{display_image('maps/rain.png')}{display_image('maps/gone.png')}",
                artifacts=[image_entry("maps/rain.png")],
                files={"maps/rain.png": image_bytes("maps/rain.png")},
            )
        return Reply("Still working on it.")

    harness = Harness(
        tmp_path,
        items,
        reply,
        user_replies={"rain map kenya": [progress("kenya"), progress("?"), progress("??")]},
    )
    harness.run(EVAL_MAX_TURNS="4")

    # The user's reads after replies 2 and 3 list the files each time and
    # download the map once; the judge downloads it on its own.
    user_requests = harness.app.file_requests[:3]
    assert user_requests == [
        ("/artifacts", None),
        ("/artifacts/content", "maps/rain.png"),
        ("/artifacts", None),
    ]
    downloads = [r for r in harness.app.file_requests if r[0] != "/artifacts"]
    assert downloads == [("/artifacts/content", "maps/rain.png")] * 2
    calls = harness.models.user_calls()
    assert [p["type"] for p in calls[2]["messages"][-2]["content"]] == [
        "text",
        "text",
        "image_url",
        "text",
    ]


def test_deliverable_kinds_presented_on_different_turns_pass(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image", "zarr"])]

    def reply(opening, number, text):
        if number == 1:
            return Reply("Which region?")
        if number == 2:
            return Reply("Here is the data: [data](out.zarr)", artifacts=[zarr_entry("out.zarr")])
        if number == 3:
            return Reply("Making the map.")
        return Reply(
            f"DONE: the map.\n{display_image('maps/rain.png')}",
            artifacts=[image_entry("maps/rain.png")],
            files={"maps/rain.png": image_bytes("maps/rain.png")},
        )

    harness = Harness(
        tmp_path,
        items,
        reply,
        user_replies={"rain map kenya": [progress("kenya"), progress("map?"), progress("ok")]},
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        ("out.zarr", "zarr", 2), ("maps/rain.png", "image", 4)
    )


def test_an_earlier_unlisted_file_is_a_note_for_the_judge(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]

    def reply(opening, number, text):
        if number == 1:
            return Reply("Which region?")
        if number == 2:
            return Reply(f"Draft.\n{display_image('maps/early.png')}")
        return Reply(
            f"DONE: final map.\n{display_image('maps/final.png')}",
            artifacts=[image_entry("maps/final.png")],
            files={"maps/final.png": image_bytes("maps/final.png")},
        )

    harness = Harness(
        tmp_path,
        items,
        reply,
        user_replies={"rain map kenya": [progress("kenya"), progress("it does not open")]},
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        ("maps/early.png", "unknown", 2), ("maps/final.png", "image", 3)
    )


def test_an_earlier_file_whose_download_failed_is_a_note_for_the_judge(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        early_then_final(early_provenance=True, final_provenance=True),
        user_replies={"rain map kenya": [progress("kenya"), progress("make it final")]},
    )
    harness.app.failing_downloads["maps/draft.png"] = 99
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        ("maps/draft.png", "image", 2), ("maps/final.png", "image", 3)
    )


def test_the_final_read_waits_the_poll_interval_before_each_retry(tmp_path):
    def run(failing: bool) -> list[float]:
        items = [task("t1", "rain map kenya", deliverables=["image"])]
        harness = Harness(tmp_path, items, map_reply(presenting=display_image("maps/rain.png")))
        if failing:
            harness.app.failures_after = 2
            harness.app.failing_listings = 1
            harness.app.failing_downloads["maps/rain.png"] = 1
        harness.run()
        assert harness.output()["outcome"] == "goal_reached"
        return harness.clock.sleeps

    # At the default EVAL_POLL_INTERVAL_SECONDS: one wait before the extra
    # listing, one before the extra download.
    extra = Counter(run(failing=True)) - Counter(run(failing=False))
    assert extra == Counter({DEFAULT_POLL_INTERVAL_SECONDS: 2})


def test_a_failed_listing_shows_the_user_each_path_as_not_opened(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(tmp_path, items, map_reply(presenting=display_image("maps/rain.png")))
    harness.app.failing_listings = 1
    harness.run()

    user_messages = harness.models.user_calls()[-1]["messages"]
    assert evidence_files(user_messages) == shown(unopened=("maps/rain.png",))
    # The judge lists the files again and sees the map.
    assert harness.output()["outcome"] == "goal_reached"


def test_an_unlisted_path_is_looked_up_again_on_the_next_turn(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]

    def reply(opening, number, text):
        if number == 1:
            return Reply("Which region?")
        if number == 2:
            # The listing does not hold the map yet.
            return Reply(f"Here it is.\n{display_image('maps/rain.png')}")
        return Reply(
            "DONE: it should open now.",
            artifacts=[image_entry("maps/rain.png")],
            files={"maps/rain.png": image_bytes("maps/rain.png")},
        )

    harness = Harness(
        tmp_path, items, reply, user_replies={"rain map kenya": [progress("kenya"), progress("?")]}
    )
    harness.run()

    calls = harness.models.user_calls()
    assert evidence_files(calls[1]["messages"]) == shown(unopened=("maps/rain.png",))
    assert [p["type"] for p in calls[2]["messages"][-2]["content"]] == [
        "text",
        "text",
        "image_url",
    ]
    assert harness.output()["outcome"] == "goal_reached"


def test_a_backtick_file_the_listing_lags_is_shown_on_the_next_turn(tmp_path):
    items = [task("t1", "rain data kenya", deliverables=["zarr"])]

    def reply(opening, number, text):
        if number == 1:
            return Reply("Which data source should I use?")
        if number == 2:
            # The listing does not hold the store yet.
            return Reply("The data is saved as `k.zarr`.")
        return Reply("DONE: it is ready now.", artifacts=[zarr_entry("k.zarr")])

    harness = Harness(
        tmp_path,
        items,
        reply,
        user_replies={"rain data kenya": [progress("era5"), progress("where is it?")]},
    )
    harness.run()

    assert [evidence_files(c["messages"]) for c in harness.models.user_calls()] == [
        [],
        [],
        shown(("k.zarr", "zarr")),
    ]
    assert harness.output()["outcome"] == "goal_reached"


@pytest.mark.parametrize("path", ["maps/rain.svg", "maps/rain"])
def test_an_image_outside_the_shown_types_is_a_note_for_the_user_and_the_judge(tmp_path, path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(
            presenting=display_image(path),
            artifacts=[image_entry(path)],
            files={path: image_bytes(path)},
        ),
    )
    harness.run()

    user_messages = harness.models.user_calls()[-1]["messages"]
    assert [p["type"] for p in user_messages[-2]["content"]] == ["text", "text"]
    assert evidence_files(user_messages) == shown((path, "image"))
    # The judge is still called; the note goes with the file's evidence.
    assert harness.output()["outcome"] == "goal_reached"
    assert [p["type"] for p in judge_parts(harness)] == ["text", "text"]
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == judged(
        (path, "image", 2)
    )
    # Such an image is never downloaded.
    assert {r[0] for r in harness.app.file_requests} == {"/artifacts"}


def test_unreadable_image_skips_the_judge(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=2, deliverables=["image"])]
    harness = Harness(
        tmp_path, items, map_reply(presenting=display_image("maps/rain.png"), files={})
    )
    harness.run()

    assert harness.models.verdict_calls() == []


def test_missing_kind_skips_the_judge(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=2, deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(presenting="[data](out.zarr)", artifacts=[zarr_entry("out.zarr")]),
    )
    harness.run()

    assert harness.models.verdict_calls() == []
    assert not [r for r in harness.app.file_requests if r[0] != "/artifacts"]


def test_every_deliverable_kind_must_be_presented(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=2, deliverables=["image", "zarr"])]
    harness = Harness(tmp_path, items, map_reply(presenting=display_image("maps/rain.png")))
    harness.run()

    assert harness.models.verdict_calls() == []


def test_task_without_deliverables_is_judged_on_the_chat_alone(tmp_path):
    items = [task("t1", "rain map kenya")]
    harness = Harness(tmp_path, items, map_reply(presenting=display_image("maps/rain.png")))
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    # The user sees the presented map; the judge reads no file.
    assert harness.app.file_requests == [
        ("/artifacts", None),
        ("/artifacts/content", "maps/rain.png"),
    ]
    assert evidence_files(harness.models.user_calls()[-1]["messages"]) == shown(
        ("maps/rain.png", "image")
    )
    messages = harness.models.verdict_calls()[-1]["messages"]
    assert all(isinstance(m["content"], str) for m in messages)
    assert PROVENANCE_CRITERION not in json.dumps(harness.models.verdict_calls()[-1]["tools"])


def test_item_with_artifact_checks_does_not_start_the_run(tmp_path):
    item = task("t1", "rain map kenya")
    item["expected_output"]["artifact_checks"] = [{"path": "*.png"}]
    harness = Harness(tmp_path, [item], reply_done_on(1))

    with pytest.raises(TaskError):
        harness.run()
    assert harness.docker.run_calls == []


def judged_at_the_cap(monkeypatch) -> list[bool]:
    """Records whether each judge panel ruling is told the play reached the turn cap."""
    at_cap = []
    rule = play_module.rule_panel

    async def recording(judges, state, *, at_turn_cap):
        at_cap.append(at_turn_cap)
        return await rule(judges, state, at_turn_cap=at_turn_cap)

    monkeypatch.setattr(play_module, "rule_panel", recording)
    return at_cap


def test_turn_cap_of_the_task_ends_as_failure_with_its_scores(tmp_path, monkeypatch):
    items = [task("t1", "rain forecast kenya", turn_limit=3)]
    harness = Harness(tmp_path, items, reply_done_on(99))
    at_cap = judged_at_the_cap(monkeypatch)
    harness.run()

    scores = harness.scores()
    output = harness.output()
    assert output["outcome"] == "turn_cap"
    # Only at the cap is the judge told the conversation reached it.
    assert at_cap == [True]
    assert scores["goal_reached"] == 0.0
    assert scores["turns_to_goal"] == 3
    assert scores["model_calls_to_goal"] == 6
    assert "time_to_answer_seconds" in scores
    # The judge rules once, on the final result.
    assert len(harness.models.verdict_calls()) == 1


def test_reply_on_the_last_allowed_turn_still_counts(tmp_path, monkeypatch):
    # With a cap of 1 the only reply is read once more, without sending anything.
    items = [task("t1", "rain forecast kenya", turn_limit=1)]
    harness = Harness(tmp_path, items, reply_done_on(1))
    at_cap = judged_at_the_cap(monkeypatch)
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert harness.scores()["turns_to_goal"] == 1
    assert len(harness.app.completions) == 1
    assert len(harness.models.user_calls()) == 1
    assert at_cap == [False]


def test_the_read_at_the_cap_counts_no_classification_or_hint(tmp_path):
    items = [task("t1", "rain forecast kenya", turn_limit=1, hidden=["Use the weekly total"])]
    hint = progress("use the weekly total", classification="clarifying_question", hint=True)
    harness = Harness(
        tmp_path, items, reply_done_on(99), user_replies={"rain forecast kenya": [hint]}
    )
    harness.run()

    scores = harness.scores()
    assert harness.output()["outcome"] == "turn_cap"
    assert len(harness.models.user_calls()) == 1
    assert scores["clarifying_questions"] == 0
    assert scores["hints_used"] == 0


def test_a_done_or_give_up_reply_never_counts_a_hint(tmp_path):
    items = [task("t1", "rain forecast kenya", hidden=["Use the weekly total"])]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(99),
        user_replies={"rain forecast kenya": [{**GIVE_UP, "revealed_hidden_step": True}]},
    )
    harness.run()

    assert harness.output()["outcome"] == "user_gave_up"
    assert harness.scores()["hints_used"] == 0


def test_an_empty_message_is_usable_at_the_cap(tmp_path):
    items = [task("t1", "rain forecast kenya", turn_limit=1)]
    harness = Harness(
        tmp_path, items, reply_done_on(99), user_replies={"rain forecast kenya": [progress("")]}
    )
    harness.run()

    assert harness.output()["outcome"] == "turn_cap"
    assert len(harness.models.user_calls()) == 1


def test_a_message_that_is_not_text_is_unusable_at_the_cap(tmp_path):
    items = [task("t1", "rain forecast kenya", turn_limit=1)]
    bad = {**progress("x"), "message": 3}
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(99),
        user_replies={"rain forecast kenya": [bad, progress("")]},
    )
    harness.run()

    assert len(harness.models.user_calls()) == 2
    assert harness.output()["outcome"] == "turn_cap"


def test_a_give_up_at_the_cap_gives_its_reason(tmp_path):
    items = [task("t1", "rain forecast kenya", turn_limit=1)]
    harness = Harness(
        tmp_path, items, reply_done_on(99), user_replies={"rain forecast kenya": [GIVE_UP]}
    )
    harness.run()

    output = harness.output()
    assert output["outcome"] == "user_gave_up"
    assert GIVE_UP["give_up_reason"] in output["reasoning"]


def test_a_done_at_the_cap_ends_as_user_done(tmp_path):
    items = [task("t1", "rain data kenya", turn_limit=1, deliverables=["zarr"])]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.run()

    assert harness.output()["outcome"] == "user_done"


def test_a_run_that_ends_below_the_cap_is_not_reached(tmp_path, monkeypatch):
    # Scenario's run is cut to 1 turn while the task's cap is 3.
    real_arun = play_module.scenario.arun

    async def short_arun(**kwargs):
        return await real_arun(**{**kwargs, "max_turns": 1})

    monkeypatch.setattr(play_module.scenario, "arun", short_arun)
    items = [task("t1", "rain forecast kenya", turn_limit=3)]
    harness = Harness(tmp_path, items, reply_done_on(99))
    harness.run()

    assert harness.output()["outcome"] == "not_reached"
    assert harness.models.user_calls() == []


def test_task_without_turn_limit_runs_to_eval_max_turns(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(99))
    harness.run(EVAL_MAX_TURNS="6")

    assert harness.output()["outcome"] == "turn_cap"
    assert harness.scores()["turns_to_goal"] == 6


def test_eval_max_turns_has_its_default(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(99))
    harness.run()

    assert harness.output()["outcome"] == "turn_cap"
    assert harness.scores()["turns_to_goal"] == DEFAULT_MAX_TURNS


def test_task_turn_limit_overrides_eval_max_turns(tmp_path):
    items = [task("t1", "rain forecast kenya", turn_limit=8)]
    harness = Harness(tmp_path, items, reply_done_on(99))
    harness.run(EVAL_MAX_TURNS="3")

    assert harness.scores()["turns_to_goal"] == 8


def test_invalid_eval_max_turns_does_not_start_the_run(tmp_path):
    harness = Harness(tmp_path, [task("t1", "rain forecast kenya")], reply_done_on(1))

    with pytest.raises(SettingsError):
        harness.run(EVAL_MAX_TURNS="0")
    assert harness.docker.run_calls == []


def test_pause_past_eval_max_turns_does_not_start_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya", pauses=[{"before_turn": 7, "seconds": 1}])]
    harness = Harness(tmp_path, items, reply_done_on(1))

    with pytest.raises(TaskError):
        harness.run(EVAL_MAX_TURNS="6")
    assert harness.docker.run_calls == []


def test_done_on_the_last_allowed_turn_is_not_judged_as_the_turn_cap(tmp_path, monkeypatch):
    # With a cap of 2, the user's second turn is the last; it reads reply 1 and is done.
    items = [task("t1", "rain forecast kenya", turn_limit=2)]
    harness = Harness(tmp_path, items, reply_done_on(1))
    at_cap = judged_at_the_cap(monkeypatch)
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert harness.scores()["turns_to_goal"] == 1
    assert at_cap == [False]


def test_user_gives_up_ends_as_failure(tmp_path):
    items = [task("t1", "rain forecast kenya", turn_limit=5, hidden=["Use the weekly total"])]
    give_up = {
        "classification": "other",
        "revealed_hidden_step": False,
        "done": False,
        "give_up": True,
        "give_up_reason": "it keeps asking the same thing",
        "message": "",
    }
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(99),
        user_replies={
            "rain forecast kenya": [progress("try the weekly total", hint=True), give_up]
        },
    )
    harness.run()

    output = harness.output()
    scores = harness.scores()
    assert output["outcome"] == "user_gave_up"
    assert scores["goal_reached"] == 0.0
    assert scores["turns_to_goal"] == 2
    assert scores["hints_used"] == 1


GIVE_UP = {
    "classification": "other",
    "revealed_hidden_step": False,
    "done": False,
    "give_up": True,
    "give_up_reason": "it is not what I need",
    "message": "",
}

UNUSABLE_REPLIES = {
    "empty-message": progress(""),
    "unknown-classification": progress("go on", classification="request"),
    "not-json": "Sure, go on.",
    "done-and-give-up": {**GIVE_UP, "done": True},
    "done-not-boolean": {**progress("go on"), "done": "no"},
}


def test_a_passing_final_check_is_marked_on_a_failed_outcome(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path, items, reply_done_on(1), user_replies={"rain forecast kenya": [GIVE_UP]}
    )
    harness.run()

    output = harness.output()
    assert output["outcome"] == "user_gave_up"
    assert output["judges"][0]["passed"] is True


def test_final_judge_without_a_verdict_ends_as_error_with_scores(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), judge_without_verdict=True)
    harness.run()

    output = harness.output()
    scores = harness.scores()
    assert output["outcome"] == "error"
    assert [j["state"] for j in output["judges"]] == ["error"]
    assert scores["goal_reached"] == 0.0
    assert scores["turns_to_goal"] == 1
    assert output["trace_scores"] == "complete"
    assert "model_calls_to_goal" in scores


@pytest.mark.parametrize("unusable", UNUSABLE_REPLIES.values(), ids=UNUSABLE_REPLIES.keys())
def test_unusable_user_reply_is_asked_for_once_more(tmp_path, unusable):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(2),
        user_replies={"rain forecast kenya": [unusable, progress("kenya")]},
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    calls = harness.models.user_calls()
    assert len(calls) == 3
    # The second call is shown the first reply.
    assert calls[1]["messages"][:-2] == calls[0]["messages"]
    text = unusable if isinstance(unusable, str) else json.dumps(unusable)
    assert calls[1]["messages"][-2] == {"role": "assistant", "content": text}
    assert harness.app.completions[1]["turn"]["user_message"]["content"] == "kenya"


def test_two_unusable_user_replies_stop_the_play_as_error(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(2),
        user_replies={"rain forecast kenya": [progress(""), progress("")]},
    )
    harness.run()

    assert harness.output()["outcome"] == "error"
    assert harness.models.verdict_calls() == []


def test_a_reply_without_text_is_asked_for_again_without_an_empty_message(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(2),
        user_replies={"rain forecast kenya": [None, progress("kenya")]},
    )
    harness.run()

    calls = harness.models.user_calls()
    assert calls[1]["messages"][:-1] == calls[0]["messages"]
    assert harness.output()["outcome"] == "goal_reached"


def test_a_cut_off_quality_rating_is_not_rated_and_keeps_the_outcome(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    cut_off = ModelReplyError("The judge reply was cut off at max_tokens 8192")
    harness = Harness(tmp_path, items, reply_done_on(1), quality_error=cut_off)
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert "answer_quality" not in harness.scores()


def test_a_cut_off_verdict_ends_the_play_as_error(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    cut_off = ModelReplyError(
        "The judge reply was cut off at max_tokens 8192 "
        "(EVAL_MODEL_MAX_TOKENS unless the call set its own)"
    )
    harness = Harness(tmp_path, items, reply_done_on(1), verdict_error=cut_off)
    harness.run()

    output = harness.output()
    assert output["outcome"] == "error"
    assert [j["state"] for j in output["judges"]] == ["error"]


def test_a_cut_off_user_reply_is_not_retried_and_ends_as_error(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    cut_off = ModelReplyError(
        "The simulated-user reply was cut off at max_tokens 8192 "
        "(EVAL_MODEL_MAX_TOKENS unless the call set its own)"
    )
    harness = Harness(
        tmp_path, items, reply_done_on(2), user_replies={"rain forecast kenya": [cut_off]}
    )
    harness.run()

    output = harness.output()
    assert output["outcome"] == "error"
    assert "EVAL_MODEL_MAX_TOKENS" in output["error"]
    assert len(harness.models.user_calls()) == 1


def test_pause_waits_before_turn_and_is_excluded_from_time_to_answer(tmp_path):
    items = [task("t1", "rain forecast kenya", pauses=[{"before_turn": 2, "seconds": 3600}])]
    harness = Harness(tmp_path, items, reply_done_on(2))
    harness.run()

    assert 3600 in harness.clock.sleeps
    scores = harness.scores()
    assert scores["goal_reached"] == 1.0
    assert scores["turns_to_goal"] == 2
    assert scores["time_to_answer_seconds"] == 10.0
    assert harness.output()["paused_seconds"] == 3600


def test_chat_error_fails_that_play_and_others_continue(tmp_path):
    items = [task("t1", "broken request"), task("t2", "rain forecast kenya")]

    def reply(opening, number, text):
        if opening == "broken request":
            return Reply(error="Model not found")
        return Reply("DONE: rain totals are 20-40 mm.")

    harness = Harness(tmp_path, items, reply)
    harness.run()

    failed = harness.output(0)
    assert failed["outcome"] == "chat_error"
    assert "Model not found" in failed["error"]
    assert harness.scores(0)["goal_reached"] == 0.0
    # The failed turn reached the app, so its trace is awaited; no reply, no rating.
    assert harness.gateway.observation_requests[0] == ("chat-1", 1)
    assert "answer_quality" not in harness.scores(0)
    assert harness.scores(0)["time_to_answer_seconds"] == 0.0
    assert harness.output(1)["outcome"] == "goal_reached"
    assert harness.container.removed


def test_chat_http_error_fails_that_play_without_waiting_for_traces(tmp_path):
    items = [task("t1", "rain forecast kenya")]

    def reply(opening, number, text):
        return Reply(error="usage cap reached", http_status=429)

    harness = Harness(tmp_path, items, reply)
    harness.run()

    output = harness.output()
    assert output["outcome"] == "chat_error"
    assert "429" in output["error"]
    # The completion request was refused, so no trace is awaited.
    assert harness.gateway.observation_requests == []
    assert "model_calls_to_goal" not in harness.scores()
    assert "answer_quality" not in harness.scores()


def test_later_failed_turn_waits_only_for_turns_that_reached_the_app(tmp_path):
    items = [task("t1", "rain forecast kenya", turn_limit=4)]

    def reply(opening, number, text):
        if number == 2:
            return Reply(error="busy", http_status=409)
        return Reply("Which region do you need?")

    harness = Harness(tmp_path, items, reply)
    harness.run()

    assert harness.output()["outcome"] == "chat_error"
    assert harness.gateway.observation_requests == [("chat-1", 1)]
    scores = harness.scores()
    assert scores["turns_to_goal"] == 2
    assert scores["time_to_answer_seconds"] == 5.0
    # The first turn got a reply, so the answer is still rated.
    assert scores["answer_quality"] == 4


def test_langfuse_error_keeps_the_other_scores(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))

    def fail(chat_id, turns):
        raise TraceReadError("ReadTimeout: timed out")

    harness.gateway.observations = fail
    harness.run()

    scores = harness.scores()
    output = harness.output()
    assert scores["goal_reached"] == 1.0
    assert scores["turns_to_goal"] == 1
    assert "model_calls_to_goal" not in scores
    assert harness.gateway.app_costs == []
    assert output["provider_models"] == []


def test_catalog_model_without_base_model_stops_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.app.base_model_id = None

    with pytest.raises(ChatError):
        harness.run()

    assert harness.gateway.experiments == []
    assert harness.container.removed


def zero_cost_observations(chat_id, turns):
    observations = app_observations(chat_id, turns)
    for obs in observations:
        if obs["type"] == "GENERATION":
            obs.update(cost_details={"total": 0.0}, total_cost=0.0)
    return observations, True


@pytest.mark.parametrize(
    "observations, trace_scores, cost",
    [
        # One turn of app_observations: 0.02 + 0.19.
        (lambda chat_id, turns: (app_observations(chat_id, turns), False), "partial", 0.21),
        (zero_cost_observations, "complete", 0.0),
    ],
    ids=["partial", "zero-cost"],
)
def test_app_cost_rollup_matches_the_cost_score(tmp_path, observations, trace_scores, cost):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.observations = observations
    harness.run()

    assert harness.output()["trace_scores"].startswith(trace_scores)
    assert harness.scores()["cost_to_goal_usd"] == pytest.approx(cost)
    assert harness.gateway.app_costs == [(harness.scores()["cost_to_goal_usd"], ["chat-1-trace-0"])]


def test_reads_that_failed_at_the_timeout_keep_the_trace_scores(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.observations = lambda chat_id, turns: (
        app_observations(chat_id, turns),
        False,
        "ReadTimeout: x",
    )
    harness.run()

    assert harness.output()["trace_scores"].startswith("partial")
    assert "model_calls_to_goal" in harness.scores()


def test_reads_that_failed_without_observations_leave_out_the_trace_scores(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.observations = lambda chat_id, turns: ([], False, "ReadTimeout: x")
    harness.run()

    assert "model_calls_to_goal" not in harness.scores()


def test_items_without_a_cost_score_add_nothing_to_the_agent_cost(tmp_path):
    items = [task("t1", "rain forecast kenya"), task("t2", "weekly rain")]
    harness = Harness(tmp_path, items, reply_done_on(1))

    def first_chat_unread(chat_id, turns):
        if chat_id == "chat-1":
            raise TraceReadError("ReadTimeout: timed out")
        return app_observations(chat_id, turns), True

    harness.gateway.observations = first_chat_unread
    summary = harness.run()

    assert "cost_to_goal_usd" not in harness.scores(0)
    assert summary.agent_cost_usd == pytest.approx(harness.scores(1)["cost_to_goal_usd"])


def test_traces_that_never_arrive_leave_out_the_trace_scores(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.observations = lambda chat_id, turns: ([], False)
    harness.run()

    assert harness.output()["trace_scores"].startswith("missing")
    assert "model_calls_to_goal" not in harness.scores()
    assert harness.scores()["goal_reached"] == 1.0


def test_broadcast_generations_use_metadata_cache_counts(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.observations = lambda chat_id, turns: (broadcast_observations(chat_id), True)
    harness.run()

    scores = harness.scores()
    # A fully cached 80k call and a 60k call that missed the cache.
    assert scores["tokens_cache_read"] == 80000
    assert scores["tokens_cache_write"] == 60000
    assert scores["tokens_uncached_input"] == 0
    assert scores["cache_miss_calls"] == 1
    # cost_details has no total on one call: its total cost is used.
    assert scores["cost_to_goal_usd"] == pytest.approx(0.016 + 0.2)


def test_hints_never_exceed_the_hidden_steps(tmp_path):
    items = [task("t1", "rain forecast kenya", turn_limit=4, hidden=["Use the weekly total"])]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(99),
        user_replies={"rain forecast kenya": [progress("hint", hint=True)] * 3},
    )
    harness.run()

    assert harness.scores()["hints_used"] == 1


def test_image_pull_failure_stops_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(1),
        pull_error=docker.errors.ImageNotFound("manifest unknown"),
    )

    with pytest.raises(AppStartError) as error:
        harness.run()

    # The app's own error, not the runner's.
    assert type(error.value) is AppStartError
    assert harness.docker.pulled == ["ghcr.io/rhiza-research/weather-skills-chat:test"]
    assert harness.docker.run_calls == []
    assert harness.docker.events == []
    assert harness.gateway.experiments == []
    assert harness.gateway.flushed


def test_pull_false_runs_a_local_image_without_a_digest(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), local=local_image())
    harness.config_path.write_text(CONFIG + "pull: false\n")

    summary = harness.run()

    # The runner image is pulled whatever pull says.
    assert harness.docker.pulled == [RUNNER_IMAGE]
    assert harness.docker.looked_up == ["ghcr.io/rhiza-research/weather-skills-chat:test"]
    assert harness.docker.run_calls[0]["image"] == "sha256:local"
    assert harness.docker.source.image_id == "sha256:local"
    assert harness.docker.create_calls[0]["image"] == "sha256:runner"
    metadata = harness.gateway.experiments[0]["metadata"]
    assert metadata["image_pulled"] == "false"
    assert metadata["image_id"] == "sha256:local"
    assert metadata["image_digest"] == ""
    assert metadata["runner_image_id"] == "sha256:runner"
    assert metadata["runner_image_digest"] == f"{RUNNER_REPOSITORY}@sha256:run"
    assert summary.goals_reached == 1
    assert harness.container.removed


def test_pull_false_records_the_local_image_repo_digest(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    local = local_image(["ghcr.io/other/mirror@sha256:111", f"{APP_REPOSITORY}@sha256:def"])
    harness = Harness(tmp_path, items, reply_done_on(1), local=local)
    harness.config_path.write_text(CONFIG + "pull: false\n")

    harness.run()

    assert harness.docker.pulled == [RUNNER_IMAGE]
    metadata = harness.gateway.experiments[0]["metadata"]
    assert metadata["image_digest"] == f"{APP_REPOSITORY}@sha256:def"
    assert metadata["image_id"] == "sha256:local"


APP_IMAGE_LINE = "image: ghcr.io/rhiza-research/weather-skills-chat:test"


def test_pull_false_records_the_pinned_digest(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    pinned = f"{APP_REPOSITORY}@sha256:second"
    local = local_image([f"{APP_REPOSITORY}@sha256:first", pinned])
    harness = Harness(tmp_path, items, reply_done_on(1), local=local)
    config = CONFIG.replace(APP_IMAGE_LINE, f"image: {pinned}") + "pull: false\n"
    harness.config_path.write_text(config)

    harness.run()

    assert harness.docker.looked_up == [pinned]
    metadata = harness.gateway.experiments[0]["metadata"]
    assert metadata["image_digest"] == pinned


def test_a_pulled_image_records_the_pinned_digest(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    pinned = f"{APP_REPOSITORY}@sha256:second"
    digests = [f"{APP_REPOSITORY}@sha256:first", pinned]
    harness = Harness(tmp_path, items, reply_done_on(1), image_digests=digests)
    harness.config_path.write_text(CONFIG.replace(APP_IMAGE_LINE, f"image: {pinned}"))

    harness.run()

    assert harness.docker.pulled == [pinned, RUNNER_IMAGE]
    metadata = harness.gateway.experiments[0]["metadata"]
    assert metadata["image_digest"] == pinned


def test_pull_false_with_missing_image_stops_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), local=None)
    harness.config_path.write_text(CONFIG + "pull: false\n")

    with pytest.raises(AppStartError) as error:
        harness.run()

    # The app's own error, not the runner's.
    assert type(error.value) is AppStartError
    assert harness.docker.events == []
    assert harness.docker.pulled == []
    assert harness.docker.run_calls == []
    assert harness.gateway.experiments == []
    assert harness.gateway.flushed


def test_pull_false_with_failing_image_lookup_stops_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(1),
        get_error=docker.errors.APIError("daemon unavailable"),
    )
    harness.config_path.write_text(CONFIG + "pull: false\n")

    with pytest.raises(AppStartError):
        harness.run()

    assert harness.docker.pulled == []
    assert harness.docker.run_calls == []
    assert harness.gateway.experiments == []


def test_container_start_failure_removes_the_container_by_name(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(1),
        run_error=docker.errors.APIError("network not found"),
    )

    with pytest.raises(AppStartError, match="network not found"):
        harness.run()

    assert harness.container.removed


def test_cleanup_error_does_not_replace_the_start_error(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    container = FakeContainer(
        ["exited"], logs=b"bad SKILL_PACKS", remove_error=docker.errors.APIError("busy")
    )
    harness = Harness(tmp_path, items, reply_done_on(1), container=container)

    with pytest.raises(AppStartError, match="bad SKILL_PACKS"):
        harness.run()


def test_container_that_disappears_stops_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), container=FakeContainer(["gone"]))

    with pytest.raises(AppStartError):
        harness.run()


def test_another_runs_container_is_left_alone(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    other = FakeContainer(["running"], name="weather-skills-eval-app-other")
    harness = Harness(tmp_path, items, reply_done_on(1), leftovers=[other])
    harness.run()

    assert harness.container.removed
    assert not other.removed


def test_missing_dataset_does_not_start_the_run(tmp_path):
    harness = Harness(tmp_path, [], reply_done_on(1))
    harness.gateway.dataset_error = RuntimeError("404 dataset not found")

    with pytest.raises(DatasetError):
        harness.run()
    assert harness.docker.run_calls == []


def test_empty_dataset_does_not_start_the_run(tmp_path):
    harness = Harness(tmp_path, [], reply_done_on(1))

    with pytest.raises(DatasetError):
        harness.run()
    assert harness.docker.run_calls == []


# Skill packs a dataset needs


@pytest.mark.parametrize("metadata", [None, {}, {"skill_packs": []}])
def test_dataset_needing_no_pack_plays_every_item(tmp_path, metadata):
    items = [task("t1", "rain forecast kenya"), task("t2", "weekly rain")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.dataset_metadata = metadata

    summary = harness.run()

    assert summary.items == 2
    assert harness.gateway.experiments[0]["metadata"]["dataset_skill_packs"] == ""


def test_dataset_packs_that_the_config_turns_on_play_every_item(tmp_path):
    items = [task("t1", "rain forecast kenya"), task("t2", "weekly rain")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.dataset_metadata = {"skill_packs": ["chc-skills"]}
    harness.repo.packs([pack(), pack(CHC_URL)])

    summary = harness.run()

    assert summary.items == 2
    assert len(harness.docker.run_calls) == 1
    assert harness.gateway.experiments[0]["metadata"]["dataset_skill_packs"] == "chc-skills"


def test_only_the_packs_the_config_does_not_turn_on_are_named(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.dataset_metadata = {"skill_packs": ["chc-skills", "a-skills", "b-skills"]}
    harness.repo.packs([pack(), pack(CHC_URL)])

    with pytest.raises(DatasetError) as error:
        harness.run()

    message = str(error.value)
    assert "a-skills" in message
    assert "b-skills" in message
    assert "chc-skills" not in message
    assert harness.docker.run_calls == []


def test_dataset_metadata_that_is_not_an_object_does_not_start_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.dataset_metadata = ["chc-skills"]

    with pytest.raises(DatasetError):
        harness.run()
    assert harness.docker.run_calls == []


@pytest.mark.parametrize(
    "packs",
    [[pack(), pack(CHC_URL, enabled_by_default=False)], [pack()], []],
    ids=["pack-off", "pack-not-listed", "no-packs"],
)
def test_dataset_pack_the_commit_does_not_turn_on_does_not_start_the_run(tmp_path, packs):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.dataset_metadata = {"skill_packs": ["chc-skills"]}
    harness.repo.packs(packs)

    with pytest.raises(DatasetError):
        harness.run()

    assert harness.docker.run_calls == []
    assert harness.gateway.experiments == []


def test_bad_dataset_skill_packs_does_not_start_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.dataset_metadata = {"skill_packs": [""]}

    with pytest.raises(DatasetError):
        harness.run()
    assert harness.docker.run_calls == []


def test_duplicate_item_ids_do_not_start_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya"), task("t1", "weekly rain")]
    harness = Harness(tmp_path, items, reply_done_on(1))

    with pytest.raises(TaskError):
        harness.run()
    assert harness.docker.run_calls == []


def test_app_container_exits_stops_the_run_with_logs(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    container = FakeContainer(["exited"], logs=b"CATALOG_MODELS is not valid JSON")
    harness = Harness(tmp_path, items, reply_done_on(1), container=container)

    with pytest.raises(AppStartError, match="CATALOG_MODELS is not valid JSON"):
        harness.run()

    assert container.removed
    assert harness.gateway.experiments == []


def test_app_never_healthy_stops_the_run_with_logs(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    container = FakeContainer(["running"] * 10_000, logs=b"still loading models")
    harness = Harness(tmp_path, items, reply_done_on(1), container=container)

    with pytest.raises(AppStartError, match="still loading models"):
        harness.run(health=lambda url: False)

    assert container.removed
    assert harness.gateway.experiments == []


@pytest.mark.parametrize(
    "edit, field",
    [
        (
            lambda text: text.replace(
                "image: ghcr.io/rhiza-research/weather-skills-chat:test\n", ""
            ),
            "image",
        ),
        (lambda text: text.replace("${APP_SECRET}", "literal"), "env.WEBUI_SECRET_KEY"),
        (lambda text: text + "replicas: 2\n", "replicas"),
        (lambda text: text + 'pull: "no"\n', "pull"),
        (lambda text: text.replace(RUNNER_IMAGE, '""'), "runner_image"),
        (lambda text: text.replace(RUNNER_IMAGE, "${UNSET_RUNNER}"), "runner_image"),
        (lambda text: text + "variant: 3\n", "variant"),
    ],
)
def test_bad_config_does_not_start_the_run(tmp_path, edit, field):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.config_path.write_text(edit(CONFIG))

    with pytest.raises(ConfigError) as error:
        harness.run()

    assert error.value.field == field
    assert harness.docker.run_calls == []
    assert harness.gateway.experiments == []
    assert harness.docker.pulled == []


# The sandbox runner


def with_env(text: str, block: str) -> str:
    """CONFIG with ``block`` added to its env, which it ends with."""
    return text + block


def test_config_setting_a_runner_variable_does_not_start_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    name = RUNNER_ENV[0]
    harness.config_path.write_text(with_env(CONFIG, f'  {name}: "x"\n'))

    with pytest.raises(ConfigError) as error:
        harness.run()

    assert error.value.field == f"env.{name}"
    assert harness.docker.pulled == []


def archive_files(data: bytes) -> dict[str, bytes]:
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        return {m.name: tar.extractfile(m).read() for m in tar.getmembers() if m.isfile()}


def test_run_starts_the_runner_and_its_volume_before_the_app(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))

    harness.run()

    docker_ = harness.docker
    [create] = docker_.create_calls
    [app] = docker_.run_calls
    run_id = app["labels"][RUN_LABEL]
    runner_name = f"weather-skills-eval-runner-{run_id}"
    volume_name = f"weather-skills-eval-data-{run_id}"
    source_name = f"weather-skills-eval-emulator-{run_id}"
    # The emulator is read from a container of the app image, never started and
    # removed at once; the runner and the volume come next, then the app; all
    # go when the run ends, the volume after the containers that use it.
    source = docker_.source
    assert source.image_id == "sha256:image"
    assert source.labels == {LABEL: EMULATOR_LABEL_VALUE, RUN_LABEL: run_id}
    assert source.read == list(APP_EMULATOR_PATHS)
    assert not source.started
    assert source.removed
    setup_name = f"weather-skills-eval-volume-setup-{run_id}"
    assert docker_.events == [
        f"create {source_name}",
        f"remove {source_name}",
        f"volume {volume_name}",
        f"create {setup_name}",
        f"remove {setup_name}",
        f"create {runner_name}",
        f"run {app['name']}",
        f"remove {app['name']}",
        f"remove {runner_name}",
        f"remove {volume_name}",
    ]
    assert docker_.volume().labels == {LABEL: VOLUME_LABEL_VALUE, RUN_LABEL: run_id}
    # The artifacts folder is created on the volume first, from a container of
    # the app image with the whole volume at the app's data directory, so the
    # volume's first mount fills it from the app image; never started.
    [setup] = docker_.setup_calls
    assert setup == {
        "image": "sha256:image",
        "name": setup_name,
        "labels": {LABEL: VOLUME_SETUP_LABEL_VALUE, RUN_LABEL: run_id},
        "volumes": {volume_name: {"bind": APP_DATA_DIR, "mode": "rw"}},
    }
    [(setup_path, setup_data)] = docker_.volume_setup.archives
    assert setup_path == APP_DATA_DIR
    with tarfile.open(fileobj=io.BytesIO(setup_data)) as tar:
        assert [(m.name, m.isdir()) for m in tar.getmembers()] == [(ARTIFACTS_SUBPATH, True)]
    assert not docker_.volume_setup.started
    assert docker_.volume_setup.removed
    # The runner: the runner image, the emulator, and only the volume's
    # artifacts folder, at the path the emulator reads.
    assert create["image"] == "sha256:runner"
    assert create["name"] == runner_name
    assert create["user"] == "0"
    assert create["command"] == ["python3", "-u", f"{EMULATOR_DIR}/emulator.py"]
    assert create["network"] == "weather-skills-eval"
    assert create["labels"] == {LABEL: RUNNER_LABEL_VALUE, RUN_LABEL: run_id}
    assert "volumes" not in create
    assert create["mounts"] == [
        {
            "Target": RUNNER_ARTIFACTS_DIR,
            "Source": volume_name,
            "Type": "volume",
            "ReadOnly": False,
            "VolumeOptions": {"Subpath": ARTIFACTS_SUBPATH},
        }
    ]
    assert create["restart_policy"] == {"Name": "on-failure"}
    token = create["environment"]["SKILL_K8S_BIND_TOKEN"]
    assert len(token) == 2 * TOKEN_BYTES
    assert create["environment"] == {
        "ARTIFACTS_DIR": RUNNER_ARTIFACTS_DIR,
        "SKILL_K8S_EMULATOR_HOST": runner_name,
        "SKILL_K8S_EMULATOR_API_PORT": EMULATOR_API_PORT,
        "SKILL_K8S_BIND_PORT": EMULATOR_BIND_PORT,
        "SKILL_K8S_EMULATOR_START_DELAY": START_DELAY,
        "SKILL_K8S_BIND_TOKEN": token,
    }
    # The emulator files reach the runner by put_archive, before it starts.
    [(path, data)] = docker_.runner.archives
    assert path == "/"
    assert archive_files(data) == {
        f"{EMULATOR_DIR.lstrip('/')}/{Path(app_path).name}": emulator_file(app_path)
        for app_path in APP_EMULATOR_PATHS
    }
    assert docker_.runner.started
    # The app: the emulator settings with this run's runner and token, the volume at its data dir.
    assert app["volumes"] == {volume_name: {"bind": APP_DATA_DIR, "mode": "rw"}}
    env = app["environment"]
    assert {name: env[name] for name in RUNNER_ENV} == {
        "SKILL_SANDLOCK": "false",
        "SKILL_SANDBOX_BACKEND": "k8s",
        "SKILL_K8S_SANDBOX_ENABLED": "true",
        "SKILL_K8S_EMULATOR": "true",
        "SKILL_K8S_APISERVER": f"http://{runner_name}:{EMULATOR_API_PORT}",
        "SKILL_K8S_BIND_TOKEN": token,
    }
    assert env["BOOTSTRAP_ADMIN_API_KEY"] == "sk-test-key"
    # The token is not recorded.
    metadata = harness.gateway.experiments[0]["metadata"]
    # The harness adds exactly the variables a configuration may not set.
    assert set(env) - set(metadata["env_var_names"].split(",")) == set(RUNNER_ENV)
    assert token not in json.dumps(metadata)
    assert "SKILL_K8S_BIND_TOKEN" not in metadata["env_var_names"]


def test_each_run_gets_its_own_bind_token(tmp_path):
    tokens = []
    for _ in range(2):
        harness = Harness(tmp_path, [task("t1", "rain forecast kenya")], reply_done_on(1))
        harness.run()
        tokens.append(harness.docker.create_calls[0]["environment"]["SKILL_K8S_BIND_TOKEN"])
    assert tokens[0] != tokens[1]


def test_app_image_without_the_emulator_stops_the_run_before_the_runner(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), app_files={})

    with pytest.raises(RunnerStartError):
        harness.run()

    # Neither the runner image nor any runner, volume or app.
    assert harness.docker.pulled == ["ghcr.io/rhiza-research/weather-skills-chat:test"]
    assert harness.docker.create_calls == []
    assert harness.docker.volumes_created == {}
    assert harness.docker.run_calls == []
    assert harness.docker.source.removed
    assert harness.gateway.experiments == []
    assert harness.gateway.flushed


def test_emulator_read_error_stops_the_run_before_the_runner(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path, items, reply_done_on(1), archive_error=docker.errors.APIError("daemon busy")
    )

    with pytest.raises(RunnerStartError):
        harness.run()

    assert harness.docker.create_calls == []
    assert harness.docker.source.removed


def test_unreadable_emulator_archive_stops_the_run_before_the_runner(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path, items, reply_done_on(1), archive_stream=lambda: iter([b"not a tar archive"])
    )

    with pytest.raises(RunnerStartError):
        harness.run()

    assert harness.docker.create_calls == []
    assert harness.docker.source.removed


def test_runner_image_pull_failure_stops_the_run_before_the_app(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(1),
        local=local_image(),
        runner_pull_error=docker.errors.ImageNotFound("manifest unknown"),
    )
    harness.config_path.write_text(CONFIG + "pull: false\n")

    with pytest.raises(RunnerStartError):
        harness.run()

    assert harness.docker.pulled == [RUNNER_IMAGE]
    assert harness.docker.create_calls == []
    assert harness.docker.volumes_created == {}
    assert harness.docker.run_calls == []
    assert harness.gateway.experiments == []


def test_runner_start_failure_stops_the_run_and_removes_what_it_created(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path, items, reply_done_on(1), create_error=docker.errors.APIError("no such network")
    )

    with pytest.raises(RunnerStartError):
        harness.run()

    assert harness.docker.run_calls == []
    for name, container in harness.docker.created.items():
        assert container.removed, name
    for volume in harness.docker.volumes_created.values():
        assert volume.removed
    assert harness.gateway.experiments == []


def test_an_artifacts_folder_that_cannot_be_created_stops_the_run_before_the_runner(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    setup = FakeContainer([], put_archive_result=False)
    harness = Harness(tmp_path, items, reply_done_on(1), volume_setup=setup)

    with pytest.raises(RunnerStartError):
        harness.run()

    assert setup.removed
    assert harness.docker.create_calls == []
    assert harness.docker.run_calls == []
    assert harness.docker.volume().removed
    assert harness.gateway.experiments == []


def test_emulator_files_that_put_archive_does_not_copy_stop_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    runner = FakeContainer([], put_archive_result=False)
    harness = Harness(tmp_path, items, reply_done_on(1), runner=runner)

    with pytest.raises(RunnerStartError):
        harness.run()

    assert not runner.started
    assert harness.docker.run_calls == []
    for name, container in harness.docker.created.items():
        assert container.removed, name
    assert harness.docker.volume().removed
    assert harness.gateway.experiments == []


def test_runner_that_exits_while_the_app_starts_stops_the_run_with_its_logs(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    runner = FakeContainer(["exited"], logs=b"missing /usr/local/bin/sandbox_runtime.py")
    harness = Harness(tmp_path, items, reply_done_on(1), runner=runner)

    with pytest.raises(RunnerStartError, match="sandbox_runtime.py"):
        harness.run()

    assert runner.removed
    assert harness.container.removed
    assert harness.docker.volume().removed


def test_runner_that_disappears_stops_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), runner=FakeContainer(["gone"]))

    with pytest.raises(RunnerStartError):
        harness.run()

    assert harness.container.removed
    assert harness.docker.volume().removed


THREE_TASKS = [
    task("t1", "rain forecast kenya"),
    task("t2", "rain map kenya"),
    task("t3", "weekly rain"),
]


# Running at /health, and through t1: two polls before it and one after.
THROUGH_T1 = ["running"] * 4
SHORT_WAIT = {"EVAL_APP_START_TIMEOUT_SECONDS": "30", "EVAL_POLL_INTERVAL_SECONDS": "5"}


def test_runner_that_dies_and_restarts_plays_the_task_on_the_restarted_runner(tmp_path, caplog):
    # Exited before t2, then Docker restarts it once: running twice with one restart.
    restarted = ("running", 1, 0)
    runner = FakeContainer([*THROUGH_T1, "exited", "restarting", restarted, restarted])
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)

    with caplog.at_level("WARNING"):
        summary = harness.run()

    assert summary.items == 3 and summary.goals_reached == 3
    assert len(harness.app.chats) == 3
    runner_name = harness.docker.create_calls[0]["name"]
    waits = [r.args for r in caplog.records if r.name == "weather_skills_eval.instance"]
    assert waits == [(runner_name, "before task t2", DEFAULT_APP_START_TIMEOUT_SECONDS)]
    # /health, 3 for t1, 4 polls and 1 check for t2, 3 for t3.
    assert runner.reloads == 12
    assert runner.removed


def test_runner_that_does_not_come_back_stops_the_run_with_its_logs(tmp_path):
    # Exited before t2 and never restarted.
    runner = FakeContainer([*THROUGH_T1, "exited"], logs=b"emulator crashed")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)

    with pytest.raises(RunnerStartError, match="emulator crashed"):
        harness.run(**SHORT_WAIT)

    # /health, 3 for t1, then every 5 s from 0 to 30 s.
    assert runner.reloads == 4 + 7
    assert len(harness.app.chats) == 1
    assert runner.removed
    assert harness.container.removed
    assert harness.docker.volume().removed


@pytest.mark.parametrize(
    "statuses",
    [["running", "gone"], ["running", "api-error"]],
    ids=["gone", "reload-error"],
)
def test_runner_lost_mid_run_stops_the_run(tmp_path, statuses):
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=FakeContainer(statuses))

    with pytest.raises(RunnerStartError):
        harness.run()

    assert harness.app.chats == {}
    assert harness.container.removed
    assert harness.docker.volume().removed


def keep_task_errors(harness, tasks_raised, then_raise=None):
    """Make the experiment record a task's runner error on its item and go on,
    leaving that item out of its results as the Langfuse SDK does, then raise
    ``then_raise`` when given."""
    record = harness.gateway.run_experiment

    def run_experiment(*, task, **kwargs):
        async def keeping(*, item, **task_kwargs):
            try:
                return await task(item=item, **task_kwargs)
            except RunnerStartError as exc:
                tasks_raised.append((item.id, exc))
                # A task never returns None; it marks the item to leave out.
                return None

        result = record(task=keeping, **kwargs)
        # The experiment's items are the same list.
        result.item_results[:] = [r for r in result.item_results if r.output is not None]
        if then_raise:
            raise then_raise
        return result

    harness.gateway.run_experiment = run_experiment


def test_runner_death_stops_the_run_when_the_experiment_keeps_task_errors(tmp_path):
    runner = FakeContainer([*THROUGH_T1, "exited"], logs=b"emulator crashed")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)
    tasks_raised = []
    keep_task_errors(harness, tasks_raised)

    with pytest.raises(RunnerStartError):
        harness.run(**SHORT_WAIT)

    # t3 is not played: it raises the same error at once.
    assert [item_id for item_id, _ in tasks_raised] == ["t2", "t3"]
    assert tasks_raised[0][1] is tasks_raised[1][1]
    assert len(harness.app.chats) == 1
    assert harness.container.removed
    assert harness.docker.volume().removed


def test_a_later_experiment_error_does_not_hide_the_runner_failure(tmp_path):
    runner = FakeContainer([*THROUGH_T1, "exited"], logs=b"emulator crashed")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)
    keep_task_errors(harness, [], then_raise=RuntimeError("Langfuse flush failed"))

    with pytest.raises(RunnerStartError, match="emulator crashed") as error:
        harness.run(**SHORT_WAIT)

    assert isinstance(error.value.__context__, RuntimeError)


def test_runner_crashing_while_the_app_starts_stops_the_run_with_its_logs(tmp_path):
    # With restart policy on-failure, a runner that crashes at start is restarting.
    runner = FakeContainer(["restarting"], logs=b"ImportError: local_runtime")
    harness = Harness(
        tmp_path, [task("t1", "rain forecast kenya")], reply_done_on(1), runner=runner
    )

    with pytest.raises(RunnerStartError, match="ImportError: local_runtime"):
        harness.run()

    assert harness.app.chats == {}
    assert runner.removed


def test_runner_in_a_crash_loop_stops_the_run_by_the_timeout(tmp_path):
    # Before t2 Docker keeps restarting it: running is never seen twice with one count.
    loop = [("running" if n % 2 else "restarting", n, 0 if n % 2 else 1) for n in range(1, 40)]
    runner = FakeContainer([*THROUGH_T1, *loop], logs=b"emulator crashed")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)

    with pytest.raises(RunnerStartError, match="emulator crashed"):
        harness.run(**SHORT_WAIT)

    assert len(harness.app.chats) == 1


def test_runner_that_restarts_on_every_poll_is_not_taken_as_running(tmp_path):
    # Running on each poll, but with a new restart count each time.
    loop = [("running", n, 0) for n in range(1, 40)]
    runner = FakeContainer([*THROUGH_T1, *loop])
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)

    with pytest.raises(RunnerStartError):
        harness.run(**SHORT_WAIT)


def test_runner_that_exits_with_code_0_stops_the_run_at_once(tmp_path):
    # Docker does not restart a container that exited with code 0.
    runner = FakeContainer([*THROUGH_T1, ("exited", 0, 0)], logs=b"emulator stopped")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)

    with pytest.raises(RunnerStartError, match="emulator stopped"):
        harness.run(**SHORT_WAIT)

    assert runner.reloads == 5
    assert len(harness.app.chats) == 1


def test_a_dead_runner_stops_the_run_at_once(tmp_path):
    # Docker never restarts a dead container.
    runner = FakeContainer([*THROUGH_T1, "dead"], logs=b"emulator removal failed")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)

    with pytest.raises(RunnerStartError, match="emulator removal failed"):
        harness.run(**SHORT_WAIT)

    assert runner.reloads == 5
    assert len(harness.app.chats) == 1
    assert runner.removed


def test_a_start_timeout_shorter_than_the_poll_interval_plays_the_task(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    runner = harness.docker.runner
    reload = runner.reload

    def slow_reload():
        # Each read of the runner takes longer than the start timeout.
        reload()
        harness.clock.now += 2

    runner.reload = slow_reload

    summary = harness.run(EVAL_APP_START_TIMEOUT_SECONDS="1", EVAL_POLL_INTERVAL_SECONDS="5")

    assert summary.goals_reached == 1
    # /health, two polls before t1 and one check after it.
    assert runner.reloads == 4


# Running at /health and on both polls before t1.
BEFORE_T1 = ["running"] * 3
RESTARTED = ("running", 1, 0)


@pytest.mark.parametrize(
    "after_t1",
    [RESTARTED, "restarting", "exited"],
    ids=["restarted", "restarting", "exited"],
)
def test_runner_that_restarts_during_a_task_costs_only_that_task(tmp_path, after_t1):
    # Not the same runner after t1; running steadily with one restart before t2.
    runner = FakeContainer([*BEFORE_T1, after_t1, RESTARTED], logs=b"emulator crashed")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)
    tasks_raised = []
    keep_task_errors(harness, tasks_raised)

    summary = harness.run()

    # t1 carries the runner error and no scores, and its item trace is linked
    # to its chat, so delete finds the chat's traces.
    [(item_id, error)] = tasks_raised
    assert item_id == "t1"
    assert "emulator crashed" in str(error)
    assert error.restartable
    # t2 and t3 are played on the restarted runner and scored; t1 is left out.
    assert [r.item.id for r in harness.gateway.experiments[0]["items"]] == ["t2", "t3"]
    assert harness.scores(0)["goal_reached"] == 1.0
    assert harness.scores(1)["goal_reached"] == 1.0
    assert harness.gateway.linked == ["chat-1", "chat-2", "chat-3"]
    assert [ids for _, ids in harness.gateway.app_costs] == [
        ["chat-2-trace-0"],
        ["chat-3-trace-0"],
    ]
    assert summary.items == 2 and summary.goals_reached == 2
    assert summary.runner_lost == ["t1"]
    # Only the scored items' costs.
    assert summary.agent_cost_usd == pytest.approx(2 * 0.21)
    # /health, then two polls and one check per task; t2's wait found the
    # runner back, so it is not waited for after the last task.
    assert runner.reloads == 10
    assert runner.removed
    assert harness.container.removed
    assert harness.docker.volume().removed


def test_runner_that_restarts_during_a_task_and_never_comes_back_stops_the_run(tmp_path):
    runner = FakeContainer([*BEFORE_T1, "exited"], logs=b"emulator crashed")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)
    tasks_raised = []
    keep_task_errors(harness, tasks_raised)

    with pytest.raises(RunnerStartError, match="emulator crashed") as error:
        harness.run(**SHORT_WAIT)

    # t1 is not scored; t2 and t3 are not played.
    assert [item_id for item_id, _ in tasks_raised] == ["t1", "t2", "t3"]
    assert tasks_raised[0][1].restartable
    assert tasks_raised[1][1] is error.value is tasks_raised[2][1]
    assert harness.gateway.linked == ["chat-1"]
    assert len(harness.app.chats) == 1
    assert harness.container.removed
    assert harness.docker.volume().removed


@pytest.mark.parametrize(
    "after_t1",
    [("exited", 0, 0), "gone", "api-error"],
    ids=["exited-0", "gone", "reload-error"],
)
def test_runner_found_stopped_for_good_after_a_task_stops_the_run_at_once(tmp_path, after_t1):
    runner = FakeContainer([*BEFORE_T1, after_t1], logs=b"emulator stopped")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)
    tasks_raised = []
    keep_task_errors(harness, tasks_raised)

    with pytest.raises(RunnerStartError) as error:
        harness.run(**SHORT_WAIT)

    assert not error.value.restartable
    # t2 and t3 raise the same error without waiting for the runner.
    assert [item_id for item_id, _ in tasks_raised] == ["t1", "t2", "t3"]
    assert all(exc is error.value for _, exc in tasks_raised)
    # /health, two polls before t1, one check after it.
    assert runner.reloads == 4
    assert harness.gateway.linked == ["chat-1"]
    assert len(harness.app.chats) == 1
    assert harness.container.removed
    assert harness.docker.volume().removed


def test_runner_restarted_during_the_last_task_is_waited_for_after_it(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    runner = FakeContainer([*BEFORE_T1, RESTARTED])
    harness = Harness(tmp_path, items, reply_done_on(1), runner=runner)
    tasks_raised = []
    keep_task_errors(harness, tasks_raised)

    summary = harness.run()

    assert [item_id for item_id, _ in tasks_raised] == ["t1"]
    assert summary.items == 0 and summary.runner_lost == ["t1"]
    # /health, two polls before t1, one check after it, two polls after the last task.
    assert runner.reloads == 6


def test_runner_lost_on_the_last_task_that_never_comes_back_stops_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    runner = FakeContainer([*BEFORE_T1, "exited"], logs=b"emulator crashed")
    harness = Harness(tmp_path, items, reply_done_on(1), runner=runner)
    keep_task_errors(harness, [])

    with pytest.raises(RunnerStartError, match="emulator crashed"):
        harness.run(**SHORT_WAIT)

    assert harness.container.removed
    assert harness.docker.volume().removed


def test_a_task_waiting_when_another_stops_the_run_is_not_played(tmp_path, monkeypatch):
    # Concurrent tasks: t2 is still waiting for the runner when the check after
    # t1 finds it exited with code 0.
    runner = FakeContainer(["running", ("exited", 0, 0)], logs=b"emulator stopped")
    harness = Harness(tmp_path, THREE_TASKS[:2], reply_done_on(1), runner=runner)
    t1_done = threading.Event()

    def wait_for_runner(self, when):
        if when == "before task t2":
            assert t1_done.wait(10)
        return 0

    monkeypatch.setattr(AppInstance, "wait_for_runner", wait_for_runner)
    outcomes = []

    def run_experiment(*, dataset, task, **kwargs):
        first, second = dataset.items

        async def t1():
            try:
                return await task(item=first)
            finally:
                t1_done.set()

        async def both():
            return await asyncio.gather(t1(), task(item=second), return_exceptions=True)

        outcomes.extend(asyncio.run(both()))
        return SimpleNamespace(run_name="r", experiment_url=None, item_results=[])

    harness.gateway.run_experiment = run_experiment

    with pytest.raises(RunnerStartError) as error:
        harness.run()

    assert outcomes == [error.value, error.value]
    assert len(harness.app.chats) == 1


def test_failed_run_removes_the_app_runner_and_volume(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.app.base_model_id = None

    with pytest.raises(ChatError):
        harness.run()

    assert harness.container.removed
    assert harness.docker.runner.removed
    assert harness.docker.volume().removed


def test_cleanup_errors_are_logged_and_do_not_replace_the_run_error(tmp_path, caplog):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(1),
        container=FakeContainer(["exited"], logs=b"bad env", remove_error=RuntimeError("busy")),
        runner=FakeContainer(["running"], remove_error=docker.errors.APIError("busy")),
        volume_remove_error=docker.errors.APIError("volume is in use"),
    )

    with pytest.raises(AppStartError, match="bad env"):
        harness.run()

    # The app container, the runner container and the volume.
    errors = [
        r
        for r in caplog.records
        if r.name.startswith("weather_skills_eval") and r.levelname == "ERROR"
    ]
    assert len(errors) == 3


# The progress log


def progress_records(caplog) -> list:
    """The run's INFO log records, in order."""
    return [
        r
        for r in caplog.records
        if r.name in ("weather_skills_eval.runner", "weather_skills_eval.instance")
        and r.levelname == "INFO"
    ]


def task_records(caplog) -> list:
    """The records of tasks starting, skipped and finishing."""
    return [r for r in progress_records(caplog) if r.funcName in ("start", "skip", "finish")]


def finished(record) -> tuple:
    """A finished task's number, total, id, result and finished count, without
    the durations."""
    number, total, task_id, result, _, done, _, _ = record.args
    return number, total, task_id, result, done


def test_the_run_logs_each_step_and_each_task_out_of_the_total(tmp_path, caplog):
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1))

    with caplog.at_level("INFO"):
        harness.run()

    records = progress_records(caplog)
    runner_name = harness.docker.create_calls[0]["name"]
    app_name = harness.docker.run_calls[0]["name"]
    image = f"{APP_REPOSITORY}:test"
    served = "anthropic/claude-haiku-5.5"
    assert [r.args for r in records[:11]] == [
        ("weather-tasks", 3),
        ("image", image),
        ("Image", image),
        ("runner image", RUNNER_IMAGE),
        ("Runner image", RUNNER_IMAGE),
        (runner_name,),
        (app_name,),
        (DEFAULT_APP_START_TIMEOUT_SECONDS, app_name),
        (app_name,),
        (served, served),
        (served, 3, 1),
    ]
    tasks = records[11:17]
    # No task lost the runner, so it is not waited for after the last task:
    # only the removal follows.
    assert [r.args for r in records[17:]] == [()]
    for number, task_id in enumerate(["t1", "t2", "t3"], start=1):
        assert tasks[2 * number - 2].args == (number, 3, task_id)
        assert finished(tasks[2 * number - 1]) == (number, 3, task_id, "goal_reached", number)


def test_tasks_are_numbered_in_the_order_they_start(tmp_path, caplog):
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1))
    record = harness.gateway.run_experiment

    def last_first(*, dataset, **kwargs):
        # The tasks start in another order than the dataset's, as they can
        # with EVAL_CONCURRENCY above 1.
        return record(dataset=SimpleNamespace(items=list(reversed(dataset.items))), **kwargs)

    harness.gateway.run_experiment = last_first

    with caplog.at_level("INFO"):
        harness.run()

    records = task_records(caplog)
    assert [r.args for r in records[0::2]] == [(1, 3, "t3"), (2, 3, "t2"), (3, 3, "t1")]
    assert finished(records[1]) == (1, 3, "t3", "goal_reached", 1)


def test_a_chat_error_logs_the_first_line_of_its_error(tmp_path, caplog):
    def reply(opening, number, text):
        return Reply(error="Model not found")

    harness = Harness(tmp_path, [task("t1", "broken request")], reply)

    with caplog.at_level("INFO"):
        harness.run()

    output = harness.output()
    assert output["outcome"] == "chat_error"
    result = f"chat_error: {output['error'].splitlines()[0]}"
    assert finished(task_records(caplog)[1]) == (1, 1, "t1", result, 1)


def test_a_task_lost_to_a_runner_restart_logs_that_instead_of_an_outcome(tmp_path, caplog):
    runner = FakeContainer([*BEFORE_T1, RESTARTED, RESTARTED], logs=b"emulator crashed")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)
    tasks_raised = []
    keep_task_errors(harness, tasks_raised)

    with caplog.at_level("INFO"):
        harness.run()

    records = task_records(caplog)
    lost = runner_module.task_result(tasks_raised[0][1])
    assert finished(records[1]) == (1, 3, "t1", lost, 1)
    # The run goes on, and the next tasks have their outcomes.
    assert finished(records[3]) == (2, 3, "t2", "goal_reached", 2)
    assert finished(records[5]) == (3, 3, "t3", "goal_reached", 3)


def test_a_task_that_raises_logs_the_first_line_of_its_error(tmp_path, caplog):
    runner = FakeContainer([*BEFORE_T1, ("exited", 0, 0)], logs=b"emulator stopped")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)
    tasks_raised = []
    keep_task_errors(harness, tasks_raised)

    with caplog.at_level("INFO"):
        with pytest.raises(RunnerStartError):
            harness.run(**SHORT_WAIT)

    records = task_records(caplog)
    error = runner_module.task_result(tasks_raised[0][1])
    assert finished(records[1]) == (1, 3, "t1", error, 1)
    # t2 and t3 raise the same error without being played.
    assert [(r.funcName, r.args) for r in records[2:]] == [
        ("skip", (2, 3, "t2")),
        ("skip", (3, 3, "t3")),
    ]
    # The runner's logs, after the first line, are left out.
    assert "emulator stopped" not in "\n".join(r.getMessage() for r in records)
