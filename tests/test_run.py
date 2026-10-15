"""The I/O and edge-case matrix, end to end through ``runner.run`` with a fake app,
fake Docker, fake Langfuse and fake harness models."""

import io
import textwrap
from pathlib import Path

import docker.errors
import pytest
from fakes import (
    APP_REPOSITORY,
    FakeApp,
    FakeClock,
    FakeContainer,
    FakeDocker,
    FakeGateway,
    FakeModels,
    Reply,
    broadcast_observations,
    local_image,
)

from weather_skills_eval.config import ConfigError
from weather_skills_eval.instance import AppStartError
from weather_skills_eval.runner import DatasetError, run
from weather_skills_eval.tasks import TaskError

ENVIRON = {
    "LANGFUSE_PUBLIC_KEY": "pk-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
    "OPENROUTER_API_KEY": "or-test",
    "LANGFUSE_TRACING_ENVIRONMENT": "eval",
    "EVAL_SIMULATED_USER_MODEL": "openrouter/test/user",
    "EVAL_JUDGE_MODEL": "openrouter/test/judge",
    "EVAL_DOCKER_NETWORK": "weather-skills-eval",
    "EVAL_TRACE_SETTLE_SECONDS": "1",
    "APP_SECRET": "webui-secret",
    "APP_ADMIN_PASSWORD": "admin-password",
    "APP_API_KEY": "sk-test-key",
    "APP_OPENROUTER_KEY": "or-app",
}

CONFIG = textwrap.dedent(
    """\
    name: production-agent
    image: ghcr.io/rhiza-research/weather-skills-chat:test
    model: weather-agent
    dataset: weather-tasks
    env:
      WEBUI_SECRET_KEY: ${APP_SECRET}
      BOOTSTRAP_ADMIN_EMAIL: eval@example.invalid
      BOOTSTRAP_ADMIN_PASSWORD: ${APP_ADMIN_PASSWORD}
      BOOTSTRAP_ADMIN_API_KEY: ${APP_API_KEY}
      OPENAI_API_BASE_URLS: https://openrouter.ai/api/v1
      OPENAI_API_KEYS: ${APP_OPENROUTER_KEY}
      SKILL_SANDLOCK: "false"
      SKILL_PACKS:
        - git_url: https://example.invalid/pack.git
          ref: main
    """
)


def task(item_id, opening, *, turn_limit=4, criteria=None, checks=None, hidden=None, pauses=None):
    return {
        "id": item_id,
        "input": opening,
        "expected_output": {
            "judge_criteria": criteria
            if criteria is not None
            else ["The assistant gives the rain forecast"],
            "artifact_checks": checks or [],
        },
        "metadata": {
            "facts": "Region: Kenya. Period: next week.",
            "hidden_steps": hidden or [],
            "pauses": pauses or [],
            "turn_limit": turn_limit,
            "tags": ["test"],
        },
    }


def progress(message="go on", classification="progress", hint=False):
    return {
        "classification": classification,
        "revealed_hidden_step": hint,
        "give_up": False,
        "give_up_reason": "",
        "message": message,
    }


class Harness:
    def __init__(self, tmp_path, items, reply, user_replies=None, container=None, **docker_options):
        self.clock = FakeClock()
        self.app = FakeApp(reply, self.clock)
        self.models = FakeModels(user_replies)
        self.gateway = FakeGateway(items, self.models)
        self.container = container or FakeContainer(["running"])
        self.docker = FakeDocker(self.container, **docker_options)
        self.config_path = tmp_path / "production-agent.yaml"
        self.config_path.write_text(CONFIG)
        self.health_checks = 0

    def health(self, url):
        self.health_checks += 1
        return True

    def run(self, health=None, config_path=None, stdin=None):
        return run(
            config_path or self.config_path,
            ENVIRON,
            stdin=stdin,
            docker_client_factory=lambda: self.docker,
            gateway_factory=lambda settings: self.gateway,
            app_transport=self.app.transport,
            instance_options={
                "health_check": health or self.health,
                "clock": self.clock,
                "sleep": self.clock.sleep_sync,
            },
            play_options={"clock": self.clock, "sleep": self.clock.sleep},
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
    assert call["environment"]["SKILL_PACKS"] == (
        '[{"git_url": "https://example.invalid/pack.git", "ref": "main"}]'
    )
    assert "volumes" not in call
    assert harness.container.removed
    assert len(harness.gateway.experiments) == 1
    experiment = harness.gateway.experiments[0]
    assert experiment["name"] == "production-agent"
    assert len(experiment["items"]) == 3
    for index in range(3):
        assert harness.scores(index)["goal_reached"] == 1.0
    metadata = experiment["metadata"]
    # The configured image is pulled, and the digest of its repository recorded.
    assert harness.docker.pulled == ["ghcr.io/rhiza-research/weather-skills-chat:test"]
    assert harness.docker.looked_up == []
    assert call["image"] == "sha256:image"
    assert metadata["image_digest"] == f"{APP_REPOSITORY}@sha256:abc"
    assert metadata["image_id"] == "sha256:image"
    assert metadata["image_pulled"] == "true"
    assert "OPENAI_API_KEYS" in metadata["env_var_names"].split(",")
    assert "1234567" in metadata["skill_packs"]
    assert metadata["tool_ids"] == "plot-2,fetch"
    assert metadata["harness_commit"] == "unrecorded"
    assert metadata["dataset"] == "weather-tasks"
    assert metadata["dataset_version"] == "latest"
    assert metadata["dataset_items"] == "3"
    assert all(isinstance(v, str) for v in metadata.values())
    assert summary.items == 3 and summary.goals_reached == 3
    assert harness.gateway.linked == ["chat-1", "chat-2", "chat-3"]
    assert harness.gateway.flushed
    for index in range(3):
        assert harness.output(index)["trace_scores"] == "complete"
    # Highest version per skill name, other tools kept.
    assert harness.app.completions[0]["tool_ids"] == ["plot-2", "fetch"]
    # The container carries its run id, and no other container is listed or touched.
    assert call["labels"]["weather-skills-eval.run"] in call["name"]
    assert harness.docker.list_filters == []


def test_config_read_from_stdin(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))

    harness.run(config_path=Path("-"), stdin=io.BytesIO(CONFIG.encode()))

    assert harness.gateway.experiments[0]["name"] == "production-agent"
    assert harness.scores()["goal_reached"] == 1.0


def test_config_from_stdin_must_name_the_experiment(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    unnamed = CONFIG.replace("name: production-agent\n", "")

    with pytest.raises(ConfigError) as error:
        harness.run(config_path=Path("-"), stdin=io.BytesIO(unnamed.encode()))

    assert error.value.field == "name"
    assert harness.docker.run_calls == []


def test_goal_met_records_turns_cost_tokens_and_time(tmp_path):
    items = [
        task(
            "t1",
            "rain forecast kenya",
            checks=[{"path": "*.png", "kind": "image"}],
        )
    ]

    def reply(opening, number, text):
        if number == 1:
            return Reply("Which region do you need?")
        return Reply(
            "DONE: here is the map.",
            artifacts=[{"path": "rain.png", "name": "rain.png", "kind": "image", "is_dir": False}],
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
    assert scores["tool_failures"] == 1
    assert scores["cache_miss_calls"] == 2
    output = harness.output()
    assert output["outcome"] == "goal_reached"
    assert output["chat_id"] == "chat-1"
    assert output["app_trace_ids"] == ["chat-1-trace-0", "chat-1-trace-1"]
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


def test_automatic_check_failing_keeps_the_play_going(tmp_path):
    items = [
        task("t1", "rain map kenya", turn_limit=2, checks=[{"path": "*.zarr", "kind": "zarr"}])
    ]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.run()

    assert harness.scores()["goal_reached"] == 0.0
    assert harness.output()["outcome"] == "turn_limit"
    # The judge model is never asked while an automatic check fails.
    assert not [c for c in harness.models.calls if c[0] == "judge" and c[2].get("tools")]


def test_turn_limit_ends_as_failure_with_its_scores(tmp_path):
    items = [task("t1", "rain forecast kenya", turn_limit=3)]
    harness = Harness(tmp_path, items, reply_done_on(99))
    harness.run()

    scores = harness.scores()
    output = harness.output()
    assert output["outcome"] == "turn_limit"
    assert scores["goal_reached"] == 0.0
    assert scores["turns_to_goal"] == 3
    assert scores["model_calls_to_goal"] == 6
    assert "time_to_answer_seconds" in scores
    # The judge's last check ends the reasoning.
    assert "(last check: verdict failed)" in output["reasoning"]


def test_user_gives_up_ends_as_failure(tmp_path):
    items = [task("t1", "rain forecast kenya", turn_limit=5, hidden=["Use the weekly total"])]
    give_up = {
        "classification": "other",
        "revealed_hidden_step": False,
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
    assert "gave up" in output["reasoning"]
    assert scores["goal_reached"] == 0.0
    assert scores["turns_to_goal"] == 2
    assert scores["hints_used"] == 1


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
    assert output["trace_scores"] == "missing: no turn reached the app"
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
        raise RuntimeError("Langfuse is down")

    harness.gateway.observations = fail
    harness.run()

    scores = harness.scores()
    output = harness.output()
    assert scores["goal_reached"] == 1.0
    assert scores["turns_to_goal"] == 1
    assert "model_calls_to_goal" not in scores
    assert output["trace_scores"].startswith("missing: Langfuse could not be read")
    comment = harness.gateway.experiments[0]["items"][0].evaluations[0].comment
    assert "trace scores missing" in comment


def test_traces_that_never_arrive_are_recorded_as_missing(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.observations = lambda chat_id, turns: ([], False)
    harness.run()

    assert harness.output()["trace_scores"] == "missing: no app traces arrived"
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

    with pytest.raises(AppStartError, match="could not be pulled"):
        harness.run()

    assert harness.docker.run_calls == []
    assert harness.gateway.experiments == []
    assert harness.gateway.flushed


def test_pull_false_runs_a_local_image_without_a_digest(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), local=local_image())
    harness.config_path.write_text(CONFIG + "pull: false\n")

    summary = harness.run()

    assert harness.docker.pulled == []
    assert harness.docker.looked_up == ["ghcr.io/rhiza-research/weather-skills-chat:test"]
    assert harness.docker.run_calls[0]["image"] == "sha256:local"
    metadata = harness.gateway.experiments[0]["metadata"]
    assert metadata["image_pulled"] == "false"
    assert metadata["image_id"] == "sha256:local"
    assert metadata["image_digest"] == ""
    assert summary.goals_reached == 1
    assert harness.container.removed


def test_pull_false_records_the_local_image_repo_digest(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    local = local_image(["ghcr.io/other/mirror@sha256:111", f"{APP_REPOSITORY}@sha256:def"])
    harness = Harness(tmp_path, items, reply_done_on(1), local=local)
    harness.config_path.write_text(CONFIG + "pull: false\n")

    harness.run()

    assert harness.docker.pulled == []
    metadata = harness.gateway.experiments[0]["metadata"]
    assert metadata["image_digest"] == f"{APP_REPOSITORY}@sha256:def"
    assert metadata["image_id"] == "sha256:local"


def test_pull_false_with_missing_image_stops_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), local=None)
    harness.config_path.write_text(CONFIG + "pull: false\n")

    with pytest.raises(AppStartError) as error:
        harness.run()

    message = str(error.value)
    assert "ghcr.io/rhiza-research/weather-skills-chat:test" in message
    assert "is not on the Docker host" in message
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

    with pytest.raises(AppStartError, match="could not be read from the Docker host"):
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

    with pytest.raises(AppStartError, match="disappeared"):
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

    with pytest.raises(DatasetError, match="could not be read"):
        harness.run()
    assert harness.docker.run_calls == []


def test_empty_dataset_does_not_start_the_run(tmp_path):
    harness = Harness(tmp_path, [], reply_done_on(1))

    with pytest.raises(DatasetError, match="has no items"):
        harness.run()
    assert harness.docker.run_calls == []


def test_duplicate_item_ids_do_not_start_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya"), task("t1", "weekly rain")]
    harness = Harness(tmp_path, items, reply_done_on(1))

    with pytest.raises(TaskError, match="more than once"):
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
        (
            lambda text: text.replace("${APP_OPENROUTER_KEY}", "sk-or-literal"),
            "env.OPENAI_API_KEYS",
        ),
        (lambda text: text + "replicas: 2\n", "replicas"),
        (lambda text: text + 'pull: "no"\n', "pull"),
        (lambda text: text + "pull: 1\n", "pull"),
    ],
)
def test_bad_config_does_not_start_the_run(tmp_path, edit, field):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.config_path.write_text(edit(CONFIG))

    with pytest.raises(ConfigError) as error:
        harness.run()

    assert error.value.field == field
    assert field in str(error.value)
    assert harness.docker.run_calls == []
    assert harness.gateway.experiments == []
