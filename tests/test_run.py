"""The I/O and edge-case matrix, end to end through ``runner.run`` with a fake app,
fake Docker, fake Langfuse and fake harness models."""

import textwrap

import pytest
from fakes import FakeApp, FakeClock, FakeContainer, FakeDocker, FakeGateway, FakeModels, Reply

from weather_skills_eval.config import ConfigError
from weather_skills_eval.instance import AppStartError
from weather_skills_eval.runner import run

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
    def __init__(self, tmp_path, items, reply, user_replies=None, container=None):
        self.clock = FakeClock()
        self.app = FakeApp(reply, self.clock)
        self.models = FakeModels(user_replies)
        self.gateway = FakeGateway(items, self.models)
        self.container = container or FakeContainer(["running"])
        self.docker = FakeDocker(self.container)
        self.config_path = tmp_path / "production-agent.yaml"
        self.config_path.write_text(CONFIG)
        self.health_checks = 0

    def health(self, url):
        self.health_checks += 1
        return True

    def run(self, health=None):
        return run(
            self.config_path,
            ENVIRON,
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
    assert metadata["image_digest"] == "ghcr.io/x/app@sha256:abc"
    assert "OPENAI_API_KEYS" in metadata["env_var_names"].split(",")
    assert "1234567" in metadata["skill_packs"]
    assert all(isinstance(v, str) for v in metadata.values())
    assert summary.items == 3 and summary.goals_reached == 3
    assert harness.gateway.linked == ["chat-1", "chat-2", "chat-3"]
    # Highest version per skill name, other tools kept.
    assert harness.app.completions[0]["tool_ids"] == ["plot-2", "fetch"]


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
    assert harness.output(1)["outcome"] == "goal_reached"
    assert harness.container.removed


def test_chat_http_error_fails_that_play(tmp_path):
    items = [task("t1", "rain forecast kenya")]

    def reply(opening, number, text):
        return Reply(error="usage cap reached", http_status=429)

    harness = Harness(tmp_path, items, reply)
    harness.run()

    output = harness.output()
    assert output["outcome"] == "chat_error"
    assert "429" in output["error"]


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
