"""The I/O and edge-case matrix, end to end through ``runner.run`` with a fake app,
fake Docker, fake Langfuse and fake harness models."""

import asyncio
import base64
import io
import json
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
import requests
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
    Reply,
    app_observations,
    broadcast_observations,
    display_image,
    display_page,
    emulator_file,
    html_entry,
    image_bytes,
    image_entry,
    local_image,
    tool_call,
    zarr_entry,
)

from weather_skills_eval import play as play_module
from weather_skills_eval.app_client import ChatError
from weather_skills_eval.config import RUNNER_ENV, ConfigError
from weather_skills_eval.instance import AppInstance, AppStartError, RunnerStartError
from weather_skills_eval.langfuse_gateway import TraceReadError
from weather_skills_eval.llm import ModelReplyError
from weather_skills_eval.runner import DatasetError, run
from weather_skills_eval.settings import SettingsError
from weather_skills_eval.simulated_user import FILES_HEADER
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
    runner_image: ghcr.io/rhiza-research/weather-skills-sandbox-runner:src-171514bbb4e1
    model: weather-agent
    dataset: weather-tasks
    env:
      WEBUI_SECRET_KEY: ${APP_SECRET}
      BOOTSTRAP_ADMIN_EMAIL: eval@example.invalid
      BOOTSTRAP_ADMIN_PASSWORD: ${APP_ADMIN_PASSWORD}
      BOOTSTRAP_ADMIN_API_KEY: ${APP_API_KEY}
      OPENAI_API_BASE_URLS: https://openrouter.ai/api/v1
      OPENAI_API_KEYS: ${APP_OPENROUTER_KEY}
      OPENAI_API_CONFIGS:
        "0":
          prefix_id: open-router-
      SKILL_PACKS:
        - git_url: https://example.invalid/pack.git
          ref: main
    """
)


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
        )
        self.gateway = FakeGateway(items, self.models)
        self.container = container or FakeContainer(["running"])
        self.docker = FakeDocker(self.container, **docker_options)
        self.config_path = tmp_path / "production-agent.yaml"
        self.config_path.write_text(CONFIG)
        self.health_checks = 0

    def health(self, url):
        self.health_checks += 1
        return True

    def run(self, health=None, config_path=None, stdin=None, **settings):
        """``settings`` are harness settings to add to ENVIRON."""
        return run(
            config_path or self.config_path,
            {**ENVIRON, **settings},
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
        )

    def scores(self, index=0):
        item = self.gateway.experiments[0]["items"][index]
        return {e.name: e.value for e in item.evaluations}

    def comments(self, index=0):
        item = self.gateway.experiments[0]["items"][index]
        return {e.name: e.comment for e in item.evaluations}

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
    assert harness.container.removed
    assert harness.docker.runner.removed
    assert harness.docker.volume().removed
    assert len(harness.gateway.experiments) == 1
    experiment = harness.gateway.experiments[0]
    # Named after the model OpenRouter answers the base model with.
    assert harness.gateway.served_requests == ["anthropic/claude-haiku-5.5"]
    assert experiment["name"] == "anthropic/claude-haiku-5.5"
    assert experiment["metadata"]["served_model"] == "anthropic/claude-haiku-5.5"
    assert experiment["metadata"]["config_name"] == "production-agent"
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
    assert metadata["harness_commit"] == "unrecorded"
    # Without EVAL_BATCH the run belongs to no batch.
    assert metadata["batch"] == ""
    assert metadata["dataset"] == "weather-tasks"
    assert metadata["dataset_version"] == "latest"
    assert metadata["dataset_items"] == "3"
    assert metadata["dataset_skill_packs"] == ""
    assert all(isinstance(v, str) for v in metadata.values())
    assert summary.items == 3 and summary.goals_reached == 3
    assert summary.runner_lost == []
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
    assert call["labels"]["weather-skills-eval.run"] in call["name"]
    assert harness.docker.list_filters == []


def test_the_batch_is_recorded_in_the_experiment_metadata(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))

    harness.run(EVAL_BATCH=" 2026-10-22T20:07:57Z ")

    assert harness.gateway.experiments[0]["metadata"]["batch"] == "2026-10-22T20:07:57Z"


def test_config_read_from_stdin(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))

    harness.run(config_path=Path("-"), stdin=io.BytesIO(CONFIG.encode()))

    assert harness.gateway.experiments[0]["metadata"]["config_name"] == "production-agent"
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


def test_config_from_stdin_must_name_the_experiment(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    unnamed = CONFIG.replace("name: production-agent\n", "")

    with pytest.raises(ConfigError) as error:
        harness.run(config_path=Path("-"), stdin=io.BytesIO(unnamed.encode()))

    assert error.value.field == "name"
    assert harness.docker.run_calls == []


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
    assert harness.comments()["turns_to_goal"] == "over the whole play (goal_reached)"
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


STARTED = (
    "Harness context, not said by the user: the conversation started on 2026-10-22 at 09:30 UTC."
)


def test_the_judge_and_the_rater_are_told_when_the_play_started(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.run()

    # In the scenario description of the verdict's system prompt, after the facts.
    [verdict] = harness.models.verdict_calls()
    system = verdict["messages"][0]["content"]
    assert f"Period: next week.\n\n{STARTED}\n</scenario>" in system
    # Not as something the user said: the transcript does not hold it.
    assert STARTED not in verdict["messages"][1]["content"]
    [quality] = [c[2] for c in harness.models.calls if c[0] == "judge" and not c[2].get("tools")]
    # Before the user's request.
    prompt = quality["messages"][0]["content"]
    assert f"{STARTED}\n\nThe user's request:\nrain forecast kenya" in prompt
    # The simulated user is not told.
    assert STARTED not in json.dumps(harness.models.user_calls())


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
    assert "provenance (data source, region" in json.dumps(verdict["tools"])
    transcript = verdict["messages"][1]["content"]
    assert "DONE: weekly rain for western Kenya from ERA5." in transcript
    assert "<details" not in transcript
    parts = judge_parts(harness)
    texts = "\n".join(p["text"] for p in parts if p["type"] == "text")
    assert "File: maps/rain.png (image), presented on turn 2" in texts
    assert '"skill": "arco-era5-fetch"' in texts
    assert '"branches"' in texts
    # Files no reply presents never reach the judge.
    assert "intermediate_results" not in texts
    images = [p["image_url"]["url"] for p in parts if p["type"] == "image_url"]
    encoded = base64.b64encode(image_bytes("maps/rain.png")).decode()
    assert images == [f"data:image/png;base64,{encoded}"]
    # The simulated user saw what a user sees: the path, the kind and the image, before
    # the instruction; no provenance and no harness problem note.
    user_messages = harness.models.user_calls()[-1]["messages"]
    assert user_messages[-2]["content"] == [
        {"type": "text", "text": FILES_HEADER},
        {"type": "text", "text": "File: maps/rain.png (image)"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}},
    ]
    assert user_messages[-1] == {"role": "user", "content": "Write your JSON reply."}
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
    assert "out.zarr (zarr)" in parts[1]["text"]
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
    assert "maps/rain.png (image)" in json.dumps(judge_parts(harness))


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
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == [
        "File: kenya_rain.zarr (zarr), presented on turn 2"
    ]
    assert evidence_files(harness.models.user_calls()[-1]["messages"]) == [
        "File: kenya_rain.zarr (zarr)"
    ]


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
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == [
        "File: weekly.zarr (zarr), presented on turn 2",
        "File: daily.zarr (zarr), presented on turn 2",
    ]


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
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == [
        "File: out.zarr (zarr), presented on turn 2",
        "File: era5_kenya.zarr (zarr), named in passing on turn 2 and not presented",
    ]


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
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == [
        "File: out.zarr (zarr), presented on turn 2"
    ]


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
    assert evidence_files(verdict["messages"]) == [
        "File: maps/rain.png (image), presented on turn 2",
        "File: kenya_rain.zarr (zarr), named in passing on turn 2 and not presented",
    ]
    # Its provenance still reaches the judge, as data.
    texts = [p["text"] for p in judge_parts(harness) if p["type"] == "text"]
    assert "Provenance:" in texts[2]
    assert texts[0] == (
        "Harness evidence, not part of the conversation: every file the assistant presented "
        "or named in passing during the conversation, in order, each once at the turn given "
        "with it, with the weather_skills_history provenance the app read from it: per branch, "
        "the skills that produced it in order, with their arguments."
    )
    # The fixed criterion covers only the presented files.
    assert (
        "Each presented file's provenance (data source, region, period, aggregation, units) "
        "matches the user's request and what was agreed in the chat; a file named in passing "
        "and not presented is not held to this"
    ) in json.dumps(verdict["tools"])


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
        ["File: out.zarr (zarr)"],
        ["File: out.zarr (zarr)"],
    ]
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == [
        "File: out.zarr (zarr), presented on turn 2"
    ]


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
    assert output["reasoning"].endswith(
        "(final check failed: No result presented: no reply presented a file)"
    )
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

    reasoning = harness.output()["reasoning"]
    assert "No presented file is an image; presented: map.html (html)" in reasoning
    assert harness.models.verdict_calls() == []
    assert evidence_files(harness.models.user_calls()[-1]["messages"]) == ["File: map.html (html)"]
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
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == [
        "File: map.html (html), presented on turn 2",
        "File: out.zarr (zarr), presented on turn 2",
    ]


def evidence_files(messages: list[dict]) -> list[str]:
    """The ``File:`` lines of the harness evidence in a model call's messages."""
    evidence = [m for m in messages if isinstance(m.get("content"), list)]
    if not evidence:
        return []
    texts = [p["text"] for p in evidence[0]["content"] if p["type"] == "text"]
    return [t.split("\n")[0] for t in texts[1:]]


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
        ["File: maps/ecmwf.png (image)"],
        ["File: maps/ecmwf.png (image)"],
        ["File: maps/gefs.png (image)"],
    ]
    # The judge gets both, in order, labeled by turn, and no intermediate file.
    assert [evidence_files(c["messages"]) for c in harness.models.verdict_calls()] == [
        [
            "File: maps/ecmwf.png (image), presented on turn 2",
            "File: maps/gefs.png (image), presented on turn 4",
        ]
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

    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == [
        "File: maps/rain.png (image), presented on turn 3"
    ]
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
    evidence = calls[1]["messages"][-2]["content"]
    assert evidence[1]["text"] == "File: maps/daily.png (image)"
    assert [p["type"] for p in evidence] == ["text", "text", "image_url"]
    sent = [body["turn"]["user_message"]["content"] for body in harness.app.completions[1:]]
    assert sent == ["era5", correction["message"]]
    assert harness.output()["outcome"] == "goal_reached"
    assert harness.scores()["turns_to_goal"] == 3
    # The judge gets both maps, labeled by turn; the criteria decide which counts.
    assert [evidence_files(c["messages"]) for c in harness.models.verdict_calls()] == [
        [
            "File: maps/daily.png (image), presented on turn 2",
            "File: maps/weekly.png (image), presented on turn 3",
        ]
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
    assert output["reasoning"].endswith("(final check failed: verdict failed)")


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
    assert output["reasoning"].endswith(
        "(final check failed: No result presented: no reply presented a file)"
    )
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

    reasoning = harness.output()["reasoning"]
    assert "Presented files have no provenance: ['maps/rain.png']" in reasoning
    assert harness.models.verdict_calls() == []
    # The user sees the map as a user would: no provenance and no problem note.
    parts = harness.models.user_calls()[-1]["messages"][-2]["content"]
    assert [p["type"] for p in parts] == ["text", "text", "image_url"]
    assert parts[1]["text"] == "File: maps/rain.png (image)"
    assert "rovenance" not in json.dumps(parts)
    assert "Problem" not in json.dumps(parts)


def test_download_that_fails_once_is_tried_again_for_the_judge(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(tmp_path, items, map_reply(presenting=display_image("maps/rain.png")))
    harness.app.failing_downloads["maps/rain.png"] = 1
    harness.run()

    # The user got the map as one it could not open; the judge read it again and saw it.
    user_parts = harness.models.user_calls()[-1]["messages"][-2]["content"]
    assert [p["text"] for p in user_parts[1:]] == ["File: maps/rain.png (could not be opened)"]
    assert "Problem" not in json.dumps(user_parts)
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
    parts = harness.models.user_calls()[-1]["messages"][-2]["content"]
    assert [p["text"] for p in parts[1:]] == ["File: out.zarr (zarr)"]
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
    parts = harness.models.user_calls()[-1]["messages"][-2]["content"]
    assert [p["type"] for p in parts] == ["text", "text", "image_url", "text"]
    assert parts[1]["text"] == "File: maps/rain.png (image)"
    assert parts[-1]["text"] == "File: maps/gone.png (could not be opened)"
    assert "Problem" not in json.dumps(parts)
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


def test_an_early_file_problem_is_a_note_for_the_judge(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        early_then_final(early_provenance=False, final_provenance=True),
        user_replies={"rain map kenya": [progress("kenya"), progress("make it final")]},
    )
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    texts = [p["text"] for p in judge_parts(harness) if p["type"] == "text"]
    assert (
        texts[1] == "File: maps/draft.png (image), presented on turn 2\nNote: it has no provenance"
    )
    assert texts[2].startswith("File: maps/final.png (image), presented on turn 3\nProvenance:")


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
    assert "Presented files have no provenance: ['maps/final.png']" in output["reasoning"]


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


def test_a_webp_image_is_attached(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(
            presenting=display_image("maps/rain.webp"),
            artifacts=[image_entry("maps/rain.webp")],
            files={"maps/rain.webp": image_bytes("maps/rain.webp")},
        ),
    )
    harness.run()

    encoded = base64.b64encode(image_bytes("maps/rain.webp")).decode()
    user_parts = harness.models.user_calls()[-1]["messages"][-2]["content"]
    assert user_parts[-1] == {
        "type": "image_url",
        "image_url": {"url": f"data:image/webp;base64,{encoded}"},
    }
    assert judge_parts(harness)[-1] == user_parts[-1]


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
    assert evidence_files(harness.models.verdict_calls()[0]["messages"]) == [
        "File: out.zarr (zarr), presented on turn 2",
        "File: maps/rain.png (image), presented on turn 4",
    ]


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
    texts = [p["text"] for p in judge_parts(harness) if p["type"] == "text"]
    assert texts[1] == (
        "File: maps/early.png (unknown), presented on turn 2\nNote: it is not in the chat's files"
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
    texts = [p["text"] for p in judge_parts(harness) if p["type"] == "text"]
    assert texts[1].startswith(
        "File: maps/draft.png (image), presented on turn 2\nNote: it could not be read:"
    )
    assert "Provenance:" in texts[1]


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

    # EVAL_POLL_INTERVAL_SECONDS defaults to 2: one wait before the extra
    # listing, one before the extra download.
    extra = Counter(run(failing=True)) - Counter(run(failing=False))
    assert extra == Counter({2.0: 2})


def test_a_failed_listing_shows_the_user_each_path_as_not_opened(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(tmp_path, items, map_reply(presenting=display_image("maps/rain.png")))
    harness.app.failing_listings = 1
    harness.run()

    parts = harness.models.user_calls()[-1]["messages"][-2]["content"]
    assert [p["text"] for p in parts[1:]] == ["File: maps/rain.png (could not be opened)"]
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
    assert evidence_files(calls[1]["messages"]) == ["File: maps/rain.png (could not be opened)"]
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
        ["File: k.zarr (zarr)"],
    ]
    assert harness.output()["outcome"] == "goal_reached"


def test_an_image_of_unknown_media_type_is_not_attached_for_the_user(tmp_path):
    items = [task("t1", "rain map kenya", deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(
            presenting=display_image("maps/rain.map"),
            artifacts=[image_entry("maps/rain.map")],
            files={"maps/rain.map": image_bytes("maps/rain.map")},
        ),
    )
    harness.run()

    parts = harness.models.user_calls()[-1]["messages"][-2]["content"]
    assert [p["type"] for p in parts] == ["text", "text"]
    assert parts[1]["text"] == "File: maps/rain.map (image) (not shown as an image)"
    # Such an image is never downloaded.
    assert {r[0] for r in harness.app.file_requests} == {"/artifacts"}


@pytest.mark.parametrize("path", ["maps/rain.svg", "maps/rain.tiff", "maps/rain.bmp", "maps/rain"])
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

    user_parts = harness.models.user_calls()[-1]["messages"][-2]["content"]
    assert [p["text"] for p in user_parts[1:]] == [f"File: {path} (image) (not shown as an image)"]
    # The judge is still called; the note goes with the file's evidence.
    assert harness.output()["outcome"] == "goal_reached"
    parts = judge_parts(harness)
    assert [p["type"] for p in parts] == ["text", "text"]
    assert parts[1]["text"].startswith(
        f"File: {path} (image), presented on turn 2 (not shown as an image)\nProvenance:"
    )


def test_unreadable_image_skips_the_judge(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=2, deliverables=["image"])]
    harness = Harness(
        tmp_path, items, map_reply(presenting=display_image("maps/rain.png"), files={})
    )
    harness.run()

    reasoning = harness.output()["reasoning"]
    assert "maps/rain.png could not be read" in reasoning
    assert "404" in reasoning
    assert harness.models.verdict_calls() == []


def test_missing_kind_skips_the_judge(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=2, deliverables=["image"])]
    harness = Harness(
        tmp_path,
        items,
        map_reply(presenting="[data](out.zarr)", artifacts=[zarr_entry("out.zarr")]),
    )
    harness.run()

    reasoning = harness.output()["reasoning"]
    assert "No presented file is an image" in reasoning
    assert "out.zarr (zarr)" in reasoning
    assert harness.models.verdict_calls() == []
    assert not [r for r in harness.app.file_requests if r[0] != "/artifacts"]


def test_every_deliverable_kind_must_be_presented(tmp_path):
    items = [task("t1", "rain map kenya", turn_limit=2, deliverables=["image", "zarr"])]
    harness = Harness(tmp_path, items, map_reply(presenting=display_image("maps/rain.png")))
    harness.run()

    assert "No presented file is a zarr" in harness.output()["reasoning"]
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
    assert evidence_files(harness.models.user_calls()[-1]["messages"]) == [
        "File: maps/rain.png (image)"
    ]
    messages = harness.models.verdict_calls()[-1]["messages"]
    assert all(isinstance(m["content"], str) for m in messages)
    assert "provenance" not in json.dumps(harness.models.verdict_calls()[-1]["tools"])


def test_item_with_artifact_checks_does_not_start_the_run(tmp_path):
    item = task("t1", "rain map kenya")
    item["expected_output"]["artifact_checks"] = [{"path": "*.png"}]
    harness = Harness(tmp_path, [item], reply_done_on(1))

    with pytest.raises(TaskError, match="expected_output.artifact_checks"):
        harness.run()
    assert harness.docker.run_calls == []


def test_turn_cap_of_the_task_ends_as_failure_with_its_scores(tmp_path):
    items = [task("t1", "rain forecast kenya", turn_limit=3)]
    harness = Harness(tmp_path, items, reply_done_on(99))
    harness.run()

    scores = harness.scores()
    output = harness.output()
    assert output["outcome"] == "turn_cap"
    # Only at the cap is the judge told the conversation reached it.
    assert "maximum number of turns" in json.dumps(harness.models.verdict_calls()[0]["messages"])
    assert scores["goal_reached"] == 0.0
    assert scores["turns_to_goal"] == 3
    assert scores["model_calls_to_goal"] == 6
    assert "time_to_answer_seconds" in scores
    # The judge rules once, on the final result, and its verdict ends the reasoning.
    assert len(harness.models.verdict_calls()) == 1
    assert output["reasoning"].endswith("(final check failed: verdict failed)")


def test_reply_on_the_last_allowed_turn_still_counts(tmp_path):
    # With a cap of 1 the only reply is read once more, without sending anything.
    items = [task("t1", "rain forecast kenya", turn_limit=1)]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert harness.scores()["turns_to_goal"] == 1
    assert len(harness.app.completions) == 1
    assert len(harness.models.user_calls()) == 1
    assert "maximum number of turns" not in json.dumps(harness.models.verdict_calls())


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


@pytest.mark.parametrize("message", [3, ["go on"], {"text": "go on"}])
def test_a_message_that_is_not_text_is_unusable_at_the_cap(tmp_path, message):
    items = [task("t1", "rain forecast kenya", turn_limit=1)]
    bad = {**progress("x"), "message": message}
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(99),
        user_replies={"rain forecast kenya": [bad, progress("")]},
    )
    harness.run()

    calls = harness.models.user_calls()
    assert len(calls) == 2
    assert "message is not text" in calls[1]["messages"][-1]["content"]
    assert harness.output()["outcome"] == "turn_cap"


def test_a_give_up_at_the_cap_gives_its_reason(tmp_path):
    items = [task("t1", "rain forecast kenya", turn_limit=1)]
    harness = Harness(
        tmp_path, items, reply_done_on(99), user_replies={"rain forecast kenya": [GIVE_UP]}
    )
    harness.run()

    output = harness.output()
    assert output["outcome"] == "user_gave_up"
    assert output["reasoning"].startswith("The simulated user gave up: it is not what I need")
    assert "Reached maximum turns" not in output["reasoning"]


def test_a_done_at_the_cap_gives_the_done_text(tmp_path):
    items = [task("t1", "rain data kenya", turn_limit=1, deliverables=["zarr"])]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.run()

    output = harness.output()
    assert output["outcome"] == "user_done"
    assert output["reasoning"].startswith("The simulated user is done (final check failed:")


def test_a_run_that_ends_below_the_cap_is_not_reached(tmp_path, monkeypatch):
    # Scenario's run is cut to 1 turn while the task's cap is 3.
    real_arun = play_module.scenario.arun

    async def short_arun(**kwargs):
        return await real_arun(**{**kwargs, "max_turns": 1})

    monkeypatch.setattr(play_module.scenario, "arun", short_arun)
    items = [task("t1", "rain forecast kenya", turn_limit=3)]
    harness = Harness(tmp_path, items, reply_done_on(99))
    harness.run()

    output = harness.output()
    assert output["outcome"] == "not_reached"
    assert output["reasoning"].endswith("(final check failed: verdict failed)")
    assert harness.models.user_calls() == []


def test_task_without_turn_limit_runs_to_eval_max_turns(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(99))
    harness.run(EVAL_MAX_TURNS="6")

    assert harness.output()["outcome"] == "turn_cap"
    assert harness.scores()["turns_to_goal"] == 6


def test_eval_max_turns_defaults_to_20(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(99))
    harness.run()

    assert harness.output()["outcome"] == "turn_cap"
    assert harness.scores()["turns_to_goal"] == 20


def test_task_turn_limit_overrides_eval_max_turns(tmp_path):
    items = [task("t1", "rain forecast kenya", turn_limit=8)]
    harness = Harness(tmp_path, items, reply_done_on(99))
    harness.run(EVAL_MAX_TURNS="3")

    assert harness.scores()["turns_to_goal"] == 8


@pytest.mark.parametrize("value", ["0", "2.5", "many"])
def test_invalid_eval_max_turns_does_not_start_the_run(tmp_path, value):
    harness = Harness(tmp_path, [task("t1", "rain forecast kenya")], reply_done_on(1))

    with pytest.raises(SettingsError, match="EVAL_MAX_TURNS"):
        harness.run(EVAL_MAX_TURNS=value)
    assert harness.docker.run_calls == []


def test_pause_past_eval_max_turns_does_not_start_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya", pauses=[{"before_turn": 7, "seconds": 1}])]
    harness = Harness(tmp_path, items, reply_done_on(1))

    with pytest.raises(
        TaskError, match="t1: metadata.pauses: pause before turn 7 is past EVAL_MAX"
    ):
        harness.run(EVAL_MAX_TURNS="6")
    assert harness.docker.run_calls == []


def test_done_on_the_last_allowed_turn_is_not_judged_as_the_turn_cap(tmp_path):
    # With a cap of 2, the user's second turn is the last; it reads reply 1 and is done.
    items = [task("t1", "rain forecast kenya", turn_limit=2)]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.run()

    assert harness.output()["outcome"] == "goal_reached"
    assert harness.scores()["turns_to_goal"] == 1
    messages = json.dumps(harness.models.verdict_calls()[0]["messages"])
    assert "maximum number of turns" not in messages


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
    assert "gave up" in output["reasoning"]
    assert output["reasoning"].endswith("(final check failed: verdict failed)")
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
    "empty-message": (progress(""), "empty message while neither done nor give_up is true"),
    "unknown-classification": (
        progress("go on", classification="request"),
        "unknown classification 'request'",
    ),
    "not-json": ("Sure, go on.", "it is not a JSON object"),
    "done-and-give-up": ({**GIVE_UP, "done": True}, "done and give_up are both true"),
    "done-not-boolean": ({**progress("go on"), "done": "no"}, "done is not true or false"),
    "give-up-missing": (
        {k: v for k, v in progress("go on").items() if k != "give_up"},
        "give_up is not true or false",
    ),
    "hint-not-boolean": (
        {**progress("go on"), "revealed_hidden_step": 1},
        "revealed_hidden_step is not true or false",
    ),
}


def test_a_passing_final_check_is_marked_on_a_failed_outcome(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path, items, reply_done_on(1), user_replies={"rain forecast kenya": [GIVE_UP]}
    )
    harness.run()

    output = harness.output()
    assert output["outcome"] == "user_gave_up"
    assert output["reasoning"].endswith("(final check passed: verdict passed)")


def test_final_judge_without_a_verdict_ends_as_error_with_scores(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), judge_without_verdict=True)
    harness.run()

    output = harness.output()
    scores = harness.scores()
    assert output["outcome"] == "error"
    assert output["error"].startswith("The final judge call failed:")
    assert "tool calls not found" in output["error"]
    assert scores["goal_reached"] == 0.0
    assert scores["turns_to_goal"] == 1
    assert output["trace_scores"] == "complete"
    assert "model_calls_to_goal" in scores


@pytest.mark.parametrize("unusable, reason", UNUSABLE_REPLIES.values(), ids=UNUSABLE_REPLIES.keys())
def test_unusable_user_reply_is_asked_for_once_more(tmp_path, unusable, reason):
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
    # The second call is shown the first reply and told why it could not be used.
    assert calls[1]["messages"][:-2] == calls[0]["messages"]
    shown = unusable if isinstance(unusable, str) else json.dumps(unusable)
    assert calls[1]["messages"][-2] == {"role": "assistant", "content": shown}
    assert calls[1]["messages"][-1]["content"] == (
        f"Your previous reply could not be used: {reason}. Write your JSON reply again, "
        "as described."
    )
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

    output = harness.output()
    assert output["outcome"] == "error"
    assert (
        "The simulated user's reply cannot be used, empty message while neither done nor "
        "give_up is true" in output["error"]
    )
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
    assert calls[1]["messages"][-1]["content"] == (
        "Your previous reply could not be used: it had no text. Write your JSON reply again, "
        "as described."
    )
    assert harness.output()["outcome"] == "goal_reached"


def test_a_whitespace_reply_counts_as_having_no_text(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(2),
        user_replies={"rain forecast kenya": ["  \n ", progress("kenya")]},
    )
    harness.run()

    calls = harness.models.user_calls()
    assert calls[1]["messages"][:-1] == calls[0]["messages"]
    assert "it had no text" in calls[1]["messages"][-1]["content"]


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
    assert output["error"].startswith("The final judge call failed:")
    assert "EVAL_MODEL_MAX_TOKENS" in output["error"]


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
        raise TraceReadError("ReadTimeout: timed out")

    harness.gateway.observations = fail
    harness.run()

    scores = harness.scores()
    output = harness.output()
    assert scores["goal_reached"] == 1.0
    assert scores["turns_to_goal"] == 1
    assert "model_calls_to_goal" not in scores
    assert output["trace_scores"] == "missing: Langfuse could not be read: ReadTimeout: timed out"
    comment = harness.gateway.experiments[0]["items"][0].evaluations[0].comment
    assert "trace scores missing" in comment
    assert harness.gateway.app_costs == []
    assert output["provider_models"] == []


def test_catalog_model_without_base_model_stops_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.app.base_model_id = None

    with pytest.raises(ChatError, match="no base model for catalog model weather-agent"):
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


def test_reads_that_failed_at_the_timeout_are_named_in_the_trace_scores(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.observations = lambda chat_id, turns: (
        app_observations(chat_id, turns),
        False,
        "ReadTimeout: x",
    )
    harness.run()

    reason = "Langfuse reads failed: ReadTimeout: x"
    assert harness.output()["trace_scores"] == f"partial: {reason}"
    assert harness.comments()["model_calls_to_goal"].endswith(reason)


def test_reads_that_failed_without_observations_are_named_as_missing(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.observations = lambda chat_id, turns: ([], False, "ReadTimeout: x")
    harness.run()

    assert harness.output()["trace_scores"] == "missing: Langfuse could not be read: ReadTimeout: x"


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

    with pytest.raises(AppStartError, match="could not be pulled") as error:
        harness.run()

    # The app's own error, not the runner's, naming the app image.
    assert type(error.value) is AppStartError
    assert "ghcr.io/rhiza-research/weather-skills-chat:test" in str(error.value)
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


@pytest.mark.parametrize(
    "digests",
    [
        [f"{APP_REPOSITORY}@sha256:first", f"{APP_REPOSITORY}@sha256:second"],
        # The pinned digest is not listed verbatim; another of the repository is.
        [f"{APP_REPOSITORY}@sha256:first"],
    ],
    ids=["listed-second", "not-listed-verbatim"],
)
def test_a_pulled_image_records_the_pinned_digest(tmp_path, digests):
    items = [task("t1", "rain forecast kenya")]
    pinned = f"{APP_REPOSITORY}@sha256:second"
    harness = Harness(tmp_path, items, reply_done_on(1), image_digests=digests)
    harness.config_path.write_text(CONFIG.replace(APP_IMAGE_LINE, f"image: {pinned}"))

    harness.run()

    assert harness.docker.pulled == [pinned, RUNNER_IMAGE]
    metadata = harness.gateway.experiments[0]["metadata"]
    assert metadata["image_digest"] == pinned


def test_a_pinned_runner_image_records_the_pinned_digest(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    pinned = f"{RUNNER_REPOSITORY}@sha256:second"
    digests = [f"{RUNNER_REPOSITORY}@sha256:first", pinned]
    harness = Harness(tmp_path, items, reply_done_on(1), runner_digests=digests)
    harness.config_path.write_text(CONFIG.replace(RUNNER_IMAGE, pinned))

    harness.run()

    assert harness.docker.pulled == ["ghcr.io/rhiza-research/weather-skills-chat:test", pinned]
    metadata = harness.gateway.experiments[0]["metadata"]
    assert metadata["runner_image"] == pinned
    assert metadata["runner_image_digest"] == pinned


def test_pull_false_with_missing_image_stops_the_run(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), local=None)
    harness.config_path.write_text(CONFIG + "pull: false\n")

    with pytest.raises(AppStartError) as error:
        harness.run()

    # The app's own error, not the runner's.
    assert type(error.value) is AppStartError
    assert harness.docker.events == []
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


# Skill packs a dataset needs

PACKS_BLOCK = "  SKILL_PACKS:\n    - git_url: https://example.invalid/pack.git\n      ref: main\n"
CHC_PACK = "    - git_url: https://github.com/rhiza-research/chc-skills.git\n"


def with_skill_packs(block: str) -> str:
    """CONFIG with its SKILL_PACKS replaced by ``block``."""
    assert CONFIG.endswith(PACKS_BLOCK)
    return CONFIG.removesuffix(PACKS_BLOCK) + block


@pytest.mark.parametrize("metadata", [None, {}, {"owner": "eval"}, {"skill_packs": []}])
def test_dataset_needing_no_pack_plays_every_item(tmp_path, metadata):
    items = [task("t1", "rain forecast kenya"), task("t2", "weekly rain")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.dataset_metadata = metadata

    summary = harness.run()

    assert summary.items == 2
    assert harness.gateway.experiments[0]["metadata"]["dataset_skill_packs"] == ""


FORECASTING_PACK = (
    "    - git_url: https://github.com/rhiza-research/forecasting-skills/\n"
    "      enabled_by_default: true\n"
)


@pytest.mark.parametrize(
    "packs, config, recorded",
    [
        (["chc-skills"], CONFIG + CHC_PACK, "chc-skills"),
        ([" chc-skills "], CONFIG + CHC_PACK, "chc-skills"),
        (
            ["chc-skills", "forecasting-skills"],
            CONFIG + FORECASTING_PACK + CHC_PACK,
            "chc-skills,forecasting-skills",
        ),
    ],
    ids=["one-pack", "padded-name", "two-packs-in-dataset-order"],
)
def test_dataset_packs_that_the_config_turns_on_play_every_item(tmp_path, packs, config, recorded):
    items = [task("t1", "rain forecast kenya"), task("t2", "weekly rain")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.dataset_metadata = {"skill_packs": packs}
    harness.config_path.write_text(config)

    summary = harness.run()

    assert summary.items == 2
    assert len(harness.docker.run_calls) == 1
    assert harness.gateway.experiments[0]["metadata"]["dataset_skill_packs"] == recorded


def test_only_the_packs_the_config_does_not_turn_on_are_named(tmp_path):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.dataset_metadata = {"skill_packs": ["chc-skills", "a-skills", "b-skills"]}
    harness.config_path.write_text(CONFIG + CHC_PACK)

    with pytest.raises(DatasetError) as error:
        harness.run()

    message = str(error.value)
    assert "a-skills, b-skills" in message
    assert "chc-skills" not in message
    assert harness.docker.run_calls == []


@pytest.mark.parametrize("metadata", ["chc-skills", ["chc-skills"]])
def test_dataset_metadata_that_is_not_an_object_does_not_start_the_run(tmp_path, metadata):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.dataset_metadata = metadata

    with pytest.raises(DatasetError, match="Dataset weather-tasks metadata must be an object"):
        harness.run()
    assert harness.docker.run_calls == []


@pytest.mark.parametrize(
    "config",
    [
        CONFIG + CHC_PACK + "      enabled_by_default: false\n",
        CONFIG,
        with_skill_packs(""),
    ],
    ids=["pack-off", "pack-not-listed", "skill-packs-unset"],
)
def test_dataset_pack_the_config_does_not_turn_on_does_not_start_the_run(tmp_path, config):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.dataset_metadata = {"skill_packs": ["chc-skills"]}
    harness.config_path.write_text(config)

    with pytest.raises(DatasetError) as error:
        harness.run()

    message = str(error.value)
    assert "weather-tasks" in message
    assert "chc-skills" in message
    assert "does not turn on" in message
    assert harness.docker.run_calls == []
    assert harness.gateway.experiments == []


@pytest.mark.parametrize("packs", ["chc-skills", [""], ["  "], [1], {"chc-skills": True}])
def test_bad_dataset_skill_packs_does_not_start_the_run(tmp_path, packs):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.gateway.dataset_metadata = {"skill_packs": packs}

    with pytest.raises(DatasetError, match="Dataset weather-tasks metadata skill_packs"):
        harness.run()
    assert harness.docker.run_calls == []


@pytest.mark.parametrize(
    "packs",
    [
        '  SKILL_PACKS: "{}"\n',
        "  SKILL_PACKS:\n    - ref: main\n",
        "  SKILL_PACKS:\n    - git_url: https://example.invalid/pack.git\n"
        '      enabled_by_default: "no"\n',
    ],
    ids=["not-a-list", "no-git-url", "enabled-not-boolean"],
)
def test_bad_skill_packs_config_does_not_start_the_run(tmp_path, packs):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.config_path.write_text(with_skill_packs(packs))

    with pytest.raises(ConfigError) as error:
        harness.run()

    assert error.value.field == "env.SKILL_PACKS"
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
        (lambda text: text.replace(f"runner_image: {RUNNER_IMAGE}\n", ""), "runner_image"),
        (lambda text: text.replace(RUNNER_IMAGE, '""'), "runner_image"),
        (lambda text: text.replace(RUNNER_IMAGE, "${UNSET_RUNNER}"), "runner_image"),
        (lambda text: text + "emulator_commit: abc\n", "emulator_commit"),
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
    assert harness.docker.pulled == []


# The sandbox runner


def with_env(text: str, block: str) -> str:
    """CONFIG with ``block`` added to its env, before SKILL_PACKS."""
    return text.replace("  SKILL_PACKS:\n", block + "  SKILL_PACKS:\n")


@pytest.mark.parametrize("name", RUNNER_ENV)
def test_config_setting_a_runner_variable_does_not_start_the_run(tmp_path, name):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1))
    harness.config_path.write_text(with_env(CONFIG, f'  {name}: "x"\n'))

    with pytest.raises(ConfigError) as error:
        harness.run()

    assert error.value.field == f"env.{name}"
    assert "set by the harness" in str(error.value)
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
    run_id = app["labels"]["weather-skills-eval.run"]
    runner_name = f"weather-skills-eval-runner-{run_id}"
    volume_name = f"weather-skills-eval-data-{run_id}"
    source_name = f"weather-skills-eval-emulator-{run_id}"
    # The emulator is read from a container of the app image, never started and
    # removed at once; the runner and the volume come next, then the app; all
    # go when the run ends, the volume after the containers that use it.
    source = docker_.source
    assert source.image_id == "sha256:image"
    assert source.labels == {"weather-skills-eval": "emulator", "weather-skills-eval.run": run_id}
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
    assert docker_.volume().labels == {
        "weather-skills-eval": "data",
        "weather-skills-eval.run": run_id,
    }
    # The artifacts folder is created on the volume first, from a container of
    # the app image with the whole volume at the app's data directory, so the
    # volume's first mount fills it from the app image; never started.
    [setup] = docker_.setup_calls
    assert setup == {
        "image": "sha256:image",
        "name": setup_name,
        "labels": {"weather-skills-eval": "volume-setup", "weather-skills-eval.run": run_id},
        "volumes": {volume_name: {"bind": "/app/backend/data", "mode": "rw"}},
    }
    [(setup_path, setup_data)] = docker_.volume_setup.archives
    assert setup_path == "/app/backend/data"
    with tarfile.open(fileobj=io.BytesIO(setup_data)) as tar:
        assert [(m.name, m.isdir()) for m in tar.getmembers()] == [("artifacts", True)]
    assert not docker_.volume_setup.started
    assert docker_.volume_setup.removed
    # The runner: the runner image, the emulator, and only the volume's
    # artifacts folder, at the path the emulator reads.
    assert create["image"] == "sha256:runner"
    assert create["name"] == runner_name
    assert create["user"] == "0"
    assert create["command"] == ["python3", "-u", "/opt/sandbox/emulator/emulator.py"]
    assert create["network"] == "weather-skills-eval"
    assert create["labels"] == {"weather-skills-eval": "runner", "weather-skills-eval.run": run_id}
    assert "volumes" not in create
    assert create["mounts"] == [
        {
            "Target": "/data/artifacts",
            "Source": volume_name,
            "Type": "volume",
            "ReadOnly": False,
            "VolumeOptions": {"Subpath": "artifacts"},
        }
    ]
    assert create["restart_policy"] == {"Name": "on-failure"}
    token = create["environment"]["SKILL_K8S_BIND_TOKEN"]
    assert len(token) >= 32
    assert create["environment"] == {
        "ARTIFACTS_DIR": "/data/artifacts",
        "SKILL_K8S_EMULATOR_HOST": runner_name,
        "SKILL_K8S_EMULATOR_API_PORT": "18080",
        "SKILL_K8S_BIND_PORT": "9090",
        "SKILL_K8S_EMULATOR_START_DELAY": "0",
        "SKILL_K8S_BIND_TOKEN": token,
    }
    # The emulator files reach the runner by put_archive, before it starts.
    [(path, data)] = docker_.runner.archives
    assert path == "/"
    assert archive_files(data) == {
        "opt/sandbox/emulator/emulator.py": emulator_file("/app/sandbox/emulator/emulator.py"),
        "opt/sandbox/emulator/local_runtime.py": emulator_file(
            "/app/sandbox/emulator/local_runtime.py"
        ),
    }
    assert docker_.runner.started
    # The app: the emulator settings with this run's runner and token, the volume at its data dir.
    assert app["volumes"] == {volume_name: {"bind": "/app/backend/data", "mode": "rw"}}
    env = app["environment"]
    assert {name: env[name] for name in RUNNER_ENV} == {
        "SKILL_SANDLOCK": "false",
        "SKILL_SANDBOX_BACKEND": "k8s",
        "SKILL_K8S_SANDBOX_ENABLED": "true",
        "SKILL_K8S_EMULATOR": "true",
        "SKILL_K8S_APISERVER": f"http://{runner_name}:18080",
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


@pytest.mark.parametrize(
    "app_files, missing",
    [
        ({}, "/app/sandbox/emulator/emulator.py"),
        (
            {"/app/sandbox/emulator/emulator.py": b"# emulator\n"},
            "/app/sandbox/emulator/local_runtime.py",
        ),
    ],
    ids=["no-emulator", "no-local-runtime"],
)
def test_app_image_without_the_emulator_stops_the_run_before_the_runner(
    tmp_path, app_files, missing
):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), app_files=app_files)

    with pytest.raises(RunnerStartError) as error:
        harness.run()

    message = str(error.value)
    assert "ghcr.io/rhiza-research/weather-skills-chat:test" in message
    assert missing in message
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

    with pytest.raises(RunnerStartError, match="could not be read from image") as error:
        harness.run()

    assert "daemon busy" in str(error.value)
    assert harness.docker.create_calls == []
    assert harness.docker.source.removed


def broken_stream():
    yield b"\0" * 10
    raise requests.exceptions.ChunkedEncodingError("connection broken")


@pytest.mark.parametrize(
    "stream",
    [lambda: iter([b"not a tar archive"]), broken_stream],
    ids=["not-a-tar", "stream-broken"],
)
def test_unreadable_emulator_archive_stops_the_run_before_the_runner(tmp_path, stream):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), archive_stream=stream)

    with pytest.raises(RunnerStartError) as error:
        harness.run()

    message = str(error.value)
    assert "/app/sandbox/emulator/emulator.py could not be read from image" in message
    assert "ghcr.io/rhiza-research/weather-skills-chat:test" in message
    assert harness.docker.create_calls == []
    assert harness.docker.source.removed


@pytest.mark.parametrize("pull", [True, False])
def test_runner_image_pull_failure_stops_the_run_before_the_app(tmp_path, pull):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(
        tmp_path,
        items,
        reply_done_on(1),
        local=local_image(),
        runner_pull_error=docker.errors.ImageNotFound("manifest unknown"),
    )
    harness.config_path.write_text(CONFIG + f"pull: {'true' if pull else 'false'}\n")

    with pytest.raises(RunnerStartError) as error:
        harness.run()

    assert f"Runner image {RUNNER_IMAGE} could not be pulled" in str(error.value)
    app_pulls = ["ghcr.io/rhiza-research/weather-skills-chat:test"] if pull else []
    assert harness.docker.pulled == [*app_pulls, RUNNER_IMAGE]
    assert harness.docker.create_calls == []
    assert harness.docker.volumes_created == {}
    assert harness.docker.run_calls == []
    assert harness.gateway.experiments == []


@pytest.mark.parametrize(
    "options",
    [
        {"volume_error": docker.errors.APIError("disk full")},
        {"create_error": docker.errors.APIError("no such network")},
        {"runner": FakeContainer([], put_archive_error=docker.errors.APIError("no space"))},
        {"runner": FakeContainer([], start_error=docker.errors.APIError("port in use"))},
    ],
    ids=["volume", "create", "put-archive", "start"],
)
def test_runner_start_failure_stops_the_run_and_removes_what_it_created(tmp_path, options):
    items = [task("t1", "rain forecast kenya")]
    harness = Harness(tmp_path, items, reply_done_on(1), **options)

    with pytest.raises(RunnerStartError, match="Runner container could not start"):
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

    with pytest.raises(RunnerStartError, match="artifacts folder could not be created"):
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

    with pytest.raises(RunnerStartError, match="emulator files could not be copied"):
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

    with pytest.raises(RunnerStartError, match="disappeared"):
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
    assert "is not running before task t2; waiting up to 600 s" in caplog.text
    # /health, 3 for t1, 4 polls and 1 check for t2, 3 for t3.
    assert runner.reloads == 12
    assert runner.removed


def test_runner_that_does_not_come_back_stops_the_run_with_its_logs(tmp_path):
    # Exited before t2 and never restarted.
    runner = FakeContainer([*THROUGH_T1, "exited"], logs=b"emulator crashed")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)

    with pytest.raises(RunnerStartError, match="emulator crashed") as error:
        harness.run(**SHORT_WAIT)

    message = str(error.value)
    assert "was not running steadily before task t2 after 30 s: it is exited" in message
    # /health, 3 for t1, then every 5 s from 0 to 30 s.
    assert runner.reloads == 4 + 7
    assert len(harness.app.chats) == 1
    assert runner.removed
    assert harness.container.removed
    assert harness.docker.volume().removed


@pytest.mark.parametrize(
    "statuses, message",
    [
        (["running", "gone"], "disappeared before task t1"),
        (["running", "exited", "gone"], "disappeared before task t1"),
        (["running", "api-error"], "could not be read before task t1"),
    ],
    ids=["gone", "gone-while-waiting", "reload-error"],
)
def test_runner_lost_mid_run_stops_the_run(tmp_path, statuses, message):
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=FakeContainer(statuses))

    with pytest.raises(RunnerStartError, match=message):
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

    with pytest.raises(RunnerStartError, match="not running steadily before task t2"):
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

    with pytest.raises(RunnerStartError, match="ImportError: local_runtime") as error:
        harness.run()

    assert "Runner container restarting before the app was ready" in str(error.value)
    assert harness.app.chats == {}
    assert runner.removed


def test_runner_in_a_crash_loop_stops_the_run_by_the_timeout(tmp_path):
    # Before t2 Docker keeps restarting it: running is never seen twice with one count.
    loop = [("running" if n % 2 else "restarting", n, 0 if n % 2 else 1) for n in range(1, 40)]
    runner = FakeContainer([*THROUGH_T1, *loop], logs=b"emulator crashed")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)

    with pytest.raises(RunnerStartError, match="emulator crashed") as error:
        harness.run(**SHORT_WAIT)

    assert "was not running steadily before task t2 after 30 s" in str(error.value)
    assert len(harness.app.chats) == 1


def test_runner_that_restarts_on_every_poll_is_not_taken_as_running(tmp_path):
    # Running on each poll, but with a new restart count each time.
    loop = [("running", n, 0) for n in range(1, 40)]
    runner = FakeContainer([*THROUGH_T1, *loop])
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)

    with pytest.raises(RunnerStartError, match="not running steadily before task t2"):
        harness.run(**SHORT_WAIT)


def test_runner_that_exits_with_code_0_stops_the_run_at_once(tmp_path):
    # Docker does not restart a container that exited with code 0.
    runner = FakeContainer([*THROUGH_T1, ("exited", 0, 0)], logs=b"emulator stopped")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)

    with pytest.raises(RunnerStartError, match="emulator stopped") as error:
        harness.run(**SHORT_WAIT)

    assert "exited with code 0 before task t2" in str(error.value)
    assert runner.reloads == 5
    assert len(harness.app.chats) == 1


def test_a_dead_runner_stops_the_run_at_once(tmp_path):
    # Docker never restarts a dead container.
    runner = FakeContainer([*THROUGH_T1, "dead"], logs=b"emulator removal failed")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)

    with pytest.raises(RunnerStartError, match="emulator removal failed") as error:
        harness.run(**SHORT_WAIT)

    assert "is dead before task t2" in str(error.value)
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
    assert "stopped during task t1" in str(error)
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

    assert "was not running steadily before task t2 after 30 s: it is exited" in str(error.value)
    # t1 is not scored; t2 and t3 are not played.
    assert [item_id for item_id, _ in tasks_raised] == ["t1", "t2", "t3"]
    assert "stopped during task t1" in str(tasks_raised[0][1])
    assert tasks_raised[1][1] is error.value is tasks_raised[2][1]
    assert harness.gateway.linked == ["chat-1"]
    assert len(harness.app.chats) == 1
    assert harness.container.removed
    assert harness.docker.volume().removed


@pytest.mark.parametrize(
    "after_t1, message",
    [
        (("exited", 0, 0), "stopped during task t1: it is exited"),
        ("dead", "stopped during task t1: it is dead"),
        ("gone", "disappeared during task t1"),
        ("api-error", "could not be read during task t1"),
    ],
    ids=["exited-0", "dead", "gone", "reload-error"],
)
def test_runner_found_stopped_for_good_after_a_task_stops_the_run_at_once(
    tmp_path, after_t1, message
):
    runner = FakeContainer([*BEFORE_T1, after_t1], logs=b"emulator stopped")
    harness = Harness(tmp_path, THREE_TASKS, reply_done_on(1), runner=runner)
    tasks_raised = []
    keep_task_errors(harness, tasks_raised)

    with pytest.raises(RunnerStartError, match=message) as error:
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

    with pytest.raises(RunnerStartError, match="emulator crashed") as error:
        harness.run(**SHORT_WAIT)

    assert "not running steadily after the last task after 30 s" in str(error.value)
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

    with pytest.raises(RunnerStartError, match="stopped during task t1") as error:
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

    logged = caplog.text
    assert "Could not remove app container" in logged
    assert "Could not remove runner container" in logged
    assert "Could not remove volume" in logged
