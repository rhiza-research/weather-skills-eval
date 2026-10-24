import asyncio
import json
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace

import httpx
import litellm
import pytest
from fakes import FakeClock, FakeRunApi
from langfuse import Langfuse
from langfuse._client.datasets import DatasetClient
from langfuse.api import Dataset, DatasetItem, DatasetStatus
from langfuse.api.core.api_error import ApiError as LangfuseApiError
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from weather_skills_eval import runner
from weather_skills_eval.instance import RunnerStartError
from weather_skills_eval.langfuse_gateway import (
    DELETE_BATCH_SIZE,
    EPOCH,
    SERVED_MODEL_MAX_TOKENS,
    TOOL_OUTPUT_FIELDS,
    TOOL_OUTPUT_PAGE_SIZE,
    ExperimentNotFound,
    LangfuseGateway,
    LangfuseRuns,
    RunTraces,
    TraceReadError,
    play_scores,
)
from weather_skills_eval.llm import ModelReplyError
from weather_skills_eval.scoring import TraceRead
from weather_skills_eval.settings import load_settings


def real_client(exporter: InMemorySpanExporter) -> Langfuse:
    return Langfuse(
        public_key=f"pk-test-{uuid.uuid4()}",
        secret_key="sk-test",
        base_url="http://langfuse.invalid",
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
        environment="eval",
    )


def gateway_for(client, **options) -> LangfuseGateway:
    settings = {
        "trace_timeout": 2,
        "trace_settle": 1,
        "poll_interval": 1,
        "model_max_tokens": 8192,
        "rate_limit_retries": 6,
        "rate_limit_backoff": 15,
    }
    settings.update(options)
    return LangfuseGateway(environment="eval", client=client, **settings)


def test_model_calls_are_root_traces_with_role_user_id_session_and_environment(monkeypatch):
    exporter = InMemorySpanExporter()
    client = real_client(exporter)
    gateway = gateway_for(client, model_retries=3)
    reply = litellm.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": '{"ok": true}'}}],
        usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    )
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return reply

    monkeypatch.setattr(litellm, "completion", completion)

    with client.start_as_current_observation(name="experiment-item-task"):
        response = gateway.traced_completion(
            "judge",
            "chat-9",
            model="openrouter/test/judge",
            messages=[{"role": "user", "content": "x"}],
        )
    client.flush()

    assert response is reply
    assert calls[0]["num_retries"] == 3
    spans = {span.name: span for span in exporter.get_finished_spans()}
    judge = spans["judge"]
    assert judge.parent is None
    assert judge.context.trace_id != spans["experiment-item-task"].context.trace_id
    assert judge.attributes["user.id"] == "judge"
    assert judge.attributes["session.id"] == "chat-9"
    assert judge.attributes["langfuse.environment"] == "eval"
    assert judge.attributes["langfuse.observation.type"] == "generation"
    assert '"input": 12' in judge.attributes["langfuse.observation.usage_details"]


def test_experiment_records_scores_and_links_the_item_trace_to_the_chat(monkeypatch):
    exporter = InMemorySpanExporter()
    client = real_client(exporter)
    gateway = gateway_for(client)
    scores = []
    monkeypatch.setattr(client, "create_score", lambda **kwargs: scores.append(kwargs))

    async def task(*, item, **kwargs):
        gateway.link_item_to_chat("chat-42")
        return {"chat_id": "chat-42", "scores": [{"name": "goal_reached", "value": 1.0}]}

    # Local data stands in for a Langfuse dataset; the experiment code path is the same.
    dataset = SimpleNamespace(
        run_experiment=lambda **kwargs: client.run_experiment(
            data=[{"input": "rain next week"}], **kwargs
        )
    )
    result = gateway.run_experiment(
        dataset=dataset,
        name="cfg",
        description="provider/model",
        task=task,
        metadata={"image": "x"},
        concurrency=1,
    )
    client.flush()

    assert [e.name for e in result.item_results[0].evaluations] == ["goal_reached"]
    assert result.description == "provider/model"
    assert [(s["name"], s["value"]) for s in scores] == [("goal_reached", 1.0)]
    spans = exporter.get_finished_spans()
    task_span = next(s for s in spans if s.name == "experiment-item-task")
    assert task_span.attributes["session.id"] == "chat-42"
    # The description travels on the item spans, where Langfuse reads it.
    assert task_span.attributes["langfuse.experiment.description"] == "provider/model"
    chat_span = next(s for s in spans if s.name == "chat")
    assert chat_span.attributes["session.id"] == "chat-42"
    assert chat_span.context.trace_id == task_span.context.trace_id
    assert scores[0]["trace_id"] == format(task_span.context.trace_id, "032x")


class Reverse:
    """A random source whose shuffle reverses the order."""

    def shuffle(self, items: list) -> None:
        items.reverse()


def test_a_real_dataset_client_plays_its_items_in_the_runners_play_order(monkeypatch):
    # Only the HTTP layer is stubbed; the project id read fails, which the SDK allows.
    client = Langfuse(
        public_key=f"pk-test-{uuid.uuid4()}",
        secret_key="sk-test",
        base_url="http://langfuse.invalid",
        httpx_client=httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(404, json={}))
        ),
        span_exporter=InMemorySpanExporter(),
        tracer_provider=TracerProvider(),
        environment="eval",
    )
    monkeypatch.setattr(client, "create_score", lambda **kwargs: None)
    created = datetime(2026, 10, 23, tzinfo=UTC)
    items = [
        DatasetItem(
            id=item_id,
            status=DatasetStatus.ACTIVE,
            input=f"task {item_id}",
            expected_output=None,
            metadata=None,
            dataset_id="d1",
            dataset_name="weather-tasks",
            created_at=created,
            updated_at=created,
            media_references=[],
        )
        for item_id in ("t1", "t2", "t3")
    ]
    dataset = DatasetClient(
        dataset=Dataset(
            id="d1",
            name="weather-tasks",
            metadata=None,
            project_id="p1",
            created_at=created,
            updated_at=created,
        ),
        items=items,
        langfuse_client=client,
    )
    started = []

    async def task(*, item, **kwargs):
        started.append(item.id)
        return {"scores": []}

    result = gateway_for(client).run_experiment(
        dataset=runner.in_play_order(dataset, Reverse()),
        name="cfg",
        description="m",
        task=task,
        metadata={},
        concurrency=1,
    )
    client.flush()

    assert started == ["t3", "t2", "t1"]
    assert [r.item.id for r in result.item_results] == ["t3", "t2", "t1"]
    # The dataset read is left in its order.
    assert [item.id for item in dataset.items] == ["t1", "t2", "t3"]


def test_a_task_lost_to_the_runner_is_recorded_as_an_error_and_the_experiment_goes_on(
    monkeypatch,
):
    exporter = InMemorySpanExporter()
    client = real_client(exporter)
    gateway = gateway_for(client)
    scores = []
    monkeypatch.setattr(client, "create_score", lambda **kwargs: scores.append(kwargs))

    async def task(*, item, **kwargs):
        if item["input"] == "first":
            gateway.link_item_to_chat("chat-1")
            raise RunnerStartError("Runner container stopped during task first")
        gateway.link_item_to_chat("chat-2")
        return {"chat_id": "chat-2", "scores": [{"name": "goal_reached", "value": 1.0}]}

    dataset = SimpleNamespace(
        run_experiment=lambda **kwargs: client.run_experiment(
            data=[{"input": "first"}, {"input": "second"}], **kwargs
        )
    )
    result = gateway.run_experiment(
        dataset=dataset, name="cfg", description="m", task=task, metadata={}, concurrency=1
    )
    client.flush()

    # The lost item is left out of the results; the second is played and scored.
    assert [r.item["input"] for r in result.item_results] == ["second"]
    assert [(s["name"], s["value"]) for s in scores] == [("goal_reached", 1.0)]
    task_spans = {
        s.attributes["session.id"]: s
        for s in exporter.get_finished_spans()
        if s.name == "experiment-item-task"
    }
    lost = task_spans["chat-1"]
    assert lost.attributes["langfuse.observation.level"] == "ERROR"
    assert "Runner container stopped" in lost.attributes["langfuse.observation.output"]
    assert task_spans["chat-2"].attributes.get("langfuse.observation.level") != "ERROR"


def test_app_cost_rollup_is_a_cost_only_generation_of_the_item(monkeypatch):
    exporter = InMemorySpanExporter()
    client = real_client(exporter)
    gateway = gateway_for(client)
    monkeypatch.setattr(client, "create_score", lambda **kwargs: None)

    async def task(*, item, **kwargs):
        gateway.record_app_cost(0.42, ["t1", "t2"])
        return {"scores": []}

    dataset = SimpleNamespace(
        run_experiment=lambda **kwargs: client.run_experiment(
            data=[{"input": "rain next week"}], **kwargs
        )
    )
    gateway.run_experiment(
        dataset=dataset, name="cfg", description="m", task=task, metadata={}, concurrency=1
    )
    client.flush()

    spans = exporter.get_finished_spans()
    task_span = next(s for s in spans if s.name == "experiment-item-task")
    rollup = next(s for s in spans if s.name == "app-cost-rollup")
    attributes = rollup.attributes
    assert rollup.context.trace_id == task_span.context.trace_id
    assert attributes["langfuse.observation.type"] == "generation"
    assert json.loads(attributes["langfuse.observation.cost_details"]) == {"total": 0.42}
    assert json.loads(attributes["langfuse.observation.input"]) == {"app_trace_ids": ["t1", "t2"]}
    # No usage and no model: no tokens counted, no price inferred.
    assert "langfuse.observation.usage_details" not in attributes
    assert "langfuse.observation.model.name" not in attributes
    # It belongs to the experiment item, so the item's cost includes it.
    assert attributes["langfuse.experiment.item.root_observation_id"] == format(
        task_span.context.span_id, "016x"
    )


def _obs(obs_id, trace_id, kind="GENERATION", name="llm", ended=True, metadata=None, output=None):
    return SimpleNamespace(
        output=output,
        id=obs_id,
        trace_id=trace_id,
        type=kind,
        name=name,
        level=SimpleNamespace(value="DEFAULT"),
        is_root_observation=kind == "SPAN",
        end_time="t" if ended else None,
        model=f"model-{obs_id}" if kind == "GENERATION" else None,
        usage_details={"input": 1, "output": 1},
        cost_details={"total": 0.1},
        total_cost=0.1,
        metadata=metadata,
    )


class FakeObservationsApi:
    """A chat session holding app traces t1 and t2 (t2 appears on read
    ``t2_from_read``, and its tool k1's output on read ``outputs_from_read``;
    before that, the output read does not return k1; ``k1_output`` is its output)
    plus the simulated user's and judge's traces, which share the session id
    but must not be counted."""

    def __init__(
        self,
        t2_from_read: int = 3,
        outputs_from_read: int = 0,
        k1_output: str | None = '{"ok": false, "error": "x"}',
        k1_ends_from_read: int = 0,
        open_k1_in_tool_read: bool = True,
        k3_from_read: int = 999,
    ):
        self.reads = 0
        self.requests: list[dict] = []
        self.t2_from_read = t2_from_read
        self.outputs_from_read = outputs_from_read
        self.k1_output = k1_output
        # Before this read, the output read finds k1 still running, without output.
        self.k1_ends_from_read = k1_ends_from_read
        # False: the output read does not return k1 while it is open.
        self.open_k1_in_tool_read = open_k1_in_tool_read
        # From this read on, t2 also holds an ended tool k3.
        self.k3_from_read = k3_from_read

    def _session(self):
        observations = [
            _obs("r1", "t1", "SPAN", "weather-skills-chat"),
            _obs("su", "user-trace", "GENERATION", "simulated-user"),
            _obs("jg", "judge-trace", "GENERATION", "judge"),
        ]
        if self.reads >= self.t2_from_read:
            observations.append(_obs("r2", "t2", "SPAN", "weather-skills-chat"))
        return observations

    def get_many(self, *, fields, limit, cursor, **filters):
        self.requests.append({"fields": fields, "limit": limit, **filters})
        if filters.get("type") == "TOOL":
            assert "io" in fields.split(",")
            if filters["trace_id"] != "t2":
                return SimpleNamespace(data=[], meta=SimpleNamespace(cursor=None))
            tools = [_obs("k2", "t2", "TOOL", "tool:fetch", output="not json")]
            ended = self.reads >= self.k1_ends_from_read
            if self.reads >= self.outputs_from_read and (ended or self.open_k1_in_tool_read):
                output = self.k1_output if ended else None
                tools.append(_obs("k1", "t2", "TOOL", "tool:plot", ended=ended, output=output))
            if self.reads >= self.k3_from_read:
                tools.append(_obs("k3", "t2", "TOOL", "tool:clip", output='{"ok": true}'))
            return SimpleNamespace(data=tools, meta=SimpleNamespace(cursor=None))
        if "session_id" in filters:
            self.reads += 1
            matches = [o for o in self._session() if o.name == filters.get("name", o.name)]
            return SimpleNamespace(data=matches, meta=SimpleNamespace(cursor=None))
        trace = filters["trace_id"]
        if trace in ("user-trace", "judge-trace"):
            raise AssertionError(f"read a harness trace: {trace}")
        # Two pages for t1, to follow the cursor.
        if trace == "t1" and cursor is None:
            return SimpleNamespace(
                data=[_obs("r1", "t1", "SPAN", "weather-skills-chat")],
                meta=SimpleNamespace(cursor="next"),
            )
        if trace == "t1":
            return SimpleNamespace(
                data=[
                    _obs(
                        "g1",
                        "t1",
                        metadata={"attributes.gen_ai.usage.input_tokens.cached": "100520"},
                    )
                ],
                meta=SimpleNamespace(cursor=None),
            )
        return SimpleNamespace(
            data=[
                _obs("r2", "t2", "SPAN", "weather-skills-chat"),
                _obs("g2", "t2"),
                _obs("k1", "t2", "TOOL", "tool:plot", ended=self.reads >= self.k1_ends_from_read),
                _obs("k2", "t2", "TOOL", "tool:fetch"),
                *(
                    [_obs("k3", "t2", "TOOL", "tool:clip")]
                    if self.reads >= self.k3_from_read
                    else []
                ),
            ],
            meta=SimpleNamespace(cursor=None),
        )


def fake_client(api) -> SimpleNamespace:
    return SimpleNamespace(api=SimpleNamespace(observations=api))


def test_chat_observations_read_only_app_traces_and_wait_until_settled():
    clock = FakeClock()
    api = FakeObservationsApi()
    gateway = gateway_for(
        fake_client(api),
        trace_timeout=600,
        trace_settle=10,
        poll_interval=5,
        clock=clock,
        sleep=clock.sleep,
    )

    read = asyncio.run(gateway.chat_observations("chat-1", 2))
    observations = read.observations

    assert read.complete
    assert read.read_error is None
    assert sorted(o["id"] for o in observations) == ["g1", "g2", "k1", "k2", "r1", "r2"]
    assert {o["level"] for o in observations} == {"DEFAULT"}
    # Flat dotted metadata keys pass through intact.
    assert next(o for o in observations if o["id"] == "g1")["metadata"] == {
        "attributes.gen_ai.usage.input_tokens.cached": "100520"
    }
    assert next(o for o in observations if o["id"] == "g2")["model"] == "model-g2"
    session_requests = [r for r in api.requests if "session_id" in r]
    assert all(r["name"] == "weather-skills-chat" for r in session_requests)
    # A tool's output is read for its "ok"; no other observation's input or output is read.
    by_id = {o["id"]: o for o in observations}
    assert by_id["k1"]["ok"] is False
    assert by_id["k2"]["ok"] is None
    assert "ok" not in by_id["g2"]
    for request in api.requests:
        fields = set(request["fields"].split(","))
        if "io" in fields:
            assert request["type"] == "TOOL"
            assert request["limit"] == TOOL_OUTPUT_PAGE_SIZE
        else:
            assert {"metadata", "model"} <= fields
    # Each trace's tool outputs are read after its full read.
    t2 = [r for r in api.requests if r.get("trace_id") == "t2"]
    assert [r.get("type") for r in t2[:2]] == [None, "TOOL"]
    # Outputs already read are not read again on later polls.
    assert [r.get("type") for r in t2].count("TOOL") == 1
    assert api.reads >= 5


def test_chat_observations_give_up_after_the_timeout():
    clock = FakeClock()
    gateway = gateway_for(
        fake_client(FakeObservationsApi()),
        trace_timeout=30,
        trace_settle=10,
        poll_interval=5,
        clock=clock,
        sleep=clock.sleep,
    )

    read = asyncio.run(gateway.chat_observations("chat-1", 5))

    assert not read.complete
    assert read.read_error is None
    assert read.observations
    assert read.traces_arriving


def test_tool_whose_output_is_not_read_yet_keeps_the_poll_going():
    """k1 is in t2's full read before its output can be read: a tool that
    appeared between the two reads."""
    clock = FakeClock()
    api = FakeObservationsApi(outputs_from_read=6)
    gateway = gateway_for(
        fake_client(api),
        trace_timeout=600,
        trace_settle=10,
        poll_interval=5,
        clock=clock,
        sleep=clock.sleep,
    )

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert read.complete
    assert {o["id"]: o.get("ok") for o in read.observations}["k1"] is False
    assert api.reads >= 8


def test_a_tool_whose_output_is_null_is_read_and_settles():
    clock = FakeClock()
    api = FakeObservationsApi(k1_output=None)
    gateway = gateway_for(
        fake_client(api),
        trace_timeout=600,
        trace_settle=10,
        poll_interval=5,
        clock=clock,
        sleep=clock.sleep,
    )

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert read.complete
    assert read.unread_tools == 0
    assert {o["id"]: o.get("ok", "unread") for o in read.observations}["k1"] is None
    # Read once, not again on later polls.
    tool_reads = [r for r in api.requests if r.get("type") == "TOOL" and r["trace_id"] == "t2"]
    assert len(tool_reads) == 1


def test_a_running_tool_is_read_again_until_it_has_ended():
    clock = FakeClock()
    api = FakeObservationsApi(k1_ends_from_read=6)
    gateway = gateway_for(
        fake_client(api),
        trace_timeout=600,
        trace_settle=10,
        poll_interval=5,
        clock=clock,
        sleep=clock.sleep,
    )

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert read.complete
    # The {"ok": false} written when the tool ended is counted.
    assert {o["id"]: o.get("ok", "unread") for o in read.observations}["k1"] is False
    tool_reads = [r for r in api.requests if r.get("type") == "TOOL" and r["trace_id"] == "t2"]
    assert len(tool_reads) >= 2


def test_a_tool_that_never_ends_is_read_once_after_the_traces_settle():
    clock = FakeClock()
    api = FakeObservationsApi(k1_ends_from_read=999)
    gateway = gateway_for(
        fake_client(api),
        trace_timeout=600,
        trace_settle=10,
        poll_interval=5,
        clock=clock,
        sleep=clock.sleep,
    )
    start = clock.now

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    # Complete well before the timeout, with the open tool's output as it was.
    assert read.complete
    assert read.unread_tools == 0
    assert {o["id"]: o.get("ok", "unread") for o in read.observations}["k1"] is None
    assert clock.now - start < 600


def test_a_forced_read_that_finds_a_new_tool_polls_on_until_settled():
    clock = FakeClock()
    # t2 appears on read 3, the traces settle on read 5, the forced read is
    # read 6, and it finds a new ended tool k3.
    api = FakeObservationsApi(k1_ends_from_read=999, k3_from_read=6)
    gateway = gateway_for(
        fake_client(api),
        trace_timeout=600,
        trace_settle=10,
        poll_interval=5,
        clock=clock,
        sleep=clock.sleep,
    )

    k3_seen_at = []
    read_session = api.get_many

    def timed(**kwargs):
        if "session_id" in kwargs and api.reads + 1 == api.k3_from_read:
            k3_seen_at.append(clock.now)
        return read_session(**kwargs)

    api.get_many = timed

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert read.complete
    oks = {o["id"]: o.get("ok", "unread") for o in read.observations}
    assert oks["k3"] is True
    assert oks["k1"] is None
    # The read that found k3 was not final: the traces settled again after it.
    assert clock.now - k3_seen_at[0] >= 10


def test_a_non_final_forced_read_keeps_no_output_of_a_tool_still_open():
    clock = FakeClock()
    # The forced read (read 6) finds k3; k1 is still open then and ends on the
    # next read with {"ok": false}.
    api = FakeObservationsApi(k1_ends_from_read=7, k3_from_read=6)
    gateway = gateway_for(
        fake_client(api),
        trace_timeout=600,
        trace_settle=10,
        poll_interval=5,
        clock=clock,
        sleep=clock.sleep,
    )

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert read.complete
    oks = {o["id"]: o.get("ok", "unread") for o in read.observations}
    assert oks["k1"] is False
    assert oks["k3"] is True


def test_missing_traces_come_before_unread_tool_outputs():
    arriving = TraceRead([], False, unread_tools=2, traces_arriving=True).partial_reason()
    assert arriving == TraceRead([], False).partial_reason()
    assert arriving != TraceRead([], False, unread_tools=2).partial_reason()


def test_a_forced_read_that_misses_the_open_tool_is_partial():
    clock = FakeClock()
    api = FakeObservationsApi(k1_ends_from_read=999, open_k1_in_tool_read=False)
    gateway = gateway_for(
        fake_client(api),
        trace_timeout=600,
        trace_settle=10,
        poll_interval=5,
        clock=clock,
        sleep=clock.sleep,
    )
    start = clock.now

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert not read.complete
    assert read.unread_tools == 1
    assert read.forced
    assert clock.now - start < 600


def test_timeout_with_unread_tool_outputs_counts_them():
    clock = FakeClock()
    api = FakeObservationsApi(outputs_from_read=999)
    gateway = gateway_for(
        fake_client(api),
        trace_timeout=60,
        trace_settle=10,
        poll_interval=5,
        clock=clock,
        sleep=clock.sleep,
    )

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert not read.complete
    assert read.unread_tools == 1
    assert not read.traces_arriving


def test_chat_observations_raise_trace_read_error_when_no_read_succeeds_by_the_timeout():
    clock = FakeClock()

    class Failing:
        def __init__(self):
            self.calls = 0

        def get_many(self, **kwargs):
            self.calls += 1
            raise httpx.ReadTimeout(f"timed out, read {self.calls}")

    api = Failing()
    gateway = gateway_for(
        fake_client(api), trace_timeout=30, poll_interval=5, clock=clock, sleep=clock.sleep
    )

    with pytest.raises(TraceReadError) as error:
        asyncio.run(gateway.chat_observations("chat-1", 1))

    assert isinstance(error.value.__cause__, httpx.ReadTimeout)
    # Retried at the poll interval until the timeout.
    assert api.calls == 7
    assert clock.sleeps == [5] * 6


def test_an_error_that_is_not_langfuses_is_raised_at_once():
    clock = FakeClock()

    class Broken:
        def __init__(self):
            self.calls = 0

        def get_many(self, **kwargs):
            self.calls += 1
            raise TypeError("unexpected keyword")

    api = Broken()
    gateway = gateway_for(
        fake_client(api), trace_timeout=30, poll_interval=5, clock=clock, sleep=clock.sleep
    )

    with pytest.raises(TraceReadError) as error:
        asyncio.run(gateway.chat_observations("chat-1", 1))
    assert isinstance(error.value.__cause__, TypeError)
    assert api.calls == 1
    assert clock.sleeps == []


class FailingFirst(FakeObservationsApi):
    """Fails the first ``failures`` session reads, then answers as the parent."""

    def __init__(self, failures: int):
        super().__init__()
        self.failures = failures
        self.status = 504

    def get_many(self, **kwargs):
        if "session_id" in kwargs and self.failures:
            self.failures -= 1
            raise LangfuseApiError(status_code=self.status, body="Read timed out")
        return super().get_many(**kwargs)


def test_chat_observations_retry_a_failed_read():
    clock = FakeClock()
    gateway = gateway_for(
        fake_client(FailingFirst(failures=1)),
        trace_timeout=600,
        trace_settle=10,
        poll_interval=5,
        clock=clock,
        sleep=clock.sleep,
    )

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert read.complete
    assert read.read_error is None
    assert sorted(o["id"] for o in read.observations) == ["g1", "g2", "k1", "k2", "r1", "r2"]


@pytest.mark.parametrize("status", [403, None])
def test_a_permanent_langfuse_error_is_raised_at_once(status):
    clock = FakeClock()
    api = FailingFirst(failures=1)
    api.status = status
    gateway = gateway_for(
        fake_client(api), trace_timeout=600, poll_interval=5, clock=clock, sleep=clock.sleep
    )

    with pytest.raises(TraceReadError) as error:
        asyncio.run(gateway.chat_observations("chat-1", 2))
    assert error.value.__cause__.status_code == status
    assert clock.sleeps == []


def test_a_permanent_error_after_a_read_returns_that_read_as_partial():
    clock = FakeClock()
    api = FakeObservationsApi()
    read_once = api.get_many

    def fail_after_the_first_read(**kwargs):
        if api.reads >= 1 and "session_id" in kwargs:
            raise LangfuseApiError(status_code=403, body="forbidden")
        return read_once(**kwargs)

    api.get_many = fail_after_the_first_read
    gateway = gateway_for(
        fake_client(api), trace_timeout=600, poll_interval=5, clock=clock, sleep=clock.sleep
    )
    start = clock.now

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert not read.complete
    assert read.read_error is not None
    assert sorted(o["id"] for o in read.observations) == ["g1", "r1"]
    # Returned at once, without waiting for the timeout.
    assert clock.now - start == 5


def test_failed_reads_do_not_restart_the_settle_time():
    clock = FakeClock()
    # Reads 2 to 4 fail; the observations have not changed, so the settle time
    # (10 s) still runs from read 1.
    api = FakeObservationsApi(t2_from_read=99)
    read = api.get_many
    failing = {"left": 3}

    def flaky(**kwargs):
        if "session_id" in kwargs and api.reads >= 1 and failing["left"]:
            failing["left"] -= 1
            raise httpx.ConnectError("refused")
        return read(**kwargs)

    api.get_many = flaky
    gateway = gateway_for(
        fake_client(api),
        trace_timeout=600,
        trace_settle=10,
        poll_interval=5,
        clock=clock,
        sleep=clock.sleep,
    )
    start = clock.now

    read = asyncio.run(gateway.chat_observations("chat-1", 1))

    assert read.complete
    # Read 1 at 0 s, failures at 5, 10 and 15 s, and the read at 20 s settles it.
    assert clock.now - start == 20


def test_chat_observations_that_fail_after_a_read_return_it_with_the_error():
    clock = FakeClock()
    api = FakeObservationsApi()
    gateway = gateway_for(
        fake_client(api), trace_timeout=30, poll_interval=5, clock=clock, sleep=clock.sleep
    )
    read = api.get_many

    def fail_after_the_first_read(**kwargs):
        if api.reads >= 1 and "session_id" in kwargs:
            raise httpx.ReadTimeout("Read timed out")
        return read(**kwargs)

    api.get_many = fail_after_the_first_read

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert not read.complete
    assert read.read_error is not None
    assert sorted(o["id"] for o in read.observations) == ["g1", "r1"]


def test_a_reply_cut_off_at_max_tokens_is_an_error(monkeypatch):
    gateway = gateway_for(real_client(InMemorySpanExporter()), model_max_tokens=50)
    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kwargs: litellm.ModelResponse(
            choices=[
                {
                    "message": {"role": "assistant", "content": '{"classif'},
                    "finish_reason": "length",
                }
            ],
            usage={"prompt_tokens": 1, "completion_tokens": 50, "total_tokens": 51},
        ),
    )

    with pytest.raises(ModelReplyError):
        gateway.traced_completion("simulated-user", "chat-1", model="m", messages=[])
    with pytest.raises(ModelReplyError):
        gateway.traced_completion(
            "judge", "chat-1", model="m", messages=[], max_completion_tokens=20
        )


def test_served_model_is_the_model_openrouter_names_in_its_reply(monkeypatch):
    gateway = gateway_for(real_client(InMemorySpanExporter()), model_retries=3)
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return litellm.ModelResponse(
            model="openrouter/anthropic/claude-opus-5.5",
            choices=[{"message": {"role": "assistant", "content": "OK"}}],
        )

    monkeypatch.setattr(litellm, "completion", completion)

    assert gateway.served_model("~anthropic/claude-opus-latest") == "anthropic/claude-opus-5.5"
    [sent] = calls
    assert sent["model"] == "openrouter/~anthropic/claude-opus-latest"
    assert sent["max_tokens"] == SERVED_MODEL_MAX_TOKENS
    assert sent["num_retries"] == 3


def test_a_reply_that_names_no_model_is_an_error(monkeypatch):
    gateway = gateway_for(real_client(InMemorySpanExporter()))
    reply = litellm.ModelResponse(choices=[{"message": {"role": "assistant", "content": "OK"}}])
    reply.model = None
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: reply)

    with pytest.raises(ModelReplyError):
        gateway.served_model("m")


def test_a_reply_that_names_the_alias_is_an_error(monkeypatch):
    gateway = gateway_for(real_client(InMemorySpanExporter()))
    reply = litellm.ModelResponse(
        model="openrouter/~anthropic/claude-opus-latest",
        choices=[{"message": {"role": "assistant", "content": "OK"}}],
    )
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: reply)

    with pytest.raises(ModelReplyError):
        gateway.served_model("~anthropic/claude-opus-latest")


def rate_limited(retry_after: str | None = None) -> litellm.RateLimitError:
    """The error LiteLLM raises for an OpenRouter 429."""
    return litellm.RateLimitError(
        "temporarily rate-limited upstream",
        llm_provider="openrouter",
        model="meta-llama/llama-4-maverick",
        response=httpx.Response(429, headers={"retry-after": retry_after} if retry_after else {}),
    )


def failing_first(monkeypatch, errors: list[Exception], reply) -> list[dict]:
    """``litellm.completion`` raises ``errors`` in order, then returns ``reply``;
    the list returned holds the kwargs of each call."""
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if len(calls) <= len(errors):
            raise errors[len(calls) - 1]
        return reply

    monkeypatch.setattr(litellm, "completion", completion)
    return calls


def ok_reply():
    return litellm.ModelResponse(
        model="openrouter/meta-llama/llama-4-maverick",
        choices=[{"message": {"role": "assistant", "content": "{}"}}],
        usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    )


def user_call(gateway):
    return gateway.traced_completion(
        "simulated-user",
        "chat-1",
        model="openrouter/meta-llama/llama-4-maverick",
        messages=[{"role": "user", "content": "x"}],
    )


def test_a_rate_limited_call_waits_then_returns_the_reply_in_one_generation(monkeypatch):
    exporter = InMemorySpanExporter()
    client = real_client(exporter)
    clock = FakeClock()
    gateway = gateway_for(client, rate_limit_sleep=clock.sleep_sync)
    reply = ok_reply()
    calls = failing_first(monkeypatch, [rate_limited(), rate_limited()], reply)

    response = user_call(gateway)
    client.flush()

    assert response is reply
    assert len(calls) == 3
    assert clock.sleeps == [15, 30]
    generations = [s for s in exporter.get_finished_spans() if s.name == "simulated-user"]
    assert len(generations) == 1
    assert generations[0].attributes["langfuse.observation.type"] == "generation"


def test_rate_limit_waits_double_from_the_backoff_and_the_last_429_is_raised(monkeypatch):
    clock = FakeClock()
    gateway = gateway_for(
        real_client(InMemorySpanExporter()),
        rate_limit_retries=6,
        rate_limit_backoff=15,
        rate_limit_sleep=clock.sleep_sync,
    )
    calls = failing_first(monkeypatch, [rate_limited() for _ in range(7)], ok_reply())

    with pytest.raises(litellm.RateLimitError):
        user_call(gateway)

    assert len(calls) == 7
    assert clock.sleeps == [15, 30, 60, 120, 240, 480]


def test_a_retry_after_longer_than_the_backoff_is_waited(monkeypatch):
    clock = FakeClock()
    gateway = gateway_for(
        real_client(InMemorySpanExporter()),
        rate_limit_backoff=15,
        rate_limit_sleep=clock.sleep_sync,
    )
    reply = ok_reply()
    failing_first(monkeypatch, [rate_limited("100"), rate_limited("1")], reply)

    assert user_call(gateway) is reply
    # A Retry-After shorter than the backoff does not shorten the wait.
    assert clock.sleeps == [100, 30]


def test_an_error_that_is_not_a_rate_limit_is_not_waited_on(monkeypatch):
    clock = FakeClock()
    gateway = gateway_for(real_client(InMemorySpanExporter()), rate_limit_sleep=clock.sleep_sync)
    error = litellm.InternalServerError("upstream failed", llm_provider="openrouter", model="m")
    calls = failing_first(monkeypatch, [error], ok_reply())

    with pytest.raises(litellm.InternalServerError):
        user_call(gateway)

    assert len(calls) == 1
    assert clock.sleeps == []


def test_a_rate_limited_served_model_request_waits_and_is_made_again(monkeypatch):
    clock = FakeClock()
    gateway = gateway_for(real_client(InMemorySpanExporter()), rate_limit_sleep=clock.sleep_sync)
    calls = failing_first(monkeypatch, [rate_limited()], ok_reply())

    assert gateway.served_model("meta-llama/llama-4-maverick") == "meta-llama/llama-4-maverick"
    assert len(calls) == 2
    assert clock.sleeps == [15]


def test_a_retry_after_longer_than_the_last_wait_is_cut_to_it(monkeypatch):
    clock = FakeClock()
    gateway = gateway_for(
        real_client(InMemorySpanExporter()),
        rate_limit_retries=6,
        rate_limit_backoff=15,
        rate_limit_sleep=clock.sleep_sync,
    )
    reply = ok_reply()
    failing_first(monkeypatch, [rate_limited("86400")], reply)

    assert user_call(gateway) is reply
    assert clock.sleeps == [480]


@pytest.mark.parametrize(
    "date", ["Fri, 01 Jan 2100 00:00:00 GMT", "Fri, 01 Jan 2100 00:00:00 -0000"], ids=repr
)
def test_a_retry_after_http_date_is_waited_up_to_the_last_wait(monkeypatch, date):
    # -0000 parses without a time zone and is taken as UTC.
    clock = FakeClock()
    gateway = gateway_for(real_client(InMemorySpanExporter()), rate_limit_sleep=clock.sleep_sync)
    reply = ok_reply()
    failing_first(monkeypatch, [rate_limited(date)], reply)

    assert user_call(gateway) is reply
    assert clock.sleeps == [480]


def test_a_retry_after_only_in_the_errors_headers_is_waited(monkeypatch):
    clock = FakeClock()
    gateway = gateway_for(real_client(InMemorySpanExporter()), rate_limit_sleep=clock.sleep_sync)
    error = litellm.RateLimitError(
        "rate-limited",
        llm_provider="openrouter",
        model="m",
        response=httpx.Response(429, headers={"retry-after": "not a date"}),
        headers={"Retry-After": "100"},
    )
    reply = ok_reply()
    failing_first(monkeypatch, [error], reply)

    assert user_call(gateway) is reply
    # The unreadable response header does not stop the error's own header being read.
    assert clock.sleeps == [100]


def test_the_default_gateway_gets_its_settings(monkeypatch):
    settings = load_settings(
        {
            "LANGFUSE_PUBLIC_KEY": "pk",
            "LANGFUSE_SECRET_KEY": "sk",
            "OPENROUTER_API_KEY": "or",
            "LANGFUSE_TRACING_ENVIRONMENT": "eval",
            "EVAL_SIMULATED_USER_MODEL": "u",
            "EVAL_JUDGE_MODELS": "j",
            "EVAL_DOCKER_NETWORK": "n",
            "EVAL_RATE_LIMIT_RETRIES": "3",
            "EVAL_RATE_LIMIT_BACKOFF_SECONDS": "2.5",
        }
    )
    built = {}
    monkeypatch.setattr(runner, "LangfuseGateway", lambda **kwargs: built.update(kwargs))
    runner._default_gateway(settings)
    # Every setting the gateway takes; the rest are its client and test seams.
    assert set(built) == {
        "environment",
        "trace_timeout",
        "trace_settle",
        "poll_interval",
        "model_max_tokens",
        "model_retries",
        "rate_limit_retries",
        "rate_limit_backoff",
    }
    # Each keyword is the Settings field of the same name, except the environment.
    fields = {"environment": "langfuse_environment"}
    assert built == {name: getattr(settings, fields.get(name, name)) for name in built}


def _completion_kwargs(monkeypatch, gateway, **kwargs) -> dict:
    calls = []

    def completion(**sent):
        calls.append(sent)
        return litellm.ModelResponse(
            choices=[{"message": {"role": "assistant", "content": "{}"}}],
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        )

    monkeypatch.setattr(litellm, "completion", completion)
    gateway.traced_completion(
        "judge", "chat-1", model="m", messages=[{"role": "user", "content": "x"}], **kwargs
    )
    return calls[0]


def test_a_model_call_is_capped_at_the_model_max_tokens_unless_it_sets_its_own(monkeypatch):
    gateway = gateway_for(real_client(InMemorySpanExporter()), model_max_tokens=8192)
    # Scenario's judge sends max_tokens=None.
    assert _completion_kwargs(monkeypatch, gateway, max_tokens=None)["max_tokens"] == 8192
    # A call that sets its own limit keeps it.
    assert _completion_kwargs(monkeypatch, gateway, max_tokens=100)["max_tokens"] == 100
    sent = _completion_kwargs(monkeypatch, gateway, max_completion_tokens=100)
    assert sent["max_completion_tokens"] == 100
    assert "max_tokens" not in sent


def test_real_client_reads_tool_outputs_through_the_http_layer():
    """Only the HTTP layer is stubbed: the Langfuse client builds the queries
    and parses the responses."""
    requests = []

    def observation(obs_id, trace_id, kind, name, **extra):
        return {
            "id": obs_id,
            "traceId": trace_id,
            "startTime": "2026-10-21T10:00:00Z",
            "endTime": "2026-10-21T10:00:01Z",
            "projectId": "p",
            "type": kind,
            "name": name,
            "level": "DEFAULT",
            **extra,
        }

    def handle(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        requests.append(params)
        if "sessionId" in params:
            data = [observation("r1", "t1", "SPAN", "weather-skills-chat", isRootObservation=True)]
        elif params.get("type") == "TOOL":
            data = [observation("k1", "t1", "TOOL", "tool:plot", output='{"ok": false}')]
        else:
            data = [
                observation("r1", "t1", "SPAN", "weather-skills-chat", isRootObservation=True),
                observation("k1", "t1", "TOOL", "tool:plot"),
            ]
        return httpx.Response(200, json={"data": data, "meta": {"cursor": None}})

    client = Langfuse(
        public_key=f"pk-test-{uuid.uuid4()}",
        secret_key="sk-test",
        base_url="http://langfuse.invalid",
        httpx_client=httpx.Client(transport=httpx.MockTransport(handle)),
        span_exporter=InMemorySpanExporter(),
        tracer_provider=TracerProvider(),
    )
    clock = FakeClock()
    gateway = gateway_for(
        client, trace_timeout=600, trace_settle=10, poll_interval=5, clock=clock, sleep=clock.sleep
    )

    read = asyncio.run(gateway.chat_observations("chat-1", 1))

    assert read.complete
    assert {o["id"]: o.get("ok") for o in read.observations} == {"r1": None, "k1": False}
    tool_reads = [r for r in requests if r.get("type") == "TOOL"]
    assert tool_reads[0]["fields"] == TOOL_OUTPUT_FIELDS
    assert tool_reads[0]["traceId"] == "t1"
    # Only the tool read asks for inputs and outputs.
    assert all("io" not in r["fields"].split(",") for r in requests if r.get("type") != "TOOL")


def test_paging_stops_when_the_cursor_repeats():
    class Looping:
        def __init__(self):
            self.calls = 0

        def get_many(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                data=[_obs(f"o{self.calls}", "t")], meta=SimpleNamespace(cursor="same")
            )

    api = Looping()
    gateway = gateway_for(fake_client(api))

    pages = gateway._pages(trace_id="t")

    assert api.calls == 2
    assert len(pages) == 2


def test_play_scores_become_evaluations():
    evaluations = play_scores(
        output={"scores": [{"name": "turns_to_goal", "value": 2, "comment": "c"}]}
    )
    assert [(e.name, e.value, e.comment) for e in evaluations] == [("turns_to_goal", 2, "c")]


# Deleting a run

ALLOWED_RUN_CALLS = {
    "experiments.list",
    "experiments.list_items",
    "observations.get_many",
    "trace.delete_multiple",
}


def run_api(**options) -> FakeRunApi:
    """Experiment e1 (cfg) with three items: two with chats, one without.
    chat-9 belongs to another run."""
    defaults = {
        "experiments": {"e1": "cfg", "e2": "empty"},
        "items": {
            "e1": [
                ("item-1", {"chat_id": "chat-1", "app_trace_ids": ["app-a"]}),
                # Langfuse can return the output as JSON text.
                ("item-2", json.dumps({"chat_id": "chat-2"})),
                ("item-3", {"chat_id": None}),
            ]
        },
        "sessions": {
            "chat-1": ["item-1", "app-a", "app-b", "user-1", "judge-1"],
            "chat-2": ["item-2", "app-c"],
            "chat-9": ["other-run"],
        },
    }
    defaults.update(options)
    return FakeRunApi(**defaults)


def test_a_run_is_its_item_traces_plus_every_trace_in_their_chat_sessions():
    api = run_api()
    runs = LangfuseRuns(client=api)

    found = runs.run_traces("e1")
    runs.delete_traces(found)

    expected = ["app-a", "app-b", "app-c", "item-1", "item-2", "item-3", "judge-1", "user-1"]
    assert found == RunTraces("cfg", 3, expected)
    assert found.item_trace_ids == ["item-1", "item-2", "item-3"]
    # The item traces go last.
    assert api.deleted == [t for t in expected if not t.startswith("item-")] + [
        "item-1",
        "item-2",
        "item-3",
    ]
    assert api.methods() <= ALLOWED_RUN_CALLS
    # The item traces go in a request of their own, though all would fit in one.
    assert delete_batches(api) == [
        [t for t in expected if not t.startswith("item-")],
        ["item-1", "item-2", "item-3"],
    ]
    # Each session is read whole, page by page.
    sessions = [
        kwargs
        for method, kwargs in api.calls
        if method == "observations.get_many" and "session_id" in kwargs
    ]
    assert [(s["session_id"], s["cursor"]) for s in sessions] == [
        ("chat-1", None),
        ("chat-1", "2"),
        ("chat-1", "4"),
        ("chat-2", None),
    ]
    # Every item page is read before anything is deleted.
    item_reads = [kwargs for method, kwargs in api.calls if method == "experiments.list_items"]
    assert [r["cursor"] for r in item_reads] == [None, "2"]
    assert all(r["experiment_id"] == "e1" for r in item_reads)
    assert [method for method, _ in api.calls][-1] == "trace.delete_multiple"


def test_an_item_without_a_chat_contributes_only_its_own_trace():
    api = run_api(
        items={"e1": [("item-3", {"chat_id": None}), ("item-4", None)]},
        traces={"item-3": [None, None]},
    )

    found = LangfuseRuns(client=api).run_traces("e1")

    assert found == RunTraces("cfg", 2, ["item-3", "item-4"])
    reads = [kwargs for method, kwargs in api.calls if method == "observations.get_many"]
    # Each item trace is read for a session; neither has one, so no session is read.
    assert [r.get("trace_id") for r in reads] == ["item-3", "item-4"]


def test_a_failed_task_contributes_the_chat_of_its_item_trace():
    # The task raised after its play linked the item trace to chat-1, so the
    # item has no output.
    api = run_api(
        items={"e1": [("item-1", None), ("item-2", {"chat_id": "chat-2"})]},
        traces={"item-1": [None, "chat-1", "chat-1"]},
    )

    found = LangfuseRuns(client=api).run_traces("e1")

    expected = ["app-a", "app-b", "app-c", "item-1", "item-2", "judge-1", "user-1"]
    assert found == RunTraces("cfg", 2, expected)
    reads = [kwargs for method, kwargs in api.calls if method == "observations.get_many"]
    # The item trace is read whole, page by page, before the sessions.
    assert [(r.get("trace_id"), r.get("session_id"), r["cursor"]) for r in reads[:2]] == [
        ("item-1", None, None),
        ("item-1", None, "2"),
    ]
    # An item with a chat id in its output is not read for one.
    assert all(r.get("trace_id") != "item-2" for r in reads)
    assert api.methods() <= ALLOWED_RUN_CALLS


def test_an_unknown_experiment_id_is_not_found():
    api = run_api()

    with pytest.raises(ExperimentNotFound):
        LangfuseRuns(client=api).run_traces("missing")

    assert api.methods() == {"experiments.list"}


def test_an_experiment_without_items_has_no_traces():
    api = run_api()

    assert LangfuseRuns(client=api).run_traces("e2") == RunTraces("empty", 0, [])
    assert api.methods() == {"experiments.list", "experiments.list_items"}


def delete_batches(api: FakeRunApi) -> list[list[str]]:
    return [
        kwargs["trace_ids"] for method, kwargs in api.calls if method == "trace.delete_multiple"
    ]


def test_trace_ids_are_deleted_in_batches_of_the_batch_size_in_order():
    api = run_api()
    size = DELETE_BATCH_SIZE
    trace_ids = [f"t{n:04d}" for n in range(2 * size + size // 2)]

    LangfuseRuns(client=api).delete_traces(RunTraces("cfg", 1, trace_ids))

    assert delete_batches(api) == [
        trace_ids[:size],
        trace_ids[size : 2 * size],
        trace_ids[2 * size :],
    ]
    assert api.deleted == trace_ids


def big_run_api(**options) -> FakeRunApi:
    """Experiment e1 whose chat-1 session holds two and a half delete batches of
    traces besides its three items."""
    session = [f"s{n:05d}" for n in range(2 * DELETE_BATCH_SIZE + DELETE_BATCH_SIZE // 2)]
    return run_api(sessions={"chat-1": ["item-1", *session], "chat-2": ["item-2"]}, **options)


def test_a_failed_last_batch_of_other_traces_leaves_every_item_trace():
    # The third request, the last batch of other traces, fails.
    api = big_run_api(
        page_size=1000, delete_errors={3: LangfuseApiError(status_code=500, body="boom")}
    )
    runs = LangfuseRuns(client=api)
    found = runs.run_traces("e1")

    with pytest.raises(LangfuseApiError):
        runs.delete_traces(found)

    assert len(api.deleted) == 2 * DELETE_BATCH_SIZE
    assert not set(found.item_trace_ids) & set(api.deleted)
    # The run is still found, with the traces left.
    assert len(runs.run_traces("e1").trace_ids) == len(found.trace_ids) - 2 * DELETE_BATCH_SIZE


def test_a_langfuse_error_while_finding_a_run_is_raised():
    api = run_api(error=LangfuseApiError(status_code=500, body="boom"))

    with pytest.raises(LangfuseApiError):
        LangfuseRuns(client=api).run_traces("e1")


def test_real_client_finds_and_deletes_a_run_through_the_http_layer():
    """Only the HTTP layer is stubbed: the Langfuse client builds the requests."""
    requests = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path, params = request.url.path, dict(request.url.params)
        if path == "/api/public/experiments":
            data = [
                {
                    "id": "e1",
                    "name": "cfg",
                    "startTime": "2026-10-20T09:00:00Z",
                    "endTime": "2026-10-20T10:00:00Z",
                    "itemCount": 1,
                }
            ]
        elif path == "/api/public/experiment-items":
            data = [
                {
                    "id": "x1",
                    "traceId": "item-1",
                    "startTime": "2026-10-20T09:00:01Z",
                    "level": "DEFAULT",
                    "environment": "sdk-experiment",
                    "experimentId": "e1",
                    "experimentName": "cfg",
                    "experimentItemId": "i1",
                    "output": json.dumps({"chat_id": "chat-1"}),
                }
            ]
        elif path == "/api/public/v2/observations":
            assert params["sessionId"] == "chat-1"
            data = [
                {
                    "id": "o1",
                    "traceId": "app-a",
                    "startTime": "2026-10-20T09:00:02Z",
                    "projectId": "p",
                    "type": "SPAN",
                }
            ]
        elif path == "/api/public/traces" and request.method == "DELETE":
            return httpx.Response(200, json={"message": "Traces deleted"})
        else:
            raise AssertionError(f"unexpected request {request.method} {path}")
        return httpx.Response(200, json={"data": data, "meta": {"cursor": None}})

    client = Langfuse(
        public_key=f"pk-test-{uuid.uuid4()}",
        secret_key="sk-test",
        base_url="http://langfuse.invalid",
        httpx_client=httpx.Client(transport=httpx.MockTransport(handle)),
        span_exporter=InMemorySpanExporter(),
        tracer_provider=TracerProvider(),
    )
    runs = LangfuseRuns(client=client)

    found = runs.run_traces("e1")
    runs.delete_traces(found)

    assert found == RunTraces("cfg", 1, ["app-a", "item-1"])
    experiments, items, session, delete, delete_items = requests
    assert dict(experiments.url.params)["id"] == "e1"
    assert dict(experiments.url.params)["fromStartTime"].startswith(
        EPOCH.replace(tzinfo=None).isoformat()
    )
    assert dict(items.url.params)["experimentId"] == "e1"
    assert dict(items.url.params)["fromStartTime"].startswith("2026-10-20T09:00:00")
    assert "io" in dict(items.url.params)["fields"].split(",")
    assert dict(session.url.params)["sessionId"] == "chat-1"
    assert json.loads(delete.content) == {"traceIds": ["app-a"]}
    assert json.loads(delete_items.content) == {"traceIds": ["item-1"]}


def test_real_client_reads_the_session_of_an_item_without_output():
    """Only the HTTP layer is stubbed: the Langfuse client builds the request
    and parses the session id of the item trace's observations."""
    requests = []

    def observation(obs_id, trace_id, **extra):
        return {
            "id": obs_id,
            "traceId": trace_id,
            "startTime": "2026-10-20T09:00:02Z",
            "projectId": "p",
            "type": "SPAN",
            **extra,
        }

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path, params = request.url.path, dict(request.url.params)
        if path == "/api/public/experiments":
            data = [
                {
                    "id": "e1",
                    "name": "cfg",
                    "startTime": "2026-10-20T09:00:00Z",
                    "endTime": "2026-10-20T10:00:00Z",
                    "itemCount": 1,
                }
            ]
        elif path == "/api/public/experiment-items":
            data = [
                {
                    "id": "x1",
                    "traceId": "item-1",
                    "startTime": "2026-10-20T09:00:01Z",
                    "level": "DEFAULT",
                    "environment": "sdk-experiment",
                    "experimentId": "e1",
                    "experimentName": "cfg",
                    "experimentItemId": "i1",
                }
            ]
        elif path == "/api/public/v2/observations" and "traceId" in params:
            data = [
                observation("task", "item-1", sessionId="chat-1"),
                observation("chat", "item-1", sessionId="chat-1"),
            ]
        elif path == "/api/public/v2/observations":
            assert params["sessionId"] == "chat-1"
            data = [observation("o1", "app-a")]
        else:
            raise AssertionError(f"unexpected request {request.method} {path}")
        return httpx.Response(200, json={"data": data, "meta": {"cursor": None}})

    client = Langfuse(
        public_key=f"pk-test-{uuid.uuid4()}",
        secret_key="sk-test",
        base_url="http://langfuse.invalid",
        httpx_client=httpx.Client(transport=httpx.MockTransport(handle)),
        span_exporter=InMemorySpanExporter(),
        tracer_provider=TracerProvider(),
    )

    found = LangfuseRuns(client=client).run_traces("e1")

    assert found == RunTraces("cfg", 1, ["app-a", "item-1"])
    _, _, trace, session = requests
    assert dict(trace.url.params)["traceId"] == "item-1"
    assert "basic" in dict(trace.url.params)["fields"].split(",")
    assert dict(session.url.params)["sessionId"] == "chat-1"
