import asyncio
import json
import uuid
from types import SimpleNamespace

import httpx
import litellm
import pytest
from fakes import FakeClock
from langfuse import Langfuse
from langfuse.api.core.api_error import ApiError as LangfuseApiError
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from weather_skills_eval import runner
from weather_skills_eval.langfuse_gateway import (
    TOOL_OUTPUT_PAGE_SIZE,
    LangfuseGateway,
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
    settings = {"trace_timeout": 2, "trace_settle": 1, "poll_interval": 1, "model_max_tokens": 8192}
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
    assert read.partial_reason() == "traces were still arriving"


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

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    # Complete well before the timeout, with the open tool's output as it was.
    assert read.complete
    assert read.unread_tools == 0
    assert {o["id"]: o.get("ok", "unread") for o in read.observations}["k1"] is None
    assert clock.now - 1000.0 < 600


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
    assert (
        TraceRead([], False, unread_tools=2, traces_arriving=True).partial_reason()
        == "traces were still arriving"
    )
    assert (
        TraceRead([], False, unread_tools=2).partial_reason()
        == "tool outputs not read by the timeout: 2"
    )


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

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert not read.complete
    assert read.unread_tools == 1
    assert read.partial_reason() == "the outputs of tools left open could not be read: 1"
    assert clock.now - 1000.0 < 600


def test_timeout_with_unread_tool_outputs_says_how_many():
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
    assert read.partial_reason() == "tool outputs not read by the timeout: 1"


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

    assert str(error.value) == f"ReadTimeout: timed out, read {api.calls}"
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

    with pytest.raises(TraceReadError, match="TypeError: unexpected keyword"):
        asyncio.run(gateway.chat_observations("chat-1", 1))
    assert api.calls == 1
    assert clock.sleeps == []


class FailingFirst(FakeObservationsApi):
    """Fails the first ``failures`` reads that ``fails`` picks, then answers as the parent."""

    def __init__(self, failures: int, fails=lambda filters: "session_id" in filters):
        super().__init__()
        self.failures = failures
        self.fails = fails
        self.status = 504

    def get_many(self, **kwargs):
        if self.fails(kwargs) and self.failures:
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


@pytest.mark.parametrize("status", [400, 401, 403, 404, None])
def test_a_permanent_langfuse_error_is_raised_at_once(status):
    clock = FakeClock()
    api = FailingFirst(failures=1)
    api.status = status
    gateway = gateway_for(
        fake_client(api), trace_timeout=600, poll_interval=5, clock=clock, sleep=clock.sleep
    )

    with pytest.raises(TraceReadError, match=f"status_code: {status}"):
        asyncio.run(gateway.chat_observations("chat-1", 2))
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

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert not read.complete
    assert "status_code: 403" in read.read_error
    assert sorted(o["id"] for o in read.observations) == ["g1", "r1"]
    # Returned at once, without waiting for the timeout.
    assert clock.now - 1000.0 == 5


def test_a_tool_output_read_that_fails_mid_poll_is_retried():
    clock = FakeClock()
    api = FailingFirst(
        failures=1,
        fails=lambda filters: filters.get("trace_id") == "t2" and filters.get("type") == "TOOL",
    )
    gateway = gateway_for(
        fake_client(api),
        trace_timeout=600,
        trace_settle=10,
        poll_interval=5,
        clock=clock,
        sleep=clock.sleep,
    )

    read = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert api.failures == 0
    assert read.complete
    assert read.read_error is None
    assert {o["id"]: o.get("ok") for o in read.observations}["k1"] is False


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

    read = asyncio.run(gateway.chat_observations("chat-1", 1))

    assert read.complete
    # Read 1 at 0 s, failures at 5, 10 and 15 s, and the read at 20 s settles it.
    assert clock.now - 1000.0 == 20


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
    assert read.read_error == "ReadTimeout: Read timed out"
    assert read.partial_reason() == "Langfuse reads failed: ReadTimeout: Read timed out"
    assert sorted(o["id"] for o in read.observations) == ["g1", "r1"]


def test_a_reply_cut_off_at_max_tokens_is_named(monkeypatch):
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

    with pytest.raises(ModelReplyError, match="cut off at max_tokens 50 .EVAL_MODEL_MAX_TOKENS"):
        gateway.traced_completion("simulated-user", "chat-1", model="m", messages=[])
    with pytest.raises(ModelReplyError, match="cut off at max_completion_tokens 20 "):
        gateway.traced_completion(
            "judge", "chat-1", model="m", messages=[], max_completion_tokens=20
        )


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


def test_default_settings_cap_every_model_call_at_8192_output_tokens(monkeypatch):
    settings = load_settings(
        {
            "LANGFUSE_PUBLIC_KEY": "pk",
            "LANGFUSE_SECRET_KEY": "sk",
            "OPENROUTER_API_KEY": "or",
            "LANGFUSE_TRACING_ENVIRONMENT": "eval",
            "EVAL_SIMULATED_USER_MODEL": "u",
            "EVAL_JUDGE_MODEL": "j",
            "EVAL_DOCKER_NETWORK": "n",
        }
    )
    built = {}
    monkeypatch.setattr(runner, "LangfuseGateway", lambda **kwargs: built.update(kwargs))
    runner._default_gateway(settings)
    assert built["model_max_tokens"] == 8192

    gateway = gateway_for(real_client(InMemorySpanExporter()), model_max_tokens=8192)
    assert _completion_kwargs(monkeypatch, gateway)["max_tokens"] == 8192
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
    assert tool_reads[0]["fields"] == "core,io"
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
