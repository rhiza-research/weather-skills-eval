import asyncio
import json
import uuid
from types import SimpleNamespace

import litellm
from fakes import FakeClock
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from weather_skills_eval.langfuse_gateway import LangfuseGateway, TraceReadError, play_scores


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
    settings = {"trace_timeout": 2, "trace_settle": 1, "poll_interval": 1}
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


def _obs(obs_id, trace_id, kind="GENERATION", name="llm", ended=True, metadata=None):
    return SimpleNamespace(
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
    """A chat session holding app traces t1 and t2 (t2 appears on the third
    read) plus the simulated user's and judge's traces, which share the
    session id but must not be counted."""

    def __init__(self):
        self.reads = 0
        self.requests: list[dict] = []

    def _session(self):
        observations = [
            _obs("r1", "t1", "SPAN", "weather-skills-chat"),
            _obs("su", "user-trace", "GENERATION", "simulated-user"),
            _obs("jg", "judge-trace", "GENERATION", "judge"),
        ]
        if self.reads >= 3:
            observations.append(_obs("r2", "t2", "SPAN", "weather-skills-chat"))
        return observations

    def get_many(self, *, fields, limit, cursor, **filters):
        self.requests.append({"fields": fields, **filters})
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
            data=[_obs("r2", "t2", "SPAN", "weather-skills-chat"), _obs("g2", "t2")],
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

    observations, complete = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert complete
    assert sorted(o["id"] for o in observations) == ["g1", "g2", "r1", "r2"]
    assert {o["level"] for o in observations} == {"DEFAULT"}
    # Flat dotted metadata keys pass through intact.
    assert next(o for o in observations if o["id"] == "g1")["metadata"] == {
        "attributes.gen_ai.usage.input_tokens.cached": "100520"
    }
    assert next(o for o in observations if o["id"] == "g2")["model"] == "model-g2"
    session_requests = [r for r in api.requests if "session_id" in r]
    assert all(r["name"] == "weather-skills-chat" for r in session_requests)
    assert all({"metadata", "model"} <= set(r["fields"].split(",")) for r in api.requests)
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

    observations, complete = asyncio.run(gateway.chat_observations("chat-1", 5))

    assert not complete
    assert observations


def test_chat_observations_raise_trace_read_error_on_langfuse_failure():
    class Failing:
        def get_many(self, **kwargs):
            raise RuntimeError("503 from Langfuse")

    gateway = gateway_for(fake_client(Failing()))

    try:
        asyncio.run(gateway.chat_observations("chat-1", 1))
    except TraceReadError as exc:
        assert "503" in str(exc)
    else:
        raise AssertionError("expected TraceReadError")


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
