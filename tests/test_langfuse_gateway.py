import asyncio
import uuid
from types import SimpleNamespace

import litellm
from fakes import FakeClock
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from weather_skills_eval.langfuse_gateway import LangfuseGateway, play_scores


def test_model_calls_are_root_traces_with_role_user_id_session_and_environment(monkeypatch):
    exporter = InMemorySpanExporter()
    client = Langfuse(
        public_key=f"pk-test-{uuid.uuid4()}",
        secret_key="sk-test",
        base_url="http://langfuse.invalid",
        span_exporter=exporter,
        tracer_provider=TracerProvider(),
        environment="eval",
    )
    gateway = LangfuseGateway(
        environment="eval", trace_timeout=1, trace_settle=1, poll_interval=1, client=client
    )
    reply = litellm.ModelResponse(
        choices=[{"message": {"role": "assistant", "content": '{"ok": true}'}}],
        usage={"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
    )
    monkeypatch.setattr(litellm, "completion", lambda **kwargs: reply)

    with client.start_as_current_observation(name="experiment-item-task"):
        response = gateway.traced_completion(
            "judge",
            "chat-9",
            model="openrouter/test/judge",
            messages=[{"role": "user", "content": "x"}],
        )
    client.flush()

    assert response is reply
    spans = {span.name: span for span in exporter.get_finished_spans()}
    judge = spans["judge"]
    assert judge.parent is None
    assert judge.context.trace_id != spans["experiment-item-task"].context.trace_id
    assert judge.attributes["user.id"] == "judge"
    assert judge.attributes["session.id"] == "chat-9"
    assert judge.attributes["langfuse.environment"] == "eval"
    assert judge.attributes["langfuse.observation.type"] == "generation"
    assert '"input": 12' in judge.attributes["langfuse.observation.usage_details"]


def _obs(obs_id, trace_id, kind="GENERATION", name="llm", ended=True):
    return SimpleNamespace(
        id=obs_id,
        trace_id=trace_id,
        type=kind,
        name=name,
        level=SimpleNamespace(value="DEFAULT"),
        is_root_observation=name == "weather-skills-chat",
        end_time="t" if ended else None,
        usage_details={"input": 1, "output": 1},
        cost_details={"total": 0.1},
        total_cost=0.1,
    )


class FakeObservationsApi:
    """Traces arrive over time: the second trace appears on the third read."""

    def __init__(self):
        self.reads = 0

    def get_many(self, *, fields, limit, cursor, **filters):
        if "session_id" in filters:
            self.reads += 1
            roots = [_obs("r1", "t1", "SPAN", "weather-skills-chat")]
            if self.reads >= 3:
                roots.append(_obs("r2", "t2", "SPAN", "weather-skills-chat"))
            return SimpleNamespace(data=roots, meta=SimpleNamespace(cursor=None))
        trace = filters["trace_id"]
        # Two pages for t1, to follow the cursor.
        if trace == "t1" and cursor is None:
            return SimpleNamespace(
                data=[_obs("r1", "t1", "SPAN", "weather-skills-chat")],
                meta=SimpleNamespace(cursor="next"),
            )
        if trace == "t1":
            return SimpleNamespace(data=[_obs("g1", "t1")], meta=SimpleNamespace(cursor=None))
        return SimpleNamespace(
            data=[_obs("r2", "t2", "SPAN", "weather-skills-chat"), _obs("g2", "t2")],
            meta=SimpleNamespace(cursor=None),
        )


def test_chat_observations_wait_for_every_turn_and_for_the_set_to_settle():
    clock = FakeClock()
    api = FakeObservationsApi()
    gateway = LangfuseGateway(
        environment="eval",
        trace_timeout=600,
        trace_settle=10,
        poll_interval=5,
        client=SimpleNamespace(api=SimpleNamespace(observations=api)),
        clock=clock,
        sleep=clock.sleep,
    )

    observations, complete = asyncio.run(gateway.chat_observations("chat-1", 2))

    assert complete
    assert sorted(o["id"] for o in observations) == ["g1", "g2", "r1", "r2"]
    assert {o["level"] for o in observations} == {"DEFAULT"}
    assert api.reads >= 5


def test_chat_observations_give_up_after_the_timeout():
    clock = FakeClock()
    gateway = LangfuseGateway(
        environment="eval",
        trace_timeout=30,
        trace_settle=10,
        poll_interval=5,
        client=SimpleNamespace(api=SimpleNamespace(observations=FakeObservationsApi())),
        clock=clock,
        sleep=clock.sleep,
    )

    observations, complete = asyncio.run(gateway.chat_observations("chat-1", 5))

    assert not complete
    assert observations


def test_play_scores_become_evaluations():
    evaluations = play_scores(
        output={"scores": [{"name": "turns_to_goal", "value": 2, "comment": "c"}]}
    )
    assert [(e.name, e.value, e.comment) for e in evaluations] == [("turns_to_goal", 2, "c")]
