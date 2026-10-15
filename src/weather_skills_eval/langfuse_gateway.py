"""Everything the harness reads from or writes to Langfuse.

- The task suite: a Langfuse dataset.
- The run: one experiment on that dataset (``DatasetClient.run_experiment``),
  with one item per task and the play's scores on each item.
- The simulated user's and the judge's model calls: each call is its own
  trace, with user id ``simulated-user`` or ``judge``, session id = the chat
  id, in the harness's Langfuse environment.
- The app's traces: one per chat turn, session id = the chat id. They are
  read back through the observations API once they stop changing.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable

import litellm
from langfuse import Evaluation, Langfuse, propagate_attributes
from opentelemetry import context as otel_context
from opentelemetry.sdk.trace import TracerProvider

from .scoring import APP_TRACE_NAME

OBSERVATION_FIELDS = "core,basic,usage"
PAGE_SIZE = 1000


def _usage(response) -> dict[str, int]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    details = {
        "input": getattr(usage, "prompt_tokens", None),
        "output": getattr(usage, "completion_tokens", None),
    }
    return {k: int(v) for k, v in details.items() if v is not None}


def _cost(response) -> dict[str, float]:
    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:  # LiteLLM has no price for some models; the call still counts
        return {}
    return {"total": float(cost)} if cost else {}


def _normalize(observation) -> dict:
    level = observation.level
    return {
        "id": observation.id,
        "trace_id": observation.trace_id,
        "type": observation.type,
        "name": observation.name,
        "level": getattr(level, "value", level),
        "is_root": bool(observation.is_root_observation),
        "end_time": observation.end_time,
        "usage_details": observation.usage_details or {},
        "cost_details": observation.cost_details or {},
        "total_cost": observation.total_cost,
    }


def play_scores(*, output, **kwargs) -> list[Evaluation]:
    """Item evaluator: the scores the play already computed, as Langfuse scores."""
    return [
        Evaluation(name=s["name"], value=s["value"], comment=s.get("comment"))
        for s in (output or {}).get("scores", [])
    ]


class LangfuseGateway:
    def __init__(
        self,
        *,
        environment: str,
        trace_timeout: float,
        trace_settle: float,
        poll_interval: float,
        client: Langfuse | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        # Its own tracer provider keeps spans of other OpenTelemetry users in
        # the process (LangWatch, used by Scenario) out of Langfuse.
        self.client = client or Langfuse(environment=environment, tracer_provider=TracerProvider())
        self._environment = environment
        self._trace_timeout = trace_timeout
        self._trace_settle = trace_settle
        self._poll_interval = poll_interval
        self._clock = clock
        self._sleep = sleep

    def dataset(self, name: str):
        return self.client.get_dataset(name)

    def run_experiment(self, *, dataset, name: str, task, metadata: dict, concurrency: int):
        return dataset.run_experiment(
            name=name,
            task=task,
            evaluators=[play_scores],
            max_concurrency=concurrency,
            metadata=metadata,
        )

    def link_item_to_chat(self, chat_id: str) -> None:
        """Give the current experiment item's trace the chat id as its session."""
        with propagate_attributes(session_id=chat_id):
            pass

    def traced_completion(self, role: str, session_id: str | None, **kwargs):
        # A fresh context makes the call a trace of its own instead of a child
        # of the experiment item, so it carries its own user id.
        token = otel_context.attach(otel_context.Context())
        try:
            with propagate_attributes(
                user_id=role,
                session_id=session_id,
                environment=self._environment,
                trace_name=role,
            ):
                with self.client.start_as_current_observation(
                    name=role,
                    as_type="generation",
                    model=kwargs.get("model"),
                    input=kwargs.get("messages"),
                ) as generation:
                    response = litellm.completion(**kwargs)
                    message = response.choices[0].message
                    generation.update(
                        output=message.model_dump() if hasattr(message, "model_dump") else message,
                        usage_details=_usage(response),
                        cost_details=_cost(response),
                    )
                    return response
        finally:
            otel_context.detach(token)

    def _pages(self, **filters) -> list:
        observations, cursor = [], None
        while True:
            page = self.client.api.observations.get_many(
                fields=OBSERVATION_FIELDS, limit=PAGE_SIZE, cursor=cursor, **filters
            )
            observations.extend(page.data)
            cursor = page.meta.cursor
            if not cursor:
                return observations

    def _app_observations(self, chat_id: str) -> list[dict]:
        roots = self._pages(session_id=chat_id, name=APP_TRACE_NAME)
        trace_ids = sorted({o.trace_id for o in roots if o.trace_id})
        observations = []
        for trace_id in trace_ids:
            observations.extend(_normalize(o) for o in self._pages(trace_id=trace_id))
        return observations

    async def chat_observations(self, chat_id: str, turns: int) -> tuple[list[dict], bool]:
        """The app's observations for a chat once they stop changing.

        Waits until there is one app trace per turn and the set of
        observations has not changed for the settle time. Returns what it has,
        with ``False``, when the timeout passes first.
        """
        deadline = self._clock() + self._trace_timeout
        last_signature = None
        stable_since = self._clock()
        while True:
            observations = await asyncio.to_thread(self._app_observations, chat_id)
            traces = {o["trace_id"] for o in observations}
            signature = tuple(sorted((o["id"], o["end_time"] is not None) for o in observations))
            if signature != last_signature:
                last_signature = signature
                stable_since = self._clock()
            elif len(traces) >= turns and self._clock() - stable_since >= self._trace_settle:
                return observations, True
            if self._clock() >= deadline:
                return observations, False
            await self._sleep(self._poll_interval)

    def flush(self) -> None:
        self.client.flush()
