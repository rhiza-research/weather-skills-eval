"""Everything the harness reads from or writes to Langfuse.

The app and the harness must use the same Langfuse project: the harness
reads the app's traces from the project its own keys belong to.

- The task suite: a Langfuse dataset.
- The run: one experiment on that dataset (``DatasetClient.run_experiment``),
  with one item per task and the play's scores on each item. Each item's
  trace carries the chat id as its session id, and an ``app-cost-rollup``
  generation whose cost is the play's ``cost_to_goal_usd``, so the
  experiment's per-item cost shows the app's cost.
- The simulated user's and the judge's model calls: each call is its own
  trace, with user id ``simulated-user`` or ``judge``, session id = the chat
  id, in the harness's Langfuse environment.
- The app's traces: one per chat turn, named ``weather-skills-chat``, with
  session id = the chat id. They are read back through the observations API
  once they stop changing.
- Deleting a run (``LangfuseRuns``): Langfuse has no call that deletes an
  experiment; it goes away once its traces are deleted. A run's traces are
  each item's trace plus every trace in the session of the item output's
  ``chat_id``, or, for an item whose output has none, in the sessions of its
  trace's observations: the item trace carries the chat when the play
  returned and the runner check after it failed.
"""

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import httpx
import litellm
from langfuse import Evaluation, Langfuse, propagate_attributes
from langfuse.api.core.api_error import ApiError
from opentelemetry import context as otel_context
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.trace import TracerProvider

from .llm import ModelReplyError
from .scoring import APP_TRACE_NAME, TraceRead

OBSERVATION_FIELDS = "core,basic,model,usage,metadata"
TOOL_OUTPUT_FIELDS = "core,io"
PAGE_SIZE = 1000
# Tool outputs can be large.
TOOL_OUTPUT_PAGE_SIZE = 100
SESSION_ATTRIBUTE = "session.id"
APP_COST_ROLLUP = "app-cost-rollup"
EXPERIMENT_ITEM_FIELDS = "core,io"
# The experiments API's maximum page size.
EXPERIMENT_PAGE_SIZE = 100
# Langfuse takes at most 1,000 trace ids per delete request:
# https://langfuse.com/faq/all/deprecated-api-migration.md
DELETE_BATCH_SIZE = 1000
# The experiments API needs a lower bound on start time; the epoch finds an
# experiment of any age.
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class TraceReadError(Exception):
    """The app's traces could not be read from Langfuse."""


class ExperimentNotFound(Exception):
    """No experiment has the given id."""


@dataclass
class RunTraces:
    experiment_name: str
    items: int
    trace_ids: list[str]
    # The items' own traces, also in ``trace_ids``.
    item_trace_ids: list[str] = field(default_factory=list, compare=False)

    def deletion_order(self) -> list[str]:
        """``trace_ids`` with the item traces last: while one is left, the
        experiment is still listed, so a delete stopped part way can be run
        again."""
        items = set(self.item_trace_ids)
        return [t for t in self.trace_ids if t not in items] + sorted(items)


def _all_pages(read: Callable[[str | None], object]) -> list:
    """Every page's data; ``read`` takes the cursor. Stops on an empty or
    repeated cursor."""
    data, cursor, seen = [], None, set()
    while True:
        page = read(cursor)
        data.extend(page.data)
        cursor = page.meta.cursor
        if not cursor or cursor in seen:
            return data
        seen.add(cursor)


def _chat_id(output) -> str | None:
    """The ``chat_id`` of an experiment item's output; None when there is none."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            return None
    chat_id = output.get("chat_id") if isinstance(output, dict) else None
    return chat_id if isinstance(chat_id, str) and chat_id else None


def _transient(exc: Exception) -> bool:
    """A read error that a later read may not hit: a transport error or
    timeout, or a 429 or 5xx response. A 4xx such as 400, 401, 403 or 404 is
    not, nor a Langfuse error without a status code."""
    if isinstance(exc, httpx.TransportError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
    elif isinstance(exc, ApiError):
        status = exc.status_code
    else:
        return False
    return status is not None and (status == 429 or status >= 500)


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


def _ok(output):
    """A tool output's ``"ok"`` value; None when the output is not a JSON object."""
    if isinstance(output, str):
        try:
            output = json.loads(output)
        except ValueError:
            return None
    return output.get("ok") if isinstance(output, dict) else None


def _normalize(observation, tool_outputs: dict[str, object]) -> dict:
    level = observation.level
    normalized = {
        "id": observation.id,
        "trace_id": observation.trace_id,
        "type": observation.type,
        "name": observation.name,
        "level": getattr(level, "value", level),
        "is_root": bool(observation.is_root_observation),
        "end_time": observation.end_time,
        "model": observation.model,
        "usage_details": observation.usage_details or {},
        "cost_details": observation.cost_details or {},
        "total_cost": observation.total_cost,
        "metadata": observation.metadata or {},
    }
    # Absent until a read has returned the tool's output.
    if observation.type == "TOOL" and observation.id in tool_outputs:
        normalized["ok"] = _ok(tool_outputs[observation.id])
    return normalized


def _shape(observations: list[dict]) -> tuple:
    """The observations and whether each has ended, without tool outputs."""
    return tuple(sorted((o["id"], o["end_time"] is not None) for o in observations))


def _unread(observations: list[dict]) -> int:
    """Tool observations whose output has not been read."""
    return sum(1 for o in observations if o["type"] == "TOOL" and "ok" not in o)


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
        model_max_tokens: int,
        model_retries: int = 2,
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
        self._model_retries = model_retries
        self._model_max_tokens = model_max_tokens
        self._clock = clock
        self._sleep = sleep

    def dataset(self, name: str):
        return self.client.get_dataset(name)

    def run_experiment(
        self, *, dataset, name: str, description: str, task, metadata: dict, concurrency: int
    ):
        return dataset.run_experiment(
            name=name,
            description=description,
            task=task,
            evaluators=[play_scores],
            max_concurrency=concurrency,
            metadata=metadata,
        )

    def link_item_to_chat(self, chat_id: str) -> None:
        """Give the current experiment item's trace the chat id as its session.

        Called inside the experiment task. The task span gets the session
        attribute directly, and a ``chat`` child span is created under it
        with the session propagated, so the trace carries it either way.
        """
        otel_trace.get_current_span().set_attribute(SESSION_ATTRIBUTE, chat_id)
        with propagate_attributes(session_id=chat_id):
            with self.client.start_as_current_observation(name="chat", input={"chat_id": chat_id}):
                pass

    def record_app_cost(self, cost_usd: float, app_trace_ids: list[str]) -> None:
        """Record the app's cost on the current experiment item's trace.

        Called inside the experiment task. Langfuse sets cost only on
        generations, and the experiment's per-item cost sums the item
        trace's generations. The rollup has no model and no usage, so Langfuse
        infers no price and counts no tokens for it.
        """
        with self.client.start_as_current_observation(
            name=APP_COST_ROLLUP,
            as_type="generation",
            input={"app_trace_ids": app_trace_ids},
            cost_details={"total": cost_usd},
        ):
            pass

    def traced_completion(self, role: str, session_id: str | None, **kwargs):
        # A fresh context makes the call a trace of its own instead of a child
        # of the experiment item, so it carries its own user id.
        kwargs.setdefault("num_retries", self._model_retries)
        # Scenario's judge sends max_tokens=None.
        if kwargs.get("max_completion_tokens") is not None:
            if kwargs.get("max_tokens") is None:
                kwargs.pop("max_tokens", None)
        elif kwargs.get("max_tokens") is None:
            kwargs["max_tokens"] = self._model_max_tokens
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
                    choice = response.choices[0]
                    message = choice.message
                    generation.update(
                        output=message.model_dump() if hasattr(message, "model_dump") else message,
                        usage_details=_usage(response),
                        cost_details=_cost(response),
                    )
                    if getattr(choice, "finish_reason", None) == "length":
                        limit = "max_tokens" if "max_tokens" in kwargs else "max_completion_tokens"
                        raise ModelReplyError(
                            f"The {role} reply was cut off at {limit} {kwargs[limit]} "
                            "(EVAL_MODEL_MAX_TOKENS unless the call set its own)"
                        )
                    return response
        finally:
            otel_context.detach(token)

    def _pages(
        self, fields: str = OBSERVATION_FIELDS, page_size: int = PAGE_SIZE, **filters
    ) -> list:
        return _all_pages(
            lambda cursor: self.client.api.observations.get_many(
                fields=fields,
                limit=page_size,
                cursor=cursor,
                **filters,
            )
        )

    def _app_observations(
        self, chat_id: str, tool_outputs: dict[str, object], force: set[str] | None = None
    ) -> list[dict]:
        """``tool_outputs`` keeps the outputs read so far, by observation id,
        across polls; a trace's tool outputs are read only while one is missing.
        A tool in ``force`` has its output read whether or not it has ended."""
        # Only the app's traces: the session also holds the simulated user's
        # and the judge's traces, and the experiment item's.
        roots = self._pages(session_id=chat_id, name=APP_TRACE_NAME)
        trace_ids = sorted({o.trace_id for o in roots if o.trace_id and o.name == APP_TRACE_NAME})
        observations = []
        for trace_id in trace_ids:
            found = self._pages(trace_id=trace_id)
            # After the full read, so every tool it found can have its output.
            # Inputs and outputs are read for tool observations only.
            if any(o.type == "TOOL" and o.id not in tool_outputs for o in found):
                tools = self._pages(
                    fields=TOOL_OUTPUT_FIELDS,
                    page_size=TOOL_OUTPUT_PAGE_SIZE,
                    trace_id=trace_id,
                    type="TOOL",
                )
                # A tool's output is read once the tool has ended, null or not;
                # a running tool may still write it.
                tool_outputs.update(
                    {
                        o.id: o.output
                        for o in tools
                        if o.end_time is not None or o.id in (force or set())
                    }
                )
            observations.extend(_normalize(o, tool_outputs) for o in found)
        return observations

    async def chat_observations(self, chat_id: str, turns: int) -> TraceRead:
        """The app's observations for a chat once they stop changing.

        ``turns`` is the number of turns the app accepted. Waits until there
        is one app trace per such turn, every tool's output has been read, and
        the observations have not changed for the settle time; the read is not
        complete when the timeout passes first. When the traces have otherwise
        settled and the only unread tools have no end time (a span a turn
        timeout left open), their outputs are read once as they are. That
        read is final when it finds the same traces and observations, complete
        when it returned every one of them; otherwise polling goes on and the
        outputs it read for tools still open are dropped. A read failing with a
        transient error (see ``_transient``) is retried at the poll interval
        until the timeout; it does not restart the settle time, which only a
        change in the observations does.
        ``TraceReadError`` is raised when no read succeeded by the timeout,
        and at once for any other error before a read succeeded; after one,
        any other error returns the last successful read as partial.
        """
        deadline = self._clock() + self._trace_timeout
        last_signature = None
        stable_since = self._clock()
        observations: list[dict] | None = None
        tool_outputs: dict[str, object] = {}
        force: set[str] = set()
        settled_shape: tuple | None = None

        def arriving(observations: list[dict]) -> bool:
            traces = {o["trace_id"] for o in observations}
            return len(traces) < turns or self._clock() - stable_since < self._trace_settle

        while True:
            try:
                observations = await asyncio.to_thread(
                    self._app_observations, chat_id, tool_outputs, force
                )
                read_error = None
            except Exception as exc:
                read_error = f"{type(exc).__name__}: {exc}"
                if not _transient(exc):
                    if observations is None:
                        raise TraceReadError(read_error) from exc
                    # After a read succeeded, that read is kept as partial.
                    return TraceRead(
                        observations,
                        False,
                        read_error=read_error,
                        unread_tools=_unread(observations),
                        traces_arriving=arriving(observations),
                    )
                if self._clock() >= deadline:
                    if observations is None:
                        raise TraceReadError(read_error) from exc
                    return TraceRead(
                        observations,
                        False,
                        read_error=read_error,
                        unread_tools=_unread(observations),
                        traces_arriving=arriving(observations),
                    )
                await self._sleep(self._poll_interval)
                continue
            traces = {o["trace_id"] for o in observations}
            shape = (len(traces), _shape(observations))
            if force and shape == settled_shape:
                unread_tools = _unread(observations)
                return TraceRead(
                    observations, not unread_tools, unread_tools=unread_tools, forced=True
                )
            if force:
                # A forced read that found new observations polls on as usual,
                # without the outputs it read for tools still open.
                still_open = {
                    o["id"] for o in observations if o["id"] in force and o["end_time"] is None
                }
                for observation in observations:
                    if observation["id"] in still_open:
                        tool_outputs.pop(observation["id"], None)
                        observation.pop("ok", None)
                force = set()
            signature = tuple(
                sorted(
                    (o["id"], o["end_time"] is not None, o.get("ok", "unread"))
                    for o in observations
                )
            )
            if signature != last_signature:
                last_signature = signature
                stable_since = self._clock()
            elif len(traces) >= turns and self._clock() - stable_since >= self._trace_settle:
                unread = [o for o in observations if o["type"] == "TOOL" and "ok" not in o]
                if not unread:
                    return TraceRead(observations, True)
                if all(o["end_time"] is None for o in unread):
                    force = {o["id"] for o in unread}
                    settled_shape = shape
                    continue
            if self._clock() >= deadline:
                return TraceRead(
                    observations,
                    False,
                    unread_tools=_unread(observations),
                    traces_arriving=arriving(observations),
                )
            await self._sleep(self._poll_interval)

    def flush(self) -> None:
        self.client.flush()


class LangfuseRuns:
    """Finds a run's traces and deletes them. Needs only the Langfuse
    settings the SDK reads, not the run's."""

    def __init__(self, client: Langfuse | None = None):
        self.client = client or Langfuse(tracer_provider=TracerProvider())

    def run_traces(self, experiment_id: str) -> RunTraces:
        """The experiment's name, item count and trace ids.

        Raises ``ExperimentNotFound`` when no experiment has the id.
        """
        api = self.client.api
        found = api.experiments.list(from_start_time=EPOCH, id=experiment_id)
        experiment = next((e for e in found.data if e.id == experiment_id), None)
        if experiment is None:
            raise ExperimentNotFound(experiment_id)
        items = _all_pages(
            lambda cursor: api.experiments.list_items(
                from_start_time=experiment.start_time,
                experiment_id=experiment_id,
                fields=EXPERIMENT_ITEM_FIELDS,
                limit=EXPERIMENT_PAGE_SIZE,
                cursor=cursor,
            )
        )
        item_trace_ids = {item.trace_id for item in items if item.trace_id}
        trace_ids = set(item_trace_ids)
        chats = {c for c in (_chat_id(item.output) for item in items) if c}
        for item in items:
            if not item.trace_id or _chat_id(item.output):
                continue
            # No chat id in the output: when the play returned and the runner
            # check after it failed, the item trace carries the chat as its
            # session id. A play that raised linked nothing.
            observations = _all_pages(
                lambda cursor, trace_id=item.trace_id: api.observations.get_many(
                    fields="core,basic",
                    limit=PAGE_SIZE,
                    cursor=cursor,
                    trace_id=trace_id,
                )
            )
            chats.update(o.session_id for o in observations if o.session_id)
        for chat_id in sorted(chats):
            # The session holds the app's, the simulated user's, the judge's
            # and the item's traces.
            session = _all_pages(
                lambda cursor, chat_id=chat_id: api.observations.get_many(
                    fields="core",
                    limit=PAGE_SIZE,
                    cursor=cursor,
                    session_id=chat_id,
                )
            )
            trace_ids.update(o.trace_id for o in session if o.trace_id)
        return RunTraces(experiment.name, len(items), sorted(trace_ids), sorted(item_trace_ids))

    def delete_traces(self, run: RunTraces) -> None:
        """Sends the run's ids in ``deletion_order``, ``DELETE_BATCH_SIZE`` per
        request, the other traces' batches first and then the item traces'
        batches; no request holds both. An error stops the delete; batches
        already sent stay deleted, and the failing one may be partly deleted.
        A partly deleted batch of other traces deletes no item trace, so
        after any interruption the run is still found and a new delete
        finishes it."""
        items = set(run.item_trace_ids)
        order = run.deletion_order()
        for group in ([t for t in order if t not in items], [t for t in order if t in items]):
            for start in range(0, len(group), DELETE_BATCH_SIZE):
                batch = group[start : start + DELETE_BATCH_SIZE]
                self.client.api.trace.delete_multiple(trace_ids=batch)
