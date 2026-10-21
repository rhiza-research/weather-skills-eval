"""Scores of one play, from the play itself and from the app's Langfuse traces.

Token counts come from each generation. In both shapes a generation's
``input`` is the whole prompt, cached tokens included:

- Generations the app records carry the cache counts in ``usage_details``
  as ``cache_read_input_tokens`` and ``cache_creation_input_tokens``.
- Generations OpenRouter Broadcast records carry only ``input``, ``output``
  and ``total`` in ``usage_details``; their cache counts are top-level
  observation metadata keys whose names contain dots:
  ``attributes.gen_ai.usage.input_tokens.cached`` and
  ``attributes.gen_ai.usage.input_tokens.cache_write``.

The cache read and cache write counts are moved out of ``input``, so
``uncached input`` holds only tokens billed at the full input price. A
cache-miss call is a model call that wrote at least ``CACHE_MISS_TOKENS``
tokens of context it could not read from the cache (uncached input plus
cache write).
"""

from dataclasses import dataclass, field

CACHE_MISS_TOKENS = 50_000

APP_CACHE_READ = "cache_read_input_tokens"
APP_CACHE_WRITE = "cache_creation_input_tokens"
BROADCAST_CACHE_READ = "attributes.gen_ai.usage.input_tokens.cached"
BROADCAST_CACHE_WRITE = "attributes.gen_ai.usage.input_tokens.cache_write"

APP_TRACE_NAME = "weather-skills-chat"
COST_SCORE = "cost_to_goal_usd"


@dataclass
class TraceMetrics:
    model_calls: int = 0
    cost_usd: float = 0.0
    calls_without_cost: int = 0
    tokens_uncached_input: int = 0
    tokens_cache_read: int = 0
    tokens_cache_write: int = 0
    tokens_output: int = 0
    tool_calls: int = 0
    tool_failures: int = 0
    cache_miss_calls: int = 0
    trace_ids: list[str] = field(default_factory=list)
    # The distinct ``model`` names of the generations.
    provider_models: list[str] = field(default_factory=list)


def _count(value) -> int:
    if value is None or value == "":
        return 0
    return int(float(value))


def split_usage(usage: dict, metadata: dict | None = None) -> tuple[int, int, int, int]:
    """(uncached input, cache read, cache write, output) of one generation."""
    usage = usage or {}
    metadata = metadata or {}
    if usage.get(APP_CACHE_READ) is not None or usage.get(APP_CACHE_WRITE) is not None:
        read, write = _count(usage.get(APP_CACHE_READ)), _count(usage.get(APP_CACHE_WRITE))
    else:
        read = _count(metadata.get(BROADCAST_CACHE_READ))
        write = _count(metadata.get(BROADCAST_CACHE_WRITE))
    uncached = max(_count(usage.get("input")) - read - write, 0)
    return uncached, read, write, _count(usage.get("output"))


def _cost(obs: dict) -> float | None:
    cost = (obs.get("cost_details") or {}).get("total")
    if cost is None:
        cost = obs.get("total_cost")
    return None if cost is None else float(cost)


def trace_metrics(observations: list[dict]) -> TraceMetrics:
    """Observations are dicts with type, name, level, model, usage_details,
    cost_details, total_cost and metadata; a TOOL observation also has ``ok``."""
    metrics = TraceMetrics()
    metrics.trace_ids = sorted({o["trace_id"] for o in observations if o.get("trace_id")})
    metrics.provider_models = sorted(
        {o["model"] for o in observations if o.get("type") == "GENERATION" and o.get("model")}
    )
    for obs in observations:
        kind, level = obs.get("type"), obs.get("level")
        if kind == "GENERATION":
            metrics.model_calls += 1
            uncached, read, write, output = split_usage(
                obs.get("usage_details") or {}, obs.get("metadata")
            )
            metrics.tokens_uncached_input += uncached
            metrics.tokens_cache_read += read
            metrics.tokens_cache_write += write
            metrics.tokens_output += output
            if uncached + write >= CACHE_MISS_TOKENS:
                metrics.cache_miss_calls += 1
            cost = _cost(obs)
            if cost is None:
                metrics.calls_without_cost += 1
            else:
                metrics.cost_usd += cost
        elif kind == "SPAN" and obs.get("name") == "llm" and level == "ERROR":
            # A model call the app could not send or the provider refused.
            metrics.model_calls += 1
        elif kind == "TOOL":
            metrics.tool_calls += 1
            # The app records a failed skill run at DEFAULT, with "ok": false in its output.
            if level == "ERROR" or obs.get("ok") is False:
                metrics.tool_failures += 1
    return metrics


@dataclass
class TraceRead:
    """The app's observations of a chat as read from Langfuse. When not
    ``complete``, ``read_error`` is the last read's error when it failed, and
    ``unread_tools`` the tool observations whose output was not read."""

    observations: list[dict]
    complete: bool
    read_error: str | None = None
    unread_tools: int = 0
    # At the timeout: traces were missing or still changing.
    traces_arriving: bool = False
    # Returned by the read of tools left open, before the timeout.
    forced: bool = False

    def partial_reason(self) -> str:
        if self.read_error:
            return f"Langfuse reads failed: {self.read_error}"
        if self.forced:
            return f"the outputs of tools left open could not be read: {self.unread_tools}"
        if self.traces_arriving or not self.unread_tools:
            return "traces were still arriving"
        return f"tool outputs not read by the timeout: {self.unread_tools}"


@dataclass
class Score:
    name: str
    value: float
    comment: str | None = None


@dataclass
class PlayFacts:
    goal_reached: bool
    outcome: str
    turns: int
    clarifying_questions: int
    hints_used: int
    time_to_answer_seconds: float
    answer_quality: int | None
    answer_quality_reasoning: str = ""


def build_scores(
    facts: PlayFacts,
    metrics: TraceMetrics | None,
    traces_complete: bool,
    trace_problem: str | None = None,
    partial_reason: str = "traces were still arriving",
) -> list[Score]:
    """The play's scores. Without ``metrics`` the trace-based scores are left
    out and the goal_reached comment says why (``trace_problem``); with
    incomplete traces, their comments give ``partial_reason``."""
    scope = f"over the whole play ({facts.outcome})"
    goal_comment = facts.outcome
    if metrics is None:
        goal_comment = f"{facts.outcome}; trace scores missing: {trace_problem or 'no app traces'}"
    scores = [
        Score("goal_reached", 1.0 if facts.goal_reached else 0.0, goal_comment),
        Score("turns_to_goal", facts.turns, scope),
        Score("clarifying_questions", facts.clarifying_questions, scope),
        Score("hints_used", facts.hints_used, scope),
        Score(
            "time_to_answer_seconds",
            round(facts.time_to_answer_seconds, 3),
            f"time the app spent on answered turns, pauses and failed turns excluded, {scope}",
        ),
    ]
    if facts.answer_quality is not None:
        scores.append(Score("answer_quality", facts.answer_quality, facts.answer_quality_reasoning))
    if metrics is None:
        return scores
    note = scope if traces_complete else f"{scope}; {partial_reason}"
    cost_note = note
    if metrics.calls_without_cost:
        cost_note = f"{note}; {metrics.calls_without_cost} model calls carried no cost"
    scores += [
        Score("model_calls_to_goal", metrics.model_calls, note),
        Score(COST_SCORE, round(metrics.cost_usd, 6), cost_note),
        Score("tokens_uncached_input", metrics.tokens_uncached_input, note),
        Score("tokens_cache_read", metrics.tokens_cache_read, note),
        Score("tokens_cache_write", metrics.tokens_cache_write, note),
        Score("tokens_output", metrics.tokens_output, note),
        Score("tool_calls", metrics.tool_calls, note),
        Score("tool_failures", metrics.tool_failures, note),
        Score(
            "cache_miss_calls",
            metrics.cache_miss_calls,
            f"calls that wrote at least {CACHE_MISS_TOKENS} tokens of context, {note}",
        ),
    ]
    return scores
