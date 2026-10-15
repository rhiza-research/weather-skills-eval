"""Scores of one play, from the play itself and from the app's Langfuse traces.

Token counts come from each generation's ``usage_details``. Both the app's
own generations and OpenRouter Broadcast generations count cached tokens
inside ``input``; the cache read and cache write counts are moved out of
``input``, so ``uncached input`` holds only tokens billed at the full input
price. A cache-miss call is a model call that wrote at least
``CACHE_MISS_TOKENS`` tokens of context it could not read from the cache
(uncached input plus cache write).
"""

from dataclasses import dataclass, field

CACHE_MISS_TOKENS = 50_000

# Key spellings of cache counts in generation usage_details.
CACHE_READ_KEYS = ("cache_read_input_tokens", "input_cache_read", "input_cached_tokens")
CACHE_WRITE_KEYS = ("cache_creation_input_tokens", "input_cache_write", "input_cache_creation")

APP_TRACE_NAME = "weather-skills-chat"


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


def _first(usage: dict, keys: tuple[str, ...]) -> int:
    for key in keys:
        if usage.get(key) is not None:
            return int(usage[key])
    return 0


def split_usage(usage: dict) -> tuple[int, int, int, int]:
    """(uncached input, cache read, cache write, output) of one generation."""
    usage = usage or {}
    read = _first(usage, CACHE_READ_KEYS)
    write = _first(usage, CACHE_WRITE_KEYS)
    uncached = max(int(usage.get("input") or 0) - read - write, 0)
    return uncached, read, write, int(usage.get("output") or 0)


def trace_metrics(observations: list[dict]) -> TraceMetrics:
    """Observations are dicts with type, name, level, usage_details, cost_details, total_cost."""
    metrics = TraceMetrics()
    metrics.trace_ids = sorted({o["trace_id"] for o in observations if o.get("trace_id")})
    for obs in observations:
        kind, level = obs.get("type"), obs.get("level")
        if kind == "GENERATION":
            metrics.model_calls += 1
            uncached, read, write, output = split_usage(obs.get("usage_details") or {})
            metrics.tokens_uncached_input += uncached
            metrics.tokens_cache_read += read
            metrics.tokens_cache_write += write
            metrics.tokens_output += output
            if uncached + write >= CACHE_MISS_TOKENS:
                metrics.cache_miss_calls += 1
            cost = (obs.get("cost_details") or {}).get("total", obs.get("total_cost"))
            if cost is None:
                metrics.calls_without_cost += 1
            else:
                metrics.cost_usd += float(cost)
        elif kind == "SPAN" and obs.get("name") == "llm" and level == "ERROR":
            # A model call the app could not send or the provider refused.
            metrics.model_calls += 1
        elif kind == "TOOL":
            metrics.tool_calls += 1
            if level == "ERROR":
                metrics.tool_failures += 1
    return metrics


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
    facts: PlayFacts, metrics: TraceMetrics | None, traces_complete: bool
) -> list[Score]:
    scope = "up to the goal" if facts.goal_reached else f"over the whole play ({facts.outcome})"
    scores = [
        Score("goal_reached", 1.0 if facts.goal_reached else 0.0, facts.outcome),
        Score("turns_to_goal", facts.turns, scope),
        Score("clarifying_questions", facts.clarifying_questions, scope),
        Score("hints_used", facts.hints_used, scope),
        Score(
            "time_to_answer_seconds",
            round(facts.time_to_answer_seconds, 3),
            f"time the app spent answering, pauses excluded, {scope}",
        ),
    ]
    if facts.answer_quality is not None:
        scores.append(Score("answer_quality", facts.answer_quality, facts.answer_quality_reasoning))
    if metrics is None:
        return scores
    note = scope if traces_complete else f"{scope}; traces were still arriving when read"
    cost_note = note
    if metrics.calls_without_cost:
        cost_note = f"{note}; {metrics.calls_without_cost} model calls carried no cost"
    scores += [
        Score("model_calls_to_goal", metrics.model_calls, note),
        Score("cost_to_goal_usd", round(metrics.cost_usd, 6), cost_note),
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
