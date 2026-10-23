"""The harness's own settings, each read from one environment variable."""

import math
from collections.abc import Mapping
from dataclasses import dataclass


class SettingsError(Exception):
    """A harness setting is missing or invalid. The message names the variable."""


@dataclass(frozen=True)
class Settings:
    simulated_user_model: str
    # The judge panel, in the order EVAL_JUDGE_MODELS names it.
    judge_models: tuple[str, ...]
    langfuse_environment: str
    docker_network: str
    app_start_timeout: float
    turn_timeout: float
    poll_interval: float
    trace_timeout: float
    trace_settle: float
    concurrency: int
    model_retries: int
    model_max_tokens: int
    max_turns: int
    harness_commit: str
    batch: str


# Read by the Langfuse SDK and LiteLLM themselves; checked here so a run fails
# before any container starts.
REQUIRED_PASSTHROUGH = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "OPENROUTER_API_KEY")


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise SettingsError(f"{name} is not set")
    return value


def _models(environ: Mapping[str, str], name: str) -> tuple[str, ...]:
    """At least one comma-separated model name; none empty, none twice."""
    raw = _required(environ, name)
    models = tuple(model.strip() for model in raw.split(","))
    if not all(models):
        raise SettingsError(f"{name} has an empty model name: {raw!r}")
    if len(set(models)) != len(models):
        raise SettingsError(f"{name} names a model more than once: {raw!r}")
    return models


def _number(
    environ: Mapping[str, str], name: str, default: float, *, minimum: float | None = None
) -> float:
    """A finite number; greater than 0, or at least ``minimum`` when given."""
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} is not a number: {raw!r}") from exc
    if not math.isfinite(value):
        raise SettingsError(f"{name} must be a finite number: {raw!r}")
    if minimum is None and value <= 0:
        raise SettingsError(f"{name} must be greater than 0")
    if minimum is not None and value < minimum:
        raise SettingsError(f"{name} must be at least {minimum:g}")
    return value


def _whole(environ: Mapping[str, str], name: str, default: int, *, minimum: int) -> int:
    value = _number(environ, name, default, minimum=minimum)
    if value != int(value):
        raise SettingsError(f"{name} must be a whole number")
    return int(value)


def load_settings(environ: Mapping[str, str]) -> Settings:
    for name in REQUIRED_PASSTHROUGH:
        _required(environ, name)
    environment = _required(environ, "LANGFUSE_TRACING_ENVIRONMENT")
    if environment.strip().lower() == "production":
        raise SettingsError("LANGFUSE_TRACING_ENVIRONMENT must not be production")
    trace_timeout = _number(environ, "EVAL_TRACE_TIMEOUT_SECONDS", 600)
    trace_settle = _number(environ, "EVAL_TRACE_SETTLE_SECONDS", 60)
    if trace_settle >= trace_timeout:
        raise SettingsError(
            "EVAL_TRACE_SETTLE_SECONDS must be less than EVAL_TRACE_TIMEOUT_SECONDS"
        )
    return Settings(
        simulated_user_model=_required(environ, "EVAL_SIMULATED_USER_MODEL"),
        judge_models=_models(environ, "EVAL_JUDGE_MODELS"),
        langfuse_environment=environment,
        docker_network=_required(environ, "EVAL_DOCKER_NETWORK"),
        app_start_timeout=_number(environ, "EVAL_APP_START_TIMEOUT_SECONDS", 600),
        turn_timeout=_number(environ, "EVAL_TURN_TIMEOUT_SECONDS", 1800),
        poll_interval=_number(environ, "EVAL_POLL_INTERVAL_SECONDS", 2),
        trace_timeout=trace_timeout,
        trace_settle=trace_settle,
        concurrency=_whole(environ, "EVAL_CONCURRENCY", 1, minimum=1),
        model_retries=_whole(environ, "EVAL_MODEL_RETRIES", 2, minimum=0),
        model_max_tokens=_whole(environ, "EVAL_MODEL_MAX_TOKENS", 8192, minimum=1),
        max_turns=_whole(environ, "EVAL_MAX_TURNS", 20, minimum=1),
        # Set at image build time from the harness checkout; see compose.yaml.
        harness_commit=environ.get("EVAL_HARNESS_COMMIT", "").strip(),
        # Groups the runs one scripts/run-eval.py invocation starts.
        batch=environ.get("EVAL_BATCH", "").strip(),
    )
