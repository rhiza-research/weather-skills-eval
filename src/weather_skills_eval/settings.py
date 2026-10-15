"""The harness's own settings, each read from one environment variable."""

from collections.abc import Mapping
from dataclasses import dataclass


class SettingsError(Exception):
    """A harness setting is missing or invalid. The message names the variable."""


@dataclass(frozen=True)
class Settings:
    simulated_user_model: str
    judge_model: str
    langfuse_environment: str
    docker_network: str
    app_start_timeout: float
    turn_timeout: float
    poll_interval: float
    trace_timeout: float
    trace_settle: float
    concurrency: int


# Read by the Langfuse SDK and LiteLLM themselves; checked here so a run fails
# before any container starts.
REQUIRED_PASSTHROUGH = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "OPENROUTER_API_KEY")


def _required(environ: Mapping[str, str], name: str) -> str:
    value = environ.get(name, "").strip()
    if not value:
        raise SettingsError(f"{name} is not set")
    return value


def _number(environ: Mapping[str, str], name: str, default: float) -> float:
    raw = environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise SettingsError(f"{name} is not a number: {raw!r}") from exc
    if value <= 0:
        raise SettingsError(f"{name} must be greater than 0")
    return value


def load_settings(environ: Mapping[str, str]) -> Settings:
    for name in REQUIRED_PASSTHROUGH:
        _required(environ, name)
    environment = _required(environ, "LANGFUSE_TRACING_ENVIRONMENT")
    if environment == "production":
        raise SettingsError("LANGFUSE_TRACING_ENVIRONMENT must not be production")
    concurrency = _number(environ, "EVAL_CONCURRENCY", 1)
    if concurrency != int(concurrency):
        raise SettingsError("EVAL_CONCURRENCY must be a whole number")
    return Settings(
        simulated_user_model=_required(environ, "EVAL_SIMULATED_USER_MODEL"),
        judge_model=_required(environ, "EVAL_JUDGE_MODEL"),
        langfuse_environment=environment,
        docker_network=_required(environ, "EVAL_DOCKER_NETWORK"),
        app_start_timeout=_number(environ, "EVAL_APP_START_TIMEOUT_SECONDS", 600),
        turn_timeout=_number(environ, "EVAL_TURN_TIMEOUT_SECONDS", 1800),
        poll_interval=_number(environ, "EVAL_POLL_INTERVAL_SECONDS", 2),
        trace_timeout=_number(environ, "EVAL_TRACE_TIMEOUT_SECONDS", 600),
        trace_settle=_number(environ, "EVAL_TRACE_SETTLE_SECONDS", 60),
        concurrency=int(concurrency),
    )
