"""``weather-skills-eval run <config>``; ``run -`` reads the configuration from stdin.

Exit codes:

- 0: the run finished and the experiment was recorded.
- 1: the run failed for any other reason.
- 2: the run did not start: a harness setting, the configuration file, the
  dataset or a dataset item is invalid, the dataset is missing or empty, or
  the dataset needs a skill pack the configuration does not turn on.
- 3: the app did not start (image pull, image not on the Docker host,
  container start, or /health), or Docker failed.
- 4: the app failed while the harness read its catalog model, tools and
  skill packs, or reported no base model for the catalog model.
- 5: Langfuse returned an error.
- 143: the harness was stopped with SIGTERM; the app container was removed.
"""

import argparse
import logging
import os
import signal
import sys
import warnings
from pathlib import Path

import docker.errors
import httpx
from langfuse.api.core.api_error import ApiError as LangfuseApiError

from .app_client import ChatError
from .config import ConfigError
from .instance import AppStartError
from .runner import DatasetError, run
from .settings import SettingsError
from .tasks import TaskError

EXIT_FAILED = 1
EXIT_NOT_STARTED = 2
EXIT_APP_NOT_STARTED = 3
EXIT_APP_FAILED = 4
EXIT_LANGFUSE = 5
EXIT_TERMINATED = 143


def _docker_client():
    import docker

    return docker.from_env()


LANGWATCH_NO_TRACE = r"No trace in context when calling langwatch\.get_current_trace\(\)"
LANGWATCH_NO_API_KEY = "No LangWatch api_key configured"


class _DropLangWatchApiKeyWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return not record.getMessage().startswith(LANGWATCH_NO_API_KEY)


def silence_langwatch() -> None:
    """Drop the LangWatch notices Scenario emits (the no-trace warning, the
    no-api-key log line and the startup banner): the harness traces to
    Langfuse, not LangWatch, so none applies to a run."""
    # Scenario prints its "LangWatch API key not configured" banner unless this is set.
    os.environ.setdefault("SCENARIO_DISABLE_SIMULATION_REPORT_INFO", "1")
    # Scenario asks LangWatch for its current trace; with no LangWatch trace it warns.
    warnings.filterwarnings(
        "ignore",
        message=LANGWATCH_NO_TRACE,
        category=UserWarning,
        module=r"langwatch\.telemetry\.context",
    )
    # Scenario's event reporter logs this once per process when no LangWatch key is set.
    reporter = logging.getLogger("scenario._events.event_reporter")
    if not any(isinstance(f, _DropLangWatchApiKeyWarning) for f in reporter.filters):
        reporter.addFilter(_DropLangWatchApiKeyWarning())


def _terminate(signum, frame):
    # Raising here unwinds the run, so the container is removed on the way out.
    raise SystemExit(EXIT_TERMINATED)


def main(argv: list[str] | None = None, *, docker_client_factory=_docker_client) -> int:
    parser = argparse.ArgumentParser(prog="weather-skills-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="Play every task in the dataset against one configuration"
    )
    run_parser.add_argument(
        "config", type=Path, help="Configuration file (YAML), or - to read it from stdin"
    )
    args = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, _terminate)
    silence_langwatch()

    def fail(code: int, message: str) -> int:
        print(message, file=sys.stderr)
        return code

    try:
        summary = run(args.config, os.environ, docker_client_factory=docker_client_factory)
    except (SettingsError, ConfigError, TaskError, DatasetError) as exc:
        return fail(EXIT_NOT_STARTED, f"Run not started: {exc}")
    except AppStartError as exc:
        return fail(EXIT_APP_NOT_STARTED, f"Run stopped, the app did not start: {exc}")
    except docker.errors.DockerException as exc:
        return fail(EXIT_APP_NOT_STARTED, f"Run stopped, Docker failed: {exc}")
    except (ChatError, httpx.HTTPError) as exc:
        return fail(EXIT_APP_FAILED, f"Run stopped, the app failed: {exc}")
    except LangfuseApiError as exc:
        return fail(EXIT_LANGFUSE, f"Run stopped, Langfuse returned an error: {exc}")
    except Exception as exc:
        return fail(EXIT_FAILED, f"Run failed: {type(exc).__name__}: {exc}")

    print(f"Experiment {summary.experiment_name}, run {summary.run_name}")
    print(f"Goal reached on {summary.goals_reached} of {summary.items} tasks")
    if summary.experiment_url:
        print(summary.experiment_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
