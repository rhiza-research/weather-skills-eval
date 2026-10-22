"""``weather-skills-eval run <config>``; ``run -`` reads the configuration from stdin.

Exit codes:

- 0: the run finished and the experiment was recorded, also when tasks
  were not scored because the sandbox runner restarted during them; the
  summary gives their number.
- 1: the run failed for any other reason.
- 2: the run did not start: a harness setting, the configuration file, the
  dataset or a dataset item is invalid, the dataset is missing or empty, or
  the dataset needs a skill pack the configuration does not turn on.
- 3: the app or the sandbox runner did not start (an app image without the
  sandbox emulator, image pull, image not on the Docker host, container
  start, or /health), the runner was not running steadily by the start
  timeout before a task, or after the last task when no task waited for it
  after it was lost, the runner was found exited with code 0, dead,
  disappeared or unreadable before or after a task, or Docker failed.
- 4: the app failed while the harness read its catalog model, tools and
  skill packs, or reported no base model for the catalog model.
- 5: Langfuse returned an error.
- 143: the harness was stopped with SIGTERM; the app and runner containers
  and the volume were removed.

``weather-skills-eval delete [--yes] <experiment-id>`` deletes every trace of
one run, after ``yes`` on stdin unless ``--yes`` is given. Exit codes:

- 0: the traces were deleted, the experiment has no items, or the answer was
  not ``yes`` and nothing was deleted.
- 1: the delete failed for any other reason.
- 2: no experiment has that id; nothing was deleted.
- 5: Langfuse returned an error.
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
from .instance import AppStartError, RunnerStartError
from .langfuse_gateway import ExperimentNotFound, LangfuseRuns
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


def _fail(code: int, message: str) -> int:
    print(message, file=sys.stderr)
    return code


def _confirmed() -> bool:
    """True only when the answer on stdin is ``yes``; end of input is no."""
    try:
        answer = input("Type yes to delete these traces: ")
    except EOFError:
        return False
    return answer.strip() == "yes"


def _rerun(deleting: bool) -> str:
    """The hint for a failure after deleting started."""
    return "; run delete again to delete the rest" if deleting else ""


def delete(experiment_id: str, *, yes: bool, runs_factory=LangfuseRuns) -> int:
    """Prints the run's counts, then deletes its traces on ``yes`` (or with
    ``--yes``); any other answer deletes nothing."""
    deleting = False
    try:
        runs = runs_factory()
        found = runs.run_traces(experiment_id)
        if not found.items:
            print(f"Experiment {found.experiment_name} has no items; nothing to delete")
            return 0
        counts = f"{found.items} items, {len(found.trace_ids)} traces"
        print(f"Experiment {found.experiment_name}: {counts}")
        if not (yes or _confirmed()):
            print("Nothing deleted")
            return 0
        deleting = True
        runs.delete_traces(found)
    except ExperimentNotFound:
        return _fail(EXIT_NOT_STARTED, f"Nothing deleted: no experiment has id {experiment_id}")
    except LangfuseApiError as exc:
        return _fail(
            EXIT_LANGFUSE, f"Delete stopped, Langfuse returned an error: {exc}{_rerun(deleting)}"
        )
    except Exception as exc:
        return _fail(EXIT_FAILED, f"Delete failed: {type(exc).__name__}: {exc}{_rerun(deleting)}")

    print(f"Deleted experiment {found.experiment_name}: {counts}")
    return 0


def main(
    argv: list[str] | None = None,
    *,
    docker_client_factory=_docker_client,
    runs_factory=LangfuseRuns,
) -> int:
    parser = argparse.ArgumentParser(prog="weather-skills-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="Play every task in the dataset against one configuration"
    )
    run_parser.add_argument(
        "config", type=Path, help="Configuration file (YAML), or - to read it from stdin"
    )
    delete_parser = commands.add_parser("delete", help="Delete every trace of one run")
    delete_parser.add_argument(
        "experiment_id", help="The experiment id: the baseline= value in the experiment URL"
    )
    delete_parser.add_argument(
        "--yes", action="store_true", help="Delete without asking for yes on stdin"
    )
    args = parser.parse_args(argv)

    if args.command == "delete":
        return delete(args.experiment_id, yes=args.yes, runs_factory=runs_factory)

    signal.signal(signal.SIGTERM, _terminate)
    silence_langwatch()

    try:
        summary = run(args.config, os.environ, docker_client_factory=docker_client_factory)
    except (SettingsError, ConfigError, TaskError, DatasetError) as exc:
        return _fail(EXIT_NOT_STARTED, f"Run not started: {exc}")
    except RunnerStartError as exc:
        return _fail(EXIT_APP_NOT_STARTED, f"Run stopped, the sandbox runner did not start: {exc}")
    except AppStartError as exc:
        return _fail(EXIT_APP_NOT_STARTED, f"Run stopped, the app did not start: {exc}")
    except docker.errors.DockerException as exc:
        return _fail(EXIT_APP_NOT_STARTED, f"Run stopped, Docker failed: {exc}")
    except (ChatError, httpx.HTTPError) as exc:
        return _fail(EXIT_APP_FAILED, f"Run stopped, the app failed: {exc}")
    except LangfuseApiError as exc:
        return _fail(EXIT_LANGFUSE, f"Run stopped, Langfuse returned an error: {exc}")
    except Exception as exc:
        return _fail(EXIT_FAILED, f"Run failed: {type(exc).__name__}: {exc}")

    print(f"Experiment {summary.experiment_name}, run {summary.run_name}")
    print(f"Goal reached on {summary.goals_reached} of {summary.items} tasks")
    if summary.runner_lost:
        lost = len(summary.runner_lost)
        print(
            f"{lost} {'task' if lost == 1 else 'tasks'} not scored because the sandbox "
            f"runner restarted during {'it' if lost == 1 else 'them'}: "
            f"{', '.join(summary.runner_lost)}"
        )
    if summary.experiment_url:
        print(summary.experiment_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
