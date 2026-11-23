"""``weather-skills-eval run`` runs one model of a configs repo commit, as its
environment describes it.

Exit codes:

- 0: the run finished and the experiment was recorded, also when tasks
  were not scored because the sandbox runner restarted during them; the
  summary gives their number.
- 1: the run failed for any other reason.
- 2: the run did not start: a harness setting, the configs repo files (another schema version, a
  connection without a key), the dataset or a dataset item is
  invalid, the dataset is missing, empty or
  could not be read from Langfuse (any error, a Langfuse error included), or
  the dataset needs a skill pack the configs repo commit does not turn on.
- 3: the app or the sandbox runner did not start (an app image without the
  sandbox emulator, image pull, a pulled image with no digest for its
  repository, image not on the Docker host, container start, or /health), the runner was not running steadily by the start
  timeout before a task, or after the last task when no task waited for it
  after it was lost, the runner was found exited with code 0, dead,
  disappeared or unreadable before or after a task, or Docker failed.
- 4: the app's skill state could not be made to match the configs repo, or
  the app failed while the harness read its catalog model, tools and
  skill packs, or reported no base model for the catalog model, or, after
  the dataset was read, a network error (an httpx error, such as a
  connection error or timeout) reaching the app or Langfuse stopped the run.
- 5: Langfuse returned an error response after the dataset was read.
- 143: the harness was stopped with SIGTERM; the app and runner containers
  and the volume were removed.

``weather-skills-eval batch`` runs one batch from its environment; see
``batch.py`` for its exit codes.

``weather-skills-eval delete [--yes] <experiment-id>`` deletes every trace of
one run, after ``yes`` on stdin unless ``--yes`` is given. Exit codes:

- 0: the traces were deleted, the experiment has no items, or the answer was
  not ``yes`` and nothing was deleted.
- 1: the delete failed for any other reason.
- 2: no experiment has that id; nothing was deleted.
- 5: Langfuse returned an error.
"""

import argparse
import contextlib
import json
import logging
import os
import signal
import sys
import warnings

import docker.errors
import httpx
from langfuse.api.core.api_error import ApiError as LangfuseApiError

from . import batch
from .app_client import ChatError
from .config import ConfigError
from .configs_repo import ConfigsRepoError, SkillStateError
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


def keep_runs_off_langwatch() -> None:
    """Remove every casing of ``LANGWATCH_API_KEY`` from ``os.environ`` before
    Scenario starts. With a key, Scenario posts every run event, transcripts
    included, to LangWatch and exports its spans there; the harness records
    runs only to Langfuse. Scenario has no setting that turns either off, skips
    both when it finds no key, and reads the key ignoring case. The run still
    gets the key in the copy of the environment ``main`` takes first."""
    for name in list(os.environ):
        if name.upper() == "LANGWATCH_API_KEY":
            del os.environ[name]


@contextlib.contextmanager
def log_to_stderr():
    """While the run lasts, print the harness's log lines, INFO and above, on
    stderr, one message per line: its progress through the run and its
    warnings. They stay off stdout, which holds the summary. The lines do not
    also reach the root logger, so a handler another
    library put there does not print them twice."""
    logger = logging.getLogger(__package__)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    level, propagate = logger.level, logger.propagate
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate


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


COST_LINE = "Cost "


def cost_line(config_name: str, served_model: str, agent_cost_usd: float) -> str:
    """One line, so it can be read from the run's mixed output."""
    cost = {
        "config_name": config_name,
        "served_model": served_model,
        "agent_cost_usd": round(agent_cost_usd, 6),
    }
    return COST_LINE + json.dumps(cost)


def main(
    argv: list[str] | None = None,
    *,
    docker_client_factory=_docker_client,
    runs_factory=LangfuseRuns,
) -> int:
    parser = argparse.ArgumentParser(prog="weather-skills-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "run", help="Play every task in the dataset against one model, as the environment says"
    )
    commands.add_parser("batch", help="Run one batch, as its environment describes it")
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
    if args.command == "batch":
        return batch.main(os.environ, docker_client_factory=docker_client_factory)

    signal.signal(signal.SIGTERM, _terminate)
    silence_langwatch()
    environ = dict(os.environ)
    keep_runs_off_langwatch()

    try:
        with log_to_stderr():
            summary = run(environ, docker_client_factory=docker_client_factory)
    except (SettingsError, ConfigError, ConfigsRepoError, TaskError, DatasetError) as exc:
        return _fail(EXIT_NOT_STARTED, f"Run not started: {exc}")
    except SkillStateError as exc:
        return _fail(EXIT_APP_FAILED, f"Run stopped, the app's skill state: {exc}")
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

    config_name = summary.metadata["config_name"]
    print(f"Experiment {summary.experiment_name} ({config_name}), run {summary.run_name}")
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
    print(cost_line(config_name, summary.metadata["served_model"], summary.agent_cost_usd))
    return 0


if __name__ == "__main__":
    sys.exit(main())
