"""``weather-skills-eval run <config>``.

Exit codes:

- 0: the run finished and the experiment was recorded.
- 1: the run failed for any other reason.
- 2: the run did not start: a harness setting, the configuration file, the
  dataset or a dataset item is invalid, or the dataset is missing or empty.
- 3: the app did not start (image pull, container start, or /health), or
  Docker failed.
- 4: the app failed while the harness read its tools and skill packs.
- 5: Langfuse returned an error.
- 143: the harness was stopped with SIGTERM; the app container was removed.
"""

import argparse
import os
import signal
import sys
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


def _terminate(signum, frame):
    # Raising here unwinds the run, so the container is removed on the way out.
    raise SystemExit(EXIT_TERMINATED)


def main(argv: list[str] | None = None, *, docker_client_factory=_docker_client) -> int:
    parser = argparse.ArgumentParser(prog="weather-skills-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="Play every task in the dataset against one configuration"
    )
    run_parser.add_argument("config", type=Path, help="Configuration file (YAML)")
    args = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, _terminate)

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
