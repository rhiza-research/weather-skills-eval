"""``weather-skills-eval run <config>``."""

import argparse
import os
import sys
from pathlib import Path

from .config import ConfigError
from .instance import AppStartError
from .settings import SettingsError
from .tasks import TaskError


def _docker_client():
    import docker

    return docker.from_env()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="weather-skills-eval")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser(
        "run", help="Play every task in the dataset against one configuration"
    )
    run_parser.add_argument("config", type=Path, help="Configuration file (YAML)")
    args = parser.parse_args(argv)

    from .runner import run

    try:
        summary = run(args.config, os.environ, docker_client_factory=_docker_client)
    except (SettingsError, ConfigError, TaskError) as exc:
        print(f"Run not started: {exc}", file=sys.stderr)
        return 2
    except AppStartError as exc:
        print(f"Run stopped, the app did not start: {exc}", file=sys.stderr)
        return 1

    print(f"Experiment {summary.experiment_name}, run {summary.run_name}")
    print(f"Goal reached on {summary.goals_reached} of {summary.items} tasks")
    if summary.experiment_url:
        print(summary.experiment_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())
