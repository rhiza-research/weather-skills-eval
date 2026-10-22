#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["python-dotenv>=1.2.4", "questionary>=2.1.1"]
# ///
"""Runs the eval for the configs picked from a checklist, with the app image,
runner image and concurrency chosen for the run."""

import os
import subprocess
import sys
import types
from pathlib import Path

import questionary
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent

# Used when neither the shell nor .env sets the value. A blank concurrency means
# the harness's own default.
DEFAULTS = {
    "APP_IMAGE": "ghcr.io/rhiza-research/weather-skills-chat:prod",
    "RUNNER_IMAGE": "ghcr.io/rhiza-research/weather-skills-sandbox-runner:prod",
    "EVAL_CONCURRENCY": "",
}


def prefilled(environ, dotenv_path):
    """Each value from the shell, else from .env, else its default."""
    # Not interpolated: a ${NAME} reference would prefill another, possibly secret, value.
    dotenv = dotenv_values(dotenv_path, interpolate=False)
    return {
        name: environ.get(name) or dotenv.get(name) or default
        for name, default in DEFAULTS.items()
    }


def env_args(values):
    return [arg for name, value in values.items() for arg in ("-e", f"{name}={value}")]


def run_eval(results):
    configs = sorted(p.name for p in (ROOT / "configs").glob("*.yaml") if p.name != "example.yaml")
    picked = questionary.checkbox("Configs to run:", choices=configs).unsafe_ask()
    if not picked:
        return 0
    values = {
        name: questionary.text(f"{name}:", default=value).unsafe_ask().strip()
        for name, value in prefilled(os.environ, ROOT / ".env").items()
    }
    pull = questionary.confirm("Pull the app image?", default=True).unsafe_ask()

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    build = subprocess.run(
        ["docker", "compose", "build", "eval"],
        cwd=ROOT,
        env={**os.environ, "HARNESS_COMMIT": commit},
    )
    if build.returncode != 0:
        return build.returncode
    if pull:
        pulled = subprocess.run(["docker", "pull", values["APP_IMAGE"]])
        if pulled.returncode != 0:
            return pulled.returncode

    command = ["docker", "compose", "run", "--rm", "-T", *env_args(values), "eval", "run", "-"]
    for name in picked:
        with open(ROOT / "configs" / name, "rb") as config:
            process = subprocess.Popen(command, cwd=ROOT, stdin=config)
            try:
                returncode = process.wait()
            except KeyboardInterrupt:
                # Ctrl-C reaches docker compose too; let it remove the run's containers.
                process.wait()
                raise
        results.append((name, returncode))
        print(f"{name}: exit {returncode}", flush=True)
    return 1 if any(returncode != 0 for _, returncode in results) else 0


def main():
    results = []
    try:
        return run_eval(results)
    except KeyboardInterrupt:
        print("\nInterrupted. Results so far:")
        for name, returncode in results:
            print(f"{name}: exit {returncode}")
        return 130


# Tests, run with:
#   uv run --isolated --no-project --with-requirements scripts/run-eval.py --with pytest \
#     pytest scripts/run-eval.py


def test_a_value_comes_from_the_shell_then_dotenv_then_the_default(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("APP_IMAGE=dotenv:app\nRUNNER_IMAGE=${APP_IMAGE}\n")
    assert prefilled({"APP_IMAGE": "shell:app"}, dotenv) == {
        "APP_IMAGE": "shell:app",
        "RUNNER_IMAGE": "${APP_IMAGE}",
        "EVAL_CONCURRENCY": "",
    }
    assert prefilled({}, tmp_path / "missing") == DEFAULTS


def test_every_value_is_passed_with_dash_e():
    values = {"APP_IMAGE": "a:1", "RUNNER_IMAGE": "r:1", "EVAL_CONCURRENCY": ""}
    assert env_args(values) == [
        "-e",
        "APP_IMAGE=a:1",
        "-e",
        "RUNNER_IMAGE=r:1",
        "-e",
        "EVAL_CONCURRENCY=",
    ]


class _Answer:
    def __init__(self, value):
        self.value = value

    def unsafe_ask(self):
        return self.value


def test_a_full_run(tmp_path, monkeypatch, capsys):
    (tmp_path / "configs").mkdir()
    for name in ("a", "b", "example"):
        (tmp_path / "configs" / f"{name}.yaml").write_text(f"name: {name}\n")
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    for name in DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(questionary, "checkbox", lambda message, choices: _Answer(choices))
    monkeypatch.setattr(questionary, "text", lambda message, default: _Answer(f" {default} "))
    monkeypatch.setattr(questionary, "confirm", lambda message, default: _Answer(default))
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, stdout="abc\n")

    def popen(command, **kwargs):
        kwargs["stdin"] = kwargs["stdin"].read()
        calls.append((command, kwargs))
        returncode = 2 if kwargs["stdin"] == b"name: a\n" else 0
        return types.SimpleNamespace(wait=lambda: returncode)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", popen)
    assert main() == 1
    (git, _), (build, build_kwargs), (pull, _), *runs = calls
    assert git == ["git", "rev-parse", "HEAD"]
    assert build == ["docker", "compose", "build", "eval"]
    assert build_kwargs["env"]["HARNESS_COMMIT"] == "abc"
    assert pull == ["docker", "pull", DEFAULTS["APP_IMAGE"]]
    # The prefilled answers, with their spaces trimmed.
    command = ["docker", "compose", "run", "--rm", "-T", *env_args(DEFAULTS), "eval", "run", "-"]
    assert runs == [
        (command, {"cwd": tmp_path, "stdin": b"name: a\n"}),
        (command, {"cwd": tmp_path, "stdin": b"name: b\n"}),
    ]
    assert capsys.readouterr().out == "a.yaml: exit 2\nb.yaml: exit 0\n"


if __name__ == "__main__":
    sys.exit(main())
