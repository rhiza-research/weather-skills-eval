#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["python-dotenv>=1.2.4", "questionary>=2.1.1"]
# ///
"""Runs the eval for the configs picked from a checklist, with the judge models,
simulated user model, app image, runner image, concurrency and repetitions
chosen for the run. Every run it starts records the same batch id."""

import os
import subprocess
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

import questionary
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent

# Used when neither the shell nor .env sets the value. A blank concurrency means
# the harness's own default; the simulated user model has no default.
DEFAULTS = {
    "EVAL_SIMULATED_USER_MODEL": "",
    "APP_IMAGE": "ghcr.io/rhiza-research/weather-skills-chat:prod",
    "RUNNER_IMAGE": "ghcr.io/rhiza-research/weather-skills-sandbox-runner:prod",
    "EVAL_CONCURRENCY": "",
}

# The judge models offered in the checklist, as LiteLLM names them.
JUDGE_MODELS = [
    "openrouter/google/gemini-4-argon",
    "openrouter/anthropic/claude-opus-5.5",
    "openrouter/openai/gpt-5.5",
]


def _dotenv(dotenv_path):
    # Not interpolated: a ${NAME} reference would prefill another, possibly secret, value.
    return dotenv_values(dotenv_path, interpolate=False)


def prefilled(environ, dotenv_path):
    """Each value from the shell, else from .env, else its default."""
    dotenv = _dotenv(dotenv_path)
    return {
        name: environ.get(name) or dotenv.get(name) or default
        for name, default in DEFAULTS.items()
    }


def ticked_judges(environ, dotenv_path):
    """The models of JUDGE_MODELS that EVAL_JUDGE_MODELS names, from the shell, else from .env."""
    current = environ.get("EVAL_JUDGE_MODELS") or _dotenv(dotenv_path).get("EVAL_JUDGE_MODELS")
    named = {model.strip() for model in (current or "").split(",")}
    return [model for model in JUDGE_MODELS if model in named]


def judges_valid(picked):
    """True for at least one judge, else the message questionary shows."""
    return bool(picked) or "Pick at least one judge"


def simulated_user_valid(text):
    """True for a model name, else the message questionary shows."""
    return bool(text.strip()) or "A simulated user model is required"


# Validators of the prefilled prompts; a prompt not named here takes any answer.
VALIDATORS = {"EVAL_SIMULATED_USER_MODEL": simulated_user_valid}


def env_args(values):
    return [arg for name, value in values.items() for arg in ("-e", f"{name}={value}")]


def new_batch():
    """The current UTC time, taken once before the runs and recorded by each as ``batch``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def repetitions_valid(text):
    """True for a whole number of at least 1, else the message questionary shows."""
    text = text.strip()
    # isdecimal, not isdigit: int() rejects digits such as "²" that isdigit accepts.
    return (text.isdecimal() and int(text) >= 1) or "A whole number, at least 1"


def run_eval(results):
    configs = sorted(p.name for p in (ROOT / "configs").glob("*.yaml") if p.name != "example.yaml")
    picked = questionary.checkbox("Configs to run:", choices=configs).unsafe_ask()
    if not picked:
        return 0
    ticked = ticked_judges(os.environ, ROOT / ".env")
    judges = questionary.checkbox(
        "Judge models:",
        choices=[questionary.Choice(model, checked=model in ticked) for model in JUDGE_MODELS],
        validate=judges_valid,
    ).unsafe_ask()
    values = {
        "EVAL_JUDGE_MODELS": ",".join(judges),
        **{
            name: questionary.text(f"{name}:", default=value, validate=VALIDATORS.get(name))
            .unsafe_ask()
            .strip()
            for name, value in prefilled(os.environ, ROOT / ".env").items()
        },
    }
    repetitions = int(
        questionary.text("Repetitions:", default="1", validate=repetitions_valid)
        .unsafe_ask()
        .strip()
    )
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

    batch = new_batch()
    print(f"Batch {batch}", flush=True)
    command = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-T",
        *env_args({**values, "EVAL_BATCH": batch}),
        "eval",
        "run",
        "-",
    ]
    # In rounds, each picked config once per round.
    for repetition in range(1, repetitions + 1):
        for name in picked:
            with open(ROOT / "configs" / name, "rb") as config:
                process = subprocess.Popen(command, cwd=ROOT, stdin=config)
                try:
                    returncode = process.wait()
                except KeyboardInterrupt:
                    # Ctrl-C reaches docker compose too; let it remove the run's containers.
                    process.wait()
                    raise
            label = f"{name} run {repetition}"
            results.append((label, returncode))
            print(f"{label}: exit {returncode}", flush=True)
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
    dotenv.write_text(
        "APP_IMAGE=dotenv:app\nRUNNER_IMAGE=${APP_IMAGE}\n"
        "EVAL_SIMULATED_USER_MODEL=openrouter/dotenv/user\n"
    )
    assert prefilled({"APP_IMAGE": "shell:app"}, dotenv) == {
        "EVAL_SIMULATED_USER_MODEL": "openrouter/dotenv/user",
        "APP_IMAGE": "shell:app",
        "RUNNER_IMAGE": "${APP_IMAGE}",
        "EVAL_CONCURRENCY": "",
    }
    from_shell = prefilled({"EVAL_SIMULATED_USER_MODEL": "openrouter/shell/user"}, dotenv)
    assert from_shell["EVAL_SIMULATED_USER_MODEL"] == "openrouter/shell/user"
    assert prefilled({}, tmp_path / "missing") == DEFAULTS


def test_judges_are_ticked_from_the_shell_then_dotenv(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"EVAL_JUDGE_MODELS={JUDGE_MODELS[0]}\n")
    assert ticked_judges({}, dotenv) == [JUDGE_MODELS[0]]
    # Spaces are ignored, the checklist's order kept, and a model not in it is left out.
    shell = {"EVAL_JUDGE_MODELS": f" {JUDGE_MODELS[2]} , openrouter/other/judge,{JUDGE_MODELS[1]}"}
    assert ticked_judges(shell, dotenv) == [JUDGE_MODELS[1], JUDGE_MODELS[2]]
    assert ticked_judges({}, tmp_path / "missing") == []


def test_at_least_one_judge_must_be_picked():
    assert judges_valid([JUDGE_MODELS[0]]) is True
    assert judges_valid([]) == "Pick at least one judge"


def test_the_simulated_user_model_must_not_be_blank():
    assert simulated_user_valid(" openrouter/test/user ") is True
    for text in ("", "   "):
        assert simulated_user_valid(text) == "A simulated user model is required"


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


def test_repetitions_must_be_a_whole_number_of_at_least_1():
    assert repetitions_valid(" 3 ") is True
    for text in ("0", "", "1.5", "-1", "two", "²"):
        assert repetitions_valid(text) == "A whole number, at least 1"


class _Answer:
    def __init__(self, value):
        self.value = value

    def unsafe_ask(self):
        return self.value


def test_a_full_run(tmp_path, monkeypatch, capsys):
    (tmp_path / "configs").mkdir()
    for name in ("a", "b", "example"):
        (tmp_path / "configs" / f"{name}.yaml").write_text(f"name: {name}\n")
    (tmp_path / ".env").write_text(
        f"EVAL_JUDGE_MODELS={JUDGE_MODELS[1]}\nEVAL_SIMULATED_USER_MODEL=openrouter/test/user\n"
    )
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    for name in [*DEFAULTS, "EVAL_JUDGE_MODELS"]:
        monkeypatch.delenv(name, raising=False)
    judge_choices = []

    def checkbox(message, choices, **kwargs):
        if message == "Judge models:":
            assert kwargs["validate"] is judges_valid
            judge_choices.extend((c.value, c.checked) for c in choices)
            # The ticked judge and the first one.
            return _Answer([c.value for c in choices if c.checked or c.value == JUDGE_MODELS[0]])
        return _Answer(choices)

    monkeypatch.setattr(questionary, "checkbox", checkbox)

    def text(message, default, **kwargs):
        if message == "Repetitions:":
            assert kwargs["validate"] is repetitions_valid
            return _Answer(" 2 ")
        assert kwargs["validate"] is VALIDATORS.get(message.removesuffix(":"))
        return _Answer(f" {default} ")

    monkeypatch.setattr(questionary, "text", text)
    monkeypatch.setattr(questionary, "confirm", lambda message, default: _Answer(default))
    monkeypatch.setitem(globals(), "new_batch", lambda: "2026-10-22T20:00:00Z")
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
    # Every judge model is offered, the one .env names ticked.
    assert judge_choices == [(model, model == JUDGE_MODELS[1]) for model in JUDGE_MODELS]
    # The picked judges, the prefilled answers with their spaces trimmed, and the batch id.
    values = {
        "EVAL_JUDGE_MODELS": f"{JUDGE_MODELS[0]},{JUDGE_MODELS[1]}",
        **DEFAULTS,
        "EVAL_SIMULATED_USER_MODEL": "openrouter/test/user",
        "EVAL_BATCH": "2026-10-22T20:00:00Z",
    }
    command = ["docker", "compose", "run", "--rm", "-T", *env_args(values), "eval", "run", "-"]
    # Two rounds of every picked config.
    assert runs == [
        (command, {"cwd": tmp_path, "stdin": b"name: a\n"}),
        (command, {"cwd": tmp_path, "stdin": b"name: b\n"}),
        (command, {"cwd": tmp_path, "stdin": b"name: a\n"}),
        (command, {"cwd": tmp_path, "stdin": b"name: b\n"}),
    ]
    assert capsys.readouterr().out == (
        "Batch 2026-10-22T20:00:00Z\n"
        "a.yaml run 1: exit 2\nb.yaml run 1: exit 0\n"
        "a.yaml run 2: exit 2\nb.yaml run 2: exit 0\n"
    )


if __name__ == "__main__":
    sys.exit(main())
