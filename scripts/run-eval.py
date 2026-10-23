#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["python-dotenv>=1.2.4", "pyyaml>=6.0.3", "questionary>=2.1.1"]
# ///
"""Runs the eval for the configs picked from a checklist, with the judge models,
simulated user model, app image, runner image, concurrency, repetitions and
number of runs at once chosen for the run. Every run it starts records the same
batch id. Before building, it prints the batch's estimated cost from the agent
costs saved in .costs.json and OpenRouter's prices."""

import http.client
import io
import json
import math
import os
import subprocess
import sys
import threading
import types
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import questionary
import yaml
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

# How many runs the script keeps going at once; prefilled like DEFAULTS, but
# not passed with -e, and the harness does not read it.
PARALLEL_RUNS = {"EVAL_PARALLEL_RUNS": "1"}

# The judge models offered in the checklist, as LiteLLM names them.
JUDGE_MODELS = [
    "openrouter/google/gemini-4-argon",
    "openrouter/anthropic/claude-opus-5.5",
    "openrouter/openai/gpt-5.5",
]

# Each config's agent cost from its latest run, by config name; not checked in.
COSTS_FILE = ".costs.json"
# The line a run's summary reports its agent cost on.
COST_LINE = "Cost "
# Public, no key needed; prices are USD per token.
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
# Rough tokens per run of a config, (input, output), measured over the
# 2026-10-22 runs: per judge model, and for the simulated user.
JUDGE_TOKENS = (102_000, 14_000)
SIMULATED_USER_TOKENS = (48_000, 15_000)


def _dotenv(dotenv_path):
    # Not interpolated: a ${NAME} reference would prefill another, possibly secret, value.
    return dotenv_values(dotenv_path, interpolate=False)


def prefilled(environ, dotenv_path, defaults=DEFAULTS):
    """Each value of ``defaults`` from the shell, else from .env, else its default."""
    dotenv = _dotenv(dotenv_path)
    return {
        name: environ.get(name) or dotenv.get(name) or default for name, default in defaults.items()
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


def whole_number_valid(text):
    """True for a whole number of at least 1, else the message questionary shows."""
    text = text.strip()
    # isdecimal, not isdigit: int() rejects digits such as "²" that isdigit accepts.
    return (text.isdecimal() and int(text) >= 1) or "A whole number, at least 1"


def config_name(path):
    """The name a run records as ``config_name``: the file's ``name``; None when
    the file sets none, since a run of it reads the config from stdin and fails."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, yaml.YAMLError):
        raw = None
    name = raw.get("name") if isinstance(raw, dict) else None
    return name.strip() if isinstance(name, str) and name.strip() else None


def parse_cost_line(line):
    """The config name, served model and agent cost on a cost line; None for any other line."""
    if not line.startswith(COST_LINE):
        return None
    try:
        cost = json.loads(line[len(COST_LINE) :])
    except ValueError:
        return None
    if not isinstance(cost, dict) or not isinstance(cost.get("config_name"), str):
        return None
    if _usd(cost.get("agent_cost_usd")) is None:
        return None
    return cost


def read_costs(path):
    """The saved costs by config name; none when the file does not exist.
    Raises OSError or ValueError when it cannot be read."""
    if not path.exists():
        return {}
    costs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(costs, dict):
        raise ValueError("it is not a JSON object")
    return costs


def save_cost(path, cost):
    """Saves ``cost``, a parsed cost line, as its config's latest cost. A file
    that cannot be read is left as it is; a failed write is reported, never raised."""
    try:
        costs = read_costs(path)
    except (OSError, ValueError) as exc:
        print(f"Cost not saved: {path.name} could not be read: {exc}", flush=True)
        return
    costs[cost["config_name"]] = {
        "served_model": cost.get("served_model"),
        "agent_cost_usd": cost["agent_cost_usd"],
    }
    try:
        path.write_text(json.dumps(costs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"Cost not saved: {path.name} could not be written: {exc}", flush=True)


def _usd(value):
    """``value`` as a float when it is a finite number; None otherwise, booleans included."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def agent_cost(costs, name):
    """The saved agent cost of config ``name``; None when there is none."""
    saved = costs.get(name)
    return _usd(saved.get("agent_cost_usd")) if isinstance(saved, dict) else None


def _price(value):
    """A per-token price sent as a string, as a float; None when it is absent,
    unparsable, not finite or negative (OpenRouter sends "-1" for a variable price)."""
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    try:
        price = float(value)
    except ValueError:
        return None
    return price if math.isfinite(price) and price >= 0 else None


def openrouter_prices():
    """USD per input and per output token of each model OpenRouter lists, by
    model id. An entry without a usable price is left out; only a list that is
    not one raises ValueError."""
    with urllib.request.urlopen(OPENROUTER_MODELS, timeout=30) as response:
        listed = json.load(response)
    models = listed.get("data") if isinstance(listed, dict) else None
    if not isinstance(models, list):
        raise ValueError("unexpected models list: no data list")
    prices = {}
    for model in models:
        if not isinstance(model, dict) or not isinstance(model.get("id"), str):
            continue
        pricing = model.get("pricing")
        if not isinstance(pricing, dict):
            continue
        price = (_price(pricing.get("prompt")), _price(pricing.get("completion")))
        if None not in price:
            prices[model["id"]] = price
    return prices


def grading_cost(model, tokens, prices):
    """One run's cost of a judge or simulated-user model; None when OpenRouter
    lists no price for it."""
    price = prices.get(model.removeprefix("openrouter/"))
    if price is None:
        return None
    return price[0] * tokens[0] + price[1] * tokens[1]


def estimate_lines(configs, repetitions, costs, judges, user_model, prices):
    """The batch's estimated cost, as lines to print: ``configs`` holds a
    (file name, config name) pair per picked config, the config name None when
    the file sets none. A config without a saved cost, or a model without a
    price, is named and left out of the total; a config is named by its config
    name, or by its file name when it has none."""
    agents = sum(agent_cost(costs, name) or 0.0 for _, name in configs) * repetitions
    graders = [(model, JUDGE_TOKENS) for model in judges]
    graders.append((user_model, SIMULATED_USER_TOKENS))
    per_run = [(model, grading_cost(model, tokens, prices)) for model, tokens in graders]
    grading = sum(cost for _, cost in per_run if cost is not None) * len(configs) * repetitions
    lines = [
        f"Estimated cost: ${agents + grading:.2f} "
        f"(agents ${agents:.2f}, judges and simulated user ${grading:.2f})"
    ]
    unknown = list(
        dict.fromkeys(name or file for file, name in configs if agent_cost(costs, name) is None)
    )
    if unknown:
        lines.append(f"Agent cost unknown, not counted: {', '.join(unknown)}")
    unpriced = list(dict.fromkeys(model for model, cost in per_run if cost is None))
    if unpriced:
        lines.append(f"No OpenRouter price, not counted: {', '.join(unpriced)}")
    return lines


def show_estimate(configs, repetitions, judges, user_model):
    """Prints the batch's estimated cost from the saved costs and OpenRouter's
    prices. A cost file or price list that cannot be read makes the estimate
    unavailable; it never stops the run."""
    try:
        costs = read_costs(ROOT / COSTS_FILE)
    except (OSError, ValueError) as exc:
        print(f"Cost estimate unavailable: {COSTS_FILE} could not be read: {exc}", flush=True)
        return
    try:
        prices = openrouter_prices()
    except (OSError, ValueError, http.client.HTTPException) as exc:
        print(f"Cost estimate unavailable: OpenRouter prices could not be read: {exc}", flush=True)
        return
    for line in estimate_lines(configs, repetitions, costs, judges, user_model, prices):
        print(line, flush=True)


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
        questionary.text("Repetitions:", default="1", validate=whole_number_valid)
        .unsafe_ask()
        .strip()
    )
    parallel = int(
        questionary.text(
            "Runs at once:",
            default=prefilled(os.environ, ROOT / ".env", PARALLEL_RUNS)["EVAL_PARALLEL_RUNS"],
            validate=whole_number_valid,
        )
        .unsafe_ask()
        .strip()
    )
    names = {name: config_name(ROOT / "configs" / name) for name in picked}
    show_estimate(
        [(name, names[name]) for name in picked],
        repetitions,
        judges,
        values["EVAL_SIMULATED_USER_MODEL"],
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
    queue = [(name, repetition) for repetition in range(1, repetitions + 1) for name in picked]
    run_queue(command, queue, names, parallel, results)
    return 1 if any(returncode != 0 for _, returncode in results) else 0


def _describe(exc):
    return f"{type(exc).__name__}: {exc}"


def pass_on(stream, out, prefix, lock, seen=None):
    """Prints each line of a run's ``stream`` on ``out`` as the run prints it,
    whole, under ``lock``, after ``prefix``; with a prefix, a last line without
    a newline gets one, so the next line printed starts on its own. ``seen``
    gets each line as the run printed it. A line that cannot be printed, or
    that ``seen`` fails on, is skipped and reading goes on to the end, so the
    run never blocks on a full pipe. Returns the first error as text, or None."""
    error = None
    try:
        for line in stream:
            try:
                newline = "" if not prefix or line.endswith("\n") else "\n"
                shown = prefix + line + newline
                with lock:
                    print(shown, end="", file=out, flush=True)
                if seen:
                    seen(line)
            except Exception as exc:
                error = error or _describe(exc)
    except Exception as exc:
        # The stream itself could not be read.
        error = error or _describe(exc)
    return error


def run_queue(command, queue, names, parallel, results):
    """Runs ``command`` once per queued run, a (config file, round) pair, with
    the config file on stdin, starting the runs in queue order with at most
    ``parallel`` going at once: the next starts when one ends. ``names`` maps a
    config file to its config name.

    Each run's stdout is passed on line by line. With ``parallel`` above 1, its
    stderr is read and passed on too, and every line of both is prefixed with
    ``[<config file> run <round>] ``; with 1, stderr is not read and reaches the
    terminal as the run prints it. When a run ends, its config's cost line is
    saved, and its exit code is added to ``results`` and printed. A run whose
    output or exit the script fails to handle is still added, as failed, after
    a line saying what failed.

    On Ctrl-C, or when a run cannot be started, no further run starts; every
    run started is read to the end and waited for, and is not added to
    ``results``; then the exception is raised."""
    prefixed = parallel > 1
    # Held to print a line, and to save a cost, record and print an exit.
    lock = threading.Lock()
    slots = threading.Semaphore(parallel)
    interrupted = threading.Event()

    def finish(process, name, repetition):
        """Reads the run's output to the end, waits for it and records it."""
        label = f"{name} run {repetition}"
        prefix = f"[{label}] " if prefixed else ""
        # The run's cost line, saved once the run has ended.
        run_cost = None
        # What the script failed to do for the run, as text.
        problems = []
        returncode = None

        def keep_cost(line):
            nonlocal run_cost
            cost = parse_cost_line(line)
            # Only the started config's own cost line.
            if cost and names[name] and cost["config_name"] == names[name]:
                run_cost = cost

        def drain(stream, out, seen=None):
            error = pass_on(stream, out, prefix, lock, seen)
            if error:
                problems.append(error)

        stderr_reader = None
        if prefixed:
            try:
                stderr_reader = threading.Thread(target=drain, args=(process.stderr, sys.stderr))
                stderr_reader.start()
            except Exception as exc:
                problems.append(_describe(exc))
                stderr_reader = None
        drain(process.stdout, sys.stdout, keep_cost)
        if stderr_reader:
            stderr_reader.join()
        elif prefixed:
            # Its reader did not start, so it is read here.
            drain(process.stderr, sys.stderr)
        try:
            returncode = process.wait()
        except Exception as exc:
            problems.append(_describe(exc))
        for stream in (process.stdout, process.stderr) if prefixed else (process.stdout,):
            try:
                stream.close()
            except Exception as exc:
                problems.append(_describe(exc))
        with lock:
            if interrupted.is_set():
                return
            if run_cost:
                try:
                    save_cost(ROOT / COSTS_FILE, run_cost)
                except Exception as exc:
                    problems.append(_describe(exc))
            # Failed, with its own code when that is one, when anything failed.
            code = (returncode or 1) if problems else returncode
            results.append((label, code))
            try:
                for problem in problems:
                    print(f"{label}: failed in run-eval.py: {problem}", flush=True)
                print(f"{label}: exit {code}", flush=True)
            except Exception:
                # stdout cannot be written; the run is recorded all the same.
                pass

    class Started:
        """A started run, finished once: by its own thread, or by the main
        thread when that thread never began."""

        def __init__(self, process, name, repetition):
            self.args = (process, name, repetition)
            self.claim = threading.Lock()
            self.done = threading.Event()

        def finish_once(self):
            if not self.claim.acquire(blocking=False):
                return
            try:
                finish(*self.args)
            finally:
                slots.release()
                self.done.set()

    started = []
    try:
        for name, repetition in queue:
            slots.acquire()
            with open(ROOT / "configs" / name, "rb") as config:
                started.append(
                    Started(
                        subprocess.Popen(
                            command,
                            cwd=ROOT,
                            stdin=config,
                            stdout=subprocess.PIPE,
                            **({"stderr": subprocess.PIPE} if prefixed else {}),
                            text=True,
                            encoding="utf-8",
                            errors="replace",
                        ),
                        name,
                        repetition,
                    )
                )
            threading.Thread(target=started[-1].finish_once).start()
        for run in started:
            run.done.wait()
    except BaseException:
        # Ctrl-C, which reaches every docker compose run too, so each removes
        # its run's containers, or a run that could not be started. Each
        # started run is finished here when its thread never began, its output
        # read to the end so it never blocks on a full pipe.
        with lock:
            interrupted.set()
        for run in started:
            run.finish_once()
            run.done.wait()
        raise


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


def test_runs_at_once_come_from_the_shell_then_dotenv_then_1(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text("EVAL_PARALLEL_RUNS=3\n")
    assert prefilled({}, dotenv, PARALLEL_RUNS) == {"EVAL_PARALLEL_RUNS": "3"}
    shell = {"EVAL_PARALLEL_RUNS": "2"}
    assert prefilled(shell, dotenv, PARALLEL_RUNS) == {"EVAL_PARALLEL_RUNS": "2"}
    assert prefilled({}, tmp_path / "missing", PARALLEL_RUNS) == {"EVAL_PARALLEL_RUNS": "1"}
    # Never among the values passed to a run with -e.
    assert prefilled(shell, dotenv).keys() == DEFAULTS.keys()


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


def test_repetitions_and_runs_at_once_must_be_a_whole_number_of_at_least_1():
    assert whole_number_valid(" 3 ") is True
    for text in ("0", "", "1.5", "-1", "two", "²"):
        assert whole_number_valid(text) == "A whole number, at least 1"


class _Answer:
    def __init__(self, value):
        self.value = value

    def unsafe_ask(self):
        return self.value


def cost_line(config, cost, served_model="provider/model"):
    """A cost line as the harness prints it."""
    data = {"config_name": config, "served_model": served_model, "agent_cost_usd": cost}
    return COST_LINE + json.dumps(data)


# USD per token, (input, output), of the models the tests price.
PRICES = {
    JUDGE_MODELS[0].removeprefix("openrouter/"): (1e-6, 2e-6),
    JUDGE_MODELS[1].removeprefix("openrouter/"): (1e-6, 2e-6),
    "test/user": (1e-6, 1e-6),
}


def test_a_full_run(tmp_path, monkeypatch, capsys):
    (tmp_path / "configs").mkdir()
    for name in ("a", "b", "example"):
        (tmp_path / "configs" / f"{name}.yaml").write_text(f"name: {name}\n")
    (tmp_path / ".env").write_text(
        f"EVAL_JUDGE_MODELS={JUDGE_MODELS[1]}\nEVAL_SIMULATED_USER_MODEL=openrouter/test/user\n"
    )
    # A saved cost for a, from an earlier run; none for b.
    (tmp_path / COSTS_FILE).write_text(
        json.dumps({"a": {"served_model": "provider/model", "agent_cost_usd": 1.5}})
    )
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    for name in [*DEFAULTS, *PARALLEL_RUNS, "EVAL_JUDGE_MODELS"]:
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
            assert kwargs["validate"] is whole_number_valid
            return _Answer(" 2 ")
        if message == "Runs at once:":
            assert kwargs["validate"] is whole_number_valid
            # Neither the shell nor .env sets it.
            assert default == "1"
            return _Answer(f" {default} ")
        assert kwargs["validate"] is VALIDATORS.get(message.removesuffix(":"))
        return _Answer(f" {default} ")

    monkeypatch.setattr(questionary, "text", text)
    asked = []
    # Output read so far; readouterr() empties capsys on each read.
    printed = []

    def seen():
        printed.append(capsys.readouterr().out)
        return "".join(printed)

    def confirm(message, default):
        asked.append(message)
        # The estimate is printed before the pull question.
        assert "Estimated cost:" in seen()
        return _Answer(default)

    monkeypatch.setattr(questionary, "confirm", confirm)
    monkeypatch.setitem(globals(), "new_batch", lambda: "2026-10-22T20:00:00Z")
    monkeypatch.setitem(globals(), "openrouter_prices", lambda: PRICES)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[:3] == ["docker", "compose", "build"]:
            # The estimate is printed before the build.
            assert "Estimated cost:" in seen()
        return subprocess.CompletedProcess(command, 0, stdout="abc\n")

    # b's run also prints a cost line of another config, which is not saved.
    b_output = f"Goal reached on 1 of 1 tasks\n{cost_line('a', 9.0)}\n{cost_line('b', 0.75)}\n"

    def popen(command, **kwargs):
        kwargs["stdin"] = kwargs["stdin"].read()
        calls.append((command, kwargs))
        failed = kwargs["stdin"] == b"name: a\n"
        output = "" if failed else b_output
        return types.SimpleNamespace(stdout=io.StringIO(output), wait=lambda: 2 if failed else 0)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", popen)
    assert main() == 1
    (git, _), (build, build_kwargs), (pull, _), *runs = calls
    # The estimate asks nothing and runs nothing.
    assert asked == ["Pull the app image?"]
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
    streamed = {
        "stdout": subprocess.PIPE,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
    }
    # Two rounds of every picked config.
    assert runs == [
        (command, {"cwd": tmp_path, "stdin": b"name: a\n", **streamed}),
        (command, {"cwd": tmp_path, "stdin": b"name: b\n", **streamed}),
        (command, {"cwd": tmp_path, "stdin": b"name: a\n", **streamed}),
        (command, {"cwd": tmp_path, "stdin": b"name: b\n", **streamed}),
    ]
    # Per run: each judge 0.102 + 0.028, the user 0.048 + 0.015; for 2 configs in 2
    # rounds. Agents: a's saved 1.5 in 2 rounds; b unknown.
    assert seen() == (
        "Estimated cost: $4.29 (agents $3.00, judges and simulated user $1.29)\n"
        "Agent cost unknown, not counted: b\n"
        "Batch 2026-10-22T20:00:00Z\n"
        "a.yaml run 1: exit 2\n"
        f"{b_output}b.yaml run 1: exit 0\n"
        "a.yaml run 2: exit 2\n"
        f"{b_output}b.yaml run 2: exit 0\n"
    )
    # a's failed runs printed no cost line, and b's run's line for a is not a's,
    # so a's saved cost stays; b's run's cost is saved.
    assert json.loads((tmp_path / COSTS_FILE).read_text()) == {
        "a": {"served_model": "provider/model", "agent_cost_usd": 1.5},
        "b": {"served_model": "provider/model", "agent_cost_usd": 0.75},
    }


class FakeRun:
    """A run's process: prints ``out`` on stdout and ``err`` on stderr, and exits
    with ``code``. A held run's stdout ends only once ``release`` is set;
    ``on_wait`` is called when the run is waited for."""

    def __init__(self, out, err="", code=0, hold=False, on_wait=None):
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self.stdout = self._stdout(out)
        self.stderr = io.StringIO(err)
        self.code = code
        self.on_wait = on_wait
        self.waited = False

    def _stdout(self, out):
        yield from io.StringIO(out)
        self.release.wait(10)

    def wait(self):
        self.waited = True
        if self.on_wait:
            self.on_wait()
        return self.code


def queue_setup(tmp_path, monkeypatch, configs):
    """ROOT at ``tmp_path``, with a config file per name in ``configs``; returns
    the queue of one round of them and their config names."""
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    for name in configs:
        (tmp_path / "configs" / f"{name}.yaml").write_text(f"name: {name}\n")
    return [(f"{name}.yaml", 1) for name in configs], {f"{n}.yaml": n for n in configs}


def config_of(kwargs):
    """The config name in the config file a run was given on stdin."""
    return kwargs["stdin"].read().decode().removeprefix("name: ").strip()


def test_runs_start_in_queue_order_with_at_most_the_runs_at_once(tmp_path, monkeypatch, capsys):
    queue, names = queue_setup(tmp_path, monkeypatch, "abcde")
    lock = threading.Lock()
    started, held = [], []
    # Runs started and not yet waited for, now and at most.
    going = {"now": 0, "most": 0}

    def ended():
        with lock:
            going["now"] -= 1

    def popen(command, **kwargs):
        assert kwargs["stdout"] is kwargs["stderr"] is subprocess.PIPE
        name = config_of(kwargs)
        process = FakeRun(f"{cost_line(name, 1.0)}\n", hold=True, on_wait=ended)
        with lock:
            started.append(name)
            held.append(process)
            going["now"] += 1
            going["most"] = max(going["most"], going["now"])
            # With every slot taken the oldest run ends; after the last start, all do.
            releasing = held if len(started) == len(queue) else held[:1]
            if going["now"] == 3 or len(started) == len(queue):
                for run in releasing:
                    run.release.set()
                del held[: len(releasing)]
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    results = []

    run_queue(["docker"], queue, names, 3, results)

    assert started == list("abcde")
    assert going["most"] == 3
    assert sorted(results) == [(f"{name}.yaml run 1", 0) for name in "abcde"]
    out = capsys.readouterr().out
    for name in "abcde":
        assert f"[{name}.yaml run 1] {cost_line(name, 1.0)}\n" in out
        assert f"\n{name}.yaml run 1: exit 0\n" in f"\n{out}"
    # Each run's cost is saved.
    assert read_costs(tmp_path / COSTS_FILE) == {
        name: {"served_model": "provider/model", "agent_cost_usd": 1.0} for name in "abcde"
    }


def test_with_runs_at_once_every_line_of_a_run_is_prefixed(tmp_path, monkeypatch, capsys):
    queue, names = queue_setup(tmp_path, monkeypatch, "ab")
    runs = {
        # a's last stdout line has no newline.
        "a": FakeRun("one\ntwo", err="Task 1 of 1 started: t1\n", code=2),
        "b": FakeRun("three\n", err="Dataset weather-tasks read: 1 items\n"),
    }
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: runs[config_of(kwargs)])
    results = []

    run_queue(["docker"], queue, names, 2, results)

    captured = capsys.readouterr()
    # Whole lines, in each run's order; the two runs' lines may interleave.
    out = captured.out.splitlines(keepends=True)
    a_lines = [line for line in out if "a.yaml" in line]
    assert a_lines == ["[a.yaml run 1] one\n", "[a.yaml run 1] two\n", "a.yaml run 1: exit 2\n"]
    b_lines = [line for line in out if "b.yaml" in line]
    assert b_lines == ["[b.yaml run 1] three\n", "b.yaml run 1: exit 0\n"]
    assert len(out) == 5
    assert sorted(captured.err.splitlines(keepends=True)) == [
        "[a.yaml run 1] Task 1 of 1 started: t1\n",
        "[b.yaml run 1] Dataset weather-tasks read: 1 items\n",
    ]
    assert sorted(results) == [("a.yaml run 1", 2), ("b.yaml run 1", 0)]


def test_ctrl_c_waits_for_every_running_run_and_records_none(tmp_path, monkeypatch, capsys):
    queue, names = queue_setup(tmp_path, monkeypatch, "abc")
    runs = []

    def popen(command, **kwargs):
        name = config_of(kwargs)
        if name == "c":
            # Ctrl-C while the third run starts; the terminal's interrupt ends
            # the two running ones a moment later.
            threading.Timer(0.5, lambda: [run.release.set() for run in runs]).start()
            raise KeyboardInterrupt
        runs.append(FakeRun(f"{name} working\n{cost_line(name, 1.0)}\n", code=130, hold=True))
        return runs[-1]

    monkeypatch.setattr(subprocess, "Popen", popen)
    results = []

    try:
        run_queue(["docker"], queue, names, 3, results)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected KeyboardInterrupt")

    # Both were read to the end and waited for, and neither is recorded or has its cost saved.
    assert [run.waited for run in runs] == [True, True]
    out = capsys.readouterr().out
    for name in "ab":
        assert f"[{name}.yaml run 1] {name} working\n" in out
        assert f"[{name}.yaml run 1] {cost_line(name, 1.0)}\n" in out
    assert "exit" not in out
    assert results == []
    assert not (tmp_path / COSTS_FILE).exists()


def test_ctrl_c_keeps_a_finished_run_and_waits_for_the_running_one(tmp_path, monkeypatch, capsys):
    queue, names = queue_setup(tmp_path, monkeypatch, "abc")
    runs = {}

    def popen(command, **kwargs):
        name = config_of(kwargs)
        if name == "c":
            # With 2 at once, c starts only once a has ended and been recorded;
            # Ctrl-C then, while b runs, which the interrupt ends a moment later.
            threading.Timer(0.5, runs["b"].release.set).start()
            raise KeyboardInterrupt
        held = name == "b"
        runs[name] = FakeRun(f"{cost_line(name, 1.0)}\n", code=130 if held else 0, hold=held)
        return runs[name]

    monkeypatch.setattr(subprocess, "Popen", popen)
    results = []

    try:
        run_queue(["docker"], queue, names, 2, results)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected KeyboardInterrupt")

    assert results == [("a.yaml run 1", 0)]
    assert read_costs(tmp_path / COSTS_FILE) == {
        "a": {"served_model": "provider/model", "agent_cost_usd": 1.0}
    }
    # b was read to the end and waited for, and is not recorded.
    assert runs["b"].waited
    out = capsys.readouterr().out
    assert f"[b.yaml run 1] {cost_line('b', 1.0)}\n" in out
    assert "b.yaml run 1: exit" not in out


def test_a_run_whose_output_the_script_fails_on_is_recorded_as_failed(
    tmp_path, monkeypatch, capsys
):
    queue, names = queue_setup(tmp_path, monkeypatch, "a")

    def failing(line):
        raise RuntimeError("cannot parse")

    monkeypatch.setitem(globals(), "parse_cost_line", failing)
    run = FakeRun("one\ntwo\n")
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: run)
    results = []

    run_queue(["docker"], queue, names, 1, results)

    # Read to the end and waited for, and recorded as failed though it exited 0.
    assert run.waited
    assert results == [("a.yaml run 1", 1)]
    assert capsys.readouterr().out == (
        "one\ntwo\n"
        "a.yaml run 1: failed in run-eval.py: RuntimeError: cannot parse\n"
        "a.yaml run 1: exit 1\n"
    )


def test_a_line_that_cannot_be_printed_is_skipped_and_reading_goes_on():
    class Broken:
        def __init__(self):
            self.written = []

        def write(self, text):
            if "bad" in text:
                raise OSError("closed")
            self.written.append(text)

        def flush(self):
            pass

    out = Broken()
    stream = io.StringIO("bad\ngood\n")

    assert pass_on(stream, out, "[a.yaml run 1] ", threading.Lock()) == "OSError: closed"
    assert "".join(out.written) == "[a.yaml run 1] good\n"
    assert stream.read() == ""


def test_runs_at_once_from_dotenv_read_stderr_and_prefix_every_line(tmp_path, monkeypatch, capsys):
    queue_setup(tmp_path, monkeypatch, "a")
    (tmp_path / ".env").write_text(
        "EVAL_SIMULATED_USER_MODEL=openrouter/test/user\nEVAL_PARALLEL_RUNS=2\n"
    )
    for name in [*DEFAULTS, *PARALLEL_RUNS, "EVAL_JUDGE_MODELS"]:
        monkeypatch.delenv(name, raising=False)
    defaults = {}

    def checkbox(message, choices, **kwargs):
        return _Answer([JUDGE_MODELS[0]] if message == "Judge models:" else choices)

    def text(message, default, **kwargs):
        defaults[message] = default
        return _Answer(default)

    monkeypatch.setattr(questionary, "checkbox", checkbox)
    monkeypatch.setattr(questionary, "text", text)
    monkeypatch.setattr(questionary, "confirm", lambda message, default: _Answer(False))
    monkeypatch.setitem(globals(), "new_batch", lambda: "2026-10-23T09:00:00Z")
    monkeypatch.setitem(globals(), "openrouter_prices", lambda: PRICES)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="abc\n"),
    )
    started = []

    def popen(command, **kwargs):
        started.append(kwargs)
        return FakeRun("Goal reached on 1 of 1 tasks\n", err="Task 1 of 1 started: t1\n")

    monkeypatch.setattr(subprocess, "Popen", popen)

    assert main() == 0

    assert defaults["Runs at once:"] == "2"
    assert started[0]["stderr"] is subprocess.PIPE
    captured = capsys.readouterr()
    assert "[a.yaml run 1] Goal reached on 1 of 1 tasks\n" in captured.out
    assert "\na.yaml run 1: exit 0\n" in captured.out
    assert "[a.yaml run 1] Task 1 of 1 started: t1\n" in captured.err


def test_the_estimate_is_one_total_with_agents_and_grading_apart():
    costs = {"a": {"agent_cost_usd": 2.0}, "b": {"agent_cost_usd": 0.5}}
    configs = [("a.yaml", "a"), ("b.yaml", "b")]
    lines = estimate_lines(configs, 3, costs, JUDGE_MODELS[:2], "openrouter/test/user", PRICES)
    # Agents: (2.0 + 0.5) * 3. Grading: (0.13 * 2 + 0.063) * 2 configs * 3 rounds.
    assert lines == ["Estimated cost: $9.44 (agents $7.50, judges and simulated user $1.94)"]


def test_a_config_without_a_saved_cost_and_a_model_without_a_price_are_named():
    costs = {"a": {"agent_cost_usd": 1.0}, "c": {"agent_cost_usd": "1.0"}}
    configs = [("a.yaml", "a"), ("a2.yaml", "a"), ("b.yaml", "b"), ("b2.yaml", "b")]
    configs += [("c.yaml", "c"), ("unnamed.yaml", None)]
    lines = estimate_lines(
        configs, 1, costs, [JUDGE_MODELS[0], JUDGE_MODELS[2]], "test/user", PRICES
    )
    # Agents: a twice. Grading: (0.13 + 0.063) * 6 configs; gpt-5.5 has no price
    # in PRICES. A config that sets no name is named by its file name.
    assert lines == [
        "Estimated cost: $3.16 (agents $2.00, judges and simulated user $1.16)",
        "Agent cost unknown, not counted: b, c, unnamed.yaml",
        f"No OpenRouter price, not counted: {JUDGE_MODELS[2]}",
    ]


def test_only_a_cost_line_is_read_as_a_cost():
    assert parse_cost_line(cost_line("a", 1.25) + "\n") == {
        "config_name": "a",
        "served_model": "provider/model",
        "agent_cost_usd": 1.25,
    }
    for line in (
        "Goal reached on 1 of 1 tasks\n",
        "Cost not json\n",
        'Cost ["a"]\n',
        'Cost {"agent_cost_usd": 1}\n',
        'Cost {"config_name": "a", "agent_cost_usd": "1"}\n',
        'Cost {"config_name": "a", "agent_cost_usd": true}\n',
        'Cost {"config_name": "a", "agent_cost_usd": NaN}\n',
        'Cost {"config_name": "a", "agent_cost_usd": Infinity}\n',
        " " + cost_line("a", 1.25),
    ):
        assert parse_cost_line(line) is None, line


def test_costs_are_saved_by_config_name_latest_first(tmp_path):
    path = tmp_path / COSTS_FILE
    assert read_costs(path) == {}
    save_cost(path, parse_cost_line(cost_line("a", 1.0)))
    save_cost(path, parse_cost_line(cost_line("b", 2.0, "m/b")))
    save_cost(path, parse_cost_line(cost_line("a", 3.0, "m/a")))
    assert read_costs(path) == {
        "a": {"served_model": "m/a", "agent_cost_usd": 3.0},
        "b": {"served_model": "m/b", "agent_cost_usd": 2.0},
    }


def test_an_unreadable_cost_file_is_never_overwritten(tmp_path, capsys):
    path = tmp_path / COSTS_FILE
    path.write_text("[1, 2]")
    save_cost(path, parse_cost_line(cost_line("a", 1.0)))
    assert path.read_text() == "[1, 2]"
    assert "Cost not saved: .costs.json could not be read" in capsys.readouterr().out


def test_a_failed_cost_file_write_is_reported_and_the_run_goes_on(tmp_path, capsys):
    path = tmp_path / "missing" / COSTS_FILE
    save_cost(path, parse_cost_line(cost_line("a", 1.0)))
    assert not path.exists()
    assert "Cost not saved: .costs.json could not be written" in capsys.readouterr().out


def test_the_config_name_is_the_files_name_only(tmp_path):
    named = tmp_path / "opus.yaml"
    named.write_text("name: opus-agent\nmodel: m\n")
    assert config_name(named) == "opus-agent"
    unnamed = tmp_path / "haiku-agent.yaml"
    unnamed.write_text("model: m\n")
    assert config_name(unnamed) is None
    broken = tmp_path / "broken.yaml"
    broken.write_text("name: [unclosed\n")
    assert config_name(broken) is None
    # yaml raises ValueError for a date that does not exist.
    bad_date = tmp_path / "bad-date.yaml"
    bad_date.write_text("name: a\ncreated: 2026-02-30\n")
    assert config_name(bad_date) is None
    not_utf8 = tmp_path / "latin1.yaml"
    not_utf8.write_bytes(b"name: caf\xe9\n")
    assert config_name(not_utf8) is None


def test_openrouter_prices_are_read_from_its_models_list(monkeypatch):
    # Shaped as OpenRouter sends it: prices are strings in USD per token.
    body = {
        "data": [
            {"id": "test/judge", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
            {"id": "test/free", "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "test/no-pricing"},
            {"id": "openrouter/auto", "pricing": {"prompt": "-1", "completion": "-1"}},
            {"id": "test/unparsable", "pricing": {"prompt": "n/a", "completion": "0.1"}},
            {"id": "test/partial", "pricing": {"prompt": "0.000001"}},
            "not an entry",
        ]
    }
    opened = []

    def urlopen(url, timeout):
        opened.append(url)
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    # An entry without a usable price is left out; the rest are still read.
    assert openrouter_prices() == {"test/judge": (1e-6, 2e-6), "test/free": (0.0, 0.0)}
    assert opened == [OPENROUTER_MODELS]


def test_a_models_list_without_data_is_unreadable(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout: io.BytesIO(b"[]"))
    try:
        openrouter_prices()
    except ValueError as exc:
        assert "no data list" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def estimate_setup(tmp_path, monkeypatch):
    """ROOT at ``tmp_path`` and OpenRouter's prices; returns the questions asked
    and the commands run, which must stay empty."""
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    monkeypatch.setitem(globals(), "openrouter_prices", lambda: PRICES)
    asked, ran = [], []

    def confirm(message, default):
        asked.append((message, default))
        return _Answer(default)

    def run(command, **kwargs):
        ran.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(questionary, "confirm", confirm)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(subprocess, "Popen", run)
    return asked, ran


def test_with_every_cost_saved_the_estimate_is_one_total(tmp_path, monkeypatch, capsys):
    asked, ran = estimate_setup(tmp_path, monkeypatch)
    (tmp_path / COSTS_FILE).write_text(json.dumps({"a": {"agent_cost_usd": 1.0}}))

    show_estimate([("a.yaml", "a")], 1, [JUDGE_MODELS[0]], "test/user")

    assert asked == [] and ran == []
    assert capsys.readouterr().out == (
        "Estimated cost: $1.19 (agents $1.00, judges and simulated user $0.19)\n"
    )


def test_a_config_without_a_saved_cost_is_named_unknown(tmp_path, monkeypatch, capsys):
    asked, ran = estimate_setup(tmp_path, monkeypatch)

    show_estimate([("a.yaml", "a")], 1, [JUDGE_MODELS[0]], "test/user")

    assert asked == [] and ran == []
    assert capsys.readouterr().out == (
        "Estimated cost: $0.19 (agents $0.00, judges and simulated user $0.19)\n"
        "Agent cost unknown, not counted: a\n"
    )
    assert not (tmp_path / COSTS_FILE).exists()


def test_an_unreadable_cost_file_makes_the_estimate_unavailable(tmp_path, monkeypatch, capsys):
    asked, ran = estimate_setup(tmp_path, monkeypatch)
    (tmp_path / COSTS_FILE).write_text("not json")

    show_estimate([("a.yaml", "a")], 1, [JUDGE_MODELS[0]], "test/user")

    assert asked == [] and ran == []
    assert capsys.readouterr().out.startswith(
        "Cost estimate unavailable: .costs.json could not be read: "
    )


def test_failed_prices_make_the_estimate_unavailable(tmp_path, monkeypatch, capsys):
    estimate_setup(tmp_path, monkeypatch)
    for error in (OSError, ValueError, http.client.HTTPException):

        def offline(error=error):
            raise error("no network")

        monkeypatch.setitem(globals(), "openrouter_prices", offline)

        show_estimate([("a.yaml", "a")], 1, [JUDGE_MODELS[0]], "test/user")

        assert capsys.readouterr().out == (
            "Cost estimate unavailable: OpenRouter prices could not be read: no network\n"
        ), error


if __name__ == "__main__":
    sys.exit(main())
