#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["python-dotenv>=1.2.4", "questionary>=2.1.1"]
# ///
"""Launches one eval batch on the Docker host it runs on, for the configs picked
from a checklist, with the judge models, simulated user model, app image, runner
image, concurrency, repetitions and number of runs at once chosen for it. It
builds the eval image, starts the batch as one detached container and exits;
the batch runs every queued run on its own, so the terminal can close. With
EVAL_HOST=vm in .env, it first moves the checkout to exactly origin/main and
runs itself again from it."""

import os
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import questionary
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = Path(__file__).resolve()

# Used when neither the shell nor .env sets the value. A blank concurrency means
# the harness's own default; the simulated user model has no default.
DEFAULTS = {
    "EVAL_SIMULATED_USER_MODEL": "",
    "APP_IMAGE": "ghcr.io/rhiza-research/weather-skills-chat:prod",
    "RUNNER_IMAGE": "ghcr.io/rhiza-research/weather-skills-sandbox-runner:prod",
    "EVAL_CONCURRENCY": "",
}

# How many runs the batch keeps going at once; prefilled like DEFAULTS.
PARALLEL_RUNS = {"EVAL_PARALLEL_RUNS": "1"}

# The fixed name is the lock: Docker refuses a second container with it.
BATCH_CONTAINER = "weather-skills-eval-batch"
LABEL = "weather-skills-eval"
BATCH_LABEL = "weather-skills-eval.batch"
EVAL_IMAGE_TAG = "weather-skills-eval"
DOCKER_SOCKET = "/var/run/docker.sock"
# docker ps output per batch container: name, state and batch id.
BATCH_FORMAT = f'{{{{.Names}}}}\t{{{{.State}}}}\t{{{{.Label "{BATCH_LABEL}"}}}}'
# Docker states of a batch container that no longer holds a batch.
STOPPED = ("exited", "dead")
# Set for the run of itself after an update, to the commit it moved to, so
# that run does not update again.
UPDATED_TO = "EVAL_CHECKOUT"


def _dotenv(dotenv_path):
    # Not interpolated: a ${NAME} reference would prefill another, possibly secret, value.
    return dotenv_values(dotenv_path, interpolate=False)


def prefilled(environ, dotenv_path, defaults=DEFAULTS):
    """Each value of ``defaults`` from the shell, else from .env, else its default."""
    dotenv = _dotenv(dotenv_path)
    return {
        name: environ.get(name) or dotenv.get(name) or default for name, default in defaults.items()
    }


def judge_models(environ, dotenv_path):
    """The models EVAL_JUDGE_MODELS names, from the shell, else from .env, in its
    order, spaces trimmed, without empty names or repeats."""
    current = environ.get("EVAL_JUDGE_MODELS") or _dotenv(dotenv_path).get("EVAL_JUDGE_MODELS")
    named = (model.strip() for model in (current or "").split(","))
    return list(dict.fromkeys(model for model in named if model))


def on_vm(dotenv_path):
    """True when .env marks this host as the eval VM; the shell does not count."""
    return (_dotenv(dotenv_path).get("EVAL_HOST") or "").strip() == "vm"


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
    """The current UTC time, taken once at launch and recorded by each run as ``batch``."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def whole_number_valid(text):
    """True for a whole number of at least 1, else the message questionary shows."""
    text = text.strip()
    # isdecimal, not isdigit: int() rejects digits such as "²" that isdigit accepts.
    return (text.isdecimal() and int(text) >= 1) or "A whole number, at least 1"


def kill_command(batch):
    """A shell command that removes the batch container first, so it starts no
    further run, then every container and volume labeled with its batch id."""
    label = f"label={BATCH_LABEL}={batch}"
    return (
        f"docker rm -f {BATCH_CONTAINER}; "
        f"for c in $(docker ps -aq --filter '{label}'); do docker rm -f -v \"$c\"; done; "
        f"for v in $(docker volume ls -q --filter '{label}'); do docker volume rm -f \"$v\"; done"
    )


def _git(*args, capture=False):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=capture, text=True)


def update_checkout():
    """On the VM: fetches origin, refuses a checkout with any change, moves it to
    exactly origin/main and runs this script again from it, so the new code
    drives the launch. Returns None when this is that second run, else the exit
    code when it cannot update."""
    head = _git("rev-parse", "HEAD", capture=True)
    if head.returncode != 0:
        print(head.stderr, end="", file=sys.stderr)
        return head.returncode
    if os.environ.get(UPDATED_TO) == head.stdout.strip():
        return None
    fetched = _git("fetch", "origin")
    if fetched.returncode != 0:
        return fetched.returncode
    status = _git("status", "--porcelain", capture=True)
    if status.returncode != 0:
        print(status.stderr, end="", file=sys.stderr)
        return status.returncode
    if status.stdout.strip():
        print(
            f"The checkout {ROOT} has changes; a launch on the VM runs only origin/main:\n"
            f"{status.stdout}",
            end="",
            flush=True,
        )
        return 1
    moved = _git("checkout", "--detach", "origin/main")
    if moved.returncode != 0:
        return moved.returncode
    commit = _git("rev-parse", "HEAD", capture=True)
    if commit.returncode != 0:
        print(commit.stderr, end="", file=sys.stderr)
        return commit.returncode
    environ = {**os.environ, UPDATED_TO: commit.stdout.strip()}
    try:
        os.execvpe("uv", ["uv", "run", str(SCRIPT), *sys.argv[1:]], environ)
    except OSError as exc:
        print(f"Could not run {SCRIPT} again with uv: {exc}", file=sys.stderr)
        return 1


def batch_containers():
    """(name, state, batch id) of each batch container, running or not; None
    when Docker cannot list them."""
    listed = subprocess.run(
        [
            "docker",
            "ps",
            "--all",
            "--filter",
            f"label={LABEL}=batch",
            "--format",
            BATCH_FORMAT,
        ],
        capture_output=True,
        text=True,
    )
    if listed.returncode != 0:
        print(listed.stderr, end="", file=sys.stderr)
        return None
    return [tuple(line.split("\t")) for line in listed.stdout.splitlines() if line.strip()]


def run_eval():
    dotenv = ROOT / ".env"
    if on_vm(dotenv):
        code = update_checkout()
        if code is not None:
            return code
    containers = batch_containers()
    if containers is None:
        return 1
    # Paused, restarting or only created counts as running: it still holds the name.
    running = [batch for _, state, batch in containers if state not in STOPPED]
    if running:
        print(f"A batch is running: {running[0]}", flush=True)
        print(f"Kill it with: {kill_command(running[0])}", flush=True)
        return 1
    offered = judge_models(os.environ, dotenv)
    if not offered:
        print("EVAL_JUDGE_MODELS names no judge model; set it in the shell or .env", flush=True)
        return 2
    configs = sorted(p.name for p in (ROOT / "configs").glob("*.yaml") if p.name != "example.yaml")
    picked = questionary.checkbox("Configs to run:", choices=configs).unsafe_ask()
    if not picked:
        return 0
    judges = questionary.checkbox(
        "Judge models:",
        choices=[questionary.Choice(model, checked=True) for model in offered],
        validate=judges_valid,
    ).unsafe_ask()
    values = {
        "EVAL_JUDGE_MODELS": ",".join(judges),
        **{
            name: questionary.text(f"{name}:", default=value, validate=VALIDATORS.get(name))
            .unsafe_ask()
            .strip()
            for name, value in prefilled(os.environ, dotenv).items()
        },
    }
    repetitions = (
        questionary.text("Repetitions:", default="1", validate=whole_number_valid)
        .unsafe_ask()
        .strip()
    )
    parallel = (
        questionary.text(
            "Runs at once:",
            default=prefilled(os.environ, dotenv, PARALLEL_RUNS)["EVAL_PARALLEL_RUNS"],
            validate=whole_number_valid,
        )
        .unsafe_ask()
        .strip()
    )
    pull = questionary.confirm("Pull the app image?", default=True).unsafe_ask()

    commit = _git("rev-parse", "HEAD", capture=True)
    if commit.returncode != 0:
        print(commit.stderr, end="", file=sys.stderr)
        return commit.returncode
    commit = commit.stdout.strip()
    with tempfile.TemporaryDirectory() as folder:
        iidfile = Path(folder) / "image-id"
        build = subprocess.run(
            [
                "docker",
                "build",
                "--target",
                "eval",
                "--build-arg",
                f"HARNESS_COMMIT={commit}",
                "--iidfile",
                str(iidfile),
                "--tag",
                EVAL_IMAGE_TAG,
                ".",
            ],
            cwd=ROOT,
        )
        if build.returncode != 0:
            return build.returncode
        image = iidfile.read_text().strip()

    # A batch container that has stopped keeps its logs until a launch replaces it.
    for name, _, _ in containers:
        removed = subprocess.run(["docker", "rm", name])
        if removed.returncode != 0:
            return removed.returncode
    batch = new_batch()
    values.update(
        {
            "EVAL_REPETITIONS": str(int(repetitions)),
            "EVAL_PARALLEL_RUNS": str(int(parallel)),
            "EVAL_PULL_APP_IMAGE": "true" if pull else "false",
            "EVAL_BATCH_CONFIGS": ",".join(picked),
            "EVAL_IMAGE": image,
            "EVAL_BATCH": batch,
        }
    )
    started = subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            BATCH_CONTAINER,
            "--restart",
            "no",
            "--label",
            f"{LABEL}=batch",
            "--label",
            f"{BATCH_LABEL}={batch}",
            "--volume",
            f"{DOCKER_SOCKET}:{DOCKER_SOCKET}",
            *(["--env-file", str(dotenv)] if dotenv.exists() else []),
            *env_args(values),
            "--entrypoint",
            "weather-skills-eval",
            image,
            "batch",
        ],
        stdout=subprocess.DEVNULL,
    )
    if started.returncode != 0:
        return started.returncode
    print(f"Batch {batch} started in container {BATCH_CONTAINER}", flush=True)
    print(f"Follow it with: docker logs -f {BATCH_CONTAINER}", flush=True)
    return 0


def main():
    try:
        return run_eval()
    except KeyboardInterrupt:
        print("\nInterrupted.")
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


# Judge models as EVAL_JUDGE_MODELS names them in the tests.
JUDGES = ["openrouter/test/judge-a", "openrouter/test/judge-b", "openrouter/test/judge-c"]


def test_judge_models_come_from_the_shell_then_dotenv(tmp_path):
    dotenv = tmp_path / ".env"
    dotenv.write_text(f"EVAL_JUDGE_MODELS={JUDGES[0]}\n")
    assert judge_models({}, dotenv) == [JUDGES[0]]
    # Spaces trimmed, its order kept, empty names and repeats dropped, any model offered.
    shell = {"EVAL_JUDGE_MODELS": f" {JUDGES[2]} ,,{JUDGES[1]}, {JUDGES[2]}"}
    assert judge_models(shell, dotenv) == [JUDGES[2], JUDGES[1]]
    assert judge_models({}, tmp_path / "missing") == []
    assert judge_models({"EVAL_JUDGE_MODELS": " , "}, tmp_path / "missing") == []


def test_at_least_one_judge_must_be_picked():
    assert judges_valid([JUDGES[0]]) is True
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


class FakeCommands:
    """``subprocess.run``: records each command and answers it from
    ``outputs``, (command prefix, exit code, stdout) entries matched in order.
    ``docker build`` writes ``image`` to its --iidfile."""

    def __init__(self, outputs=(), image="sha256:eval"):
        self.outputs = list(outputs)
        self.image = image
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[:2] == ["docker", "build"]:
            Path(command[command.index("--iidfile") + 1]).write_text(f"{self.image}\n")
        for prefix, code, stdout in self.outputs:
            if command[: len(prefix)] == prefix:
                return subprocess.CompletedProcess(command, code, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def commands(self):
        return [command for command, _ in self.calls]


def launch_setup(tmp_path, monkeypatch, dotenv, outputs=()):
    """ROOT at ``tmp_path`` with configs a and b and ``dotenv`` as .env; the
    questions take their defaults. Returns the fake commands and the questions asked."""
    (tmp_path / "configs").mkdir()
    for name in ("a", "b", "example"):
        (tmp_path / "configs" / f"{name}.yaml").write_text(f"name: {name}\n")
    (tmp_path / ".env").write_text(dotenv)
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    for name in [*DEFAULTS, *PARALLEL_RUNS, "EVAL_JUDGE_MODELS", UPDATED_TO]:
        monkeypatch.delenv(name, raising=False)
    asked = []

    def checkbox(message, choices, **kwargs):
        asked.append(message)
        if message == "Judge models:":
            return _Answer([c.value for c in choices if c.checked])
        return _Answer(choices)

    def text(message, default, **kwargs):
        asked.append(message)
        return _Answer(f" {default} ")

    def confirm(message, default):
        asked.append(message)
        return _Answer(default)

    monkeypatch.setattr(questionary, "checkbox", checkbox)
    monkeypatch.setattr(questionary, "text", text)
    monkeypatch.setattr(questionary, "confirm", confirm)
    monkeypatch.setitem(globals(), "new_batch", lambda: "2026-11-23T10:00:00Z")
    commands = FakeCommands(outputs)
    monkeypatch.setattr(subprocess, "run", commands)
    return commands, asked


DOTENV = f"EVAL_JUDGE_MODELS={JUDGES[0]}\nEVAL_SIMULATED_USER_MODEL=openrouter/test/user\n"
LIST_BATCHES = ["docker", "ps", "--all", "--filter", f"label={LABEL}=batch", "--format"]


def test_a_launch_builds_the_image_and_starts_one_detached_batch(tmp_path, monkeypatch, capsys):
    # A batch container from an earlier batch has stopped.
    old = f"{BATCH_CONTAINER}\texited\t2026-11-22T09:00:00Z\n"
    commands, asked = launch_setup(
        tmp_path,
        monkeypatch,
        DOTENV + "EVAL_PARALLEL_RUNS=2\n",
        [(LIST_BATCHES, 0, old), (["git", "rev-parse"], 0, "abc\n")],
    )

    assert main() == 0

    listed, commit, build, removed, run = commands.commands()
    assert listed == [*LIST_BATCHES, BATCH_FORMAT]
    # Locally the checkout is only read.
    assert commit == ["git", "rev-parse", "HEAD"]
    # Only once the image is built, so a failed build keeps the old logs.
    assert removed == ["docker", "rm", BATCH_CONTAINER]
    assert build[:8] == ["docker", "build", "--target", "eval", "--build-arg", "HARNESS_COMMIT=abc"]
    assert build[-3:] == ["--tag", EVAL_IMAGE_TAG, "."]
    assert commands.calls[2][1]["cwd"] == tmp_path
    values = {
        "EVAL_JUDGE_MODELS": JUDGES[0],
        **DEFAULTS,
        "EVAL_SIMULATED_USER_MODEL": "openrouter/test/user",
        "EVAL_REPETITIONS": "1",
        "EVAL_PARALLEL_RUNS": "2",
        "EVAL_PULL_APP_IMAGE": "true",
        "EVAL_BATCH_CONFIGS": "a.yaml,b.yaml",
        "EVAL_IMAGE": "sha256:eval",
        "EVAL_BATCH": "2026-11-23T10:00:00Z",
    }
    assert run == [
        "docker",
        "run",
        "--detach",
        "--name",
        BATCH_CONTAINER,
        "--restart",
        "no",
        "--label",
        f"{LABEL}=batch",
        "--label",
        f"{BATCH_LABEL}=2026-11-23T10:00:00Z",
        "--volume",
        f"{DOCKER_SOCKET}:{DOCKER_SOCKET}",
        "--env-file",
        str(tmp_path / ".env"),
        *env_args(values),
        "--entrypoint",
        "weather-skills-eval",
        "sha256:eval",
        "batch",
    ]
    assert "Pull the app image?" in asked
    assert capsys.readouterr().out == (
        f"Batch 2026-11-23T10:00:00Z started in container {BATCH_CONTAINER}\n"
        f"Follow it with: docker logs -f {BATCH_CONTAINER}\n"
    )


def test_a_running_batch_is_named_with_its_kill_command_and_nothing_starts(
    tmp_path, monkeypatch, capsys
):
    running = f"{BATCH_CONTAINER}\trunning\t2026-11-23T08:00:00Z\n"
    commands, asked = launch_setup(tmp_path, monkeypatch, DOTENV, [(LIST_BATCHES, 0, running)])

    assert main() == 1

    assert len(commands.calls) == 1
    assert asked == []
    assert capsys.readouterr().out == (
        "A batch is running: 2026-11-23T08:00:00Z\n"
        f"Kill it with: {kill_command('2026-11-23T08:00:00Z')}\n"
    )
    # The batch container first, then its runs' containers and volumes, by label.
    command = kill_command("2026-11-23T08:00:00Z")
    label = f"label={BATCH_LABEL}=2026-11-23T08:00:00Z"
    assert command.index(f"docker rm -f {BATCH_CONTAINER}") < command.index(label)
    assert f"docker ps -aq --filter '{label}'" in command
    assert f"docker volume ls -q --filter '{label}'" in command


def test_without_judge_models_the_script_stops_before_asking(tmp_path, monkeypatch, capsys):
    commands, asked = launch_setup(tmp_path, monkeypatch, "")

    assert main() == 2

    assert asked == []
    assert capsys.readouterr().out == (
        "EVAL_JUDGE_MODELS names no judge model; set it in the shell or .env\n"
    )


def test_on_the_vm_a_checkout_with_changes_is_refused_before_asking(tmp_path, monkeypatch):
    commands, asked = launch_setup(
        tmp_path,
        monkeypatch,
        DOTENV + "EVAL_HOST=vm\n",
        [(["git", "status"], 0, " M scripts/run-eval.py\n"), (["git", "rev-parse"], 0, "abc\n")],
    )

    assert main() == 1

    assert commands.commands() == [
        ["git", "rev-parse", "HEAD"],
        ["git", "fetch", "origin"],
        ["git", "status", "--porcelain"],
    ]
    assert asked == []


def test_on_the_vm_the_checkout_moves_to_origin_main_and_the_script_runs_again(
    tmp_path, monkeypatch
):
    commands, asked = launch_setup(
        tmp_path, monkeypatch, DOTENV + "EVAL_HOST=vm\n", [(["git", "rev-parse"], 0, "def\n")]
    )
    execs = []

    def execvpe(file, args, env):
        execs.append((file, args, env))
        raise SystemExit(0)

    monkeypatch.setattr(os, "execvpe", execvpe)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT)])

    try:
        main()
    except SystemExit:
        pass

    assert commands.commands() == [
        ["git", "rev-parse", "HEAD"],
        ["git", "fetch", "origin"],
        ["git", "status", "--porcelain"],
        ["git", "checkout", "--detach", "origin/main"],
        ["git", "rev-parse", "HEAD"],
    ]
    [(file, args, env)] = execs
    assert (file, args) == ("uv", ["uv", "run", str(SCRIPT)])
    assert env[UPDATED_TO] == "def"
    assert asked == []

    # The run of itself finds the checkout at that commit and goes on to launch.
    monkeypatch.setenv(UPDATED_TO, "def")
    commands.calls.clear()
    assert update_checkout() is None
    assert commands.commands() == [["git", "rev-parse", "HEAD"]]


if __name__ == "__main__":
    sys.exit(main())
