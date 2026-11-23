#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["python-dotenv>=1.2.4", "pyyaml>=6.0.3", "questionary>=2.1.1"]
# ///
"""Launches one eval batch on the Docker host it runs on: it resolves the configs
repo ref given to one commit with git, and runs the models picked from that
commit with the eval config, judge models, simulated user model, app image,
runner image, concurrency, repetitions and number of runs at once chosen for
it. It builds the eval image, starts the batch as one detached container with
the configs repo files at that commit copied in, and exits; the batch runs
every queued run on its own, so the terminal can close. With EVAL_HOST=vm in
.env, it first moves the checkout to exactly origin/main and runs itself again
from it."""

import fcntl
import io
import os
import signal
import subprocess
import sys
import tarfile
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path

import questionary
import yaml
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

# The configs repo; no default, since it is set in .env or the shell.
CONFIGS_REPO = {"EVAL_CONFIGS_REPO": ""}
# Where the batch reads the configs repo files: <ARCHIVE_PREFIX> under /app.
ARCHIVE_PREFIX = "configs-repo/"

# The fixed name is the lock: Docker refuses a second container with it.
BATCH_CONTAINER = "weather-skills-eval-batch"
LABEL = "weather-skills-eval"
BATCH_LABEL = "weather-skills-eval.batch"
# The LABEL value of a batch's run containers.
HARNESS_ROLE = "harness"
EVAL_IMAGE_TAG = "weather-skills-eval"
DOCKER_SOCKET = "/var/run/docker.sock"
# Where the launcher copies the configs archive in the batch container.
COPY_TARGET = "/app"
# docker ps output per batch container: ID, state and batch id.
BATCH_FORMAT = f'{{{{.ID}}}}\t{{{{.State}}}}\t{{{{.Label "{BATCH_LABEL}"}}}}'
# Docker states of a batch container that holds no batch. One only created
# never started: the launch lock rules out a launch still starting it.
STOPPED = ("exited", "dead", "created")
# Set for the run of itself after an update, to the commit it moved to, so
# that run does not update again.
UPDATED_TO = "EVAL_CHECKOUT"
# One launch at a time on a host, from the checkout update to the batch's
# start. Outside the checkout, so moving the checkout leaves it alone.
# /tmp, not tempfile.gettempdir(): TMPDIR differs per user, and the members
# on the VM share one lock.
LAUNCH_LOCK = Path("/tmp") / "weather-skills-eval-launch.lock"
# The descriptor of the held lock, passed to the run of itself after an update.
LOCK_FD = "EVAL_LAUNCH_LOCK_FD"
# How long the batch, and each run in it, gets to stop before Docker kills it.
BATCH_STOP_SECONDS = 90
HARNESS_STOP_SECONDS = 60


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


def repo_valid(text):
    """True for an HTTPS URL without credentials, else the message questionary shows."""
    host = text.strip().removeprefix("https://").split("/", 1)[0]
    if text.strip().startswith("https://") and "@" not in host:
        return True
    return "An HTTPS URL of the configs repo, without a user name or token"


def ref_valid(text):
    """True for a ref, else the message questionary shows."""
    # git fetch takes a full SHA, never an abbreviated one.
    # A leading dash would reach git as an option.
    ref = text.strip()
    return (bool(ref) and not ref.startswith("-")) or "A branch, a tag or a full commit SHA"


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
    """A shell command that stops the batch first, so it starts no further run
    and stops its runs, then stops any run still going, so each removes its
    own containers; then removes every container labeled with the batch id
    until none is left, as a run stopping late may still create one; then its
    volumes."""
    label = f"label={BATCH_LABEL}={batch}"

    def stop(role, seconds):
        # By label, never by name: the name may be a newer batch's by now.
        return (
            f"ids=$(docker ps -q --filter '{label}' --filter 'label={LABEL}={role}'); "
            f'[ -n "$ids" ] && docker stop -t {seconds} $ids; '
        )

    return (
        stop("batch", BATCH_STOP_SECONDS)
        + stop(HARNESS_ROLE, HARNESS_STOP_SECONDS)
        + f"for i in 1 2 3 4 5 6 7 8 9 10; do ids=$(docker ps -aq --filter '{label}'); "
        f'[ -z "$ids" ] && break; docker rm -f -v $ids; sleep 1; done; '
        f'[ -n "$ids" ] && echo "Containers left: $ids" >&2; '
        f"for v in $(docker volume ls -q --filter '{label}'); do docker volume rm -f \"$v\"; done"
    )


def fetch_configs(url, ref, folder):
    """Fetches ``ref`` of the configs repo ``url`` with the git access of this
    machine into a new repository in ``folder``, and writes its files to
    ``folder/configs.tar`` under ARCHIVE_PREFIX. Returns the commit SHA and the
    archive's path, or None after printing why it failed."""
    archive = Path(folder) / "configs.tar"
    for args in (
        ["init", "-q", str(folder)],
        ["-C", str(folder), "fetch", "-q", "--depth", "1", url, ref],
        [
            "-C",
            str(folder),
            "archive",
            f"--prefix={ARCHIVE_PREFIX}",
            "-o",
            str(archive),
            "FETCH_HEAD",
        ],
    ):
        done = subprocess.run(["git", *args], capture_output=True, text=True)
        if done.returncode != 0:
            print(f"Configs repo {url} at {ref}: {done.stderr}", end="", file=sys.stderr)
            return None
    commit = subprocess.run(
        ["git", "-C", str(folder), "rev-parse", "FETCH_HEAD"], capture_output=True, text=True
    )
    if commit.returncode != 0:
        print(commit.stderr, end="", file=sys.stderr)
        return None
    return commit.stdout.strip(), archive


def model_ids(archive):
    """The id of each model file in the configs archive, sorted."""
    ids = []
    with tarfile.open(archive) as tar:
        for member in tar.getmembers():
            folder, _, name = member.name.removeprefix(ARCHIVE_PREFIX).rpartition("/")
            if member.isfile() and folder == "models" and name.endswith(".yaml"):
                data = yaml.safe_load(tar.extractfile(member).read())
                model = data.get("model") if isinstance(data, dict) else None
                if isinstance(model, dict) and isinstance(model.get("id"), str):
                    ids.append(model["id"])
    return sorted(ids)


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
    # The run of itself keeps the launch lock: no other launch moves the
    # checkout between this update and its start.
    # take_launch_lock checked this descriptor, so it is an int and open.
    if LOCK_FD in os.environ:
        os.set_inheritable(int(os.environ[LOCK_FD]), True)
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


def take_launch_lock():
    """Holds LAUNCH_LOCK until this process ends, waiting while another launch
    on this host holds it. A run of itself after an update already holds it,
    by the descriptor it inherited."""
    if LOCK_FD in os.environ:
        try:
            # flock on the description this process already holds succeeds again.
            fcntl.flock(int(os.environ[LOCK_FD]), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except (ValueError, OSError):
            # Set in the shell, or stale: not a lock this launch holds.
            del os.environ[LOCK_FD]
    # Read only, so a lock file another user created is still usable; never
    # through a symlink planted in /tmp.
    lock = os.open(LAUNCH_LOCK, os.O_RDONLY | os.O_CREAT | os.O_NOFOLLOW, 0o666)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Another launch is under way on this host; waiting for it", flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
    os.environ[LOCK_FD] = str(lock)


def _stop_launch(number, frame):
    # Unwinds the launch, so its container is removed on the way out.
    raise SystemExit(128 + number)


def run_eval():
    dotenv = ROOT / ".env"
    take_launch_lock()
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
    with tempfile.TemporaryDirectory() as folder:
        return launch(dotenv, containers, offered, Path(folder))


def launch(dotenv, containers, offered, folder):
    """Asks for the batch, builds the eval image and starts the batch, with
    the configs repo files fetched into ``folder``."""
    repo = (
        questionary.text(
            "EVAL_CONFIGS_REPO:",
            default=prefilled(os.environ, dotenv, CONFIGS_REPO)["EVAL_CONFIGS_REPO"],
            validate=repo_valid,
        )
        .unsafe_ask()
        .strip()
    )
    ref = questionary.text("Configs ref:", default="main", validate=ref_valid).unsafe_ask().strip()
    fetched = fetch_configs(repo, ref, folder)
    if fetched is None:
        return 1
    configs_commit, archive = fetched
    print(f"Configs {repo} at {ref} is commit {configs_commit}", flush=True)
    configs = sorted(p.name for p in (ROOT / "configs").glob("*.yaml") if p.name != "example.yaml")
    if not configs:
        print("configs/ holds no eval config besides example.yaml", flush=True)
        return 2
    try:
        offered_models = model_ids(archive)
    except (tarfile.TarError, yaml.YAMLError, UnicodeDecodeError) as exc:
        print(f"The configs repo files at {configs_commit} cannot be read: {exc}", flush=True)
        return 2
    if not offered_models:
        print(f"The configs repo at {configs_commit} has no model file", flush=True)
        return 2
    config = questionary.select("Eval config:", choices=configs).unsafe_ask()
    models = questionary.checkbox("Models to run:", choices=offered_models).unsafe_ask()
    if not models:
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
    # Read only: off the VM the image holds the working tree, changes included.
    # No optional locks: git status may otherwise refresh the index.
    status = _git("--no-optional-locks", "status", "--porcelain", capture=True)
    if status.returncode != 0:
        print(status.stderr, end="", file=sys.stderr)
        return status.returncode
    if status.stdout.strip():
        commit = f"{commit}-dirty"
    iidfile = folder / "image-id"
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

    # A batch container that has stopped keeps its logs until a launch
    # replaces it. By ID: the name may already be another container's.
    for container_id, _, _ in containers:
        removed = subprocess.run(["docker", "rm", container_id])
        if removed.returncode != 0:
            return removed.returncode
    batch = new_batch()
    values.update(
        {
            "EVAL_REPETITIONS": str(int(repetitions)),
            "EVAL_PARALLEL_RUNS": str(int(parallel)),
            "EVAL_PULL_APP_IMAGE": "true" if pull else "false",
            "EVAL_BATCH_CONFIG": config,
            "EVAL_MODELS": ",".join(models),
            "EVAL_CONFIGS_REPO": repo,
            "EVAL_CONFIGS_COMMIT": configs_commit,
            "EVAL_IMAGE": image,
            "EVAL_BATCH": batch,
        }
    )
    # A hangup or termination from here to the start removes the container
    # too; a signal ignored, as under nohup, stays ignored.
    previous = {}
    for number in (signal.SIGHUP, signal.SIGTERM):
        if signal.getsignal(number) is not signal.SIG_IGN:
            previous[number] = signal.signal(number, _stop_launch)
    # Filled in as the steps succeed, so a signal mid-way still finds the ID.
    launched = {"id": None, "started": False}
    try:
        code = _create_and_start(dotenv, values, image, batch, archive, launched)
    finally:
        if launched["id"] and not launched["started"]:
            # A container never started would hold the name of the next batch.
            subprocess.run(["docker", "rm", "-f", launched["id"]], stdout=subprocess.DEVNULL)
        for number, handler in previous.items():
            # None: a handler not set from Python, which cannot be put back.
            if handler is not None:
                signal.signal(number, handler)
    if not launched["started"]:
        return code
    print(f"Batch {batch} started in container {BATCH_CONTAINER}", flush=True)
    print(f"Follow it with: docker logs -f {BATCH_CONTAINER}", flush=True)
    return 0


def _create_and_start(dotenv, values, image, batch, archive, launched):
    """Creates the batch container, copies the configs archive in and starts
    it, recording its ``id`` and whether it ``started`` in ``launched``.
    Returns the exit code of the last step. Every step after the create acts
    on the ID, never on the name."""
    created = subprocess.run(
        [
            "docker",
            "create",
            "--name",
            BATCH_CONTAINER,
            "--restart",
            "no",
            # Only the batch reaches the metadata server, for Secret Manager;
            # the VM blocks it for containers on other networks.
            "--network",
            "host",
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
        stdout=subprocess.PIPE,
        text=True,
    )
    if created.returncode != 0:
        return created.returncode
    launched["id"] = created.stdout.strip()
    # The configs repo files at the commit, copied in before it starts: the
    # batch container has no git access of its own.
    with open(archive, "rb") as files:
        copied = subprocess.run(
            ["docker", "cp", "-", f"{launched['id']}:{COPY_TARGET}"], stdin=files
        ).returncode
    if copied != 0:
        return copied
    start = subprocess.run(["docker", "start", launched["id"]], stdout=subprocess.DEVNULL)
    launched["started"] = start.returncode == 0
    return start.returncode


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


CONFIGS_COMMIT = "c" * 40


class FakeCommands:
    """``subprocess.run``: records each command and answers it from
    ``outputs``, (command prefix, exit code, stdout) entries matched in order.
    ``docker build`` writes ``image`` to its --iidfile; ``git archive`` writes
    a configs archive with models m-b and m-a; ``git rev-parse FETCH_HEAD``
    answers CONFIGS_COMMIT; ``docker cp`` records the archive it reads."""

    def __init__(self, outputs=(), image="sha256:eval"):
        self.outputs = list(outputs)
        self.image = image
        self.calls = []
        self.copied = None

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[:2] == ["docker", "build"]:
            Path(command[command.index("--iidfile") + 1]).write_text(f"{self.image}\n")
        if "archive" in command:
            with tarfile.open(command[command.index("-o") + 1], "w") as tar:
                for model in ("m-b", "m-a"):
                    data = f"schema_version: 1\nmodel:\n  id: {model}\n".encode()
                    info = tarfile.TarInfo(f"{ARCHIVE_PREFIX}models/{model}.yaml")
                    info.size = len(data)
                    tar.addfile(info, io.BytesIO(data))
        if command[-1] == "FETCH_HEAD" and "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, stdout=f"{CONFIGS_COMMIT}\n", stderr="")
        if command[:2] == ["docker", "cp"]:
            self.copied = kwargs["stdin"].read()
        for prefix, code, stdout in self.outputs:
            if command[: len(prefix)] == prefix:
                return subprocess.CompletedProcess(command, code, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def commands(self):
        return [command for command, _ in self.calls]


def launch_setup(tmp_path, monkeypatch, dotenv, outputs=()):
    """ROOT at ``tmp_path`` with the eval config weather-eval and ``dotenv``
    as .env; the questions take their defaults, and every model. Returns the
    fake commands and the questions asked."""
    (tmp_path / "configs").mkdir()
    for name in ("weather-eval", "example"):
        (tmp_path / "configs" / f"{name}.yaml").write_text("dataset: d\n")
    (tmp_path / ".env").write_text(dotenv)
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    for name in [*DEFAULTS, *PARALLEL_RUNS, *CONFIGS_REPO, "EVAL_JUDGE_MODELS", UPDATED_TO]:
        monkeypatch.delenv(name, raising=False)
    # A copy, so the lock descriptor the launch records stays in this test.
    monkeypatch.setattr(os, "environ", {k: v for k, v in os.environ.items() if k != LOCK_FD})
    monkeypatch.setitem(globals(), "LAUNCH_LOCK", tmp_path / "launch.lock")
    asked = []

    def select(message, choices, **kwargs):
        asked.append(message)
        [choice] = choices
        return _Answer(choice)

    monkeypatch.setattr(questionary, "select", select)

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


REPO_URL = "https://github.com/rhiza-research/weather-skills-configs"
DOTENV = (
    f"EVAL_JUDGE_MODELS={JUDGES[0]}\nEVAL_SIMULATED_USER_MODEL=openrouter/test/user\n"
    f"EVAL_CONFIGS_REPO={REPO_URL}\n"
)
LIST_BATCHES = ["docker", "ps", "--all", "--filter", f"label={LABEL}=batch", "--format"]


CREATED_ID = "f00d" * 16
LAUNCH_OUTPUTS = [
    (["git", "rev-parse"], 0, "abc\n"),
    (["docker", "create"], 0, f"{CREATED_ID}\n"),
]


def test_a_launch_builds_the_image_and_starts_one_detached_batch(tmp_path, monkeypatch, capsys):
    # Batch containers from earlier batches: one stopped, one created and never started.
    old = "old1\texited\t2026-11-22T09:00:00Z\nold2\tcreated\t2026-11-22T10:00:00Z\n"
    commands, asked = launch_setup(
        tmp_path,
        monkeypatch,
        DOTENV + "EVAL_PARALLEL_RUNS=2\n",
        [(LIST_BATCHES, 0, old), *LAUNCH_OUTPUTS],
    )

    assert main() == 0

    (
        listed,
        init,
        fetch,
        archive,
        fetched,
        commit,
        status,
        build,
        removed,
        removed_created,
        create,
        cp,
        start,
    ) = commands.commands()
    # Removed by ID, never by name.
    assert removed_created == ["docker", "rm", "old2"]
    # Read only; a clean tree records the commit as it is.
    assert status == ["git", "--no-optional-locks", "status", "--porcelain"]
    # The launch lock is held, outside the checkout.
    assert os.environ[LOCK_FD]
    assert listed == [*LIST_BATCHES, BATCH_FORMAT]
    # The ref, main by default, resolved to one commit with this machine's git.
    folder = init[-1]
    assert init == ["git", "init", "-q", folder]
    assert fetch == ["git", "-C", folder, "fetch", "-q", "--depth", "1", REPO_URL, "main"]
    assert archive[2:5] == [folder, "archive", f"--prefix={ARCHIVE_PREFIX}"]
    assert fetched == ["git", "-C", folder, "rev-parse", "FETCH_HEAD"]
    # Locally the checkout is only read.
    assert commit == ["git", "rev-parse", "HEAD"]
    # Only once the image is built, so a failed build keeps the old logs.
    assert removed == ["docker", "rm", "old1"]
    assert build[:8] == ["docker", "build", "--target", "eval", "--build-arg", "HARNESS_COMMIT=abc"]
    assert build[-3:] == ["--tag", EVAL_IMAGE_TAG, "."]
    assert commands.calls[7][1]["cwd"] == tmp_path
    values = {
        "EVAL_JUDGE_MODELS": JUDGES[0],
        **DEFAULTS,
        "EVAL_SIMULATED_USER_MODEL": "openrouter/test/user",
        "EVAL_REPETITIONS": "1",
        "EVAL_PARALLEL_RUNS": "2",
        "EVAL_PULL_APP_IMAGE": "true",
        "EVAL_BATCH_CONFIG": "weather-eval.yaml",
        # Every model at the commit was picked, offered in id order.
        "EVAL_MODELS": "m-a,m-b",
        "EVAL_CONFIGS_REPO": REPO_URL,
        "EVAL_CONFIGS_COMMIT": CONFIGS_COMMIT,
        "EVAL_IMAGE": "sha256:eval",
        "EVAL_BATCH": "2026-11-23T10:00:00Z",
    }
    # The configs files at the commit go in before the batch starts, both by
    # the ID docker create printed.
    assert cp == ["docker", "cp", "-", f"{CREATED_ID}:{COPY_TARGET}"]
    with tarfile.open(fileobj=io.BytesIO(commands.copied)) as tar:
        assert f"{ARCHIVE_PREFIX}models/m-a.yaml" in tar.getnames()
    assert start == ["docker", "start", CREATED_ID]
    assert create == [
        "docker",
        "create",
        "--name",
        BATCH_CONTAINER,
        "--restart",
        "no",
        "--network",
        "host",
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
        f"Configs {REPO_URL} at main is commit {CONFIGS_COMMIT}\n"
        f"Batch 2026-11-23T10:00:00Z started in container {BATCH_CONTAINER}\n"
        f"Follow it with: docker logs -f {BATCH_CONTAINER}\n"
    )


STUB_DOCKER = """#!/bin/sh
# Logs each call; lists a run, two containers and a volume once each.
echo "$*" >> "$LOG"
case "$*" in
  "ps -q "*"=batch") echo b1-container ;;
  "ps -q "*"=harness") [ -e "$LOG.ran" ] || { touch "$LOG.ran"; echo h1; } ;;
  "ps -aq "*) [ -e "$LOG.swept" ] || { touch "$LOG.swept"; echo c1; echo c2; } ;;
  "volume ls "*) echo v1 ;;
esac
"""


def test_the_kill_command_stops_then_sweeps_until_nothing_is_left(tmp_path):
    (tmp_path / "docker").write_text(STUB_DOCKER)
    (tmp_path / "docker").chmod(0o755)
    log = tmp_path / "calls"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}", "LOG": str(log)}
    # The real subprocess.run: the command runs in sh against the stub.
    subprocess.run(["sh", "-c", kill_command("b1")], env=env, check=True)

    label = f"label={BATCH_LABEL}=b1"
    assert log.read_text().splitlines() == [
        # The batch by its labels, never by its name.
        f"ps -q --filter {label} --filter label={LABEL}=batch",
        f"stop -t {BATCH_STOP_SECONDS} b1-container",
        f"ps -q --filter {label} --filter label={LABEL}=harness",
        f"stop -t {HARNESS_STOP_SECONDS} h1",
        f"ps -aq --filter {label}",
        "rm -f -v c1 c2",
        # Swept again, until none is left.
        f"ps -aq --filter {label}",
        f"volume ls -q --filter {label}",
        "volume rm -f v1",
    ]


def test_a_batch_container_the_files_cannot_be_copied_into_is_removed(tmp_path, monkeypatch):
    commands, _ = launch_setup(
        tmp_path, monkeypatch, DOTENV, [*LAUNCH_OUTPUTS, (["docker", "cp"], 1, "")]
    )

    assert main() == 1

    *_, cp, removed = commands.commands()
    assert cp[:2] == ["docker", "cp"]
    assert removed == ["docker", "rm", "-f", CREATED_ID]


def test_a_termination_before_the_start_removes_the_batch_container(tmp_path, monkeypatch):
    commands, _ = launch_setup(tmp_path, monkeypatch, DOTENV, LAUNCH_OUTPUTS)
    copy = commands.__call__

    def terminated_during_copy(command, **kwargs):
        if command[:2] == ["docker", "cp"]:
            # A real SIGTERM, which the handler installed for this window takes.
            os.kill(os.getpid(), signal.SIGTERM)
        return copy(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", terminated_during_copy)
    before = signal.getsignal(signal.SIGTERM)

    try:
        main()
    except SystemExit as exc:
        assert exc.code == 128 + signal.SIGTERM
    else:
        raise AssertionError("expected SystemExit")

    assert commands.commands()[-1] == ["docker", "rm", "-f", CREATED_ID]
    assert ["docker", "start", CREATED_ID] not in commands.commands()
    assert signal.getsignal(signal.SIGTERM) is before


def test_a_second_launch_waits_for_the_launch_lock(tmp_path, monkeypatch, capsys):
    launch_setup(tmp_path, monkeypatch, DOTENV)
    # Another launch holds it, on its own open file description.
    other = os.open(LAUNCH_LOCK, os.O_RDONLY | os.O_CREAT)
    fcntl.flock(other, fcntl.LOCK_EX)
    released = []

    def release():
        released.append(True)
        os.close(other)

    threading.Timer(0.2, release).start()

    take_launch_lock()

    assert released
    assert capsys.readouterr().out == "Another launch is under way on this host; waiting for it\n"
    assert os.environ[LOCK_FD]


def test_off_the_vm_a_checkout_with_changes_is_recorded_dirty_and_left_as_it_is(
    tmp_path, monkeypatch
):
    commands, _ = launch_setup(
        tmp_path,
        monkeypatch,
        DOTENV,
        [*LAUNCH_OUTPUTS, (["git", "--no-optional-locks"], 0, " M src/x.py\n")],
    )

    assert main() == 0

    [build] = [c for c in commands.commands() if c[:2] == ["docker", "build"]]
    assert "HARNESS_COMMIT=abc-dirty" in build
    # Only reads of the checkout; the configs fetch is in its own folder (-C).
    git = [c[1] for c in commands.commands() if c[0] == "git" and c[1] != "-C"]
    assert set(git) <= {"init", "rev-parse", "--no-optional-locks"}


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
    # The run of itself inherits the held lock.
    assert os.get_inheritable(int(env[LOCK_FD]))
    assert asked == []

    # The run of itself finds the checkout at that commit and goes on to launch.
    monkeypatch.setenv(UPDATED_TO, "def")
    commands.calls.clear()
    assert update_checkout() is None
    assert commands.commands() == [["git", "rev-parse", "HEAD"]]


if __name__ == "__main__":
    sys.exit(main())
