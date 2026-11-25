#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "google-cloud-secret-manager>=2.32.0",
#     "python-dotenv>=1.2.4",
#     "pyyaml>=6.0.3",
#     "questionary>=2.1.1",
# ]
# ///
"""Launches and runs eval batches, in one of two modes.

Interactive mode, with no arguments, runs with uv on a member's machine. It
asks for the configs repo and a ref, which it resolves to one commit with git,
the models at that commit to run, the judge models, simulated user model, app
image, runner image, concurrency, repetitions, number of runs at once, whether
to pull the app image and the eval image tag, and prints the batch's estimated
cost from the agent costs saved in .costs.json and OpenRouter's prices. Then it
prints two ways to launch the batch, the two gcloud commands that create a VM
that runs it and the docker run command of the eval image, and runs the one
the member picks, if any.

Non-interactive mode, with every batch value as a flag, runs in the eval
container and runs the batch on the Docker host whose socket it has. It reads
each secret the environment leaves empty from Secret Manager."""

import argparse
import http.client
import importlib.util
import io
import json
import math
import os
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
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
# The eval image tag the batch runs; prefilled like DEFAULTS.
IMAGE_TAG = {"EVAL_IMAGE_TAG": "main"}

# The configs repo; no default, since it is set in .env or the shell.
CONFIGS_REPO = {"EVAL_CONFIGS_REPO": ""}
# Where each run reads the configs repo files.
CONFIGS_DIR = "/configs-repo"
# Each run's container command: unpacks the tar of the configs repo files it
# gets on stdin, so no host path is mounted and a remote Docker context works.
UNPACK_AND_RUN = (
    f"mkdir -p {CONFIGS_DIR} && tar -x -f - -C {CONFIGS_DIR} && exec weather-skills-eval run"
)

# Batch values. A variable NAME is the flag --name-lowercase-dashes and the VM
# metadata key name-lowercase-dashes. Each is required; an empty value is a value.
# The questions' answers.
ANSWERS = (
    "EVAL_CONFIGS_REPO",
    "EVAL_CONFIGS_COMMIT",
    # The model ids to run, comma-separated.
    "EVAL_MODELS",
    "EVAL_JUDGE_MODELS",
    "EVAL_SIMULATED_USER_MODEL",
    "APP_IMAGE",
    "RUNNER_IMAGE",
    "EVAL_CONCURRENCY",
    "EVAL_REPETITIONS",
    "EVAL_PARALLEL_RUNS",
    # true or false: whether the batch pulls APP_IMAGE once, before the runs.
    "EVAL_PULL_APP_IMAGE",
    "EVAL_IMAGE_TAG",
)
# The non-secret settings .env.example sets that are not answers.
SETTINGS = (
    "EVAL_SECRETS_PROJECT",
    "LANGFUSE_BASE_URL",
    "LANGFUSE_TRACING_ENVIRONMENT",
    "EVAL_CONNECTION_KEYS",
    "EVAL_DATASET",
    "EVAL_ADMIN_EMAIL",
)
# The settings .env.example leaves commented out; a batch value only when set.
OPTIONAL_SETTINGS = (
    "EVAL_APP_START_TIMEOUT_SECONDS",
    "EVAL_TURN_TIMEOUT_SECONDS",
    "EVAL_POLL_INTERVAL_SECONDS",
    "EVAL_TRACE_TIMEOUT_SECONDS",
    "EVAL_TRACE_SETTLE_SECONDS",
    "EVAL_MODEL_RETRIES",
    "EVAL_RATE_LIMIT_RETRIES",
    "EVAL_RATE_LIMIT_BACKOFF_SECONDS",
    "EVAL_MODEL_MAX_TOKENS",
    "EVAL_MAX_TURNS",
)
# Batch values the batch itself uses, which no run is given.
BATCH_ONLY = ("EVAL_MODELS", "EVAL_REPETITIONS", "EVAL_PARALLEL_RUNS", "EVAL_IMAGE_TAG")

# The eval image, holding the harness and this script.
EVAL_IMAGE = "ghcr.io/rhiza-research/weather-skills-eval"
# The Python that has this script's dependencies in the eval image, and the script there.
IMAGE_PYTHON = "/opt/run-eval/bin/python"
IMAGE_SCRIPT = "/app/scripts/run-eval.py"
# The batch container's name on a Docker host: one batch at a time.
BATCH_CONTAINER = "weather-skills-eval-batch"
# Seconds docker stop waits for the batch to stop its runs before it kills it.
STOP_TIMEOUT = "120"
# A batch VM: created from this instance template, named this prefix and the time.
INSTANCE_TEMPLATE = "weather-skills-eval"
VM_PREFIX = "weather-skills-eval-"
VM_ZONE = "us-central1-a"

# The app image labels each run records.
REVISION_LABEL = "org.opencontainers.image.revision"
PR_NUMBER_LABEL = "weather-skills-chat.pr-number"

# The secrets every run needs: the harness's own and those it sets the app's
# from. The variables EVAL_CONNECTION_KEYS names are needed too.
SECRETS = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "OPENROUTER_API_KEY",
    "APP_WEBUI_SECRET_KEY",
    "APP_ADMIN_PASSWORD",
    "APP_ADMIN_API_KEY",
)
# The secrets the batch needs and no run does: the configs repo read token.
BATCH_SECRETS = ("GIT_READ_TOKEN",)
# Secret Manager secret id of an environment variable: this prefix, then the
# name lowercased with underscores as dashes. Shared with the VM's Terraform.
SECRET_PREFIX = "weather-skills-eval-"

# Each model's agent cost from its latest run, by model id; not checked in.
COSTS_FILE = ".costs.json"
# The line a run's summary reports its agent cost on.
COST_LINE = "Cost "
# Public, no key needed; prices are USD per token.
OPENROUTER_MODELS = "https://openrouter.ai/api/v1/models"
# Rough tokens per run of a model, (input, output), measured over the
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


def judge_models(environ, dotenv_path):
    """The models EVAL_JUDGE_MODELS names, from the shell, else from .env, in its
    order, spaces trimmed, without empty names or repeats."""
    current = environ.get("EVAL_JUDGE_MODELS") or _dotenv(dotenv_path).get("EVAL_JUDGE_MODELS")
    named = (model.strip() for model in (current or "").split(","))
    return list(dict.fromkeys(model for model in named if model))


def judges_valid(picked):
    """True for at least one judge, else the message questionary shows."""
    return bool(picked) or "Pick at least one judge"


def simulated_user_valid(text):
    """True for a model name, else the message questionary shows."""
    return bool(text.strip()) or "A simulated user model is required"


def image_tag_valid(text):
    """True for a tag, else the message questionary shows."""
    return bool(text.strip()) or "An eval image tag is required"


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


def metadata_key(name):
    """``EVAL_IMAGE_TAG`` -> ``eval-image-tag``: the VM metadata key of a batch value."""
    return name.lower().replace("_", "-")


def flag(name):
    """``EVAL_IMAGE_TAG`` -> ``--eval-image-tag``: the flag of a batch value."""
    return "--" + metadata_key(name)


def metadata_keys():
    """The VM metadata key of every batch value."""
    return [metadata_key(name) for name in (*ANSWERS, *SETTINGS, *OPTIONAL_SETTINGS)]


def batch_flags(values):
    """``values`` as flags, ``--name=value``, so a value may start with a dash."""
    return [f"{flag(name)}={value}" for name, value in values.items()]


def vm_name(now):
    """The name of a batch VM created at ``now``: VM_PREFIX and the UTC time."""
    return VM_PREFIX + now.strftime("%Y%m%d-%H%M%S")


def vm_commands(vm, values):
    """The two gcloud commands that launch a batch VM: create it from the
    instance template without metadata, so the template's cloud-init stays,
    then add every batch value as metadata. The metadata list is split on a
    character no value holds (gcloud topic escaping); ValueError when every
    candidate is in a value."""
    delimiter = next((c for c in ";|#~" if not any(c in v for v in values.values())), None)
    if delimiter is None:
        raise ValueError("every metadata delimiter ;|#~ is in a batch value")
    pairs = delimiter.join(f"{metadata_key(name)}={value}" for name, value in values.items())
    return [
        ["gcloud", "compute", "instances", "create", vm, "--zone", VM_ZONE]
        + ["--source-instance-template", INSTANCE_TEMPLATE],
        ["gcloud", "compute", "instances", "add-metadata", vm, "--zone", VM_ZONE]
        + ["--metadata", f"^{delimiter}^{pairs}"],
    ]


def docker_command(values, secrets):
    """The docker run command that runs the batch in the eval image in
    non-interactive mode, on the current Docker context's host. Each secret is
    passed by name only, from the environment docker run is given."""
    return [
        "docker",
        "run",
        "--detach",
        "--rm",
        "--pull",
        "always",
        "--name",
        BATCH_CONTAINER,
        "--stop-timeout",
        STOP_TIMEOUT,
        # The Docker host's own socket, so no path from this machine is mounted.
        "--volume",
        "/var/run/docker.sock:/var/run/docker.sock",
        *(arg for name in secrets for arg in ("-e", name)),
        f"{EVAL_IMAGE}:{values['EVAL_IMAGE_TAG']}",
        IMAGE_PYTHON,
        IMAGE_SCRIPT,
        *batch_flags(values),
    ]


def batch_running():
    """True when the batch container exists on the current Docker context's host."""
    inspected = subprocess.run(
        ["docker", "container", "inspect", BATCH_CONTAINER], capture_output=True, text=True
    )
    return inspected.returncode == 0


def fetch_configs(url, ref, folder):
    """Fetches ``ref`` of the configs repo ``url`` with the git access of this
    machine into a new repository in ``folder`` and checks it out there.
    Returns the commit SHA, or None after printing why it failed."""
    for args in (
        ["init", "-q", str(folder)],
        ["-C", str(folder), "fetch", "-q", "--depth", "1", url, ref],
        ["-C", str(folder), "checkout", "-q", "--detach", "FETCH_HEAD"],
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
    return commit.stdout.strip()


def configs_archive(folder):
    """The tar of the configs repo checkout in ``folder``, next to it."""
    return Path(f"{folder}.tar")


def github_repo(url):
    """``<owner>/<repo>`` of ``url`` when it is ``https://github.com/<owner>/<repo>``
    (a trailing ``/`` or ``.git`` allowed); None, after printing why, otherwise."""
    path = url.removeprefix("https://github.com/").removesuffix("/").removesuffix(".git")
    owner, _, repo = path.partition("/")
    if not url.startswith("https://github.com/") or not owner or not repo or "/" in repo:
        print(f"Configs repo {url} is not https://github.com/<owner>/<repo>", flush=True)
        return None
    return path


def _open(request):
    return urllib.request.urlopen(request, timeout=120)


def download_configs(url, commit, token, folder):
    """Downloads the files of the GitHub repository ``url`` at ``commit`` with
    the read token ``token``, through GitHub's tarball API, into ``folder``.
    Returns the folder holding them, or None after printing why it failed."""
    path = github_repo(url)
    if path is None:
        return None
    request = urllib.request.Request(
        f"https://api.github.com/repos/{path}/tarball/{commit}",
        headers={"Accept": "application/vnd.github+json"},
    )
    # Unredirected: GitHub sends the download to codeload.github.com with its
    # own token in the URL, and the read token never follows a redirect.
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    try:
        with _open(request) as response, tarfile.open(fileobj=response, mode="r|gz") as tar:
            # "data" refuses absolute paths, links out of the folder and device files.
            tar.extractall(folder, filter="data")
    except (OSError, tarfile.TarError, http.client.HTTPException) as exc:
        print(f"Configs repo {url} at {commit}: {_describe(exc)}", flush=True)
        return None
    # The tarball's one top folder, <owner>-<repo>-<short SHA>, holds the files.
    tops = list(Path(folder).iterdir())
    if len(tops) != 1 or not tops[0].is_dir():
        print(f"Configs repo {url} at {commit}: the tarball has no single top folder", flush=True)
        return None
    return tops[0]


def model_ids(folder):
    """The id of each model file in the configs repo checkout in ``folder``, sorted."""
    ids = []
    for path in (Path(folder) / "models").glob("*.yaml"):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        model = data.get("model") if isinstance(data, dict) else None
        if isinstance(model, dict) and isinstance(model.get("id"), str):
            ids.append(model["id"])
    return sorted(ids)


def parse_cost_line(line):
    """The model id, served model and agent cost on a cost line; None for any other line."""
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
    """The saved costs by model id; none when the file does not exist.
    Raises OSError or ValueError when it cannot be read."""
    if not path.exists():
        return {}
    costs = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(costs, dict):
        raise ValueError("it is not a JSON object")
    return costs


def save_cost(path, cost):
    """Saves ``cost``, a parsed cost line, as its model's latest cost. A file
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


def agent_cost(costs, model):
    """The saved agent cost of ``model``; None when there is none."""
    saved = costs.get(model)
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


def estimate_lines(models, repetitions, costs, judges, user_model, prices):
    """The batch's estimated cost of running ``models``, as lines to print. A
    model without a saved cost, or a judge or simulated user model without a
    price, is named and left out of the total."""
    agents = sum(agent_cost(costs, model) or 0.0 for model in models) * repetitions
    graders = [(model, JUDGE_TOKENS) for model in judges]
    graders.append((user_model, SIMULATED_USER_TOKENS))
    per_run = [(model, grading_cost(model, tokens, prices)) for model, tokens in graders]
    grading = sum(cost for _, cost in per_run if cost is not None) * len(models) * repetitions
    lines = [
        f"Estimated cost: ${agents + grading:.2f} "
        f"(agents ${agents:.2f}, judges and simulated user ${grading:.2f})"
    ]
    unknown = [model for model in models if agent_cost(costs, model) is None]
    if unknown:
        lines.append(f"Agent cost unknown, not counted: {', '.join(unknown)}")
    unpriced = list(dict.fromkeys(model for model, cost in per_run if cost is None))
    if unpriced:
        lines.append(f"No OpenRouter price, not counted: {', '.join(unpriced)}")
    return lines


def show_estimate(models, repetitions, judges, user_model):
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
    for line in estimate_lines(models, repetitions, costs, judges, user_model, prices):
        print(line, flush=True)


def secret_id(name):
    """``APP_OPENROUTER_API_KEY`` -> ``weather-skills-eval-app-openrouter-api-key``."""
    return SECRET_PREFIX + name.lower().replace("_", "-")


def read_secret(project, secret):
    """The latest version of ``secret`` in ``project``, read with the
    application default credentials: the machine's Google login, or the VM's
    service account on the VM."""
    # Imported here: only a batch with an empty secret reads Secret Manager.
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project}/secrets/{secret}/versions/latest"
    # Stripped: a secret added with echo ends in a newline no key holds.
    return client.access_secret_version(name=name).payload.data.decode("utf-8").strip()


def run_secrets(connection_keys):
    """The secrets every run needs: SECRETS and each variable
    ``connection_keys``, an EVAL_CONNECTION_KEYS value, names."""
    keys = (pair.rpartition("=")[2].strip() for pair in (connection_keys or "").split(","))
    return list(dict.fromkeys([*SECRETS, *(name for name in keys if name)]))


def load_secrets(names, project, dotenv):
    """Puts each secret ``names`` lists into this process's environment: its
    value from the environment, else from ``dotenv``, when one sets it, else its
    latest version from Secret Manager in ``project``. Returns ``names``; None,
    after printing why, when one cannot be read."""
    project = (project or "").strip()
    for name in names:
        value = os.environ.get(name) or dotenv.get(name)
        if value:
            os.environ[name] = value
            continue
        if not project:
            print(f"Secret {name} is empty and EVAL_SECRETS_PROJECT is not set", flush=True)
            return None
        try:
            os.environ[name] = read_secret(project, secret_id(name))
        except Exception as exc:
            print(
                f"Secret {secret_id(name)} could not be read from project {project}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return None
    return names


def image_labels(image):
    """The labels of ``image`` on the Docker host; None, after printing why,
    when Docker cannot read them."""
    inspected = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{json .Config.Labels}}", image],
        capture_output=True,
        text=True,
    )
    if inspected.returncode != 0:
        print(inspected.stderr, end="", flush=True)
        return None
    return json.loads(inspected.stdout) or {}


# The launch choices interactive mode offers.
VM_LAUNCH = "On a new VM"
LOCAL_LAUNCH = "On the current Docker context's host"
NO_LAUNCH = "Neither"


def run_interactive():
    """Interactive mode: asks for the batch, prints its launch commands and
    runs the one picked. Returns the exit code."""
    dotenv = ROOT / ".env"
    offered = judge_models(os.environ, dotenv)
    if not offered:
        print("EVAL_JUDGE_MODELS names no judge model; set it in the shell or .env", flush=True)
        return 2
    settings = prefilled(os.environ, dotenv, dict.fromkeys(SETTINGS, ""))
    unset = [name for name, value in settings.items() if not value]
    if unset:
        print(f"Not set in the shell or .env: {', '.join(unset)}", flush=True)
        return 2
    optional = prefilled(os.environ, dotenv, dict.fromkeys(OPTIONAL_SETTINGS, ""))
    # The configs repo checkout, only read for its model ids.
    folder = Path(tempfile.mkdtemp(prefix="run-eval-configs-"))
    try:
        answers = ask(dotenv, offered, folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)
    if isinstance(answers, int):
        return answers
    values = {name: answers[name] for name in ANSWERS}
    values.update(settings)
    values.update((name, value) for name, value in optional.items() if value)
    return launch(values, dotenv)


def launch(values, dotenv):
    """Prints the batch's two launch commands and runs the one picked."""
    secrets = [*run_secrets(values["EVAL_CONNECTION_KEYS"]), *BATCH_SECRETS]
    vm = vm_name(datetime.now(UTC))
    try:
        gcloud = vm_commands(vm, values)
    except ValueError as exc:
        print(f"No VM launch: {exc}", flush=True)
        return 2
    docker = docker_command(values, secrets)
    print("To run the batch on a new VM, which deletes itself when the batch ends:", flush=True)
    for command in gcloud:
        print(f"  {shlex.join(command)}", flush=True)
    print("To run it on the current Docker context's host, with each secret set:", flush=True)
    print(f"  {shlex.join(docker)}", flush=True)
    choice = questionary.select(
        "Launch:", choices=[VM_LAUNCH, LOCAL_LAUNCH, NO_LAUNCH]
    ).unsafe_ask()
    if choice == VM_LAUNCH:
        for index, command in enumerate(gcloud):
            done = subprocess.run(command)
            if done.returncode != 0:
                print(f"Failed, exit {done.returncode}: {shlex.join(command)}", flush=True)
                if index > 0:
                    # The VM exists and waits for metadata that will not come.
                    print(f"VM {vm} was created; delete it with:", flush=True)
                    print(f"  gcloud compute instances delete {vm} --zone {VM_ZONE}", flush=True)
                return done.returncode
        print(f"VM {vm} runs the batch and deletes itself when the batch ends.", flush=True)
        print(f"Stop it with: gcloud compute instances delete {vm} --zone {VM_ZONE}", flush=True)
        return 0
    if choice == LOCAL_LAUNCH:
        if batch_running():
            print(f"A batch is running: container {BATCH_CONTAINER}", flush=True)
            print(f"Stop it with: docker stop {BATCH_CONTAINER}", flush=True)
            return 1
        # Each secret is passed by name, so its value is never on a command line.
        if load_secrets(secrets, values["EVAL_SECRETS_PROJECT"], _dotenv(dotenv)) is None:
            return 1
        done = subprocess.run(docker)
        if done.returncode != 0:
            print(f"Failed, exit {done.returncode}: docker run", flush=True)
            return done.returncode
        print(f"Batch started in container {BATCH_CONTAINER}", flush=True)
        print(f"Follow it with: docker logs -f {BATCH_CONTAINER}", flush=True)
        print(f"Stop it with: docker stop {BATCH_CONTAINER}", flush=True)
    return 0


def ask(dotenv, offered, folder):
    """Asks for the batch, with the configs repo checked out in ``folder``,
    and prints its estimated cost. Returns the answers by batch value name,
    or an exit code."""
    repo = (
        questionary.text(
            "EVAL_CONFIGS_REPO:",
            default=prefilled(os.environ, dotenv, CONFIGS_REPO)["EVAL_CONFIGS_REPO"],
        )
        .unsafe_ask()
        .strip()
    )
    # The batch downloads the commit from GitHub, so only that form is accepted.
    if github_repo(repo) is None:
        return 2
    ref = questionary.text("Configs ref:", default="main").unsafe_ask().strip()
    configs_commit = fetch_configs(repo, ref, folder)
    if configs_commit is None:
        return 1
    print(f"Configs {repo} at {ref} is commit {configs_commit}", flush=True)
    models = questionary.checkbox("Models to run:", choices=model_ids(folder)).unsafe_ask()
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
    repetitions = int(
        questionary.text("Repetitions:", default="1", validate=whole_number_valid)
        .unsafe_ask()
        .strip()
    )
    parallel = int(
        questionary.text(
            "Runs at once:",
            default=prefilled(os.environ, dotenv, PARALLEL_RUNS)["EVAL_PARALLEL_RUNS"],
            validate=whole_number_valid,
        )
        .unsafe_ask()
        .strip()
    )
    show_estimate(models, repetitions, judges, values["EVAL_SIMULATED_USER_MODEL"])
    pull = questionary.confirm("Pull the app image?", default=True).unsafe_ask()
    image_tag = (
        questionary.text(
            "EVAL_IMAGE_TAG:",
            default=prefilled(os.environ, dotenv, IMAGE_TAG)["EVAL_IMAGE_TAG"],
            validate=image_tag_valid,
        )
        .unsafe_ask()
        .strip()
    )
    return {
        **values,
        "EVAL_CONFIGS_REPO": repo,
        "EVAL_CONFIGS_COMMIT": configs_commit,
        "EVAL_MODELS": ",".join(models),
        "EVAL_REPETITIONS": str(repetitions),
        "EVAL_PARALLEL_RUNS": str(parallel),
        "EVAL_PULL_APP_IMAGE": "true" if pull else "false",
        "EVAL_IMAGE_TAG": image_tag,
    }


def _whole_number(text):
    text = text.strip()
    if whole_number_valid(text) is not True:
        raise argparse.ArgumentTypeError("a whole number, at least 1")
    return text


def _model_list(text):
    models = dict.fromkeys(model.strip() for model in text.split(",") if model.strip())
    if not models:
        raise argparse.ArgumentTypeError("at least one model id, comma-separated")
    return ",".join(models)


def _non_empty(text):
    if not text:
        raise argparse.ArgumentTypeError("must not be empty")
    return text


def _commit(text):
    if len(text) != 40 or any(c not in "0123456789abcdef" for c in text):
        raise argparse.ArgumentTypeError("a full 40-character commit SHA")
    return text


# Checks of the required batch values. Every other one must not be empty,
# but EVAL_CONCURRENCY, which is empty for the harness's default.
FLAG_OPTIONS = {
    "EVAL_CONFIGS_COMMIT": {"type": _commit},
    "EVAL_MODELS": {"type": _model_list},
    "EVAL_CONCURRENCY": {},
    "EVAL_REPETITIONS": {"type": _whole_number},
    "EVAL_PARALLEL_RUNS": {"type": _whole_number},
    "EVAL_PULL_APP_IMAGE": {"choices": ("true", "false")},
}


def batch_parser():
    """The non-interactive mode's flags: every batch value, the optional
    settings not required. A missing or invalid flag exits 2, naming it."""
    parser = argparse.ArgumentParser(
        prog="run-eval.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    for name in (*ANSWERS, *SETTINGS):
        options = FLAG_OPTIONS.get(name, {"type": _non_empty})
        parser.add_argument(flag(name), dest=name, required=True, **options)
    for name in OPTIONAL_SETTINGS:
        parser.add_argument(flag(name), dest=name)
    return parser


def parse_batch(argv):
    """The batch values ``argv`` gives, by name; an optional setting only when
    given and not empty, as interactive mode leaves out an empty one."""
    parsed = vars(batch_parser().parse_args(argv))
    return {
        name: value
        for name, value in parsed.items()
        if value is not None and (value or name not in OPTIONAL_SETTINGS)
    }


def run_noninteractive(values):
    """Non-interactive mode: runs the batch ``values`` describes. SIGTERM, as
    from docker stop, stops it with the results so far."""
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    folder = Path(tempfile.mkdtemp(prefix="run-eval-configs-"))
    results = []
    try:
        return run_batch(values, folder, results)
    except KeyboardInterrupt:
        print("\nStopped. Results so far:", flush=True)
        for name, returncode in results:
            print(f"{name}: exit {returncode}", flush=True)
        return 143
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        configs_archive(folder).unlink(missing_ok=True)


def run_batch(values, folder, results):
    """Reads the secrets, downloads the configs repo files into ``folder``,
    pulls the app image when asked to, reads its labels, and runs each model
    once per round, as many rounds as the repetitions, with ``docker compose
    run`` of the eval service in the eval image this batch runs."""
    # Each secret is passed by name, so its value is never on a command line.
    secrets = run_secrets(values["EVAL_CONNECTION_KEYS"])
    if load_secrets([*secrets, *BATCH_SECRETS], values["EVAL_SECRETS_PROJECT"], {}) is None:
        return 1
    files = download_configs(
        values["EVAL_CONFIGS_REPO"],
        values["EVAL_CONFIGS_COMMIT"],
        os.environ["GIT_READ_TOKEN"],
        folder,
    )
    if files is None:
        return 1
    if values["EVAL_PULL_APP_IMAGE"] == "true":
        pulled = subprocess.run(["docker", "pull", values["APP_IMAGE"]])
        if pulled.returncode != 0:
            return pulled.returncode
    labels = image_labels(values["APP_IMAGE"])
    if labels is None:
        return 1

    archive = configs_archive(folder)
    # Readable only by this user, like the files; without a .git, which the
    # harness does not read.
    with (
        open(os.open(archive, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as out,
        tarfile.open(fileobj=out, mode="w") as tar,
    ):
        tar.add(
            files,
            arcname=".",
            filter=lambda member: None if member.name == "./.git" else member,
        )

    # compose.yaml's eval service runs this image, already on the Docker host.
    os.environ["EVAL_IMAGE"] = f"{EVAL_IMAGE}:{values['EVAL_IMAGE_TAG']}"
    batch_id = new_batch()
    print(f"Batch {batch_id}", flush=True)
    command = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-T",
        "--entrypoint",
        "sh",
        *env_args(
            {
                **{name: value for name, value in values.items() if name not in BATCH_ONLY},
                "EVAL_BATCH": batch_id,
                "EVAL_CONFIGS_DIR": CONFIGS_DIR,
                # Each run uses the app image whose labels were read above.
                "EVAL_PULL_APP_IMAGE": "false",
                "EVAL_APP_IMAGE_REVISION": labels.get(REVISION_LABEL, ""),
                "EVAL_APP_IMAGE_PR_NUMBER": labels.get(PR_NUMBER_LABEL, ""),
            }
        ),
        *(arg for name in secrets for arg in ("-e", name)),
    ]
    # In rounds, each picked model once per round.
    queue = [
        (model, repetition)
        for repetition in range(1, int(values["EVAL_REPETITIONS"]) + 1)
        for model in values["EVAL_MODELS"].split(",")
    ]
    run_queue(command, queue, archive, int(values["EVAL_PARALLEL_RUNS"]), results)
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


def run_queue(command, queue, archive, parallel, results):
    """Runs ``command`` once per queued run, a (model, round) pair, with
    ``-e EVAL_MODEL=<model> eval -c UNPACK_AND_RUN`` added and the tar file
    ``archive`` on stdin, starting the runs in queue order with at most
    ``parallel`` going at once: the next starts when one ends.

    Each run's stdout is passed on line by line. With ``parallel`` above 1, its
    stderr is read and passed on too, and every line of both is prefixed with
    ``[<model> run <round>] ``; with 1, stderr is not read and reaches this
    process's stderr as the run prints it. When a run ends, its model's cost
    line is saved, and its exit code is added to ``results`` and printed. A run
    whose output or exit the script fails to handle is still added, as failed,
    after a line saying what failed.

    On SIGTERM, or when a run cannot be started, no further run starts; on
    SIGTERM each started run is sent SIGTERM; every run started is read to the
    end and waited for, and is not added to ``results``; then the exception is
    raised."""
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
            # Only the started model's own cost line.
            if cost and cost["config_name"] == name:
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

        def stop(self):
            try:
                self.args[0].send_signal(signal.SIGTERM)
            except OSError:
                # It has already ended.
                pass

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
            # Opened per run, so runs at once never share a file offset.
            with open(archive, "rb") as configs:
                started.append(
                    Started(
                        subprocess.Popen(
                            [*command, "-e", f"EVAL_MODEL={name}", "eval", "-c", UNPACK_AND_RUN],
                            cwd=ROOT,
                            stdin=configs,
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
    except BaseException as exc:
        # SIGTERM, or a run that could not be started. Each started run is
        # finished here when its thread never began, its output read to the
        # end so it never blocks on a full pipe.
        with lock:
            interrupted.set()
        if isinstance(exc, KeyboardInterrupt):
            # Only this process got SIGTERM: each docker compose run gets it
            # too, so each removes its run's containers.
            for run in started:
                run.stop()
        for run in started:
            run.finish_once()
            run.done.wait()
        raise


def main(argv=()):
    """Non-interactive mode with arguments, interactive mode without."""
    argv = list(argv)
    if argv:
        return run_noninteractive(parse_batch(argv))
    try:
        return run_interactive()
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


def cost_line(model, cost, served_model="provider/model"):
    """A cost line as the harness prints it."""
    data = {"config_name": model, "served_model": served_model, "agent_cost_usd": cost}
    return COST_LINE + json.dumps(data)


# USD per token, (input, output), of the models the tests price.
PRICES = {
    JUDGES[0].removeprefix("openrouter/"): (1e-6, 2e-6),
    JUDGES[1].removeprefix("openrouter/"): (1e-6, 2e-6),
    "test/user": (1e-6, 1e-6),
}


CONFIGS_COMMIT = "c" * 40
REPO_URL = "https://github.com/rhiza-research/weather-skills-configs"
LABELS = {REVISION_LABEL: "8dd46ed4", PR_NUMBER_LABEL: "42"}


class FakeCommands:
    """``subprocess.run``: records each command and answers it from
    ``outputs``, (command prefix, exit code, stdout) entries matched in order.
    ``git checkout`` of FETCH_HEAD writes model files m-b and m-a; otherwise
    ``git rev-parse FETCH_HEAD`` answers CONFIGS_COMMIT, ``docker image
    inspect`` answers LABELS, and ``docker container inspect`` finds no container."""

    def __init__(self, outputs=()):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if "checkout" in command and command[-1] == "FETCH_HEAD":
            models = Path(command[2]) / "models"
            models.mkdir()
            for model in ("m-b", "m-a"):
                data = f"schema_version: 1\nmodel:\n  id: {model}\n"
                (models / f"{model}.yaml").write_text(data)
        for prefix, code, stdout in self.outputs:
            if command[: len(prefix)] == prefix:
                return subprocess.CompletedProcess(command, code, stdout=stdout, stderr="")
        if command[-1] == "FETCH_HEAD" and "rev-parse" in command:
            return subprocess.CompletedProcess(command, 0, stdout=f"{CONFIGS_COMMIT}\n", stderr="")
        if command[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, stdout=json.dumps(LABELS), stderr="")
        if command[:3] == ["docker", "container", "inspect"]:
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    def commands(self):
        return [command for command, _ in self.calls]


CONNECTION_KEYS = "https://openrouter.ai/api/v1=APP_OPENROUTER_API_KEY"
# The settings of the tests' .env, by name.
TEST_SETTINGS = {
    "EVAL_SECRETS_PROJECT": "test-project",
    "LANGFUSE_BASE_URL": "https://us.cloud.langfuse.com",
    "LANGFUSE_TRACING_ENVIRONMENT": "eval",
    "EVAL_CONNECTION_KEYS": CONNECTION_KEYS,
    "EVAL_DATASET": "weather-eval-tasks",
    "EVAL_ADMIN_EMAIL": "admin@example.com",
}
DOTENV = (
    f"EVAL_JUDGE_MODELS={JUDGES[0]}\nEVAL_SIMULATED_USER_MODEL=openrouter/test/user\n"
    f"EVAL_CONFIGS_REPO={REPO_URL}\nEVAL_MAX_TURNS=5\nEVAL_MODEL_RETRIES=\n"
    + "".join(f"{name}={value}\n" for name, value in TEST_SETTINGS.items())
)
# The batch values of a launch with DOTENV and EVAL_PARALLEL_RUNS=2 that takes
# every default, in order.
BATCH = {
    "EVAL_CONFIGS_REPO": REPO_URL,
    "EVAL_CONFIGS_COMMIT": CONFIGS_COMMIT,
    "EVAL_MODELS": "m-a,m-b",
    "EVAL_JUDGE_MODELS": JUDGES[0],
    "EVAL_SIMULATED_USER_MODEL": "openrouter/test/user",
    "APP_IMAGE": DEFAULTS["APP_IMAGE"],
    "RUNNER_IMAGE": DEFAULTS["RUNNER_IMAGE"],
    "EVAL_CONCURRENCY": "",
    "EVAL_REPETITIONS": "1",
    "EVAL_PARALLEL_RUNS": "2",
    "EVAL_PULL_APP_IMAGE": "true",
    "EVAL_IMAGE_TAG": "main",
    **TEST_SETTINGS,
    # Set in DOTENV; EVAL_MODEL_RETRIES is empty there, so it is left out.
    "EVAL_MAX_TURNS": "5",
}
# The secrets a batch of BATCH needs.
BATCH_SECRET_NAMES = [*SECRETS, "APP_OPENROUTER_API_KEY", "GIT_READ_TOKEN"]


def launch_setup(tmp_path, monkeypatch, dotenv, outputs=(), choice=NO_LAUNCH):
    """ROOT at ``tmp_path`` with ``dotenv`` as .env; the questions take their
    defaults, and every model, and the launch is ``choice``; the VM is named
    weather-skills-eval-test. Returns the fake commands and the questions asked."""
    (tmp_path / ".env").write_text(dotenv)
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    monkeypatch.setitem(globals(), "openrouter_prices", lambda: PRICES)
    monkeypatch.setitem(globals(), "vm_name", lambda now: "weather-skills-eval-test")
    for name in [
        *DEFAULTS,
        *PARALLEL_RUNS,
        *IMAGE_TAG,
        *CONFIGS_REPO,
        *SETTINGS,
        *OPTIONAL_SETTINGS,
        "EVAL_JUDGE_MODELS",
    ]:
        monkeypatch.delenv(name, raising=False)
    asked = []

    def select(message, choices):
        asked.append(message)
        assert choices == [VM_LAUNCH, LOCAL_LAUNCH, NO_LAUNCH]
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

    def mkdtemp(prefix):
        (tmp_path / "configs").mkdir()
        return str(tmp_path / "configs")

    monkeypatch.setattr(tempfile, "mkdtemp", mkdtemp)
    commands = FakeCommands(outputs)
    monkeypatch.setattr(subprocess, "run", commands)
    return commands, asked


def no_secret_read(project, secret):
    raise AssertionError(f"{secret} read from Secret Manager")


def test_the_vm_launch_prints_every_command_and_runs_the_two_gcloud_commands(
    tmp_path, monkeypatch, capsys
):
    commands, asked = launch_setup(
        tmp_path, monkeypatch, DOTENV + "EVAL_PARALLEL_RUNS=2\n", choice=VM_LAUNCH
    )
    (tmp_path / COSTS_FILE).write_text(json.dumps({"m-a": {"agent_cost_usd": 1.5}}))
    # The VM reads the secrets itself.
    monkeypatch.setitem(globals(), "read_secret", no_secret_read)

    assert main() == 0

    init, fetch, checkout, fetched, create, add = commands.commands()
    # The ref, main by default, resolved to one commit with this machine's git.
    folder = init[-1]
    assert init == ["git", "init", "-q", folder]
    assert fetch == ["git", "-C", folder, "fetch", "-q", "--depth", "1", REPO_URL, "main"]
    assert checkout == ["git", "-C", folder, "checkout", "-q", "--detach", "FETCH_HEAD"]
    assert fetched == ["git", "-C", folder, "rev-parse", "FETCH_HEAD"]
    # The checkout is only read for its model ids.
    assert not Path(folder).exists()
    assert asked[-3:] == ["Pull the app image?", "EVAL_IMAGE_TAG:", "Launch:"]
    vm = "weather-skills-eval-test"
    assert create == [
        *("gcloud", "compute", "instances", "create", vm, "--zone", "us-central1-a"),
        *("--source-instance-template", "weather-skills-eval"),
    ]
    assert add[:-1] == [
        *("gcloud", "compute", "instances", "add-metadata", vm, "--zone", "us-central1-a"),
        "--metadata",
    ]
    # Every batch value under its metadata key, split on a character no value holds.
    assert add[-1].startswith("^;^")
    assert add[-1].removeprefix("^;^").split(";") == [
        f"{metadata_key(name)}={value}" for name, value in BATCH.items()
    ]
    assert f"eval-connection-keys={CONNECTION_KEYS}" in add[-1]
    out = capsys.readouterr().out
    assert "Estimated cost:" in out
    for command in (create, add, docker_command(BATCH, BATCH_SECRET_NAMES)):
        assert f"  {shlex.join(command)}\n" in out
    assert f"gcloud compute instances delete {vm} --zone us-central1-a" in out


def test_a_failed_gcloud_command_stops_the_vm_launch(tmp_path, monkeypatch, capsys):
    create = ["gcloud", "compute", "instances", "create"]
    commands, _ = launch_setup(
        tmp_path, monkeypatch, DOTENV, outputs=[(create, 1, "")], choice=VM_LAUNCH
    )

    assert main() == 1

    assert commands.commands()[-1][:4] == create
    assert "Failed, exit 1: gcloud compute instances create" in capsys.readouterr().out


def test_without_judge_models_the_script_stops_before_asking_more(tmp_path, monkeypatch, capsys):
    commands, asked = launch_setup(tmp_path, monkeypatch, "")

    assert main() == 2

    assert asked == []
    assert capsys.readouterr().out == (
        "EVAL_JUDGE_MODELS names no judge model; set it in the shell or .env\n"
    )


def test_a_running_batch_is_named_with_its_stop_command_and_nothing_starts(
    tmp_path, monkeypatch, capsys
):
    commands, _ = launch_setup(
        tmp_path,
        monkeypatch,
        DOTENV,
        outputs=[(["docker", "container", "inspect"], 0, "[]")],
        choice=LOCAL_LAUNCH,
    )
    monkeypatch.setitem(globals(), "read_secret", no_secret_read)

    assert main() == 1

    assert commands.commands()[-1] == ["docker", "container", "inspect", BATCH_CONTAINER]
    assert capsys.readouterr().out.endswith(
        "A batch is running: container weather-skills-eval-batch\n"
        "Stop it with: docker stop weather-skills-eval-batch\n"
    )


def test_a_local_launch_reads_empty_secrets_and_passes_each_by_name_only(
    tmp_path, monkeypatch, capsys
):
    dotenv = (
        DOTENV + "EVAL_PARALLEL_RUNS=2\nLANGFUSE_SECRET_KEY=\nLANGFUSE_PUBLIC_KEY=from-dotenv\n"
    )
    commands, _ = launch_setup(tmp_path, monkeypatch, dotenv, choice=LOCAL_LAUNCH)
    # Set in .env and in the shell; every other secret is empty, so it is read.
    given = {"LANGFUSE_PUBLIC_KEY": "from-dotenv", "OPENROUTER_API_KEY": "from-shell"}
    for name in BATCH_SECRET_NAMES:
        # Restored after the test; the launch sets each.
        monkeypatch.setenv(name, "from-shell" if name == "OPENROUTER_API_KEY" else "")
    read = []

    def reader(project, secret):
        read.append((project, secret))
        return f"value-of-{secret}"

    monkeypatch.setitem(globals(), "read_secret", reader)

    assert main() == 0

    inspect, docker = commands.commands()[-2:]
    assert inspect == ["docker", "container", "inspect", BATCH_CONTAINER]
    assert docker == docker_command(BATCH, BATCH_SECRET_NAMES)
    assert docker[:13] == [
        *("docker", "run", "--detach", "--rm", "--pull", "always"),
        *("--name", "weather-skills-eval-batch", "--stop-timeout", "120"),
        *("--volume", "/var/run/docker.sock:/var/run/docker.sock", "-e"),
    ]
    assert docker[-len(BATCH) - 3 :] == [
        "ghcr.io/rhiza-research/weather-skills-eval:main",
        "/opt/run-eval/bin/python",
        "/app/scripts/run-eval.py",
        *(f"--{metadata_key(name)}={value}" for name, value in BATCH.items()),
    ]
    empty = [name for name in BATCH_SECRET_NAMES if name not in given]
    assert read == [("test-project", secret_id(name)) for name in empty]
    for name in BATCH_SECRET_NAMES:
        assert os.environ[name] == given.get(name, f"value-of-{secret_id(name)}")
        assert docker[docker.index(name) - 1] == "-e"
    assert not any("value-of-" in arg or "from-" in arg for arg in docker)
    assert "docker stop weather-skills-eval-batch" in capsys.readouterr().out


def test_a_missing_flag_exits_2_naming_it_and_nothing_runs(monkeypatch, capsys):
    commands = FakeCommands()
    monkeypatch.setattr(subprocess, "run", commands)
    monkeypatch.setattr(subprocess, "Popen", commands)

    try:
        main(batch_flags({name: value for name, value in BATCH.items() if name != "EVAL_MODELS"}))
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit")

    assert "the following arguments are required: --eval-models" in capsys.readouterr().err
    assert commands.calls == []


def test_an_invalid_flag_value_exits_2_naming_it_and_nothing_runs(monkeypatch, capsys):
    commands = FakeCommands()
    monkeypatch.setattr(subprocess, "run", commands)
    monkeypatch.setattr(subprocess, "Popen", commands)
    # Each case, with pytest not a dependency of the script.
    for name, value in (
        ("EVAL_PULL_APP_IMAGE", "yes"),
        ("EVAL_REPETITIONS", "0"),
        ("EVAL_MODELS", ","),
        ("EVAL_IMAGE_TAG", ""),
        ("EVAL_CONFIGS_COMMIT", "main"),
    ):
        try:
            main(batch_flags({**BATCH, name: value}))
        except SystemExit as exc:
            assert exc.code == 2, name
        else:
            raise AssertionError(f"expected SystemExit for {name}")
        assert f"argument {flag(name)}:" in capsys.readouterr().err, name
    assert commands.calls == []
    # Spaces around a whole number are ignored, and an empty optional setting is left out.
    parsed = parse_batch(batch_flags({**BATCH, "EVAL_REPETITIONS": " 2", "EVAL_MODEL_RETRIES": ""}))
    assert parsed["EVAL_REPETITIONS"] == "2"
    assert "EVAL_MODEL_RETRIES" not in parsed


def test_an_unset_setting_stops_interactive_mode_before_asking(tmp_path, monkeypatch, capsys):
    dotenv = DOTENV.replace("EVAL_DATASET=weather-eval-tasks\n", "")
    commands, asked = launch_setup(tmp_path, monkeypatch, dotenv)

    assert main() == 2

    assert asked == []
    assert commands.calls == []
    assert capsys.readouterr().out == "Not set in the shell or .env: EVAL_DATASET\n"


def test_the_metadata_delimiter_is_one_no_value_holds():
    values = {"EVAL_DATASET": "a;b", "EVAL_JUDGE_MODELS": "x,y"}
    _, add = vm_commands("weather-skills-eval-test", values)
    assert add[-1] == "^|^eval-dataset=a;b|eval-judge-models=x,y"


def test_the_batch_settings_are_the_non_secret_settings_of_env_example():
    text = (Path(__file__).resolve().parent.parent / ".env.example").read_text()
    uncommented = dotenv_values(stream=io.StringIO(text), interpolate=False)
    secrets = {*run_secrets(uncommented["EVAL_CONNECTION_KEYS"]), *BATCH_SECRETS}
    names = {name for name in uncommented if name not in secrets}
    assert set(SETTINGS) <= names <= {*ANSWERS, *SETTINGS}
    commented = {
        line[2:].partition("=")[0]
        for line in text.splitlines()
        if line.startswith("# ") and "=" in line and line[2:].partition("=")[0].isidentifier()
    }
    # EVAL_DOCKER_NETWORK is set by compose.yaml, EVAL_HARNESS_COMMIT by the image.
    assert commented == {
        *OPTIONAL_SETTINGS,
        "EVAL_PARALLEL_RUNS",
        "EVAL_IMAGE_TAG",
        "EVAL_DOCKER_NETWORK",
        "EVAL_HARNESS_COMMIT",
    }


def noninteractive_setup(tmp_path, monkeypatch, **values):
    """ROOT at ``tmp_path``, each secret set in the environment and the configs
    repo download faked; returns the fake commands, the flags of BATCH with
    models a and b, one run at once, and ``values``, and the downloads made."""
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    monkeypatch.setitem(globals(), "new_batch", lambda: "2026-10-22T20:00:00Z")
    monkeypatch.setitem(globals(), "read_secret", no_secret_read)
    # The SIGTERM handler stays out of the test process.
    monkeypatch.setattr(signal, "signal", lambda signum, handler: None)
    # Restored after the test; the batch sets EVAL_IMAGE.
    monkeypatch.setenv("EVAL_IMAGE", "")
    for name in BATCH_SECRET_NAMES:
        monkeypatch.setenv(name, f"test-{name}")
    downloads = []

    def download(url, commit, token, folder):
        downloads.append((url, commit, token, folder))
        files = Path(folder) / "rhiza-research-weather-skills-configs-ccccccc"
        files.mkdir()
        (files / "settings.yaml").write_text("schema_version: 1\n")
        return files

    monkeypatch.setitem(globals(), "download_configs", download)
    commands = FakeCommands()
    monkeypatch.setattr(subprocess, "run", commands)
    values = {**BATCH, "EVAL_MODELS": "a,b", "EVAL_PARALLEL_RUNS": "1", **values}
    return commands, batch_flags(values), downloads


def model_of(command):
    """The model a run's command names with EVAL_MODEL."""
    [model] = [arg.removeprefix("EVAL_MODEL=") for arg in command if arg.startswith("EVAL_MODEL=")]
    return model


def test_the_batch_runs_each_model_once_per_round_and_saves_its_cost(tmp_path, monkeypatch, capsys):
    commands, argv, downloads = noninteractive_setup(tmp_path, monkeypatch, EVAL_REPETITIONS="2")
    # A saved cost for a, from an earlier run.
    (tmp_path / COSTS_FILE).write_text(
        json.dumps({"a": {"served_model": "provider/model", "agent_cost_usd": 1.5}})
    )
    # b's run also prints a cost line of another model, which is not saved.
    b_output = f"Goal reached on 1 of 1 tasks\n{cost_line('a', 9.0)}\n{cost_line('b', 0.75)}\n"
    runs = []

    def popen(command, **kwargs):
        # compose runs the eval service in the batch's own eval image.
        assert os.environ["EVAL_IMAGE"] == "ghcr.io/rhiza-research/weather-skills-eval:main"
        # Each run gets the configs repo files as a tar on stdin.
        with tarfile.open(fileobj=kwargs["stdin"]) as tar:
            files = [member for member in tar.getmembers() if member.isfile()]
            kwargs["stdin"] = {member.name: tar.extractfile(member).read() for member in files}
        runs.append((command, kwargs))
        failed = model_of(command) == "a"
        return FakeRun("" if failed else b_output, code=2 if failed else 0)

    monkeypatch.setattr(subprocess, "Popen", popen)

    assert main(argv) == 1

    # The configs repo files at the commit, read with the read token.
    [(url, commit, token, folder)] = downloads
    assert (url, commit, token) == (REPO_URL, CONFIGS_COMMIT, "test-GIT_READ_TOKEN")
    pull, inspect = commands.commands()
    assert pull == ["docker", "pull", DEFAULTS["APP_IMAGE"]]
    assert inspect[-1] == DEFAULTS["APP_IMAGE"]
    given = {name: value for name, value in BATCH.items() if name not in BATCH_ONLY}
    values = {
        **given,
        "EVAL_BATCH": "2026-10-22T20:00:00Z",
        "EVAL_CONFIGS_DIR": CONFIGS_DIR,
        "EVAL_PULL_APP_IMAGE": "false",
        "EVAL_APP_IMAGE_REVISION": "8dd46ed4",
        "EVAL_APP_IMAGE_PR_NUMBER": "42",
    }
    # No host path is mounted into a run, and the read token is not passed.
    command = ["docker", "compose", "run", "--rm", "-T", "--entrypoint", "sh"]
    command += env_args(values)
    command += [arg for name in [*SECRETS, "APP_OPENROUTER_API_KEY"] for arg in ("-e", name)]
    # Two rounds of every model.
    assert [run for run, _ in runs] == [
        [*command, "-e", f"EVAL_MODEL={model}", "eval", "-c", UNPACK_AND_RUN]
        for model in ("a", "b", "a", "b")
    ]
    assert [kwargs["stdin"] for _, kwargs in runs] == [
        {"./settings.yaml": b"schema_version: 1\n"}
    ] * 4
    assert "a run 2: exit 2\n" in capsys.readouterr().out
    # a's failed runs printed no cost line, and b's run's line for a is not a's,
    # so a's saved cost stays; b's run's cost is saved.
    assert json.loads((tmp_path / COSTS_FILE).read_text()) == {
        "a": {"served_model": "provider/model", "agent_cost_usd": 1.5},
        "b": {"served_model": "provider/model", "agent_cost_usd": 0.75},
    }
    # The configs repo files and their tar are gone once the batch ends.
    assert not Path(folder).exists()
    assert not configs_archive(folder).exists()


def test_sigterm_stops_the_batch_with_the_results_so_far(tmp_path, monkeypatch, capsys):
    install, previous = signal.signal, signal.getsignal(signal.SIGTERM)
    _, argv, downloads = noninteractive_setup(tmp_path, monkeypatch)
    # The real handler, this once.
    monkeypatch.setattr(signal, "signal", install)

    def popen(command, **kwargs):
        if model_of(command) == "b":
            # docker stop, while b starts.
            os.kill(os.getpid(), signal.SIGTERM)
        return FakeRun("")

    monkeypatch.setattr(subprocess, "Popen", popen)

    try:
        assert main(argv) == 143
    finally:
        install(signal.SIGTERM, previous)

    assert capsys.readouterr().out.endswith("\nStopped. Results so far:\na run 1: exit 0\n")
    assert not Path(downloads[0][3]).exists()


def test_the_configs_repo_files_are_downloaded_at_the_commit_with_the_token(
    tmp_path, monkeypatch, capsys
):
    data = b"schema_version: 1\n"
    tarball = io.BytesIO()
    with tarfile.open(fileobj=tarball, mode="w:gz") as tar:
        member = tarfile.TarInfo("rhiza-research-weather-skills-configs-ccccccc/settings.yaml")
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))
    tarball.seek(0)
    requests = []

    def fake_open(request):
        requests.append(request)
        return tarball

    monkeypatch.setitem(globals(), "_open", fake_open)

    files = download_configs(f"{REPO_URL}.git", CONFIGS_COMMIT, "t0ken", tmp_path / "configs")

    assert (files / "settings.yaml").read_bytes() == data
    [request] = requests
    assert request.full_url == (
        f"https://api.github.com/repos/rhiza-research/weather-skills-configs/tarball/{CONFIGS_COMMIT}"
    )
    assert download_configs("https://example.com/a/b", CONFIGS_COMMIT, "t0ken", tmp_path) is None
    assert "is not https://github.com/<owner>/<repo>" in capsys.readouterr().out
    assert len(requests) == 1


def test_the_read_token_is_an_unredirected_header(tmp_path, monkeypatch):
    requests = []

    def fake_open(request):
        requests.append(request)
        raise OSError("offline")

    monkeypatch.setitem(globals(), "_open", fake_open)

    assert download_configs(REPO_URL, CONFIGS_COMMIT, "t0ken", tmp_path) is None

    [request] = requests
    # urllib copies only request.headers to a redirect's request.
    assert request.unredirected_hdrs == {"Authorization": "Bearer t0ken"}
    assert "Authorization" not in request.headers


def vm_entry():
    """scripts/vm-entry.py, loaded as a module."""
    path = Path(__file__).resolve().parent / "vm-entry.py"
    spec = importlib.util.spec_from_file_location("vm_entry", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_vm_runs_its_metadata_s_batch_and_is_deleted_whatever_the_exit(monkeypatch, capsys):
    entry = vm_entry()
    # cloud-init's own key is not a batch value.
    attributes = {"user-data": "#cloud-config", "eval-image-tag": "main", "app-image": "a:1"}
    monkeypatch.setattr(entry, "metadata", lambda path: json.dumps(attributes))
    runs, deleted = [], []
    # No main: vm-entry runs non-interactive mode directly, never interactive mode.
    batch = types.SimpleNamespace(metadata_keys=metadata_keys, parse_batch=lambda argv: argv)
    monkeypatch.setattr(entry, "load_run_eval", lambda: batch)
    monkeypatch.setattr(entry, "delete_instance", lambda: deleted.append(True))

    def fails(values):
        runs.append(values)
        return 1

    batch.run_noninteractive = fails
    assert entry.main() == 1
    assert runs == [["--app-image=a:1", "--eval-image-tag=main"]]
    # A missing value: parsing exits 2.
    batch.parse_batch = lambda argv: sys.exit(2)
    assert entry.main() == 2

    def raises(values):
        raise RuntimeError("broken")

    def forbidden():
        raise OSError("forbidden")

    batch.parse_batch = lambda argv: argv
    batch.run_noninteractive = raises
    monkeypatch.setattr(entry, "delete_instance", forbidden)
    assert entry.main() == 1

    assert deleted == [True, True]
    err = capsys.readouterr().err
    assert "Batch failed in vm-entry.py: RuntimeError: broken\n" in err
    assert "VM not deleted: OSError: forbidden\n" in err


def test_the_vm_deletes_itself_through_the_compute_api(monkeypatch):
    entry = vm_entry()
    answers = {
        "project/project-id": "test-project",
        "instance/zone": "projects/123/zones/us-central1-a",
        "instance/name": "weather-skills-eval-test",
        "instance/service-accounts/default/token": json.dumps({"access_token": "t0ken"}),
    }
    monkeypatch.setattr(entry, "metadata", lambda path: answers[path])
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        return io.BytesIO(b"{}")

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)

    entry.delete_instance()

    [request] = requests
    assert request.get_method() == "DELETE"
    assert request.full_url == (
        "https://compute.googleapis.com/compute/v1/projects/test-project"
        "/zones/us-central1-a/instances/weather-skills-eval-test"
    )
    assert request.get_header("Authorization") == "Bearer t0ken"


def test_read_secret_reads_the_latest_version_and_strips_it(monkeypatch):
    from google.cloud import secretmanager

    names = []

    class Client:
        def access_secret_version(self, name):
            names.append(name)
            return types.SimpleNamespace(payload=types.SimpleNamespace(data=b" s3cret\n"))

    monkeypatch.setattr(secretmanager, "SecretManagerServiceClient", Client)

    assert read_secret("test-project", "weather-skills-eval-x") == "s3cret"
    assert names == ["projects/test-project/secrets/weather-skills-eval-x/versions/latest"]


class FakeRun:
    """A run's process: prints ``out`` on stdout and ``err`` on stderr, and exits
    with ``code``. A held run's stdout ends only once ``release`` is set, or
    once it gets a signal; ``on_wait`` is called when the run is waited for."""

    def __init__(self, out, err="", code=0, hold=False, on_wait=None):
        self.release = threading.Event()
        if not hold:
            self.release.set()
        self.stdout = self._stdout(out)
        self.stderr = io.StringIO(err)
        self.code = code
        self.on_wait = on_wait
        self.waited = False
        self.signals = []

    def _stdout(self, out):
        yield from io.StringIO(out)
        self.release.wait(10)

    def send_signal(self, signum):
        self.signals.append(signum)
        self.release.set()

    def wait(self):
        self.waited = True
        if self.on_wait:
            self.on_wait()
        return self.code


def queue_setup(tmp_path, monkeypatch, models):
    """ROOT at ``tmp_path``; returns the queue of one round of ``models`` and
    an empty configs tar."""
    monkeypatch.setitem(globals(), "ROOT", tmp_path)
    archive = tmp_path / "configs.tar"
    tarfile.open(archive, "w").close()
    return [(model, 1) for model in models], archive


def test_runs_start_in_queue_order_with_at_most_the_runs_at_once(tmp_path, monkeypatch, capsys):
    queue, archive = queue_setup(tmp_path, monkeypatch, "abcde")
    lock = threading.Lock()
    started, held = [], []
    # Runs started and not yet waited for, now and at most.
    going = {"now": 0, "most": 0}

    def ended():
        with lock:
            going["now"] -= 1

    def popen(command, **kwargs):
        assert kwargs["stdout"] is kwargs["stderr"] is subprocess.PIPE
        name = model_of(command)
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

    run_queue(["docker"], queue, archive, 3, results)

    assert started == list("abcde")
    assert going["most"] == 3
    assert sorted(results) == [(f"{name} run 1", 0) for name in "abcde"]
    out = capsys.readouterr().out
    for name in "abcde":
        assert f"[{name} run 1] {cost_line(name, 1.0)}\n" in out
        assert f"\n{name} run 1: exit 0\n" in f"\n{out}"
    # Each run's cost is saved.
    assert read_costs(tmp_path / COSTS_FILE) == {
        name: {"served_model": "provider/model", "agent_cost_usd": 1.0} for name in "abcde"
    }


def test_with_runs_at_once_every_line_of_a_run_is_prefixed(tmp_path, monkeypatch, capsys):
    queue, archive = queue_setup(tmp_path, monkeypatch, "ab")
    runs = {
        # a's last stdout line has no newline.
        "a": FakeRun("one\ntwo", err="Task 1 of 1 started: t1\n", code=2),
        "b": FakeRun("three\n", err="Dataset weather-tasks read: 1 items\n"),
    }
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: runs[model_of(command)])
    results = []

    run_queue(["docker"], queue, archive, 2, results)

    captured = capsys.readouterr()
    # Whole lines, in each run's order; the two runs' lines may interleave.
    out = captured.out.splitlines(keepends=True)
    a_lines = [line for line in out if "a run" in line]
    assert a_lines == ["[a run 1] one\n", "[a run 1] two\n", "a run 1: exit 2\n"]
    b_lines = [line for line in out if "b run" in line]
    assert b_lines == ["[b run 1] three\n", "b run 1: exit 0\n"]
    assert len(out) == 5
    assert sorted(captured.err.splitlines(keepends=True)) == [
        "[a run 1] Task 1 of 1 started: t1\n",
        "[b run 1] Dataset weather-tasks read: 1 items\n",
    ]
    assert sorted(results) == [("a run 1", 2), ("b run 1", 0)]


def test_sigterm_passes_on_to_every_running_run_and_records_none(tmp_path, monkeypatch, capsys):
    queue, archive = queue_setup(tmp_path, monkeypatch, "abc")
    runs = []

    def popen(command, **kwargs):
        name = model_of(command)
        if name == "c":
            # SIGTERM while the third run starts.
            raise KeyboardInterrupt
        runs.append(FakeRun(f"{name} working\n{cost_line(name, 1.0)}\n", code=143, hold=True))
        return runs[-1]

    monkeypatch.setattr(subprocess, "Popen", popen)
    results = []

    try:
        run_queue(["docker"], queue, archive, 3, results)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected KeyboardInterrupt")

    # Both got SIGTERM, were read to the end and waited for, and neither is
    # recorded or has its cost saved.
    assert [run.signals for run in runs] == [[signal.SIGTERM], [signal.SIGTERM]]
    assert [run.waited for run in runs] == [True, True]
    out = capsys.readouterr().out
    for name in "ab":
        assert f"[{name} run 1] {name} working\n" in out
        assert f"[{name} run 1] {cost_line(name, 1.0)}\n" in out
    assert "exit" not in out
    assert results == []
    assert not (tmp_path / COSTS_FILE).exists()


def test_sigterm_keeps_a_finished_run_and_waits_for_the_running_one(tmp_path, monkeypatch, capsys):
    queue, archive = queue_setup(tmp_path, monkeypatch, "abc")
    runs = {}

    def popen(command, **kwargs):
        name = model_of(command)
        if name == "c":
            # With 2 at once, c starts only once a has ended and been recorded;
            # SIGTERM then, while b runs.
            raise KeyboardInterrupt
        held = name == "b"
        runs[name] = FakeRun(f"{cost_line(name, 1.0)}\n", code=143 if held else 0, hold=held)
        return runs[name]

    monkeypatch.setattr(subprocess, "Popen", popen)
    results = []

    try:
        run_queue(["docker"], queue, archive, 2, results)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("expected KeyboardInterrupt")

    assert results == [("a run 1", 0)]
    assert read_costs(tmp_path / COSTS_FILE) == {
        "a": {"served_model": "provider/model", "agent_cost_usd": 1.0}
    }
    # b got SIGTERM, was read to the end and waited for, and is not recorded.
    assert runs["b"].signals == [signal.SIGTERM]
    assert runs["b"].waited
    out = capsys.readouterr().out
    assert f"[b run 1] {cost_line('b', 1.0)}\n" in out
    assert "b run 1: exit" not in out


def test_a_run_whose_output_the_script_fails_on_is_recorded_as_failed(
    tmp_path, monkeypatch, capsys
):
    queue, archive = queue_setup(tmp_path, monkeypatch, "a")

    def failing(line):
        raise RuntimeError("cannot parse")

    monkeypatch.setitem(globals(), "parse_cost_line", failing)
    run = FakeRun("one\ntwo\n")
    monkeypatch.setattr(subprocess, "Popen", lambda command, **kwargs: run)
    results = []

    run_queue(["docker"], queue, archive, 1, results)

    # Read to the end and waited for, and recorded as failed though it exited 0.
    assert run.waited
    assert results == [("a run 1", 1)]
    assert capsys.readouterr().out == (
        "one\ntwo\na run 1: failed in run-eval.py: RuntimeError: cannot parse\na run 1: exit 1\n"
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

    assert pass_on(stream, out, "[a run 1] ", threading.Lock()) == "OSError: closed"
    assert "".join(out.written) == "[a run 1] good\n"
    assert stream.read() == ""


def test_the_estimate_is_one_total_with_agents_and_grading_apart():
    costs = {"a": {"agent_cost_usd": 2.0}, "b": {"agent_cost_usd": 0.5}}
    lines = estimate_lines(["a", "b"], 3, costs, JUDGES[:2], "openrouter/test/user", PRICES)
    # Agents: (2.0 + 0.5) * 3. Grading: (0.13 * 2 + 0.063) * 2 models * 3 rounds.
    assert lines == ["Estimated cost: $9.44 (agents $7.50, judges and simulated user $1.94)"]


def test_a_model_without_a_saved_cost_and_a_model_without_a_price_are_named():
    costs = {"a": {"agent_cost_usd": 1.0}, "c": {"agent_cost_usd": "1.0"}}
    lines = estimate_lines(["a", "b", "c"], 1, costs, [JUDGES[0], JUDGES[2]], "test/user", PRICES)
    # Agents: a. Grading: (0.13 + 0.063) * 3 models; judge-c has no price in PRICES.
    assert lines == [
        "Estimated cost: $1.58 (agents $1.00, judges and simulated user $0.58)",
        "Agent cost unknown, not counted: b, c",
        f"No OpenRouter price, not counted: {JUDGES[2]}",
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


def test_costs_are_saved_by_model_latest_first(tmp_path):
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

    show_estimate(["a"], 1, [JUDGES[0]], "test/user")

    assert asked == [] and ran == []
    assert capsys.readouterr().out == (
        "Estimated cost: $1.19 (agents $1.00, judges and simulated user $0.19)\n"
    )


def test_a_model_without_a_saved_cost_is_named_unknown(tmp_path, monkeypatch, capsys):
    asked, ran = estimate_setup(tmp_path, monkeypatch)

    show_estimate(["a"], 1, [JUDGES[0]], "test/user")

    assert asked == [] and ran == []
    assert capsys.readouterr().out == (
        "Estimated cost: $0.19 (agents $0.00, judges and simulated user $0.19)\n"
        "Agent cost unknown, not counted: a\n"
    )
    assert not (tmp_path / COSTS_FILE).exists()


def test_an_unreadable_cost_file_makes_the_estimate_unavailable(tmp_path, monkeypatch, capsys):
    asked, ran = estimate_setup(tmp_path, monkeypatch)
    (tmp_path / COSTS_FILE).write_text("not json")

    show_estimate(["a"], 1, [JUDGES[0]], "test/user")

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

        show_estimate(["a"], 1, [JUDGES[0]], "test/user")

        assert capsys.readouterr().out == (
            "Cost estimate unavailable: OpenRouter prices could not be read: no network\n"
        ), error


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
