"""``weather-skills-eval batch``: every run of one batch, on the Docker host whose
socket is mounted. ``scripts/run-eval.py`` starts it as the detached container
``weather-skills-eval-batch``; the batch holds the queue, so the launching
terminal can close.

Order of a batch:
1. Read its settings from the environment, and the configs repo files at ``EVAL_CONFIGS_COMMIT``, which the launcher
   copied into the batch container; stop on any error (exit 2).
2. Create the ``weather-skills-eval`` network when it is missing. It is never
   removed.
3. Pull the app image unless ``EVAL_PULL_APP_IMAGE`` is ``false``, and read
   its revision and PR number labels. A Docker error up to here stops the
   batch (exit 3).
4. For each secret a run needs, take it from the batch's environment when
   set, else read it from Secret Manager. Only the batch reads Secret
   Manager, over the host network the launcher runs it on; each run gets
   only the secrets it needs, as env.
5. Run the queue: each model once per round, as many rounds as
   ``EVAL_REPETITIONS``, at most ``EVAL_PARALLEL_RUNS`` at once. Each run is a
   child container of the eval image ID running ``weather-skills-eval run`` on
   one model, with the configs repo files copied in.

Exit codes: 0 when every run exited 0; 1 when any did not; 2 when the batch
did not start (a setting or the configs repo files);
3 when Docker failed before the first run.
"""

import codecs
import sys
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .config import APP_ENV, is_secret_name
from .configs_repo import (
    ConfigsRepoError,
    load_app_definition,
    parse_connection_keys,
)
from .instance import BATCH_LABEL, DOCKER_ERRORS, LABEL, _archive
from .settings import REQUIRED_PASSTHROUGH, SettingsError, _required, _whole

EXIT_FAILED = 1
EXIT_NOT_STARTED = 2
EXIT_DOCKER = 3

NETWORK = "weather-skills-eval"
HARNESS_LABEL_VALUE = "harness"
NETWORK_LABEL_VALUE = "network"
DOCKER_SOCKET = "/var/run/docker.sock"
# Where the launcher copies the configs repo files into the batch container,
# and where the batch copies them into each run container.
CONFIGS_REPO_DIR = "/app/configs-repo"
# The app image labels each run records.
REVISION_LABEL = "org.opencontainers.image.revision"
PR_NUMBER_LABEL = "weather-skills-chat.pr-number"
# Secret Manager secret id of an environment variable: this prefix, then the
# name lowercased with underscores as dashes. Shared with the VM's Terraform.
SECRET_PREFIX = "weather-skills-eval-"


@dataclass(frozen=True)
class BatchSettings:
    batch: str
    # Model ids at the configs commit, in the order each round runs them.
    models: tuple[str, ...]
    configs_repo: str
    configs_commit: str
    repetitions: int
    parallel_runs: int
    eval_image: str
    app_image: str
    runner_image: str
    pull_app_image: bool
    # Empty when unset; needed only when the environment lacks a secret.
    secrets_project: str


@dataclass(frozen=True)
class PinnedImages:
    # The app image's labels; empty when it does not carry them.
    app_revision: str
    app_pr_number: str


def load_batch_settings(environ: Mapping[str, str]) -> BatchSettings:
    raw_models = _required(environ, "EVAL_MODELS")
    models = tuple(model.strip() for model in raw_models.split(","))
    commit = _required(environ, "EVAL_CONFIGS_COMMIT")
    pull = environ.get("EVAL_PULL_APP_IMAGE", "").strip().lower() or "true"
    return BatchSettings(
        batch=_required(environ, "EVAL_BATCH"),
        models=models,
        configs_repo=_required(environ, "EVAL_CONFIGS_REPO"),
        configs_commit=commit,
        repetitions=_whole(environ, "EVAL_REPETITIONS", 1, minimum=1),
        parallel_runs=_whole(environ, "EVAL_PARALLEL_RUNS", 1, minimum=1),
        eval_image=_required(environ, "EVAL_IMAGE"),
        app_image=_required(environ, "APP_IMAGE"),
        runner_image=_required(environ, "RUNNER_IMAGE"),
        pull_app_image=pull == "true",
        secrets_project=environ.get("EVAL_SECRETS_PROJECT", "").strip(),
    )


def needed_secrets(connection_keys: Iterable[str]) -> set[str]:
    """The secret variables a run needs: the harness's own, those the app's
    secret-named variables come from, and the ``connection_keys`` variables
    that hold the commit's connection keys."""
    app = {setting for name, setting in APP_ENV.items() if is_secret_name(name)}
    return set(REQUIRED_PASSTHROUGH) | app | set(connection_keys)


def repo_files(root: Path) -> dict[str, bytes]:
    """Every file under ``root``, by its path in a run container."""
    return {
        f"{CONFIGS_REPO_DIR}/{path.relative_to(root).as_posix()}": path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def secret_id(name: str) -> str:
    """``APP_OPENROUTER_API_KEY`` -> ``weather-skills-eval-app-openrouter-api-key``."""
    return SECRET_PREFIX + name.lower().replace("_", "-")


def read_secret(project: str, secret: str) -> str:
    """The latest version of ``secret`` in ``project``, read with the
    application default credentials: the VM's service account on the VM."""
    # Imported here: only a batch whose environment lacks a secret needs it.
    from google.cloud import secretmanager

    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project}/secrets/{secret}/versions/latest"
    # Stripped: a secret added with echo ends in a newline no key holds.
    return client.access_secret_version(name=name).payload.data.decode("utf-8").strip()


def resolve_secrets(
    names: Iterable[str],
    environ: Mapping[str, str],
    project: str,
    reader: Callable[[str, str], str],
) -> tuple[dict[str, str], dict[str, str]]:
    """For each of ``names`` the environment leaves unset or empty: its value
    read from Secret Manager, or why it could not be read. Returns (values,
    errors), each by variable name."""
    values, errors = {}, {}
    for name in sorted(names):
        if environ.get(name, "").strip():
            continue
        if not project:
            errors[name] = f"{name} is not set, and EVAL_SECRETS_PROJECT is not set"
            continue
        try:
            value = reader(project, secret_id(name))
        except Exception as exc:
            errors[name] = (
                f"{name} is not set, and secret {secret_id(name)} could not be read "
                f"from project {project}: {type(exc).__name__}: {exc}"
            )
            continue
        if not value:
            errors[name] = f"{name} is not set, and secret {secret_id(name)} is empty"
            continue
        values[name] = value
    return values, errors


def ensure_network(docker, batch: str, say: Callable[[str], None]) -> None:
    """Create the network runs join when it is missing; never remove it."""
    # The names filter also matches names that only contain it.
    if any(network.name == NETWORK for network in docker.networks.list(names=[NETWORK])):
        return
    docker.networks.create(NETWORK, labels={LABEL: NETWORK_LABEL_VALUE, BATCH_LABEL: batch})
    say(f"Network {NETWORK} created")


def pin_images(docker, settings: BatchSettings, say: Callable[[str], None]) -> PinnedImages:
    """The app image's labels, from the image pulled unless
    ``EVAL_PULL_APP_IMAGE`` is ``false``."""
    if settings.pull_app_image:
        say(f"Pulling app image {settings.app_image}")
        app = docker.images.pull(settings.app_image)
    else:
        app = docker.images.get(settings.app_image)
    labels = (app.attrs.get("Config") or {}).get("Labels") or {}
    return PinnedImages(
        app_revision=labels.get(REVISION_LABEL, ""),
        app_pr_number=labels.get(PR_NUMBER_LABEL, ""),
    )


def run_environment(
    environ: Mapping[str, str],
    needed: Iterable[str],
    secrets: Mapping[str, str],
    pinned: PinnedImages,
    model: str,
) -> dict[str, str]:
    """A run's environment: the batch's own, which holds the dotenv file and
    the launcher's choices, without any secret-named variable the run does not
    need in ``needed``, plus the secrets read for it, its model, where its
    configs repo files are, the app image's labels and the network."""
    needed = set(needed)

    def forwarded(name: str) -> bool:
        # The harness's EVAL_ settings hold no secret, EVAL_CONNECTION_KEYS included.
        return name.startswith("EVAL_") or not is_secret_name(name) or name in needed

    return {
        **{name: value for name, value in environ.items() if forwarded(name)},
        **secrets,
        "EVAL_MODEL": model,
        "EVAL_CONFIGS_DIR": CONFIGS_REPO_DIR,
        "EVAL_DOCKER_NETWORK": NETWORK,
        "EVAL_APP_IMAGE_REVISION": pinned.app_revision,
        "EVAL_APP_IMAGE_PR_NUMBER": pinned.app_pr_number,
    }


def pass_on(
    chunks: Iterable[bytes], out, prefix: str, lock: threading.Lock, errors: list[str]
) -> None:
    """Prints a run's output on ``out`` line by line, each line whole, under
    ``lock``, after ``prefix``; a last line without a newline gets one. A
    stream that breaks adds its error to ``errors``."""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""

    def show(line: str) -> None:
        with lock:
            print(prefix + line, file=out, flush=True)

    try:
        for chunk in chunks:
            pending += decoder.decode(chunk)
            *lines, pending = pending.split("\n")
            for line in lines:
                show(line)
        pending += decoder.decode(b"", final=True)
    except Exception as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    if pending:
        show(pending)


def run_one(
    docker,
    image: str,
    batch: str,
    files: Mapping[str, bytes],
    environment: Mapping[str, str],
    prefix: str,
    lock: threading.Lock,
    out,
    err,
) -> int:
    """One run in a child container of the eval ``image``, with ``files``, by
    path, copied in and the run's stdout and stderr passed on; returns its
    exit code. The container is removed when the run ends."""
    container = docker.containers.create(
        image,
        entrypoint=["weather-skills-eval"],
        command=["run"],
        name=f"weather-skills-eval-harness-{uuid.uuid4().hex[:12]}",
        environment=dict(environment),
        labels={LABEL: HARNESS_LABEL_VALUE, BATCH_LABEL: batch},
        network=NETWORK,
        volumes={DOCKER_SOCKET: {"bind": DOCKER_SOCKET, "mode": "rw"}},
    )
    try:
        # No bind mount, as for the runner's emulator files.
        container.put_archive("/", _archive(dict(files)))
        container.start()
        errors: list[str] = []
        readers = [
            threading.Thread(
                target=pass_on,
                args=(
                    container.logs(stream=True, follow=True, stdout=stdout, stderr=not stdout),
                    out if stdout else err,
                    prefix,
                    lock,
                    errors,
                ),
            )
            for stdout in (True, False)
        ]
        for reader in readers:
            reader.start()
        for reader in readers:
            reader.join()
        return container.wait()["StatusCode"]
    finally:
        container.remove(force=True)


def run_queue(
    queue: list[tuple[str, int]],
    parallel: int,
    start: Callable[[str, int, str], int],
    results: list[tuple[str, int]],
    lock: threading.Lock,
    out,
) -> None:
    """Runs each queued (model, round) with ``start(model, round, prefix)``,
    in queue order, at most ``parallel`` at once: the next starts when one
    ends. With ``parallel`` above 1 every line of a run is prefixed with
    ``[<model> run <round>] ``. Each run's exit code is added to
    ``results`` and printed."""
    slots = threading.Semaphore(parallel)
    threads = []

    def go(name: str, repetition: int) -> None:
        label = f"{name} run {repetition}"
        prefix = f"[{label}] " if parallel > 1 else ""
        try:
            code = start(name, repetition, prefix)
            with lock:
                results.append((label, code))
                print(f"{label}: exit {code}", file=out, flush=True)
        finally:
            slots.release()

    for name, repetition in queue:
        slots.acquire()
        thread = threading.Thread(target=go, args=(name, repetition))
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()


def _docker_client():
    import docker

    return docker.from_env()


def main(
    environ: Mapping[str, str],
    *,
    docker_client_factory=_docker_client,
    secret_reader: Callable[[str, str], str] = read_secret,
    configs_repo_dir: Path = Path(CONFIGS_REPO_DIR),
    out=sys.stdout,
    err=sys.stderr,
) -> int:
    lock = threading.Lock()

    def say(line: str) -> None:
        with lock:
            print(line, file=out, flush=True)

    try:
        settings = load_batch_settings(environ)
        definition = load_app_definition(configs_repo_dir)
        files = repo_files(configs_repo_dir)
        connection_keys = parse_connection_keys(environ.get("EVAL_CONNECTION_KEYS", ""))
        key_variables = [connection_keys[u.rstrip("/")] for u in definition.connections]
    except (SettingsError, ConfigsRepoError, OSError) as exc:
        print(f"Batch not started: {exc}", file=err, flush=True)
        return EXIT_NOT_STARTED
    say(f"Batch {settings.batch}: {settings.configs_repo} at {settings.configs_commit}")

    try:
        docker = docker_client_factory()
        ensure_network(docker, settings.batch, say)
        pinned = pin_images(docker, settings, say)
    except DOCKER_ERRORS as exc:
        print(f"Batch not started, Docker failed: {exc}", file=err, flush=True)
        return EXIT_DOCKER

    # The same for every model: the commit's connections are shared.
    needed = needed_secrets(key_variables)
    values, errors = resolve_secrets(needed, environ, settings.secrets_project, secret_reader)
    missing = [errors[secret] for secret in sorted(needed) if secret in errors]
    secrets = {secret: values[secret] for secret in needed if secret in values}

    def start(model: str, repetition: int, prefix: str) -> int:
        if missing:
            say(f"{model} run {repetition}: not started: {'; '.join(missing)}")
            return EXIT_NOT_STARTED
        # Every variable the app's environment comes from reaches the run.
        referenced = needed | set(APP_ENV.values())
        environment = run_environment(environ, referenced, secrets, pinned, model)
        return run_one(
            docker,
            settings.eval_image,
            settings.batch,
            files,
            environment,
            prefix,
            lock,
            out,
            err,
        )

    # In rounds, each model once per round.
    queue = [
        (model, repetition)
        for repetition in range(1, settings.repetitions + 1)
        for model in settings.models
    ]
    results: list[tuple[str, int]] = []
    run_queue(queue, settings.parallel_runs, start, results, lock, out)
    passed = sum(1 for _, code in results if code == 0)
    say(f"Batch {settings.batch} finished: {passed} of {len(results)} runs exited 0")
    return 0 if passed == len(results) else EXIT_FAILED
