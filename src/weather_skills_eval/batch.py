"""``weather-skills-eval batch``: every run of one batch, on the Docker host whose
socket is mounted. ``scripts/run-eval.py`` starts it as the detached container
``weather-skills-eval-batch``; the batch holds the queue, so the launching
terminal can close.

Order of a batch:
1. Read its settings from the environment and its configuration files; stop
   on any error (exit 2).
2. Remove the containers, then the volumes, that carry a
   ``weather-skills-eval.run`` label or another batch's
   ``weather-skills-eval.batch`` label, left over from a run or batch that was
   killed.
3. Create the ``weather-skills-eval`` network when it is missing. It is never
   removed.
4. Pin the eval, app and runner images to image IDs: the eval image is read
   from the host, the app image is pulled unless ``EVAL_PULL_APP_IMAGE`` is
   ``false``, the runner image is pulled. Read the app image's revision and PR
   number labels. A Docker error up to here stops the batch (exit 3).
5. For each secret a run needs, take it from the batch's environment when
   set, else read it from Secret Manager.
6. Run the queue: each configuration once per round, as many rounds as
   ``EVAL_REPETITIONS``, at most ``EVAL_PARALLEL_RUNS`` at once. Each run is a
   child container of the eval image ID running ``weather-skills-eval run`` on
   one configuration.

Exit codes: 0 when every run exited 0; 1 when any did not; 2 when the batch
did not start (a setting or a configuration file); 3 when Docker failed before
the first run.
"""

import codecs
import sys
import threading
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import ConfigError, secret_references
from .instance import BATCH_LABEL, DOCKER_ERRORS, IMAGE_ID, LABEL, RUN_LABEL, _archive
from .settings import REQUIRED_PASSTHROUGH, SettingsError, _required, _whole

EXIT_FAILED = 1
EXIT_NOT_STARTED = 2
EXIT_DOCKER = 3

NETWORK = "weather-skills-eval"
HARNESS_LABEL_VALUE = "harness"
NETWORK_LABEL_VALUE = "network"
DOCKER_SOCKET = "/var/run/docker.sock"
# Where the Dockerfile copies configs/ in the eval image.
CONFIGS_DIR = Path("/app/configs")
# Where each run container gets its configuration.
RUN_CONFIG_DIR = "/app/run"
# The app image labels each run records.
REVISION_LABEL = "org.opencontainers.image.revision"
PR_NUMBER_LABEL = "weather-skills-chat.pr-number"
# Secret Manager secret id of an environment variable: this prefix, then the
# name lowercased with underscores as dashes. Shared with the VM's Terraform.
SECRET_PREFIX = "weather-skills-eval-"
# Docker sets it per container, so the batch's own is not passed on.
NOT_FORWARDED = ("HOSTNAME",)


@dataclass(frozen=True)
class BatchSettings:
    batch: str
    # Configuration file names, in the order each round runs them.
    configs: tuple[str, ...]
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
    eval_id: str
    app_id: str
    runner_id: str
    # The app image's labels; empty when it does not carry them.
    app_revision: str
    app_pr_number: str


def load_batch_settings(environ: Mapping[str, str]) -> BatchSettings:
    raw_configs = _required(environ, "EVAL_BATCH_CONFIGS")
    configs = tuple(name.strip() for name in raw_configs.split(","))
    if not all(configs):
        raise SettingsError(f"EVAL_BATCH_CONFIGS has an empty file name: {raw_configs!r}")
    pull = environ.get("EVAL_PULL_APP_IMAGE", "").strip().lower() or "true"
    if pull not in ("true", "false"):
        raise SettingsError(f"EVAL_PULL_APP_IMAGE must be true or false, not {pull!r}")
    return BatchSettings(
        batch=_required(environ, "EVAL_BATCH"),
        configs=configs,
        repetitions=_whole(environ, "EVAL_REPETITIONS", 1, minimum=1),
        parallel_runs=_whole(environ, "EVAL_PARALLEL_RUNS", 1, minimum=1),
        eval_image=_required(environ, "EVAL_IMAGE"),
        app_image=_required(environ, "APP_IMAGE"),
        runner_image=_required(environ, "RUNNER_IMAGE"),
        pull_app_image=pull == "true",
        secrets_project=environ.get("EVAL_SECRETS_PROJECT", "").strip(),
    )


def load_configs(names: Iterable[str], configs_dir: Path) -> dict[str, str]:
    """Each configuration's text, by file name, from ``configs_dir``."""
    texts = {}
    for name in names:
        path = configs_dir / name
        if Path(name).name != name or not path.is_file():
            available = sorted(p.name for p in configs_dir.glob("*.yaml"))
            raise ConfigError(name, f"is not a configuration; the files are {', '.join(available)}")
        try:
            texts[name] = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(name, f"cannot be read: {exc}") from exc
    return texts


def needed_secrets(text: str) -> set[str]:
    """The secret variables a run of the configuration ``text`` needs: the
    harness's own and those its secret-named keys reference."""
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        # The run reports the configuration error itself.
        raw = None
    return set(REQUIRED_PASSTHROUGH) | secret_references(raw)


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


def remove_leftovers(docker, batch: str, say: Callable[[str], None]) -> None:
    """Remove the containers, then the volumes, of any run, and of any batch
    but ``batch``. One that cannot be removed is reported and the batch goes
    on: it only takes up room."""

    def leftover(labels: Mapping[str, str]) -> bool:
        return RUN_LABEL in labels or labels.get(BATCH_LABEL, batch) != batch

    containers = {}
    for label in (RUN_LABEL, BATCH_LABEL):
        for container in docker.containers.list(all=True, filters={"label": label}):
            if leftover(container.labels or {}):
                containers[container.id] = container
    for container in containers.values():
        try:
            container.remove(force=True, v=True)
        except DOCKER_ERRORS as exc:
            say(f"Could not remove leftover container {container.name}: {exc}")
        else:
            say(f"Removed leftover container {container.name}")
    volumes = {}
    for label in (RUN_LABEL, BATCH_LABEL):
        for volume in docker.volumes.list(filters={"label": label}):
            if leftover(volume.attrs.get("Labels") or {}):
                volumes[volume.name] = volume
    for volume in volumes.values():
        try:
            volume.remove(force=True)
        except DOCKER_ERRORS as exc:
            say(f"Could not remove leftover volume {volume.name}: {exc}")
        else:
            say(f"Removed leftover volume {volume.name}")


def ensure_network(docker, batch: str, say: Callable[[str], None]) -> None:
    """Create the network runs join when it is missing; never remove it."""
    # The names filter also matches names that only contain it.
    if any(network.name == NETWORK for network in docker.networks.list(names=[NETWORK])):
        return
    docker.networks.create(NETWORK, labels={LABEL: NETWORK_LABEL_VALUE, BATCH_LABEL: batch})
    say(f"Network {NETWORK} created")


def pin_images(docker, settings: BatchSettings, say: Callable[[str], None]) -> PinnedImages:
    """The image ID of each image, resolved once, so no rebuild or pull
    during the batch changes what its runs start from."""
    eval_image = docker.images.get(settings.eval_image)
    # An image ID cannot be pulled.
    if settings.pull_app_image and not IMAGE_ID.fullmatch(settings.app_image):
        say(f"Pulling app image {settings.app_image}")
        app = docker.images.pull(settings.app_image)
    else:
        app = docker.images.get(settings.app_image)
    say(f"Pulling runner image {settings.runner_image}")
    runner = docker.images.pull(settings.runner_image)
    for what, ref, image in (
        ("Eval", settings.eval_image, eval_image),
        ("App", settings.app_image, app),
        ("Runner", settings.runner_image, runner),
    ):
        say(f"{what} image {ref} pinned as {image.id}")
    labels = (app.attrs.get("Config") or {}).get("Labels") or {}
    return PinnedImages(
        eval_id=eval_image.id,
        app_id=app.id,
        runner_id=runner.id,
        app_revision=labels.get(REVISION_LABEL, ""),
        app_pr_number=labels.get(PR_NUMBER_LABEL, ""),
    )


def run_environment(
    environ: Mapping[str, str], secrets: Mapping[str, str], pinned: PinnedImages
) -> dict[str, str]:
    """A run's environment: the batch's own, which holds the dotenv file and
    the launcher's choices, plus the secrets it needs, the pinned images and
    the network."""
    return {
        **{name: value for name, value in environ.items() if name not in NOT_FORWARDED},
        **secrets,
        "APP_IMAGE": pinned.app_id,
        "RUNNER_IMAGE": pinned.runner_id,
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
    pinned: PinnedImages,
    batch: str,
    name: str,
    text: str,
    environment: Mapping[str, str],
    prefix: str,
    lock: threading.Lock,
    out,
    err,
) -> int:
    """One run in a child container of the eval image, with the configuration
    copied in and the run's stdout and stderr passed on; returns its exit
    code. The container is removed when the run ends."""
    path = f"{RUN_CONFIG_DIR}/{name}"
    container = docker.containers.create(
        pinned.eval_id,
        entrypoint=["weather-skills-eval"],
        command=["run", path],
        name=f"weather-skills-eval-harness-{uuid.uuid4().hex[:12]}",
        environment=dict(environment),
        labels={LABEL: HARNESS_LABEL_VALUE, BATCH_LABEL: batch},
        network=NETWORK,
        volumes={DOCKER_SOCKET: {"bind": DOCKER_SOCKET, "mode": "rw"}},
    )
    try:
        # No bind mount, as for the runner's emulator files.
        if not container.put_archive("/", _archive({path: text.encode("utf-8")})):
            raise RuntimeError("the configuration could not be copied into the run container")
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
        status = container.wait()
        if errors:
            raise RuntimeError(f"the run's output could not be read: {'; '.join(errors)}")
        if "StatusCode" not in status:
            raise RuntimeError(f"Docker reported no exit code: {status}")
        return status["StatusCode"]
    finally:
        try:
            container.remove(force=True)
        except Exception as exc:
            with lock:
                print(
                    f"Could not remove run container {container.name}: {exc}", file=err, flush=True
                )


def run_queue(
    queue: list[tuple[str, int]],
    parallel: int,
    start: Callable[[str, int, str], int],
    results: list[tuple[str, int]],
    lock: threading.Lock,
    out,
) -> None:
    """Runs each queued (configuration, round) with ``start(name, round,
    prefix)``, in queue order, at most ``parallel`` at once: the next starts
    when one ends. With ``parallel`` above 1 every line of a run is prefixed
    with ``[<configuration> run <round>] ``. Each run's exit code is added to
    ``results`` and printed; a run ``start`` raises on is recorded as
    failed, after a line saying why."""
    slots = threading.Semaphore(parallel)
    threads = []

    def go(name: str, repetition: int) -> None:
        label = f"{name} run {repetition}"
        prefix = f"[{label}] " if parallel > 1 else ""
        try:
            try:
                code = start(name, repetition, prefix)
            except Exception as exc:
                with lock:
                    print(
                        f"{label}: failed in the batch: {type(exc).__name__}: {exc}",
                        file=out,
                        flush=True,
                    )
                code = EXIT_FAILED
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
    configs_dir: Path = CONFIGS_DIR,
    out=sys.stdout,
    err=sys.stderr,
) -> int:
    lock = threading.Lock()

    def say(line: str) -> None:
        with lock:
            print(line, file=out, flush=True)

    try:
        settings = load_batch_settings(environ)
        texts = load_configs(settings.configs, configs_dir)
    except (SettingsError, ConfigError) as exc:
        print(f"Batch not started: {exc}", file=err, flush=True)
        return EXIT_NOT_STARTED
    say(f"Batch {settings.batch}")

    try:
        docker = docker_client_factory()
        remove_leftovers(docker, settings.batch, say)
        ensure_network(docker, settings.batch, say)
        pinned = pin_images(docker, settings, say)
    except DOCKER_ERRORS as exc:
        print(f"Batch not started, Docker failed: {exc}", file=err, flush=True)
        return EXIT_DOCKER

    needed = {name: needed_secrets(text) for name, text in texts.items()}
    values, errors = resolve_secrets(
        set().union(*needed.values()), environ, settings.secrets_project, secret_reader
    )

    def start(name: str, repetition: int, prefix: str) -> int:
        missing = [errors[secret] for secret in sorted(needed[name]) if secret in errors]
        if missing:
            say(f"{name} run {repetition}: not started: {'; '.join(missing)}")
            return EXIT_NOT_STARTED
        secrets = {secret: values[secret] for secret in needed[name] if secret in values}
        environment = run_environment(environ, secrets, pinned)
        return run_one(
            docker, pinned, settings.batch, name, texts[name], environment, prefix, lock, out, err
        )

    # In rounds, each configuration once per round.
    queue = [
        (name, repetition)
        for repetition in range(1, settings.repetitions + 1)
        for name in settings.configs
    ]
    results: list[tuple[str, int]] = []
    run_queue(queue, settings.parallel_runs, start, results, lock, out)
    passed = sum(1 for _, code in results if code == 0)
    say(f"Batch {settings.batch} finished: {passed} of {len(results)} runs exited 0")
    return 0 if passed == len(results) else EXIT_FAILED
