"""One throwaway app container per run, with a sandbox runner container next to it.

Skills run on the runner, not in the app: the runner container starts from the
sandbox runner image (the image the cluster's tool runners use, with Chromium)
and runs the sandbox emulator of the app image under test, which serves the
claim, bind and exec APIs the app's ``k8s`` sandbox backend calls. The harness
creates a container from the app image without starting it, reads
``emulator.py`` and ``local_runtime.py`` from ``/app/sandbox/emulator`` with
``get_archive``, removes that container, and copies the files into the runner
with ``put_archive``, so a remote Docker context works the same as a local
one. The runner has the Docker restart policy ``on-failure``, so a runner
that dies is restarted, and the harness waits for it to be running again
before each task. The app gets the emulator settings and a bind token
generated for the run; the runner gets the same token. One Docker volume per run is mounted at
the app's data directory and at ``/data`` in the runner, so the files a skill
writes land in the chat's artifacts folder that the app reads.

By default the app image is pulled before every run, so a stale local tag
never runs. With ``pull`` off, the app image already on the Docker host is
used, which is how an image built there and never pushed is run; it is never
pulled, and a missing image stops the run. The runner image is a published
image and is pulled on every run, whatever ``pull`` says. The app container
starts from the image's id with only the configuration's environment plus the
emulator settings, and the run's volume, which starts empty. The containers
and the volume are removed when the run ends, whether the run succeeded or
not, and when the harness gets SIGTERM (the CLI turns it into an exit that
unwinds through this cleanup).

The containers and the volume are labeled with their run's id. A run only ever
touches the objects it created, by name, so runs sharing one Docker daemon do
not interfere. A harness killed with SIGKILL cannot clean up; its objects keep
the ``weather-skills-eval.run`` label and are removed by hand.
"""

import io
import logging
import secrets
import tarfile
import time
import uuid
from collections.abc import Callable

import docker.errors
import httpx

log = logging.getLogger(__name__)

LABEL = "weather-skills-eval"
LABEL_VALUE = "app"
RUNNER_LABEL_VALUE = "runner"
EMULATOR_LABEL_VALUE = "emulator"
VOLUME_LABEL_VALUE = "data"
RUN_LABEL = "weather-skills-eval.run"

# Where the volume is mounted: the app's data directory, and the runner's.
APP_DATA_DIR = "/app/backend/data"
RUNNER_DATA_DIR = "/data"
# Where the app image ships the emulator, and where the runner gets it.
APP_EMULATOR_DIR = "/app/sandbox/emulator"
EMULATOR_FILES = ("emulator.py", "local_runtime.py")
EMULATOR_DIR = "/opt/sandbox/emulator"
EMULATOR_API_PORT = "18080"
EMULATOR_BIND_PORT = "9090"


class AppStartError(Exception):
    """The app image could not be pulled or is not on the Docker host, or its
    container exited, disappeared or never answered /health. Carries the
    container logs when there are any."""


class RunnerStartError(AppStartError):
    """The app image has no sandbox emulator, the runner image could not be
    pulled, or the runner container could not start, exited or disappeared
    before the app was ready, was gone or not running again in time before a
    task or after the last task, or was not the same running runner after a
    task (``restartable`` says whether Docker restarts it). Carries the runner
    logs when there are any."""

    # True when Docker restarts the runner by itself: it restarted, is
    # restarting, or exited with a non-zero code.
    restartable = False


def _health_ok(url: str) -> bool:
    try:
        response = httpx.get(url, timeout=5)
    except httpx.HTTPError:
        return False
    if response.status_code != 200:
        return False
    try:
        data = response.json()
    except ValueError:
        return False
    return isinstance(data, dict) and data.get("status") is True


def repository_of(ref: str) -> str:
    """``ghcr.io/org/app:tag`` and ``ghcr.io/org/app@sha256:...`` -> ``ghcr.io/org/app``."""
    if "@" in ref:
        return ref.split("@", 1)[0]
    head, _, last = ref.rpartition("/")
    if ":" in last:
        last = last.split(":", 1)[0]
    return f"{head}/{last}" if head else last


def _archive(files: dict[str, bytes]) -> bytes:
    """A tar of ``files`` (absolute paths) with their parent directories, to
    extract at ``/``."""
    directories: set[str] = set()
    for path in files:
        parents = path.strip("/").split("/")[:-1]
        directories.update("/".join(parents[: i + 1]) for i in range(len(parents)))
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as tar:
        for directory in sorted(directories):
            info = tarfile.TarInfo(directory)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            tar.addfile(info)
        for path, data in files.items():
            info = tarfile.TarInfo(path.strip("/"))
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _file_from_archive(chunks, path: str) -> bytes:
    """The one file in the tar ``get_archive`` returned for ``path``."""
    with tarfile.open(fileobj=io.BytesIO(b"".join(chunks))) as tar:
        for member in tar.getmembers():
            if member.isfile():
                return tar.extractfile(member).read()
    raise FileNotFoundError(f"{path} is not a file")


class AppInstance:
    def __init__(
        self,
        *,
        docker_client,
        image: str,
        runner_image: str,
        env: dict[str, str],
        network: str,
        port: str,
        start_timeout: float,
        poll_interval: float,
        pull: bool = True,
        health_check: Callable[[str], bool] = _health_ok,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self._docker = docker_client
        self._image_ref = image
        self._runner_image_ref = runner_image
        self._env = env
        self._network = network
        self._port = port
        self._start_timeout = start_timeout
        self._poll_interval = poll_interval
        self._pull_first = pull
        self._health_check = health_check
        self._clock = clock
        self._sleep = sleep
        self.run_id = uuid.uuid4().hex[:12]
        self.name = f"weather-skills-eval-app-{self.run_id}"
        self.runner_name = f"weather-skills-eval-runner-{self.run_id}"
        self.emulator_source_name = f"weather-skills-eval-emulator-{self.run_id}"
        self.volume_name = f"weather-skills-eval-data-{self.run_id}"
        self._bind_token = secrets.token_hex(32)
        self.container = None
        self.runner = None
        self.image_digest: str | None = None
        self.image_id: str | None = None
        self.runner_image_digest: str | None = None
        self.runner_image_id: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.name}:{self._port}"

    @property
    def app_env(self) -> dict[str, str]:
        """The configuration's environment plus the settings that send skill
        runs to this run's runner."""
        return {
            **self._env,
            "SKILL_SANDLOCK": "false",
            "SKILL_SANDBOX_BACKEND": "k8s",
            "SKILL_K8S_SANDBOX_ENABLED": "true",
            "SKILL_K8S_EMULATOR": "true",
            "SKILL_K8S_APISERVER": f"http://{self.runner_name}:{EMULATOR_API_PORT}",
            "SKILL_K8S_BIND_TOKEN": self._bind_token,
        }

    @property
    def runner_env(self) -> dict[str, str]:
        return {
            "ARTIFACTS_DIR": f"{RUNNER_DATA_DIR}/artifacts",
            "SKILL_K8S_EMULATOR_HOST": self.runner_name,
            "SKILL_K8S_EMULATOR_API_PORT": EMULATOR_API_PORT,
            "SKILL_K8S_BIND_PORT": EMULATOR_BIND_PORT,
            "SKILL_K8S_EMULATOR_START_DELAY": "0",
            "SKILL_K8S_BIND_TOKEN": self._bind_token,
        }

    def _labels(self, value: str) -> dict[str, str]:
        return {LABEL: value, RUN_LABEL: self.run_id}

    def _emulator_files(self, image) -> dict[str, bytes]:
        """The emulator files of the app image, keyed by their path in the
        runner, read from a container created from the image and never started."""
        try:
            source = self._docker.containers.create(
                image.id,
                name=self.emulator_source_name,
                labels=self._labels(EMULATOR_LABEL_VALUE),
            )
        except docker.errors.DockerException as exc:
            raise RunnerStartError(
                f"The emulator could not be read from image {self._image_ref}: {exc}"
            ) from exc
        try:
            files = {}
            for name in EMULATOR_FILES:
                path = f"{APP_EMULATOR_DIR}/{name}"
                try:
                    chunks, _ = source.get_archive(path)
                    data = _file_from_archive(chunks, path)
                except (docker.errors.NotFound, FileNotFoundError) as exc:
                    raise RunnerStartError(f"Image {self._image_ref} has no {path}") from exc
                except (docker.errors.DockerException, tarfile.TarError, OSError) as exc:
                    # OSError covers a broken archive stream (requests' errors are OSErrors).
                    raise RunnerStartError(
                        f"{path} could not be read from image {self._image_ref}: {exc}"
                    ) from exc
                files[f"{EMULATOR_DIR}/{name}"] = data
            return files
        finally:
            self._remove_container(self.emulator_source_name, "emulator source")

    def _pull(self, ref: str, error: type[AppStartError], what: str):
        try:
            return self._docker.images.pull(ref)
        except docker.errors.DockerException as exc:
            raise error(f"{what} {ref} could not be pulled: {exc}") from exc

    def _local(self, ref: str, error: type[AppStartError], what: str):
        try:
            return self._docker.images.get(ref)
        except docker.errors.ImageNotFound as exc:
            raise error(f"{what} {ref} is not on the Docker host") from exc
        except docker.errors.DockerException as exc:
            raise error(f"{what} {ref} could not be read from the Docker host: {exc}") from exc

    @staticmethod
    def _repo_digest(ref: str, image) -> str | None:
        repository = repository_of(ref)
        for digest in image.attrs.get("RepoDigests") or []:
            if digest.split("@", 1)[0] == repository:
                return digest
        return None

    def _image(self, ref: str, error: type[AppStartError], what: str, pull: bool):
        """The image to start, and its digest for its repository."""
        if not pull:
            # An image built on the host has no repo digest until it is pushed.
            image = self._local(ref, error, what)
            return image, self._repo_digest(ref, image)
        image = self._pull(ref, error, what)
        digest = self._repo_digest(ref, image)
        if digest is None:
            raise error(
                f"{what} {ref} has no digest for {repository_of(ref)} "
                f"(digests: {image.attrs.get('RepoDigests') or []})"
            )
        return image, digest

    @staticmethod
    def _logs(container) -> str:
        try:
            return container.logs(tail=200).decode("utf-8", errors="replace")
        except Exception as exc:  # the logs are a diagnostic; never mask the start error
            return f"<logs unavailable: {exc}>"

    def _start_runner(self, image, files: dict[str, bytes]) -> None:
        try:
            self._docker.volumes.create(
                name=self.volume_name, labels=self._labels(VOLUME_LABEL_VALUE)
            )
            self.runner = self._docker.containers.create(
                image.id,
                command=["python3", "-u", f"{EMULATOR_DIR}/emulator.py"],
                name=self.runner_name,
                user="0",
                environment=self.runner_env,
                network=self._network,
                labels=self._labels(RUNNER_LABEL_VALUE),
                volumes={self.volume_name: {"bind": RUNNER_DATA_DIR, "mode": "rw"}},
                # Docker restarts a runner that dies, as the cluster replaces a dead runner pod.
                restart_policy={"Name": "on-failure"},
            )
            if not self.runner.put_archive("/", _archive(files)):
                raise RunnerStartError("The emulator files could not be copied into the runner")
            self.runner.start()
        except docker.errors.DockerException as exc:
            raise RunnerStartError(f"Runner container could not start: {exc}") from exc

    def _runner_state(self, when: str) -> tuple[str, object, object]:
        """The runner's status, Docker restart count and exit code;
        ``RunnerStartError`` when it is gone or cannot be read."""
        try:
            self.runner.reload()
        except docker.errors.NotFound as exc:
            raise RunnerStartError(f"Runner container disappeared {when}") from exc
        except docker.errors.DockerException as exc:
            raise RunnerStartError(f"Runner container could not be read {when}: {exc}") from exc
        attrs = self.runner.attrs or {}
        state = attrs.get("State") or {}
        return self.runner.status, attrs.get("RestartCount"), state.get("ExitCode")

    def _runner_error(self, message: str) -> RunnerStartError:
        return RunnerStartError(f"{message}\n--- runner logs ---\n{self._logs(self.runner)}")

    def wait_for_runner(self, when: str) -> object:
        """Return the runner's restart count once it is running on two
        consecutive polls with the same restart count, so a runner Docker keeps
        restarting is not taken for a running one. Polls at the poll interval up
        to the start timeout, and at least twice, so a start timeout shorter
        than the poll interval does not fail a running runner;
        ``RunnerStartError`` (with its logs) when it is not running steadily by
        then, and at once when it exited with code 0 or is dead, which Docker
        does not restart, or is gone."""
        deadline = self._clock() + self._start_timeout
        running_with = None  # (restart count,) at the last poll that found it running
        waiting_logged = False
        polls = 0
        while True:
            status, restarts, exit_code = self._runner_state(when)
            polls += 1
            if status == "exited" and exit_code == 0:
                raise self._runner_error(
                    f"Runner container exited with code 0 {when}; Docker does not restart it."
                )
            if status == "dead":
                raise self._runner_error(
                    f"Runner container is dead {when}; Docker does not restart it."
                )
            if status == "running" and running_with == (restarts,):
                return restarts
            if status != "running" and not waiting_logged:
                log.warning(
                    "Runner container %s is not running %s; waiting up to %g s for it to restart",
                    self.runner_name,
                    when,
                    self._start_timeout,
                )
                waiting_logged = True
            running_with = (restarts,) if status == "running" else None
            if polls >= 2 and self._clock() >= deadline:
                raise self._runner_error(
                    f"Runner container was not running steadily {when} after "
                    f"{self._start_timeout:g} s: it is {status}, restarted {restarts} times."
                )
            self._sleep(self._poll_interval)

    def check_runner_unchanged(self, when: str, restarts: object) -> None:
        """Raise ``RunnerStartError`` (with its logs) unless the runner is
        running and has not restarted since it had ``restarts`` restarts. The
        error is ``restartable`` when the runner is running again, restarting,
        or exited with a non-zero code; not when it exited with code 0, is
        dead or in any other state, or is gone or cannot be read."""
        status, now, exit_code = self._runner_state(when)
        if status != "running" or now != restarts:
            error = self._runner_error(
                f"Runner container stopped {when}: it is {status}, "
                f"restarted {now} times, {restarts} before."
            )
            error.restartable = status in ("running", "restarting") or (
                status == "exited" and exit_code != 0
            )
            raise error

    def check_runner(self, when: str = "before the app was ready") -> None:
        """Raise ``RunnerStartError`` when the runner container has exited or
        is being restarted after a crash (with its logs), or cannot be read,
        such as when it is gone."""
        status, _, _ = self._runner_state(when)
        if status in ("exited", "dead", "restarting"):
            raise self._runner_error(f"Runner container {status} {when}.")

    def _wait_for_health(self) -> None:
        deadline = self._clock() + self._start_timeout
        url = f"{self.base_url}/health"
        while True:
            self.check_runner()
            try:
                self.container.reload()
            except docker.errors.NotFound as exc:
                raise AppStartError("App container disappeared before answering /health") from exc
            if self.container.status in ("exited", "dead"):
                raise AppStartError(
                    f"App container exited before answering /health.\n"
                    f"--- container logs ---\n{self._logs(self.container)}"
                )
            if self._health_check(url):
                return
            if self._clock() >= deadline:
                raise AppStartError(
                    f"App did not answer {url} within {self._start_timeout:g} s.\n"
                    f"--- container logs ---\n{self._logs(self.container)}"
                )
            self._sleep(self._poll_interval)

    def _remove_container(self, name: str, what: str) -> None:
        try:
            self._docker.containers.get(name).remove(force=True, v=True)
        except docker.errors.NotFound:
            pass
        except Exception as exc:
            log.error("Could not remove %s container %s: %s", what, name, exc)

    def remove(self) -> None:
        """Remove this run's emulator source, app container, runner container
        and volume by name, in that order, since a volume in use cannot be
        removed. Never
        raises: a cleanup error must not replace the error that ended the run."""
        self.container = None
        self.runner = None
        self._remove_container(self.emulator_source_name, "emulator source")
        self._remove_container(self.name, "app")
        self._remove_container(self.runner_name, "runner")
        try:
            self._docker.volumes.get(self.volume_name).remove(force=True)
        except docker.errors.NotFound:
            pass
        except Exception as exc:
            log.error("Could not remove volume %s: %s", self.volume_name, exc)

    def __enter__(self) -> "AppInstance":
        image, self.image_digest = self._image(
            self._image_ref, AppStartError, "Image", self._pull_first
        )
        self.image_id = image.id
        try:
            files = self._emulator_files(image)
            # A published image: pulled whatever ``pull`` says.
            runner_image, self.runner_image_digest = self._image(
                self._runner_image_ref, RunnerStartError, "Runner image", True
            )
            self.runner_image_id = runner_image.id
            self._start_runner(runner_image, files)
            self.container = self._docker.containers.run(
                image.id,
                detach=True,
                name=self.name,
                environment=self.app_env,
                network=self._network,
                labels=self._labels(LABEL_VALUE),
                volumes={self.volume_name: {"bind": APP_DATA_DIR, "mode": "rw"}},
            )
            self._wait_for_health()
        except BaseException as exc:
            self.remove()
            if isinstance(exc, docker.errors.DockerException):
                raise AppStartError(f"App container could not start: {exc}") from exc
            raise
        return self

    def __exit__(self, *exc_info) -> None:
        self.remove()
