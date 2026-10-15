"""One throwaway app container per run.

By default the configured image is pulled before every run, so a stale local
tag never runs. With ``pull`` off, the image already on the Docker host is
used, which is how an image built there and never pushed is run; it is never
pulled, and a missing image stops the run. The container starts from the
image's id with only the configuration's environment and no volumes, so its
storage starts empty. It is removed when the run ends, whether the run
succeeded or not, and when the harness gets SIGTERM (the CLI turns it into an
exit that unwinds through this cleanup).

Each container is labeled with its run's id. A run only ever touches the
container it started, by name, so runs sharing one Docker daemon do not
interfere. A harness killed with SIGKILL cannot clean up; its container
keeps the ``weather-skills-eval.run`` label and is removed by hand.
"""

import logging
import time
import uuid
from collections.abc import Callable

import docker.errors
import httpx

log = logging.getLogger(__name__)

LABEL = "weather-skills-eval"
LABEL_VALUE = "app"
RUN_LABEL = "weather-skills-eval.run"


class AppStartError(Exception):
    """The app image could not be pulled or is not on the Docker host, or its
    container exited, disappeared or never answered /health. Carries the
    container logs when there are any."""


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


class AppInstance:
    def __init__(
        self,
        *,
        docker_client,
        image: str,
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
        self.container = None
        self.image_digest: str | None = None
        self.image_id: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.name}:{self._port}"

    def _pull(self):
        try:
            return self._docker.images.pull(self._image_ref)
        except docker.errors.DockerException as exc:
            raise AppStartError(f"Image {self._image_ref} could not be pulled: {exc}") from exc

    def _local(self):
        try:
            return self._docker.images.get(self._image_ref)
        except docker.errors.ImageNotFound as exc:
            raise AppStartError(f"Image {self._image_ref} is not on the Docker host") from exc
        except docker.errors.DockerException as exc:
            raise AppStartError(
                f"Image {self._image_ref} could not be read from the Docker host: {exc}"
            ) from exc

    def _repo_digest(self, image) -> str | None:
        repository = repository_of(self._image_ref)
        for digest in image.attrs.get("RepoDigests") or []:
            if digest.split("@", 1)[0] == repository:
                return digest
        return None

    def _digest(self, image) -> str:
        digest = self._repo_digest(image)
        if digest is not None:
            return digest
        repository = repository_of(self._image_ref)
        raise AppStartError(
            f"Image {self._image_ref} has no digest for {repository} "
            f"(digests: {image.attrs.get('RepoDigests') or []})"
        )

    def _logs(self) -> str:
        try:
            return self.container.logs(tail=200).decode("utf-8", errors="replace")
        except Exception as exc:  # the logs are a diagnostic; never mask the start error
            return f"<logs unavailable: {exc}>"

    def _wait_for_health(self) -> None:
        deadline = self._clock() + self._start_timeout
        url = f"{self.base_url}/health"
        while True:
            try:
                self.container.reload()
            except docker.errors.NotFound as exc:
                raise AppStartError("App container disappeared before answering /health") from exc
            if self.container.status in ("exited", "dead"):
                raise AppStartError(
                    f"App container exited before answering /health.\n"
                    f"--- container logs ---\n{self._logs()}"
                )
            if self._health_check(url):
                return
            if self._clock() >= deadline:
                raise AppStartError(
                    f"App did not answer {url} within {self._start_timeout:g} s.\n"
                    f"--- container logs ---\n{self._logs()}"
                )
            self._sleep(self._poll_interval)

    def remove(self) -> None:
        """Remove this run's container by name. Never raises: a cleanup error
        must not replace the error that ended the run."""
        self.container = None
        try:
            self._docker.containers.get(self.name).remove(force=True, v=True)
        except docker.errors.NotFound:
            pass
        except Exception as exc:
            log.error("Could not remove app container %s: %s", self.name, exc)

    def __enter__(self) -> "AppInstance":
        if self._pull_first:
            image = self._pull()
            self.image_digest = self._digest(image)
        else:
            # An image built on the host has no repo digest until it is pushed.
            image = self._local()
            self.image_digest = self._repo_digest(image)
        self.image_id = image.id
        try:
            self.container = self._docker.containers.run(
                image.id,
                detach=True,
                name=self.name,
                environment=self._env,
                network=self._network,
                labels={LABEL: LABEL_VALUE, RUN_LABEL: self.run_id},
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
