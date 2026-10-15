"""One throwaway app container per run.

The container starts from the image with only the configuration's
environment and no volumes, so its storage starts empty. It is removed when
the run ends, whether the run succeeded or not.
"""

import time
import uuid
from collections.abc import Callable

import docker.errors
import httpx


class AppStartError(Exception):
    """The app container exited or never answered /health. Carries its logs."""


def _health_ok(url: str) -> bool:
    try:
        response = httpx.get(url, timeout=5)
    except httpx.HTTPError:
        return False
    if response.status_code != 200:
        return False
    try:
        return response.json().get("status") is True
    except ValueError:
        return False


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
        self._health_check = health_check
        self._clock = clock
        self._sleep = sleep
        self.name = f"weather-skills-eval-app-{uuid.uuid4().hex[:12]}"
        self.container = None
        self.image_digest: str | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.name}:{self._port}"

    def _image(self):
        try:
            return self._docker.images.get(self._image_ref)
        except docker.errors.ImageNotFound:
            return self._docker.images.pull(self._image_ref)

    def _logs(self) -> str:
        try:
            return self.container.logs(tail=200).decode("utf-8", errors="replace")
        except Exception as exc:  # the logs are a diagnostic; never mask the start error
            return f"<logs unavailable: {exc}>"

    def _wait_for_health(self) -> None:
        deadline = self._clock() + self._start_timeout
        url = f"{self.base_url}/health"
        while True:
            self.container.reload()
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
        if self.container is None:
            return
        try:
            self.container.remove(force=True, v=True)
        except docker.errors.NotFound:
            pass
        self.container = None

    def __enter__(self) -> "AppInstance":
        image = self._image()
        digests = image.attrs.get("RepoDigests") or []
        self.image_digest = digests[0] if digests else image.id
        self.container = self._docker.containers.run(
            image.id,
            detach=True,
            name=self.name,
            environment=self._env,
            network=self._network,
            labels={"weather-skills-eval": "app"},
        )
        try:
            self._wait_for_health()
        except BaseException:
            self.remove()
            raise
        return self

    def __exit__(self, *exc_info) -> None:
        self.remove()
