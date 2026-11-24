"""The eval side of a run, from harness settings: which app and runner images to
run, the dataset, and the app's environment the eval owns, its secrets
included. The configs repo adds the rest of the app's environment (see
``configs_repo.py``).
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass

from .settings import _required

# The harness sets these on the app to run skills on the sandbox runner.
RUNNER_ENV = (
    "SKILL_SANDLOCK",
    "SKILL_SANDBOX_BACKEND",
    "SKILL_K8S_SANDBOX_ENABLED",
    "SKILL_K8S_EMULATOR",
    "SKILL_K8S_APISERVER",
    "SKILL_K8S_BIND_TOKEN",
)

# The app's variables the eval sets, each from the harness setting named. The
# harness signs in with BOOTSTRAP_ADMIN_API_KEY; the app does not start
# without WEBUI_SECRET_KEY, and creates the admin from the bootstrap three. The
# app must trace into the Langfuse project the harness reads.
APP_ENV = {
    "WEBUI_SECRET_KEY": "APP_WEBUI_SECRET_KEY",
    "BOOTSTRAP_ADMIN_EMAIL": "EVAL_ADMIN_EMAIL",
    "BOOTSTRAP_ADMIN_PASSWORD": "APP_ADMIN_PASSWORD",
    "BOOTSTRAP_ADMIN_API_KEY": "APP_ADMIN_API_KEY",
    "LANGFUSE_BASE_URL": "LANGFUSE_BASE_URL",
    "LANGFUSE_PUBLIC_KEY": "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY": "LANGFUSE_SECRET_KEY",
}

# The image listens on PORT, 8080 unless its environment sets it.
DEFAULT_APP_PORT = "8080"


@dataclass(frozen=True)
class RunConfig:
    name: str
    image: str
    runner_image: str
    model: str
    dataset: str
    env: dict[str, str]
    pull: bool = True
    # Appended to the experiment name in parentheses; empty leaves the name as is.
    variant: str = ""

    @property
    def env_names(self) -> list[str]:
        return sorted(self.env)

    @property
    def api_key(self) -> str:
        return self.env["BOOTSTRAP_ADMIN_API_KEY"]

    @property
    def port(self) -> str:
        return self.env.get("PORT", DEFAULT_APP_PORT)

    @property
    def connections(self) -> list[dict]:
        """The connection configs of ``OPENAI_API_CONFIGS``, in the order written."""
        return _connections(self.env.get("OPENAI_API_CONFIGS"))

    @property
    def skill_packs_on(self) -> list[str]:
        """The names of the packs in ``SKILL_PACKS`` that the app turns on."""
        return [
            pack_name(pack["git_url"])
            for pack in _skill_packs(self.env.get("SKILL_PACKS"))
            if pack.get("enabled_by_default", True)
        ]


def pack_name(git_url: str) -> str:
    """A pack's repository name: the last path segment of its URL, ignoring
    surrounding whitespace and trailing slashes, without ``.git``."""
    return git_url.strip().rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")


def _skill_packs(text: str | None) -> list[dict]:
    """The pack entries of ``SKILL_PACKS``."""
    return json.loads(text) if text is not None else []


def _connections(text: str | None) -> list[dict]:
    """The connection entries of ``OPENAI_API_CONFIGS``, in the order written."""
    return list(json.loads(text).values()) if text is not None else []


def run_config(environ: Mapping[str, str]) -> RunConfig:
    """The eval side of a run: ``APP_IMAGE``, ``RUNNER_IMAGE``, ``EVAL_DATASET``,
    ``EVAL_PULL_APP_IMAGE`` (``true``, the default, or ``false``) and the app
    variables of APP_ENV, each required."""
    pull = environ.get("EVAL_PULL_APP_IMAGE", "").strip().lower() or "true"
    return RunConfig(
        name="",
        image=_required(environ, "APP_IMAGE"),
        runner_image=_required(environ, "RUNNER_IMAGE"),
        model="",
        dataset=_required(environ, "EVAL_DATASET"),
        env={app: _required(environ, setting) for app, setting in APP_ENV.items()},
        pull=pull == "true",
    )
