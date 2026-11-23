"""The eval side of a run, from harness settings: which app and runner images to
run, the dataset, and the app's environment the eval owns, its secrets
included. The configs repo adds the rest of the app's environment (see
``configs_repo.py``).

A name looks like a secret when one of its words is key, secret, password or
token, singular or plural, except "tokens". Words are separated by
underscores, hyphens, dots and case changes, so ``OPENAI_API_KEYS`` and
``apiKey`` are secret names and ``max_tokens`` and ``TIKTOKEN_ENCODING_NAME``
are not.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass

from .settings import SettingsError, _required

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

SECRET_WORDS = {"key", "keys", "secret", "secrets", "password", "passwords", "token", "apikey"}
_WORD = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
# The image listens on PORT, 8080 unless its environment sets it.
DEFAULT_APP_PORT = "8080"


class ConfigError(Exception):
    """The app's environment is invalid. The message names the field."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field


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
    """The pack entries of ``SKILL_PACKS``, with ``git_url`` and
    ``enabled_by_default`` checked for their types."""
    if text is None:
        return []
    field = "env.SKILL_PACKS"
    try:
        packs = json.loads(text)
    except ValueError as exc:
        raise ConfigError(field, f"is not JSON: {exc}") from exc
    if not isinstance(packs, list) or not all(isinstance(p, dict) for p in packs):
        raise ConfigError(field, "must be a list of pack objects")
    for index, pack in enumerate(packs):
        git_url = pack.get("git_url")
        if not isinstance(git_url, str) or not git_url.strip():
            raise ConfigError(field, f"entry {index} needs a non-empty string git_url")
        if not isinstance(pack.get("enabled_by_default", True), bool):
            raise ConfigError(field, f"entry {index} enabled_by_default must be true or false")
    return packs


def _connections(text: str | None) -> list[dict]:
    """Each connection's ``enable``, ``prefix_id`` and ``model_ids``, the keys
    the app reads to pick a connection, checked for their types."""
    if text is None:
        return []
    field = "env.OPENAI_API_CONFIGS"
    try:
        configs = json.loads(text)
    except ValueError as exc:
        raise ConfigError(field, f"is not JSON: {exc}") from exc
    if not isinstance(configs, dict) or not all(isinstance(c, dict) for c in configs.values()):
        raise ConfigError(field, "must map each connection to an object")
    for key, config in configs.items():
        if not isinstance(config.get("enable", True), bool):
            raise ConfigError(f"{field}.{key}.enable", "must be true or false")
        if not isinstance(config.get("prefix_id") or "", str):
            raise ConfigError(f"{field}.{key}.prefix_id", "must be a string")
        model_ids = config.get("model_ids") or []
        if not isinstance(model_ids, list) or not all(isinstance(m, str) for m in model_ids):
            raise ConfigError(f"{field}.{key}.model_ids", "must be a list of model ids")
    return list(configs.values())


def is_secret_name(name: str) -> bool:
    return any(word.lower() in SECRET_WORDS for word in _WORD.findall(name))


def run_config(environ: Mapping[str, str]) -> RunConfig:
    """The eval side of a run: ``APP_IMAGE``, ``RUNNER_IMAGE``, ``EVAL_DATASET``,
    ``EVAL_PULL_APP_IMAGE`` (``true``, the default, or ``false``) and the app
    variables of APP_ENV, each required."""
    pull = environ.get("EVAL_PULL_APP_IMAGE", "").strip().lower() or "true"
    if pull not in ("true", "false"):
        raise SettingsError(f"EVAL_PULL_APP_IMAGE must be true or false, not {pull!r}")
    return RunConfig(
        name="",
        image=_required(environ, "APP_IMAGE"),
        runner_image=_required(environ, "RUNNER_IMAGE"),
        model="",
        dataset=_required(environ, "EVAL_DATASET"),
        env={app: _required(environ, setting) for app, setting in APP_ENV.items()},
        pull=pull == "true",
    )
