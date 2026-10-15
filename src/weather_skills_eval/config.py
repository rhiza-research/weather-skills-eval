"""The configuration file: which app image to run and the environment it runs with.

A value may reference one of the harness's own environment variables as
``${NAME}``. A variable whose name looks like a secret (it contains KEY,
SECRET, PASSWORD or TOKEN) must be exactly one such reference, so no secret
is ever written into a configuration file.
"""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

ALLOWED_KEYS = ("name", "image", "model", "dataset", "env")
REQUIRED_KEYS = ("image", "model", "dataset", "env")

# The harness signs in with BOOTSTRAP_ADMIN_API_KEY; the app does not start
# without WEBUI_SECRET_KEY, and creates the admin from the other two.
REQUIRED_ENV = (
    "WEBUI_SECRET_KEY",
    "BOOTSTRAP_ADMIN_EMAIL",
    "BOOTSTRAP_ADMIN_PASSWORD",
    "BOOTSTRAP_ADMIN_API_KEY",
)

SECRET_NAME = re.compile(r"KEY|SECRET|PASSWORD|TOKEN", re.IGNORECASE)
REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# The image listens on PORT, 8080 unless the configuration sets it.
DEFAULT_APP_PORT = "8080"


class ConfigError(Exception):
    """The configuration file is invalid. The message names the field."""

    def __init__(self, field: str, message: str):
        super().__init__(f"{field}: {message}")
        self.field = field


@dataclass(frozen=True)
class RunConfig:
    name: str
    image: str
    model: str
    dataset: str
    env: dict[str, str]

    @property
    def env_names(self) -> list[str]:
        return sorted(self.env)

    @property
    def api_key(self) -> str:
        return self.env["BOOTSTRAP_ADMIN_API_KEY"]

    @property
    def port(self) -> str:
        return self.env.get("PORT", DEFAULT_APP_PORT)


def _text(raw: Mapping, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(key, "must be a non-empty string")
    return value.strip()


def _resolve(field: str, value: str, environ: Mapping[str, str]) -> str:
    def substitute(match: re.Match) -> str:
        name = match.group(1)
        if name not in environ:
            raise ConfigError(field, f"references ${{{name}}}, which is not set")
        return environ[name]

    return REFERENCE.sub(substitute, value)


def _env_value(name: str, value: object) -> str:
    field = f"env.{name}"
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    if isinstance(value, str):
        return value
    raise ConfigError(field, "must be a string, or a mapping or list written as JSON")


def parse_config(raw: object, *, default_name: str, environ: Mapping[str, str]) -> RunConfig:
    if not isinstance(raw, Mapping):
        raise ConfigError("<root>", "must be a mapping")
    for key in raw:
        if key not in ALLOWED_KEYS:
            raise ConfigError(str(key), "is not a known key")
    for key in REQUIRED_KEYS:
        if key not in raw:
            raise ConfigError(key, "is missing")

    name = _text(raw, "name") if "name" in raw else default_name
    image = _text(raw, "image")
    model = _text(raw, "model")
    dataset = _text(raw, "dataset")

    env_raw = raw["env"]
    if not isinstance(env_raw, Mapping):
        raise ConfigError("env", "must be a mapping of variable names to values")

    env: dict[str, str] = {}
    for name_key, value in env_raw.items():
        if not isinstance(name_key, str) or not ENV_NAME.match(name_key):
            raise ConfigError(f"env.{name_key}", "is not a valid variable name")
        field = f"env.{name_key}"
        text = _env_value(name_key, value)
        if SECRET_NAME.search(name_key) and not REFERENCE.fullmatch(text.strip()):
            raise ConfigError(
                field, "holds a secret and must be a single ${NAME} reference, not a literal"
            )
        env[name_key] = _resolve(field, text, environ)

    for required in REQUIRED_ENV:
        if not env.get(required):
            raise ConfigError(f"env.{required}", "is missing")

    return RunConfig(name=name, image=image, model=model, dataset=dataset, env=env)


def load_config(path: Path, environ: Mapping[str, str]) -> RunConfig:
    try:
        text = path.read_text()
    except OSError as exc:
        raise ConfigError(str(path), f"cannot be read: {exc}") from exc
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(str(path), f"is not valid YAML: {exc}") from exc
    return parse_config(raw, default_name=path.stem, environ=environ)
