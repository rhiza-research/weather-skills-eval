"""The configuration file: which app image to run and the environment it runs with.

``image``, ``runner_image`` and the values in ``env`` may reference the harness's own
environment variables as ``${NAME}``. A variable, or a key inside a mapping
value, whose name looks like a secret (one of its words is key, secret,
password or token, singular or plural, except "tokens") must be exactly one
such reference, so no secret is ever written into a configuration file.
Words are separated by underscores, hyphens, dots and case changes, so
``OPENAI_API_KEYS`` and ``apiKey`` are secret names and ``max_tokens`` and
``TIKTOKEN_ENCODING_NAME`` are not.
"""

import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import yaml

ALLOWED_KEYS = ("name", "variant", "image", "runner_image", "pull", "model", "dataset", "env")
REQUIRED_KEYS = ("image", "runner_image", "model", "dataset", "env")

# The harness sets these on the app to run skills on the sandbox runner.
RUNNER_ENV = (
    "SKILL_SANDLOCK",
    "SKILL_SANDBOX_BACKEND",
    "SKILL_K8S_SANDBOX_ENABLED",
    "SKILL_K8S_EMULATOR",
    "SKILL_K8S_APISERVER",
    "SKILL_K8S_BIND_TOKEN",
)

# The harness signs in with BOOTSTRAP_ADMIN_API_KEY; the app does not start
# without WEBUI_SECRET_KEY, and creates the admin from the other two.
REQUIRED_ENV = (
    "WEBUI_SECRET_KEY",
    "BOOTSTRAP_ADMIN_EMAIL",
    "BOOTSTRAP_ADMIN_PASSWORD",
    "BOOTSTRAP_ADMIN_API_KEY",
)

SECRET_WORDS = {"key", "keys", "secret", "secrets", "password", "passwords", "token", "apikey"}
_WORD = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")
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


def _text(raw: Mapping, key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(key, "must be a non-empty string")
    return value.strip()


def is_secret_name(name: str) -> bool:
    return any(word.lower() in SECRET_WORDS for word in _WORD.findall(name))


def _lookup(field: str, name: str, environ: Mapping[str, str]) -> str:
    value = environ.get(name)
    if value is None:
        raise ConfigError(field, f"references ${{{name}}}, which is not set")
    if not value.strip():
        raise ConfigError(field, f"references ${{{name}}}, which is empty")
    return value


def _resolve_text(field: str, key: str, value: str, environ: Mapping[str, str]) -> str:
    """Substitute references in one string. A secret-named key's value must be
    a single reference (surrounding whitespace ignored) and resolves to the
    referenced value alone."""
    if is_secret_name(key):
        match = REFERENCE.fullmatch(value.strip())
        if not match:
            raise ConfigError(
                field, "holds a secret and must be a single ${NAME} reference, not a literal"
            )
        return _lookup(field, match.group(1), environ)
    return REFERENCE.sub(lambda m: _lookup(field, m.group(1), environ), value)


def _resolve_tree(field: str, key: str, value: object, environ: Mapping[str, str]) -> object:
    """Resolve references inside a mapping or list, keeping its structure."""
    if isinstance(value, Mapping):
        resolved = {}
        for child_key, child in value.items():
            if not isinstance(child_key, str):
                raise ConfigError(f"{field}.{child_key}", "keys must be strings")
            resolved[child_key] = _resolve_tree(f"{field}.{child_key}", child_key, child, environ)
        return resolved
    if isinstance(value, list):
        return [
            _resolve_tree(f"{field}[{index}]", key, child, environ)
            for index, child in enumerate(value)
        ]
    if isinstance(value, str):
        return _resolve_text(field, key, value, environ)
    if is_secret_name(key) and value is not None:
        raise ConfigError(
            field, "holds a secret and must be a single ${NAME} reference, not a literal"
        )
    return value


def _env_value(name: str, value: object, environ: Mapping[str, str]) -> str:
    field = f"env.{name}"
    if isinstance(value, (Mapping, list)):
        return json.dumps(_resolve_tree(field, name, value, environ))
    if isinstance(value, str):
        return _resolve_text(field, name, value, environ)
    raise ConfigError(field, "must be a string, or a mapping or list written as JSON")


def parse_config(raw: object, *, default_name: str | None, environ: Mapping[str, str]) -> RunConfig:
    """``default_name`` is used when the file has no ``name``; with ``None``
    (a configuration read from stdin) ``name`` is required."""
    if not isinstance(raw, Mapping):
        raise ConfigError("<root>", "must be a mapping")
    for key in raw:
        if key not in ALLOWED_KEYS:
            raise ConfigError(str(key), "is not a known key")
    for key in REQUIRED_KEYS:
        if key not in raw:
            raise ConfigError(key, "is missing")
    if default_name is None and "name" not in raw:
        raise ConfigError("name", "is missing; a configuration read from stdin needs one")

    name = _text(raw, "name") if "name" in raw else default_name
    variant = raw.get("variant")
    if variant is None:
        variant = ""
    if not isinstance(variant, str):
        raise ConfigError("variant", "must be a string")
    variant = variant.strip()
    image = _resolve_text("image", "image", _text(raw, "image"), environ).strip()
    if not image:
        raise ConfigError("image", "must be a non-empty string")
    runner_image = _resolve_text(
        "runner_image", "runner_image", _text(raw, "runner_image"), environ
    ).strip()
    if not runner_image:
        raise ConfigError("runner_image", "must be a non-empty string")
    model = _text(raw, "model")
    dataset = _text(raw, "dataset")
    pull = raw.get("pull", True)
    if not isinstance(pull, bool):
        raise ConfigError("pull", "must be true or false")

    env_raw = raw["env"]
    if not isinstance(env_raw, Mapping):
        raise ConfigError("env", "must be a mapping of variable names to values")

    env: dict[str, str] = {}
    for name_key, value in env_raw.items():
        if not isinstance(name_key, str) or not ENV_NAME.match(name_key):
            raise ConfigError(f"env.{name_key}", "is not a valid variable name")
        if name_key in RUNNER_ENV:
            raise ConfigError(
                f"env.{name_key}", "is set by the harness to run skills on the sandbox runner"
            )
        env[name_key] = _env_value(name_key, value, environ)

    for required in REQUIRED_ENV:
        if not env.get(required, "").strip():
            raise ConfigError(f"env.{required}", "is missing")

    port = env.get("PORT")
    if port is not None and not (port.isdigit() and 1 <= int(port) <= 65535):
        raise ConfigError("env.PORT", f"must be a port number from 1 to 65535, not {port!r}")
    _connections(env.get("OPENAI_API_CONFIGS"))
    _skill_packs(env.get("SKILL_PACKS"))

    return RunConfig(
        name=name,
        image=image,
        runner_image=runner_image,
        model=model,
        dataset=dataset,
        env=env,
        pull=pull,
        variant=variant,
    )


STDIN = "-"


def _from_text(
    text: str, *, source: str, default_name: str | None, environ: Mapping[str, str]
) -> RunConfig:
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(source, f"is not valid YAML: {exc}") from exc
    return parse_config(raw, default_name=default_name, environ=environ)


def load_config(path: Path, environ: Mapping[str, str], stdin: BinaryIO | None = None) -> RunConfig:
    """Read the configuration from ``path``, or from ``stdin`` when ``path`` is ``-``."""
    if str(path) == STDIN:
        source = "<stdin>"
        stream = stdin if stdin is not None else sys.stdin.buffer
        try:
            text = stream.read().decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ConfigError(source, f"is not UTF-8 text: {exc}") from exc
        return _from_text(text, source=source, default_name=None, environ=environ)
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(str(path), f"is not UTF-8 text: {exc}") from exc
    except OSError as exc:
        raise ConfigError(str(path), f"cannot be read: {exc}") from exc
    return _from_text(text, source=str(path), default_name=path.stem, environ=environ)
