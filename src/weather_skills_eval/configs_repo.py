"""The app definition at one commit of the configs repo (weather-skills-configs),
read from a copy of the repo's files at that commit. The harness reads the files
and requires the schema version it supports, in each schema under ``schemas/``
and each data file; it checks nothing else about them.

What each file gives the app, as the repo's README says:

- ``models/*.yaml``: each file's ``model`` is one ``CATALOG_MODELS`` entry,
  passed unchanged. ``eval.variant`` names the experiment and never reaches
  the app.
- ``skill-packs.yaml``: each pack's ``git_url``, ``ref`` and
  ``enabled_by_default`` form its ``SKILL_PACKS`` entry. Its ``is_active`` and
  ``skills`` are state the harness sets through the app's API after it starts.
- ``connections.yaml``: the keys, in file order, are ``OPENAI_API_BASE_URLS``;
  each value is the ``OPENAI_API_CONFIGS`` entry at the same index.
  ``OPENAI_API_KEYS`` comes from ``EVAL_CONNECTION_KEYS``, in the same order.
- ``settings.yaml``: each key is the app's environment variable of that name.
"""

import dataclasses
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

from .config import ConfigError, RunConfig
from .settings import SettingsError, _required

# The one schema version the harness reads: each schema's x-schema-version and
# each data file's schema_version.
SCHEMA_VERSION = 1
SCHEMAS = (
    "schemas/model.json",
    "schemas/skill-packs.json",
    "schemas/connections.json",
    "schemas/settings.json",
)
COMMIT = re.compile(r"[0-9a-f]{40}")
# The exit code of a run whose app ignored its catalog; a batch starts no
# further run after it.
EXIT_CATALOG = 6
# App variables the configs repo defines; an eval config may not set them.
APP_DEFINED = (
    "CATALOG_MODELS",
    "SKILL_PACKS",
    "OPENAI_API_BASE_URLS",
    "OPENAI_API_KEYS",
    "OPENAI_API_CONFIGS",
)


class ConfigsRepoError(Exception):
    """The configs repo files cannot be read, are of another schema version,
    or do not define what a run needs. The message names the file or setting."""


class CatalogMismatchError(Exception):
    """The app's catalog ids are not the ids it was given: the app image does
    not provision its catalog from the environment."""


class SkillStateError(Exception):
    """The app's skill pack or skill state differs from the configs repo, or a
    pack or skill the repo records is not in the app."""


@dataclass(frozen=True)
class ConfigsSource:
    directory: Path
    # HTTPS, without a trailing slash or .git, as each run records it.
    repo: str
    commit: str
    model: str
    # The env variable holding each connection's key, by base URL.
    connection_keys: dict[str, str]


@dataclass(frozen=True)
class AppDefinition:
    # Each CATALOG_MODELS entry by its id.
    models: dict[str, dict]
    # Each model's experiment name suffix; empty for none.
    variants: dict[str, str]
    # Each pack as the file holds it, with is_active and skills.
    skill_packs: list[dict]
    # Each connection's OPENAI_API_CONFIGS entry by base URL, in file order.
    connections: dict[str, dict]
    settings: dict[str, object]


def normalize_repo(url: str) -> str:
    """``https://github.com/org/repo.git/`` -> ``https://github.com/org/repo``."""
    repo = url.strip().rstrip("/").removesuffix(".git")
    if not repo.startswith("https://"):
        raise SettingsError(f"EVAL_CONFIGS_REPO must be an HTTPS URL, not {url!r}")
    # Recorded in every experiment, so it must carry no credentials.
    if "@" in repo.removeprefix("https://").split("/", 1)[0]:
        raise SettingsError("EVAL_CONFIGS_REPO must not hold a user name or token")
    return repo


def parse_connection_keys(text: str) -> dict[str, str]:
    """``<base URL>=<VARIABLE>`` pairs, comma-separated."""
    keys = {}
    for pair in (part.strip() for part in text.split(",")):
        if not pair:
            continue
        url, _, name = pair.rpartition("=")
        url, name = url.strip().rstrip("/"), name.strip()
        if not url or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
            raise SettingsError(
                f"EVAL_CONNECTION_KEYS needs <base URL>=<VARIABLE> pairs, not {pair!r}"
            )
        keys[url] = name
    return keys


def load_source(environ: Mapping[str, str]) -> ConfigsSource:
    commit = _required(environ, "EVAL_CONFIGS_COMMIT")
    if not COMMIT.fullmatch(commit):
        raise SettingsError(
            f"EVAL_CONFIGS_COMMIT must be a 40-character commit SHA, not {commit!r}"
        )
    return ConfigsSource(
        directory=Path(_required(environ, "EVAL_CONFIGS_DIR")),
        repo=normalize_repo(_required(environ, "EVAL_CONFIGS_REPO")),
        commit=commit,
        model=_required(environ, "EVAL_MODEL"),
        connection_keys=parse_connection_keys(environ.get("EVAL_CONNECTION_KEYS", "")),
    )


def _read(root: Path, name: str) -> dict:
    """A data file, refused unless it is a mapping of the supported version."""
    try:
        data = yaml.safe_load((root / name).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ConfigsRepoError(f"{name} cannot be read: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigsRepoError(f"{name} must be a mapping")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ConfigsRepoError(
            f"{name} is schema version {data.get('schema_version')!r}; "
            f"the harness reads only version {SCHEMA_VERSION}"
        )
    return data


def _field(data: Mapping, name: str, key: str, kind: type):
    value = data.get(key)
    if not isinstance(value, kind):
        raise ConfigsRepoError(f"{name}: {key} must be a {kind.__name__}")
    return value


def load_app_definition(root: Path) -> AppDefinition:
    """Every data file at ``root``, a copy of the repo at one commit."""
    for schema_name in SCHEMAS:
        try:
            schema = json.loads((root / schema_name).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ConfigsRepoError(f"{schema_name} cannot be read: {exc}") from exc
        version = schema.get("x-schema-version") if isinstance(schema, dict) else None
        if version != SCHEMA_VERSION:
            raise ConfigsRepoError(
                f"{schema_name} is version {version!r}; "
                f"the harness reads only version {SCHEMA_VERSION}"
            )

    models, variants = {}, {}
    for path in sorted((root / "models").glob("*.yaml")):
        name = f"models/{path.name}"
        data = _read(root, name)
        model = _field(data, name, "model", dict)
        model_id = _field(model, name, "id", str)
        variant = _field(data, name, "eval", dict).get("variant") or ""
        if not isinstance(variant, str):
            raise ConfigsRepoError(f"{name}: eval.variant must be a string or null")
        variant = variant.strip()
        models[model_id] = model
        variants[model_id] = variant
    if not models:
        raise ConfigsRepoError("models/ holds no model file")

    packs = _field(_read(root, "skill-packs.yaml"), "skill-packs.yaml", "skill_packs", list)
    connections = _field(_read(root, "connections.yaml"), "connections.yaml", "connections", dict)
    settings = _field(_read(root, "settings.yaml"), "settings.yaml", "settings", dict)
    return AppDefinition(models, variants, packs, connections, settings)


def app_env(
    definition: AppDefinition, connection_keys: Mapping[str, str], environ: Mapping[str, str]
) -> dict[str, str]:
    """The app's environment the configs repo defines, with each connection's
    key from the variable ``connection_keys`` names for its base URL."""
    keys = []
    for url in definition.connections:
        name = connection_keys.get(url.rstrip("/"))
        if name is None:
            raise ConfigsRepoError(
                f"EVAL_CONNECTION_KEYS names no key variable for connection {url}; "
                f"it names {', '.join(connection_keys) or 'none'}"
            )
        value = environ.get(name, "")
        if not value.strip():
            raise ConfigsRepoError(f"{name}, the key of connection {url}, is not set")
        keys.append(value)
    return {
        "CATALOG_MODELS": json.dumps(list(definition.models.values())),
        "SKILL_PACKS": json.dumps(
            [
                {
                    "git_url": pack.get("git_url"),
                    "ref": pack.get("ref"),
                    "enabled_by_default": pack.get("enabled_by_default"),
                }
                for pack in definition.skill_packs
            ]
        ),
        "OPENAI_API_BASE_URLS": ";".join(definition.connections),
        "OPENAI_API_KEYS": ";".join(keys),
        "OPENAI_API_CONFIGS": json.dumps(
            {str(index): entry for index, entry in enumerate(definition.connections.values())}
        ),
        **{
            name: value if isinstance(value, str) else json.dumps(value)
            for name, value in definition.settings.items()
        },
    }


def with_app_definition(
    config: RunConfig,
    definition: AppDefinition,
    source: ConfigsSource,
    environ: Mapping[str, str],
) -> RunConfig:
    """The eval config ``config`` running model ``source.model``: its env plus
    the app env the configs repo defines. The run's config name is the model id."""
    for key in ("model", "variant"):
        if getattr(config, key):
            raise ConfigError(key, "comes from the configs repo commit; remove it")
    for name in (*APP_DEFINED, *definition.settings):
        if name in config.env:
            raise ConfigError(f"env.{name}", "comes from the configs repo commit; remove it")
    if source.model not in definition.models:
        raise ConfigsRepoError(
            f"Model {source.model} is not in the configs repo commit; "
            f"it has {', '.join(sorted(definition.models))}"
        )
    return dataclasses.replace(
        config,
        name=source.model,
        model=source.model,
        variant=definition.variants[source.model],
        env={**config.env, **app_env(definition, source.connection_keys, environ)},
    )


async def check_catalog(app, definition: AppDefinition) -> None:
    """The app's catalog ids, read as a platform admin, must be the ids it was given."""
    listed = await app.platform_models()
    ids = sorted(str(m.get("id")) for m in listed if isinstance(m, dict))
    given = sorted(definition.models)
    if ids != given:
        raise CatalogMismatchError(
            f"The app lists catalog models {ids}, not the {given} it was given in "
            "CATALOG_MODELS: the app image does not provision from env"
        )


def _skill_differences(wanted: list[dict], listed: list) -> tuple[list[str], list[tuple]]:
    """What is missing from the app, and each change that makes it match
    ``wanted``: (description, kind, pack id, tool id, value)."""
    by_url: dict[str, dict] = {}
    for pack in listed:
        if isinstance(pack, dict) and isinstance(pack.get("git_url"), str):
            by_url.setdefault(pack["git_url"], pack)
    missing, changes = [], []
    for want in wanted:
        url = want.get("git_url")
        pack = by_url.get(url)
        if pack is None:
            missing.append(f"skill pack {url} is not in the app")
            continue
        if pack.get("is_active") != want.get("is_active"):
            changes.append(
                (f"pack {url} is_active", "pack", pack.get("id"), None, want.get("is_active"))
            )
        skills = {s.get("name"): s for s in pack.get("skills") or [] if isinstance(s, dict)}
        for skill in want.get("skills") or []:
            have = skills.get(skill.get("name"))
            where = f"skill {skill.get('name')} of pack {url}"
            if have is None:
                missing.append(f"{where} is not in the app")
                continue
            tool_id = have.get("tool_id")
            if have.get("is_active") != skill.get("is_active"):
                changes.append(
                    (f"{where} is_active", "skill", pack.get("id"), tool_id, skill.get("is_active"))
                )
            if have.get("enabled_by_default") != skill.get("enabled_by_default"):
                changes.append(
                    (
                        f"{where} enabled_by_default",
                        "default",
                        pack.get("id"),
                        tool_id,
                        skill.get("enabled_by_default"),
                    )
                )
    return missing, changes


async def apply_skill_state(app, definition: AppDefinition) -> None:
    """Makes the app's pack and skill ``is_active`` and skill
    ``enabled_by_default`` match the configs repo (AD-24). The toggle routes
    flip the state, so each is called only on a difference. Reads the state
    again and raises ``SkillStateError`` when anything still differs, or when a
    recorded pack or skill is not in the app."""
    missing, changes = _skill_differences(definition.skill_packs, await app.platform_skill_packs())
    if missing:
        raise SkillStateError("; ".join(missing))
    for _, kind, pack_id, tool_id, value in changes:
        if kind == "pack":
            await app.toggle_skill_pack(pack_id)
        elif kind == "skill":
            await app.toggle_skill(pack_id, tool_id)
        else:
            await app.set_skill_enabled_by_default(pack_id, tool_id, value)
    missing, changes = _skill_differences(definition.skill_packs, await app.platform_skill_packs())
    if missing or changes:
        differing = missing + [f"{description} differs" for description, *_ in changes]
        raise SkillStateError(
            "The app's skill state does not match the configs repo: " + "; ".join(differing)
        )
