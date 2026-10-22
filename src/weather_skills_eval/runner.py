"""A run: one configuration, every task in the dataset played once, one Langfuse experiment.

Order of a run:
1. Read the harness settings and the configuration file; stop on any error.
2. Read the dataset, check that the configuration turns on every skill pack
   the dataset's metadata names in ``skill_packs``, and check every item
   against the task format.
3. Pull the app image, read the sandbox emulator from it, pull the runner
   image, start the runner container and its volume, then the app
   container, and wait for /health.
4. Read the resolved configuration from the app (the catalog model's base
   model, tools, skill pack commits), and ask OpenRouter which model answers
   for that base model; the experiment is named after it.
5. Run the experiment: each item's task waits for the runner to be running
   (Docker restarts one that died), plays the task and returns its scores,
   which the evaluator records on the item. A runner that Docker restarted,
   is restarting, or exited with a non-zero code after the play leaves that
   task unscored, with the runner error on its item, and the run goes on;
   when no task waited for the runner after the latest such loss, it is
   waited for once after the last task. A runner that is gone, exited with
   code 0, is dead, or is not running steadily by the start timeout stops
   the run.
6. Remove the containers and the volume and flush Langfuse, whatever happened.
"""

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .app_client import AppClient, ChatError, highest_skill_versions
from .config import RunConfig, load_config
from .instance import AppInstance, RunnerStartError
from .langfuse_gateway import LangfuseGateway
from .play import PlayDeps, play
from .scoring import COST_SCORE
from .settings import Settings, load_settings
from .tasks import parse_tasks


class DatasetError(Exception):
    """The dataset is missing, cannot be read, has no items, has invalid
    ``skill_packs`` metadata, or needs a skill pack the configuration does not
    turn on."""


@dataclass
class RunSummary:
    experiment_name: str
    run_name: str
    experiment_url: str | None
    items: int
    goals_reached: int
    metadata: dict
    # Ids of the tasks not scored because the runner restarted during their play.
    runner_lost: list[str]


def base_model(entry: dict, catalog_id: str, connections: list[dict]) -> str:
    """The catalog model's base model id, without the app's connection prefix.

    The app names a connection's models ``<prefix_id>.<id>``. It serves a
    model from the last enabled connection whose ``model_ids`` allow it (an
    empty list allows any), and removes that connection's prefix before it
    calls the provider; this picks the connection the same way.
    """
    base = entry.get("base_model_id")
    if not isinstance(base, str) or not base.strip():
        raise ChatError(f"The app reports no base model for catalog model {catalog_id}")
    base = base.strip()

    def bare(prefix: str | None) -> str:
        return base[len(prefix) + 1 :] if prefix and base.startswith(f"{prefix}.") else base

    serving: dict = {}
    for connection in connections:
        allowed = connection.get("model_ids") or []
        prefix = connection.get("prefix_id")
        if connection.get("enable", True) and (
            not allowed or bare(prefix) in allowed or base in allowed
        ):
            serving = connection
    return bare(serving.get("prefix_id"))


async def _resolved(app: AppClient, config: RunConfig) -> tuple[list[str], list[dict], str]:
    model = base_model(await app.catalog_model(config.model), config.model, config.connections)
    tool_ids = highest_skill_versions(await app.tool_summary())
    packs = [
        {
            "git_url": p.get("git_url"),
            "git_ref": p.get("git_ref"),
            "commit_sha": p.get("commit_sha"),
        }
        for p in await app.skill_packs()
        if isinstance(p, dict)
    ]
    return tool_ids, packs, model


def run(
    config_path: Path,
    environ: Mapping[str, str],
    *,
    docker_client_factory: Callable[[], object],
    gateway_factory: Callable[[Settings], LangfuseGateway] | None = None,
    app_transport=None,
    instance_options: dict | None = None,
    play_options: dict | None = None,
    stdin=None,
) -> RunSummary:
    """``config_path`` is a file, or ``-`` to read the configuration from stdin."""
    settings = load_settings(environ)
    config: RunConfig = load_config(config_path, environ, stdin=stdin)
    gateway = (gateway_factory or _default_gateway)(settings)
    try:
        return _run(
            config,
            settings,
            gateway,
            docker_client_factory=docker_client_factory,
            app_transport=app_transport,
            instance_options=instance_options,
            play_options=play_options,
        )
    finally:
        gateway.flush()


def _read_dataset(gateway: LangfuseGateway, name: str):
    try:
        dataset = gateway.dataset(name)
    except Exception as exc:
        raise DatasetError(f"Dataset {name} could not be read from Langfuse: {exc}") from exc
    if not dataset.items:
        raise DatasetError(f"Dataset {name} has no items")
    return dataset


def _required_packs(dataset, name: str) -> list[str]:
    """The pack names in the dataset metadata's ``skill_packs``; none when the
    metadata or the key is absent."""
    metadata = getattr(dataset, "metadata", None)
    if metadata is None:
        return []
    if not isinstance(metadata, Mapping):
        raise DatasetError(f"Dataset {name} metadata must be an object")
    packs = metadata.get("skill_packs")
    if packs is None:
        return []
    if not isinstance(packs, list) or not all(isinstance(p, str) and p.strip() for p in packs):
        raise DatasetError(
            f"Dataset {name} metadata skill_packs must be a list of non-empty pack names"
        )
    return [p.strip() for p in packs]


def _check_packs(required: list[str], config: RunConfig, name: str) -> None:
    """Stop the run before the app starts when the configuration does not turn
    on a pack the dataset needs, so every run of a dataset plays every item."""
    on = config.skill_packs_on
    missing = [pack for pack in required if pack not in on]
    if missing:
        raise DatasetError(
            f"Dataset {name} needs skill packs {', '.join(missing)}, "
            "which the configuration does not turn on in SKILL_PACKS"
        )


def _run(
    config: RunConfig,
    settings: Settings,
    gateway: LangfuseGateway,
    *,
    docker_client_factory,
    app_transport,
    instance_options,
    play_options,
) -> RunSummary:
    dataset = _read_dataset(gateway, config.dataset)
    required_packs = _required_packs(dataset, config.dataset)
    _check_packs(required_packs, config, config.dataset)
    tasks = parse_tasks(dataset.items, settings.max_turns)
    dataset_version = getattr(dataset, "version", None)

    with AppInstance(
        docker_client=docker_client_factory(),
        image=config.image,
        runner_image=config.runner_image,
        pull=config.pull,
        env=config.env,
        network=settings.docker_network,
        port=config.port,
        start_timeout=settings.app_start_timeout,
        poll_interval=settings.poll_interval,
        **(instance_options or {}),
    ) as instance:

        def new_app() -> AppClient:
            return AppClient(instance.base_url, config.api_key, transport=app_transport)

        async def read_resolved():
            app = new_app()
            try:
                return await _resolved(app, config)
            finally:
                await app.aclose()

        tool_ids, packs, model = asyncio.run(read_resolved())
        served_model = gateway.served_model(model)
        metadata = {
            "config_name": config.name,
            "served_model": served_model,
            "image": config.image,
            "image_digest": instance.image_digest or "",
            "image_id": instance.image_id or "",
            "image_pulled": "true" if config.pull else "false",
            "runner_image": config.runner_image,
            "runner_image_digest": instance.runner_image_digest or "",
            "runner_image_id": instance.runner_image_id or "",
            "model": model,
            "catalog_model": config.model,
            "env_var_names": ",".join(config.env_names),
            "skill_packs": json.dumps(packs),
            "tool_ids": ",".join(tool_ids),
            "simulated_user_model": settings.simulated_user_model,
            "judge_model": settings.judge_model,
            "max_turns": str(settings.max_turns),
            "harness_commit": settings.harness_commit or "unrecorded",
            "batch": settings.batch,
            "dataset": config.dataset,
            "dataset_version": dataset_version.isoformat() if dataset_version else "latest",
            "dataset_items": str(len(tasks)),
            "dataset_skill_packs": ",".join(required_packs),
        }

        # The first runner failure; once set, no further task is played, and
        # it is raised after the experiment whether or not run_experiment
        # raised it from the task.
        runner_failure: list[RunnerStartError] = []
        # Each task the runner stopped or restarted during; the run goes on.
        runner_lost: list[str] = []
        # How many of the losses in runner_lost a wait that started after
        # them found the runner running steadily again.
        runner_back_after = 0

        async def task_fn(*, item, **kwargs):
            nonlocal runner_back_after
            if runner_failure:
                raise runner_failure[0]
            lost_before_wait = len(runner_lost)
            try:
                restarts = await asyncio.to_thread(
                    instance.wait_for_runner, f"before task {item.id}"
                )
            except RunnerStartError as exc:
                runner_failure.append(exc)
                raise
            runner_back_after = max(runner_back_after, lost_before_wait)
            # Another task may have stopped the run while this one waited.
            if runner_failure:
                raise runner_failure[0]
            app = new_app()
            try:
                result = await play(
                    tasks[item.id],
                    PlayDeps(
                        app=app,
                        model=config.model,
                        tool_ids=tool_ids,
                        simulated_user_model=settings.simulated_user_model,
                        judge_model=settings.judge_model,
                        max_turns=settings.max_turns,
                        complete=gateway.traced_completion,
                        chat_observations=gateway.chat_observations,
                        turn_timeout=settings.turn_timeout,
                        poll_interval=settings.poll_interval,
                        **(play_options or {}),
                    ),
                )
            finally:
                await app.aclose()
            # Before the runner check, so the item trace carries the chat as its
            # session even when the check fails, and delete finds the chat.
            if result.chat_id:
                gateway.link_item_to_chat(result.chat_id)
            # A runner that stopped or restarted during the play would score the
            # task as an agent failure. The error goes on the item instead, with
            # no scores. A runner Docker restarts leaves the run going, and a
            # later wait decides whether it goes on; any other state stops it.
            try:
                await asyncio.to_thread(
                    instance.check_runner_unchanged, f"during task {item.id}", restarts
                )
            except RunnerStartError as exc:
                if exc.restartable:
                    runner_lost.append(item.id)
                else:
                    runner_failure.append(exc)
                raise
            # Absent when the trace scores are missing; then nothing is recorded.
            app_cost = next((s.value for s in result.scores if s.name == COST_SCORE), None)
            if app_cost is not None:
                gateway.record_app_cost(app_cost, result.app_trace_ids)
            return result.output()

        try:
            result = gateway.run_experiment(
                dataset=dataset,
                name=served_model,
                description=model,
                task=task_fn,
                metadata=metadata,
                concurrency=settings.concurrency,
            )
        except Exception:
            # The runner failure is the cause; it is not hidden behind a later error.
            if runner_failure:
                raise runner_failure[0]  # noqa: B904 - the other error stays as its context
            raise
        if runner_failure:
            raise runner_failure[0]
        # No wait started after the latest loss found the runner back.
        if len(runner_lost) > runner_back_after:
            instance.wait_for_runner("after the last task")

    outputs = [r.output or {} for r in result.item_results]
    return RunSummary(
        experiment_name=served_model,
        run_name=result.run_name,
        experiment_url=result.experiment_url,
        items=len(result.item_results),
        goals_reached=sum(1 for o in outputs if o.get("goal_reached")),
        metadata=metadata,
        runner_lost=runner_lost,
    )


def _default_gateway(settings: Settings) -> LangfuseGateway:
    return LangfuseGateway(
        environment=settings.langfuse_environment,
        trace_timeout=settings.trace_timeout,
        trace_settle=settings.trace_settle,
        poll_interval=settings.poll_interval,
        model_retries=settings.model_retries,
        model_max_tokens=settings.model_max_tokens,
    )
