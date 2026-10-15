"""A run: one configuration, every task in the dataset played once, one Langfuse experiment.

Order of a run:
1. Read the harness settings and the configuration file; stop on any error.
2. Read the dataset and check every item against the task format.
3. Start the app container and wait for /health.
4. Read the resolved configuration from the app (tools, skill pack commits).
5. Run the experiment: each item's task plays the task and returns its
   scores, which the evaluator records on the item.
6. Remove the container and flush Langfuse, whatever happened.
"""

import asyncio
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .app_client import AppClient, highest_skill_versions
from .config import RunConfig, load_config
from .instance import AppInstance
from .langfuse_gateway import LangfuseGateway
from .play import PlayDeps, play
from .settings import Settings, load_settings
from .tasks import parse_tasks


class DatasetError(Exception):
    """The dataset is missing, cannot be read, or has no items."""


@dataclass
class RunSummary:
    experiment_name: str
    run_name: str
    experiment_url: str | None
    items: int
    goals_reached: int
    metadata: dict


async def _resolved(app: AppClient) -> tuple[list[str], list[dict]]:
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
    return tool_ids, packs


def run(
    config_path: Path,
    environ: Mapping[str, str],
    *,
    docker_client_factory: Callable[[], object],
    gateway_factory: Callable[[Settings], LangfuseGateway] | None = None,
    app_transport=None,
    instance_options: dict | None = None,
    play_options: dict | None = None,
) -> RunSummary:
    settings = load_settings(environ)
    config: RunConfig = load_config(config_path, environ)
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
    tasks = parse_tasks(dataset.items)
    dataset_version = getattr(dataset, "version", None)

    with AppInstance(
        docker_client=docker_client_factory(),
        image=config.image,
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
                return await _resolved(app)
            finally:
                await app.aclose()

        tool_ids, packs = asyncio.run(read_resolved())
        metadata = {
            "config_name": config.name,
            "image": config.image,
            "image_digest": instance.image_digest or "",
            "model": config.model,
            "env_var_names": ",".join(config.env_names),
            "skill_packs": json.dumps(packs),
            "tool_ids": ",".join(tool_ids),
            "simulated_user_model": settings.simulated_user_model,
            "judge_model": settings.judge_model,
            "harness_commit": settings.harness_commit or "unrecorded",
            "dataset": config.dataset,
            "dataset_version": dataset_version.isoformat() if dataset_version else "latest",
            "dataset_items": str(len(tasks)),
        }

        async def task_fn(*, item, **kwargs):
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
                        complete=gateway.traced_completion,
                        chat_observations=gateway.chat_observations,
                        turn_timeout=settings.turn_timeout,
                        poll_interval=settings.poll_interval,
                        **(play_options or {}),
                    ),
                )
            finally:
                await app.aclose()
            if result.chat_id:
                gateway.link_item_to_chat(result.chat_id)
            return result.output()

        result = gateway.run_experiment(
            dataset=dataset,
            name=config.name,
            task=task_fn,
            metadata=metadata,
            concurrency=settings.concurrency,
        )

    outputs = [r.output or {} for r in result.item_results]
    return RunSummary(
        experiment_name=config.name,
        run_name=result.run_name,
        experiment_url=result.experiment_url,
        items=len(result.item_results),
        goals_reached=sum(1 for o in outputs if o.get("goal_reached")),
        metadata=metadata,
    )


def _default_gateway(settings: Settings) -> LangfuseGateway:
    return LangfuseGateway(
        environment=settings.langfuse_environment,
        trace_timeout=settings.trace_timeout,
        trace_settle=settings.trace_settle,
        poll_interval=settings.poll_interval,
        model_retries=settings.model_retries,
    )
