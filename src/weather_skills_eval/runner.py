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
5. Run the experiment, playing the tasks in a random order drawn for this
   run, which makes runs at once less likely to reach the same data
   providers with the same task at the same time: each item's task waits for
   the runner to be running
   (Docker restarts one that died), plays the task and returns its scores,
   which the evaluator records on the item. A runner that Docker restarted,
   is restarting, or exited with a non-zero code after the play leaves that
   task unscored, with the runner error on its item, and the run goes on;
   when no task waited for the runner after the latest such loss, it is
   waited for once after the last task. A runner that is gone, exited with
   code 0, is dead, or is not running steadily by the start timeout stops
   the run.
6. Remove the containers and the volume and flush Langfuse, whatever happened.

The run logs each step of starting it, and each task when it starts and when
it finishes, numbered out of the total in the order the tasks start.
"""

import asyncio
import copy
import hashlib
import importlib.metadata
import json
import logging
import random
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .app_client import AppClient, ChatError, highest_skill_versions
from .config import RunConfig, load_config
from .instance import AppInstance, RunnerStartError
from .judge import (
    EVIDENCE_HEADER,
    PROVENANCE_CRITERION,
    QUALITY_PROMPT,
    SCENARIO_DESCRIPTION,
    STARTED_LINE,
)
from .langfuse_gateway import LangfuseGateway
from .play import PlayDeps, play
from .scoring import COST_SCORE
from .settings import Settings, load_settings
from .simulated_user import (
    CONVERSATION_HEADER,
    FILES_HEADER,
    REPLY_REQUEST,
    RETRY_REQUEST,
)
from .simulated_user import SYSTEM_PROMPT as SIMULATED_USER_PROMPT
from .tasks import parse_tasks

log = logging.getLogger(__name__)

# Bumped by hand when how a play is graded changes in a way the prompt texts
# in the comparability hash do not show.
GRADING_VERSION = 1


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
    # The sum of the items' cost_to_goal_usd; an item without that score adds nothing.
    agent_cost_usd: float


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


def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def comparability_hash(
    *,
    judge_models: Iterable[str],
    simulated_user_model: str,
    items: Iterable,
    max_turns: int,
    model_max_tokens: int,
    turn_timeout: float,
) -> str:
    """The first 16 hex characters of the SHA-256 of what grades a run: two runs
    with the same hash were graded the same way. What is under test (the agent's
    model, the app and runner images, skill packs, the app's system prompt) is not
    in it. ``items`` are the dataset items, each with id, input, expected_output
    and metadata."""

    def value_of(item, key):
        return item.get(key) if isinstance(item, dict) else getattr(item, key, None)

    dataset = sorted(
        (
            {key: value_of(item, key) for key in ("id", "input", "expected_output", "metadata")}
            for item in items
        ),
        key=lambda item: str(item["id"]),
    )
    prompts = {
        "judge": [SCENARIO_DESCRIPTION, STARTED_LINE, PROVENANCE_CRITERION, EVIDENCE_HEADER],
        # Scenario's judge system prompt lives in this package.
        "langwatch_scenario": importlib.metadata.version("langwatch-scenario"),
        "simulated_user": [
            SIMULATED_USER_PROMPT,
            CONVERSATION_HEADER,
            FILES_HEADER,
            REPLY_REQUEST,
            RETRY_REQUEST,
        ],
        "answer_quality": QUALITY_PROMPT,
    }
    graded = {
        "judge_models": sorted(judge_models),
        "simulated_user_model": simulated_user_model,
        "dataset": dataset,
        "grading": {
            "version": GRADING_VERSION,
            "prompts": hashlib.sha256(_canonical(prompts)).hexdigest(),
        },
        "settings": {
            "max_turns": max_turns,
            "model_max_tokens": model_max_tokens,
            # A float, so 1800 and 1800.0 hash alike.
            "turn_timeout": float(turn_timeout),
        },
    }
    return hashlib.sha256(_canonical(graded)).hexdigest()[:16]


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


def duration(seconds: float) -> str:
    """Whole seconds as hours, minutes and seconds, leaving out each that is
    0: ``4m12s``, ``18m``, ``1h5s``; ``0s`` for none."""
    hours, rest = divmod(round(seconds), 3600)
    minutes, secs = divmod(rest, 60)
    parts = [f"{n}{unit}" for n, unit in ((hours, "h"), (minutes, "m"), (secs, "s")) if n]
    return "".join(parts) or "0s"


def first_line(text: str) -> str:
    lines = text.splitlines()
    return lines[0] if lines else ""


def outcome_result(output: dict) -> str:
    """What the finished line says of a task that returned: its outcome, and
    for ``chat_error`` or ``error`` the first line of its error."""
    error = output.get("error")
    return f"{output['outcome']}: {first_line(error)}" if error else output["outcome"]


def task_result(exc: BaseException) -> str:
    """What the finished line says of a task that raised instead of an outcome."""
    if isinstance(exc, RunnerStartError) and exc.restartable:
        return "not scored, the sandbox runner restarted during it"
    # The first line only: a runner error goes on with the runner's logs.
    message = first_line(str(exc))
    return f"raised {type(exc).__name__}: {message}" if message else f"raised {type(exc).__name__}"


class Progress:
    """Logs each task when it starts and when it finishes. Tasks are numbered
    in the order they start, so with concurrency above 1 a number is not the
    task's place in the dataset."""

    def __init__(self, total: int, run_started: float, clock: Callable[[], float]):
        self._total = total
        self._run_started = run_started
        self._clock = clock
        self._lock = threading.Lock()
        self._started = 0
        self._finished = 0

    def start(self, task_id: str) -> tuple[int, float]:
        """The task's number and the time it started."""
        with self._lock:
            self._started += 1
            number = self._started
        log.info("Task %d of %d started: %s", number, self._total, task_id)
        return number, self._clock()

    def skip(self, task_id: str) -> None:
        """A task not played because the sandbox runner had already failed."""
        with self._lock:
            self._started += 1
            number = self._started
        log.info(
            "Task %d of %d skipped: %s, the sandbox runner failed", number, self._total, task_id
        )

    def finish(self, number: int, task_id: str, result: str, started: float) -> None:
        """``result`` is the outcome, or what happened to a task that raised."""
        with self._lock:
            self._finished += 1
            finished = self._finished
        now = self._clock()
        log.info(
            "Task %d of %d finished: %s, %s, %s; %d of %d finished, %s elapsed",
            number,
            self._total,
            task_id,
            result,
            duration(now - started),
            finished,
            self._total,
            duration(now - self._run_started),
        )


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
    clock: Callable[[], float] = time.monotonic,
    rng: random.Random | None = None,
) -> RunSummary:
    """``config_path`` is a file, or ``-`` to read the configuration from stdin.
    ``clock`` times the tasks and the run for the progress log. ``rng``
    shuffles the order the tasks are played in; an unseeded one when None."""
    run_started = clock()
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
            clock=clock,
            run_started=run_started,
            rng=rng or random.Random(),
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


def in_play_order(dataset, rng: random.Random):
    """A copy of ``dataset`` whose ``items``, the list its ``run_experiment``
    starts in order, are shuffled by ``rng``; ``dataset`` is left as read."""
    played = copy.copy(dataset)
    played.items = list(dataset.items)
    rng.shuffle(played.items)
    return played


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
    clock: Callable[[], float],
    run_started: float,
    rng: random.Random,
) -> RunSummary:
    dataset = _read_dataset(gateway, config.dataset)
    log.info("Dataset %s read: %d items", config.dataset, len(dataset.items))
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
        log.info("OpenRouter serves %s as %s", model, served_model)
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
            "judge_models": ",".join(sorted(settings.judge_models)),
            "comparability_hash": comparability_hash(
                judge_models=settings.judge_models,
                simulated_user_model=settings.simulated_user_model,
                items=dataset.items,
                max_turns=settings.max_turns,
                model_max_tokens=settings.model_max_tokens,
                turn_timeout=settings.turn_timeout,
            ),
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
        progress = Progress(len(tasks), run_started, clock)

        async def task_fn(*, item, **kwargs):
            if runner_failure:
                progress.skip(item.id)
                raise runner_failure[0]
            number, started = progress.start(item.id)
            try:
                output = await play_task(item)
            except Exception as exc:
                progress.finish(number, item.id, task_result(exc), started)
                raise
            progress.finish(number, item.id, outcome_result(output), started)
            return output

        async def play_task(item):
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
                        judge_models=list(settings.judge_models),
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

        played = in_play_order(dataset, rng)
        log.info(
            "Experiment %s started: %d tasks, concurrency %d",
            served_model,
            len(tasks),
            settings.concurrency,
        )
        try:
            result = gateway.run_experiment(
                dataset=played,
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
            log.info("Waiting for the runner after the last task")
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
        agent_cost_usd=sum(
            score["value"]
            for o in outputs
            for score in o.get("scores", [])
            if score.get("name") == COST_SCORE
        ),
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
