"""The dataset item format: one task per Langfuse dataset item.

- ``input``: the opening message the simulated user sends, as a string.
- ``expected_output``: how the goal is checked. ``deliverables`` are the
  kinds of file the result must include (``image``, ``zarr``); when it is
  not empty, the files the assistant presents are the result, and each must
  carry provenance. ``judge_criteria`` are read by the judge. The goal is
  reached when the simulated user is done and the judge passes every
  criterion.
- ``metadata``: what the simulated user knows and how the play runs:
  ``facts`` (the facts sheet the user answers from), ``hidden_steps`` (hints
  the user reveals only when the assistant is stuck), ``pauses`` (waits
  before given turns), ``turn_limit`` (optional; overrides the harness's
  ``EVAL_MAX_TURNS`` for the task) and ``tags``.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class TaskError(Exception):
    """A dataset item does not follow the task format. The message names the item."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Deliverable = Literal["image", "zarr"]


class ExpectedOutput(_Strict):
    deliverables: list[Deliverable] = Field(
        default_factory=list, description="Kinds of file the presented result must include"
    )
    judge_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_a_goal(self):
        if not self.deliverables and not self.judge_criteria:
            raise ValueError("needs at least one deliverable or judge criterion")
        return self


class Pause(_Strict):
    before_turn: int = Field(ge=2, description="The pause runs before this turn (1 is the first)")
    seconds: float = Field(gt=0)


class TaskMetadata(_Strict):
    facts: str = Field(min_length=1)
    hidden_steps: list[str] = Field(default_factory=list)
    pauses: list[Pause] = Field(default_factory=list)
    turn_limit: int | None = Field(default=None, ge=1)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _pauses_fit(self):
        turns = [p.before_turn for p in self.pauses]
        if len(turns) != len(set(turns)):
            raise ValueError("pauses name the same turn twice")
        if self.turn_limit is not None:
            _check_pauses(self.pauses, self.turn_limit, "the turn limit")
        return self


def _check_pauses(pauses: list[Pause], turns: int, limit: str) -> None:
    for pause in pauses:
        if pause.before_turn > turns:
            raise ValueError(f"pause before turn {pause.before_turn} is past {limit}")


class Task(_Strict):
    id: str
    opening_message: str = Field(min_length=1)
    expected: ExpectedOutput
    metadata: TaskMetadata

    def max_turns(self, cap: int) -> int:
        """The most user messages the play sends: ``turn_limit``, else ``cap``."""
        return self.metadata.turn_limit or cap

    def pause_before(self, turn: int) -> float:
        for pause in self.metadata.pauses:
            if pause.before_turn == turn:
                return pause.seconds
        return 0.0


def parse_task(item) -> Task:
    """Build a task from a dataset item (an object or a dict with input/expected_output/metadata)."""
    get = item.get if isinstance(item, dict) else lambda key: getattr(item, key, None)
    item_id = str(get("id") or "<no id>")
    opening = get("input")
    if not isinstance(opening, str) or not opening.strip():
        raise TaskError(
            f"dataset item {item_id}: input must be the opening message as a non-empty string"
        )
    parts = {}
    for field, model in (("expected_output", ExpectedOutput), ("metadata", TaskMetadata)):
        try:
            parts[field] = model.model_validate(get(field) or {})
        except ValidationError as exc:
            raise TaskError(f"dataset item {item_id}: {_problems(field, exc)}") from exc
    try:
        return Task(
            id=item_id,
            opening_message=opening,
            expected=parts["expected_output"],
            metadata=parts["metadata"],
        )
    except ValidationError as exc:
        raise TaskError(f"dataset item {item_id}: {_problems('input', exc)}") from exc


def _problems(field: str, exc: ValidationError) -> str:
    return "; ".join(
        ".".join([field, *(str(p) for p in error["loc"])]) + f": {error['msg']}"
        for error in exc.errors()
    )


def parse_tasks(items, max_turns: int) -> dict[str, Task]:
    """All tasks of a dataset, keyed by item id. Item ids must be unique. The
    pauses of a task without ``turn_limit`` must fit in ``max_turns``
    (``EVAL_MAX_TURNS``)."""
    tasks: dict[str, Task] = {}
    for item in items:
        task = parse_task(item)
        if task.id in tasks:
            raise TaskError(f"dataset item {task.id}: the id appears more than once")
        if task.metadata.turn_limit is None:
            try:
                _check_pauses(task.metadata.pauses, max_turns, "EVAL_MAX_TURNS")
            except ValueError as exc:
                raise TaskError(f"dataset item {task.id}: metadata.pauses: {exc}") from exc
        tasks[task.id] = task
    return tasks
