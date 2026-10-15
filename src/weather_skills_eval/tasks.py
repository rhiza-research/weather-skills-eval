"""The dataset item format: one task per Langfuse dataset item.

- ``input``: the opening message the simulated user sends, as a string.
- ``expected_output``: how the goal is checked. ``artifact_checks`` are
  automatic checks on the chat's files; ``judge_criteria`` are read by the
  judge. The goal is reached when every check and every criterion passes.
- ``metadata``: what the simulated user knows and how the play runs:
  ``facts`` (the facts sheet the user answers from), ``hidden_steps`` (hints
  the user reveals only when the assistant is stuck), ``pauses`` (waits
  before given turns), ``turn_limit`` and ``tags``.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class TaskError(Exception):
    """A dataset item does not follow the task format. The message names the item."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactCheck(_Strict):
    """Passes when at least one chat file matching ``path`` meets every condition."""

    path: str = Field(min_length=1, description="Glob matched against the file's chat path")
    kind: Literal["file", "image", "zarr", "zarr_view", "directory"] | None = None
    variables: list[str] = Field(default_factory=list, description="Zarr data variables")
    dims: list[str] = Field(default_factory=list, description="Zarr dimensions")
    contains: str | None = Field(default=None, description="Text the file content contains")


class ExpectedOutput(_Strict):
    artifact_checks: list[ArtifactCheck] = Field(default_factory=list)
    judge_criteria: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _has_a_goal(self):
        if not self.artifact_checks and not self.judge_criteria:
            raise ValueError("needs at least one artifact check or judge criterion")
        return self


class Pause(_Strict):
    before_turn: int = Field(ge=2, description="The pause runs before this turn (1 is the first)")
    seconds: float = Field(gt=0)


class TaskMetadata(_Strict):
    facts: str = Field(min_length=1)
    hidden_steps: list[str] = Field(default_factory=list)
    pauses: list[Pause] = Field(default_factory=list)
    turn_limit: int = Field(ge=1)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _pauses_fit(self):
        turns = [p.before_turn for p in self.pauses]
        if len(turns) != len(set(turns)):
            raise ValueError("pauses name the same turn twice")
        for pause in self.pauses:
            if pause.before_turn > self.turn_limit:
                raise ValueError(f"pause before turn {pause.before_turn} is past the turn limit")
        return self


class Task(_Strict):
    id: str
    opening_message: str = Field(min_length=1)
    expected: ExpectedOutput
    metadata: TaskMetadata

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
    if not isinstance(opening, str):
        raise TaskError(f"dataset item {item_id}: input must be the opening message as a string")
    parts = {}
    for field, model in (("expected_output", ExpectedOutput), ("metadata", TaskMetadata)):
        try:
            parts[field] = model.model_validate(get(field) or {})
        except ValidationError as exc:
            problems = "; ".join(
                ".".join([field, *(str(p) for p in error["loc"])]) + f": {error['msg']}"
                for error in exc.errors()
            )
            raise TaskError(f"dataset item {item_id}: {problems}") from exc
    return Task(
        id=item_id,
        opening_message=opening,
        expected=parts["expected_output"],
        metadata=parts["metadata"],
    )
