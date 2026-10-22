"""The judge: once the play has ended, Scenario's judge rules on the final result.

When the task expects files, the result is every file the assistant
presented during the play, one per path, at its latest presentation (see
``checks``); the criteria decide which of them the user wanted at the end.
A file of the last reply that presents files blocks the ruling when it has a
problem; an earlier file's problem is a note next to it instead. When no
result can be judged (nothing presented, a kind missing, or a last-reply
file missing from the chat's listing, unreadable or without provenance), the
verdict is that reason and no judge model is called. Otherwise the judge
rules on the task's criteria plus ``PROVENANCE_CRITERION``, from the visible
transcript and one message of harness evidence: each presented file's path,
kind, the turn that presented it and provenance (the listing's
``provenance_detail``) as JSON, and each image. A file no reply names other
than in backticks is labeled as named in passing and not presented, and the
provenance criterion does not cover it, unless it is of a kind the task
needs and no file of that kind was presented outright. A task that expects no file is
judged on the transcript and its criteria alone. The judge's scenario
description, and the answer-quality prompt, state when the play started.

The judge never ends a play: the simulated user or the turn cap does.
"""

import asyncio
import base64
import json
from collections.abc import Awaitable, Callable
from datetime import datetime

from scenario import AgentInput, JudgeAgent, JudgmentRequest, ScenarioResult
from scenario.scenario_state import ScenarioState

from .checks import Evidence
from .llm import JUDGE, Complete, ModelReplyError, reply_json

PROVENANCE_CRITERION = (
    "Each presented file's provenance (data source, region, period, aggregation, units) "
    "matches the user's request and what was agreed in the chat; a file named in passing "
    "and not presented is not held to this"
)

EVIDENCE_HEADER = (
    "Harness evidence, not part of the conversation: every file the assistant presented "
    "or named in passing during the conversation, in order, each once at the turn given with "
    "it, with the weather_skills_history provenance the app read from it: per branch, the "
    "skills that produced it in order, with their arguments."
)


def started_line(started: datetime) -> str:
    """The play's start in UTC, for the judge and the answer-quality rater: they
    see only the visible conversation, never the app's system prompt, so
    without it a request relative to today cannot be checked."""
    return (
        "Harness context, not said by the user: the conversation started on "
        f"{started:%Y-%m-%d} at {started:%H:%M} UTC."
    )


def evidence_message(evidence: Evidence) -> dict:
    """The judge's evidence; it is built only for evidence without a problem.
    A file presented before the last reply that presents files may carry a
    note of its own problem."""
    parts: list[dict] = [{"type": "text", "text": EVIDENCE_HEADER}]
    for file in evidence.files:
        if file.named_in_passing:
            how = f"named in passing on turn {file.turn} and not presented"
        else:
            how = f"presented on turn {file.turn}"
        text = f"File: {file.path} ({file.kind}), {how}"
        if file.kind == "image" and not file.shown_as_image:
            text += " (not shown as an image)"
        if file.note:
            text += f"\nNote: {file.note}"
        if file.provenance is not None:
            provenance = json.dumps(file.provenance, indent=2, sort_keys=True)
            text += f"\nProvenance:\n{provenance}"
        parts.append({"type": "text", "text": text})
        if file.image is not None and file.shown_as_image:
            encoded = base64.b64encode(file.image).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{file.media_type};base64,{encoded}"},
                }
            )
    return {"role": "user", "content": parts}


class GoalJudge(JudgeAgent):
    name = "judge"

    def __init__(
        self,
        *,
        criteria: list[str],
        deliverables: list[str],
        model: str,
        complete: Complete,
        session_id: Callable[[], str | None],
        collect_evidence: Callable[[], Awaitable[Evidence]],
    ):
        super().__init__(criteria=criteria, model=model, include_traces=False)
        self._deliverables = deliverables
        self._complete_fn = complete
        self._session_id = session_id
        self._collect_evidence = collect_evidence
        self._evidence: Evidence | None = None
        self.verdict: str | None = None

    def _completion(self, **kwargs):
        if self._evidence is not None:
            # After the system prompt and the transcript.
            messages = list(kwargs["messages"])
            messages.insert(2, evidence_message(self._evidence))
            kwargs["messages"] = messages
        return self._complete_fn(JUDGE, self._session_id(), **kwargs)

    async def rule(self, state: ScenarioState, *, at_turn_cap: bool) -> bool:
        """The verdict on the play's final result; its reasoning is ``verdict``.

        Called after ``scenario.arun`` returns, in the same context: Scenario's
        judge reads the run it left current there. Scenario's judge calls the
        model synchronously from its async code, so it runs on a thread with
        its own event loop, and other plays keep polling the app meanwhile;
        the thread gets a copy of this context. Raises ``ModelReplyError``
        when the judge returns no verdict.

        Scenario's judge tells the model the conversation reached the turn cap
        whenever the last turn was the cap's; unless the play ended at the
        turn cap, it gets a state whose ``max_turns`` is past the last turn.
        """
        # Verified against langwatch-scenario 1.6.0: JudgeAgent.call reads
        # scenario_state.config.max_turns and current_turn, and its cache reads
        # the run arun leaves current.
        if not at_turn_cap:
            config = state.config.model_copy(update={"max_turns": state.current_turn + 2})
            state = state.model_copy(update={"config": config})
        criteria = list(self.criteria)
        evidence = None
        if self._deliverables:
            evidence = await self._collect_evidence()
            if evidence.problem:
                self.verdict = evidence.problem
                return False
            criteria.append(PROVENANCE_CRITERION)
        request = AgentInput(
            thread_id=state.thread_id,
            messages=list(state.messages),
            new_messages=[],
            judgment_request=JudgmentRequest(criteria=criteria),
            scenario_state=state,
        )
        self._evidence = evidence

        async def verdict():
            # Cleared in the thread: a cancelled await leaves the call running.
            try:
                return await self.call(request)
            finally:
                self._evidence = None

        result = await asyncio.to_thread(asyncio.run, verdict())
        if not isinstance(result, ScenarioResult):
            raise ModelReplyError(f"The judge returned no verdict: {result!r}")
        self.verdict = result.reasoning or ("passed" if result.success else "no reasoning given")
        return bool(result.success)


QUALITY_PROMPT = """You rate how well a weather assistant answered a user, from 1 to 5.

5: complete, correct and clear; 4: good with small gaps; 3: usable but incomplete or \
unclear; 2: mostly unhelpful; 1: no answer or wrong.

{started}

The user's request:
{opening}

What the answer had to achieve:
{criteria}

The conversation:
{transcript}

Reply with one JSON object and nothing else: {{"score": an integer from 1 to 5, \
"reasoning": one or two sentences}}"""


def rate_answer(
    *,
    complete: Complete,
    model: str,
    session_id: str | None,
    started: datetime,
    opening: str,
    criteria: list[str],
    transcript: str,
) -> tuple[int, str]:
    response = complete(
        JUDGE,
        session_id,
        model=model,
        messages=[
            {
                "role": "user",
                "content": QUALITY_PROMPT.format(
                    started=started_line(started),
                    opening=opening,
                    criteria="\n".join(f"- {c}" for c in criteria) or "(the user's request)",
                    transcript=transcript,
                ),
            }
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    data = reply_json(response)
    score = data.get("score")
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError(f"Answer quality is not an integer from 1 to 5: {data!r}")
    return score, str(data.get("reasoning") or "")
