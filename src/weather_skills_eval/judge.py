"""The judge: after every assistant reply, Scenario's judge rules on the goal.

When the task expects files, the result is the files the assistant presents
(see ``checks``). While no result can be judged (nothing presented, a kind
missing, a file unreadable or without provenance), the play continues and
no judge model is called. Otherwise the judge rules on the task's criteria
plus ``PROVENANCE_CRITERION``, from the visible transcript and one message
of harness evidence: each presented file's path, kind and provenance (the
listing's ``provenance_detail``) as JSON, and each image. A task that expects no file is judged on the
transcript and its criteria alone.

A passing verdict ends the play as a success; a failing one lets the play
continue, so the goal can still be reached on a later turn. Plays that never
reach the goal end at the turn limit or when the simulated user gives up.
"""

import base64
import json
from collections.abc import Awaitable, Callable

from scenario import AgentInput, JudgeAgent, JudgmentRequest, ScenarioResult

from .checks import Evidence
from .llm import JUDGE, Complete, reply_json

PROVENANCE_CRITERION = (
    "Each presented file's provenance (data source, region, period, aggregation, units) "
    "matches the user's request and what was agreed in the chat"
)

EVIDENCE_HEADER = (
    "Harness evidence, not part of the conversation: the files the assistant presented "
    "as its result, each with the weather_skills_history provenance the app read from it: "
    "per branch, the skills that produced it in order, with their arguments."
)


def evidence_message(evidence: Evidence) -> dict:
    parts: list[dict] = [{"type": "text", "text": EVIDENCE_HEADER}]
    for file in evidence.files:
        provenance = json.dumps(file.provenance, indent=2, sort_keys=True)
        parts.append(
            {
                "type": "text",
                "text": f"File: {file.path} ({file.kind})\nProvenance:\n{provenance}",
            }
        )
        if file.image is not None:
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
        self.last_verdict: str | None = None

    def _completion(self, **kwargs):
        if self._evidence is not None:
            # After the system prompt and the transcript.
            messages = list(kwargs["messages"])
            messages.insert(2, evidence_message(self._evidence))
            kwargs["messages"] = messages
        return self._complete_fn(JUDGE, self._session_id(), **kwargs)

    async def call(self, input: AgentInput):
        criteria = list(self.criteria)
        evidence = None
        if self._deliverables:
            evidence = await self._collect_evidence()
            if evidence.problem:
                self.last_verdict = evidence.problem
                return []
            criteria.append(PROVENANCE_CRITERION)
        forced = input.model_copy(update={"judgment_request": JudgmentRequest(criteria=criteria)})
        self._evidence = evidence
        try:
            verdict = await super().call(forced)
        finally:
            self._evidence = None
        if isinstance(verdict, ScenarioResult):
            self.last_verdict = verdict.reasoning
            if verdict.success:
                return verdict
        return []


QUALITY_PROMPT = """You rate how well a weather assistant answered a user, from 1 to 5.

5: complete, correct and clear; 4: good with small gaps; 3: usable but incomplete or \
unclear; 2: mostly unhelpful; 1: no answer or wrong.

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
