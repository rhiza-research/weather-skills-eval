"""The judge: per-turn automatic checks, then Scenario's judge on the task's criteria.

After every assistant reply the judge runs the task's artifact checks. While
any fails, the play continues. Once all pass, it asks Scenario's
``JudgeAgent`` for a verdict on the judge criteria. A passing verdict ends the
play as a success; a failing one lets the play continue, so the goal can
still be reached on a later turn. Plays that never reach the goal end at the
turn limit or when the simulated user gives up.
"""

from collections.abc import Awaitable, Callable

from scenario import AgentInput, JudgeAgent, JudgmentRequest, ScenarioResult

from .checks import CheckResult
from .llm import JUDGE, Complete, reply_json


class GoalJudge(JudgeAgent):
    name = "judge"

    def __init__(
        self,
        *,
        criteria: list[str],
        model: str,
        complete: Complete,
        session_id: Callable[[], str | None],
        run_checks: Callable[[], Awaitable[list[CheckResult]]],
    ):
        super().__init__(criteria=criteria, model=model, include_traces=False)
        self._complete_fn = complete
        self._session_id = session_id
        self._run_checks = run_checks
        self.last_checks: list[CheckResult] = []
        self.last_verdict: str | None = None

    def _completion(self, **kwargs):
        return self._complete_fn(JUDGE, self._session_id(), **kwargs)

    async def call(self, input: AgentInput):
        self.last_checks = await self._run_checks()
        failed = [r for r in self.last_checks if not r.passed]
        if failed:
            self.last_verdict = "Automatic checks failed: " + "; ".join(r.detail for r in failed)
            return []
        if not self.criteria:
            return ScenarioResult(
                success=True,
                messages=list(input.messages),
                reasoning="Every automatic check passed",
            )
        forced = input.model_copy(
            update={"judgment_request": JudgmentRequest(criteria=list(self.criteria))}
        )
        verdict = await super().call(forced)
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
    if not isinstance(score, int) or not 1 <= score <= 5:
        raise ValueError(f"Answer quality is not an integer from 1 to 5: {data!r}")
    return score, str(data.get("reasoning") or "")
