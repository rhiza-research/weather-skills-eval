"""One play: one task played once through the app's chat API with LangWatch Scenario.

Each Scenario turn is: the simulated user speaks, the app answers, the judge
checks the goal. Scenario's ``max_turns`` is the task's turn limit, so a play
that has not reached the goal by then ends as a failure. A pause scheduled
before turn k runs at the start of that turn, before the user speaks; it is
outside every app answer, so the time to answer does not include it.

Outcomes: ``goal_reached``; ``turn_limit`` (the limit passed without the
goal); ``user_gave_up``; ``chat_error`` (the app returned an error, or a
reply the harness cannot read); ``error`` (anything else stopped the play,
such as a harness model reply that cannot be used); ``not_reached`` (the
play ended without the goal for none of these reasons).
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import scenario
from scenario import AgentAdapter, AgentInput, AgentRole

from .app_client import AppClient, Conversation, visible_text
from .checks import collect_evidence, presented_paths
from .judge import GoalJudge, rate_answer
from .llm import Complete
from .scoring import PlayFacts, Score, TraceMetrics, build_scores, trace_metrics
from .simulated_user import SimulatedUser
from .tasks import Task

ChatObservations = Callable[[str, int], Awaitable[tuple[list[dict], bool]]]


class AppAgent(AgentAdapter):
    """The agent under test: the app, reached over HTTP. Keeps the exchanges
    that got a reply, so the conversation survives a play that errors, and
    the files presented by the latest reply that presents any."""

    role = AgentRole.AGENT
    name = "weather-skills-chat"

    def __init__(self, conversation: Conversation):
        self._conversation = conversation
        self.exchanges: list[tuple[str, str]] = []
        self.presented: list[str] = []

    async def call(self, input: AgentInput):
        text = input.last_new_user_message_str()
        answer = await self._conversation.send(text)
        presented = presented_paths(answer.content)
        if presented:
            self.presented = presented
        reply = visible_text(answer.content) or "(the reply has no text)"
        self.exchanges.append((text, reply))
        return reply


def neutral_transcript(exchanges: list[tuple[str, str]]) -> str:
    return "\n\n".join(f"User: {user}\n\nAssistant: {reply}" for user, reply in exchanges)


@dataclass
class PlayResult:
    task_id: str
    chat_id: str | None
    outcome: str
    goal_reached: bool
    reasoning: str
    error: str | None
    scores: list[Score]
    trace_scores: str
    app_trace_ids: list[str] = field(default_factory=list)
    paused_seconds: float = 0.0

    def output(self) -> dict:
        return {
            "chat_id": self.chat_id,
            "outcome": self.outcome,
            "goal_reached": self.goal_reached,
            "reasoning": self.reasoning,
            "error": self.error,
            "trace_scores": self.trace_scores,
            "app_trace_ids": self.app_trace_ids,
            "paused_seconds": self.paused_seconds,
            "scores": [
                {"name": s.name, "value": s.value, "comment": s.comment} for s in self.scores
            ],
        }


@dataclass
class PlayDeps:
    app: AppClient
    model: str
    tool_ids: list[str]
    simulated_user_model: str
    judge_model: str
    complete: Complete
    chat_observations: ChatObservations
    turn_timeout: float
    poll_interval: float
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep


async def play(task: Task, deps: PlayDeps) -> PlayResult:
    conversation = Conversation(
        deps.app,
        model=deps.model,
        tool_ids=deps.tool_ids,
        turn_timeout=deps.turn_timeout,
        poll_interval=deps.poll_interval,
        clock=deps.clock,
        sleep=deps.sleep,
    )

    def session_id() -> str | None:
        return conversation.chat_id

    agent = AppAgent(conversation)
    user = SimulatedUser(
        task=task,
        model=deps.simulated_user_model,
        complete=deps.complete,
        session_id=session_id,
    )
    judge = GoalJudge(
        criteria=list(task.expected.judge_criteria),
        deliverables=list(task.expected.deliverables),
        model=deps.judge_model,
        complete=deps.complete,
        session_id=session_id,
        collect_evidence=lambda: collect_evidence(
            deps.app, conversation.chat_id, agent.presented, list(task.expected.deliverables)
        ),
    )
    paused = 0.0

    async def before_turn(state) -> None:
        nonlocal paused
        # Scenario counts turns from 0 and calls this after starting a new
        # one, before checking the turn limit.
        if state.current_turn >= task.metadata.turn_limit:
            return
        seconds = task.pause_before(state.current_turn + 1)
        if seconds:
            await deps.sleep(seconds)
            paused += seconds

    error: str | None = None
    try:
        result = await scenario.arun(
            name=task.id,
            description=(
                f"A user asks a weather assistant: {task.opening_message}\n\n"
                f"What the user knows:\n{task.metadata.facts}"
            ),
            agents=[agent, user, judge],
            script=[scenario.proceed(on_turn=before_turn)],
            max_turns=task.metadata.turn_limit,
            verbose=False,
        )
        goal_reached = bool(result.success)
        reasoning = result.reasoning or ""
    except Exception as exc:
        goal_reached = False
        error = conversation.error or f"{type(exc).__name__}: {exc}"
        reasoning = error
    if goal_reached:
        outcome = "goal_reached"
    elif error is not None:
        outcome = "chat_error" if conversation.error else "error"
    elif user.gave_up:
        outcome = "user_gave_up"
    elif conversation.turns >= task.metadata.turn_limit:
        outcome = "turn_limit"
    else:
        outcome = "not_reached"
    if outcome in ("turn_limit", "user_gave_up", "not_reached") and judge.last_verdict:
        reasoning = f"{reasoning} (last check: {judge.last_verdict})"

    quality: int | None = None
    quality_reasoning = ""
    if agent.exchanges:
        try:
            quality, quality_reasoning = await asyncio.to_thread(
                rate_answer,
                complete=deps.complete,
                model=deps.judge_model,
                session_id=conversation.chat_id,
                opening=task.opening_message,
                criteria=list(task.expected.judge_criteria),
                transcript=neutral_transcript(agent.exchanges),
            )
        except Exception as exc:
            quality_reasoning = f"answer quality not rated: {exc}"

    facts = PlayFacts(
        goal_reached=goal_reached,
        outcome=outcome,
        turns=conversation.turns,
        clarifying_questions=user.clarifying_questions,
        hints_used=user.hints_used,
        time_to_answer_seconds=sum(conversation.answer_seconds),
        answer_quality=quality,
        answer_quality_reasoning=quality_reasoning,
    )

    metrics: TraceMetrics | None = None
    traces_complete = False
    problem: str | None = None
    if conversation.chat_id is None or conversation.turns_reached == 0:
        problem = "no turn reached the app"
    else:
        try:
            observations, traces_complete = await deps.chat_observations(
                conversation.chat_id, conversation.turns_reached
            )
        except Exception as exc:
            observations = []
            problem = f"Langfuse could not be read: {exc}"
        if observations:
            metrics = trace_metrics(observations)
        elif problem is None:
            problem = "no app traces arrived"

    if metrics is None:
        trace_scores = f"missing: {problem}"
    else:
        trace_scores = "complete" if traces_complete else "partial: traces were still arriving"

    return PlayResult(
        task_id=task.id,
        chat_id=conversation.chat_id,
        outcome=outcome,
        goal_reached=goal_reached,
        reasoning=reasoning,
        error=error,
        scores=build_scores(facts, metrics, traces_complete, problem),
        trace_scores=trace_scores,
        app_trace_ids=metrics.trace_ids if metrics else [],
        paused_seconds=paused,
    )
