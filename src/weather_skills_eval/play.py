"""One play: one task played once through the app's chat API with LangWatch Scenario.

Each Scenario turn is: the simulated user speaks, the app answers. The play
ends when the simulated user is done or gives up, or when the turn cap
passes: the task's ``turn_limit``, else ``EVAL_MAX_TURNS``, given to Scenario
as ``max_turns``. At the cap the simulated user reads the last reply once
more, outside Scenario's run, so it can still be done or give up; nothing
more is sent to the app. The judge panel then rules once on the final result: every
judge on the same evidence, the majority of the judges that ruled deciding; a judge
whose call failed is recorded as errored and left out. A pause scheduled before turn
k runs at the start of that turn, before the user speaks; it is outside every
app answer, so the time to answer does not include it.

Outcomes: ``goal_reached`` (the user is done and the panel passed);
``chat_error`` (the app returned an error, or a reply the harness cannot
read, a request to the app failed without a response, or the reply was not
done within ``EVAL_TURN_TIMEOUT_SECONDS``); ``error`` (anything else stopped the play, such as a harness model
reply that cannot be used, or every judge call failed); ``user_done`` (the
user is done, and the panel failed or no result could be judged);
``user_gave_up``; ``turn_cap`` (the play sent as many user messages as the cap allows, and the user was neither
done nor gave up); ``not_reached`` (the run ended for none of these reasons).
"""

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime

import scenario
from scenario import AgentAdapter, AgentInput, AgentRole, ScenarioResult
from scenario.scenario_state import ScenarioState

from .app_client import AppClient, Conversation, visible_text
from .checks import Evidence, ReplyFiles, collect_evidence, reply_files
from .judge import GoalJudge, PanelRuling, rate_answer, rule_panel, scenario_description
from .llm import Complete
from .scoring import (
    PlayFacts,
    Score,
    TraceMetrics,
    TraceRead,
    build_scores,
    judge_label,
    trace_metrics,
)
from .simulated_user import SimulatedUser
from .tasks import Task

ChatObservations = Callable[[str, int], Awaitable[TraceRead]]


class AppAgent(AgentAdapter):
    """The agent under test: the app, reached over HTTP. Keeps the exchanges
    that got a reply, so the conversation survives a play that errors, and
    the files each reply names, in order. Which of them a reply presents is
    known only once the chat's file listing is read (see ``checks``)."""

    role = AgentRole.AGENT
    name = "weather-skills-chat"

    def __init__(self, conversation: Conversation):
        self._conversation = conversation
        self.exchanges: list[tuple[str, str]] = []
        self.replies: list[ReplyFiles] = []

    async def call(self, input: AgentInput):
        text = input.last_new_user_message_str()
        answer = await self._conversation.send(text)
        files = reply_files(answer.content, len(self.exchanges) + 1)
        if files.paths:
            self.replies.append(files)
        reply = visible_text(answer.content) or "(the reply has no text)"
        self.exchanges.append((text, reply))
        return reply


def last_reply_input(state: ScenarioState) -> AgentInput:
    """The simulated user's input for reading the reply to the last allowed turn."""
    return AgentInput(
        thread_id=state.thread_id,
        messages=list(state.messages),
        new_messages=[],
        scenario_state=state,
    )


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
    provider_models: list[str] = field(default_factory=list)
    paused_seconds: float = 0.0
    # Each judge's model, state, verdict and reasoning; empty when the panel did not rule.
    judges: list[dict] = field(default_factory=list)

    def output(self) -> dict:
        return {
            "chat_id": self.chat_id,
            "outcome": self.outcome,
            "goal_reached": self.goal_reached,
            "reasoning": self.reasoning,
            "error": self.error,
            "judges": self.judges,
            "trace_scores": self.trace_scores,
            "app_trace_ids": self.app_trace_ids,
            "provider_models": self.provider_models,
            "paused_seconds": self.paused_seconds,
            "scores": [
                {"name": s.name, "value": s.value, "comment": s.comment} for s in self.scores
            ],
        }


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass
class PlayDeps:
    app: AppClient
    model: str
    tool_ids: list[str]
    simulated_user_model: str
    judge_models: list[str]
    # EVAL_MAX_TURNS, for tasks without a turn_limit.
    max_turns: int
    complete: Complete
    chat_observations: ChatObservations
    turn_timeout: float
    poll_interval: float
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    # The UTC date and time, told to the judge and the rater as the play's start.
    now: Callable[[], datetime] = _utc_now


async def play(task: Task, deps: PlayDeps) -> PlayResult:
    started = deps.now()
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
    max_turns = task.max_turns(deps.max_turns)

    # The user's evidence, the latest presenting reply's files, as (replies
    # that name files, evidence): collected again only after a reply names
    # files, so each presented image is downloaded once per presenting reply.
    collected: tuple[int, Evidence] | None = None
    # Images downloaded for the user, by path and the turn that presented them.
    downloaded: dict[tuple[str, int], bytes] = {}

    async def evidence() -> Evidence:
        nonlocal collected
        if collected is not None and collected[0] == len(agent.replies):
            return collected[1]
        found = await collect_evidence(
            deps.app,
            conversation.chat_id,
            list(agent.replies),
            list(task.expected.deliverables),
            known_images=downloaded,
        )
        downloaded.update({(f.path, f.turn): f.image for f in found.files if f.image is not None})
        # Dropped when the listing or a download failed, or a path is not
        # listed yet: the listing and the failed downloads are read again on
        # the next turn, and the images already downloaded for the same
        # presentation are reused by path and turn. While a backtick path in
        # the latest reply naming files is unlisted, the evidence is read
        # again each turn, at the cost of one listing per turn.
        stale = found.read_failed or found.unlisted or found.unlisted_candidates
        collected = None if stale else (len(agent.replies), found)
        return found

    user = SimulatedUser(
        task=task,
        model=deps.simulated_user_model,
        complete=deps.complete,
        session_id=session_id,
        collect_evidence=evidence,
    )

    # The judges' evidence, read once after the play and shared by every judge.
    final_evidence: asyncio.Future[Evidence] | None = None

    def every_presented_file() -> Awaitable[Evidence]:
        nonlocal final_evidence
        if final_evidence is None:
            final_evidence = asyncio.ensure_future(
                collect_evidence(
                    deps.app,
                    conversation.chat_id,
                    list(agent.replies),
                    list(task.expected.deliverables),
                    every=True,
                    # A failed listing or download is tried once more, after the poll
                    # interval, before it counts.
                    retries=1,
                    wait=lambda: deps.sleep(deps.poll_interval),
                )
            )
        return final_evidence

    judges = [
        GoalJudge(
            criteria=list(task.expected.judge_criteria),
            deliverables=list(task.expected.deliverables),
            model=model,
            complete=deps.complete,
            session_id=session_id,
            collect_evidence=every_presented_file,
        )
        for model in deps.judge_models
    ]
    panel: PanelRuling | None = None
    paused = 0.0
    # The run's state, which the judges rule on once the run has ended.
    final_state = None

    async def before_turn(state) -> None:
        nonlocal paused, final_state
        final_state = state
        # Scenario counts turns from 0 and calls this after starting a new
        # one, before checking the turn cap.
        if state.current_turn >= max_turns:
            return
        seconds = task.pause_before(state.current_turn + 1)
        if seconds:
            await deps.sleep(seconds)
            paused += seconds

    error: str | None = None
    passed = False
    try:
        result = await scenario.arun(
            name=task.id,
            description=scenario_description(task.opening_message, task.metadata.facts, started),
            agents=[agent, user],
            script=[scenario.proceed(on_turn=before_turn)],
            max_turns=max_turns,
            verbose=False,
        )
        reasoning = result.reasoning or ""
    except Exception as exc:
        error = conversation.error or f"{type(exc).__name__}: {exc}"
        reasoning = error
    reached_cap = conversation.turns >= max_turns
    if error is None and final_state is not None and reached_cap and not user.ended:
        try:
            last = await user.read_last_reply(last_reply_input(final_state))
            if isinstance(last, ScenarioResult):
                reasoning = last.reasoning or ""
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            reasoning = error
    at_turn_cap = reached_cap and not user.ended
    if error is None and final_state is None:
        error = "Scenario ended the run before any turn, so there is nothing to judge"
        reasoning = error
    if error is None:
        try:
            panel = await rule_panel(judges, final_state, at_turn_cap=at_turn_cap)
        except Exception as exc:
            error = f"The final judge call failed: {type(exc).__name__}: {exc}"
            reasoning = error
        else:
            if panel.error is not None:
                error = f"No judge ruled: {panel.error}"
                reasoning = error
            passed = panel.passed
    goal_reached = error is None and user.done and passed
    verdict = panel.verdict if panel is not None else ""
    if goal_reached:
        outcome = "goal_reached"
        reasoning = verdict
    elif error is not None:
        outcome = "chat_error" if conversation.error else "error"
    elif user.done:
        outcome = "user_done"
    elif user.gave_up:
        outcome = "user_gave_up"
    elif at_turn_cap:
        outcome = "turn_cap"
    else:
        outcome = "not_reached"
    if outcome not in ("goal_reached", "chat_error", "error"):
        result_word = "passed" if passed else "failed"
        reasoning = f"{reasoning} (final check {result_word}: {verdict})"

    async def rate(model: str) -> tuple[str, int | None, str]:
        try:
            score, why = await asyncio.to_thread(
                rate_answer,
                complete=deps.complete,
                model=model,
                session_id=conversation.chat_id,
                started=started,
                opening=task.opening_message,
                criteria=list(task.expected.judge_criteria),
                transcript=neutral_transcript(agent.exchanges),
            )
        except Exception as exc:
            return model, None, f"answer quality not rated: {exc}"
        return model, score, why

    # Every judge rates; none when no turn got a reply.
    quality: list[tuple[str, int | None, str]] = []
    if agent.exchanges:
        quality = list(await asyncio.gather(*(rate(model) for model in deps.judge_models)))

    # The agreement of the judges that ruled; None when fewer than two ruled or
    # no judge model was called.
    agree = panel.agree if panel is not None else None
    judges_comment = ""
    if panel is not None:
        judges_comment = "; ".join(
            f"{judge_label(r.model)} errored: {r.reasoning}"
            if r.passed is None
            else f"{judge_label(r.model)} {r.word}"
            for r in panel.rulings
        )

    facts = PlayFacts(
        goal_reached=goal_reached,
        outcome=outcome,
        turns=conversation.turns,
        clarifying_questions=user.clarifying_questions,
        hints_used=user.hints_used,
        time_to_answer_seconds=sum(conversation.answer_seconds),
        answer_quality=quality,
        judges_agree=agree,
        judges_comment=judges_comment,
    )

    metrics: TraceMetrics | None = None
    traces_complete = False
    partial = ""
    problem: str | None = None
    if conversation.chat_id is None or conversation.turns_reached == 0:
        problem = "no turn reached the app"
    else:
        try:
            read = await deps.chat_observations(conversation.chat_id, conversation.turns_reached)
        except Exception as exc:
            problem = f"Langfuse could not be read: {exc}"
        else:
            traces_complete, partial = read.complete, read.partial_reason()
            if read.observations:
                metrics = trace_metrics(read.observations)
            elif read.read_error:
                problem = f"Langfuse could not be read: {read.read_error}"
            else:
                problem = "no app traces arrived"

    if metrics is None:
        trace_scores = f"missing: {problem}"
    else:
        trace_scores = "complete" if traces_complete else f"partial: {partial}"

    return PlayResult(
        task_id=task.id,
        chat_id=conversation.chat_id,
        outcome=outcome,
        goal_reached=goal_reached,
        reasoning=reasoning,
        error=error,
        scores=build_scores(facts, metrics, traces_complete, problem, partial),
        trace_scores=trace_scores,
        app_trace_ids=metrics.trace_ids if metrics else [],
        provider_models=metrics.provider_models if metrics else [],
        paused_seconds=paused,
        judges=panel.output() if panel is not None else [],
    )
