"""The simulated user.

It opens with the task's opening message. On every later turn it reads the
assistant's last reply, classifies it, and answers from the facts sheet. It
reveals a hidden step only when the assistant is stuck, and it may give up,
which ends the play as a failure.
"""

import asyncio
from collections.abc import Callable

from scenario import AgentAdapter, AgentInput, AgentRole, ScenarioResult

from .app_client import visible_text
from .llm import SIMULATED_USER, Complete, ModelReplyError, reply_json
from .tasks import Task

CLASSIFICATIONS = ("clarifying_question", "progress", "answer", "other")

SYSTEM_PROMPT = """You play a person who asked a weather assistant for help. You are the user, \
never the assistant. Write the way a busy person types: short and plain.

What you want, in your own words when you started:
{opening}

Facts you know. Answer the assistant's questions from these facts only. If a question is \
not covered by them, say you do not know or have no preference:
{facts}

Hidden steps. You know these, but you only mention one when the assistant is stuck: it \
repeats itself, says it cannot go on, or asks you how to proceed. Reveal at most one per \
reply, in order, and only those not yet revealed:
{hidden_steps}

You may give up when the assistant keeps failing, goes in circles, or clearly cannot help.

Reply with one JSON object and nothing else:
{{"classification": one of "clarifying_question" (the assistant's last message asks you \
something it needs to go on), "progress" (it reports work or partial results), "answer" \
(it presents what you asked for), "other",
 "revealed_hidden_step": true if your message reveals a hidden step, else false,
 "give_up": true or false,
 "give_up_reason": a short reason, or "",
 "message": your next message to the assistant, or "" when you give up}}"""


def transcript(messages: list) -> str:
    lines = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        speaker = "You" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {visible_text(content)}")
    return "\n\n".join(lines)


class SimulatedUser(AgentAdapter):
    role = AgentRole.USER
    name = "simulated-user"

    def __init__(
        self,
        *,
        task: Task,
        model: str,
        complete: Complete,
        session_id: Callable[[], str | None],
    ):
        self._task = task
        self._model = model
        self._complete = complete
        self._session_id = session_id
        self._opened = False
        self.classifications: list[str] = []
        self.hints_used = 0
        self.gave_up = False
        self.give_up_reason = ""

    @property
    def clarifying_questions(self) -> int:
        return self.classifications.count("clarifying_question")

    def _system_prompt(self) -> str:
        hidden = self._task.metadata.hidden_steps
        return SYSTEM_PROMPT.format(
            opening=self._task.opening_message,
            facts=self._task.metadata.facts,
            hidden_steps="\n".join(f"- {step}" for step in hidden) or "(none)",
        )

    async def call(self, input: AgentInput):
        if not self._opened:
            self._opened = True
            return self._task.opening_message

        response = await asyncio.to_thread(
            self._complete,
            SIMULATED_USER,
            self._session_id(),
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": "The conversation so far:\n\n"
                    + transcript(input.messages)
                    + "\n\nWrite your JSON reply.",
                },
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        data = reply_json(response)
        classification = data.get("classification")
        if classification not in CLASSIFICATIONS:
            raise ModelReplyError(f"Unknown classification from the simulated user: {data!r}")
        self.classifications.append(classification)
        # A claimed reveal counts only while unrevealed hidden steps remain.
        if data.get("revealed_hidden_step") is True and self.hints_used < len(
            self._task.metadata.hidden_steps
        ):
            self.hints_used += 1
        if data.get("give_up") is True:
            self.gave_up = True
            self.give_up_reason = str(data.get("give_up_reason") or "")
            return ScenarioResult(
                success=False,
                messages=list(input.messages),
                reasoning=f"The simulated user gave up: {self.give_up_reason}",
            )
        message = data.get("message")
        if not isinstance(message, str) or not message.strip():
            raise ModelReplyError(f"The simulated user sent no message: {data!r}")
        return message
