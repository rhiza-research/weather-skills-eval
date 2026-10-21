"""The simulated user.

It opens with the task's opening message. On every later turn it reads the
assistant's last reply and what a real user sees of the latest reply that
presents files (each file's path and kind, and each image), classifies the
reply, checks the result against what it asked for, and answers from the
facts sheet. It reveals the next hidden step not yet revealed or done when
the assistant is stuck, or has presented a result or moved on past where that
step belongs without doing it. It ends the play when it is done or gives up.
"""

import asyncio
import base64
from collections.abc import Awaitable, Callable

from scenario import AgentAdapter, AgentInput, AgentRole, ScenarioResult

from .app_client import visible_text
from .checks import Evidence
from .llm import SIMULATED_USER, Complete, ModelReplyError, reply_json, reply_text
from .tasks import Task

CLASSIFICATIONS = ("clarifying_question", "progress", "answer", "other")

SYSTEM_PROMPT = """You play a person who asked a weather assistant for help. You are the user, \
never the assistant. Write the way a busy person types: short and plain.

What you want, in your own words when you started:
{opening}

Facts you know. Answer the assistant's questions from these facts only. If a question is \
not covered by them, say you do not know or have no preference:
{facts}

Hidden steps. You know these. The next hidden step is the first one below that you have \
not revealed and that is not done. A step the conversation or the files you were shown \
already show as done counts as used: skip it and consider the next one. Mention the next \
hidden step in either case:
- The assistant is stuck: it repeats itself, says it cannot go on, or asks you how to proceed.
- The assistant has presented a result, or moved on past where that step belongs, without \
doing that step.
Do not mention it while the assistant is still working toward it: a progress reply that has \
not reached that step yet is not a reason. Reveal at most one per reply:
{hidden_steps}

When the assistant has presented files, you also get them after the conversation: each \
file's path and kind, and each image. Compare what you received, the answer's text and any \
files, with what you asked for and your facts (place, variable, period, aggregation, \
units, layout). When something does not match or a requested part is missing, say so \
plainly and ask for the fix, the way a busy person would.

You are done when the assistant has presented everything you asked for, as you asked. \
Before you are done or give up, ask for anything still wrong or missing, unless you already \
asked for that exact fix. Give up only when the assistant says it cannot do what you asked, \
or fails again at a step you already asked it to fix.

Reply with one JSON object and nothing else:
{{"classification": one of "clarifying_question" (the assistant's last message asks you \
something it needs to go on), "progress" (it reports work or partial results), "answer" \
(it presents what you asked for), "other",
 "revealed_hidden_step": true if your message reveals a hidden step, else false,
 "done": true when the assistant has presented everything you asked for, as you asked, \
else false,
 "give_up": true or false; never true together with "done",
 "give_up_reason": a short reason, or "",
 "message": your next message to the assistant, or "" when you are done or give up}}"""


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


def unusable(data: dict, *, message_required: bool = True) -> str | None:
    """Why a reply cannot be used, or None when it can. Without
    ``message_required`` an empty message is usable: it is never sent."""
    if data.get("classification") not in CLASSIFICATIONS:
        return f"unknown classification {data.get('classification')!r}"
    for flag in ("revealed_hidden_step", "done", "give_up"):
        if not isinstance(data.get(flag), bool):
            return f"{flag} is not true or false"
    done, give_up = data["done"], data["give_up"]
    if done and give_up:
        return "done and give_up are both true"
    message = data.get("message")
    if message is not None and not isinstance(message, str):
        return "message is not text"
    empty = message is None or not message.strip()
    if message_required and not (done or give_up) and empty:
        return "empty message while neither done nor give_up is true"
    return None


FILES_HEADER = "Files the assistant showed you in its latest reply that had files:"


def files_message(evidence: Evidence) -> dict:
    """What a real user sees of the presented files: paths, kinds and images.
    An image that could not be downloaded, or of a type outside
    ``IMAGE_MEDIA_TYPES``, is shown as its path with a note."""
    parts: list[dict] = [{"type": "text", "text": FILES_HEADER}]
    for file in evidence.files:
        if file.download_failed:
            parts.append({"type": "text", "text": f"File: {file.path} (could not be opened)"})
            continue
        line = f"File: {file.path} ({file.kind})"
        if file.kind == "image" and not file.shown_as_image:
            line += " (not shown as an image)"
        parts.append({"type": "text", "text": line})
        if file.image is not None and file.shown_as_image:
            encoded = base64.b64encode(file.image).decode("ascii")
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{file.media_type};base64,{encoded}"},
                }
            )
    for path in evidence.unlisted:
        parts.append({"type": "text", "text": f"File: {path} (could not be opened)"})
    return {"role": "user", "content": parts}


def parse_reply(response, *, message_required: bool = True) -> tuple[dict | None, str | None]:
    """The reply as a JSON object, and why it cannot be used (None when it can)."""
    try:
        data = reply_json(response)
    except ModelReplyError:
        return None, "it is not a JSON object"
    return data, unusable(data, message_required=message_required)


def _text(response) -> str:
    try:
        return reply_text(response)
    except ModelReplyError:
        return ""


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
        collect_evidence: Callable[[], Awaitable[Evidence]],
    ):
        self._task = task
        self._model = model
        self._complete = complete
        self._session_id = session_id
        self._collect_evidence = collect_evidence
        self._opened = False
        self.classifications: list[str] = []
        self.hints_used = 0
        self.done = False
        self.gave_up = False
        self.give_up_reason = ""

    @property
    def ended(self) -> bool:
        return self.done or self.gave_up

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

    async def _complete_once(self, messages: list[dict]):
        return await asyncio.to_thread(
            self._complete,
            SIMULATED_USER,
            self._session_id(),
            model=self._model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
        )

    async def _reply(self, messages: list[dict], *, message_required: bool) -> dict:
        """When the reply cannot be used, the call is made once more, shown the
        reply and told why. A model call that fails, such as a reply cut off at
        max_tokens, is not retried."""
        response = await self._complete_once(messages)
        data, reason = parse_reply(response, message_required=message_required)
        if reason is None:
            return data
        text = _text(response)
        shown = [{"role": "assistant", "content": text}] if text.strip() else []
        if not text.strip():
            reason = "it had no text"
        retry = [
            *messages,
            *shown,
            {
                "role": "user",
                "content": f"Your previous reply could not be used: {reason}. Write your JSON "
                "reply again, as described.",
            },
        ]
        data, reason = parse_reply(
            await self._complete_once(retry), message_required=message_required
        )
        if reason is not None:
            raise ModelReplyError(f"The simulated user's reply cannot be used, {reason}: {data!r}")
        return data

    async def call(self, input: AgentInput):
        if not self._opened:
            self._opened = True
            return self._task.opening_message
        return await self._respond(input, counted=True)

    async def read_last_reply(self, input: AgentInput):
        """Read the reply to the last turn the cap allows. Its message is never
        sent, so only its done or give-up counts: no classification, no hint,
        and an empty message is usable."""
        return await self._respond(input, counted=False)

    async def _respond(self, input: AgentInput, *, counted: bool):
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {
                "role": "user",
                "content": "The conversation so far:\n\n" + transcript(input.messages),
            },
        ]
        evidence = await self._collect_evidence()
        if evidence.files or evidence.unlisted:
            messages.append(files_message(evidence))
        messages.append({"role": "user", "content": "Write your JSON reply."})
        data = await self._reply(messages, message_required=counted)
        ends = data["done"] or data["give_up"]
        if counted:
            self.classifications.append(data["classification"])
        # A claimed reveal counts only when its message is sent, and only while
        # unrevealed hidden steps remain.
        if (
            counted
            and not ends
            and data["revealed_hidden_step"]
            and self.hints_used < len(self._task.metadata.hidden_steps)
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
        if data.get("done") is True:
            self.done = True
            return ScenarioResult(
                success=True,
                messages=list(input.messages),
                reasoning="The simulated user is done",
            )
        return data["message"]
