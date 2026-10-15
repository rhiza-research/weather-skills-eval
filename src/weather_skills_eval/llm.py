"""The harness's own model calls (simulated user and judge).

Every call goes through a ``Complete`` function:
``complete(role, session_id, **litellm_kwargs)``. In a run it is
``LangfuseGateway.traced_completion``, which calls LiteLLM and records the
call in Langfuse under user id ``role``. Tests pass a fake.
"""

import json
from collections.abc import Callable
from typing import Any

Complete = Callable[..., Any]

SIMULATED_USER = "simulated-user"
JUDGE = "judge"


class ModelReplyError(Exception):
    """A harness model replied with something the harness cannot use."""


def reply_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError) as exc:
        raise ModelReplyError(f"Unexpected model response: {response!r}") from exc
    if not isinstance(content, str):
        raise ModelReplyError(f"Model returned no text: {response!r}")
    return content


def reply_json(response: Any) -> dict:
    text = reply_text(response).strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ModelReplyError(f"Model reply is not a JSON object: {text!r}")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ModelReplyError(f"Model reply is not valid JSON: {text!r}") from exc
    if not isinstance(data, dict):
        raise ModelReplyError(f"Model reply is not a JSON object: {text!r}")
    return data
