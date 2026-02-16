"""
Agent reasoning module.

This is the SAME agent for both pipelines. It receives structured context
(either JSON or SKN) plus a question, and returns an answer.

The agent is identical. The only variable in the experiment is the format
of the context it receives from the extraction step.
"""

import json
import logging
import os
import time
from typing import Any

from anthropic import Anthropic, APIError, RateLimitError
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_client: Anthropic | None = None


def _get_client() -> Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY not set. "
                "Create a .env file or export the variable."
            )
        _client = Anthropic(api_key=api_key)
    return _client


def _call_api(
    system: str,
    user_message: str,
    model: str,
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> str:
    client = _get_client()
    for attempt in range(1, max_retries + 1):
        try:
            message = client.messages.create(
                model=model,
                max_tokens=4096,
                temperature=0,
                system=system,
                messages=[{"role": "user", "content": user_message}],
            )
            return message.content[0].text
        except RateLimitError:
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning(
                "Rate-limited (attempt %d/%d). Retrying in %.1fs",
                attempt, max_retries, wait,
            )
            if attempt == max_retries:
                raise
            time.sleep(wait)
        except APIError as exc:
            if exc.status_code and exc.status_code >= 500:
                wait = base_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Server error %s (attempt %d/%d). Retrying in %.1fs",
                    exc.status_code, attempt, max_retries, wait,
                )
                if attempt == max_retries:
                    raise
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Exhausted retries without a response")


AGENT_SYSTEM_PROMPT = """\
You are an AI agent. Based on the provided context, answer the question.

You must return ONLY a valid JSON object with this exact structure:
{
  "answer": "your answer here",
  "confidence": 0.85,
  "reasoning": "step by step explanation"
}

Do not include any text outside the JSON object.
Confidence must be between 0.0 and 1.0.
Base your confidence on how well the context supports your answer.
If the context indicates missing information or high uncertainty, lower your confidence accordingly."""


def agent_reason(
    context: str | dict[str, Any],
    question: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict[str, Any]:
    """Run agent reasoning over extracted context.

    This is the same agent for both JSON and SKN pipelines. It receives
    whatever the extraction step produced and a question, then answers.

    Parameters
    ----------
    context : str | dict
        Extracted context. Either a JSON dict or a SKN string.
    question : str
        The question to answer based on the context.
    model : str
        Claude model identifier.

    Returns
    -------
    dict
        {"answer": str, "confidence": float, "reasoning": str}
    """
    if isinstance(context, dict):
        context_str = json.dumps(context, indent=2)
    else:
        context_str = str(context)

    user_message = f"Context:\n{context_str}\n\nQuestion: {question}"

    logger.info("Agent reasoning (model=%s)", model)
    raw = _call_api(
        system=AGENT_SYSTEM_PROMPT,
        user_message=user_message,
        model=model,
    )

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].rstrip()

    try:
        parsed: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Failed to parse agent response")
        parsed = {
            "answer": raw,
            "confidence": 0.0,
            "reasoning": "Failed to parse structured response",
        }

    parsed.setdefault("answer", "")
    parsed.setdefault("confidence", 0.0)
    parsed.setdefault("reasoning", "")

    try:
        parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
    except (TypeError, ValueError):
        parsed["confidence"] = 0.0

    return parsed
