"""
Agent reasoning module.

This is the SAME agent for all pipelines. It receives structured context
(JSON, JSON+conf, or any SKN variant) plus a question, and returns an answer.

The agent is identical across all extraction formats. The only variable
in the experiment is the format of the context it receives.
"""

import json
import logging
from typing import Any

from src.llm import call_llm

logger = logging.getLogger(__name__)

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
    model: str = "qwen2.5:7b",
) -> dict[str, Any]:
    """Run agent reasoning over extracted context.

    Parameters
    ----------
    context : str | dict
        Extracted context (JSON dict or SKN string).
    question : str
        The question to answer.
    model : str
        Ollama model name.

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
    raw = call_llm(
        system=AGENT_SYSTEM_PROMPT,
        user_message=user_message,
        model=model,
        json_mode=True,
    )

    try:
        parsed: dict[str, Any] = json.loads(raw)
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
