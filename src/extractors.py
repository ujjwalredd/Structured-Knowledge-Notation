"""
Extraction module: JSON vs Structured Knowledge Notation (SKN).

This module sits at the core of the experiment. In a real AI pipeline:

    User Query -> Web Search -> Raw HTML/Text -> EXTRACTION -> Agent -> Answer

Current systems use JSON for the extraction step. This module provides both
a JSON extractor and a SKN extractor so the downstream agent can be
benchmarked with each format as its input.
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


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

JSON_SYSTEM_PROMPT = """\
You are an information extraction engine.
Your task is to extract structured information from the provided text and output a strict JSON object.

Rules:
1. Output must be valid JSON.
2. Do not include explanations.
3. Do not include markdown.
4. Do not include comments.
5. Only include information explicitly supported by the text.
6. Do not infer unstated causal relationships.
7. Do not add confidence scores.
8. Maintain neutral formatting.

Required Schema:
{
  "entities": [
    {
      "name": "",
      "type": "",
      "attributes": {"key": "value"}
    }
  ],
  "relationships": [
    {
      "source": "",
      "relation": "",
      "target": ""
    }
  ],
  "claims": [
    {
      "statement": "",
      "category": "fact | statistic | opinion | prediction"
    }
  ]
}

If information is missing, use empty arrays."""

SKN_SYSTEM_PROMPT = """\
You are a structured knowledge extraction engine optimized for downstream AI reasoning.
Your task is to convert the provided text into Structured Knowledge Notation (SKN).

Output must follow this exact format. Do not include explanations, markdown, or JSON.
Do not add any text outside the structure. Be concise. Every token must carry information.

[SKN]
@src <domain_category> | fresh:<high|medium|low> | reliability:<0.0-1.0>

@facts
  ! "highest priority fact directly from the text" [<confidence 0.0-1.0>]
  ! "second highest priority fact" [<confidence>]
  . "supporting detail" [<confidence>]
  . "supporting detail" [<confidence>]
  ~ "uncertain or inferred claim" [<confidence>]

@causal
  <cause> -> <effect> [<strength 0.0-1.0>]
  <cause> -> <effect> [<strength>]

@gaps
  - <specific information missing from the text>
  - <another piece of missing information>

@risk misdirection:<low|medium|high> | missing_context:<low|medium|high>
[/SKN]

Symbol key:
  ! = high importance fact (use for critical information)
  . = supporting detail (use for context and secondary facts)
  ~ = uncertain or inferred (use when the text implies but does not state)
  -> = causal link between two elements

Rules:
1. Confidence scores must reflect only what the text directly supports.
2. Order facts by importance. Put the most critical information first.
3. Extract causal links only when the text strongly implies them.
4. The @gaps section must list what is NOT in the text but would be needed for a complete answer.
5. Do not fabricate information not grounded in the text.
6. Keep it compact. Aim for under 200 tokens total.
7. Every fact must be traceable to the source text.
8. The @risk line must assess whether the question or text could lead to misdirection."""


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------


def extract_json(
    raw_text: str, model: str = "claude-sonnet-4-20250514"
) -> dict[str, Any]:
    """Extract structured JSON from raw text (simulating web search results).

    This is the current industry-standard approach. The raw text is converted
    to a flat JSON structure with entities, relationships, and claims.

    Returns
    -------
    dict
        Parsed JSON with entities, relationships, and claims.
    """
    logger.info("Extracting JSON format (model=%s)", model)
    raw = _call_api(system=JSON_SYSTEM_PROMPT, user_message=raw_text, model=model)

    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].rstrip()

    try:
        parsed: dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse JSON response: %s", exc)
        parsed = {"entities": [], "relationships": [], "claims": []}

    return parsed


def extract_skn(
    raw_text: str, model: str = "claude-sonnet-4-20250514"
) -> str:
    """Extract Structured Knowledge Notation from raw text.

    This is the proposed replacement for JSON extraction. The raw text is
    converted to SKN format carrying inline confidence scores, causal chains,
    knowledge gaps, and risk assessment.

    Returns
    -------
    str
        SKN-formatted string.
    """
    logger.info("Extracting SKN format (model=%s)", model)
    raw = _call_api(system=SKN_SYSTEM_PROMPT, user_message=raw_text, model=model)
    return raw.strip()
