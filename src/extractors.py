"""
Extraction module: JSON vs Structured Knowledge Notation (SKN).

Provides multiple extraction variants for ablation study:
  - json:           Flat JSON, no metadata (baseline)
  - json_conf:      JSON with per-claim confidence scores
  - skn:            Full SKN (all sections)
  - skn_no_gaps:    SKN without @gaps section
  - skn_no_causal:  SKN without @causal section
  - skn_no_risk:    SKN without @risk section
"""

import json
import logging
from typing import Any

from src.llm import call_llm

logger = logging.getLogger(__name__)

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

JSON_CONF_SYSTEM_PROMPT = """\
You are an information extraction engine.
Your task is to extract structured information from the provided text and output a strict JSON object.

Rules:
1. Output must be valid JSON.
2. Do not include explanations.
3. Do not include markdown.
4. Do not include comments.
5. Only include information explicitly supported by the text.
6. Do not infer unstated causal relationships.
7. Add a confidence score (0.0-1.0) to each claim based on how strongly the text supports it.
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
      "category": "fact | statistic | opinion | prediction",
      "confidence": 0.85
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

SKN_NO_GAPS_PROMPT = """\
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
  ~ "uncertain or inferred claim" [<confidence>]

@causal
  <cause> -> <effect> [<strength 0.0-1.0>]

@risk misdirection:<low|medium|high> | missing_context:<low|medium|high>
[/SKN]

Symbol key:
  ! = high importance fact
  . = supporting detail
  ~ = uncertain or inferred
  -> = causal link

Rules:
1. Confidence scores must reflect only what the text directly supports.
2. Order facts by importance.
3. Extract causal links only when the text strongly implies them.
4. Do NOT include a @gaps section.
5. Do not fabricate information not grounded in the text.
6. Keep it compact. Aim for under 200 tokens total.
7. Every fact must be traceable to the source text.
8. The @risk line must assess misdirection potential."""

SKN_NO_CAUSAL_PROMPT = """\
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
  ~ "uncertain or inferred claim" [<confidence>]

@gaps
  - <specific information missing from the text>

@risk misdirection:<low|medium|high> | missing_context:<low|medium|high>
[/SKN]

Symbol key:
  ! = high importance fact
  . = supporting detail
  ~ = uncertain or inferred

Rules:
1. Confidence scores must reflect only what the text directly supports.
2. Order facts by importance.
3. Do NOT include a @causal section. No cause-effect chains.
4. The @gaps section must list what is NOT in the text.
5. Do not fabricate information not grounded in the text.
6. Keep it compact. Aim for under 200 tokens total.
7. Every fact must be traceable to the source text.
8. The @risk line must assess misdirection potential."""

SKN_NO_RISK_PROMPT = """\
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
  ~ "uncertain or inferred claim" [<confidence>]

@causal
  <cause> -> <effect> [<strength 0.0-1.0>]

@gaps
  - <specific information missing from the text>
[/SKN]

Symbol key:
  ! = high importance fact
  . = supporting detail
  ~ = uncertain or inferred
  -> = causal link

Rules:
1. Confidence scores must reflect only what the text directly supports.
2. Order facts by importance.
3. Extract causal links only when the text strongly implies them.
4. The @gaps section must list what is NOT in the text.
5. Do NOT include a @risk line.
6. Do not fabricate information not grounded in the text.
7. Keep it compact. Aim for under 200 tokens total.
8. Every fact must be traceable to the source text."""


# ---------------------------------------------------------------------------
# Public extraction functions
# ---------------------------------------------------------------------------

def extract_json(
    raw_text: str, model: str = "qwen2.5:7b"
) -> dict[str, Any]:
    """Extract structured JSON from raw text (baseline, no metadata)."""
    logger.info("Extracting JSON format (model=%s)", model)
    raw = call_llm(
        system=JSON_SYSTEM_PROMPT,
        user_message=raw_text,
        model=model,
        json_mode=True,
    )
    try:
        parsed: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse JSON response: %s", exc)
        parsed = {"entities": [], "relationships": [], "claims": []}
    return parsed


def extract_json_conf(
    raw_text: str, model: str = "qwen2.5:7b"
) -> dict[str, Any]:
    """Extract JSON with per-claim confidence scores (ablation variant)."""
    logger.info("Extracting JSON+confidence format (model=%s)", model)
    raw = call_llm(
        system=JSON_CONF_SYSTEM_PROMPT,
        user_message=raw_text,
        model=model,
        json_mode=True,
    )
    try:
        parsed: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse JSON+conf response: %s", exc)
        parsed = {"entities": [], "relationships": [], "claims": []}
    return parsed


def extract_skn(
    raw_text: str, model: str = "qwen2.5:7b"
) -> str:
    """Extract full SKN (all sections)."""
    logger.info("Extracting SKN format (model=%s)", model)
    raw = call_llm(
        system=SKN_SYSTEM_PROMPT,
        user_message=raw_text,
        model=model,
    )
    return raw


def extract_skn_no_gaps(
    raw_text: str, model: str = "qwen2.5:7b"
) -> str:
    """Extract SKN without @gaps section (ablation variant)."""
    logger.info("Extracting SKN-no-gaps format (model=%s)", model)
    raw = call_llm(
        system=SKN_NO_GAPS_PROMPT,
        user_message=raw_text,
        model=model,
    )
    return raw


def extract_skn_no_causal(
    raw_text: str, model: str = "qwen2.5:7b"
) -> str:
    """Extract SKN without @causal section (ablation variant)."""
    logger.info("Extracting SKN-no-causal format (model=%s)", model)
    raw = call_llm(
        system=SKN_NO_CAUSAL_PROMPT,
        user_message=raw_text,
        model=model,
    )
    return raw


def extract_skn_no_risk(
    raw_text: str, model: str = "qwen2.5:7b"
) -> str:
    """Extract SKN without @risk line (ablation variant)."""
    logger.info("Extracting SKN-no-risk format (model=%s)", model)
    raw = call_llm(
        system=SKN_NO_RISK_PROMPT,
        user_message=raw_text,
        model=model,
    )
    return raw


# Registry for benchmark.py to iterate over
EXTRACTORS = {
    "json": extract_json,
    "json_conf": extract_json_conf,
    "skn": extract_skn,
    "skn_no_gaps": extract_skn_no_gaps,
    "skn_no_causal": extract_skn_no_causal,
    "skn_no_risk": extract_skn_no_risk,
}

# Default pipelines (without ablation)
DEFAULT_PIPELINES = ["json", "skn"]

# All pipelines (with ablation)
ABLATION_PIPELINES = [
    "json", "json_conf",
    "skn_no_gaps", "skn_no_causal", "skn_no_risk", "skn",
]
