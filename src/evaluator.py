"""
Evaluation metrics for the SKN benchmark.

Measures six dimensions per extraction format:
1. Accuracy          Did the agent get the right answer?
2. Confidence (ECE)  Are confidence scores well-calibrated?
3. Hallucination     Did the agent fabricate unsupported claims?
4. Groundedness      Is the answer backed by the source text?
5. Abstention        Does the agent correctly refuse when info is missing?
6. Normalized hallucination  fabricated_claims / total_claims

Includes bootstrap confidence intervals for all metrics.
"""

import json
import logging
import re
from typing import Any

import numpy as np

from src.llm import call_llm

logger = logging.getLogger(__name__)


def _parse_json_response(raw: str) -> dict[str, Any]:
    """Extract and parse a JSON object from an LLM response."""
    cleaned = raw.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if code_block:
        try:
            return json.loads(code_block.group(1).strip())
        except json.JSONDecodeError:
            pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            pass

    logger.debug("Raw LLM response that failed parsing: %s", raw[:500])
    raise json.JSONDecodeError("No valid JSON found in response", cleaned, 0)


# ===================================================================
# 1. ACCURACY
# ===================================================================

ACCURACY_JUDGE_PROMPT = """\
You are an evaluation judge. Compare the predicted answer to the ground truth.
Determine if the prediction captures the core meaning of the ground truth.

Be lenient: the prediction does not need to match word-for-word.
It should capture the same root cause or key insight.

Respond with ONLY this JSON, no other text before or after:
{"correct": true}
or
{"correct": false}"""


def _judge_correct(
    prediction: str, ground_truth: str, model: str = "qwen2.5:7b"
) -> bool:
    user_msg = f"Ground truth: {ground_truth}\n\nPrediction: {prediction}"
    raw = call_llm(
        system=ACCURACY_JUDGE_PROMPT,
        user_message=user_msg,
        model=model,
        json_mode=True,
    )
    try:
        result = _parse_json_response(raw).get("correct", False)
        logger.debug("Accuracy judge raw: %s -> %s", raw[:200], result)
        return bool(result)
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Accuracy judge unparseable: %s", raw[:300])
        return False


# ===================================================================
# 2. HALLUCINATION (with normalized claim count)
# ===================================================================

HALLUCINATION_JUDGE_PROMPT = """\
You are a hallucination detector. Given the ORIGINAL SOURCE TEXT and an ANSWER,
determine whether the answer contains any claims that are NOT supported by or
cannot be reasonably inferred from the source text.

Count the TOTAL number of distinct claims in the answer, and list any fabricated ones.

Severity scale:
0.0 = no hallucination
0.1-0.3 = minor embellishment (adding plausible but unstated detail)
0.4-0.6 = moderate fabrication (inventing specific facts not in the text)
0.7-1.0 = severe hallucination (contradicts the source or invents wholesale)

Respond with ONLY this JSON object, no other text before or after:
{"hallucination_detected": true, "hallucination_severity": 0.5, "total_claims": 8, "fabricated_claims": ["claim1", "claim2"]}"""


def _judge_hallucination(
    source_text: str, answer: str, model: str = "qwen2.5:7b"
) -> dict[str, Any]:
    user_msg = (
        f"ORIGINAL SOURCE TEXT:\n{source_text}\n\n"
        f"ANSWER TO EVALUATE:\n{answer}"
    )
    raw = call_llm(
        system=HALLUCINATION_JUDGE_PROMPT,
        user_message=user_msg,
        model=model,
        json_mode=True,
    )
    try:
        result = _parse_json_response(raw)
        fabricated = result.get("fabricated_claims", [])
        total = max(int(result.get("total_claims", 1)), 1)
        return {
            "hallucination_detected": bool(result.get("hallucination_detected", False)),
            "hallucination_severity": float(result.get("hallucination_severity", 0.0)),
            "total_claims": total,
            "fabricated_claims": fabricated,
            "normalized_hallucination": len(fabricated) / total,
        }
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Hallucination judge unparseable")
        return {
            "hallucination_detected": False,
            "hallucination_severity": 0.0,
            "total_claims": 1,
            "fabricated_claims": [],
            "normalized_hallucination": 0.0,
        }


# ===================================================================
# 3. GROUNDEDNESS
# ===================================================================

GROUNDEDNESS_JUDGE_PROMPT = """\
You are a groundedness evaluator. Given the ORIGINAL SOURCE TEXT and an ANSWER,
score how well the answer is grounded in (supported by) the source text.

groundedness_score: 1.0 means every claim in the answer is directly traceable
to the text. 0.0 means the answer has no connection to the source text.
unsupported_fraction: proportion of claims in the answer that lack support.

Respond with ONLY this JSON object, no other text before or after:
{"groundedness_score": 0.8, "unsupported_fraction": 0.2}"""


def _judge_groundedness(
    source_text: str, answer: str, model: str = "qwen2.5:7b"
) -> dict[str, Any]:
    user_msg = (
        f"ORIGINAL SOURCE TEXT:\n{source_text}\n\n"
        f"ANSWER TO EVALUATE:\n{answer}"
    )
    raw = call_llm(
        system=GROUNDEDNESS_JUDGE_PROMPT,
        user_message=user_msg,
        model=model,
        json_mode=True,
    )
    try:
        result = _parse_json_response(raw)
        return {
            "groundedness_score": float(result.get("groundedness_score", 0.0)),
            "unsupported_fraction": float(result.get("unsupported_fraction", 0.0)),
        }
    except (json.JSONDecodeError, AttributeError):
        logger.warning("Groundedness judge unparseable")
        return {"groundedness_score": 0.0, "unsupported_fraction": 1.0}


# ===================================================================
# 4. ABSTENTION
# ===================================================================

ABSTENTION_KEYWORDS = [
    "insufficient information",
    "not enough information",
    "cannot determine",
    "cannot be determined",
    "unable to determine",
    "not specified",
    "not provided",
    "not enough data",
    "insufficient data",
    "no information",
    "cannot answer",
    "impossible to determine",
    "not available",
    "lack of information",
    "missing information",
    "does not contain",
    "do not contain",
    "cannot provide",
    "unable to provide",
    "cannot identify",
    "unable to identify",
    "no relevant information",
    "no specific information",
    "not contain any information",
    "i cannot",
    "i'm unable",
    "i am unable",
    "no details about",
    "no mention of",
    "does not include",
    "does not reference",
]


def _detected_abstention(answer: str) -> bool:
    answer_lower = answer.lower()
    return any(kw in answer_lower for kw in ABSTENTION_KEYWORDS)


# ===================================================================
# 5. BOOTSTRAP CONFIDENCE INTERVALS
# ===================================================================


def bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> dict[str, float]:
    """Compute bootstrap confidence interval for a metric.

    Returns
    -------
    dict
        {"mean": float, "ci_low": float, "ci_high": float, "std": float}
    """
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0}

    arr = np.array(values)
    rng = np.random.default_rng(42)
    boot_means = np.array([
        rng.choice(arr, size=len(arr), replace=True).mean()
        for _ in range(n_bootstrap)
    ])

    alpha = (1 - ci) / 2
    return {
        "mean": round(float(arr.mean()), 4),
        "ci_low": round(float(np.percentile(boot_means, alpha * 100)), 4),
        "ci_high": round(float(np.percentile(boot_means, (1 - alpha) * 100)), 4),
        "std": round(float(boot_means.std()), 4),
    }


# ===================================================================
# Public metric functions
# ===================================================================


def calculate_accuracy(
    results: list[dict[str, Any]], agent_type: str, model: str = "qwen2.5:7b"
) -> float:
    answerable = [r for r in results if r.get("answerable", True)]
    if not answerable:
        return 0.0

    correct = 0
    for entry in answerable:
        prediction = entry.get(f"{agent_type}_answer", "")
        ground_truth = entry.get("ground_truth", "")
        is_ok = _judge_correct(prediction, ground_truth, model=model)
        entry[f"{agent_type}_correct"] = is_ok
        if is_ok:
            correct += 1

    for entry in results:
        if not entry.get("answerable", True):
            entry[f"{agent_type}_correct"] = None

    acc = correct / len(answerable)
    logger.info(
        "Accuracy (%s): %.2f (%d/%d answerable)",
        agent_type, acc, correct, len(answerable),
    )
    return acc


def calculate_ece(
    results: list[dict[str, Any]], agent_type: str, n_bins: int = 10
) -> float:
    answerable = [
        r for r in results
        if r.get("answerable", True) and r.get(f"{agent_type}_correct") is not None
    ]
    if not answerable:
        return 0.0

    confidences = np.array(
        [float(r.get(f"{agent_type}_confidence", 0.0)) for r in answerable]
    )
    correctness = np.array(
        [1.0 if r.get(f"{agent_type}_correct", False) else 0.0 for r in answerable]
    )

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(answerable)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (
            (confidences >= lo) & (confidences <= hi)
            if i == n_bins - 1
            else (confidences >= lo) & (confidences < hi)
        )
        bin_size = mask.sum()
        if bin_size == 0:
            continue
        ece += (bin_size / total) * abs(
            correctness[mask].mean() - confidences[mask].mean()
        )

    logger.info("ECE (%s): %.4f", agent_type, ece)
    return float(ece)


def calculate_hallucination(
    results: list[dict[str, Any]], agent_type: str, model: str = "qwen2.5:7b"
) -> dict[str, Any]:
    if not results:
        return {
            "hallucination_rate": 0.0,
            "avg_severity": 0.0,
            "normalized_hallucination": 0.0,
        }

    detections = 0
    severities: list[float] = []
    normalized_scores: list[float] = []

    for entry in results:
        source = entry.get("raw_text", "")
        answer = entry.get(f"{agent_type}_answer", "")
        result = _judge_hallucination(source, answer, model=model)
        entry[f"{agent_type}_hallucination"] = result
        if result["hallucination_detected"]:
            detections += 1
        severities.append(result["hallucination_severity"])
        normalized_scores.append(result["normalized_hallucination"])

    rate = detections / len(results)
    avg_sev = float(np.mean(severities))
    avg_norm = float(np.mean(normalized_scores))

    logger.info(
        "Hallucination (%s): rate=%.2f, severity=%.4f, normalized=%.4f",
        agent_type, rate, avg_sev, avg_norm,
    )
    return {
        "hallucination_rate": round(rate, 4),
        "avg_severity": round(avg_sev, 4),
        "normalized_hallucination": round(avg_norm, 4),
    }


def calculate_groundedness(
    results: list[dict[str, Any]], agent_type: str, model: str = "qwen2.5:7b"
) -> dict[str, Any]:
    if not results:
        return {"avg_groundedness": 0.0, "avg_unsupported_fraction": 0.0}

    scores: list[float] = []
    unsupported: list[float] = []
    for entry in results:
        source = entry.get("raw_text", "")
        answer = entry.get(f"{agent_type}_answer", "")
        result = _judge_groundedness(source, answer, model=model)
        entry[f"{agent_type}_groundedness"] = result
        scores.append(result["groundedness_score"])
        unsupported.append(result["unsupported_fraction"])

    logger.info(
        "Groundedness (%s): avg=%.4f", agent_type, float(np.mean(scores)),
    )
    return {
        "avg_groundedness": round(float(np.mean(scores)), 4),
        "avg_unsupported_fraction": round(float(np.mean(unsupported)), 4),
    }


def calculate_abstention(
    results: list[dict[str, Any]], agent_type: str
) -> dict[str, Any]:
    tp = fp = fn = tn = 0
    for entry in results:
        answerable = entry.get("answerable", True)
        abstained = _detected_abstention(entry.get(f"{agent_type}_answer", ""))
        entry[f"{agent_type}_abstained"] = abstained

        if not answerable and abstained:
            tp += 1
        elif answerable and abstained:
            fp += 1
        elif not answerable and not abstained:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    logger.info(
        "Abstention (%s): P=%.2f R=%.2f F1=%.2f (tp=%d fp=%d fn=%d tn=%d)",
        agent_type, precision, recall, f1, tp, fp, fn, tn,
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
    }


# ===================================================================
# Master metrics
# ===================================================================


def calculate_metrics(
    results: list[dict[str, Any]],
    pipelines: list[str] | None = None,
    model: str = "qwen2.5:7b",
) -> dict[str, Any]:
    """Calculate all metrics for specified pipeline formats.

    Parameters
    ----------
    results : list[dict]
        Per-sample evaluation rows.
    pipelines : list[str] | None
        List of format names to evaluate (e.g. ["json", "skn"]).
        Defaults to ["json", "skn"].
    model : str
        Ollama model for LLM-as-judge calls.
    """
    if pipelines is None:
        pipelines = ["json", "skn"]

    metrics: dict[str, Any] = {}

    for fmt in pipelines:
        logger.info("========== Evaluating: %s format ==========", fmt)

        acc = calculate_accuracy(results, fmt, model=model)
        ece = calculate_ece(results, fmt)

        answerable = [r for r in results if r.get("answerable", True)]
        confidences = [
            float(r.get(f"{fmt}_confidence", 0.0)) for r in answerable
        ]
        correct_confs = [
            c for c, r in zip(confidences, answerable)
            if r.get(f"{fmt}_correct", False)
        ]
        wrong_confs = [
            c for c, r in zip(confidences, answerable)
            if r.get(f"{fmt}_correct") is False
        ]

        hall = calculate_hallucination(results, fmt, model=model)
        grnd = calculate_groundedness(results, fmt, model=model)
        abst = calculate_abstention(results, fmt)

        # Per-sample accuracy for bootstrap
        per_sample_acc = [
            1.0 if r.get(f"{fmt}_correct", False) else 0.0
            for r in answerable
        ]
        per_sample_hall = [
            r.get(f"{fmt}_hallucination", {}).get("normalized_hallucination", 0.0)
            for r in results
        ]
        per_sample_grnd = [
            r.get(f"{fmt}_groundedness", {}).get("groundedness_score", 0.0)
            for r in results
        ]

        metrics[fmt] = {
            "accuracy": round(acc, 4),
            "accuracy_ci": bootstrap_ci(per_sample_acc),
            "ece": round(ece, 4),
            "avg_confidence": round(
                float(np.mean(confidences)) if confidences else 0.0, 4
            ),
            "confidence_when_correct": round(
                float(np.mean(correct_confs)) if correct_confs else 0.0, 4
            ),
            "confidence_when_wrong": round(
                float(np.mean(wrong_confs)) if wrong_confs else 0.0, 4
            ),
            "hallucination_rate": hall["hallucination_rate"],
            "hallucination_severity": hall["avg_severity"],
            "normalized_hallucination": hall["normalized_hallucination"],
            "normalized_hallucination_ci": bootstrap_ci(per_sample_hall),
            "groundedness": grnd["avg_groundedness"],
            "groundedness_ci": bootstrap_ci(per_sample_grnd),
            "unsupported_fraction": grnd["avg_unsupported_fraction"],
            "abstention_precision": abst["precision"],
            "abstention_recall": abst["recall"],
            "abstention_f1": abst["f1"],
        }

    return metrics
