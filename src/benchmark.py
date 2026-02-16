"""
Main benchmark runner for JSON vs SKN extraction comparison.

Real-world AI pipeline:

    Question -> Web Search -> Raw Results -> [Extraction] -> Same Agent -> Answer

The extraction step is the ONLY variable. Both pipelines receive the same
search results. One uses JSON extraction, the other uses SKN extraction.
The same agent receives the extracted context and the same question.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import agent_reason
from src.evaluator import calculate_metrics
from src.extractors import extract_skn, extract_json
from src.searcher import search_web

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_dataset(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        data: list[dict[str, Any]] = json.load(fh)
    required_keys = {"id", "question", "ground_truth"}
    for idx, entry in enumerate(data):
        missing = required_keys - entry.keys()
        if missing:
            raise ValueError(f"Sample {idx} missing keys: {missing}")
    logger.info("Loaded %d samples from %s", len(data), path)
    return data


def _save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False)
    logger.info("Saved: %s", path)


def _save_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    logger.info("Saved: %s", path)


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _context_size(obj: Any) -> tuple[int, int]:
    text = json.dumps(obj, indent=2) if isinstance(obj, dict) else str(obj)
    return len(text), _estimate_tokens(text)


def _delta(skn_val: float, json_val: float, higher_is_better: bool = True) -> str:
    diff = skn_val - json_val
    if abs(diff) < 0.0001:
        return "  (tied)"
    if higher_is_better:
        winner = "SKN" if diff > 0 else "JSON"
    else:
        winner = "SKN" if diff < 0 else "JSON"
    return f"  (delta {diff:+.4f}, {winner} wins)"


def run_benchmark(
    dataset_path: str,
    output_dir: str = "results",
    model: str = "claude-sonnet-4-20250514",
) -> dict[str, Any]:
    """Run the full JSON vs SKN extraction benchmark."""
    start_time = time.time()
    samples = _load_dataset(dataset_path)

    n_answerable = sum(1 for s in samples if s.get("answerable", True))
    n_unanswerable = len(samples) - n_answerable

    search_results: list[dict[str, Any]] = []
    extractions: list[dict[str, Any]] = []
    agent_responses: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    search_times: list[float] = []
    json_extract_times: list[float] = []
    skn_extract_times: list[float] = []
    json_token_counts: list[int] = []
    skn_token_counts: list[int] = []

    # Phase 1: Web Search (same search for both pipelines)
    logger.info("=== Phase 1: Web Search ===")
    for sample in tqdm(samples, desc="Searching", unit="sample"):
        sid = sample["id"]
        question = sample["question"]

        t0 = time.time()
        try:
            raw_text = search_web(question)
        except Exception as exc:
            logger.error("Search failed for sample %s: %s", sid, exc)
            raw_text = ""
        search_times.append(time.time() - t0)

        sample["raw_text"] = raw_text
        search_results.append({
            "id": sid,
            "question": question,
            "raw_text": raw_text,
            "raw_text_chars": len(raw_text),
        })

    _save_json(search_results, os.path.join(output_dir, "search_results.json"))

    # Phase 2: Extraction (the variable being tested)
    logger.info("=== Phase 2: Extraction ===")
    for sample in tqdm(samples, desc="Extracting", unit="sample"):
        sid = sample["id"]
        raw = sample["raw_text"]

        t0 = time.time()
        try:
            json_out = extract_json(raw, model=model)
        except Exception as exc:
            logger.error("JSON extraction failed for sample %s: %s", sid, exc)
            json_out = {"entities": [], "relationships": [], "claims": []}
        json_extract_times.append(time.time() - t0)

        t0 = time.time()
        try:
            skn_out = extract_skn(raw, model=model)
        except Exception as exc:
            logger.error("SKN extraction failed for sample %s: %s", sid, exc)
            skn_out = ""
        skn_extract_times.append(time.time() - t0)

        json_chars, json_tokens = _context_size(json_out)
        skn_chars, skn_tokens = _context_size(skn_out)
        json_token_counts.append(json_tokens)
        skn_token_counts.append(skn_tokens)

        extractions.append({
            "id": sid,
            "json_extraction": json_out,
            "skn_extraction": skn_out,
            "json_tokens_est": json_tokens,
            "skn_tokens_est": skn_tokens,
        })

    _save_json(extractions, os.path.join(output_dir, "extractions.json"))

    # Phase 3: Agent reasoning (same agent, different context)
    logger.info("=== Phase 3: Agent Reasoning ===")
    json_reason_times: list[float] = []
    skn_reason_times: list[float] = []

    for sample, ext in tqdm(
        zip(samples, extractions),
        total=len(samples),
        desc="Reasoning",
        unit="sample",
    ):
        sid = sample["id"]
        question = sample["question"]

        t0 = time.time()
        try:
            json_resp = agent_reason(ext["json_extraction"], question, model=model)
        except Exception as exc:
            logger.error("Agent (JSON context) failed for sample %s: %s", sid, exc)
            json_resp = {"answer": "", "confidence": 0.0, "reasoning": "error"}
        json_reason_times.append(time.time() - t0)

        t0 = time.time()
        try:
            skn_resp = agent_reason(ext["skn_extraction"], question, model=model)
        except Exception as exc:
            logger.error("Agent (SKN context) failed for sample %s: %s", sid, exc)
            skn_resp = {"answer": "", "confidence": 0.0, "reasoning": "error"}
        skn_reason_times.append(time.time() - t0)

        agent_responses.append({
            "id": sid,
            "question": question,
            "json_response": json_resp,
            "skn_response": skn_resp,
        })

        eval_rows.append({
            "id": sid,
            "category": sample.get("category", "unknown"),
            "raw_text": sample["raw_text"],
            "ground_truth": sample["ground_truth"],
            "answerable": sample.get("answerable", True),
            "json_answer": json_resp["answer"],
            "json_confidence": json_resp["confidence"],
            "json_reasoning": json_resp["reasoning"],
            "skn_answer": skn_resp["answer"],
            "skn_confidence": skn_resp["confidence"],
            "skn_reasoning": skn_resp["reasoning"],
        })

    _save_json(agent_responses, os.path.join(output_dir, "agent_responses.json"))

    # Phase 4: Evaluation
    logger.info("=== Phase 4: Evaluation ===")
    metrics = calculate_metrics(eval_rows)

    # Efficiency metrics
    avg_search = sum(search_times) / len(search_times)
    avg_json_ext = sum(json_extract_times) / len(json_extract_times)
    avg_skn_ext = sum(skn_extract_times) / len(skn_extract_times)
    avg_json_rsn = sum(json_reason_times) / len(json_reason_times)
    avg_skn_rsn = sum(skn_reason_times) / len(skn_reason_times)
    avg_json_tok = sum(json_token_counts) / len(json_token_counts)
    avg_skn_tok = sum(skn_token_counts) / len(skn_token_counts)

    metrics["json"]["avg_search_time_s"] = round(avg_search, 2)
    metrics["json"]["avg_extraction_time_s"] = round(avg_json_ext, 2)
    metrics["json"]["avg_reasoning_time_s"] = round(avg_json_rsn, 2)
    metrics["json"]["avg_total_latency_s"] = round(
        avg_search + avg_json_ext + avg_json_rsn, 2
    )
    metrics["json"]["avg_context_tokens"] = round(avg_json_tok)

    metrics["skn"]["avg_search_time_s"] = round(avg_search, 2)
    metrics["skn"]["avg_extraction_time_s"] = round(avg_skn_ext, 2)
    metrics["skn"]["avg_reasoning_time_s"] = round(avg_skn_rsn, 2)
    metrics["skn"]["avg_total_latency_s"] = round(
        avg_search + avg_skn_ext + avg_skn_rsn, 2
    )
    metrics["skn"]["avg_context_tokens"] = round(avg_skn_tok)

    token_ratio = avg_skn_tok / avg_json_tok if avg_json_tok > 0 else 0.0
    metrics["efficiency"] = {
        "token_ratio_skn_vs_json": round(token_ratio, 2),
    }

    _save_json(metrics, os.path.join(output_dir, "metrics.json"))
    _save_json(eval_rows, os.path.join(output_dir, "detailed_results.json"))

    # Summary
    elapsed = time.time() - start_time
    j = metrics["json"]
    c = metrics["skn"]

    lines = [
        "=" * 70,
        "  JSON vs SKN Extraction Benchmark",
        "=" * 70,
        "",
        "  Pipeline: Question -> Search -> [Extraction] -> Agent -> Answer",
        "  The extraction format is the ONLY variable.",
        "",
        f"    Model             {model}",
        f"    Total samples     {len(samples)}",
        f"    Answerable        {n_answerable}",
        f"    Unanswerable      {n_unanswerable}",
        f"    Duration          {elapsed:.1f}s",
        "",
        "  " + "." * 66,
        "  1. Accuracy (higher is better)",
        "  " + "." * 66,
        f"    JSON              {j['accuracy']:.2%}",
        f"    SKN              {c['accuracy']:.2%}" + _delta(c["accuracy"], j["accuracy"]),
        "",
        "  " + "." * 66,
        "  2. Confidence Calibration (lower ECE is better)",
        "  " + "." * 66,
        f"    JSON ECE          {j['ece']:.4f}",
        f"    SKN ECE          {c['ece']:.4f}" + _delta(c["ece"], j["ece"], False),
        f"    JSON Conf/Right   {j['confidence_when_correct']:.4f}",
        f"    SKN Conf/Right   {c['confidence_when_correct']:.4f}",
        f"    JSON Conf/Wrong   {j['confidence_when_wrong']:.4f}",
        f"    SKN Conf/Wrong   {c['confidence_when_wrong']:.4f}",
        "",
        "  " + "." * 66,
        "  3. Hallucination (lower is better)",
        "  " + "." * 66,
        f"    JSON Rate         {j['hallucination_rate']:.2%}",
        f"    SKN Rate         {c['hallucination_rate']:.2%}" + _delta(c["hallucination_rate"], j["hallucination_rate"], False),
        f"    JSON Severity     {j['hallucination_severity']:.4f}",
        f"    SKN Severity     {c['hallucination_severity']:.4f}" + _delta(c["hallucination_severity"], j["hallucination_severity"], False),
        "",
        "  " + "." * 66,
        "  4. Groundedness (higher is better)",
        "  " + "." * 66,
        f"    JSON Score        {j['groundedness']:.4f}",
        f"    SKN Score        {c['groundedness']:.4f}" + _delta(c["groundedness"], j["groundedness"]),
        f"    JSON Unsupported  {j['unsupported_fraction']:.2%}",
        f"    SKN Unsupported  {c['unsupported_fraction']:.2%}" + _delta(c["unsupported_fraction"], j["unsupported_fraction"], False),
        "",
        "  " + "." * 66,
        "  5. Abstention (higher F1 is better)",
        "  " + "." * 66,
        f"    JSON F1           {j['abstention_f1']:.4f}",
        f"    SKN F1           {c['abstention_f1']:.4f}" + _delta(c["abstention_f1"], j["abstention_f1"]),
        "",
        "  " + "." * 66,
        "  6. Token Efficiency",
        "  " + "." * 66,
        f"    Avg Search Time   {avg_search:.2f}s",
        f"    JSON Avg Tokens   {j['avg_context_tokens']}",
        f"    SKN Avg Tokens   {c['avg_context_tokens']}",
        f"    JSON Extract Time {j['avg_extraction_time_s']:.2f}s",
        f"    SKN Extract Time {c['avg_extraction_time_s']:.2f}s",
        f"    JSON Total Lat    {j['avg_total_latency_s']:.2f}s",
        f"    SKN Total Lat    {c['avg_total_latency_s']:.2f}s",
        "",
        "=" * 70,
        "  Verdict",
        "=" * 70,
    ]

    skn_wins = json_wins = 0
    for name, skn_v, json_v, higher in [
        ("Accuracy", c["accuracy"], j["accuracy"], True),
        ("Calibration (ECE)", c["ece"], j["ece"], False),
        ("Hallucination Rate", c["hallucination_rate"], j["hallucination_rate"], False),
        ("Groundedness", c["groundedness"], j["groundedness"], True),
        ("Abstention F1", c["abstention_f1"], j["abstention_f1"], True),
    ]:
        if abs(skn_v - json_v) < 0.0001:
            lines.append(f"    {name:25s}  TIED")
        elif (higher and skn_v > json_v) or (not higher and skn_v < json_v):
            skn_wins += 1
            lines.append(f"    {name:25s}  SKN wins")
        else:
            json_wins += 1
            lines.append(f"    {name:25s}  JSON wins")

    lines.append("")
    lines.append(f"    SKN wins {skn_wins}/5, JSON wins {json_wins}/5")
    if skn_wins > json_wins:
        lines.append("    SKN is the superior extraction format for AI reasoning.")
    elif json_wins > skn_wins:
        lines.append("    JSON outperforms SKN in this benchmark.")
    else:
        lines.append("    Results are inconclusive.")
    lines.append("=" * 70)

    summary = "\n".join(lines)
    _save_text(summary, os.path.join(output_dir, "summary.txt"))
    print("\n" + summary)
    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run JSON vs SKN extraction benchmark"
    )
    parser.add_argument("--dataset", default="data/samples.json")
    parser.add_argument("--output", default="results")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    args = parser.parse_args()

    run_benchmark(args.dataset, args.output, args.model)
