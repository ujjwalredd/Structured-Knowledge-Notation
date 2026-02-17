"""
Main benchmark runner for JSON vs SKN extraction comparison.

Supports two modes:
  --ablation    Run all 6 extraction variants (JSON, JSON+conf, SKN variants)
  (default)     Run JSON vs full SKN only

Pipeline:
    Question -> Web Search -> Raw Results -> [Extraction] -> Same Agent -> Answer

The extraction step is the ONLY variable.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import agent_reason
from src.evaluator import calculate_metrics
from src.extractors import EXTRACTORS, DEFAULT_PIPELINES, ABLATION_PIPELINES
from src.searcher import search_web

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


def _delta(val_a: float, val_b: float, higher_is_better: bool = True) -> str:
    diff = val_a - val_b
    if abs(diff) < 0.0001:
        return "  (tied)"
    if higher_is_better:
        winner = "A" if diff > 0 else "B"
    else:
        winner = "A" if diff < 0 else "B"
    return f"  (delta {diff:+.4f}, {winner} better)"


def run_benchmark(
    dataset_path: str,
    output_dir: str = "results",
    model: str = "qwen2.5:7b",
    ablation: bool = False,
) -> dict[str, Any]:
    """Run the extraction benchmark.

    Parameters
    ----------
    dataset_path : str
        Path to samples JSON file.
    output_dir : str
        Directory for output files.
    model : str
        Ollama model name.
    ablation : bool
        If True, run all 6 extraction variants. Otherwise just JSON vs SKN.
    """
    start_time = time.time()
    samples = _load_dataset(dataset_path)

    pipelines = ABLATION_PIPELINES if ablation else DEFAULT_PIPELINES
    n_answerable = sum(1 for s in samples if s.get("answerable", True))
    n_unanswerable = len(samples) - n_answerable

    logger.info(
        "Running %s mode: %d pipelines x %d samples",
        "ABLATION" if ablation else "DEFAULT",
        len(pipelines),
        len(samples),
    )

    search_results: list[dict[str, Any]] = []
    extractions: list[dict[str, Any]] = []
    agent_responses: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    # Timing and token tracking per pipeline
    extract_times: dict[str, list[float]] = {p: [] for p in pipelines}
    reason_times: dict[str, list[float]] = {p: [] for p in pipelines}
    token_counts: dict[str, list[int]] = {p: [] for p in pipelines}
    search_times: list[float] = []

    # Phase 1: Web Search (shared across all pipelines)
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

    # Phase 2: Extraction (run each pipeline variant)
    logger.info("=== Phase 2: Extraction ===")
    for sample in tqdm(samples, desc="Extracting", unit="sample"):
        sid = sample["id"]
        raw = sample["raw_text"]
        ext_row: dict[str, Any] = {"id": sid}

        for pipe_name in pipelines:
            extractor_fn = EXTRACTORS[pipe_name]
            t0 = time.time()
            try:
                output = extractor_fn(raw, model=model)
            except Exception as exc:
                logger.error(
                    "%s extraction failed for sample %s: %s",
                    pipe_name, sid, exc,
                )
                output = "" if "skn" in pipe_name else {
                    "entities": [], "relationships": [], "claims": []
                }
            extract_times[pipe_name].append(time.time() - t0)

            _, tokens = _context_size(output)
            token_counts[pipe_name].append(tokens)
            ext_row[f"{pipe_name}_extraction"] = output
            ext_row[f"{pipe_name}_tokens_est"] = tokens

        extractions.append(ext_row)

    _save_json(extractions, os.path.join(output_dir, "extractions.json"))

    # Phase 3: Agent reasoning (same agent, different context per pipeline)
    logger.info("=== Phase 3: Agent Reasoning ===")
    for sample, ext in tqdm(
        zip(samples, extractions),
        total=len(samples),
        desc="Reasoning",
        unit="sample",
    ):
        sid = sample["id"]
        question = sample["question"]
        resp_row: dict[str, Any] = {"id": sid, "question": question}
        eval_row: dict[str, Any] = {
            "id": sid,
            "category": sample.get("category", "unknown"),
            "raw_text": sample["raw_text"],
            "ground_truth": sample["ground_truth"],
            "answerable": sample.get("answerable", True),
        }

        for pipe_name in pipelines:
            context = ext[f"{pipe_name}_extraction"]
            t0 = time.time()
            try:
                resp = agent_reason(context, question, model=model)
            except Exception as exc:
                logger.error(
                    "Agent (%s) failed for sample %s: %s",
                    pipe_name, sid, exc,
                )
                resp = {"answer": "", "confidence": 0.0, "reasoning": "error"}
            reason_times[pipe_name].append(time.time() - t0)

            resp_row[f"{pipe_name}_response"] = resp
            eval_row[f"{pipe_name}_answer"] = resp["answer"]
            eval_row[f"{pipe_name}_confidence"] = resp["confidence"]
            eval_row[f"{pipe_name}_reasoning"] = resp["reasoning"]

        agent_responses.append(resp_row)
        eval_rows.append(eval_row)

    _save_json(agent_responses, os.path.join(output_dir, "agent_responses.json"))

    # Phase 4: Evaluation
    logger.info("=== Phase 4: Evaluation ===")
    metrics = calculate_metrics(eval_rows, pipelines=pipelines, model=model)

    # Efficiency metrics per pipeline
    avg_search = sum(search_times) / len(search_times)

    for pipe_name in pipelines:
        avg_ext = sum(extract_times[pipe_name]) / len(extract_times[pipe_name])
        avg_rsn = sum(reason_times[pipe_name]) / len(reason_times[pipe_name])
        avg_tok = sum(token_counts[pipe_name]) / len(token_counts[pipe_name])

        metrics[pipe_name]["avg_search_time_s"] = round(avg_search, 2)
        metrics[pipe_name]["avg_extraction_time_s"] = round(avg_ext, 2)
        metrics[pipe_name]["avg_reasoning_time_s"] = round(avg_rsn, 2)
        metrics[pipe_name]["avg_total_latency_s"] = round(
            avg_search + avg_ext + avg_rsn, 2
        )
        metrics[pipe_name]["avg_context_tokens"] = round(avg_tok)

    # Cross-pipeline efficiency
    if "json" in pipelines and "skn" in pipelines:
        json_tok = metrics["json"]["avg_context_tokens"]
        skn_tok = metrics["skn"]["avg_context_tokens"]
        metrics["efficiency"] = {
            "token_ratio_skn_vs_json": round(skn_tok / json_tok, 2) if json_tok > 0 else 0.0,
        }

    _save_json(metrics, os.path.join(output_dir, "metrics.json"))
    _save_json(eval_rows, os.path.join(output_dir, "detailed_results.json"))

    # Summary
    elapsed = time.time() - start_time

    lines = [
        "=" * 70,
        f"  SKN Benchmark ({'Ablation' if ablation else 'Default'} Mode)",
        "=" * 70,
        "",
        "  Pipeline: Question -> Search -> [Extraction] -> Agent -> Answer",
        "  The extraction format is the ONLY variable.",
        "",
        f"    Model             {model}",
        f"    Total samples     {len(samples)}",
        f"    Answerable        {n_answerable}",
        f"    Unanswerable      {n_unanswerable}",
        f"    Pipelines         {', '.join(pipelines)}",
        f"    Duration          {elapsed:.1f}s",
        "",
    ]

    for pipe_name in pipelines:
        m = metrics[pipe_name]
        lines.extend([
            "  " + "." * 66,
            f"  {pipe_name.upper()}",
            "  " + "." * 66,
            f"    Accuracy          {m['accuracy']:.2%}  "
            f"(95% CI: [{m['accuracy_ci']['ci_low']:.2%}, {m['accuracy_ci']['ci_high']:.2%}])",
            f"    ECE               {m['ece']:.4f}",
            f"    Hallucination     {m['hallucination_rate']:.2%}",
            f"    Norm. Halluc.     {m['normalized_hallucination']:.4f}  "
            f"(95% CI: [{m['normalized_hallucination_ci']['ci_low']:.4f}, "
            f"{m['normalized_hallucination_ci']['ci_high']:.4f}])",
            f"    Groundedness      {m['groundedness']:.4f}  "
            f"(95% CI: [{m['groundedness_ci']['ci_low']:.4f}, "
            f"{m['groundedness_ci']['ci_high']:.4f}])",
            f"    Abstention F1     {m['abstention_f1']:.4f}",
            f"    Avg Tokens        {m['avg_context_tokens']}",
            f"    Avg Latency       {m['avg_total_latency_s']:.2f}s",
            "",
        ])

    # Head-to-head verdict (JSON vs full SKN)
    if "json" in pipelines and "skn" in pipelines:
        j = metrics["json"]
        s = metrics["skn"]

        lines.extend([
            "=" * 70,
            "  Verdict: JSON vs SKN",
            "=" * 70,
        ])

        skn_wins = json_wins = 0
        for name, skn_v, json_v, higher in [
            ("Accuracy", s["accuracy"], j["accuracy"], True),
            ("Calibration (ECE)", s["ece"], j["ece"], False),
            ("Hallucination Rate", s["hallucination_rate"], j["hallucination_rate"], False),
            ("Groundedness", s["groundedness"], j["groundedness"], True),
            ("Abstention F1", s["abstention_f1"], j["abstention_f1"], True),
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
    parser.add_argument("--model", default="qwen2.5:7b")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run all 6 extraction variants (ablation study)",
    )
    args = parser.parse_args()

    run_benchmark(args.dataset, args.output, args.model, args.ablation)
