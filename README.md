# SKN vs JSON Extraction Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Thesis

Current AI pipelines extract information from web search results into **JSON** before passing it to a reasoning LLM. JSON carries raw facts but zero epistemic metadata: no per-claim confidence, no causal structure, no knowledge gaps, no risk signals. The reasoning model is left to guess what is reliable and what is missing.

**Structured Knowledge Notation (SKN)** is a drop-in replacement for JSON at the extraction layer. It encodes confidence, causality, gaps, and risk inline using a token-efficient notation. This benchmark isolates the extraction format as the single variable and measures its downstream impact on agent reasoning.


## What Changed (v2)

| Feature | v1 | v2 |
|---------|----|----|
| LLM Provider | Anthropic Claude API (paid) | **Ollama (local, free)** |
| Default Model | `claude-sonnet-4-20250514` | **`qwen2.5:7b`** |
| Dataset Size | 15 samples | **50 samples** |
| Categories | 6 | **7** (added `multi_step`) |
| Extraction Variants | 2 (JSON, SKN) | **6** (ablation study) |
| Hallucination Metric | Binary detection only | **Normalized (fabricated/total claims)** |
| Statistical Rigor | Point estimates | **Bootstrap 95% confidence intervals** |
| Code Duplication | 3 files with copy-pasted API client | **Shared `llm.py` module** |
| API Key Required | Yes | **No** |


## Architecture

```
                    +---------------------------------------------------+
                    |                   SHARED                          |
  Question ------  |  DuckDuckGo Web Search                            |
  (from dataset)   |  Returns top 5 results (title + snippet + URL)    |
                    +------------------------+--------------------------+
                                             |
                                same raw search results
                                             |
              +----------+----------+----------+----------+----------+
              |          |          |          |          |          |
              v          v          v          v          v          v
          json       json_conf  skn_no_gaps skn_no_causal skn_no_risk  skn
        (baseline)  (+ confidence) (- @gaps) (- @causal)  (- @risk)  (full)
              |          |          |          |          |          |
              v          v          v          v          v          v
            Same agent_reason() for all  -->  Same evaluator for all
```

Default mode runs 2 pipelines (JSON vs full SKN). Ablation mode runs all 6.

### Execution Phases

| Phase | What Happens | Calls per Sample (default) | Calls per Sample (ablation) |
|-------|-------------|---------------------------|----------------------------|
| 1. Search | `search_web(question)` via DuckDuckGo | 1 HTTP | 1 HTTP |
| 2. Extraction | Extract using each pipeline variant | 2 LLM | 6 LLM |
| 3. Reasoning | `agent_reason(context, q)` per variant | 2 LLM | 6 LLM |
| 4. Evaluation | Accuracy, hallucination, groundedness judges + ECE + abstention | 6 LLM | 18 LLM |

Default: ~11 calls/sample x 50 samples = ~550 local LLM calls.
Ablation: ~31 calls/sample x 50 samples = ~1,550 local LLM calls.


## Ablation Study

The ablation isolates which SKN component drives the accuracy improvement. Without this, someone could argue "SKN wins because it carries more information, not because of its format."

| Variant | What It Tests |
|---------|--------------|
| `json` | Baseline: flat entities/relationships/claims, no metadata |
| `json_conf` | JSON + per-claim confidence scores (tests: does adding confidence to JSON close the gap?) |
| `skn_no_gaps` | Full SKN minus `@gaps` (tests: how much does explicit gap listing matter?) |
| `skn_no_causal` | Full SKN minus `@causal` (tests: do causal chains help or hurt?) |
| `skn_no_risk` | Full SKN minus `@risk` (tests: does misdirection signaling matter?) |
| `skn` | Full SKN with all sections (proposed format) |

Run with:
```bash
python -m src.benchmark --ablation
```


## New Metrics

### Normalized Hallucination Rate

The raw hallucination detection rate penalizes longer answers. A 10-sentence answer has more surface area for hallucination flags than a 3-sentence answer.

The normalized metric fixes this:
```
normalized_hallucination = fabricated_claims_count / total_claims_count
```

This measures hallucination density rather than presence, making comparisons fairer across formats that produce different answer lengths.

### Bootstrap Confidence Intervals

All key metrics now include 95% bootstrap confidence intervals (1000 resamples, seed 42). This answers: "Could this result be due to chance?"

```
Accuracy: 0.85 (95% CI: [0.78, 0.92])
```

If the confidence intervals for two formats overlap, the difference is not statistically meaningful.


## Previous Results (Claude Sonnet 4, n=15)

These results were obtained with `claude-sonnet-4-20250514` on the original 15-sample dataset. They demonstrate the format's potential but should be interpreted with the caveats noted in Limitations.

| Metric | JSON | SKN | Winner | Delta |
|--------|------|-----|--------|-------|
| **Accuracy** | 75.0% | **100.0%** | SKN | +25.0 pp |
| **ECE (calibration)** | **0.1867** | 0.2325 | JSON | +0.046 |
| **Hallucination rate** | **40.0%** | 60.0% | JSON | +20.0 pp |
| **Groundedness** | **0.750** | 0.727 | JSON | -0.023 |
| **Abstention F1** | 0.857 | **1.000** | SKN | +0.143 |
| **Avg context tokens** | 686 | **273** | SKN | 2.5x fewer |
| **Avg total latency** | 19.39s | **15.86s** | SKN | 18% faster |

Run the benchmark with your local Ollama model to generate fresh results on the expanded 50-sample dataset.


## SKN Format Specification

```
[SKN]
@src <domain> | fresh:<high|medium|low> | reliability:<0.0-1.0>

@facts
  ! "critical fact from source" [<confidence 0.0-1.0>]
  . "supporting detail" [<confidence>]
  ~ "uncertain / inferred claim" [<confidence>]

@causal
  <cause> -> <effect> [<strength 0.0-1.0>]

@gaps
  - <information missing from the source>

@risk misdirection:<low|medium|high> | missing_context:<low|medium|high>
[/SKN]
```

### Symbol Reference

| Symbol | Meaning | Usage |
|--------|---------|-------|
| `!` | High-priority fact | Critical information directly from source |
| `.` | Supporting detail | Secondary facts, context |
| `~` | Uncertain / inferred | Text implies but does not explicitly state |
| `->` | Causal link | Directional cause-effect with strength score |
| `[0.0-1.0]` | Confidence / strength | Inline score reflecting source support |

### Sections

| Section | Purpose |
|---------|---------|
| `@src` | Source domain classification, freshness, reliability |
| `@facts` | Priority-ordered claims with per-claim confidence |
| `@causal` | Cause-effect chains with directional strength |
| `@gaps` | Explicit list of what the source does NOT contain |
| `@risk` | Misdirection and missing context risk assessment |


## Evaluation Metrics

### 1. Accuracy
LLM judge compares agent answer to ground truth semantically. Scored on answerable samples only.

```
Metric: correct_count / answerable_count
With 95% bootstrap CI.
```

### 2. Expected Calibration Error (ECE)
Measures gap between agent confidence and actual correctness using 10 equal-width bins.

```
ECE = sum over bins: (bin_size / total) * |accuracy_in_bin - avg_confidence_in_bin|
Lower is better. 0.0 = perfectly calibrated.
```

### 3. Hallucination Rate + Normalized Hallucination
LLM judge checks each answer against search results for fabricated claims.

```
Returns: {hallucination_detected: bool, hallucination_severity: 0.0-1.0, total_claims: int, fabricated_claims: [...]}
Metrics: detection_rate, avg_severity, normalized_hallucination (fabricated/total)
With 95% bootstrap CI on normalized metric.
```

### 4. Groundedness
LLM judge scores how well the answer is traceable to the search results.

```
Returns: {groundedness_score: 0.0-1.0, unsupported_fraction: 0.0-1.0}
Metrics: avg_groundedness (higher is better), avg_unsupported (lower is better)
With 95% bootstrap CI.
```

### 5. Abstention
Keyword-based detection of refusal to answer on unanswerable samples.

```
Metrics: precision, recall, F1 (higher is better)
```

### 6. Token Efficiency
Estimated token count of extraction output and end-to-end latency.

```
Metrics: avg_context_tokens, avg_search_time_s, avg_extraction_time_s, avg_total_latency_s
```


## Project Structure

```
csb-benchmark/
+-- data/
|   +-- samples.json            # 50 questions across 7 categories
+-- src/
|   +-- __init__.py
|   +-- llm.py                  # Shared Ollama client (all modules import from here)
|   +-- searcher.py             # search_web(query) via DuckDuckGo
|   +-- extractors.py           # 6 extraction variants + registry
|   +-- agent.py                # agent_reason(context, question)
|   +-- evaluator.py            # LLM judges + ECE + abstention + bootstrap CI
|   +-- benchmark.py            # 4-phase orchestrator with ablation support
+-- results/                    # Generated after benchmark run
|   +-- search_results.json
|   +-- extractions.json
|   +-- agent_responses.json
|   +-- detailed_results.json
|   +-- metrics.json            # Includes bootstrap CIs and normalized hallucination
|   +-- summary.txt
+-- .env.example
+-- requirements.txt
+-- README.md
+-- LICENSE
```

### Module API

**llm.py**
```python
def call_llm(system: str, user_message: str, model: str = "qwen2.5:7b",
             json_mode: bool = False) -> str
```

**searcher.py**
```python
def search_web(query: str, num_results: int = 5) -> str
```

**extractors.py**
```python
def extract_json(raw_text: str, model: str = "qwen2.5:7b") -> dict[str, Any]
def extract_json_conf(raw_text: str, model: str = "qwen2.5:7b") -> dict[str, Any]
def extract_skn(raw_text: str, model: str = "qwen2.5:7b") -> str
def extract_skn_no_gaps(raw_text: str, model: str = "qwen2.5:7b") -> str
def extract_skn_no_causal(raw_text: str, model: str = "qwen2.5:7b") -> str
def extract_skn_no_risk(raw_text: str, model: str = "qwen2.5:7b") -> str

EXTRACTORS: dict[str, Callable]       # name -> function registry
DEFAULT_PIPELINES = ["json", "skn"]
ABLATION_PIPELINES = ["json", "json_conf", "skn_no_gaps", "skn_no_causal", "skn_no_risk", "skn"]
```

**agent.py**
```python
def agent_reason(context: str | dict, question: str, model: str = "qwen2.5:7b") -> dict[str, Any]
# Returns: {"answer": str, "confidence": float, "reasoning": str}
```

**evaluator.py**
```python
def calculate_metrics(results: list[dict], pipelines: list[str] = None, model: str = "qwen2.5:7b") -> dict[str, Any]
def bootstrap_ci(values: list[float], n_bootstrap: int = 1000, ci: float = 0.95) -> dict[str, float]
# Returns per pipeline: accuracy, accuracy_ci, ece, hallucination_rate, normalized_hallucination,
#   normalized_hallucination_ci, groundedness, groundedness_ci, abstention_f1, ...
```

**benchmark.py**
```python
def run_benchmark(dataset_path: str, output_dir: str = "results",
                  model: str = "qwen2.5:7b", ablation: bool = False) -> dict[str, Any]
```


## Dataset

50 samples across 7 categories. No raw_text provided. The system searches the internet live for each question.

| Category | Count | Purpose |
|----------|-------|---------|
| `diagnostic` | 15 | Root cause analysis questions searchable online |
| `optimization` | 7 | Performance tuning with known best practices |
| `architecture` | 5 | Distributed systems design questions |
| `insufficient_info` | 8 | Questions about fictitious internal systems (no online info exists) |
| `ambiguous` | 5 | Multiple valid answers (tests nuanced reasoning) |
| `misleading` | 5 | Question implies wrong root cause (tests misdirection resistance) |
| `multi_step` | 5 | Debugging requiring chaining 2-3 pieces of information |

Technologies covered: Python, Java, Go, Rust, TypeScript, Node.js, Docker, Kubernetes, PostgreSQL, Redis, Elasticsearch, Kafka, RabbitMQ, Terraform, AWS Lambda, gRPC.


## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ujjwalredd/Structured-Knowledge-Notation.git
cd Structured-Knowledge-Notation
```

### 2. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Pull the model

```bash
ollama pull qwen2.5:7b
```

Other recommended models:
| Model | VRAM | Quality | Speed |
|-------|------|---------|-------|
| `qwen2.5:7b` | ~4.7 GB | Good (recommended default) | Fast |
| `qwen2.5:14b` | ~9 GB | Better | Moderate |
| `llama3.1:8b` | ~4.7 GB | Good | Fast |
| `llama3.3:70b` | ~40 GB | Best | Slow |

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

No API key required. DuckDuckGo search and Ollama are both free.


## Running the Benchmark

```bash
# Default: JSON vs full SKN (50 samples, ~550 local LLM calls)
python -m src.benchmark

# Ablation: all 6 extraction variants (~1,550 local LLM calls)
python -m src.benchmark --ablation

# Custom model
python -m src.benchmark --model qwen2.5:14b

# All options
python -m src.benchmark --dataset data/samples.json --output results --model qwen2.5:7b --ablation
```


## Technical Details

| Parameter | Value |
|-----------|-------|
| LLM Provider | Ollama (local inference) |
| Search Provider | DuckDuckGo (via `ddgs` package) |
| Default Model | `qwen2.5:7b` |
| Temperature | 0 (deterministic) |
| Max Tokens | 4096 |
| Search Results | Top 5 per query |
| Retry Strategy | Exponential backoff (base 2s, max 3 retries) |
| ECE Bins | 10 equal-width bins [0.0, 1.0] |
| Token Estimation | `len(text) // 4` |
| Bootstrap Resamples | 1000 (seed 42) |
| Confidence Level | 95% |
| JSON Mode | Ollama `format="json"` for structured output |


## Limitations

1. **LLM-as-judge variance.** Accuracy, hallucination, and groundedness are judged by the same model that generates the answers. LLM judges can exhibit systematic biases. Using a separate, larger model as judge would reduce this bias.

2. **Single model per run.** Results depend heavily on the chosen model. A model with weaker instruction-following might not leverage `@gaps` and `@risk` effectively. Run with multiple models to compare.

3. **Search result volatility.** DuckDuckGo results change over time. Running on different days may produce different search results, affecting all downstream metrics. The `search_results.json` output file preserves the exact results used in each run.

4. **Token estimation is approximate.** Using `len(text) // 4` is a rough heuristic. Actual tokenization varies by model. The relative comparison (SKN vs JSON) remains directionally valid.

5. **Local model quality vs cloud APIs.** Smaller local models (7B-14B parameters) may produce lower absolute scores than cloud models (Claude, GPT-4) but the relative comparison between extraction formats remains meaningful.


## Prior Work and Gap Analysis

| System | What It Does | What It Lacks |
|--------|-------------|---------------|
| OpenAI Web Search | URL citations with text spans | No per-source reliability score |
| Google Gemini Grounding | URI + byte-offset chunk mapping | No per-claim confidence |
| Microsoft GraphRAG | Entity-relationship knowledge graphs | Treats extraction as deterministic, no uncertainty |
| LangChain / LlamaIndex | Document objects with metadata dicts | Untyped metadata, no enforced quality schema |
| CRAG (2024) | Classifies docs as correct/incorrect/ambiguous | Confidence used only as retrieval gate, not passed to generator |
| Bayesian RAG (2025) | Monte Carlo dropout for uncertainty | Used solely for re-ranking, not structured for reasoning |
| Self-RAG (ICLR 2024) | Reflection tokens for quality assessment | Baked into model weights, not a separable data structure |
| DRAGIN (ACL 2024) | Token-level entropy triggers retrieval | Entropy signal is ephemeral, never structured |

**Gap:** No existing system passes a unified structured block with per-claim confidence, causal chains, knowledge gaps, and risk signals from the retrieval layer to the reasoning layer. SKN fills this gap with minimal token overhead.


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
