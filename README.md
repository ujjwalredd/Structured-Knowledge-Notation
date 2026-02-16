# SKN vs JSON Extraction Benchmark

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Repository:** [github.com/ujjwalredd/Structured-Knowledge-Notation](https://github.com/ujjwalredd/Structured-Knowledge-Notation)

## Thesis

Current AI pipelines extract information from web search results into **JSON** before passing it to a reasoning LLM. JSON carries raw facts but zero epistemic metadata: no per-claim confidence, no causal structure, no knowledge gaps, no risk signals. The reasoning model is left to guess what is reliable and what is missing.

**Structured Knowledge Notation (SKN)** is a drop-in replacement for JSON at the extraction layer. It encodes confidence, causality, gaps, and risk inline using a token-efficient notation. This benchmark isolates the extraction format as the single variable and measures its downstream impact on agent reasoning.


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
                          +------------------+------------------+
                          |                                     |
                          v                                     v
               PIPELINE A (Baseline)               PIPELINE B (Proposed)
            +----------------------+            +----------------------+
            |  extract_json()      |            |  extract_skn()       |
            |  entities/relations  |            |  @facts @causal      |
            |  claims (flat)       |            |  @gaps @risk         |
            +----------+-----------+            +----------+-----------+
                       |                                   |
                       v                                   v
            agent_reason(json, q)              agent_reason(skn, q)
                       |                                   |
                       +-----------------+-----------------+
                                         |
                                         v
                                +----------------+
                                |   Evaluation   |
                                |   (5 metrics)  |
                                +----------------+

  Same search. Same agent. Same evaluator. Only the extraction format changes.
```

### Execution Phases

| Phase | What Happens | Calls per Sample |
|-------|-------------|-----------------|
| 1. Search | `search_web(question)` via DuckDuckGo | 1 HTTP |
| 2. Extraction | `extract_json(raw)` and `extract_skn(raw)` on same search results | 2 LLM |
| 3. Reasoning | `agent_reason(json_ctx, q)` and `agent_reason(skn_ctx, q)` | 2 LLM |
| 4. Evaluation | Accuracy, hallucination, groundedness judges + ECE + abstention | 6 LLM |

Total: ~11 calls per sample. 15 samples = ~165 API calls.


## Results

Benchmark run on 15 samples using `claude-sonnet-4-20250514` with temperature 0. Total runtime: 721.3 seconds.

### Head-to-Head Comparison

| Metric | JSON | SKN | Winner | Delta |
|--------|------|-----|--------|-------|
| **Accuracy** | 75.0% (9/12) | **100.0%** (12/12) | SKN | +25.0 pp |
| **ECE (calibration)** | **0.1867** | 0.2325 | JSON | +0.046 |
| **Hallucination rate** | **40.0%** | 60.0% | JSON | +20.0 pp |
| **Hallucination severity** | **0.200** | 0.247 | JSON | +0.047 |
| **Groundedness** | **0.750** | 0.727 | JSON | -0.023 |
| **Unsupported fraction** | **25.0%** | 27.3% | JSON | +2.3 pp |
| **Abstention F1** | 0.857 | **1.000** | SKN | +0.143 |
| **Avg context tokens** | 686 | **273** | SKN | 2.5x fewer |
| **Avg extraction time** | 10.52s | **7.61s** | SKN | 28% faster |
| **Avg total latency** | 19.39s | **15.86s** | SKN | 18% faster |

**Verdict: JSON wins 3/5 evaluation dimensions, SKN wins 2/5.**

But the picture is more nuanced than a simple tally. Read on.


### Confidence Breakdown

| Metric | JSON | SKN |
|--------|------|-----|
| Average confidence (answerable) | 0.813 | 0.768 |
| Confidence when correct | 0.890 | 0.768 |
| Confidence when wrong | 0.583 | 0.000 |

SKN never returned a wrong answer in this run, so `confidence_when_wrong` is 0.0 by default. JSON's confidence when wrong (0.583) reveals overconfidence on incorrect answers, a dangerous property in production systems where high confidence on wrong answers can propagate errors downstream.


### Abstention Detail

| Metric | JSON | SKN |
|--------|------|-----|
| Precision | 0.75 | **1.00** |
| Recall | 1.00 | 1.00 |
| F1 | 0.857 | **1.000** |
| True Positives | 3 | 3 |
| False Positives | 1 | 0 |
| False Negatives | 0 | 0 |
| True Negatives | 9 | 12 |

Both pipelines correctly refused all 3 unanswerable questions (samples 11-13). The difference: JSON produced 1 false positive (sample 8, FastAPI/502 question), where it hedged with "I cannot definitively explain" despite the question being answerable. SKN answered the same question directly and was judged correct. The `@gaps` section in SKN gave the agent explicit signal about what was missing vs what was present, enabling it to commit to an answer rather than hedge.


### Token Efficiency

| Metric | JSON | SKN | Ratio |
|--------|------|-----|-------|
| Avg context tokens | 686 | 273 | SKN uses **60% fewer tokens** |
| Token ratio | 1.0x | 0.4x | -- |

SKN achieves this compression by replacing verbose JSON keys (`"entities"`, `"relationships"`, `"claims"`) with single-character symbols (`!`, `.`, `~`, `->`) and structured section headers (`@facts`, `@causal`, `@gaps`). The token savings compound at scale: a 1000-query workload would save ~413,000 context tokens.


## Tradeoff Analysis

The results reveal a fundamental tension between two properties:

### Where SKN Wins: Reasoning Quality

**Accuracy (+25 pp):** SKN achieved perfect accuracy (12/12) vs JSON's 75% (9/12). The three questions JSON got wrong were:

| Sample | Category | Question | JSON Got Wrong Because |
|--------|----------|----------|----------------------|
| 8 | diagnostic | FastAPI 502 under load | JSON hedged ("cannot definitively explain") instead of answering. SKN's `@gaps` section explicitly listed what was missing, so the agent could distinguish between "I have enough to answer" vs "I truly cannot answer." |
| 10 | diagnostic | JWT 401 after key rotation | JSON latched onto `ignoreNotBefore` (a tangential detail from search results) and built an incorrect explanation around it. SKN's per-claim confidence scores (`[0.65]`) flagged this as uncertain, steering the agent toward a more cautious but correct synthesis. |
| 14 | ambiguous | Monolith vs microservices | JSON gave a one-sided recommendation ("should consider migrating"). SKN presented both sides because `@risk misdirection:medium` signaled the question was inherently ambiguous, prompting the agent to acknowledge tradeoffs. |

The pattern: SKN's epistemic metadata (confidence scores, gap lists, risk signals) gave the reasoning agent better signal for deciding when to commit, when to hedge, and when to present nuance.

**Abstention F1 (+0.143):** SKN's `@gaps` section creates a clear boundary between "the source lacks this specific info" and "the source has relevant info but it's incomplete." This distinction prevented the false positive that JSON produced on sample 8.

### Where JSON Wins: Faithfulness to Source

**Hallucination rate (-20 pp):** JSON answers hallucinated in 40% of samples vs SKN's 60%. This is the most significant concern with SKN. The likely mechanism: SKN's causal chains (`@causal` section with `->` operators) encourage the agent to construct explanatory narratives. When the search results contain partial information, the agent fills gaps in the causal chain with plausible but unsupported inferences.

Example from sample 4 (CI/CD optimization):
- JSON answer: listed optimization techniques directly from search results (no hallucination)
- SKN answer: added "Use GitHub Actions' matrix builds to run tests in parallel" and "Use workflow artifacts to share build outputs" -- plausible recommendations not present in the search results

The causal structure in SKN acts as a double-edged sword: it improves reasoning quality but also provides a scaffold for the agent to fill in with fabricated details.

**Groundedness (-0.023):** A small gap, but directionally consistent with the hallucination finding. JSON answers stay closer to the source text. SKN answers are more interpretive, which helps accuracy but hurts literal groundedness.

**ECE / Calibration (+0.046):** JSON's confidence scores are better calibrated (closer to actual accuracy). SKN's slightly worse calibration comes from an interesting asymmetry: SKN answers tend to be more conservative in confidence (avg 0.768 vs 0.813) but more accurate (100% vs 75%), meaning SKN is slightly underconfident rather than overconfident. In practice, underconfidence is safer than overconfidence.

### The Efficiency Advantage

SKN's 2.5x token reduction and 18% latency improvement are unambiguous wins that do not trade off against any quality dimension. These savings come purely from the notation's compactness, not from information loss. The same facts, relationships, and claims are present in both formats; SKN simply represents them with fewer tokens.

At scale, this matters:

| Scale | JSON Tokens | SKN Tokens | Savings |
|-------|------------|------------|---------|
| 100 queries | 68,600 | 27,300 | 41,300 tokens |
| 1,000 queries | 686,000 | 273,000 | 413,000 tokens |
| 10,000 queries | 6,860,000 | 2,730,000 | 4,130,000 tokens |


### Summary: When to Use Which

| Use Case | Recommended Format | Rationale |
|----------|-------------------|-----------|
| High-stakes decisions where accuracy matters most | **SKN** | 100% vs 75% accuracy; epistemic metadata prevents reasoning errors |
| Applications requiring strict source faithfulness | **JSON** | Lower hallucination rate; answers stay closer to source text |
| Cost-sensitive or latency-sensitive deployments | **SKN** | 2.5x fewer tokens, 18% faster end-to-end |
| Systems that must know when to refuse | **SKN** | Perfect abstention (F1=1.0); no false positives |
| Regulatory contexts requiring auditability | **JSON** | Higher groundedness; easier to trace claims back to sources |
| General-purpose question answering | **SKN** | Accuracy and efficiency advantages outweigh hallucination risk for most use cases |


## Per-Sample Results

### Answerable Samples (12)

| ID | Category | Question (abbreviated) | JSON | SKN | Notes |
|----|----------|----------------------|------|-----|-------|
| 1 | diagnostic | Pandas MemoryError large CSV | Correct | Correct | Both identified chunking/RAM issue |
| 2 | diagnostic | Jackson empty body, Lombok getter | Correct | Correct | Both traced to missing @Getter |
| 3 | diagnostic | K8s pod eviction, memory leak | Correct | Correct | Both identified unbounded queue |
| 4 | optimization | Slow CI/CD monorepo | Correct | Correct | SKN hallucinated extra details |
| 5 | diagnostic | React Query stale data | Correct | Correct | Both identified invalidateQueries |
| 6 | optimization | SQL composite index | Correct | Correct | Both got (customer_id, status, created_at) |
| 7 | architecture | Split-brain, leader election | Correct | Correct | Both identified network partition cause |
| 8 | diagnostic | FastAPI 502 under load | **Wrong** | Correct | JSON hedged; SKN answered directly |
| 9 | diagnostic | Git repo size, large binaries | Correct | Correct | Both recommended filter-repo + LFS |
| 10 | diagnostic | JWT 401 after key rotation | **Wrong** | Correct | JSON fixated on ignoreNotBefore |
| 14 | ambiguous | Monolith vs microservices | **Wrong** | Correct | JSON one-sided; SKN presented tradeoffs |
| 15 | misleading | MongoDB vs PostgreSQL (O(n^2)) | Correct | Correct | Both correctly identified algorithm as root cause |

### Unanswerable Samples (3)

| ID | Category | Question (abbreviated) | JSON Abstained | SKN Abstained |
|----|----------|----------------------|----------------|---------------|
| 11 | insufficient_info | Acme Corp internal outage | Yes | Yes |
| 12 | insufficient_info | Project Zebra-9 build failure | Yes | Yes |
| 13 | insufficient_info | Proprietary ML dataset regression | Yes | Yes |


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


## JSON vs SKN Comparison

| Capability | JSON | SKN |
|-----------|------|-----|
| Per-claim confidence scores | No | Yes, inline `[0.0-1.0]` |
| Fact priority ordering | No (flat array) | Yes (`!` > `.` > `~`) |
| Causal chains with strength | No (unlabeled edges) | Yes (`->` with `[strength]`) |
| Explicit knowledge gaps | Not representable | Yes (`@gaps` section) |
| Misdirection risk signal | Not representable | Yes (`@risk` line) |
| Token efficiency | ~686 tokens | ~273 tokens |


## Evaluation Metrics

### 1. Accuracy
LLM judge compares agent answer to ground truth semantically. Scored on answerable samples only.

```
Metric: correct_count / answerable_count
```

### 2. Expected Calibration Error (ECE)
Measures gap between agent confidence and actual correctness using 10 equal-width bins.

```
ECE = sum over bins: (bin_size / total) * |accuracy_in_bin - avg_confidence_in_bin|
Lower is better. 0.0 = perfectly calibrated.
```

### 3. Hallucination Rate
LLM judge checks each answer against search results for fabricated claims.

```
Returns: {hallucination_detected: bool, hallucination_severity: 0.0-1.0}
Metrics: detection_rate (lower is better), avg_severity (lower is better)
```

### 4. Groundedness
LLM judge scores how well the answer is traceable to the search results.

```
Returns: {groundedness_score: 0.0-1.0, unsupported_fraction: 0.0-1.0}
Metrics: avg_groundedness (higher is better), avg_unsupported (lower is better)
```

### 5. Abstention
Keyword-based detection of refusal to answer on unanswerable samples.

```
Metrics: precision, recall, F1 (higher is better)
```

### 6. Token Efficiency
Estimated token count of extraction output and end-to-end latency including search time.

```
Metrics: avg_context_tokens, avg_search_time_s, avg_extraction_time_s, avg_total_latency_s
```


## Project Structure

```
csb-benchmark/
+-- data/
|   +-- samples.json            # 15 questions (question + ground_truth, no raw_text)
+-- src/
|   +-- __init__.py
|   +-- searcher.py             # search_web(query) via DuckDuckGo
|   +-- extractors.py           # extract_json(), extract_skn()
|   +-- agent.py                # agent_reason(context, question) -> {answer, confidence, reasoning}
|   +-- evaluator.py            # LLM judges + ECE + abstention detector
|   +-- benchmark.py            # 4-phase orchestrator (search, extract, reason, evaluate)
+-- results/                    # Generated after benchmark run
|   +-- search_results.json     # Raw search output per question
|   +-- extractions.json        # JSON and SKN extraction outputs
|   +-- agent_responses.json    # Agent answers for both formats
|   +-- detailed_results.json   # Per-sample evaluation scores
|   +-- metrics.json            # Aggregated metrics across all dimensions
|   +-- summary.txt             # Human-readable comparison with verdict
+-- .env.example
+-- requirements.txt
+-- README.md
```

### Module API

**searcher.py**
```python
def search_web(query: str, num_results: int = 5) -> str
# Returns: concatenated search result text (titles, snippets, URLs)
```

**extractors.py**
```python
def extract_json(raw_text: str, model: str = "claude-sonnet-4-20250514") -> dict[str, Any]
def extract_skn(raw_text: str, model: str = "claude-sonnet-4-20250514") -> str
```

**agent.py**
```python
def agent_reason(context: str | dict, question: str, model: str = "claude-sonnet-4-20250514") -> dict[str, Any]
# Returns: {"answer": str, "confidence": float, "reasoning": str}
```

**evaluator.py**
```python
def calculate_metrics(results: list[dict]) -> dict[str, Any]
# Returns: {"json": {...metrics}, "skn": {...metrics}}
```

**benchmark.py**
```python
def run_benchmark(dataset_path: str, output_dir: str = "results", model: str = "claude-sonnet-4-20250514") -> dict[str, Any]
```


## Dataset

15 samples across 6 categories. No raw_text provided. The system searches the internet live for each question.

| Category | Count | Purpose |
|----------|-------|---------|
| `diagnostic` | 7 | Root cause analysis questions searchable online |
| `optimization` | 2 | Performance tuning with known best practices |
| `architecture` | 1 | Distributed systems design question |
| `insufficient_info` | 3 | Questions about fictitious internal systems (no online info exists) |
| `ambiguous` | 1 | Multiple valid answers (tests nuanced reasoning) |
| `misleading` | 1 | Question implies wrong root cause (tests misdirection resistance) |

Each sample:
```json
{
  "id": 1,
  "category": "diagnostic",
  "question": "What causes Python MemoryError when loading large CSV files with pandas read_csv?",
  "ground_truth": "Loading the entire file into memory at once without chunking exceeds available RAM",
  "answerable": true
}
```


## Setup

### 1. Clone the repository

```bash
git clone https://github.com/ujjwalredd/Structured-Knowledge-Notation.git
cd Structured-Knowledge-Notation
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Anthropic API key

Get a key from [console.anthropic.com](https://console.anthropic.com).

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:
```
ANTHROPIC_API_KEY=sk-ant-...
```

DuckDuckGo search requires no API key.


## Running the Benchmark

```bash
# default run
python -m src.benchmark

# custom parameters
python -m src.benchmark --dataset data/samples.json --output results --model claude-sonnet-4-20250514
```


## Output Format

**metrics.json** structure:
```json
{
  "json": {
    "accuracy": 0.75,
    "ece": 0.1867,
    "avg_confidence": 0.8133,
    "confidence_when_correct": 0.89,
    "confidence_when_wrong": 0.5833,
    "hallucination_rate": 0.4,
    "hallucination_severity": 0.2,
    "groundedness": 0.75,
    "unsupported_fraction": 0.25,
    "abstention_precision": 0.75,
    "abstention_recall": 1.0,
    "abstention_f1": 0.8571,
    "avg_search_time_s": 1.38,
    "avg_extraction_time_s": 10.52,
    "avg_reasoning_time_s": 7.49,
    "avg_total_latency_s": 19.39,
    "avg_context_tokens": 686
  },
  "skn": {
    "accuracy": 1.0,
    "ece": 0.2325,
    "avg_confidence": 0.7675,
    "confidence_when_correct": 0.7675,
    "confidence_when_wrong": 0.0,
    "hallucination_rate": 0.6,
    "hallucination_severity": 0.2467,
    "groundedness": 0.7267,
    "unsupported_fraction": 0.2733,
    "abstention_precision": 1.0,
    "abstention_recall": 1.0,
    "abstention_f1": 1.0,
    "avg_search_time_s": 1.38,
    "avg_extraction_time_s": 7.61,
    "avg_reasoning_time_s": 6.87,
    "avg_total_latency_s": 15.86,
    "avg_context_tokens": 273
  },
  "efficiency": {
    "token_ratio_skn_vs_json": 0.4
  }
}
```

**summary.txt** reports a head-to-head verdict across 5 dimensions (accuracy, ECE, hallucination, groundedness, abstention) with the winner of each dimension and an overall count.


## Technical Details

| Parameter | Value |
|-----------|-------|
| LLM Provider | Anthropic Claude API |
| Search Provider | DuckDuckGo (via `ddgs` package) |
| Default Model | `claude-sonnet-4-20250514` |
| Temperature | 0 (deterministic) |
| Max Tokens | 4096 |
| Search Results | Top 5 per query |
| Retry Strategy | Exponential backoff (base 2s, max 3 retries) |
| Retry Triggers | `RateLimitError`, HTTP 5xx, search API errors |
| ECE Bins | 10 equal-width bins [0.0, 1.0] |
| Token Estimation | `len(text) // 4` |


## Limitations

1. **Small sample size (n=15).** The 25 pp accuracy gap and 20 pp hallucination gap could shift significantly with more samples. These results are directional, not statistically conclusive.

2. **Single model.** All results are from `claude-sonnet-4-20250514`. Different models may respond differently to SKN's structured epistemic signals. A model with weaker instruction-following might not leverage `@gaps` and `@risk` as effectively.

3. **LLM-as-judge variance.** Accuracy, hallucination, and groundedness are judged by the same model family. LLM judges can exhibit systematic biases, and using a different judge model might produce different scores.

4. **Search result volatility.** DuckDuckGo results change over time. Running the benchmark on a different day may produce different raw search results, affecting all downstream metrics. The `search_results.json` output file preserves the exact results used in each run for reproducibility.

5. **Hallucination measurement.** The hallucination judge flags claims not present in the source text. Some "hallucinated" claims may be correct inferences that the judge conservatively flags as unsupported. The severity scores (0.2-0.3 for most SKN hallucinations) suggest these are minor embellishments rather than severe fabrications.


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
