# Multi-Hall-SA Evaluation Adapter v0

**Status:** scorer implementation draft with synthetic contract fixtures only. No released dataset has been downloaded, ingested or scored by this change.

Parent programme: [SA-FACT-001 / Symthaea #10531](https://github.com/Luminous-Dynamics/symthaea/issues/10531).  
Harness owner: the existing Rust `tools/knowledge-benchmark-harness` crate.

## Why a separate scorer path is needed

Multi-Hall-SA rows contain four aligned statements: English factual, English hallucinated, target-language factual and target-language hallucinated. This is not the same construct as the current MythologyBench metamorphic-pair scorer. The adapter adds a narrow deterministic classifier-evaluation path; it does not replace the shared harness or create a second evidence system.

## Data minimization

The pair manifest stores only:
- pair ID;
- source-file ID;
- fact index;
- target-language label;
- hallucination-type label.

It deliberately stores **no statement text**. The licensed dataset remains external and read-only. The scorer binds to the exact normalized pair-manifest bytes by SHA-256. Any future text retrieval or corpus use must undergo its own rights and stewardship review.

## Normalized pair manifest

One JSONL record per released dataset row:

```json
{
  "schema_version": "multi-hall-sa-pair-v0",
  "pair_id": "dataset-pair-id",
  "file_id": "source-file-id",
  "fact_index": 0,
  "language": "isiZulu",
  "hallucination_type": "temporal shift"
}
```

The released builder assigns `file_id` as a document ordinal within each language-pair build. The original evaluator uses it to look up the source document. It is a valid cluster key **within that language split**, not a globally stable source ID across independently built language splits. Therefore all bootstrap comparisons, including English-vs-English-system comparisons, stay within one target-language split. A future release should include a stable source-document hash or canonical document-family ID.

**Important:** the released builder constructs `pair_id` from language, `fact_index` and error type, but `fact_index` resets for every source document. Therefore `pair_id` is not globally unique. The normalized row identity is the composite tuple `(file_id, pair_id, fact_index, language, hallucination_type)`; the scorer validates and joins on the complete tuple.

## Prediction receipt

Each system emits four result rows per pair:
- `statement_language=english`, `statement_kind=factual`;
- `statement_language=english`, `statement_kind=hallucinated`;
- `statement_language=target`, `statement_kind=factual`;
- `statement_language=target`, `statement_kind=hallucinated`.

The result includes a predicted label (`factual`, `hallucinated`, or `abstain`) and optionally a calibrated `p_hallucinated`. Language and kind are separate fields; target resolves to the pair manifest's language.

Every composite pair/system must have exactly one result for each of the four combinations. Each prediction repeats `file_id`, `fact_index`, `language` and `hallucination_type` alongside `pair_id`. This is intentional: `pair_id` alone collides across source documents in the released builder. Duplicates, omissions, unknown composite identities, unknown systems, invalid probabilities and inconsistent plan IDs fail closed.

## Metrics

The scorer reports per system and per language condition:
- TP / FP / TN / FN;
- accuracy, precision, recall, F1 and macro-F1 on non-abstained decisions;
- false-negative rate among covered hallucinated examples;
- unconditional missed-hallucination rate (abstentions count as not detected);
- false-hallucination rate on factual statements;
- coverage and selective risk;
- Brier score and 10-bin ECE when probabilities are supplied;
- latency, token and retrieval-call summaries where recorded.

For each target language it compares target-language and English missed-hallucination rates on the same `pair_id` values. It reports the paired difference and a deterministic cluster bootstrap interval over `file_id`. All four statements in a pair remain together. All pairs sharing a `file_id` are resampled as one cluster.

It also computes paired candidate-minus-baseline accuracy comparisons for English and each target language separately. The pair-balanced accuracy is the mean of the factual and hallucinated statement decisions for that fact; abstentions count as incorrect for this strict accuracy endpoint and are reported separately through coverage/selective-risk metrics. System comparisons are matched on identical pair IDs and use the same document-cluster bootstrap. Confidence intervals are emitted only for qualifying held-out runs; synthetic and development runs get point estimates only.

A held-out comparison requires a preregistered, rights-approved, source-family-disjoint split, confirmed cluster semantics, at least 30 independent clusters, at least 1,000 bootstrap replicates and a sealed/independent evaluation process. Smaller synthetic/development runs remain exploratory and report no qualifying confidence interval.

## Commands

From the repository root:

```bash
cargo run --manifest-path tools/knowledge-benchmark-harness/Cargo.toml --locked --release --bin score-multihall-sa -- \
  --plan benchmarks/multi-hall-sa/v0/synthetic-plan.json \
  --pairs benchmarks/multi-hall-sa/v0/synthetic-pairs.jsonl \
  --results benchmarks/multi-hall-sa/v0/synthetic-results.jsonl \
  --out target/multi-hall-sa-synthetic-report.json
```

The synthetic fixture uses four pairs across the four target languages and four error types. It exercises the complete four-prediction matrix for two systems, probability calibration inputs, and an intentional missed target-language hallucination. With only four synthetic source clusters, the scorer must emit point metrics but no qualifying bootstrap interval. The fixture is designed to test the scorer contract, not simulate the published benchmark or provide a model comparison.

The synthetic plan and output are scorer-contract fixtures only. They are not model outputs or benchmark results.

## Claim ceilings

- The scorer evaluates submitted labels against benchmark metadata; it does not independently validate historical or factual truth.
- It does not execute models, retrieve evidence or check citations.
- It cannot establish source-document independence merely because `file_id` strings differ.
- It does not turn confidence scores into calibrated probabilities; calibration metrics are reported only when the submitting system supplies probabilities.
- A reduction in cross-language disparity alone is not success if absolute accuracy deteriorates.
- No SOTA or deployment claim follows from synthetic fixtures or schema validation.


## Matched inference budget and usage receipt

Every plan contains one shared `budget` envelope for all candidate and baseline systems:
- `max_evidence_tokens`;
- `max_input_tokens`;
- `max_output_tokens`;
- `max_retrieval_calls`;
- `max_latency_ms`;
- optional `max_cost_microusd` (micro-USD; when set, every result must report cost).

Each statement-level prediction reports `evidence_tokens`, `input_tokens`, `output_tokens`, `retrieval_calls`, `latency_ms` and optional `cost_microusd`. The scorer rejects any row that exceeds the declared envelope and reports mean observed usage/cost per slice. Cost is only comparable when providers report it consistently; record the provider pricing snapshot and distinguish metered API cost from amortized hardware cost in the run notes.

Use the same evidence/context/token/call/latency envelope for every system in a comparison. The scorer does not itself enforce identical prompt quality, source access or retrieval implementation; those are part of the frozen run plan and external runner receipt. Synthetic fixture usage values are invented solely to test contract behavior and do not represent real inference.


**Normalization note:** the released builder assigns `file_id` as an integer document ordinal. The adapter normalizes it to a canonical decimal string (for example, integer `7` becomes `"7"`) in the metadata-only manifest. Synthetic fixtures may use readable string labels. Never strip or merge distinct source IDs during normalization.

The budget caps are **per statement inference**, matching the four predictions per composite pair. Every system gets the same caps. A non-null `max_cost_microusd` requires a reported per-statement cost; costs are integer micro-USD. The scorer reports means from submitted receipts and does not verify provider billing. The run manifest should also state the pricing snapshot or local hardware amortization method.

For a held-out run with multiple target languages, each target-language split must independently meet the minimum source-cluster count. A large combined cluster count across languages cannot compensate for an underpowered split.
