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

The `file_id` is a **provisional source-document cluster key**. Confirm its semantics against the pinned dataset files and source metadata before using it for statistical inference. `pair_id` is the matched fact unit.

## Prediction receipt

Each system emits four result rows per pair:
- `statement_language=english`, `statement_kind=factual`;
- `statement_language=english`, `statement_kind=hallucinated`;
- `statement_language=target`, `statement_kind=factual`;
- `statement_language=target`, `statement_kind=hallucinated`.

The result includes a predicted label (`factual`, `hallucinated`, or `abstain`) and optionally a calibrated `p_hallucinated`. Language and kind are separate fields; target resolves to the pair manifest's language.

Every pair/system must have exactly one result for each of the four combinations. Duplicates, omissions, unknown pair IDs, unknown systems, invalid probabilities and inconsistent plan IDs fail closed.

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
