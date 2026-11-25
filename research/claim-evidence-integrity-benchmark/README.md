# Claim-Evidence Integrity Benchmark v0.1

Owner: [#13635](https://github.com/Luminous-Dynamics/symthaea/issues/13635)

## Purpose

This pack tests whether a research system distinguishes:
- a retrieved page from a source that actually supports a particular claim;
- a valid URL from a valid citation;
- topic relevance from entailment;
- multiple URLs from independent source families;
- a historical report from evidence available at a specified reference time;
- claim-level support from independent ground-truth adjudication.

The central contract is: **no claim receives support merely because a page is topically relevant, highly rated, or associated with a confident answer.**

## Paradigm shift: from claim-only to context-aware verification

The NeurIPS 2026 position paper *From Claims to Context* argues that factuality alone is insufficient without temporal, source-attribution, source-network, conflict and nuance signals. It separates claim reference time, evidence publication time, retrieval time and model knowledge cutoff. See https://ekmpa.github.io/context/

The benchmark therefore treats a verification as a structured relation among a scoped claim, source artifact, exact quoted span, source-family lineage, temporal context and label provenance. Each claim carries population, outcome/measure, quantifier/effect and domain/time boundary fields. Expected synthetic labels have explicit evaluator-only provenance and are never live calibration labels. It does not emit a single unqualified confidence score.

## External comparison points

| Track | Public reference | What it measures |
|---|---|---|
| Deep-research citation audits | *Cited but Not Verified* (2026): >94% link validity and >80% relevance, but 39–77% factual accuracy across tested models; one study found factual-check accuracy fell ~42% as tool calls increased from 2 to 150 | Link validity, topical relevance and fact support must be separate |
| Dynamic response factuality | FactBench / VERIFY: 1,000 prompts, 150 topics, Supported / Unsupported / Undecidable labels | Real-world generated-content factuality and appropriate undecidability |
| Temporal/numeric claims | TSVer: 287 real-world claims, 38 fact-checking organizations, 400 time series; Gemini-2.5-Pro 63.37 verdict accuracy and 48.63 justification score | Time ranges, numeric evidence and justifications |
| Scientific claim-source retrieval | CheckThat! 2026 study: verification-based ranking reached MRR@5 0.758 | Finding the actual publication behind a paraphrased scientific claim |
| Long-form RAG attribution | TREC 2025 RAG Track uses nugget coverage and sentence-support rate | Evidence coverage and sentence-level attribution |

These scores use different datasets and constructs and must not be pooled into a single SOTA number.

## Label semantics

### Evidence relation

Each evidence item has one claim-specific relation:
- `supports`: the quoted passage entails the claim within its scope and time.
- `refutes`: the quoted passage contradicts the claim within its scope and time.
- `qualifies`: the passage narrows, conditions or partially supports the claim but does not settle it.
- `irrelevant`: the passage is topically related or unrelated but does not bear on the claim.
- `unresolved`: source text or context is insufficient to classify.

### Final verdict

- `SUPPORTED`: sufficiently valid, claim-specific support is present, without an unresolved material contradiction.
- `REFUTED`: sufficient valid evidence contradicts the claim.
- `MIXED`: independent, admissible evidence materially conflicts or the claim holds only under divergent conditions.
- `UNDECIDABLE`: evidence is missing, weak, stale, ambiguous, non-independent, invalid or insufficient.
- `OUT_OF_SCOPE`: the claim cannot be assessed under the declared task or source/time boundary.

A `qualifies` item is not automatically positive support. A `retracted` or `compromised` item is preserved for audit, but cannot be treated as ordinary admissible support.

## Temporal contract

Every case distinguishes:
1. claim reference time — the time at which the proposition is intended to hold;
2. evidence publication time;
3. evidence retrieval time;
4. model knowledge cutoff.

Evidence published after a historical reference time must not be silently used as though it were available to a contemporaneous decision. It may be used for a retrospective verdict only when the task says retrospective verification is permitted. Keep both modes explicit.

## Source lineage and independence

A source family identifies common origin, syndication or copied lineage. Three mirrors of one wire report remain one information lineage. Domain reputation is a contextual prior, not a truth label. The source-family and compromise fields are evaluator-known in blind tests; candidate processes must not receive the hidden answer key.

## Files

- `fixtures/claim-evidence-integrity-v0.1.json`: 15 known-answer synthetic fixtures.
- `schemas/claim-evidence-integrity-v0.1.schema.json`: strict JSON Schema.
- `src/lib.rs`: deterministic validator for keys, fixture identity, timestamps, exact quote byte spans and artifact SHA-256.
- `src/bin/validate.rs`: validation CLI and negative-control self-test.
- `Cargo.lock`: pinned standalone dependency resolution.
- `.github/workflows/claim-evidence-integrity-benchmark.yml`: exact Rust 1.96 format/test/self-test/Clippy workflow.

Every synthetic evidence artifact includes an SHA-256 digest of its exact text and a byte span for the quoted passage. Each expected fixture label is tagged as synthetic, evaluator-only known-answer data, never ground truth for production meta-learning.

## Local execution

From the repository root:

```bash
cargo fmt --manifest-path research/claim-evidence-integrity-benchmark/Cargo.toml -- --check
cargo test --manifest-path research/claim-evidence-integrity-benchmark/Cargo.toml --locked --all-targets
cargo run --manifest-path research/claim-evidence-integrity-benchmark/Cargo.toml --locked -- --self-test
cargo clippy --manifest-path research/claim-evidence-integrity-benchmark/Cargo.toml --locked --all-targets -- -D warnings
```

The self-test runs negative controls for modified artifact bytes, altered quote spans and duplicate fixture IDs. Unit tests also reject missing claim scope, unsupported relation enums, missing temporal axes, duplicate evidence IDs and candidate-visible synthetic labels. The validator proves pack integrity and contract shape only; it does not decide whether natural-language evidence entails a claim.

## Synthetic fixture families

The 15 cases cover topical relevance without entailment, explicit support/refutation, conflict, syndicated duplicates, retrospective correction, contemporaneous verification, future leakage, quantifier scope, underspecified claims, time-series claims, broken factual attribution despite a working citation, searched-no-match, compromised source context and retraction.

Success means the implementation respects the contract, not that it has learned to fact-check. All fixtures are source-visible synthetic examples, not expert gold or empirical capability evidence.

## Promotion protocol

Before reporting external superiority:
- freeze benchmark version, task-set identity, retrieval date and scorer;
- use independently adjudicated claim-source pairs and report inter-annotator agreement;
- retain all abstentions, undecidable outcomes, invalid outputs, refusals and missing data;
- separate retrieval, attribution, claim relation, temporality, source-family independence and adjudication scores;
- publish per-domain/per-case-family rows, calibration, selective risk and full cost;
- run paired matched-budget ablations for retrieval depth and source context;
- keep held-out evidence, labels and evaluator-only metadata outside candidate training and process.

## Proposed targets (prospective only)

After exact external dataset and scorer parity:
- CheckThat!-style claim-source retrieval MRR@5 >= 0.80 against the published 0.758 reference;
- claim-support precision >= 0.95 at >= 0.80 coverage;
- claim-refutation recall >= 0.90 on blinded contradiction cases;
- false-support rate <= 0.01 on topical-only, source-conflict, duplicate-lineage and future-leakage controls;
- temporal verdict accuracy >= 0.95 on paired time-swap cases;
- no additional independent-support credit for syndicated copies;
- ECE <= 0.05 with reliability and risk-coverage curves.

Do not apply these numerical goals to this synthetic pack. They require a human-adjudicated confirmatory set.

## Claim boundary

This benchmark does not provide an entailment model, authenticate sources, determine real-world truth, or prove the current WebResearcher is accurate. It establishes a fail-closed interface contract and a reproducible evaluation plan.
