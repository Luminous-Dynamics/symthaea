# MATH-REP-001D2 — Semantic report validation

Status: **evidence-coherence gate; no execution result claimed**

Validator:

`.github/scripts/validate-math-equivalence-property-report.py`

## Purpose

Validate that a MATH-REP-001D1 JSON report is internally coherent and exactly bound to the frozen D0 generator configuration.

JSON Schema alone can constrain shape, but it cannot conveniently express all of the cross-field and manifest-dependent laws needed here. D2 therefore adds a stdlib semantic validator on top of the closed report schema.

## Evidence versus promotion

A failing experiment is still valid evidence.

Therefore the default validator accepts both PASS and failure reports when their structure and accounting are honest:

```bash
python3 .github/scripts/validate-math-equivalence-property-report.py report.json
```

Promotion is a distinct policy gate:

```bash
python3 .github/scripts/validate-math-equivalence-property-report.py \
  report.json --require-pass
```

`--require-pass` must never be used as a reason to discard a structurally valid failing report.

## Semantic checks

The validator requires:

- exact report/evaluator/generator/normalizer/authority IDs;
- exact top-level and nested field sets (no hidden extras);
- split is exactly `development` or `evaluation`;
- seed list exactly matches the frozen manifest for that split and order;
- total cases exactly match the manifest;
- class totals derive from the frozen family cycle;
- pair-family keys exactly match all 24 frozen pair families;
- refusal-family keys exactly match all 8 frozen refusal families;
- each family total matches its deterministic occurrence count;
- family pass counts reconcile to class pass counts;
- family normalization-error counts reconcile to class normalization-error counts;
- refusal classes cannot carry pair-normalization-error counts;
- top-level `passed_cases` reconciles to nested class summaries;
- `all_passed` is exactly consistent with total/passed accounting;
- `passed <= total` and error counts remain in range.

The validator computes expected family counts from:

- manifest family order;
- cases per seed;
- exact split seed count.

It does not trust the report to tell it those expected counts.

## What the validator does not do

D2 does not establish mathematical truth and does not recompute normal forms.

It checks **evidence integrity/accounting**, not:

- whether the generator's algebraic oracle is complete;
- whether v2 is mathematically universal;
- whether a theorem is proved;
- whether retrieval improves search;
- whether HDC helps.

## Exact-head binding

The D1 report deliberately does not contain a hard-coded Git commit because source cannot self-reference its eventual commit hash without changing that hash.

A qualified research evidence record should bind externally:

```text
report bytes/digest
exact Git head
Rust/toolchain identity
command / split
workflow or operator execution identity
```

D2 validates the report content before that external evidence binding.

## Recommended execution chain

```text
D0 manifest validator
        ↓
focused v2 Rust regression
        ↓
D1 development run -> JSON
        ↓
D2 validate JSON
        ↓
classify failures
        ↓
if eligible: explicit D1 evaluation run -> JSON
        ↓
D2 validate JSON
        ↓
D2 --require-pass only for promotion
        ↓
Evidence plane binds report + exact execution lineage
```

Queued/skipped workflows remain non-evidence.
