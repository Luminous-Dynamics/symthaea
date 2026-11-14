# Independent evaluation tranche — Institutional Intent Compiler

## Why this increment exists

A production validator cannot establish its own correctness by passing tests written against its own implementation. The next evidence step is a separately implemented oracle that consumes the same synthetic manifests but does not import or invoke the production validator.

## Scope

The reference evaluator is intentionally independent and limited to a tractable subset: trace presence, source existence/span bounds, review-decision subject/action/baseline, modality-change approval, accepted dependency state, and nonempty manifests. The oracle is specified separately in `reference-evaluator/`; its known-answer catalog is frozen alongside the fixtures.

## Corpus v0.1

Nine synthetic cases exercise a valid record, missing trace, missing modality approval, missing human acceptance, unknown source, out-of-bounds span, empty requirement list, dependency on a non-accepted requirement, and a stale-baseline decision.

The catalog stores expected diagnostic code sets, not merely a pass/fail label. This prevents a broad false rejection from counting as a correct result.

## Evaluation protocol

1. Freeze the corpus and known answers before evaluating a candidate implementation.
2. Run the production validator and reference evaluator as separate processes.
3. Compare status and diagnostic-code sets; preserve false positives and false negatives.
4. Add mutation tests for deleted, duplicated, altered and misbound evidence.
5. Expand the corpus only through versioned additions; do not silently rewrite prior known answers.
6. Keep hidden holdout cases out of the development fixtures.
7. Record compiler version, dependency lock, exact source SHA, command, exit code and stdout/stderr digest in any qualification receipt.

## Not established

No Rust compilation or execution is claimed in this environment. The new evaluator source is authored but pending compilation, exact-head tests and independent review. Passing this structural oracle would not establish semantic requirement quality, legal authority, source authenticity or safety.
