# SYM-RSI-SEM-003 — Implementation Repair 1

Status: **SOURCE REPAIR RECORDED — EXECUTABLE QUALIFICATION PENDING**

## Frozen protocol lineage

- original preregistration: `af8162162e13a2b3c95fc70a0af9b62a3183ac4f`
- descriptive-estimator amendment: `418f057387eacf2747edba6e8efaffca05d4b04c`

Neither protocol document is changed by this repair.

## Failed implementation candidate

The first compact SEM-003 implementation subject was:

`df3eedea80cb21ffc7f4896b64ff6480ea8d2680`

Static source audit after freeze found that `sym_rsi_semantic_set_experiment.rs` contained a duplicated helper block. In particular, the exact file contained two definitions of `fn candidate_index`, together with duplicate generator/disjointness helpers.

That is a Rust compile error. Therefore `df3eedea…` is preserved as an **unqualified compile-invalid implementation candidate** and must never be described as a compiled, tested, or measured SEM-003 subject.

No SEM-003 benchmark result from that subject is accepted as evidence.

## Replacement implementation candidate

Replacement source subject:

`dd533164f8aff636f06e49a161fa75b4e70359d1`

The replacement keeps the frozen external scientific contract:

- schema `symthaea.sym-rsi-sem-003.v1`;
- generator version `semantic-set-generator.v1`;
- context dimensions `[4, 8, 16]`;
- 64 candidates per dimension;
- 32 calibration queries per dimension;
- 32 clean held-out queries per dimension;
- 24 ambiguous queries per dimension;
- 24 unrelated queries per dimension;
- 16 OOD queries per dimension;
- alpha `0.10`;
- unchanged candidate/source generator equations;
- unchanged split-calibration quantile rule;
- unchanged support rejection semantics;
- unchanged eight primary success criteria;
- unchanged receipt field surface and receipt-integrity domain strings.

The internal implementation is refactored into one canonical corpus builder, threshold calibrator, evaluator, and primary-summary path. Those internal seams are crate-parent-visible so the preregistered SEM-003S stability audit can reuse the exact same evaluation semantics without defining a second primary classifier.

## Static repair evidence

Static connector audit of the replacement source confirms exactly one occurrence of:

- `fn candidate_index`;
- `fn candidate_context`;
- `fn validate_exact_disjointness`;
- `pub fn run_sym_rsi_sem_003`.

This removes the known duplicate-definition compile blocker.

Static review is not rustfmt, parsing, compilation, Clippy, unit-test, or runtime evidence.

## Qualification boundary

`dd533164…` remains an implementation **candidate** until an exact-head executable gate successfully establishes at least:

- `cargo fmt --check -p symthaea`;
- root-library compile with inline unit tests;
- strict Clippy with warnings denied;
- recursive-improvement unit tests;
- protected-measurement boundary scan.

A queued/skipped workflow is not PASS.

## Measurement boundary

No protected SYM-RSI partitions 201–204, 301–304, 401–404, or 1201–1204 are consumed by this repair.

No qualified SEM-003 benchmark result is claimed by this document.

## Failure discipline

If executable qualification finds another source defect, preserve `dd533164…` unchanged as the next failed candidate and repair under a new exact subject. Do not rewrite this lineage or reinterpret static review as executable evidence.