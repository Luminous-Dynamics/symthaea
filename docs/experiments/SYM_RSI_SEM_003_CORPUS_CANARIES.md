# SYM-RSI-SEM-003 — Independent Corpus Identity Canaries

Status: **FROZEN BEFORE ACCEPTED SEM-003 MEASUREMENT**

Primary preregistration:

`af8162162e13a2b3c95fc70a0af9b62a3183ac4f`

Descriptive-estimator amendment:

`418f057387eacf2747edba6e8efaffca05d4b04c`

Compile-invalid first implementation candidate retained in history:

`df3eedea80cb21ffc7f4896b64ff6480ea8d2680`

Replacement primary implementation candidate:

`dd533164f8aff636f06e49a161fa75b4e70359d1`

## Purpose

These canaries independently freeze the exact synthetic corpus identities expected from the SEM-003 generator.

They are **not** benchmark outcomes, evidence of retrieval quality, confidence, empirical validation, or executable qualification. Their only purpose is to detect accidental generator, float-arithmetic, support-diagnostic, ordering, target-label, or digest-domain drift before any SEM-003 receipt is accepted.

## Independent derivation

The canaries below were reconstructed outside the Rust SEM-003 implementation from the frozen equations and digest contracts.

The reconstruction used:

- dimensions `4, 8, 16`;
- 64 candidate contexts per dimension;
- 32 calibration queries per dimension;
- 32 clean held-out queries per dimension;
- 24 ambiguous queries per dimension;
- 24 unrelated queries per dimension;
- 16 OOD queries per dimension;
- IEEE-754 single-precision rounding for every generator arithmetic step;
- canonical little-endian `f32` bytes for exact context identity;
- BLAKE3-256 for exact context and corpus commitments;
- the frozen SEM-003 domain-separation strings;
- default semantic clamp `1.0` when reconstructing candidate support diagnostics.

The independent BLAKE3 implementation used for reconstruction was first checked against the official empty-input BLAKE3-256 vector:

`af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262`

The reconstructed corpus cardinalities were:

- candidate bank: `192` exact-distinct contexts;
- calibration queries: `96`;
- clean queries: `96`;
- ambiguous queries: `72`;
- unrelated queries: `72`;
- OOD queries: `48`.

The reconstruction also rechecked exact-identity disjointness between all frozen partitions before computing these commitments.

## Frozen corpus canaries

| Corpus | BLAKE3-256 digest |
|---|---|
| candidate bank | `6f58f0ccefafd5da69f757c0fd24e8b6e80f680eba0d3bb521eb7c13bce985a5` |
| calibration queries | `fe3230303a59c7e4681582e76b4323c19ee64bbbac7f8e797d51437932565aac` |
| clean held-out queries | `52ace5c8fac61dd8951cf3bf4f76ac80c7ac787ecadcc1eb46e8ea7e32160f12` |
| ambiguous held-out queries | `afc959c2cc2289eb2f87fd4de234829905aeb4af8efca05bbffa2a028da2e399` |
| unrelated held-out queries | `fef0dfde5b9b90bd1cf8d67c954066b9b9484deda7734539d0d2c1783b481ff3` |
| OOD held-out queries | `7ee1c9e0e744bec62357f56b209e7b00fe91c7e3efe5a8dc2050e63c3114c3b3` |

## Acceptance rule

Before an SEM-003 receipt from the replacement implementation candidate can be accepted as belonging to the frozen synthetic corpus, all six receipt corpus digests must equal the canaries above.

A mismatch is an **integrity / implementation-lineage failure**, not an adverse scientific benchmark outcome. Do not reinterpret a mismatched corpus as SEM-003 data, and do not retune these canaries after observing benchmark results.

The machine-checkable integration test `tests/sym_rsi_sem003_corpus_canaries.rs` enforces the same rule through the public SEM-003 runner.

## Claim boundary

Passing these canaries establishes only that the public SEM-003 runner reconstructed the frozen corpus identities expected by this independent derivation.

It does not establish:

- that the Rust subject compiles outside the environment that ran the test;
- that SEM-003 passes any retrieval-quality criterion;
- that SEM-003S is calibration-stable;
- factual correctness of retrieved memories;
- empirical evidence authority;
- belief confidence;
- consciousness or general recursive self-improvement.
