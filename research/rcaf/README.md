# CETB RCAF kernel

A small, dependency-free Rust reference kernel for Repeatability-Calibrated Architecture Fidelity (RCAF).

This is a research metric library, not a consciousness detector. It does not read EEG/fMRI files, estimate correlation matrices, perform preprocessing, or make clinical classifications. It consumes already-computed correlation matrices / edge weights and participant-level within- and between-condition similarities.

## Contents

- Weighted Jaccard for non-negative edge-weight vectors.
- Pair-aware correlation-matrix modes: absolute, positive edges in both matrices, negative edges in both matrices, and signed-shift.
- Matrix validation for symmetry, correlation range and shape.
- Participant-paired repeatability summaries and deterministic percentile bootstrap intervals for the median paired gap.
- Nine unit-test cases for hand-computed values, sign-mask semantics, invalid inputs and paired summaries.

The positive-only and negative-only modes apply a shared sign mask: an edge is included only when it has the requested sign in both matrices. This matches the pairwise masking convention used in the published ds006623 revision code. It is not equivalent to replacing non-matching edges with zero and including them in the denominator.

## Run

The crate is intentionally outside the root Cargo workspace and has no third-party dependencies:

    cargo test --manifest-path research/rcaf/Cargo.toml

No such command has been run in the current environment because rustc/cargo are unavailable. The source is staged for review; it is not represented as compiled or tested.

## Statistical contract

For each participant, first collapse repeated segment placements using the prespecified within-condition and between-condition summaries. Then pass equal-length participant vectors to summarize_rcaf. The participant is the inferential unit; repeated windows must never be passed as independent participants.

Positive paired gap means between-state similarity is lower than the within-state reference. The normalized excess-distance fraction is descriptive only; it is not a variance decomposition or causal attribution.
