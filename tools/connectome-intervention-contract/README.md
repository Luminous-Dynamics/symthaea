# Connectome intervention contract verifier

Standalone Rust semantic verifier for CONNECTOME-INTERVENTION-RUN-V1. It complements JSON Schema validation by checking cross-object references, real pseudonymous split membership, recomputed digests, baseline coverage and paired-comparison arithmetic.

It deliberately does not fetch datasets, authenticate rights authorities, recover the secret HMAC key, or establish biological validity.

## Run

```sh
cargo fmt --manifest-path tools/connectome-intervention-contract/Cargo.toml -- --check
cargo test --manifest-path tools/connectome-intervention-contract/Cargo.toml --locked
cargo clippy --manifest-path tools/connectome-intervention-contract/Cargo.toml --locked --all-targets -- -D warnings
cargo run --manifest-path tools/connectome-intervention-contract/Cargo.toml --locked
cargo run --manifest-path tools/connectome-intervention-contract/Cargo.toml --locked -- path/to/manifest.json
```

Without an argument, the executable reads the deliberately incomplete scorecard example.

## Dispositions

- INCOMPLETE: planned/running/incomplete manifests are not promoted to a successful result.
- COMPLETE: all implemented semantic checks passed. This is not biological validation.
- INVALID: cross-references, digests, partitions, baseline coverage, paired arithmetic, uncertainty or claim gates are inconsistent.
- NOT_COMPLETE: source manifest records failed/invalid run state.

Malformed JSON, file errors and INVALID reports exit nonzero. INCOMPLETE exits zero for the CI example check, but is never called a pass.

## Verified contract

- HMAC token format, duplicate detection and pairwise train/validation/test disjointness.
- Recomputed SHA-256 membership hashes using sorted lowercase token strings, newline-delimited with a trailing newline.
- Exact measured scorecard unit cohorts and recomputed membership digests.
- Candidate, baseline and paired-comparison unit sets must be identical; equal counts alone are insufficient.
- Paired-unit subset-of-test rule, unit count and paired membership digest.
- Model/hypothesis/metric/stratum references and declared baseline-family coverage.
- Matched data scope and training budgets.
- Preregistered feasible interventions and required controls.
- Numeric score/CI consistency, no fabricated score for failed/abstained states, paired mean reconciliation and margin checks.
- Calibration references, immutable source/result receipt hashes and permission gates for measured-intervention claims.

## HMAC limitation

The secret HMAC key is never written to the manifest. This verifier checks token shape, set membership and digest consistency but cannot independently prove that the tokens were generated from the claimed raw identities using the committed secret. That requires a controlled tokenization authority or separate signed attestation. Raw identities must not be published in the manifest.


## CPG numerical receipt verifier

The second binary validates `CPG_NUMERICAL_RUN_RECEIPT_V1` and requires the raw state file for a complete run:

```sh
cargo run --bin cpg-numerical-receipt --manifest-path tools/connectome-intervention-contract/Cargo.toml --locked
cargo run --bin cpg-numerical-receipt --manifest-path tools/connectome-intervention-contract/Cargo.toml --locked -- path/to/receipt.json
```

The receipt's `numerics.raw_state_artifact.relative_path` is resolved relative to the receipt file. The artifact is contiguous little-endian float32 or float64. The verifier recomputes SHA-256, checks that the byte count equals `state_dimension × actual_saved_timepoints × bytes_per_value`, scans every value for NaN/Inf, and reconciles the declared finite-state summary.

A complete CPG receipt is rejected for any non-finite value, non-success solver status, missing saved output, post-solver sanitization, clipping, undocumented transformation, unapplied requested noise, digest mismatch or inconsistent tensor dimensions. The binary does not assume that `throw_on_failure=false` implies failure; it requires an explicit successful solver status and an independent raw-state scan.

Seventeen Rust tests are authored for positive and adversarial controls. The receipt schema validation script also checks a complete synthetic-shaped receipt and rejects non-finite, sanitized, clipped, failed-solver and unapplied-noise variants. Their execution is a CI requirement, not claimed until an exact-head workflow completes. 
Noise requests also declare a realized sample count. A non-zero request must be marked applied, contain at least one sample and bind a sample digest. The verifier does not infer non-application merely from an empirical standard deviation of zero: a small finite sample can have zero observed spread. The current contract binds the sample digest and summary statistics but does not yet load and recompute the full noise-sample array; that remains an explicit provenance limitation.

A passing numerical receipt establishes execution integrity only, not model validity or biological causality.
