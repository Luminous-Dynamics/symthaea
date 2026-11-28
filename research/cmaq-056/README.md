# CMAQ-056 — Typed null contracts and content-addressed monitor manifests

## Why this exists

CMAQ-054 monitors a conditional paired-sample feature difference. Under the direct null

    Law(X_t | Z_t, F_(t-1)) = Law(Y_t | Z_t, F_(t-1)),

the bounded product feature difference has conditional mean zero. A reference sample is not needed to justify that zero target.

CMAQ-052 serves a different job: it estimates a bounded-moment target from a reference stream and spends a nonzero reference-envelope failure probability delta_reference. That requires the common-conditional-mean martingale assumptions and a reference/target/method receipt.

The risk plan must distinguish these modes. A direct conditional test should not reserve a fictitious reference-failure budget; a reference-calibrated test must account for its coverage failure exactly once.

There is also an evidence-integrity requirement: a 64-character string is not a content commitment unless it is recomputed from canonical contents. CMAQ-056 hashes the actual ordered expert IDs, rational priors, Fourier frequency banks, targets and reference IDs.

## Calibration modes

- DirectConditionalPairedEquality: alpha_test <= alpha_local; no reference failure term is used. The betting target must contain the zero vector. This contract does not imply iid sampling and does not claim that a finite Fourier bank detects every alternative.
- ReferenceCalibratedBoundedMean: alpha_test + delta_reference <= alpha_local; a theorem version and positive reference failure budget are mandatory. The same reference event is shared by all experts and restarts.

The modes are distinct and cannot silently downgrade into one another.

## Canonical digest contract

Version 1 uses domain-separated binary encodings with:
- fixed big-endian integer widths;
- u64 length prefixes for strings and vectors;
- exact u128 rational numerators/denominators;
- f64 values encoded by IEEE-754 bits after rejecting NaN/infinity and normalizing negative zero to positive zero;
- expert order preserved as registered;
- explicit format version and calibration mode.

SHA-256 is provided by the sha2 crate. Separate digests are emitted for the complete portfolio inventory, target envelope, reference bundle and calibration contract.

These digests establish content identity, not source authenticity, physical independence, common-conditional-mean assumptions or truth of a claimed stochastic bound.

## Implementation

The standalone Rust crate:
- builds a canonical manifest directly from CMAQ-054 KernelExpertConfig values;
- checks positive exact prior weights and their exact sum-to-one;
- checks map/target dimensional consistency and, in direct paired-equality mode, zero containment;
- rebuilds the canonical manifest from the actual runtime expert configs before accepting a typed risk plan;
- computes SHA-256 digests from actual frequency and target contents;
- constructs an exact fleet alpha plan with an optional reference-failure budget;
- verifies a legacy CMAQ-053 receipt only in reference-calibrated mode and only when all canonical digests/weights match.

This is an integration bridge. The next change should make CMAQ-053 carry the typed calibration mode directly and make CMAQ-054/055 consume the verified manifest receipt at construction.

## Tests

- SHA-256 standard known-answer vectors.
- deterministic canonical bytes and digest.
- negative-zero normalization.
- expert order, weight, frequency, target and reference mutations change the relevant digest.
- zero-excluding target rejected.
- direct mode admits alpha_test up to the local budget with no reference reservation.
- reference mode enforces alpha_test + delta_reference exactly.
- invalid or mismatched legacy CMAQ-053 receipts are rejected.

Run:

    cargo +1.100.0 generate-lockfile --manifest-path research/cmaq-056/Cargo.toml
    cargo +1.100.0 fmt --manifest-path research/cmaq-056/Cargo.toml --all -- --check
    cargo +1.100.0 test --manifest-path research/cmaq-056/Cargo.toml --locked
    cargo +1.100.0 clippy --manifest-path research/cmaq-056/Cargo.toml --locked --all-targets -- -D warnings
    cargo +1.100.0 run --release --manifest-path research/cmaq-056/Cargo.toml --locked --bin manifest_probe

The probe emits canonical digests and is a manifest-wiring check, not a calibration experiment.
