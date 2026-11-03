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


## Ordered and state-conditioned protocols

The V1 manifest can declare protocol_semantics = ordered_steps with at least two protocol_steps. The verifier checks contiguous zero-based indices, unique non-empty targets per step, finite non-negative inter-step intervals, and equality between the parent intervention target set and the union of step target sets. State context at the intervention or step level requires a state-measurement manifest SHA-256 receipt. These checks validate declaration consistency, not biological efficacy or whether the physical assay can deliver the protocol.

The schema exposes distinct metric identifiers for expected value of sample information and Fisher D-/A-optimal gains. They must remain separately labelled and use their declared units; they are not interchangeable with EIG bits.
