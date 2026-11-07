# SAM policy-trajectory support v0.1

A small, dependency-free Rust reference for the finite path-support conformance slice in Symthaea #10514.

This crate distinguishes exact membership in a declared finite joint-trajectory support set, absence under a declared exhaustive finite support universe, absence when the evidence universe is not known to be exhaustive, incomplete trajectories, and policy/context scope mismatches.

It intentionally does **not** estimate propensities, learn a support classifier, infer real-world support, run a simulator, or certify a policy as safe. Exact path membership is not a practical sample-size guarantee.

## Run

    cargo fmt --manifest-path tools/sam-policy-trajectory-support/Cargo.toml -- --check
    cargo clippy --locked --manifest-path tools/sam-policy-trajectory-support/Cargo.toml --all-targets -- -D warnings
    cargo test --locked --manifest-path tools/sam-policy-trajectory-support/Cargo.toml

## Conformance cases

The unit tests include:
1. Marginal-action support can falsely accept a path absent from the joint trajectory corpus.
2. Per-agent marginal support does not imply joint ego/actor action support.
3. Full history identity prevents compressed-state aliasing.
4. An incomplete/censored path cannot be admitted.
5. Absence in a non-exhaustive corpus remains unknown.
6. Target-policy version and context are bound to the support manifest.
7. Canonical action ordering is independent of input order and duplicate actor commands reject.
8. Exact conditional chain-rule mass can become small without becoming structural zero.

The chain-rule helper accepts probabilities only when each input is already the conditional probability given the full preceding **joint** history under one declared estimand. Multiplying marginal probabilities or per-agent propensities is not justified by this helper.

## Claim boundary

Passing the finite fixtures establishes correct bookkeeping for these declared finite cases. It does not establish practical support in an empirical dataset, valid propensities, causal identification, human-driver behavior, simulator fidelity, collision-risk calibration, vehicle safety, or SAE Level 4/5 capability.

## Canonical artifact export

The support crate now exposes deterministic content bytes for three exact artifacts:

- `SupportManifest::canonical_bytes()`: evidence-manifest identifier, target policy, context, finite-universe exhaustiveness and the complete canonical supported-path set.
- `TrajectoryQuery::canonical_bytes()`: target policy, context, query completeness and every ordered decision step; joint action ordering is normalized the same way as path membership.
- `Assessment::canonical_bytes()`: versioned disposition, decision-point count and finite-path-membership flag.

These are canonical **content exports**, not hashes or signatures. Consumers choose and apply their digest algorithm to the returned bytes. The format is domain-separated and length-prefixed; a later schema/version change must use a new domain/version. The exports do not authenticate the SAM executable or the source manifest and do not validate the truth of caller-provided evidence.

Tests check joint-action order normalization, query completeness/partial-query sensitivity, deterministic finite manifest content export, and byte-for-byte golden vectors shared with Symtropy's reciprocal decision-evidence adapter.


## Source-bound assessment receipt

The SAM crate also exposes `assess_with_receipt`. It computes the normal finite assessment and retains the exact canonical `SupportManifest` and `TrajectoryQuery` bytes used for that call. `AssessmentReceipt::canonical_bytes()` emits one domain-separated, length-prefixed artifact containing those two canonical inputs, the source-derived evidence-manifest/policy/context/exhaustiveness metadata, the query policy/context/completeness/decision-point metadata, and the exact `Assessment::canonical_bytes()` result.

A downstream adapter can now reject a stale or cross-wired assessment: its receipt must reproduce from the exact support-manifest bytes, exact query bytes and recorded disposition. This is still an unkeyed content commitment, not proof that the SAM binary produced it or that the input evidence is truthful; build provenance must be anchored independently when authenticity is required.
