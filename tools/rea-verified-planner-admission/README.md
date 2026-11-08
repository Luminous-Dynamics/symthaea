# REA verified Evidence → planner admission adapter

Standalone Rust adapter that composes the existing REA raw Evidence verifier with the existing REA experiment planners. It does not implement a new Evidence identity algorithm or a new planner.

## Contract

- Callers provide exact raw Evidence JSON bytes. The adapter calls `symthaea-rea-compat::verify_evidence_json` and uses only the returned computed Evidence ID, semantic digest and raw-file digest.
- Caller origin/lineage declarations are a separate sidecar. They do not change REA Evidence v1 bytes or semantic identity.
- Parent IDs must resolve inside the same admitted catalog; duplicate parents, dangling references and cycles reject.
- The original caller-declared origin is retained in the admission report with `CallerDeclaredUnverified` qualification.
- Because the existing planner has no unverified-origin state, v0 deliberately projects caller-declared `IndependentObservation` and `IndependentDerivation` records as `ExternalReference`. They cannot ground a production planner hypothesis until a separate acquisition-attestation policy is implemented.
- The planner wrapper verifies every hypothesis support/contradiction ID is present in the raw-byte-admitted catalog and rejects common-mode root multiplicity within a hypothesis support closure before invoking the existing planner.
- The adapter computes a canonical `planner_request_sha256` over the admitted raw/semantic digests, lineage manifest, hypotheses, likelihood rows, experiment constraints and planner configuration. It is a request integrity commitment, not a signed run receipt.
- The returned claim ceiling explicitly leaves producer authentication, origin authentication, source independence, observation truth and issuer-history completeness unestablished.

## Run

```bash
cargo test --manifest-path tools/rea-verified-planner-admission/Cargo.toml --locked
cargo clippy --manifest-path tools/rea-verified-planner-admission/Cargo.toml --all-targets --locked -- -D warnings
```

## Deliberate limit

This v0 adapter handles individually supplied Evidence JSON records. It does not accept an EvidenceBundle as a planner catalog yet. Bundle verification remains available through the existing compatibility crate; an explicit bundle-member raw-binding representation must be added before the adapter accepts bundle input. No bundle completeness, producer authenticity, independent-source status or real-world observation truth is inferred.

The adapter can preserve a caller-declared `IndependentObservation` label because the existing planner API accepts that classification. Its report always marks that classification `CallerDeclaredUnverified`, and the planner's existing claim ceiling remains unchanged. Do not present a planned rank/abstention or `independent_root_evidence_ids` field as authenticated source independence.