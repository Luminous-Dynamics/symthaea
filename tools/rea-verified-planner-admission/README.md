# REA verified Evidence → planner admission adapter

Standalone Rust adapter that composes the existing REA raw Evidence verifier with the existing REA experiment planners. It does not implement a new Evidence identity algorithm or a new planner.

## Contract

- Callers provide exact raw Evidence JSON bytes. The adapter calls `symthaea-rea-compat::verify_evidence_json` and uses only the returned computed Evidence ID, semantic digest and raw-file digest.
- Caller origin/lineage declarations are a separate sidecar. They do not change REA Evidence v1 bytes or semantic identity.
- Parent IDs must resolve inside the same admitted catalog; duplicate parents, dangling references and cycles reject.
- The resulting planner `EvidenceProvenance` projection preserves caller-declared origin classes verbatim. It does **not** authenticate those classifications.
- The planner wrapper verifies that every hypothesis support/contradiction ID is present in the raw-byte-admitted catalog before invoking the existing planner.
- The returned claim ceiling explicitly leaves producer authentication, origin authentication, source independence, observation truth and issuer-history completeness unestablished.

## Run

```bash
cargo test --manifest-path tools/rea-verified-planner-admission/Cargo.toml --locked
cargo clippy --manifest-path tools/rea-verified-planner-admission/Cargo.toml --all-targets --locked -- -D warnings
```

## Deliberate limit

This v0 adapter handles individually supplied Evidence JSON records. It does not accept an EvidenceBundle as a planner catalog yet. Bundle verification remains available through the existing compatibility crate; an explicit bundle-member raw-binding representation must be added before the adapter accepts bundle input. No bundle completeness, producer authenticity, independent-source status or real-world observation truth is inferred.

The adapter can preserve a caller-declared `IndependentObservation` label because the existing planner API accepts that classification. Its report always marks that classification `CallerDeclaredUnverified`, and the planner's existing claim ceiling remains unchanged. Do not present a planned rank/abstention or `independent_root_evidence_ids` field as authenticated source independence.