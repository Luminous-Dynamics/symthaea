# MACE v0.1 — Qualified Candidate Evidence Projection

**Status:** research implementation increment; draft PR. All eight new Rust tests are authored but unexecuted.

## Purpose

The decision-scoped qualification gate and pinned Ed25519 verifier can validate a receipt, but a caller still needs a safe way to apply that receipt to the exact candidate batch. This increment adds that missing pure projection seam.

The projection accepts a typed CandidateEvidenceBatchV1, a VerifiedEvidenceQualificationV1 produced by the signature-verifying gate, the full EvidenceQualificationBindingV1 re-presented at consumption, and an independently supplied expectation for candidate ID, backend identity/revision, query scope and query purpose. It returns either a distinct QualifiedCandidateEvidenceBatchV1 or an explicit abstention. It never falls back to the original unqualified batch.

## Invariants

1. Rebuild the canonical batch digest from typed records; public-field mutation cannot reuse a stale digest.
2. Match run ID, query digest and candidate-set digest to the signed use binding.
3. Match candidate ID, backend identity/revision, query scope and purpose to independently supplied request expectations.
4. Require exact equality between sorted batch evidence IDs and the use binding's candidate-ID set.
5. Revalidate the complete use binding, including subject scope, DecisionNeed, purpose, policy, lifecycle snapshot, verifier profile and expiry.
6. Remove rejected and indeterminate evidence.
7. Remove derived evidence if any required parent is absent or not independently admissible in this same batch.
8. Reject cycles in the remaining dependency graph with an iterative topological check.
9. Re-encode the surviving subset and retain both its digest and the original batch digest, plus the qualification receipt digest.

The qualified type does not include an authorized flag or issue an action permit. It also retains the complete use-binding snapshot and verified-at time. A consumer must call `validate_at_use` again before every later consumption; a different query, subject, DecisionNeed, purpose, policy, lifecycle snapshot, environment or verifier profile is rejected, as is expiry or a consuming-clock rollback. This is still not a substitute for live authorization and lifecycle revalidation.

## Parent closure

A derived record is not safe merely because its own payload was marked admissible. If a required parent was rejected, indeterminate, missing from the batch or itself dependent on a failed parent, the derived record is removed. Out-of-batch parents are not silently treated as admissible. A future multi-batch DAG service can relax this only by verifying and binding the complete transitive closure.

## Claim boundary

This is not a source-authentication system, rights engine, truth oracle, independent qualifier service, executor authorization, consume-once ledger, rollback-resistant witness, or production sidecar integration. The LongMemEval-V2 sidecar still emits unqualified candidate evidence. It must remain diagnostic-only until a separately operated qualifier can issue a standard externally verifiable receipt and the sidecar can obtain and consume it for the exact batch.

The current MACE signature is a versioned custom Ed25519 input, not DSSE. A follow-up should select and implement a standard DSSE/in-toto or COSE binding with canonical payload bytes and public test vectors. The trust-profile digest and monotonic version floor must remain outside the candidate backend and memory-store rollback domain.

## Tests

Eleven Rust tests cover admissible-only projection, expected scope mismatch, stale-batch mutation, exact ID-set mismatch, rejected-parent propagation, missing-parent abstention, cycle rejection and zero-admissible abstention, projection expiry at later consumption, context-rebinding rejection and consuming-clock rollback. They are authored but not executed. Rust compilation, formatting, Clippy, dependency resolution and exact-head CI remain unverified.
