# MACE use-time evidence revalidation v0.1

**Status:** implementation and deterministic negative-control tests authored on a child branch of the measured-cost paired runner. Rust execution is pending a toolchain-enabled environment; no pass is claimed here.

## Why this increment exists

An event that was valid when ingested is not automatically valid when retrieved later. Between ingestion and use, the event may be retracted, superseded, become outside its valid-time interval, lose trusted-for-scope status, or acquire a changed source/assessment/dependency identity.

The new `revalidate_for_use_v1` path reuses the existing `EvidenceVerifier` and `ingest_as_known` contract immediately before downstream use. It does not create a second source authority or lifecycle ledger.

## Receipt and gates

The caller supplies the original raw event, its prior ingress-admitted value, a current `AsKnownQuery`, and a downstream `EvidenceUseContextV1` containing:
- a non-personal purpose slug;
- the exact required scope;
- the frozen downstream-policy SHA-256;
- whether complete dependency ancestry is mandatory.

A successful `EvidenceUseReceiptV1` binds the prior and current source snapshots, event/validity/verification times, source and root identity, trust-policy revision, lifecycle state, dependency lineage, calibration, causal assessment, outcome, purpose and consumer-policy identity.

The path fails closed when:
- the original event or prior snapshot does not match;
- current ingress validation rejects the event (including future-known receipts, retraction, supersession, invalid time or failed receipt verification);
- the immutable source/root/issuer identity changes;
- the trust-policy revision, calibration, causal assessment, outcome or dependency declaration changes;
- the source is no longer trusted for the requested scope;
- a strict consumer requires complete ancestry but the receipt reports Partial or Unknown.

The receipt's digest is a deterministic integrity commitment, not a signature. The returned Rust value carries a private in-process verification capability. JSON export intentionally labels its custody as `originating_process_only`; decoding or copying that JSON must not recreate the capability.

## Trust boundary

This is not a user/tenant RBAC decision, a signed cross-service authorization token, or a distributed freshness service. `TrustedForScope` is a source-policy classification, not proof that a proposition is true. Remote consumers must independently verify the underlying issuer receipts or obtain a separately signed use receipt from an authorized service. A purpose code and policy digest bind context; they do not grant permissions.

## Validation authored

Eight deterministic tests cover:
1. success with an exact prior/current bitemporal pair and scope/policy binding;
2. rejection when a previously admitted source is retracted before use;
3. rejection for changed source identity and loss of trusted-for-scope status;
4. strict veto of unknown ancestry and rejection of evidence recorded after the requested known-at time;
5. malformed purpose and downstream-policy digest rejection;
6. rejection of known-at time regression relative to the prior admitted snapshot;
7. rejection when the verified measurement-error bound changes, even if a faulty verifier reuses the old calibration digest;
8. rejection at the half-open valid-until boundary.

The strict JSON Schema is at [evidence-use-receipt-v1.schema.json](../../schemas/evidence-use-receipt-v1.schema.json). Its role is structural validation only; it does not authenticate issuers or recreate in-process verifier custody.

## Research and benchmark implications

This adds a new measurable surface that retrieval-only memory benchmarks generally do not isolate: **freshness at consumption time**. Future evaluation should report separate axes:
- stale/revoked/superseded evidence-use escape rate (veto metric; target zero on exhaustive fixtures);
- valid-evidence false-rejection rate;
- use-time revalidation coverage and time-to-detect;
- revalidation latency p50/p95/p99;
- correction-to-projection invalidation lag;
- evidence lineage completeness and common-root sensitivity;
- downstream paired decision regret/harm after revalidation.

Do not combine these with answer accuracy into a compensatory scalar. A single stale-use escape can invalidate a safety-critical run regardless of average recall.

## Next integration

1. Project this receipt into the candidate-visible DSM source-event envelope without exposing evaluator labels.
2. Require one current use receipt for every evidence ID passed to the downstream policy; distinguish retrieved IDs from decision-support IDs.
3. Add paired before/after-correction interventions and measure whether a stale projection can change the chosen action.
4. Reuse the existing paired cluster bootstrap and phase-cost receipts; do not create a second evaluator.
5. For cross-process deployment, add a separately governed signing/verifying service and explicit role/purpose authorization, rather than treating this in-process receipt as a portable capability.
