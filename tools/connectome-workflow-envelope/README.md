# connectome-workflow-envelope

A dependency-free Rust validator for cross-stage proof-reading evidence chains.

## Purpose

The crate composes independent stage receipts without duplicating their domain algorithms. It validates that those receipts refer to the same dataset operation, sample hash, task route and graph lineage, then computes the actual workflow status from the present evidence.

It does not run models, select views, allocate reviewers, apply graph edits, verify biology or hash external bytes.

## Evidence-chain model

A workflow envelope contains immutable references and operation identity plus optional stage receipts:

1. dataset identity and license approval;
2. group-disjoint split audit;
3. model prediction and calibration;
4. active-view decision or stop receipt;
5. review allocation and observed human disposition;
6. independently verified edit transaction;
7. query-specific graph-stability report;
8. optional finite-horizon future-drift calibration, bound to the same query and exact graph-history snapshots.

The validator advances only while the evidence supports the next transition. A missing receipt remains pending. A deferred review is not a completed review. A rejected or quarantined edit never advances the canonical graph head. A query report is accepted only if its first snapshot equals the envelope base snapshot and its latest snapshot equals the committed edit result.

## Workflow statuses

- `DatasetLicensePending`: license review has not approved the dataset.
- `SplitAuditPending`: no split audit receipt.
- `BlockedBySplitLeakage`: group audit reports overlap; no generalization claim is eligible.
- `PredictionPending` / `PredictionUncalibrated`: prediction missing or has no qualified calibration.
- `ViewDecisionPending`: no active-perception receipt.
- `ReviewAllocationPending`, `ReviewPending`, `ReviewDeferred`, `ReviewRejected`: allocation and actual human disposition are distinct.
- `EditPending`, `EditRejected`, `EditQuarantined`: proposals and non-committed outcomes never masquerade as committed edits.
- `QueryAuditPending`, `QueryInsufficientHistory`, `QueryUnstable`: a committed edit alone does not establish query readiness.
- `QueryReadyObservedWindow`: the named query was stable across the observed graph-history window only; no future-drift calibration is attached.
- `QueryStopCalibrationInsufficient`, `QueryStopCalibrationGroupOverlap`, `QueryStopCalibrationExchangeabilityUnsupported`: finite-horizon calibration abstains because its support or assumptions are inadequate.
- `QueryProjectedUnstableWithinHorizon`: calibrated upper future drift or current uncertainty exceeds the declared query materiality tolerance.
- `QueryProjectedStableWithinHorizon`: observed query stability plus a finite-sample future-drift upper bound within tolerance for the declared horizon. This is a marginal projection, not a future-stability guarantee.

Every computed report hard-codes `future_stability_certified = false` and `universal_connectome_completeness_claimed = false`.

## Required cross-stage invariants

- Dataset license review digest is present before downstream work is considered.
- Split receipt binds exact operation ID, sample hash, task route and split.
- Prediction binds exact operation/task/sample, checkpoint, inference configuration, calibration profile and prediction artifact. Probabilities are finite in [0,1].
- View decision binds exact operation/task, policy, scenario set, loss matrix and acquisition-cost profile. Acquired views bind sample/task/artifact and cost units to the loss unit. `EmBest` is rejected for false-merge identification. A stopped view records a non-empty stop reason.
- Review receipt binds operation/task/sample. A deferred allocation has no reviewer/disposition/time; a completed disposition requires a reviewer and positive observed duration.
- Edit receipt binds operation/task/sample, operation-binding digest, base graph snapshot, event stream, independent verifier and annotation-propagation receipts. A committed edit must change the snapshot, include a result digest, and have passing verifier and annotation attestations.
- Query receipt binds query/analysis/uncertainty references and source volume. Its first snapshot must equal the envelope base; its latest snapshot must equal the committed edit result. It cannot claim future stability or universal completeness.
- Optional query-stop receipt binds the exact observed query definition, analysis, source volume and first/latest snapshots. It recomputes the finite-sample rank `ceil((n+1)*(1-alpha))`, checks the evidence-backed group/exchangeability flags, and rejects status/quantile contradictions. The envelope keeps `future_stability_certified=false` even for `QueryProjectedStableWithinHorizon`.

## Canonical serialization and digest boundary

The v0.1 Rust crate is an in-memory validator over typed receipts. It intentionally does not add a production serialization or cryptography dependency.

The adapter contract should use RFC 8785 JSON Canonicalization Scheme (JCS) for envelope/receipt serialization and SHA-256 over the canonical UTF-8 bytes. Domain-separated envelopes must include schema version and context before canonicalization. For the operation-binding digest, canonicalize the complete dataset identity plus operation identity (including source-group IDs and task/split) and hash those bytes. Do not hash concatenated un-delimited fields.

Every referenced digest must be independently recomputed against the actual artifact bytes at ingestion. In particular, adapters must verify:
- split assignment and group metadata for the referenced operation;
- prediction output and calibration evaluation artifact;
- selected view image/geometry bytes and leakage-policy manifest;
- reviewer identity, disposition and observed time;
- exact matching ledger event, actor separation, verifier artifact and annotation propagation;
- actual query report and first/latest graph snapshot bytes.

A well-formed 64-character digest string is not proof of artifact authenticity. The booleans in receipts (for example `calibrated`, `license_approved`, `verification_passed`) are assertions to be checked against referenced evidence, not trust anchors.

## JSON contracts and fixtures

- Input schema: `research/connectomics/connectome-workflow-envelope-v0.1.schema.json`.
- Computed report schema: `research/connectomics/connectome-workflow-audit-report-v0.1.schema.json`.
- Optional future-stop calibration receipt: `query_stop_calibration` in the input schema, matching [#10073](https://github.com/Luminous-Dynamics/symthaea/pull/10073).
- Fixture schema: `research/connectomics/connectome-workflow-envelope-fixture-v0.1.schema.json`.
- Known-answer cases: `research/connectomics/fixtures/connectome-workflow-envelope-known-answer-v0.1.json`.

The JSON schema defines the normalized wire representation. The current Rust types are the in-memory domain API; a future adapter must implement and test the explicit conversion to/from the tagged JSON representation.

## Tests

Run in a Rust environment:

```sh
cargo test --locked --manifest-path tools/connectome-workflow-envelope/Cargo.toml
```

Thirty-four unit tests are authored for pending/blocked statuses, operation/task/split consistency, uncalibrated predictions, view leakage and unit mismatch, stopped views, pending/deferred/rejected human review, verified commit, rejected/quarantined edit, snapshot continuity, query readiness, instability, missing history, source-volume mismatch, forbidden future/universal claims, malformed identities and digests, finite-horizon conformal rank, distinct group/exchangeability abstentions, snapshot binding and projection-status contradictions.

They have **not** been compiled or executed in the current authoring environment.

## Finite-horizon stopping boundary

The optional calibration receipt is separate from observed-window stability. It carries the frozen query/materiality context, group-disjointness and exchangeability assessment digests, unique source-volume/independence-cluster support, alpha, exact conformal rank, residual quantile, target observed drift, current uncertainty width and future edit horizon. The envelope checks rank and the arithmetic relation `projected upper = observed ratio + residual quantile` within numerical tolerance. A projected-stable status is allowed only when the observed query is stable and both current width and projected upper drift are <= 1. The report still hard-codes future stability certification to false.

## Integration and performance benchmarks

The envelope validator is a small deterministic CPU-side gate, not the bottleneck for image or graph operations. Benchmark it separately from adapters:
- validation throughput and p50/p95/p99 latency by envelope size and stage count;
- canonical serialization and digest-verification throughput;
- receipt verification cache hit/miss behavior;
- concurrent ingest contention and append-only audit latency;
- replay determinism across process/platform versions;
- memory per receipt and history compaction without loss of provenance.

The end-to-end benchmark must report model inference, view acquisition, human review, graph application, independent verification, annotation propagation and query-readiness latency separately. Do not compare the in-memory validator to CAVE's graph-operation latency as if they were equivalent.

See [SCI-CONNECTOMICS-059](https://github.com/Luminous-Dynamics/symthaea/issues/10028).