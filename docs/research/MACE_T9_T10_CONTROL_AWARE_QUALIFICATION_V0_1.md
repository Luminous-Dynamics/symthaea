# MACE T9/T10 control-aware qualification v0.1

**Status:** Rust result-validator/reference model authored on top of lifecycle kernel #13327 and file-witness adapter #13373. Tests are authored but not compiled or executed in the current environment. This is not a live SUT adapter, storage snapshot tool, production erasure mechanism, independent witness deployment, or T9/T10 security result.

## Purpose

The November 2026 Agent Memory Integrity Internet-Draft -00 separates two at-rest restore attacks:

- **T9 whole-store rollback:** restore a complete pre-write store snapshot after a genuine write.
- **T10 erasure resurrection:** restore a complete pre-erasure store snapshot after the SUT has acknowledged its own erasure operation.

T9 and T10 must be evaluated independently. A monotonic counter or write high-water mark may detect the lost write in T9 while missing the restored erased data in T10. MACE therefore reports independent T9 and T10 outcomes and never hides one behind a combined score.

This increment adds:

- research/mace-memory-lifecycle-kernel/src/at_rest_qualification.rs
- module export from research/mace-memory-lifecycle-kernel/src/lib.rs

The typed Rust API validates evaluator evidence and reports per-family scores. It does not invoke the SUT, stop processes, create snapshots, restore data, or call erase/write APIs. Those are adapter obligations.

## Mandatory controls

A result cannot be scored unless both controls succeed:

### C1 — no-op reload

Stop and restart the SUT with no edit. The seeded memory must remain available and unchanged, with no integrity signal.

### C2 — genuine write after restart

Stop and restart the SUT, write one new record through the SUT's own write path, and read it back. The record must be served.

Each control receipt is bound to a SHA-256 evidence digest. Missing or failed controls invalidate the report; they never become a pass or disappear from the denominator.

### C3 — edit landed exactly

For every applicable test episode, compare the full store before and after the intended operation, verify the exact normalized delta, restore the pre-operation image, and compare the restored image with the original. A failed or ambiguous edit is an Error and is excluded from the denominator, not a pass.

The validator requires:

- full-store snapshot digests (including database files, sidecars, indexes, journals and metadata);
- a semantic record inventory before, after the intended operation, and after restore;
- expected and observed normalized edit-delta digests;
- an operation receipt and acknowledgement from the SUT's own interface;
- proof that the SUT was stopped during external restore and restarted before the ordinary read;
- invocation of the ordinary SUT read path after restore.

For T9, C3 requires exactly one new record and no other logical record changes. For T10, C3 requires exactly the targeted record or records to be removed and all other logical records unchanged. The full restored snapshot digest and inventory must both match the pre-operation image.

The storage adapter must define and independently calculate the normalized edit-delta digest. A caller-provided boolean is not a substitute for the before/after evidence.

## Ordered experiment evidence

Each applicable T9/T10 episode now carries a run-bound event trace with a contiguous sequence number, unique event ID, monotonic elapsed time, one pinned clock-profile digest, typed event kind/source, and digest of the original receipt bytes.

The required event sequence is explicit:

- T9: SUT restarted before write → write requested through the SUT write API → write acknowledged → SUT stopped → restore harness applies the pre-write image → SUT restarted → ordinary SUT read completes.
- T10: SUT restarted before erasure → erasure requested through the SUT erasure API → erasure acknowledged → SUT stopped → restore harness applies the pre-erasure image → SUT restarted → ordinary SUT read completes.

The acknowledgement event must bind the exact operation/erasure receipt digest. The verifier rejects missing/reordered events, source-role substitution, cross-run event transplantation, duplicate event IDs, non-monotonic event time, clock-profile drift, receipt rebinding and event-trace digest mismatch.

The entire ordered trace is committed with a domain-separated SHA-256 root and retained in the case evidence. This is still a consumer-side integrity check: the live adapter must authenticate the event issuers, preserve original receipts and pin the clock/observer/restore-harness profiles independently. A self-authored event log with a recomputed digest is not independent proof.

## T9 procedure

1. Start from a fresh seeded store copy.
2. Through the SUT write path, commit one new genuine record and obtain its acknowledgement.
3. Stop the SUT.
4. Restore the complete pre-write snapshot, including every file and sidecar in the SUT's storage boundary.
5. Confirm that the restored full-store digest and semantic inventory match the pre-write image.
6. Restart the SUT and read the affected context through its ordinary read path.
7. Classify the result as Rejected, Reported, or Accepted and record detection point and observation class.

The validator checks that the post-write inventory equals the pre-write inventory plus exactly the new record, and that restoring the old snapshot exactly reproduces the pre-write snapshot digest and inventory.

## T10 procedure

1. Start from a fresh seeded store copy containing the protected target record.
2. Take a complete pre-erasure snapshot.
3. Invoke the SUT's own erasure interface and retain its erasure acknowledgement/receipt.
4. Verify that exactly the target record or context was removed and that unrelated records remain.
5. Stop the SUT.
6. Restore the complete pre-erasure snapshot.
7. Restart the SUT and read the affected context through its ordinary read path.
8. Classify the result independently from T9.

The validator checks that the pre-erasure inventory contains the exact target digest, the post-erasure inventory is exactly the pre-erasure inventory minus the target, the erasure was acknowledged, and the restored full-store image and inventory exactly match the pre-erasure snapshot.

**A successful erasure API response is not evidence that T10 is resisted.** The test deliberately reintroduces a genuine pre-erasure snapshot and asks what the restarted SUT actually serves.

## Verdicts and scoring

The typed observations preserve the Internet-Draft distinction:

- Rejected on the read path: pass.
- Reported at or before serving on the read path: pass.
- Reported only by a separate later audit: partial pass.
- Accepted as genuine without a signal: fail.

The observation class distinguishes the DUT boundary, DUT-reported audit result and endpoint-software boundary. Audit-only outcomes are never promoted to read-time protection.

The report returns separate T9 and T10 counts and metric values. For each family:

\`Metric = (read\_passes + 0.5 \times audit\_partials) / applicable\_cases\`

Not Applicable cases require a reason and are excluded from the denominator. Error cases also require a reason and are excluded; an edit that did not land is not a pass. A zero-acceptance gate is only evaluable when the family has at least one applicable observation. N/A must never be presented as a successful security result.

## Authored negative and positive tests

The module includes tests for:

- independently scored T9 and T10 outcomes;
- T9 success alongside T10 failure, demonstrating that a combined score must not conceal the T10 weakness;
- audit-only reports scoring as partial rather than read-time passes;
- failed or missing C1/C2 invalidating the report;
- T10 requiring an acknowledged SUT erasure and exact single-record removal;
- T9 requiring exactly one new record and restoration of the complete pre-write image;
- expected versus observed raw-store delta mismatch;
- N/A and Error exclusion without pass credit;
- duplicate case-run identifiers;
- invalid observation-class/detection-point combinations.

These tests are source additions only and have not been executed with Rust. No compile, format, Clippy, CI, benchmark run or security qualification is claimed.

## Integration sequence

1. Run the exact lifecycle-kernel crate tests, rustfmt and strict Clippy.
2. Implement a non-destructive storage adapter that captures complete store snapshots and normalized deltas for a real supported SUT.
3. Integrate C1/C2 and T9/T10 procedures with the existing #13327 lifecycle kernel and #13373 reference witness.
4. Keep the witness outside the candidate store's backup and rollback domain; test both data-only restore and common local data+witness restore.
5. Publish per-case raw evidence, protocol version, environment, SUT build/configuration, observation boundary and witness identity.
6. Run the same adapter on at least one external memory-system baseline before claiming comparative robustness.

Reference method: draft-khandelwal-bpm-agent-memory-integrity-00, an active individual Internet-Draft published 1 November 2026. It is work in progress and is not an adopted IETF standard.
