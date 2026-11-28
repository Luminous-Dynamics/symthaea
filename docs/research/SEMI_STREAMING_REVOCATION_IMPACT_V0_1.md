# SEMI-STREAM-REVOCATION-001 — update journal and K1 revocation impact v0.1

**Issue:** #14190  
**Implementation:** `symthaea-semiconductor::benchmark::streaming_revocation_impact`  
**Status:** append-only journal and read-only impact evaluator implemented; exact-head validation pending.

## Why this is required

A streaming model is a recurrent state machine. A qualified label can update the feature NLMS adapter and global/local residual EMAs. Later predictions and updates consume that changed state.

If calibration/reference evidence is revoked after update (k), the direct update at (k) is not the only affected object. Every subsequent accepted update in the same generation has used a state descended from the affected transition.

A list of accepted label IDs is not enough. The engine needs a digest-linked transition history and a way to determine which suffix requires a new generation replay.

## Engine additions

The streaming engine now stores:
- deterministic generation ID;
- frozen genesis adapter state and digest;
- genesis journal anchor digest binding engine/model/feature/policy identities;
- ordered accepted-update journal;
- current journal-head digest.

Each accepted update appends an entry binding:
- update and arrival sequence;
- parent/resulting adapter-state digests;
- parent journal digest;
- immutable prediction receipt (including prediction-time feature vector);
- complete label-delivery event and K1 report;
- immutable update receipt;
- entry digest.

The engine appends the journal record only after candidate state mutation and receipt generation succeed. Rejected/held/duplicate deliveries do not enter the accepted-update journal.

## Read-only K1 requalification

The impact evaluator verifies:
- engine genesis state and journal-head digests;
- journal count equals accepted update count;
- contiguous update/arrival sequences;
- parent state and journal hash chain;
- every prediction and update receipt;
- label ID, report-attested revision, full label-payload digest and measurement-event binding;
- original K1 report and update receipt digests;
- uniqueness of accepted label revisions, measurement events and samples;
- live adapter state digest equals the final journal state.

It then requires one current K1 report for every accepted label revision. A missing or stale report yields UNKNOWN. A mismatched payload or duplicate current report is REJECTED. A now-BLOCKED or policy-ineligible label directly invalidates its accepted update.

Because adapter state is recursive, the evaluator marks every later transition as inherited/tainted. If a later blocked label occurs after an earlier unknown, blocked status takes precedence for the overall result; inherited suffix entries preserve the strongest previous blocked dependency.

## Important replay boundary

This v0.1 evaluator is impact-only:
- it never rewrites historical prediction/update receipts;
- it never changes the running adapter state;
- it does not run a new model generation;
- it does not roll back deployment or trigger retraining.

A future replay executor must start from the frozen genesis policy/state and replay raw feature-observation events and label-delivery events in recorded arrival order, using current K1 reports. It must regenerate later predictions because a removed earlier update may have changed the model state that produced those predictions. Reusing old prediction receipts against a cleaned adapter would be an invalid shortcut.

## Tests

Twelve Rust tests are authored:
1. two accepted updates form a digest-linked state/journal chain;
2. revoking update 1 directly invalidates it and taints update 2;
3. revoking update 2 leaves update 1 unaffected;
4. UNKNOWN K1 status creates an indeterminate suffix;
5. missing current report returns UNKNOWN;
6. tampered journal entry digest is rejected;
7. changed payload in current K1 report is rejected;
8. duplicate current report is rejected;
9. stale K0 frontier is UNKNOWN, not clean;
10. read-only report is deterministic and preserves old receipts;
11. missing/out-of-order journal sequence is rejected;
12. empty journal has no replay requirement.

The test count will be reconciled against the exact source at validation time.

## Claim ceiling

This is a synthetic software lineage/revocation-impact contract. It does not authenticate the physical calibration issuer, prove source authenticity from SHA-256, execute replay, perform rollback, authorize model deployment or control fab equipment. The report identifies a conservative dependency suffix, not the exact predictive performance loss caused by a revoked update.
