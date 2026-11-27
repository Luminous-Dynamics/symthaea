# CMAQ-034 — Quarantine-to-recovery handoff

CMAQ-034 connects a CMAQ-031 anytime-monitor quarantine to the CMAQ-028 independent recovery-option evaluator.

## Decision contract

A monitor alarm invalidates the active calibration profile for new candidate commitments. This gate does not let the quarantined candidate continue while it looks for a backup.

The handoff requires:

1. A genuine CMAQ-031 quarantine update.
2. A reconciled current execution outcome and digest-bound current-state receipt.
3. A CMAQ-028 request containing exactly one prefix: the current state at step zero, with a state-evidence digest equal to the reconciled state digest and a separate reachability-evidence receipt digest binding the supplied reachable-set bound.
4. A recovery portfolio assessed by the actual CMAQ-028 evaluator.
5. At least k+1 independent selected failure domains, so that after one declared domain failure at least k remain.
6. No selected recovery option may reuse the active recovery/controller profile that was quarantined.

A normal portfolio CommitPrefix result can be accepted only when last_safe_commit_step equals zero and the request contains only the current prefix. That is interpreted as zero additional candidate continuation followed by an immediate recovery switch. A nonzero candidate continuation prefix is rejected.

Unknown or unreconciled execution returns BlockUntilReconciliation without running the portfolio assessor. Missing or insufficient options fail closed. The function returns the complete portfolio assessment for audit.

## Trust boundary

The adapter checks identity correspondence, declared failure-domain uniqueness, plant/context scope, exact prefix-state digest equality, reachability-receipt digest shape, and SHA-256 syntax. CMAQ-028 rejects prefixes missing either the state evidence or reachability evidence digest. It does not authenticate receipts, verify signers, prove physical independence, or command actuators. A separately governed evidence-ingestion boundary must authenticate the state receipt and recovery certificates. The runtime caller must treat every error or non-switch disposition as no new commitment.

The k+1 rule is deliberate: a requirement of k usable recovery options after one declared domain failure requires at least k+1 distinct options before that failure.

## Qualification

The dedicated workflow runs format, deterministic handoff tests and strict Clippy for this crate and its CMAQ-028 / CMAQ-031 path dependencies. Passing it validates software decision flow only. Physical recovery, certificate validity, and actuator response remain pending.
