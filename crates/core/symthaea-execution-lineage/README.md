# Symthaea Execution Lineage

`CognitiveExecutionLineageV1` identifies one concrete cognitive execution without granting epistemic or action authority.

The contract deliberately separates four identities:

```text
source generation
    +
configuration projection
    +
genesis commitment
    +
fresh execution instance
    ↓
CognitiveExecutionLineageV1
```

## Core invariants

- A genesis seed is reproducibility material, not execution-instance identity.
- A package version is not exact source-generation identity.
- A cycle counter is unique only inside an execution lineage.
- Source-generation identity is supplied by an outer build/evidence boundary; this crate validates its commitment shape but does not self-certify that source as qualified.
- Configuration bytes are bound together with an explicit configuration-profile identity and profile-contract digest.
- `None` genesis and an explicitly empty genesis value have distinct commitments.
- Every issued lineage receives a fresh 256-bit operating-system entropy nonce. Entropy failure is fatal; there is no clock/PID/version/genesis fallback.
- The raw execution nonce is never persisted; only its domain-separated BLAKE3 commitment is retained.
- Persisted lineage records recompute and verify their complete lineage commitment on deserialization.

## Non-scope

This crate does not:

- inspect `CognitiveLoopService`;
- create or consume `CycleResult`;
- qualify source code or builds;
- perform epistemic admission;
- grant GWT/workspace authority;
- grant external-action authority;
- authorize recursive self-improvement.

RCA-002.0b may later bind this contract at the cognitive-loop constructor boundary. RCA-002.1 may then project `(execution_lineage_digest, cycle_index)` into the already-isolated RCA shadow observer.
