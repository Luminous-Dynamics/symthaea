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
- Empty configuration projections fail closed.
- `None` genesis and an explicitly empty genesis value have distinct commitments.
- Every live-issued lineage receives a fresh 256-bit operating-system entropy nonce. Entropy failure is fatal; there is no clock/PID/version/genesis fallback.
- The raw execution nonce is never persisted; only its domain-separated BLAKE3 commitment is retained.
- Persisted lineage records recompute and verify their complete lineage commitment on deserialization.

## Live issuance is not archival replay

Live issuance returns `IssuedCognitiveExecutionLineageV1`, an affine-style capability that deliberately implements neither `Serialize` nor `Deserialize`.

The serializable `CognitiveExecutionLineageV1` is an archival identity record only. Deserialization proves that the record is internally commitment-consistent; it does **not** recreate live issuance authority.

A future qualified cognitive-loop constructor should therefore require the issued capability, not an arbitrary archival record loaded from disk.

## Non-scope

This crate does not:

- inspect `CognitiveLoopService`;
- create or consume `CycleResult`;
- qualify source code or builds;
- prove that the externally supplied source generation is qualified;
- perform epistemic admission;
- grant GWT/workspace authority;
- grant external-action authority;
- authorize recursive self-improvement.

RCA-002.0b may later bind the **issued** capability at the cognitive-loop constructor boundary. RCA-002.1 may then project `(execution_lineage_digest, cycle_index)` into the already-isolated RCA shadow observer.
