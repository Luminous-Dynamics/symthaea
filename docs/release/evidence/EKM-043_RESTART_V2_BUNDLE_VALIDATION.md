# EKM-043 — Restart V2 Cross-Component Validation

## Purpose

EKM-042 transports a legacy V1 base-state snapshot beside an explicit typed policy/decision sidecar. EKM-043 independently verifies that those two snapshots describe the same revision decisions and that the claimed EKM-039 V2 digest binds them consistently.

## Central invariant

**A V2 bundle is semantically admissible only when the base epistemic evidence and the typed policy schema reproduce the same belief-revision decision that was persisted.**

## Validation sequence

EKM-043:

1. runs the independent EKM-035 base-state semantic validator,
2. runs the independent EKM-041 typed-schema semantic validator,
3. requires exact capture-cycle, revision-count, next-ID, receipt-ID, claim, delta, and evaluation-cycle agreement,
4. reconstructs a verification-only `EpistemicLedger` from the base snapshot using normal public append APIs,
5. rebuilds the typed policy from `BeliefRevisionPolicySchemaV1`,
6. reconstructs the original proposal basis, calibration snapshot, and uncertainty assessment,
7. re-runs the existing `BeliefRevisionGate`,
8. converts that fresh decision into `BeliefRevisionDecisionSnapshotV1`,
9. requires exact equality with the typed schema decision,
10. requires the old V1 receipt's eligibility and declared provenance-root count to agree with that recomputed decision,
11. independently canonical-hashes the typed schema snapshot using the same explicit V2 field/tag contract,
12. requires the claimed V2 digest to equal the domain-separated binding of the embedded V1 manifest digest and recomputed typed-schema digest.

## Scientific / epistemic consequence

The gate re-run means V2 validation does not merely compare duplicated decision text. The decision is recomputed from the base claim/evidence/provenance state, basis IDs, calibration, uncertainty assessment, and typed policy schema. This brings provenance diversity, evidence polarity, causal claim kind, intervention requirements, calibration thresholds, uncertainty freshness, contradiction policy, and bounded deltas back through the original proposal-only gate.

## Legacy V1 manifest caveat

The V2 digest check proves that the claimed digest correctly binds:

- the **embedded** V1 manifest digest, and
- the independently canonical typed-schema digest.

It does **not** independently re-derive the legacy V1 manifest digest from untrusted bytes. V1's historical revision digest still contains the explicitly documented `Debug`-based receipt representation.

Independent re-derivation of that legacy manifest requires reconstructing the typed receipt/history objects and regenerating the EKM-032 manifest. That is intentionally deferred to the later wire-to-quarantine conversion tranche.

The validation report therefore names this result `claimed_digest_binding_valid`, not `manifest_verified` or `restart_authenticated`.

## Authority boundary

The reconstructed ledger exists only inside validation and is never returned. EKM-043 returns only a validation report or typed error.

It does not create an EKM restart capsule/quarantine, writable support store/history, authorization state, activation handle, evidence/belief mutation, causal/world-model/action mutation, file I/O, or network I/O.

## Qualification boundary

Exact-head CI remains authoritative. Queued runs do not establish format, compile, Clippy, test, or runtime PASS.
