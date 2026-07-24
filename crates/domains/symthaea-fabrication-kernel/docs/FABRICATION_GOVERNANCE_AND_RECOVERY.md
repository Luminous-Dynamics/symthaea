# Fabrication Governance and Recovery Boundary

Version 0.11.0 adds lifecycle-aware governance above the v0.10.0 cryptographic
release path. A mathematically valid signature is no longer sufficient for the
new governed path: the signer must also be active for fabrication-manifest use,
inside its key-validity interval, and evaluated against a fresh, monotonic trust
snapshot.

## Governed authority chain

```text
FabricationManifest
  -> detached cryptographic signatures
  -> fresh TrustSnapshot
  -> revocation- and usage-aware verification
  -> VerifiedAttestation retaining snapshot digest and evaluation time
  -> FabricationGovernance ceremony
  -> hash-chained AuditJournal
  -> single-use AuthorizedPrintJob
  -> SubmittedJobReceipt
  -> durable ExecutionGuardCheckpoint
  -> explicit fresh-session restart re-authorization, when policy permits
```

`FabricationGovernance` is the preferred high-level ceremony. It makes
lifecycle verification and audit insertion inseparable for attestation,
authorization, submission, and containment transitions. Lower-level APIs remain
available for callers that need custom orchestration.

## Trust snapshots

`TrustSnapshot` is a bounded, sequence-numbered registry of signer identities.
Each `KeyTrustRecord` binds:

- signature algorithm and key identifier;
- activation and optional expiration time;
- active, retired, or revoked lifecycle state;
- permitted usages.

`TrustSnapshotTracker` rejects sequence rollback, conflicting snapshots with the
same sequence, and regressed issuance times. Long-lived keys may span multiple
snapshot lifetimes; snapshot freshness and key validity are distinct clocks.

The snapshot is policy evidence, not a certificate-chain implementation. The
caller must authenticate the snapshot source and persist the latest accepted
sequence across process restarts.

## Audit durability

`AuditJournal` uses domain-separated SHA-256 records. Every event binds its
sequence, timestamp, actor, action, subject, optional details, and previous
record hash. Verification detects content modification, deletion, reordering,
sequence drift, timestamp regression, and broken predecessor links.

A hash chain detects mutation but does not make storage immutable. Production
deployments should anchor journal heads in append-only or externally witnessed
storage.

## Governed 3MF packages

`export_3mf_package_with_governance` packages:

- geometry and fabrication manifest;
- detached attestation and manifest digest;
- trust snapshot and snapshot digest;
- audit journal, whole-journal digest, and chain head.

`inspect_3mf_package` verifies ZIP/OPC structure, CRCs, metadata copies, all
cryptographic digests, snapshot validity, journal integrity, and chain-head
consistency. `verify_governed_3mf_package` additionally evaluates signer
lifecycle at an explicit Unix time.

## Checkpoint and recovery truth boundary

`ExecutionGuardCheckpoint` serializes the complete deterministic guard state and
can be restored only after schema, policy, time, progress, heartbeat, and state
consistency checks pass. Its domain-separated digest detects checkpoint drift.

`reauthorize_print_restart` deliberately does **not** infer a safe mid-G-code
resume. It can only issue a new single-use authorization to restart the exact
attested program from the beginning when:

- interruption evidence and checkpoint digest are intact;
- the manifest and machine identity are unchanged;
- the previous session nonce is not reused;
- interruption age remains inside policy;
- progress is partial;
- the guard latched Pause, or an explicit policy permits a clean disconnect;
- the prior state was not Cancel or EmergencyStop.

True mid-program continuation requires firmware-specific modal-state capture,
physical position proof, extrusion-state reconstruction, and independently
validated resume semantics. Those remain outside this crate's authority.

## Governed replay

`GovernedFabricationReplayContract` extends deterministic replay evidence with
the trust-snapshot digest, full audit-journal digest, and audit head. A replay is
not governance-equivalent when any of those values drift, even if geometry and
G-code remain identical.

## Remaining deployment requirements

1. Authenticate and persist trust-snapshot sequence state outside the process.
2. Use reviewed Ed25519 and ML-DSA providers backed by protected keys.
3. Anchor audit heads in durable append-only or externally witnessed storage.
4. Bind machine capability advertisements and session nonces to authenticated
   machine identities.
5. Implement printer-specific pause/restart ceremonies and physical state
   confirmation before considering any resume behavior.
6. Run the full canonical workspace build, tests, Clippy, docs, fuzzing, and
   supervised hardware fault injection.
