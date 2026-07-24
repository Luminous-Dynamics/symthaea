# Fleet lifecycle continuity

The v11 continuation closes five deployment gaps that remain after staged fleet
rollout: bounded operation during a network partition, permanent retirement,
replacement continuity, component supply-chain provenance, and long-horizon
evidence retention.

## Partition behavior

A network partition never creates new command authority. `PartitionPolicy`
permits only a short local grace period and only while command freshness,
trusted-time uncertainty, operator presence, local supervisor quorum, and the
emergency stop remain healthy. `PartitionCheckpoint` persists every decision in
an anti-rollback chain. A `PartitionContinuationPermit` is bound to the exact
checkpoint, control session, sequence ceiling, policy digest, action set, and
grace expiry.

Before fleet rejoin, `PartitionExecutionSummary` accounts for the complete
offline command interval and its audit segment. `PartitionReconciliationRecord`
classifies the result as accepted, recovery-required, or quarantine-required.
Offline grace overruns and emergency shutdowns require quarantine.

## Retirement and replacement

`DecommissionOrder` identifies the exact device, robot, hardware fingerprint,
authority epoch, sanitization profile, and final inventory generation. A
`DecommissionTombstone` is valid only after the inventory marks that exact
device retired and identity-key destruction is evidenced.

A replacement receives a new device ID and hardware fingerprint. It is linked
to the retired identity through `DeviceReplacementContinuity`, including
commissioning, calibration review, HIL evidence, and the later fleet inventory.
The old and new identities are never aliases.

Physical custody transfer uses a separate dual-party ceremony. Releasing and
receiving organizations must sign the same handoff with distinct keys after
physical inspection.

## Supply-chain provenance

Each safety-relevant component has a signed provenance record containing its
manufacturer, model, serial/lot, hardware identity, firmware/SBOM identity,
manufacturing evidence, supplier, integrator, confidence level, and inspection
time.

`DeviceBomManifest` forms an acyclic, generation-linked component tree. A
`DeviceProvenanceBundle` requires exactly one verified record for every BOM
component and can require component classes, firmware SBOMs, current evidence,
and non-legacy provenance.

Component replacement requires consecutive BOM generations. The old component
must be marked replaceable, the new component must occupy the same role, and the
event binds maintenance, custody, calibration review, and post-service HIL
evidence.

## Long-horizon evidence

`EvidenceRetentionPolicy` gives each evidence class a finite or permanent
retention rule, replication requirement, fault-domain requirement, encryption
requirement, and deletion mode. Legal-hold evidence is always permanent.

Deletion is not inferred from age. It requires:

1. An `EligibleForDeletion` retention decision.
2. A current hash-linked legal-hold checkpoint with no active hold.
3. A signed, short-lived deletion permit bound to both artifacts.
4. A signed deletion receipt containing destruction evidence.

Archives are hash-linked segments. Replica receipts must come from distinct
providers, media, signing keys, and fault domains. Periodic scrub receipts prove
that the actual stored bytes remain readable; corruption prevents a passing
scrub report.

## Fleet admission

`FleetLifecycleAssuranceBundle` cross-binds inventory, per-device provenance,
retirement tombstones, replacement continuity, partition checkpoints,
retention state, archive replication, and scrub reports.

`hal-fleet-lifecycle-verify` evaluates this bundle against a production policy
and exits nonzero when any required check fails. Signature verification of each
underlying signed artifact remains an independent prerequisite; the lifecycle
bundle provides cross-artifact closure rather than replacing signature checks.
