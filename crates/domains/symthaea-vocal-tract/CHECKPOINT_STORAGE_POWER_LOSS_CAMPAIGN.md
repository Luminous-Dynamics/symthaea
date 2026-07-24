# Checkpoint Storage and Sudden-Power-Loss Campaign — Series 16

Status: preregistered hardware campaign. This document defines the evidence
required before checkpoint persistence, replay state, audit export, and audit
compaction may be promoted for a specific storage stack.

## Purpose

Series 15 established deterministic process-death behavior. Process death does
not remove power from the filesystem, block layer, controller cache, or storage
device. Series 16 therefore separates three evidence classes:

1. `ProcessCrashSimulation`
2. `VirtualMachinePowerCut`
3. `PhysicalDevicePowerCut`

Only `PhysicalDevicePowerCut` counts toward the physical promotion gate.

## Frozen contracts

- `symthaea.checkpoint-storage-profile.v1`
- `symthaea.checkpoint-storage-profile-attestation.v1`
- `symthaea.checkpoint-power-loss-campaign.v1`
- `symthaea.checkpoint-power-loss-result.v1`
- `symthaea.checkpoint-power-loss-evidence.v1`
- `symthaea.checkpoint-power-loss-promotion-report.v1`
- `symthaea.checkpoint-operational-evidence.v6` (Series 16 storage/power-loss gates; retained as the historical schema)
- `symthaea.checkpoint-operational-evidence.v7` (Series 17 adds independent lease/journal/receipt gates)

Earlier process-crash and operational-evidence reports remain historical
artifacts. They cannot satisfy the Series 16 physical-power-loss gates.

## Lane A — authenticated storage profiles

Create one authenticated `CheckpointStorageProfileManifest` for every tested
combination of:

- Filesystem and filesystem instance.
- Exact mount-option digest.
- Kernel-release digest.
- Block-device and complete storage-stack binding.
- Logical and physical sector sizes.
- Claimed atomic-write unit.
- Volatile write-cache policy.
- Flush and Force Unit Access support.
- Barrier policy and stable-write support.

Raw device paths and mount paths should not be placed in public evidence. Use
nonzero bindings produced by the controlled lab inventory.

Required gates:

- The manifest validates and its keyed attestation verifies.
- Volatile cache is never described as safe without flush or FUA support.
- Barrier policy and write-cache policy are explicit rather than `Unknown`.
- The profile digest is frozen before any trial begins.
- Changing kernel, mount options, firmware, controller mode, cache policy, or
  device replaces the profile rather than amending an old result.

## Lane B — preregistered physical matrix

Minimum promotion matrix:

- At least two independently bound storage profiles.
- All six durability boundaries.
- At least two physical repetitions per boundary across the profile matrix.
- At least 24 physical trials are recommended; the minimum code gate is 12.

Durability boundaries:

1. After data write, before file synchronization.
2. After file synchronization, before publication.
3. After publication, before directory synchronization.
4. After directory synchronization, before acknowledgement.
5. During atomic replacement.
6. During audit-compaction replacement.

The campaign plan must freeze:

- Test-harness digest.
- Power-controller identity binding.
- Power-controller calibration digest.
- Operator-protocol digest.
- Workload digest and expected pre-power-loss state for every trial.
- Evidence class and durability boundary for every trial.

## Lane C — power-event evidence

Every result must contain a nonzero `power_event_evidence_digest` bound to the
external event record. For physical tests this should cover, as applicable:

- Relay or programmable power-distribution-unit event log.
- Controller timestamp and command acknowledgement.
- Independent voltage/current trace.
- Host heartbeat loss and restart observation.
- Device power-good transition.

A process exit, guest shutdown, graceful detach, or host-side `sync` after the
injection command is not a physical power-cut result.

## Lane D — recovery classification

Classify every restarted trial as exactly one of:

- `CleanRecovery`
- `FailClosedIndeterminate`
- `DetectedCorruption`
- `SilentCorruption`
- `Unrecoverable`

`CleanRecovery` requires both filesystem consistency and application-level
checkpoint/audit validation. Filesystem mount success alone is insufficient.

Promotion accepts clean recovery and, when explicitly permitted, a fail-closed
indeterminate state. Promotion rejects detected corruption, silent corruption,
and unrecoverable state.

Required application checks include:

- Envelope authentication and checkpoint configuration binding.
- Replay-state authentication and membership/order invariants.
- Rollback-chain position and predecessor digest.
- Audit record or segment-anchor authentication and absolute sequence.
- Export, archive receipt, retention commitment, and compaction continuity when
  the trial targets those paths.
- Exact old-or-new-state classification at atomic publication boundaries.

## Lane E — multi-lab evidence merge

Labs may return partial result bundles for one frozen campaign. Merge only with
`merge_checkpoint_power_loss_evidence`.

Required gates:

- Every result validates against the exact campaign digest.
- Trial identifiers are unique across all bundles.
- Duplicate trial claims fail closed, even if their bytes appear identical.
- The merged output is deterministically ordered by trial identifier.
- A result cannot change its evidence class, profile, boundary, workload, or
  expected pre-power-loss digest.

## Lane F — promotion evaluation

Run `checkpoint_series16_power_loss_evaluator` against the canonical campaign
and merged evidence artifacts.

Default promotion requirements:

- At least 12 physical device-power-cut trials.
- At least two storage profiles.
- All six durability boundaries.
- Every preregistered trial completed.
- Zero silent corruption.
- Zero detected corruption.
- Zero unrecoverable trials.
- Every completed trial is clean or explicitly fail-closed.

The evaluator does not manufacture trial results. Missing physical trials fail;
process-crash or VM-only results do not count as physical evidence.

## Lane G — operational evidence V6

Populate operational metrics with
`apply_authenticated_storage_power_loss_evidence`. Do not manually copy counts
from an unverified spreadsheet.

Required independent gates:

- `authenticated_storage_profile`
- `explicit_storage_durability_semantics`
- `physical_power_loss_trial_coverage`
- `power_loss_storage_profile_matrix`
- `power_loss_durability_boundary_coverage`
- `power_loss_no_silent_corruption`
- `power_loss_recovery_is_clean_or_fail_closed`

Missing Series 16 lanes are `not_exercised`, never pass.

## Promotion scope

Promotion is limited to the exact authenticated storage profiles in the
campaign. Passing results do not transfer automatically across:

- Filesystems or mount options.
- Kernel versions.
- Device firmware.
- RAID/HBA/controller modes.
- Write-cache or battery-backed-cache state.
- Hypervisor storage backends.
- Cloud volume classes.
- Sector sizes or atomic-write units.

## Explicit non-claims

Series 16 does not claim:

- That authenticated evidence proves the lab physically performed the trial
  without an independently trusted lab or instrument authority.
- Safety for an untested storage profile.
- Protection from malicious firmware or a dishonest power controller.
- Hardware-rooted measurement attestation.
- That a finite campaign proves the absence of every device failure mode.


## Series 17 operational execution

Assignment, resumable execution journals, stale-writer negatives, and exact sealed-result-artifact binding are preregistered separately in `CHECKPOINT_POWER_LOSS_OPERATIONS_CAMPAIGN.md`. Those gates do not alter the physical evidence classification or the Series 16 recovery requirements.
