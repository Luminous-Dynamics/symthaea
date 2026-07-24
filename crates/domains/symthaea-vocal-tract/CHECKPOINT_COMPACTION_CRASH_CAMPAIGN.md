# Checkpoint Audit Compaction and Crash Campaign — Series 15

Status: preregistered engineering campaign. This document defines the evidence
required before the Series 15 multi-process admission and audit-compaction path
may be promoted.

## Scope

Series 15 adds four claims that were not established by earlier campaigns:

1. One active-connection ceiling is shared by cooperating daemon processes.
2. A synchronized, independently acknowledged audit segment may be replaced by
   an authenticated continuation anchor without resetting sequence or hash
   continuity.
3. Promoted compaction additionally requires an authenticated retention
   commitment with an adequate expiry, replica floor, and storage-class binding.
4. Process termination at deterministic publication boundaries has an explicit,
   recoverable result.

The Series 15 operational evidence contract is
`symthaea.checkpoint-operational-evidence.v5`. Physical storage promotion is a
separate Series 16 claim under `symthaea.checkpoint-operational-evidence.v6`.

## Fixed artifacts and identities

Record before execution:

- Source commit and final Git tree.
- Host kernel and libc versions.
- Filesystem type, mount options, block device, and write-cache policy.
- Effective UID/GID and daemon supervisor.
- Shared-admission root and configured capacity.
- Audit key ID, archive-authority ID, archive repository binding, and storage
  class binding. Do not record secret key bytes.
- Export, archive receipt, retention commitment, and continuation-anchor schema
  identifiers.

## Lane A — cross-process admission

Start two independent processes using the same shared-admission root and the
same capacity.

Required gates:

- The first process acquires one kernel-lock-backed slot.
- The competing process is rejected before protocol parsing or backend work.
- Killing the holder releases the slot without PID-file cleanup.
- A fresh process can acquire the released slot.
- Zero and excessive capacities fail at configuration time.
- Different configured capacities against the same root fail with an explicit
  configuration-mismatch result.
- Slot, capacity, and lock files are regular, same-owner, and private.

Repeat with capacities 1, 2, 8, and the deployment maximum. Include at least
100 competing-process iterations per capacity.

## Lane B — exact segment export and independent retention

Create a nonempty authenticated audit segment, synchronize it, export it, and
obtain an independently keyed archive receipt.

Required gates:

- Empty anchor-only segments cannot be exported as misleading zero-record
  archives.
- Receipt metadata and the exact receipt bytes are read once and digest-bound
  during promoted compaction.
- The retention commitment binds the exact receipt digest, repository binding,
  archive ID, export ID, expiry, replica floor, and storage class.
- Expired commitments, insufficient replicas, wrong storage class, wrong
  repository, wrong authority key, and altered receipt bytes fail closed.

## Lane C — authenticated compaction continuity

Compact the exact live segment only after export, independent receipt, and
retention verification.

Required gates:

- The live segment still matches the archived record count, byte length, and
  head digest.
- The replacement contains an authenticated continuation anchor.
- The next append uses the pre-compaction next sequence number.
- The next record's `previous_record_digest` equals the archived segment head.
- Segment index increments monotonically.
- Wrong export, wrong receipt, stale live head, and altered anchor fail closed.
- Reopening the log after compaction reproduces the same anchor and continuation
  state.

Run at segment sizes 1, 2, 127, 1,024, and the configured practical maximum.
Run at least three consecutive compaction cycles and verify absolute sequence
and hash continuity across every cycle.

## Lane D — pre-publication process death

Terminate the compaction process at:

- `BeforeAuditCompactionWrite`
- `AfterAuditCompactionFileSync`

Required gates:

- The original live segment remains valid and complete.
- No replacement anchor is accepted.
- Any temporary artifact is either absent or safely ignorable.
- Retrying the same exact compaction succeeds.

The deterministic child-process exit lane is an engineering approximation. It
is not evidence of sudden power-loss survival.

## Lane E — post-publication process death

Terminate the process at `AfterAuditCompactionPublish` before the directory-sync
result can be returned.

Required gates:

- The caller receives no false synchronized-success claim.
- Reopening the log yields either the complete old segment or the complete new
  anchored segment; a mixed or partial state fails verification.
- When the new anchor is visible, digest-bound reconciliation succeeds.
- Reconciliation with any other digest fails.
- Appending after reconciliation preserves absolute sequence and hash
  continuity.

## Lane F — legacy real-power-loss outline

Execute on the actual deployment storage stack, not merely a child-process
exit. At minimum test ext4, XFS, and Btrfs when they are deployment candidates,
including their intended mount options and storage hardware.

Inject sudden loss at the same logical boundaries by using a VM/device harness
that can cut power or detach the block device without graceful shutdown.

Required observations:

- Recovered state classification: old segment, new anchored segment, or invalid.
- Filesystem repair activity.
- Whether file data and directory entry survived independently.
- Whether the final artifact requires reconciliation.
- No invalid state is silently accepted.

Promotion is filesystem/device specific. Passing on one storage stack does not
promote another. Execute the frozen, authenticated Series 16 contract in
`CHECKPOINT_STORAGE_POWER_LOSS_CAMPAIGN.md`; this Series 15 outline alone cannot
satisfy the V6 physical-power-loss gates.

## Lane G — operational evidence V5

Generate `symthaea.checkpoint-operational-evidence.v5` directly from measured
results.

Required independent gates:

- `shared_supervised_connection_limit`
- `audit_archive_retention_commitment`
- `audit_compaction_execution`
- `audit_compaction_reconciliation`
- `process_crash_recovery_coverage`

Missing lanes are `not_exercised`, never pass. Series 14 V4 reports remain
historical evidence and cannot satisfy Series 15 gates.

## Promotion rule

Promotion requires every required Series 15 gate plus all required checkpoint
confidentiality, persistence, recovery, rollback, operational, supervision,
attestation, replay, and archive gates from earlier campaigns.

## Explicit non-claims

Series 15 does not claim:

- Hardware-rooted archive retention or trusted wall-clock time.
- Protection from privileged writers that ignore advisory locks.
- Distributed admission across unrelated hosts.
- Automatic secure deletion of archived or compacted material.
- Sudden-power-loss survival before the real storage matrix is executed.
- That a retention commitment proves the repository actually retained all
  replicas without independent repository evidence.
