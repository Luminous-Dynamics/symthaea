# Checkpoint Durable Replay and Archive Campaign

Date: 2026-07-20
Status: preregistered Series 14 campaign

## Purpose

This campaign evaluates the operational boundaries introduced after the Series
13 bounded-work repairs:

- Authenticated replay state that survives daemon restart.
- Separate replay contexts for the key agent and monotonic service.
- Cross-process serialization of replay-state updates.
- Hard active-connection admission before protocol parsing.
- Independently keyed archive acknowledgment for audit exports.
- Exact-live-head compaction eligibility proofs.
- Deterministic replay-state publication faults.

The campaign does not infer any Series 14 result from a Series 13 in-memory
replay or request-rate test.

## Frozen contracts

- `symthaea.checkpoint-replay-state.v1`
- `symthaea.checkpoint-audit-archive-receipt.v1`
- `symthaea.checkpoint-audit-compaction-proof.v1`
- `symthaea.checkpoint-operational-evidence.v4`

Legacy operational-evidence V1, V2, and V3 reports remain recognizable but
cannot satisfy Series 14 gates.

## Lane A — key-agent replay persistence

1. Start with an empty replay-state directory and a fixed nonzero request ID.
2. Accept and durably record the request under the key-agent replay context.
3. Drop every guard and reopen the state from a fresh object.
4. Submit the exact same request ID.

Required gates:

- The first request is accepted only after the authenticated state file and
  parent directory are synchronized.
- The restarted guard reports `RestartDurable`.
- The repeated ID is rejected.
- A wrong replay-state key, context, capacity, authentication tag, or malformed
  membership/order relation fails closed.
- Zero request IDs are rejected without changing the state.

## Lane B — monotonic-service replay persistence

Repeat Lane A using the monotonic replay context and an independent state root
and key.

Required gates:

- Key-agent and monotonic contexts cannot open each other's state.
- The monotonic request remains rejected after restart.
- A replay-state availability or authentication failure produces an unavailable
  protocol response, not an authorization denial that could hide state loss.

## Lane C — cross-process replay serialization

Run at least two cooperating processes or independently opened guards against
one replay-state root. Release them simultaneously with the same fresh request
ID.

Required gates:

- Exactly one writer accepts and commits the identifier.
- Every other writer rejects the identifier as replay after acquiring the
  kernel lock.
- The resulting authenticated state contains one instance of the identifier.
- Replay root, lock, and state objects remain owned by the effective user and private to that user.
- The configured capacity and context remain unchanged.

## Lane D — replay publication faults

Exercise each deterministic replay-state fault point:

- Before state-file write.
- After state-file synchronization but before publication.
- After publication but before directory synchronization.

Required gates:

- Pre-publication failures do not replace the previous valid state.
- A post-publication failure blocks the protected key/rollback operation.
- If the newly published state is visible, retrying the same request is rejected.
- Temporary files are cleaned after handled failures.

This is a handled-fault lane. It does not claim survival across real power loss
until the same cases are executed on the intended filesystem and storage stack.

## Lane E — active-connection admission

Configure a supervised listener with a maximum of one active connection.

Required gates:

- The first accepted connection owns one RAII permit.
- A second admission attempt is rejected before `accept(2)` or protocol parsing.
- Dropping the first connection releases the permit.
- The pending second connection may then be accepted.
- Zero connection capacity and values above `MAX_CHECKPOINT_AGENT_CONNECTIONS` are rejected at configuration time.

Request-rate limiting and active-connection limiting remain separate gates.

## Lane F — independent audit archive receipt

1. Create and synchronize a nonempty authenticated audit export.
2. Have a separately keyed archive authority seal a receipt for the exact
   export artifact digest, export ID, head digest, count, size, repository
   binding, archive ID, and retention time.
3. Persist the receipt with no-overwrite semantics.
4. Verify it with the archive authority key.

Required gates:

- Only `Synced` exports may receive receipts.
- The caller supplies the expected repository binding again when persisting the receipt; a self-asserted binding is insufficient.
- Wrong authority key, repository binding, export identifier, export digest, export metadata, or modified receipt fails closed.
- Receipt directories and files remain effective-user-owned and private.
- The authority key is distinct from the live audit-log key.
- The receipt does not expose or authorize deletion by itself.

## Lane G — exact-head compaction proof

Generate a compaction proof from the verified export and independent receipt.

Required gates:

- The live record vector, byte length, and head digest still exactly match the
  exported segment.
- The archive receipt exactly matches the export artifact digest and metadata.
- The proof records independent retention and permits operator-managed
  compaction only for that exact head.
- Appending one additional audit record invalidates the old proof.
- The crate does not truncate or delete the live log as a side effect of proof
  generation.

## Lane H — evidence integrity

Generate `symthaea.checkpoint-operational-evidence.v4` directly from measured
results.

Required independent gates:

- Key-agent restart-durable replay.
- Monotonic-service restart-durable replay.
- Replay publication fault fail-closed behavior.
- Supervisor active-connection limit.
- Independent archive-receipt verification.
- Exact-head compaction proof.

Missing lanes are `not_exercised`, never pass.

## Promotion rule

Promotion requires every required Series 14 gate plus all required checkpoint
confidentiality, persistence, recovery, rollback, operational, supervision,
attestation, and Series 13 repair gates.

## Explicit non-claims

Series 14 does not claim:

- Hardware-rooted replay state or privileged rollback resistance.
- Real sudden-power-loss survival on untested filesystems or devices.
- Protection from privileged writers that ignore advisory locks.
- Distributed connection admission across multiple unsupervised daemons.
- That an archive receipt proves the repository will retain data forever.
- Automatic destructive audit compaction or secure deletion.
- Confidentiality of replay identifiers or audit-export metadata.

## Series 15 companion campaign

Run `CHECKPOINT_COMPACTION_CRASH_CAMPAIGN.md` for shared cross-process admission,
retention-bound compaction execution, authenticated segment continuity,
process-death recovery, reconciliation, and the filesystem/device-specific
power-loss matrix. Series 14 archive receipts and read-only compaction proofs do
not by themselves authorize or validate destructive live-segment replacement.
