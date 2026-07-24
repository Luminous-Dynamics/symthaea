# Symthaea Wisdom Series VIII Integration Guide

Series VIII closes the operational-history side of bounded retention. Series VII
can restore executable authority from a signed authority checkpoint and a retained
suffix, but it intentionally cannot reconstruct the evicted operational prefix.
Series VIII verifies signed archive segments, reconstructs the exact historical
record stream, replays operational state, restores authority separately, and only
then admits the runtime service.

## Safety model

Operational state and executable authority are different proofs:

- **Archive replay** proves what operational state follows from the complete
  evidence history.
- **Authority recovery** proves which permits, execution attempts, in-doubt
  actions, and runtime cursors remain valid after retention.
- **Startup admission** proves those reconstructions agree with the exact current
  durable ledger, trust registry, policy, and runtime-source identities.

No one proof substitutes for either of the others.

## Archive persistence

`EvidenceArchiveSegment::to_canonical_bytes()` and
`EvidenceArchiveSegment::from_canonical_bytes()` define a strict transport format.
The decoder rejects malformed domains, invalid UTF-8, oversized fields, truncated
payloads, trailing data, and non-canonical re-encodings.

`EvidenceArchiveStore` supplies an append-only storage boundary. The included
`ProcessFileArchiveStore` is crash-durable for one process through temporary-file
creation, file synchronization, rename, and directory synchronization where the
platform supports it. It is not a multi-process transaction or consensus system.

Production deployments with multiple writers must place the store behind
transactional SQL, consensus KV, or another backend that provides a genuinely
atomic append contract.

## Archive signing-key rotation

A complete archive may contain segments and embedded retention checkpoints signed
by different authorized identities. Use:

- `ArchiveVerifierSet` to resolve bounded sets of algorithm/key identities;
- `verify_archive_chain_with_trust()` to authorize each identity at the segment
  or checkpoint creation time;
- the exact `TrustRegistry` expected by the deployment.

A mathematically valid signature from an unknown, expired, retired, or revoked
identity is insufficient.

## Complete operational restoration

Call `restore_operational_state_from_archive()` with:

1. the ordered signed archive segments;
2. the exact current retained `EvidenceLedger`;
3. a key-resolving archive verifier;
4. the deployment trust registry;
5. the expected `WisdomConfig` and replay policy;
6. an explicit `ArchiveOperationalRestorePolicy`.

The restoration path:

1. verifies every segment and embedded checkpoint;
2. verifies segment indices, predecessor fingerprints, creation-time monotonicity,
   and retention overlap;
3. merges overlapping snapshots by evidence sequence;
4. rejects divergent duplicate records and sequence gaps;
5. enforces a bounded materialization budget;
6. replays the final archived prefix and checks its signed operational-state claim;
7. replays the complete record stream through the current retained tail;
8. installs the exact current durable ledger into the restored state.

The default policy requires history to begin at sequence zero, requires overlap
with the current durable head, verifies the final signed state claim, and limits
materialization to one million records.

## Authority restoration and startup

After operational restoration:

1. set the configured structural ethics policy on the restored state;
2. call `recover_authority_from_checkpoint()` using the signed authority and
   retention checkpoints plus the current retained ledger;
3. call `validate_archive_operational_startup_with_authority()`;
4. mint an `OperationalStartupPermit` from the ready report;
5. activate `WisdomRuntimeService` immediately while the permit still binds the
   same ledger revision, state, policy, trust registry, and source set.

`WisdomRuntimeService::bootstrap_from_archive_with_preflight()` performs this
sequence under one fenced durable writer and returns both the archive restoration
and startup reports.

The compatibility helper `bootstrap_from_archive()` accepts an externally minted
admission permit. New production integration should prefer the preflight variant
so restoration, validation, permit minting, and activation remain adjacent.

## Retention rotation

At an exact committed service head:

1. stop or fence mutations for the rotation transaction;
2. call `prepare_retention_rotation()` to create an archive segment and authority
   checkpoint over the same `LedgerRevision`;
3. call `verify_retention_rotation_bundle()` before persistence;
4. serialize the complete `RetentionRotationBundle` with
   `to_canonical_bytes()` and durably persist that atomic artifact;
5. append its archive segment to the configured archive store;
6. record the authority checkpoint location and fingerprint in deployment state;
7. only after every required artifact is durable may historical evidence be
   evicted from the live ledger;
8. reopen or revalidate the persisted bundle before considering rotation complete.

The canonical bundle binds the exact ledger revision, archive segment, retention
checkpoint, authority checkpoint, and operational-state fingerprint. Substitution
of either nested artifact fails decoding or verification.

`persist_retention_rotation()` verifies the bundle and appends the archive segment,
but its storage trait does not itself atomically persist the authority checkpoint.
Deployments must persist the canonical complete bundle—or provide an equivalent
transactional store—before eviction.

## Startup sequence after retention

1. Load and verify the exact trust-registry snapshot.
2. Load the ordered archive segment chain.
3. Load the current durable ledger head and its fenced revision.
4. Decode and verify the current authority/retention checkpoint pair.
5. Restore complete operational state from archives and the retained head.
6. Restore executable authority from the checkpoint and retained suffix.
7. Run archive-aware startup preflight.
8. Mint and immediately consume startup admission.
9. Accept scheduler work only after the service is active.
10. Reconcile every in-doubt execution before release or automated retry.

## Resource and failure policy

- Archive segment count, ledger bytes, verifier count, strings, signatures, and
  reconstructed records are bounded.
- Unknown keys, malformed encodings, missing history, divergent overlap, stale
  state claims, and current-head mismatch fail closed.
- Archive unavailability is a startup failure when complete history is required.
- An archive verifier error never falls back to trusting the current retained head.
- A valid archive never grants action authority by itself.

## Non-claims

Series VIII does not provide:

- archive replication, geographic durability, or availability guarantees;
- Byzantine consensus or multi-process filesystem transactions;
- secure wall-clock time;
- hardware key custody;
- automatic trust-registry distribution;
- truthfulness of signed runtime producers;
- atomic commitment between an external side effect and local evidence;
- a built-in transactional store for the complete rotation bundle;
- compilation or workspace verification in environments without the Rust toolchain.
