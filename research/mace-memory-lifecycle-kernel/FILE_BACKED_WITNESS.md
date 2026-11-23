# MACE file-backed monotonic witness adapter v0.1

**Status:** local filesystem reference/conformance adapter. The code and tests are authored but have not been compiled or executed in this environment. This is not a production independent witness, remote trust anchor, cryptographic issuer verifier or defense against whole-filesystem rollback.

## Purpose

`file_witness::FileBackedWitnessStore` implements the existing `MonotonicWitnessStore` trait using immutable, append-only checkpoint files. It provides a concrete persistence adapter for restart and compare-and-append integration tests without adding a new crate dependency.

Each checkpoint binds:
- monotonically increasing generation;
- canonical memory-store head SHA-256;
- revocation tombstones;
- erasure tombstones;
- preceding checkpoint digest;
- its own checkpoint digest.

The adapter replays and validates the complete generation sequence on open and each latest-checkpoint read. Gaps, malformed names, non-regular checkpoint entries, invalid encodings, invalid checkpoint digests, chain discontinuities and tombstone regressions fail closed.

## Atomic publication model

For each append, the adapter:
1. reloads and validates the current checkpoint history;
2. checks the caller's expected checkpoint digest and next generation;
3. serializes the checkpoint into a bounded binary record;
4. creates a unique temporary file in the same directory;
5. writes and fsyncs that file;
6. publishes the generation by a hard-link create-if-absent operation, which cannot overwrite an existing generation;
7. removes the temporary name and fsyncs the directory on Unix;
8. reads the published checkpoint back and verifies that it exactly matches the requested checkpoint.

A competing writer for the same generation receives a conflict rather than overwriting the winner. Checkpoint files are immutable by naming convention and append-only API.

### Ambiguous failure handling

A failure after the final hard link is created but before directory fsync/read-back can leave the new generation visible even though the caller received an error. Treat that result as **unknown**, reopen the witness and reconcile against its latest checkpoint before retrying a mutation. Never assume every returned I/O error means no commit occurred.

## Security boundary — critical

A local file-backed witness is not an independently administered trust anchor.

If the memory database and this witness directory are included in the same filesystem image, VM snapshot, volume backup or restore set, an attacker/operator can restore both to an earlier generation. This recreates the exact common rollback-domain problem the witness abstraction is intended to avoid.

This adapter also does not:
- sign checkpoints or authenticate a witness issuer;
- protect against offline modification of the directory by a privileged account;
- detect replacement of the entire directory with an older valid checkpoint chain;
- provide quorum, remote attestation, a hardware monotonic counter or transparency-log inclusion;
- authenticate the `IdentityAuthority` implementation;
- prove semantic truth or source correctness.

A production deployment requires a separately administered witness service or hardware-backed monotonic facility, with its own authenticated key/trust-anchor lifecycle, independent backup/restore policy, access controls, monitored administration and explicit disaster-recovery protocol. Keep it outside the memory-store rollback and erasure domain. The file adapter is useful for contract tests and single-host crash/restart testing only.

The Internet-Draft's T9 and T10 cases must remain separate. A filesystem adapter in the same restore domain can pass ordinary process-restart tests and still fail both whole-store rollback and erasure-resurrection tests.

## Limits

- Maximum checkpoint file: 16 MiB.
- Maximum revoked IDs: 100,000 per checkpoint.
- Maximum erased IDs: 100,000 per checkpoint.
- Maximum UTF-8 field: 4,096 bytes.
- Maximum replayed generations: 1,000,000.
- Unix directory fsync is used after publication. On non-Unix platforms the reference fallback does not claim directory-entry durability; use a platform-specific, tested durable-publication primitive before production use.

## Build and tests

```sh
cargo test --manifest-path research/mace-memory-lifecycle-kernel/Cargo.toml
cargo clippy --manifest-path research/mace-memory-lifecycle-kernel/Cargo.toml --all-targets -- -D warnings
```

The nine authored tests cover restart persistence, tombstone round-trip, stale writer conflict, concurrent publication, tombstone regression rejection, malformed-name rejection, tamper detection, generation-gap rejection, and interrupted pending files. An additional end-to-end T10 fixture commits a protected record, takes a stale pre-erasure data backup, erases the record, reopens the current file witness and verifies that reconciliation removes the resurrected record. The same fixture restores both the pre-erasure data and pre-erasure local witness and demonstrates that the pair is mutually consistent again: the erased record is accepted. This is an explicit test of the adapter's common-rollback-domain limitation, not a security pass. The tests have not been executed here.

## Next gates

1. Compile, format, Clippy and execute tests at the exact PR head.
2. Add crash injection at every write/fsync/link/unlink boundary and test the “unknown outcome → reopen/reconcile” protocol.
3. Integrate this adapter with `commit_record`, `revoke_record`, `erase_record` and backup reconciliation using a persisted memory snapshot.
4. Add T9/T10 fixtures where only the data store is restored and where both store and local witness are restored. The latter is expected to demonstrate this adapter's limitation.
5. Implement the production witness in a genuinely separate administrative/failure domain and verify key rotation, quorum/availability, trust-anchor recovery and cross-domain backup/restore.
