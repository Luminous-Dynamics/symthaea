# MACE memory lifecycle reference kernel v0.1

**Status:** research reference model; uncompiled and unexecuted in the current authoring environment. This is not a production memory store, cryptographic verifier, or agent defense.

## Purpose

This crate makes a small set of lifecycle invariants executable as a reference API:

1. A content digest establishes byte consistency only. It does not authenticate the issuer, grant authority, establish semantic truth, or prove admissibility.
2. Issuer identity and action authority come from an injected `IdentityAuthority` profile, not from fields asserted by a memory record.
3. A derived record is eligible only when its own issuer is authorized and every required parent is currently eligible for the same scope and purpose.
4. Revoking a parent makes every dependent descendant unusable at current-at-use, even when the descendant remains physically stored.
5. Erasure writes a tombstone for the full derived-descendant closure to a separate `TrustedWitness`, then removes that closure from the working store.
6. A store snapshot cannot be served merely because its internal hashes are self-consistent. The snapshot generation and canonical head must match the separately supplied witness.
7. Recovery from a pre-erasure backup is allowed only after applying the witness tombstones and reproducing the witness's committed post-erasure head. If unrelated writes are missing or bytes differ, recovery fails closed.

## Trust boundaries

- `IdentityAuthority` is an adapter contract. A production implementation must authenticate issuer credentials, key status/rotation, scope, purpose and revocation using an independently trusted key/profile system. The fixture implementation in unit tests is not an authenticator.
- `MonotonicWitnessStore` is the pluggable persistence interface. Its contract requires atomic compare-and-append, monotonic generations, checkpoint-chain continuity, and durable storage outside the memory snapshot's rollback domain. Each `WitnessCheckpoint` commits the store head, generation, revocation tombstones, erasure tombstones and previous checkpoint hash.
- `TrustedWitness` is only the in-memory reference implementation of that interface. It does not implement remote attestation, a transparency log, quorum, a hardware monotonic counter, or secure persistence. A production adapter must authenticate the checkpoint chain and retain a trust anchor outside the data-store rollback domain. If the witness can be rolled back with the store, the model's rollback protection does not hold.
- The compare-and-append interface is expected to be atomic: on failure, the previous witness checkpoint remains authoritative. The kernel restores its working snapshot when witness append fails, and a deterministic test covers that path.
- SHA-256 is used for deterministic byte/head commitments. It is not a signature and does not authenticate the origin of a statement.
- Semantic truth, source correctness, poison intent and downstream action safety are outside this kernel. A genuinely authenticated source can still be wrong or malicious; a policy/evidence adjudicator must separately assess the claim and the requested decision.
- The crate does not execute tools, network requests, attacks, or destructive operations. Tests use inert fixture records and in-memory snapshots only.

## Run

```sh
cargo test --manifest-path research/mace-memory-lifecycle-kernel/Cargo.toml
cargo clippy --manifest-path research/mace-memory-lifecycle-kernel/Cargo.toml --all-targets -- -D warnings
```

These commands are prescribed but have not been executed here. The reference package must be independently reviewed and run on an exact commit before any qualification claim.

## Filesystem witness reference adapter

The crate now includes `file_witness::FileBackedWitnessStore`, an append-only filesystem reference implementation of `MonotonicWitnessStore`. It uses owner-only Unix directory/file permissions, bounded checkpoint records, atomic create-if-absent publication, generation-chain replay, tombstone monotonicity checks and Unix directory fsync. It is intended for local restart/conformance testing, not production independent trust.

**Critical limitation:** if the witness directory is restored together with the memory database from the same snapshot/backup/VM image, both can be rolled back together. A local hash chain does not prevent whole-directory rollback or privileged offline replacement. Production still requires a separately administered witness service or hardware-backed monotonic facility with an independently authenticated trust anchor. See [FILE_BACKED_WITNESS.md](FILE_BACKED_WITNESS.md).

## Next integration gates

1. Supply a production `IdentityAuthority` backed by a real issuer/key registry and cryptographic receipt verifier.
2. Supply a `TrustedWitness` implementation backed by a separately administered monotonic service or verifiable append-only log.
3. Define authenticated correction/supersession receipts and bitemporal as-known replay.
4. Add an actual memory-backend adapter and isolated agent harness; reproduce MemSecBench, MemoryDojo, MemPoison/MPBench and the storage-integrity Internet-Draft in their own protocols.
5. Evaluate poisoning persistence, retrieval exposure, harmful action, selective repair, benign-memory preservation, rollback and erasure resurrection separately. Do not combine native benchmark scores.
