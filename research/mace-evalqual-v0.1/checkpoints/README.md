# Detached signed checkpoint verifier

This package defines a narrow Ed25519 verification primitive for database/journal head checkpoints. It does **not** run a checkpoint service, create a hardware-backed key, or persist the consumer's trust state.

## Security contract

A checkpoint binds:

- schema version;
- witness key ID and database identity;
- monotonic witness checkpoint sequence;
- destination journal sequence and SHA-256 head;
- previous signed-checkpoint digest;
- issue time.

The Ed25519 signature covers a versioned binary representation, not JSON. Text fields are 32-bit big-endian length-prefixed UTF-8 bytes; sequence fields and issue time are 64-bit big-endian; journal head and predecessor are raw 32-byte SHA-256 values (the predecessor is length-tagged as zero or 32 bytes). The signing domain is `LUMINOUS-MACE-CHECKPOINT\0v1\0`. The envelope digest is SHA-256 over `LUMINOUS-MACE-CHECKPOINT-DIGEST\0v1\0`, signing bytes and signature. This gives external implementations an unambiguous format without relying on JSON canonicalization.

`Bootstrap` verifies a supplied public key plus expected key/database IDs. It does not prove that the initial checkpoint is fresh; provisioning must obtain its first anchor from a trusted process.

`AcceptNext` rejects an older signed checkpoint, a same-sequence fork, a skipped witness checkpoint sequence, a broken predecessor link, a changed key/database identity, a non-advancing journal sequence, or an issue-time rollback. Exact replay of the current envelope is idempotent. The caller must persist the returned `TrustedState` atomically **outside the destination database and outside the database administrator's trust domain**. Resetting or rolling back that state defeats rollback detection.

## What this does not provide

- a remote signer, witness quorum, transparency log, or independent administrator;
- secure key storage, HSM integration, key rotation or revocation;
- protection if the same actor controls both the database and signing key;
- proof that every real-world effect was observed;
- freshness during bootstrap without an out-of-band trusted starting state.

This is a verification building block, not a deployed E3/E4 witness. The tests use ephemeral keys and synthetic database identifiers only. A fixed synthetic test vector is published in `testdata/checkpoint-v1.json`; it includes public verification material and deterministic signing bytes, not a production key.

## Tests

The package tests valid signature and monotonic advance, exact replay, rollback, same-sequence fork, field/signature tampering, wrong database/key, predecessor mismatch, sequence gap, no-progress checkpoints, issue-time rollback, genesis validation and JSON transport round trip.
