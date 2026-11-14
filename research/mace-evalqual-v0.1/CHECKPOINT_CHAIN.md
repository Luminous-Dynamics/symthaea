# MACE detached checkpoint chain v1

**Status:** proposed wire/signature contract with a locally tested Go verifier. The signer, checkpoint transport, secure key custody and independent witness service are not implemented in the public PR.

## Purpose

The SQLite observer's receipt hash chain provides internal consistency; it does not authenticate a total rewrite by an actor able to modify all rows and recompute their hashes. A checkpoint lets a separately administered witness sign a particular database identity and journal head. A consumer can then compare the observed database against an out-of-band signed statement and retain a monotonic cursor to detect rollback or forks.

## Signed statement

The v1 checkpoint binds these exact fields:

- `schema_version`: `mace-checkpoint/v1`.
- `key_id`: stable identifier of the pinned witness public key.
- `database_id`: immutable identity of the logical destination/database.
- `checkpoint_sequence`: strictly increasing witness checkpoint number.
- `journal_sequence`: terminal-record sequence visible at the destination.
- `journal_head_sha256`: lowercase SHA-256 of the destination journal head.
- `previous_checkpoint_sha256`: digest of the previous signed envelope; absent only for checkpoint sequence 1.
- `issued_at_unix`: witness issue time in Unix seconds.

The transport envelope is `{checkpoint, signature}`. JSON encodes the Ed25519 signature as base64. The signature is over a fixed binary encoding, **not** over serialized JSON, so cross-language implementations do not depend on JSON key ordering or whitespace.

## Canonical signing bytes

Domain prefix (ASCII bytes): `LUMINOUS-MACE-CHECKPOINT\0v1\0`.

After the prefix, append fields in this exact order:

1. `schema_version`, `key_id`, `database_id`: each UTF-8 byte string preceded by a 4-byte unsigned big-endian byte length.
2. `checkpoint_sequence`: unsigned 64-bit big-endian.
3. `journal_sequence`: unsigned 64-bit big-endian.
4. `journal_head_sha256`: 32 raw bytes decoded from lowercase hexadecimal.
5. predecessor length: one byte, either 0 for genesis or 32; then the 32 raw predecessor digest bytes when present.
6. `issued_at_unix`: positive Unix seconds encoded as 64-bit big-endian.

Sign the resulting bytes with Ed25519. The envelope digest used as the next checkpoint's predecessor is SHA-256 over `LUMINOUS-MACE-CHECKPOINT-DIGEST\0v1\0` followed by the canonical signing bytes and the 64-byte signature.

## Consumer state machine

Bootstrap requires the consumer to pin a trusted public key, key ID and database ID out-of-band. Bootstrap accepts an initial valid signed checkpoint, but does not prove that it is fresh; provisioning must obtain the initial state from a trusted process.

For each subsequent checkpoint, the consumer verifies the signature and identity, then requires:

- exact replay of the current envelope is idempotent;
- checkpoint sequence equals current sequence + 1;
- predecessor digest equals the current envelope digest;
- journal sequence strictly advances;
- issued time does not move backward;
- database ID and key ID do not change implicitly.

Older checkpoints are rollback. A different envelope at the current sequence is a fork. Missing checkpoint numbers, incorrect predecessor hashes, stale journal heads or identity changes are rejected. Key rotation requires a separately specified and authorized transition; v1 does not silently accept a new key.

The consumer persists its returned monotonic trust state atomically in a domain **outside the destination database and outside the database administrator's trust boundary**. Storing that cursor beside the database makes rollback protection ineffective. If both the database and the witness signing key are controlled by the same adversary, that adversary can sign a malicious replacement chain.

## Conformance vector

`fixtures/checkpoint-v1.json` publishes one synthetic test vector with the checkpoint, public key, exact signing bytes, signature in hex/base64 and envelope digest. The private test key is not published. The vector can be implemented in another language without importing the Go package.

The local companion contains a standard-library Ed25519 verifier under `checkpoints/`. Its tests cover valid signatures, advancement, exact replay, rollback, same-sequence fork, field/signature tamper, wrong key/database, predecessor mismatch, sequence gap, journal non-progress, issue-time rollback, malformed trusted state and JSON transport round-trip. That source is currently a separate local companion artifact, not part of this public PR's code diff.

## What v1 does not provide

- remote signer or witness service;
- secure private-key custody, HSM integration, signing policy or key revocation;
- quorum witnesses, transparency log, split-view/fork gossip or independent timestamping;
- automatic key rotation;
- durable external storage of the consumer's trust cursor;
- proof that every downstream real-world effect was observed.

Therefore v1 is a signature/monotonic-cursor building block, not an E3/E4 witness deployment or certification. The next implementation step is a genuinely separate witness service with separately administered keys, durable monotonic checkpoint storage and split-view detection.



## Reference verifier implementation

The public branch now includes `checkpoints/`, a standard-library Go implementation of the v1 signing preimage, Ed25519 verifier, envelope digest, bootstrap and next-checkpoint consumer state machine. Its tests validate the published vector and exercise signature tampering, identity mismatch, exact replay, rollback, fork, gap, broken predecessor, no-progress journal head, issue-time rollback and corrupt trusted state. The verifier contains a `Sign` helper for deterministic tests and controlled witness implementations; it is not a deployed signer service and must not be used as a production key-management system.

The repository's public test script includes the package in `go test ./...` and checks all three public CLIs. TLC remains a separate, conditional verification tier and is explicitly reported as not run unless a pinned JAR is supplied.


The public verifier now has an optional cross-implementation test that verifies the same published signing preimage and Ed25519 signature with OpenSSL's CLI. This is independent crypto-provider interoperability evidence, not an independent implementation of the checkpoint state machine. The test skips explicitly where OpenSSL is absent; the mandatory Go vector test still runs.


## Join checkpoint to the observed destination

The reference API includes `AcceptObserved`, which requires the independently observed database ID, journal sequence and journal-head digest to match the signed checkpoint before it advances `TrustedState`. This removes a common integration mistake: verifying a valid signature but forgetting to compare it with the database snapshot being audited. The observer must obtain all three observed values from one coherent transaction snapshot. `AcceptObserved` cannot upgrade the observer's trust domain or prove events outside its coverage.


The current public reference package has 21 checkpoint tests, including the `AcceptObserved` join cases and an optional OpenSSL cross-provider signature check. The integration API checks identity, sequence and head equality before accepting an advanced checkpoint. It still relies on the caller to obtain a coherent observer snapshot and keep the monotonic trust cursor in an independent domain.


A checkpoint is not considered progress merely because its integer sequence advances: the journal-head digest must also differ from the currently trusted digest. A new witness checkpoint that repeats the old head is rejected as `ErrNoProgress`.


## Peer consistency and split-view detection

The checkpoint verifier includes `VerifyPeerConsistency(older, newer, bridge, publicKey)`. It validates that the newer trusted cursor is a descendant of the older cursor under the same witness key/database, using every intervening signed checkpoint. It detects a same-sequence fork, a missing/broken bridge, identity substitution, rollback, or a peer cursor not reached by the signed chain.

This is a **proof comparison primitive**, not split-view prevention. A witness may present two signed histories to isolated clients; the clients detect the fork only after exchanging cursors/bridges or submitting them to a common monitor. Without gossip, transparency-log consistency proofs or an independent witness quorum, an undisclosed fork can remain hidden. Peer consistency therefore complements but does not replace independent checkpoint issuance, durable external cursor storage and key custody.
