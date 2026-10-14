# Durable Receiver Inbox and Commit-Before-ACK Contract

**Status:** Deterministic research prototype. Exact-head CI remains the verification gate; no live sensors or satellite links were used.  
**Implementation:** `tools/energy_telemetry/receiver_inbox.py`  
**Tests:** `tests/test_energy_telemetry_receiver_inbox.py`

## Contract

A positive receiver ACK means: “the receiver authenticated this exact protected envelope and durably recorded its ciphertext and deduplication key before issuing the ACK.” It does **not** mean the measurement is true, the payload is safe to act on, or downstream processing succeeded.

The receiver does the following, in this order:

1. Bound the incoming wire bytes and verify the canonical envelope signature against a caller-supplied trusted Ed25519 key map.
2. AEAD-decrypt the envelope in memory using an authorized key map; validate strict metadata, event identity and time constraints.
3. Reject expired envelopes and observation timestamps beyond the configured future skew. Stale but unexpired data may be archived/accepted but is explicitly non-actionable.
4. Begin an SQLite write transaction and look up the event ID.
   - If the same ID and exact digest/bytes already exist, classify it as an idempotent duplicate.
   - If the event ID exists with different protected bytes/digest, fail closed.
   - Otherwise insert a record containing the event ID, SHA-256 of the exact protected bytes, exact ciphertext BLOB, accepted time and expiry.
5. Commit the inbox transaction.
6. Only after the commit returns, create a canonical Ed25519-signed `telemetry-ack/v1` positive ACK bound to the exact event ID and digest.

This order permits recovery if the receiver commits but crashes before writing the ACK: the sender can retry the exact same ciphertext, the inbox detects an exact duplicate, and the receiver signs a fresh ACK. A different ciphertext reusing that event ID is a conflict, even if a sender claims it represents equivalent cleartext. Senders therefore must replay the exact stored wire bytes, not re-encrypt with a fresh nonce on retry.

## Persistence and privacy

The inbox persists **ciphertext only** for the sensor/application payload. The inner metadata and payload were encrypted together by the envelope layer. It does store event ID, envelope digest, acceptance timestamp and expiry in plaintext for replay prevention and retention/queue management. Those metadata remain sensitive: use a private parent directory, restrictive file permissions, encrypted storage/backup where required, and a bounded retention/archive policy.

The receiver returns the decrypted payload to the authorized caller in process memory after durable acceptance, along with the signed ACK and a separate freshness/actionability assessment. It does not write the plaintext payload to the inbox database. Callers must still authenticate their own downstream sink and separate receipt of data from decisions or actuation.

## Remaining boundaries

- **Key registry:** sender and receiver key IDs are resolved using caller-supplied mappings. Provisioning, producer authorization, key rotation, revocation and hardware-backed key storage are external.
- **Nonce lifecycle:** the envelope creator generates a fresh random nonce, but durable per-key usage state, conservative invocation caps and recovery/restore rules are not enforced here.
- **Receiver-side processing:** the inbox commits receipt before ACK. It does not atomically commit domain-specific processing side effects. Those need an outbox/transaction or idempotent consumer contract of their own.
- **At-most-one inbox row, not exactly-once effects:** unique event IDs plus exact digest comparison provide durable deduplication at this inbox. They do not guarantee exactly-once execution in downstream systems.
- **Storage security:** SQLite data/metadata files are not encrypted by this module. File permissions are not a replacement for device-level/application-level encryption where physical theft or privileged local access is in the threat model.
- **Privacy accounting:** transport retransmission dedupe is independent of the differential-privacy release ledger. A duplicate packet should not spend budget again, but a new private statistic/query must pass the separate privacy accountant.
- **Network semantics:** this is an application-level receiver inbox, not a replacement for IETF Bundle Protocol v7 or a CCSDS space-link profile.

## Deterministic test coverage

The suite uses synthetic event metadata and fixed test-only keys. It covers successful signed-ACK round-trip, ciphertext-only persistence, restart and duplicate retries, conflicting event ID replay, unknown sender keys, wrong encryption keys, bad signatures, expired/future-dated envelopes, stale-but-not-expired events, capacity and configuration bounds, foreign/unowned database rejection, file permissions, concurrent duplicate acceptance, and an injected transaction abort proving no positive ACK is returned when the durable insert fails.

These are test cases added to the repository; they must not be described as passing until the dedicated exact-head GitHub Actions run completes successfully.
