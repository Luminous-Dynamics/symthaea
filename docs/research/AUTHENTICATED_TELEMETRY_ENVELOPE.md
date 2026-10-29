# Authenticated Encrypted Telemetry Envelope: Prototype Contract

**Status:** Deterministic research prototype using a pinned maintained cryptography package. Not production-approved and not yet validated by completed exact-head CI.  
**Implementation:** `tools/energy_telemetry/authenticated_envelope.py`  
**Dependency:** `cryptography==50.0.2`, pinned in `tools/energy_telemetry/requirements.txt`. PyPI lists 50.0.2 as released September 30, 2026: https://pypi.org/project/cryptography/

## Security objective

Provide a bounded envelope whose event metadata and payload are confidential in transit, integrity-protected by authenticated encryption, and tied to an explicitly trusted sender key by a digital signature. Keep the satellite-facing outer header small. This does not replace aggregation-at-source, access control, key management, anti-replay state, transport security, nor a safe response policy.

## Cryptographic profile

- **AEAD:** ChaCha20-Poly1305, with a 32-byte symmetric key and a 96-bit nonce. The maintained `cryptography` API provides authenticated encryption and associated data; its documentation warns that a nonce must never be reused with the same key: https://cryptography.io/en/latest/hazmat/primitives/aead/
- **Signature:** Ed25519, verifying that the outer canonical envelope was signed by the private key associated with a locally trusted signer key ID. Verification uses the maintained `cryptography` API: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/
- **Canonical encoding:** sorted compact ASCII JSON with non-finite values disallowed, duplicate-key rejection, strict base64 and a closed set of fields.
- **Nonce generation:** a fresh 96-bit value is sampled from the operating system CSPRNG for each sealing call. The code does not prove global uniqueness or maintain a durable per-key message counter. Production key management must enforce a conservative per-key usage cap, rotate keys, and prevent counter/state rollback across device restore.
- **Key IDs:** only opaque, non-identifying IDs are permitted by syntax. A key ID is not an identity proof; the caller's key registry maps signer IDs to trusted public keys and encryption IDs to authorized keys.

## Envelope construction

1. Validate the event metadata using the shared telemetry contract.
2. Canonically encode event metadata, length-prefix it, and append the binary application payload.
3. Encrypt that inner frame with ChaCha20-Poly1305. The associated data binds protocol version and the opaque encryption/signer key IDs.
4. Construct a closed outer object containing only protocol version, encryption key ID, signer key ID, nonce, and ciphertext.
5. Sign the canonical unsigned outer object with Ed25519 and serialize the final outer object canonically.
6. Derive the queue digest as SHA-256 of the exact final wire bytes and the wire size as their byte length.

The outer envelope intentionally does not expose event ID, domain, sequence, observation time, expiry, quality or payload. It still exposes key IDs, approximate message size and transmission timing. This does **not** conceal radio metadata or traffic analysis, and it cannot compensate for low-entropy private values included in a payload. Aggregate locally before sealing whenever the purpose permits it.

## Receiver contract

1. Bound the received envelope length before parsing.
2. Reject malformed, non-canonical JSON, duplicate keys and unknown/missing fields.
3. Resolve `signer_key_id` only through an independently configured trusted-key registry.
4. Verify the Ed25519 signature over the canonical unsigned outer object **before decryption**.
5. Resolve the encryption key from an authorized key registry and authenticate/decrypt with ChaCha20-Poly1305.
6. Parse the decrypted length-prefixed metadata, reject non-canonical or invalid metadata, and derive the queue identity digest from the exact received envelope bytes.
7. Apply receiver-side replay protection and freshness/quality policy separately. A valid signature or successful decryption does not make stale data actionable or establish that a sensor told the truth.

No trust-on-first-use behavior or automatic signer-key enrollment is implemented.

## Durable queue integration

Schema version 3 adds a nullable `envelope_bytes` BLOB. The new `enqueue_protected` API accepts the sealed-envelope result only when its metadata digest and wire size match the exact ciphertext bytes. The new `select_protected_batch` returns those same persisted bytes, checks the stored digest and size again, and records an attempt transactionally. It leaves the event pending until the ACK path accepts the matching digest.

The legacy metadata-only `enqueue` and `select_batch` interface remains for policy tests; it cannot select ciphertext rows, and protected-batch selection cannot fabricate payload bytes for legacy rows. This separation is deliberate.

The SQLite row still stores parsed event metadata in plaintext for local sorting, freshness and capacity policy. The BLOB is encrypted by this envelope profile; the database itself is **not** encrypted by this module. Restrictive local file permissions help but are not a substitute for device-level/app-level storage encryption and a private parent directory.

The authenticated-envelope module defines a canonical receiver-signed ACK (`telemetry-ack/v1`) and verifies it against an explicitly provided trusted Ed25519 receiver-key map. `DurableTelemetryQueue.acknowledge_protected` accepts only a valid positive signed ACK and checks its timestamp against sender clock-skew policy and event expiry. The legacy unsigned `acknowledge` method refuses rows containing protected envelope bytes. This authenticates the asserting key; it does not prove that the receiver actually committed delivery. The receiver must persist and deduplicate the event before signing acceptance, and that receiver-side transaction/replay ledger is not implemented here.

## Tests added

The authenticated-envelope test suite has 22 cases using fixed test-only keys and synthetic payloads. It covers round-trip, hidden metadata, payload/digest binding, signature tampering, AEAD tag tampering with a valid signature, wrong/missing keys, unknown fields, duplicate JSON keys, non-canonical serialization, malformed base64, metadata validation, key bounds, envelope length limits and signed receiver ACK verification.

Durable queue tests cover ciphertext persistence across reopen, byte-for-byte replay, digest-bound enqueue, wire-byte capacity, separation from metadata-only rows, and schema migration to version 3. These tests have been committed but must not be reported as passing until the exact-head CI run is complete.

## Explicit deployment blockers

- Key generation, secure provisioning, rotation, revocation, hardware-backed storage and incident response.
- Enforcement and audit of per-key AEAD message limits/nonce policy across restarts, backup restoration and multiple producers.
- Production key provisioning/rotation/revocation, target-filesystem crash/power-loss tests and a reviewed downstream outbox/transaction. The receiver inbox added here commits its own replay record before ACK but does not provide exactly-once downstream effects.
- Encryption of local persistent metadata as required by the device threat model.
- Canonical signed trust/policy context, authorization, and an independently reviewed key-to-producer binding.
- Multi-process, crash/power-loss, storage-full, key rotation, backup/restore and target-device tests.
- Differential-privacy accounting and secure aggregation remain independent components; neither is implemented by this envelope.
- Completed exact-head CI and security review. No meter, household, satellite or live production system was used for these tests.

## Related standards

- ChaCha20-Poly1305 API and nonce rule: https://cryptography.io/en/latest/hazmat/primitives/aead/
- Ed25519 signing and verification API: https://cryptography.io/en/latest/hazmat/primitives/asymmetric/ed25519/
- Delay-tolerant transport context (this is not a BPv7 implementation): https://www.rfc-editor.org/rfc/rfc9171.html
