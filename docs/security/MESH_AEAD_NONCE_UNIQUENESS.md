# Mesh AEAD Nonce Uniqueness Review

**Date:** 2026-10-22  
**Status:** security design finding; remediation required before relying on restart-safe AEAD  
**Scope:** feature-gated `mesh-encryption` helpers in `src/swarm/mesh/mod.rs`  
**Deployment statement:** this review establishes source-level conditions and counterexamples, not that an affected build is deployed or has been exploited.

## Executive finding

The current helper forms a 96-bit ChaCha20-Poly1305 nonce as:

```text
source_id[0..6] | payload_type[1] | epoch[1] | sequence_le[4]
```

The implementation and its comments previously described an 8-bit epoch as preventing nonce reuse across restarts. That statement is too strong. `RotatingKeyPair::new` draws a random 8-bit epoch, but `DualLayerMesh::new` initializes `encryption_epoch` to **zero** and does not randomize it; the ordinary `with_encryption_key` builder does not change that field. In that path, a restart with the same key/source/sequence namespace can repeat a nonce deterministically. Where independent uniform one-byte epochs are used, they match with probability **1/256**. The nonce also uses only the first 48 bits of the packet's 64-bit `source_id`, and the 32-bit sequence repeats after wrap if the key and epoch remain unchanged.

RFC 8439 requires a different 96-bit nonce for each invocation under the same key and explains that nonce reuse repeats both the keystream and the one-time Poly1305 key. RFC 8439 also highlights the multiple-sender case: nonce namespaces must not collide when senders share a key. See https://www.rfc-editor.org/rfc/rfc8439.html and https://www.rfc-editor.org/rfc/rfc5116.html.

**Classification:** a confirmed nonce-design weakness in the current API's stated restart-safety guarantee. Real-world exploitability depends on configuration: the key must be reused and the sender must repeat the relevant nonce fields. This review does not assert that all deployments reuse encryption keys or that any deployment is currently exposed.

## Source-grounded failure conditions

### N1 — Eight-bit epoch is not a restart uniqueness guarantee

`RotatingKeyPair::new(key)` accepts supplied key material and generates `epoch: u8`. A later process may receive the same key and draw the same epoch as before. If source identity, payload type and sequence repeat—as they can when process state restarts—the resulting nonce is identical.

This is not a claim that random bytes are biased; it is a claim that a 256-value namespace is too small to provide a robust uniqueness guarantee. A statistical test that samples random epochs is not an adequate regression test: the failure must be represented as a deterministic state-machine counterexample.

### N2 — Sender identity is truncated inside the nonce

`build_nonce` accepts an eight-byte `source_id` but copies only `source_id[..6]`. Two IDs that differ only in their final two bytes produce the same nonce when type, epoch and sequence match. This becomes an AEAD nonce collision when both invocations use the same effective key.

Whether this is reachable in a deployment depends on identity allocation and key sharing. If all senders use one group key, the protocol must deliberately partition nonce space across senders; it must not assume that a truncated identifier is unique. If senders use independently derived keys, the key schedule must actually provide independent per-sender secrets—not merely derive all sender keys from a group secret known to every member if member-level identity authentication is required.

### N3 — The sequence namespace wraps

The nonce includes a 32-bit sequence. Once the sequence repeats under the same key, sender prefix, payload type and epoch, the nonce repeats. At 50 increments per second, a continuously incremented 32-bit counter wraps in roughly 2.72 years; lower-frequency streams wrap later. Wrapping is not safe under an unchanged AEAD key unless the protocol also changes to a fresh key namespace before reuse.

### N4 — Fragment-level AEAD repeats nonce after 16-bit thought ID wrap

The independent fragment-encryption path builds its nonce as:

```text
source_id[0..8] | thought_id[0..2] | fragment_index | 0
```

`WisdomPacket::thought_id()` returns `sequence as u16`. Therefore, for the same key, source ID and fragment index, sequence values that differ by 65,536 produce the same fragment nonce. At 50 packet-sequence increments per second, the low 16-bit identifier wraps in about 21.8 minutes. Even at lower message rates, wrap remains inevitable if the key and sender persist long enough. The fragment nonce does not contain `DualLayerMesh::encryption_epoch`, so that epoch cannot separate fragment ciphertexts across restarts. Resetting a sender and reusing its fragment sequence namespace can repeat nonces immediately.

The new deterministic test `legacy_fragment_nonce_reuses_after_16_bit_sequence_wrap` constructs packets whose full sequences differ but whose `thought_id()` matches, then confirms that fragment ciphertexts reuse the nonce and expose the XOR of equal-length plaintexts. This is a known-broken behavior witness, not a security-passing assertion. It has been added but not compiled or run in this environment.

**Impact condition:** the same effective key is used for both fragment encryptions. If callers use independently established per-session keys, the collision in nonce bytes alone is not a same-key nonce reuse. The code and configuration must establish that guarantee rather than assume it.

### N5 — Fragment-level encryption exceeds the declared LoRa MTU

`LORA_MTU` is 222 bytes and the receive path allocates a fixed `[u8; LORA_MTU]` buffer. A serialized full-size `LoRaFragment` is 222 bytes. `encrypt_fragment` wraps that whole serialized frame with a 12-byte nonce and a 16-byte AEAD tag, yielding **250 bytes**. The send path passes that oversized buffer to `send_raw`, while the receiver's fixed-size read buffer can hold only 222 bytes. A physical transport must reject, truncate or otherwise fail to convey the encrypted frame as a single declared LoRa frame unless a lower layer explicitly supports larger frames—which would contradict the stated MTU contract.

This path is selected only when the encryption key and fragment-encryption flag are both configured. The sender already encrypts the packet envelope before fragmentation, so fragment-level encryption is an additional wrapper with meaningful size and latency costs. It cannot be treated as a free upgrade. A complete fix must either:
- remove/deprecate the extra fragment wrapper where packet-level AEAD supplies end-to-end integrity, or
- version the framing and account for the 28-byte wrapper in the fragmentation capacity, FEC recovery, reassembly and transport MTU; the smaller payload changes fragment count and radio cost.

Do not merely increase the read buffer: that does not make a 250-byte frame fit a 222-byte physical MTU.

The deterministic test `legacy_fragment_encryption_exceeds_lora_mtu` builds a maximal legacy fragment and asserts its wrapped output is exactly `LORA_MTU + AEAD_NONCE_SIZE + AEAD_TAG_SIZE` (250 bytes). This test is a legacy-failure witness, not a transport-safety pass. It has not been compiled or run in this environment.

### N6 — Authentication and encryption identities are separate protocol questions

An HMAC-authenticated outer packet and a separately encrypted envelope must not be assumed to share the same identity or replay semantics automatically. Every key/nonce tuple must be evaluated using the *effective encryption key*, not only the outer packet's source ID or MAC. A shared group key also permits any key holder to construct packets that appear to come from another group member; nonce hygiene cannot fix that identity-authentication limitation.

## Immediate disposition

- Do not treat the current one-byte epoch as a guarantee against nonce reuse.
- Do not enable the encryption path with a key reused across process restarts unless the deployment supplies a separately reviewed uniqueness mechanism.
- Do not silently patch the legacy helper's nonce layout in place: the helper's output is on-wire data and an incompatible change requires an explicit protocol version and migration plan.
- Keep the new construction research-only until the property suite, key-management assumptions and interoperability plan are independently reviewed.

The packet-level and fragment-level source comments have been corrected to remove false restart-safety guarantees. Three deterministic legacy witnesses are now present in `src/swarm/mesh/tests.rs`: `legacy_nonce_source_id_truncation_reveals_plaintext_xor`, `legacy_fragment_nonce_reuses_after_16_bit_sequence_wrap`, and `legacy_fragment_encryption_exceeds_lora_mtu`. Both are deliberately named as vulnerability witnesses; passing them reproduces the old flaw, it does not validate the protocol. They have not been compiled or run here. This is containment plus executable witness authoring, not the complete cryptographic remediation.

## Protocol-v3 direction

The exact wire design needs to be selected against the real key-establishment model. The following are mandatory requirements, not yet an implemented protocol.

### 1. Define the key domain

Use independent per-sender secret material if peers must authenticate which member originated a message. Bind the full, canonical sender identity and protocol identifier into the key schedule. HKDF-SHA-256 can derive separated subkeys from established secret input; public identity/context alone is not key material and must never be described as secret entropy.

A shared group secret may provide group confidentiality and group membership authentication, but it cannot distinguish one legitimate group member from another. If that is an accepted trust model, state it explicitly. Otherwise establish sender-specific keys or use an appropriate signature/credential scheme for origin authentication.

### 2. Make restarts create a new cryptographic namespace

Prefer a fresh, authenticated key epoch with enough entropy (for example, a 128-bit session/boot identifier) and derive the effective sender/session key using a reviewed KDF. Authenticate the epoch in the packet. A new random epoch is still probabilistic; with a sufficiently large namespace and suitable key derivation, collision probability becomes negligible, while persisted state or fresh independent session keys can provide additional assurance.

A key epoch must not be accepted solely because the packet claims it. Define how a sender is authorized to advance epochs, how concurrent starts are handled, and how rollback, cloned images and snapshot restoration are treated.

### 3. Use a monotonic counter within each effective key

With a per-sender/per-epoch key, reserve a deterministic nonce/counter format and ensure each counter value is used at most once under that key. For example, a reviewed format could use a fixed domain prefix plus a 64-bit counter; packet type can be included in a separately derived subkey or in non-overlapping nonce bits. Persist/reserve counters safely if the same key survives restart. Rotate or fail closed *before* a counter wraps. The final 96-bit encoding, field byte order and domain-separation rules must be specified and versioned exactly.

Do not combine a fresh random nonce, a short epoch, a wrap-prone counter and a truncated identity without calculating the joint collision model. The acceptance criterion is uniqueness per effective key, not merely that each field appears different in ordinary logs.

### 4. Bind protocol context

Authenticate a canonical encoding of at least: protocol/version, full sender identity or its reviewed key identity, key epoch/version, payload type, sequence/counter, message length and any security-relevant timestamp/route metadata. Use a standard AEAD API and preserve full authentication tags. If intermediate relays are permitted to mutate hop TTL, distinguish end-to-end origin authentication from per-hop forwarding authentication instead of letting a shared group key blur the two properties.

### 5. Fail closed on ambiguous state

If durable nonce state is corrupt, rolled back, absent when required, or unavailable, the sender must not continue encrypting under the possibly reused key. Use a new authorized key epoch or stop sending encrypted traffic. Never “recover” by choosing a smaller random field or resetting the counter without changing the effective key.

## Required adversarial test matrix

These are proposed acceptance tests; they have not been implemented or run by this review.

| ID | Stimulus | Required result |
|---|---|---|
| NONCE-01 | Same key, sender, type, epoch and sequence after simulated restart | Legacy counterexample reproduces; v3 refuses reuse or uses a provably distinct key namespace |
| NONCE-02 | Two source IDs share the first six bytes but differ in the last two | `legacy_nonce_source_id_truncation_reveals_plaintext_xor` is the deterministic legacy witness; CI pending. V3 must prevent the collision under every supported key-sharing model |
| NONCE-03 | Packet-level sequence reaches its maximum | Sender rotates key or fails closed before counter reuse |
| NONCE-04 | Power loss after nonce reservation but before send, and after send but before durable commit | Restart cannot reuse an already-reserved nonce |
| NONCE-05 | Snapshot rollback / cloned sender state | Both instances cannot encrypt under the same key/counter namespace |
| NONCE-06 | Two senders start concurrently with the same group key | Nonce namespaces remain disjoint, or configuration is rejected when it cannot guarantee this |
| NONCE-07 | Key rotation, epoch transition and grace period | No key/nonce pair repeats; old keys have explicit receive-only expiry |
| NONCE-08 | Unknown epoch, old epoch, skipped epoch and downgrade packet | Reject unless the protocol's authenticated transition rule explicitly permits it |
| NONCE-09 | Valid ciphertext and every one-field tamper (type, identity, epoch, sequence, length) | Untampered input decrypts; all mutations fail authentication |
| NONCE-10 | Wrong key and cross-sender identity | Decryption fails; identity claims are checked against the configured key binding |
| NONCE-11 | All supported payload types at counter zero | No nonce collision under one effective key |
| NONCE-12 | Sequence serialization and byte-order boundary values | Canonical test vectors agree across independent implementations |
| NONCE-13 | Fragment sequences differ by 65,536 for the same source/key/index | `legacy_fragment_nonce_reuses_after_16_bit_sequence_wrap` documents the deterministic legacy collision; v3 must use a unique per-key fragment nonce namespace |
| NONCE-14 | Fragment sender restarts with sequence reset and same key | No fragment nonce is reused; current epoch field must not be assumed to protect fragments |
| NONCE-15 | Maximum-size 222-byte LoRa frame with fragment AEAD enabled | Detect that the legacy wrapper expands the frame to 250 bytes; v3 either fits within the declared MTU with versioned fragmentation or rejects/deprecates this mode |

Tests must include deterministic witnesses, not only random statistical sampling. Property/model-based tests should generate restart, key rotation, counter reservation, rollback, multi-sender and malformed-packet traces. Report a digest of the frozen test corpus and retain minimized failing traces. A deterministic source-truncation witness has now been added in `src/swarm/mesh/tests.rs` as `legacy_nonce_source_id_truncation_reveals_plaintext_xor`; it intentionally demonstrates the legacy flaw and is not a security-passing test. It has not yet been compiled or executed in this environment.

## Benchmark and release gates

Measure correctness first, then cost. The following are proposed targets, not achieved numbers:

- **Nonce uniqueness:** zero duplicate `(effective-key-id, nonce)` pairs in the complete deterministic state-machine suite; each seeded truncation/epoch/wrap defect must be detected.
- **Valid traffic preservation:** 100% of canonical positive vectors decrypt and authenticate under the intended configuration.
- **Mutation rejection:** 100% of the pre-registered field-tamper and downgrade vectors reject.
- **Cross-implementation agreement:** 100% agreement on packet and KDF known-answer vectors.
- **Recovery:** zero nonce reuse across simulated crash/restart, rollback and key-rotation sequences.
- **Operational costs:** packet overhead in bytes, p50/p95/p99 encrypt/decrypt latency, throughput, memory for durable state, flash writes per message/session, restart recovery time and behavior under storage failure.
- **Adversarial resource use:** bounded allocations and bounded per-peer state when packets contain malformed or unrecognized epochs/identities.

Compare with the protocol requirements in RFC 8439/RFC 5116 and with a reviewed AEAD implementation using the same packet corpus and environment. A faster implementation does not compensate for a single known nonce-reuse path. No “beats SOTA” claim is appropriate until the uniqueness property and key-establishment assumptions are independently validated.

## Exit criteria

1. An ADR pins the v3 wire format, key derivation, sender identity model, nonce/counter allocation and restart behavior.
2. A reference implementation and independently computed known-answer vectors are checked into the repository.
3. The negative controls above reproduce the legacy failures and are rejected by v3; all positive controls remain green.
4. A state-machine/property test and malformed-input/fuzz campaign are reproducible with source revision, toolchain, seed and corpus digests recorded.
5. An independent security reviewer signs off on the threat model and protocol composition. Do not infer approval from ordinary CI, a proof of serialization correctness, or success on the known-break suite.
