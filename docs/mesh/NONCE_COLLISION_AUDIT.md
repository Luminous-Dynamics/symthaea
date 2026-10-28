# Mesh nonce collision audit — exact runtime counterexample

**Status:** Source-level audit and deterministic reproduction. Not a runtime exploit report, key-recovery claim, or evidence that a particular deployed device emitted this vector.

## Summary

The current source has two 96-bit nonce encodings used with ChaCha20-Poly1305:

- Packet encryption, `build_nonce` in `src/swarm/mesh/mod.rs`:
  `source_id[0..6] || payload_type || epoch || sequence_le32`
- Fragment encryption, `encrypt_fragment` in the same file:
  `source_id[0..8] || thought_id_le16 || fragment_index || 0`

`DualLayerMesh::send` can encrypt the compressed whole-packet envelope, then encrypt each LoRa fragment with the same configured 32-byte key. The layered transform mismatch is tracked separately in [#8612](https://github.com/Luminous-Dynamics/symthaea/issues/8612).

## Exact collision vector

| Field | Value |
|---|---|
| `source_id` | `01 02 03 04 05 06 02 07` |
| `payload_type` | `0x02` |
| `epoch` | `0x07` |
| `sequence` | `0x00051234` |
| `thought_id` | `sequence as u16 = 0x1234` |
| `fragment_index` | `0x05` |

Both nonce layouts yield:

`01 02 03 04 05 06 02 07 34 12 05 00`

This is an exact arithmetic overlap, not just a birthday-bound possibility. Whether a specific deployed identity/epoch configuration reaches this tuple is configuration-dependent; the source proves these nonce namespaces are not disjoint by construction. If both invocations occur with the same effective key and tuple, the pair (key, nonce) repeats.

The test named `legacy_packet_fragment_nonce_overlap_negative_control` reproduces this through the actual packet and fragment encryption functions. A green result means the negative control detected the known legacy overlap. It is **not** a security pass.

## Independent repeat paths

### Random 8-bit packet epoch

The packet nonce uses a random one-byte epoch. Conditional on the same source, type, key and sequence recurring across restart, the epoch matches for one restart pair with probability 1/256 = 0.390625%. If the same key/sequence conditions recur across multiple starts, the probability of any collision among independent uniform epochs follows the birthday bound:

| Starts sharing the same key and recurring sequence | Probability of any epoch collision |
|---:|---:|
| 5 | 3.8531% |
| 10 | 16.3055% |
| 16 | 38.0292% |
| 20 | 53.3167% |
| 32 | 86.7643% |

These are calculations, not fleet observations. A larger random epoch reduces probability but cannot establish strict uniqueness and does not defeat cloned PRNG state.

### 16-bit fragment thought ID

`WisdomPacket::thought_id()` uses the low 16 bits of `sequence`. Fragment nonces contain this value but no epoch or upper sequence bits. With a long-lived fragment key, the same (source, thought ID, fragment index) repeats every 65,536 sequence values: about 21.85 minutes at 50 messages/s or 18.20 hours at one message/s.

## Required remediation

1. Make whole-envelope and fragment encryption mutually exclusive unless a reviewed protocol profile defines transform order and key usage. See [#8612](https://github.com/Luminous-Dynamics/symthaea/issues/8612).
2. Use separate, domain-separated effective keys for packet, fragment, direction and protocol version, derived with an established KDF from authenticated session key material.
3. Solve within-domain uniqueness too: an effective key's nonce/sequence allocator must survive concurrency, crash/restart, snapshot/fork/clone, backup restore, rollback and wrap. Fresh authenticated session-key contexts or a genuinely durable rollback-resistant allocator are candidates; random epochs alone are not.
4. Instrument every AEAD call site with one test-only oracle over (effective-key fingerprint, nonce), without logging raw key material.
5. Include the exact vector, 8-bit epoch reuse, 16-bit thought-ID wrap, 32-bit packet-sequence wrap, source-ID truncation, rotation, VM/filesystem clone, process fork, restore/rollback, concurrent senders, and missing authority state as negative controls.
6. Fail closed before encryption/send whenever uniqueness cannot be established. Preserve whole-batch MTU preflight.

Tracked work: [#8660 exact packet/fragment collision](https://github.com/Luminous-Dynamics/symthaea/issues/8660), [#8534 clone-safe key/nonce runtime implementation](https://github.com/Luminous-Dynamics/symthaea/issues/8534), [#8612 secure-fragment V2](https://github.com/Luminous-Dynamics/symthaea/issues/8612).

## Standards reference

- [RFC 8439](https://www.rfc-editor.org/rfc/rfc8439.html) requires distinct nonces for ChaCha20-Poly1305 invocations under the same key.
- [RFC 5869 / HKDF](https://www.rfc-editor.org/rfc/rfc5869.html) provides an established key-derivation building block. Domain separation alone does not establish fresh session state or protect against rollback.
- [RFC 8931](https://www.rfc-editor.org/rfc/rfc8931.html) is a standards-track selective-recovery reference for constrained multi-hop networks; it does not replace cryptographic lifecycle requirements.

## Assurance boundary

This is source-layout reasoning plus a deterministic test witness. It does not prove that a deployed configuration emitted the exact tuple, execute a full Rust build, or establish production exploitability. The next step is an exact-head uniqueness oracle plus runtime clone/restart/rollback fault injection and independent review.
