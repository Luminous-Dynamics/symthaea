# Mesh fragment V3: stream identity and reassembly — 2026-10-29

## Status

**Research prototype / draft PR only. Not compiled or executed in Rust in this environment. Not connected to production sender/receiver. Not an authenticated transport.**

Branch: `feat/mesh-fragment-v3-stream-identity-20261029`
Related work:
- [#8820 MESH-STREAM-ID-001](https://github.com/Luminous-Dynamics/symthaea/issues/8820)
- [#8534 clone-safe key/nonce lifecycle](https://github.com/Luminous-Dynamics/symthaea/issues/8534)
- [#8612 authenticated fragment V2 protocol](https://github.com/Luminous-Dynamics/symthaea/issues/8612)
- [#8634 reliability–airtime–energy benchmark](https://github.com/Luminous-Dynamics/symthaea/issues/8634)
- [#8807 adaptive FEC and deadline-aware repair](https://github.com/Luminous-Dynamics/symthaea/issues/8807)

## Motivation

The isolated V2 prototype identifies a stream using a 16-bit `thought_id`, encoded length, fragment count and capacity profile. Two different same-length messages can reuse that identifier while stale fragments are still in flight. A receiver that validates only those metadata can combine individually valid fragments from separate messages and create a mixed payload.

V3 makes the wire stream identity explicit: a 64-bit `session_generation` plus a 32-bit `message_sequence`. The reassembler also requires the full 32-byte authenticated peer context supplied by the caller, so two peers cannot collide merely by choosing the same numeric pair. The caller must pass this context only after frame/source authentication; it must not derive it from an unverified source-ID field. The reassembler rejects a frame before state advances when peer context or any wire identity component differs. It also poisons the assembly if a duplicate fragment index arrives with conflicting payload bytes.

## Wire format

V3 uses a 22-byte header, with a V3 magic/version prefix and a version discriminator in the reserved-flag location so the hardened legacy parser rejects V3 instead of ambiguously decoding it:

| Offset | Size | Field |
|---:|---:|---|
| 0 | 2 | V3 magic/version = `D3 03` |
| 2 | 1 | Fragment index |
| 3 | 1 | Total count including XOR parity |
| 4 | 1 | Payload length |
| 5 | 1 | Flags: V3 marker `0x80`, parity `0x01`, AEAD size profile `0x04` |
| 6 | 2 | CRC-16/CCITT |
| 8 | 8 | Session generation, little-endian |
| 16 | 4 | Message sequence, little-endian |
| 20 | 2 | Exact encoded message length, little-endian |
| 22 | variable | Payload |

The codec offers an allocation-free `write_to(&mut [u8])` path. The vector serializer remains a convenience wrapper. The maximum source-fragment count remains 254 plus one parity shard.

## Identity lifecycle requirements

The header is a protocol binding, not an identity allocator. It does **not** fix a cloned or rolled-back sender on its own.

- The 32-byte peer context must be derived from authenticated peer/security state and kept separate per authenticated principal.
- `session_generation` must come from the reviewed clone-safe key/session-generation lifecycle; it must not be recycled under the same effective traffic key.
- `message_sequence` must not repeat inside one session generation. The sender must rotate to a genuinely fresh session/key generation before sequence wrap or rollback can cause reuse.
- Authenticated identity, flags, exact length and payload must be cryptographically bound before the receiver trusts a frame. CRC only detects accidental corruption and is forgeable.
- A missing or uncertain generation allocation must fail closed. Do not substitute a random restart label or a process-local counter and call it clone-safe.
- The AEAD-overhead profile below is only accounting; the codec performs no encryption and does not resolve the production packet/fragment transform-order issue.

## Independent model check and remaining lifecycle falsifier

A standalone Python wire/identity mirror (separate from the unexecuted Rust tests) validates the pinned CRC vector, rejects 1,776 individual wire-bit flips, and shows the identity predicate rejects mixed fragments when the generation, message sequence, or authenticated peer context differs.

The model also contains an explicit negative control: if the **same peer context + same generation + same sequence** is reused for two different 2,048-byte messages, it accepts 12 data fragments across the mixed streams and reports that the data-shard count is complete. This is expected: the framing codec cannot prevent identity reuse on its own. That counterexample is why the upstream generation/key allocator must be clone/rollback resistant under #8534 and must fail closed if uniqueness is uncertain. Python model is not Rust runtime evidence and performs no AEAD.

Local model:
- `/mnt/data/mesh_v3_stream_identity_model.py`, SHA-256 `dadaaf5f85f0690d680a556ef0c627ddfcb480d31958897115f1cd9a4a042844`
- `/mnt/data/mesh_v3_stream_identity_model_results.json`, SHA-256 `5c0cac22d84e4348c470178df9df985edf90f524954db0e8c35fc5f5d8012fc2`

## MTU and analytical airtime

Frozen model assumptions: 222-byte MTU; 28-byte external nonce/tag budget; bandwidth 125 kHz; coding rate 4/5; eight-symbol preamble; explicit header and PHY CRC; one XOR parity frame. Semtech-style time-on-air equations are used. No radios were measured.

V3 has 22 header bytes and therefore a budgeted shard payload of `222 - 22 - 28 = 172` bytes, compared with V2's `184` bytes.

| Encoded bytes | SF7 V2/V3 frames | SF7 V2/V3 airtime | Airtime delta | Wire-byte delta |
|---:|---:|---:|---:|---:|
| 1,800 | 11 / 12 | 3.776256 / 4.047872 s | +7.193% | +170 |
| 2,048 | 13 / 13 | 4.299008 / 4.508928 s | +4.883% | +144 |
| 2,105 | 13 / 14 | 4.380928 / 4.688384 s | +7.018% | +194 |
| 2,133 | 13 / 14 | 4.421888 / 4.729344 s | +6.953% | +194 |

The full CSV includes SF7/SF9/SF12 and the exact frame-length arithmetic. This is an analytical size/airtime estimate, not measured latency, delivery probability, battery energy or a comparison of implementations.

## Tests authored (not executed)

- Reconstruct boundary sizes through every single data-fragment erasure with parity available.
- Reconstruct from all data shards when the parity shard is lost.
- Verify a caller-owned output buffer matches Vec serialization and a short output is rejected without partial writes.
- Confirm external AEAD-size accounting never exceeds 222 bytes.
- Reject fragments from a different authenticated peer context, session generation or message sequence.
- Explicitly preserve the negative control: if the exact peer + generation + sequence is reused for two different messages, framing alone can mix them; this is an upstream allocator failure that #8534 must prevent.
- Accept sequence zero as a valid first message sequence while rejecting zero session generation.
- Poison the assembly on a conflicting duplicate.
- Reject V3 from the hardened legacy V1 parser.
- Reject every single-bit mutation of one representative frame.
- Pin a known checksum/wire-layout vector.
- Reject zero generation, empty payload and oversized input.

The Rust toolchain is unavailable in this environment; these tests have not run. CI/workflow skips are not passes. Before promotion, run exact-head Rust fmt/check/test/Clippy and feature builds, add independent negative controls, and review identity allocation and authenticated-context design.

## Next qualification plan

1. Execute the V3 Rust unit suite on the exact commit.
2. Add a mixed-message regression reproducing the V2 counterexample using the actual Rust codec/reassembler.
3. Specify the secure session-generation allocator jointly with #8534 and bind all header fields to AEAD associated data or an equivalent reviewed authenticated construction.
4. Pin a Rust Reed–Solomon backend and check known-answer parity vectors against the pure-Python GF(256) reference before replacing the benchmark's ideal-MDS policy.
5. Benchmark fixed XOR, multi-parity coding, selective repair and a held-out adaptive policy under the same channel seeds, actual ACK bytes, deadlines and MTU; eventually calibrate energy and airtime on the target radio.
6. Keep this codec out of production until message identity, nonce uniqueness, replay defense, symmetric transform order and exact final-frame MTU preflight all pass.

## Reproducible sizing artifact

The current local analytical script is `/mnt/data/mesh_v3_identity_budget.py` (SHA-256 `52b6b567027b9fffd43454e423db03e8be8c956be4126edef466ed9d8aa497ef`) and CSV is `/mnt/data/mesh_v3_identity_budget.csv` (SHA-256 `23c9158443748065f2faa568b69653febbe18610b0a7e5f129b949dc4e96f3bc`).
