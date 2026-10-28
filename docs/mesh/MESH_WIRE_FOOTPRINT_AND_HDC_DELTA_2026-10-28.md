# Wire-cost correction and representation-first HDC delta frontier — 28 October 2026

**Evidence class:** Source audit, deterministic synthetic Hamming-distance fixtures, and analytical SX127x airtime. This is not a Rust test receipt, real Symthaea workload result, or hardware measurement.

## 1. Important correction to the deadline benchmark

The earlier event-time model used a raw 2,048-byte `BinaryHV` passed to the fragmenter. The production `DualLayerMesh::send` path serializes the full `WisdomPacket` first (`WISDOM_PACKET_SIZE = 2,104`), wraps it in a one-byte envelope tag, and may then apply packet AEAD (12-byte nonce + 16-byte tag) before fragmentation. The old deadline CSV is therefore explicitly a **raw-HV-only baseline**, not the complete current WisdomPacket wire footprint.

The Rust test module on [PR #8635](https://github.com/Luminous-Dynamics/symthaea/pull/8635) now checks all three sizes with the actual repository fragmenter:
- Raw BinaryHV only: 2,048 bytes; final data frame 130 bytes including the 8-byte fragment header.
- Full uncompressed packet envelope: 2,105 bytes; final data frame 187 bytes.
- Packet-AEAD envelope: 2,133 bytes; final data frame 215 bytes.

All fit in ten data frames plus an XOR parity frame at 222-byte MTU, but airtime is different. Under the frozen SF/BW/CR/preamble/header/CRC assumptions:

| Profile | SF7 data | SF7 + XOR | SF9 data | SF9 + XOR | SF12 data | SF12 + XOR |
|---|---:|---:|---:|---:|---:|---:|
| Raw BinaryHV only (2,048 B) | 3.351 s | 3.699 s | 10.660 s | 11.767 s | 77.251 s | 85.287 s |
| Full uncompressed envelope (2,105 B) | 3.433 s | 3.781 s | 10.906 s | 12.013 s | 79.217 s | 87.253 s |
| Packet-AEAD envelope (2,133 B) | 3.474 s | 3.822 s | 11.028 s | 12.135 s | 80.036 s | 88.072 s |

These are analytical PHY sums over data/parity frames, not completion latency or measured radio time. ACK, retries, receive energy, duty-cycle scheduling, collision and queueing are excluded. Compression can reduce wire size when the actual packet is compressible; the table is the uncompressed/ineffective-compression baseline.

Tracking issue: [#8710 — wire cost accounting](https://github.com/Luminous-Dynamics/symthaea/issues/8710).

## 2. Paradigm shift: optimize the representation, not only retransmission

Before spending all effort on more FEC/retries, test whether adjacent HDC vectors can be transmitted as a full snapshot or as an authenticated base-bound delta. This should only be considered if real successive `WisdomPacket.wisdom` values have sufficiently small Hamming distances. Independent random-like BinaryHVs are not automatically compressible; the gain must be measured from a representative corpus.

The synthetic proposal uses sorted changed-bit indices encoded as unsigned-varint gaps. Its fixed wire budget is 95 bytes: one version/type byte, the current 56-byte metadata/auth field, a 4-byte base sequence, a full 32-byte base digest and a 2-byte changed-bit count, plus encoded gaps. This is only a cost model, **not a deployable wire format**. The current fixed-size `WisdomPacket` decoder does not accept this variable-length payload.

### Controlled Hamming-distance frontier

The independent Python reference tested 16 changed-bit rates from 0% through 100%, with 100 deterministic seeds per rate. It verifies that varint gap decoding recovers the original ordered bit-index list and that applying the decoded delta to the base reproduces the exact target bytes.

| Changed-bit distance | Mean proposed delta size | SF7 + XOR airtime | Savings vs full uncompressed envelope |
|---:|---:|---:|---:|
| 0% | 95 B | 0.523 s | 86.2% |
| 1% | 304 B | 0.866 s | 77.1% |
| 5% | 915 B | 1.865 s | 50.7% |
| 8% | 1,406 B | 2.654 s | 29.8% |
| 10% | 1,733 B | 3.203 s | 15.3% |
| 12.5% | 2,143 B | 3.874 s | -2.4% |
| 15% | 2,553 B | 4.509 s | -19.2% |
| 20% | 3,372 B | 5.851 s | -54.7% |
| 50% | 8,287 B | 13.855 s | -266.4% |
| 100% | 16,479 B | 27.218 s | -619.8% |

The rough break-even is around 11–12.5% changed bits for this specific format, this header budget, this FEC behavior and this PHY profile. It is not a universal threshold. The delta must also be compared against the packet-AEAD full profile and charge the authentication/replay and cache lifecycle overhead of the new format.

### LZ4 sanity check

Across 100 deterministic synthetic packets with 56-byte metadata and a random-like 2-KiB hypervector, the Python LZ4 reference selected compressed framing in 0/100 cases; mean stored-compressed size was 2,118 bytes, larger than the 2,104-byte raw packet, so the fallback was appropriate. This is not byte-for-byte validation against Rust `lz4_flex`, and the synthetic vectors are not a substitute for real HDC traces. It does show why random-like vector compressibility must not be assumed.

The local independent experiment and output are available as task artifacts: `mesh_hdc_delta_frontier.py` and `mesh_hdc_delta_frontier.csv`.

Tracking issue: [#8712 — qualify representation-first BinaryHV delta coding](https://github.com/Luminous-Dynamics/symthaea/issues/8712).

## 3. Required gates before implementation

1. Collect a representative, privacy-safe corpus of adjacent actual BinaryHV values; report Hamming-distance distributions, current LZ4 ratios and exact source SHA/config receipts.
2. Compare full snapshots, current LZ4, sparse bit-index deltas, gap-coded deltas and periodic full refresh on held-out workload traces and identical network seeds.
3. Bind sender/session, base sequence, full base digest, protocol version, target metadata and authenticated delta semantics. Missing/stale/wrong base must reject before any state mutation and trigger a full-snapshot request.
4. Test lost, reordered, duplicated, stale, replayed and rollback deltas; sender restart, cache eviction and key rotation; invalid/truncated/overflowing varints; bounded memory and reassembly timeout.
5. Select delta only when its total cost is lower after headers, auth/AEAD, parity, ACKs, retries, deadline misses, memory retained and full-refresh recovery.
6. Keep the raw-HV event model labelled separately from full envelope/packet-AEAD results. Do not claim state-of-the-art performance from synthetic distances.

## Sources and implementation context

- Current mesh serialization/compression/encryption order is in `src/swarm/mesh/dual_layer.rs` and `src/swarm/mesh/mod.rs`.
- The analytical time-on-air equation uses the Semtech SX1276/77/78/79 datasheet, section 4.1.1.7: [datasheet](https://cdn.sparkfun.com/assets/7/7/3/2/2/SX1276_Datasheet.pdf).
- [RFC 8931](https://www.rfc-editor.org/rfc/rfc8931.html) is a useful constrained-link selective recovery comparator, not a wire-compatible replacement for this raw-LoRa profile.

## Qualification boundary

The added Rust test source has not been run locally because this environment has no Rust toolchain. The earlier exact-head CI/Showroom/benchmark/governance workflows were skipped, not passed. Keep PR #8635 in draft until relevant exact-head tests, format, Clippy, and review have real passing receipts. The analytical/Synthetic outputs are research aids, not production performance claims.