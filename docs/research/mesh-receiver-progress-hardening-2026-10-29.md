# Legacy mesh receiver hardening — 2026-10-29

## Status

Experimental fix branch: `fix/mesh-receiver-progress-and-fragment-shape-20261029`.
Issue: [#8852 — MESH-RECEIVER-PROGRESS-001](https://github.com/Luminous-Dynamics/symthaea/issues/8852).

**The Rust tests are authored but not executed.** The current environment has no cargo, rustc or rustfmt. This is source work plus regression tests, not a validated build or security certification.

## Findings

### 1. Duplicate/rejected frames refreshed activity time

MeshReceiver::receive_fragment previously updated `last_fragment_at` and `last_fragment_index` before duplicate detection and before reassembly admission. Replaying a valid CRC fragment—or delivering metadata that the assembler would reject—could refresh the inactivity deadline without contributing a new shard.

The receiver now refreshes time and progress index only after the assembler confirms the shard index is stored. The tests inspect the timestamp directly after a duplicate, and also after an individually CRC-valid fragment advertises a conflicting total count.

It also applies a configurable encoded-envelope size cap (default `WISDOM_PACKET_SIZE + 30` = 2,134 bytes) and rejects a valid-CRC but implausibly large advertised fragment count before allocating pending state. The +30 budget covers a compression prefix, optional key-version prefix, 12-byte nonce and 16-byte AEAD tag. The `fragments_oversize` metric separates these events from CRC corruption.

### 2. Legacy parser accepted impossible FEC metadata

LoRaFragment::from_bytes previously accepted the FEC flag at any fragment index, allowed a final fragment without FEC, and did not require the parity payload to be full-size. FragmentAssembler::feed did not compare total_fragments against the initial assembly and could mark an index received while leaving its data slot empty. Existing expect calls could panic during assembly.

Parser and assembler now reject unknown flags, impossible wire lengths (including trailing bytes), total counts below two, FEC/index inconsistencies, zero-length data, short parity, total-count mismatches, checksum failures, and impossible payload lengths before changing assembly state. Assembly uses checked arithmetic and Option paths instead of expect for untrusted reassembly state. Output allocation is bounded by the maximum bytes represented by the configured shard count.

### 3. Exact encoded length is absent from the legacy header

DualLayerMesh::send compresses/encapsulates the packet before fragmenting it. The legacy fragment header contains index/count/payload_len but does not bind the total encoded byte length. MeshReceiver's nominal WisdomPacket size therefore cannot reliably determine the length of a recovered final data fragment.

The revised policy is deliberately conservative:

- A final data fragment may be reconstructed by FragmentAssembler::new only when the caller contract guarantees the supplied expected_payload_size is exact.
- MeshReceiver and the compressed/enveloped loopback reassembler now call new_without_final_fragment_recovery. They still recover one missing non-final data fragment using XOR.
- All observed data fragments can still be assembled using their actual payload_len values, including variable-length compressed envelopes.
- If the final data shard is absent and legacy metadata cannot establish its true length, the receiver does not report the assembly as complete; it waits for the actual final data shard or expires the incomplete stream. It must not guess the tail length.

This is a correctness-first compatibility tradeoff until the versioned length-bound V2/V3 protocol is reviewed, authenticated and integrated. It can reduce FEC recovery for one particular loss position in the legacy compressed path; that is preferable to silent truncation/panic.

## Regression tests authored

- Reject FEC flag on a non-final index with a recomputed valid CRC.
- Reject final fragment without FEC with a recomputed valid CRC.
- Reject conflicting total count and misplaced FEC directly in FragmentAssembler before state advances.
- Accept a fully received variable-length payload even when the nominal expected size is stale; bound allocation even for usize::MAX nominal size.
- Exact-size mode rejects payload-size/count mismatches and checked-size overflow before accepting a shard, and verifies reconstructed output equals the declared size.
- Reject inferred final-tail recovery when the nominal length is inconsistent; return None instead of panicking.
- Preserve normal full reassembly, middle-shard recovery, last-shard recovery for callers with a truly exact payload size, lost-parity behavior, and legal reordering.
- Ensure duplicate and CRC-valid but count-mismatched receiver frames do not refresh last_fragment_at or last_fragment_index.
- Ensure a valid-CRC frame advertising 255 fragments is rejected before any pending state is allocated under the default packet-size cap.
- Default max encoded envelope is 2,134 bytes: 10 data shards plus parity are admitted; a valid-CRC 255-shard offer is rejected before allocation.
- Ensure MeshReceiver does not claim success when only the legacy final data fragment is missing, and completes if that actual shard later arrives.

## Validation boundary

For the default maximum encoded size of 2,134 bytes and the legacy 214-byte shard capacity, the receiver now admits at most 10 data shards plus parity (11 total) rather than a peer-selected 254 data shards. Raw payload storage for data slots plus parity is therefore capped at 2,354 bytes per pending stream instead of 54,570 bytes; across 64 pending streams this is 150,656 bytes instead of 3,492,480 bytes, roughly 23.2× less payload storage (ignoring allocator and `Option` metadata). The ceiling is configurable for deployments with larger payloads. The source changes were reviewed for line lengths; no modified Rust source line exceeds 100 characters. This is not a rustfmt result. Local tooling check finds no cargo, rustc or rustfmt. Draft PR [#8856](https://github.com/Luminous-Dynamics/symthaea/pull/8856) is open. At the last exact-head check, hosted workflows concluded `skipped`, not passed; the Rust tests remain unexecuted. No test PASS is claimed.

## Follow-on

The longer-term fix is not to permanently lose final-shard recovery. It is to adopt the reviewed V3 length-bound wire identity and authenticated framing path after proving clone-safe session-generation allocation, direction-separated traffic keys, atomic non-repeating message sequences, replay defense and strict MTU preflight. See [#8836](https://github.com/Luminous-Dynamics/symthaea/pull/8836), [#8534](https://github.com/Luminous-Dynamics/symthaea/issues/8534), [#8612](https://github.com/Luminous-Dynamics/symthaea/issues/8612), and [#8820](https://github.com/Luminous-Dynamics/symthaea/issues/8820).
