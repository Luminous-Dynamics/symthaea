# Legacy mesh reassembly hardening — 2026-10-29

## Status

Research patch on a draft branch. The local environment has Python but no cargo, rustc, or rustfmt. Rust tests are authored but not run; exact-head workflows report skipped, not passed. Nothing is merged or claimed production-validated.

Related issue: [#8852 MESH-RECEIVER-PROGRESS-001](https://github.com/Luminous-Dynamics/symthaea/issues/8852).

## Source-audit findings

1. The legacy fragment parser accepted impossible FEC/index combinations as long as the frame checksum matched. The assembler did not independently require the fragment count to match its configured stream, and it could treat an FEC frame at a data index as satisfying the received-index bitmap.
2. The assembler's previous `expect` paths and unchecked inferred tail length could panic for internally inconsistent count/size metadata.
3. A fixed `expected_payload_size` cannot safely infer the true encoded tail length when a final data shard is missing from a compressed/enveloped packet. The raw legacy frame does not carry exact encoded length.
4. The receiver refreshed inactivity time before duplicate, reordering and assembler-admission checks. A peer repeating a valid fragment could keep an incomplete assembly alive without delivering new information.
5. The receiver must bound peer-controlled fragment-count metadata before allocating pending assembler vectors. The branch includes a configured maximum encoded-envelope bound and a regression test for a CRC-valid oversized count.

These are source-derived issues; Rust runtime reproduction is not yet available in this environment.

## Changes on this branch

- The legacy parser fails closed on unknown flag bits, total counts below data-plus-parity minimum, zero payloads, parity not located at the final index, and parity whose shard is not full size.
- `FragmentAssembler::feed` checks stream identity, exact total count, checksum, flag set, FEC/index relation and shard length before changing state.
- Exact-size mode now checks that encoded size and data-fragment count agree, validates the final data-shard length, and verifies the assembled output length equals the configured exact size.
- A separate `new_without_final_fragment_recovery` compatibility mode uses observed shard lengths for variable-size legacy envelopes and declines recovery when the missing shard is the final data shard, whose length is not carried on wire. Missing non-final shards remain recoverable when XOR parity and the final data shard are present.
- Reconstruction replaces panic-prone assumptions with checked arithmetic/optional lookups. Reservation is capped by the maximum payload bytes the configured shards can represent, not an untrusted nominal payload-size hint.
- `MeshReceiver` uses the variable-length compatibility mode, bounds total fragment count by the configured maximum encoded envelope before allocation, and refreshes `last_fragment_at` / `last_fragment_index` only after a new fragment is admitted.
- The loopback reassembly helper uses the compatibility mode too, so compressed/enveloped payloads do not silently acquire an exact-size contract they cannot satisfy.

## Tests authored (not executed)

- Parser rejects FEC on a non-final index and rejects a final index without FEC even when the attacker recomputes a valid CRC.
- Assembler rejects a count mismatch or FEC/index mismatch without marking a fragment received.
- Exact-size mode rejects mismatched payload size/count and overflowing exact-size metadata.
- Variable-length mode reconstructs all present data shards even when the nominal payload-size hint is stale or extremely large, with allocation bounded by shard capacity.
- Variable-length mode fails closed when the final data shard is missing; valid final data arriving later lets assembly complete.
- Receiver duplicate and malformed-count tests backdate the inactivity timestamp deterministically and assert rejected frames do not refresh it.
- Receiver rejects a CRC-valid excessive fragment count before pending-assembly allocation.
- Existing positive controls cover round trips, middle/final loss recovery with exact-size mode, lost parity, duplicates, reverse order and capacity/expiry behavior.

## Explicit trade-off

The legacy wire format cannot distinguish the true byte length of a lost final shard from its XOR-padded buffer. This patch intentionally does not guess: final-shard recovery is disabled for variable-length legacy envelopes. The correct way to restore that recovery is a versioned length-bound frame protocol (the separate V3 prototype in [draft PR #8836](https://github.com/Luminous-Dynamics/symthaea/pull/8836)), subject to its own identity/key lifecycle and authentication gates.

## Acceptance gates

- Zero panics for malformed or inconsistent fragment metadata.
- Zero assembler-state changes for rejected index/count/FEC/profile metadata.
- Zero timeout refresh from duplicates/rejected fragments.
- Exact byte reconstruction for the declared valid corpus.
- No MTU relaxation, no cleartext fallback, and no assumption that CRC authenticates the sender.
- Exact-head rustfmt/check/test/Clippy/feature builds plus negative controls before merging. Skipped CI workflows do not satisfy this gate.

## Industry-facing benchmark

Compare the hardened legacy path and V3 path against matched fixed-XOR/selective-repair/multi-parity coding under identical seeded traces, deadline and maximum frame size. Report delivery-by-deadline, p95/p99 latency, analytical and measured airtime separately, measured or explicitly modeled energy per delivered byte, ACK cost, goodput/fairness, pending assemblies and pending byte-seconds. Test on held-out loss traces and then hardware-in-the-loop; do not call a simulator-only result better than SOTA.

Standards comparators: [RFC 8931 selective fragment recovery](https://www.rfc-editor.org/rfc/rfc8931.html), [RFC 9011 SCHC over LoRaWAN](https://www.rfc-editor.org/rfc/rfc9011.html). These are comparisons, not drop-in protocols for raw LoRa.