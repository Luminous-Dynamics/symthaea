# Alpha.7 Hash Roundtrip Fix

Alpha.7 fixes the fixture workflow failure where `packet.stable_hash64()` changed after a valid `.svmp` text roundtrip.

## Root cause

The `.svmp` prototype format serializes floating-point coefficients with a bounded textual precision for readability. Alpha.6 hashed raw in-memory `f32` bits, so the original packet and the parsed packet could represent the same persisted artifact while producing different hashes.

## Fix

`VisualMemoryPacket::stable_hash64()` now hashes the canonical `.svmp` text representation instead of raw `f32` bits. This makes the checksum an artifact hash rather than an in-memory-layout hash.

This remains non-cryptographic and should be used for regression fixtures, corpus indexing, and experiment manifests only.

## Verification

Run:

```bash
cargo test -p symthaea-visual-compression-probe
```

Expected result: all library tests and `tests/fixture_workflow.rs` pass.
