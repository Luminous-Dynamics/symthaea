# Bootstrap Survivability Reference

This dependency-free Rust reference now has two deliberately separate layers:

1. A fail-closed decision kernel for bootstrap and payload qualification.
2. A synthetic physical-bootstrap codec implementing the declared GF(256) [6,4,3] known-erasure profile.

## Codec responsibilities

- Canonical, versioned manifest serialization with bytewise-sorted object IDs.
- Explicit lengths, exact manifest byte length, and zero-padding rules.
- Systematic 4+2 shard encoding over GF(2^8), polynomial 0x11d.
- Recovery from any two known shard erasures (all 15 patterns have independent algebraic KAT coverage).
- Cross-check all surviving shards for inconsistency when redundancy remains.
- Reject fewer than four surviving shards, inconsistent surplus shards, malformed framing, noncanonical record order, truncation, trailing bytes, and nonzero padding.

The parity rows are `[1,1,1,1]` and `[1,2,3,4]`. This is a synthetic reference profile, not a standard or a qualified production media format.

## Cryptographic and physical trust boundary

The codec carries each object's expected SHA-256 but does not calculate or verify it. Callers must independently verify every reconstructed object's digest and provenance before use. The codec does not authenticate signatures, validate calibration, establish reader independence, or qualify physical media/retention.

This is **known-erasure decoding**, not unknown-error correction. If exactly four shards survive, a corrupt survivor can be indistinguishable from a different codeword. External digests remain mandatory even when all six shards are internally consistent.

## Run locally

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test
```

The assistant execution environment has no Cargo, rustc or rustfmt. The authored Rust tests have **not** been executed here. The independent finite-field oracle verified the KAT values and all 15 two-erasure combinations; CI or a local Rust toolchain must still compile and run the crate at the exact branch head.

## Parent contracts

- `BOOTSTRAP_SURVIVABILITY_001.md`: placement/fault assumptions and qualification semantics.
- `BOOTSTRAP_CODEC_PROFILE_001.md`: wire format, decoder contract, failure semantics and evidence ceiling.
- `fixtures/bootstrap_codec_kat_v0_1.json`: fixed bytes and erasure corpus.
- `fixtures/bootstrap_survivability_cases_v0_1.json`: higher-level control-plane fault cases.

Mycelix #5762 remains the canonical operational common-cause/dependency authority. Symtropy #7571 owns Recovery Capsule package and representation-information semantics.
