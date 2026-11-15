# SYMPsi pinned Quicknet beacon verifier

**Status:** offline cryptographic verification component and published-vector tests. It is not a full experimental custody protocol, and does not establish that psi exists.

## What it verifies

This standalone Rust package uses the explicitly pinned `drandite-core = 0.1.0` API to verify a Quicknet BLS12-381 G1 unchained signature against an out-of-band pinned trust root:

- beacon ID: `quicknet`
- scheme: `bls-unchained-g1-rfc9380`
- chain hash: `52db9ba70e0cc0f6eaf7803dd07447a1f5477735fd3f661792ba94600c84e971`
- genesis Unix time: `1692803367`
- period: 3 seconds
- public key, genesis seed: pinned in `src/lib.rs`

It additionally requires the source's returned round to equal the requested round, rejects round zero, verifies the signature, and derives the scheduled Unix timestamp from the pinned round timeline using the crate's checked arithmetic.

## Reproducible known-answer test

The published Quicknet round-1000 example has:
- signature: `b44679b9a59af2ec876b1a6b1ad52ea9b1615fc3982b19576350f93447cb1125e342b73a8dd2bacbe47e4b6b63ed5e39`
- verified public randomness: `fe290beca10872ef2fb164d2aa4442de4566183ec51c56ff3cd603d930e54fdd`
- scheduled time: `1692806364`

The test suite checks the valid signature, modified signature rejection, requested/returned round mismatch, round-zero rejection, and trust-root mismatch for public key and beacon ID.

## Reproduce

    cargo fmt --manifest-path tools/sympsi-beacon-verifier/Cargo.toml --all -- --check
    cargo test --manifest-path tools/sympsi-beacon-verifier/Cargo.toml
    cargo clippy --manifest-path tools/sympsi-beacon-verifier/Cargo.toml --all-targets -- -D warnings

The CLI accepts a requested round and a text file containing exactly one 48-byte signature in hexadecimal (whitespace is ignored):

    cargo run --manifest-path tools/sympsi-beacon-verifier/Cargo.toml -- 1000 /path/to/signature.hex

The CLI has no HTTP client, does not fetch a beacon, and passes the requested round as the expected returned round only because it consumes a raw signature file rather than a relay response. Production adapters must preserve and compare the actual returned round before calling the verifier.

## Explicit non-claims

A valid signature establishes only that the bytes verify for the selected scheme and pinned public key/chain information. It does **not** establish:
- that the round was released on schedule or is fresh;
- that a signature was not published early by threshold signers;
- that the response commitment existed before the target could be known;
- that the relay, timestamp authority, or witness is independent;
- that a future round was selected before the response;
- that the beacon's public randomness is secret or a cryptographic key.

The crate deliberately performs no network I/O and reads no system clock. Do not route signature verification success into an AIA trial's valid disposition until SCI-PSI-003 also qualifies external response commitment timestamps, target round selection, freshness, witness independence, and abort/withholding policy.

## Dependency and provenance

The dependency is exact-version pinned for this isolated research component. Before production adoption, separately audit the crate's source, transitive dependency graph, license, maintainers and cryptographic implementation. A passing test vector is not a cryptographic audit.
