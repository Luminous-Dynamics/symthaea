# REA APK Signing Block structural harness

This isolated Rust harness locates and inventories the outer Android APK Signing Block. It is not a signature verifier.

## Exact structural contract

Android specifies a little-endian outer container immediately before the ZIP Central Directory:

- uint64 block size excluding the leading uint64 field;
- an ordered sequence of uint64-length-prefixed ID/value pairs;
- a trailing uint64 equal to the leading size;
- the 16-byte ASCII magic `APK Sig Block 42`.

The scanner first validates a unique EOCD candidate, classic single-disk ZIP layout, bounded central-directory count/size, exact central-directory-to-EOCD adjacency and no trailing data. It then validates the signing block's size fields and every pair boundary with checked arithmetic.

Recognized scheme-presence IDs:
- v2: `0x7109871a`
- v3: `0xf05368c0`
- v3.2 hybrid PQC: `0x70e1c89f`

Every pair remains an ordered occurrence. Duplicate IDs are not collapsed. Unknown IDs remain visible as opaque observations. For each value, the report stores absolute offset, length and SHA-256, without parsing signer contents.

## Claim ceiling

A reported v2/v3/v3.2 ID means only that an outer ID/value pair with that identifier was structurally located. It does not establish that the inner signer block is valid or even fully parseable. No certificate, signature, trust anchor, proof-of-rotation, SDK range, PQC algorithm or signer identity is validated here.

The scanner does not verify APK v1/JAR signatures. Android v4 is a separate companion `.apk.idsig` file and is not embedded in the APK Signing Block. ZIP64 is explicitly Unsupported in this first implementation.

The source is opened once, hashed before/after the scan and never reopened by path. This detects persistent modifications but is not a claim that a malicious writer cannot mutate and restore bytes between observations.

## Build and test

```bash
cargo fmt --manifest-path tools/rea-apk-signing-block-harness/Cargo.toml -- --check
cargo test --manifest-path tools/rea-apk-signing-block-harness/Cargo.toml --locked
cargo clippy --manifest-path tools/rea-apk-signing-block-harness/Cargo.toml --locked --all-targets -- -D warnings
```

The harness includes generated golden outer containers and malformed-length, duplicate-ID, unknown-ID, EOCD-ambiguity, ZIP64-abstention and byte-budget controls.
