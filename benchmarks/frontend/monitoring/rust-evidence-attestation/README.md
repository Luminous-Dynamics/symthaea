# Symthaea evidence-attestation preflight

Standalone Rust preflight for strict source-receipt parsing, RFC 8785 JCS and DSSE v1 Ed25519 verification.

This directory declares its own Cargo workspace to keep an uncompiled research dependency set out of the main Symthaea workspace until a Rust toolchain can regenerate and validate its lockfile. It is not yet a production crate or a dependency of Symthaea.

## Run

```sh
cargo test --manifest-path benchmarks/frontend/monitoring/rust-evidence-attestation/Cargo.toml
cargo clippy --manifest-path benchmarks/frontend/monitoring/rust-evidence-attestation/Cargo.toml --all-targets -- -D warnings
```

## Contract

- DSSE authenticates the exact serialized payload bytes and payload type through PAE. It does not canonicalize the signed payload.
- keyid is optional and unauthenticated. The verifier tries the trusted key set and does not allow a spoofed/missing hint to decide signer identity.
- Signature thresholds count distinct trusted public keys, not signature entries.
- Trust records separately bind issuer, tenant, source, validity interval and revocation time.
- Strict serde deserialization rejects duplicate object keys, including escaped-equivalent names, before values are collapsed into a generic JSON map.
- RFC 8785 JCS canonicalization is a separate function for semantic content digests. It does not replace DSSE exact-byte verification.
- Both standard and URL-safe Base64 encodings are accepted, in padded and unpadded forms, per the DSSE protocol.

## Dependencies and qualification

The preflight pins the same Ed25519-Dalek major/minor already used by the root workspace, plus `serde_json_canonicalizer 0.4.1` for current RFC 8785 serialization. The canonicalizer does not itself provide duplicate-key rejection; this crate's custom visitor adds that gate before canonicalization.

No Rust compiler/Cargo is available in the current execution environment. Source and tests are committed but remain **not compiled or executed**. The separate workspace is deliberate: it avoids silently changing root-workspace Cargo.lock or CI behavior before dependency resolution is verified.

No production root-of-trust, HSM, durable revocation feed, source-adapter attestation, live database, broker or persistent evidence ledger is implemented.
