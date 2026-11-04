# REA member-verification receipt harness

This isolated Rust harness validates and canonicalizes the versioned member-verification receipt defined at:

`docs/research/rea-zip-content-verification-receipt-v0.schema.json`

It is intentionally outside the root Cargo workspace and is not yet a production service or receipt writer.

## Contract

- Structural validation uses JSON Schema Draft 2020-12 via `jsonschema 0.60.2`; remote reference resolution is disabled.
- Cross-field semantic validation additionally requires a verified result to bind a replayable whole-archive SHA-256, qualified source binding, held-descriptor and selected-path checks, post-attempt source revalidation, exact declared/observed output size, both decoder and independent CRC results, metadata parity, and supported unencrypted regular-file compression.
- Verified outcomes use `asserted_claim=member_content_integrity_only`; rejected, unsupported, source-changed, limit-exceeded and abstained outcomes must use `asserted_claim=none`. Every outcome retains an explicit claim ceiling. Producer authenticity, producer identity, APK/IPA signature validity, archive completeness, installability, runtime behaviour, MASVS control pass, semantic truth and absence of a malicious writer are never inferred from member SHA-256/CRC alone.
- Raw JSON parsing rejects duplicate object property names before constructing `serde_json::Value`. A conventional Value parse can collapse duplicate keys and hide non-I-JSON input.
- A verified receipt also checks canonical standard-base64 filename bytes, local/data/archive offset ordering, archive-size and occurrence-count budgets, per-member/entry/aggregate output bounds, lowercase digest/CRC syntax, and a null failure disposition. Non-verified outcomes require a failure kind consistent with the outcome and cannot publish a completed payload digest or assert a successful integrity claim.
- JCS canonicalization uses `serde_json_canonicalizer 0.4.1`, pinned to the crate release that advertises RFC 8785 compatibility.
- Public receipt digest functions validate the schema and cross-field semantics first. The preferred raw-input path performs duplicate-key rejection, schema validation, semantic validation, JCS canonicalization and SHA-256 in that order; no digest is produced for an invalid receipt.
- Receipt digests are SHA-256 over JCS bytes. They identify the canonical receipt statement; they do not authenticate the producer or elevate the underlying observation.

## Exact integer representation

RFC 8785 builds on I-JSON and ECMAScript/IEEE-754 number serialization. JSON numbers are therefore restricted to exactly representable, small bounded integers in this receipt. Potentially 64-bit values—archive sizes, offsets, occurrence indices, declared and observed lengths, and configured byte/count budgets—are encoded as canonical unsigned decimal strings:

`^(0|[1-9][0-9]*)$`

This preserves `u64::MAX` exactly across JavaScript-compatible implementations. Flags, compression-method codes and the bounded 1 MiB read buffer remain small JSON integers. Floating point, negative and out-of-range numbers are rejected before canonicalization.

## Duplicate names and source replay

- Receipt occurrence identity binds the central-directory index and exact raw filename bytes (base64); decoded filename is presentation-only.
- A verified receipt requires an archive SHA-256 so an independent replay can bind itself to exact source bytes. Filesystem identity is recorded separately; it is not an archive digest or an immutable snapshot.
- An attestation envelope is a separate optional reference. A signature authenticates the issuer's statement under a trust policy; it does not establish the truth of the measured artifact.

## Build and test

```bash
cargo fmt --manifest-path tools/rea-receipt-harness/Cargo.toml -- --check
cargo test --manifest-path tools/rea-receipt-harness/Cargo.toml --locked
cargo clippy --manifest-path tools/rea-receipt-harness/Cargo.toml --locked --all-targets -- -D warnings
```

The test suite includes schema meta-validation, verified/rejected receipt outcomes, CRC/size/source/offset/archive-budget/claim-ceiling negative controls, canonical-base64 checks, duplicate raw-property rejection at root and nested levels, decimal-string precision through `u64::MAX`, and RFC 8785 Section 3.2.2 number/string plus UTF-16 property-order vectors.

A green harness run is evidence only for this receipt schema, implementation revision, dependency lock and fixtures. An independent replay tool against original content-addressed archive bytes remains a separate gate.
