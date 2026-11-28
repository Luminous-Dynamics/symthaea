# MON-047 — Cross-language canonical claim encoding

**Status:** research contract and golden vectors; runtime implementation pending  
**Date:** 2026-11-28  
**Tracks:** MON-046; RFC 8785

## Objective

Make a recovery claim envelope produce the same canonical bytes and digest in independent Go and Rust implementations. This is a prerequisite for interoperable signatures, content-addressed evidence graphs and reproducible appraisal. It is not itself authentication, semantic validation or authorization.

## Normative profile

1. Canonicalize JSON using RFC 8785, the JSON Canonicalization Scheme (JCS).
2. Reject duplicate object member names before canonicalization. Do not silently keep first/last values.
3. Reject inputs outside the interoperable JSON/I-JSON and JCS number domain; do not round arbitrary-precision integers into IEEE-754 binary64 and claim identity preservation.
4. Preserve Unicode code points exactly. JCS does **not** perform NFC/NFD normalization. If a domain needs normalized identifiers, normalize and validate them in a separately versioned semantic layer before constructing the signed object.
5. Hash UTF-8 canonical bytes. Never hash pretty-printed JSON, transport framing, or a parsed/re-serialized value under an unspecified serializer.
6. Domain-separate claim digests using the exact prefix bytes `MON-046:CLAIM-ENVELOPE:v1\0` followed by canonical UTF-8 bytes. This is a MON profile decision, not an RFC 8785 requirement.
7. The envelope's profile/version are inside the canonicalized object. Changing canonicalization, domain prefix, or envelope semantics requires a new profile version.

## Digest definitions

`canonical_sha256 = SHA256(JCS(envelope))`

`claim_digest = SHA256(ASCII("MON-046:CLAIM-ENVELOPE:v1") || 0x00 || JCS(envelope))`

Use fixed byte concatenation as above; do not concatenate variable-length fields without framing. The canonical byte string is the source of truth for both hashes.

## Go/Rust conformance requirements

- Both implementations consume the same raw JSON fixture bytes.
- Both reject duplicate keys and out-of-profile numeric values consistently.
- Both emit byte-identical JCS UTF-8 output.
- Both recompute SHA-256 from emitted bytes; fixture-provided hashes are expected values only.
- Tests compare bytes, not merely parsed object equality.
- No implementation may use a host-language map iteration order or default JSON serializer as a substitute for JCS.
- Pin JCS library versions and record implementation/build provenance in the test report.
- Run the same corpus on little- and big-endian targets where supported; output must not vary.

## Security and semantic boundaries

Canonicalization answers only: "What exact JSON value is represented by these bytes under this profile?" It does not establish who produced the bytes, whether the source observation is true, whether the inventory is complete, whether a signer is trusted, whether a claim is fresh, or whether a production action is safe.

Keep the following predicates separate:
- canonical encoding validity;
- schema validity;
- domain semantic validity;
- artifact byte identity;
- signature validity;
- trust-root and revocation appraisal;
- scope completeness;
- recovery outcome qualification;
- relying-party admission.

A successful canonicalization test must never be reported as a successful trust or recovery test.

## Golden corpus

See `JCS-VECTORS-v1.json` for five positive known-answer vectors and `JCS-NEGATIVE-v1.json` for malformed/ambiguous cases. The corpus intentionally starts with simple, independently auditable examples. Expand it with the complete RFC 8785 Appendix B number vectors and Appendix E property sorting guidance before declaring full JCS conformance.

## Acceptance gates

1. Go and Rust pass every positive vector byte-for-byte.
2. Go and Rust reject every negative vector.
3. Mutations to key ordering, escaping, number formatting, Unicode handling or domain prefix are detected.
4. Repeated execution and process/platform changes do not alter outputs.
5. A second independent implementation reproduces all expected digests.
6. Reports state exact commit, compiler, dependency lock digest and vector corpus digest.

## Explicit non-claims

No Go/Rust runtime implementation, RFC-complete conformance, signature interoperability, trusted timestamp, trust bootstrap, durable nonce ledger, live recovery campaign, production authorization or SOTA result is claimed by this research artifact.
