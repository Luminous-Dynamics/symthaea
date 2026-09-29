# EPF-010 retrieval receipt vectors

These vectors define the byte-level contract for canonical retrieval receipts.

## Encoding

- Hash: SHA-256.
- Domain separator: `epistemic-retrieval-receipt:v1\\0`.
- Strings are UTF-8 and encoded as a 4-byte unsigned big-endian byte length followed by raw bytes.
- Optional frontier references use a one-byte presence tag (0/1), followed by the length-prefixed string when present.
- `mode` is a fixed one-byte tag: Historical=0, Live=1.
- Collections that are semantically sets are sorted and deduplicated before encoding.
- Representation bindings are sorted by identity then representation digest.
- Exclusions are sorted by canonical identity then reason tag.
- No JSON serialization, field ordering, platform endianness, or Rust enum representation participates in the digest.

The encoding is intentionally a narrow typed binary contract rather than a general-purpose JSON/RDF canonicalization format. RFC 8785 establishes the need for invariant representations for cryptographic hashing/signing and explicitly calls out overflow/input sanity checks. W3C Data Integrity likewise treats canonicalization correctness as security-critical.

## Independent verification

Run `python3 scripts/verify_epf010_vectors.py` from this crate directory. The verifier uses only Python's standard library and the vectors in `vectors/epf-010.json`.

## Security boundary

A matching digest establishes byte-level integrity under this encoding. It does **not** establish producer authentication, truth, evidence strength, corroboration, currentness, or authority. Producer authentication requires a separate trust/key/revocation policy.
