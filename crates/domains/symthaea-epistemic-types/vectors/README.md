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
- Projection-identity bindings are sorted by identity then projection-identity digest and bind the projection semantics exposed to reasoning, not just its representation bytes.
- Exclusions are sorted by canonical identity then reason tag.
- No JSON serialization, field ordering, platform endianness, or Rust enum representation participates in the digest.

The encoding is intentionally a narrow typed binary contract rather than a general-purpose JSON/RDF canonicalization format. RFC 8785 establishes the need for invariant representations for cryptographic hashing/signing and explicitly calls out overflow/input sanity checks. W3C Data Integrity likewise treats canonicalization correctness as security-critical.

## Independent verification

Run `python3 scripts/verify_epf010_vectors.py` from this crate directory. The verifier uses only Python's standard library and the vectors in `vectors/epf-010.json`.

## Security boundary

A matching digest establishes byte-level integrity under this encoding. It does **not** establish producer authentication, truth, evidence strength, corroboration, currentness, or authority. Producer authentication requires a separate trust/key/revocation policy.


## Adversarial vectors

`epf-010-negative.json` defines rejection cases that must remain invalid across implementations:

- duplicate selected identities;
- duplicate `(identity, representation_digest)` bindings;
- duplicate exclusions;
- representation bindings referring to an unselected identity;
- changed retrieval-profile versions with a stale receipt digest;
- duplicate retrieval-profile versions;
- empty retrieval-profile versions;
- whitespace-only retrieval-profile versions;
- altered canonical bytes.

Run `python3 scripts/verify_epf010_negative_vectors.py` to verify the negative fixtures independently. A conforming implementation should reject these cases for the stated reason rather than normalize them into a different valid receipt.

The reasoning-facing `EvidenceView::from_retrieval` boundary also performs full receipt verification before constructing a view. This prevents a caller from supplying a structurally invalid receipt, recomputing its unkeyed digest, and bypassing the `VerifiedRetrievalReceipt` integrity gate.
