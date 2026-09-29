# EPF-008 canonical identity vectors

These vectors define the byte-level identity contract for memory projection references.

## Encoding

- Hash: SHA-256.
- Domain separators are ASCII UTF-8 strings terminated by one NUL byte.
- Strings are UTF-8 and encoded as a 4-byte unsigned big-endian byte length followed by raw bytes.
- schema_version is an unsigned 16-bit big-endian integer.
- memory_kind is a fixed one-byte tag: Working=0, Episodic=1, Semantic=2, Procedural=3, KnowledgeGraph=4, Vector=5, Hdc=6.
- Optional source_frontier is encoded as a one-byte presence tag (0/1), followed by the length-prefixed string when present.
- No JSON serialization, field ordering, platform endianness, or Rust enum representation participates in the digest.

The contract is deliberately narrower than general-purpose JSON/RDF canonicalization. RFC 8785 establishes why cryptographic operations need an invariant representation, while W3C RDF Dataset Canonicalization targets RDF dataset normalization; EPF-008 only needs a small stable encoding for typed memory-reference identities.

## Independent verification

Run python3 scripts/verify_epf008_vectors.py from this crate directory. The verifier uses only Python's standard library and the vectors in vectors/epf-008.json.

## Non-authority invariant

A matching digest establishes byte-level identity under this encoding. It does not establish truth, evidence strength, corroboration, currentness, authority, or semantic validity.
