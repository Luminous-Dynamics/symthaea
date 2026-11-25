# RFC 9162 Merkle proof preflight

A Go standard-library implementation of SHA-256 Merkle hashing, inclusion proofs, and append-only consistency proofs.

## Implemented

- `LeafHash`: SHA-256 over `0x00 || leaf`.
- `NodeHash`: SHA-256 over `0x01 || left || right`.
- RFC 9162 tree splitting and Merkle Tree Hash calculation.
- Inclusion proof generation for fixtures and `VerifyInclusion`.
- Consistency proof generation for fixtures and `VerifyConsistency`.
- Exact-power-of-two old-tree normalization from RFC 9162 Section 2.1.4.2.

## Tests and local performance

Tests cover every inclusion position and every consistency pair for all tree sizes through 256: 65,536 generated proof checks. They also cover three seven-leaf examples from RFC 9162 Section 2.1.5, wrong roots, altered proofs, truncation, excess nodes, invalid sizes and suffix mutations.

An independent Node implementation using Node's built-in SHA-256 verifies a Go-generated corpus of 136 inclusion and 120 consistency vectors. The Node fixture builder independently emits the same roots and proof paths; Go verifies 136 inclusion and 120 consistency proofs from the Node corpus. This is bidirectional differential testing, not a standards certification.

On Go 1.23.2, linux/amd64, AMD EPYC 9V74, five 300 ms benchmark runs measured 0 B/op and 0 allocations/op. Median consistency verification: 534.5 ns (16→256), 529.3 ns (256→4,096), 527.5 ns (4,096→65,536). Median inclusion verification: 525.9 ns (16 leaves), 1,046 ns (256), 1,564 ns (4,096).

## Trust boundary

RFC 9162 is an IETF Experimental Protocol, not a Standards Track RFC. This package accepts already-decoded 32-byte proof nodes and supplied roots. It does not parse CBOR/COSE, verify RFC 9942 receipts, authenticate checkpoint signatures, validate TS/issuer keys or registration policies, perform witness gossip, or prevent split views by itself. A valid consistency proof only establishes the relationship between supplied roots; checkpoint authentication and independent view comparison are separate requirements.

```sh
go test -count=1 ./rfc9162
go test -race -count=1 ./rfc9162
go test -cover ./rfc9162
go test -run '^$' -bench 'Verify(Inclusion|Consistency)' -benchmem ./rfc9162
```
