# ANN search from shared readers

`HdcStoreReader::open_with_index_policy` applies the same persisted-index
contract as mutable open without requiring write access. Ordinary
`HdcStoreReader::open` intentionally constructs no ANN index.

A shared reader computes the canonical snapshot identity from its selected
header generation and ascending live records. A sidecar is accepted only when
all counts, LSH dimensions, seed, logical fingerprint, ordering, and checksums
match that exact read generation.

Policies:

- `PreferSnapshot`: load a compatible sidecar or rebuild deterministically;
- `Rebuild`: ignore the sidecar and hash every live vector;
- `RequireSnapshot`: fail closed when the sidecar is missing, stale, or corrupt.

`scan_similar` remains exact. `scan_similar_approx` is an explicit opt-in and
returns `SearchOutcome` diagnostics including candidate count, exact fallback,
and whether every live vector was examined.

When `scan_similar_approx` is called on an exact-only reader, it performs a full
exact scan and reports `fell_back_to_exact = true`. Correctness is preserved
without silently paying index construction cost at open time.
