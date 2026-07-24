# ANN validation contract

The format-v2 store keeps canonical BinaryHV data on disk and rebuilds an
in-memory random-hyperplane LSH index when opened. The index is an acceleration
hint, not part of the correctness boundary.

## Public search semantics

- `scan_similar` is exact. It evaluates every live vector and returns a
  deterministic top-k ordering.
- `scan_similar_approx` is explicitly approximate. Its `SearchOutcome` reports
  the number of vectors examined, the total live population, and whether the
  result was exact because every vector was examined.
- Candidate count is not treated as evidence that the true nearest neighbors
  were included. Callers choose an exact-fallback policy through
  `ApproximateSearchOptions`.

## Default LSH configuration

New stores use 32 bands with 8 rows per band. This replaces the former 10 by 32
configuration, whose long bands strongly suppressed collisions for nearby
vectors. The new default uses 256 hyperplanes rather than 320 and favors recall
at the cost of larger candidate sets.

`estimated_lsh_candidate_probability` exposes the standard random-hyperplane
model as a tuning aid. It converts Hamming agreement to bipolar cosine
similarity, estimates per-row collision probability, then calculates the chance
of sharing at least one complete band. This model is not a substitute for
measurement on Symthaea's real vector distributions.

## Required empirical gate before performance claims

A release benchmark should compare approximate results against exact top-k
ground truth across representative stores and report:

1. recall@1, recall@5, and recall@10;
2. mean and percentile candidate fractions;
3. reopen/index-rebuild duration;
4. query latency for exact and approximate paths;
5. results stratified by target similarity and corpus size;
6. deterministic seeds and machine-readable raw output.

No production recall percentage should be documented until that harness exists
and is run against representative Symthaea memories rather than only synthetic
self-queries.
