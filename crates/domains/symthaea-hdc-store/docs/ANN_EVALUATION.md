# Empirical ANN evaluation

`AnnValidationReport` compares `scan_similar_approx` against the exact
`scan_similar` contract for every supplied query. It records per-query neighbor
IDs and aggregates:

- mean and worst recall@k;
- mean, p50, and p95 candidate fractions;
- the fractions of queries that examined the full store and that explicitly fell back to an exact scan;
- total similarity evaluations.

`AnnValidationSuite` runs the same query corpus at multiple unique top-k values,
which is intended for recall@1, recall@5, and recall@10 evidence. Results are
deterministic for a fixed store, query set, index configuration, and options.

## Release gating

`AnnValidationThresholds` makes acceptance criteria explicit. Its defaults are
starting points rather than production claims: at least 100 queries, mean recall
of 0.95, worst-query recall of 0.80, mean candidate fraction no greater than
0.50, exhaustive evaluation on no more than 50% of queries, and explicit fallback
on no more than 5% of queries.

Real release thresholds should be selected from representative Symthaea memory
workloads and checked separately for corpus size, target similarity, and memory
type. Self-queries are useful only as a wiring sanity test and must not be cited
as realistic recall evidence.

## Evidence artifacts

`AnnValidationReport::to_csv` emits one deterministic row per query, including
exact and approximate neighbor IDs. CI can retain this raw file alongside the
aggregate values and the store/query generation manifest so regressions remain
auditable.
