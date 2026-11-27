# MON-022 database surface coverage benchmark v0.1

**Status:** benchmark harness and prospective targets; no numeric result frozen.

## Reproducible commands

From surface-coverage/:
- go test -race -count=5 ./...
- go vet ./...
- go test -count=1 -cover ./...
- go test -run=^$ -fuzz=FuzzSurfaceScopeIdentity -fuzztime=2s
- go test -run=^$ -bench=BenchmarkEvaluateCoverage -benchtime=3x

## Workload

The Go benchmark generates 12, 100, 1,000, 10,000 and 100,000 pre-normalized surface attestations. Each object has a typed inventory scope digest, source/target identity, method profile, evidence digest and freshness boundary. The benchmark measures in-memory surface-inventory evaluation only. It excludes PostgreSQL catalog queries, snapshots, sequence reads, LO byte streaming, generated-column recomputation, observer deployment, cryptography and restore tests.

## Prospective targets (not achieved)

- Zero false full-state-qualified outcomes when any required surface is missing, unknown, stale or excluded.
- Zero false independence when observer control root equals either endpoint root.
- 100% classification accuracy on 36 frozen scenario seeds, then 500+ independently authored blind holdouts.
- p95 under 250 ms for 100,000 normalized surface attestations on a pinned reference host after optimization.
- Incremental invalidation of only surfaces affected by a changed schema/extension/publication profile.

Report p50/p95/p99, memory, allocations, evidence bytes/object, observer cost, missing-surface recall and total assurance cost. Do not compare with raw CDC rows/s; this is an inventory qualification benchmark.

## Qualification boundary

No exact-head Go test, fuzz run, numeric benchmark, live PostgreSQL adapter or independent holdout is claimed yet. The local environment can run Go 1.23.2 but could not resolve GitHub for a fresh exact branch checkout; current source must be tested against the exact commit before numeric results are frozen.
