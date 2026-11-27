# MON-023 restore-to-serve benchmark v0.1

**Status:** semantic evaluator benchmark harness and prospective targets; no numeric result frozen.

## Workload and commands

The harness varies required dependencies across 1, 10, 100 and 1,000 entries and evaluates complete service-recovery claims.

From restore-assurance/:
- go test -race -count=5 ./...
- go vet ./...
- go test -count=1 -cover ./...
- go test -run=^$ -fuzz=FuzzRPOUpperBound -fuzztime=2s
- go test -run=^$ -bench=BenchmarkEvaluateServiceRecovery -benchtime=3x

This measures normalized in-memory evidence evaluation only. It excludes pgBackRest, WAL retrieval, PostgreSQL restore/replay, real dependency startup, application traffic and cryptographic verification.

## Recovery rehearsal benchmark matrix

Pin identical source backup, required WAL range, server minor version, database size, storage/network, compression/encryption, surface inventory, dependency graph, app release and invariant suite. Run:
- 1 GB, 100 GB and 1 TB restore sets;
- cold restore vs warm cache;
- full/differential/incremental backup paths;
- target modes: LSN, time, named restore point and end-of-archive;
- injected missing WAL, corrupt backup page, stale secrets, unavailable KMS/DNS/broker, failed app invariant, stale endpoint, split-brain old primary;
- crash schedules before/after restore, replay, surface verification, dependency readiness, app probes and traffic enablement.

Measure RPO upper bound, database RTO, service RTO, artifact retrieval, replay, surface reconciliation, dependency readiness, time-to-first-valid-read/write, p50/p95/p99, resource cost, operator interventions and cleanup.

## Prospective targets (not achieved)

- Zero false service-recovered results on the finite corpus and at least 500 independently authored blind faults.
- Every RPO/RTO pass uses conservative clock uncertainty bounds.
- p95 recovery-to-serve within the declared tier target on matched hardware/workloads.
- Report service recovery success by fault class, not only aggregate success.
- Maximize independently qualified service recoveries per total recovery cost, not raw restore GB/s.

No SOTA superiority is claimed. Numeric comparisons require identical restore target, topology, data, storage, dependencies and application validation.
