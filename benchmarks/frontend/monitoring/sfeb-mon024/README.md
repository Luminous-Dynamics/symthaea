# MON-024 — End-to-End Evidence Graph Qualification

**Status:** executable standard-library Go composition oracle, adversarial tests, fuzz target and microbenchmarks; upstream adapters, cryptographic receipt verification, live PostgreSQL/Kafka recovery exercises and independent production qualification pending.

MON-024 composes the five prior MON increments into one bounded decision:

> Can an independent verifier establish that a specific causal history was correctly ordered, continuously captured, completely scoped, restored to the intended point, and safely validated before service traffic is enabled?

It is not another CDC connector, backup system or orchestration engine. It is a claim-composition and activation-gate layer over independently produced evidence.

## The compositional model

```text
MON-019 causal provenance
          |
          v
MON-020 temporal/event-order assurance
          |
          v
MON-021 PostgreSQL CDC continuity
          |
          v
MON-022 database-surface coverage
          |                         \
          |                          \
          +--------------------------+-----> MON-023 restore-to-service
                                                |
                                                v
                                      MON-024 traffic gate
```

Every receipt must bind the same incident, claim, source cluster, database, source epoch, timeline, snapshot, terminal source position, scope digest and inventory digest. Parent links bind each stage to the exact parent receipt ID and output digest; restore consumes both the CDC and surface-coverage receipts.

The evaluator returns one of three outcomes:

- `qualified`: all five stage claims join, are fresh and runtime-independently graded, have required parent edges, satisfy typed checks, and have at least two observer control domains. Only this outcome sets `may_enable_traffic=true`.
- `blocked`: contradictory identities, invalid parent links, explicit failed checks, unsupported stages, duplicated stage/evidence IDs, full-state exclusions or observer/producer control-root aliasing.
- `indeterminate`: missing, stale, unknown, non-runtime, insufficiently independent or clock-unqualified evidence; unresolved mutation outcomes also hold the gate.

No missing value is silently converted into a pass. An unknown outcome from a prior mutating command must be reconciled before a new mutation is considered.

## Implementation

`proof-graph/` contains a Go 1.23, standard-library-only semantic reference:

- typed stage and evidence models;
- exact parent-edge and parent-output digest checks;
- shared recovery-boundary identity checks;
- evidence freshness, evaluation clock-domain and uncertainty bounds;
- observer/producer control-root separation and minimum observer-domain diversity;
- tri-state required checks for causal, temporal, CDC, surface, restore and service-readiness claims;
- separate `Covered`, `NotApplicable`, `Excluded` and `Unknown` surface counts;
- fail-closed traffic eligibility;
- adversarial tests, a fuzz target and allocation-reporting benchmarks.

## Current limits

The evaluator validates the *shape and declared status* of normalized receipts. It does not verify digital signatures, attest observer administration boundaries, connect to PostgreSQL, decode WAL, run pgBackRest, inspect Kafka offsets, perform application probes or control a load balancer. Runtime adapters must produce signed receipts and independent observers must verify them before a production claim can be made. Unit-test fixtures are policy test data, not production evidence.

This branch does not claim SOTA superiority. The immediate result is a compositional contract and a locally executable fail-closed reference that can be compared fairly against existing tools on matched fault scenarios.

## Files

- `END-TO-END-EVIDENCE-GRAPH-CONTRACT-v1.md` — proof graph, claim semantics, trust boundaries, threats and integration plan.
- `SCENARIOS-v1.json` / `SCENARIOS-v1.schema.json` — descriptive known-answer scenario seeds.
- `BENCHMARKS-v0.1.md` — local preflight measurements, benchmark protocol and prospective targets.
- `proof-graph/` — semantic evaluator, tests, fuzz target and benchmark harness.

## Local preflight

Recorded on Linux/amd64, Go 1.23.2, virtual AMD EPYC 9V74 host:

- `go test ./...` — pass.
- `go test -race ./...` — pass.
- `go vet ./...` — pass.
- `go test -run '^$' -fuzz FuzzEvaluateNeverPanics -fuzztime=3s` — pass; 92,576 executions in the recorded run.
- `go test -run '^$' -bench BenchmarkEvaluate -benchmem -benchtime=1s` — pass; exploratory numbers are recorded in `BENCHMARKS-v0.1.md`.

These are local package checks, not GitHub Actions results, exact-head checks after publication, runtime integration tests or independent qualification.
