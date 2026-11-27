# MON-024 proof-graph reference

This is a standard-library-only Go 1.23 semantic evaluator for a fixed five-stage recovery proof graph.

## Run

```sh
go test ./...
go test -race ./...
go vet ./...
go test -run '^$' -fuzz FuzzEvaluateNeverPanics -fuzztime=30s
go test -run '^$' -bench BenchmarkEvaluate -benchmem -count=10
```

## Use

Construct `composition.Input` with a single evaluation time and bounded evaluation clock, plus exactly one `Receipt` for each required stage. Each receipt must carry the common `Boundary`, exact `Parents`, a production-grade evidence status, observer/producer control roots, freshness data, typed checks and (for the surface stage) its declared surface claim and count manifest.

Call `composition.Evaluate(input)`. Only `Result{Verdict: "qualified", MayEnableTraffic: true}` is eligible to be consumed by a separate activation controller. The evaluator itself has no network, database, filesystem, process or control-plane side effects.

`MayEnableTraffic` is a policy decision based on normalized inputs, not an instruction to perform the operation. A production controller must separately authenticate receipts, enforce authorization, reconcile command outcomes and preserve an auditable idempotency key for any activation.

## Trust boundary

This package does not validate signatures or prove that observer control roots are genuinely independent. Test fixtures use synthetic in-memory values for policy behavior only. It also does not parse PostgreSQL LSNs, confirm WAL durability, inspect schema surfaces or perform an actual restore. Those remain domain-adapter responsibilities in MON-021/022/023.
