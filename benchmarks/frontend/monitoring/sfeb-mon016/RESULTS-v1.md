# MON-016 local preflight results v1

Environment: Go 1.23.2 / linux-amd64, AMD EPYC 9V74.

- `go test -race -count=5 ./...` — PASS.
- `go vet ./...` — PASS.
- `go test -count=1 -cover ./...` — PASS, 77.0% statement coverage.
- `go test -run '^$' -bench . -benchmem` — PASS.

Two-second fuzz smoke:

| Target | Executions | Result |
|---|---:|---|
| FuzzGapNeverQualifiesComplete | 88,164 | PASS |
| FuzzSharedRootNeverProducesWitnessedNegative | 88,787 | PASS |
| FuzzQuerySubstitutionNeverQualifies | 119,843 | PASS |

Microbenchmarks on this host:

| Benchmark | ns/op | B/op | allocs/op |
|---|---:|---:|---:|
| VerifyBoundedCaptureSmall (4 items) | 1,368 | 2,056 | 5 |
| VerifyBoundedCaptureLarge (1,000 items) | 133,674 | 475,440 | 8 |

These are local Go semantic-model measurements, not source connector throughput or database capture cost.

Not run: PostgreSQL WAL/slot integration, outbox reconciliation, SCITT/RFC 9162 cryptographic verification, independent second-source enumeration, third-party holdout, matched industry benchmark, or production capture trials.
