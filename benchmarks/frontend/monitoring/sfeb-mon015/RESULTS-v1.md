# MON-015 local semantic preflight results v1

**Environment:** Go 1.23.2, linux/amd64, AMD EPYC 9V74 host.

## Tests

- go test -race -count=5 ./... — PASS.
- go vet ./... — PASS.
- go test -count=1 -cover ./... — PASS, 81.7% statement coverage.
- 18 deterministic mutation cases execute under TestEveryMutantCorpusEntryHasAnExpectedVerdict.
- Separate tests cover AAC's five effect statuses, attestation grades, effect-mode overclaim, timeout/errored verdict pairings, independent assurance axes, optional/malformed cross-party blocks, producer/sink/observer reconciliation, and OAR lifecycle edges.

## Bounded fuzz smoke

Five Go fuzz targets were each run for two seconds on the final recorded source snapshot:

| Fuzz target | Executions | Result |
|---|---:|---|
| FuzzDigestMutationNeverEqualsAcrossContextChanges | 96,472 | PASS |
| FuzzOARConfirmationCannotAppearWithoutExecution | 97,595 | PASS |
| FuzzAACOverclaimNeverPasses | 102,413 | PASS |
| FuzzCrossTenantScopeNeverQualifies | 95,638 | PASS |
| FuzzOARCrossRunReferenceNeverConfirms | 98,094 | PASS |

The fuzz targets are bounded smoke tests, not statistically meaningful security proof. Run the 30-second command set in run-local.sh and preserve full transcripts for qualification.

## Microbenchmarks

A local go test -run '^$' -bench . -benchmem run reported:

| Benchmark | ns/op | B/op | allocs/op |
|---|---:|---:|---:|
| BenchmarkCompareDigestEqual | 235.0 | 0 | 0 |
| BenchmarkCompareDigestIncomparable | 246.8 | 0 | 0 |
| BenchmarkQualifyDigitalEffect | 549.2 | 0 | 0 |
| BenchmarkProjectOARLifecycle | 534.3 | 864 | 6 |

These are semantic-model microbenchmarks on this host. They are not database throughput, end-to-end action latency, cost per action, or a vendor comparison.

## Not run / not qualified

- No maintained upstream OAR verifier run; pinned OAR vectors were not executed in this environment.
- No AAC parser, COSE or SCITT signature/receipt verifier was run here.
- No PostgreSQL runtime/transaction/RLS/race integration: the local environment lacks psql and a PostgreSQL server/container runtime.
- No DBOS/Temporal matched benchmark, external independent holdout, third-party review, or physical W6 experiment.
- This package consumes normalized verification results; passing these tests does not establish source authenticity, transparency-log completeness, or a real-world effect.
