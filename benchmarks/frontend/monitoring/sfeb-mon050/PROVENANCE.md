# MON-050 dependency-impact oracle provenance

- Repository: `Luminous-Dynamics/symthaea`
- Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/14277
- Branch: `research/sfeb-mon050-dependency-impact-oracle-v0`
- Base commit: `a7e14b36154df5421bc6c5f445832c8563c49da6` (MON-049)
- Scope: deterministic typed-DAG invalidation semantics and six known-answer scenarios only.

## Local verification

Executed in the active container with Go 1.23.2:
- `gofmt`
- `go test ./...`
- `go test -race ./...`
- `go test -count=50 ./...`
- `go vet ./...`
- Draft 2020-12 JSON Schema validation of the scenario corpus

The 1k/10k/100k benchmark figures in BENCHMARKS-v0.1.md are one-iteration smoke observations only. They are not statistically stable, cross-vendor results or SOTA claims.

No restore system, provider integration, authenticated event source, temporal admission boundary, or real-world reliability experiment has been exercised. Hosted CI for this branch remains independent and pending.
