# MON-053 reference-model provenance

- Repository: `Luminous-Dynamics/symthaea`
- Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/14313
- Branch: `research/sfeb-mon053-temporal-admission-v0`
- Base: MON-050 head `60280bb75a188785ff374c88c9c3b3f6a4949d81`
- Scope: temporal interval appraisal, mandatory dependency lease minima and in-memory one-use admission serialization only.

## Local toolchain and verification

- Go: 1.23.2
- Platform: linux/amd64
- `gofmt`
- `go test ./...`
- `go test -race ./...`
- `go test -count=50 ./...`
- `go test -race -count=100 -run '^TestConcurrentRevocationIsLinearizedWithAdmission$' ./...`
- `go vet ./...`
- `go build ./...`

The race-detector invocation repeats 100 schedules per test run 100 times (10,000 paired goroutine schedules) and passed in this local run. This validates only the in-memory mutex model.

No durable nonce ledger, cross-process store, consensus service, signed permit, external system-of-record transaction, hardware fencing, provider restore or SOTA evaluation is implemented.
