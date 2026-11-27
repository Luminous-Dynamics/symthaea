# MON-028 scorer module

Standard-library Go 1.23 package. No network access or third-party dependency is required.

## 1. Prepare and sign the independent suite

The benchmark authority authors a suite JSON conforming to `../SUITE-v1.schema.json`. It fixes case identities, families, expected verdicts, replicate counts, fault-matrix digest, oracle digest, workload, environment and resource budget.

Compute the suite digest:

```sh
go run ./cmd/suite-digest --suite ../SUITE-example.json
```

Sign/custody the suite with MON-026/027. Distribute the same signed suite to every benchmark operator. Do not let a candidate system author expected outcomes.

## 2. Candidate result file

The system under test emits a result JSON conforming to `../RESULT-v1.schema.json`. Each assigned case/replicate must appear exactly once. A result entry contains only its observed verdict and execution state; it does not contain an expected verdict.

Every executed observation must include a SHA-256 of its result record and an evidence-bundle digest. Skipped, timed-out and infrastructure-error observations remain represented but cannot carry a verdict.

## 3. Score and compare

```sh
go run ./cmd/score \
  --suite ../SUITE-example.json \
  --result ./candidate-result.json

# Optional baseline comparison. Both must use the same suite, workload,
# fault matrix, environment, resource budget and full case/replicate set.
go run ./cmd/score \
  --suite ../SUITE-example.json \
  --result ./candidate-result.json \
  --baseline ./baseline-result.json
```

Exit codes: 0 safety gate passed, 2 malformed/unsafe result or safety-gate failure, 3 campaign incomplete, 64 CLI usage error, 65 JSON parse error, 66 file access error, 70 output error.

## 4. Verification commands

```sh
gofmt -w .
go vet ./...
go test ./...
go test -race ./...
go test -run '^$' -fuzz FuzzPairKeyDeterministic -fuzztime=30s
go test -run '^$' -bench BenchmarkScoreRun -benchmem -count=10
```

The performance target is measured separately from the target system's recovery RTO.

## Statistical interpretation

The aggregate and per-family false-qualified rates use the completed unsafe-case denominator. Planned unsafe observations are always reported separately. Incomplete campaigns cannot pass the safety gate. The one-sided Wilson 95% upper bound is an approximate screening interval; common-host, common-reset and common-injector clustering requires cluster-aware confirmatory analysis.

A score file is a derived artifact. Preserve the signed suite, raw SUT outputs, per-case evidence artifacts and score report together under MON-026/027 custody.
