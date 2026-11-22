# Local verification record

Reference environment: Linux/amd64, Go 1.23.2, AMD EPYC 9V74.

- `./verify.sh`: PASS
- `go test -count=1 ./...`: PASS
- `go test -count=1 -race ./...`: PASS
- `go vet ./...`: PASS
- `gofmt` check: PASS
- CLI build: PASS
- 38 top-level test functions plus 18 subtest cases.
- `go test -cover ./...`: join package 76.7%; CLI package 66.2%.
- The CLI tests cover missing out-of-band trust/policy files, agent-supplied trust fields in the evidence, duplicate JSON keys, unknown pinned fields, empty files, missing files and oversized input.
- The global receipt-ID test rejects collisions across all five roles, including a witness ID colliding with an authorization receipt.

Five-run verifier microbenchmark:

| Run | ns/op | B/op | allocs/op |
|---:|---:|---:|---:|
| 1 | 370,140 | 17,292 | 100 |
| 2 | 373,659 | 17,301 | 100 |
| 3 | 369,918 | 17,296 | 100 |
| 4 | 361,841 | 17,304 | 100 |
| 5 | 347,239 | 17,299 | 100 |

Median: 369,918 ns/op (369.9 µs), 17,299 B/op, 100 allocations/op.

These numbers measure local verifier overhead only. They do not measure agent runtime, trusted observer overhead, witness network latency, downstream effect completion, security efficacy or any external benchmark.

Local checks are not GitHub Actions CI, independent implementation verification, cryptographic interoperability qualification, runtime integration or production qualification.
