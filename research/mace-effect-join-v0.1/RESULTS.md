# Local verification record

Reference environment: Linux/amd64, Go 1.23.2, AMD EPYC 9V74.

- `./verify.sh`: PASS
- `go test -count=1 ./...`: PASS
- `go test -count=1 -race ./...`: PASS
- `go vet ./...`: PASS
- `gofmt` check: PASS
- CLI build: PASS
- 29 named tests; join package statement coverage 76.7%. CLI has no dedicated test file yet.

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
