# Local verification record

Reference environment: Linux/amd64, Go 1.23.2, AMD EPYC 9V74.

- `./verify.sh`: PASS
- `go test -count=1 ./...`: PASS
- `go test -count=1 -race ./...`: PASS
- `go vet ./...`: PASS
- `gofmt` check: PASS
- CLI build: PASS
- 29 named tests; join package statement coverage 76.3%. CLI has no dedicated test file yet.

Five-run verifier microbenchmark:

| Run | ns/op | B/op | allocs/op |
|---:|---:|---:|---:|
| 1 | 367,797 | 17,168 | 99 |
| 2 | 370,990 | 17,173 | 99 |
| 3 | 404,806 | 17,163 | 99 |
| 4 | 384,539 | 17,177 | 99 |
| 5 | 369,602 | 17,166 | 99 |

Median: 370,990 ns/op (371.0 µs), 17,168 B/op, 99 allocations/op.

These numbers measure local verifier overhead only. They do not measure agent runtime, trusted observer overhead, witness network latency, downstream effect completion, security efficacy or any external benchmark.

Local checks are not GitHub Actions CI, independent implementation verification, cryptographic interoperability qualification, runtime integration or production qualification.
