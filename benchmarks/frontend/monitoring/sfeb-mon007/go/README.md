# MON-007 Go semantic and Merkle preflight

This workspace contains a standard-library-only external-effect semantic model plus an RFC 9162 SHA-256 Merkle proof package. The semantic model covers source identity/digest, tenant/resource scope, authority-at-commit, sink fencing, stable operation identity, sink receipt, separate-reader readback and causality. The Merkle package handles inclusion and append-only consistency proofs over already-decoded 32-byte digests.

## Reproduce from the repository root

```sh
mkdir -p /tmp/sfeb-mon007-go/testdata /tmp/sfeb-mon007-go/rfc9162/testdata
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/go/model.go.gz > /tmp/sfeb-mon007-go/model.go
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/go/model_test.go.gz > /tmp/sfeb-mon007-go/model_test.go
cp benchmarks/frontend/monitoring/sfeb-mon007/go/go.mod /tmp/sfeb-mon007-go/go.mod
cp benchmarks/frontend/monitoring/sfeb-mon007/go/testdata/effect-core-vectors-v1.json.gz /tmp/sfeb-mon007-go/testdata/
for f in merkle.go merkle_test.go consistency.go consistency_test.go; do
  gzip -dc "benchmarks/frontend/monitoring/sfeb-mon007/go/rfc9162/${f}.gz" > "/tmp/sfeb-mon007-go/rfc9162/${f}"
done
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/go/rfc9162/differential_test.go.gz > /tmp/sfeb-mon007-go/rfc9162/differential_test.go
cp benchmarks/frontend/monitoring/sfeb-mon007/go/rfc9162/testdata/rfc9162-node-generated-v1.json.gz /tmp/sfeb-mon007-go/rfc9162/testdata/
cd /tmp/sfeb-mon007-go
go test -count=1 ./...
go test -race -count=1 ./...
go vet ./...
go test -cover ./...
go test -run '^$' -bench 'Verify(Inclusion|Consistency)' -benchmem ./rfc9162
```

## Local evidence (2026-11-25)

- Go 1.23.2, linux/amd64, AMD EPYC 9V74.
- 30 test functions per suite: 18 semantic, 5 inclusion, 6 consistency, and 1 cross-implementation differential test. Three fresh suite runs passed; race detector and `go vet` passed.
- 65,536 Go proof checks: 32,896 inclusion positions and 32,640 consistency size-pairs for tree sizes through 256. Three seven-leaf RFC examples are checked.
- Cross-language differential tests: Go verifies 136 Node-generated inclusion and 120 Node-generated consistency proofs. Node independently verifies an equally-sized Go-generated corpus. Both independent fixture builders emit identical roots and proof paths.
- Coverage: 72.9% semantic core; 95.5% RFC 9162 package. Inclusion and consistency verifier functions each have 100% statement coverage.
- Five-run proof microbenchmarks: zero allocations/op. Median consistency: 534.5 ns (16→256), 529.3 ns (256→4,096), 527.5 ns (4,096→65,536). Median inclusion: 525.9 ns (16 leaves), 1,046 ns (256), 1,564 ns (4,096).

The semantic `*Verified` flags and `VerifiedFields` snapshots are pre-verified facts; real adapters must derive them from authenticated bytes and observations. The Merkle package does not authenticate checkpoints or parse COSE. There is no RFC 9942 receipt verifier, SCITT policy-history verifier, witness gossip, split-view protection, or live external sink here.
