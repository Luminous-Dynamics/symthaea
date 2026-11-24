# SFEB MON-007 independent Go core preflight

This directory contains a dependency-free Go implementation of the **external-effect semantic core** of MON-007. It independently models source provenance, tenant/resource scope, authority-at-commit, sink-enforced fencing, stable operation identity, sink acceptance, separate-reader postcondition readback, and causal links between source, operation, sink transaction, and resource revision.

It is deliberately narrower than the JavaScript semantic model. It does not implement or verify COSE/CBOR, Ed25519 signatures, RFC 9942 receipts, RFC 9162 Merkle proofs, SCITT registration policies, transparency-service APIs, distributed witnesses, cryptographic trust-root rotation, a live source/sink, or real readback. The `*Verified` values and `VerifiedFields` snapshots are pre-verified facts that production adapters must derive from the exact authenticated bytes/observations. A unit-test pass is not a security or standards-conformance claim.

## Run

Extract to a temporary directory from the repository root:

```sh
mkdir -p /tmp/sfeb-mon007-go
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/go/model.go.gz > /tmp/sfeb-mon007-go/model.go
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/go/model_test.go.gz > /tmp/sfeb-mon007-go/model_test.go
cp benchmarks/frontend/monitoring/sfeb-mon007/go/go.mod /tmp/sfeb-mon007-go/go.mod
cd /tmp/sfeb-mon007-go
go test ./...
go test -race ./...
go vet ./...
go test -cover ./...
```

No third-party Go dependencies, network access, npm, database, container, or CI runner is required.

## Local evidence (2026-11-25)

- Go 1.23.2, linux/amd64.
- **17/17 tests passed**, including 5 repeated test-suite invocations and 2 repeated race-detector invocations.
- `go vet ./...` passed.
- Statement coverage: **72.6%**.
- Additional negative controls bind the source identity and source digest into the action and sink receipt; a substituted source digest breaks both sink acceptance and the causal edge.
- Missing authority is reported as missing evidence rather than inventing a successful authorization decision.

The Go implementation is an independent core preflight, not yet a cross-language conformance verifier. No JS↔Go differential corpus has been run. The next step is to define canonical typed fixtures and assert equivalent predicate/status/reason outputs across implementations, then connect independently maintained cryptographic and SCITT/COSE verifier libraries.
