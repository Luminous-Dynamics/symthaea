# SFEB MON-008 — authenticated checkpoint preflight

**Status:** research reference implementation; draft PR stacked on [MON-007 #13551](https://github.com/Luminous-Dynamics/symthaea/pull/13551), tracked by [MON-008 issue #13583](https://github.com/Luminous-Dynamics/symthaea/issues/13583).

MON-008 moves beyond checking Merkle proof relationships. This increment adds domain-separated Ed25519 checkpoint authentication, external bootstrap pins (no TOFU), sequential dual-threshold trust-root rotation, monotonic tree/time/policy state, RFC 9162 extension checks, and witness quorums counted by distinct configured failure domains. An older witness view needs its own consistency path to the accepted head. Go and Node independently generate and verify each other's checkpoint and witness fixtures.

## Critical boundaries

The binary signing profile here is a test profile, **not COSE and not wire-compatible with RFC 9942**. It does not parse RFC 9942 receipts, RFC 9943 Signed Statements or historical policy collateral. It is a pre-adapter executable model, not a replacement transparency service.

Verifier state is in memory. Snapshot rollback resistance is a deployment responsibility. The verifier retains the active trust root only; historical archive replay needs a separately retained and validated root-update chain. Witness failure-domain labels are configured out of band and require independent operational auditing. Signatures, Merkle proofs and witness agreement do not establish that an external action actually happened.

## Reproduce from the repository root

```sh
for f in checkpoint/profile.go checkpoint/profile_test.go cmd/checkpoint-vectors/main.go; do
  gzip -dc "benchmarks/frontend/monitoring/sfeb-mon008/go/${f}.gz" > "benchmarks/frontend/monitoring/sfeb-mon008/go/${f}"
done
for f in checkpoint.mjs checkpoint.test.mjs generate-vectors.mjs; do
  gzip -dc "benchmarks/frontend/monitoring/sfeb-mon008/js/${f}.gz" > "benchmarks/frontend/monitoring/sfeb-mon008/js/${f}"
done
for f in go-generated-v1.json node-generated-v1.json; do
  gzip -dc "benchmarks/frontend/monitoring/sfeb-mon008/js/testdata/${f}.gz" > "benchmarks/frontend/monitoring/sfeb-mon008/js/testdata/${f}"
done
cd benchmarks/frontend/monitoring/sfeb-mon008/go
go test -count=1 ./...
go test -race -count=1 ./...
go vet ./...
go test -cover ./...
go test -run '^$' -bench 'BenchmarkVerifyCheckpointWitnessQuorum' -benchmem -count=5 ./checkpoint
cd ../js
node --test checkpoint.test.mjs
```

The parent MON-007 Go module supplies RFC 9162 consistency verification. The Node implementation uses built-in `node:crypto`; no npm dependencies are required.

## Local preflight, not CI

Go: 18 test functions, three fresh complete suite runs, one separate race-detector run, `go vet`, and 75.3% checkpoint-package statement coverage. Node: five tests across five fresh invocations (25 passing executions). Both directions of cross-language Ed25519 fixtures pass.

Five-run verifier-only medians on Xeon Platinum 8573C: 132.729 µs (one witness), 202.110 µs (two), 292.231 µs (three). These exclude network, disk, COSE, SCITT, sink mutation, readback and offline audit. They are not end-to-end or SOTA measurements.

The detailed assurance-gap benchmark protocol, test report, benchmark report, and language READMEs are stored as gzip assets alongside the source. The downloadable self-contained preflight bundle is attached to the conversation.
