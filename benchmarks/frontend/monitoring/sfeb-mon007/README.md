# SFEB MON-007 executable action assurance

MON-007 is a semantic reference model for SFEB's action-evidence claim graph. It separates source provenance, authority, sink-enforced execution, independent readback, transparency registration, and trust continuity. A signature or transparency receipt does not prove that the underlying world-state claim is true.

## Reproduce JavaScript semantic assurance

```sh
mkdir -p /tmp/sfeb-mon007
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/sfeb-action-assurance-model-v1.mjs.gz > /tmp/sfeb-mon007/sfeb-action-assurance-model-v1.mjs
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/sfeb-action-assurance-model-v1.test.mjs.gz > /tmp/sfeb-mon007/sfeb-action-assurance-model-v1.test.mjs
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/benchmark.mjs.gz > /tmp/sfeb-mon007/benchmark.mjs
node --check /tmp/sfeb-mon007/sfeb-action-assurance-model-v1.mjs
node --test /tmp/sfeb-mon007/sfeb-action-assurance-model-v1.test.mjs
node /tmp/sfeb-mon007/benchmark.mjs
```

## Reproduce independent Node/Go RFC 9162 proof checks

See `js/README.md` and `go/README.md` for complete extraction commands. The two independently implemented verifiers use Node's built-in `node:crypto` and Go's standard library. They verify each other's inclusion and consistency corpora (136 inclusion + 120 consistency vectors in each direction) and independently generate identical roots and proof paths. Go additionally runs exhaustive generated tests for 32,896 inclusion positions and 32,640 consistency-size pairs through tree size 256.

RFC 9162 is an IETF Experimental Protocol. This work verifies proof relationships against caller-supplied roots; it does not authenticate signed checkpoints or prevent split views without independent witness comparison.

## Reproduced local evidence (2026-11-25)

- JavaScript semantic model: 31/31 tests in five fresh invocations (155 executions); 36 single-field security mutations.
- Node RFC 9162 implementation: 7/7 tests in five fresh invocations (35 executions).
- Go semantic and Merkle suite: 30 test functions per suite, three fresh suite runs; race detector and `go vet` passed in the completed run.
- Go coverage: 72.9% external-effect semantic core; 95.5% RFC 9162 package. Both Merkle verifier functions have 100% statement coverage.
- Go proof microbenchmark medians: consistency 534.5 ns (16→256), 529.3 ns (256→4,096), 527.5 ns (4,096→65,536); inclusion 525.9 ns (16 leaves), 1,046 ns (256), 1,564 ns (4,096). Five runs, 0 B/op and 0 allocations/op.
- JavaScript semantic evaluator: median 39,124 evaluations/s across five 30,000-evaluation runs.

The semantic evaluator measures only in-process predicates. The Go benchmark measures only Merkle proof verification. Neither is an end-to-end workflow or SOTA result. The final combined clean-extraction race run hit the session time limit partway through the exhaustive Merkle package; a prior complete race run on the same source set passed.

## Trust boundary and next work

The semantic model consumes adapter-produced pre-verified facts and `verifiedFields` snapshots. Production adapters must derive these from exact authenticated bytes and independent observations. Outstanding work includes COSE/CBOR and RFC 9942 receipt verification, RFC 9943 historical registration-policy collateral, signed checkpoint/key-history verification, independent witness gossip and split-view detection, and a real sink-enforced conditional write with separately authorized readback.

See `SOTA-COMPARISON-PLAN-v1.md` for the proposed matched PostgreSQL-outbox, DBOS, Temporal, transparency-only and SFEB-over-baseline benchmark. Vendor no-op throughput, local semantic evaluation and complete external-effect qualification are separate workload classes.
