# SFEB MON-007 executable action assurance

MON-007 is a semantic reference model for SFEB's action-evidence claim graph. It separates source provenance, authority, sink-enforced execution, independent readback, transparency registration, and trust continuity. It does not claim that a signature or transparency receipt proves the underlying world-state claim is true.

## Run the JavaScript model

From the repository root:

```sh
mkdir -p /tmp/sfeb-mon007/go/testdata
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/sfeb-action-assurance-model-v1.mjs.gz > /tmp/sfeb-mon007/sfeb-action-assurance-model-v1.mjs
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/sfeb-action-assurance-model-v1.test.mjs.gz > /tmp/sfeb-mon007/sfeb-action-assurance-model-v1.test.mjs
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/benchmark.mjs.gz > /tmp/sfeb-mon007/benchmark.mjs
cp benchmarks/frontend/monitoring/sfeb-mon007/go/testdata/effect-core-vectors-v1.json.gz /tmp/sfeb-mon007/go/testdata/
node --check /tmp/sfeb-mon007/sfeb-action-assurance-model-v1.mjs
node --check /tmp/sfeb-mon007/benchmark.mjs
node --test /tmp/sfeb-mon007/sfeb-action-assurance-model-v1.test.mjs
node /tmp/sfeb-mon007/benchmark.mjs
```

## Qualified semantic distinctions

- `effect_verified` does not require transparency registration.
- `externally_anchored` does not imply that an external effect happened or was authorized.
- `action_claim_anchored` binds the signed action statement to exact source, authority, fence, sink receipt, and readback evidence digests.
- Unknown sink outcomes and unavailable readback remain `indeterminate`; reconcile the original operation identity before another mutation.
- Source identity and digest are bound through the action, source receipt, sink receipt, and causal edge. Cross-system timestamps are not used to infer causality.
- Trust-root rotation requires sequential progression and old/new threshold verification. Checkpoint rollback, same-size forks, stale authority, fence replay, tenant/resource substitution, and untrusted readers fail closed.

Nine shared JavaScript/Go golden vectors exercise valid and adversarial external-effect cases. Both implementations produce the same predicate statuses and reason-code sets for the qualified subset.

## Independent Go RFC 9162 proof preflight

`go/rfc9162` implements SHA-256 Merkle leaf/node hashing, inclusion proofs, and append-only consistency proofs. Exhaustive local tests checked 32,896 inclusion positions and 32,640 old/new tree-size pairs for every size through 256, plus RFC 9162 Section 2.1.5's seven-leaf examples and malformed-proof controls. Both verifier functions have 100% statement coverage; the package has 95.5% statement coverage. Five-run local Go microbenchmarks measured zero allocations per verification.

RFC 9162 is an IETF Experimental Protocol, not a Standards Track RFC. The package verifies already-decoded digests against supplied roots; it does not authenticate checkpoints, verify COSE receipts, validate SCITT policies, or prevent split views without independent witness comparison.

## Reproduced local evidence (2026-11-25)

- JavaScript: 31/31 tests passed in five fresh invocations (155 successful test executions); 36 single-field security mutations; five semantic-only benchmark runs, median 39,124 evaluations/s.
- Go: 29 test functions passed across three fresh suite runs; race detector and `go vet` passed. External-effect semantic coverage: 72.9%; RFC 9162 package coverage: 95.5%.
- Go proof verification medians: consistency 534.5 ns (16→256), 529.3 ns (256→4,096), 527.5 ns (4,096→65,536); inclusion 525.9 ns (16 leaves), 1,046 ns (256 leaves), 1,564 ns (4,096 leaves). All measured 0 B/op and 0 allocations/op.

The JavaScript benchmark measures only in-process semantic evaluation. The Go microbenchmarks measure only Merkle proof verification. Neither is an end-to-end workflow, product, or SOTA comparison. Full run details and hashes are in the benchmark and artifact reports.

## Trust boundary and next work

The semantic model consumes adapter-produced pre-verified facts and `verifiedFields` snapshots. Production adapters must derive these from exact authenticated bytes and independent observations. Outstanding work includes independent COSE/CBOR and RFC 9942 receipt verification, RFC 9943 historical registration-policy collateral, signed checkpoint and key-history verification, witness gossip/split-view detection, and a real sink-enforced conditional write with separately authorized readback. These are not represented as implemented or qualified.
