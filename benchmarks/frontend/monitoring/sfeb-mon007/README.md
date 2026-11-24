# SFEB MON-007 executable action-assurance model v1

A dependency-free JavaScript reference model for the **semantics** of SFEB's action-evidence claim graph. It follows the MON-007 specification but deliberately does not implement cryptography, COSE/CBOR, SCITT APIs, Merkle proof verification, a witness service, or a real mutation sink.

## Run

From the repository root:

```sh
mkdir -p /tmp/sfeb-mon007
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/sfeb-action-assurance-model-v1.mjs.gz > /tmp/sfeb-mon007/sfeb-action-assurance-model-v1.mjs
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/sfeb-action-assurance-model-v1.test.mjs.gz > /tmp/sfeb-mon007/sfeb-action-assurance-model-v1.test.mjs
gzip -dc benchmarks/frontend/monitoring/sfeb-mon007/benchmark.mjs.gz > /tmp/sfeb-mon007/benchmark.mjs
node --check /tmp/sfeb-mon007/sfeb-action-assurance-model-v1.mjs
node --check /tmp/sfeb-mon007/benchmark.mjs
node --test /tmp/sfeb-mon007/sfeb-action-assurance-model-v1.test.mjs
node /tmp/sfeb-mon007/benchmark.mjs
```

The extracted files share one directory so the test and benchmark imports resolve. No npm packages, Cargo workspace, network, database, container runtime or CI runner is required. Uses Node's built-in test runner and performance APIs.

## What it qualifies semantically

- Separate predicates for source provenance, authority, sink fencing, sink acceptance, idempotency, readback, causality, issuer authentication, registration policy, transparency registration, checkpoint freshness and trust-root continuity.
- `effect_verified` does not require transparency registration.
- `externally_anchored` does not imply the effect happened or was authorized.
- `action_claim_anchored` additionally requires the signed statement to bind the exact action, source, authority revision, fence, sink receipt and readback digests.
- A sink result of `unknown` and an unavailable readback remain `indeterminate`.
- A replayed fence token, stale authority, different tenant/resource, mismatched request digest, untrusted reader, old checkpoint, same-size fork, failed consistency proof, duplicate TS identity, insufficient failure-domain diversity, historical policy substitution or trust-root rollback fail closed.
- Cross-system timestamps are not compared to infer causality. Causal links use source event ID, stable operation ID, sink transaction ID and resource revisions.
- Root rotation requires sequential version progression and both old- and new-root threshold verification.

## Crucial trust boundary

Fields such as `signatureVerified`, `receiptSignatureVerified`, `inclusionProofVerified`, `registrationPolicyProofVerified`, `trustAnchorHistoryVerified`, `policyEffectiveAtRegistration`, `receiptVerified`, `observationVerified` and `enforcedAtSink` are **pre-verified facts**. The model does not validate the underlying signed bytes or determine whether a source/sink is truthful. Production adapters must derive these facts through real cryptographic verification, independently trusted key/policy state, authenticated sink interfaces and separately authorized readback. The verifier-side `trustPolicy` must not be supplied by the untrusted statement producer.

This model is a policy-semantics test seam, not a production security boundary. Passing its unit tests is not evidence of RFC conformance, distributed safety, operational independence or physical durability.

## Deliberate non-goals / next work

1. Add typed schema and canonical test vectors for the profile.
2. Integrate an independently maintained COSE/CBOR + RFC 9942/9162 verifier and validate the MON-006 wire fixtures.
3. Implement independent Rust and Go verifiers with different critical libraries.
4. Bind the profile to a real SCITT Transparency Service and policy-history collateral.
5. Integrate a low-risk external sink with sink-enforced conditional write/fencing and independent readback.
6. Run process/host fault tests and matched PostgreSQL/DBOS/Temporal baselines.

The benchmark driver measures only this pure semantic evaluator. It is a reproducibility baseline for future implementations, not an industry or SOTA comparison.

## Reproduced local evidence (2026-11-25)

- `node --check` passed for the model and benchmark drivers.
- **28/28 tests passed in five separate fresh Node.js test-runner invocations** (140 successful test executions; 0 failures).
- The adversarial mutation matrix changes 36 security-critical fields one at a time and asserts that full qualification cannot remain `verified`.
- Five benchmark runs, each with 3,000 warm-up evaluations and 30,000 measured evaluations (30 batches), all returned `verified` for every baseline fixture.
- Median semantic-evaluator throughput: **38,208 evaluations/s** (range 36,847–40,483) on Node.js v22.16.0, Linux x64, 3 logical CPUs.
- Median per-evaluation sample latency across those runs: p50 **0.024556 ms**, p95 **0.033526 ms**, p99 **0.044903 ms**.

Full run-by-run measurements are in `benchmark-report-v1.json`. These figures cover only the semantic model and must not be compared to workflow engine or complete verifier throughput.

## Authenticated-field binding contract

Each verified evidence object carries a `verifiedFields` snapshot emitted by the trusted decoding/verification adapter. The model compares the interpreted fields against that snapshot before accepting dependent predicates. This catches mutation of a field while a detached `signatureVerified: true` flag is left unchanged. Production adapters must derive the snapshot directly from the exact bytes they verified; an untrusted producer must never be allowed to author the snapshot.
