# SFEB MON-008 — authenticated checkpoint preflight

**Status:** research reference implementation; draft PR [#13620](https://github.com/Luminous-Dynamics/symthaea/pull/13620), stacked on [MON-007 PR #13551](https://github.com/Luminous-Dynamics/symthaea/pull/13551). Follow-on programme: [issue #13583](https://github.com/Luminous-Dynamics/symthaea/issues/13583).

MON-008 extends MON-007's independent RFC 9162 proof checks with authenticated, monotonic checkpoint observations. It adds domain-separated Ed25519 signatures, an externally provisioned bootstrap pin (no TOFU), sequential dual-threshold root rotation, policy-version/digest continuity, tree-extension proofs, witness quorums counted by distinct configured failure domains, per-witness last-seen state, and Go/Node cross-language fixtures.

## Reproduce

From a clean checkout **or from the extracted self-contained bundle**, run:

```sh
bash benchmarks/frontend/monitoring/sfeb-mon008/scripts/reproduce.sh
```

The runner expands committed gzip source and fixture files into sibling working-tree paths, including the MON-007 Go module used by MON-008. It then runs the semantic, Merkle and checkpoint suites, race detector, vet and coverage. Expanded source files are local build inputs and need not be committed. Set `RUN_BENCHMARKS=1` to run the verifier-only MON-008 benchmark after tests.

## Local preflight results (2026-11-25; not CI)

- MON-007 semantic model: 31 tests; five fresh invocations (155 passing executions).
- MON-007 Go suite: 30 test functions; repeated suite, race and vet evidence is recorded in its own reports.
- MON-007 Node Merkle verifier: 7 tests per invocation; five fresh invocations (35 passing executions).
- MON-008 Go: 18 test functions, three fresh full-suite runs; separate race-detector run passed; `go vet` passed; checkpoint package statement coverage 75.3%.
- MON-008 Node: 5 test cases across five fresh invocations (25 passing executions).
- Go/Node generated fixtures verify bidirectionally.
- Five-run Go verifier-only medians: 132.729 µs (one witness), 202.110 µs (two), 292.231 µs (three).

The benchmark covers in-process signature, quorum and verifier-state checks only. It excludes network, durable storage, COSE, SCITT registration, external sink mutation, independent readback and offline audit. It is not an end-to-end latency or SOTA result. Machine details and raw samples are in `benchmark-report-v1.json.gz`; test counts and scope are in `test-report-v1.json.gz`.

## Critical trust and standards boundaries

The signed binary envelope is a **custom research test profile**, not COSE and not wire-compatible with RFC 9942. It does not parse RFC 9942 receipts, RFC 9943 Signed Statements, or historical registration-policy/trust-anchor collateral. It is a semantic preflight, not a replacement transparency service.

The initial log trust root, bootstrap pin, witness registry and failure-domain assignments are provisioned out of band. Verifier state is in memory; production rollback resistance requires separately governed durable monotonic storage. The verifier retains the active root only; a historical archive auditor must retain and validate the complete root-update chain and applicable policy collateral. Witness identities/failure domains require independent operational governance.

A signature proves that a key signed exact bytes. A valid Merkle proof proves a relationship to the authenticated root. A witness quorum indicates that distinct configured failure domains attested to the same checkpoint. None of those alone proves the underlying statement is true or that an external action occurred. Sink-side fencing, a stable operation ID, the sink receipt and separately authorized readback remain distinct assurance claims.

## Paradigm-shift benchmark

See `ASSURANCE-GAP-BENCHMARK-v1.md`. The Assurance Gap Challenge deliberately removes trust in the original orchestrator, its database and its credentials after the action. A fresh auditor receives only portable evidence plus independently configured trust roots, then reconstructs authority, source, action, sink result, readback, checkpoint, append-only history and historical policy context. Scores are verified, missing, contradicted, stale or indeterminate—not just throughput.

The benchmark matches PostgreSQL outbox, DBOS, Temporal, SCITT/RFC 9942+9943, Rekor, SFEB overlays and a native sink control across W0–W6 workload tiers. It keeps no-op orchestration separate from external-effect qualification.


## Interoperability landscape (updated 2026-11-25)

MON-008 is deliberately not positioned as a new agent receipt format. See `INTEROPERABILITY-AND-NOVELTY-MAP-v1.md` for the current RFC 9942/9943, AEE, Vaara Receipt and Sigstore comparison. The AEE draft already defines caller/serving-party paired evidence and offline bundles; the next differentiator is to qualify the real destination-side state change, independent readback and declared observation completeness.

`ASSURANCE-GAP-BENCHMARK-v2.md` adds AEE, the SCITT AI Agent Execution profile and Vaara Receipt as comparison arms, and adds omission/coverage adversarial tests. It forbids treating missing receipt evidence as proof that an action did not occur outside an independently qualified observation boundary.
