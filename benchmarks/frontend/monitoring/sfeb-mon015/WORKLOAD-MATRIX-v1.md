# MON-015 matched workload and qualification matrix v1

All comparisons must record code commit, verifier versions, PostgreSQL version/config, hardware, region, instance sizes, worker counts, network profile, data size, durability settings, evidence retention, costs and run seeds.

| Tier | Question | Workload and fault injection | Required evidence / outcome | Primary metrics |
|---|---|---|---|---|
| W0 orchestration | How fast is durable dispatch? | No-op workflow, fixed-size signed payload, equal queue/transaction semantics; single node and sharded topology. | Completed orchestration events; no effect claim. | throughput, p50/p95/p99, CPU, storage, cost/event, queue age. |
| W1 operation safety | Can retries duplicate a destination mutation? | Same-key retries, simultaneous same-key contenders, same key with cross-tenant payload, lost ACK before/after COMMIT, client restart, lease expiry. | One immutable destination operation identity; exact retries return original outcome; changed bindings conflict. | duplicate committed transitions, collision/conflict accuracy, lock waits, retry latency. |
| W2 effect assurance | Does the digital destination state match the producer claim? | Producer false-success, sink rollback, delayed commit, exact readback, stale replica, wrong revision, wrong state digest, readback gap. | observed-committed only after exact independent readback; otherwise missing/contradicted/unresolved. | false-success rate, readback delay, stale observation rate, qualified effects/s. |
| W3 independence | Are the evidence sources actually independent? | Shared DBA, cloud account, KMS root, operator, build pipeline, network admin, backup/restore admin; shared DB login; compromised sealer. | Shared/unknown roots block independence claim; report a dependency graph. | false-independence acceptances, root coverage, independent reviewer findings. |
| W4 custody and retention | Can a third party reconstruct the evidence later? | Key rotation, sealer restart, timestamp missing, checkpoint gaps, vacuum-aged commit timestamps, archive export, offline verification. | Cryptographic verification and explicit retention/expiry limitations; no upgrade when metadata is missing. | offline reproduction success, verification time, bytes/effect, retention cost, timestamp coverage. |
| W5 authority and adversarial semantics | Can policy/authority changes cross the action boundary? | Revoke, expiry at commit, early deferred check, stale token, action mismatch, approval rejection, replayed nonce, contradictory evidence, fuzz/mutation cases. | Zero false accepts in mandatory corpus; invalid or missing authority never promotes to qualified. | false accept/reject, mutation score, coverage, time-of-check-to-commit, race outcomes. |
| W6 physical effect | Did the real system do what digital state says? | Actual actuator/site/robot/payment settlement with independent calibrated sensor, witness, or external system-of-record. | Separate physical observation with calibration, identity, timestamp and control-root provenance. | discrepancy rate, sensor coverage, calibration age, false physical confirmations. |

## Mandatory test ladder

1. Valid path: producer, sink and observer bind the same destination, operation, request, revision and state; authority valid; independent observer.
2. Lost acknowledgement: COMMIT succeeds but producer never sees the ACK; identical retry must not duplicate the effect.
3. Rollback: database function returns commit-pending, caller rolls back; observer must not later see a committed operation.
4. Retry conflict: same operation ID with different request or scope must conflict, not replay.
5. Cross-tenant attack: tenant A's writer/observer attempts tenant B's resource; no mutation or disclosure.
6. Authority race: revocation and expiry race the effect commit; final classification is based on actual commit time, not function return time.
7. Observer substitution: same authenticated principal, wrong principal, shared control root, stale replica, wrong revision and changed state digest.
8. Evidence substitution: same-looking hex but changed digest algorithm, profile, canonicalization, encoding, purpose or subject.
9. Lifecycle attack: OAR execution after denial, approval after allowed, confirmation linked to wrong execution, cross-run or cross-agent parent.
10. Effect overclaim: AAC confirmed without a response digest, planned with phantom attestation, dispatched with response digest, unknown attestation promoted, or timeout mislabeled as confirmed.
11. Capture gap: disable trigger, truncate/restore, interrupt WAL/CDC, observer lag, missing checkpoint, missing receipt.
12. Physical mismatch: digital state says complete but independent physical observation contradicts it.

## Metrics and denominators

Report separately:
- requested/dispatch events;
- transaction commits;
- refused attempts;
- unresolved outcomes;
- observed committed effects;
- qualified digital effects;
- qualified physical effects.

Latency segments: policy/authority check, dispatch, receiver idempotency, destination transaction, commit, observer readback, external timestamp/receipt verification, final join.

Never remove timeouts, refusals, failed dispatches, rollbacks, unresolved records, contradicted results, or missing evidence from the denominator. A benchmark that counts only successful calls is not an assurance benchmark.

## Proposed targets (not measured achievements)

- 0 unauthorized acceptance, 0 cross-tenant disclosure, 0 duplicate committed effects, 0 false-success and 0 false-independence acceptances on the mandatory labelled corpus.
- 100% of missing authority/readback/trust-root evidence remains explicit as indeterminate or unqualified; no optimistic defaults.
- 100% of the pinned OAR upstream positive, compatibility and negative vectors pass in the upstream implementation. MON does not replace that verifier.
- 100% of expected outcomes match the independent deterministic mutation corpus.
- Blinded holdout: target 0 false-accepts across at least 500 independently authored adversarial cases before claiming robust qualification; publish sample count and confidence bounds.
- W0 initial overhead target: at most 5% throughput reduction or latency increase only for a matched workload with identical durability and semantics. This is a target for discussion, not a current result.
- W2/W5 primary objective: maximize qualified effects per dollar and per joule subject to the safety constraints above; raw TPS alone is not the objective.
- W6: no physical claim without a declared sensor/witness boundary, calibration status, and independent provenance.
