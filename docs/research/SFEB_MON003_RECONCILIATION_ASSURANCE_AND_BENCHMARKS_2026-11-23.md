# MON-003 v3: authorized reconciliation and proof-carrying actions

**Date:** 2026-11-23 · **Status:** Executed in-memory reference increment · **Related PR:** #13398

## Why v3 exists

A security review of the 14-case v2 kernel found two untested model gaps:

1. An idempotent retry returned the complete stored mutation result before validating the worker's current fence and authorization. The separate `reconcile(actionKey)` method also returned the full result with no tenant or reader authorization.
2. The sink keyed fencing epochs by objective ID alone. Two tenants using an identical objective ID against a shared sink could share one fencing namespace.

These are defects in the prior reference-model contract and coverage, not evidence of a live production compromise. A test suite cannot prove an invariant that it never exercises.

## Changes in v3

### 1. Execution and reconciliation use separate authorities

An action retry must still pass the live tenant/objective fence, action grant, current policy revision, source-receipt validity, target scope and request-digest checks. When a stored result already exists, the execution endpoint fails closed with `REPLAY_REQUIRES_RECONCILIATION`; it never returns the full stored payload.

A separate read-only reconciliation grant is scoped to tenant, objective, principal, one operation key, read-policy revision, issue time and expiry. It returns only a minimized operation receipt: operation identity, request digest, sink status and commit time. It omits worker identity, target payload and postcondition detail.

This separation permits an independently authorized auditor to determine what happened even after action permission is revoked, without granting permission to perform another mutation. Reader revocation or read-policy revision change invalidates outstanding reconciliation grants.

### 2. Fencing is tenant-scoped

The fence namespace is now the composite `(tenantId, objectiveId)`. Reusing an objective identifier in another tenant cannot share or advance the first tenant's epoch. Lease identity also carries tenant scope and is checked at the sink.

## Executed evidence

Source: [`mon003-objective-kernel-v3.mjs`](../../benchmarks/frontend/monitoring/mon003-objective-kernel-v3.mjs)  
Report: [`mon003-objective-kernel-v3-report-v1.json`](../../benchmarks/frontend/monitoring/mon003-objective-kernel-v3-report-v1.json)

Node.js v22.16.0 executed `node --check` and the full test program. **20/20 deterministic cases passed.** Added cases specifically exercise minimized lost-response reconciliation, audit access after action revocation, unauthorized/expired/revoked reader grants, operation/tenant scope binding, fence isolation for identical objective IDs, and stale-worker replay denial.

Source Git blob SHA: `fa4e86f6ec2761154bfa1b0ccecf5535be99b5f8`  
Source SHA-256: `42f018f8ea2fb94267d6fc32d725cf836322a48f08af443483aa34c6418be936`

V2 remains unchanged as the 14/14 historical baseline. V3 is additive.

**Qualification boundary:** Pure in-memory reference model. The sink assumes atomic fence/policy/idempotency/effect/result persistence. There is no database, broker, distributed lease service, production authorization server, signer/key registry, live source probe or external action sink. Source receipts and tenant identity remain trusted model inputs. These results do not prove exactly-once behavior for arbitrary external systems or production safety.

## Design implications: proof-carrying objectives

The product direction is an assurance contract over existing workflow engines, not a replacement for Temporal, DBOS, event buses or durable timers. Each objective should carry four independently verifiable evidence links:

1. **Truth receipt:** source identity, event identity/sequence, tenant, payload digest and capability-lease status.
2. **Authority receipt:** principal, objective/version, target, policy revision, fence epoch and validity interval.
3. **Execution receipt:** idempotency key, request digest, sink operation ID and durable commit status.
4. **Outcome receipt:** independent readback of the postcondition—or an explicit `outcome_unknown` state.

An objective can claim success only when all four links refer to the same tenant-scoped operation. Unknown outcome is neither success nor failure; it blocks creation of a new operation identity until reconciliation.

## Standardized trust formats

The current local digest encoder is test-only. It is not a cross-language cryptographic contract.

- Use [RFC 8785 / JCS](https://www.rfc-editor.org/rfc/rfc8785.html) where semantic JSON hashing must be identical across implementations. Its I-JSON constraints and property-ordering/serialization rules require full test vectors, including number formatting and Unicode edge cases.
- Evaluate [DSSE](https://github.com/secure-systems-lab/dsse/blob/master/protocol.md) for signing immutable serialized receipts. DSSE signs a typed payload's exact bytes through pre-authentication encoding and avoids a prerequisite canonicalization step. Its optional key ID is an unauthenticated hint, and DSSE does not provide key management or issuer authorization; those remain separate trust services.

Preferred direction: DSSE for typed source and execution receipts; JCS only for the request-digest use case that actually requires semantic JSON equivalence. Never treat a signature as proof that its issuer is authorized for a tenant/resource.

## Industry comparison

- **Temporal** already supplies durable ordered workflow history and replay; SFEB should integrate with it rather than rebuild orchestration. [Durable execution](https://docs.temporal.io/durable-execution) · [Event history](https://docs.temporal.io/evaluate/understanding-temporal)
- **DBOS** already uses workflow ownership tokens checked transactionally on checkpoint writes to reject zombie executors. It also documents at-least-once step execution while workflow outcomes are persisted exactly once. Therefore fencing and workflow recovery are not novel SFEB claims. The differentiation to test is source provenance + current external-sink authority + tenant-scoped idempotency + separately authorized reconciliation and independent outcome proof at the mutation boundary. [DBOS concurrent executions](https://docs.dbos.dev/explanations/concurrent-executions)
- **CloudEvents** provides the source-scoped `(source, id)` event identity model; SFEB should preserve it and reject changed-content reuse.
- **OpenTelemetry** event conventions describe named point-in-time occurrences and state changes. Emit compatible events, but do not mistake sampled diagnostic telemetry for a durable audit/authorization ledger. [Event conventions](https://opentelemetry.io/docs/specs/semconv/general/events/)
- **SentinelBench** remains the agent-monitoring baseline for wait/poll efficiency and task success. Its current 10-minute GPT-5.4 row reports 75/100 with `wait_for` versus 68/100 with `sleep`, and median cost $0.23 versus $1.17. The successful-task median reaction time is 51.7s versus 42.3s, so `wait_for` is not a latency win on every measure. At 40 minutes, the Microsoft report gives 69/100 versus 56/100 and median cost $0.48 versus $4.65. These are monitoring-task results, not SFEB scores; the benchmark authors also identify ephemeral must-act-now tasks as a gap. SFEB must retain this baseline and add source replay, authorization, tenant isolation, reconciliation and outcome-proof strata. [Interactive leaderboard](https://sentinelbench.com/) · [Microsoft Research methodology and limitations](https://www.microsoft.com/en-us/research/articles/sentinelbench-a-benchmark-for-long-running-monitoring-agents/)

## Benchmark protocol

Keep the existing nine matched arms: sleep/inspect; SentinelBench-style `wait_for`; fixed polling; notification plus source reconciliation; durable edge receipt + current auth + idempotent sink; native Temporal; Temporal + SFEB; native DBOS + idempotent external sink; DBOS + SFEB.

Across all arms, run identical trace seeds and add adversarial strata for transient edge events, reconnect/replay gaps, connector drift, duplicate delivery, event-ID collisions, permission revocation after grant, reader revocation, timeout-after-commit, stale-worker retry and same objective IDs across tenants.

| Metric | Proposed target (not yet achieved) |
|---|---|
| Unauthorized mutation | 0 observed, with exact per-family denominator |
| Cross-tenant mutation/read leakage | 0 observed |
| Stale-owner commit acceptance | 0 observed in model and live race schedules |
| Unauthorized reconciliation disclosure | allow-listed metadata only; no postcondition/worker/payload fields |
| Outcome recovery | 100% where durable sink evidence exists; otherwise remain unknown |
| Eligible edge recall | reported separately by source capability tier and retention horizon |
| Added latency | ≤10% over matched native sink-adapter baseline; publish absolute p50/p95 |
| Evidence completeness | 100% of claimed successes carry all required linked receipts |
| Economics | total cost per independently verified action, including replay, probes, storage and verification |

A 20/20 deterministic suite is regression evidence, not a field reliability estimate. Publish per-family denominators, clustered uncertainty, exact source head and reproducible artifacts.

## Promotion gates

A. Model: deterministic/property-based tests and bounded schedule exploration.  
B. Persistence: crash injection around real DB transaction/commit/acknowledgement boundaries.  
C. Concurrency: shared sink, multi-tenant schedules, stale owners, retries and revocation races with a linearizability checker.  
D. Trust: JCS cross-language vectors or DSSE exact-byte verification, issuer scope, key rotation and revocation.  
E. Integration: paired Temporal and DBOS adapters plus a real idempotent/conditional external sink.  
F. Outcome: independent system-of-record readback; executor cannot self-certify success.

## Executed DSSE source-receipt verification preflight

The follow-on trust increment now has an executable verifier rather than relying only on boolean signature fixtures:

- Source: `benchmarks/frontend/monitoring/mon003-dsse-receipt-v1.mjs`
- Report: `benchmarks/frontend/monitoring/mon003-dsse-receipt-report-v1.json`
- Node.js v22.16.0: **22/22 deterministic cases passed**, including real Ed25519 signing and verification using Node's built-in `crypto` implementation.
- The harness checks DSSE v1 PAE with UTF-8 byte lengths and exact payload bytes; payload/signature tampering; payload-type substitution; untrusted or spoofed key IDs; key validity and revocation; issuer identity; tenant and source scope; receipt expiry/future timestamps; malformed base64/JSON; and a fail-closed single-signer profile.
- The trust record—not DSSE `keyid`—supplies issuer identity and authorized tenant/source scopes. The key ID is only an untrusted lookup hint.

This closes the prior *test-fixture crypto boolean* gap for the verifier logic. It does not establish source truth: it proves that a payload was signed by the private key corresponding to a configured public key, and that the current in-memory trust record authorizes that key for the declared issuer/tenant/source at the supplied verification time.

**Qualification boundary:** ephemeral generated test keys and an in-memory registry only. No production trust-root provisioning, durable key-revocation feed, rotation controller, HSM, multi-signature threshold, live source adapter or external key management service is included. This remains a preflight, not production PKI.

## Immediate next work

1. Port operation identity, tenant/objective scope, execution gate and minimized reconciliation to a Rust core.
2. Implement a persistent evidence/operation ledger and define the real transaction boundary before claiming atomicity.
3. Add DSSE verification and a separately administered trust-root/key-revocation interface; evaluate JCS for cross-language request digests.
4. Implement a real external sink adapter with an idempotency or conditional-write contract.
5. Integrate Temporal and DBOS and execute the paired nine-arm benchmark.

**Decision:** v3 closes two gaps in the reference model. The next frontier is a persistent, independently verified evidence ledger tested against native durable-execution engines—not another in-memory case-count increase. This remains research, not production certification.
