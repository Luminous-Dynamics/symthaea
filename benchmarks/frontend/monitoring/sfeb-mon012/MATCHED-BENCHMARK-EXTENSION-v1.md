# MON-012 — matched benchmark extension v1

**Status:** proposed qualification protocol. No PostgreSQL integration or matched external-product benchmark has run.

## 1. Extend W0-W6

Keep MON-011 W0-W6 intact. Add these negative controls:

| Tier | Extension | Required controls |
|---|---|---|
| W1 | Destination-global identity and principal-bound immutable operation | Same ID across tenants, resources, principals, requests, policies, authority revisions, fences; concurrent submission and lost ack |
| W2 | Principal-scoped mutation and exact-resource read authorization | Cross-tenant mutation, cross-resource read, stale revision, wrong digest, caller-supplied principal, GUC spoof |
| W3 | Control-dependency graph rather than service/role labels | Shared identity admin, credential root, key custodian, cloud, storage, operator, network and deployment roots |
| W4 | Capture-boundary reconciliation and bypass inventory | Direct write, trigger disable, owner/superuser path, rollback/restore, CDC slot loss, truncation, retention gap and observer outage |
| W5 | Privileged-path, identity-root and verifier-version challenges | Role escalation, grant substitution, stale/revoked grants, policy TOCTOU, wrong VDS, checkpoint rollback and digest-context confusion |
| W6 | Actor/site/operator control-domain separation | Dispatch/receipt mismatch, stale sensor/attestation, shared operator, unobserved actuation and sensor disagreement |

A missing implementation is not evaluated, not pass and not zero.

## 2. Mandatory metrics

Targets apply to the mandatory adversarial corpus. Zero observed failures in a finite synthetic corpus is a conformance result for that corpus, not a statistical guarantee about every deployment.

| Metric | Target | Measurement boundary |
|---|---:|---|
| Cross-tenant mutation false acceptance | 0 | Principal lacks named tenant/resource/action grant |
| Cross-tenant observer disclosure | 0 | Rows/state digests outside observer grant |
| Cross-tenant operation-ID replay acceptance | 0 | Same destination/operation ID with changed immutable binding |
| Unauthorized mutation acceptance | 0 | Invalid, missing, stale, expired, revoked or mismatched authority at commit |
| False independence | 0 | Shared or unknown mandatory control root |
| Unknown-to-success promotion | 0 | Missing, stale, disputed, contradictory or uncheckable evidence promoted to committed/qualified |
| False completeness | 0 | Bypass/gap inside declared capture boundary |
| Historical retroactivity error | 0 | As-of view changed by a correction not yet recorded at cutoff |
| Official vector disagreement | 0 | Independent maintained verifiers on same pinned vectors |

Publish coverage, recall and confidence intervals on independently annotated holdout data. Do not claim population-wide zero false-accept from a small fixture set.

## 3. Assurance-cost frontier

For each W0-W6 run publish:
- committed operations/sec and independently qualified effects/sec as separate rates;
- p50/p95/p99 mutation, readback and qualification latency;
- added latency per adapter and verification stage;
- CPU, memory, storage bytes per action, receipt/capsule size and verifier CPU;
- queue age, backlog, CDC/WAL lag, checkpoint lag and recovery time;
- cost per million dispatched actions and cost per million independently qualified effects;
- attempts in committed, refused, unresolved, contradicted, stale and uncheckable states;
- offline audit time with original operator/database/credentials removed.

Do not improve cost by dropping refusals, unresolved outcomes, failed attempts or disputed records.

## 4. Industry comparisons

- **DBOS:** reproduce its published W0 methodology (20K workflows/sec on one PostgreSQL server and 1M/sec across 50 shards in its 28 Oct 2026 vendor article). Those are no-op workflows with no steps: a W0 reference, not a W2/W6 comparator. Match hardware, database, sharding, worker concurrency, durability and cost.
- **Temporal:** use equivalent retry, lost-ack and worker-crash traces. Temporal says Activities may execute more than once and recommends idempotency at the called service. Test receiver-enforced identity and effect evidence; do not imply Temporal lacks orchestration durability.
- **PostgreSQL outbox/logical decoding:** measure commit-to-capture latency, slot retention, replay, rollback, truncation and recovery. An outbox/WAL record is capture evidence, not independent proof of downstream effect.
- **SCITT/Rekor/OAR/Agent Action Capsule:** use maintained verifiers and the same signed bundles/vectors. Do not compare a synthetic semantic checker with cryptographic verifiers or claim draft conformance without official-vector execution.

## 5. SOTA acceptance criteria

A credible result requires:
1. Matched W0 baseline with hardware, source commit, workload and resource budget published.
2. W1-W5 traces with equivalent attempts, faults, recovery, destination and readback semantics.
3. Zero false-success and unauthorized mutation acceptance across mandatory negative controls.
4. No false independence under declared shared-control attacks and no false completeness within the declared boundary.
5. Better or non-inferior latency/cost/throughput on a predeclared Pareto frontier while retaining stronger evidence guarantees.
6. Independent implementations, verifiers and hidden holdouts. Expected answers are authored separately from the implementation.
7. Offline third-party replay without original operator, database or credentials.

Proposed first gate: preserve the best matched W0 result within a predeclared 5% non-inferiority margin, then measure assurance cost at W1-W5. Five percent is a proposed target, not an observed result or industry standard. The long-term goal is to minimize cost per independently qualified effect subject to safety gates.

## 6. Independent holdout

Separate corpus author, implementation author, independent runner and auditor. Include false but well-signed sources, valid producer confirmations of spoofed responses, sink commits without readback, wrong-revision readback, shared control roots and actions omitted before the capture boundary.

The 15-case CPDC and MON-012 synthetic tests are starter fixtures, not independent holdout data and do not establish superiority over DBOS, Temporal, Rekor or other vendors.
