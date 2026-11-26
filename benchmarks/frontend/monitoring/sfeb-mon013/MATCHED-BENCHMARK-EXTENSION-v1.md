# MON-013 / MON-011 matched benchmark extension v1

**Status:** proposed workload and measurement protocol. No PostgreSQL suite or matched vendor comparison has run.

## 1. Outcome semantics under test

MON-013 adds an explicit outcome sequence:

- commit_pending: a stored procedure accepted a conditional mutation within the caller's still-open transaction; commit or rollback is not yet known.
- observed_committed: a separate observer reads the exact operation record and resulting state-history revision after commit.
- qualified: the effect is observed and authorization/policy validity at the actual database commit time is established.
- observed_committed with not_qualified or indeterminate: the state change occurred, but authorization cannot be established or was invalid.

Do not collapse these states into one success bit. Do not count a return from the SQL function as an independently verified effect.

## 2. Fair industry baselines

### DBOS

DBOS's 28 October 2026 vendor benchmark reports 20,000 no-op workflows/sec on one PostgreSQL server and 1 million/sec over 50 RDS shards. Each shard is an RDS db.m7i.4xlarge with 16 vCPU and 64 GB RAM; the 50-shard run uses 48 worker processes per shard. The workflow has no steps. This is a W0 orchestration reference, not a W2 external-effect or W6 physical-effect comparison.

The same article describes a metastable dequeue feedback loop around 8–16 shards / approximately 200K workflows/sec: a transient shard slowdown (for example, vacuum or IO) makes workers poll more, increases lock contention and can collapse cluster throughput. DBOS describes bounded dequeue concurrency using advisory locks as a mitigation. MON should include controlled VACUUM/IO perturbations, shard skew, lock wait, poll concurrency, queue age and recovery time in W0, not just peak throughput.

### Temporal

Temporal documentation explains that Activities can execute more than once, including a worker completing an Activity and crashing before reporting completion. It recommends idempotency at the called service. MON should integrate with Temporal rather than compare itself as a replacement: use stable receiver-enforced operation IDs, exact request digests and separate post-commit readback.

### PostgreSQL RLS

PostgreSQL documentation says table owners normally bypass RLS unless FORCE ROW LEVEL SECURITY is enabled, and superusers/BYPASSRLS always bypass it. Policies combine permissively with OR by default. MON's benchmark must test role membership, policy composition, owner and BYPASSRLS paths; an RLS test under an ordinary role does not qualify the privileged boundary.

### Outbox / logical decoding

Use PostgreSQL transactional outbox and WAL/logical-decoding baselines for capture completeness and consumer-retry tests. An outbox row or WAL change is durable database evidence, not by itself proof that a downstream service committed the corresponding effect or that a separate observer read the exact resulting state.

## 3. Workload ladder additions

| Tier | Workload / negative control | Required result |
|---|---|---|
| W1a | Same destination operation ID, simultaneous cross-tenant contenders | One durable operation identity; at most one mutation; all changed bindings conflict |
| W1b | Lost client acknowledgement followed by retry after transaction commit | Original outcome returned; no second state revision |
| W2a | Function returns before transaction commit; client rolls back | No post-commit operation/readback; outcome remains unresolved |
| W2b | Commit succeeds; separate observer reads exact operation and revision | observed_committed only with operation, request digest, revision and state digest match |
| W2c | Producer returns success but transaction rolls back or readback is missing | Never map to committed |
| W3a | Different login roles share the same DBA, cloud, storage, credential or operator root | Shared control, not independent |
| W4a | Commit timestamp setting disabled or metadata aged out by VACUUM | Effect may be observed; authority-at-commit stays uncheckable |
| W5a | Deferred check is forced early, then grant expires before commit | Post-commit timestamp check marks authority invalid / qualification not_qualified |
| W5b | Grant revoked concurrently with a mutation holding FOR SHARE | Revoke and mutation serialize; commit-time result agrees with recorded commit timestamp |
| W5c | Writer/observer attempts direct table access or privilege escalation | No unauthorized row disclosure or direct target-state mutation |
| W6 | Actor, actuator and independent site instrumentation | Separate physical-effect evidence; SQL readback is never physical proof |

W5a is important: deferred constraints can be forced immediate by a transaction participant. The database readback must evaluate actual commit time, not rely solely on the deferred trigger.

## 4. Mandatory measurements

Report per build, PostgreSQL version, schema hash, configuration, data scale and run:

- Dispatches/sec, transaction commits/sec and qualified effects/sec as three separate measures.
- p50/p95/p99 for authorization, lock acquisition, mutation, transaction commit, observer readback and final qualification.
- False-success, unauthorized acceptance, cross-tenant disclosure, duplicate committed mutation and false-independence counts.
- Commit timestamp coverage: fraction of observed commits with a retained timestamp; missing/aged-out percentage and age distribution.
- Time-of-check to commit interval; commit-time grant-validity failures; revoke-vs-commit race outcome.
- WAL bytes, outbox bytes, evidence bytes/action, index growth, vacuum cost and retained history cost.
- CPU, memory, storage, IOPS, connections, lock waits, deadlocks, queue age, replica lag, checkpoint lag and recovery time.
- Cost per million attempted actions and cost per million qualified effects.
- Offline audit reproducibility after the original operator, database and credentials are removed.

Do not exclude refused, timed-out, rolled-back, indeterminate, contradicted or uncheckable records from denominators.

## 5. Proposed targets

The following are qualification targets, not measured achievements or standards requirements:

- Zero duplicate committed mutations per (destination_id, operation_id) across the mandatory replay/contender corpus.
- Zero false-success or unauthorized-acceptance outcomes in mandatory W2/W5 controls.
- Zero cross-tenant mutation and readback disclosure in the mandatory RLS corpus.
- Zero false-independence claims when any required control root is shared or unknown.
- Zero false-complete outcomes within the published observation boundary.
- 100% exact operation/request/revision/state-digest agreement on the labelled readback corpus.
- 100% explicit uncheckable classification when commit timestamps or trust anchors are missing.
- Initial W0 non-inferiority goal: no more than 5% throughput regression against a matched orchestration baseline, provided workload and durability semantics are identical. This is a proposed target to revisit after measurement.
- Publish the assurance cost frontier rather than optimize only peak throughput.

A zero failure count on a small synthetic corpus is conformance to that corpus, not a population-wide security guarantee. Report corpus size, mutation score, holdout source, confidence intervals and independent review.

## 6. Pareto objective

Optimize qualified_effects_per_second divided by total assurance cost, subject to zero false success and unauthorized acceptance on mandatory tests; no weaker durability, transaction or retry semantics than the baseline; bounded and disclosed capture scope; portable offline verification; and explicit missing/indeterminate outcomes.

DBOS W0 and Temporal retry durability are strong baseline capabilities. MON's differentiator is stronger W1–W5 evidence assurance at a measured overhead, not a claim that the assurance layer replaces their orchestration engines.
