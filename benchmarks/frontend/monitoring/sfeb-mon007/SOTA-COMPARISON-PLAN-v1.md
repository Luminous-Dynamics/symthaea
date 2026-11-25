# SFEB MON-007 matched SOTA comparison plan

**Status:** benchmark protocol proposal; no end-to-end SOTA result has yet been measured.

## 1. Comparison question

Do not compare the pure semantic evaluator's ~39k evaluations/s or the Go Merkle microbenchmarks to workflow throughput. Those numbers measure different units of work and omit persistence, network, cryptography, transparency registration, sink effects and independent readback.

The target question is:

> At equivalent durability and failure semantics, what additional cost does independently qualifying an external action impose, and how much stronger is the evidence an auditor can verify after the original operator is unavailable or untrusted?

## 2. Baseline arms

Run these arms on the same machines, database shape, external sink, data, concurrency and fault schedule:

1. **PostgreSQL transactional outbox:** application transaction + outbox + unique operation key; dispatch and reconcile through an idempotent receiver.
2. **DBOS durable transaction:** transactional database state/queue and DBOS step; use the vendor's documented external-effect/idempotency pattern.
3. **Temporal Activity:** deterministic workflow with Activity retries and stable idempotency key; persist the workflow event history.
4. **SCITT / transparency only:** same signed action statement registered in the chosen Transparency Service; no assertion that this proves the effect.
5. **Rekor / transparency only:** same artifact/statement registered in a transparency log with authenticated checkpoint and inclusion proof.
6. **SFEB over DBOS.**
7. **SFEB over Temporal.**
8. **SFEB over the PostgreSQL outbox.**
9. **Native control:** direct conditional sink write/readback, to isolate orchestration cost from assurance cost.

Do not compare a DBOS no-op workflow to an SFEB operation with signatures, a real external write, proof registration and readback. Those are different workloads.

## 3. Workload tiers

| Tier | Workload | What it isolates |
|---|---|---|
| W0 | No-op durable workflow, no steps | Raw orchestration throughput; useful context only |
| W1 | One transactional database write | Durable state and outbox overhead |
| W2 | Conditional external write with stable idempotency key | Actual side effect and replay handling |
| W3 | W2 + independent readback + signed action statement | Effect qualification cost |
| W4 | W3 + transparency registration + inclusion + consistency verification | Complete evidence path |
| W5 | W4 + independent witness observations and checkpoint/key-history checks | Non-equivocation and root continuity |
| W6 | W5 + operator data-store loss/compromise and offline audit | Evidence portability and operator-independence |

Use small, medium and large payload classes (for example 1 KiB, 16 KiB and 256 KiB) and a stated number of evidence nodes. Report exact payload distributions and proof sizes.

## 4. Fault and adversarial schedule

Inject faults at every boundary, with deterministic seeds and a published attempt ledger:

- before authorization, after authorization, during revocation, and at sink commit;
- after sink commit but before acknowledgement, after acknowledgement but before readback, and after readback but before transparency registration;
- worker crash, process kill, host restart, network partition, delayed/lost/duplicated messages, clock skew and database failover;
- lost or duplicated idempotency key, stale fence, resource/tenant substitution, inconsistent read replica, and readback principal compromise;
- transparency registration timeout, stale checkpoint, same-size fork, invalid consistency proof, witness outage, correlated witness compromise and root rollback;
- operator database snapshot restore, log deletion, and operator refusal to cooperate after action completion.

Include negative controls. An auditor must reject forged receipts, stale authority, cross-tenant evidence, contradictory readbacks, invalid proof paths, old roots, same-size forks and evidence that only the original operator can resolve.

## 5. Metrics

### Performance and cost

- intent acceptance/s, sink-accepted effects/s, independently effect-verified actions/s, externally anchored actions/s;
- end-to-end p50/p95/p99 at each workload tier;
- recovery time after each fault, time-to-reconcile, time-to-anchor, and age of unresolved `indeterminate` actions;
- CPU-seconds, memory, database IOPS, network egress, signing/verification CPU, witness cost, stored bytes and total cost per qualified action;
- proof bytes per action, evidence graph node count, and verifier memory/time versus tree size and chain depth.

### Assurance quality

- false-accept count (target zero observed in the campaign), false-reject count, and indeterminate rate;
- duplicate external effects, stale-authority writes, cross-tenant acceptance, and post-revocation writes;
- percentage independently verifiable with the operator's services unavailable;
- percentage auditable from only the portable package plus public/trusted roots;
- time for a fresh auditor to reproduce the result without credentials to the original orchestrator;
- checkpoint equivocation detection time and witness failure-domain diversity.

## 6. Proposed qualification gates (targets, not achieved)

| Gate | Proposed target | Measurement discipline |
|---|---|---|
| Unauthorized effect | 0 accepted in 1,000,000 seeded negative attempts per release candidate | Record the full attempt corpus and all outcomes; not a proof of zero risk |
| Cross-tenant effect | 0 accepted | Include identity, resource and readback substitutions |
| Revocation/fencing race | 0 stale writes accepted at sink | Race revocation against commit with sink-side enforcement |
| Duplicate effect | 0 duplicates under replay of the same operation ID | Verify actual sink state, not only workflow history |
| Merkle portability | 100% of accepted proofs verify in two independent implementations | Use official vectors plus generated and mutated vectors |
| Evidence portability | 100% of qualified bundles independently audit without operator DB | Fresh auditor, fresh process, no operator secrets |
| p95 overhead | Initially target ≤10% for cached/local qualification and ≤20% for full external proof path versus the matched same-semantics baseline | Report overhead by stage; revise only with documented evidence |
| p99 recovery | Within the declared recovery SLO for every injected fault | No silent reset of unknown outcomes |
| Evidence retention | Every action's policy/key/checkpoint context reconstructible for the declared retention period | Test expired keys, policy updates and root rotations |

The million-attempt campaign is a qualification test, not a statistical guarantee or a substitute for formal analysis and code review.

## 7. Published industry claims and correct interpretation

- DBOS published a benchmark claiming over 1 million durable workflows/s using 50 PostgreSQL shards, each an AWS `db.m7i.4xlarge` RDS PostgreSQL instance, with 2,400 worker processes and simple no-op workflows with no steps. This is a vendor-published raw orchestration benchmark; it is not an external mutation, cryptographic evidence or readback benchmark. It is still a valuable W0 reference and a test of SFEB's ability to scale as an overlay. Source: https://www.dbos.dev/blog/one-million-workflows
- Temporal documents that Activities may execute more than once under retries; it recommends idempotency keys and describes the distinction between an Activity being observed as completed once and its code executing once. Source: https://docs.temporal.io/activity-definition
- Sigstore Rekor already provides a transparency log, signed tree-head information, inclusion proof verification and integrity tooling. SFEB should integrate such infrastructure rather than duplicate a generic log. Sources: https://docs.sigstore.dev/logging/overview/ and https://docs.sigstore.dev/logging/cli/
- RFC 9942 defines COSE Receipts for VDS proofs; RFC 9943 defines an interoperable single-issuer signed-statement transparency architecture. SFEB should build its action semantics and effect qualification above those formats and services, not fork them. Sources: https://www.rfc-editor.org/rfc/rfc9942.html and https://www.rfc-editor.org/rfc/rfc9943.html

## 8. Paradigm-shift contribution

The main result should be an **assurance gap benchmark**: execute an action, then remove trust in the original orchestrator and its database. Give an independent auditor only the portable evidence package and the relevant trusted roots. Measure which claims can be reproduced, which must remain unknown, what contradictory evidence is detected, and how long/costly the audit is.

This shifts the field's central metric from "how many workflows were recorded?" to "how many consequential external actions remain independently auditable after the operator is no longer trusted?"

## 9. Reproducibility requirements

Publish hardware/firmware, OS/kernel, runtime/compiler, database version/config, durability settings, shard topology, worker topology, queue semantics, sink implementation, payload sizes, concurrency, retries, fault seeds, raw samples and analysis scripts. Use at least five independent runs per performance point, confidence intervals, and a preregistered workload protocol. Do not mix vendor claims, local microbenchmarks and end-to-end results in one leaderboard column.
