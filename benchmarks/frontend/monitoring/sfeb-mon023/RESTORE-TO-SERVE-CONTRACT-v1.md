# MON-023 — Restore-to-Serve Recovery Assurance Contract v1

**Status:** research contract; semantic Go reference and live recovery adapters pending.  
**Relationship:** composes MON-021 row-stream continuity and MON-022 database surface coverage into an independently qualified restore-to-service claim.

## 1. Research question

Can an independent verifier establish that a recovery operation restored the intended source state to a declared recovery point, met its RPO/RTO objectives, restored all required database surfaces, re-established dependencies, and passed application invariants before the service was called recovered?

The output is not a binary backup-success badge. It is a bounded recovery claim.

## 2. Three recovery claims

### 2.1 Artifact validated
Requires exact source cluster/database and backup-set identity; repository/backup manifest; integrity evidence bound to the artifact digest; required WAL/archive range from backup consistency point to requested recovery target; no known missing WAL interval; exact target and timeline lineage; and pinned tool/configuration profile.

This means inputs are suitable for a restore attempt. It does not imply PostgreSQL started or serves data.

### 2.2 Database recovered
Requires artifact validation plus:
- isolated restore target identity and proof it is separate from production;
- PostgreSQL reached a terminal consistent state (or an explicitly validation-only paused target);
- actual recovery point and source timeline lineage;
- requested vs achieved target comparison;
- database/catalog/system identity checks;
- MON-022 surface coverage for the declared claim;
- measured RPO and RTO within plan bounds;
- no unexplained replay or recovery error.

A process exit code alone is insufficient. A recovery target may pause for inspection; paused-for-validation is not service-ready.

### 2.3 Service recovered
Requires database recovery plus:
- required dependencies individually identified and ready (DNS, IAM, KMS, secrets, identity, storage, brokers, external APIs and secondary databases);
- intended application/configuration release running;
- read-only business invariants pass;
- controlled write/readback probes pass, with safe rollback or disposable state;
- outbox/queue state reconciled;
- old-primary fencing and stale endpoint control verified;
- independent service observer records traffic-ready/accepted state.

A database recovered without its dependencies is not service recovery. HTTP liveness while business invariants fail is not qualified.

## 3. Recovery point and lineage

A recovery point binds source cluster/system identifier, database, timeline and authenticated timeline history, requested target mode (LSN, transaction, named restore point, time or end-of-archive), exact target and inclusivity, actual achieved LSN/time/XID, and proof the achieved point belongs to the intended source history.

LSNs from divergent timelines cannot be compared as one linear number. Time-based targets require a common clock domain and bounded uncertainty. If archive recovery ends before the target, the target was not reached.

## 4. RPO and RTO

RPO is the age of the latest committed source state represented by the recovered database at the declared incident boundary.

Given a common clock domain with bounded uncertainty:
- incident upper bound = incident timestamp + incident uncertainty;
- recovery-point lower bound = source recovery-point timestamp - recovery-point uncertainty;
- conservative RPO upper bound = incident upper bound - recovery-point lower bound.

RTO is elapsed time from incident start to service-ready acceptance by an independent service observer—not backup download completion, restore command completion, PostgreSQL process start or a non-application HTTP health check.

Conservative RTO upper bound = service-ready upper timestamp - incident-start lower timestamp. Both objectives must be non-negative and within plan maxima. Incomparable clocks or unbounded uncertainty mean indeterminate.

## 5. Evidence model

### Plan
Plan ID/version/digest; recovery tier/scope; source/target cluster and database identity; backup method/config; target mode/value; maximum RPO/RTO; required MON-022 surface inventory; required dependencies; application invariant suite/version; traffic/fencing policy.

### Artifact/WAL
Exact backup/repository identity and digest; tool/config profile; page/archive integrity results; WAL start/stop and required interval; archive gap/error state; PITR target and timeline lineage; evidence digest and freshness.

### Restore execution
Isolated restore environment/control root; command/profile; start/end/consistency timestamps plus clock domain/uncertainty; process result and PostgreSQL recovery/paused/promoted state; recovered cluster/database/timeline identity; achieved recovery point; restore log/manifest and verifier references.

### Surfaces/dependencies/service
MON-022 inventory digest and per-surface result; excluded/not-applicable counts; dependencies with identity/control root/version/readiness; application release/config digest; invariant suite ID/version/digest; per-probe results; safe write probe/rollback evidence; independent traffic acceptance observer; service-ready time and clock uncertainty.

## 6. Decision semantics

| Claim | Required evidence | Does not imply |
|---|---|---|
| artifact_validated | Backup/WAL identity, integrity and target evidence | Restore completed |
| database_recovered | Isolated restore, consistency, lineage, surface scope and RPO/RTO | Dependencies or service ready |
| service_recovered | Database plus dependencies, invariants, safe probes and independent acceptance | Future availability outside test scope |

Decisions: qualified, denied, indeterminate, divergent. A lower claim may qualify when a higher claim does not. Artifact validation must never be promoted to database recovery, nor database recovery to service recovery, by inference.

## 7. Budget and safety

RTO components include detection/decision, backup/WAL retrieval, restore/replay, surface reconciliation, dependency recovery, application warmup/probes and operator fencing/promotion. Do not add average timings and call them guaranteed RTO; report p50/p95/p99 and worst observed.

Restore rehearsal must be isolated from production, use scoped/rotated credentials, private networking and controlled egress. Writes must be rolled back or contained in disposable state. Do not promote or accept traffic without explicit authorization, old-primary fencing and stale endpoint controls. A separate VM name is not independence proof: control root, credentials, storage and egress matter.

## 8. Industry comparison

| System | Existing capability | MON-023 addition |
|---|---|---|
| pgBackRest | Backup/archive consistency checks, repository verification, PITR | Actual restored point, RPO/RTO, surface proof, app invariants and service acceptance |
| AWS Backup Restore Testing | Scheduled restores and restore duration, optional validation | Source lineage, full surface scope, dependency/application correctness and conservative objective bounds |
| PostgreSQL recovery target pause | Inspect a recovery point before promotion/continue | Typed target/achieved point and independent invariants |
| Kubernetes health checks | Process/container liveness/readiness | Application semantics, data invariants, dependency lineage and recovery objectives |
| MON-021/022 | Row-stream continuity and non-row state coverage | Actual restore-to-serve proof |

Compare on identical backup set/WAL range, database size/version, topology, storage/network, encryption/compression, surface inventory, dependency set and app suite. Restore command duration alone is not RTO.

## 9. Benchmark programme

Finite-corpus goals:
- zero false database_recovered when requested target was not reached;
- zero false service_recovered when a mandatory dependency or invariant fails;
- zero RPO/RTO pass when conservative clock-uncertainty bound exceeds objective;
- zero traffic acceptance before fencing and dependency checks;
- 100% expected classifications on frozen cases, including positive controls.

Dimensions: 1 GB/100 GB/1 TB backups; full/differential/incremental; WAL rates/archive gaps; crash before/after restore, replay, pause, promotion, surface checks, dependency start, probes and traffic; KMS/DNS/IAM/secret/broker/external-service failures; invariant suites for referential counts, ledger balance, uniqueness, outbox consistency and read/write probes.

Prospective targets (not achieved): zero false service-recovered results on the finite corpus plus 500 blind faults; conservative RPO/RTO calculations; service-ready claims include every required dependency/invariant; recovery-to-serve p95 inside each declared tier objective; maximize qualified recovered services per total cost, not raw restore GB/s.

## 10. Implementation sequence

1. Freeze claim levels and evidence model.
2. Implement normalized Go evaluator for claims and conservative RPO/RTO.
3. Add deterministic fault schedules with independent expected outcomes.
4. Add pgBackRest repository/WAL adapter.
5. Add PostgreSQL 18/19 recovery target/timeline/catalog probes.
6. Consume MON-021 lineage and MON-022 surface results.
7. Add dependency and application-specific invariant adapters.
8. Add isolated orchestration, fencing, cleanup and independent readiness witness.
9. Run matched restore rehearsals and total assurance-cost benchmark.

## 11. Claim ceiling

The semantic evaluator consumes normalized evidence. It does not run backup/restore, authenticate cryptography, start PostgreSQL, measure actual application availability, validate secrets or guarantee future operation. No live pgBackRest, AWS Backup, PostgreSQL or application recovery rehearsal is claimed.
