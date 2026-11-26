# MON assurance matched benchmark protocol v1

**Status:** research protocol and target gates. No matched external-product benchmark has run. Targets below are proposed qualification goals, not measured results.

## 1. Research thesis

The field should move from *durable logs and retry orchestration* to **causally bound, independently verifiable effect assurance**.

The qualified chain is:

`versioned source assertion -> claim/transformation lineage -> separately authorized decision -> exact action intent -> atomic sink commit -> independent readback -> externally anchored completeness/checkpoint -> physical observation where applicable`

The point is not to replace a workflow engine, event log, database or transparency service. It is to qualify the semantic joins between them and state precisely what each evidence layer proves.

A workflow completion, signed receipt, valid citation, commit record, trace ID or producer response is not interchangeable with an independently observed real-world effect.

## 2. Fair industry comparison

| System / standard | What it establishes well | Important limit for this benchmark | MON integration stance |
|---|---|---|---|
| PostgreSQL outbox + logical decoding | Transactional change capture and durable database-to-consumer delivery patterns | A transaction or WAL event is not proof of a downstream side effect; replication slots, lag, retention, rollback and consumer acknowledgement must be included in failure tests | Use as W1/W2 destination and CDC baseline |
| DBOS | Durable workflow execution using PostgreSQL; published benchmark scales from 20K workflows/sec on one RDS Postgres shard to 1M/sec on 50 shards | Its Oct 28, 2026 public benchmark uses no-op workflows with no steps. It is W0 orchestration throughput, not an external-effect/readback result. The article also documents dequeue feedback-loop/metastability and its per-shard advisory-lock mitigation | Reproduce W0 under matched hardware, then add identical W1-W6 effects/faults |
| Temporal | Durable workflow orchestration, event history, retry policies and cross-SDK Activities | Temporal states that Activities may execute more than once and recommends idempotency at the called service; exactly-once observation of Activity completion does not mean exactly-once physical execution | Treat Temporal as a leading orchestration layer; test receiver-enforced operation identity and MON evidence adapter |
| Sigstore Rekor / transparency logs | Verifiable inclusion and append-only transparency mechanisms | Log inclusion is not factual truth, authorization, sink commit or physical state; checkpoint consistency and trusted witnesses are separate concerns | Reuse maintained transparency ecosystem; qualify checkpoint/position/consistency rather than inventing a receipt format |
| RFC 9942 / RFC 9943 | Published IETF standards for COSE receipts and trustworthy/transparent digital supply-chain architecture | A receipt proves only claims defined by the profile; RFC9162_SHA256 and MMR receipts have different proof semantics and must not be conflated | Interoperate through maintained libraries and official vectors |
| Open Action Record (OAR) | Action lifecycle evidence: decision, exact approval, authority recheck, state commitments and read-only confirmation | OAR -01 is an Internet-Draft; the public repo implements -00 constructions, not every -01 addition. Action receipts do not replace source/claim lineage or cross-layer join tests | Adopt OAR outcome and lifecycle semantics; MON qualifies source-to-effect composition |
| Figure Statements, Decision Provenance, Agent Action Capsule, Measurement Capsules | Emerging profile work for source figures, decisions, agent actions and independent measurement | Internet-Drafts are work-in-progress, not published RFCs; Measurement Capsules are authority-free and cannot authorize; an agent response digest is not proof of external-world state | Pin draft version and map semantics; no proprietary duplicate profiles |

### DBOS reference point

DBOS's Oct 28, 2026 article reports 20K durable workflows/sec on one PostgreSQL server and 1M/sec on 50 shards. The described configuration uses db.m7i.4xlarge RDS Postgres per shard (16 cores, 64 GB RAM), 48 worker processes per shard (2,400 at 50 shards), and explicitly simple no-op workflows with no steps. It also records a metastability problem under dequeue contention and the use of advisory locks to limit per-shard dequeue concurrency.

Therefore:
- report those numbers as a published vendor benchmark, not an independently reproduced result;
- use them as a W0 target/reference point only;
- do not compare them against W2-W6 throughput without adding the same effect, observer, cryptographic and fault workload to all systems;
- report hardware, cost, configuration, queue age, p50/p95/p99 and recovery alongside throughput.

### Temporal reference point

Temporal's Activity documentation recommends Activity idempotency. It explains that a worker can finish an Activity and crash before reporting completion, so that Activity may be retried; the called service must enforce an idempotency key. MON should make that destination-level operation ID and exact request digest observable and independently auditable, rather than claim orchestration systems remove retry ambiguity.

### OAR as the action-layer interoperability baseline

OAR -01 (25 Nov 2026) is now the preferred action-layer mapping. It defines authority-set references, commit-time re-checks, exact approval/display binding, before/after state commitments, read-only system-of-record confirmation, and the outcome classes `committed`, `refused`, and `unresolved`. The OAR public repository reports 18 main and 6 compatibility vectors. Its draft implementation-status section says the repository implements -00 constructions while new -01 additions remain planned. Accordingly, MON should run the current maintained verifiers over the public OAR 0.3/-00 vector corpus, then build a version-pinned -01 adapter once the new fields and vectors are implemented. Do not claim OAR -01 conformance from the current public implementation.

The MON-011 mapping is published in `OAR-INTEROP-v1.md`. MON does not need another action-receipt format: its unique contribution is source/claim/transformation lineage and exact causal joins from OAR-compatible action records to MON-009 sink commits and readbacks.

### RFC9162 vs MMR transparency profile

The SCITT WG's `draft-ietf-scitt-receipts-mmr-profile-00` is an active Internet-Draft (21 Nov 2026). It defines a distinct MMR VDS profile. Its protected `tree-size-2` rule applies to consistency receipts; it is not a generic substitute for every inclusion proof's tree-size metadata. An empty MMR inclusion path does not establish an exact leaf index by itself. MON's adapter should explicitly test domain-separated/position-bound leaf serialization, protected consistency target equality, proof shape and accumulator reconstruction. The separate qualification note is in MON-009 `TRANSPARENCY-PROFILE-QUALIFICATION-v1.md`.

For RFC9942 RFC9162_SHA256, reported Errata 9316/9317 concern tree-size/leaf-index position binding. A 25 Nov Internet-Draft notes that their status was not re-verified and describes an issue around size and position, not arbitrary membership forgery. We must pin the exact VDS profile and not generalize the report to all RFC9942 receipts.

## 3. Workload and qualification ladder

Every run must include successful, rejected, duplicate, lost-ack, timeout, crash, rollback, partition, failover, stale evidence, revocation and recovery cases. All implementations receive equivalent input traces, operation counts, state-transition semantics, hardware budget, and fault schedule.

| Tier | Workload | Required measurements / proposed safety gate |
|---|---|---|
| W0 | Durable no-op orchestration and queue throughput | Match DBOS single-/multi-shard methodology. Record workflows/sec, p50/p95/p99, CPU/RAM, cost per million, queue age, shard skew, lock/slot lag and post-perturbation recovery. This is not an effect claim. |
| W1 | At-least-once dispatch, retries, duplicate delivery, lost acknowledgement | Zero duplicate committed mutations per globally scoped operation ID; cross-tenant operation-ID reuse must conflict; prove idempotency at the receiving sink. |
| W2 | Atomic sink commit followed by separate readback | Every accepted action maps to an exact destination revision and state digest, or an explicit non-success/indeterminate result; zero false-success classification. |
| W3 | Independence / common-cause controls | Detect shared identity, privilege, operator, key custody, storage, network, account and administrator domains. Require zero false-independence claims under mandatory shared-domain negative controls. |
| W4 | Completeness / anti-entropy | External high-water marks, capture boundaries, crash recovery, tail truncation, rollback, retention gaps and observer outages. Zero false-complete results for the declared bounded interval. |
| W5 | Adversarial integrity | ID/digest substitution, authority revocation, policy TOCTOU, stale evidence, signer rotation/compromise, split-view log, checkpoint rollback, receipt size/index mutations, privileged bypass. All mandatory critical controls must fail closed. |
| W6 | Physical / external effect | Independently instrumented actuator or environment readback, separate calibration/observer provenance and safety constraints. Never use producer "confirmed" as physical success without independent effect observation. Unknown stays indeterminate. |

W0 throughput must always be reported separately from W6 qualified effect throughput. A system that has not implemented or safely tested a tier is marked **not evaluated**, not zero and not pass.

## 4. Claim Provenance and Drift Challenge (CPDC)

Extend the current 15-case synthetic preflight into an independently authored, versioned benchmark with eight capability classes:

- C0 — source authenticity and exact artifact/edition identity;
- C1 — semantic identity (unit, numerator/denominator, population, geography, time period, methodology);
- C2 — bitemporal valid-time and observed/recorded-time evaluation;
- C3 — publisher correction/retraction and pending-action propagation;
- C4 — independent dispute and authority-free measurement evidence;
- C5 — exact decision/approval bytes bound to exact action request;
- C6 — portable offline audit after operator/database/credential removal;
- C7 — deterministic transformation reproduction and dependency closure.

Mandatory metrics:
- false-current rate: target 0 on labeled correction/retraction corpus;
- missed safety-critical pending-action reauthorization: target 0;
- historical retroactivity errors: target 0;
- unauthorized effects / false-success: target 0 on mandatory controls;
- dependency closure recall: target 100% for labeled safety-critical edges; initial precision goal >=99%, then tighten with independent annotation;
- exact source/edition and transformation replay: target 100% on pinned corpus;
- offline audit reproducibility: target 100% on portable bundles;
- critical mutation kill rate: target 100%; overall >=99% is an aspirational goal after a sufficiently large, independently authored mutation corpus.

The current CPDC v1.1 result validates only the semantic rules encoded by its synthetic fixtures. The lineage-aware 15/15 exact result is expected by construction. It is not an external benchmark score.

## 5. Cross-layer Causal Evidence Join benchmark

The CEJI benchmark composes MON-009 and MON-010. A single test vector contains a source assertion, claim/transform path, decision, approval, intent, authority-at-commit evidence, sink ledger record, observer readback and optional transparency checkpoint.

A case passes only when the exact source digest, approval digest, principal, tenant, resource, mutation path, operation ID, request digest, policy revision/digest, commit-time authority scope and state revision/digest are joined across the complete chain. IDs, trace IDs and labels are insufficient without digest/reference binding.

Required mutation families:
- substitute each evidence ID and digest independently;
- cross-tenant and cross-resource replay;
- change principal, path, operation ID or request bytes at each edge;
- mismatch approval or policy revision;
- move authority validity/revocation across commit time;
- replace an independent observer with a shared operator/account/key/storage domain;
- change sink revision/state digest after readback;
- reorder timestamps, stale readback, truncate capture after final event;
- preserve valid inclusion proof while mutating unprotected RFC 9942 tree-size/leaf-index metadata and test checkpoint/index binding.

Target: zero accepted wrong joins, zero false independence, zero false-complete intervals, zero unauthorized acceptance. Unknown/uncheckable remains indeterminate.

## 6. Reproducible comparison protocol

1. Pin source commits, SDK/library versions, kernel/OS, database versions, schema, shard topology, worker counts and runtime configuration.
2. Publish the exact fixture and workload generator, including deterministic seeds and every injected fault.
3. Compare PostgreSQL outbox/logical decoding, DBOS, Temporal, maintained SCITT/Rekor adapters and the MON composition on equivalent effect workloads. Keep vendor defaults and separately document tuned configurations.
4. Run a W0-only no-op test, then W1-W5 digital effect tests, and W6 only in a safe instrumented physical testbed.
5. Perform at least five independent repetitions; publish raw traces and p50/p95/p99, throughput, queue age, recovery time, CPU/RAM, cost, evidence completeness and safety metrics.
6. Include negative controls for shared credentials, administrator bypass, log truncation, observer outage and post-commit lost acknowledgements.
7. Have a separate evaluator author a hidden holdout corpus. The system under test must not receive the gold expected outcomes during evaluation.
8. Report unexecuted, skipped and unavailable tests as not evaluated. No score may treat these as passes.
9. Publish both native system result and composed system result. The goal is not to weaken or replace orchestration; it is to add a verifiable assurance layer with measured overhead.

## 7. What would count as surpassing SOTA

A credible claim requires a matched benchmark, independently authored holdout data, reproducible adapter configurations and external review. Aim for a Pareto improvement:

- throughput/latency competitive with the same orchestration baseline at the same workload and hardware;
- zero false-success and zero unauthorized acceptance on mandatory safety vectors;
- higher dependency-closure recall and lower false-current rate;
- independent offline audit without original operator credentials;
- measured low overhead and bounded recovery under partitions/failover.

Do not claim "beats DBOS" by comparing a 15-case Go preflight to DBOS's no-op throughput. Do not claim "exactly once" from workflow completion semantics. Demonstrate exactly-once *committed mutation per operation key at the sink*, then independently read back the resulting state.

## References

- DBOS, *1M Durable Workflows per Second on Postgres*, Oct 28, 2026: https://www.dbos.dev/blog/one-million-workflows (vendor-published no-op benchmark).
- Temporal, *Activity Definition*, idempotency and retry semantics: https://docs.temporal.io/activity-definition
- RFC 9942, *CBOR Object Signing and Encryption (COSE) Receipts*: https://www.rfc-editor.org/rfc/rfc9942
- RFC 9943, *An Architecture for Trustworthy and Transparent Digital Supply Chains*: https://www.rfc-editor.org/rfc/rfc9943
- Figure Statements -00, Internet-Draft: https://datatracker.ietf.org/doc/html/draft-yetis-scitt-figure-receipts-00
- Decision Provenance -03, Internet-Draft: https://datatracker.ietf.org/doc/html/draft-lemonnier-scitt-decision-provenance-03
- Measurement Capsules -01, Internet-Draft: https://datatracker.ietf.org/doc/draft-templeman-scitt-measurement-capsule/01/
- Agent Action Capsule -05, Internet-Draft: https://datatracker.ietf.org/doc/draft-mih-scitt-agent-action-capsule/05/
- Open Action Record -01, Internet-Draft: https://www.ietf.org/ietf-ftp/internet-drafts/draft-malhotra-scitt-oar-01.html
- SCITT COSE Receipts for MMRs -00, Internet-Draft: https://datatracker.ietf.org/doc/draft-ietf-scitt-receipts-mmr-profile/00/
- SCITT pain-point resolutions -00, Internet-Draft (reported RFC 9942 errata discussion): https://datatracker.ietf.org/doc/draft-jones-scitt-pain-point-resolutions/00/

**Evidence boundary:** this document defines the fair comparison protocol and target gates only. It does not contain new benchmark measurements or product results.
