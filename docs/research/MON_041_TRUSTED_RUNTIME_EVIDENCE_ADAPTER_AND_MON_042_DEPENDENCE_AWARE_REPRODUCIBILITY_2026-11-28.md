# MON-041 — Trusted Runtime Evidence Adapter and Recovery Claim Attestation v0.1

Status: research contract and implementation plan. This document does not implement a live adapter, authenticated runtime receipt, production authorization, or recovery qualification.

## Objective

Bridge the offline MON-037–MON-040 evaluators to independently captured runtime evidence without allowing a candidate runtime, recovery operator, model, or evaluator to authenticate its own claims.

MON-041 specifies a trust-boundary protocol for PostgreSQL, Kafka, Debezium, backup/PITR, transaction, surface-inventory, service-probe, fault-injection, and custody evidence. It is an adapter and protocol boundary, not a new database, broker, CDC engine, backup engine, or production orchestrator.

## Core hypothesis

A recovery claim is admissible to the MON campaign only when every required evidence item is:
1. bound to one immutable recovery claim and recovery boundary;
2. captured by an identified principal with a current, scoped capability;
3. fresh at evaluation and protected against replay;
4. complete relative to an independently enumerated inventory and declared capture interval;
5. cryptographically authenticated from a trust root outside the recovery candidate;
6. linked to the exact software/configuration revision and dependency state;
7. independently checked by the relevant MON semantic verifier; and
8. authorized for the requested action by a separate authority gate.

Hashes establish byte identity, not origin, truth, freshness, completeness, independence, or authority.

## Architectural separation

Five roles have separate identities and permissions:

- **Subject:** the source/recovery system whose state is asserted.
- **Capture agent:** reads source, broker, backup and service evidence and emits raw receipts. It cannot issue trust roots or authorize promotion.
- **Custody verifier:** verifies signer identity, key status, scope, freshness, nonce, chain of custody and artifact availability. It cannot change semantic verdicts.
- **Semantic adjudicators:** MON-037/038/039/040 and prior MON evaluators. They consume authenticated evidence and emit scoped results; they cannot mint capture receipts or authorize actions.
- **Promotion authority:** evaluates policy over adjudicator results, current authority, revocation state and requested action. It cannot rewrite evidence or semantic results.

Symthaea may propose experiments, dependency hypotheses, likely failure interactions and next-best evidence requests. It is not a trust root, capture agent, semantic oracle, or promotion authority.

## Claim and boundary identity

Every capture session MUST bind:
- claim ID and trial ID;
- source cluster/database identity and recovery target identity;
- source epoch and PostgreSQL system identifier/timeline ancestry where applicable;
- backup artifact and snapshot identity;
- Kafka cluster ID, topic IDs, partition inventory and per-partition capture ranges;
- Debezium connector identity, configuration digest, schema-history identity and offset state;
- required database-surface inventory digest and transaction-report scope;
- service/invariant profile digest;
- exact code revision, binary/image digest, configuration digest and schema versions;
- declared time interval and capture policy version.

Any change to a bound identity or scope creates a new claim or explicit invalidation/revalidation event. It MUST NOT be silently normalized into the old claim.

## Challenge-response capture protocol

1. The custody service creates a cryptographically random, single-use challenge nonce and bounded capture lease.
2. The capture agent authenticates using a registered workload identity and presents a capability scoped to requested sources, claim, evidence classes and time window.
3. The adapter reads metadata and data from source APIs. It records exact API/version, request parameters, response status, principal, source identity and observed timestamps.
4. The adapter emits one signed receipt per evidence class plus a session manifest. Receipts bind the challenge nonce, claim/boundary digest, capture-agent identity, source identity, tool/API version, exact artifact digests, completeness declaration and monotonic sequence number.
5. The custody verifier checks signature chain, key validity at capture time, revocation, capability scope, nonce uniqueness, freshness, sequence continuity, manifest closure and independent artifact retrieval.
6. MON semantic verifiers consume only custody-admitted evidence. They preserve blocked, indeterminate and qualified as separate outcomes.
7. Promotion authority obtains a fresh policy decision and current revocation/authority snapshot. It authorizes only the exact action, target, claim digest and short-lived epoch named in the decision.
8. The enforcement point checks the authorization epoch and fencing token at the mutation boundary. Expired, replayed, revoked, stale or scope-mismatched decisions fail closed.

A lost acknowledgement after an external mutation remains an unknown outcome until independently reconciled. It MUST NOT create a new operation identity or bypass the original recovery boundary.

## Evidence receipt requirements

A signed receipt MUST include:
- protocol/profile version;
- receipt ID and parent/session-manifest ID;
- challenge nonce and single-use marker;
- claim, trial and recovery-boundary digests;
- capture principal and workload identity;
- source identity and evidence class;
- API/client/broker/server version and relevant configuration digest;
- capture start/end timestamps, clock source and clock uncertainty;
- monotonic sequence number and previous-receipt digest;
- declared range and completeness/inventory digest;
- artifact media type, byte length and SHA-256;
- signer key ID, signature algorithm and signature;
- key-status/revocation evidence reference;
- explicit limitations and unavailable fields.

Receipts are append-only. Corrections are new receipts that supersede prior receipts by explicit relation; prior bytes are never rewritten.

## Kafka capture adapter contract

The adapter MUST:
- retrieve cluster ID, topic IDs, partition inventory, leader epochs, log-start offsets, high watermarks and LSO using version-pinned APIs;
- capture read_committed records and explicitly record isolation level;
- bind a half-open range per partition, with end-exclusive offset at the verified LSO for the capture snapshot;
- preserve legal offset holes and transaction/control-marker semantics;
- capture aborted-transaction metadata where the API exposes it;
- bind producer-fence evidence to the same cluster/topic/partition scope and prove fencing precedes capture;
- detect topic recreation, partition-count drift, leader-epoch changes, truncation, retention movement and range instability;
- take repeated boundary observations or otherwise prove the capture snapshot is stable;
- reconcile each captured business record to independent SQL/CDC truth without inventing cross-partition order.

A read_committed consumer may stop at the LSO below the high watermark while transactions remain open. This is not by itself proof of missing committed business events. A capture whose boundary is moving or cannot be stabilized is indeterminate.

**V1 offset-hole disposition:** MON-038 v1 receives application-visible `read_committed` records and stable offset/LSO/HW metadata, but it does not contain raw record-batch/control-marker evidence sufficient to account for every exact offset slot. Therefore range-width minus visible-record-count is diagnostic only. Any positive slot count remains `offset_gap_slots_unclassified` and holds the component/campaign indeterminate. A hole MUST NOT be labeled legal merely because both the captured-record array and independently supplied event envelope omit the same event. A future v2 may classify a hole only with independently authenticated, version-pinned broker-log/record-batch evidence that accounts for the exact offsets as committed data, aborted data, or transaction control markers. The Kafka Fetch aborted-transactions list is useful metadata for filtering aborted records but, by itself, is not a complete per-offset explanation of every invisible control marker.

**Offset-hole accounting is a hard requirement:** the adapter/verifier MUST NOT infer that every slot absent from the returned record array is a legal control/aborted-transaction gap. Each hole in the half-open range must be accounted for by broker-provided aborted-transaction metadata, control-marker evidence, or another version-pinned and independently checkable explanation. If the capture API cannot expose enough information to classify a hole, report the count and exact ranges as unclassified and keep the partition/campaign indeterminate. A record-count subtraction such as (end offset - start offset - visible record count) is a diagnostic only; it is not proof that the missing offsets are legal. Include a negative control that removes an in-range business record and verify that the evaluator cannot classify that hole as benign merely because source/event evidence was also omitted or incomplete.

## PostgreSQL/source-truth adapter contract

The adapter MUST:
- bind database system identity, timeline and ancestry, WAL/LSN range, snapshot identity and server version;
- export source truth under a documented consistent-snapshot/isolation contract;
- bind exact query text or query digest, parameters, role, schema and row ordering/canonicalization rules;
- report transaction identities, commit status and closure information where available;
- export required database-surface inventory independently from the recovery candidate;
- record replication-slot/connector state and relevant WAL retention conditions;
- distinguish source snapshot time from capture time and recovery target time;
- preserve uncertainty where exact cross-system atomic snapshotting is unavailable.

No adapter may claim a globally atomic PostgreSQL-plus-Kafka snapshot unless a defensible protocol establishes that relationship. Otherwise the campaign must use explicit watermarks/causal cuts and return indeterminate when the cut cannot be established.

## Freshness, revocation and authority

- Capture leases are short-lived and scoped to one claim and evidence class.
- Challenge nonces are single-use; duplicate nonce use blocks.
- Key revocation, workload identity revocation, capability expiry or policy change invalidates dependent claims according to an explicit dependency graph.
- A key revocation effective at or before receipt capture invalidates that receipt. A later revocation does not retroactively invalidate a historical signature unless the incident policy establishes an earlier compromise-effective time; that effective time must be authenticated and recorded.
- The verifier rechecks revocation and policy immediately before issuing an action decision.
- The enforcement point checks the fencing epoch at the point of mutation; a preflight-only check is insufficient.
- Clock uncertainty is included in freshness decisions. Wall-clock timestamps alone do not establish ordering.
- A stale but authentic receipt is not admissible as current evidence.
- A valid receipt from a compromised or non-independent capture domain is not sufficient for independent adjudication.

## Failure and attack model

Required negative controls include:
- forged, unsigned, malformed and wrong-key receipts;
- valid signature from revoked/expired key;
- wrong tenant, claim, cluster, topic, partition or recovery boundary;
- nonce replay and receipt replay across trials;
- stale receipt and future-dated receipt;
- clock skew and clock-source compromise;
- truncated manifest, omitted inventory member and hidden extra member;
- artifact substitution after signing;
- API response pagination omission;
- topic deletion/recreation under the same name;
- partition expansion during capture;
- leader-epoch change, log truncation and retention advancement;
- LSO/HW movement during capture;
- producer fencing after capture rather than before it;
- source SQL snapshot drift and schema change during export;
- custody verifier outage and unavailable revocation status;
- authorization revocation racing with promotion;
- stale recovery worker attempting mutation after lease expiry;
- lost response after an external side effect;
- compromised capture agent attempting to self-assert completeness;
- Symthaea-generated plausible but false root-cause hypothesis.

Known policy violations block; missing, unavailable or unresolvable evidence holds indeterminate. Neither permits promotion.

## MON-041 outputs

- capture session manifest;
- authenticated evidence receipts;
- custody verification report;
- revocation snapshot;
- dependency-domain inventory;
- semantic adjudication bundle;
- promotion authorization, only when separately authorized;
- recovery claim lineage;
- machine-readable reason codes.

The campaign retains component verdicts and never collapses missing evidence into a clean pass.

## MON-042 — Dependence-Aware Reproducibility (planned companion)

MON-040's Clopper–Pearson result is valid only when Bernoulli trial units satisfy independent, identically distributed assumptions. Distinct seeds and replay IDs are necessary bookkeeping controls but do not establish independent trials.

MON-042 will specify:
- a preregistered experimental-unit definition (process, host, cluster, region, independent deployment, or other justified unit);
- dependency-domain manifests for host, hypervisor, storage, network, clock, identity, orchestration, control plane, image, configuration and external services;
- randomized blocking and stratification across domains;
- separate denominators for within-cluster repetitions and independent cluster-level replications;
- cluster-aware confidence intervals or hierarchical models when warranted by the sampling design;
- common-cause fault schedules and correlated-failure scenarios;
- sensitivity analysis under plausible intra-cluster correlation;
- a strict rule that the nominal 29/29 Clopper–Pearson result is reported only for a defensible independent Bernoulli unit;
- benchmark-the-benchmark tests for interval coverage under dependence, missingness, selection and adaptive stopping.

Do not replace Clopper–Pearson with a more complex model by default. Choose the estimator from the preregistered sampling design and publish assumptions, estimand, unit, dependence structure and raw observations.

## Benchmark programme

All targets below are proposed gates, not achieved results.

| Dimension | Proposed gate |
|---|---:|
| Receipt authenticity, scope, nonce and revocation negative controls | 100% detected |
| Required inventory omissions and additions | 100% detected |
| Stale, replayed and wrong-boundary evidence | 100% rejected |
| Runtime source/capture completeness | 100% required manifests admitted only when independently closed |
| Unauthorized promotion | 0 in 1,000,000 deterministic adversarial software mutations |
| Promotion race / stale fencing | 0 stale or revoked mutation accepted |
| Unknown-outcome handling | 100% remain unknown absent authoritative reconciliation |
| Capture stability | 100% unstable boundaries held indeterminate |
| Custody overhead | report p50/p95/p99 and bytes/CPU; target <= 10% p95 overhead on matched capture workload, subject to absolute SLO |
| Evidence verification throughput | report receipts/s and bytes/s at 1/10/100/1,000 partitions and 10^3/10^5/10^7 records |
| Dependency-domain classification | 100% known-answer controls; unknown domains remain unknown |
| Statistical coverage | nominal interval coverage tested over preregistered independent and correlated simulation families |

The million deterministic mutations are evaluator regression controls, not a million independent operational trials and not a rare-event probability estimate.

## Matched industry comparison

Compare the same pinned PostgreSQL, pgBackRest, Debezium, Kafka and infrastructure versions under identical data, snapshots, transaction workloads, fault schedules, RPO/RTO targets and independent source truth.

Arms:
1. Native backup/PITR and native verification.
2. Native Debezium/Kafka checks.
3. Row-count-only reconciliation.
4. Event-ID-only reconciliation.
5. MON-037–040 offline semantic evaluators.
6. MON-041 authenticated capture/custody plus MON-037–040.
7. Independent SRE/maintainer adjudication.

Report false-qualified, false-blocked, indeterminate, event completeness, transaction closure, surface coverage, stale/revoked evidence detection, promotion safety, RPO, RTO, throughput, p50/p95/p99, CPU, memory, storage, operator hours and total cost per independently supported recovery.

Do not compare a verifier microbenchmark with database restore throughput. Report assurance gain and operational overhead separately. A claim of superiority requires paired results and uncertainty intervals, not feature-count comparisons.

## Execution sequence

1. Freeze MON-037–040 exact heads and run their offline checks independently; preserve skipped/unexecuted statuses as pending.
2. Implement receipt and claim-boundary schemas plus canonicalization test vectors.
3. Implement a local deterministic custody verifier with synthetic trust roots, nonce replay controls, expiry and revocation fixtures.
4. Implement Kafka metadata/capture adapter behind read-only credentials; do not grant promotion rights.
5. Implement PostgreSQL source-truth adapter with read-only consistent snapshots.
6. Bind outputs through MON-026 custody and MON-037 composition.
7. Add a disposable integration environment and fault schedules; keep live credentials and production endpoints out of fixtures.
8. Add a separate promotion-gate prototype with no production authority, testing fencing and revocation races.
9. Start MON-042 experimental design before interpreting repeated runtime trials.
10. Run independent maintainer review and publish raw receipts, manifests, scripts, exact revisions and limitations.

## Explicit non-claims

This specification does not implement runtime capture, cryptographic key management, a production trust root, a revocation service, independent physical failure domains, a live Kafka/PostgreSQL adapter, a promotion gate, certification, or production qualification. Synthetic receipts are protocol fixtures only. Hashes are not signatures; signatures are not truth; truth is not authority; authority is not safe execution.
