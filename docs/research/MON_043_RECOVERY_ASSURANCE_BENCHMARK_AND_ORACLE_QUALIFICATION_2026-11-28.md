# MON-043 — Recovery Assurance Benchmark and Oracle Qualification v0.1

Status: benchmark protocol specification. No benchmark suite, live trial, competitor result, or SOTA claim is implemented or measured by this document.

## Objective

Create a reproducible, version-pinned, independently adjudicated benchmark for end-to-end recovery assurance. The benchmark must test whether MON-037–MON-042 detect unsafe recovery states that component-native checks and simpler reconciliation strategies miss, while quantifying false blocks, indeterminate decisions, operational cost and runtime overhead.

The benchmark evaluates assurance quality, not whether MON replaces PostgreSQL, pgBackRest, Debezium, Kafka, Jepsen, Chaos Mesh, or cloud fault-injection products.

## Primary research questions

1. Which cross-layer recovery faults are missed by component-native health and backup checks?
2. How much incremental detection does each MON layer contribute?
3. Does authenticated custody reduce forged, stale, replayed or scope-confused evidence admission?
4. Does partition-vector and causal-cut reasoning detect errors hidden by global-offset or row-count-only methods?
5. Do counterfactual reduction and repeatability improve diagnosis without overstating causality or statistical confidence?
6. What is the marginal latency, throughput, storage, operator effort and total cost of each assurance layer?
7. Are the benchmark oracles themselves sensitive to seeded defects and robust against evaluator gaming?

## Benchmark arms

All arms receive the same source workload, backup snapshot, database version, broker/connector versions, fault schedule, recovery target and independent ground-truth package.

| Arm | Configuration | Purpose |
|---|---|---|
| A0 | Native PostgreSQL backup/PITR and native backup verification | Component baseline |
| A1 | A0 + pgBackRest verification and restore workflow | Backup-tool baseline |
| A2 | A1 + Debezium/Kafka native capture and offset checks | CDC baseline |
| A3 | A2 + row-count reconciliation | Aggregate reconciliation baseline |
| A4 | A2 + event-ID reconciliation | Identity reconciliation baseline |
| A5 | A4 + transaction closure and partition-vector capture | MON semantic subset |
| A6 | A5 + MON-037–040 | Integrated offline assurance |
| A7 | A6 + MON-041 authenticated capture/custody | Runtime evidence admission |
| A8 | Independent SRE/maintainer adjudication using sealed source truth | External adjudication reference |

Where a product does not expose a comparable capability, mark it not applicable rather than silently substituting a different check. Report both native and augmented variants when augmentation is technically fair.

## Scenario families

The public benchmark should include a development corpus and a separately controlled holdout corpus. Every scenario has a manifest, expected disposition, fault schedule, source-truth package, recovery boundary, oracle version, and explicit evidence limitations.

### F1 — Backup and timeline lineage
- missing or corrupted backup artifact;
- incomplete WAL archive;
- wrong system identifier;
- wrong timeline or fork ancestry;
- invalid PITR target;
- backup from a different schema/configuration epoch;
- incremental backup dependency invalidated by version transition;
- valid backup bytes with unusable restored database.

### F2 — CDC event identity and transaction closure
- missing, extra, duplicate, conflicting and payload-mismatched events;
- BEGIN/END missing or inconsistent;
- END event_count mismatch;
- transaction total_order and collection_order gaps;
- logical-message accounting mismatch;
- schema-history drift;
- connector restart boundary error;
- source acknowledgement without durable capture.

### F3 — Kafka partition and transaction boundaries
- sparse offsets due to control/aborted records;
- read_uncommitted versus read_committed mismatch;
- LSO below HW due to open transaction;
- unstable LSO/HW during capture;
- leader epoch change and log truncation;
- topic deletion/recreation and partition expansion;
- producer fencing after rather than before capture;
- cross-partition transaction without a defensible causal cut;
- retention advancement during evidence collection.

### F4 — Full-state and service invariants
- omitted required database surface;
- unknown or unjustified N/A surface;
- restored state passes row counts but violates a business invariant;
- duplicate external financial effect;
- stale materialized view;
- service probe reaches the wrong cluster;
- traffic enabled before fencing;
- revoked principal retains write authority;
- rollback path unavailable or not independently tested.

### F5 — Evidence and trust attacks
- forged, unsigned, revoked, stale and replayed receipts;
- valid receipt bound to wrong claim or source;
- artifact substitution after receipt creation;
- capture agent self-asserts completeness;
- missing inventory item or hidden extra item;
- custody service unavailable;
- authorization revocation races with promotion;
- stale fencing token accepted;
- compromised shared failure domain presented as independent observer.

### F6 — Fault interaction and reproducibility
- single-fault controls;
- pairwise and higher-order interactions;
- non-monotonic fault interactions;
- equivalent failure signatures from different fault sets;
- non-equivalent signatures sharing a generic blocked verdict;
- deterministic and stochastic fault schedules;
- repeated trials within one host/cluster and across independent domains;
- missing, censored and incomplete trial observations.

### F7 — Operational and economic recovery
- degraded but safe recovery;
- full restore versus PITR;
- replay from CDC versus backup;
- partial service restoration;
- read-only service before write service;
- recovery under constrained CPU, memory, storage and network budgets;
- operator intervention and manual evidence collection.

## Oracle qualification

No benchmark score is meaningful until its oracle is qualified.

Required oracle controls:
- positive controls for every expected verdict class;
- negative controls for every required invariant and evidence boundary;
- seeded mutations against the evaluator implementation;
- mutation score by fault family, not only aggregate score;
- metamorphic tests for invariant-preserving transformations;
- independent reference implementation or independently authored adjudication for critical claims;
- blinded holdout cases;
- evaluator version pinning and mutation corpus digest;
- explicit unresolved cases and oracle disagreement.

A mutation is counted as detected only when the evaluator emits the exact expected disposition and reason family. A generic blocked verdict does not count as detecting the correct fault class. Indeterminate is not a successful detection when the expected result is a known block, and vice versa.

Do not let the system under test generate the ground truth, expected verdicts, or mutation labels for its own evaluation.

## Primary metrics

### Safety and correctness
- false-qualified rate: unsafe scenarios incorrectly admitted;
- false-blocked rate: safe scenarios incorrectly rejected;
- indeterminate rate: scenarios for which evidence is insufficient;
- exact disposition accuracy by fault family;
- invariant violation escape rate;
- forged/stale/replayed evidence acceptance;
- cross-claim and cross-tenant evidence confusion;
- unauthorized promotion and stale-fence acceptance;
- event and transaction reconciliation completeness.

### Recovery performance
- RPO distribution and achieved recovery point;
- RTO distribution and time to read-only/write-ready service;
- time to establish evidence completeness;
- time to independently adjudicate;
- replay/reconstruction time;
- time to detect an unsafe promotion candidate.

### Resource and operational cost
- throughput and recovery throughput;
- p50/p95/p99 latency;
- CPU, memory, disk I/O, network and storage growth;
- evidence bytes per event/transaction/partition;
- verifier cost per million records;
- operator hours and manual interventions;
- total cost per independently supported recovery.

### Diagnostic quality
- fault-set size before/after reduction;
- number of replays required;
- exact-signature preservation rate;
- number of reduction orderings explored;
- one-minimal versus exhaustive global-minimum claim;
- diagnosis stability across independent runs;
- unresolved competing explanations.

Never collapse these metrics into one score that allows a speed gain to compensate for an unsafe promotion.

## Statistical design

- Preregister the primary estimand, experimental unit, sampling frame, sample size, confidence level and stopping rules.
- Distinguish deterministic software mutation tests from stochastic operational trials.
- Distinguish repeated runs within one cluster from independent cluster/deployment replications.
- Use paired scenario schedules for baseline comparisons.
- Randomize arm order where carryover effects are controlled; otherwise use fresh equivalent environments.
- Record shared dependency domains and estimate or bound intra-cluster dependence.
- Use exact binomial intervals only where the Bernoulli independence assumptions are defensible.
- Use cluster-aware or hierarchical methods when the experimental design requires them.
- Publish all failures, missing trials, exclusions, censored outcomes and deviations from preregistration.
- Keep development and holdout scenarios separate; do not tune thresholds on holdout results.

A 10,000- or 1,000,000-case deterministic mutation campaign measures evaluator regression sensitivity over a finite mutation set. It is not an operational rare-event probability estimate.

## Proposed gates (not achieved)

| Gate | Proposed target |
|---|---:|
| Known-answer positive controls | 100% exact disposition |
| Critical invariant negative controls | 100% exact detection |
| Unsafe promotion in deterministic mutation corpus | 0 escapes |
| Forged, stale, replayed or wrong-scope receipt admission | 0 accepted |
| Required surface omission acceptance | 0 accepted |
| Missing acknowledged business event acceptance | 0 accepted |
| Transaction closure violation acceptance | 0 accepted |
| Unstable Kafka capture boundary acceptance | 0 accepted |
| Stale/revoked fencing acceptance | 0 accepted |
| False-blocked rate on sealed safe controls | <= 1%, with confidence interval |
| Indeterminate rate on fully evidenced controls | <= 1%, with confidence interval |
| p95 overhead on matched workload | <= 10% target, plus absolute SLO |
| Cost per independently supported recovery | Reported and compared; no arbitrary universal threshold |
| Oracle mutation sensitivity | 100% for the declared critical mutation set |

Zero escapes in a finite deterministic suite is a regression gate, not proof of zero production risk. Any false-qualified critical case blocks benchmark qualification regardless of average score.

## Matched experiment protocol

1. Freeze software versions, configuration, hardware, storage class, durability settings and source workload.
2. Generate and independently retain source truth before fault injection.
3. Snapshot the backup, WAL, schema, Kafka topic/partition and connector state.
4. Preregister scenario, fault schedule, recovery boundary, expected invariants and stopping conditions.
5. Execute each arm in an equivalent isolated environment.
6. Capture raw histories, receipts, manifests, logs and service probes.
7. Run each arm's native checks without access to the sealed oracle labels.
8. Run MON semantic and custody evaluators on the same evidence budget.
9. Have independent adjudicators inspect disputed and holdout cases.
10. Publish raw artifacts, exact commands, tool versions, seeds, environment manifests, exclusions and analysis scripts.
11. Repeat across independent failure domains for confirmatory operational claims.
12. Publish failures and negative results, not only successful recovery demonstrations.

## Benchmark-the-benchmark

Before any public leaderboard:
- verify that each seeded defect is actually present in the test environment;
- verify that each negative control can trigger the intended failure;
- verify that positive controls remain accepted;
- test whether a deliberately weakened evaluator can still pass;
- test whether an evaluator can pass by returning indeterminate for every case;
- test whether a system can game the score by omitting hard scenarios;
- test oracle disagreement and adjudicator consistency;
- publish mutation and holdout corpus digests;
- prohibit self-generated expected labels without independent confirmation.

## Reporting and comparison rules

Publish per-arm and per-fault-family confusion matrices. Report exact denominators and uncertainty intervals. Separate component capability, end-to-end assurance, operational performance and cost.

A claim that MON exceeds a baseline requires:
- identical recovery target and source truth;
- identical fault schedules and evidence budgets;
- pinned compatible versions;
- independent adjudication;
- paired effect sizes and uncertainty intervals;
- no critical safety regression;
- measured operational overhead.

A claim that MON beats SOTA is prohibited until the benchmark has a public protocol, sealed holdout, independent replication, and comparable measured baselines.

## Execution roadmap

1. Create benchmark schemas and scenario manifest format.
2. Convert the existing MON-037/038/039/040 synthetic scenarios into a common index without changing their source fixtures.
3. Add F1–F7 family manifests and known-answer labels.
4. Build independent oracle controls and evaluator mutation harness.
5. Implement a local runner that executes only deterministic offline controls.
6. Add matched PostgreSQL/Kafka lab adapters after MON-041 custody is available.
7. Add Jepsen-style operation histories and consistency checks for application-visible histories.
8. Add Chaos Mesh/AWS FIS experiment adapters only as fault executors; retain MON as evidence adjudicator.
9. Seal holdout corpus and preregister statistical design.
10. Run independent maintainer/SRE replication before publishing comparative claims.

## Explicit non-claims

This document does not implement a benchmark runner, public leaderboard, sealed holdout, independent oracle, live database/broker experiment, competitor measurement, certification, or SOTA result. All thresholds are prospective. Existing synthetic scenarios remain development fixtures until their exact heads and checks are executed and independently reviewed.
