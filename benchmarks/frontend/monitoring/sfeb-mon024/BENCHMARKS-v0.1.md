# MON-024 Benchmark Plan v0.1

## 1. What is being measured

MON-024 is a proof-composition evaluator, not a database replication engine. Benchmark it on the correctness and cost of deciding whether an independently evidenced recovery claim is eligible for service activation. Run CDC throughput and recovery duration measurements in the underlying products as separate, matched experiments.

## 2. Locally observed semantic-reference preflight

Environment: Linux/amd64, Go 1.23.2, virtual AMD EPYC 9V74 host. Single exploratory run; not a portable performance claim.

| Check | Result |
|---|---:|
| `go test ./...` | pass |
| `go test -race ./...` | pass |
| `go vet ./...` | pass |
| Fuzz run, 3 seconds | pass; 92,576 executions |
| Qualified graph evaluation | 3,536 ns/op; 5,360 B/op; 14 allocs/op |
| Adversarial graph evaluation | 3,710 ns/op; 5,472 B/op; 17 allocs/op |

These numbers are a single local microbenchmark sample, not a statistically characterized distribution or a comparison to vendor products. Re-run with pinned Go version, CPU governor, kernel, repetitions and benchstat before using them in a performance claim.

## 3. Known-answer scenario corpus

`SCENARIOS-v1.json` describes positive and negative controls. The key invariant is zero false-qualified outcomes: no injected contradiction, missing required proof, stale/unknown evidence, scope hole, unresolved mutation or premature-activation condition may produce `may_enable_traffic=true`.

Positive controls matter just as much: a valid full-state claim with properly attested NotApplicable surfaces and a valid row-stream claim with explicit exclusions should qualify. This prevents a system from “passing” by rejecting everything.

## 4. Proposed benchmark targets (not achieved results)

| Dimension | Target | Qualification method |
|---|---|---|
| False assurance | 0 false-qualified across 10,000 seeded and generated fault cases | mutation/fault injection; independent expected outcomes |
| Positive-control precision | 100% of valid known-answer controls qualify | blind control set and exact verdict matching |
| Scope accounting | 100% of required surfaces accounted for; N/A never counted as Covered | inventory reconciliation against independent catalog observer |
| Evidence-graph integrity | 100% detection of missing, duplicate, substituted and cross-boundary parent receipts | edge and digest mutation corpus |
| Safety-gate behavior | 0 traffic enables when any prerequisite is failed, unknown, stale, synthetic or unresolved | model-based state transition and integration tests |
| Freshness and clock safety | 100% rejection/indeterminate classification for stale or out-of-bound clock evidence | clock skew, drift, expiry and partition experiments |
| Runtime overhead | p95 ≤ 1 ms and p99 ≤ 5 ms for the five-stage graph on a documented 2-vCPU reference | 30 independent runs, pinned environment |
| Memory overhead | ≤ 32 KiB transient allocation per five-stage evaluation | allocation benchmark and heap profile |
| Operator burden | ≤ 1 manual reconciliation per ordinary recovery drill; zero hidden/implicit approvals | timed operator study with event logs |

Targets are prospective acceptance criteria. We have not yet run the 10,000-case corpus, a blinded control set, a 2-vCPU reference benchmark or an operator study.

## 5. Fair industry comparison

Do not compare MON-024's microseconds per graph evaluation against Debezium's rows/s or pgBackRest's restore throughput. They measure different work. Instead, run *identical incident scenarios* through each stack and measure:

1. source event capture and replay continuity;
2. source identity, snapshot and timeline correctness;
3. state-surface completeness;
4. recovery-point correctness and achieved RPO;
5. database-ready and service-ready RTO separately;
6. detection of fault injection before traffic activation;
7. false-qualified and false-blocked rates;
8. evidence exportability, independent-verifier effort and operator recovery burden.

Required comparison profiles:

- PostgreSQL 18 and 19 logical replication / logical decoding;
- Debezium 3.7.1.Final with the tested Kafka Connect 4.3.1 compatibility profile;
- Kafka Connect source exactly-once enabled only where the connector supports it and in distributed mode;
- pgBackRest backup/PITR workflows;
- the same restore target, data volume, write workload, fault schedule and independent verifier across profiles.

Do not assume the same feature set across product combinations. Record connector versions, plugins, replication identity, publications/filters, Kafka EOS configuration, PostgreSQL settings, backup retention, restore topology and application dependencies.

## 6. Research experiments to pursue

- Crash after source event emission but before durable consumer offset; verify replay is duplicate-safe and no event is lost.
- Crash after durable evidence persistence but before ACK; verify replay deduplicates exact events and rejects conflicting identities.
- Promote/fail over with a timeline fork; verify no numeric-only LSN comparison crosses the fork.
- Change a publication, schema, sequence, large object, generated column, partition, extension-owned table or ACL independently; check that MON-022 updates its inventory evidence.
- Restore to a target before/after a committed transaction boundary; compare exact source/target state.
- Restore database while dependency is unavailable; ensure service claim remains unqualified.
- Leave old primary or stale endpoint reachable; verify activation is blocked.
- Inject uncertain command completion; verify no subsequent mutation before reconciliation.
- Corrupt or stale one receipt, parent digest, observer identity, clock domain or expiry; check verdict deterministically.

## 7. How to beat the state of the art responsibly

The differentiation hypothesis is *cross-product, independently verifiable recovery correctness*, not raw throughput. Existing tools should remain the execution engines. MON should aim to provide the strongest evidence boundary across them, with a measured reduction in false recovery claims and no unacceptable RTO/RPO or operational overhead penalty.

A superiority claim requires matched hardware, versions, source workload, target point, injected faults, independent truth oracle and at least 30 repeated trials per configuration. Report confidence intervals and all failures; do not rank products on dissimilar workload numbers.
