# MON-025 Live Recovery Benchmark Plan v0.1

## 1. Purpose

Compare end-to-end recovery assurance on a matched fault schedule. Do not compare unlike operations such as Kafka events/s, PostgreSQL commits/s, pgBackRest MB/s and MON proof-evaluation nanoseconds as if they shared one performance dimension.

## 2. Matrix

| Axis | Baseline | Extension |
|---|---|---|
| PostgreSQL | 18.7 | 19.0 |
| Kafka | 4.3.1 (Debezium 3.7.1 tested pairing) | 4.3.2, then 4.4.0 after compatibility validation |
| CDC | Debezium PostgreSQL connector 3.7.1.Final | Native PostgreSQL logical replication, matched output semantics |
| Backup | PostgreSQL physical base backup + WAL archive | pgBackRest 2.60.1 |
| Target | same-version isolated restore | separately measured cross-major migration is not part of v0.1 |
| Evidence | independent source and target SQL observers | observer/common-cause adversarial lane |

## 3. Proposed acceptance targets (not measured)

| Measure | Target |
|---|---:|
| False-qualified challenge runs | 0 |
| Positive controls correctly accepted | 100% |
| Missing or conflicting expected event IDs | 0 |
| Unknown/failed prerequisite followed by service enable | 0 |
| Full-state inventory accounting | 100% |
| Recovery point correctness | exact target or explicit bounded gap; never silently rounded forward |
| RPO measurement | report p50/p95/p99 and worst-case upper bound |
| Database-ready RTO | p50/p95/p99 and worst case, separate from service-ready |
| Service-ready RTO | p50/p95/p99 and worst case, separate from database-ready |
| Independent evidence export | 100% of runs with raw artifact hashes and run manifest |
| Operator interventions | report per run; target median ≤1 for baseline drill |

## 4. Required experimental design

- At least 30 independent repetitions per cell for exploratory variance; use a power calculation for confirmatory claims.
- Pair all candidate stacks on same initial data, generated event IDs, transaction shape, write rate, target point and injected failure.
- Pin image digests, source commits, PostgreSQL settings, connector config, Kafka Connect worker config, hardware, filesystem and Docker version.
- Randomize experiment ordering and isolate persistent state between runs.
- Include baseline/no-fault and positive controls as well as fault cases.
- Record all missing, failed, timed-out and indeterminate runs. Do not delete bad runs.
- Report cluster-aware confidence intervals; episodes sharing the same database reset, failure injection mechanism or host are not automatically independent samples.

## 5. Performance and correctness measures

1. **Capture correctness:** expected unique event IDs, payload digest matches, duplicates, conflicting identities, missing IDs and transaction-envelope closure.
2. **Capture performance:** event throughput, commit-to-Kafka latency p50/p95/p99, WAL retained bytes, slot lag, Kafka bytes, CPU, memory and disk I/O.
3. **Recovery correctness:** requested vs achieved recovery point, database identity, timeline lineage and per-surface source/target digest comparison.
4. **RPO:** incident upper time minus recovery-point lower time, with clock uncertainty.
5. **RTO:** database-ready and service-ready measured independently, using bounded common clock domain.
6. **Safety:** false-qualified, false-blocked and indeterminate outcomes; detection latency for each fault.
7. **Operations:** operator interventions, manual reconciliations, restore retries, storage consumed and total cost.

## 6. Anti-shortcut and adversarial checks

- No-op workload and empty database control must not score as event capture.
- Deterministic fixture IDs prevent accidental success from row counts alone.
- Equal row counts with altered values must fail the digest check.
- A duplicate replay with identical payload is reported as replay; conflicting payload under the same event ID is a hard mismatch.
- Missing and out-of-order transaction records must be identified separately.
- A restored target is compared to source truth at the exact requested recovery point, not against source's later live state.
- A service process can be healthy while dependencies, fencing or invariants are broken; such a run must not pass service readiness.
- Benchmark scorer and expected data must be controlled separately from the implementation under test.

## 7. How to beat current practice

Our hypothesis is a measurable reduction in false recovery assurance through evidence composed across systems. The first success criterion is not beating PostgreSQL or Debezium throughput: it is detecting incorrect recovery claims that each component's own health endpoint may not detect.

To claim an improvement over a vendor or open-source baseline, replay the same injected failure set and use one independent truth oracle for every system. Report:
- false-qualified rate and false-blocked rate;
- missed events and conflicting replay count;
- RPO/RTO and service-readiness;
- operational intervention count;
- total compute/storage/egress and evidence-retention overhead.

Do not publish a “SOTA” label until these measurements are independently reproduced and all compared systems have matched configuration and resource budgets.
