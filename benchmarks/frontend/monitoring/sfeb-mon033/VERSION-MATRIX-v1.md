# MON-033 — Debezium PostgreSQL Restart Regression Version Matrix v1

**Evidence cut:** 2026-11-27  
**Purpose:** separate the historical/fixed upstream code paths from actual MON-033 runtime observations.

## Upstream version facts

| Version | Upstream classification | MON-033 role | Runtime status |
|---|---|---|---|
| 3.7.0.Final (2026-09-29) | Pre-3.7.1 candidate; 3.7.1 release notes list DBZ-2889 and DBZ-2894 as fixes | Candidate reproduction arm; pin exact image/source digest | Not run |
| 3.7.1.Final (2026-11-06) | Stable 3.7 release; release notes list DBZ-2889 and DBZ-2894 among fixes | Stable fixed-version regression control | Not run |
| 3.8.0.Alpha1 (2026-11-02) | Development prerelease; release notes list DBZ-2889 among fixes | Optional forward-compatibility control | Not run |
| 3.8.0.Alpha2 (2026-11-19) | Development prerelease; release notes list DBZ-2894 and DBZ-2901 among fixes | Separate decoderbufs and COPY/logical-message follow-up lanes | Not run |

Debezium 3.7.1.Final was built against Kafka Connect 4.3.1 and tested with Kafka brokers 4.3.1. Debezium 3.8.0.Alpha2 reports the same Kafka Connect/broker pairing. Keep that pairing fixed for the first controlled comparison; test other Kafka versions as explicit compatibility dimensions.

## Upstream references

- [Debezium 3.7 release notes](https://debezium.io/releases/3.7/release-notes) — 3.7.1.Final release date and Kafka compatibility; DBZ-2889 and DBZ-2894 are listed under fixes.
- [Debezium 3.8 release notes](https://debezium.io/releases/3.8/release-notes) — 3.8 is development; Alpha1 lists DBZ-2889, Alpha2 lists DBZ-2894 and DBZ-2901.
- [DBZ-2889](https://issues.redhat.com/browse/DBZ-2889) — PostgreSQL restart-loss issue.
- [DBZ-2894](https://issues.redhat.com/browse/DBZ-2894) — decoderbufs BEGIN / stored-LSN interaction.
- [DBZ-2901](https://issues.redhat.com/browse/DBZ-2901) — DML lost after restarting from a transactional logical message following COPY.

Release-note classification does not establish the exact first affected patch or prove that a particular lab build reproduces a bug. The 3.7.0.Final arm is therefore a hypothesis to validate by source diff and live fault injection, not a pre-declared failure.

## Matched test plan

1. Pin the full Debezium connector artifact SHA-256, container image digest, Kafka Connect and broker patch versions, PostgreSQL exact major/minor, JDBC/decoder plugin, JDK, connector config digest, and replication-slot identity.
2. Use one PostgreSQL image/data volume snapshot and deterministic transaction script for each paired trial. Reset to a known state between trials.
3. Exercise separately:
   - pgoutput row-event boundary with `lsn_events_processed=1`;
   - pgoutput transaction-scoped logical MESSAGE boundary;
   - decoderbufs BEGIN sharing the stored LSN (DBZ-2894);
   - COPY followed by transactional logical MESSAGE and DML restart boundary (DBZ-2901).
4. Capture the exact target offset before restart; if a target cannot be reached, report indeterminate and do not substitute a nearby state.
5. After restart, reconcile the independent SQL primary-key/version truth set against every row, message and transaction event. Count missing, duplicate, conflicting and out-of-order events separately.
6. Repeat each cell at least 100 times initially, randomize version order, and retain raw evidence. Expand to 1,000+ per cell before making low-rate reliability claims.

## Acceptance

- Pre-fix candidate: reproduce only when the exact trigger state is observed and SQL-vs-CDC truth shows the predicted loss/duplication.
- 3.7.1.Final control: reach the same boundary and show no loss/duplicate under the same workload, with complete event-set reconciliation.
- Alpha versions: exploratory only; do not silently blend prerelease outcomes into stable product claims.
- All versions: zero false-qualified trials; the evaluator must mark incomplete, ambiguous or unobserved cases as indeterminate, never pass.

At 0 false qualifications across 10,000 independent unsafe trials, the one-sided 95% upper bound on the false-qualification probability is still approximately 0.03%; zero observed failures is not proof of zero risk.

## What this work does not claim

No version in this matrix has been run in the MON lab by this PR. The current local checks validate shell/offset predicate logic only. This matrix is the experiment design and upstream release-note mapping, not a completed compatibility certification.
