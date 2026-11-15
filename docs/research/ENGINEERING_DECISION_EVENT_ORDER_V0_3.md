# Event-order v0.3: deterministic virtual-time replay

**Status:** research implementation, synthetic schedules only. This document describes authored source and intended gates; it is not a claim that Rust tests, CI, clock adapters, owner receipt verification, or physical systems have passed.

## Why v0.3 exists

v0.1 supplied the first event-order evaluator and hand-authored trace/mutation corpus. v0.2 added qualified clock-basis descriptors, interval comparison, bitemporal visibility at decision frontier K, unique attempt lineage, complete attempt-inventory commitments, scoped stop obligations and exact per-attempt-to-discharge relations.

The remaining gap is test realism. A collection of static traces is useful but cannot expose the interaction surface created by clock evolution, causal production, network delivery, observation, durability, cancellation, irreversible effects, stop-policy changes, process failure and replay. v0.3 makes the scenario itself a versioned program: fixed seed + generator version + typed actions + deterministic scheduler + canonical digest + counterexample reducer.

## Implemented in this increment

- Versioned schedule contract: schema version, generator version, seed, scenario family and typed action list.
- Independent virtual time per source clock series, including clock domain, basis descriptor, epoch, scale, nanosecond unit, offset, uncertainty, calibration validity, receipt reference and qualification state. Each produced event binds back to that exact source clock series; `_ns` fields reject non-nanosecond units.
- Event production with explicit causal predecessor edges.
- Deterministic network delivery, duplicate/drop controls, partition and reconnect.
- Observation and durability steps; cancellation request/acknowledgement; stop trigger, policy revision and discharge; effect attempt, commit and terminal reconciliation; process crash/restart; event supersession.
- Canonical action ordering by virtual time and unique ordinal. Input vector order is not semantic; causal predecessor lists are normalized as sets.
- Replay report with schedule digest, ordered action records and scheduler dispositions.
- Fixed-seed SplitMix64 delivery/partition generator plus a second generator for both stop/cancel/commit race orders (commit-before-accepted-ack and commit-after-accepted-ack).
- Delta-debugging shrinker that only accepts a candidate when structural validation still passes and the supplied counterexample predicate still reproduces.
- Fifteen module unit tests and three Proptest properties (256 cases each configured), covering determinism, permutation invariance, receipt shape, clock epoch regression, exact stop action binding, crash/persistence, conflicting reconciliation, and both cancellation/commit race orders.

The scheduler is not the v0.2 safety oracle. It emits reproducible schedule observations and precondition outcomes. Clock regressions and event-time regressions are diagnosed within an exact source clock series; a new epoch starts a new series. Failed/unresolved clock qualifications and semantic unsafe orderings are retained as replay inputs; malformed identity, causal references, or receipt-digest shapes are rejected. The next adapter must translate those into v0.2 owner-qualified inputs and run the matched evaluator ladder. A schedule may intentionally contain unsafe orderings; the scheduler must preserve them rather than silently "repair" them. A local clock regression is an observation about that source clock, not a cross-domain ordering claim.

## Paradigm shifts

1. **From trace fixtures to executable histories.** Scenarios describe how facts are produced, delivered, observed and persisted, not only the final JSON snapshot.
2. **From wall-clock timestamps to qualified temporal evidence.** Each source clock advances independently; calibration, basis, uncertainty and validity become explicit faultable inputs. No numeric cross-domain comparison is added here.
3. **From event order to knowledge order.** Production, delivery, first observation and durable recording are distinct. This supports historical replay without allowing a late-arriving fact to rewrite what was knowable at K.
4. **From retry logic to effect-lineage assurance.** Start, cancel, commit and reconcile are separate events; an unknown result is not silently converted into not-applied.
5. **From label matching to exact action lineage.** A stop discharge binds the exact trigger and current policy-revision action IDs; substituted or stale references do not inherit a prior discharge.
6. **From duplicate suppression to contradiction detection.** An exact repeat is idempotent, but a changed cancellation answer or terminal receipt is recorded as a conflict.
7. **From cancellation intent to commit-race evidence.** Both commit-before-acknowledgement and commit-after-accepted-acknowledgement schedules remain visible as different outcomes.
8. **From hand-selected negatives to counterexample discovery.** Seeded schedules, delivery permutations and property-preserving shrinking aim to find the smallest causal schedule that still violates an invariant.
9. **From one aggregate score to a safety/usefulness frontier.** False-safe, false-hold, unresolved, reason-code mismatch and per-fault-family outcomes remain separate.

## Benchmarks and target gates

Targets are preregistered objectives, not measured results.

| Dimension | Target | Evidence unit |
|---|---:|---|
| False-safe | 0 on frozen and independently authored held-out hard negatives | Independent schedule lineage |
| Attempt cardinality | 100% one start / at most one terminal | Attempt identity |
| Retry lineage | 100% exact predecessor, qualified NOT_APPLIED and strict BEFORE | Retry chain |
| Stop discharge | 100% exact obligation/discharge/attempt binding | Applicable attempt-obligation pair |
| Completeness | 100% missing, stale, or unresolved inventory blocks | Decision frontier |
| Determinism | 100% byte-identical replay for fixed seed, generator, build and input | Replay |
| Permutation | 100% canonical report invariance under action delivery permutation | Schedule equivalence class |
| Five-way ordering | ≥95% on blinded, independently adjudicated traces | Independent schedule lineage |
| False hold | Proposed ≤5% only after adequate eligible-ACT denominator | Paired ACT/HOLD case |
| Performance | Report p50/p95 assessment latency, event/s, peak memory, generation cost, shrink iterations and replay time | Fixed hardware/build/corpus |

Do not use mutations of one parent trace as independent observations. Split data by scenario lineage and causal skeleton before creating held-out sets. A mutation suite is a falsification check, not a reliability estimate. Use confidence intervals and publish denominator counts; zero misses on a small corpus is not a zero real-world failure rate.

## Matched evaluator ladder

Run identical frozen schedule inputs, budgets and adjudicated labels through:

1. wall-clock point sorting;
2. same-domain sequence-only ordering;
3. HLC / causal-only ordering;
4. causal graph only;
5. qualified interval-only comparison;
6. v0.1 event-order evaluator;
7. v0.2 temporal + bitemporal + attempt + stop-relation composition;
8. v0.2 plus independently qualified complete history inventory;
9. v0.3 schedule/replay plus each evaluator above.

Compare paired decisions and per-family confusion matrices. Report false-safe and false-hold separately, plus unknown/indeterminate rates and reason-code quality. Include ablations so any gains can be attributed to a specific qualification mechanism rather than a larger corpus or more conservative policy.

## Industry comparison and claim ceiling

- **FoundationDB simulation testing** is the closest methodology comparator for deterministic fault-driven history exploration and reproducible failures. The relevant comparison is test methodology, not throughput or correctness superiority.
- **Jepsen / Elle and Porcupine** are appropriate comparators on the subset that can be expressed as transactional histories with matching consistency models. They are not general engineering stop/effect authority evaluators.
- **Hybrid logical clocks and Spanner/TrueTime** provide important clock/causal-order design baselines. v0.2's qualified interval and common-basis gate should be compared on the exact same traces; it does not replace TrueTime's infrastructure assumptions or Spanner's transaction guarantees.
- **OpenTelemetry timestamp and observed-timestamp separation** is an interoperability clue: event time and observation time are not interchangeable. v0.2 bitemporal fields add durable-recording and decision-frontier semantics, but do not replace the telemetry system.
- **NIST manufacturing traceability guidance** is an integration opportunity for provenance and cross-system traceability. Symthaea's potential contribution is decision admissibility over owner-qualified evidence, not a replacement for PLM, ALM, MES, ERP or quality-management systems.

The differentiator to test is not "better clock sync" or a universal distributed database. It is a cross-owner admissibility layer for irreversible engineering operations when time bases, evidence authorities, and system-of-record boundaries differ. Demonstrate this only with matched traces and independent ground truth.

## Known limitations / next increment

- The implemented generator families currently cover event production/delivery/partition and the two principal stop/cancel/commit orderings. Dedicated generators are still needed for calibration expiry/drift, crash between effect and durable reconciliation, multi-scope stops, bitemporal late arrival, revocation and inventory omission.
- Observation and durable-persistence actions are currently ordered by simulator virtual time; they do not yet carry separate observer-clock and store-clock basis/epoch/timestamp evidence. The next increment must add these domains so the v0.2 bitemporal envelope can be populated without promoting simulator time into physical/owner time.
- The current scheduler records owner receipt fields but does not authenticate them. Owner-native signature, trust-root, scope, completeness and semantic verifiers must qualify those receipts.
- The v0.3 scheduler is not yet wired to the v0.2 evaluator. That adapter and an independent oracle are the next critical step.
- Deterministic replay depends on exact code build and input digest. The schedule seed alone is not a full reproducibility receipt.
- Fourteen Rust module unit tests and three Proptest properties (256 cases each configured) are authored at this point. No Rust toolchain is available in the execution environment used for this authoring pass. Source checks and independent reference tests must not be represented as Rust execution or CI pass.

## Reproduction

```sh
cargo test -p symthaea-decision-assessment
cargo clippy -p symthaea-decision-assessment --all-targets -- -D warnings
cargo fmt --all -- --check
```

Record exact commit SHA, rustc/cargo versions, command lines, exit codes, test counts, Proptest cases, and input corpus digest. Skipped CI is not a pass.
