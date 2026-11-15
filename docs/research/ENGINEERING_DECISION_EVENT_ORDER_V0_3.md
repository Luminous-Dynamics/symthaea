# Event-order v0.3: deterministic virtual-time replay

**Status:** research implementation, synthetic schedules only. This document describes authored source and intended gates; it is not a claim that Rust tests, CI, clock adapters, owner receipt verification, or physical systems have passed.

## Why v0.3 exists

v0.1 supplied the first event-order evaluator and hand-authored trace/mutation corpus. v0.2 added qualified clock-basis descriptors, interval comparison, bitemporal visibility at decision frontier K, unique attempt lineage, complete attempt-inventory commitments, scoped stop obligations and exact per-attempt-to-discharge relations.

The remaining gap is test realism. A collection of static traces is useful but cannot expose the interaction surface created by clock evolution, causal production, network delivery, observation, durability, cancellation, irreversible effects, stop-policy changes, process failure and replay. v0.3 makes the scenario itself a versioned program: fixed seed + generator version + typed actions + deterministic scheduler + canonical digest + counterexample reducer.

## Implemented in this increment

- Versioned schedule contract: schema version, generator version, seed, scenario family and typed action list.
- Independent virtual time per source clock series, including clock domain, basis descriptor, epoch, scale, nanosecond unit, offset, uncertainty, calibration validity, receipt reference and qualification state. Each produced event binds back to that exact source clock series; `_ns` fields reject non-nanosecond units.
- Event production with explicit causal predecessor edges and an exact `clock_mapping_action_id` binding; the event's source timestamp is checked against that specific mapping's tick, not merely the latest clock tick.
- Deterministic network delivery, duplicate/drop controls, partition and reconnect.
- Observer and durable-store timestamps on independent, exact clock series with receipt/qualification fields and an owner-qualified v0.2 bitemporal projection; cancellation request/acknowledgement; stop trigger, policy revision and discharge; effect attempt, commit and terminal reconciliation; process crash/restart; event supersession.
- Canonical action ordering by virtual time and unique ordinal. Input vector order is not semantic; causal predecessor lists are normalized as sets.
- Replay report with schedule digest, ordered action records and scheduler dispositions.
- Fixed-seed SplitMix64 delivery/partition generator, a clock-mapping fault generator (unqualified, expired, stale mapping reference), and a second generator for both stop/cancel/commit race orders (commit-before-accepted-ack and commit-after-accepted-ack).
- Delta-debugging shrinker that only accepts a candidate when structural validation still passes and the supplied counterexample predicate still reproduces.
- Thirty module unit tests and seven Proptest properties (256 cases each configured; 1,792 potential property cases), covering deterministic replay, action permutation, exact source/observer/store mapping identity, unqualified/expired/stale/not-yet-valid/invalid clock mappings, separate owner-time projection, missing/substituted owner resolutions, event-level first-observation cardinality, occurrence/observation/persistence ordering, exact stop action binding, crash/persistence, conflicting reconciliation, and cancellation/commit races.

The scheduler is not the v0.2 safety oracle. It emits reproducible schedule observations and precondition outcomes. Clock regressions and event-time regressions are diagnosed within an exact source clock series; a new epoch starts a new series. Failed/unresolved clock qualifications and semantic unsafe orderings are retained as replay inputs; malformed identity, causal references, or receipt-digest shapes are rejected. The next adapter must translate those into v0.2 owner-qualified inputs and run the matched evaluator ladder. A schedule may intentionally contain unsafe orderings; the scheduler must preserve them rather than silently "repair" them. A local clock regression is an observation about that source clock, not a cross-domain ordering claim.

## v0.3.1 qualified capture-clock projection

The stacked v0.3.1 branch is `research/event-order-v0-3-1-bitemporal-clocks-2026-11-15` and is tracked by [issue #12292](https://github.com/Luminous-Dynamics/symthaea/issues/12292). It introduces a versioned v0.3.1 schema without changing the original v0.3 schema path.

`EventObserved` and `EventPersisted` now have distinct source clock series, local timestamps, exact clock-mapping action IDs, receipt references/digests and qualification states. The replay classifier diagnoses unqualified, expired, not-yet-valid, invalid-validity and stale-reference mappings for source, observer and durable-store clocks. The simulator `at_ns` remains scheduling time only.

`project_bitemporal_event` requires the exact event production, unique event-level first observation and unique durable write to be replay-applied. It resolves all three owner times from a caller-supplied `QualifiedEventTime` map keyed by (mapping action ID, source-local timestamp); verifies clock receipt and validity binding; enforces occurrence-not-after-observation and observation-before-durable-record; then emits the v0.2 `BitemporalEventEnvelope`. Its evidence sidecar preserves the schedule digest, action IDs, mapping-action IDs, source-local timestamps and receipt digests. Missing owner resolution is Indeterminate; substituted identity or contradictory chronology is CriticalFailure. The adapter does not normalize clocks, authenticate signatures/trust roots, or reconstruct attempt/stop inventory yet.

A standalone independent reference model is available at [event-order-v0.3.1-independent-reference.mjs](../../benchmarks/engineering-change-impact/event-order-v0.3.1-independent-reference.mjs), with its run record at [event-order-v0.3.1-independent-reference.json](../../benchmarks/engineering-change-impact/event-order-v0.3.1-independent-reference.json). It covers 8,000 capture-clock fault checks, 100,000 interval-order pairs and 100,000 projection-order cases. It is JavaScript reference evidence only, not execution of the Rust module.

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
| False-safe | 0 on frozen and independently authored held-out hard negatives; aim for 3,000 independent negative lineages aggregate | Independent schedule lineage |
| Attempt cardinality | 100% one start / at most one terminal | Attempt identity |
| Retry lineage | 100% exact predecessor, qualified NOT_APPLIED and strict BEFORE | Retry chain |
| Stop discharge | 100% exact obligation/discharge/attempt binding | Applicable attempt-obligation pair |
| Completeness | 100% missing, stale, or unresolved inventory blocks | Decision frontier |
| Determinism | 100% byte-identical replay for fixed seed, generator, build and input | Replay |
| Permutation | 100% canonical report invariance under action delivery permutation | Schedule equivalence class |
| Clocked bitemporality | 100% separate source/observer/store clock identity; no simulator-time substitution | Event/observer/store triple |
| Owner mapping | 100% missing, stale, failed, out-of-validity or substituted mapping blocks projection | Exact mapping-action/timestamp key |
| Capture order | 100% unknown/overlapping observation→durability order remains Indeterminate; reverse order is critical | Bitemporal envelope |
| Five-way ordering | ≥95% on blinded, independently adjudicated traces | Independent schedule lineage |
| False hold | Proposed ≤5% only after adequate eligible-ACT denominator | Paired ACT/HOLD case |
| Performance | Report p50/p95 assessment latency, event/s, peak memory, generation cost, shrink iterations and replay time | Fixed hardware/build/corpus |

Do not use mutations of one parent trace as independent observations. Split data by scenario lineage and causal skeleton before creating held-out sets. A mutation suite is a falsification check, not a reliability estimate. Publish exact denominators and one-sided 95% binomial confidence bounds: with zero failures, 3,000 independent trials imply an upper bound of approximately 0.1% (the rule-of-three approximation); about 30,000 are needed for approximately 0.01%. These are statistical bounds on the sampled distribution, not guarantees against unmodelled faults or distribution shift. Report per-fault-family results as well as the aggregate.

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

This is a capability comparison, not a claim of current superiority. The systems below solve adjacent but non-identical problems.

- **FoundationDB Simulation** is the strongest methodological reference for deterministic fault exploration: its simulation is integrated with the same Flow runtime used for production execution, and FoundationDB documents simulation alongside live performance and hardware failure testing. Symthaea's schedule/replay approach is still much earlier: currently a research scheduler and generator, not a production-equivalent execution runtime. The target is to adopt its discipline—deterministic reproductions, fault injection, live-system follow-through—not to claim parity. [FoundationDB testing documentation](https://github.com/apple/foundationdb/blob/main/documentation/sphinx/source/testing.rst)
- **Jepsen / Elle** analyze observed transactional histories and detect consistency anomalies using dependency graphs and cycles. This is a strong comparator for transaction subsets and for independently adjudicating histories. Elle is not a universal stop/effect authority system; its own documentation notes limits around unobserved transactions and incomplete inference. Symthaea should integrate or differential-test against Elle where the event history is a database transaction history rather than recreate that checker. [Elle project](https://github.com/jepsen-io/elle)
- **Porcupine** is a fast linearizability checker for histories of concurrent objects. It is an appropriate baseline for single-operation or register-like semantics, not for bitemporal evidence qualification, stop discharge, cross-owner receipt scope or engineering-effect retry lineage. [Porcupine project](https://github.com/anishathalye/porcupine)
- **Google Spanner / TrueTime** provides a substantially stronger infrastructure guarantee than this repository currently does: Spanner's externally consistent transactions use time intervals and timestamp ordering to preserve real-time transaction order across servers. Symthaea must not claim to outperform this. The useful comparison is whether a cross-owner engineering decision can combine Spanner-like qualified time evidence with non-database evidence (calibration, permissions, test results, inventory completeness, stop obligations) and preserve fail-closed admissibility. [Spanner TrueTime and external consistency](https://docs.cloud.google.com/spanner/docs/true-time-external-consistency)
- **Hybrid logical clocks** are an important logical/physical time baseline. They can preserve causal order while retaining physical-time proximity, but they are not a substitute for owner-authenticated clock calibration, mapping validity, source identity or durable observation evidence. Use matched replay traces and compare HLC-only, qualified-interval and combined order oracles.
- **OpenTelemetry** explicitly distinguishes event Timestamp from collection ObservedTimestamp; Symthaea's v0.2 model goes further by distinguishing first observation, durable recording and decision-frontier visibility. The relationship is complementary: ingest telemetry without conflating these times; preserve its source semantics and add owner qualification on top. [OpenTelemetry log data model](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/logs/data-model.md)
- **NIST IR 8536 (final, September 2026)** frames manufacturing traceability as decision-centric and risk-scaled, with interoperable cryptographic links and selective disclosure across organizations. This is a direct integration opportunity: reuse NIST-style traceability structures and existing sector standards; Symthaea's proposed addition is a decision-admissibility/reconciliation layer for irreversible actions, not a replacement for PLM, ALM, MES, ERP, QMS or the NIST framework. [NIST IR 8536 final](https://csrc.nist.gov/pubs/ir/8536/final)

The differentiator to test is not "better clock sync" or a universal distributed database. It is a cross-owner admissibility layer for irreversible engineering operations when time bases, evidence authorities, policy scopes and systems of record differ. Demonstrate this only with identical frozen inputs, independently adjudicated ground truth, ablations and performance measurements.

## v0.3.1 qualified observer and durable-store clocks

The stacked follow-on adds a separate observer clock series and durable-store clock series to the schedule action model. Both carry exact clock domain/basis/epoch/scale/nanosecond unit, exact mapping-action identity, source-local time, receipt ref/digest and qualification. Simulator action time remains only a replay scheduler field and is never promoted into the v0.2 time contract.

A deterministic generator creates event → delivery → observer clock/observation → independent store clock/durable record. The first v0.2 adapter, `project_bitemporal_event`, requires exactly one production record, one selected observer record and one selected store record; it requires the observation and persistence actions to have Applied scheduler dispositions; and it consumes externally supplied `QualifiedEventTime` resolutions keyed by (mapping action ID, source-local timestamp). It checks mapping-action identity, source domain, receipt reference/digest, owner qualification, mapping validity and local sample bounds before building `BitemporalEventEnvelope`.

Missing owner time is Indeterminate; explicit owner FAIL, substituted clock identity or receipt mismatch is CriticalFailure. It does not normalize source clocks, authenticate signatures or trust roots, or infer durable time from simulator time. Occurrence time remains optional if no independently qualified occurrence-time resolution is supplied. The projected envelope is suitable for the v0.2 bitemporal visibility evaluator; attempt lineage, scoped stops and complete attempt inventory are not yet reconstructed by this adapter.

The v0.3.1 branch historically contained 24 module unit tests, six Proptest properties (256 cases each configured), and one frozen integration known-answer fixture. The stacked v0.3.2 branch extends this with the action-history projection and additional tests documented above.

## v0.3.2: action-level evidence and attempt/stop projection

The follow-on is tracked by [issue #12316](https://github.com/Luminous-Dynamics/symthaea/issues/12316) on branch `research/event-order-v0-3-2-action-evidence-2026-11-15`, based on the exact v0.3.1 head. It introduces the separate `symthaea.event-order-schedule.v0.3.2` schema; it does not overwrite the v0.3.1 schema.

### Contract changes

- `ScheduledAction.owner_time_ref` names an exact earlier mapping-action ID and source-local timestamp. The reference must match native event source fields for production, observation and persistence actions. Scheduler `at_ns` remains a replay-ordering coordinate only.
- `ActionEvidence` sidecars are typed by action kind: attempt start and retry parent; effect-commit receipt; cancellation-ack receipt; terminal-reconciliation exact-start binding and qualification; stop trigger and policy revision receipts plus scope; discharge scope/frontier/manifest; and complete inventory attestation.
- The owner verifier input is now bound to the exact action ID, receipt reference and SHA-256 that were checked. Its injected gates cover identity, digest integrity, authenticity, scope and semantics; inventory completeness and in-flight reconciliation remain separately represented. This is a trust boundary, not an implementation of signature or trust-root verification.
- `project_action_history` constructs `OperationAttemptRecord`, `ScopedStopObligation`, `StopDischargeReceipt` and `AttemptHistoryInventory` from canonical replay order and externally qualified owner-time/receipt inputs. It calls the v0.2 lifecycle and start-eligibility validators.
- An inventory frontier must be exactly one replay-applied produced event and owner-time-before its inventory attestation. Its manifest digest is recomputed by the v0.2 attempt-manifest projection.
- A stop discharge binds exact trigger action, latest policy-revision action, unchanged scope, unique complete inventory attestation, frontier and manifest digest. Per-attempt relations bind exact start and discharge event IDs; pre-discharge attempts require exact terminal-before-discharge evidence.
- Cancellation acknowledgements have their own owner time, request-action binding and receipt, but never imply `NOT_APPLIED`. Retry requires the exact predecessor, a qualified `NOT_APPLIED` terminal, and terminal-before-retry-start.
- The action projector rejects effect-commit-after-terminal and `NOT_APPLIED`-with-commit contradictions. If any owner-time, receipt, lifecycle, stop or inventory check fails, the combined report suppresses a positive start-eligibility flag.

### v0.3.2 source and reference evidence

- 43 module `#[test]` cases are authored.
- Eight Proptest properties are configured for 256 cases each (2,048 potential property cases per full run).
- Two integration tests cover the frozen clocked schedule and action-evidence variant rejection.
- A standalone independent reference model is stored at `benchmarks/engineering-change-impact/event-order-v0.3.2-independent-reference.mjs` with result record `event-order-v0.3.2-independent-reference.json`. On Node.js v22.16.0 with seed `0x4c554d49` it exercised 100,000 interval pairs, 150,000 receipt identity checks, 100,000 retry cases, 90,000 stop/inventory cases, 10,000 action-vector permutations, 50,000 cancellation semantics checks, and 50,000 effect/terminal chronology cases. All recorded semantic violation counters were zero. This is a separately authored JavaScript reference model, not Rust execution or evidence of production reliability.
- JSON Schema syntax/JSON parsing and source-level identity/cardinality audits are separate from a full schema-engine test and Rust compile. No `cargo test`, `cargo clippy`, `cargo fmt` or physical qualification is claimed. The authoring environment has no `rustc`, `cargo` or `rustfmt`, and skipped CI is not pass.

### Remaining hard gates

1. Run exact-head Rust tests, clippy and format checks; resolve all compile or lint failures without weakening safety semantics.
2. Implement owner-native receipt verification (signature chain, revocation, trust-root version, semantic scope, inventory completeness and freshness).
3. Add crash windows between physical effect and durable reconciliation, retry-after-restart, policy-revision races, stop-scope expansion, receipt revocation, late arrivals and inventory omission generators.
4. Validate action/effect projection against independently authored oracles and real adapter traces; report per-family false-safe, false-hold and indeterminate rates.
5. Benchmark projection latency, throughput, memory, counterexample shrinking and deterministic replay under increasing action cardinality.

## Known limitations at the v0.3.1 stage

- The implemented generator families cover event production/delivery/partition, source-clock unqualified/expired/stale mapping faults, eight observer/store mapping faults (unqualified, expired, not-yet-valid and invalid validity interval), and the two principal stop/cancel/commit orderings. Dedicated generators are still needed for gradual calibration drift/step changes, crash precisely between physical effect and durable reconciliation, multi-scope stops, bitemporal late arrival, revocation and inventory omission.
- The current scheduler records owner receipt fields but does not authenticate them. Owner-native signature, trust-root, scope, completeness and semantic verifiers must qualify those receipts.
- The v0.3.1 bitemporal projection alone does not reconstruct attempt lifecycle, retry lineage, scoped stop obligations or complete attempt inventory; the v0.3.2 action projector now supplies a synthetic-schedule adapter for these contracts, but production owner verification and real effect-system adapters remain unimplemented.
- Deterministic replay depends on exact code build and input digest. The schedule seed alone is not a full reproducibility receipt.
- At v0.3.2, 46 Rust module unit tests, eight Proptest properties (256 cases each configured; 2,048 potential cases), and two integration tests are authored, not executed. The frozen action-history known-answer fixture contains 14 actions, seven owner-time resolutions and five owner receipt assessments. The projector returns exact terminal, stop, inventory, effect and cancellation sidecars plus schedule/replay digests. The local authoring environment has no Rust toolchain. Independent JavaScript reference checks must not be represented as Rust execution or CI pass.

## Reproduction

The v0.3.1 follow-on branch is `research/event-order-v0-3-1-bitemporal-clocks-2026-11-15`, stacked on v0.3 PR #12287. Its observer/store clock fields and bitemporal projection are in the same crate module, with dedicated unit and property tests.

```sh
cargo test -p symthaea-decision-assessment
cargo clippy -p symthaea-decision-assessment --all-targets -- -D warnings
cargo fmt --all -- --check
```

Record exact commit SHA, rustc/cargo versions, command lines, exit codes, test counts, Proptest cases, and input corpus digest. Skipped CI is not a pass.
