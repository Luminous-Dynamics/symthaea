# Cross-System Temporal Semantics Contract v0.1

**Status:** Research / implementation RFC. The contract is proposed; the four integrations below are audit findings and migration targets, not claims of completed cross-repository qualification.  
**Scope:** Symthaea, Mycelix, Sol Atlas, Symtropy, and adapters between them.

## Executive decision

**Do not model time as one universal scalar.** Establish a shared temporal *type system* and conversion rules instead. An event may have a physical timestamp, a causal position, a simulation tick, a valid-time interval, and a cognitive observation cycle. Those values answer different questions.

The shared rule is:

> Every temporal value MUST identify what clock or reference system it belongs to, what its unit means, and what comparisons are valid. Converting between time domains requires an explicit, versioned mapping and must preserve uncertainty. If no valid mapping exists, the comparison is unknown—not guessed.

This RFC draws on W3C OWL-Time's separation of instants, intervals, durations and temporal reference systems; Lamport's distinction between physical clock readings and causal ordering; Rust's distinction between `Instant` and wall-clock time; and NASA/JPL SPICE's explicit astronomical time scales such as TT and TDB. These are complementary building blocks, not a claim that any one source supplies our complete ontology.

## 1. The temporal domains we actually need

| Domain | Question answered | Canonical representation | Must not be used as |
|---|---|---|---|
| **Monotonic runtime time** | How much real execution time elapsed on this process/clock? | Monotonic instant + duration; clock/process scope | A portable event timestamp or cross-machine ordering |
| **Civil / wall time** | When does an event claim to have occurred on the human calendar? | UTC instant with precision, source and uncertainty; local timezone only for presentation | Proof of causality, unique event ID, or an exact clock truth without uncertainty |
| **Causal / logical time** | What is ordered by program, message or ledger dependencies? | Source-chain sequence, Lamport clock, vector clock, or explicitly specified HLC | Physical elapsed duration or civil date |
| **Simulation / worldline time** | At what model step and model time does a simulated event occur? | Run ID + branch/worldline ID + integer tick + tick scale + model version | Wall time, another run's tick, or evidence about reality without calibration |
| **Astronomical / ephemeris time** | At what physical dynamical epoch is an orbital position evaluated? | Named time scale (e.g. TDB/TT), epoch, units, reference frame and ephemeris/model version | UI animation seconds or a generic Gregorian year |
| **Cognitive / embodied time** | Over what elapsed duration, cognitive cycles, or predicted horizon was a state observed? | Cycle/observation sequence plus monotonic elapsed time and declared temporal window | A claim about subjective duration based solely on loop frequency |
| **Calendar / social / institutional time** | When does a schedule, contract, care rhythm or policy become effective or expire? | Calendar rule, timezone, valid-from/valid-until interval and rule version | A fixed number of seconds where calendar semantics matter |
| **Ecological / evolutionary time** | Over what season, generation or climate regime does change occur? | Domain interval, unit, baseline and scenario provenance | A universal “year” without specifying local/body/calendar and model resolution |

A codebase may use several domains at once. That is a feature if the distinctions are explicit.

## 2. Minimum temporal wire contract

Agree on the field semantics before extracting another shared crate. The following is a conceptual schema, not yet a generated JSON Schema or production API.

### 2.1 A temporal coordinate

Every coordinate needs:

- `domain`: one of the defined domains, or a registered domain identifier;
- `value`: integer ticks/nanoseconds where exact integer representation is appropriate; decimal or a library-specific value for scales that require it;
- `unit`: an explicit unit such as `ns`, `s`, `cycle`, `month_tick`, `model_year` or `TDB_seconds`;
- `reference_system`: UTC/Unix epoch, local calendar and IANA zone, J2000/TDB, a named simulation run, or another named reference;
- `origin_id` / `clock_id`: where the coordinate began and which clock generated it;
- `uncertainty` and `precision`: stated bounds where known; never imply a precise measurement merely because storage is precise;
- `provenance`: source, schema version, and conversion/software version where applicable.

Use typed Rust structs/enums rather than untyped floats or ambiguous field names. Do not serialize Rust `Instant` as if it were portable; it is only meaningfully comparable within its clock domain.

### 2.2 Event envelope: happened, observed, recorded, valid

Keep these concepts separate:

- **event/valid time:** when the represented event or fact applies in the modeled domain;
- **observed time:** when a sensor/agent obtained the observation;
- **recorded time:** when the system persisted or received the record;
- **causal position:** the explicit parent/dependency and sequence information;
- **simulation position:** run/worldline plus tick, if generated by a simulation;
- **validity interval / expiry:** when a claim, credential, plan or policy is applicable.

The times can differ legitimately. A delayed sensor observation can describe an earlier physical event. A future appointment has future valid time but is recorded now. An imported causal event can arrive after a dependent-looking local wall timestamp because the source clock is skewed.

### 2.3 Interval rules

Use instants, durations and intervals as distinct types. Apply Allen's interval relations only when intervals inhabit the same reference system or have been converted by an explicit mapping. `start: f64` and `end: f64` without a unit and reference origin are not sufficient for cross-system reasoning.

For recurring schedules, store the recurrence rule and calendar/timezone rather than repeatedly adding a fixed number of seconds. For forecast data, carry issue time, valid interval, horizon and source version. For open-ended sustainability, report rolling finite horizons and renewal capabilities rather than asserting that an infinite horizon has been proven.

## 3. System-specific contract

### Symthaea — lived/process time, memory time, prediction time

Current source: `src/consciousness/temporal/temporal_consciousness.rs` and `src/consciousness/temporal/temporal_primitives.rs`.

1. Separate observation-cycle count from actual monotonic elapsed time. A 100-sample history is a sample window, not necessarily a 100-second window.
2. Define the units of `phi_velocity` and `phi_acceleration` explicitly (currently reported as `Φ/s` and `Φ/s²`); preserve sample timestamps or elapsed offsets so derivatives can be reproduced.
3. Make smoothing cadence-aware. A sample-based EMA such as `0.1 * x + 0.9 * ema` has a different time constant at 10 Hz and 1 Hz. Prefer `alpha = 1 - exp(-dt / tau)` with a declared `tau`, or explicitly declare the filter to be per-cycle and refrain from interpreting it as a physical-time filter.
4. Use time-weighted or explicitly sample-weighted stability statistics when observations are irregular. Publish sample count and elapsed coverage beside the metric.
5. Give `TemporalInterval` a temporal reference/clock domain and a unit. Reject interval comparisons across incompatible clocks until conversion is supplied.
6. Retention depth/protention depth are currently counts; do not label their sum “time” without mapping those counts to an elapsed horizon or explicitly naming it a horizon-in-steps.
7. Distinguish prediction target time from prediction depth/index. Future-self identity alignment should compare the stated predicted horizon and provenance, not silently treat a present-state proxy as an actual future state.

**Acceptance test:** feeding the same piecewise-constant trajectory at different sample cadences should produce equivalent elapsed-time EWMA and slope within declared tolerance; duplicate or backwards timestamps must be rejected or explicitly classified as unusable, not divided through. Replay with a fixed observation clock must be deterministic.

### Mycelix — distributed causal time, authoring time, expiry

Current architecture: Holochain source-chain and DHT events plus multi-agent coordination. Wall-clock timestamps are useful metadata but are not a universal total ordering in distributed operation.

1. Treat a signed action/header timestamp as the author's claimed physical time with the protocol's actual precision and validation limits. Do not use it alone to prove causal order or to resolve every concurrent event.
2. Preserve authoritative source-chain sequence/hash and explicit message/receipt dependencies. Where events from different agents are concurrent, retain concurrency rather than inventing order from close timestamps.
3. For expiry, leases, credentials, replay windows and deadline policies, declare the time basis and tolerance for clock skew. Security-sensitive replay protection should bind identity/action digest/nonce or sequence as appropriate, not trust a wall timestamp alone.
4. For replicated records, distinguish `created_at`, `observed_at`, `received_at`, `valid_from`, and `valid_until`. Do not overload `timestamp`.
5. A shared event schema can carry both causal position and physical-time claims, but each validator must state which one a rule depends on.

**Acceptance test:** out-of-order delivery, different-agent concurrent events, a skewed wall clock, duplicate timestamp and replay attempts produce explicit outcomes and do not rewrite causal history.

### Sol Atlas — simulation years, UI playback, and ephemeris time

Current source: `sol-atlas-core/src/timeline.rs`, `simulation.rs`, and `solar_system.rs`.

1. Keep civilization projection `model_year`, user-selected timeline year, elapsed animation time, and physical ephemeris time as distinct fields/types.
2. Clearly label the current orbit animation as **illustrative/non-ephemeris** wherever display speeds are intentionally compressed or exaggerated. Do not silently interpret `body_position(..., time_secs)` as an accurate physical position.
3. If physically meaningful ephemerides are needed, use a named scale and epoch (NASA/JPL SPICE describes TT/TDB and the J2000 epoch), body/frame identity and a versioned ephemeris source. The UI clock is not the dynamics clock.
4. Give timeline data a source date/version, baseline, model horizon, and validity/revision status. “Year 0 = present day” must resolve to a frozen baseline date for each dataset version, not float forward unnoticed.
5. Separate scenario time from data publication/observation date. A 2100 forecast can be published today; the two dates answer different questions.

**Acceptance test:** playback speed changes never alter model results; moving the timeline slider is deterministic; physical and illustrative orbital modes are visibly and programmatically distinct; baseline-date changes require a versioned data update.

### Symtropy — worldline time, chronicle time, and causal order

Current source: `src/systems/chronicle.rs` and `src/systems/chronicle_recorder.rs`.

1. The current `logical_time` is populated from `SystemTime::now()` seconds since Unix epoch. Rename this field to a physically meaningful name such as `recorded_at_unix_seconds` in a versioned schema, or implement a genuine logical counter and give it its own field. Do not conflate both semantics.
2. Seconds-granularity IDs (`evt_<unix-seconds>`) can collide when multiple events are recorded within one second. Use a stable unique event identifier (e.g. UUID/ULID or worldline-scoped monotone sequence plus collision-resistant component).
3. If chronological replay is important, define a worldline-scoped event sequence and explicit parent/causal references. A wall-clock timestamp is metadata, not sufficient sequence state.
4. For deterministic replay, separate the replay's logical/worldline step from the host clock. Re-running the same worldline should not shift state because a new wall-clock value was sampled.
5. Hash-chain / signature integrity is a separate gate from time semantics. A field named `prev_hash` is only a chain if it references the actual prior committed event; placeholder hash/signature values are not integrity evidence. Do not treat chronology work as solving those integrity gaps.

**Acceptance test:** rapid event bursts have unique IDs; wall clock rollback does not reorder the worldline; deterministic replay produces the same logical sequence; a fork explicitly identifies its parent event and branch/worldline.

## 4. Conversion and interoperation rules

1. **No implicit casts.** Do not compare `SystemTime`, `Instant`, Holochain/source-chain ordering, simulation ticks and TDB numbers as if their numeric representations meant the same thing.
2. **Conversion is a function with provenance.** Record input and output domain, conversion implementation/version, epoch, model/reference frame as applicable, and uncertainty.
3. **Causality is not chronology.** “A's timestamp is smaller” does not prove A caused or preceded B. Use event dependencies or logical clocks for causal questions.
4. **Time uncertainty travels with the value.** When uncertainty intervals overlap, do not claim a strict physical order unless another causal relation establishes it.
5. **Replay should be deterministic by design.** Persist external observations and their source/recorded times. Do not let wall-clock reads leak into simulation transition logic unless explicitly part of the experiment.
6. **Time horizons must be explicit.** A forecast, cache, credential or plan needs a unit, validity window, staleness policy and source version; absence of an expiry must not silently mean “eternal”.
7. **Unknown stays unknown.** An unavailable clock, missing source timestamp or model outside its valid horizon is not silently replaced with now, zero, or an accurate-looking date.

## 5. Qualification matrix

Build a shared temporal contract test corpus and adapt it to each runtime. Minimum cases:

- monotonic clock vs wall clock moving backwards;
- duplicate timestamps and same-second event bursts;
- different process/host clock origins;
- delayed, reordered, duplicated and concurrent network events;
- expired and not-yet-valid objects at skew boundaries;
- leap day, timezone change and DST boundaries for civil schedules;
- simulation branch/fork and deterministic replay at a fixed run/tick;
- prediction horizons represented in cycles vs elapsed time;
- variable observation cadence produces cadence-stable elapsed-time metrics;
- interval comparisons across mismatched reference systems are rejected;
- TDB/TT/UTC conversion is explicit and carries an ephemeris/version provenance;
- source data whose publication time differs from the modeled valid interval;
- requested horizon beyond executed simulation duration fails closed.

Every result is reported against an exact commit, toolchain, schema and fixture corpus. Source presence and authored tests are not test passes.

## 6. Recommended rollout

**Phase 0 — terminology and schema (this RFC).** Agree on field meanings and versioning; document domain mapping per repository.

**Phase 1 — fix high-confidence semantic errors.** In Symthaea, test deterministic elapsed-time observations and cadence-aware filtering. In Symtropy, separate wall-clock recording time from logical sequence and remove second-based event ID generation. In Sol Atlas, label animation time vs model year and freeze the baseline date. In Mycelix, document which validation rules rely on chain order, wall time, or expiry time.

**Phase 2 — versioned envelopes and adapters.** Generate a JSON Schema or equivalent typed contract; add Rust DTOs and explicit adapters per repository. Do not add a universal Rust dependency until the wire semantics and compatibility rules are accepted.

**Phase 3 — cross-system conformance.** Run the same golden event corpus through the four implementations and compare normalization, interval results, causality and replay behavior. Add property tests for monotonicity, transitivity where applicable, conversion round-trips within stated tolerance, and rejection of invalid cross-domain comparisons.

**Phase 4 — temporal planning.** Combine Symthaea's learned predictions with Mycelix's causal/evidence history, Sol Atlas' physical and civilizational horizons, and Symtropy's persistent worldline memory through explicit mappings. Keep the models' uncertainties and timescales separate; integrate evidence, not just scalar scores.

## 7. Grounding references

- Lamport, L. (1978), *Time, Clocks, and the Ordering of Events in a Distributed System*. Defines the “happened-before” partial order and logical clocks. [Microsoft Research](https://www.microsoft.com/en-us/research/publication/time-clocks-ordering-events-distributed-system/)
- W3C, *Time Ontology in OWL*. Models temporal entities as instants/intervals, with duration, temporal reference systems and Allen interval relations. [W3C](https://www.w3.org/TR/owl-time/)
- Rust standard library, `std::time::Instant`. Describes a monotonic nondecreasing clock for measuring elapsed duration, with platform-specific caveats. [Rust documentation](https://doc.rust-lang.org/stable/std/time/struct.Instant.html)
- NASA/JPL NAIF, *SPICE Time Subsystem*. Defines astronomical scales including TT/TDB and the J2000 epoch for ephemeris use. [SPICE documentation](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html)

**Qualification boundary:** this RFC does not claim that the four systems have been fully audited, that every listed concern is a live defect, or that conformance tests pass. Its purpose is to give us a precise, testable agreement before cross-repository integrations become more tightly coupled.
