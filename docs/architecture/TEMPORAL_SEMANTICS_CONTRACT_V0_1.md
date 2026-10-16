# Cross-System Temporal Semantics Contract v0.1

**Status:** Research / implementation RFC. The contract is proposed; the four integrations below are audit findings and migration targets, not claims of completed cross-repository qualification.  
**Scope:** Symthaea, Mycelix, Sol Atlas, Symtropy, and adapters between them.

**Companion:** [Ancient Wisdom to Engineering Invariants v0.1](./ANCIENT_WISDOM_TO_ENGINEERING_INVARIANTS_V0_1.md) translates selected philosophical lenses into proposed, testable engineering properties. It is a research RFC, not a claim of implementation or proof of ethical correctness.

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

### 2.3 Bitemporal history, retroactive corrections, and “known as of”

For claims, policies, evidence, and other facts that can be corrected after the fact, keep at least these temporal meanings distinct:

- **Valid time:** when the proposition applies in the modeled world.
- **Transaction/recorded time:** when this system accepted and stored this version of the proposition.
- **Event time:** when the represented event occurred, according to the best available evidence and its stated precision.
- **Observed time:** when this observer obtained the evidence.

Do not collapse these into one `timestamp`. In particular, **“what do we now believe was true at time T?”** differs from **“what did this system record as its belief at time R?”** Both queries can be legitimate and must not silently return the same history.

A correction received later may change the current valid-time interpretation of an earlier period, but it must not erase the earlier record of what was known at transaction time. Preserve the original claim, its recording boundary, the correction, their provenance, and the supersession/dispute relation. The current view can prefer a corrected claim while an as-of query can still reconstruct the earlier recorded belief. A correction is not retroactive proof that the system knew the corrected fact earlier.

**Conformance case BT-01:** Record claim A as valid over `[t1,t3)` at transaction time `r1`. At later transaction time `r2`, receive correction B stating that the fact was valid only over `[t0,t2)`. A current valid-time query at `t2.5` must not treat A as current; an as-of-transaction-time query for `r1` must still reconstruct that A was the recorded claim then. Both versions and the correction link remain auditable.

This distinction is especially useful for Mycelix claims/disputes, Symtropy chronicles, audit receipts, civic/economic decisions, and any workflow that can receive delayed evidence. It is grounded in temporal-database work distinguishing event, valid and transaction time, not a new claim that all repositories currently implement bitemporal storage. [Jensen et al., “Resolution of time concepts in temporal databases” (1994)](https://doi.org/10.1016/0020-0255(94)90059-0).

### 2.4 Interval rules

Use instants, durations and intervals as distinct types. Apply Allen's interval relations only when intervals inhabit the same reference system or have been converted by an explicit mapping. `start: f64` and `end: f64` without a unit and reference origin are not sufficient for cross-system reasoning.

For recurring schedules, store the recurrence rule and calendar/timezone rather than repeatedly adding a fixed number of seconds. For forecast data, carry issue time, valid interval, horizon and source version. For open-ended sustainability, report rolling finite horizons and renewal capabilities rather than asserting that an infinite horizon has been proven.

## 3. System-specific contract

### Symthaea — lived/process time, memory time, prediction time

Current source: `src/consciousness/temporal/temporal_consciousness.rs` and `src/consciousness/temporal/temporal_primitives.rs`.

1. Separate observation-cycle count from actual monotonic elapsed time. A 100-sample history is a sample window, not necessarily a 100-second window.
2. Define the units of `phi_velocity` and `phi_acceleration` explicitly (currently reported as `Φ/s` and `Φ/s²`); preserve sample timestamps or elapsed offsets so derivatives can be reproduced.
3. Make smoothing cadence-aware. A sample-based EMA such as `0.1 * x + 0.9 * ema` has a different time constant at 10 Hz and 1 Hz. Prefer `alpha = 1 - exp(-dt / tau)` with a declared `tau`, or explicitly declare the filter to be per-cycle and refrain from interpreting it as a physical-time filter.
4. Use time-weighted or explicitly sample-weighted stability statistics when observations are irregular. Publish sample count and elapsed coverage beside the metric.
5. Enforce `min_samples_for_trajectory` before treating a trajectory as qualified, with an effective minimum of at least three observations because stability requires three retained samples. A configured minimum of one or two must not turn default placeholder stability into a measured healthy trajectory. Distinguish insufficient sample evidence from a measured unhealthy trajectory, and mark qualification impossible when history capacity is smaller than the effective minimum. Publish retained count, capacity, effective required minimum, and qualification status in the report.
6. Give `TemporalInterval` a temporal reference/clock domain and a unit. Reject interval comparisons across incompatible clocks until conversion is supplied.
7. Retention depth/protention depth are currently counts; do not label their sum “time” without mapping those counts to an elapsed horizon or explicitly naming it a horizon-in-steps. Honor constructor-provided retention and protention capacities exactly, including zero-capacity configurations; do not substitute hidden hard-coded bounds.
8. Distinguish prediction target time from prediction depth/index. Future-self identity alignment should compare the stated predicted horizon and provenance, not silently treat a present-state proxy as an actual future state.
9. The current predictive-self model forecasts scalar Φ/coherence values but does not produce an independently projected HDC self-representation. Therefore the temporal identity analyzer must not copy the present vector into the future slot or report perfect present–future alignment by construction. Until a genuine projected representation with a declared horizon and provenance is available, mark present–future and past–future alignment as unavailable, expose measured-pair coverage, and keep full identity coherence unqualified. The scalar forecast and vector-identity question are separate interfaces. The current implementation resets alignment availability on each accepted observation, renders missing pairwise values as `N/A`, scales the aggregate identity contribution by measured-pair coverage, and counts evidence gaps separately from measured low-alignment warnings. A deterministic full-analyzer `TemporalConsciousnessAnalyzer::observe_at` path exercises timestamp admission through downstream-state mutation.

**Aggregate metric rule:** compute the overall coherence score over sufficiently measured components only, and publish a separate evidence-coverage ratio against the nominal component weights. A high score over sparse evidence must not be presented as equivalent to a fully observed score. In particular, a single present/primal observation is not evidence of temporal coherence: the Husserlian metric requires at least one observed relation between retained, present, or predicted states. When no relation is measured, report the metric as unavailable and exclude its nominal weight from aggregate coverage rather than treating an empty comparison set as a perfect score. Per-metric output must likewise expose availability: Φ velocity/trend require at least two retained samples, sample stability requires at least three, acceleration requires two velocity intervals, continuity requires a pair of states, pairwise identity needs actual representations, and binding requires at least one recorded binding. Public continuity/unification predicates must fail closed when their evidence sets are empty, and recommendations should distinguish pending evidence from a measured unhealthy value. Temporal binding strength inputs must be finite and within the declared [0, 1] range; invalid values are rejected before eviction or aggregate-score mutation. Binding component identifiers must also be non-empty after trimming; invalid labels or strengths leave the binding set untouched, even at capacity. Reports render these unqualified values as `N/A`; numeric defaults remain storage values, not evidence. Running temporal-coherence averages exclude accepted observations whose aggregate evidence coverage is zero; the report tracks the contributing observation count and mean evidence coverage so partial-signal averages are not mistaken for full qualification. Health qualification has three states: `HEALTHY` only when all required evidence is complete and measured thresholds pass; `UNHEALTHY` when a measured metric violates its threshold; and `INSUFFICIENT EVIDENCE` when no measured failure is established but one or more required sources are absent. Keep the legacy boolean false unless the state is fully qualified healthy.

**Acceptance test:** when each observation represents the interval ending at its timestamp, feeding equivalent piecewise-constant input at different sample cadences should produce equivalent elapsed-time EWMA values at a common endpoint within declared tolerance. The velocity field is a two-point finite difference and transient slopes are **not** claimed cadence-invariant; compare them only with matched endpoints/sampling semantics or introduce a separately specified windowed estimator. Acceleration is not qualified until three accepted samples provide two velocity intervals. For irregular intervals, compare adjacent interval velocities at their time midpoints, rather than dividing their difference by only the most recent interval. Duplicate or backwards timestamps must be rejected or explicitly classified as unusable, not divided through. Extreme finite samples whose EMA, velocity, or acceleration cannot be represented as finite f64 values must be rejected before mutating the trajectory or any downstream analyzer state. Finite-difference calculations should avoid unnecessary intermediate overflow, and variance/stability calculations must remain finite when finite inputs exceed naive sum/square ranges. EMA interpolation must also avoid false overflow when `1 - decay` rounds to one for a tiny nonzero decay and same-sign extreme finite inputs; preserve the convex interpolation result instead of rejecting it due to intermediate rounding. The lower-level continuity API must enforce finite Φ inputs itself rather than relying on the analyzer wrapper; rejection must leave retained state and event evidence unchanged. Its replay entry point must additionally reject duplicate/backwards elapsed offsets. One continuity instance must not mix host-monotonic and replay-elapsed clock domains, because their coordinates are not comparable, and callers that need an admission result must have a result-bearing host-clock entry point as well as the deterministic replay entry point. The analyzer must preflight both Φ-trajectory and continuity admission before mutating either; because component trackers remain public for compatibility, a caller may already have bound continuity to a different clock domain. A rejected analyzer sample may increment only the rejection counter and must not advance Φ, Husserlian retention, identity evidence, or accepted-observation statistics. Fixed-clock replay must produce reproducible derived metrics; any host-clock timestamps used only for diagnostics must not affect qualification.

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
- retroactive corrections where valid-time and transaction-time queries must preserve different answers;
- requested horizon beyond executed simulation duration fails closed.

Every result is reported against an exact commit, toolchain, schema and fixture corpus. Source presence and authored tests are not test passes.

## 6. Recommended rollout

**Phase 0 — terminology and schema (this RFC).** Agree on field meanings and versioning; document domain mapping per repository.

**Phase 1 — fix high-confidence semantic errors.** In Symthaea, test deterministic elapsed-time observations and cadence-aware filtering. In Symtropy, separate wall-clock recording time from logical sequence and remove second-based event ID generation. In Sol Atlas, label animation time vs model year and freeze the baseline date. In Mycelix, document which validation rules rely on chain order, wall time, or expiry time.

**Phase 2 — versioned envelopes and adapters.** Generate a JSON Schema or equivalent typed contract; add Rust DTOs and explicit adapters per repository. Do not add a universal Rust dependency until the wire semantics and compatibility rules are accepted.

**Phase 3 — cross-system conformance.** Run the same golden event corpus through the four implementations and compare normalization, interval results, causality and replay behavior. Add property tests for monotonicity, transitivity where applicable, conversion round-trips within stated tolerance, and rejection of invalid cross-domain comparisons.

**Phase 4 — temporal planning.** Combine Symthaea's learned predictions with Mycelix's causal/evidence history, Sol Atlas' physical and civilizational horizons, and Symtropy's persistent worldline memory through explicit mappings. Keep the models' uncertainties and timescales separate; integrate evidence, not just scalar scores.

## 7. Greek temporal semantics: lenses, not competing clocks

Ancient Greek vocabulary offers a useful set of conceptual lenses for asking different questions about time. This is **not a canonical ancient taxonomy of mutually exclusive kinds of time**: word meanings overlap, vary by period and author, and later philosophical/religious uses can differ from earlier everyday usage. Use the lenses as explicitly named semantic interpretations over typed temporal data, not as substitutes for clock domains, units, or reference systems.

| Lens | Greek | Historically grounded sense | Engineering question | Candidate use in this ecosystem |
|---|---|---|---|---|
| **Chronos** | χρόνος | Time or duration; in Aristotle's *Physics* IV, time is counted/numbered motion with respect to before and after | How much time elapsed? How long until a deadline? What is the ordered duration? | Monotonic durations, simulation horizon, physical-time rates and forecast calibration |
| **Kairos** | καιρός | The fitting, due, critical or opportune time; in rhetoric, a moment fitted to situation, audience and action | Is this the right moment to act, intervene, speak, or defer? | Symthaea's action window; Mycelix proposal/deadline/expiry decisions; urgency and intervention timing |
| **Aion** | αἰών | In early Greek usage, life, vital force, whole lifetime or generation; in some cosmological and later uses, an age or eternity | What spans a life, generation, era or open-ended continuity? | Civilizational and ecological horizons, intergenerational obligations, continuity and renewal beyond a single project or life |
| **Hora** | ὥρα | A season or naturally/otherwise fixed period, and later or contextually an hour/time of day | What changes with the season, time of day or recurring environmental phase? | Sol Atlas's day/seasonal contexts, light and energy availability, biological/care rhythms |
| **Eniautos** | ἐνιαυτός | A year, yearly cycle, and in some usages a larger recurring period or cycle | What recurs annually, and what does one complete cycle change? | Model-year accounting, annual resource budgets, season-to-season comparison and frozen-year baselines |
| **Rhythmos** | ῥυθμός | Rhythm, cadence or an organized pattern of movement; not simply another word for a clock | What pattern repeats, at what cadence, and how does it adapt? | Symthaea observation cadence and bodily rhythms; Symtropy's event/game/energy cycles; seasonal and civic rhythms |
| **Nun / to nyn** | νῦν / τὸ νῦν | “Now”; in philosophical analysis, the now can be treated as a boundary relating past and future rather than a duration in itself | What is observed or decided now, and what evidence was available at that point? | Decision snapshots, event-time cutoffs, causal audits and leakage-free forecasting |

### How to apply these without collapsing their meanings

- **One event may have several lenses at once.** A heatwave may have a Chronos duration (five days), a Hora context (a seasonal anomaly), a Kairos intervention window (water dispatch must happen before storage runs out), a Rhythmos effect (recurrent demand peaks), and Aion consequences (long-lived infrastructure or ecosystem damage).
- **Kairos is not just “timestamp + priority.”** A meaningful opportunity estimate needs context: the action available, the deadline or window, current conditions, expected consequence, uncertainty and the cost of acting too early or too late. It is a decision assessment, not an objective clock.
- **Aion does not certify literal eternity.** It can motivate life-, generation- or era-scale analysis, and open-ended renewal as a research objective. A finite model can compare finite horizons and resilience under stated assumptions; it cannot prove infinite survival.
- **Hora and Eniautos are not identical.** A Hora-style cycle can be a season, hour or other period; an Eniautos-style lens emphasizes a yearly cycle. Neither supplies a correct calendar by itself: the implementation must name the calendar, timezone, body/location and baseline when those details affect the question.
- **Rhythmos is a pattern over time, not a time reference system.** A 20 ms cognitive sampling cadence, a daily care rhythm and a thousand-year ecological renewal pattern are not interchangeable simply because each has a rhythm.
- **Nun is not the same as the entire present interval.** Store the observation/decision instant and the interval of evidence separately when evaluating what a system could know at decision time.

### A practical type design

Keep two independent pieces of metadata:

1. **Clock domain** — how the coordinate is represented and compared: monotonic elapsed time, UTC wall time, causal/logical position, simulation/worldline tick, astronomical scale, or local calendar.
2. **Temporal lens** — what role the time plays in a claim or decision: Chronos, Kairos, Aion, Hora, Eniautos, Rhythmos, Now, or a typed domain-specific extension.

That allows a Mycelix claim to have a UTC recording timestamp and source-chain causal position, while also being evaluated with a Kairos deadline and Aion-scale consequences. It also lets Sol Atlas use a model-year coordinate with an Eniautos lens without confusing that with elapsed UI playback time.

A future schema might encode the lens as `temporal_aspect`, but it must not make the lens determine units, causality, or physical truth. If the lens is absent or disputed, keep the underlying timestamp and event semantics usable and mark only the interpretation as unknown.

## 8. Grounding references

- Lamport, L. (1978), *Time, Clocks, and the Ordering of Events in a Distributed System*. Defines the “happened-before” partial order and logical clocks. [Microsoft Research](https://www.microsoft.com/en-us/research/publication/time-clocks-ordering-events-distributed-system/)
- Aristotle, *Physics* IV.11, on time as number/count of motion with respect to before and after. [MIT Classics Archive translation](https://classics.mit.edu/Aristotle/physics.4.iv.html)
- Plato, *Timaeus* 37d, on generated time as a “moving image of Eternity.” This is Plato's cosmological account, not a universal definition shared by all Greek authors. [Perseus Digital Library](https://www.perseus.tufts.edu/hopper/text?doc=Perseus%3Atext%3A1999.01.0180%3Atext%3DTim.%3Asection%3D37d)
- Oxford Classical Dictionary, “Aion”: early senses include life, vital force, lifetime and generation, with eternity emerging in cosmological usage. [Oxford Academic](https://academic.oup.com/edited-volume/61673/chapter-abstract/548066302)
- Oxford Classical Dictionary, “Kairos”: the personification of Opportunity and a concept distinguished from Chronos. [Oxford Academic](https://academic.oup.com/edited-volume/61673/chapter-abstract/549669909)
- Liddell–Scott–Jones lexicon via the Greek Language Portal, “ἐνιαυτός”: year and recurring period/cycle. [Greek Language Portal](https://www.greek-language.gr/digitalResources/ancient_greek/tools/liddell-scott/search.html?lq=%E1%BC%90%CE%BD%CE%B9%CE%B1%CF%85%CF%84%CF%8C%CF%82)
- Liddell–Scott–Jones lexicon, “ὥρα”: season, hour and fixed period. [Lexicon entry](https://bible.by/strong/greek/5610/dict/liddell-scott-jones/)
- Oxford Classical Dictionary, “Rhythmics”: rhythmos as organized movement/rhythm, including bodily and musical cadence. [Oxford Academic](https://academic.oup.com/edited-volume/61673/chapter-abstract/552822120)
- W3C, *Time Ontology in OWL*. Models temporal entities as instants/intervals, with duration, temporal reference systems and Allen interval relations. [W3C](https://www.w3.org/TR/owl-time/)
- Rust standard library, `std::time::Instant`. Describes a monotonic nondecreasing clock for measuring elapsed duration, with platform-specific caveats. [Rust documentation](https://doc.rust-lang.org/stable/std/time/struct.Instant.html)
- NASA/JPL NAIF, *SPICE Time Subsystem*. Defines astronomical scales including TT/TDB and the J2000 epoch for ephemeris use. [SPICE documentation](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/time.html)

**Qualification boundary:** this RFC does not claim that the four systems have been fully audited, that every listed concern is a live defect, or that conformance tests pass. Its purpose is to give us a precise, testable agreement before cross-repository integrations become more tightly coupled.
