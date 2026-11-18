# symthaea-aqua-bench

Product-level comparator-context adapter for Luminous Aqua.

## Scope

This crate does **not** implement a water ledger, energy ledger, hydraulic/thermal solver, water-quality assessment, metrology passport, safety interlock, hygiene assessor or comfort model.

It checks that a pair of Aqua benchmark runs use matching:
- task profile and duration;
- inlet-water profile;
- supply/delivery conditions;
- room conditions;
- measurement protocol;
- complete-service accounting boundary.

Each run must point to canonical owner evidence:
- Symtropy water-service T0 receipt;
- Mycelix metrology admission;
- Mycelix intended-use water-quality assessment;
- independent task-outcome assessment;
- Symtropy resource-closure receipt when energy is compared.

Device configuration and cleaning protocol are allowed to differ between comparators, but their exact digests must be recorded. That is essential: different filter/cleaning requirements are part of the product's real resource burden and must remain visible in the upstream receipts.

## Non-duplication

- Symtropy #4978 owns deterministic water-service T0 accounting/evaluation.
- Symtropy #5364 owns household/community water-service continuity.
- Symtropy #5425 owns water-specific lifecycle cost per verified service.
- Symtropy #2152 owns physical resource stock/flow/energy/residual closure.
- Mycelix #6025/#6029/#6034/#6048/#6054/#6057 own metrology, measurand, calibration, uncertainty, timebase and applicability.
- Mycelix #6100 owns raw-water observation versus use-specific quality assessment.
- Sol-Atlas #765/#766/#770/#772 own pool/water-feature thermal, evaporation and microclimate qualification.

The Aqua adapter only binds the product comparison to those owner outputs. It cannot validate or replace them.

## Authored test inventory

- Matched-run evidence adapter: 18 unit tests.
- Spray-field summary and matched degradation comparison: 15 unit tests.
- Service-continuity classifier: 17 unit tests, including exhaustive enumeration of all 57,600 categorical input combinations.
- Exact-point pressure/flow evidence: 40 unit tests, including one-sided manufacturer pressure-range declarations, uncertainty-aware paired retention and pressure-datum mismatch negative controls.
- Method-specific UWL / ETL pressure and boost-function records: 22 unit tests for separate hot/cold inlet channels, asymmetric P1/P2/P3 intervals, faceplate-centre P2, calibration receipts, stabilized-flow admission, boost reset, Qreg/Qboost/Qmax and labelled regular-flow evidence.
- Task-phase capability validator: 20 unit tests for exact variant/configuration/method/conditions binding, consent, local hold-to-boost auto-return, duration limits, phase order and rejection of unqualified capabilities.

**132 tests are authored across the six modules.** This is a source inventory only; no exact-head run, rustfmt or Clippy result is claimed until the dedicated Aqua workflow reports it.

## Run

```sh
cargo test -p symthaea-aqua-bench
```


## Aqua programme interfaces

Product and benchmark contexts are defined in:
- `docs/AQUA_SHOWER_SERVICE_CONTRACT_V0_1.md`
- `docs/AQUA_SHOWER_BENCHMARK_V0_1.md`
- `docs/AQUA_WASHSTATION_BENCHMARK_V0_1.md`
- `docs/AQUA_THERMAL_SUITE_BENCHMARK_V0_1.md`
- `docs/AQUA_DHW_CASCADE_BENCHMARK_V0_1.md`
- `docs/AQUA_REFERENCE_BATHROOM_V0_1.md`
- `docs/AQUA_REFERENCE_ROOM_ARCHETYPES_V0_1.md`
- `docs/AQUA_OWNER_INTEGRATION_MAP_V0_1.md`

The adapter remains deliberately small: it binds paired Aqua test runs to equivalent conditions and canonical owner evidence. New fixture classes and room modes must not add a second water, energy, hydraulic, water-quality, metrology, thermal, pool or safety authority.
- `docs/AQUA_MODULAR_PLATFORM_ARCHITECTURE_V0_1.md`
- `docs/AQUA_SPRAY_FIELD_METROLOGY_PROTOCOL_V0_1.md`
- `docs/AQUA_SPRAY_GEOMETRY_DOE_PROTOCOL_V0_1.md`
- `docs/AQUA_BOUNDED_BATHING_SERVICE_COMPILER_V0_1.md`
- `docs/AQUA_TASK_SELECTIVE_DELIVERY_PROTOCOL_V0_1.md` — base-flow / explicit boost task phases and matched complete-task benchmark.
- `docs/AQUA_ESSENTIAL_SERVICE_CONTINUITY_PROTOCOL_V0_1.md`
- `docs/AQUA_WASHSTATION_BENCHMARK_V0_1.md` — South African SANS Table 3 fixture-specific flow and design-pressure references.
- `docs/AQUA_LOW_PRESSURE_VALIDATION_PROTOCOL_V0_1.md` — pressure datum, 50–550 kPa test matrix, comparator eligibility and unoccupied validation sequence.
- `docs/AQUA_COMPETITOR_LEDGER_V0_1.md` — SANS clause/table ambiguity, Johannesburg by-law boundaries and exact commercial shower comparators.
- `docs/research/aqua-evidence-corpus-v0.1.json` and `docs/research/AQUA_EVIDENCE_CORPUS_V0_1.md` — source-classified evidence claims and synchronized rendering; schema/anti-laundering validation runs in the dedicated Aqua workflow.
- `docs/AQUA_AQS1_INTERFACE_AND_FAULT_CONTRACT_V0_1.md`
- `AQUA-ADAPTIVE-EXPERIENCE-001 #12712` — optional privacy-preserving preference/task assistance.


## Spatial spray-field analysis

The dependency-free spray_field module summarizes a complete, externally qualified deposition grid:
- target-area coverage against a preregistered deposition-density threshold;
- area-weighted target mean and coefficient of variation;
- mapped off-target mass fraction and raw mass totals;
- fail-closed handling for incomplete capture, duplicate/missing cells, invalid areas/masses and zero mapped mass.

A second helper, `compare_spray_field_to_baseline`, reports matched baseline-to-scale, partial-blockage or post-clean deltas for the same specimen, geometry revision and frozen test context. It rejects cross-specimen, cross-geometry and cross-condition comparisons and validates the public summary's derived-field consistency.

This is descriptive post-processing only. It does not calibrate instruments, prove capture completeness, assess comfort or hygiene, replace ASME/CSA or WaterSense methods, or authorize a physical test. The full protocol is in docs/AQUA_SPRAY_FIELD_METROLOGY_PROTOCOL_V0_1.md.


## Essential-service continuity evaluator

The dependency-free `service_continuity` module classifies externally supplied, owner-qualified state into:
- QUALIFIED_FULL
- QUALIFIED_REDUCED
- MANUAL_ONLY
- ISOLATED_STOP
- RECOVERY_PENDING

It fails closed on unqualified/unknown water-use status, local protection uncertainty, stale or contradictory critical evidence, pressure outside/unknown relative to the admitted envelope, blocked/unknown drainage and detected/unknown leaks. Reduced and manual-only modes are admitted only when explicitly qualified by the local device authority. Loss of optional digital recommendations does not reduce a qualified local service state. A cleared fault cannot automatically clear the requalification-required flag.

This is a product-level classification function only. It does not actuate isolation, control water, establish safety limits, approve water quality, or replace independently qualified hardware protection. Seventeen unit tests are authored, including exhaustive enumeration of all 57,600 combinations in the finite input vocabulary. No execution pass is claimed until exact-head CI reports it.


## Exact-point pressure / flow evidence

The dependency-free `pressure_curve` module validates exact-point pressure / flow observations. It:
- requires dynamic pressure at an explicit datum (terminal fitting inlet or point of discharge) and does not substitute static or upstream residual pressure;
- checks the measured pressure plus its uncertainty against a preregistered setpoint tolerance and the complete test-article pressure-rating envelope;
- reports manufacturer-declared operating-range status separately from test-rig admission, so an out-of-declared-range comparator point remains visibly exploratory;
- validates exact-point matched comparisons using a shared context/protocol digest and a preregistered maximum pressure separation;
- does not interpolate an untested pressure or infer a task/comfort result.

Forty pressure-curve / pressure-retention contract tests are authored. Manufacturer pressure evidence may specify a minimum only, a maximum only, or a complete range; the validator preserves that distinction and never invents an unpublished bound. If the source pressure datum is missing or differs from the observed datum, it returns DeclaredDatumNotSpecified or DeclaredDatumMismatch instead of claiming a pressure-range match. The paired low/high-pressure comparison requires the same specimen, exact variant, mode, geometry, pressure-invariant context, protocol, rating evidence and instrument IDs; it reports central flow retention plus conservative measurement-uncertainty bounds. This is not a confidence interval across specimens, does not interpolate and does not determine task success. The module validates measurement context only; Mycelix retains metrology admission and Symtropy retains hydraulic/resource authority.


The paired pressure-response metric has a useful external comparator: the UK government's non-domestic Energy Technology List requires at least 60% low-pressure flow relative to maximum flow at 3 bar, tested at 1.5 bar, when the 3-bar flow is no more than 8 L/min. Aqua's proposed 70% initial and 80% stretch values are internal research targets, not certification criteria. See docs/AQUA_LOW_PRESSURE_VALIDATION_PROTOCOL_V0_1.md.


## Dynamic supply qualification

The low-pressure protocol defines exact steady-state pressure points. Time-varying pressure, hot/cold supply disturbances, second-fixture demand, pump/regulator cycling and restoration are a separate event-driven qualification programme: see docs/AQUA_DYNAMIC_SUPPLY_TRANSIENT_PROTOCOL_V0_1.md and issue #12801. Event scripts, measurement channels, timebase, uncertainty, recovery and safe-stop outcomes must be frozen separately; no transient performance is claimed until unoccupied tests exist.

## Power-coupled shower / wastewater heat recovery

The 2025/26 UK ETL electric-shower pathway now makes the coupled system boundary especially relevant: the programme retains 4.5 L/min as standard electric-shower maximum, with conditional 9 L/min (power modulation + compliant wastewater heat recovery) and 10.6 L/min (plus UBF) pathways. These are non-domestic UK criteria, not Aqua results or South African requirements. The detailed benchmark is in `docs/AQUA_POWER_COUPLED_SHOWER_DWHR_BENCHMARK_V0_1.md` and issue #12816. It requires whole-system purchased-energy measurement, receiver-side useful heat, fouling and task-level water/energy accounting. No duplicate thermal ledger is implemented here; Symtropy remains the canonical heat/resource authority.


## Shower test-method and pressure-datum crosswalk

See docs/AQUA_METHOD_CROSSWALK_V0_1.md and issue #12817. The pressure_curve module records TerminalFittingInlet or PointOfDischarge and rejects cross-datum comparisons. Its single-channel record deliberately does not represent UWL's simultaneous hot/cold inlet pressures or ETL's faceplate-centre P2; those require separate method-specific channel records and apparatus. The generic helper cannot establish EN 18021, UWL, ETL or WaterSense conformity.


## Method-specific UWL and ETL pressure records

The dependency-free standard_methods module keeps method-specific channels separate from the generic pressure_curve record:
- UWL pressure points retain hot and cold inlet channels independently for Type 1 P1 / P2 and Type 2 P3, with the published asymmetric pressure intervals and per-channel instrument / calibration receipts.
- ETL maximum-flow evidence is represented as a distinct faceplate-centre P2 pressure observation plus stabilized flow and its stabilization/calibration receipts.

These validators admit the shape and numerical limits of method evidence only. They do not perform the complete UWL/ETL procedures, validate all spray coverage or flow requirements, or establish product conformity. Formal claims still require the controlled protocol and independent method execution.


## Task-selective delivery and UWL boost qualification

The `standard_methods::UwlBoostFunctionObservation` record validates the UWL 2023 Clause 12.1 boost sequence at Type 1 P2: regular flow, boosted flow, flow-control close, reopen without boost, then post-close flow. It preserves separate hot/cold inlet pressure channels, flow calibration provenance and a digest for evidence that Qreg is listed on the scheme label. The check requires boost reset after closure, a resolved flow increase, a boosted upper measurement bound below twice the regular lower bound, and post-close flow consistent with regular flow within combined declared measurement uncertainty. This is a method-evidence validator, not UWL product conformity.

The `task_phase` module validates advisory PREPARE / BASE_DELIVERY / OPTIONAL_BOOST / PAUSE / RINSE / RETURN_TO_READY schedules against exact independently qualified capabilities, market variant, hardware and delivery-condition digests. Boost phases require explicit task consent, a direct local user hold with auto-return qualification, and the capability's locally qualified maximum duration. Manufacturer-declared-only or ambiguous boost modes cannot be selected as qualified task-plan capabilities. The summary contains durations only; Symtropy retains water/energy closure and no AI actuator command is emitted.

Full preregistration and exact competitor ambiguity: `docs/AQUA_TASK_SELECTIVE_DELIVERY_PROTOCOL_V0_1.md`, issue #12798. The Hansgrohe 28036000 public sheet's boosted 6.6 / 9.4 L/min values conflict with its listed 4 L/min maximum/regulator; this remains unresolved pending manufacturer clarification or independent exact-SKU test.


## Repairability and serviceability evidence

The dependency-free serviceability module validates exact repair-job observations and summarizes a preregistered cohort. It records access outcome, disassembly depth, tools, spare availability, technician skill/time, downtime, source resource observations, post-repair function and independently authorized return-to-service evidence.

It only counts a routine job as a non-destructive qualified repair when:
- dry-side or wet-side access was completed without destructive finish removal;
- the post-repair function check passed;
- the return-to-service state is explicitly QualifiedReady;
- an independent device-authority receipt exists; and
- any required Mycelix water-quality or metrology requalification receipt exists.

Unknown access and pending/failed/unknown requalification remain visible and do not count as successes. A zero routine-job denominator yields an undefined fraction, not 0%. The evaluator reports median technician and downtime minutes and tool count, but does not calculate lifecycle cost or water/energy/environmental totals. Those remain with Symtropy and Mycelix.

Twenty unit tests are authored; no current-head test or Clippy pass is claimed.


## Repairability evidence and post-repair qualification

The new dependency-free serviceability module validates repair-job observations and summarizes a frozen cohort. It keeps destructive finish removal, unknown access, failed/pending repair and missing return-to-service evidence visible. A job only counts as a qualified non-destructive routine repair when access is non-destructive, the post-repair function check passes, an independent device-authority receipt exists, and any required Mycelix water-quality or metrology receipt exists.

The module reports medians and counts, not lifecycle cost or resource totals. It leaves canonical cost/resource/environmental accounting with Symtropy and Mycelix. The protocol and frozen job manifest are in docs/AQUA_REPAIRABILITY_SERVICEABILITY_PROTOCOL_V0_1.md and issue #12844.


## Product-category and standard-clause-bound task capabilities

The task-phase compiler now binds each plan and qualified mode to an explicit fixture category (Tap, ShowerOutlet, ShowerSystem, or OtherSanitaryFixture) and a qualification method scope. It rejects an EN 18021 tap boost clause (7.3.3) being used to qualify a shower-outlet mode (8.3.2), and it binds shower-system boost to its own clause (10.3.8). UWL 2023 boost qualification is recorded as its own route. The generic independently qualified method route is explicitly not a claim of conformity to one of those standards.

This is a scope-integrity contract only. It does not implement EN 18021 shower clauses, UWL certification, boost hardware, or actuator control. The shower-specific numerical acceptance requirements of EN 18021 need the controlled full standard before implementation. The task-phase module now has 23 authored tests; none is claimed executed on the current head.
