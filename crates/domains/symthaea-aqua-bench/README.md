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
- `docs/AQUA_ESSENTIAL_SERVICE_CONTINUITY_PROTOCOL_V0_1.md`
- `docs/AQUA_WASHSTATION_BENCHMARK_V0_1.md` — South African SANS Table 3 fixture-specific flow and design-pressure references.
- `docs/AQUA_LOW_PRESSURE_VALIDATION_PROTOCOL_V0_1.md` — pressure datum, 50–550 kPa test matrix, comparator eligibility and unoccupied validation sequence.
- `docs/AQUA_COMPETITOR_LEDGER_V0_1.md` — SANS clause/table ambiguity, Johannesburg by-law boundaries and exact commercial shower comparators.
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
- requires dynamic pressure at the terminal fitting inlet and does not substitute static or upstream residual pressure;
- checks the measured pressure plus its uncertainty against a preregistered setpoint tolerance and the complete test-article pressure-rating envelope;
- reports manufacturer-declared operating-range status separately from test-rig admission, so an out-of-declared-range comparator point remains visibly exploratory;
- validates exact-point matched comparisons using a shared context/protocol digest and a preregistered maximum pressure separation;
- does not interpolate an untested pressure or infer a task/comfort result.

Twenty negative/contract tests are authored. Manufacturer pressure evidence may specify a minimum only, a maximum only, or a complete range; the validator preserves that distinction and never invents an unpublished bound. The module validates measurement context only; Mycelix retains metrology admission and Symtropy retains hydraulic/resource authority.
