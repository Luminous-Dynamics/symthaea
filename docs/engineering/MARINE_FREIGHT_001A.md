# MARINE-FREIGHT-001A — fixed-battery screening contract

Status: research contract and synthetic arithmetic fixture. The implementation and authored tests require exact-head Rust/CI execution before they can be considered verified.

Parent: [MARINE-FREIGHT-001 #7444](https://github.com/Luminous-Dynamics/symthaea/issues/7444)  
Related owner map: [TRANSPORT-ENG-001A #6245](https://github.com/Luminous-Dynamics/symthaea/issues/6245) and proposed owner-matrix PR [#6246](https://github.com/Luminous-Dynamics/symthaea/pull/6246)  
Maritime substrate: [PR #1258](https://github.com/Luminous-Dynamics/symthaea/pull/1258) and follow-on [#1259](https://github.com/Luminous-Dynamics/symthaea/issues/1259)  
Maritime external benchmark profile: [#6260](https://github.com/Luminous-Dynamics/symthaea/issues/6260)

## Why this first slice

The useful initial question is not “which ship is best?” It is whether a fully declared candidate has any obvious dimensional, payload, energy, charging, or time-window contradiction before anyone spends effort on higher-fidelity solvers or real-site study.

The standalone Rust crate `symthaea-marine-bench` performs bounded screening arithmetic over explicit inputs. It intentionally does not depend on the maritime runtime substrate: this avoids making a screening calculation a runtime authority and permits development while the separate maritime-core PR is still under review.

## Artifacts

- `crates/domains/symthaea-marine-bench/schemas/marine_freight_reference_case_v1.schema.json` — JSON Schema 2020-12 structure and per-field units.
- `crates/domains/symthaea-marine-bench/fixtures/marine_freight_reference_case_v1.synthetic.json` — reproducible fixture; every input value is synthetic and marked not-for-decision.
- `crates/domains/symthaea-marine-bench/src/assessor.rs` — deterministic checks with explicit `within_declared_bound`, `violates_declared_bound`, or `indeterminate` findings.
- `crates/domains/symthaea-marine-bench/examples/screen_reference_case.rs` — prints a stable JSON report from the frozen fixture.

## Model boundary

The input energy numbers are *given integrated energy values*. The crate does not infer them from hull geometry. Hydrodynamics, propulsion maps, waves/current, shallow-water and blockage corrections must come from an appropriately validated solver or reference data. No new drag coefficient or surrogate is presented as naval-architecture truth.

Current checks are:

1. vessel length, beam, and draft against a declared route envelope;
2. draft plus required under-keel clearance against declared water depth;
3. vessel length and beam against a declared berth envelope;
4. payload demand against declared cargo capacity;
5. loaded-leg and return-leg propulsion plus auxiliary energy against derated usable battery capacity;
6. charge energy available from the lesser of charger power and shore-connection power;
7. idealized recharge duration plus the declared round-trip duration against the service window.

The battery-energy arithmetic is:

`E_usable = E_nominal × f_temperature × f_end_of_life × (1 − f_reserve)`

The duty-cycle energy is the sum of the four declared leg loads. The idealized battery-side recharge energy is:

`E_charge = min(P_charger, P_shore) × t_window × η_charge`

The turnaround estimate assumes constant power and a constant declared efficiency. It does not model power taper, dynamic battery temperature, electrical transients, charging queues, weather-driven load, or maintenance disruptions.

## Evidence and unknowns

- Every known quantity names a provenance record; provenance IDs must be unique.
- Units are checked by the evaluator, not inferred or silently converted.
- Unknown values remain explicit and produce `indeterminate` findings. Unknown is not zero.
- Invalid or missing evidence does not produce a successful comparison.
- Findings remain independent; there is no weighted score that can hide a failed hard constraint.
- The report carries a frozen claim ceiling. Input data cannot rewrite the evaluator into a seaworthiness or operating-authority claim.
- The synthetic fixture is useful only for regression-testing arithmetic and negative controls. Passing it is not evidence about a real vessel, route, battery, dock, or service.

## Hand-checkable synthetic oracle

These values are authored regression inputs, not measurements:

- Duty-cycle energy: `1850 + 125 + 1150 + 110 = 3235 kWh`.
- Derated usable battery energy: `5400 × 0.90 × 0.85 × (1 − 0.15) = 3511.35 kWh`.
- Battery-side recharge energy: `min(600, 750) × 8 × 0.92 = 4416 kWh`.
- Idealized recharge duration: `3235 / (600 × 0.92) ≈ 5.860507 h`.
- Round trip plus recharge: approximately `20.860507 h`, within the synthetic 24 h window.

The arithmetic fixture is arranged to exercise within-bound calculations; negative tests deliberately lower battery/grid limits or make evidence unknown. No fixture value should be reused as a real design assumption.

## Benchmark and exit gates

Proposed gates—not measured results:

- reproduce the hand-calculated duty-cycle and capacity values to within (10^{-9}) kWh;
- reproduce the idealized turnaround calculation to within (10^{-12}) h for this deterministic fixture;
- retain independent findings for every hard constraint;
- verify negative controls for under-capacity batteries, inadequate shore power, route-envelope violations, payload overload, unknown route depth, unknown energy, missing provenance, unknown provenance class, unit mismatch, duplicated provenance IDs, invalid reserve factors, and attempted claim-ceiling escalation;
- ensure no unknown/malformed input can become a favorable result through a default value;
- compare alternative designs only on matched mission schedules and declared boundaries, then report a Pareto set rather than a magic scalar.

The first subsequent expansion should add configuration composition for conventional-fuel reference, fixed battery, swappable energy containers, and optional wind assistance. Those should be treated as separate architectures, not as a presumed ranking. Only after inputs and owners are mapped should the project consume validated hydrodynamics. The ITTC maintains a community benchmark repository and a recommended CFD validation database for resistance and propulsion: https://www.ittc.info/benchmark-repository/ and https://www.ittc.info/media/11956/75-03-02-02.pdf.

## Not claimed

This tranche provides no validated hydrodynamic prediction, optimized hull, stability or strength result, safe docking design, zero lifecycle emissions, cost superiority, statutory/class approval, seaworthiness, operating authority, or physical qualification. Real-site and prototype work requires appropriately qualified naval-architecture, battery/electrical, terminal/structural, operator, and approval owners.
