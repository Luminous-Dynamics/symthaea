# Atelier Lightshell — Zone Requirements Contract v0.1

## Purpose

This contract makes functionally graded textile design executable. Each robot zone receives its own requirement vector, test methods and hard constraints. It prevents a global material score from hiding a trade-off between impact response, vibration isolation, ventilation, articulation, thermal effects, sensor visibility and repairability.

**This is a requirement-admission contract, not a material selector or qualification engine.** A valid proposed manifest says that the design intent is structurally complete enough to discuss. A frozen manifest means the exact robot zone, frame, operating domain, method and acceptance rule are bound. Neither state proves that a textile performs, that the garment is safe, or that a robot is certified.

## Functional-role to metric vocabulary

| Functional role | Compatible measurements |
| --- | --- |
| impact_compression | compression_energy_j_g; transmitted_force_n |
| vibration_isolation | vibration_transmissibility_ratio |
| airflow | air_permeability_mm_s |
| thermal_management | temperature_delta_c |
| articulation | joint_rom_delta_deg; flex_fatigue_cycles; minimum_clearance_mm |
| abrasion | abrasion_cycles |
| fluid_chemical_barrier | fluid_ingress_mass_g_m2; chemical_mass_change_pct |
| cleanability | cleaning_residue_fraction |
| sensor_accommodation | sensor_occlusion_fraction |
| repairability | repair_time_min |

Units are canonical per metric. Comparisons use higher_is_better, lower_is_better or within_declared_band, with exact direction semantics checked by Rust. Numeric limits may remain null in proposed/requirements_pending states; a frozen requirement requires valid bounds, method references and acceptance rules. This avoids inventing thresholds before robot CAD, ODD, material capability and measurement systems are known.

## Lifecycle

1. **proposed** — synthetic or exploratory zones; CAD identity, ODD and thresholds may be unknown.
2. **requirements_pending** — requirements are being reconciled with the real robot article, hazard profile, clearance, thermal budget, service access and measurement capability.
3. **frozen** — exact article revision and geometry digest are bound; every zone has an explicit frame and clearance-review reference; each requirement has canonical units, compatible role/metric semantics, numeric bounds, method and acceptance rule.

A frozen requirement is not 'qualified'. Physical coupon, module and integrated-robot evidence remain separate stages in the Lightshell qualification manifest. Failed and unknown evidence cannot be converted into passes by this contract.

## Design principles

- **Zone-local Pareto portfolios:** choose separately per zone; do not collapse all material functions into one weighted protection score.
- **Hard constraints first:** motion/clearance, heat, cooling, optical access, cable routes, retention and service access are non-compensatory.
- **Evidence follows the claim:** photographs may classify visible construction, but cannot establish strength, modulus, impact response, vibration transmissibility, wash retention or fatigue life.
- **Uncertainty and unknowns stay visible:** a missing CAD frame, test method, calibration or acceptance rule blocks freezing.
- **Replaceability is engineered:** a replaceable zone must have a named mechanical interface and inspection access before the brief can freeze.
- **No authority escalation:** the contract cannot authorize motion or manufacture, nor replace robot/system risk assessment.

## P0 validation controls

The Rust validator checks duplicate zones and function/metric pairs, known role and metric names, compatible role-to-metric mappings, canonical units, exact comparison direction, ordered bounds, geometry-digest format, CAD/frame/clearance references, frozen method and acceptance references, serviceability, and fail-closed evidence policy. Unknown JSON fields are rejected.

Authored tests cover a valid proposed portfolio; duplicate zones; unknown roles; a role without a compatible measurement; duplicate metrics; contradictory bounds; unit mismatch; frozen briefs missing CAD/method evidence; replaceable zones missing service interfaces; malformed geometry digests; and unknown fields.

## Benchmark plan

Compare zone-based design decomposition against (a) experienced manual engineering decomposition and (b) a single-global-material baseline. Record: function/zone coverage; critical hard-gate omission and false-accept rates; traceability from zone to CAD/ODD/method/acceptance rule; number of requirement edits and rework loops; and, after physical pilot data exists, physical fit iterations and test rework.

Critical false acceptance should be treated as a hard failure, not averaged into a weighted score. Numeric improvement targets should be preregistered only after the robot article, measurement system, process capability and human-expert baseline are frozen.

## Evidence and standards context

The 2026 Yan, Wang and Yu study reports that spacer-course architecture shifts the tested compression/impact versus vibration-isolation trade-off; square-wave constructions showed multiple resonant peaks, and durability/washability were not tested. Lightshell must add resonance-aware frequency-response testing and post-aging verification rather than treating a paper's force-reduction percentage as an acceptance threshold.

Spacer fabrics need structure-aware fixtures: Schwager et al. (2022) document that conventional flat-textile test arrangements can pre-compress thick spacer samples or fail to accommodate them, distorting comparisons.

RGBench (AAAI 2026) is a comparator for cloth dynamics/manipulation simulation, not a protective robot-skin qualification benchmark. Its public real-capture subset should be treated as a limited starting point, not full evidence of robot-mounted garment fit or lifecycle durability.

Robot safety references must be selected by robot class and application. ISO 10218-1:2025 and ISO 10218-2:2025 address industrial robot/system scopes; ISO 18646-5:2026 evaluates legged-robot locomotion performance and explicitly is not a safety verification standard; ISO/TS 15066:2016 addresses collaborative industrial robot systems and is in a revision state. No Lightshell manifest should declare blanket standards compliance.

## References

- Yan, Wang & Yu (2026), [Influence of Connective Architectures of Inlaid Weft-Knitted Spacer Fabric](https://doi.org/10.3390/polym18020151).
- Schwager et al. (2022), [Development and Standardization of Testing Equipment and Methods for Spacer Fabrics](https://doi.org/10.1007/s10443-021-09959-y).
- [RGBench — Real Garment Benchmark](https://rgbench.github.io/).
- [ISO 10218-1:2025](https://www.iso.org/standard/73933.html) and [ISO 10218-2:2025](https://www.iso.org/standard/73934.html).
- [ISO 18646-5:2026](https://www.iso.org/standard/86850.html).
- [ISO/TS 15066:2016](https://www.iso.org/standard/62996.html).

## Current state

The example manifest is synthetic and intentionally proposed. No CAD article revision, numeric performance threshold, material lot, machine process, coupon, physical test, or robot qualification is claimed. Rust tests are authored in source; compilation/test execution and exact-head CI remain pending.