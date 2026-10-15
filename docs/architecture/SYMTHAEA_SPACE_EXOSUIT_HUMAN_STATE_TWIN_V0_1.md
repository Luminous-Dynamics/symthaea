# Symthaea Space Exosuit: human-state evidence twin v0.1

**Status:** simulation/research contract and initial Rust implementation. Not a medical monitor, human-rated subsystem, or operational suit controller.

**Track:** SX-023 / [issue #1537](https://github.com/Luminous-Dynamics/symthaea/issues/1537). This first patch is intentionally a narrow, simulation-only leaf on the SX-022 branch. It adds evidence fusion and a water-inventory ledger; it does not implement a complete physiology model or change the assist/survival control path.

## Why this layer exists

The integrated EVA work already couples workload and metabolism to oxygen demand, CO2 burden, humidity, equipment heat, PLSS thermal storage, radiator derating, powered-glove work, power buses, pressure-integrity estimates, and safe-haven planning. The remaining human-state concern is not simply “read more wearable sensors.” It is to preserve what is known, how it is known, when it was sampled, and when measurements disagree.

NASA treats a spacesuit as a habitable life-support system, not just an exoskeleton. Relevant functions include ventilation/CO2 removal, oxygen and pressure, thermal management, hydration, communications, and waste management. See the [NASA Life Support Subsystems overview](https://www.nasa.gov/reference/jsc-life-support-subsystems/), [Spacesuit Basics](https://www.nasa.gov/centers-and-facilities/johnson/spacewalk-spacesuit-basics/), and active [NASA-STD-3001 Volume 2, Revision F](https://standards.nasa.gov/standard/NASA/NASA-STD-3001_VOL_2) (dated 2026-07-14).

## Implemented in v0.1

`crates/domains/symthaea-exoskeleton/src/human_state.rs` provides:

- Typed metrics for heart rate, respiratory rate, core/skin temperature, potable-water inventory, local fatigue, whole-body metabolic power, and wearer-reported exertion.
- Evidence provenance (`Instrument`, `WearerReported`, `ModelEstimate`, `ModelPrediction`), monotonic mission timestamp, confidence, absolute uncertainty, and source identity.
- A freshness gate; future-dated, invalid, low-confidence, and stale observations are excluded and counted.
- Per-source latest-sample de-duplication, so a time series from one sensor is not misrepresented as multiple simultaneous independent sensors.
- Unit-specific, explicitly illustrative disagreement tolerances. Conflicting fresh sources yield `Conflicted` with no fused value; they are not averaged into a falsely reassuring result.
- An explicit `Unknown`/`Missing` state when there is no acceptable evidence. No missing value defaults to a nominal/healthy value.
- Conservative intervals spanning accepted uncertainty ranges and source confidence that does not exceed the weakest accepted observation.
- A hydration inventory ledger that accounts for starting water, independently verified replenishment, and consumption. Invalid updates and overdraw are rejected without mutating inventory.

`human_state_fault_campaign.rs` supplies deterministic regression cases for stale data, contradictory temperatures, water-inventory disagreement, low-confidence wearable input, invalid/outlier data, and separating local fatigue from whole-body workload.

## Safety and authority boundary

This module intentionally has **no** actuator command, motor permission, oxygen/pressure valve, purge, thermal controller, rescue propulsion, medical diagnosis, or emergency-response command surface. Its outputs are evidence records for later planning/diagnostic consumers only.

The following invariants are non-negotiable:

1. Missing, stale, future-dated, non-finite, out-of-domain, or low-confidence data never becomes a nominal value.
2. Conflicting accepted sources remain visibly conflicted; the fusion layer does not conceal disagreement through averaging.
3. Self-reported symptoms/exertion remain separately inspectable when compared with instruments or models.
4. Local fatigue and whole-body workload stay separate dimensions; no aggregate score may erase a limiting dimension.
5. Water inventory is conserved; a predicted demand is not evidence that water exists, and a service advertisement is not a verified transfer.
6. Human-state estimates cannot mask pressure, oxygen, CO2, radiation, or other hard survival-system limits. Those limits remain independent and take precedence.
7. Simulation outputs cannot be upgraded to bench, human-in-the-loop, or qualification evidence by configuration, model confidence, signatures, or a passing software test.

## Tolerances and evidence

Every numerical tolerance in `HumanStateConfig::simulation_reference()` is a **synthetic software-test parameter**. It is not a physiological reference range, calibrated sensor specification, diagnostic cutoff, individualized baseline, NASA requirement, or medical prescription. Domain checks for normalized fields (for example, fatigue fraction and exertion scale) are input-shape checks only.

Likewise, no universal hydration prescription is inferred. NASA's suited-operations water provisions are a design/reference input for requirements analysis, not an individualized clinical recommendation. Future demand models must state their workload/environment assumptions, uncertainty, hydration source, and wearer-specific evidence protocol explicitly.

## Next engineering increments

1. **Policy envelope:** add an advisory-only recommendation layer that can suggest reducing workload, rest/hydration, or return-to-safe-haven planning. It must consume uncertainty and explicit mission context, and must never replace independent pressure/oxygen/CO2/thermal/radiation limits.
2. **Thermal-human coupling:** separately track human heat-storage/temperature observations and the PLSS thermal store; include response lag and model uncertainty. Do not label a heuristic/HDC output equivalent to validated physiology models.
3. **Metabolic/fatigue model:** combine locomotion, pressure-garment resistance, powered-glove effort, task duration, and recovery intervals. Keep local muscular fatigue distinct from whole-body aerobic workload and separate observed, estimated, predicted, and unknown states.
4. **Broader PLSS accounting:** deepen ventilation flow and CO2 washout, humidity/condensation, trace contaminants, oxygen-path state, thermal rejection, potable-water quality/inventory, waste capture, power and thermal budgets, sensor disagreement, maintenance, and degradation.
5. **Evidence-backed fault campaigns:** exercise coupled failures such as high workload plus radiator degradation, primary physiology-sensor loss during communications outage, apparently nominal vitals during a pressure emergency, wearer injury plus unavailable rover, and heat/water shortfall with no reachable safe haven. The campaign must prove that optimistic human-state estimates never clear a hard survival hazard.
6. **Calibration ladder:** software synthetic cases → instrumented sensor replay → calibrated bench/thermal rigs → healthy unsuited human protocol → exercise/pressure-glove protocol → partial/full suited human-in-loop work → relevant environmental analogs → program-specific verification and qualification.

## What Symthaea can and cannot engineer

Symthaea can contribute to requirements traceability, Rust reference models, deterministic simulation, design-space exploration, parameter sensitivity, fault injection, evidence manifests, and reviewable CAD/thermal/structural-analysis workflows when suitable tools and validated solvers are connected. It cannot establish that it has safely “engineered its own exosuit” merely by generating a design or passing software tests. Human life-support hardware requires domain-engineer review, instrumented physical tests, independent verification, documented failure analysis, and the applicable program's qualification process.

This patch is a small step toward that workflow: make the evidence richer and more honest before adding more autonomy.
