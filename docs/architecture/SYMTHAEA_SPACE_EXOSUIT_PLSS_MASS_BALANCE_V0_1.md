# Symthaea Space Exosuit: PLSS atmosphere mass-balance reference model v0.1

**Status:** simulation-only software model. Not a validated atmosphere/physiology model, medical monitor, or physical PLSS controller.

**Implementation branch:** `fix/space-exosuit-plss-mass-balance`

**Research track:** [SX-024 — PLSS atmosphere, CO2 and humidity model](https://github.com/Luminous-Dynamics/symthaea/issues/7459), following [SX-022](https://github.com/Luminous-Dynamics/symthaea/pull/1535).

## Problem addressed

The prior reference implementation used total ventilation flow times an efficiency fraction as the amount of CO2 removed. That quantity is not a justified CO2 capture rate: fan throughput is circulated gas, not CO2 captured. It could also force the calculated residual CO2 burden to zero independent of the CO2 production rate or sorbent exhaustion.

The prior humidity burden was unitless, while generation and removal were computed using unlike implied units. Those values should not be interpreted physically.

This patch replaces both shortcuts with explicitly dimensioned, finite-capacity *reference models*. It does not claim the coefficients are measured or representative of any qualified suit.

## CO2 model

- `co2_burden_l` is excess unremoved CO2-equivalent volume above a synthetic initial bulk-circuit condition. Its reference pressure and temperature are explicit configuration values; the metabolism model's original gas-volume convention is still an assumption that requires calibration.
- `circuit_co2_partial_pressure_pa()` converts that excess reference-condition volume to a bulk-circuit partial-pressure estimate using the ideal gas law, circuit free volume, and uniform circuit gas temperature.
- `co2_scrubber_removal_rate_l_min` is the configured maximum CO2-equivalent capture rate at the configured reference conditions. It is separate from `ventilation_flow_l_min` and is not calibrated against a particular sorbent.
- `max_circuit_co2_partial_pressure_pa` is a synthetic mission-planning screening ceiling. It is **not** an exposure limit and must not be compared directly with the inspired CO2 partial-pressure (PICO2) values in human-suit requirements.
- `co2_scrubber_capacity_l` is a finite total capacity. Remaining capacity is stateful and decremented only by modeled capture.
- The modeled capture per step is bounded by all three: CO2-equivalent burden available, rate × step duration (and modeled path availability), and remaining cartridge capacity.
- Scrubber degradation halves the synthetic reference rate; a failed scrubber captures zero. A degraded ventilation path reduces the declared scrubber rate; a failed ventilation path retains the existing mission-abort behavior.
- Mission preflight uses both current and projected bulk-circuit partial pressure. Invalid estimates fail closed to a safe-haven recommendation; a projected excursion past the synthetic ceiling prevents task start.
- This is a uniformly mixed circuit estimate. It does **not** model helmet flow fields, breathing waveform, dead space, local rebreathing, inlet/outlet placement, or inspired CO2; it cannot demonstrate compliance with a human-suit PICO2 requirement.
- No scrubber regeneration, pressure-dependent gas flow, leak coupling, sorbent chemistry, breakthrough curve, or temperature/moisture cross-sensitivity is claimed in v0.1.
- A failed path or exhausted cartridge/collector returns a `ReturnToSafeHaven` planning disposition before the next task segment begins. The segment-level model does not determine a validated safe exposure duration.

## Humidity model

- `humidity_generation_g_per_min` is water-vapor generation as water mass, in grams per minute.
- `humidity_vapor_burden_g` is the water mass awaiting removal in this reduced-order model. It is not relative humidity, vapor partial pressure, dew point, or a validated cabin moisture state.
- `humidity_removal_rate_g_min` sets the maximum modeled removal rate.
- `humidity_collector_capacity_g` is finite condensate storage. The remaining capacity is decremented by removed water mass; when the collector is full, removal stops and vapor burden accumulates.
- A failed humidity-control path removes no water. A degraded path is a synthetic 50% rate multiplier.
- Condensation thermodynamics, surface temperatures, distributed humidity, liquid transport, microbial/contaminant handling, and recovered-water processing are not modeled here.

## Safety and numerical behavior

- All numeric reference values remain tagged `Simulation`.
- Non-finite derived mass-balance quantities are rejected rather than converted to zero.
- Oxygen inventories, residual gas/water state, and the thermal store are checked for finite, in-range state before a step is accepted.
- Oxygen allocation is planned against both usable inventories before either one is mutated. An insufficient-supply error does not partially drain primary or secondary oxygen.
- Thermal arithmetic (including source summation and step integration) is checked before inventory or gas/water mutation. The established `NoThermalPath` semantics remain explicit: the modeled segment exchange is recorded, then the unavailable independent cooling path is reported.
- Negative oxygen/CO2 generation rates and negative metabolic input values are invalid.
- CO2 removal cannot exceed available modeled CO2; capture cannot exceed the declared rate or remaining capacity.
- Humidity removal cannot exceed the available modeled water vapor, declared rate, or remaining collector capacity.
- The module has no command surface for oxygen regulators, pressure valves, ventilation hardware, scrubbers, thermal control, motors, or propulsion.
- The existing deterministic assist/survival boundary remains independent of this simulation.

## Regression coverage added

The PLSS and integrated EVA regression suites now include:

1. CO2 capture capped by finite sorbent capacity even at high fan flow.
2. CO2 capture rate remains independent of artificially increasing fan throughput.
3. Ideal-gas conversion from one liter of reference-condition excess to bulk-circuit pressure.
4. Projected pressure uses bounded capture and does not mutate the live state.
5. Failed scrubber/ventilation paths are represented conservatively.
6. Humidity removal is capped by finite condensate-storage capacity and failed control lets vapor accumulate.
7. Overflowing humidity input is rejected before oxygen inventory is consumed.
8. Insufficient total oxygen leaves both inventories unchanged.
9. Heat-source summation and thermal step-integration overflow are rejected before model state mutation.
10. Mission preflight refuses a segment when current or projected bulk-circuit pressure exceeds the synthetic ceiling.

These tests are intended to prove arithmetic/accounting invariants. They do not validate suitability of chosen rates/capacities or establish human safety.

## Validation strategy informed by external test evidence

NASA's integrated PLSS testbed work evaluates subsystem behavior under varying metabolic rate, suit pressure, ventilation flow, transient/contingency operation, and system interactions. Published PLSS breadboard testing also reports boundary anomalies involving CO2/water-vapor saturation, degraded vacuum quality, fan control, and thermal valve response. The implication for this software model is concrete: a scalar removal-rate fit is not enough. Future calibration must preserve coupled operating conditions and include adverse boundary points.

Recommended staged acceptance:
1. **Software invariants:** exact-head Rust tests, property tests for mass/resource bounds, state-unchanged assertions on rejected numerical steps, and deterministic fault campaigns. These establish accounting semantics only.
2. **Component maps:** characterize scrubber/collector removal rates across inlet CO2 and water loading, flow, temperature, pressure, and remaining capacity; preserve the source data, instrument calibration, uncertainty, and raw runs.
3. **Integrated loop:** sweep metabolic gas/water load, ventilation flow, circuit pressure/temperature, thermal rejection, and transient faults together. Test fan degradation, sorbent/collector saturation, valve or flow-path anomalies, and recovery behavior.
4. **Inspired-CO2 validation:** use a validated suit/helmet flow and breathing model plus the standardized human-in-the-loop CO2 washout method; do not substitute the bulk-circuit estimate for PICO2.
5. **Independent review:** require an independent verifier to consume versioned raw measurements and model parameters, recreate the fit and holdout errors, and issue a receipt tied to exact source/data hashes before any quantitative performance claim.

External research: [NASA PLSS 1.0 development/testing report](https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20120007410.pdf), [NASA PLSS 1.0 breadboard development and results](https://ntrs.nasa.gov/citations/20120004300), [active NASA-STD-3001 Volume 2 Revision F](https://standards.nasa.gov/node/237), and [NASA's standardized PICO2 measurement method (2026)](https://ntrs.nasa.gov/citations/20260001545).

## Required next steps before quantitative use

1. Calibrate the assumed CO2 gas-reference convention and ideal-gas reduction against an instrumented atmosphere loop; include pressure/temperature bounds and gas-composition measurements.
2. Replace the well-mixed bulk estimate with coupled loop/helmet flow, breathing-phase and dead-space models, and validate inspired CO2 using NASA's standardized in-suit CO2 washout measurement procedure.
3. Model the scrubber with empirically supported removal behavior as a function of flow, concentration, humidity, temperature, consumable loading, and bypass/leak paths; represent breakthrough/capacity with data.
4. Model water vapor using mass balance plus temperature-dependent saturation and condensation; track condensate collection, possible re-evaporation, and collector capacity.
5. Couple gas/thermal state without conflating the human thermal model and PLSS thermal store.
6. Add sensor evidence states and age/confidence semantics for gas composition, flow, temperature, pressure, cartridge loading, condensate, and leaks.
7. Run property and fault campaigns, record exact source identity and deterministic artifacts, and calibrate against an instrumented gas-loop/thermal bench.
8. Only after bench and independent engineering review should human-in-the-loop or program qualification be considered.

## External references

NASA-STD-3001 Volume 2 Revision F requires suits to monitor/record bulk suit atmospheric variables including ppCO2, and separately establishes inspired CO2 partial-pressure requirements. It explains that inspired CO2 depends on suit design, ventilation, helmet geometry, and wearer metabolic rates, and points to standardized human-in-the-loop testing. NASA's 2026 updated measurement method is therefore relevant to validation; it does not make the current bulk-circuit model a compliance demonstration.

- [NASA-STD-3001 Volume 2, §11.3 Spacesuit Atmosphere](https://www.nasa.gov/reference/11-0-spacesuits-vol-2/)
- [NASA 2026 Standardized Measurement Method for CO2 Partial Pressure Within Pressure Suits (NASA/TM-20260001545, published March 1, 2026)](https://ntrs.nasa.gov/citations/20260001545)
- [NASA Life Support Subsystems](https://www.nasa.gov/reference/jsc-life-support-subsystems/)
- [NASA development of pressure-swing adsorption for CO2 and humidity removal](https://ntrs.nasa.gov/citations/20060020747)

No claim is made that this module is flight-ready, medically valid, human-rated, or equivalent to EMU, xEMU, or AxEMU.
