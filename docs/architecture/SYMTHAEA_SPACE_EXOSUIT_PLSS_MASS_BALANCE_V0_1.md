# Symthaea Space Exosuit: PLSS atmosphere mass-balance reference model v0.1

**Status:** simulation-only software model. Not a validated atmosphere/physiology model, medical monitor, or physical PLSS controller.

**Implementation branch:** `fix/space-exosuit-plss-mass-balance`

**Research track:** [SX-024 — PLSS atmosphere, CO2 and humidity model](https://github.com/Luminous-Dynamics/symthaea/issues/7459), following [SX-022](https://github.com/Luminous-Dynamics/symthaea/pull/1535).

## Problem addressed

The prior reference implementation used total ventilation flow times an efficiency fraction as the amount of CO2 removed. That quantity is not a justified CO2 capture rate: fan throughput is circulated gas, not CO2 captured. It could also force the calculated residual CO2 burden to zero independent of the CO2 production rate or sorbent exhaustion.

The prior humidity burden was unitless, while generation and removal were computed using unlike implied units. Those values should not be interpreted physically.

This patch replaces both shortcuts with explicitly dimensioned, finite-capacity *reference models*. It does not claim the coefficients are measured or representative of any qualified suit.

## CO2 model

- `co2_burden_l` is the unremoved generated CO2 quantity represented as liters of CO2-equivalent at a declared reference condition. It is a bookkeeping proxy, **not** suit CO2 partial pressure, ppm, physiological exposure, or time-to-harm.
- `co2_scrubber_removal_rate_l_min` is the configured maximum CO2-equivalent capture rate. It is separate from `ventilation_flow_l_min`.
- `co2_scrubber_capacity_l` is a finite total capacity. Remaining capacity is stateful and decremented only by modeled capture.
- The modeled capture per step is bounded by all three: CO2-equivalent burden available, rate × step duration (and modeled path availability), and remaining cartridge capacity.
- Scrubber degradation halves the synthetic reference rate; a failed scrubber captures zero. A degraded ventilation path reduces the declared scrubber rate; a failed ventilation path retains the existing mission-abort behavior.
- No scrubber regeneration, pressure, gas-volume mixing, leakage coupling, sorbent chemistry, breakthrough curve, or temperature/moisture cross-sensitivity is claimed in v0.1.
- The EVA mission planner uses a conservative no-assist workload to check whether the declared rate and finite capacity can cover an entire planned segment. A failed path, nonzero residual burden, inadequate rate, or insufficient capacity returns a `ReturnToSafeHaven` planning disposition before the next task segment begins.
- When a resource is exhausted within a step, the segment report escalates to return at the step boundary. This segment-level model does not determine the real partial pressure or a safe accumulated burden; its values must not be interpreted as human exposure limits.

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
- Negative oxygen/CO2 generation rates and negative metabolic input values are invalid.
- CO2 removal cannot exceed available modeled CO2; capture cannot exceed the declared rate or remaining capacity.
- Humidity removal cannot exceed the available modeled water vapor, declared rate, or remaining collector capacity.
- The module has no command surface for oxygen regulators, pressure valves, ventilation hardware, scrubbers, thermal control, motors, or propulsion.
- The existing deterministic assist/survival boundary remains independent of this simulation.

## Regression coverage added

The PLSS unit tests now include:

1. CO2 capture capped by finite sorbent capacity even at high fan flow.
2. CO2 capture rate remains independent of artificially increasing fan throughput.
3. A failed scrubber causes CO2 burden to accumulate under nominal ventilation.
4. Humidity removal is capped by finite condensate-storage capacity.
5. A failed humidity-control path causes water-vapor mass to accumulate.
6. Overflowing humidity input is rejected before oxygen inventory is consumed.

These tests are intended to prove arithmetic/accounting invariants. They do not validate suitability of chosen rates/capacities or establish human safety.

## Required next steps before quantitative use

1. Replace gas-equivalent volume with an explicit thermodynamic state: circuit free volume, gas temperature, total pressure, dry/wet basis, CO2 mole fraction/partial pressure, and declared reference conditions.
2. Model the scrubber with empirically supported removal efficiency as a function of flow, concentration, humidity, temperature, consumable loading, and bypass/leak paths; represent sorbent breakthrough or usable capacity using calibration data.
3. Model water vapor using mass balance plus temperature-dependent saturation and condensation; track condensate collection, possible re-evaporation, and collector capacity.
4. Couple gas/thermal state without conflating the human thermal model and PLSS thermal store.
5. Add sensor evidence states and age/confidence semantics for gas composition, flow, temperature, pressure, cartridge loading, condensate, and leaks.
6. Run property and fault campaigns, record exact source identity and deterministic artifacts, and calibrate against an instrumented gas-loop/thermal bench.
7. Only after bench and independent engineering review should human-in-the-loop or program qualification be considered.

## External references

NASA's life-support and spacesuit material describes oxygen/pressure, ventilation/CO2 removal, moisture, thermal control, and the link between human workload, O2 demand, CO2 production, and heat rejection. These references motivate the modeled subsystems; they do not calibrate the synthetic values above.

- [NASA Life Support Subsystems](https://www.nasa.gov/reference/jsc-life-support-subsystems/)
- [NASA Spacesuit Basics](https://www.nasa.gov/centers-and-facilities/johnson/spacewalk-spacesuit-basics/)
- [NASA-STD-3001 Volume 2](https://standards.nasa.gov/standard/NASA/NASA-STD-3001_VOL_2)

No claim is made that this module is flight-ready, medically valid, human-rated, or equivalent to EMU, xEMU, or AxEMU.
