# Thermal interval envelope benchmark v1

**Status:** contract and synthetic math-oracle suite. No empirical thermal or food-safety qualification is claimed.

## Paradigm shift

A measurement is not a point when its uncertainty is explicitly known. Do not throw away lower/nominal/upper readings by converting the series into the point-only `ThermalTrajectory` and then reporting one dose value. Instead preserve three ordered tracks throughout the conditional calculation and publish the evidence/assumptions alongside the outputs.

The new `evaluate_thermal_interval_envelope_v1` function:
- binds to the existing process trace and validated observation log;
- requires an explicit sensor and measurement kind;
- rejects unknown selected readings, invalid UTC timestamps, duplicate sample times, non-positive/non-finite D/z parameters, and series that change placement description;
- calculates the exact piecewise-linear D/z integral independently for lower, nominal and upper temperature tracks;
- reports peak-temperature bounds, selected process-step sequence, elapsed time, model scope, and qualification blockers;
- emits no safe/unsafe, pasteurization or doneness verdict.

## Mathematical oracle

For a segment with linear temperature between (T_0) and (T_1), the assumed lethality rate is (r(T)=10^{(T-T_{ref})/z}/D_{ref}). The exact segment integral can be expressed as the logarithmic mean of the endpoint rates multiplied by elapsed time:

`segment = dt * (r1 - r0) / (ln(r1) - ln(r0))`

For equal endpoint rates, the limiting value is `dt * r0`. Production uses the equivalent `expm1(x) / x` form; the integration test uses the endpoint-rate logarithmic-mean form as an independent oracle. Both only verify the chosen mathematical model, not that D/z kinetics fit a real organism or food matrix.

## Synthetic regression gates

1. Lower, nominal and upper outputs agree with the independent oracle within absolute (10^{-9}) log10 units for the frozen synthetic ramp.
2. Peak and integral bounds are ordered: lower ≤ nominal ≤ upper.
3. Permuting observation array order does not change the report.
4. Widening a temperature interval cannot narrow the corresponding monotonic integral envelope.
5. Shifting the whole curve by one z-value scales this model's conditional integral by ten.
6. Unknown selected readings, invalid dates/parameters, absent selectors, synthetic-evidence promotion, and a changed sensor location fail closed or produce explicit qualification blockers.
7. No public result field encodes safety, doneness, pasteurization, or a recommendation.
8. The exact-head Rust tests, formatting and Clippy must execute before qualification; skipped/queued/absent CI is not pass evidence.

## Empirical qualification plan

Before making predictions about real cooking:
- collect repeated, timestamped temperature series on preregistered food geometries and cooking methods;
- compare each instrument against a traceable reference and retain calibration uncertainty;
- record core, surface, medium and ambient temperatures separately; use a new trajectory when a probe moves;
- freeze the kinetic model, organism, food matrix and parameter source before test-set evaluation;
- measure residuals and, only where bounds declare a statistical coverage level, report empirical coverage and interval width on held-out batches;
- stratify error, missing samples, dropout rate and repeatability by sensor type, placement, geometry and heating method;
- keep mathematical conformance, sensor calibration, thermal-model validity, sensory quality and food-safety assessment as separate gates.

## Qualification boundary

The fixture and D/z values are synthetic. A mathematically exact integral conditional on assumed parameters is not evidence that a real product reached a safe state. USDA FSIS explains that probe placement, product geometry and within-food temperature variation affect representativeness, and distinguishes sensory doneness from safety:
- https://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/food-safety-basics/food-thermometers

NIST's thermocouple calibration guidance documents uncertainty budgets and the need to account for instrument/calibration effects; actual kitchen instrumentation requires appropriate range- and method-specific evidence:
- https://www.nist.gov/pml/sensor-science/temperature-humidity/thermocouples-calibrations-services

**Execution status:** Rust tests have been authored but not run in this environment. Static/source review is not Rust compilation or exact-head CI evidence.
