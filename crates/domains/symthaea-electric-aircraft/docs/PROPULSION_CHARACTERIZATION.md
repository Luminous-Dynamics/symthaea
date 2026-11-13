# Propulsion characterization contract

**Model increment:** electric-fixed-wing-energy-v0.2.0  
**Evidence class:** model-interface specification. Synthetic unit-test curves are not measurements.

## Why this exists

A single efficiency constant (for example, "motor efficiency = 94%") hides operating-point dependence and offers no defensible way to handle a point outside the measured region. v0.2 introduced evidence-bound, piecewise-linear efficiency curves and rejects extrapolation.

The evaluator uses:

\[
\eta_{\\mathrm{used}} = \eta_{\\mathrm{nominal}} - u_{\\mathrm{absolute}}
\]

where u_absolute is a supplied absolute uncertainty bound. This is a conservative screening convention; it is not a statistical confidence interval, a tolerance guarantee, or a certification method.

## Data contract

Each curve records:

- stable curve ID;
- independent-variable name and unit;
- ordered calibration points;
- evidence dataset ID and URI;
- evidence class: physical measurement, manufacturer-reported, published research, or simulation;
- SHA-256 digest string;
- human-readable evidence description.

Each point is a tuple (x, efficiency, absolute_uncertainty). Independent-variable values must be strictly increasing. Efficiency must be finite and in (0, 1]. Absolute uncertainty must be finite, nonnegative, and smaller than nominal efficiency.

Ordinary characterization evaluation validates the digest's 64-character hexadecimal syntax. PropulsionCharacterization::verify_evidence_payloads recomputes SHA-256 over caller-supplied immutable bytes and rejects missing or mismatched payloads; evaluate_mission_fully_verified applies this check to both propulsion and battery datasets. The crate does **not** download the referenced URI, verify signatures, authenticate publishers/authority, or approve configuration applicability. Those responsibilities remain in the surrounding evidence pipeline.

## Current map meanings

| Curve | Independent variable | Rating denominator |
|---|---|---|
| Motor efficiency | Mechanical shaft-load fraction | Declared motor rated shaft power |
| Inverter efficiency | Inverter input-load fraction | Declared inverter rated input power |
| Propeller efficiency | Advance ratio J = V/(nD) | Declared fixed RPM and propeller diameter |

The propeller model uses fixed propeller RPM and diameter in this increment. Motor and inverter maps are one-dimensional projections. The curves must describe the same intended configuration and operating conditions as the mission evaluation.

## Evaluation order

1. Compute useful aircraft propulsive power from steady-segment aerodynamic drag and positive climb power.
2. Calculate propeller advance ratio from true airspeed, fixed RPM, and diameter.
3. Look up the conservative propeller efficiency; reject a point outside the curve.
4. Calculate required propeller shaft power and motor shaft-load fraction.
5. Look up the conservative motor efficiency; reject a point outside the curve.
6. Calculate inverter input power and inverter-load fraction.
7. Look up the conservative inverter efficiency; reject a point outside the curve.
8. Calculate battery power, battery current and the downstream pack energy/thermal response.
9. Record the exact evidence dataset IDs in the mission report.

The curve evaluator interpolates nominal value and absolute uncertainty linearly between adjacent calibration points. It does not extrapolate below the minimum or above the maximum calibration x-value.

## Measurement protocol for real maps

### Motor
Record calibrated shaft torque, shaft RPM, DC input voltage/current, temperatures, cooling conditions, and steady-state acceptance criteria. Derive mechanical output power as P=Tω. Sweep load and speed; record repeatability, hysteresis, stabilization time and sensor calibration. A one-dimensional load curve is only a projection and must not be used outside its stated speed/temperature/voltage regime.

### Inverter
Record DC input and phase output electrical measurements, switching/control mode, coolant/ambient temperature, bus voltage, current, power factor where applicable, and operating limits. Keep the measurement boundary explicit so motor and inverter losses are not double-counted.

### Propeller
Measure thrust, torque, RPM, airspeed, density/temperature, blade pitch, and geometry. Build a map across advance ratio and power coefficient, with uncertainty and blade/configuration identifiers. A curve against advance ratio alone is a screening projection, not a complete propeller performance map.

### Evidence handling
Freeze raw data before fitting or smoothing. Preserve instrument calibration certificates, test article serial numbers, software/firmware versions, test procedure revision, environmental conditions, raw traces, data-cleaning code, exclusions, and uncertainty budget. The map's evidence URI must resolve to immutable content whose digest is independently verified.

## Synthetic test fixtures

Unit tests use invented efficiency curves to verify:
- interpolation of nominal efficiency and absolute uncertainty;
- strict rejection of extrapolation;
- invalid or duplicate independent-axis points;
- digest syntax checks and exact caller-supplied byte verification;
- use of conservative lower bounds in the mission estimator;
- dataset IDs recorded in the report;
- characterized mission rejection when an operating point is outside the measured domain.

These tests validate software behaviour only. They do not qualify any motor, inverter, propeller, aircraft, test rig, or dataset.

## Next fidelity upgrade

The next model increment should replace these one-dimensional projections with bounded multi-dimensional maps:

- motor efficiency f(RPM, torque) plus voltage and temperature validity conditions;
- inverter efficiency f(V_DC, I_DC) with switching mode and thermal state;
- propeller coefficient surfaces over advance ratio and power coefficient, with Reynolds/Mach/blade configuration boundaries;
- battery OCV and internal-resistance surfaces over SOC, temperature, SOH and charge/discharge direction.

The model should never fill missing regions with guessed extrapolation. It should return Incomplete / out-of-domain until a validated map or a justified conservative envelope is supplied.
