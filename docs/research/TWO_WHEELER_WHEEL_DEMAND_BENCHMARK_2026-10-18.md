# TWO-WHEELER-WHEEL-DEMAND-001 — A reusable wheel-load reference kernel

**Status:** Synthetic reference implementation and verification contract, v0.1  
**Parent:** [TWO-WHEELER-ENG-001](https://github.com/Luminous-Dynamics/symthaea/issues/8044)  
**Powertrain frontier:** [POWERTRAIN-ENG-001](https://github.com/Luminous-Dynamics/symthaea/issues/8056)  
**Review date:** 2026-10-18  
**Evidence status:** Model equations and synthetic oracle fixtures; not validated against physical vehicle data.

## Why this slice

Do not duplicate the recent full-powertrain digital twins published by the Stellenbosch/Roam research team. The more useful first Rust contribution is a **small, inspectable wheel-demand kernel** that can act as an independent equation-level cross-check and later compose with motor/inverter maps, battery models, combustion maps and the energy ledger.

A validated model for the Roam Air electric motorcycle was published in February 2026. It used dynamometer-derived motor/inverter efficiency maps and six Nairobi trips; reported energy-per-distance errors ranged from 0.4% to 11.3%. The authors identified gradient estimation as an important source of bias, and showed that payload, posture, surface and operating point matter: https://doi.org/10.3390/wevj1702090.

A subsequent full-powertrain digital twin was published on 20 August 2026 and validated against six Stellenbosch drive cycles. It reports battery-power correlations of 0.905–0.981, motor-speed correlations of 0.912–0.996, and energy prediction within 2.71% on five of six trips. The sixth is a short downhill/high-transient case reported at 10.65%. Its declared scope excludes mechanical braking, auxiliary loads, battery temperature/ageing, and adaptation to other powertrain configurations; wind was estimated rather than measured: https://doi.org/10.3390/wevj1708432.

These are important existing results. We should not claim novelty by reimplementing a full twin without its raw data, component maps, or a validation plan. The potential contribution here is a small Rust reference contract with explicit signs/units, negative controls, reproducible force decomposition, and a path to stronger out-of-sample benchmarks.

## Model contract

Inputs are one point in a declared forward-travel reference frame:

- `mass_kg`: total translating vehicle + rider + payload mass, strictly positive.
- `rotating_equivalent_mass_kg`: non-negative added equivalent mass used only in the inertial term.
- `speed_mps`: non-negative forward ground speed. Reverse motion is outside v0.1.
- `acceleration_mps2`: signed acceleration along the forward axis.
- `grade_rad`: signed road angle in radians; positive uphill; strictly between `-π/2` and `+π/2`.
- `rolling_resistance_coefficient`: non-negative coefficient.
- `air_density_kg_m3`: positive air density.
- `drag_area_m2`: non-negative drag area (C_dA).
- `wind_speed_mps`: signed along-track wind speed; positive means tailwind.

The reference equations are:

```text
F_inertial = (mass + rotating_equivalent_mass) * acceleration
F_rolling  = Crr * mass * g * cos(grade) * direction
F_grade    = mass * g * sin(grade)
u_air      = ground_speed - along_track_wind
F_aero     = 0.5 * air_density * CdA * u_air * abs(u_air)
F_wheel    = F_inertial + F_rolling + F_grade + F_aero
P_wheel    = F_wheel * ground_speed
P_traction = max(P_wheel, 0)
P_overrun  = max(-P_wheel, 0)
```

where (g=9.80665\,\mathrm{m/s^2}). At standstill, rolling direction is set to the forward launch direction if forward acceleration is positive, otherwise zero; wheel power remains zero at exactly zero ground speed. A tailwind faster than ground speed may produce a negative aerodynamic force. All intermediate and output quantities are checked for finite values.

## Sign and claim discipline

- `F_wheel > 0` means positive wheel traction force is demanded.
- `P_wheel < 0` is a mechanical overrun/braking opportunity in this single-point screening model. It is **not** measured braking performance and is **not** credited as battery-regenerated energy.
- Regenerative recovery depends on battery charge acceptance, SoC, temperature, power electronics, motor/control limits, tire grip, and braking policy. It requires a separate qualified subsystem model.
- This is a pointwise force/power calculation, not a drive-cycle simulator. It does not integrate energy over time, predict battery power, or establish motor/fuel conversion efficiency.
- It does not model suspension load transfer, tire slip, transient tire dynamics, drivetrain backlash, braking, rider control, aerodynamic yaw, thermal behaviour, component degradation or real road surfaces.
- All fixture values are synthetic and are not measured parameters for the Roam Air or any other physical vehicle.

## Files and execution

- Synthetic reference fixture: [`two_wheeler_wheel_demand_v0_1.json`](data/two_wheeler_wheel_demand_v0_1.json)
- Published validation-summary transcription (not raw traces): [`published_sa_electric_motorcycle_validation_summary_v0_1.json`](data/published_sa_electric_motorcycle_validation_summary_v0_1.json)
- Rust source: [`xtask/src/two_wheeler_wheel_demand.rs`](../../xtask/src/two_wheeler_wheel_demand.rs)
- CLI dispatch: [`xtask/src/main.rs`](../../xtask/src/main.rs)

On a Rust-capable checkout:

```sh
cargo test -p xtask two_wheeler_wheel_demand
cargo run -p xtask -- two-wheeler-wheel-demand \
  --input docs/research/data/two_wheeler_wheel_demand_v0_1.json \
  --out reports/two_wheeler_wheel_demand_report.json
```

The fixture contains eight computed positive controls and four invalid-input controls. It exercises steady cruise, grade, acceleration, headwind/tailwind, downhill negative power, standstill power, zero demand and fail-closed validation for invalid mass, reverse speed, a vertical/out-of-domain grade and negative drag area.

The Rust source and command are authored but have not been compiled/executed in the current environment. A source-side formula replay can check fixture arithmetic only; it is not Rust test evidence or physical validation.

## Benchmarks and evidence ladder

### Gate W0 — Arithmetic contract

- 100% of frozen synthetic outputs fall within per-case tolerances.
- 100% of invalid-input controls are rejected with their expected error class.
- Stable input fixture digest and deterministic report.
- Positive and negative controls are both required.
- Changing expected values, signs, or fixture revision should cause visible verification failure; the verifier must not rewrite expected outputs.

### Gate W1 — Independent equation implementation

Implement an independently reviewed oracle (not a copy/paste of the production function), compare it on deterministic generated inputs and boundary tests, and report maximum absolute/relative force and power differences. Target exact agreement within the declared floating-point tolerance; this proves equation implementation agreement only.

### Gate W2 — Measured coast-down / dynamometer inputs

Use documented test setup, calibration, environmental measurements, mass/payload, tire pressure/type, gear ratio, road/roller force or coast-down traces, grade source, and signed wind. Fit parameters on training runs only and freeze separate validation runs. Unmeasured properties remain unknown or carry an explicit prior/range.

### Gate W3 — Held-out route validation

Adapt the published SA benchmark with author-provided or openly reusable input traces and component maps, if available. The published paper's summary table is not a substitute for raw telemetry. Use complete routes held out from calibration; also run a leave-one-route-out view and keep route metadata. Report per-route:
- Wh/km absolute error and relative error;
- integrated energy error and temporal battery-power error once the powertrain layer exists;
- force/power residuals vs speed, grade, acceleration, payload, wind and surface;
- uncertainty intervals and model-domain violations;
- all outlier routes, including short downhill or near-zero-net-energy trips.

**Do not rank by mean percentage error alone.** Short downhill trips can have a small denominator and high percentage error. Report absolute Wh/km error and route total Wh error alongside relative error; never silently exclude the outlier.

After sufficient independently held-out data exist, a stretch target is median absolute Wh/km error no greater than 3%, P90 no greater than 5%, and explicit review of every route above 10% relative error. These are proposed targets, not existing results or a claim that the current kernel can reach them.

### Gate W4 — Full-powertrain composition

Only after wheel-demand validity is established, connect battery ECM, inverter/motor efficiency maps, controller, drivetrain, regenerative braking, thermal derating and auxiliaries as separate modules with exact configurations. Validate them independently, then end-to-end. Treat ICE fuel-flow/engine maps and plasma-ignition parasitics as other architecture adapters; the wheel-demand kernel itself does not assume any specific prime mover.

### Gate W5 — Design optimization

Search gearing, wheel radius, mass, (C_dA), rolling resistance, powertrain sizing, thermal constraints and battery configuration over declared bounds. Report Pareto alternatives and sensitivity; do not optimize safety away. Candidate rankings must be robust to measurement uncertainty and regional duty-cycle variation.

## Suggested next steps

1. Compile/test `xtask` at this exact branch head and resolve any code/test mismatch.
2. Add mutation tests for reversed sign, omitted inertia, invalid/negative density, duplicated IDs, malformed fixture and output tampering.
3. Define a time-series trace schema with timestamps, grade source/uncertainty, wind direction/speed, payload, throttle/brake stage, battery/motor measurements, and missingness.
4. Establish data-use permission and retrieve raw, independently measured South African validation data before claiming reproduction.
5. Integrate the wheel-demand output with the existing powertrain energy-ledger tool, then add separately verified motor/inverter and battery models.

## References

- Adams, Botha & Booysen, *A Validated Physics-Based Powertrain Model for an Electric Motorcycle in Sub-Saharan Africa*, World Electric Vehicle Journal 17(2), 90 (10 February 2026): https://doi.org/10.3390/wevj1702090.
- Adams, Botha & Booysen, *A Validated Full-Powertrain Digital Twin of an Electric Motorcycle Developed for Sub-Saharan African Conditions*, World Electric Vehicle Journal 17(8), 432 (20 August 2026): https://doi.org/10.3390/wevj1708432.
- UNECE World Motorcycle Test Cycle background: https://unece.org/transport/standards/transport/vehicle-regulations-wp29/global-technical-regulations-gtrs.

**Claim ceiling:** This kernel establishes only the output of the declared simple equations under its stated assumptions. Passing its synthetic fixture does not establish real-road accuracy, battery energy use, range, efficiency, durability, roadworthiness, or any safety approval.
