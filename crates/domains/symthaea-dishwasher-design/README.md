# symthaea-dishwasher-design

First-principles engineering primitives for Luminous InstantClean research.

## Scope

- Water and effective-solid sensible-heat demand.
- Heater electricity and ideal warm-up time.
- Per-cycle heater, pump, controls and drying energy accounting.
- Fresh water kept distinct from recirculated process-water volume.
- Single-orifice nozzle velocity, Reynolds number, dynamic pressure and ideal pump-power estimates.
- Pareto-front extraction over residual soil, energy, water and duration.

The crate intentionally does not model cleaning chemistry, spray atomization, splash-mediated soil removal, surface damage, microbiology, sanitization or multiphase CFD. Those need calibrated experiments and independent verification.

## Run

~~~sh
cargo test -p symthaea-dishwasher-design
~~~

The workspace glob includes `crates/domains/*`; no root manifest edit is needed.

## Model boundaries

- Water density is approximated as 1 kg/L in the thermal model.
- Solid mass is an effective heat-coupled mass to be calibrated, not automatically the whole load.
- Heater efficiency and recovered-heat fraction are explicit assumptions.
- Pipe/cabinet thermal losses, evaporation, pump curves, filter losses, cavitation, nozzle coefficients, spray breakup and cleaning kinetics are not inferred.
- The nozzle model uses measured flow and pressure drop as inputs; it is not a nozzle-design solver.
- Pareto comparisons use measured or proposed values and cannot establish cleanliness or safety.
