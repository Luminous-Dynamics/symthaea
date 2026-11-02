# symthaea-dishwasher-design

First-principles and calibration-bounded engineering primitives for Luminous InstantClean research.

## Modules

- `energy`: ideal sensible heating, heater electricity and heat-up-time estimates.
- `cycle`: heater, pump, controls and drying energy accounting. Fresh-water use is distinct from recirculated process-water charge.
- `hydraulics`: single-orifice bulk-flow estimates from measured flow and pressure drop.
- `hydraulic_network`: steady parallel-branch network solver using a measured quadratic pump curve and equivalent branch resistances.
- `kinetics`: bounded empirical first-order soil-removal response with temperature, detergent-dose and shear covariates.
- `pareto`: non-dominated candidate extraction over residual-soil fraction, energy, fresh water and duration.

## Run

From the Symthaea workspace:

~~~sh
cargo test -p symthaea-dishwasher-design
cargo clippy -p symthaea-dishwasher-design --all-targets -- -D warnings
~~~

The workspace already includes `crates/domains/*`; no root workspace manifest change is needed.

## Required calibration and model boundaries

### Thermal and energy models

- Water density is approximated as 1 kg/L.
- Solid mass is an effective heat-coupled mass to be calibrated, not automatically the entire load.
- Heater efficiency and recovered-heat fraction are explicit assumptions.
- Cabinet/pipe thermal losses, evaporation and pump curve losses are not inferred unless measured inputs account for them.

### Hydraulic models

- The nozzle model uses measured flow, pressure drop, geometry, fluid properties and pump efficiency.
- The network model assumes steady incompressible flow, one common pressure at parallel branches and quadratic pump/system curves.
- Each branch resistance must be measured or derived from a separately verified component model.
- Rotating spray arms, transient pump acceleration, leakage, air entrainment, cavitation, water accumulation in the cabinet, nozzle discharge coefficients and spray breakup are outside this model.
- These modules are not CFD and do not predict cleaning or coverage by themselves.

### Soil kinetics

- Each calibration is specific to the tested soil/surface/detergent family.
- Parameters must be fitted to repeated physical experiments with held-out validation.
- The Arrhenius/dose/shear functional form is an empirical response approximation, not a universal chemical mechanism.
- The implementation rejects conditions outside the recorded calibration envelope; do not widen the envelope merely to make an optimizer return a result.
- Distinct carbohydrate, protein, oil, starch, tea/coffee and mixed soils require separate models unless data demonstrate a justified shared model.
- This is not a microbiological or sanitization model.

### Evidence boundaries

No model output establishes cleaning effectiveness, drying quality, hygiene, microbiological reduction, material safety or regulatory compliance. Report predictions separately from measured outcomes and retain raw observations, instrument calibration identifiers, protocol version, geometry digest, solver version and replicate identity.

See `experiments/measurement-template.csv` for a minimal physical-run data contract.
