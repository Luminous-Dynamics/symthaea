# symthaea-electric-aircraft

A small, inspectable Rust engineering kernel for reduced-order fixed-wing electric-aircraft mission-energy feasibility.

## First increment

The v0.1 model estimates, for a supplied sequence of steady-flight mission segments:

- all-up mass and MTOW compliance;
- aerodynamic drag from a declared lift-to-drag ratio;
- climb power (without assuming descent regeneration);
- motor, inverter, and propeller conversion losses;
- battery terminal power, current, voltage sag, and I²R heat;
- a lumped first-order battery thermal response;
- state-of-charge depletion, state-of-health-adjusted capacity, and protected reserve;
- an explicit user-configured uncertainty margin.

It is dependency-free Rust and uses fail-closed errors for invalid inputs, MTOW violations, battery power/current/voltage/temperature limits, and reserve infringement.

## Build and test

From this crate directory:

\`\`\`sh
cargo test
\`\`\`

The crate is included automatically by the Symthaea workspace's crates/domains/* member glob. It intentionally has no third-party dependencies in this first increment.

## Characterized propulsion (v0.2)

`evaluate_mission_characterized` accepts three `EfficiencyCurve` datasets: motor efficiency versus shaft-load fraction, inverter efficiency versus input-load fraction, and propeller efficiency versus advance ratio. Each curve identifies its independent-axis units, source URI, evidence class, dataset ID and SHA-256 digest. The evaluator interpolates only within the supplied data domain, subtracts the declared absolute uncertainty bound, and records the dataset IDs used. It rejects unsupported operating points. The digest is format-checked but the current crate does not fetch artifacts or recompute their hashes; consumers must verify artifact bytes in the surrounding evidence pipeline.

The scalar `evaluate_mission` path remains available for screening and regression continuity. It is not equivalent to a calibrated map run.

## Important boundary

This is a research/mission-planning model, not an aircraft performance prediction, flight simulator, battery qualification tool, flight-control system, or certification artefact. It assumes steady segments and a supplied L/D ratio. It does not model takeoff transients, stall, compressibility, detailed propeller maps, voltage/SOC/temperature-dependent cell electrochemistry, thermal runaway, cell imbalance, battery fault propagation, aeroelasticity, structural loads, control laws, or regulatory reserve rules. A successful software test is not evidence of airworthiness.

See docs/MODEL_LIMITATIONS.md and docs/BENCHMARK_PLAN.md.
