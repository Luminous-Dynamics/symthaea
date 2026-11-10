# symthaea-electric-aircraft

A small, inspectable Rust engineering kernel for reduced-order fixed-wing electric-aircraft mission-energy feasibility.

## First increment

The v0.3 crate estimates, for a supplied sequence of steady-flight mission segments:

- all-up mass and MTOW compliance;
- aerodynamic drag from a declared lift-to-drag ratio;
- climb power (without assuming descent regeneration);
- motor, inverter, and propeller conversion losses;
- battery terminal power, current, voltage sag, and I²R heat;
- a lumped first-order battery thermal response;
- state-of-charge depletion, state-of-health-adjusted capacity, and protected reserve;
- an explicit user-configured uncertainty margin;
- evidence-bound propulsion efficiency curves with conservative uncertainty bounds and no extrapolation.



The dedicated uncertainty module adds deterministic adverse-corner screening for both scalar and fully characterized missions. The characterized path takes propulsion-efficiency and battery OCV/resistance uncertainty from their evidence-bound maps and rejects duplicated scalar perturbations to those same fields. It preserves the nominal result, budget, exact adjusted inputs and adverse-corner outcome. It is not Monte Carlo, a confidence interval, a probabilistic worst-case proof or a certification method.

The standalone thermal_network module provides an implicit multi-node RC network for coupled battery, inverter, motor, enclosure and coolant lumped nodes; heat inputs and conductance parameters must be characterized independently. The mission_thermal module now replays mission-derived battery/motor/inverter heat through that network and checks temperature-dependent component power derating on every substep. The overlay also rechecks the source report's battery electrical identities, energy/SOC transitions, reserve/current/power/temperature limits and aggregate totals, and preserves upstream characterization records. The upstream evidence-digest flag is carried as a source-report claim, not re-verified by the overlay. This is a thermal feasibility overlay, not yet a fully iterated electro-thermal solver: updated temperatures do not feed back into OCV/resistance or propulsion maps. See docs/THERMAL_NETWORK.md and docs/MISSION_THERMAL_DERATING.md.

Battery OCV and internal-resistance maps are two-dimensional over SOC and pack temperature. Both map families reject extrapolation and record the dataset IDs used. Evidence SHA-256 fields are syntax-checked at construction. The verified-map API can hash caller-supplied bytes and reject missing or mismatched payloads; URI resolution, signatures, source authority and configuration approval remain external. See docs/ROBUST_UNCERTAINTY.md.

It is Rust and uses fail-closed errors for invalid inputs, MTOW violations, battery power/current/voltage/temperature limits, and reserve infringement.

## Build and test

From the Symthaea workspace root:

```sh
cargo fmt --manifest-path crates/domains/symthaea-electric-aircraft/Cargo.toml -- --check
cargo test --locked -p symthaea-electric-aircraft
cargo clippy --locked -p symthaea-electric-aircraft --all-targets -- -D warnings
```

The crate is included automatically by the Symthaea workspace's crates/domains/* member glob. Its only direct dependency is the workspace-provided sha2 crate for SHA-256 evidence-payload verification.

## Evidence-bound characterization (v0.2/v0.3)

`evaluate_mission_characterized` accepts three `EfficiencyCurve` datasets: motor efficiency versus shaft-load fraction, inverter efficiency versus input-load fraction, and propeller efficiency versus advance ratio. Each curve identifies its independent-axis units, source URI, evidence class, dataset ID and SHA-256 digest. The evaluator interpolates only within the supplied data domain, subtracts the declared absolute uncertainty bound, and records the dataset IDs used. It rejects unsupported operating points. The digest is format-checked by ordinary characterization calls. Use evaluate_mission_fully_verified with a map of immutable caller-supplied artifact bytes to require all propulsion and battery evidence payloads and recompute every SHA-256 digest before evaluation. This still does not fetch URIs or authenticate signatures, publisher authority or configuration approval.

Mission reports retain complete characterization evidence records (dataset ID, URI, evidence class, description and SHA-256 digest), not only IDs. The evidence_content_digests_verified flag is true only for evaluate_mission_fully_verified after every caller-supplied payload matches its recorded digest. This attests content-digest checks only; it does not establish signature validity, publisher/authority trust, URI-to-bytes authenticity or configuration approval.

The scalar `evaluate_mission` path remains available for screening and regression continuity. `evaluate_mission_battery_characterized` uses the SOC/temperature pack maps with scalar propulsion efficiencies; `evaluate_mission_fully_characterized` uses both map families. OCV uncertainty is subtracted and resistance uncertainty is added before solving the equivalent pack circuit.
 
The pack maps are still reduced-order: they do not model cell-level electrochemistry, hysteresis, current-rate effects, SOH-dependent OCV/resistance, dynamic polarization or thermal runaway. See docs/BATTERY_CHARACTERIZATION.md.

## Important boundary

This is a research/mission-planning model, not an aircraft performance prediction, flight simulator, battery qualification tool, flight-control system, or certification artefact. It assumes steady segments and a supplied L/D ratio. It does not model takeoff transients, stall, compressibility, detailed propeller maps, voltage/SOC/temperature-dependent cell electrochemistry, thermal runaway, cell imbalance, battery fault propagation, aeroelasticity, structural loads, control laws, or regulatory reserve rules. A successful software test is not evidence of airworthiness.

See docs/ARCHITECTURE_TRADE_STUDY.md for the mission-first architecture decision framework, docs/MODEL_LIMITATIONS.md for promotion gates, and docs/BENCHMARK_PLAN.md plus docs/METRIC_BOUNDARIES.md for industry comparisons.
