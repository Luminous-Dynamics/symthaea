# Marine energy-budget benchmark (V0)

This dependency-free Rust crate implements a narrow accounting and comparison gate for [MARINE-ENERGY-BUDGET-001](https://github.com/Luminous-Dynamics/symthaea/issues/8092), under the broader [HYDRO-CO-DESIGN-001 program](https://github.com/Luminous-Dynamics/symthaea/issues/8082).

It answers: **do two candidate cases declare the same mission and evidence basis, and does the candidate's total electrical input-energy interval support a robust saving?**

## Run

```sh
cargo test --locked --offline --manifest-path crates/tools/marine-energy-budget-bench/Cargo.toml
```

No third-party crates are required.

## V0 boundary

All energy inputs are electrical energy measured at the named `energy_boundary_id`, in watt-hours (Wh). The crate does not mix electrical input, shaft energy, fuel chemical energy, or lifecycle energy. It does not model fluid flow, thrust, drag, batteries, or powertrain behavior.

Every case includes propulsion input energy and auxiliary input energy. Auxiliary energy includes relevant components such as air injection, control actuators, cooling, and other support loads. Missing auxiliary energy or its uncertainty makes the comparison `Indeterminate`, never zero.

The delivered-service fingerprint includes mission ID, condition digest, distance, optional payload, target speed, and arrival policy. V0 requires exact agreement on those declarations and on energy boundary, evidence origin, and uncertainty method before comparing two cases.

## Uncertainty semantics

V0 uses conservative absolute bounds, not statistical confidence intervals. Total uncertainty is the sum of declared propulsion and auxiliary absolute uncertainty bounds, without assuming independence or cancellation.

For the baseline total-energy band `[B_low, B_high]` and candidate band `[C_low, C_high]`, provided `B_low > 0`, the conservative relative-saving band is:

```text
[1 - C_high / B_low, 1 - C_low / B_high]
```

A result is:
- `RobustSavingWithinDeclaredBounds` only if the lower bound is strictly above zero;
- `RobustWorseWithinDeclaredBounds` only if the upper bound is strictly below zero;
- `Indeterminate` if the bound crosses or touches zero or the baseline lower bound is not positive;
- `Incomparable` if mission, condition, service, evidence origin, uncertainty method or energy boundary do not match;
- `Invalid` if supplied numeric inputs are invalid.

These names describe arithmetic under the supplied declarations. **They do not mean the physical model is validated or that real-world energy savings have been demonstrated.** Evidence references and hashes are identifiers, not proof of authenticity by themselves.

## Metrics

Each complete case returns:
- propulsion input Wh;
- auxiliary input Wh;
- total input Wh and a conservative total-energy band;
- Wh/km when route distance is positive;
- Wh/(tonne·km) only when payload is present and positive.

The benchmark compares equivalent declared service before interpreting energy numbers. A drag reduction or lower propulsion subcomponent does not establish a reduction in total electrical input energy.

## Test design

The unit tests use frozen synthetic inputs and hand-calculable expected results. They include auxiliary energy that erases a propulsion-only saving, uncertainty overlap, absent auxiliary data, mismatched missions/evidence bases, invalid inputs, and non-positive baseline lower bounds.

Passing these tests establishes arithmetic/contract consistency only. Physical calibration requires appropriately matched independent measurements, uncertainty analysis, and a stated validation domain.
