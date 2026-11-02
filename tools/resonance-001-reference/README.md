# RESONANCE-001 analytical reference

This crate is the closed-form reference layer for passive lumped-element
RESONANCE-001 experiments.

Included:
- Ideal series RLC resonance, Q, half-power bandwidth and complex input impedance.
- Ideal parallel RLC resonance, Q, half-power bandwidth and complex input impedance.
- Two lossless magnetically coupled resonators, mutual inductance and lower /
  upper normal-mode frequencies.

All quantities use SI units. Frequency arguments are in hertz.

## Scope and claim ceiling

These equations assume linear, lumped, time-invariant components. They exclude
component tolerances, parasitic and distributed effects, coil geometry,
external loading, core loss, skin/proximity effects, radiation, enclosure
perturbations and measurement-system error.

The coupled-mode model is an ideal lossless mutual-inductance reference. It is
not a model of arbitrary Tesla coils or a qualified model for every resonator
geometry. Use it as an analytical oracle only where its assumptions hold.

A matching result qualifies the formula implementation against its analytical
known answers. It does not validate ngspice, FDTD, a physical circuit, a
measurement instrument, or any claim of anomalous energy transfer.

## Run

    cargo test --manifest-path tools/resonance-001-reference/Cargo.toml

## Benchmark ladder

1. Exact analytical R0 cases in research/resonance-001/fixtures/analytic-v0.1.json.
2. Compare a pinned circuit solver only after its full input/output parser and
   convergence boundary are qualified.
3. Measure passive low-voltage fixtures using calibrated impedance / network
   analysis and record raw complex response, fixture compensation, component
   tolerances and uncertainty.
4. Hold out complete coil geometries and reassembly sessions.
5. Compare fixed logarithmic sweeps, adaptive refinement, random selection,
   and Symthaea expected-information-gain selection under equal measurement
   budgets.
6. Report frequency, Q, bandwidth, complex transfer and uncertainty
   separately. A win on one metric is not a win on all metrics.
