# RESONANCE-001 analytical reference

This crate is the closed-form reference layer for passive lumped-element
RESONANCE-001 experiments.

Included:
- Ideal series RLC resonance, Q, half-power bandwidth and complex input impedance.
- Ideal parallel RLC resonance, Q, half-power bandwidth and complex input impedance.
- Two lossless magnetically coupled resonators, mutual inductance and lower /
  upper normal-mode frequencies.
- Two driven, lossy series-RLC loops with complex primary input impedance,
  secondary current/voltage transfer, winding-dot polarity and normalized
  secondary resistor dissipation.

All quantities use SI units. Frequency arguments are in hertz.

## Scope and claim ceiling

These equations assume linear, lumped, time-invariant components. They exclude
component tolerances, parasitic and distributed effects, coil geometry,
external loading, core loss, skin/proximity effects, radiation, enclosure
perturbations and measurement-system error.

The normal-mode model is an ideal lossless mutual-inductance reference. The
driven two-loop model adds positive series resistances and sinusoidal
steady-state response, but still assumes linear lumped elements and ideal
mutual inductance. Neither model includes coil geometry, parasitics, distributed
transmission-line effects, core loss, external loading beyond the declared loop
resistance, radiation or enclosure perturbations. Neither is a complete model
of a Tesla coil or a qualified model for arbitrary resonator geometry.

A matching result qualifies the formula implementation against its analytical
known answers. It does not validate ngspice, FDTD, a physical circuit, a
measurement instrument, or any claim of anomalous energy transfer.

## Run

    cargo test --manifest-path tools/resonance-001-reference/Cargo.toml

## Benchmark ladder

1. Exact analytical R0 cases in research/resonance-001/fixtures/analytic-v0.1.json, including driven coupled-loop transfer and loss.
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
