# Measurement and history oracle

The standalone crate now exposes a second binary that makes two additional synthetic semantics inspectable outside the library unit tests.

## Run

From the repository root:

```sh
cargo run --locked --manifest-path tools/connectome-perturbation-oracle/Cargo.toml --bin measurement
```

The report is deterministic JSON and is governed by:
- `docs/neuroscience/CONNECTOME_MEASUREMENT_DYNAMICS_REPORT.schema.json`
- `tools/connectome-perturbation-oracle/fixtures/measurement-suite.expected.json`

## Pulse-order dependence

The same two interventions are delivered in opposite orders to a declared adaptive-threshold network. The model includes a bounded contribution from the previous target output. Under the frozen values:
- A then B yields [0, 1, 1, 0].
- B then A yields [0, 0, 1, 0].

This is a generic history-dependent test system. It is not a validated model of a particular neuron or a claim that the biological circuit has this transfer function.

## Calcium observation dynamics

A separate low-pass observation kernel maps latent neural activity x[t] to an observed fluorescence-like trace:

`F[t] = F0 + decay * (F[t-1] - F0) + gain * x[t]`

For latent [1, 0, 0], decay 0.5, gain 2 and baseline 0, the observed trace is [2, 1, 0.5]. The latent trace remains separately available and unchanged.

This is a measurement-model fixture, not a full calcium-indicator model. Real data adapters must pin the actual indicator, sampling, deconvolution, bleaching, noise and preprocessing model.

## Claim ceiling

Both cases are synthetic known answers. They validate separation of temporal response dynamics and observation filtering in the evaluator; they do not demonstrate biological fit, causal identification, or state-of-the-art performance.
