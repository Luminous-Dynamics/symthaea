# Symthaea Chaos Reference Suite

Standalone, standard-library-only Rust fixture generator for qualification of Symthaea's nonlinear-dynamics estimators.

It produces known-regime reference trajectories without embedding any candidate estimator. This separation is deliberate: the fixture generator must not decide whether a generated trace is chaotic.

## Included generators

- Stable linear contraction with analytic Lyapunov exponent.
- Logistic map: stable fixed point, period-2 cycle and fully chaotic parameter regime.
- Hénon map.
- Quasiperiodic torus negative control.
- Seeded AR(1) coloured-noise negative control.
- Lorenz-63 continuous flow integrated with RK4.

The machine-readable contract lives in fixtures.json. Regime labels are properties of the generators, not classifications made by Symthaea.

## Run

cargo test --manifest-path research/chaos-reference-suite/Cargo.toml
cargo run --manifest-path research/chaos-reference-suite/Cargo.toml --bin chaos-fixtures -- 2048 20261116 > chaos-fixtures.csv

The executable prints fixture metadata as comment lines, followed by CSV records. Sample count is capped at 1,000,000 per fixture.

A dedicated pull-request workflow runs formatting, strict Clippy, all tests and a small frozen CSV smoke generation for this standalone package.

## Qualification rules

- Keep fixture definitions, parameters, seeds, burn-in and sample count frozen per suite version.
- Do not tune estimator thresholds after observing held-out outputs.
- Treat AR(1) as a stochastic control, never as a chaos-positive case.
- Report inconclusive when the data length or estimator convergence is inadequate.
- Compare Lyapunov, attractor and dimension estimates against independent methods.
- Do not interpret any output as evidence of consciousness or cryptographic hardness.

The FNV digest is non-cryptographic and must never authenticate evidence or security decisions.

## Forecast-evaluator integration

This generator owns only known-truth trajectory construction. It does not train or rank a predictor. Its comment/header/CSV output is the versioned file contract consumed by the independent decision-predictability evaluator in [PR #12551](https://github.com/Luminous-Dynamics/symthaea/pull/12551). The file boundary keeps the reference generator and candidate baseline/evaluator packages independently buildable; there is no Cargo path dependency between the two draft PRs.

Once both packages are present in the same checkout:

```bash
cargo run --manifest-path research/chaos-reference-suite/Cargo.toml --bin chaos-fixtures -- 2048 20261116 > chaos-fixtures.csv
cargo run --manifest-path research/decision-predictability-benchmark/Cargo.toml --bin forecast-fixtures -- chaos-fixtures.csv > forecast-report.csv
```

The current evaluator protocol is a within-trajectory chronological continuation diagnostic (60% train / 10% validation / 30% test, recursive free-running from one fixed origin). It is not a held-out equation-family generalization result, and the fixture-generator's deterministic sample steps are not independent statistical replicates. Keep the generator's known regime labels evaluator-side; do not pass them to candidate code or use them as model predictions.

The evaluator smoke fixture tests the CSV adapter independently. It does not replace a full generator-to-evaluator execution after both PRs are integrated.
