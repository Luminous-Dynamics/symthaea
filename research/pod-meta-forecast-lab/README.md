# POD-Meta Forecast Calibration Lab

A standalone Rust microbenchmark for a missing capability in the POD-Meta Efficiency Lab: calibration of the operation-success forecasts that drive value-of-computation decisions.

This is a deterministic synthetic experiment, not a production forecaster, external benchmark score, measured FLOP/energy result, or state-of-the-art claim.

## Run

~~~sh
cargo fmt --manifest-path research/pod-meta-forecast-lab/Cargo.toml -- --check
cargo test --locked --manifest-path research/pod-meta-forecast-lab/Cargo.toml
cargo clippy --locked --manifest-path research/pod-meta-forecast-lab/Cargo.toml --all-targets -- -D warnings
cargo run --locked --manifest-path research/pod-meta-forecast-lab/Cargo.toml --bin pod-meta-forecast
~~~

The output is tab-separated and includes three record families:

- FORECAST: Brier score, log loss, ten-bin expected calibration error, mean forecast and observed rate for all test cases, seen regimes, and a held-out regime.
- RELIABILITY: counts, mean predicted probability, and empirical success rate in each forecast bin.
- SELECTION: fixed-order, random, global, family, context-family, and evaluator-only oracle selection across five regimes under identical episode outcomes and a hard per-episode budget.

## Forecast arms

- constant_half: no learned base-rate estimate.
- global_bayes: one smoothed rate from training observations.
- family_bayes: operation-family rates shrunk toward the global rate.
- context_family_shrinkage: context-by-operation rates shrunk toward the corresponding family rate.

The training split uses four evidence regimes. The fifth regime, held_out_shift, is absent from fitting and exists to expose dependence on context coverage. The test split uses a disjoint seed and includes all five regimes. No evaluation labels are used to fit or update the forecaster.

## Selection arms

Selection uses one logical proof obligation with four alternative operation families. Each operation has a fixed synthetic work-unit cost. A policy ranks candidates by predicted probability of verified progress per work unit, attempts affordable actions under the hard budget, and stops only after an independently generated simulator success. Fixed-order and budget-matched-random are controls. The oracle upper bound sees the simulator's probability table and is diagnostic only.

All policies are scored on the same episode outcome arrays. The selection metrics preserve success rate, total work, progress per work unit, work per verified proof, maximum per-episode work, and budget violations separately. No weighted composite is computed.

## Claim boundary

The simulator probabilities are synthetic and stationary within each named regime; the held-out regime is a deliberately shifted generator row. It does not measure calibration on real Symthaea operations, live model/tool outcomes, FLOPs, GPU time, energy, money, or end-to-end task completion.

A held-out context is not equivalent to a held-out task family or a protected evaluator. The next external step is a preregistered adapter to real operation telemetry and an independently administered evaluator, with task-family splits, temporal splits, and distribution-shift evaluation.

See PROTOCOL_V1.md for scoring definitions, controls, and external comparison rules.
