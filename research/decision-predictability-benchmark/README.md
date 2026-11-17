# Decision-Relevant Predictability Metrics

Dependency-free Rust metric contracts for the benchmark constitution in [#12546](https://github.com/Luminous-Dynamics/symthaea/issues/12546).

This first executable slice deliberately contains **no candidate forecasting model** and no hidden-world generator. It implements small evaluator-side mathematical contracts that can be tested independently:

- qualified Lyapunov-time normalization;
- observed forecast-horizon extraction with threshold and monotonic-time checks;
- information-set Bayes risk, policy risk, perfect-information lower bound and regret;
- expected value of information from expected posterior Bayes risk and acquisition cost;
- componentwise resource-vector Pareto dominance.

The decision evaluator takes conditional state probabilities and state/action losses. It computes the Bayes-optimal risk given each information set separately from the clairvoyant perfect-information risk. This prevents charging a candidate for irreducible uncertainty as though it were model error.

## Run

```bash
cargo fmt --manifest-path research/decision-predictability-benchmark/Cargo.toml -- --check
cargo clippy --manifest-path research/decision-predictability-benchmark/Cargo.toml --all-targets -- -D warnings
cargo test --manifest-path research/decision-predictability-benchmark/Cargo.toml --all-targets
```

The crate uses only the Rust standard library and is an isolated Cargo workspace.

## Scope and limits

These functions are metric/oracle primitives, not a complete benchmark, forecasting system, sensor planner or safety controller. The subsequent work is to add frozen paired-world generators, candidate/evaluator separation, observation permissions and deterministic known-answer fixtures. Lyapunov normalization is rejected unless the reference exponent is marked `Reliable`; a candidate's own unqualified estimate must not define its score.

Resource vectors are comparable only when their units, measurement boundaries, hardware and workload contract match. The crate does not measure physical energy or hardware performance.

No output from this crate establishes SOTA, physical reliability, consciousness or causal reconstruction.
