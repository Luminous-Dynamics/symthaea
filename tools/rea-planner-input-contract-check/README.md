# REA planner input contract checker v0.1

Standalone Rust checker for the planner input contract and its synthetic known-answer example. It validates Draft 2020-12 and cross-field completeness for priors, Evidence references, candidate likelihoods, robust scenarios and terminal decision utilities.

It deliberately does not call the existing planners. The checked-in example uses Synthetic Evidence and is qualified for pure mathematical known-answer work only. The current EIG/EVSI planners reject synthetic-only hypothesis support. Do not relabel these records as independent observations.

Run:

```bash
cargo test --manifest-path tools/rea-planner-input-contract-check/Cargo.toml --locked
cargo clippy --manifest-path tools/rea-planner-input-contract-check/Cargo.toml --all-targets --locked -- -D warnings
```

The checker enforces probability normalization, unique identities, full outcome/hypothesis matrix coverage, complete scenario experiment coverage, complete decision × hypothesis utility matrices, Evidence reference resolution, and separation between math-only synthetic and observed-planner modes.

It does not authenticate Evidence, establish calibration, choose experiments, or measure planner performance.
