# MEL temporal-grounding qualification harness

This nested Rust package imports the exact production temporal-grounding scorer and SHA-256 digest module by path. It stays outside the main workspace so focused qualification does not inherit unrelated legacy workspace build or lint debt.

Run the exact-source checks with:

    cargo +1.96.0 test --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml
    cargo +1.96.0 clippy --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --all-targets -- -D warnings

## Temporal-grounding transformation equivariance

The temporal_transform module accepts a strict case containing canonical ground truth, a frozen scorer plan, baseline and transformed-view predictions, and a piecewise monotonic frame map from the view timeline back to the canonical source timeline. It verifies contiguous view coverage, monotonic canonical coverage, source/view artifact clock identities, and the opaque upstream task-qualification receipt digest syntax. It maps half-open prediction intervals back to canonical source frames using integer rational arithmetic, flooring starts and ceiling ends so boundaries are never silently shrunk, then scores both predictions through the production temporal-grounding scorer.

Run the synthetic two-times time-stretch control with:

    cargo +1.96.0 run --locked --manifest-path tools/mel-temporal-grounding-qualifier/Cargo.toml --bin mel-temporal-transform-audit -- tools/mel-temporal-grounding-qualifier/fixtures/mel-temporal-equivariance-2x.synthetic.json

This is only a coordinate-mapping evaluator. It does not transform or hash audio bytes, authenticate the upstream qualification receipt, or establish perceptual equivalence. Those responsibilities remain with the transform materialization and task-specific qualification owners (#13260).
