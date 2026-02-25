# Testing

This project has a large test surface. Use these tiers to keep feedback fast and predictable.

## Quick checks

- Lib/unit tests only:
  - `RUSTC_WRAPPER= cargo test -p symthaea --lib`
- Single suite:
  - `RUSTC_WRAPPER= cargo test -p symthaea --test module_ablation`

## Full test suite

- Default (fast enough for local, but still large):
  - `RUSTC_WRAPPER= cargo test -p symthaea`

## Integration/e2e suites (feature-gated)

Some heavier end-to-end/integration test files are behind a feature to keep the
baseline run tight. Enable them explicitly:

- `RUSTC_WRAPPER= cargo test -p symthaea --features integration-tests`

## Psych-bench baselines

Run the regression check against the current baseline:

- `RUSTC_WRAPPER= cargo test -p symthaea-psych-bench --test full_battery regression_against_baseline -- --nocapture`

Update the baseline snapshot (current baseline is `crates/symthaea-psych-bench/baselines/v0.6.0.json`):

- `RUSTC_WRAPPER= cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --snapshot crates/symthaea-psych-bench/baselines/v0.6.0.json`

## Ignored/expensive tests

Some benchmarks and model-download tests are marked `#[ignore]`.
Run them explicitly when needed:

- `cargo test -p symthaea -- --ignored`
- For neural-bridge tests (large model downloads):
  - `cargo test -p symthaea --features neural-bridge -- --ignored`

## LLM backend tests

- Real API tests are ignored by default and require env vars:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`

Run with:

- `OPENAI_API_KEY=... ANTHROPIC_API_KEY=... cargo test -p symthaea --test llm_backend_integration -- --ignored`
