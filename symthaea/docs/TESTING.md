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

## Coding Validation

Run the focused validation lane for Symthaea's Rust coding path:

- `./scripts/run_coding_validation.sh`

This checks that simulated execution cannot receive success credit, verifies the
real Rust executor path, and runs the Rust-adapted HumanEval smoke testbench.
The JSON report is written to `target/coding-validation/humaneval-smoke.json`
unless `SYMTHAEA_CODING_VALIDATION_REPORT` is set.

The same lane is available through:

- `./scripts/ci_check_lanes.sh coding-validation`

## Meta-study (WebResearcher)

Run the self-research pipeline (requires network access and an LLM backend):

- `RUSTC_WRAPPER= cargo run --example meta_study --features "web_research_module school_learning"`

Optional search providers (for real results):
- `SYMTHAEA_SEARCH_PROVIDER=serper SYMTHAEA_SEARCH_API_KEY=...`
- `SYMTHAEA_SEARCH_PROVIDER=brave SYMTHAEA_SEARCH_API_KEY=...`
- `SYMTHAEA_SEARCH_PROVIDER=searxng SYMTHAEA_SEARCH_ENDPOINT=https://your-searxng`

Research time controls:
- `SYMTHAEA_RESEARCH_BUDGET_SECS=300` (0 disables)
- `SYMTHAEA_RESEARCH_YIELD_TIMEOUT_SECS=60`
- `SYMTHAEA_RESEARCH_TARGET_CHARS=8000`
- `SYMTHAEA_RESEARCH_MAX_EXPANSIONS=1`
- `SYMTHAEA_RESEARCH_MIN_OBJECTIVES_PER_5K=4`
- `SYMTHAEA_RESEARCH_SIMILARITY_THRESHOLD=0.85`
- `SYMTHAEA_GLOBAL_SIMILARITY_THRESHOLD=0.9`
- `SYMTHAEA_POLYMATH_COLLISIONS=1` (0 disables)
- `SYMTHAEA_POLYMATH_MAX_ATTEMPTS=8`
- `SYMTHAEA_POLYMATH_MIN_SIMILARITY=0.25`
- `SYMTHAEA_POLYMATH_MAX_SIMILARITY=0.85`

## Curriculum research benchmark

End-to-end timing for research → ingest → recall:

- `RUSTC_WRAPPER= cargo run --example curriculum_research_bench --features "web_research_module school_learning"`

## Psych-bench baselines

Run the regression check against the current baseline:

- `RUSTC_WRAPPER= cargo test -p symthaea-psych-bench --test full_battery regression_against_baseline -- --nocapture`

Update the baseline snapshot (current baseline is `crates/symthaea-psych-bench/baselines/v0.6.0.json`):

- `RUSTC_WRAPPER= cargo run -p symthaea-psych-bench --example run_psych_benchmarks -- --snapshot crates/symthaea-psych-bench/baselines/v0.6.0.json`

## SSM sensor demo (INA219)

Run the diagonal SSM over power draw (simulated by default):

- `RUSTC_WRAPPER= cargo run -p symthaea-ssm --bin ssm_sensor_demo`

Enable real INA219 via I2C (requires feature + hardware):

- `RUSTC_WRAPPER= cargo run -p symthaea-ssm --features ina219-linux --bin ssm_sensor_demo`

Tuning (sim defaults shown):
- `SYMTHAEA_SSM_HZ=10`
- `SYMTHAEA_SSM_SECONDS=20`
- `SYMTHAEA_SIM_BASE_WATTS=6.0`
- `SYMTHAEA_SIM_AMPLITUDE_WATTS=0.4`
- `SYMTHAEA_SIM_PERIOD_S=6.0`

## Symthaea power SSM sensor

Enable the SSM power sensor inside Symthaea (simulated by default):

- `SYMTHAEA_POWER_SSM=1 RUSTC_WRAPPER= cargo run -p symthaea --features \"ssm-power\" --example meta_study`

Standalone power SSM demo (no web research, fast loop):

- `RUSTC_WRAPPER= cargo run -p symthaea --features \"ssm-power\" --example power_ssm_demo`

Enable real INA219 in Symthaea (requires hardware + linux I2C):

- `SYMTHAEA_POWER_SSM=1 SYMTHAEA_INA219_REAL=1 RUSTC_WRAPPER= cargo run -p symthaea --features \"ssm-power-hal\" --example meta_study`

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
