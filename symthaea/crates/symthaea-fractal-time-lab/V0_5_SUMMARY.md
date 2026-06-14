# Symthaea Fractal Time Lab v0.5 File Bundle

This bundle adds evidence/report discipline on top of v0.4:

- `BenchmarkRun` metadata object.
- Explicit claim/non-claim boundaries in machine-readable output.
- Markdown report generator.
- `--markdown` and `--markdown-out`.
- `--json-run` and `--json-run-out`.
- `--fail-on-benchmark-fail` for optional benchmark gating.
- Additional claim-scope tests.
- Updated CI workflow scaffold.

Copy into:

```text
/srv/luminous-dynamics/symthaea/crates/symthaea-fractal-time-lab/
```

Then run:

```bash
cargo test -p symthaea-fractal-time-lab --lib
cargo test -p symthaea-fractal-time-lab --test benchmark_sanity
cargo run -p symthaea-fractal-time-lab --example run_experiments -- --seed 42 --trials 32
cargo run -p symthaea-fractal-time-lab --example run_experiments -- --markdown --seed 42 --trials 32
```
