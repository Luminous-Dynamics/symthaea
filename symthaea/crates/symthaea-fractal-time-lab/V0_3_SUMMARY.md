# Symthaea Fractal Time Lab v0.3 File Bundle

This bundle adds:

- CSV output mode.
- Integration tests in `tests/benchmark_sanity.rs`.
- Greedy `BoxCoveringCoarseGrainer`.
- True Laplacian/Fiedler `SpectralCoarseGrainer`.
- Explicit `OddEvenCoarseGrainer` baseline instead of mislabeling odd/even grouping as spectral.
- Experiment docs under `docs/experiments/fractal-time-lab/`.

Copy into:

```text
/srv/luminous-dynamics/symthaea/crates/symthaea-fractal-time-lab/
```

Then run:

```bash
cargo test -p symthaea-fractal-time-lab --lib
cargo test -p symthaea-fractal-time-lab --test benchmark_sanity
cargo run -p symthaea-fractal-time-lab --example run_experiments -- --seed 42 --trials 32
cargo run -p symthaea-fractal-time-lab --example run_experiments -- --json --seed 42 --trials 32
cargo run -p symthaea-fractal-time-lab --example run_experiments -- --csv --seed 42 --trials 32
```
