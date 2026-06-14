# Symthaea Fractal Time Lab

Experimental benchmark records for `symthaea-fractal-time-lab`.

## Scope

This lab tests computational structure:

- scale-recursive spectral similarity,
- persistent subharmonic temporal response,
- multi-scale integration survival,
- graph coarse-graining diagnostics,
- greedy box-covering and box-dimension heuristics.

It does **not** prove fractal time, quantum consciousness, cosmological recurrence, or any physical claim beyond the implemented simulations.

## Recommended run commands

```bash
cargo test -p symthaea-fractal-time-lab --lib
cargo test -p symthaea-fractal-time-lab --test benchmark_sanity

cargo run -p symthaea-fractal-time-lab --example run_experiments -- \
  --seed 42 --trials 32

cargo run -p symthaea-fractal-time-lab --example run_experiments -- \
  --json-run --seed 42 --trials 32

cargo run -p symthaea-fractal-time-lab --example run_experiments -- \
  --markdown --seed 42 --trials 32

cargo run -p symthaea-fractal-time-lab --example run_experiments -- \
  --seed 42 --trials 32 \
  --json-run-out docs/experiments/fractal-time-lab/latest-run.json \
  --json-out docs/experiments/fractal-time-lab/latest-scorecards.json \
  --csv-out docs/experiments/fractal-time-lab/latest-scorecard.csv \
  --markdown-out docs/experiments/fractal-time-lab/latest-report.md
```
