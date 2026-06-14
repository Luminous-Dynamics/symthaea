# Symthaea Fractal Time Lab v0.5 — replacement files

Copy these into:

```text
/srv/luminous-dynamics/symthaea/crates/symthaea-fractal-time-lab/
```

Suggested verification:

```bash
cd /srv/luminous-dynamics/symthaea

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

Scope:

- Exploratory benchmark crate.
- Not physical proof of fractal time, quantum consciousness, or cosmological recurrence.
- Do not modify `symthaea-wisdom` while integrating this.
