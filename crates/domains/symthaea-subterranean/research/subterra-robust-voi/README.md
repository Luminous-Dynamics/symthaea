# Subterra robust EVSI reference v0.1

A dependency-light Rust wrapper around the one-step EVSI reference. It evaluates a frozen candidate measurement set across a finite, explicitly declared ensemble of plausible prior / likelihood models.

## Core rule

A candidate measurement is robustly selectable only when:
1. every model describes the same hypothesis IDs and ordering;
2. every model uses the same mission decision IDs, ordering and loss matrix;
3. every model uses the same candidate action IDs, outcome IDs, modalities and resource costs;
4. each model independently admits the action through its supplied external gate and hard resource budget;
5. the minimum net EVSI across the entire declared model set exceeds the configured threshold.

Priors and conditional outcome likelihood values may vary between models. A model-specific denied external gate or over-budget result excludes the action; it is never averaged away.

## Output

For each action the report provides:
- per-model status and net EVSI;
- worst-case net EVSI across the finite declared model set;
- descriptive arithmetic mean across models;
- best-case net EVSI;
- selected / not selected / excluded disposition.

No model weights are invented. The mean is not a probabilistic expectation unless weights are introduced and justified by a separate study protocol.

## Scope and limitations

This is a finite model-ensemble, one-step EVSI wrapper. It does not:
- construct or calibrate the model ensemble;
- prove that the true physical model is included in the finite set;
- establish universal worst-case safety or risk bounds;
- infer sensor lineage or common-cause dependence automatically;
- implement robust finite-horizon survival planning;
- evaluate CVaR, distributionally robust optimization, or minimax regret;
- plan geometry or issue robot commands.

A worst-case guarantee is only with respect to the explicit model set supplied to the function. Model selection, calibration, OOD detection and sensor-dependence assumptions require independent evaluation.

## Run locally

```bash
cargo fmt --manifest-path crates/domains/symthaea-subterranean/research/subterra-robust-voi/Cargo.toml -- --check
cargo test --manifest-path crates/domains/symthaea-subterranean/research/subterra-robust-voi/Cargo.toml
cargo clippy --manifest-path crates/domains/symthaea-subterranean/research/subterra-robust-voi/Cargo.toml --all-targets -- -D warnings
```

The only dependency is the adjacent standalone `subterra-voi-reference` crate. Neither crate joins the production Symthaea workspace.

## Evidence state

- Robust model-ensemble wrapper and 10 unit tests: AUTHORED.
- Rust formatting, compilation, tests and Clippy: NOT EXECUTED (no Rust toolchain available in the authoring environment).
- CI: NOT CHECKED / skipped jobs are not passes.
- Sensor calibration, cave analogue, robot-in-the-loop, flight and planetary validation: NOT PERFORMED.
