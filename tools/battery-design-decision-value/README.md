# Battery design decision-value benchmark v0.1

A dependency-free Rust reference evaluator for tiny, finite, known-truth battery design decisions. It compares the value of proposed experiments by their ability to reduce decision loss, while keeping probabilistic EVSI separate from distribution-free minimax-regret analysis.

This is an evaluation oracle, not an optimizer, test scheduler, laboratory controller or physical battery model.

## Run from the repository root

    cargo metadata --manifest-path tools/battery-design-decision-value/Cargo.toml --locked --no-deps
    cargo fmt --manifest-path tools/battery-design-decision-value/Cargo.toml -- --check
    cargo test --manifest-path tools/battery-design-decision-value/Cargo.toml --locked
    cargo clippy --manifest-path tools/battery-design-decision-value/Cargo.toml --all-targets --locked -- -D warnings

The schema and synthetic benchmark fixture are stored under `schemas/transport/` and `fixtures/ev_battery/`.

## Semantics

- Probability-bearing EVSI requires an explicit prior/likelihood model and an admitted calibration/known-truth receipt.
- Set-valued analysis requires no probabilities and reports worst-case regret separately.
- The evaluator never converts intervals or sets into probabilities.
- Hard feasibility is non-compensable: a design must remain feasible in every world still considered possible before it can be selected.
- Failed, censored, invalid and missing experiment outcomes stay in the possible-outcome ledger.
- Resource and experiment costs are reported independently; a declared decision-loss-equivalent cost is used only for net-value comparison.
- No positive net value means STOP / NO_EXPERIMENT.

A passing result establishes only decision arithmetic and protocol behavior on the declared finite fixture. It is not evidence of a physical battery improvement.
