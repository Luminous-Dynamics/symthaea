# Subterra EVSI reference kernel v0.1

A dependency-free Rust reference implementation for decision-focused selection of the next scientific measurement in a partially observed planetary subsurface.

## Why this exists

Entropy reduction and raw information gain are not always useful mission objectives. A measurement can reduce uncertainty without changing the best available decision. This kernel calculates expected value of sample information (EVSI), then subtracts declared resource shadow prices and ranks only pre-admitted candidate actions.

The explicit open-world hypothesis is mandatory: every problem must include a non-zero-probability hypothesis with ID `unknown`. This prevents the planner from forcing every site into the currently enumerated geological categories.

## Scope

Included:
- finite hypothesis priors, explicit unknown;
- finite mission decision loss matrix;
- per-action conditional outcome likelihoods;
- one-step Bayesian posterior calculation;
- expected pre- and post-measurement decision loss;
- EVSI and separately reported entropy reduction;
- energy, duration, data and exposure budgets and shadow prices;
- external-gate denial, hard budget rejection, deterministic tie-break;
- abstention when no eligible action has net value above the declared minimum;
- unit tests for mathematical controls and adversarial inputs.

Not included:
- fitting or calibrating priors, losses or outcome likelihoods;
- arbitrary joint/correlated sensor likelihoods;
- source authentication, sensor trust or evidence verification;
- geological reconstruction, SLAM or coordinate transforms;
- route planning, motor commands or actuator authority;
- safety, planetary-protection or sampling authorization;
- physical or planetary qualification.

The caller supplies the likelihood model and must calibrate it independently. An external reviewed gate must decide whether an action is eligible before it is passed to the planner. That gate is a hard constraint, not a soft cost.

## Run locally

From the Symthaea checkout:

```bash
cargo fmt --manifest-path crates/domains/symthaea-subterranean/research/subterra-voi-kernel/Cargo.toml -- --check
cargo test --manifest-path crates/domains/symthaea-subterranean/research/subterra-voi-kernel/Cargo.toml
cargo clippy --manifest-path crates/domains/symthaea-subterranean/research/subterra-voi-kernel/Cargo.toml --all-targets -- -D warnings
```

This crate has its own Cargo workspace and does not join or alter the production Symthaea workspace.

## Evidence state

- Source kernel and unit-test source: AUTHORED.
- Compilation and Rust test execution: NOT EXECUTED in the authoring environment.
- CI: NOT CHECKED.
- Hardware, cave analogue, flight or planetary validation: NOT PERFORMED.

A mathematical worked example is included in the unit tests. That is a deterministic reference control, not empirical measurement-selection performance.
