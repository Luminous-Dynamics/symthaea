# Subterra survival-conditioned planner v0.1

A dependency-free Rust reference planner for finite-horizon scientific measurement selection when retreat, interruption, robot loss, or relay failure can remove future sensing opportunities.

## Why this exists

A greedy measurement selector can overvalue an action that is informative now but has a material probability of terminating the mission before later measurements can be acquired. This planner represents a contingent policy tree:
- Observation outcomes update the hypothesis posterior and may continue.
- Safe retreat ends that survey branch without being scored as robot loss.
- Mission interruption and robot loss are explicit absorbing outcomes with caller-supplied terminal losses.
- Relay loss with store-and-forward is a separate terminal outcome, retaining a caller-supplied consequence rather than assuming all data are lost.
- Every continuation receives the remaining resource budget on that branch.
- The planner can stop at any node and select the best current mission decision.
- Mission-interruption and robot-loss probabilities are constrained by separate hard ceilings.
- Search expansion has an explicit upper bound and deterministic action ordering.

## Relation to the one-step EVSI reference

The sibling `subterra-voi-kernel` remains the one-step mathematical reference. This crate is a separate research successor and does not replace it. The current finite-horizon implementation assumes:
- a finite, static hypothesis set;
- caller-supplied, normalized priors and conditional outcome likelihoods;
- a conditional independence model for repeated observations unless the caller encodes shared latent state in the hypothesis set;
- fixed outcome likelihoods per action;
- caller-supplied terminal losses and resource prices.

It does not fit or calibrate likelihoods, model arbitrary correlated measurements automatically, authenticate sensor lineage, solve SLAM or geomechanics, plan geometric robot trajectories, issue work orders, or control actuators. The external gate is caller-supplied and cannot be generated or widened by this planner.

## Run locally

From the Symthaea checkout:

```bash
cargo fmt --manifest-path crates/domains/symthaea-subterranean/research/subterra-survival-planner/Cargo.toml -- --check
cargo test --manifest-path crates/domains/symthaea-subterranean/research/subterra-survival-planner/Cargo.toml
cargo clippy --manifest-path crates/domains/symthaea-subterranean/research/subterra-survival-planner/Cargo.toml --all-targets -- -D warnings
```

This crate has its own Cargo workspace and does not join or alter the production Symthaea workspace.

## Evidence state

- Source and 10 unit-test cases: AUTHORED.
- Mathematical policy-tree sanity check: independently recalculated in a separate executable reference script; not a Rust execution.
- Rust formatting, compile, tests and Clippy: NOT EXECUTED in the current authoring environment (no cargo/rustc present).
- CI: NOT CHECKED; any skipped job is not a pass.
- Robot-in-the-loop, terrestrial analogue, flight or planetary validation: NOT PERFORMED.
