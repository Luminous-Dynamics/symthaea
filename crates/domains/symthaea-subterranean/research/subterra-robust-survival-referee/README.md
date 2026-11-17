# Subterra Robust Survival Policy Referee v0.1

**Status:** authored standalone reference implementation; Rust compilation, tests, formatting and Clippy have not yet been executed in this environment.

## Purpose

This crate evaluates one supplied contingent policy tree against a finite family of plausible prior/observation models. It reuses the policy and outcome types from `subterra-survival-reference`, but independently recomputes model-specific probabilities, posteriors, terminal losses, interruption/robot-loss risk and resource use.

It is a **referee, not an optimizer or controller**. It does not choose a different policy for each model and does not authorize a robot action.

## Why a referee before an optimizer

The single-step EVSI, finite-horizon survival planner and finite-model robust EVSI references solve adjacent problems. Naively planning independently under each model gives the planner knowledge of which model is true.

This referee tests a candidate tree unchanged across all models. At a shared observation history the action is identical; the models differ only in the probability of the observed outcomes and resulting posterior. This prevents model clairvoyance and provides a measurement boundary before a joint robust optimizer is attempted.

## Independent calculations

For model (m), history (h), action (a), and outcome (o):

`P(o | m,h,a) = Σ_i P(i | m,h) P(o | i,a,m)`

`P(i | m,h,o,a) ∝ P(i | m,h) P(o | i,a,m)`

The referee deliberately ignores the candidate `PolicyBranch.probability` and `PolicyBranch.posterior` caches. It recomputes both quantities from the model and reports cache mismatches as diagnostics.

## Hard checks

A policy is admissible only when every supplied model:
- admits every selected action through its external gate;
- remains within energy, duration, data and exposure budgets on every supported path;
- stays within the declared mission-interruption and robot-loss probability limits.

Hard failures are not averaged away. The unweighted mean across models is descriptive only; no model weights are inferred.

## Scope and limits

- The ambiguity set is finite and caller-supplied; it is not guaranteed to contain the real sensor/geology model.
- Observation likelihoods, priors, costs and loss matrices require independent calibration/review.
- A `RelayLostStoreAndForward` outcome is only meaningful when a separate evidence process verifies local retention; this crate does not authenticate that evidence.
- Synthetic test conformance is not geological, structural, dynamic, physical, planetary-protection or regulatory qualification.
- A referee PASS is not authority to dispatch, enter, bore, sample, repair or alter infrastructure.

## Run

From this directory:

```sh
cargo fmt --all -- --check
cargo test --offline
cargo clippy --offline --all-targets -- -D warnings
```

The only dependency is the adjacent standalone survival-planner reference crate. No third-party crate is required by this referee.

## Core tests

The unit tests include hand-computable expected-loss/risk values, model-specific risk divergence, external-gate denial, path-wise budget exhaustion, policy-cache tampering, missing branches and model-schema mismatch. They are source-authored tests until an exact-head Rust toolchain run records execution evidence.
