# Subterra likelihood calibration reference v0.1

A standalone Rust research package with one local dependency on the Subterra EVSI reference for estimating finite categorical observation likelihoods from independently labelled cave-site cases, scoring the estimates on disjoint calibration/evaluation sites and converting fully supported likelihood tables into EVSI inputs.

## Why this exists

The EVSI and robust-EVSI planners accept caller-supplied values of P(outcome | hypothesis, measurement). Those values must not be guessed from qualitative claim states. This package defines a data contract and a small empirical estimator for deriving those likelihoods from independent, labelled cases.

## Core design

- Exactly one non-zero-support hypothesis ID named `unknown` must be present in the contract.
- Each case binds a site, a physical acquisition-lineage ID, an action, an independently adjudicated reference hypothesis, an observed outcome, split assignment and opaque evidence references.
- A site may appear in only one split. The same acquisition lineage may appear only once. Each site/action pair may contribute only one case.
- This prevents repeated processing of one acquisition or multiple sessions at one site from silently inflating the effective sample count.
- The fit uses only the Training split. Calibration and Evaluation cases do not affect counts or likelihoods.
- A symmetric Dirichlet prior with caller-declared alpha smooths categorical outcome counts.
- A minimum independent training-site count gates whether a likelihood cell is emitted. Under-supported cells carry no probabilities.
- The table is immutable through the public API and includes the ordered action IDs, sensor modalities and hypothesis IDs.
- A conversion helper builds a one-step EVSI problem only when every hypothesis/action cell is supported and the caller supplies matching priors, decision losses, resource costs and external gate decisions.
- Calibration and Evaluation splits are scored separately using multiclass Brier score and negative log-likelihood (NLL), both proper scoring rules.
- The Evaluation split cannot be used to fit the model, and the Training split cannot be passed to the holdout scorer.

## Important limitations

- The estimator is a simple categorical frequency model with Dirichlet smoothing; it is not a neural calibration method, causal sensor model, hierarchical Bayesian model, conformal predictor or guarantee under distribution shift. The EVSI conversion is a type adapter, not a claim that the empirical probabilities are calibrated or fit for a particular mission.
- It does not infer priors, mission losses, decision costs or action admissibility.
- It does not authenticate reference labels or observation evidence; all references are opaque identifiers.
- It assumes one independent case per site/action and unique acquisition-lineage IDs. More complex repeated-measurement or cross-sensor dependence requires a richer joint model, not duplicate rows.
- A held-out score is only evidence on the declared calibration/evaluation sites. It does not establish planetary transfer.
- A likelihood cell with too little support is not emitted and must not be passed into EVSI as an invented distribution.

## Run locally

~~~bash
cargo fmt --manifest-path crates/domains/symthaea-subterranean/research/subterra-likelihood-calibration/Cargo.toml -- --check
cargo test --manifest-path crates/domains/symthaea-subterranean/research/subterra-likelihood-calibration/Cargo.toml --all-targets
cargo clippy --manifest-path crates/domains/symthaea-subterranean/research/subterra-likelihood-calibration/Cargo.toml --all-targets -- -D warnings
~~~

This is a standalone nested Cargo workspace and is not included in the production Symthaea workspace. Its only dependency is the adjacent subterra-voi-reference package.

## Evidence state

- Data contract, empirical estimator and 19 unit-test cases: AUTHORED.
- Rust formatting, compilation, tests and Clippy: NOT EXECUTED in the current authoring environment (no cargo/rustc/rustfmt).
- Calibration quality, site transfer, robot-in-the-loop, cave analogue, flight and planetary validation: NOT PERFORMED.
