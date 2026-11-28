# Symthaea GameBench Decision Bridge v0.1

This is a small adapter from game-specific, read-only mechanics probes to the repository's existing finite-world Bayesian decision evaluator.

It intentionally does **not** implement EVSI, a posterior engine, a robust oracle, a sequential planner, or an execution authority.

## Reused decision authority

The adapter depends on the existing research/decision-predictability-benchmark crate:

- Bayesian posterior update over evaluator-owned hidden-state priors and likelihoods.
- Expected posterior Bayes risk, gross/net expected value of sample information (EVSI), and mutual information as a separate diagnostic.
- Permission, acquisition-cost, deadline, and risk-admission checks.
- A separate prior-free robust evaluator.

That library is the decision-value authority. This bridge maps a GameBench mechanics-probe vocabulary onto it and adds proper scoring for candidate-authored pre-probe forecasts.

The repository also has a separate prior-free sequential policy-oracle contribution on draft PR #13571. That owner remains responsible for set-valued, multi-step robust policy trees. This bridge does not copy its recurrence or evaluator.

## Supported v0.1 scope

Only **read-only, non-mutating probes** are supported, such as inspecting a switch label, reading a visible panel, or querying a non-invasive sensor. The candidate emits its forecast before the probe; after the environment returns an observed outcome, its forecast can be scored with the multiclass Brier score.

World-changing actions (pulling a lever, consuming a key, advancing a hazard, opening a gate) are rejected by this adapter. The shared one-step EVSI contract does not have outcome-specific transition/utility matrices. Those actions require an explicit transition model and composition with the sequential/robust decision owners; they must not be passed off as read-only observations.

## Candidate/evaluator separation

Candidate-visible:
- allowed terminal decision IDs;
- visible evidence and public probe IDs;
- declared probe costs/deadlines;
- public outcome labels for the selected probe;
- candidate-authored outcome forecast.

Evaluator-only:
- hidden game state and mechanism;
- priors and likelihood tables;
- decision-loss table;
- posterior and Bayes-risk outputs;
- oracle value and information-gain metrics.

The shared crate's type separation is not a process sandbox. Confirmatory evaluation must keep evaluator truth in a separately permissioned evaluator process and release only the actual observation to the candidate.

## Development known-answer task

The unit test encodes four paired hidden worlds:
- two competing switch mechanics;
- a nuisance lamp state independent of the terminal decision;
- a high-information but decision-irrelevant lamp inspection;
- a lower-information, decision-relevant non-mutating switch trace.

With the frozen synthetic prior and loss table, the shared evaluator should report:
- nuisance probe: 1 bit of information gain, 0 gross EVSI, negative net EVSI;
- switch trace: approximately 0.1887 bits of information gain, 0.15 gross EVSI, 0.11 net EVSI.

These are reference-fixture values, not agent results. The tests assert the adapter receives them from the shared evaluator; it does not reimplement those equations.

## Qualification

From the repository root:

~~~sh
cargo fmt --manifest-path research/symthaea-gamebench-decision-bridge-v0-1/Cargo.toml -- --check
cargo test --manifest-path research/symthaea-gamebench-decision-bridge-v0-1/Cargo.toml
cargo clippy --manifest-path research/symthaea-gamebench-decision-bridge-v0-1/Cargo.toml --all-targets -- -D warnings
~~~

The 6 tests in this crate are authored source tests until exact-head CI runs them. The dependency's earlier recorded 16/16 pass does not qualify this bridge or any GameBench candidate.

No candidate campaign, hidden-mechanic learning, transfer, visual competence, or SOTA result is claimed.
