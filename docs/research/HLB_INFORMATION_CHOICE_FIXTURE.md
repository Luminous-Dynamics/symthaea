# HLB information-choice fixture

**Status:** deterministic reference environment implemented; Symthaea integration not implemented or measured.  
**Entry point:** `cargo run --locked --release -p symthaea-core --example hlb_information_choice_fixture` with `SYMTHAEA_SOURCE_SHA` set to the exact 40-character Git commit SHA.  
**Tests:** `cargo test --locked --release -p symthaea-core --example hlb_information_choice_fixture`.

## The task

The world has a hidden binary goal, Left or Right, with equal prior probability. The agent observes an initial noisy cue. It may then choose one of three actions:
- commit Left;
- commit Right;
- pay for one optional probe, which reveals a second noisy signal.

Correct commitment earns 10,000 basis points (one reward unit); incorrect commitment earns zero. The probe's cost is paid even if the final choice is incorrect. At most one probe is allowed, after which the agent must commit. Repeated inspection and failure to commit are invalid actions with an explicit penalty.

The hidden goal and pre-generated probe signal live in the private environment state. They are not members of `Observation`; the probe is revealed to the policy only after a valid inspection. Every policy runs against the same generated world for each scenario/episode seed.

The seed generator is a fixed SplitMix64 implementation. The checked-in task manifest pins four parameter regimes:
- a high-quality initial cue where the probe is too expensive for its expected gain;
- an ambiguous cue with a sufficiently cheap, high-quality probe;
- the same ambiguous cue with a prohibitively expensive probe;
- a lower-quality probe that is not worth its cost.

The Rust validator recalculates the expected inspect/decline decision from the manifest's observation accuracies, prior, reward and cost. A contradictory declared label is a hard failure.

## Reference policies

The fixture includes four deterministic reference policies:
- `cue_greedy`: act on the maximum-likelihood initial cue, never inspect;
- `always_inspect`: always pays for the optional probe, then uses a maximum-likelihood decision;
- `bayes_value_of_information`: computes the expected increase in optimal accuracy from the probe and inspects only when that value exceeds the cost;
- `random`: seeded random inspection/commit choices.

The fully informed perfect-information oracle is recorded as an upper-bound reference only; it is not a deployable baseline, and no oracle information is exposed to the policies.

The run receipt reports accuracy, mean reward, probe rate, invalid actions and regret against perfect information. Scores from the reference policies are **environment sanity-checks, not Symthaea benchmark scores**.

## What the fixture can and cannot establish

It can establish that:
- hidden observations are not present in the initial public observation;
- repeated inspection is rejected;
- a fixed seed produces fixed hidden states and policy results;
- the declared cheap/high-value and expensive/low-value observation scenarios separate the reference value-of-information policy from simple comparators.

It cannot establish that Symthaea's FEP implementation computes expected information gain correctly, that HDC or CfC improves task performance, or that the full Holographic Liquid Brain beats SOTA. The environment is intentionally small and binary. The next slice must connect the real active-core HDC/LTC incumbent and FEP policy through an explicit observation/action adapter, prove adapter activation, then evaluate on richer held-out tasks including model-structure discrimination and compositional generalization.

## Research motivation

Friston et al. (2026), [*Active inference and artificial reasoning*](https://doi.org/10.1038/s41467-026-77209-5), examine selecting observations that disambiguate competing models using expected information gain over states, parameters and model structure. Symthaea should move beyond a simple binary probe task toward that type of multi-hypothesis measurement-selection test after this fixture qualifies.

Chou et al. (2026), [*A systematic empirical comparison of active inference and reinforcement learning models in accounting for decision-making under uncertainty*](https://doi.org/10.1016/j.jmp.2026.103011), report similar model choice accuracy and overlapping but distinct explanatory strengths in their bandit-task comparison. The right benchmark therefore tests complexity-matched policy and model comparisons rather than assumes that active inference always wins.

For the temporal dimension, Liang et al. (2026), [*Rederived Closed-Form Continuous-Time Neural Networks*](https://doi.org/10.1109/TNNLS.2026.3669345), derive a Lagrange-interpolation based approximation and an error bound for LTC dynamics, reporting results on six tasks. This should be a next replication target for irregularly sampled forecasting, not a promise of an automatic gain for Symthaea.
