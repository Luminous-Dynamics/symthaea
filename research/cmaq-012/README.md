# CMAQ-012: Causal Attribution of Action-Conditioned World-Model Advantage

**Status:** protocol v0.1 + executable Rust synthetic panel; no external benchmark result  
**Tracking issue:** [#9201](https://github.com/Luminous-Dynamics/symthaea/issues/9201)  
**Claim ceiling:** synthetic causal-attribution measurement test; not a trained Symthaea model, robot result, VLA reproduction or SOTA claim

## Why this benchmark exists

World-model prediction metrics and embodied-policy task success often live in separate benchmark families. A survey posted 30 August 2026 catalogues 160 robotics benchmarks and reports only 11 explicit VLA-versus-world-model contrasts and four prediction-to-action bridges. Its key question is whether world modelling creates measurable closed-loop advantage over a matched direct policy, and for which capabilities. [Survey](https://arxiv.org/abs/2609.29669).

The benchmark therefore treats access to action-conditioned predictions as a **causal intervention**. It holds the exogenous episode tape, candidate actions, controller baseline and reward fixed, then compares action-conditioned planning with controls designed to isolate the source of any gain.

## Synthetic panel

The dependency-free Rust harness runs **512 paired episodes per scenario**, each 64 decisions long. Every policy sees the same pre-generated potential outcomes and warning observations for a given episode. There are four lanes:

- **Predictive signal:** a warning observation is informative about a latent hazard. A fast action has high reward in safe contexts and negative reward under hazard; a safe action has a lower but stable reward.
- **Action invariant:** both candidate actions have stable values independent of the warning. The world model should not invent an action advantage.
- **Low information:** the warning channel is near-uninformative (sensitivity and specificity 0.52), while a frozen model incorrectly assumes its training calibration (0.80 / 0.85).
- **Sensor shift:** at step 32 the warning sensor inverts (sensitivity 0.15, specificity 0.20), but the frozen model retains its original calibration.

The task is deliberately small and interpretable. Its role is to test the measurement contract and negative controls before adapting to real environments.

## Policy arms

1. **Direct controller:** chooses the fast action without querying a world model.
2. **Action-blind placebo:** spends the same two model queries as a predictive arm but assigns the same score to both candidate actions. Stable tie-breaking reproduces the direct controller.
3. **Frozen-calibration action model:** computes expected action value conditioned on the warning using fixed training sensor calibration (0.80 / 0.85). It is useful on the matching predictive lane but can transfer negatively under low information or a sensor inversion.
4. **Calibration-oracle action model:** uses the current sensor's true likelihood parameters. This is an upper bound on calibration knowledge, not a deployable baseline.
5. **Shuffled action labels:** deliberately swaps the action-value labels to test whether the apparent benefit depends on correct action conditioning.
6. **Hidden-state oracle:** sees the latent hazard and selects the best action. This is an evaluator-only ceiling.

## Relationship to CMAQ-011's matched estimand

This v0.1 crate is a **synthetic mechanism test** for causal-attribution controls. It is not yet the complete D0/D1/W1/W2/W3 protocol.

In particular:
- The action-blind placebo matches the number of model queries, but does not equalize FLOPs or wall-clock time. It is not the compute-matched direct arm D1.
- The shuffled-action-label arm permutes the two action-value labels. External adapters must implement CMAQ-011's stratified shuffled-future control across candidate sequences.
- The direct controller is intentionally simple and frozen to make the synthetic mechanism legible; it is not evidence of competitiveness against a trained VLA or strong direct policy.
- The calibration-oracle arm is supplied with current sensor likelihood parameters and is an upper bound, not an online calibration algorithm.

Therefore the reported paired deltas are synthetic uplift against this direct controller, not the external benchmark's final Advantage-of-Prediction estimand. Any robot result must implement the full CMAQ-011 D0/D1/W1/W2/W3 design.

## Metrics

- Mean cumulative return and success rate.
- Paired per-episode return delta against the direct controller with a 95% normal-approximation interval.
- CVaR95 return (mean of the worst 5% of episode returns).
- Conditional action regret versus the best action under the true observation-conditional value.
- Action disagreement and counterfactual action-ranking accuracy.
- Prediction MAE and model queries per episode.

The paired confidence interval is computed over episode-level return differences, not over individual timesteps. CVaR95 is reported as a tail diagnostic; no independent-episode guarantee is implied.

## Preregistered controls

- The action-blind placebo must reproduce direct-controller return/action choices while consuming 128 model queries per episode.
- The action-invariant lane must not manufacture prediction lift.
- A calibrated action-conditioned model should improve return on the predictive-signal lane.
- Frozen calibration must expose negative transfer on low-information and sensor-shift lanes; these results are not "fixed" by changing thresholds after looking at the panel.
- Calibration-oracle and hidden-state policies are upper bounds, not industry baselines.
- All model costs remain visible; this panel does not claim equal compute between direct and model-assisted policies.

## Run

```sh
cargo test --locked
cargo run --quiet --locked > report.json
```

The focused GitHub workflow runs the unit tests, generates JSON, validates the 4 × 6 scenario/policy census and checks the preregistered negative controls.

## Limits and next step

This is not a simulator for manipulation, locomotion or a trained Symthaea model. The posterior calculation is analytically matched to the synthetic warning generator. The next version should replace the toy panel with adapters to a pinned task suite and compare a frozen direct controller, the same controller with action-conditioned model access, action-blind/shuffled controls, and an oracle ceiling using identical candidate actions and paired state/seed blocks.

The first external targets should be capability slices where prediction plausibly matters: occlusion/hidden-state inference, long-horizon dynamics and contact-sensitive manipulation. Candidate suites include LIBERO/CALVIN and World-In-World, but adapters and source revisions must be verified before any result is called reproduced.
