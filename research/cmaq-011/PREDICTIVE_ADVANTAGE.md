# CMAQ-011: What Does Prediction Actually Buy?

## Primary claim

The benchmark must distinguish three statements:

1. A world model predicts future states accurately.
2. A world-model planner succeeds at a downstream task.
3. **Using those predictions causes better task outcomes than a matched direct policy.**

Only statement 3 establishes the advantage of prediction. The first two are necessary supporting evidence, not substitutes.

## Controlled decision graph

`frozen proposer -> shared candidate action sequences -> {direct evaluator | learned-world-model evaluator | oracle evaluator} -> executed action -> environment feedback`

All arms use paired initial states, identical candidate sequences, action horizons, task reward, resets and evaluation seeds. The world model must not receive a stronger proposer or additional action candidates unless those are reported as a separate factor.

## Five mandatory arms

- **D0:** strong frozen direct policy, no world-model rollouts.
- **D1:** compute/latency-matched direct policy; spend the world-model inference budget on direct-policy refinement or ensembling, not on future prediction.
- **W1:** learned action-conditioned world model ranks the shared candidates.
- **W2:** shuffle the predicted futures across candidate action sequences within matched state/task strata, preserving marginal predictions but breaking action/future alignment.
- **W3:** exact simulator/oracle transitions for the same candidates and horizon.

D0 answers whether the whole deployed system helps over a direct policy. D1 isolates the contribution of predictive modeling from extra compute. W2 tests whether the planner actually uses correct action-conditioned futures. W3 identifies the residual gap due to model error versus proposer/planner quality.

## Primary estimands

Per capability and task:

`AoP = E[U(W1) - U(D1)]`

`Deployment gain = E[U(W1) - U(D0)]`

`Prediction mediation = E[U(W1) - U(W2)]`

`Oracle gap = E[U(W3) - U(W1)]`

`Compute-only gain = E[U(D1) - U(D0)]`

Estimate each with paired episodes and paired bootstrap confidence intervals. Publish seed-level paired differences, not just unpaired means. For success-rate metrics, report absolute percentage-point difference and relative ratio separately.

Interpretation:
- W1 > D0 but W1 ~= D1: benefit may be extra compute, not prediction.
- W1 ~= W2: the action-conditioned predictions may not causally influence the selected action.
- W3 ~= W1 but both poor: improve proposal/planning/reward design, not model fidelity first.
- W3 >> W1: model fidelity, uncertainty or horizon is likely limiting performance.
- W1 < D1: world modeling is a net liability at this budget, even if one-step prediction metrics look good.

## Capability strata

At minimum, pre-register separate task slices for:
- hidden dynamics / latent state inference;
- occlusion and active perception;
- delayed action effects and long-horizon dependencies;
- contact-rich interaction and object dynamics;
- recovery after perturbation;
- out-of-distribution object, geometry or dynamics parameter families.

For each slice report task success, return/regret, lower-tail success, failure mode, calibration and latency. Do not average away a task where prediction causes regressions.

## Controls and anti-gaming

- Candidate sets and proposer outputs are hashed and shared across arms.
- Oracle transitions are used only to score the selected action/candidate, never leaked to D0/D1/W1/W2.
- Shuffled-future permutations are seeded, stratified and independently validated.
- Replanning cadence, rollout depth, reset handling and termination rules are identical.
- Failed, timed-out and resource-exhausted episodes remain in the result denominator with an explicit status.
- A model-side confidence or value estimate cannot serve as ground truth for its own selected action.
- Report raw prediction scores and decision outcomes side by side; no composite score may conceal a trade-off.

## Relation to current research

- *Do World Models Make Better Robots? A Survey of Evaluation Benchmarks for Predictive Embodied Intelligence* (arXiv:2609.29669, 30 August 2026) identifies a measurement gap: its 160-benchmark corpus contains 11 explicit direct-VLA/world-model comparisons and four prediction-to-action bridges. The authors propose per-capability closed-loop advantage, counterfactual competence, prediction-to-action retention, and capability coverage. CMAQ-011 operationalizes those questions with matched D0/D1/W1/W2/W3 arms.
- *World-In-World* (ICLR 2026 Oral) provides a usable closed-loop visual-world-model task family, but the upstream repository still lists portions of evaluation setup and documentation as under construction. Treat adapter readiness and reproducibility as explicit deliverables.
- *VLAW* (ICML 2026) reports iterative co-improvement of a VLA and action-conditioned video world model, including real-robot experiments. It motivates a later training-data feedback study; it does not replace the evaluation-only D0/D1/W1/W2/W3 protocol.

## Research target

Demonstrate positive AoP on at least two capabilities and two task families with:
- paired 95% confidence intervals excluding zero;
- no material CVaR95 or failure-rate regression;
- W1 outperforming W2, showing that action-conditioned predictions matter;
- W1 outperforming D1 at equalized inference resources;
- held-out task-parameter families untouched by tuning;
- a fully reproducible independent rerun.

These are proposed acceptance targets, not results already achieved.
