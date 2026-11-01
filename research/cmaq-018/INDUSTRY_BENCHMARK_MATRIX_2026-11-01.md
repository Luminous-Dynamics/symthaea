# CMAQ-018 / VIS-005 frontier benchmark matrix — 2026-11-01

**Purpose:** establish a defensible route from predictive quality to better decisions, with matched direct-policy baselines, explicit counterfactuals, selection-aware calibration and resource accounting.

**Status:** research protocol and target gates. These are not measured Symthaea results and are not a claim of SOTA.

## 1. What the current field is measuring — and what is missing

Recent benchmark reviews support an important strategic distinction:

- The 2026 Survey of World Model Benchmarks indexes 102 representative benchmarks, with 85 spanning multiple target categories. Its task coding identifies 98 open-loop versus 13 closed-loop protocols and 99 prediction-level versus 14 downstream-outcome metric tracks. Categories include visual/temporal quality, spatial/state consistency, long-horizon memory, physical plausibility, causal/counterfactual reasoning, control fidelity and functional utility. Counts overlap by benchmark and evaluation target. [Survey and benchmark explorer](https://world-model-benchmarks.github.io/).
- Do World Models Make Better Robots? maps 160 benchmarks and reports that only 11 (7%) explicitly compare VLA and world-model approaches, while only four connect prediction to an executed action. This is a preprint survey, not an empirical SOTA leaderboard, but it identifies a meaningful evaluation gap: does prediction earn decision value against a matched direct policy? [arXiv:2609.29669](https://arxiv.org/abs/2609.29669).
- [WorldOracle](https://arxiv.org/abs/2610.18826), dated 2026-10-13, compares action-conditioned rollouts with paired real executions and evaluates physical-state correctness, interaction alignment, key interaction regions and multi-view action coherence.
- [MiraBench](https://arxiv.org/abs/2605.29360), dated 2026-05-28, separates physical consistency, action-following fidelity and optimism-bias detection. Its reported findings warn that visual fidelity is a poor proxy for action fidelity, larger scale does not reliably fix action following, and optimistic failure prediction is common.
- [CaliBench](https://arxiv.org/abs/2608.16829), dated 2026-08-17, measures stochastic physical-outcome calibration in an interpretable discrete space and separates scorability from calibration. This is a useful complement for aleatoric distribution fidelity, not a substitute for intervention- or decision-level calibration.
- [World-In-World](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html), ICLR 2026, is a closed-loop platform. Its reported findings emphasize controllability and post-training action-observation data rather than visual quality alone, and show that inference-time compute can change task performance.
- [Rollback](https://arxiv.org/abs/2610.36068), dated 2026-10-26, uses rollback to decision points, alternative action branches and real robot feedback to improve policies. Its reported task gains are task-specific and must not be compared numerically with a different task suite.

**Implication:** do not enter a visual realism leaderboard and claim world-model superiority. Build a bridge benchmark with the same histories, actions, direct-policy baseline, simulator truth, resource budget and downstream task outcomes. This is where the current programme can make a distinct contribution.

## 2. Evaluation ladder

| Level | Question | Primary metrics | Evidence needed |
|---|---|---|---|
| P0 — representation / visual | Does the prediction preserve task-relevant visual structure? | geometry/state error, temporal consistency, track identity, object permanence | held-out observations; geometry that the model actually emits |
| P1 — physical / stochastic | Does the predicted distribution match physical outcomes? | scorability, discrete-outcome TV distance, physical constraints, coverage, interval score | repeated outcomes, interpretable state, calibrated held-out set |
| P2 — action following | Does the prediction reflect the commanded action? | action-conditioned state error, interaction alignment, contact/terminal transition accuracy, optimism false-safe rate | paired execution with the same initial history and action sequence |
| P3 — counterfactual effect | Does changing the action produce the correct predicted difference? | paired effect sign/magnitude error, rank correlation, action reversal detection, branch coverage | exact cloned simulator state or genuinely randomized matched intervention |
| P4 — selected-action reliability | Does the uncertainty remain valid after selecting from the full action bank? | simultaneous action-set coverage, regret-bound coverage, false-dominance rate, abstention utility | frozen candidate set; evaluator-only outcomes; episode as statistical unit |
| P5 — decision advantage | Does prediction improve decisions over a direct policy? | task success/return, realized regret, net decision value, failures/recovery | matched D0/D1/W1/W2/W3 on same episodes and budget |
| P6 — robust utility | Does the advantage persist under policy-induced shift and exploit attempts? | OOD degradation, worst-stratum regret, exploitability, constraint violation, recovery | held-out mechanism families, adversarial optimizer, real execution bridge |

A model does not inherit the evidence of a higher level from a lower one. A P0 visual win is not a P5 decision win.

## 3. Mandatory comparator panel

Every primary evaluation must include these arms with matched input history, candidate action set, observation/action semantics, environment reset, training data, inference/training compute and accounting:

| Arm | Definition | Why it is required |
|---|---|---|
| D0 | reactive/direct policy with no learned world-model calls | opportunity-cost baseline |
| D1 | direct value/action policy using permitted state/history but no learned transition model | controls for extra representation/value capacity |
| W1 | world-model planning with a fixed, preregistered planner and no privileged oracle | measures ordinary planning benefit |
| W2 | uncertainty-aware world-model planning with abstention / value-of-information option | measures whether calibrated uncertainty improves selection |
| W3 | adaptive compute/search policy with a hard inference budget and early-stop rule | tests whether extra inference compute is worth its cost |
| Oracle | evaluator-only exact dynamics / ground truth | ceiling and instrumentation control; never a deployable comparator |

Also include shuffled-action labels, shuffled history, no-coupling/independent-draw, point predictor, explicit-effect-head, support-aware and optimistic-failure negative controls. Do not omit failed, timed-out or invalid episodes.

## 4. Decision-centered metrics

### 4.1 Prediction and effect fidelity

For an episode with history h and action pair (a_i, a_j), score the paired effect:

    Delta Y(h; a_i, a_j) = Y(do(a_i) | h) - Y(do(a_j) | h)

Report effect sign accuracy, effect magnitude MAE, rank Kendall tau-b, top-1 regret, null-effect false-positive rate, terminal false-safe rate and branch outcome/state MAE. Report by horizon and mechanism family, not only pooled.

### 4.2 Calibration after adaptive selection

For each episode, score the entire candidate action set, not each action as an independent sample:

- marginal coverage by action and outcome channel;
- simultaneous coverage: all action outcomes covered in the same episode;
- mean interval width and proper interval score;
- selected-action regret-bound coverage;
- false-dominance certificate count/rate;
- abstention rate and abstention utility / opportunity cost.

The episode/history is the sampling unit. Do not treat 21 pairs from seven actions, or 128 joint draws from one history, as 21 or 128 independent episodes. Confidence intervals must cluster at the original history/episode.

### 4.3 Net decision value

    NDV = realized utility(selected action) - realized utility(D0) - inference cost - acquisition cost - failure/recovery cost

Report task return and success, but also compute, memory, energy, wall-clock latency, number of candidate branches, model calls and the cost of abstention / asking for another observation. Use a common declared utility/cost scale and keep safety-critical failure penalties separate from ordinary reward.

## 5. Preregistered qualification targets

These are proposed gates, not claimed current performance. Adjust them only before seeing sealed evaluation results and record the change:

| Gate | Minimum proposed target | Failure interpretation |
|---|---|---|
| Payload integrity | 100% rejection of byte mutation, stale digest, duplicate IDs, missing draws, non-finite samples and mismatched action/draw cardinality | artifact or contract integrity failure |
| Paired effect correctness | exact known-answer effects and permutation invariance on all synthetic controls | adapter/math failure |
| Calibration | nominal 90% and 95% coverage reported with episode-clustered uncertainty; supported strata's lower confidence bound must be within 2 percentage points of nominal | miscalibration / insufficient sample |
| Selection reliability | zero false-dominance certificates on at least 3,000 targeted adversarial selection episodes; report a one-sided confidence bound, not just 0 observed | unsupported selection certificate |
| Abstention | abstain on unsupported/OOD action sets; separately report missed-opportunity cost | safety/utility trade-off |
| Decision value | lower 95% episode-clustered CI of net decision-value advantage over D0 and D1 greater than zero under equal budget | no demonstrated advantage |
| Robustness | no hidden-regime or action-reversal stratum may be rescued by pooled averages; report worst-stratum regret, terminal false-safe and OOD coverage | brittle model |
| Efficiency | publish Pareto curve of NDV versus latency, energy, memory and inference calls; W2/W3 must beat an equal-compute comparator | compute does not buy value |
| Reproduction | at least 5 independently seeded runs for stochastic training/policy claims; retain exact source, environment, weights, profile and per-episode receipts | no reproducible SOTA claim |

Zero false certificates in 3,000 independent targeted episodes gives a one-sided 95% exact binomial upper bound of approximately 0.0998% (about 1 per 1,002 episodes), assuming independent representative trials. It does not establish universal safety, and clustered or mechanism-correlated episodes reduce the effective sample size. The benchmark must publish the sampling design, effective independent episode count and confidence construction.

## 6. SOTA comparison protocol

1. Freeze task versions, initial histories, candidate actions, observation/action spaces, horizon, success definition and cost utility.
2. Compare published baselines on their own named benchmarks where possible: WorldOracle for paired real rollout fidelity; MiraBench for action-following/optimism; CaliBench for stochastic outcome calibration; World-In-World for closed-loop utility. Do not combine scores across tasks or infer cross-paper ranking.
3. Add the same Symtropy hidden-ground-truth branch set to D0/D1/W1/W2/W3. Keep the predictor blind to hidden state, unchosen branch truth, future frames and evaluator labels.
4. Match data and compute budgets. Publish both equal-training-data and equal-total-compute comparisons. Include the cost of sampling every action in W2/W3.
5. Use paired episodes and episode-clustered bootstrap or appropriate randomization inference. Report all seeds, means, dispersion, confidence intervals and failure counts.
6. Release exact raw per-episode receipts and a replay command. An aggregate report without replayable evidence is a diagnostic, not a qualification result.
7. Claim “better than SOTA” only for a precisely named task/metric/protocol when the preregistered confidence interval excludes the strongest matched baseline, and when the gain does not come from hidden information or extra uncharged compute.

## 7. Paradigm shifts worth pursuing

### A. From prediction accuracy to intervention fidelity

Train/evaluate on paired action differences from matched histories. Preserve common random numbers where structurally valid. Give the model credit for getting the effect right, not merely predicting the most common future.

### B. From per-action uncertainty to joint action-set uncertainty

Generate all candidate action outcomes from one shared posterior/noise draw bank. Compute paired effect distributions from same-column differences. The action-bank content digest must bind the exact values; producer coupling still needs independent provenance.

### C. From confidence scores to decision certificates with abstention

Only issue an interval-dominance certificate when one action's lower bound clears every competitor's upper bound. Otherwise abstain or request a new observation. A certificate is conditional on simultaneous coverage and is never actuator authority.

### D. From passive world models to falsifiable counterfactual engines

Allow rollback and exact branching in Symtropy, but keep evaluator truth sealed. Use the resulting branch data to audit action effects, selection regret and model exploitation before policy updates.

### E. From aggregate benchmark scores to a value/compute frontier

Measure the marginal net decision value from each additional rollout, candidate action, sensory query and planner step. Stop inference when expected value of information is below its total cost or when uncertainty is unsupported.

### F. From one simulator leaderboard to a model–simulator–real triangulation

Use Symtropy for controlled hidden truth, WorldOracle-style paired real execution for grounding, and independent physical constraints/metrology for validation. Agreement among two learned models is not independent ground truth.

## 8. Immediate implementation queue

1. Finish the action-bank canonical digest and Rust tests in [PR #9573](https://github.com/Luminous-Dynamics/symthaea/pull/9573).
2. Use the fail-closed unique-dominance selector and selected-action evaluation metrics; preserve abstention outcomes in episode receipts.
3. Add a report aggregation layer for the P0–P6 metric set with episode-level denominators, worst-stratum reporting and paired D0/D1/W1/W2/W3 contrasts.
4. Extend the Symtropy adapter with frozen-history branch IDs, canonical action manifests, independent outcome receipts and common-disturbance tape identity.
5. Connect CMAQ-012 profile/receipt content verification to an independent runner/recorder attestation. Hash integrity is not hardware provenance.
6. Execute real matched runs only after the above is independently reproducible; then benchmark against the named external suites.

## 9. Claim ceiling

This document is a research protocol. It does not assert that Symthaea currently outperforms any published model, has calibrated action-conditioned visual forecasts, or is safe for physical action. The new Rust action-bank API validates payload structure and content digest; the selected-action evaluator measures decision-relevant metrics on supplied intervals/outcomes. Real predictor quality, calibration, coupling authenticity, domain transfer and downstream advantage remain empirical questions.


### Executable net-decision-value evaluator

The current Rust API now includes `CostedDecisionOutcome`, `DecisionValueEpisode` and `evaluate_net_decision_value`. For each matched episode, the evaluator subtracts inference, acquisition and failure/recovery costs from realized utility for the world-model arm, D0 and D1. It then reports paired mean gross and net advantages and net win rates versus both direct baselines. Negative costs and non-finite values are rejected.

The synthetic known-answer fixture illustrates why gross reward is not enough: mean gross advantage is +2.333 versus D0, but after costs the advantage is only +0.333; versus D1 the net advantage is +0.167. These values only test arithmetic and do not represent a measured policy. A real advantage claim still requires enough independent episodes, paired uncertainty intervals and equal-budget accounting.


### VIS-005 production-readiness distinction

The [VIS-005 source audit](VIS005_ACTION_CONDITIONED_GAP_AUDIT_2026-11-01.md) records that existing delayed-horizon visual forecasting is observation-grounded but autonomous, the cross-manifold path is vision-to-cognition, and the current JEPA path uses a generated CfC target. These are important baselines, not action-conditioned visual world-model evidence. P2/P3 claims require a true action-conditioned predictor trained/scored against later observations and same-history alternative branches.


## 10. New frontier: structured belief and action-causal metrics

The action-bank implementation now has a structured-channel sibling. This is important for visual world models because a single scalar task utility can hide a physically wrong rollout: an object can be in the wrong place while a coarse success proxy looks plausible, or the geometry can be right while the predicted contact state is wrong.

Require separate, versioned channels for object geometry/pose, linear and angular motion, contact state/probability, visibility/occlusion, object existence, task utility and terminal risk. Report channel units and sample counts independently. Use same-draw paired effects within each channel; never subtract quantiles across channels or pool unlike dimensions.

A planner may create a scalar utility only through a frozen utility/constraint definition with declared weights, units and failure penalties. That scalar is a downstream decision functional, not a replacement for the full physical-belief diagnostics.
