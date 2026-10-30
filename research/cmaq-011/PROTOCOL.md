# CMAQ-011: Decision-Centric Benchmark Bridge

**Status:** protocol proposal v0.1; no benchmark result claimed  
**Tracking issue:** [#9094](https://github.com/Luminous-Dynamics/symthaea/issues/9094)

## Thesis

A world model is useful only insofar as it improves decisions under a fixed resource budget. Prediction error remains a diagnostic metric, not the score that determines whether a model is better. The benchmark therefore measures the chain:

`state → counterfactual predictions → action ranking → selected action → closed-loop outcome`.

Every reported result must identify which link it measures. Do not pool results across incompatible tracks.

## External reference points

- **MRS.Q (ICML 2026):** an important model-based RL/search baseline. Its paper reports that search can hurt despite high model accuracy and that pessimistic ensemble value aggregation is critical. Its public repo uses Gym, DMC, HumanoidBench, Atari and visual-DMC prefixes. It declares CC BY-NC 4.0; benchmark it as an external dependency and preserve its license, rather than copying its implementation into Symthaea. The source is pinned at commit `b12abaf309833f28619dfd87f4251074abed3ae3` in [`mrsq-baseline.lock.json`](mrsq-baseline.lock.json).
- **MRS.Q upstream reference results:** the pinned commit's checked-in logs contain 10 seeds for Gym-HalfCheetah-v4 at 1,000,000 training steps (mean return **14,649.00**, sample SD **876.97**) and Dmc-quadruped-walk at 500,000 steps (mean return **935.47**, sample SD **79.45**). The lockfile stores all seed values, upstream defaults and exact result files. These are *author-supplied reference artifacts*, not local reproductions by Symthaea. Reward scales are task-specific and must not be compared across environments; reproduce wrappers, reward transforms, physics versions, seeds and evaluation settings before matched-performance claims.
- **World-In-World (ICLR 2026 Oral):** a separate closed-loop visual/embodied track covering Active Recognition, Active Embodied QA, Image-Goal Navigation and Robotic Manipulation. Its standardized online planning/action interface emphasizes task success and controllability, not visual realism alone. Adapter readiness and exact upstream code/license must still be verified before reproduction.
- **WorldArena 2.0 (2026 benchmark protocol):** the published [WorldArena 2.0 page](https://world-arena.ai/) extends evaluation across vision-to-visuotactile reasoning, offline metrics to online RL rollouts, and simulation to real-robot deployment. It names RoboTwin 2.0, LIBERO and an AgileX split-type ALOHA platform. This is a high-value manipulation/contact track. However, the linked [WorldArena GitHub repository](https://github.com/tsinghua-fib-lab/WorldArena) has latest inspected commit `2da2ae253b8637ba9de3afc7bea4e087f778ee4d` (2026-07-22), and the GitHub API did not identify a license. We have not established that the published 2.0 protocol, its full code, datasets and reuse terms align. Treat the website as a protocol reference only until a versioned release and reuse terms are confirmed.
- **LIBERO via LeRobot:** current LeRobot docs list five suites and 130 tasks: Spatial, Object, Goal, LIBERO-90 and Long. Linux and MuJoCo are required; the documented install path uses the `libero` extra and EGL for headless rendering. This provides a concrete manipulation adapter, but evaluation must use policy-visible observations and avoid privileged simulator state.
- **World-model benchmark surveys:** the 2018–2026 survey reports 102 representative world-model benchmarks, with 13 closed-loop protocols and 14 downstream-outcome metric tracks. A separate robotics benchmark survey posted 30 August 2026 lists 160 benchmarks, 11 explicit VLA/world-model contrasts and four prediction-to-action bridges. These counts are survey-scoped, not universal censuses.

## Tracks

### A. Vector control (first implementation target)

Start with continuous-control tasks for which state snapshots and counterfactual simulation can be made reproducible.

| Order | Environment | Role |
|---|---|---|
| P0 | Gymnasium/MuJoCo, HalfCheetah-v4 if the pinned MRS.Q adapter supports it | Search and model-based RL comparison |
| P0 | DeepMind Control, quadruped-walk | Cross-suite continuous-control transfer |
| P1 | HumanoidBench task family | Higher-dimensional contact-rich control |
| P2 | Additional held-out Gym/DMC tasks | Generalization, not leaderboard tuning |

The first two tasks are the minimum useful cross-suite slice. Exact task IDs, package versions, physics versions, wrappers, reward transforms, observation normalization and reset semantics belong in a lockfile generated from the environment adapter—not inferred from a task name.

### B. Manipulation / embodied control (separate scorecard)

Use a staged path rather than trying to reproduce a large vision-model leaderboard immediately:

1. **LIBERO-Spatial and LIBERO-Long** as the first manipulation slices. The current LeRobot docs list 10 tasks per suite. Freeze policy-visible camera/state inputs, language instruction, action representation, reward/termination, reset and physics versions. The Long suite tests multi-step dependencies; Spatial isolates geometric reasoning.
2. **RoboTwin 2.0 / WorldArena 2.0** as the contact-rich extension. Add visuo-tactile input only after the same D0/D1/W1/W2/W3 protocol works without tactile input. Separate slip, contact, force and object-interaction strata.
3. **World-In-World** as a separate visual closed-loop suite: Active Recognition, Active Embodied QA, Image-Goal Navigation and Robotic Manipulation.
4. **AgileX ALOHA sim-to-real** only after simulation results are independently reproducible and a robot-specific safety case, action limits and emergency-stop procedure exist.

Do not pool image/video, visuo-tactile, vector-state and physical-robot scores. Where Symthaea cannot supply a compatible observation/action interface, report adapter infeasibility rather than manufacture a proxy score. Do not use privileged simulator state as input to a deployed policy; it is allowed only for oracle scoring and clearly labelled upper bounds.

## Model and policy arms

At minimum, evaluate where the environment adapter permits:

1. No-world-model direct policy.
2. Random or fixed-action smoke control (sanity baseline only).
3. MPC/CEM with the same proposal candidates and simulator-query budget.
4. MRS.Q from a pinned external revision and its published settings.
5. DR.Q as a strong representation/off-policy control baseline; report its training budget and clarify that it does not test predictive planning at inference.
6. Pessimistic ensemble-value search (minimum over value heads), with ensemble size and training budget stated.
7. Symthaea one-step / short-horizon model.
8. Symthaea with uncertainty-aware or integrity-gated planning.
9. Oracle environment-model upper bound (diagnostic ceiling, never a deployable baseline).

For predictive-advantage claims, also run the matched D0/D1/W1/W2/W3 arms specified in `PREDICTIVE_ADVANTAGE.md`. A published algorithmic baseline such as DR.Q or MRS.Q does not replace the within-task matched direct-policy control.

Do not treat paper-reported MRS.Q scores as a reproduced baseline. A row is “reproduced” only when the exact configuration is run in this harness and the artifact is retained.

## Predictive advantage: mandatory matched ablation

CMAQ-011 is not merely a world-model leaderboard. Its primary scientific question is whether prediction changes task success relative to a strong direct policy when the proposal policy, task, data, action budget and compute are controlled.

The specialized robotics survey *Do World Models Make Better Robots?* (arXiv:2609.29669, 30 August 2026) catalogues 160 web-verified benchmarks and reports only 11 explicit direct-VLA/world-model comparisons and four prediction-to-action bridges. This is a survey-specific audit, not a universal census. It motivates making the prediction contribution itself the primary estimand.

For each capability and task, create paired runs from the same initial states and seed manifest with these arms:

| Arm | Policy / predictive access | Purpose |
|---|---|---|
| D0 direct | Frozen direct policy; no future predictions | Strong no-world-model baseline |
| D1 compute-matched direct | Same direct policy family with the same inference compute/latency budget allocated to policy-side refinement or ensembling | Controls for extra test-time compute |
| W1 learned WM | Same frozen proposal policy; action ranking may use learned predicted futures | Candidate world-model system |
| W2 shuffled-future control | Same W1 pipeline, but future predictions are permuted across candidate action sequences within matched state/task strata | Tests whether correctly action-conditioned prediction mediates the gain |
| W3 oracle dynamics | Same proposer and planner with ground-truth environment transitions or exact simulator queries | Upper bound on available planning value; not deployable |

Freeze the proposal policy and candidate sequence set across D0/D1/W1/W2/W3. The direct arms must not receive hidden candidate-specific future information. The shuffled-future control must preserve marginal prediction quality while breaking action-to-future correspondence; document exactly how strata and permutations are formed. The oracle arm must use the same candidate count and rollout horizon.

For paired episode i and capability c:

`AoP(c) = mean_i[U_W1(i,c) - U_D1(i,c)]`

where AoP is **Advantage of Prediction** and D1 is the compute-matched direct policy. Also report the more deployment-realistic gain over D0, but do not use it alone to claim prediction added value. Use paired bootstrap intervals and per-task distributions.

Mandatory mediation checks:

- `W1 - W2`: whether action-conditioned predicted futures improve decisions beyond matching marginal prediction statistics;
- `W1 - D1`: whether the world model beats the compute-matched direct policy;
- `W3 - W1`: remaining gap to perfect dynamics under the chosen proposer/planner;
- `D1 - D0`: benefit from extra inference-time compute without prediction.

If W1 beats D0 but not D1, the gain may be extra compute, not prediction. If W1 and W2 are indistinguishable, the system may not be using action-conditioned predictions. If W3 is weak, the bottleneck is likely proposal/planning rather than predictive fidelity.

Report four advantage-aware quantities:

1. **Closed-loop prediction advantage:** AoP per capability, with paired confidence interval.
2. **Counterfactual competence:** success/regret on episodes requiring inference about unobserved or hypothetical dynamics, with occlusion, hidden-state and memory slices.
3. **Prediction-to-action retention:** gap between open-loop prediction scores and closed-loop success on the same task/data split.
4. **Capability coverage:** fraction of preregistered capabilities with a valid D1-versus-W1 head-to-head; missing or incompatible arms count as not covered, not zero performance.

Also publish inference FLOPs, p50/p95 latency, memory, candidate count, model queries, simulator queries and energy only when measured. Use a cost-performance Pareto frontier rather than a single scalar. Capabilities must remain separate: manipulation, navigation, contact reasoning, occlusion inference and delayed-effect reasoning should not be averaged into one headline number without their constituent results.

## Four evaluation levels

### L1. Predictive fidelity (diagnostic)

Report normalized state RMSE/MAE, reward MAE, termination prediction, physical invariant violations and calibration. Use the same input history and held-out transition tape for all methods.

### L2. Counterfactual decision fidelity (core)

From identical saved simulator states, score the same candidate action sequences. Report Spearman and Kendall rank correlation, top-1 agreement, top-k oracle regret, selected-action regret and action-ranking calibration. The oracle action is defined by actual simulator rollouts under a declared horizon and discount, not by a model ensemble.

The key measure is:

`regret(s) = Q_env(s, a*_env) - Q_env(s, a_selected)`

For stochastic tasks, estimate each action's value with paired simulator seeds and report uncertainty. Never use the model's own value as its ground truth.

### L3. Closed-loop utility (deployment-facing)

Report task success, undiscounted return, median, mean, lower-tail return, CVaR95 loss, failure/catastrophe rate and recovery after perturbation. Use multiple episodes and paired seed manifests. Report both the fixed policy and model-planner policy: this distinguishes model quality from proposer/controller quality.

### L4. Exploitability and robustness (CMAQ differentiator)

Sweep planning search count, rollout horizon and controlled model-error growth. Report:
- unsupported-action/invalid-state selection rate;
- optimism gap between predicted and environment-realized value;
- decision-ranking inversion rate;
- performance change versus the no-search/direct-policy baseline;
- OOD fallback and probe rates;
- false revocations and missed hazardous shifts;
- worst-case and CVaR95 regret.

A strong claim must show that additional search does not silently exploit model error. Publish the complete budget-performance frontier rather than a single best number.

## Matched-budget rules

Every model/policy pair must share:
- the exact task, reward, wrapper, reset distribution, episode horizon and seed manifest;
- the same candidate proposal set and action-sequence horizon;
- the same maximum environment/model queries per decision;
- a declared inference-time compute budget, wall-clock budget and memory ceiling;
- the same observation history and training data access.

Where exact equality is not possible, publish both the unconstrained result and a normalized cost-performance frontier; never hide the mismatch. Separate training compute from evaluation compute.

Record query count, simulator calls, probe/intervention count, model forward passes, wall-clock p50/p95/p99, peak memory, device/runtime and estimated energy if measured. Do not compare “steps” when one system performs many hidden substeps or parallel branches.

## Splits and statistics

- Train/development/qualification environments must be disjoint by task parameter family, not merely random seeds.
- Reserve a final hidden qualification set before tuning thresholds or prompts.
- Use at least 20 evaluation seeds for smoke; use 100+ paired seeds for a confirmatory comparison unless power analysis justifies another number.
- Report mean and median, bootstrap or paired confidence intervals, CVaR95 and per-task values. Do not average away a failing task.
- Register primary hypotheses and thresholds before running the qualification set.
- Preserve failed, timed-out and resource-exhausted runs in the ledger.

## Required claim ladder

- **Implemented:** adapter and schema checks pass.
- **Smoke-tested:** a small deterministic set runs end to end.
- **Reproduced:** external baseline and reference settings run from pinned source/config.
- **Matched:** candidate and baseline share the required budgets and splits.
- **Competitive:** statistically non-inferior or better on the pre-registered primary outcomes.
- **SOTA:** only when comparing against all relevant current published baselines on the same task/protocol/split and matching compute; cite exact papers and release artifacts.

Never promote an implementation or smoke result to competitive/SOTA.

## Provenance and artifact binding

Every row must bind:
- source commit and clean-tree status;
- model checkpoint digest;
- environment source/release and physics-engine version;
- wrapper/reward/normalization configuration;
- task split and seed-manifest digest;
- policy/proposer implementation and version;
- compute/runtime/hardware budget;
- raw episode records and aggregation code;
- validator version and final report digest.

A result without complete provenance is exploratory, not leaderboard eligible. An independent rerun should reproduce aggregate metrics within declared tolerance.

## Implementation sequence

1. Land the canonical JSON schemas and validator contract.
2. Reproduce the pinned MRS.Q reference artifacts for Gym-HalfCheetah-v4 and Dmc-quadruped-walk before making any matched-performance claim; then build the vector-control pair with pinned wrappers, physics, rewards and seed manifests.
3. Add no-model, compute-matched direct, MPC/CEM, MRS.Q, TD-MPC2/TD7 controls and Symthaea D0/D1/W1/W2/W3 variants where the task APIs support them.
4. Produce paired L1–L4 and advantage-of-prediction reports under fixed query/compute budgets.
5. Build LIBERO-Spatial/Long adapters as a separate manipulation track; keep all observation interfaces policy-visible.
6. Assess WorldArena 2.0's versioned code/data/license correspondence before treating it as a reproduced baseline. Add contact/tactile and ALOHA only as independently validated extensions.
