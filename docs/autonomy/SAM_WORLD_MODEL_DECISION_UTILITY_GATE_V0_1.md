# SAM World-Model Decision-Utility Promotion Gate v0.1

Status: **specification only; no model is promoted by this document**.  
Primary owner: [CMAQ-004 / #9010](https://github.com/Luminous-Dynamics/symthaea/issues/9010).  
Road-autonomy composition: #10237, #10252, #10256, #10257, #10266, #10318, #10320, #10356, #10372, #10012, #6256 and #11258.  
Execution and reciprocal interaction: Symtropy #3387, #3471 and #3532.

## 1. The paradigm shift

A world model should not be promoted into a driving decision path because it generates attractive video, predicts trajectories with a good average error, or reduces a hidden-state entropy proxy.

The relevant question is:

**Does access to this exact model improve decisions enough to justify its compute, latency, memory, energy and failure-mode costs, under matched evidence privilege and a held-out interaction distribution?**

Keep three different claims distinct:

1. **Fidelity:** does the model represent or predict the scene and its dynamics accurately enough for the declared task?
2. **Controllability:** do its predictions respond correctly to the candidate action and causal interaction history?
3. **Decision value:** does using the model improve task decisions over a matched direct policy after accounting for query and inference cost?

A model can do well at (1) and poorly at (2) or (3). Such a model may remain useful for perception research, synthetic data or visualization, but it is not thereby a better planner.

## 2. Candidate arms

Every evaluation generation freezes the policy, model, memory, evaluator/reference portfolio, task/action set, scene history, scenario corpus and compute budget.

| Arm | Candidate behavior | Purpose |
|---|---|---|
| A — direct reactive | Select actions from observations without querying a learned transition/reward model | Decision baseline |
| B — direct + same memory | Same direct policy and the same persistent memory, without predictive model queries | Isolate memory contribution |
| C — model-assisted | Policy may query the candidate world model for action-conditioned rollouts | Candidate under test |
| D — shuffled-dynamics control | Same model interface/representation with action-transition mapping deliberately shuffled or causally broken | Detect reliance on plausible-but-wrong dynamics |
| E — stale-model control | Preserve an older model through a hidden dynamics/actor-policy regime shift | Test stale-model exploitation |
| F — qualified oracle | Evaluator-only reference / analytic upper bound, inaccessible to policy | Diagnostic only; never leaderboard eligible |
| G — distilled student | Smaller policy distilled from a model-assisted teacher, with separate training and observation privileges | Edge deployment / teacher-student efficiency comparison |

A teacher model's use for training is not the same as its availability to the deployed student. Declare observation, action blending, supervision, optimizer/replay, evaluator and inference-state privilege separately under #11258.

## 3. Evaluation layers

### Layer 1 — visual and scene fidelity

Use DrivingGen only for the capabilities it measures:
- visual distribution / image quality;
- temporal scene consistency;
- agent appearance, persistence and disappearance consistency;
- trajectory plausibility;
- ego-conditioning / controllability.

DrivingGen's open-domain and ego-conditioned tracks are not closed-loop task-success evidence. Its paper reports 400 evaluation clips (200 per track), while the current hosted dataset card reports a different evaluation example count. Pin dataset revision, split and sample IDs before reproduction. Video metrics such as FVD or CLIP-IQA are never safety surrogates.

### Layer 2 — metric and temporal prediction

Use #10256 and #10257:
- trajectory error by horizon, including ADE/FDE and miss rate where benchmark-defined;
- prediction-set coverage, sharpness/set measure and worst-family coverage;
- signed/absolute model residual by horizon;
- calibration and data-shift behavior;
- rollout discrepancy and planner optimism.

A larger uncertainty set may improve coverage while making planning useless. Report the coverage–sharpness frontier and downstream action effects separately.

### Layer 3 — decision utility

Use #10237 / #9010:
- paired task-success / completion delta versus the direct baseline;
- selected-action regret and better-feasible-alternative frequency;
- action-ranking reversal rate across qualified references;
- hard-constraint disposition, false-safe and false-intervention outcomes;
- intervention/abstention and no-admissible-action outcomes;
- model query count, latency, compute, memory and energy.

The model-assisted arm must pay for the model query and rollout costs. Report both matched-budget and best-attainable-budget frontiers.

### Layer 4 — interaction and closed loop

Keep protocol classes separate:
- **WOD-E2E:** open-loop rater-preference trajectory evaluation on long-tail logged scenes; not interactive future feedback.
- **NAVSIM v2:** two-stage pseudo-simulation with reactive-agent support, but the official README states it is not sequential or interactive.
- **ReactSim-Bench:** surrounding-agent response to pre-generated ego deviations; one-sided reactive evaluation, not reciprocal ego replanning.
- **nuPlan-R:** online ego planning with top-k selected reactive diffusion agents and log-replay background actors; selective reciprocal interaction.
- **Symtropy #3532:** bounded reciprocal policy-in-loop execution for the declared interactive actor subset, with independent observations, action latching, causal delivery and replay.

World-In-World (ICLR 2026) is a valuable *methodological* comparator for closed-loop world-model utility, but its tasks are active recognition, embodied QA, navigation and manipulation—not an automotive leaderboard. Never import its task scores into an AV benchmark.

## 4. Matched-experiment protocol

Each episode must bind:
- scenario/family/actor-policy IDs and split lineage;
- exact initial state, scene history, map, road/dynamics and actuator profiles;
- model/policy/memory/evaluator/reference artifact digests;
- complete candidate-action set and action payload digests;
- observation profile, sensor privilege and source-role manifest;
- teacher action/blend/supervision and evaluator-only truth timing;
- model query count, query latency, rollout horizon and resource budget;
- exogenous disturbance stream identity and whether paired common-random-number alignment is causally justified;
- exact execution and replay receipt.

Commit the action before exposing counterfactual/evaluator outcomes to the candidate. Freeze the subject before the sealed holdout. Scenario frames and sibling variants are not independent experimental episodes.

## 5. Promotion gate

A candidate may be promoted into a declared planner profile only when all of the following are satisfied:

1. **Prediction/fidelity:** task-relevant accuracy and uncertainty/coverage evidence meets the declared context of use.
2. **Decision lift:** paired task utility improves or regret decreases relative to the direct policy under matched compute; the effect and interval are reported over scenario-family clusters.
3. **Controllability:** shuffled-dynamics and stale-model controls expose expected failure; candidate predictions respond to changed actions only through the declared causal channel.
4. **Reference robustness:** disagreements across qualified references remain explicit; action ranking is stable or the result is marked reference-sensitive/indeterminate.
5. **Interaction robustness:** improvement persists on an independent interaction protocol, not only an open-loop log or pseudo-simulation.
6. **Safety constraints:** no hidden regression in hard constraint violations, false-safe outcomes or adequately sampled worst-family collision/road-boundary metrics; RTA remains independent.
7. **Compute:** p50/p95/p99 latency, deadline misses, memory, energy and query cost meet the deployment profile.
8. **Information integrity:** no evaluator-to-policy leakage, no undeclared teacher supervision, and no mislabeling of simulator-oracle input as local perception.

If fidelity improves but decision lift is zero or negative, retain the model as a research/perception/simulation asset but do not promote it into the planner. If average utility improves while an adequately sampled worst-family constraint regresses, promotion is blocked pending a targeted remedy and independent holdout.

## 6. Prospective target design

Targets are preregistered per selected benchmark and are not achieved results:

- First reproduce the direct baseline and strongest matched model-assisted baseline exactly.
- Then target at least +2 percentage points of closed-loop success/CLS or statistically supported regret reduction on a sealed actor/failure-family slice, while maintaining All-Core pass and worst-family safety-surrogate performance.
- Require prediction-set coverage and sharpness not to regress beyond the declared tolerance.
- Require model-query costs to be included; report matched-compute and best-attainable-compute tracks.
- Report a Pareto frontier over decision lift, worst-family outcome, latency, memory and energy rather than one weighted score.
- A result on DrivingGen alone cannot pass the decision-utility gate; a result on World-In-World alone cannot establish driving performance.

## 7. Claim ceiling

This gate determines whether a specified model contributes measurable decision value for a specified task, benchmark profile and artifact generation. It does not prove the model spans reality, establish human behavioral fidelity, establish physical sensor validity, prove public-road safety, or qualify SAE Level 4/5 autonomy.
