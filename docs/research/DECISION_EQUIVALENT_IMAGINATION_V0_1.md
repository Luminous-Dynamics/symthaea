# Decision-Equivalent Imagination and Anytime Qualification v0.1

**Status:** Research specification; prospective, not an achieved Symthaea capability  
**Created:** 2026-11-24  
**Parent:** Symthaea #13445 — ROB-DECISION-SUFFICIENCY-001  
**Benchmark integration:** #13389  
**Scope:** Decision-value of world-model inference/rollout budgets; not a new world model, policy, simulator, minimax-regret oracle, calibration method, scheduler or action authority.

## Executive thesis

A robot does not need to reconstruct every detail of every possible future. It needs enough predictive evidence to select an admissible action whose regret remains within a declared task tolerance, under the model's qualified scope and before the action deadline.

The research hypothesis is:

> The most useful world model is not necessarily the one that predicts the most. It is the one that identifies when its remaining uncertainty is decision-irrelevant, when more prediction could change the decision, and when no amount of additional computation available before the deadline can justify acting.

This turns imagination into an **anytime, decision-qualified resource**. Prediction may be expanded in bounded increments, but a larger search budget does not inherit trust from a smaller one: selection pressure can expose optimizer-exploitable model errors.

The research programme must separately measure:
1. **Predictive fidelity:** whether predicted futures match the declared reference outcomes.
2. **Decision sufficiency:** whether the available predictive evidence supports a sufficiently good common action.
3. **Execution validity:** whether that action is independently admissible and executable.
4. **Net value:** whether using the model improves the actual decision after inference, observation, delay, energy and failure costs.

A pass on one axis never implies a pass on another.

## Why now: external evidence and its limits

### Planning-range limits

Alrasheed, Azam and Akhtar, *The Planning Limits of Latent World Models* (30 September 2026), evaluate action-conditioned predictors over Meta-World and real-robot BridgeData V2 interactions. Their five-step rollout models rank actions reliably mainly for targets around 5–10 control steps ahead, while task goals span 16–53 steps. An 81-fold larger predictor and longer-rollout training did not extend that measured range. Their experiments also report that nearby expert subgoals and feedback can outperform pure imagination on distant-goal tasks. The paper reports a VLA action-reranking comparison in which selecting among eight VLA-proposed actions improves success from 65% to 77% across 16 tasks.

Source: https://arxiv.org/abs/2609.39235

These findings motivate range-aware subgoals, decision-focused candidate reranking and explicit feedback. They are not Symthaea results and do not establish that the same values transfer to different robots or tasks.

### Decision-centric evaluation gap

Yu et al., *How Should World Models Be Evaluated? A Decision-Making-Centric Position* (13 June 2026), argues that a world-model claim should be evaluated according to the downstream use it purports to support, including counterfactual action fidelity, closed-loop validity, policy ranking, optimization lift, exploitability and uncertainty calibration.

Source: https://arxiv.org/abs/2606.15032

Jena et al., *Do World Models Make Better Robots?* (30 August 2026), audits 160 benchmarks and reports that 11 explicitly compare VLA and world-model systems, while four carry predictions through to executed action. These are the survey's corpus-specific counts, not a universal census.

Source: https://arxiv.org/abs/2609.29669

### Embodied execution and reliability diagnostics

- MiraBench defines physics adherence, action-following fidelity and optimism-bias diagnostics, supported by over 16,000 human judgments. Source: https://arxiv.org/abs/2605.29360
- RoboTrustBench provides 1,207 expert-validated instruction–image pairs across normal, constraint-sensitive, counterfactual and adversarial cases. Source: https://aclanthology.org/2026.findings-emnlp.1462/
- RoboWM-Bench converts generated manipulation behaviour into embodied action sequences and checks execution in reconstructed simulation; it highlights spatial, contact and non-physical deformation errors. Source: https://arxiv.org/abs/2604.19092
- H2RBench evaluates real-to-sim human-to-robot transfer across four manipulation tasks. It reports Pearson r=0.89, Spearman rho=0.85 and MMRV=0.06 for one robot-plus-maximum-human-data setting across 16 method-task pairs. These are rank-agreement diagnostics, not universal physical-fidelity guarantees. Source: https://arxiv.org/abs/2609.24778

### Asynchronous world-action models

Streaming-WAM reports 98.35% on standard LIBERO, 91.74% on RoboTwin 2.0 and 75.35% on RoboCasa, with runtime gains in its respective protocols. The 98.35% figure is **not** a LIBERO-Dynamic result.

Sources:
- https://arxiv.org/abs/2609.28927
- https://sjtu-deng-lab.github.io/Streaming-WAM/

AHA-WAM reports 92.80% on RoboTwin and 78.3% across its four real-world tasks, under its own setup:
https://arxiv.org/abs/2606.09811

TempoWAM reports progress-triggered inference reductions on easier tasks and improved success on difficult real-robot tasks:
https://arxiv.org/abs/2608.09492

These systems are complementary comparators, not one directly rankable leaderboard. Preserve each benchmark's native protocol, hardware, checkpoint, task population and timing rules.

## Decision-equivalence semantics

Let a frozen decision context contain:
- a declared finite action set A;
- an evaluator-supported world set W for the exact task/episode scope;
- a hard admissibility predicate Adm(w,a);
- a task utility U(w,a), with higher values preferred;
- a preregistered tolerance epsilon in the declared utility scale.

For each evaluator world w, define the admissible action set:

`A_adm(w) = { a in A : Adm(w,a) = true }`

For non-empty A_adm(w), define the world-specific optimum:

`U*(w) = max_{a in A_adm(w)} U(w,a)`

and the epsilon-optimal action set:

`A_epsilon(w) = { a in A_adm(w) : U*(w) - U(w,a) <= epsilon }`

The common epsilon-optimal set across the supported worlds is:

`A_common_epsilon(W) = intersection over w in W of A_epsilon(w)`

Interpretation:

- If A_common_epsilon(W) is non-empty, at least one common action is epsilon-optimal in every declared supported world.
- If it is empty, no action in the declared action set is epsilon-optimal across all supported worlds. The evidence is decision-sensitive under this finite scope.
- If a hard-admissibility set differs across supported worlds, preserve that fact explicitly. Do not average a constraint violation into soft utility.
- An incomplete, unsupported or unqualified world/action set cannot produce a positive sufficiency certificate.

This finite-set definition is a measurement contract, not a global optimality theorem. It depends on completeness and validity of the declared worlds, actions, utility and constraint semantics.

### Distinguish three notions

1. **World equivalence:** two world hypotheses imply the same decision-relevant admissibility and epsilon-optimal action sets for the declared decision scope.
2. **Representation sufficiency:** a representation does not merge full-state information classes in a way that destroys the required action distinctions across relevant future decision epochs.
3. **Model-budget sufficiency:** the exact rollout/query/search budget gives enough evidence to qualify a decision within the declared regret and time bounds.

Do not conflate these. Representation compression is owned by #13239/#13255. Calibrated selected-action sufficiency over contiguous future ticks is owned by #11646/#11655. This specification focuses on world-model inference/rollout budget and composes those owners.

## Anytime imagination protocol

A candidate may request computation in bounded stages, for example:
- add a short action-conditioned rollout;
- evaluate another candidate action;
- add a counterfactual branch;
- refine a task-relevant latent state;
- evaluate another independently qualified dynamics profile;
- request a new observation or reference computation.

At every stage, record the exact additional work and its output identity. Do not equate configured capacity with completed work or completed work with useful work.

Candidate-visible stopping logic may use only current observations, admitted evidence, its own model outputs and declared resource/deadline state. Hidden evaluator outcomes, future observations, labels and oracle utilities remain evaluator-only.

A candidate may propose to stop and act when:
1. its own admitted support indicates a common epsilon-good action under the frozen decision rule;
2. the selected-action adequacy/calibration profile is current for the exact model, policy, task family, horizon and shift scope;
3. the expected action-application time remains inside evidence freshness and deadline limits;
4. independent action-authority and execution-envelope checks pass;
5. the estimated marginal value of another compute/observation increment does not justify its cost under the preregistered stopping policy.

The evaluator independently checks support coverage, realised regret, false-sufficiency, omitted worlds/actions and timing. The candidate's stopping declaration is not its own qualification receipt.

A new observation, environment epoch, dynamics shift, model artifact, action set or relevant authority change revokes the prior decision certificate and requires re-evaluation.

### Non-monotonicity rule

More compute, candidates, rollout depth or branches are not presumed to improve decisions. Each expanded search profile is a distinct selected-action distribution and must be requalified. Include optimizer-exploitability controls for narrow optimism pockets and selection-induced distribution shift.

## Keep resource and horizon variables separate

| Symbol | Meaning |
|---|---|
| K | World-model rollout horizon |
| L | Target/subgoal lookahead |
| H | Planner or policy-tree lookahead |
| H_train | Recursive training horizon |
| A | Physical action prefix committed before feedback |
| R | Observation refresh/delivery interval |
| N | Candidate action count |
| B | Branch count |
| M | Prediction sample count |
| F | Measured model forward passes or declared compute-work units |
| D | Representation size/dimension |
| T | Wall-clock latency, including the declared pipeline boundary |
| E | Measured energy, only when instrumented and calibrated |
| S | Peak and retained memory under a declared measurement boundary |

N, B, M, F, D, T, E and S are resource/profile coordinates, not interchangeable units. Do not collapse them into a scalar cost without an explicit, preregistered application-specific pricing model.

K, L, H, H_train, A and R are different temporal/control variables. In particular:
- Increasing K does not establish support for a larger L.
- Increasing H does not extend the empirically supported model range P*(K).
- Increasing K or H does not authorize a longer A.
- Changing R changes what evidence is available at the decision time.
- A long imagined future must not be replayed as a shared physical trajectory after closed-loop actions diverge.

## Experimental design

### Reuse existing policy/evaluator arms

On compatible robotics tasks reuse the existing CMAQ-012 / #13389 arms:
- D0: direct policy without predictive rollouts;
- D1: compute-matched direct/value controller;
- W1: learned action-conditioned world-model policy;
- W2: shuffled/stale-future negative control;
- W3: evaluator-only oracle-dynamics diagnostic.

Sweep W1 across preregistered model-budget profiles while freezing the backbone, action proposer, task, action semantics, checkpoint, relevant runtime and initial/exogenous world commitment.

Do not introduce a new evaluator, bootstrap implementation or second five-arm taxonomy.

### Budget profiles

Use a boundary-focused/fractional-factorial design rather than a full Cartesian product. At minimum vary:
- K: short / boundary / extended;
- L: below / at / beyond measured P*(K);
- N: small / medium / large candidate set;
- B or M: minimal / intermediate / expanded where supported;
- stable / shifted dynamics;
- early / late / unavailable observations;
- fixed / adaptive search;
- short / long physical action prefix A, independently of K and H.

Record exact executed work, not only requested budgets. If a candidate does not support an axis, mark it unavailable rather than silently substituting another method.

### Required scenario families

1. **Decision-invariant uncertainty:** worlds differ in prediction detail but share an epsilon-good admissible action.
2. **Decision-boundary ambiguity:** the preferred or feasible action changes across supported worlds.
3. **Hard-constraint conflict:** a high predicted-reward action is infeasible in at least one relevant supported world.
4. **Beyond-range goal:** L exceeds empirically qualified P*(K).
5. **Contact and interaction:** small friction, geometry, timing or contact changes flip a decision.
6. **Failure preservation:** failed grasp, lost support, occlusion, collision, deformation or impossible contact must not become imagined success.
7. **Optimizer exploitability:** adaptive search discovers local optimism pockets while global/behaviour-distribution MSE remains similar.
8. **Moving-world latency:** the environment evolves during model inference; score at actual application time.
9. **Future-decision representation collision:** a representation preserves the current choice but destroys a distinction needed after a later observation or contact/phase transition.
10. **Unnecessary-imagination control:** additional computation cannot change action feasibility or the epsilon-good action set.

### Baselines and ablations

- D0/D1/W1/W2/W3, from the existing CMAQ evaluator;
- official native timing baselines, including synchronous and asynchronous execution where supported;
- fixed-horizon and progress-triggered replanning, including TempoWAM-style controls where compatible;
- nearest-qualified-subgoal plus feedback/MPC;
- pessimistic ensemble aggregation, as a separate risk-control baseline;
- selection-conditioned calibration / risk-sensitive selection, using a faithful pinned comparator where available;
- maximum-budget W1, as a comparator but not ground truth;
- PCA, random projection, fixed quantization and full-state controls for the separate recurrent-compression lane;
- evaluator-only finite oracle, never candidate-visible or deployable.

MRS.Q (ICML 2026) is a useful vector-control/search comparator. Its study finds that search can hurt even with accurate models and that pessimistic value aggregation can matter more than improving mean prediction accuracy. Respect its CC BY-NC 4.0 code licence; do not vendor or relicense it.
https://proceedings.mlr.press/v306/chang26a.html

## Primary metrics

Keep each measure separate and stratify by task, capability, horizon, model profile, shift and failure family.

### Decision quality
- action-time task utility and regret;
- paired net decision lift against D0/D1;
- common epsilon-good action-set non-emptiness / decision-sufficiency rate;
- action agreement and ranking reversals;
- hard-admissibility set disagreement;
- false-sufficiency and unsupported-action rates;
- harmful predictive intervention rate;
- recovery success, recovery latency and tail loss.

### Prediction and support
- horizon-conditioned residuals and optimism bias (#10257);
- latent rollout consistency and task-relevant state sufficiency;
- selected-action calibration (#9127);
- prediction-set coverage/sharpness (#10256);
- model disagreement and ranking stability (#10252);
- evaluator-measured support coverage and omitted-world/action diagnostics.

### Resource and timing
- actual model calls, rollout/branch/sample counts and compute work;
- p50/p95/p99 end-to-end and stage latency;
- measured energy and memory with explicit boundaries;
- observation count, bytes, freshness and delivery latency;
- action prefix A, refresh interval R, queue underflow and deadline misses;
- compute, observation, latency and joint interaction effects using existing #13281/#13176 owners.

### Robustness and completeness
- worst-family and lower-tail outcomes;
- CVaR or tail-regret measures under a preregistered loss definition;
- false-sufficiency by shift/failure group;
- direct fallback, abstention, no-output and unsupported-scope rates;
- denominator, attrition and missing/invalid evidence counts.

Never allow a scalar aggregate to hide an unsafe action, missing evidence or hard-constraint failure.

## Deterministic semantic fixtures

The examples below are exact arithmetic fixtures for the contract. The reward values are illustrative and evaluator-only; these are not simulator timings or benchmark measurements.

### F1 — Decision-invariant uncertainty

| World | Action A | Action B | Best action |
|---|---:|---:|---|
| W0 | 100 | 70 | A |
| W1 | 80 | 60 | A |

A low-budget profile cannot distinguish W0 from W1 but chooses A. A full-budget profile distinguishes them and also chooses A. Both have zero regret. Additional detail has zero decision value; any compute/latency cost remains charged.

### F2 — Decision-boundary ambiguity

| World | Action A | Action B | Best action |
|---|---:|---:|---|
| W0 | 100 | 70 | A |
| W1 | 20 | 85 | B |

The low-budget profile aliases W0 and W1 and selects A. Its regret is 0 in W0 and 65 in W1, giving mean regret 32.5 for equal weighting. A higher-budget profile distinguishes the worlds and chooses A in W0, B in W1, yielding zero regret. A further profile that makes the same choices but costs more is dominated.

### F3 — Hard-constraint override

The model predicts high reward for action A, but the independent evaluator finds A violates a hard contact/geometry constraint. A must never be admitted based on confidence or reward. The expected outcome is a pre-authorized fallback or BLOCK if no fallback is authorized. This is not a soft cost-regret trade-off.

### F4 — Unsupported target range

The exact evidence profile has P*(K)=10 steps and the target L=20. A longer planner horizon H or high-confidence score cannot expand P*(K). Expected disposition: qualified nearer subgoal, fresh evidence, authorized fallback or UNKNOWN/DEFER.

### F5 — Optimizer exploitation

Two models have matched average error on random/behaviour-policy actions. One contains a narrow optimistic action region that adaptive search discovers. The evaluator must report selected-action error, realised regret and search-induced exploitability separately. Pessimistic or risk-controlled selection may reduce downside, but its coverage and cost must also be reported.

### F6 — Future-decision representation collision

Two full states map to the same compressed representation and support the same current action. A later valid observation or contact transition separates the states and makes different actions necessary. A current-step compression pass must not be reported as a future-decision pass. Reuse #13239/#13255 and #11646/#11655.

## Qualification targets (prospective, not results)

1. 100% exact agreement with an independent brute-force oracle on finite known-answer fixtures.
2. Zero model-use admissions for unsupported L, stale/shifted evidence, incomplete required evidence, invalid hard constraints or missing action authority.
3. No positive decision-value credit for added compute in decision-invariant controls; all cost still reported.
4. Positive held-out paired net decision lift on at least one preregistered family, with a cluster-aware 95% interval excluding zero after costs, before claiming operational benefit.
5. At least one held-out decision-boundary family where added compute/observation improves the decision.
6. At least one family where the direct or short-horizon policy wins; the adaptive system must identify and preserve that outcome.
7. A prospective efficiency target: reduce p95 latency or physically measured inference energy by at least 25% versus maximum-budget W1 while remaining within a preregistered 1 percentage-point task-success noninferiority margin. Claim energy reduction only with calibrated instrumentation.
8. No increase in independent hard-constraint violations; preserve failures, timeouts, no-output and abstentions in the denominator.
9. Demonstrate that expanded search is requalified and does not inherit an earlier profile's trust.
10. Report all resource/decision Pareto points and worst-family results; no single composite leaderboard score.

Targets must be preregistered after a pilot establishes adequate independent episode/cluster counts. They are not current measured results.

## Existing ownership and integration map

| Concern | Existing owner | This document's relationship |
|---|---|---|
| Model decision-use gate / ten evidence dimensions | #9029 / PR #13413 | Consume; do not add another gate |
| Horizon residuals, LRC, P*(K), state probes | #10257 / #10276 | Consume |
| Cross-model action ranking | #10252 | Consume |
| Prediction-set coverage and sharpness | #10256 | Consume |
| Selected-action uncertainty calibration | #9127 | Consume |
| Decision-sensitive observation / measurement | #10320 / #9563 | Request through existing policy |
| Recoverability-conditioned planning depth | #11396 | Keep H separate from K/L/A |
| Physical action prefix / commitment | #11581 | Never infer action authority from K/H |
| Compute while world evolves | #11561 / #13216 | Charge real application-time cost |
| Compute-observation interaction | #13281 / #13176 | Reuse existing paired evaluators |
| Decision-preserving state representation | #13239 / #13255 | Separate D/representation compression from rollout budget |
| Calibrated action-sufficiency horizon | #11646 / #11655 | Reuse; do not duplicate |
| Optimizer-induced model exploitability | #9396 | Include as adversarial lane |
| Causal trajectory pairing | #13375 / PR #13377 | Reuse |
| Robot benchmark adapters and matched arms | #13389 / #9122 | Reuse |
| Evaluator isolation and signed receipts | #9437 | Required for confirmatory claims |
| Statistical cluster analysis | #8578 and existing statistical owner | Reuse exact approved procedure |

### Required interpretation boundary

A candidate-visible decision-stability assertion is not a cryptographic or scientific proof. A hash identifies bytes; it does not authenticate source truth, completeness or physical validity. Evaluator-issued receipts bind a result to a declared process and artifact but do not prove that the model family spans reality. Independent runtime assurance and physical execution authority remain separate.

## Implementation sequence

1. Freeze the versioned model-budget profile in #13445; include K, N, B, M, F and actual measured T/E/S separately.
2. Add the finite semantic cases F1–F6 to the existing decision-value fixtures / test owners. Do not create a second regret oracle.
3. Connect per-profile resource and outcome rows to the existing #13389 adapter and #9122 auditor.
4. Reuse #13239/#13255 to test representation collisions separately from rollout-budget changes.
5. Use #9396 to test adaptive search over localized optimism errors.
6. Reuse #10320 for decision-sensitive observation requests and #11646/#11655 for the calibrated future action-sufficiency horizon.
7. Execute one low-cost numeric-control benchmark slice through the pinned CMAQ-011/MRS.Q bridge before acquiring large visual model assets.
8. Then reproduce one compatible robot manipulation slice; keep standard LIBERO, LIBERO-Dynamic, RoboDyna, RoboTwin and H2RBench as separate native protocols.
9. Require #9437 evaluator isolation and independent rerun before any external or physical safety claim.

## Claim ceiling

This specification does not establish a learned decision-equivalent representation, an adaptive compute policy, a better robot policy, physical safety, general world completeness or SOTA. A document/fixture pass establishes only the finite contract it checks. Benchmark claims require exact source/checkpoint/data/runtime pins, matched native protocols, held-out task/world clusters, full failure accounting, independently controlled evaluation and reproduced end-to-end outcomes.


## Additional November 2026 synthesis: more inference compute can help, and search can still hurt

Two results that appear contradictory are useful together:

- **World-In-World (ICLR 2026 Oral)** reports that allocating more inference-time compute can substantially improve closed-loop performance in its four embodied environments. It also finds that controllability and action-observation post-training matter more than visual quality alone. https://world-in-world.github.io/ and https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html
- **MRS.Q (ICML 2026)** finds that search can reduce performance even when the predictive model is accurate. Its method addresses overestimation using the minimum over an ensemble of value functions; this is a value-aggregation method, not evidence that a collection of dynamics models is independent or physically complete. https://proceedings.mlr.press/v306/chang26a.html

The implication is not "always use less compute" or "always use more compute." It is to allocate extra computation where it changes a decision, while separately controlling the optimism and distribution shift introduced by search.

This suggests a useful experiment not to be collapsed into a static average budget:
- Compare fixed compute per episode with adaptive compute per decision.
- Hold total episode compute (and, separately, wall-clock deadline) matched.
- Report where extra inference improves the chosen action, where it only improves prediction metrics, and where search makes realized decisions worse.
- Compare greedy predicted-value ranking with pessimistic value-ensemble and calibrated/risk-sensitive selection.
- Report compute allocation by task phase and decision-boundary sensitivity, not only episode-wide totals.

The practical architectural principle from *World Models for Robotic Manipulation: A Survey* is to use the cheapest predictive mechanism that addresses the active bottleneck: auxiliary prediction, candidate filtering, short-horizon MPC, or a learned environment should be chosen according to the decision problem and available system budget. https://onlinelibrary.wiley.com/doi/10.1002/smb2.70053

### Benchmark track addition: World-In-World

Add World-In-World as a separate closed-loop visual/embodied track alongside (not pooled with) RoboDyna, LIBERO-Dynamic, RoboTwin and H2RBench. Its four task families are Active Recognition, Active Embodied QA, Image-Goal Navigation and Robotic Manipulation. Preserve its native API, planner, scene/task population, checkpoint, inference budget and metric definitions.

This benchmark is particularly relevant to the adaptive-compute hypothesis because it varies world-model inference compute in closed-loop tasks. It does not replace RoboDyna's dynamic manipulation stress, LIBERO-Dynamic's contact/time/latency protocol, RoboWM-Bench's video-to-executable-action bridge, or H2RBench's sim-to-real transfer analysis.


### Benchmark-provenance note: standard LIBERO success is near saturation

The Streaming-WAM project page's own standard-LIBERO table reports:
- FastWAM-Joint-CD: 98.85%;
- FastWAM-Joint: 98.60%;
- Streaming-WAM: 98.35%.

Streaming-WAM's distinctive result in that table is its asynchronous efficiency (41 ms chunk preparation, 12× faster than FastWAM's 493 ms), not the highest raw standard-LIBERO success percentage. The project separately reports 91.74% on RoboTwin 2.0 and 75.35% on RoboCasa. Preserve these as three native benchmark tracks.

Source: https://sjtu-deng-lab.github.io/Streaming-WAM/

Therefore, Symthaea should not use 98.35% as a universal world-model SOTA threshold. On near-saturated standard LIBERO, a more discriminating result would be matched-budget latency/energy, robustness under dynamic/contact perturbations, harmful-intervention rate, and performance on the unsolved tail. For LIBERO-Dynamic and RoboDyna, compare only against their own protocol-specific baselines.


## Benchmark protocol ledger: native denominators and fair targets

This crosswalk is a planning-time source audit, not a reproduction. It prevents the same-looking percentages from being treated as matched results.

### RoboDyna

The official project describes 30 dynamic manipulation tasks, 3,500 demonstrations and ten evaluation seeds per task/condition. Its main policy table reports success averaged over synchronous and asynchronous execution. The displayed 18% π0.5 overall figure aggregates its Easy, Medium and household slices and **excludes the Hard split**. Human study numbers use a separate slice/control protocol: 12 people played hard conceptual tasks and household tasks, using teleoperation or direct scene-interactive controls. Do not compare the policy's 18% overall denominator directly to the human 50.2% teleoperation or 77.1% interactive results.

Official sources:
- https://robodyna.github.io/robodyna/
- https://robodyna.github.io/robodyna/index.html

For Symthaea, report the published overall population and the Hard held-out composition separately. The prospective +5 percentage-point target applies only to the exact same policy/episode/metric denominator. A higher score on Easy or the human-study slice cannot be used to claim that overall target.

### LIBERO-Dynamic

The official project describes 40 tasks and a shared 1,800-episode evaluation population per configuration: 40 tasks × 3 motion families × 5 reviewed trajectories × 3 target speeds. The world advances during calibrated inference latency.

Published references must remain separated:
- Frozen/static-trained π0.5 (JAX), asynchronous best measured action horizon: 57.8% across 1,800 episodes.
- π0.5 mixed dynamic+static fine-tuning: 69.3%; a separately trained recipe, with a different training budget, not a frozen-backbone augmentation result.
- On the same π0.5 PyTorch checkpoint, the reported execution-strategy references include Async Queue 48.4%, measured-latency Sync Hold 43.1%, and RTC fixed=10 37.8%. Zero-latency Sync Hold at 50.4% is an idealized timing diagnostic, not a deployable baseline. VLASH at 55.4% uses a separately trained delay-aware checkpoint and is a system reference, not a same-checkpoint controller ablation.

Official sources:
- https://renshaojie233.github.io/LIBERO-Dynamic/
- https://github.com/renshaojie233/LIBERO-Dynamic

The first Symthaea comparison should target the frozen 57.8% reference under the exact 1,800 episodes and calibrated clock. The 69.3% recipe is a later end-to-end system target, not evidence of the isolated Symthaea contribution.

### Standard LIBERO and Streaming-WAM

Standard LIBERO is not LIBERO-Dynamic. The Streaming-WAM project's standard-LIBERO table reports 98.85% for FastWAM-Joint-CD, 98.60% for FastWAM-Joint and 98.35% for Streaming-WAM. Streaming-WAM's notable advantage is runtime: 41 ms chunk preparation versus 493 ms for FastWAM, with 12× reported chunk-time speedup. Use it as an asynchronous execution/latency comparator, not as a LIBERO-Dynamic policy result or the highest raw standard-LIBERO success result.

Official source: https://sjtu-deng-lab.github.io/Streaming-WAM/

### World-In-World

Keep its Active Recognition, Active Embodied QA, Image-Goal Navigation and Robotic Manipulation tasks as a separate closed-loop visual-world-model track. Its report that inference-time compute can improve outcomes motivates adaptive budget allocation, but it does not license pooling its score with RoboDyna, LIBERO-Dynamic, RoboTwin, RoboWM-Bench or H2RBench.

Official source: https://world-in-world.github.io/

### Targeting strategy

- **Near-saturated standard LIBERO:** prioritize p95/p99 latency, resource use, long-tail failures, controllability and matched execution quality, not an arbitrary 1-point success chase.
- **RoboDyna:** target dynamic and compositional failure families; report Easy/Medium/household and Hard separately; match sync/async conditions and denominators.
- **LIBERO-Dynamic:** first isolate frozen-policy Symthaea lift from action-timing effects; then test fine-tuned full-system recipes separately.
- **World-In-World:** test whether adaptive inference compute improves task success on its native four tasks at matched total episode compute and deadline.
- **RoboWM-Bench:** measure whether predicted futures can be converted into executable actions.
- **H2RBench:** retain per-task simulated and physical results plus Pearson, Spearman and MMRV as separate rank-agreement diagnostics.

A result only beats a comparator when the exact benchmark, population, checkpoint/training recipe, action semantics, timing, hardware/resource envelope and failure accounting are matched or the difference is explicitly treated as a separate track.

## DEEP v2: single-step value versus sequential closed-loop value — 2026-11-24

Pham Cong and Luong, *The Decision Value of Perception Compute* (arXiv:2609.35910v2, revised 16 November 2026), is directly relevant prior art for decision-value-aware compute allocation:

- It values an escalation by the change in downstream decision loss, not by perception accuracy alone.
- It explicitly allows escalation to have negative decision value.
- It compares pre-escalation allocators with a budget-constrained decision oracle, charges allocator cost, and treats all-full inference as a baseline rather than an upper bound.
- Its v2 reports that 34–62% of escalations that change downstream loss make it worse across its evaluated driving settings.

Sources:
- https://arxiv.org/abs/2609.35910
- https://arxiv.org/html/2609.35910v2

### Exact scope distinction

DEEP's core estimand is a **single-step escalation value** for a fixed input and reference scene. Where the downstream decision depends on prior action, both counterfactual branches use the all-cheap prior action. The paper explicitly says this is not the escalation's closed-loop value (Section 3, “Single-step scope”). Its measured budget is an average per-input budget, not a per-frame deadline.

Symthaea must therefore not claim to invent decision-value-aware compute allocation. The research question here is the sequential extension:

> When additional imagination changes an action, does its apparent local benefit survive the state, observation, contact and recovery consequences of that changed action?

The required comparison has two distinct estimands:

1. **Fixed-context local value:** evaluate low- and higher-budget decisions against the same frozen decision context. This is the closest compatible DEEP-style diagnostic.
2. **Branch-evolving episode value:** execute each budget policy from the same initial state and exogenous forcing commitment, but let each branch evolve under its own actions and observations. After action divergence, do not replay a shared prerecorded future state trajectory. Use #13375/#13377 for causal trajectory pairing and event identity.

A positive local value does not imply positive episode value. The exact semantic counterexample is F7 in the machine-readable fixture.

### Required comparison arms

- Fixed low-budget imagination.
- Fixed high-budget imagination.
- Fixed per-decision and fixed per-episode compute schedules.
- Adaptive per-decision allocation using only candidate-visible evidence.
- Entropy/uncertainty-threshold allocation as a diagnostic baseline, not the target objective.
- DEEP-style per-decision budget oracle, evaluator-only and not deployable.
- Episode-level hindsight oracle, evaluator-only diagnostic and not a candidate baseline.

Use #13281 for paired compute/observation interaction where the treatment design matches; #9122 for the existing paired direct-versus-predictive advantage auditor; #13375/#13377 for branch-evolving causal pairing; #13176 for resource/exposure attribution; and #9437 for evaluator isolation. Do not add another regret oracle, VOI estimator, trajectory replayer, scheduler, bootstrap engine or action authority.

### Additional diagnostics

Report, without combining them into a new composite score:

- local-to-closed-loop value sign-reversal rate;
- local and episode-level net decision value, with all compute, sensing, queueing, action delay, recovery and failure costs included at their actual timestamps;
- harmful-imagination intervention rate and realized regret after selection;
- episode success, verified progress, recovery success/time and no-output/abstention exposure;
- total and stage p50/p95/p99 latency, measured energy, peak memory and inference work;
- performance by task family, contact regime, supported horizon and deadline stratum.

The local-to-closed-loop sign-reversal rate is a diagnostic of how often a local escalation conclusion fails after feedback changes the trajectory. It is not a safety probability and is not a replacement for episode-level task outcomes.

### Experimental priority

1. Reproduce the cheap/high decision-value contrast on a small numerical-control slice, with the fixed-context and branch-evolving outcomes kept separate.
2. Test the same distinction on World-In-World at matched episode compute and deadline, preserving its four native task families.
3. Move to one RoboDyna or LIBERO-Dynamic slice only after the exact checkpoint, native denominator, timing profile and causal-pairing prerequisites are pinned.
4. Compare against DEEP's allocator/oracle methodology only on a compatible perception-escalation subproblem. Do not compare its driving results directly with manipulation success.

This addendum extends the existing #13445 experiment contract; it is not a new standalone score or execution authority.

## Numeric-control baseline parity: MRS.Q and WIMLE — 2026-11-24

The first external numerical-control experiment should reuse the already pinned CMAQ-011 protocol rather than introduce a new benchmark adapter.

### Source and licence pins

- MRS.Q: `facebookresearch/MRSQ@b12abaf309833f28619dfd87f4251074abed3ae3`, CC BY-NC 4.0. Treat as an isolated upstream dependency; do not vendor, relicense or imply commercial reuse permission.
- WIMLE: `mehranagh20/wimle@be730d6e7cd1538ca5519b8f505feff126b92938`, MIT, subject to preserving upstream attribution and checking dependency licences.

These are source pins recorded by #9095, not evidence that either baseline has been reproduced in the current Symthaea campaign.

### Two different experiments

1. **Paper-native reproduction:** run each baseline with its published wrappers and settings. This establishes whether its own reference result can be reproduced, not whether it is directly comparable with the other system.
2. **Wrapper-normalized head-to-head:** run compatible agents through one frozen environment adapter with identical action cadence, action transforms, reward aggregation, controller decisions, termination/reset rules and base physics-step horizon. Label this as a normalized adaptation, not a paper reproduction.

The DMC wrapper audit in #9095 found a concrete parity hazard:
- MRS.Q defaults to action-repeat 2, aggregates two underlying DMC rewards and uses 500 controller decisions / 1,000 base simulator steps.
- WIMLE uses one underlying DMC step per selected action, action rescaling and SinglePrecision, with 1,000 controller decisions.

The approximately equal number of base physics transitions does not make the decision opportunities or reward semantics equivalent. Both decision count and base simulator transitions must be captured in receipts. Compatible shared DMC tasks identified by #9095 are `quadruped-run`, `humanoid-run` and `dog-run`; MRS.Q's `quadruped-walk` reference is MRS.Q-only and must remain a separate native reproduction.

### Comparison method and interpretation

Use #9122's existing five-arm auditor:
- D0: direct policy, no predictive rollout;
- D1: compute-matched direct/value policy;
- W1: learned action-conditioned world-model planner;
- W2: shuffled-future negative control;
- W3: evaluator-only oracle-dynamics diagnostic.

Add MRS.Q's minimum-over-value-functions selector as a named **value-aggregation/search baseline**. Do not describe it as an independent ensemble of dynamics models. Reuse #9396 for adaptive search/optimism-pocket stress and #9127 for selection-conditioned uncertainty calibration. If observations are an experimental treatment, reuse #13281's signed compute×observation contrasts.

First use the checked-in MRS.Q Gym-HalfCheetah-v4 and DMC quadruped-walk reference rows as source/parity diagnostics; then run the shared-task normalized DMC comparison. Report prediction error, selected-action regret, return, lower-tail/CVaR outcomes, search-induced optimism, selected-action coverage, inference work, simulator calls, p50/p95/p99 latency and memory separately. Preserve each environment/task and protocol as its own result row; no pooled score across vector-control, visual-world and manipulation tracks.

This is the low-cost falsification stage before visual and robot acquisition. It tests whether Symthaea's decision-focused controls improve search decisions under a known environment contract, not whether it has already beaten the world-action-model leaderboard.

## Individual counterfactual effect retention — 2026-11-24

A further distinction is required beyond population-average action value and decision-equivalent state compression.

Okita, *Where Individual-Effect Information Survives: A Diagnostic Ladder for World Models* (4 November 2026; DOI 10.3390/make8110363), asks whether a world model can predict what would have happened to **this same individual** under another action. The paper separates:
- factual prediction / reconstruction;
- average intervention effect across individuals;
- individual treatment-effect variation that requires inferring instance-specific latent causes from its history (abduction).

Its diagnostic chain measures information available in the environment, retained in the model latent, and expressed by the imagined prediction. It reports cases in which information survives in the recurrent latent but is lost in rollout/output. It also requires checking whether the environment formulation supports the question at all: action efficacy, branchability and individual dependence; and whether the estimate is measurable above its estimation floor and converged. The paper reports 12 environment formulations, six of which fail at least one condition. Its evidence is bounded: the model-side results use small low-dimensional systems and do not establish the same pattern for large pixel-based or modern robotics world models.

Source: https://www.mdpi.com/2504-4990/8/11/363

### Why this extends #13255 rather than replacing it

Decision equivalence asks whether two states can safely share a representation for a declared action/decision scope. Temporal representation preservation asks whether that remains true across future decision epochs. Individual counterfactual fidelity asks a different question: does the representation retain the history-dependent, instance-specific information needed to predict how *the same instance* responds differently under alternative actions?

An average action effect can be correct while all individual effects are wrong because positive and negative effects cancel. A latent can preserve the instance identity while the transition/decoder maps every imagined intervention to the same population-average effect. Neither reconstruction accuracy nor average treatment-effect accuracy catches this.

For individual (i) and two interventions (a,b), define the task-specific individual effect:

`Delta_i(a,b) = U_i(a) - U_i(b)`.

The mean effect `E_i[Delta_i]` is a second-rung/population quantity. It does not determine the individual `Delta_i` values. Preserve and report both.

### Required measurement chain

For a declared counterfactual-capable environment formulation, report separately:

1. **Environment opportunity:** whether the action set changes outcomes, branchable paired interventions can be executed from the same exogenous condition, and the effect varies across individuals.
2. **History/latent retention:** whether the encoded recurrent state retains information about individual effect heterogeneity beyond the branch-point observation alone.
3. **Imagination/decoder expression:** whether action-conditioned rollout and decoded output preserve the same individual's effect under each intervention.
4. **Decision consequence:** whether losing that individual contrast changes feasibility, selected action, realized regret or recovery.
5. **Measurement validity:** whether the estimator's error floor and sample-size convergence make its reported effect interpretable.

The abducible-share decomposition `A -> A_lat -> A_pred` from the paper is a useful external diagnostic, not an automatic truth oracle. Only report an environment-level `A` when the environment prerequisites, parameterization, intervention set, estimation floor, sample size and convergence checks are all disclosed. The paper's `A >= 3 phi` criterion is its source-specific convention; Symthaea must preregister its own threshold and include a 2x/3x/4x sensitivity analysis rather than treating 3x as universal.

### Branch construction and leakage controls

- Replay the same factual history to the branch point for each paired run.
- Save/restore the same individual state and its exogenous-variable commitment before each intervention.
- Keep exogenous disturbances paired within the same individual; allow endogenous state/event consequences to differ when caused by the intervention.
- Bind branch-point observation, full history, latent-state digest, intervention identity, event/draw bundle, policy/model digest and evaluator ground truth.
- Do not expose the hidden individual identity or branch outcomes to the candidate except through the factual history it is entitled to observe.
- Use #13375/#13377 and the Symtropy stochastic correspondence verifier for source-bound branch and event identity. Common random numbers do not permit replaying an impossible post-intervention trajectory.
- Treat missing, unpaired, non-branchable or unsupported counterfactual outcomes as incomplete, not zero effect.

### HDC/CfC-LTC experiment

Extend #13255's existing HDC-only, CfC/LTC-only, HDC+CfC/LTC, reconstruction, value-aligned and simple-projection comparisons with two readouts:
- a frozen-latent probe/readout of history-specific individual effects;
- the model's own action-conditioned imagined contrast from that same latent.

This distinguishes representation failure from rollout/decoder failure. Train the representation on development individuals and intervention pairs; keep held-out individuals, intervention pairs and dynamics families separate. Report population-average effect error, individual-effect MAE/RMSE, effect-sign accuracy, per-individual ranking, calibration slope, lower-tail error, decision regret and the `A/A_lat/A_pred` chain (only when statistically interpretable). A probe is diagnostic only; a latent readout that uses evaluator labels is not a deployed capability.

### Machine-readable controls

The semantic fixture set now includes:
- **F8 — mean cancellation:** the population-average effect is exactly correct (zero), but individual effects are +10 and -10. The history-conditioned latent recovers both; the rollout collapses both to zero. Its exact latent MAE is 0 and rollout individual-effect MAE is 10.
- **F9 — formulation gate:** action efficacy, branchability, individual dependence, estimation-floor measurability and convergence are separately checked. Missing/failed conditions suppress a third-rung claim; the source's 3x floor convention is explicitly not treated as universal.

These are deterministic semantic controls, not empirical abducible-share estimates, model results or physical-safety evidence.

### Benchmark placement

Start with deterministic, branchable simulation formulations where the same individual can be restored and multiple interventions actually executed. Robosuite/Meta-World-style manipulation formulations can be adapted, but each task/intervention definition must pass the five-condition audit; a simulator name alone is not sufficient. Then add held-out dynamic manipulation cases through #13389. Observational real-world datasets may support population-effect estimation under appropriate assumptions, but they do not supply exact individual counterfactual ground truth by themselves. Do not label an observational comparison as a third-rung reproduction.

This extension belongs to #13255's representation/temporal experiment and #13445's decision-value evaluation. It adds no new policy, simulator, causal oracle, calibration engine, scorer, scheduler or action authority.
