# World-Model Assurance Benchmark Matrix 2026 — CMAQ-029

**Status:** proposed benchmark protocol; no comparison result is claimed.  
**Purpose:** establish a fair, falsifiable comparison between current world-model calibration/reachability methods and the CMAQ assurance stack.

## 1. Competitive positioning

The objective is not to beat large world models at photorealistic generation or replace advanced dynamics learners. The objective is to produce a reusable assurance layer that can sit over an HDC, Koopman, recurrent, latent-video or VLA transition model and answer:

- Is the prediction identifiable from permitted evidence?
- Does its uncertainty envelope cover the full registered decision block?
- Is the current trajectory still recoverable by a qualified backup?
- How many independent recovery modes remain?
- Is another observation/probe worth its cost?
- Can planner search exploit unsupported optimistic regions?

The comparison must distinguish prediction quality, calibration validity, reachability, runtime assurance and executed task utility. These are separate axes.

## 2. Closest published calibration/reachability comparators

| Comparator | Published capability | Important boundary | Matched CMAQ-029 test |
|---|---|---|---|
| CAFT (Zhou, Lindemann, Sesia; ICML 2024) | Simultaneous forecasting bands for a full random trajectory | Its unit is a trajectory; it is not by itself a mechanism-clustered execution/assurance pipeline | Same fixed rollout/action/query block, trajectory-level score, same nominal coverage and predictor |
| Koopman + conformal reachability (Nath, Yin, Chou; L4DC 2026) | Learned Koopman lift, reachable-set propagation, conformal inflation; 11D Hopper, 28D Swimmer, 12D quadrotor results | Different model family and reachability implementation; no directly interchangeable HDC benchmark score | Re-run the same tasks and state metrics; measure coverage, set volume, runtime and downstream recovery |
| CDR-LWM (Wang; Electronics 2026) | Five-member latent ensemble; normalized joint next-state residual; runtime backup under tested dynamics/actuator/observation shifts | The paper states its conformal theorem is one-step and exchangeable, not trajectory-wise or shift-invariant; episode-disjoint splitting still leaves within-episode dependence | Extend the same CartPole/Pendulum tasks to whole-block and independent-cluster calibration; compare trigger rate and safety/return |
| Transported Beta Laws (Ramos et al.; COPA 2026) | Models calibration-conditional coverage distributions and departures caused by clustering, dependence and scale/test shift | Statistical coverage-law analysis, not a world-model reachable-set or recovery controller | Use as a dependence-aware calibration audit; compare nominal split-CP coverage with observed cluster-level coverage and bad-calibration probability |
| Conformal Coverage of Time Series (Zhai, Cheng, Wu; 2026) | Non-asymptotic temporal-dependence coverage bounds and block-based inference for realized coverage | Requires an appropriate functional-dependence representation; not an automatic arbitrary-dependent-data guarantee | Use block-based uncertainty intervals and temporal-dependence diagnostics on sequential residual streams |

### Primary references

- CAFT: https://proceedings.mlr.press/v235/zhou24l.html
- Koopman conformal reachability: https://proceedings.mlr.press/v331/nath26a.html
- CDR-LWM: https://www.mdpi.com/2079-9292/15/20/4765
- Transported Beta Laws: https://proceedings.mlr.press/v329/r-ramos26a.html
- Conformal Coverage of Time Series: https://arxiv.org/abs/2609.33868

## 3. Matched predictor baselines

All predictors must see the same model-fitting clusters, action inputs, context features, intervention/query/horizon contract and evaluation blocks.

| Predictor | Why include it |
|---|---|
| Global per-action HDC mean delta | Actual NixWard additive-delta control; measures loss from pooling away context |
| Contextual HDC delta | Actual current candidate intended to preserve observable context structure |
| Context + realized-action-parameter HDC | Tests whether action parameter encoding adds predictive sufficiency |
| Identity / no-op | Copying shortcut and false-merge negative control |
| Full-state memorization | Upper-bound memory and over-splitting control |
| Symbolic/SCM baseline | Intervention-structured comparison where variables/equations are available |
| Recurrent transition model | Temporal baseline with actual training, checkpoint and inference integration |
| Koopman dynamics + conformal reachability | Published high-dimensional reachability baseline |
| Latent ensemble + CDR-LWM-style governor | Published latent-world-model assurance baseline |
| Direct VLA/WAM policy without a separate world-model shield | End-to-end control baseline |

Do not describe HDC as CfC/LTC unless that transition model has an actual training, checkpoint and inference path. Do not compare a small HDC transition module against a generative foundation model by parameter count alone. Report the end-to-end system cost of every model and the incremental value of adding the same assurance layer to each.

## 4. Matched uncertainty and reachability baselines

| Uncertainty method | Unit/coverage target | Required scorecard |
|---|---|---|
| No uncertainty / prequential MSE | None; diagnostic only | Unsafe and optimistic error rate, not a calibrated coverage claim |
| Ensemble disagreement | Model-member disagreement | Calibration error, false confidence under shared error |
| One-step split conformal | Next-state sample | One-step marginal coverage and interval volume |
| CAFT-style simultaneous trajectory conformal | Full trajectory | Whole-path coverage and tube volume |
| CDR-LWM-style normalized joint residual | One-step state vector | Reproduce published trigger/backup baseline, then stress multi-step rollouts |
| Koopman + conformal reachability | Propagated reachable set | State-set coverage, volume, propagation compute |
| CMAQ-029 cluster-block maximum | Entire pre-registered cluster block | Full-block coverage, cluster count, radius, worst context |
| Dependence-aware transported-Beta / block inference | Calibration-conditional coverage distribution / dependent residual process | Bad-calibration probability, coverage CI, dependence sensitivity |

The cluster-block maximum is valid only if the independent cluster blocks are exchangeable under the frozen protocol. If clusters remain dependent through a shared mechanism, asset, facility or acquisition process, collapse to a higher-level independent unit or use a justified dependence-aware method. A string-valued cluster label does not establish statistical independence.

## 5. Runtime-assurance baselines

| Runtime architecture | Role in comparison |
|---|---|
| CBF / HJ reachability / viability filter | Classical hard-invariant baseline; record conservatism and compute |
| Backup-CBF / predictive safety filter | Trusted fallback plus online filtering |
| Fixed-threshold Simplex | High-performance controller with one high-assurance controller and fixed switching condition |
| CDR-LWM uncertainty/novelty governor | Latent model with empirical runtime backup trigger |
| Context-dependent multi-candidate Simplex | Recovery profiles conditioned on the declared context and parameter region |
| CMAQ-027 recoverability-qualified commitment | High-performance controller may continue while every prefix remains in one qualified recovery tube |
| CMAQ-028 recovery-option portfolio | Requires a minimum count of distinct, externally justified recovery failure domains across every committed prefix |

Compare on matched action sequences first, then end-to-end closed-loop policies. Report the separate contribution of uncertainty-envelope calibration, recoverability and recovery-option redundancy; otherwise a better controller may be mistaken for a better assurance layer.

## 6. Core factorial evaluation

| Factor | Pre-registered levels |
|---|---|
| Predictor | Global HDC; contextual HDC; context/action HDC; symbolic/SCM; recurrent; Koopman; latent ensemble |
| Calibration unit | Transition; episode; full trajectory; independent environment/mechanism cluster |
| Coverage block | One step; fixed horizon; fixed action/query portfolio; complete cluster block |
| Context | Known scope; unseen cluster/same context; observable shift; unidentifiable hidden alias; unsupported scope |
| Dynamics shift | Nominal; actuator; observation; friction; mass/inertia; latency; contact; disturbance; compound |
| Planner selection | Direct policy; fixed search; higher search; adversarial search over optimistic errors |
| Recovery mode | No backup; one invariant backup; one recoverability profile; multiple recovery options |
| Execution outcome | Success; failure; partial completion; rollback; delayed effect; unknown reconciliation |

Training, scale-validation, calibration and test clusters must be disjoint. Family/environment IDs stay evaluator-only. Any action/query selection process that can inspect multiple candidate trajectories must be frozen and included in the calibration protocol; expanding search invalidates a calibration that did not score the expanded candidate block.

## 7. Metrics — report separately

### Calibration
- whole-block simultaneous coverage at the declared nominal level;
- one-step coverage only as a diagnostic;
- cluster-level realized coverage and cluster confidence interval;
- calibration-conditional coverage variability / bad-calibration probability where a dependence model permits it;
- radius and reachable-set volume;
- coverage by observable context cell and worst-family reporting;
- number of independent calibration units and rank/support sufficiency.

### World-model quality
- intervention × query × horizon response error;
- pairwise false merges and false splits;
- state abstraction class count and all-in retained memory;
- worst-cluster and worst-context error;
- point / set / UNKNOWN coverage and selective risk.

### Runtime assurance and recovery
- unsafe steps and envelope violations;
- recoverability-entry margin and latest switch deadline;
- successful recovery rate and time/energy to recovery;
- high-performance utilization relative to an invariance-only shield;
- recovery-option count at every committed prefix;
- option loss and common-cause fault-injection outcomes.

### Decision quality
- selected-action regret, not only open-loop model error;
- decision value per probe cost;
- unnecessary probes and missed decision-critical probes;
- false-split cost during refinement;
- optimizer exploitability as search budget grows;
- mission success, intervention rate and safe fallback rate.

### Resource cost
- total model + calibration + evidence-index memory;
- training, calibration, update and inference latency;
- tail latency, power/energy and bytes retained;
- cost per successful task and per recovered fault.

## 8. Proposed qualification targets (not achieved results)

| Gate | Proposed target |
|---|---|
| Deterministic calibration witnesses | Exact expected quantile and no pseudo-replication |
| 95% split-CP support | At least 19 independent calibration clusters merely to produce a finite non-randomized rank; use 100+ for practical width estimation and 200+ held-out clusters for informative coverage audits where feasible |
| Full-block coverage | Empirical cluster-level coverage consistent with the pre-registered nominal target; publish confidence intervals, not a pass/fail count alone |
| Envelope efficiency | At the same cluster-level coverage, smaller reachable-set volume than a conservative matched baseline |
| Recovery | No missed switch in deterministic certified witnesses; zero false certificate admission |
| Recovery portfolio | Preserve k distinct failure domains after injected loss of any one declared domain in the single-failure experiment |
| Compression | Preserve the earlier stretch target of at least 25% lower all-in representation cost with no more than 1 percentage point task-success loss |
| Planner robustness | More search does not increase hard-invariant violations; selected-action regret no worse than matched direct-policy baseline |

The 19-cluster number is a mathematical minimum for a finite 95% order-statistic quantile, not a sufficient scientific sample size. The 100+/200+ values are proposed study-design targets, not theorems. The actual number should be set using power and interval-width calculations at the independent-cluster level.

## 9. What would constitute a credible advantage

A credible result is not “higher average prediction accuracy.” It would show, on shared tasks and matched model/compute interfaces, that the CMAQ stack:

1. achieves target full-block coverage with fewer or comparably sized reachable sets;
2. detects when evidence is insufficient and refuses to apply a stale calibration after scope, checkpoint, action-selection or horizon changes;
3. preserves higher high-performance utilization than an invariance-only shield while maintaining recovery success under the certificate's covered conditions;
4. retains multiple independent recovery modes when required and correctly detects common-cause concentration;
5. spends probes only when their result can reduce decision regret;
6. resists optimizer exploitation as the action/search budget grows;
7. does all of the above with a smaller all-in model/evidence/resource footprint.

All claims require exact checkpoint, source-manifest, contract, cluster split and evidence receipts. A synthetic fixture is software qualification, not SOTA, physical safety or real-world validation.
