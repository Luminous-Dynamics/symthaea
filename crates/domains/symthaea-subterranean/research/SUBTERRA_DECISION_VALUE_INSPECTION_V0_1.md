# Subterra Decision-Value Inspection Benchmark v0.1

**Status:** synthetic-first research protocol  
**Date:** 2026-11-17  
**Tracking:** [#12628 — SUBTERRA-INSPECTION-VOI-001](https://github.com/Luminous-Dynamics/symthaea/issues/12628)

## 1. Research question

Can an underground robotic system choose the *next admissible observation* that most reduces the risk of a consequentially wrong engineering or maintenance decision, while respecting inspection cost, time, energy, exposure, evidence quality and operating constraints?

The target is not maximum entropy reduction, number of scans, number of points, waypoint visits or number of tests.

The target is:

> Reduce held-out decision regret with the least admissible evidence acquisition burden, without increasing false acceptance, false-clear claims, unauthorized actions, or unresolved physical-state risk.

This is a task adapter and benchmark for existing planning/referee components. It does not create a generic active-learning engine or a second Subterra mission executive.

## 2. Reuse the existing machinery

The initial implementation should adapt existing components:

| Existing component | Reuse |
|---|---|
| `subterra-survival-planner` | Finite-horizon contingent measurement policies over hypotheses, decisions, likelihoods, resource budgets, mission interruption and robot-loss outcomes |
| `subterra-robust-survival-referee` | Independently recompute one immutable policy across multiple model scenarios; ignore candidate-cached branch probability/posterior values for scoring |
| #10143 — POWERTRAIN-DISCOVERY-001 | General active experiment selection, decision-regret reduction and multi-objective experiment cost accounting |
| #10822 — CAVE-SURVEY-001 | Decision-relevant active sensing, separate observed/inferred/occluded/unknown geometry and resource/fault baselines |
| #11277 — EV-DIAGNOSTIC-VOI-001 | Domain-specific EVSI precedent and no-test / periodic / rule / entropy / EVSI baseline ladder |
| #12606 / #12609 | Construction and asset-condition task definitions, hidden truth and task-specific metrics |
| #12621 | Integrated accepted-work mission-level evaluation |

Do not duplicate the generic EVSI planner, survival-policy referee, SLAM/mapping engine, sensor physics, work-order authority or operation state machine.

## 3. Decision-value formulation

Let:
- (H) be a finite set of mutually exclusive task hypotheses, including an explicit open-world `unknown` hypothesis with non-zero prior;
- (b(h)) be the current belief from admissible observations;
- (D) be the set of currently admissible downstream decisions;
- (L(d,h)) be the evaluator-owned loss for decision (d) if hypothesis (h) is true;
- (a) be an admissible inspection action;
- (O_a) be the action's declared outcome set;
- (P(o\mid h,a)) be a calibrated or explicitly synthetic outcome likelihood;
- (c(a)) be the declared resource-cost vector and acquisition burden.

The no-further-inspection baseline is:

[
R_0(b)=\min_{d\in D}\sum_{h\in H} b(h)L(d,h)
]

For one inspection action, the outcome probability and posterior are:

[
P(o\mid b,a)=\sum_h b(h)P(o\mid h,a)
]

[
b(h\mid o,a)=\frac{b(h)P(o\mid h,a)}{P(o\mid b,a)}
]

The expected posterior decision risk is:

[
R_a(b)=\sum_{o\in O_a}P(o\mid b,a)
\left[\min_{d\in D}\sum_h b(h\mid o,a)L(d,h)\right]
]

The **gross EVSI** is:

[
\operatorname{EVSI}(a\mid b)=R_0(b)-R_a(b)
]

A task-specific monetary or operational **net value** may subtract a predeclared acquisition-cost utility, but never by silently scalarizing unlike units. Report the full resource vector and the decision-loss term separately. If scalar resource prices are used by the existing planner, freeze and publish those prices before evaluating candidates. Always compare the Pareto frontier in decision loss, energy, duration, data, exposure and human/verifier effort.

For a multi-step contingent policy, calculate expected posterior loss over the policy tree, and separately report the worst supported model/branch resource and risk envelope. The independent referee evaluates the *same policy* across model scenarios; candidate-side posterior caches must not be trusted as score inputs.

## 4. What makes an observation valuable?

Two deliberately different cases are required:

1. **High information, zero decision value.** The observation sharply changes a posterior or reduces entropy, but every plausible outcome still yields the same admissible decision. EVSI is zero (or effectively zero within declared numerical tolerance); the test should not be preferred merely for novelty.
2. **Modest information, positive decision value.** The observation changes the optimal admissible decision near a high-consequence boundary, preventing a false clear, missed defect or unnecessary intervention. Its decision value may exceed that of the higher-entropy test.

The benchmark must include both. Expected information gain is a diagnostic metric, not the optimization target.

## 5. Task adapters

### A. Pipe/cable corridor

**Hypotheses** may distinguish:
- serviceable within declared inspection scope;
- defect present below a defined severity threshold;
- consequential defect present;
- multiple or interacting defects;
- unknown/unobservable/out-of-domain condition.

**Candidate observations** may include:
- an additional RGB/depth/LiDAR viewpoint;
- thermal scan;
- acoustic/vibration measurement;
- electromagnetic measurement where a qualified modality and protocol exist;
- a separate sensor pose or separately calibrated inspection chain;
- human/specialist review.

**Admissible decisions** may include continued monitoring until a declared expiry, gathering another observation, marking the segment unknown/hold, requesting specialist review, or presenting an approved maintenance work-package proposal.

Detection, localization, extent, severity, coverage and calibration metrics remain per defect class. An inspection result does not itself authorize isolation, repair or electrical work.

### B. Drawpoint hang-up volumetric assessment

**Hypotheses** may distinguish no visible blockage within declared coverage, small/large blockage bands, ambiguous topology, and unknown due to occlusion or invalid sensor geometry.

**Candidate observations** may include a new stand-off viewpoint, alternate scan geometry, a second LiDAR/depth observation, or specialist volumetric assessment.

**Admissible decisions** may include retaining unknown/blocked status, gathering additional views, requesting human review, or using a bounded volume estimate for a separately approved downstream engineering decision.

A plausible mesh, estimated empty volume or high-confidence free-space score never grants permission to enter, blast, dislodge or manipulate a drawpoint.

### C. Post-construction/support verification

**Hypotheses** may distinguish required postcondition met, measurable nonconformance, partial installation, damage, and insufficient/occluded evidence.

**Candidate observations** may include independent metrology, alternate modality, a separate inspection pose, or specialist review.

**Admissible decisions** may include acceptance only for the explicit checked postconditions, rejection, further evidence collection or hold for engineering disposition.

An acknowledgement or actuator telemetry from the executing robot is not independent evidence of physical installation or structural adequacy.

Do not pool likelihoods, loss vectors, labels, thresholds or acceptance criteria between these three adapters.

## 6. Adapter contract to the existing planner

Each task adapter supplies a frozen `PlanningProblem`:
- hypothesis identifiers and priors;
- downstream decision identifiers and evaluator-owned loss-by-hypothesis vectors;
- measurement actions, modalities, resource costs, external admissibility-gate results;
- outcome likelihoods by hypothesis and terminal loss only for genuinely terminal mission outcomes.

Each action catalogue entry separately carries:
- task/asset and work-scope IDs;
- exact sensor/profile, calibration, coordinate frame and applicability domain;
- evidence and common-cause dependencies;
- outcome definitions, likelihood source/class and uncertainty;
- energy, duration, data, exposure, downtime and human/verifier burden;
- required safety prerequisites, external approval and expiry.

A boolean external gate passed to the planner is only an input from the reviewed authority/admissibility system. It is not derived from confidence, trust or expected value. If a prerequisite is unknown, the action is ineligible.

The task adapter also supplies the initial belief and no-test decision baseline. An independent evaluator owns the hidden truth, loss function, outcome generator, cost model and final decision score. The candidate must not receive hidden labels or future observations.

## 7. Baseline ladder

Run all baselines on identical initial beliefs, task sets, sensor/action eligibility, information access, resource budgets and fault schedules:

1. no further inspection / current-evidence decision;
2. fixed periodic or fixed coverage schedule;
3. deterministic risk-threshold / age / condition rule;
4. random eligible observation;
5. maximum predictive entropy / uncertainty sampling;
6. expected information gain;
7. one-step myopic EVSI;
8. finite-horizon contingent EVSI policy;
9. the exact same candidate policy independently refereed across model scenarios;
10. human/specialist inspection under matched information and time budgets where practical.

Also compare the fleet allocation layer separately: nearest eligible platform, greedy capability assignment, charging-aware route planner, and decision-value/verification-aware assignment. Do not conflate path planning gains with inspection-policy gains.

## 8. Metrics — no single weighted score

**Decision quality**
- no-test expected Bayes risk;
- expected posterior decision risk;
- gross EVSI and decision-regret reduction;
- realized hidden-truth regret;
- fraction of inspections that actually change the admissible decision;
- wrong-action and missed-defect cost by task/defect family;
- EVSI prediction error and calibration.

**Evidence quality**
- defect recall at fixed false-alarm rate;
- false-clear / false-closeout rate;
- observed coverage and unknown/occluded surface or volume debt;
- localization/extent/volume error;
- calibration, Brier score, ECE and selective risk/coverage;
- common-cause sensitivity and independent-verifier agreement.

**Resource and operational cost**
- energy, duration, data, exposure and downtime;
- human and independent-verifier minutes;
- inspection cost per accepted asset / work package;
- recovery time and unresolved-unknown age;
- planner runtime, expanded search nodes and memory.

**Robustness**
- per-model and worst-model expected loss;
- worst supported branch resource use;
- model disagreement and OOD abstention;
- correct NO_TEST / HOLD / HUMAN_REVIEW behavior;
- correct handling of unobservable and non-discriminating actions.

## 9. Frozen adversarial cases

The companion JSON includes 20 deterministic case definitions. Core controls include:
- high-EIG observation that cannot change the decision;
- low/modest-EIG observation that crosses a high-consequence decision boundary;
- omitted hidden defect and non-zero unknown prior;
- stale work-package/design/frame/calibration;
- common-cause dependencies between prior mapping and proposed verification;
- high-confidence out-of-domain modality/material/lighting;
- occluded required region and false-clear pressure;
- conflicting sensor outcomes;
- unavailable inspection action or unsafe test prerequisites;
- test cost exceeding return reserve/exposure budget;
- same policy inadmissible in one plausible geology/asset model;
- candidate access to evaluator hidden truth;
- high planner confidence with independently insufficient evidence.

A failed/unsupported/unknown observation must not be relabeled as a negative defect result.

## 10. Statistical protocol

- Freeze task manifests, hidden truth, likelihood models, loss vectors, sensor/action catalogue, model families, seeds, failure schedules, planner/referee versions and evaluator code before tuning.
- Split by site, asset, geological/material family, sensor campaign and time period; never randomly split adjacent frames.
- Use paired seeds and the same fault schedules for each baseline.
- Report denominators, confidence intervals, model-specific metrics, unsupported cases and every exclusion.
- Where outcome likelihoods are estimated from data, calibrate them on separate data and evaluate realized regret on sealed holdouts.
- Report robustness under misspecified likelihoods and omitted hypotheses, not only performance under the candidate's assumed model.
- Report zero-failure finite-suite bounds; do not say zero risk.

## 11. Proposed targets — hypotheses, not results

- At matched total inspection budget and the same hard gates, target **≥15% lower held-out expected decision regret** than the strongest non-adaptive baseline.
- Target **≥20% less inspection time or energy** at non-inferior defect recall and false-alarm rate where a matched comparison is possible.
- Zero false-clear or false-closeout claims in frozen negative controls.
- 100% correct NO_TEST / HOLD / INSUFFICIENT_EVIDENCE behavior in the declared out-of-domain, missing-coverage and inadmissible-action controls.
- Demonstrate improvements across pipe/cable and drawpoint tasks, at least two material/geology families, and one held-out fault distribution.
- Publish per-model worst-case loss, paired confidence intervals and the complete cost/decision Pareto frontier.

These are research targets, not measured results. Do not promote a numerical target if the underlying outcome likelihoods, cost assumptions or independent ground truth are not yet qualified.

## 12. Current research and industry comparisons

The 2026 charging-aware mine-inspection study by Xu et al. is a useful **fleet-routing baseline**. It jointly considers task allocation, obstacle/road-environment factors, battery/charging feasibility and path planning. Its simulation reports roughly 7.20–11.16% lower objective cost than its selected heuristic comparators, with 120 s average runtime; the authors state their study is 2-D and simulated, lacks physical-robot validation, uses simplified energy/charging assumptions and accepts a possible optimality loss from decoupled two-stage planning. Reproduce its exact task/cost definition before comparing. It does not establish defect calibration, inspection completeness or independent as-built acceptance. [Paper](https://www.nature.com/articles/s41598-026-65759-z).

The 2026 underground mine infrastructure resilience review identifies dedicated pipe/cable inspection and drawpoint hang-up volumetric assessment as substantially less mature than re-entry and ground-condition assessment. This is an evidence-backed research opening, not proof no commercial/adjacent solutions exist. [Review](https://www.frontiersin.org/journals/built-environment/articles/10.3389/fbuil.2026.2017682/abstract).

Commercial underground automation is advancing on equipment autonomy, 3-D perception and fleet operations. Subterra should not claim novelty for generic autonomous mining or multi-robot routing. Its differentiated hypothesis is **decision-valued inspection connected to independently verified physical work and unresolved-state recovery**.

## 13. Claim ceiling

This benchmark qualifies only the declared synthetic likelihoods, priors, loss functions, sensor/action catalogue, model scenarios and evaluator. It does not prove real sensor accuracy, defect-detection performance, electrical or structural adequacy, geotechnical safety, mine safety, field productivity, or authority to intervene. Physical qualification requires lawful ground-truth data, mine-owner approval, qualified specialist review, instrumented trials and independent measurements.
