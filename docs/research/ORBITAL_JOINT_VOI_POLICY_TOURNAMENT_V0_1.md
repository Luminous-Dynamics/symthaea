# Orbital Joint Sequential Evidence Policy Tournament v0.1

**Tracker:** [ORBITAL-JOINT-VOI-POLICY-TOURNAMENT-001 #13125](https://github.com/Luminous-Dynamics/symthaea/issues/13125)  
**One-step adapter:** [ORBITAL-JOINT-EVIDENCE-VOI #13111 / PR #13112](https://github.com/Luminous-Dynamics/symthaea/pull/13112)  
**Generic policy benchmark owner:** [SE-OPT-BENCH-001 #7803](https://github.com/Luminous-Dynamics/symthaea/issues/7803)  
**Status:** P0 protocol; no tournament outcome claimed.  
**Claim ceiling:** finite synthetic sequential-policy behavior only.

## 1. What changes from the one-step selector

The current one-step adapter ranks one next evidence action from a frozen decision context. This P3 benchmark evaluates a **sequence**:

`public context -> policy chooses one action -> hidden simulator reveals only its observation -> public evidence state updates -> policy chooses again -> validated stop or budget stop`

The runner owns the hidden world and actual execution simulator. Policy implementations receive only a typed public policy view. The realized hidden state is not a field on that view. No policy receives the evaluator's final truth, future outcomes or hidden-world index.

This is a benchmark of evidence acquisition, not a physical flight/qualification tool.

## 2. Two input planes and a strict information boundary

The fixture format is split:
- `policy_scenarios.json`: public decision states/sets, priors where admitted, candidate action metadata and declared response models;
- `hidden_worlds.json`: evaluator-only truth assignments, paired seed IDs and deterministic observation selections. The runner does not pass this record to the policy.

The policy input type must not contain hidden world identity, true final state, future observations, oracle score, expected winner, or post-hoc result. The simulator may use those values only after an action commitment to reveal the action's observation.

The response model is public and declared as part of the policy's epistemic context (e.g. a deterministic set-valued mapping or calibrated likelihood model). The realized outcome is produced by the simulator and only that outcome is returned to the policy.

## 3. Baseline tournament policies

The tournament must include these methods without renaming or silently changing their assumptions:

1. **FIXED_SEQUENCE:** fixed campaign order and preregistered stop rule.
2. **SEEDED_RANDOM:** deterministic seeded choice among admissible, budget-feasible actions.
3. **SPACE_FILLING_DOE:** greedily covers the largest number of uncovered declared factors/interaction cells.
4. **HIGHEST_UNCERTAINTY:** chooses highest model uncertainty, regardless of decision relevance.
5. **HYPOTHESIS_DISAGREEMENT:** adapter to the existing Symthaea population prediction-disagreement baseline.
6. **INFORMATION_GAIN / KG / MES:** only where the declared calibrated probabilistic model supports it.
7. **CI_MFBO:** Consequential Improvement multi-fidelity baseline. Chooses action and fidelity jointly, charging deferred high-fidelity validation required after low-fidelity evaluation.
8. **ONE_STEP_DECISION_VALUE:** current #13112 selector as a baseline applied repeatedly, with observations updated between turns.
9. **SEQUENTIAL_EVSI:** expected decision-value policy only in a calibrated probability regime.
10. **SEQUENTIAL_MINIMAX:** robust finite-set decision-regret reduction without fabricating probabilities.
11. **EXPERT_SEQUENCE:** human-authored action order and rationale, with engineering/review time counted.

Baselines are allowed to abstain, stop for a human decision, or report an unvalidated candidate. The tournament must not force every policy to produce an acceptance.

## 4. Action model and cost semantics

Each action binds:
- exact claim(s), provider/client profile, physical unit, configuration and ICD revision;
- class, method/procedure/model revision and applicability;
- admissibility receipt, event-frontier/change-impact receipt and dependencies;
- fidelity: analytical, reduced-order, coarse simulation, high-fidelity simulation, coupon/test, NDE/metrology or assembled article;
- public response model / observation likelihoods and expected output semantics;
- resource vector and any mandatory follow-up action for validation;
- whether the action covers required test interactions and lifecycle claims.

Resource vectors remain separate:
- engineering hours;
- solver/core hours;
- facility/instrument hours;
- specimens/material;
- calibration/setup;
- review hours;
- external cost.

No arbitrary scalar conversion. Every action must fit the remaining component-wise budget, including required follow-up validation when determining whether it is a feasible route to a validated final decision.

### Deferred high-fidelity validation

A low-fidelity action can be valuable information, but it is not a validated final decision when a high-fidelity validation is mandatory. The benchmark records:
- immediate low-fidelity cost;
- deferred mandatory validation cost;
- total route cost;
- actual spend to date;
- unpaid deferred cost at stop;
- whether the route completed its mandatory high-fidelity/physical validation.

This is essential to make a fair CI-MFBO comparison. A cheap LF sample must not be scored as cheaper than a complete LF→HF route while its validation obligation is silently ignored.

## 5. Sequential update rules

After each committed action:
1. simulator obtains the hidden world's observation for that selected action;
2. only the observation, action receipt and resource consumption are returned to the policy;
3. the policy updates its supported state set or calibrated posterior using the declared model;
4. evidence currentness, common-cause dependencies, mandatory obligations and change-impact receipts are rechecked;
5. the next admissible action is chosen, or the policy stops/abstains/escalates.

A failed, aborted, invalid, censored or out-of-domain action consumes actual resources and provides no favorable result. No policy may treat its own prediction or surrogate output as new independent evidence.

## 6. Frozen scenario families

P0 uses at least eight scenario families with paired hidden worlds and held-out variants:

- **S1 — high-variance distraction:** irrelevant high variance versus low-variance but decision-changing measurement.
- **S2 — hard feasibility boundary:** targeted evidence can prevent a false-feasible decision.
- **S3 — cheap-model bias + deferred HF:** low-fidelity model optimistic near constraint boundary, mandatory HF validation still required.
- **S4 — correlated finite state set:** preserving allowed joint states beats rectangularized marginal ranges that invent impossible combinations.
- **S5 — no information:** every available action is irrelevant; stop/abstain is correct.
- **S6 — dependency/calibration:** two nominal evidence channels share a common-cause root; stale or invalid calibration blocks actions.
- **S7 — event-driven change:** complete/incomplete owner-native invalidation frontier changes the required claim set.
- **S8 — lifecycle/abort:** initial mating cannot satisfy wear/requalification; aborted test preserves actual costs, creates no pass.

Metamorphic controls: action and input order permutations, duplicated action IDs, stale profile/configuration, wrong physical serial/ICD, missing dependency root, unsupported probability regime, budget vector overrun, hidden-world leakage attempts and non-deterministic replay.

## 7. Scorecard

Do not hide a critical miss in a weighted average. Report:
- false-feasible / missed-feasible counts;
- final decision regret and worst-case regret;
- expected regret/EVSI only in calibrated probability regime;
- time/cost to first feasible candidate;
- total cost to independently validated decision;
- unpaid deferred HF cost and unvalidated final decisions;
- full resource vector and actual vs planned spend;
- critical interaction/mandatory obligation coverage;
- change-impact/requalification correctness;
- common-cause evidence reuse/independence exposure;
- failures, aborts, censored actions and invalid actions;
- abstention/human-review accuracy;
- policy overhead and expert review time.

A policy wins only if it improves regret or full validated-decision cost **without worsening** hard false-feasible rate, mandatory coverage, change-impact integrity, evidence independence or validation completion.

## 8. Current SOTA and comparison rules

The tournament must include CI-MFBO as a first-class baseline, not merely generic Bayesian optimization:
- Aydogdu et al. 2026, *Consequential improvement acquisition function for efficient multi-fidelity Bayesian optimization*, explicitly chooses solution/fidelity and charges the deferred HF validation cost after an LF sample: https://doi.org/10.1007/s11081-026-10078-5.
- Mainini et al. 2026 volume, *Analytical benchmark problems and methodological framework for assessment/comparison of multifidelity optimization methods*: https://doi.org/10.1007/s11831-025-10392-8.
- 2026 production-engineering BO robustness study with 15,360 configurations across eight problems: https://doi.org/10.3389/fmtec.2025.1614335.
- Current Symthaea disagreement selector and the one-step #13112 selector remain separately named baselines.

Matched evaluation requires identical scenario worlds, public context, action catalogue, response models, resource budgets, mandatory validation and held-out split. Use paired seeds and report confidence intervals/effect sizes. Do not compare an optimized policy on more favorable action catalogs or ignore deferred validation.

## 9. Owners and authority

- #7803 owns reusable VOI policy metrics, tournament protocol and generic baseline semantics.
- #13111/#13112 owns the aerospace one-step action/receipt adapter.
- Symtropy #6523 profile compatibility; #6540 mandatory campaign; #6484 lifecycle; #4165 inspectability/NDE; #6256 physical test-cell/authority.
- Mycelix #6219 event-driven invalidation and bitemporal evidence; #6345/#6348 article/build/semantic evidence identity.
- Sol-Atlas #405 facility/site context only.

No action in this benchmark actuates a machine, starts a physical test or grants engineering release. A physical experiment remains under competent external test authority. Symthaea proposes and records; it does not authorize.

## 10. Deliverables and gates

P0 (current): protocol and split public/hidden fixture contract.  
P1: machine-readable policy-scenario schema and eight public response-model scenarios plus evaluator-world assignments.  
P2: typed simulator boundary and deterministic replay receipt proving policies cannot receive hidden-world fields.  
P3: sequential baseline implementations and paired-budget runner.  
P4: held-out tournament report against baselines, including CI-MFBO and expert sequence.  
P5: independent engineering review and, only then, real evidence-domain adapter.

**Claim ceiling:** a pass proves behavior on the finite synthetic simulator only. It does not demonstrate physical model validity, aerospace qualification or universal SOTA superiority.
