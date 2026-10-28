# CASTING Decision Value of Information and Cost-Aware Stopping v1

**Workstream:** CASTING-0032 ([issue #8789](https://github.com/Luminous-Dynamics/symthaea/issues/8789))  
**Parent context:** [CASTING-0031 / PR #8777](https://github.com/Luminous-Dynamics/symthaea/pull/8777)  
**Status:** research contract plus a deterministic non-probabilistic gate; no calibrated VOI policy or comparative performance result is claimed  
**Evidence classes:** synthetic, public analytic benchmark, engineering simulator, and physical manufacturing remain separate

## 1. Research question

Can an optimizer choose *which evaluation to buy next—and when to stop* by estimating how much that information can improve the eventual engineering decision, rather than maximizing an objective-only acquisition score?

The hypothesis is deliberately narrow and falsifiable:

> At matched end-to-end cost, a decision-aware, cost-aware policy improves lower-tail feasible decision quality under preregistered model shifts while preserving nominal decision quality within a frozen non-inferiority margin.

This is not a claim that value-of-information (VOI), robust optimization, multi-fidelity optimization, or cost-aware stopping is individually new. The candidate contribution is a reproducible integration of these ideas with a decision-level audit, shift evaluation, and full-cost ledger in the casting suite.

## 2. The paradigm shift

A simulation evaluation is valuable only to the extent that its result can improve a decision or avoid a consequential mistake. A lower-fidelity evaluation that merely confirms a stable ranking may be wasteful; an evaluation near a hard feasibility boundary can be valuable even when its expected objective improvement is small.

Separate three questions that ordinary nominal optimization often conflates:

1. **Search value:** could this observation improve the feasible design set?
2. **Decision value:** could it change which candidate should be recommended?
3. **Risk value:** could it prevent a false-feasible choice or reveal that the evidence is insufficient?

The optimizer must not receive high-fidelity truth, sealed shifts, or future validation observations when choosing actions. Those are evaluator/auditor inputs only.

## 3. Decision state and action contract

At each decision step, the policy receives a frozen observation history, available candidate/fidelity actions, explicit costs, objective and constraint semantics, and only the predictions/uncertainty it is genuinely entitled to use.

A decision state should expose, at minimum:

- candidate identity and design-variable vector;
- observed objective and constraint values by fidelity, with missingness/failure states explicit;
- current recommended candidate or feasible Pareto set, if any;
- nominal prediction and uncertainty fields, including their source/method and calibration status;
- declared feasibility margins and thresholds;
- known cost schedule and observed wall/CPU time;
- policy configuration and history digest.

An action is a tuple of candidate, fidelity, and any required replication/measurement choice. The policy returns one of: evaluate, escalate fidelity, replicate, abstain/request review, or stop. Unsupported actions must be rejected instead of silently mapped to another fidelity.

## 4. Candidate decision-value acquisition

Use this as a policy contract, not as an assertion that the current selector already computes a posterior.

Let D be the current evidence state, a a permissible evaluation action, Y_a its not-yet-observed outcome, d(D) the current design recommendation, L(d, z) a preregistered loss on an evaluator-only reference state z, and C(a) the full marginal cost of performing and processing the action.

For a method with a defensible predictive distribution:

**Current decision risk:**  
R(D) = E_z[L(d(D), z) | D]

**Expected value of information:**  
VOI(a | D) = R(D) - E_{Y_a | D}[R(D updated with Y_a)]

**Cost-normalized score:**  
score(a | D) = VOI(a | D) / C(a)

The reference state z is used only to evaluate policy quality, never as an input to action selection. The expectation over possible Y_a must use the policy's actual predictive model. Do not manufacture a probability distribution for a deterministic heuristic.

Use a declared decomposable loss (or independently report the components) covering:
- feasible-set quality / normalized feasible regret;
- false-feasible decision penalty tied to declared consequence;
- instability of the recommended design under plausible outcomes;
- constraint-margin severity and uncertainty.

Avoid double-counting the same failure in multiple weighted terms without a documented rationale. Freeze scaling, loss weights, objective normalization, reference point and any thresholds before unsealing shifts. Report sensitivity to weight choices rather than tuning weights on the holdout.

### Exact discrete Bayesian reference calculator

The module `tools/casting-benchmark/discrete_decision_voi.py` provides an exact finite-state reference calculation for a policy with an explicit prior, decision-loss table and likelihood model for each candidate observation. It computes current Bayes risk, outcome probabilities, posterior Bayes risk, expected value of information (VOI), and VOI per cost. It does not know the realized latent world or read sealed holdout outcomes; the caller supplies the *predictive* prior/likelihoods available to the policy.

Known-answer example: with two equally likely worlds and a loss of 10 for choosing the wrong decision, current Bayes risk is 5. A 90%-accurate query reduces expected posterior risk to 1 (VOI 4); at cost 2 its VOI/cost is 2. A perfect query yields absolute VOI 5, but at cost 5 its VOI/cost is 1, so the noisy query is correctly ranked first when budget is scarce. These are mathematical fixture results, not casting results.

The calculator is a reference oracle for validating future probabilistic acquisition implementations. It assumes the supplied distributions and loss table are appropriate; it does not calibrate them, learn them, establish model correctness, or demonstrate that a real Symthaea selector can achieve the computed VOI. The deterministic gate and this exact-discrete calculator are deliberately kept distinct.

### Non-probabilistic policies

Where a baseline has no calibrated posterior, compare it fairly through a labelled non-probabilistic decision proxy or native action rule. Possible diagnostics include ranking-gap margins, observed residual envelopes, and deterministic constraint-margin screens. These diagnostics must not be renamed “calibrated VOI.” Evaluate whether the proxy actually ranks useful queries on separate known-answer fixtures before using it in a benchmark.

## 5. Stop, escalate, replicate, abstain

Do not reduce stopping to “no positive expected improvement.” A candidate can have little expected objective improvement and still carry substantial constraint risk.

- **Stop:** no admissible action exceeds the preregistered marginal decision-value threshold, the frozen quality target is met, or the cost cap is reached.
- **Escalate:** a cheap fidelity cannot resolve a decision-critical ranking or feasibility ambiguity and a more expensive measurement has positive expected risk-adjusted value.
- **Replicate:** measurement noise or stochastic solver output makes a replicate more decision-relevant than evaluating a new design.
- **Abstain / request review:** the recommended design cannot meet the declared evidence or risk gate. Abstention is a valid outcome, not a hidden optimizer failure and not proof that the design is safe.
- **Continue search:** a feasible-quality improvement remains worth more than its fully accounted cost.

Every transition needs a reason code, policy version, state digest, action/cost estimate, threshold, and output digest. Record both intended and realized costs. Cost prediction error must be visible.

## 6. Full cost contract

Record at least these distinct quantities:

1. Objective/constraint evaluations per fidelity, with immutable fidelity IDs.
2. Cost units per evaluation and the frozen mapping to high-fidelity-equivalent units.
3. Actual wall-clock and CPU time where available; GPU time and peak memory where measured.
4. Acquisition and surrogate fitting time.
5. Initialization, hyperparameter/configuration tuning, failed attempts and restarts.
6. Nominal reference-prediction work.
7. Shift-audit and independent validation work.
8. Human review / physical test cost when available; otherwise mark unavailable rather than zero.

Publish both the **policy search budget** and the **end-to-end full cost**. Do not subtract the cost of audits that are necessary to make the recommendation trustworthy. For fair algorithm comparisons, give all policies the same allowed objective/constraint oracle budget and use an additional full-cost view that includes method overhead.

## 7. Benchmark tracks and comparators

### A. Standard / shared-assumption track

Constraints are identical across fidelity levels when that is an assumption of the comparator. Use matched tasks, seeds, initialization opportunities, reference points, costs, and stopping budgets.

Required controls:
- high-fidelity-only;
- low-fidelity-only;
- random / fixed-fidelity;
- current Symthaea policy;
- a cost-aware stopping comparator when a reproducible implementation is available;
- MF-SCBO with immutable source revision/environment;
- a multi-objective multi-fidelity method such as BoTorch MF-HVKG/MOMF when its assumptions and implementation can be reproduced.

### B. Fidelity-dependent constraint stress track

The high- and low-fidelity constraint outcomes may disagree. Methods whose declared assumptions require shared constraints are still reportable here, but results must be labelled **out-of-assumption stress** and not combined with the standard track into a single score.

### C. Shift-robustness track

Use the versioned shift families from the decision-robustness contract, sealed until policies/configurations are frozen. Include zero-shift, parameter drift, boundary drift, localized defect, structural discrepancy, discretization and measurement-regime changes. Add simulator-backed families only after parameter ranges, solver identity, numerical tolerances, outcome units and costs are frozen.

### D. External public benchmark track

Use adapters to public benchmark suites only after licensing, semantics, fidelity mapping, objective/constraint direction, transformations, and known optima are checked. Candidate suites:
- the fourteen analytic multi-fidelity problems and experimental protocol described at https://doi.org/10.1007/s11831-025-10392-8;
- COCO BBOB-constrained (54 functions) from https://coco-platform.org/data-archive/index.html;
- reproducible benchmark tasks released with a locked MF-SCBO revision.

Keep these results separate from the compositional casting fixtures and any physical foundry claims.

## 8. Preregistered primary and secondary metrics

Primary:
- feasible hypervolume versus total-cost curve and area under the cost-normalized curve;
- cost to reach a frozen feasible-quality target, including right-censored failures to reach it;
- paired per-task finite-pool feasible hypervolume retention/regret for each shift;
- worst-shift and lower-tail performance, not only means;
- feasibility and false-feasible outcomes on independent validation, with uncertainty procedures appropriate to the sampling units.

Secondary:
- number and fraction of evaluation actions that change the final recommendation;
- decision flips avoided by escalation or abstention;
- evaluations allocated to candidates near constraint boundaries;
- prediction calibration/coverage only when true probabilistic output and an appropriate held-out design exist; for VOI, separately validate the observation likelihoods/prior with held-out predictive log score or Brier score and reliability checks, since a mathematically correct VOI calculation can still rank queries badly under a misspecified model;
- nominal regression, wall time, memory, and cost accounting discrepancies.

Finite candidate-pool regret is not global regret. Synthetic candidates/shifts do not establish physical risk bounds. Candidate × shift rows are dependent; do not treat them as IID observations to manufacture narrow confidence intervals. Use a preregistered cluster-aware procedure over independently generated tasks/seeds and state exactly what the independent unit is.

## 9. Provisional gates (targets, not results)

Treat these as aspirational acceptance targets to freeze before the main run:

- >=5% median improvement in cost-normalized feasible-quality AUC over the strongest reproduced baseline;
- win on >=70% of preregistered tasks with paired per-task analysis;
- no more than 5% median regression on any critical task family;
- >=20% lower total cost to the same frozen quality target where both methods reach it;
- publish all paired task results, uncertainty intervals, tail metrics, ablations and failures.

These gates do not imply adequate statistical power. If the test set cannot distinguish the target effect, call it inconclusive. Do not tune the thresholds after viewing holdout results.

For physical false-feasible risk, define the independent experimental unit, sampling design, consequence threshold and confidence method before evaluating. Do not convert synthetic pass rates into physical risk claims.

## 10. Required ablations and hostile controls

- objective-value-only acquisition;
- constraint-risk-only acquisition;
- joint decision-value acquisition;
- joint policy without cost-aware stopping;
- joint policy without escalation;
- joint policy without abstention;
- full cost versus high-fidelity-call-only accounting;
- no-shift and known-shift decision reversal controls;
- hidden-fidelity-truth leakage attempt;
- holdout-seed/configuration leakage attempt;
- zero-cost or negative-cost action rejection;
- missing/duplicate evaluation, NaN/Inf, solver failure, timeout and incompatible fidelity IDs;
- a case where the most valuable action is replication rather than a new candidate;
- a case where every available action is too costly and the correct outcome is stop/abstain.

Check that a recommendation does not become safer merely by excluding difficult candidates from the reporting denominator. Every selected candidate × shift pair and every attempted evaluation needs a disposition.

## 11. Deliverables in implementation order

1. **Contract and known-answer fixtures:** define decision loss, policy action API, cost semantics, abstention and failure behavior.
2. **Trace ledger:** store candidate/fidelity action, observation, expected/actual cost, decision before and after, reason code and immutable state digests.
3. **Simple baselines first:** fixed allocation, high-fidelity-only, cost-aware baseline and transparent decision proxy. Avoid making a bespoke complex policy the only comparator.
4. **VOI policy and ablations:** implement one change at a time and run known-answer tests.
5. **Paired-cost experiment runner:** lock seeds, suite revision, source/environment digest, stopping rules and report format.
6. **Independent audit:** verify receipts and holdout isolation; compare cost traces and decision metrics independently.
7. **Simulator/physical integration:** only after separate parameter, metrology, simulator-discrepancy and process-validation contracts are complete.

## 12. Literature and industry context

- Palazzolo, Binois & Giraldi, **MF-SCBO** (2026 preprint): https://arxiv.org/abs/2609.29941; implementation https://github.com/Nemo-Virtual-Lab/MF-SCBO.
- Xie et al., **Cost-aware Stopping for Bayesian Optimization**, ICML 2026: https://proceedings.mlr.press/v306/xie26c.html.
- Aydogdu et al., **Consequential improvement acquisition function for efficient multi-fidelity Bayesian optimization**, published 27 March 2026 / issue September 2026: https://doi.org/10.1007/s11081-026-10078-5. This already jointly selects a sample/fidelity around consequential high-fidelity improvement; include as a comparator if an executable implementation can be obtained and locked. The paper reports code/data available upon reasonable request rather than a clearly linked public implementation. It notes future work on multi-objective extension and more complex cost structures, making multi-objective casting robustness a better differentiator than claiming generic joint sample/fidelity selection as new.
- NIST AMS 100-85 (5 October 2026), risk-based uncertainty quantification and VVUQ: https://doi.org/10.6028/NIST.AMS.100-85.
- Multifidelity benchmark methodology and fourteen analytical problems (2025): https://doi.org/10.1007/s11831-025-10392-8.
- COCO constrained suite: https://coco-platform.org/data-archive/index.html.
- Metal casting AI/digital twin review (2026), reproducibility/UQ/generalization gaps: https://doi.org/10.1007/s43939-026-00685-5.
- MAGMASOFT autonomous casting optimization is an existing industrial baseline, including multi-objective optimization and robust process windows: https://www.magmasoft.com/en/solutions/magmasoft/.

## 13. First executable slice (2026-10-28)

The first deterministic policy gate is implemented in `tools/casting-benchmark/decision_value_gate.py`, with known-answer and hostile-input tests in `tools/casting-benchmark/test_decision_value_gate.py`. The gate ranks permitted actions by caller-supplied decision-loss-reduction proxy divided by explicit marginal cost. When a risk gate is unresolved, it only admits actions meeting a positive predeclared normalized risk-reduction proxy; when none is available within budget, it abstains instead of silently declaring a safe design. It also supports explicit replication and escalation actions, deterministic tie-breaking, and a cost reservation in the returned decision receipt.

A separate JSONL action/outcome ledger is implemented in `tools/casting-benchmark/decision_action_ledger.py`, with tests in `tools/casting-benchmark/test_decision_action_ledger.py`. Each record binds a canonical event to the previous record hash, records state and policy SHA-256 digests, and links evaluation outcomes to their originating actions. The verifier recomputes the hash chain and reconciles estimated with actual cost, while detecting byte tampering, duplicate outcomes, missing decision links and partial trailing records. It is single-writer only and tamper-evident, not a signature, trusted timestamp, or non-repudiation system.

`tools/casting-benchmark/forecast_validation.py` now scores forecast distributions against later outcomes in a separate evaluator: multiclass Brier score, log loss without probability clipping, zero-probability surprise counts, top-label reliability bins, and per-regime scoring. Every outcome must bind to the exact canonical forecast digest and a digest of the outcome fields. This detects post-hoc forecast alteration and allows nominal-versus-shift strata to be reported, but does not prove the forecast was published before the outcome, the samples are independent, or the model is calibrated under future shifts. Independent timestamps/signatures and data-collection controls would be needed for stronger provenance.

These are small transparent components, **not** a trained optimizer or evidence of calibrated model probabilities/performance. A caller must ensure its proxy inputs derive only from information visible at decision time. The configured risk-reduction proxy is not a probability or physical risk bound. The deterministic gate, exact-discrete Bayesian VOI calculator, forecast scorer, hash-chained action ledger, and integration fixture each have dedicated standard-library unittest modules wired into focused CI; source/test execution results must be tied to the current exact head SHA.

## 14. Claim ceiling

This contract does not establish that a decision-value acquisition rule is novel, that the code is implemented, that a benchmark target has been met, or that a physical casting is safe. Synthetic known-answer success validates tested semantics only. Any physical claim requires independently traceable measurements, quantified uncertainty, simulator-to-physical discrepancy evidence and relevant process qualification.
