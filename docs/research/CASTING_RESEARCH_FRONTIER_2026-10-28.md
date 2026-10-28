# Casting optimization research frontier — 2026-10-28

## Executive position

The promising field-level shift is **from “find a low-cost proxy optimum” to “buy the next piece of evidence that most reduces risk-adjusted manufacturing decision regret.”** This should jointly reason about design, fidelity, constraint confidence, deferred validation cost, robustness and information transfer between process regimes.

This is a research agenda, not evidence that the current Symthaea optimizer beats an established baseline. Current casting-like tests are deterministic synthetic fixtures. The 9,072-run default matrix is a plan; it has not been run.

## What the field already does

- **MF-HVKG** uses a one-step look-ahead acquisition to choose design/fidelity for Pareto-set knowledge and hypervolume under posterior mean.
- **MOMF** adds a fidelity/trust objective to multi-objective optimization.
- **MF-JESMOC** targets constrained multi-objective multi-fidelity optimization and selects both design and fidelity using cost-normalized joint-entropy reduction; the paper reports experiments on synthetic and real-world problems, and links public code.
- **MF-SCBO** (preprint posted 24 September 2026) extends scalable constrained Bayesian optimization to multiple fidelities, including non-nested sampling. It is a relevant new comparator for constrained and higher-dimensional settings, but its task assumptions still need to be matched explicitly.
- **Consequential improvement (CI)** is a recent acquisition method for multi-fidelity Bayesian optimization that accounts for deferred high-fidelity validation cost. Its cited paper positions its multi-objective extension as future work, so do not call the Symthaea heuristic an implementation of CI.
- **COCO BBOB-biobj** offers 55 bi-objective functions with official performance assessment. It is a complementary test of optimizer quality, not a casting simulator.

References:
- BoTorch MF-HVKG / MOMF tutorial: https://botorch.org/docs/v0.13.0/tutorials/Multi_objective_multi_fidelity_BO
- MF-JESMOC paper: https://doi.org/10.1016/j.neucom.2025.131674
- MF-JESMOC public code: https://github.com/fernandezdaniel/MOBOCMF
- MF-SCBO paper: https://arxiv.org/abs/2609.29941
- MF-SCBO public code: https://github.com/Nemo-Virtual-Lab/MF-SCBO
- NIST (2026), risk-based VVUQ methods for computational modelling and simulation: https://doi.org/10.6028/NIST.AMS.100-85
- Conformal Policy Control (ICML 2026): https://proceedings.mlr.press/v306/prinster26a.html
- Consequential improvement paper: https://doi.org/10.1007/s11081-026-10078-5
- COCO archive: https://coco-platform.org/data-archive/index.html
- Multi-fidelity BO experimental best-practices review: https://www.nature.com/articles/s43588-025-00822-9

## Paradigm shifts to test

### 1. Candidate-and-fidelity is one action

Do not commit to a static low/medium/high quota. Compare every currently legal pair `(design, fidelity)` at each decision. Include immediate cost **and the expected follow-on validation cost** of low/medium data; record the proxy and its calibration separately. The new `joint_action_cost_aware` policy is the first transparent heuristic for this hypothesis, not the endpoint.

The next algorithmic leap should replace its hand-built uncertainty/validation proxies with calibrated posteriors and decision regret reduction. Candidate method families include MF-HVKG, MF-JESMOC, correlated multi-task GPs or deep GPs, and calibrated discrepancy ensembles. Compare them rather than assuming one is best.

### 2. Calibrate error where the decision changes

Global RMSE can look good while a proxy reverses a Pareto ranking or marks an unsafe design feasible. Measure:
- objective rank inversions and Pareto-set recall at each fidelity;
- constraint sign errors and false-feasible promotions, stratified by margin-to-boundary;
- uncertainty calibration (Brier score, reliability plots / ECE, and interval coverage);
- performance under local bias islands, coordinate warps, discontinuities, heteroscedastic noise and unseen compositions.

A policy should abstain or pay for more fidelity when calibrated constraint confidence is insufficient. “No false feasible found in four promotions” is not evidence of a low false-feasible rate.

### 3. Optimize risk-adjusted real decisions, not only surrogate metrics

For casting, the eventual objective vector should be agreed with domain experts and mapped to measurable units. Candidate dimensions might include defect / yield risk, mechanical properties, cycle time, energy, material cost, rework and emissions. Constraints should express actual safety, equipment and product requirements. Do not combine incomparable dimensions with arbitrary weights merely to make one scalar leaderboard.

Report Pareto quality and decision regret alongside the consequence of wrong decisions. A tiny hypervolume gain is not beneficial if it incurs significant false-feasible risk or relies on an invalid proxy.

### 4. Treat uncertainty and shift as first-class axes

Evaluate transfer across alloy families, geometries, molds, process windows and solver regimes. Hold out entire mismatch families and contexts rather than randomly splitting points from the same synthetic function. Test whether the method:
- detects when cheap-fidelity correlation collapses;
- chooses to obtain high-fidelity anchors when calibration is weak;
- returns “insufficient evidence” instead of an ungrounded feasible recommendation;
- retains performance under cost changes, missing fidelities, and correlated errors across objectives and constraints.

### 5. Build a fidelity ladder tied to physical validation

The intended stack should be documented by actual sources of error, not just labels:
1. analytic / rule-based screening;
2. cheap reduced-order or coarse-grid simulation;
3. finer numerical simulation with convergence and solver-error estimates;
4. validated multiphysics / casting simulation;
5. prospective physical measurements.

At every transition record model revision, mesh / solver settings, material and boundary-condition assumptions, convergence evidence, measured fidelity correlation, and the accounting unit (compute-hours, machine-hours, material, foundry trial slots). Never pretend one universal “cost unit” represents all real expense; synthetic normalized units are only for controlled comparisons.

### 6. Make evidence independently replayable

The raw trace should be sufficient to reconstruct the action sequence, fidelity costs, observed archive and metrics. Hashes prove byte identity, not who ran a test or whether a simulator is physically valid. Keep source, environment, suite, initial design, random seeds, termination criteria and scoring convention frozen; keep failed, refused, and incomplete runs in the accounting.

## Benchmark ladder and gates

| Level | Required evidence | Main outputs | Gate |
|---|---|---|---|
| A. Correctness controls | Severity-zero exact-truth control, diagonal-composition equivalence, tamper/negative controls | Exact equality; independent replay accepts clean traces and rejects mutations | All authored tests execute and pass in a declared environment |
| B. Synthetic discrepancy stress | 6×6 objective/constraint compositions, dimensions 2/5/10, severity 0/0.5/1/2, more seeds | Feasible high-fidelity HV vs cost, false-feasible risk, regret, fidelity allocation | Exploratory 3-seed matrix first; then ≥20 paired seeds, target 30 |
| C. Public optimizer benchmarks | COCO BBOB-biobj with its native assessment; a separately locked multi-fidelity suite with known truth | COCO runtime/ECDF on COCO; matched-cost HV-AUC/target cost on the MF suite | Exact package/commit, adapter, parameters, and runtime environment locked |
| D. Casting simulator | Held-out designs and process regimes; solver convergence and validation data | Domain-unit quality, feasibility calibration, simulator wall clock, measured discrepancy | Protocol frozen before comparison; report simulator uncertainty separately |
| E. Prospective foundry study | Predeclared trial plan, independent metrology, controls, multiple runs / batches | Yield/defect / properties, energy and time, cost and failure modes | Physical claims only after measurements; synthetic or simulator performance is not a substitute |

### External comparison set

For multi-fidelity constrained multi-objective acquisition, require **MF-HVKG, MOMF and MF-JESMOC** once their code and environments are pinned. Also retain existing repository policies, high-fidelity-only, Sobol / random sampling, and a competitive evolutionary baseline such as NSGA-II. For the single-fidelity COCO control track, use suite-appropriate strong methods (e.g. qLogNEHVI / qLogNParEGO where applicable) and keep COCO-native metrics.

Do not report literature numbers side-by-side as if they were measured on the same instances and budgets. Re-run baselines on the same frozen tasks and cost model. The 2026 consequential-improvement paper is an auxiliary single-objective scalarization track until author code or a separately described implementation/environment is locked; it is not a drop-in multi-objective baseline.

## Proposed aspirational targets (not current results)

These are proposed engineering gates for a serious superiority claim, and must be frozen before result inspection:

1. **Quality per cost:** at least 10% lower median high-fidelity-equivalent cost to reach a predeclared feasible-HV target than the strongest matched external baseline, with paired confidence intervals supporting improvement after multiplicity adjustment.
2. **Anytime utility:** better area under the feasible-HV-versus-actual-cost curve, not just a better final point; publish per-task and aggregate curves.
3. **Safety:** false-feasible risk's one-sided 95% upper confidence bound no greater than 5% on a suitably powered held-out promotion set; use abstention / extra fidelity where the bound cannot be established.
4. **Robustness:** no hidden subgroup collapse on held-out discrepancy families; publish worst-case and lower-decile results as well as averages.
5. **Efficiency:** separately report optimizer wall time, peak memory, acquisition cost, and every fidelity call. Do not count only cheap evaluations or omit deferred validation.
6. **Reproducibility:** every claimed run has a replayable raw trace, immutable code/environment/suite hashes, declared seed, and retained failures.

These targets are not universal industry standards and do not, by themselves, prove industrial usefulness. They are proposed gates; obtain foundry stakeholder agreement on the actual quality, risk and economic outcomes before the physical track.

## New safety paradigm: promotion risk must be statistically controlled

A candidate that a cheap model labels feasible but high fidelity rejects is a **false-feasible promotion**. Treat this as a decision-risk event, not just another optimization metric. The new `promotion_risk_gate.py` computes the one-sided exact Clopper–Pearson upper confidence bound for this event rate and returns `allow` only when a 5% risk target is met by a predeclared **independent held-out validation sample**.

The key negative control: at 95% one-sided confidence, 0 failures in 8 independent Bernoulli trials still permits a failure probability of about 31.2%; 0/58 still does not establish a 5% upper bound; 0/59 reaches about 4.95%. Optimizer-selected promotion outcomes cannot certify their own risk because selection is adaptive. The scorer therefore reports a descriptive bound for its optimization trace but never treats that trace as independent validation. The guarantee also depends on representative/exchangeable validation data; under distribution shift it is not a deployment guarantee. Required action under unsupported shift: abstain, collect representative high-fidelity evidence, and re-calibrate before relying on the gate.

This is aligned with the risk-based credibility direction in NIST's October 2026 VVUQ report, and with 2026 work on conformal policy control. The present module is a small exact-binomial gate, **not** a complete conformal or distribution-free shift-robust guarantee.

## What is implemented now vs still open

Implemented in the current stacked branches: an adaptive heuristic for joint candidate/fidelity choice; local residual correction; a deferred validation-cost proxy; per-action audit trace; independent action replay; an exact binomial promotion-risk gate that requires independent held-out validation for certification; and tests authored for budget response, tamper rejection, known bounds and fail-closed behavior.

Not yet demonstrated: local test execution, any full sweep, advantage over DIWA / MF-HVKG / MOMF / MF-JESMOC, calibration of its uncertainty, validated casting-simulator performance, or physical foundry benefit. The method should be revised or discarded if matched-cost trials do not support the hypothesis.
