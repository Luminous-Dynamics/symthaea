# Casting optimization research frontier — 2026-10-28

## Executive position

The promising field-level shift is **from “find a low-cost proxy optimum” to “buy the next piece of evidence that most reduces risk-adjusted manufacturing decision regret.”** This should jointly reason about design, fidelity, constraint confidence, deferred validation cost, robustness and information transfer between process regimes.

This is a research agenda, not evidence that the current Symthaea optimizer beats an established baseline. Current casting-like tests are deterministic synthetic fixtures. The 9,072-run default matrix is a plan; it has not been run.

## What the field already does

- **MF-HVKG** uses a one-step look-ahead acquisition to choose design/fidelity for Pareto-set knowledge and hypervolume under posterior mean.
- **MOMF** adds a fidelity/trust objective to multi-objective optimization.
- **MF-JESMOC** targets constrained multi-objective multi-fidelity optimization and selects both design and fidelity using cost-normalized joint-entropy reduction; the paper reports experiments on synthetic and real-world problems, and links public code.
- **Consequential improvement (CI)** is a recent acquisition method for multi-fidelity Bayesian optimization that accounts for deferred high-fidelity validation cost. Its cited paper positions its multi-objective extension as future work, so do not call the Symthaea heuristic an implementation of CI.
- **COCO BBOB-biobj** offers 55 bi-objective functions with official performance assessment. It is a complementary test of optimizer quality, not a casting simulator.

References:
- BoTorch MF-HVKG / MOMF tutorial: https://botorch.org/docs/v0.13.0/tutorials/Multi_objective_multi_fidelity_BO
- MF-JESMOC paper: https://doi.org/10.1016/j.neucom.2025.131674
- MF-JESMOC public code: https://github.com/fernandezdaniel/MOBOCMF
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

## What is implemented now vs still open

Implemented in this branch: an adaptive heuristic for joint candidate/fidelity choice; local residual correction; a deferred validation-cost proxy; per-action audit trace; independent action replay; and tests authored for budget response and tamper rejection.

Not yet demonstrated: local test execution, any full sweep, advantage over DIWA / MF-HVKG / MOMF / MF-JESMOC, calibration of its uncertainty, validated casting-simulator performance, or physical foundry benefit. The method should be revised or discarded if matched-cost trials do not support the hypothesis.
