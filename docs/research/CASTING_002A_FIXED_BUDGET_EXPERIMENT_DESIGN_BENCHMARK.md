# CASTING-002A — Fixed-Budget Experiment Design and Multi-Fidelity Benchmark

**Status:** Preregistered research contract; implementation and results pending  
**Depends on:** CASTING-001A (#8351), CASTING-001B (#8390), CASTING-001C (#8404 / PR #8406)  
**Claim ceiling:** Benchmark protocol only; no optimizer, model or process has yet won this benchmark

## 1. Research question

Which method produces the best **validated, constraint-satisfying casting decision per unit of total engineering cost**, given the same starting information and a fixed budget for experiments and simulations?

The benchmark must distinguish four questions that are often collapsed:

1. Does a model predict a measured quantity accurately on a held-out population?
2. Does its uncertainty help make better decisions?
3. Does the experiment-selection method reach a declared quality target with fewer expensive evaluations?
4. Does the resulting physical process improve yield, accepted-part cost, energy or cycle time without violating required quality or safety constraints?

Evidence for one question must not be represented as evidence for the others.

## 2. Industry and research baseline

Casting is already a mature simulation and process-optimization field. Commercial systems such as MAGMASOFT support filling, flow, solidification/feeding analysis and parameter optimization. A September 2026 foundry case report describes 31 simulation versions for a particular part and reports a 51% defect-rate reduction; this is a vendor/customer case report, not a universal benchmark and not independently verified by this repository.

Academic small-data optimization is also a substantive baseline. Pei et al. (2025) reported revised Latin-hypercube sampling plus Bayesian optimization and phase-field simulation for a specific squeeze-casting application: an initial dataset of 25 observations followed by six experimental iterations, with the best reported UTS and elongation exceeding the initial dataset's best values by 17.6% and 18.4%. That target process, alloy, component and pair of properties differ from the initial sand-cast coupon / solidification-time model here. Reproducing the methodology comparison is valid; copying its numeric material-property gains as our target is not.

A 2026 review of casting computation/AI/digital twins identifies reproducible benchmarking, uncertainty quantification, generalization across geometry/alloy/process populations, interoperability and industrial validation as persistent gaps. NIST's October 2026 uncertainty report calls for credibility assessment scaled to model influence and decision consequences.

References:
- Pei et al. (2025), *Optimizing casting process using a combination of small data machine learning and phase-field simulations*, npj Computational Materials 11, 27: https://www.nature.com/articles/s41524-025-01524-6
- Bayesian optimization-driven CFD optimization for continuous casting (30 August 2026) reports performance on par with commercial CAESES for submerged-entry-nozzle geometry optimization using a single-phase water approximation. It is an algorithm-comparison baseline, not physical validation of molten-metal casting: https://www.sciencedirect.com/science/article/pii/S0045793026002276
- Essam et al. (2026), *A review of computational modeling, artificial intelligence, and digital twins in metal casting and foundry operations*: https://link.springer.com/article/10.1007/s43939-026-00685-5
- NISTIR 8620 (2026), *Digital Twins Workshops Summary Report*: https://www.nist.gov/publications/digital-twins-workshops-summary-report
- NIST AMS 100-85 (2026), *On Methods of Uncertainty Quantification in Computational Modeling and Simulation: Harmonizing Modeling and Metrology Practices*: https://www.nist.gov/publications/methods-uncertainty-quantification-computational-modeling-and-simulation-harmonizing
- PI Castings (18 September 2026), report of MAGMASOFT adoption and a part-specific result: https://pi-castings.co.uk/introduction-of-simulation-capabilities/

## 3. Paradigm shift: score decisions, not just predictions

The proposed differentiation is not to claim a novel optimizer before testing it. It is to produce a reproducible *cost–quality–risk frontier* linking:

- process/model assumptions and applicability;
- low- and high-fidelity predictions;
- physical observations and inspection;
- calibrated uncertainty and domain-shift detection;
- proposed experiments and why they were selected;
- actual cost, time, energy, yield and accepted-part disposition.

The comparison must make it impossible to win by using more expensive trials, dropping failed/infeasible experiments, abstaining without penalty, tuning against the final holdout, or improving one property while hiding deterioration in another.

## 4. Methods to compare

Every method sees the same initial observations, valid candidate domain, objective/constraint definitions and final holdout. Include only baselines compatible with the benchmark task.

| Family | Baseline / candidate | Role |
|---|---|---|
| Analytical | Fixed Chvorinov exponent n = 2 | Transparent predeclared reference for solidification time where applicable |
| Analytical | Learned-exponent Chvorinov model | Existing empirical baseline; not a filling/defect solver |
| Sampling | Random feasible sampling | Budget-matched lower-complexity control |
| DOE | Factorial / central-composite / response-surface design where valid | Established structured experimental-design baseline |
| Space filling | Latin hypercube; revised/valid-combination LHS when applicable | Reproduces relevant small-data sampling baseline |
| Single fidelity | Bayesian optimization with a frozen surrogate/acquisition implementation | Adaptive selection using a single declared evaluation fidelity |
| Multi fidelity | Surrogate-assisted or multi-fidelity Bayesian optimization | Candidate only when meaningful low/high-fidelity sources and their cost/discrepancy are available |
| Engineering comparator | Documented foundry-expert choice and/or validated commercial solver workflow | Include only under a matched task, with source/configuration and resource burden recorded |

Do not rank a Chvorinov formula against a multiphysics solver as though they solve the same task. Chvorinov predicts a narrow solidification-time proxy. Full filling, turbulence/air entrainment, thermal fields, feeding, shrinkage, microstructure and residual stress are separate modeled quantities that require appropriate model and validation evidence.

## 5. Synthetic harness before physical trials

Create a versioned, deterministic function suite with explicit known optimum and constraint truth. Synthetic cases qualify the benchmark harness and optimization semantics only.

### Required cases

1. **Smooth single optimum:** a simple low-dimensional objective with known optimum and a feasible region.
2. **Nonlinear interactions:** factors interact; additive or one-factor-at-a-time methods are inadequate.
3. **Multi-objective conflict:** quality endpoints trade off, yielding a known or numerically characterized Pareto frontier.
4. **Constrained feasible island:** invalid regions and costly failures exist; the method must not optimize by evaluating illegal candidates.
5. **Heteroscedastic noise:** observation variance depends on the input; uncertainty-aware selection is tested under unequal measurement noise.
6. **Multi-fidelity discrepancy:** cheap and expensive objective sources correlate imperfectly; the low-fidelity model has a controlled bias.
7. **Domain shift:** the held-out geometry/process family changes selected function terms or parameter distributions.
8. **Sensor degradation:** missing, noisy or drifting observations challenge calibration and decision robustness.
9. **Censored resource cost:** runs can fail, time out, or consume different amounts of time/resource; incurred cost remains counted.

Freeze function versions, domains, noise process, fidelity relationship, reference optimum, feasibility rules and pseudo-random seeds. Include known-answer tests that verify the suite itself, plus negative controls for leakage and incorrect optimum/constraint data.

## 6. Budget and run protocol

A proposed harness starts with at least 100 paired seeds per synthetic case; if runtime precludes this, declare the smaller count before results are inspected and publish run-level results. No post-hoc seed selection.

For each seed:

1. Freeze one shared initial dataset and candidate set.
2. Run each eligible method using the same available information.
3. Impose equal budgets for physical-trial equivalents and separately itemized low-/high-fidelity calls. A multi-fidelity strategy may trade cheaper calls against expensive ones only through a predeclared cost schedule.
4. Use a separate untouched final evaluation set; optimizer query/stop decisions cannot observe its outcomes.
5. Preserve the entire run, including failures, invalid recommendations, constraint violations and early stops.
6. Repeat paired conditions, changing only the method under comparison.

Cost-accounting dimensions must be visible separately:
- physical experiments and destructive tests;
- high-fidelity solver calls;
- low-fidelity/surrogate calls (surrogate execution still has compute cost);
- solver and model preparation, mesh setup and calibration work;
- wall-clock time, peak memory and, where measurable, compute energy;
- instrumentation, metrology and calibration labor;
- foundry/operator and expert-review hours;
- failed tests, rework and material consumed.

Never convert simulations into “physical samples saved” without reporting both counts.

## 7. Metrics and proposed promotion target

### Primary decision-quality metrics

- **Cost to feasibility:** cumulative cost until the first candidate satisfying every predeclared hard constraint.
- **Cost to target:** cumulative cost to reach the predeclared quality target; report runs that never reach it as censored failures, not zero cost.
- **Simple regret:** distance from the known optimum on synthetic cases at fixed cumulative cost.
- **Best-feasible-objective / Pareto hypervolume versus cost:** integrate the feasible incumbent curve using one frozen normalization/reference point.
- **Constraint violations:** count, severity and incurred cost; required safety/quality constraints are not compensated by a weighted score.

### Secondary credibility and operations metrics

- frozen holdout MAE/RMSE and endpoint-specific metrics;
- empirical prediction-interval coverage and width;
- error at matched prediction coverage, plus natural coverage and abstention fraction;
- degradation under held-out geometry/process families and controlled drift;
- robustness to missing/noisy sensor channels and solver disagreement;
- convergence variability across seeds, not just mean best result;
- total experiment cost, compute time, memory, energy and human effort.

### Proposed research bar (not an industry standard)

For a candidate to earn a *synthetic benchmark improvement* label, predeclare the specific comparison and require at least a 20% reduction in median cost-to-target versus the strongest eligible baseline across the frozen suite, with a paired 95% uncertainty interval supporting improvement and no increase in hard-constraint violation. Publish each case separately: an aggregate may not hide regressions on an important case.

This threshold is a proposed internal promotion gate, not a demonstrated result or universally valid acceptance criterion. If the candidate fails it, report the result plainly. If baselines tie, prefer the simpler/cheaper/clearer method.

## 8. Physical validation track

A physical study cannot start from synthetic success as though it had validated a foundry process. Before the physical protocol is frozen, the responsible foundry engineer and metrology reviewer should specify:

- one named non-critical part/coupon and exact alloy/process/tooling population;
- valid parameter ranges and the hard material/inspection constraints;
- independently reviewed instrumentation, calibration, measurement uncertainty and defect classification;
- trial/batch/geometry units and a sample-size or sequential stopping plan;
- a frozen control recipe/process and a prospective candidate-selection policy;
- all outcome definitions and exclusion rules;
- energy, yield, scrap/rework, cycle-time and accepted-part cost accounting;
- an explicit stop/review rule for missing critical sensors, out-of-domain inputs, uncertainty beyond tolerance, unexpected defects or unsafe conditions.

Report process effects with uncertainty and all attempted runs. Keep quality, cost, energy, yield and safety evidence separate; do not roll them up into one score until weights and veto constraints are justified in advance. Any actual machine execution remains within qualified foundry procedures and existing fabrication/HAL/operator authority.

## 9. What “better than SOTA” would mean

Do not claim superiority against a named solver merely because our software is open, integrates HDC/AI, or generates a plausible prediction. A defensible claim requires a matched task, documented competitor configuration, identical data split and decision budget, comparable output definitions, competent independent validation, and transparent compute/setup/human costs.

Potential differentiators worth testing:
- **evidence-first optimization:** every recommendation retains the exact model, split, applicability domain and source lineage;
- **cost-aware multi-fidelity planning:** explicitly choose when a cheap approximation is adequate and when an expensive solver or physical trial is worth buying;
- **risk-aware feasibility:** prioritize meeting critical properties and constraint robustness over nominal objective alone;
- **shift-aware abstention:** detect unsupported geometry/alloy/tooling populations and require review or higher-fidelity evidence;
- **outcome-based proof:** demonstrate reductions in physical trials, defect/rework, cost or energy on prospective matched comparisons.

Each bullet is a hypothesis to test, not a result.

## 10. Architecture boundary and validation state

Compose with CASTING-001A/B/C and MFG-PROC #5686. Do not create a new manufacturing process registry, material identity authority, metrology authority, recipe system, or machine-execution path.

This document preregisters an evaluation strategy. The first test-only synthetic harness is drafted in [CASTING-002B PR #8441](https://github.com/Luminous-Dynamics/symthaea/pull/8441). Follow-on drafts now include a nearest-neighbor adaptive heuristic and a fixed-hyperparameter GP-LCB baseline with a Rastrigin multimodal fixture. These changes have not been compiled or executed in this authoring environment; no benchmark result, commercial solver head-to-head, physical experiment, or production benefit is claimed. Freeze scoring refinements before inspecting comparative benchmark results.
