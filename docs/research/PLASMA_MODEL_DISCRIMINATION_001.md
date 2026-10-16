# PLASMA-MD-001 — Decision-aware model discrimination for plasma discovery

**Status:** Research protocol proposal. No solver runs, numerical results, or physical validation are claimed by this document.  
**Parent benchmark:** [Symthaea PLASMA-VV-001 #7588](https://github.com/Luminous-Dynamics/symthaea/issues/7588)  
**Reference qualification:** [PLASMA-REF-001 #7622](https://github.com/Luminous-Dynamics/symthaea/issues/7622)  
**Existing-module audit:** [PLASMA-CRED-001 #7600](https://github.com/Luminous-Dynamics/symthaea/issues/7600)  
**Execution/evidence contract:** [Symtropy PLASMA-TWIN-002 #1703](https://github.com/Luminous-Dynamics/symtropy/issues/1703)

## 1. Core shift: choose models by decision risk, not a universal fidelity ranking

A single ordered ladder—cheap model, medium model, expensive model—can be misleading. Different solvers can resolve different physics, and a model can be accurate for one observable but inadequate for another. The practical question is:

> Given this regime, design, requested observable, uncertainty, and engineering decision, what is the cheapest defensible next calculation or measurement?

Treat model selection as a **decision policy that is itself benchmarked**. A low-cost model can screen candidates where its applicability is supported. An explicit escalation policy requests a more detailed model or new observation where the low-cost model could change the decision, violate a constraint, or miss a known mechanism. It must abstain when no model has adequate support.

This is not a claim that an AI can infer a solver's validity from its outputs alone. The policy is allowed to use only declared capability envelopes, qualified benchmarks, measured/estimated uncertainty, retained model discrepancies, and source provenance. Out-of-envelope conditions stay out of profile until independently qualified.

## 2. Why this is now worth benchmarking

The existing external anchors illustrate three distinct error layers that should not be collapsed:

1. **Numerical error:** The Zapdos verification and validation study uses manufactured solutions to test spatial/temporal convergence and compares selected RF capacitively coupled plasma (CCP) results against LSODI and measurements. It reports good 1D code-to-code agreement, discrepancies in some 2D peak values, and low-pressure disagreement with selected experimental results. A converged solve is therefore not by itself proof of model adequacy. [Zapdos V&V paper](https://doi.org/10.1016/j.cpc.2023.108837) · [Mendeley dataset V1](https://doi.org/10.17632/j76jr9m66p.1)

2. **Model-form / closure error:** A 2026 2D study compares fluid and electrostatic PIC–MCC RF-CCP simulations in symmetric and asymmetric geometries. Its reported diagnostics include electron energy probability functions, nonlocal electron-temperature profiles and ion/electron pressure tensors. It describes regimes in which scalar or local fluid closures miss kinetic structure, particularly in low-pressure/nonlocal conditions and asymmetric geometries. Use these observations to design a model-discrimination challenge, not as fixed numeric tolerances until exact input cases, data, and comparison definitions have been pinned. [Kim, Lee & Lee, *Materials* 19(9), 1900 (2026)](https://doi.org/10.3390/ma19091900) · [Full article](https://pmc.ncbi.nlm.nih.gov/articles/PMC13164984/)

3. **Input-data / computational-chain error:** A 2026 cross-code study of LTE thermophysical properties for argon, krypton and xenon traced discrepancies through composition, collision integrals, input datasets and transport calculations, finding that source-data inconsistencies explained some errors. This is a different thermal-plasma regime from the low-temperature RF-CCP benchmark; its value here is methodological: trace each computed result back through the full scientific dependency graph, not just the top-level solver version. [Vagnoni et al., *Entropy* 28(7), 830 (2026)](https://doi.org/10.3390/e28070830)

A related 2026 study of uncertainty-aware ML for a microwave atmospheric-pressure plasma jet explicitly separates interpolated from extrapolated experimental conditions and reports prediction-interval metrics on independent operating conditions. That is a useful evaluation pattern, but its results must not be transferred to RF-CCP or another regime without testing. [RSC PCCP article](https://doi.org/10.1039/D5CP04364F)

A general precedent for combining fundamental-science and application-driven experiment choices is the 2025 PARADIGM framework, which optimizes experiment selection against uncertainties in shared nuclear data. This is not plasma evidence; it motivates optimizing for uncertainty reduction relevant to an actual downstream decision rather than for a generic model score. [Physical Review X (2025)](https://doi.org/10.1103/PhysRevX.15.021086)

## 3. Scope and non-goals

Initial scope is a source-pinned, low-temperature RF-CCP case family connected to the Zapdos benchmark program. Qualification applies only to its exact regime, model assumptions, source data and observables.

Do not combine results from the following regimes as though they were a single benchmark:

- low-temperature, non-equilibrium RF-CCP;
- atmospheric-pressure plasma jets;
- LTE thermal-plasma properties and arcs;
- magnetized MHD and fusion plasmas;
- laser-plasma and other kinetic regimes.

Cross-domain methods may be reused; numerical thresholds, closure assumptions and accuracy claims may not.

Non-goals: implementing a new in-house plasma PDE/PIC solver, direct hardware actuation, ignition or commissioning authority, treating Symthaea HDC/Φ scores as physical constitutive laws, or using generated synthetic data as experimental truth.

## 4. Evidence and uncertainty contract

For each model result, preserve separate fields for:

- **Model identity:** exact solver/revision, governing formulation, closure assumptions, options and documented omissions.
- **Data lineage:** each geometry, initial/restart state, species/reaction/transport table, cross section and calibration dataset with source, revision, license terms and content hash.
- **Numerical uncertainty:** discretization/refinement, time-step sensitivity, solver tolerances, convergence diagnostics and stochastic sampling error where applicable.
- **Parameter/input uncertainty:** source measurements, input ranges, posterior/covariance or other justified uncertainty representation.
- **Observation uncertainty:** instrument/calibration/source-specific uncertainty; missing uncertainty is marked unavailable, not zero.
- **Model-form discrepancy:** observable-aligned residuals between model classes or against independent data, with the regimes where comparison is valid.
- **Surrogate error:** training-source identities, held-out performance, calibration, applicability and extrapolation state.
- **Decision consequence:** which objective, constraint, or choice can change if the result is wrong.

Never compress these into one all-purpose confidence/Phi score. An error bar based on parameter variation is not automatically an estimate of model-form error. Agreement between two codes with shared input tables or shared assumptions is not fully independent evidence.

## 5. Proposed model-escalation policy

The benchmark should compare policies rather than presuppose a winning one.

### Policies to compare

1. **Always-low-cost:** use the qualified fluid model on every in-envelope candidate; abstain out of envelope.
2. **Always-high-cost:** use a qualified kinetic/PIC model for every case where an applicable implementation exists; count full compute cost.
3. **Static rule:** escalate using a preregistered regime indicator derived from published physics and model envelope (for example, a properly defined nonlocality indicator if the necessary inputs and assumptions are available).
4. **Uncertainty-aware rule:** escalate when predictive uncertainty or discrepancy estimates make a constraint or ranking decision unstable.
5. **Disagreement-aware rule:** request a high-fidelity comparison where qualified models disagree on an observable material to the decision.
6. **Decision-aware active rule:** pick the next solver run or observation by expected reduction in decision regret or expected discrimination between explicit competing model hypotheses per unit cost. Keep this distinct from optimizing the design objective itself.

The model-selection policy cannot rewrite validity envelopes at inference time. A policy that cannot justify a choice must return **abstain / insufficient evidence**.

### Cost ledger

Record more than solver evaluations: wall time, CPU/GPU type and hours, peak memory, particle count or mesh size where applicable, failed-run cost, preprocessing and artifact-verification cost. Report both wall-clock and normalized compute budget where fair normalization is defensible. Because fluid and PIC simulations do different work, a bare count of runs is insufficient.

## 6. Challenge-set design

Exact cases and numerical tolerances remain provisional until the source-pinned base case is qualified under #7622. Do not manufacture a synthetic reproduction of the literature and label it published reference data.

| Challenge | What it tests | Required output |
|---|---|---|
| Manufactured solution / refinement | Numerical implementation and convergence | Error norms, observed order, mesh/time-step series and solver diagnostics |
| Within-envelope interpolation | Baseline predictive accuracy | Per-observable residuals and uncertainty calibration |
| Held-out parameter region | Generalization, not random-row leakage | Performance and abstention by held-out regime/region |
| Symmetric vs asymmetric geometry | Whether the policy notices geometry-linked model limitations | Observable-specific fluid/kinetic discrepancies and capability rationale |
| Kinetic-sensitive observables | Whether scalar summaries conceal distribution/tensor differences | EEPF and temperature profiles; pressure-tensor quantities only where outputs and definitions are available |
| Input-table perturbation | Dependency-graph and sensitivity visibility | Changed dataset identity, predicted impact, recalculation requirement and retained old receipt |
| Non-converged but zero-exit process | Process-vs-numerical distinction | Explicit non-converged disposition; no completed-result promotion |
| Missing cross section / unknown reaction | Input completeness and solver capability | Unsupported/incomplete disposition, not default coefficients |
| Solver disagreement | Model-form review behavior | Both outputs retained, assumptions aligned or explicitly incomparable, no automatic averaging |
| Out-of-envelope case | Calibration and capability honesty | Abstention; no success-looking extrapolation |
| Surrogate false winner | Surrogate calibration and confirmation discipline | Exact-solver confirmation, miss retained in campaign ledger |
| Validation leakage | Independence of evidence | Contamination detected or claim ceiling reduced and run excluded from independent-validation claims |

A change to one required physical input, mesh, boundary condition, species dataset or solver revision must change the immutable scenario identity. Source-control hashes are not enough unless the full dependency closure is represented.

## 7. Preregistered metrics

Freeze the dataset split, observables, physical assumptions, objective/constraint functions, model-selection thresholds, budgets and analysis method before comparing policies.

### Scientific credibility

- observable-specific bias, MAE/RMSE or another justified error measure;
- spatial/temporal profile error as well as scalar/global error where reference data support it;
- normalized residuals using uncertainty only where that uncertainty is justified;
- convergence order and sensitivity to numerical controls;
- prediction-interval coverage and width, reported separately by regime and interpolation/extrapolation status;
- disagreement and model-form discrepancy, retained instead of hidden by an aggregate score;
- calibration/validation leakage and dependency provenance.

### Model-selection quality

- **false-safe escalation miss:** a low-cost model is accepted even though an applicable independent comparator or held-out observation crosses a predeclared material-error or engineering-decision threshold;
- **unnecessary escalation rate:** expensive model calls that did not materially change the qualified decision, while acknowledging that a run can still provide reusable evidence;
- **decision regret:** cost of the selected action/design relative to the best action supported by the preregistered reference/uncertainty model;
- **abstention quality:** unsupported cases abstained correctly, and qualified cases were not refused unnecessarily;
- **model-discrimination success:** fraction of preregistered, genuinely distinguishable hypothesis pairs for which the selected next observation/run separates predictions according to a predeclared criterion;
- **information gain per cost:** use a declared utility (e.g. expected entropy reduction, expected reduction in decision regret, or an appropriate optimal-design criterion), report which utility is used and why. Do not compare incomparable utilities as a single leaderboard score.

### Discovery efficiency

- feasible Pareto quality under frozen normalization;
- expensive solver-hours to hit a frozen decision-quality threshold;
- feasible-candidate rate and constraint violations;
- solver-call count alongside actual compute cost;
- robustness across independent seeds and held-out regions.

The previous aspirational goal of reducing expensive high-fidelity evaluations by at least 25% remains a **hypothesis to test**, not a benchmark result. It only counts if design quality is held constant on held-out cases, feasibility violations do not increase, failed evaluations remain counted, and variability supports the claim.

## 8. Acceptance goals and statistical discipline

1. Every submitted attempt has an explicit disposition, including failure, refusal, cancellation and incomplete output.
2. Every advertised reference or physical validation record has a complete provenance path, and all unverified sources are marked as unresolved.
3. All preregistered adversarial faults in the defined suite are detected. Report the corpus size and fault classes; 100% detection in a finite suite is not a proof against every possible attack.
4. No unsupported/out-of-envelope case becomes a successful model result through fallback defaults or serialization.
5. A zero-false-miss result in a small challenge set must include a one-sided confidence bound; zero observed failures must not be presented as zero real-world risk.
6. Report paired policy comparisons with per-seed data, effect sizes and uncertainty intervals. Use enough independent runs to support the prespecified power/precision; 20 seeds may be a screening study, not automatically a conclusive study.
7. An active experiment policy must beat a preregistered random/space-filling selection baseline under the same cost budget on held-out discrimination or decision regret. Report inconclusive if it does not.
8. No global plasma validity claim is inferred from a qualified low-temperature RF-CCP case.

Thresholds requiring expert judgment must be chosen before evaluation and tied to the intended observable/decision. If the available paper or measurement data do not provide enough information to predefine a defensible tolerance, state the limitation and treat that claim as not yet scoreable.

## 9. Execution order

1. Finish source/data provenance and case mapping in [#7622](https://github.com/Luminous-Dynamics/symthaea/issues/7622).
2. Implement typed receipts and fail-closed cases in [Symtropy #1703](https://github.com/Luminous-Dynamics/symtropy/issues/1703).
3. Independently reproduce the selected numerical reference and qualify the relevant observables.
4. Build the paired challenge set and register metrics / decision thresholds.
5. Run a cheap policy-screening study; retain all failures and abstentions.
6. Expand to independent seeds and held-out regimes once statistical precision justifies stronger claims.
7. Only then report any improvement in solver-call efficiency or scientific discovery quality.

No step depends on waiting for CI runners to start planning or auditing the next step. CI, local numerical reproduction, numerical verification, comparison with published data, and validation against physical measurements must still be reported as separate evidence states.

## References

- DeChant et al. (2023), “Verification and validation of the open-source plasma fluid code: Zapdos,” *Computer Physics Communications* 291, 108837. https://doi.org/10.1016/j.cpc.2023.108837
- DeChant et al. (2023), published dataset V1. https://doi.org/10.17632/j76jr9m66p.1
- Kim, Lee & Lee (2026), “Comparison of Kinetic and Fluid Simulation Models for RF Capacitively Coupled Plasmas in Semiconductor Processing,” *Materials* 19(9), 1900. https://doi.org/10.3390/ma19091900
- Vagnoni, Murphy & Ghedini (2026), “Cross-Code Verification for Improved Thermophysical Properties of Argon, Krypton and Xenon Plasmas,” *Entropy* 28(7), 830. https://doi.org/10.3390/e28070830
- Uncertainty-aware ML for a microwave atmospheric-pressure plasma jet (2026). https://doi.org/10.1039/D5CP04364F
- Neudecker et al., PARADIGM Collaboration (2025), “Machine Learning to Select Experiments Driven by Fundamental Science and Applications for Targeted Nuclear Data Improvement,” *Physical Review X* 15, 021086. https://doi.org/10.1103/PhysRevX.15.021086


## 10. Kinetic benchmark companion: WarpX / Turner four-case PIC-MCC

Source reconnaissance has now identified an open-source kinetic comparator with an existing published benchmark rather than a hand-built proxy:

- WarpX source tip inspected 2026-10-16: `4f2c08099faddfab540fb925a02f2c13bc5eacdd`.
- Existing driver: `Examples/Physics_applications/capacitive_discharge/inputs_base_1d_picmi.py`, blob `7fd5827b4f6c7126a9e27aeb00852e1c266edcb5`.
- Published-reference analyzer: `Examples/Physics_applications/capacitive_discharge/analysis_1d.py`, blob `d4c791e6a015dd6b8006abba7c8cf2587de044fb`.
- Test registration: `Examples/Physics_applications/capacitive_discharge/CMakeLists.txt`, blob `5e733eeeb4dea1405460f8992c01cbf35f677299`.
- WarpX itself uses a BSD-style license in `LICENSE.txt`; the cross-section data are separately licensed, so software and data permission must not be collapsed into one license field.
- Cross-section source repo tip inspected: `BLAST-WarpX/warpx-data@3033347525b9cf24cad99a724fb3d15345764840`. The helium directory documents electron-neutral/excitation/ionization tables from Biagi/MAGBOLTZ v7.1 and ion-neutral data from Phelps. Preserve that provenance and the dataset's separate Community Data License Agreement — Permissive 2.0; do not copy any files into Symthaea until license and contributor attribution are checked.

The upstream WarpX README describes four helium RF capacitive-discharge cases following Turner et al. (2013), with 6.7 cm gap and 13.56 MHz drive, and points to the published benchmark: https://doi.org/10.1063/1.4775084. The analyzer includes the 129-point reference time-averaged ion-density profile for Case 1, excludes wall-endpoint nodes due to a documented deposition artifact, and asserts interior RMS-relative-density error below 6.5%. The CMake `--test` mode is a lower-cost regression using 320 RF cycles, so it must be labelled smoke/regression mode and not presented as the full canonical-resolution benchmark.

**Critical separation:** WarpX's helium kinetic cases are not automatically comparable to the Zapdos fluid cases. First qualify each case independently. Only define a fluid-vs-PIC score if gas species, collision data, geometry, drive, boundary conditions, initial state, averaging window and observables are made commensurate; otherwise preserve them as distinct regime-specific benchmarks.

## 10.1. PIC-MCC convergence and stochastic error separation

A single close profile match is a first reproduction check, not sufficient numerical verification. Parodi & Petronio (2025) explicitly distinguish spatial discretization, time discretization and macroparticle statistical weight as contributors to PIC-MCC error, use repeated pseudo-random-number seeds, and apply a three-level refinement/error-estimation method. Their work also motivates staged simple/analytical checks before a fully coupled case. https://doi.org/10.1063/5.0241527

Add two deliberately separate gates:

**Gate A — canonical reference reproduction**
- Freeze the exact upstream WarpX commit, case number, original input parameters, every collision-data file and actual SHA-256 computed over the downloaded bytes, compilation/build flags, RNG seed, run duration, averaging interval and analyzer commit.
- Run the canonical upstream case, not the reduced `--test` setting. The test mode is useful for integrity and quick regression only.
- Compare the same published observable with the same masking rule: Case 1 time-averaged ion density on the analyzer's 129-point reference grid, preserving the documented endpoint exclusions.
- Report profile bias/error and overlay the spatial profile; report the upstream <6.5% interior RMS-relative-density criterion only as the existing reference criterion for that matched case/analyzer—not as a universal plasma accuracy tolerance.
- Run a screening set of at least five independent RNG seeds with all physical/numerical parameters otherwise frozen, report the mean profile and confidence intervals, and increase the replication count if uncertainty straddles the registered acceptance decision.

**Gate B — numerical convergence**
- Change `Δx`, `Δt`, macroparticle weight (or its equivalent particles-per-cell setting) and averaging duration as separately traceable experimental factors. Do not change all factors simultaneously and then attribute the difference to a single cause.
- Begin with at least three systematically refined levels for each numerical factor and at least five independent RNG seeds per level as a screening floor; grow the replication count when Monte Carlo uncertainty prevents a clear conclusion. This is a proposed project floor, not a universal theorem that five seeds are sufficient.
- Estimate stochastic scatter separately from systematic grid/time-step/particle-weight trends. Report confidence intervals over independent seeds for the registered quantity of interest.
- Estimate observed convergence order and a Richardson/GCI-style error only if the responses show an identifiable asymptotic convergence regime and the assumptions of that estimator are supported. If not, report non-asymptotic/inconclusive; do not force a convergence order onto noisy or non-monotone data.
- Preserve failures, divergent runs, seed-specific outliers and non-convergent refinement levels. Do not remove them just because the resulting average is harder to qualify.
- Use staged analytical or independently verified submodule tests wherever applicable before the coupled discharge benchmark, so an error source can be localized rather than attributed to the full physics stack.

The key output is not simply one number. It is a decomposition of **reference discrepancy**, **stochastic sampling uncertainty**, **spatial/temporal discretization error**, **particle-weight error**, and **model-form/physical discrepancy**, with evidence claims kept separate. Agreement among multiple codes with the same numerical settings is useful, but it does not alone establish that the shared numerical framework is converged or experimentally valid.

A new implementation/preflight work item tracks the next step: [PLASMA-PIC-REF-001 #7640](https://github.com/Luminous-Dynamics/symthaea/issues/7640). The immediate acceptance condition is exact dependency closure, independent SHA-256 computed over downloaded bytes, a frozen case manifest, and explicit canonical-vs-smoke status. No WarpX run or benchmark pass is claimed by this source inspection.

## 11. Measured disruption-prediction benchmark — separate evidence track

Disruption prediction is a useful adjacent benchmark for the Symthaea plasma domain, but it is **not a substitute** for verifying the governing equations or numerics of an RF-CCP solver.

Research anchors identified for a future measured-corpus track:

- [DisruptionBench](https://github.com/MIT-PSFC/DisruptionBench) describes model-agnostic tasks across Alcator C-Mod, DIII-D and EAST, including zero-shot, few-shot and many-shot settings. Its paper is “DisruptionBench: A robust benchmarking framework for machine learning-driven disruption prediction,” DOI: https://doi.org/10.1007/s10894-025-00495-2. The repository is archived/read-only; exact commit, task/data manifests, label conventions and paper/code correspondence still need to be pinned and audited.
- The repository's MIT license and README files were inspected through the GitHub connector (README blob `936719dd7020688da0870d2ef171f0d1eeaca5ca`; license blob `d674d4d67f0c1d191ed21338292cbbbb24bc64bb`). These are file blob identities, **not** a substitute for a complete immutable repository commit and dataset closure.
- [DisruptionPy](https://github.com/MIT-PSFC/disruption-py) provides data access/analysis tooling for several tokamak datasets. Its current development README lists C-Mod, DIII-D, EAST, HBT-EP and MAST; the MIT license was inspected (README blob `2705fc2bc4c118cbb92ba50c9bf4ce28e1c6d233`, license blob `43d39319e26a6b4034fd229d585225b3125a88b8`). Availability and terms are machine-dependent. An access library does not grant redistribution rights to underlying diagnostics, and raw/restricted machine data must not be committed to Symthaea.
- The concrete follow-on is [PLASMA-ML-REF-001 #7665](https://github.com/Luminous-Dynamics/symthaea/issues/7665); the current metric-semantics correction is [PR #7663](https://github.com/Luminous-Dynamics/symthaea/pull/7663).

Protocol requirements for this track:

1. Keep a rights-cleared, source-pinned data manifest and immutable hashes for any locally acquired permitted artifact. If credentials or a data-use agreement are required and unavailable, record the gate as blocked rather than swapping in synthetic data.
2. Use a Rust-native scoring/validation contract even if an upstream dataset requires a separately pinned extraction utility. Preserve the input schema, preprocessing revision, time base and label-generation provenance.
3. Group all partitions by complete shot, then by session/campaign and machine when available. No adjacent-sample random splits; no normalization, feature selection or pretraining leakage across held-out groups.
4. Report event-level metrics separately from sample/time-window metrics. Include event recall/precision/F1, false alarms per non-disruption shot and per unit observation time, false-positive rate, warning-lead-time distribution, calibration, scoreability/censoring counts, and uncertainty intervals bootstrapped over shots. State the exact useful-warning horizon and repeated-alarm policy before running the benchmark.
5. Report per machine and regime before an aggregate. Zero-shot/few-shot/many-shot transfer and leave-one-machine-out are separate experimental conditions, not interchangeable names for one holdout.
6. Include integrity adversaries: absent/corrupt diagnostics, malformed/irregular time axes, post-event alarms, duplicate/reordered samples, repeated warnings, partial/censored shots, duplicate shot IDs and split leakage. All injected faults in the finite preregistered suite should be detected; show suite size and coverage.
7. Treat a 95% event-recall objective as a preregistered operating-point hypothesis only when paired with an explicit false-alarm constraint, a useful lead-time requirement, held-out results and uncertainty. It is not, by itself, a safety claim or a hardware-operating recommendation.

The target distinction remains: synthetic demonstrations test implementation; source/data integrity tests establish lineage; numerical verification tests equations and discretization; comparison with published references checks reproducibility; experimental agreement tests a model against measurements. Passing one state does not promote it to another.

## 12. Decision-reversal challenge: benchmark the value of each expensive run

The next paradigm shift is to choose a simulation or experiment based on its expected ability to change a consequential engineering decision, not simply on the surrogate's predicted improvement or on uncertainty magnitude alone. This extends multifidelity optimization; it does not imply that an information-gain policy is automatically superior. Published multifidelity work explicitly finds that gains depend on the cost and informativeness of the lower-fidelity source, and recent multifidelity expected-information-gain estimators reduce Monte Carlo estimation variance only under their stated assumptions. Those are reasons to test selection policies under controls, not assume their advantage.

References:
- Sabanza-Gil et al. (2025), “Best practices for multi-fidelity Bayesian optimization in materials and molecular research,” *Nature Computational Science* 5, 572–581. https://doi.org/10.1038/s43588-025-00822-9
- Coons & Huan (2025), “A Multifidelity Estimator of the Expected Information Gain for Bayesian Optimal Experimental Design,” *SIAM/ASA Journal on Uncertainty Quantification* 13(4), 1990–2021. https://doi.org/10.1137/25M1731812
- Fan et al. (2024), “Multi-fidelity Bayesian Optimization with Multiple Information Sources of Input-dependent Fidelity,” *UAI 2024*, PMLR 244, 1271–1293. https://proceedings.mlr.press/v244/fan24a.html

### Freeze the decision before selecting a run

For each challenge case, define a quantity of interest, units, permitted regime, uncertainty/tolerance, and decision boundary before evaluating candidate policies. Example form: a proposed design is feasible only if its predicted response and uncertainty remain within a preregistered physical or engineering constraint. The decision threshold must be supported by the intended use and evidence; if it cannot be justified, the case remains not scoreable.

For each candidate design/run/fidelity, store:
- cost estimate and observed wall time / CPU-GPU solver-hours;
- the predicted distribution for the registered quantity of interest, not only a point estimate;
- which physical regime and assumption envelope qualify the prediction;
- expected change to the decision and expected reduction in decision regret;
- expected information gain, if calculated, as a separate metric with its estimator, prior, likelihood/noise model and Monte Carlo error;
- whether a more costly model or experiment is required to qualify the final decision.

Do not blend information gain, physical error and decision regret into a single leaderboard score unless a decision-maker has pre-registered and justified a common utility function.

### Policies to compare under one fixed total-cost budget

1. Random/space-filling sampling.
2. Cheap-model-only optimization.
3. Fixed-ratio escalation to the higher-cost solver.
4. Classical surrogate expected-improvement policy.
5. Uncertainty-only escalation.
6. Decision-aware escalation based on predicted risk of decision reversal / expected reduction in regret.
7. Information-gain-per-cost selection, only when the hypotheses and observation likelihood are well-defined.

All policies receive the same candidate pool, costs, initial evidence, high-fidelity budget and hidden evaluation cases. The experiment-selection policy must not read the high-fidelity target for a candidate before choosing that candidate. The final candidate must still be confirmed against the high-fidelity reference or a physical measurement appropriate to the claim.

Implementation follow-on: [PLASMA-ACT-001 #7685](https://github.com/Luminous-Dynamics/symthaea/issues/7685). Reuse the generic deterministic active-selection/evaluator/receipt architecture proposed in [MFG-ACTIVE-002 #7638](https://github.com/Luminous-Dynamics/symthaea/issues/7638); add only plasma-specific quantities, solver-fidelity adapters, decision boundaries, physical validity envelopes and reference cases.

### Decision-reversal challenge set

Include cases where the low-cost model:
- is accurate in most of the domain but badly biased near a decision boundary;
- has input-dependent fidelity (accurate in one region, unreliable in another);
- is overconfident outside the qualified envelope;
- disagrees with the expensive model for a physically meaningful reason;
- is poorly correlated with the expensive model and therefore provides little useful information;
- returns convergence failure, timeout, invalid units, a stale result or incomplete dependencies;
- has uncertainty that grows, shrinks incorrectly, or fails calibration under distribution shift.

The challenge must include cases in which the cheapest model is enough and cases in which it would cause a wrong decision. A policy that escalates every case may avoid some false-safe decisions but is not cost efficient; a policy that never escalates is cheap but can be dangerously overconfident. Report both failure modes.

### Acceptance metrics

Report each of the following independently:
- **False-safe rate:** frequency that the chosen action/design is declared acceptable but the registered high-fidelity reference violates the predefined constraint beyond tolerance.
- **Decision regret:** loss in the registered decision utility relative to the best decision supported by the reference evidence.
- **Qualified-decision rate:** fraction of cases where the policy provides an in-envelope, sufficiently supported decision instead of unsupported certainty.
- **Escalation precision and recall:** whether expensive calls occur on cases where they materially alter the qualified decision; include avoided-but-needed escalations separately.
- **Cost to decision:** solver-hours, wall-clock, memory/GPU-hours and total attempted/failed calls required to meet the registered decision quality.
- **Calibration and abstention:** coverage/width by regime; unsupported cases must abstain, not silently revert to defaults.
- **Reproducibility:** per-seed paired results, dependency closure, and exact source/config/input hashes.

**Proposed target, not an observed result:** demonstrate at least 25% fewer high-cost solver-hours than fixed-ratio escalation at equal or better held-out decision quality, with no increase in false-safe decisions or physical constraint violations. Require paired uncertainty intervals and retain failed attempts. A reduction in calls alone does not pass.

For the first screen, use a rights-cleared deterministic analytical/synthetic challenge with known answers, then a source-pinned solver reference. Only after those gates should any claim be tested against physical measurements. Simulation-derived labels must never be described as experimental validation.
