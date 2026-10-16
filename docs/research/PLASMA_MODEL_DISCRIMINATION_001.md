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
