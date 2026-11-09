# Availability-aware hypothesis-discriminating experiment design v0.2

Date: 2026-11-09  
Status: extension specification for existing model-discrimination and budgeted-design engines  
Parent: [UAP-ACTIVE-001 #11187](https://github.com/Luminous-Dynamics/symthaea/issues/11187)  
Programme: [CTGP PR #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## 1. Architectural decision

The current CTGP branch already implements two distinct design layers:

- `symthaea-science-evidence::model_discrimination` computes derivation-backed, open-world-aware model-identity EIG for calibrated Gaussian predictive distributions. It supports multiple prior scenarios, typed non-comparability, quadrature convergence checks, cost-normalized ranking and a separate minimum directed predictive-KL comparator.
- `symthaea-uap-observatory::budgeted_design` performs multi-scenario budgeted D-optimal selection with mutual-exclusion groups, greedy selection, an exhaustive small-set oracle and separate data-only observability diagnostics.

A second scalar-Gaussian EIG implementation in the Observatory would be redundant and weaker. The proposed design therefore extends the shared model-discrimination engine rather than creating a parallel likelihood/entropy kernel.

The Observatory continues to own the physical measurement geometry, calibration lineage, observability, target-region FIM placement and global ambiguity checks. The science-evidence crate owns model prediction comparability, model-identity EIG and evidence-bound experiment ranking.

## 2. The remaining gap: measurement availability

For a candidate experiment (e_j), let:

- (q_j) be the probability of obtaining a usable observation;
- (I_j = I(M;Y mid 	ext{usable},e_j)) be conditional model-identity EIG;
- (C_j) be cost.

Under the v0.1 assumption that a missing observation is uninformative:

[
I_{mathrm{expected},j}=q_j I_j
]

[
mathbb{E}[H(Mmid 	ext{outcome},e_j)]
=H(M)-q_j I_j
]

Keep conditional and availability-adjusted EIG separate in every scenario and aggregate. Rank cost utility with the availability-adjusted value. A candidate with (q=0) remains a valid candidate record and retains its conditional EIG, but has zero expected information under this model.

This is deliberately not a general missing-data model. Visibility, occlusion, sensor dropout, radar cross-section, instrument-health state and non-detection can themselves be informative. Those must be represented as explicit observation outcomes in a later model, not hidden inside a scalar availability multiplier.

## 3. Existing literature and comparison boundary

The programme must compare against established results rather than claim novelty over active sensing or Bayesian design:

- Weiss et al. (2026), *Joint Self-Calibration of Receiver Geometry, Timing, and Target Positions for Multistatic Radar Autofocus*, Sensors 26(15), 4954, DOI 10.3390/s26154954. Joint geometry/clock/target calibration, anchor-dependent gauge analysis and coherent image-sharpness refinement; demonstrates that discrete mirror ambiguity can survive local rank checks.
- *Multistatic Localization of a Moving Object Aided by Calibration Objects: Analysis and Solutions*, IEEE Transactions on Signal Processing 74 (2026), DOI 10.1109/TSP.2026.3726231. Asynchronous multistatic localization with calibration objects and CRLB analysis.
- *Multifidelity sensor placement in Bayesian state estimation problems* (2026), DOI 10.3934/ammc.2026006. Cost/noise-fidelity-constrained Bayesian D-optimal design.
- *Accuracy Analysis and Optimal Configuration for AOA Localization*, IEEE Signal Processing Letters (2026-09-07), DOI 10.1109/LSP.2026.3731095. Bearing sensor geometry derived from Fisher-information eigenstructure.

The competitive target is a reproducible end-to-end decision contract, not a new claim for EIG, FIM, Bayesian experimental design or sensor fusion.

## 4. How the objectives compose

These outputs answer different questions and must remain distinct:

| Question | Existing primitive | Output |
|---|---|---|
| Which named model best predicts a candidate observation? | `model_discrimination` | Conditional and availability-adjusted model-identity EIG |
| How sensitive is that ranking to prior assumptions? | `model_discrimination` | Per-prior-scenario EIG and worst-case/weighted summaries |
| How much do models differ in predictive distributions? | `gaussian_predictive_kl_nats` | Directed pairwise KL; not the same as EIG |
| Which sensor geometry improves local position information? | `triangulation` | FIM / CRLB placement criteria |
| Which set of options improves local parameter information under cost/exclusivity? | `budgeted_design` | Greedy and exhaustive D-optimal design |
| Are local parameter directions observable? | `observability` | Data-only Jacobian singular values, rank and null directions |
| Are there distinct global solutions with indistinguishable predictions? | `global_identifiability` | Residual-qualified ambiguity pairs |

A candidate that scores well on one row does not automatically score well on the others. A full-rank local Jacobian does not establish global uniqueness; a low CRLB does not establish model discrimination; a high model EIG does not imply the model set is complete.

## 5. Qualification plan

1. Add and validate `availability_probability` on `CandidateExperiment`.
2. Preserve per-scenario conditional EIG; add availability-adjusted EIG and expected posterior entropy.
3. Preserve conditional worst-case and weighted-mean EIG; add adjusted summaries.
4. Rank by adjusted EIG per cost while showing (q), conditional EIG and adjusted EIG.
5. Keep open-world, `NotComparable`, `QuadratureNotConverged` and unresolved-model states intact.
6. Add (q=0), (q=0.5), (q=1), invalid-(q), prior-sensitivity and ranking tests.
7. Extend the synthetic reference output to display both conditional and adjusted scores.
8. Later compose the resulting experiment recommendations with physical sensor placement, budget/exclusivity, observability and global ambiguity checks.

## 6. Benchmark objectives

Report, separately:
- conditional model-identity EIG;
- availability-adjusted EIG;
- expected posterior entropy;
- directed pairwise predictive KL;
- model-recovery accuracy under planted models;
- abstention and open-world rejection rates;
- candidate choice regret versus exhaustive small-set design;
- geometric position covariance / CRLB;
- data-only numerical rank and weak directions;
- global ambiguity detection rate;
- cost, latency, compute and memory.

Baselines:
1. random candidate;
2. cheapest candidate;
3. maximum conditional EIG per cost;
4. maximum availability-adjusted EIG per cost;
5. FIM D/A/E-optimal ranking;
6. greedy budgeted D-optimal design;
7. exhaustive oracle on small candidate sets.

The evaluation must use identical model predictions, calibration assumptions, candidate availability and resource budgets. Hold out entire hypothesis families and measurement geometries. Report confidence intervals and prior-sensitivity, not a single average.

## 7. Prospective gates

- Availability probability is finite and in [0,1].
- Conditional EIG is unchanged by availability.
- Adjusted EIG equals (q I).
- Expected posterior entropy equals prior entropy minus adjusted EIG.
- (q=0) leaves entropy unchanged; (q=1) recovers conditional EIG.
- Invalid availability fails closed.
- Existing open-world and non-comparability gates are preserved.
- Cost ranking uses adjusted EIG per cost.
- No claim that missingness is informative until it is explicitly represented in the predictive model.
- No SOTA claim without sealed model-family holdouts and matched baselines.

## 8. Claim ceiling

This extension qualifies an availability-adjusted decision calculation under an explicit missingness assumption. It does not establish model truth, identify an observed object, validate a UAP anomaly or imply new physics. The correct output may be that the named models are not comparable, the model universe is open, or no available measurement is expected to discriminate the current hypotheses.
