# CTGP model-misspecification and M-open experiment-design benchmark v0.1

Date: 2026-11-09  
Status: integration specification; no robust-acquisition result claimed  
Parent: [CTGP-ROBUST-001 #11262](https://github.com/Luminous-Dynamics/symthaea/issues/11262)  
Programme: [CTGP draft PR #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## Architectural decision

Do not create another EIG, R-IDeA, MMD, model-challenge or robust-decision engine. Reuse:
- `symthaea-science-evidence::theory_prediction` for typed observable/model coverage and the `OTHER_UNRESOLVED` open-world state;
- `model_discrimination` and `model_recovery` for quantitative prediction comparability, EIG, detection/non-detection, recovery and abstention;
- `symthaea-uap-observatory::global_identifiability` for global ambiguity;
- `symthaea-uap-observatory::budgeted_design` for candidate portfolio feasibility/cost/exclusivity;
- QARCH-010 #9067 and QARCH-011 #9088 for R-I/R-IDeA benchmarking and paper-faithful primitives;
- QARCH-012 #9092 for the sign-identifiability audit of the paper's magnitude-based de-amplification proxy;
- SCI-CONNECTOMICS-094 #10774 for response-model expansion and discrepancy challenges.

This document is a CTGP applied-physics benchmark and integration contract, not a competing general-purpose active-learning implementation.

## Motivation and research basis

The AISTATS 2026 paper by Tang, Sloman and Kaski studies Bayesian active learning when the assumed model is misspecified and the experiment-selection distribution differs from the target/test distribution. It develops R-IDeA using representativeness, informativeness and de-amplification concepts. Its results do not imply that a magnitude-only discrepancy proxy certifies the sign of an error interaction; Symthaea's QARCH-012 audit already identifies that exact limitation.

Primary reference: Tang, R., Sloman, S. J., & Kaski, S. (2026), *Representative, Informative, and De-Amplifying: Requirements for Robust Bayesian Active Learning under Model Misspecification*, PMLR 300:3016–3024. https://proceedings.mlr.press/v300/tang26d.html

The physical-science setting makes the problem especially important: candidate predictions are often simplified, observation channels are incomplete, calibration is uncertain, and the hypothesis set is almost certainly not exhaustive. An experiment may have high EIG among named models while every named model is wrong.

The scientific decision must therefore be able to return **none of the current models is sufficiently supported**.

## Paradigm shift

Move from “choose the experiment that best distinguishes our current theories” to:

> Choose experiments that improve decisions over the declared target domain while actively challenging the adequacy of the model family and observation process.

Keep four components separate:
1. **Within-family discrimination:** EIG / predictive KL among named models under the declared likelihood.
2. **Representativeness:** distance between selected experiments and the target/operating distribution.
3. **Misspecification and de-amplification:** whether selected observations improve held-out error when the assumed response model is wrong.
4. **Open-world adequacy:** whether a valid challenger or unseen planted process invalidates a nominal conclusion.

None is a certificate of model completeness; no single composite score is permitted to hide these distinctions.

## CTGP benchmark worlds

### Causal sets
Truth families: 1+1D, 2+1D and 3+1D Minkowski sprinklings; curved 1+1D conformal-FRW; alternative finite-region windows; non-Poisson controls.

Observation regimes: finite cardinality, interval censoring, thinning, finite-cone sampling, boundary effects, coordinate/measurement perturbations.

Challengers: continuum-dimension mismatch, curved-vs-flat mismatch, non-Poisson process, altered sampling window.

Endpoints: dimension-posterior coverage, abundance/interval-profile prediction error, held-out log score, false-confident dimension claims.

### Time-crystal sensing
Truth families: noninteracting perfect-pi control, detuned unlocked model, interacting kicked Ising, anisotropic Heisenberg/Trotter model, metastable/scarred response, open-system decoherence variants.

Observation regimes: drive jitter, unknown phase, finite windows, readout noise, disorder, initial-state shift.

Challengers: omitted bath, pulse error, incorrect interaction graph, unknown timing/phase, long-lived but not phase-rigid response.

Endpoints: frequency-locking coverage, lifetime prediction error, QFIM/CFI calibration, false phase classification, nuisance-aware amplitude-bound coverage.

### Metric geometry and GravCraft
Truth families: Minkowski, Schwarzschild, analytic weak-field cases, perturbed metrics, deliberately inconsistent metric/source proposals.

Observation regimes: finite derivative step, coordinate-chart changes, measurement noise, incomplete source/boundary conditions and parameter uncertainty.

Challengers: trajectory-fitting metric that violates constraints, omitted stress-energy, heuristic shift-as-acceleration simulator, incorrect boundary conditions.

Endpoints: geodesic/curvature residual, Einstein constraint violation, source reconstruction error, false physically-realizable classification.

### Multi-sensor anomaly observations
Truth families: conventional trajectories, sensor/optical artifacts, parallax, clock drift, correlated pose errors, planted unknown dynamics.

Observation regimes: asynchronous timestamps, calibration drift, missingness, clutter, sensor dropout.

Challengers: wrong clock map, boresight error, unmodelled common-mode calibration, unknown detection channel, target-model mismatch.

Endpoints: trajectory error, interval coverage, observability abstention, false-confident trajectory rate, false anomaly rate.

Every generator labels its evidence class: analytic reference, reduced simulator, physical simulator, or adversarial software fixture. Synthetic data are not physical discoveries.

## Acquisition contract and baselines

Each candidate carries:
- observable, operational definition and units;
- model/version/specification references;
- quantitative predictive distribution or typed non-quantitative disposition;
- calibration/noise model and its provenance;
- execution/availability probability and total cost;
- target-regime distribution descriptor;
- nuisance parameters and uncertainty;
- held-out regime identity and leakage restrictions.

Compare on identical frozen candidates, priors, measurements, calibration information and budget:
1. Random feasible choice.
2. Nominal EIG/cost.
3. Worst-prior-scenario EIG/cost.
4. Pairwise worst-confusable predictive KL.
5. Model-recovery or expected decision-value acquisition.
6. R-I representativeness/informativeness proxy.
7. Paper-faithful R-IDeA from QARCH-011, preserving the QARCH-012 sign-identifiability limitation.
8. Model-challenge / ambiguity-envelope method from SCI-CONNECTOMICS-094.
9. Random and fixed-order domain coverage.
10. Evaluator-only oracle for regret measurement only; never usable for candidate selection.

The QARCH-011 paper-faithful acquisition and its magnitude-based DeA gate must not be described as a formal signed de-amplification certificate. Include the QARCH sign-pair negative control: same absolute gap, opposite signed error interaction.

## Required metrics

Report separately:
- conditional and availability-adjusted model-identity EIG;
- pairwise directed KL and worst-confusable alternative;
- MMD / target-domain representativeness;
- signed evaluator-only error interaction and held-out generalization error;
- discrepancy detection precision/recall and time-to-detection;
- model-recovery confusion matrix;
- false-confident model-selection/physical-interpretation rate;
- open-world and correct abstention rate;
- dimension, frequency, trajectory and metric posterior coverage;
- decision regret and robust worst-case utility;
- selected-regime coverage and OOD held-out error;
- calibration/model-challenge withdrawal rate;
- acquisition and compute cost.

The primary endpoint for each campaign is its downstream scientific decision loss. Maximizing EIG, minimizing MMD or maximizing worst-case robustness alone does not establish a better scientific decision policy.

## Preregistered adversarial cases

1. **All named theories wrong:** plant an unseen process outside the candidate set. No forced winner.
2. **High-EIG narrow regime:** nominal EIG selects an extreme unrepresentative regime; trade-offs must remain visible.
3. **R-IDeA sign-pair:** identical absolute disagreement but opposite signed interactions; a magnitude proxy cannot certify both.
4. **Nuisance-model expansion:** add a valid clock, pose, bath or boundary-condition challenger. A robust worst-case claim cannot improve after the admissible model set expands.
5. **Unresolved theory:** a theory without a quantitative prediction (e.g. no derived Chain Theory photon propagator) remains NotComparable, not a fabricated zero-likelihood model.
6. **Calibration drift:** a nominally informative sensor is misleading under shared calibration shift.
7. **Selection leakage:** evaluator truth, held-out observations or discrepancy labels enter selection/tuning; reject the run.
8. **Decision-irrelevant discrepancy:** predictive error changes but downstream decision does not; report discrepancy without inventing decision impact.
9. **Out-of-domain abstention:** outside validated model/sensor domain, return typed non-comparability or unresolved.
10. **Metric fit without physical consistency:** trajectory fits while ADM/Einstein constraints fail; do not promote physical realizability.

## Prospective acceptance gates

- 100% agreement with independent analytic EIG/MMD/FIM/CRLB fixtures.
- Zero false-confident physical/model claims on planted M-open fixtures under frozen confidence rules.
- 100% typed non-comparability where quantitative predictions are missing.
- R-IDeA reproduction agrees with QARCH-011 paper equations; QARCH-012 limitation stays explicit.
- Model-family expansion cannot improve worst-case loss or preserve an invalid resolved claim.
- All policy thresholds and hyperparameters are frozen before sealed evaluation worlds.
- Held-out regimes are separated by generator family, not random sample only.
- Baselines receive identical candidates, observations, priors, costs and compute.
- Claims remain scoped to the tested generator and physical domain.

## Implementation sequence

1. Freeze this benchmark and map existing CTGP examples to truth/observation regimes.
2. Reuse theory-prediction coverage and model-discrimination outputs to build typed candidate manifests. Never fabricate predictions for missing theories.
3. Compose with QARCH R-IDeA and SCI-CONNECTOMICS-094 through adapters; do not duplicate their solvers.
4. Start with a small M-open causal-set and sensor-calibration challenge.
5. Add evaluation-only truth records, signed discrepancy diagnostics and held-out target distributions.
6. Retain all failed and abstained trials.
7. Add DTC and GR/ADM worlds only after independent numerical-reference gates pass.
8. Bind source, candidate pool, model set, target distribution, seed manifest, calibration, policy and outcomes to science-evidence receipts.

## Claim ceiling

A pass validates behaviour only on the frozen candidate family and challenge worlds. It does not prove theory truth, model completeness, new physics or UAP identity. The correct outcome may be **no available experiment can distinguish these theories under the present observation model**.
