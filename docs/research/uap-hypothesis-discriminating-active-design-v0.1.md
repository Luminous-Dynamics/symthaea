# Hypothesis-discriminating active experiment design v0.1

Date: 2026-11-09  
Status: executable scalar-Gaussian design primitive; synthetic qualification only  
Parent: [UAP-ACTIVE-001 #11187](https://github.com/Luminous-Dynamics/symthaea/issues/11187)  
Programme: [CTGP PR #10802](https://github.com/Luminous-Dynamics/symthaea/pull/10802)

## Why this is different from sensor placement

The existing bearing-sensor placement APIs optimize local geometry at an assumed target point or across a finite target-location hypothesis set. The newer budgeted design API chooses measurement options to improve prior-regularized D-optimal information and separately reports data-only Jacobian rank. Both are valuable.

This module adds a complementary objective: **choose a measurement that is expected to distinguish competing explanations**, not simply localize a target more precisely.

A sensor may improve a target-position covariance while producing nearly identical predictions under two competing physical models. Such a measurement is useful for localization but weak for model discrimination. Another measurement may have a less favorable CRLB yet strongly separate the models.

The active-design API should be used alongside:
- data-only local observability (`observability::analyze_jacobian`);
- global ambiguity screening (`global_identifiability::screen_global_ambiguities`);
- geometric / target-region FIM placement (`triangulation`);
- multi-option D-optimal placement (`budgeted_design`).

It does not replace any of them.

## Literature / SOTA boundary

The software deliberately builds on existing research rather than claiming novelty over:
- Weiss et al. (2026), *Joint Self-Calibration of Receiver Geometry, Timing, and Target Positions for Multistatic Radar Autofocus*, Sensors 26(15), 4954, DOI 10.3390/s26154954. The work jointly estimates receiver geometry, clocks and targets; identifies anchor-count-dependent gauge freedoms; and refines calibration against coherent image sharpness. It also shows that coplanar anchors can retain a discrete mirror ambiguity not exposed by local rank alone.
- *Multistatic Localization of a Moving Object Aided by Calibration Objects: Analysis and Solutions*, IEEE Transactions on Signal Processing 74 (2026), DOI 10.1109/TSP.2026.3726231, which studies asynchronous multistatic localization with calibration objects and CRLB-level estimators.
- *Multifidelity sensor placement in Bayesian state estimation problems* (2026), DOI 10.3934/ammc.2026006, which formulates cost-constrained sensor placement with multiple noise/cost levels and Bayesian D-optimal design.
- *Accuracy Analysis and Optimal Configuration for AOA Localization*, IEEE Signal Processing Letters (2026-09-07), DOI 10.1109/LSP.2026.3731095, which derives bearing sensor configurations from Fisher-information eigenstructure.

The near-term contribution is a reproducible model-discrimination utility and abstention contract that can later be coupled to those calibration-aware, geometric and budgeted design methods.

## v0.1 mathematical contract

For candidate measurement j, under hypothesis H_i:

    Y | H_i, j ~ Normal(mu_ij, sigma_j^2)

where

    sigma_j^2 = sigma_measurement,j^2 + sigma_calibration,j^2.

The prior is a finite discrete distribution p_i with sum p_i = 1. The conditional expected information gain is

    EIG(j) = sum_i p_i integral f_i(y) log(f_i(y) / sum_k p_k f_k(y)) dy.

This is mutual information I(H;Y), measured in nats. The implementation computes the expectation under each hypothesis in its own standard-normal coordinate. This avoids undersampling narrow likelihoods when two candidate means are far apart.

The result includes a deterministic Simpson quadrature comparison at 1,024 and 2,048 intervals. The difference is reported as a convergence diagnostic; it is not a formal interval bound.

Availability q_j is handled as

    EIG_available(j) = q_j * EIG(j)

under the v0.1 assumption that a missing observation is uninformative. Therefore:

    E[posterior entropy] = H(prior) - q_j * EIG(j).

This assumption is intentionally narrow. In real sensor systems, occlusion, non-detection, radar cross-section and instrument state can make a missing observation informative. Those outcomes must be modelled explicitly in a later observation model, not hidden inside q_j.

The selection utility is

    utility(j) = EIG_available(j) / cost(j)

subject to a hard scalar budget and a predeclared minimum EIG. If no candidate is eligible, return an explicit abstention.

## Executable implementation

The module is `symthaea_uap_observatory::active_design`.

Inputs:
- `HypothesisPrior`: unique ID and strictly positive normalized prior probability.
- `ExperimentCandidate`: unique ID, cost, availability, measurement sigma, calibration sigma and one mean prediction per hypothesis.

Outputs:
- conditional EIG;
- availability-adjusted EIG;
- expected posterior entropy;
- information per cost;
- effective marginal sigma;
- quadrature convergence difference;
- budget eligibility;
- selected candidate or abstention.

The likelihood model is scalar Gaussian with common effective sigma per candidate. It assumes independent measurement and calibration variance for this single observation. It does not model persistent shared clock bias or cross-candidate covariance.

## Required follow-on benchmarks

Compare the chosen experiment against:
1. random candidate selection;
2. cheapest available candidate;
3. highest geometric D-optimal / logdet FIM;
4. lowest A-optimal covariance trace;
5. largest E-optimal minimum eigenvalue;
6. existing budgeted greedy D-optimal selection;
7. exhaustive oracle EIG selection for small candidate sets.

All baselines must receive identical candidate costs, availability and noise/calibration assumptions. Report:
- oracle selection accuracy;
- expected-information regret;
- entropy reduction;
- cost and availability;
- abstention precision and recall when all candidates are nondiscriminating;
- robustness to prior misspecification;
- run time and memory.

Hold out hypothesis families and measurement geometries, not only random candidates. A larger EIG on the training candidate set is not evidence of a better experimental-design policy.

## Prospective gates

- identical predictive distributions yield zero EIG within tolerance;
- EIG is non-negative and no greater than prior entropy within numerical tolerance;
- increasing independent measurement or calibration variance cannot increase EIG beyond quadrature error;
- candidate and hypothesis reordering leave each candidate's score invariant;
- budget-ineligible candidates are never selected;
- an indistinguishable candidate set correctly abstains;
- the 1,024/2,048 quadrature difference remains below a preregistered threshold on the qualification grid;
- no performance or SOTA claim until held-out selection regret beats matched D/A/E-optimal and random/cheap baselines with uncertainty reported.

## Current claim ceiling

A passing unit test establishes only the scalar Gaussian calculation for the declared assumptions. It does not infer a real object's identity, establish a UAP anomaly, validate a physical theory or imply exotic propulsion. If candidate predictions do not distinguish the hypotheses, abstention is the correct result.
