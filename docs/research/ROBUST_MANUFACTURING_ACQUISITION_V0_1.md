# Robust Manufacturing Acquisition v0.1

**Status:** research selector and finite synthetic test; not validated on real factory campaigns.

## Paradigm shift

Candidate-specific noise addresses measurement heterogeneity. It does not address structural uncertainty shared by the supplied hypothesis set. A design can have high nominal expected information gain and still amplify a wrong physical model.

This work adds a deliberately conservative finite-scenario maximin baseline:
`robust_score(e) = min_s EIG(e | model_s, measurement_noise_e) / cost(e)`.

The eligibility check is external and hard. A candidate is scored only after it passes the route, calibration, safety and resource gates. It must also have complete, finite, strictly positive EIG under every declared model-discrepancy scenario. One adverse scenario is sufficient to make the candidate unrankable.

This is a **finite-set maximin EIG** baseline. It is not the full R-IDeA criterion, it does not prove robustness outside the declared scenario set, and it can be conservative when a low-probability scenario collapses the information gain.

## Implementation

`select_experiment_by_worst_case_information_gain_per_cost` in `symthaea-core` accepts:

- candidates and a common hypothesis set;
- a non-empty finite array of alternative model scenarios;
- hard external candidate eligibility;
- candidate-specific standard measurement uncertainty;
- a prediction closure conditioned on candidate and model scenario;
- comparable acquisition cost.

For each eligible candidate, it computes EIG separately under every scenario and retains the minimum. Invalid noise/cost, empty scenarios, incomplete predictions, non-finite information or zero information in any scenario fail closed. Ties retain the first candidate.

The original common-noise and candidate-noise EIG selectors remain available as comparison baselines.

## Required model-discrepancy families

For transformer and electrical-component manufacturing, the challenge set should cover:

1. Omitted electromagnetic/thermal coupling (e.g., harmonics, local hot spots, fringing, eddy-current effects).
2. Constitutive/material-law shift (steel grade, lot, stress/annealing history, temperature-dependent properties).
3. Boundary-condition shift (ambient, cooling, load waveform, mounting, enclosure).
4. Measurement-model shift (bias, drift, heteroscedasticity, correlated errors, calibration age).
5. Process regime transfer (supplier, machine, tool wear, station, operator, winding program).
6. Geometric and scale transfer (core construction, lamination, coil arrangement, rated power, frequency).

Each scenario needs a versioned description, provenance and a held-out challenge population. Scenarios must not be selected after observing which candidate they penalize.

## Benchmark

Compare, at identical admissible candidate pools and total acquisition budgets:

- uniform random among eligible candidates;
- prediction variance;
- common-noise EIG;
- candidate-noise EIG;
- candidate-noise EIG per cost;
- finite-set worst-case EIG per cost.

Primary metrics: worst-scenario decision-relevant information, correct engineering decision recovery, regret against a frozen within-pool evaluator oracle, cost to resolve the predeclared decision, calibration by discrepancy family, OOD/shift strata, abstention, and ineligible-action attempts.

Pair all policies on the same product, component lot, manufacturing campaign, prior evidence and measurement budget. Cluster inference by independent lots/campaigns, not repeated reads. Preserve at least one unseen manufacturer/material lot/station family for external transfer.

### Advancement criteria (internal research targets)

- Exactly zero ineligible selections in all negative controls.
- Exactly zero false-safe or conformity regressions relative to the deterministic quality gate.
- Worst-case decision regret no worse than the best non-robust baseline under every frozen scenario family.
- At least 10% lower verified acquisition cost to a predeclared decision, or at least 5 percentage points higher held-out decision recovery, with paired 95% lot/campaign-cluster confidence intervals excluding zero.
- Calibration/coverage assessed on independent campaign clusters; missing or out-of-domain evidence must abstain.

These are proposed pilot thresholds, not industry norms or claimed results.

## Limitations and next steps

The current EIG implementation still uses a uniform prior over hypothesis instances and independent Gaussian observation noise. The robustness envelope is a finite set, not an adversarial continuous model class. It does not model correlated metrology errors, Bayesian posterior weights over equivalence classes, inferential error distributions, representativeness weighting, or explicit error de-amplification.

Next research steps:

- Add prior-weighted EIG over equivalence classes rather than a uniform prior over raw hypotheses.
- Add correlated and non-Gaussian measurement error, including shared calibration bias and drift.
- Add a representativeness/de-amplification metric, with an ablation against the maximin baseline.
- Add sequential posterior updates after each real measurement; the present selector is a one-step ranking function.
- Calibrate scenario-set coverage on independent manufacturer/lot/station holdouts.
- Keep all physical acceptance, safety and release authority outside the cognitive selector.

## Research position

Callahan & Catanach (AISTATS 2026) show that nominal Bayesian experiment utility can become harmful under model or inferential misspecification, and develop robust experimental-utility axioms. Tang, Sloman & Kaski (AISTATS 2026) develop R-IDeA, combining representativeness, informativeness and error de-amplification. This implementation is a deliberately narrower finite-set maximin comparator and should be benchmarked against those methods rather than described as equivalent.

- Callahan & Catanach (2026): https://proceedings.mlr.press/v300/callahan26a.html
- Tang, Sloman & Kaski (2026): https://proceedings.mlr.press/v300/tang26d.html
