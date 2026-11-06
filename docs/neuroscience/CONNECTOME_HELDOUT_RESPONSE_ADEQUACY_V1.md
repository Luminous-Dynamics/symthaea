# Held-out connectome intervention-response adequacy benchmark V1

**Tracking:** #10781 — SCI-CONNECTOMICS-095
**Composes:** #10774, #10779, #10675, #10670, #9587, #9255
**Status:** synthetic full-trajectory scoring contract; no biological calibration result is claimed.

## 1. Objective

Evaluate whether a response model predicts the full observable trajectory under a held-out intervention, at the level of independent biological units. Detect when a held-out response falls outside the model's declared predictive support, without forcing an invalid posterior update.

## 2. Unit of evidence

The independent unit is the animal, specimen, preparation, target-neuron/session, or another explicitly justified biological cluster. Timepoints, channels, cells from one animal and repeated presentations are not independent biological replicates merely because they create separate rows.

Freeze three disjoint unit partitions:
- training units used for model fitting;
- calibration units used only for predeclared calibration/interval construction;
- lockbox units used only for final evaluation.

The primary split must be by independent biological unit. Random timepoint splits may be shown only as a leakage negative control, never as the primary generalization result.

## 3. Predictive object

Each model predicts a joint distribution over a complete trajectory, not a separate independent distribution for each timepoint. A trajectory binds its ordered observation vector, time grid, measurement units and observation-kernel revision.

Keep separate:
- latent neural-state trajectory;
- structural graph and direct edge evidence;
- observation / indicator kernel;
- observed fluorescence/electrophysiological/behavioral trajectory;
- mechanism attribution and query consequence.

A response does not prove a direct synapse; an anatomical edge does not by itself establish a functional response. Extrasynaptic and state-mediated effects must not be relabelled as direct EM edges.

## 4. Metrics

Categorical trajectories:
- negative log predictive score when the observed trajectory has positive predicted probability;
- zero-support / out-of-envelope count, reported separately rather than silently replaced by a clipped probability;
- multiclass Brier score over the frozen trajectory category universe;
- empirical coverage of highest-probability predictive sets at preregistered nominal levels (e.g. 80%, 90%, 95%).

Continuous scalar trajectories should additionally report CRPS and interval width/coverage. Multivariate trajectories should additionally report energy score and a variogram/dependence-sensitive score. Marginal pointwise metrics cannot substitute for joint trajectory scoring.

Always report calibration and sharpness separately. A wide predictive interval can cover everything but remain uninformative. A proper score can rank distributions but does not alone establish calibration across all subgroups.

## 5. Model-discrepancy handling

If an observed categorical trajectory has zero probability under every declared model for that analysis, mark `MODEL_DISCREPANCY_DETECTED` and `POSTERIOR_UPDATE_WITHHELD`. Preserve the raw observation and receipt; do not force an arbitrary epsilon likelihood or silently renormalize remaining models.

For continuous predictive distributions, use an explicit predictive-region / tail-probability rule and preregistered threshold. A low-density outcome is not automatically a model falsification; calibration uncertainty, measurement error and multiple testing must be accounted for.

Compare posterior predictive checks (PPC) with genuinely held-out predictive checks (HPC). PPC reuses data for fitting and checking; holdout checks use independent data for evaluation. Report which units and observations entered each phase.

## 6. Dependence and conformal guarantees

Classical split conformal finite-sample marginal coverage relies on exchangeability. Neural trajectories are temporally dependent, and repeated measurements within animals/sessions violate naive row-level exchangeability. Use animal/session/block-aware methods with a stated guarantee where available; otherwise report `NOT_GUARANTEED` rather than advertising nominal coverage as a guarantee.

Where covariate shift or sequential drift is present, a weighted/online/dependence-aware method requires its own assumptions, diagnostics and validation. Coverage claims do not transfer automatically from an independent-unit split to correlated timepoints.

## 7. Synthetic known-answer fixture

Five independent synthetic units are declared: two calibration, two lockbox observations, and one failed lockbox acquisition. Two model families predict full discrete four-step trajectories.

- Baseline model predicts only trajectory A (0.8) and B (0.2).
- Challenger predicts A (0.6), B (0.3), and C (0.1).
- The first lockbox unit observes B; the second observes C.
- C is outside the baseline model's support but inside the challenger's support.
- The failed acquisition remains in the lockbox denominator ledger and is not converted into a negative or silently discarded.

Expected software outcomes:
- baseline lockbox zero-support count: 1;
- challenger lockbox zero-support count: 0;
- baseline and challenger lockbox 90% highest-probability-set coverage: 1/2 each;
- observed lockbox units: 2; failed lockbox units: 1;
- calibration status: `SYNTHETIC_KNOWN_ANSWER_ONLY_NOT_A_CALIBRATION_CLAIM`.

The fixture deliberately demonstrates that the model with no zero-support event is not thereby proven calibrated. Two lockbox observations are far too few to establish population coverage.

## 8. Promotion gates

Targets, not current results:
- 100% correct rejection of unit leakage and duplicated independent-unit IDs;
- 100% retention of failed/inaccessible units in the disposition ledger;
- 100% fail-closed handling of zero-support lockbox outcomes;
- 0 posterior updates forced from zero-likelihood observations;
- proper-score improvements assessed on independent lockbox units with cluster-aware uncertainty;
- no calibration claim without adequate independent-unit support and a valid dependence assumption;
- model-discrepancy detection evaluated separately from query-critical consequence.

## 9. Industry comparators

- In-sample posterior predictive checks (PPC) versus holdout predictive checks (HPC).
- Nominal Bayesian simulator ensemble versus #10774's adversarial finite model envelope.
- Standard split conformal only where exchangeability is defensible; blocked/cluster/time-series conformal methods otherwise.
- Proper distributional scores: log score and Brier for categorical outcomes; CRPS for scalar traces; energy and variogram scores for joint trajectories.
- Expert range predictions and simple mean-trajectory/pointwise-interval baselines.

Relevant methods: the calibrated holdout predictive check framework (JRSSB 2024, https://doi.org/10.1093/jrsssb/qkad105); conformal prediction's exchangeability assumptions and time-series caveats (Kaiser & Herzog 2025, https://doi.org/10.1177/25152459251380452); multivariate proper scoring rules and dependence-sensitive evaluation (https://ascmo.copernicus.org/articles/11/23/2025/).

## 10. Claim ceiling

A passing synthetic audit qualifies scoring and disposition semantics only. It does not establish biological calibration, population coverage, global model adequacy, a direct synaptic mechanism, or prospective intervention efficacy. Biological promotion requires source-bound held-out experiments, independent-unit lockboxes, adequate sample size, dependence-aware uncertainty and external replication.