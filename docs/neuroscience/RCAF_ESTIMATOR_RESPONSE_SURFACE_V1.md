# RCAF Estimator Response on Matched Factorial Cells v1

**Status:** synthetic estimator-level qualification primitive; no empirical fMRI or biological claim is made.  
**Base:** matched four-cell counterfactuals and common-random-numbers dose-response.  
**Purpose:** measure how the existing RCAF estimators respond after each raw time × ROI cell is converted into connectivity.

## 1. From time series to connectivity

Each factorial point contains four raw time × ROI matrices:

- X: clean-null observations;
- X + S: signal-only observations;
- X + A: nuisance-only observations;
- X + S + A: signal-plus-nuisance observations.

The estimator-response module independently computes a Pearson ROI-connectivity matrix from each matrix. It then evaluates each existing RCAF matrix-similarity estimator on the four corresponding connectivity matrices.

This order matters: connectivity estimation is nonlinear. Even if raw interventions are additive, the signal-induced connectivity change can be altered by nuisance. That nonlinear interaction is one of the things the benchmark should reveal.

## 2. Existing estimators compared

The current response surface evaluates the three existing implementations in `simulation.rs`:

1. **Absolute weighted Jaccard:** compares absolute correlation weights, emphasizing edge magnitude regardless of sign.
2. **Signed-shift weighted Jaccard:** shifts signed edge weights into non-negative mass before comparing them, retaining sensitivity to polarity changes.
3. **Normalized mean absolute edge error:** measures mean absolute upper-triangle edge deviation with a fixed normalization.

All similarities are reported in [0, 1], with larger values indicating closer matrices. The estimator implementation and its normalization are pinned by the source revision; do not treat these metrics as interchangeable or as a validated universal functional-connectivity score.

## 3. Per-estimator measurements

For each estimator E, with C(·) denoting the per-cell Pearson connectivity estimator:

- **Target signal distance:** 1 − E(C(X), C(X + S)).
- **Nuisance-only displacement:** 1 − E(C(X), C(X + A)).
- **Residual displacement on signal background:** 1 − E(C(X + S), C(X + S + A)).
- **Signal retention fraction:** 1 − residual displacement / target signal distance.
- **Nuisance-to-signal distance ratio:** nuisance-only displacement / target signal distance.

The two ratios are null when the target signal distance is at or below the declared numerical denominator threshold. Values are not clipped: negative retention or retention greater than one should remain visible rather than being made to look like a valid [0,1] score.

These are finite-sample matrix-response quantities. They are not participant-level statistical power, false-positive rate, confidence interval coverage, or behavioural prediction. Those require repeated independent simulations or participant/site samples and a statistical decision procedure.

## 4. Connectivity-space directional recovery

For each cell, construct the upper-triangle connectivity changes:

[
\Delta_S = C(X+S)-C(X),
qquad
\Delta_{S|A}=C(X+S+A)-C(X+A).
]

Report:
- projection gain: (\langle\Delta_S,\Delta_{S|A}\rangle / \|\Delta_S\|^2);
- directional cosine;
- relative residual: (\|\Delta_{S|A}-\Delta_S\| / \|\Delta_S\|).

Without nuisance, the matched construction should yield a projection gain and cosine of one (within floating-point tolerance) and zero residual. With nuisance, the connectivity transform can change the signal response; the resulting deviation is an observed signal × nuisance interaction in connectivity space, not proof of causal direction.

For zero signal, these metrics are null because no target direction exists. Do not replace an undefined direction with an arbitrary zero or one.

## 5. Integrated dose-response surface

The 4 × 4 common-random-numbers grid is now evaluated with all three estimators. Signal scales [0, 0.5, 1, 2] and nuisance component-variance fractions [0, 0.05, 0.10, 0.20] reuse the same raw signal and nuisance innovations throughout the grid.

Each point therefore includes:
- the four raw factorial cells and their invariant checks;
- the three estimator-specific similarity/displacement summaries;
- connectivity-space signal projection gain, cosine and residual;
- clean/nuisance variance, covariance and combined variance.

The focused workflow checks estimator identity/order, score ranges, zero-signal null semantics, dose calibration, common-random-numbers fingerprints and content-addressed archival.

## 6. What the next benchmark must add

This implementation is still a single synthetic transient instance. The estimator-level confirmatory surface should next run:
- ≥1,000 independently seeded innovation sets for screening;
- ≥10,000 Monte Carlo null replicates for confirmatory null calibration where feasible;
- multiple ROI/network sizes, atlas mappings and connectivity estimators;
- common-mode, graph-smooth, heat-kernel, transient and measured physiology/motion nuisances;
- preprocessing operator graphs from one-step regression, pre-whitening, physiological lags, censoring and filtering;
- site-held-out and participant-grouped evaluation;
- confidence intervals, coverage, null false-positive rate, direction-aware recovery, wrong-direction rejection, attrition and retained DoF.

Do not confuse M Monte Carlo replicates with N independent participants. The current estimator-response score is a deterministic per-matrix summary, not an inferential result.

## Claim limits

The module evaluates existing RCAF matrix similarity estimators on synthetic Pearson connectivity matrices. It does not establish superiority to fMRIPrep, XCP-D, CONN or current denoising SOTA. It does not estimate empirical artifact distributions or qualify biological/clinical outcomes. A superiority claim still requires matched public data, native comparator settings, frozen external evaluation and independent replication.
