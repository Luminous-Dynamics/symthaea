# RCAF Matched Factorial Dose-Response Surface v1

**Status:** executable synthetic response-surface construction; not an estimator benchmark or empirical fMRI qualification.  
**Base:** four-cell counterfactual engine and family-agnostic nuisance variance calibrator.  
**Purpose:** vary target strength and nuisance fraction without changing the underlying synthetic innovations.

## 1. Design: common random numbers across the grid

Let X be one frozen clean observation matrix, S one frozen target-signal innovation, and A one frozen raw nuisance innovation. Construct the four cells at every grid point:

- X
- X + sS
- X + A_v
- X + sS + A_v

Here s is a declared signal-amplitude multiplier. A_v is obtained by applying the family-agnostic matrix calibrator to the same raw A at a requested component-variance fraction v:

[
v = \frac{\operatorname{Var}(A_v)}
{\operatorname{Var}(X)+\operatorname{Var}(A_v)}.
]

The signal innovation and raw nuisance innovation are not regenerated between points. This is a common-random-numbers design: changing s or v changes only the declared intervention scale, not the stochastic realization. Stable raw-input fingerprints are included in the report.

The current smoke grid is 4 × 4:

- Signal scales: 0, 0.5, 1, 2.
- Nuisance component-variance fractions: 0, 0.05, 0.10, 0.20.

Each of the 16 points includes all four matched counterfactual cells, their diagnostic fingerprints, requested and realized nuisance fractions, the calibration multiplier, clean/nuisance variance, covariance, combined observed variance and construction-invariant residuals.

## 2. Executable implementation

The module `research/rcaf/src/factorial_dose_response.rs`:

1. Validates non-empty, finite, non-negative and strictly increasing dose axes.
2. Validates all time × ROI matrix dimensions and finite input values.
3. Scales the same raw target-signal matrix at each signal dose.
4. Calibrates the same raw nuisance matrix at each component-variance dose.
5. Runs the four-cell factorial engine at each Cartesian-product point.
6. Reports stable input fingerprints and point-level cell fingerprints.
7. Fails closed on calibration error, invalid grids, dimension mismatch, non-finite output or invariant failure.

The CLI smoke example uses an explicit event-triggered synthetic transient, but the sweep module itself is nuisance-family agnostic. The same module can be used with common-mode, graph-smooth, heat-kernel, physiological or future empirical nuisance matrices, provided the observation geometry is matched.

## 3. Metamorphic controls

The focused workflow independently checks:

- all 16 Cartesian-product points and deterministic row-major ordering;
- each realized component fraction matches its requested fraction;
- nuisance component variance rises monotonically with the dose axis;
- nuisance-only cell fingerprints are identical across signal scales at a fixed nuisance dose;
- zero nuisance collapses nuisance-only to clean-null and combined to signal-only;
- zero signal collapses signal-only to clean-null and combined to nuisance-only;
- all four-cell matched-component and additivity invariants remain within numerical tolerance;
- covariance-inclusive combined variance is consistent with Var(X) + Var(A) + 2 Cov(X,A).

These are construction and magnitude-calibration checks. They do not yet quantify whether an estimator recovers the signal or rejects nuisance.

## 4. What the response surface makes possible

The next estimator-level surface should run the same frozen four-cell matrices through each pinned connectivity estimator and preprocessing operator graph, then calculate at every (signal scale, nuisance fraction) point:

- nuisance-only false-positive rate;
- directional effect recovery and wrong-direction rejection;
- effect attenuation, inflation and drift;
- confidence interval coverage and width;
- network topology error and edge-sign error;
- retained temporal degrees of freedom;
- operator-order sensitivity and final nuisance leakage;
- correct abstention on equivalent worlds and false abstention on identifiable controls.

Repeat with independent nuisance instances and participant/site clusters. The present common-random-numbers surface is one replicate-level design. Screening should span at least 1,000 independently seeded raw innovation sets; confirmatory null calibration should target 10,000 Monte Carlo replicates where feasible. Monte Carlo replicate count M does not substitute for independent participant count N.

## 5. Proposed qualification targets

Targets below are proposed engineering gates, not standards or achieved results:

| Qualification | Proposed target |
|---|---:|
| Null false-positive rate | ≤5% |
| Wrong-direction acceptance | ≤1% |
| Moderate-effect directional recovery | ≥80% |
| Lower 95% recovery bound | ≥75% |
| Nominal 95% interval coverage | 93–97% |
| Correct abstention on exact-equivalence controls | 100% |
| False abstention on clear identifiable controls | ≤5% |
| Artifact-induced drift | ≤10% of minimum meaningful effect |

A single average score is insufficient. The result should be a multi-objective surface with confidence intervals, participant/site stratification and Pareto-frontier comparisons. A denoiser that increases tSNR by removing relevant task signal is not an improvement.

## 6. Claim limits

The current 4 × 4 example constructs matched synthetic counterfactual cells for one transient nuisance instance. It does not run a connectivity estimator, benchmark fMRIPrep/XCP-D/CONN, estimate empirical scanner artifacts or qualify a biological endpoint. The response surface is the input-design substrate for the next estimator-level qualification stage, not a SOTA result.
