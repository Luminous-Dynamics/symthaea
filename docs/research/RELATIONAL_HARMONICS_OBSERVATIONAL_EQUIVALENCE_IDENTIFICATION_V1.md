# RH-006 Observational-Equivalence / Identification Gate — v1

## Status

**Research diagnostic only. Not approved for formal inference.**

Current selection: `stop-assumption-failure`.

The gate is intentionally before nuisance estimation, inference-method selection, and bootstrap execution.

## Core distinction

RH-006 currently observes a `RelationalPredictionSample`: feature-time variables, outcome-time variables, relational features, common context, and an independently observed future outcome.

That predictive observable is not automatically a measurement model for a latent state.

Let `T(theta)` be the observable-data law induced by latent DGP parameters `theta`, and let `psi(theta)` be a proposed latent target. At observable law `P`, define:

```text
I_psi(P) = { psi(theta) : T(theta) = P }
```

Point identification means `I_psi(P)` is a singleton. A non-singleton set is partial identification. Two admissible observationally equivalent witnesses with different target values establish non-identification for the stated model class.

Therefore:

```text
coordinate invariance != statistical identification
```

An invariant target can remain nonidentified when multiple latent DGPs induce the same observable law.

## Adversarial latent model

Use the minimal construction:

```text
F = q + v
Y = b q + e
v independent of (q, e)
```

Assume joint Gaussian variables and observable covariance:

```text
Sigma_(F,Y) = [[2, 0.5],
               [0.5, 1.5]]
```

Gaussianity matters: equality of the observable mean and covariance parameters determines the full observable Gaussian law.

## Executed witness result

The harness is `scripts/rh006_observational_equivalence.py`.

The independent local run uses deterministic parameters and 100,000 Gaussian draws per witness. The simulation is a sanity check only; the identification result comes from the analytic construction.

### Witness A

```text
Var(q)     = 1.0
Cov(e,q)   = +0.4
K          = +0.4
C          = 0.3368607684...
b          = 0.1
Var(e)     = 1.41
Var(v)     = 1.0
```

### Witness B

```text
Var(q)     = 0.5
Cov(e,q)   = -0.4
K          = -0.8
C          = -0.4923659639...
b          = 1.8
Var(e)     = 1.32
Var(v)     = 1.5
```

Both produce:

```text
Var(F)     = 2.0
Cov(F,Y)   = 0.5
Var(Y)     = 1.5
```

So:

```text
same observable law
    !=> same Xi
    !=> same K
    !=> same C
```

## Identified-set construction

Let `t = Cov(F,Y)`, `V = Var(Y)`, and `s = Var(q)`.

For admissible `s` with `t^2 / V < s < Var(F)`, define:

```text
tau^2 = V - t^2 / s
```

For any `c` in `(-1, 1)` choose:

```text
k      = c / sqrt(1-c^2) * sqrt(s * tau^2)
b      = (t - k) / s
Var(e) = k^2 / s + tau^2
```

Then `C = c` while the observable covariance remains unchanged. The PSD boundary also permits `C = -1` and `C = +1`.

For the canonical observable covariance, the minimal-model sets are:

| Quantity | Class | Identified set / value |
|---|---|---|
| `Var(F)` | O | `{2}` |
| `Cov(F,Y)` | O | `{0.5}` |
| `Var(Y)` | O | `{1.5}` |
| `Var(q)` | PI | `[1/6, 2]` if `Var(v)=0` is allowed |
| `Var(v)` | PI | `[0, 11/6]` |
| `Xi = Cov(e,q)` | NI | `R` |
| `K = Xi / Var(q)` | NI | `R` |
| `C` | NI | `[-1,1]` |
| `b` | NI | `R` |
| `Var(e)` | NI | non-singleton |
| `D(C)=0.25-C^2` | NI | `[-0.75,0.25]` |
| branch existence | NI | two roots / double root / no real root |

If strictly positive `Var(v)` is required, the upper endpoint of `Var(q)` becomes open.

## Branch-regime attack

For `D(C)=0.25-C^2`:

```text
C = 0.00  -> D = +0.25    -> two real branches
C = 0.50  -> D =  0.00    -> double root
C = 0.99  -> D = -0.7301  -> no real branch
```

Thus a nonidentified nuisance can make the downstream qualitative regime itself nonidentified.

This is stronger than poor numerical conditioning.

## Estimation trap

If the same sample is used to fit `Y = b q + e` by OLS, the fitted residual is sample-orthogonal to the fitted regressor by construction.

Therefore:

```text
fit b
-> compute residual
-> correlate residual with q
```

cannot identify `Cov(e,q)` under endogeneity. It mechanically manufactures near-zero sample covariance.

This is a rejection rule for future latent-nuisance estimators. It is not a criticism of the current fixed-ridge forecasting estimator.

## Required classification

Every proposed nuisance quantity should carry one of:

- **O — Observable:** direct functional of `RelationalPredictionSample`.
- **MI — Model-identified:** unique only after an explicit identification model and its assumptions pass.
- **PI — Partially identified:** the data determine a non-singleton identified set.
- **NI — Not identified:** admissible observationally equivalent DGPs produce different target values.

The executable qualification order is:

```text
observable law
-> observational-equivalence witnesses
-> target variation
-> identified set
-> downstream surface image
-> only then estimation
```

A target that varies across admissible observationally equivalent witnesses must fail closed as `NI` rather than become a point-estimated nuisance.

## Exact-law requirement

Covariance equality is sufficient for the current adversarial witness because the model is Gaussian.

For real RH-006 time series, covariance matching alone is insufficient unless Gaussianity is itself part of the maintained model. The eventual harness should therefore prove observational equivalence at the law level, using an analytically equivalent process family or a common-innovation construction.

Empirical distance can be used as a falsification sanity check, but it must not be used as the proof of identification.

## Identification rescues

The harness demonstrates one explicit point-identification rescue:

1. two calibrated independent indicators of `q` with known unit loading, recovering `Var(q)` from cross-covariance;
2. a valid excluded instrument `Z` with `Cov(Z,e)=0` and `Cov(Z,q) != 0`, identifying `b`;
3. recovery of `Xi` from `Cov(F,Y) - b * Var(q)` and then recovery of `C`.

The rescue recovers:

```text
b  = 0.1
Xi = 0.4
C  = 0.3368607684...
```

The repository must not treat a predictive signal as a valid instrument without explicit exclusion and exogeneity justification.

Other defensible rescue families are randomized intervention, a rank-identified latent measurement model, IV/control-function designs, proximal proxy designs with explicit completeness-style assumptions, and honest partial-identification bounds.

The last option is important: **PI is a scientific result, not a failed experiment.** The identified set can be propagated through the downstream decision surface.

## Placement in RH-006

The current inference path already stops because the estimator-applicability artifact is not approved.

The identification gate should precede that layer:

```text
observable schema
-> identification audit
-> identified-set status
-> nuisance estimation
-> finite-sample calibration
-> inferential procedure selection
-> bootstrap / interval / p-value
```

A future inferential receipt should bind at least:

```text
observed_schema_id
target_id
classification
identification_model_id
identification_assumption_digest
observational_equivalence_suite_digest
identified_set_digest
downstream_surface_digest
```

The receipt should fail closed unless the target is `O` or `MI` and every required assumption has an explicit qualification result.

## Recommended adversarial order

```text
1. coordinate invariance
2. observational equivalence
3. identification classification
4. identified-set construction
5. downstream branch/surface image
6. estimator identification and leakage audit
7. finite-sample calibration
8. dependence characterization
9. inferential-method bridge
10. bootstrap re-estimation
```

This prevents a correct estimator for a nonidentified target from being mistaken for a solution to the identification problem.

## Literature boundary

Latent-variable and SEM identification research treats identification as a structural problem involving scaling, rank, measurement, or graphical assumptions. Recent work combining graphical and algebraic approaches follows the same separation.

Proxy and proximal approaches provide important rescue families, but their point-identification results depend on explicit proxy-separation and completeness-style assumptions. Recent 2026 work also shows that structural violations can break proximal identification even when proxies are predictive.

Nested forecast bootstrap methods are designed for the forecast-comparison problem and can accommodate recursive or rolling estimation, but they do not establish identification of a separate latent nuisance parameter.

## Nonclaims

- No RH-006 latent nuisance has been shown to be point identified from the current experiment.
- No valid instrument has been established.
- No current RH-006 feature is promoted to an instrument merely because it predicts the target.
- The finite-sample simulation is not the identification proof.
- The harness does not validate the full RH-006 estimator.
- No formal p-value or confidence interval is enabled.
- No bootstrap validity theorem is established.
- No hosted CI/Test/Clippy PASS is claimed by this branch.
