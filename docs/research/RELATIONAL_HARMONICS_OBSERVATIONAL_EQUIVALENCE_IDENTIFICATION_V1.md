# RH-006 Observational-Equivalence / Identification Gate — v1

## Status

**Research diagnostic only. Not approved for formal inference.**

Current selection remains:

`stop-assumption-failure`

The purpose of this gate is to sit **before** nuisance estimation, inference-method selection, and bootstrap execution.

## Scientific question

RH-006 currently observes a `RelationalPredictionSample` containing feature-time variables, outcome-time variables, relational features, common context, and the independently observed future outcome. The current estimator-compatibility contract identifies how the declared forecasting estimator behaves, but it does not by itself establish point identification of a future latent nuisance quantity.

Any future latent construction of the form

[
F=q+v,qquad Y=bq+e,
]

with (vperp(q,e)), jointly Gaussian variables, and observable covariance

[
Sigma_{F,Y} =
egin{pmatrix}
2 & 0.5\
0.5 & 1.5
end{pmatrix}
]

must pass an observational-equivalence attack before (K), (Xi), (C), or any downstream branch quantity is treated as point identified.

## Executed result

The deterministic harness is:

`scripts/rh006_observational_equivalence.py`

It checks:

1. two explicit latent DGP witnesses with identical observable covariance;
2. a continuum of admissible (C) values, including the contraction boundary;
3. unbounded (K) with the observable law fixed;
4. branch-regime changes under (D(C)=0.25-C^2);
5. an explicit identification-rescue construction using a second calibrated indicator of (q) and a valid excluded instrument.

The harness was executed independently with deterministic parameters and a 100,000-draw Gaussian sanity simulation. The finite-sample simulation is calibration only; the identification result is analytic.

## Exact observational-equivalence witness

Model A:

- (operatorname{Var}(q)=1)
- (operatorname{Cov}(e,q)=0.4)
- (K=0.4)
- (C=0.3368607684ldots)
- (b=0.1)
- (operatorname{Var}(e)=1.41)
- (operatorname{Var}(v)=1)

Model B:

- (operatorname{Var}(q)=0.5)
- (operatorname{Cov}(e,q)=-0.4)
- (K=-0.8)
- (C=-0.4923659639ldots)
- (b=1.8)
- (operatorname{Var}(e)=1.32)
- (operatorname{Var}(v)=1.5)

Both satisfy:

[
operatorname{Var}(F)=2,qquad
operatorname{Cov}(F,Y)=0.5,qquad
operatorname{Var}(Y)=1.5.
]

Because the construction is jointly Gaussian, equality of the observable mean/covariance parameters gives equality of the entire observable Gaussian law.

Therefore:

[
oxed{
	ext{same observable law}

otRightarrow
	ext{same }Xi

otRightarrow
	ext{same }K

otRightarrow
	ext{same }C
}
]

under the minimal model.

## Identified-set result

For

[
t=operatorname{Cov}(F,Y),qquad
V=operatorname{Var}(Y),qquad
s=operatorname{Var}(q),
]

take any admissible (s) with

[
rac{t^2}{V}<s<operatorname{Var}(F).
]

Define

[
	au^2=V-rac{t^2}{s}.
]

For any (cin(-1,1)), set

[
k =
rac{c}{sqrt{1-c^2}}sqrt{s	au^2},
qquad
b=rac{t-k}{s},
qquad
operatorname{Var}(e)=rac{k^2}{s}+	au^2.
]

Then (C=c) while the observable covariance remains unchanged.

The exact PSD-boundary construction also admits (C=pm1).

The resulting minimal-model identified sets are:

| Quantity | Classification | Identified set / value |
|---|---|---|
| (operatorname{Var}(F)) | O | ({2}) |
| (operatorname{Cov}(F,Y)) | O | ({0.5}) |
| (operatorname{Var}(Y)) | O | ({1.5}) |
| (operatorname{Var}(q)) | PI | ([1/6,2]), subject to boundary convention |
| (operatorname{Var}(v)) | PI | ([0,11/6]), subject to boundary convention |
| (operatorname{Cov}(e,q)=Xi) | NI | (mathbb R) |
| (K=Xi/operatorname{Var}(q)) | NI | (mathbb R) |
| (C) | NI | ([-1,1]) |
| (b) | NI | (mathbb R) |
| (operatorname{Var}(e)) | NI | non-singleton |
| (D(C)=0.25-C^2) | NI | ([-0.75,0.25]) |
| branch existence | NI | can be two roots, a double root, or no real roots |

The critical point is that **branch existence is itself not identified** when the branch discriminant depends on (C).

## Estimation trap

Fitting

[
Y=bq+e
]

by OLS on the same sample forces the fitted residual to be sample-orthogonal to the fitted regressor. Consequently, residual-then-correlate cannot recover (operatorname{Cov}(e,q)) under endogeneity; it mechanically manufactures near-zero sample covariance.

This is a validity boundary for any future RH-006 nuisance estimator, not a property of the current forecasting estimator.

## Required machine gate

Every proposed nuisance quantity should carry one of:

- **O — Observable:** directly a functional of the observed `RelationalPredictionSample`.
- **MI — Model-identified:** uniquely determined under an explicit identification model and its checked assumptions.
- **PI — Partially identified:** only an identified set is determined.
- **NI — Not identified:** observationally equivalent admissible DGPs produce different target values.

The minimum executable attack is:

[
oxed{
	ext{observable law}
ightarrow
	ext{latent witness family}
ightarrow
	ext{target variation}
ightarrow
	ext{identified set}
ightarrow
	ext{branch/surface image}
}
]

A target that varies across observationally equivalent witnesses must not be promoted to a point-estimated nuisance parameter.

## What can rescue identification?

The harness demonstrates one explicit rescue, not an assertion that RH-006 already possesses it.

A defensible point-identification structure is:

1. two calibrated independent indicators of (q) with known unit loading, so (operatorname{Var}(q)) is recovered from their cross-covariance;
2. a valid excluded instrument (Z) with (operatorname{Cov}(Z,e)=0) and (operatorname{Cov}(Z,q)
eq0), so (b) is identified;
3. then
   [
   Xi=operatorname{Cov}(F,Y)-boperatorname{Var}(q)
   ]
   and (operatorname{Var}(e)) are identified, giving (C).

The executable rescue recovers the Model-A value:

[
C=0.3368607684ldots
]

under those added assumptions.

The repository must therefore not infer that a currently available signal is an instrument merely because it is predictive. Exogeneity and exclusion are structural assumptions requiring experimental or design justification.

## Placement in RH-006

The existing inference path is already conservative: the current estimator-applicability flag is false and the compiled selector resolves to `stop-assumption-failure` when compatibility is not approved.

The identification gate should become an earlier prerequisite:

[
	ext{observable schema}
ightarrow
	ext{identification audit}
ightarrow
	ext{identified-set status}
ightarrow
	ext{nuisance estimation}
ightarrow
	ext{finite-sample calibration}
ightarrow
	ext{inference selection}.
]

No inference selector should be allowed to turn an NI or PI nuisance quantity into a point-null or point-alternative claim without an explicit model-identification receipt.

## Literature boundary

The result is consistent with latent-variable and SEM identification work: multiple indicators can be essential for identification, and graphical/algebraic instrumental-variable criteria can identify latent parameters when the required measurement and exclusion assumptions hold. Proxy-variable approaches likewise require explicit rank/completeness-style conditions; proxies are not automatically identifying merely because they correlate with a latent confounder. Neyman-orthogonal / DML methods protect inference against nuisance-estimation error after the estimand is defined and identified; orthogonality is not an identification theorem.

Forecast-comparison literature also reinforces the existing RH-006 separation between identification and inference: nested-model bootstrap procedures are tied to the exact forecast-estimation design, while rolling/recursive dependence and model instability can invalidate naive conditional procedures.

## Nonclaims

- No RH-006 latent nuisance has been shown to be point identified from the current experiment.
- No valid instrument has been established.
- The finite-sample simulation is not a proof of equal observable laws.
- The harness does not validate the current RH-006 estimator.
- Formal p-values, confidence intervals, and bootstrap-based inference remain disabled.
