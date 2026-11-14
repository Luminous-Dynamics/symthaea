# AIRROAD-HIERARCHICAL-FRF-UQ-001 — repeat-period covariance and shared nuisance separation

**Status:** V0.1 exact-rational statistical-estimator reference  
**Tracking issue:** [#11986](https://github.com/Luminous-Dynamics/symthaea/issues/11986)  
**Integrates:** #11914 cross-frequency shared calibration; #11946 matched IQC benchmark; #11948 solver core

> This reference verifies covariance arithmetic and uncertainty-source semantics on a synthetic exact-rational fixture. It is not a physical noise estimate or a hard uncertainty certificate.

## 1. Research motivation

Wind, Meijer, Dolk and Heemels (2026-11-06), [*Sample-Covariance Regularization for Data-Driven Predictive Control from Frequency Response Functions under Colored Noise*](https://arxiv.org/abs/2611.10226), derive an FRF sample covariance from repeated independent periods. Their covariance is formed over the complete vector of positive-frequency FRF estimates rather than only as separate per-frequency marginal variances. This retains cross-frequency covariance from colored measurement noise.

For P independent repetitions with full FRF vectors (h^{[s]}\in\mathbb C^M), define

[
\bar h=\frac{1}{P}\sum_{s=1}^{P}h^{[s]},\qquad
\Delta h^{[s]}=h^{[s]}-\bar h.
]

Their estimator for the covariance of the mean FRF vector is

[
\widehat C_H=
\frac{1}{P(P-1)}
\sum_{s=1}^{P}\Delta h^{[s]}(\Delta h^{[s]})^H.
]

The estimator can be obtained from repeated periods without a parametric noise model. Its statistical interpretation still depends on the repeated-period assumptions; sample covariance is not a deterministic all-realizations bound.

Primary source details: arXiv:2611.10226, Section II-E, equations (33)–(38), and Section IV, equations (60)–(63).

## 2. Hierarchical uncertainty model

AIRROAD must not compress every uncertainty source into one radius or one covariance matrix without preserving its origin.

Represent an FRF vector schematically as

[
\hat h=h_{\rm true}+e_{\rm period}+J_{\rm cal}\xi_{\rm cal}+e_{\rm model}.
]

- (e_{\rm period}): repeat-varying stochastic measurement noise;
- (\xi_{\rm cal}): shared calibration/acquisition nuisance, fixed over the scope of the campaign;
- (J_{\rm cal}): sensitivity mapping the same nuisance into every affected frequency and channel;
- (e_{\rm model}): separate physical/model discrepancy with its own justified hard-set or statistical semantics.

The centered repeat-period estimator removes any additive campaign-fixed offset. It therefore cannot, by itself, identify or estimate a shared calibration offset fixed across those repetitions.

If period noise and calibration are independent random sources, then

[
C_{\rm total}=C_{\rm period}+J_{\rm cal}C_{\rm cal}J_{\rm cal}^H.
]

If they are correlated, the cross-covariance terms must be included. If dependence is unknown, do not silently set the cross terms to zero.

A deterministic calibration interval and deterministic model discrepancy remain hard sets. They are not Gaussian covariance merely because a downstream method accepts covariance.

## 3. Exact-rational V0.1 fixture

Use two frequency coordinates and three synthetic repeated-period FRF estimates:

[
h^{[1]}=\begin{bmatrix}2\\3\end{bmatrix},\quad
h^{[2]}=\begin{bmatrix}0\\2\end{bmatrix},\quad
h^{[3]}=\begin{bmatrix}1\\1\end{bmatrix}.
]

Their mean is (ar h=[1,2]^T), and their centered deviations are

[
d_1=[1,1]^T,\quad d_2=[-1,0]^T,\quad d_3=[0,-1]^T.
]

The full-vector covariance of the mean is

[
C_{\rm period}
=\frac{1}{3(3-1)}\sum_s d_sd_s^T
=\begin{bmatrix}1/3&1/6\\1/6&1/3\end{bmatrix}.
]

It is positive definite with determinant (1/12). The off-diagonal (1/6) is the frozen cross-frequency covariance discriminator.

For a separate campaign-fixed random calibration gain coordinate with variance (1/100) and sensitivity

[
J_{\rm cal}=[1,2]^T,
]

the calibration covariance is

[
C_{\rm cal}=\frac{1}{100}J_{\rm cal}J_{\rm cal}^T
=\begin{bmatrix}1/100&2/100\\2/100&4/100\end{bmatrix}.
]

Only under the explicitly declared independence assumption may the sources be added:

[
C_{\rm total}=C_{\rm period}+C_{\rm cal}
=\begin{bmatrix}103/300&14/75\\14/75&28/75\end{bmatrix},
\qquad \det C_{\rm total}=7/75.
]

A diagonal-only approximation keeps the same two marginal variances but changes the joint covariance and joint law. It is an ablation, not an equivalent representation.

## 4. Required checks

The exact Fraction verifier:
- derives the mean and centered deviations from raw period vectors;
- implements the exact (1/[P(P-1)]) estimator;
- verifies symmetry and exact 2×2 PSD principal minors;
- derives the shared calibration covariance from the declared sensitivity and variance;
- adds sources only under an explicit zero-cross-covariance assumption;
- verifies the combined covariance and determinant;
- creates a same-marginal diagonal-only counterfactual and proves it is different;
- shows that adding a common campaign-fixed offset to every repetition leaves centered sample covariance unchanged;
- rejects P < 2, stale campaign/acquisition identity, calibration-ID mismatch, wrong denominator, uncentered covariance, lost cross-frequency covariance, unknown cross-source dependence, and statistical-to-hard-set promotion.

## 5. Matched benchmarks

Compare campaigns with identical excitation, period count and per-frequency marginals:

1. Full-vector repeated-period sample covariance.
2. Per-frequency block-diagonal covariance ablation.
3. Shared calibration covariance under declared independence.
4. Explicit cross-covariance between period noise and calibration.
5. Hard-set calibration kept separate from stochastic covariance.
6. Incorrect pooling of fixed calibration into repeat-varying covariance.
7. Finite-sample covariance-estimation uncertainty and empirical confidence-region coverage.
8. Rank-deficient and ill-conditioned covariance cases.

Report empirical coverage, proper scoring/calibration, off-diagonal covariance retention, joint versus marginal uncertainty, covariance-estimation error versus P, predictor/controller cost where appropriate, measurement time, and downstream robust decisions only when a valid hard-set or chance-constrained bridge exists.

The source paper reports that its regularized predictive-control method can reduce the required number of measured periods in its tested setting (including a three-versus-ten-period comparison). Reproduce that objective under its own assumptions; do not turn it into a universal AIRROAD sample-complexity claim.

## 6. Claim ceiling

V0.1 validates exact estimator arithmetic and uncertainty-source composition semantics only. It does not establish Gaussianity, actual period independence, finite-sample coverage, physical calibration variance, a deterministic hard bound, IQC feasibility, robust stability or flight safety.
