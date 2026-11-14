# AIRROAD-MIMO-REPEATED-PERIOD-COMPLEX-UQ-001 — joint MIMO C/P estimator

**Status:** V0.1 exact-rational two-coordinate repeated-period reference  
**Tracking issue:** [#12013](https://github.com/Luminous-Dynamics/symthaea/issues/12013)  
**Composes with:** #11986 hierarchical FRF UQ, #11990 complementary covariance, #11988 MIMO excitation admissibility, #11914 shared calibration identity

> This reference computes second-order sample statistics from synthetic repeated complex FRF vectors. It does not classify physical measurement noise or create a hard uncertainty bound.

## 1. Research basis

González, Classens, Rojas, Oomen and Hjalmarsson (2025-10-30), [Finite Sample MIMO System Identification with Multisine Excitation](https://arxiv.org/abs/2510.26929), derives finite-sample MIMO FRF estimator distribution/covariance results while treating aliasing and the conditions for unbiasedness, uncorrelatedness and consistency.

Wind, Meijer, Dolk and Heemels (2026-11-06), [Sample-Covariance Regularization for Data-Driven Predictive Control from Frequency Response Functions under Colored Noise](https://arxiv.org/abs/2611.10226), uses repeated periods to construct the sample covariance of the mean FRF vector and preserve cross-frequency dependence. This is an adjacent predictive-control/statistical estimator comparator, not an IQC synthesis method.

## 2. Full-vector sample estimators

For \(P\) repeated complex FRF vectors \(h^{[r]}\\in\\mathbb C^n\), with canonical coordinate order fixed by the acquisition passport:
\[
\\bar h=\\frac1P\\sum_{r=1}^{P}h^{[r]},\\qquad
d^{[r]}=h^{[r]}-\\bar h.
\]

Compute both second-order moments of the mean:
\[
\\widehat C=\\frac{1}{P(P-1)}\\sum_{r=1}^{P}d^{[r]}(d^{[r]})^H,
\\qquad
\\widehat P=\\frac{1}{P(P-1)}\\sum_{r=1}^{P}d^{[r]}(d^{[r]})^T.
\]

The augmented covariance is:
\[
\\widehat C_a=
\\begin{bmatrix}
\\widehat C&\\widehat P\\\\
\\widehat P^*&\\widehat C^*
\\end{bmatrix}.
\]
For complex errors, \(C\) alone is complete only under a justified properness hypothesis \(P=0\).

Let the real vectorization be \([\\Re h^T,\\Im h^T]^T\). The verifier independently computes:
\[
\\widehat\\Sigma_{xy}=\\frac{1}{P(P-1)}
\\sum_{r=1}^{P}
\\begin{bmatrix}\\Re d^{[r]}\\\\\\Im d^{[r]}\\end{bmatrix}
\\begin{bmatrix}\\Re d^{[r]}\\\\\\Im d^{[r]}\\end{bmatrix}^{T}.
\]
The real covariance must map back to the exact same \(C,P\), and vice versa.

## 3. Frozen exact two-coordinate fixture

Five repeated period vectors are:
\[
h^{[1]}=[1,0]^T,\\quad h^{[2]}=[0,1]^T,\\quad
h^{[3]}=[i,0]^T,\\quad h^{[4]}=[0,i]^T,\\quad
h^{[5]}=[-1-i,-1-i]^T.
\]
The sample mean is zero. The real coordinate order is \([\\Re z_0,\\Re z_1,\\Im z_0,\\Im z_1]\), giving:
\[
\\Sigma_{xy}=\\frac1{20}
\\begin{bmatrix}
2&1&1&1\\\\
1&2&1&1\\\\
1&1&2&1\\\\
1&1&1&2
\\end{bmatrix}.
\]
The exact determinant is \(1/32000\); its leading principal minors are \(1/10,3/400,1/2000,1/32000\).

The complex estimates are:
\[
C=\\begin{bmatrix}1/5&1/10\\\\1/10&1/5\\end{bmatrix},
\\qquad
P=\\frac{i}{10}\\begin{bmatrix}1&1\\\\1&1\\end{bmatrix}.
\]
The augmented matrix has determinant \(1/2000=16\\det(\\Sigma_{xy})\) and leading principal minors \(1/5,3/100,1/250,1/2000\). All are positive. This is an improper vector-valued complex sample even though \(C\) is itself real and symmetric.

## 4. Finite-period rank semantics

The centered real sample matrix has at most \(P-1\) independent rows. For \(n\) complex coordinates, its dimension is \(2n\), so:
\[
\\operatorname{rank}(\\widehat C_a)
=\\operatorname{rank}(\\widehat\\Sigma_{xy})\\leq\\min(P-1,2n).
\]
With four periods and two complex coordinates, full rank is impossible; singularity is expected, not a failed covariance estimator. The result must be represented as a lower-dimensional support. No diagonal jitter, pseudo-observation or undocumented shrinkage may turn this into a positive-definite certificate.

## 5. Calibration and dependence semantics

A campaign-fixed additive complex calibration offset cancels when each period vector is centered. Consequently, the repeated-period estimator does not identify that offset or its variance. Keep it under #11914/#11986 as a shared nuisance with a stable identity across all frequencies and channels.

A separate calibration covariance may be added to period-noise covariance only when the cross-source independence assumption is declared. Otherwise, retain the cross-covariance and cross-complementary-covariance terms or return NOT_EVALUABLE. Statistical moments remain distinct from deterministic calibration intervals and hard model-discrepancy sets.

## 6. Claim ceiling

V0.1 is exact-rational estimator arithmetic on synthetic data. It does not establish physical period independence, Gaussianity, properness, calibration variance, confidence coverage, deterministic hard bounds, continuous-frequency adequacy, IQC feasibility, robust stability or airworthiness.
