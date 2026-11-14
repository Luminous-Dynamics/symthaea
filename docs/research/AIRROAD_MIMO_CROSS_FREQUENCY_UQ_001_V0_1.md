# AIRROAD-MIMO-CROSS-FREQUENCY-UQ-001 — shared calibration dependencies across frequency knots

**Status:** V0.1 exact-rational two-frequency dependency reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11914  
**Composes with:** #11858 shared channel calibration, #11891 pointwise data consistency, #11882/#11885 full-band residual certificates, #11893 composed envelope, #11801 robust MIMO stability

## 1. Why this matters to current SOTA

The October 14, 2026 preprint *Robust Data-Driven Controller Synthesis from Sampled and Uncertain Frequency Response Estimates using Integral Quadratic Constraints* develops finite-dimensional IQC/LMI synthesis using uncertain sampled MIMO FRFs and piecewise-convex inter-frequency generator sets. It demonstrates a wire-bonder application.

In its stochastic FRF generator construction, the paper explicitly assumes FRF estimates at different frequency points are statistically independent and their noise contributions are circular complex Gaussian. The wire-bonder example uses 247 measured multisine frequencies and reports approximately 95.45% (2-sigma) confidence generator sets. See:
- https://arxiv.org/abs/2610.20248
- https://arxiv.org/abs/2610.20248 (Sections 5.2.2 and 6.2)

This is a valid declared stochastic profile when its assumptions fit the acquisition. AIRROAD's research question is whether the acquisition/calibration evidence justifies frequency independence in a given campaign. A common calibration or acquisition nuisance can induce cross-frequency dependence. The profile must preserve that dependency rather than silently block-diagonalize it.

## 2. V0.1 deterministic shared hard-set fixture

Two nominal 2x2 matrices are frozen:

    G0(omega_1) = I2
    G0(omega_2) = 2*I2

A single deterministic real calibration factor applies to every matrix element at both knots:

    q = 1+u
    u in [-1/10,1/10]

The measured matrices are:

    Gm(omega_1) = q*I2
    Gm(omega_2) = 2*q*I2

Therefore, for every admissible q:

    Gm(omega_2) - 2*Gm(omega_1) = 0

The shared joint uncertainty is one-dimensional: every point lies on this affine relation.

The marginal sets are:
- omega_1: each diagonal entry in [9/10,11/10], off-diagonal entries zero;
- omega_2: each diagonal entry in [9/5,11/5], off-diagonal entries zero.

If the common nuisance is discarded and independent q_1,q_2 are permitted at the two frequencies, the product set admits:

    Gm(omega_2)-2*Gm(omega_1)
      = 2*(q_2-q_1)
      in [-2/5,2/5]

This product is strictly larger than the true shared-calibration joint set. It preserves the frequency-wise marginals but loses a physically meaningful cross-frequency dependency.

## 3. Evidence needed to assert sharing

The fixture binds one nuisance identity, acquisition root and calibration certificate to every matrix element at both knots. A shared identity is not inferred from equal numeric domains. A real campaign needs evidence that the same calibration factor and scope genuinely apply to those channels, frequencies and runs.

When that evidence is absent, do not assume either sharing or independence. Preserve the unresolved dependency and abstain from a stronger joint claim.

## 4. Separate statistical covariance companion

For a distinct statistical model, let q=1+xi with zero-mean xi and variance sigma^2=1/100. For one diagonal channel:

    z1=1+xi
    z2=2*(1+xi)

Then:

    Cov([z1,z2]^T)
      = (1/100)*[[1,2],[2,4]]

The cross-frequency covariance is 1/50. A block-diagonal covariance with the same marginals but zero cross term is a different joint distribution.

This companion is not a deterministic hard set and does not create a 100% guarantee. Statistical covariance, confidence regions and deterministic hard bounds remain distinct uncertainty classes.

## 5. Required evidence contract

Each frequency/matrix observation binds:
- frequency ID, exact frequency and unit;
- observed/driven port identities and matrix row/column;
- nominal model, topology and operating-point digest;
- calibration nuisance ID, acquisition root and calibration certificate;
- explicit nuisance scope over every covered record;
- deterministic or statistical uncertainty semantics;
- exact marginal and joint uncertainty representation.

A downstream uncertainty consumer may request marginal projections, but it must not mistake those projections for the complete joint multi-frequency model.

## 6. Benchmark targets (targets, not results)

- 100% exact-rational agreement with shared and independent joint-set fixtures.
- Zero shared-dependency claims without full scope and identity coverage.
- Zero frequency-independence claims when a common root is declared.
- Zero statistical-to-deterministic hard-set promotion.
- Report marginal widths, joint dimension, cross-frequency covariance, product-set inflation and downstream stability-margin impact separately.
- Compare frequency-independent generator products against joint affine/ellipsoidal sets on identical marginals.
- Preserve containment: exploiting a dependency must never remove any physically admitted realization.
- No robust-stability PASS if cross-frequency dependency semantics or inter-knot behavior remain unqualified.

## 7. Claim ceiling

V0.1 is a synthetic two-frequency shared-calibration reference. It does not establish the calibration scope of any real campaign, implement IQC controller synthesis, qualify measured aircraft FRFs, prove continuous-frequency plant uncertainty, or establish aircraft stability/airworthiness/certification.
