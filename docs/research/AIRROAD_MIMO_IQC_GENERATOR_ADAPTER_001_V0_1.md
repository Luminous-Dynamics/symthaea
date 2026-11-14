# AIRROAD-MIMO-IQC-GENERATOR-ADAPTER-001 — shared evidence coordinates for inter-frequency generator sets

**Status:** V0.1 exact-rational two-frequency adapter reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11926  
**Composes with:** #11914 cross-frequency UQ, #11919 exact generator remainder, #11924 adaptive mesh, #11801 robust MIMO stability

## 1. Purpose and SOTA boundary

The October 2026 preprint by Spin, Heemels, van de Wouw and Toth demonstrates finite-data MIMO IQC controller synthesis from uncertain sampled FRFs using piecewise-convex inter-frequency generator sets and finite-dimensional LMIs/SDP, with an industrial wire-bonder validation: https://arxiv.org/abs/2610.20248

AIRROAD does not claim to invent IQC, generator interpolation or finite-data robust MIMO synthesis. It provides a narrow evidence adapter that binds joint calibration/acquisition nuisance coordinates to every frequency/matrix record, and attaches independently replayable inter-frequency remainder certificates.

The output is a typed uncertainty representation, not an IQC or controller/stability PASS.

## 2. V0.1 shared generator coordinates

Nominal matrices are:

    Gbar_1 = I2 at omega_1=1
    Gbar_2 = 2 I2 at omega_2=2

One common deterministic hard coordinate theta∈[-1,1] applies to every matrix element at both knots:

    G_1(theta) = I2 + (1/10)*theta*I2
    G_2(theta) = 2I2 + (1/5)*theta*I2

The coordinate identity, acquisition root and calibration certificate are shared. Therefore:

    G_2(theta) - 2G_1(theta) = 0

for every admissible theta. The joint uncertainty set has one scalar coordinate, not two independent frequency coordinates.

If independent theta_1 and theta_2 are substituted while preserving the same marginal domains [-1,1], the diagonal relation becomes:

    G_2(theta_2)-2G_1(theta_1)
      = (theta_2-theta_1)/5 * I2

and its scalar relation interval is [-2/5,+2/5]. This is a strictly larger product set. Equal marginals do not imply the same joint generator object.

## 3. Inter-frequency remainder

Add the exact translated residual from #11919:

    R(omega) = 16*(omega-1)*(2-omega)*(omega-3/2)^2 * P
    P = [[0,1],[0,0]]
    omega in [1,2]

With x=omega-1, the scalar polynomial is exactly:

    d(x)=16*x*(1-x)*(x-1/2)^2

The factor P has Frobenius norm one, and #11919 certifies the exact interval supremum is 1/4. Thus the piecewise generator artifact includes an additive Frobenius ball of radius 1/4 over the entire interval.

The resulting object is:

    G(omega,theta) in Gbar_interp(omega)
                         + Gtheta_interp(omega)*theta
                         + (1/4)*B_F

where the same theta coordinate is preserved at both knots. The last term bounds only the declared interpolation residual, not all physical/model uncertainty.

## 4. Required evidence semantics

A consumer must receive:
- frequency knot and matrix element identities;
- nominal matrix and per-coordinate generator matrices;
- stable shared nuisance coordinate IDs and exact domains;
- acquisition root and calibration certificate scope;
- model/topology/operating-point/port/frequency identity;
- hard-set versus statistical class;
- inter-frequency interval and certified remainder radius/method;
- explicit statement that IQC synthesis is not yet performed.

A marginal projection may be generated for a solver interface, but it must not erase the original joint coordinate dependency. If dependency evidence is unknown, preserve UNKNOWN instead of assuming shared or independent.

## 5. Negative controls and benchmarks

The exact oracle verifies the shared affine identity for all theta endpoints and symbolic coefficient equality; it derives the independent product relation interval; verifies the translated exact remainder radius; and rejects distinct IDs presented as one shared coordinate, incomplete element coverage, stale calibration/topology, statistical-to-hard promotion, interval gaps, underestimated remainder and an IQC-PASS mislabel.

Targets, not achieved results:
- 100% exact Fraction agreement with the shared generator/remainder fixture;
- zero loss of common nuisance dimension when deriving marginal projections;
- zero frequency-independence claims without experiment/acquisition evidence;
- zero statistical-to-deterministic promotion;
- zero full-band gaps;
- report shared dimension, independent product dimension, marginal widths, remainder radius, set-inflation factor and solver feasibility separately.

## 6. Claim ceiling

V0.1 is a synthetic two-frequency shared-generator and analytic-remainder adapter. It does not implement IQC synthesis, qualify physical FRFs, establish aircraft robust stability, or create flight-safety/airworthiness/certification evidence.
