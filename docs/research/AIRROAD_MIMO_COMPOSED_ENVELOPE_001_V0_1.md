# AIRROAD-MIMO-COMPOSED-ENVELOPE-001 — pointwise data consistency to full-band matrix-error bounds

**Status:** V0.1 synthetic exact-rational end-to-end composition  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11893  
**Composes:** #11891 pointwise LMI, #11885 Lipschitz envelope, #11882 rational continuous certificate, #11858 shared calibration, #11801 robust MIMO stability

## 1. Objective

Demonstrate one end-to-end chain from bounded MIMO input/output observations at frequency knots to a continuous-band induced-spectral-norm residual bound.

This is a proof-composition reference, not a new LMI solver, uncertainty library, system identifier, or circuit simulator.

## 2. Nominal model, residual and band

The nominal model is the 2x2 identity matrix. The declared matrix residual is:

    E(omega) = r(omega) P
    r(omega) = omega*(omega-1/2)*(omega-1)
    P = [0 1]
        [1 0]

The matrix P is orthogonal: P^T P = I. Therefore:

    ||E(omega)||_2 = |r(omega)|
    ||dE/domega||_2 = |r'(omega)|

The complete declared normalized-frequency band is [0,1]. Three measurement knots cover both endpoints:

    omega = {0, 1/2, 1}

The residual is exactly zero at every knot, but not between them.

## 3. Pointwise data consistency

At each knot, apply both basis inputs U=I2. The two measured output vectors equal the nominal predictions, and each measured output column has deterministic hard Euclidean error radius epsilon=1/100.

For each input column u=e_j:
- the nominal residual is zero;
- the true output error is bounded by epsilon;
- the exact squared gain bound is gamma=epsilon^2/||u||^2=1/10000.

The 3x3 block LMI is:

    [ gamma I_2       r_worst ]
    [ r_worst^T       u^T u   ] >= 0

with a worst-case residual witness of norm epsilon. The verifier checks every principal minor exactly. Because gamma I bounds the residual in every direction, this establishes a column norm bound for the deterministic hard ball, not merely the selected witness direction.

The two orthogonal inputs span the complete input space. Summing their column bounds gives:

    ||E(omega_i)||_F^2 <= 2/10000 = 1/5000
    ||E(omega_i)||_2^2 <= 1/5000

For a simple rational norm radius, use 1/70 because:

    (1/70)^2 = 1/4900 >= 1/5000

Thus every pointwise matrix error is bounded by 1/70 in induced spectral norm.

A rank-deficient input ensemble may retain per-direction information, but it cannot support this full 2x2 matrix claim.

## 4. Independent global derivative proof

The scalar residual polynomial is:

    r(omega)=omega^3 - (3/2)omega^2 + (1/2)omega

Its derivative is:

    r'(omega)=3*omega^2 - 3*omega + 1/2

This derivative is convex. Its maximum on [0,1] is 1/2 at either endpoint; its minimum is -1/4 at omega=1/2. Hence:

    sup_[0,1] |r'(omega)| = 1/2

Since P is orthogonal:

    sup_[0,1] ||dE/domega||_2 = 1/2

This bound is bound to the exact model, topology, operating point, frequency-unit and derivative-proof identities. It is derived from the synthetic polynomial itself, not guessed from the measurements.

## 5. Compose pointwise LMI and full-band coverage

The largest knot gap is h=1/2. Because both band endpoints are sampled, every frequency is no farther than h/2=1/4 from a knot.

For any omega, choose its nearest knot omega_i:

    ||E(omega)||_2
      <= ||E(omega_i)||_2
         + (1/2)*|omega-omega_i|
      <= 1/70 + (1/2)*(1/4)
      = 39/280

Therefore the composed result is:

    sup_(omega in [0,1]) ||E(omega)||_2 <= 39/280

This is the exact-rational full-band outer bound for this synthetic profile.

## 6. Conservatism known-answer

The actual residual family has a tighter exact supremum. Let t=omega-1/2:

    r = t^3 - t/4
    r' = 3t^2 - 1/4

At the two interior critical points, t^2=1/12 and r^2=1/432. The endpoints have zero residual, hence:

    sup_[0,1] ||E(omega)||_2^2 = 1/432

The squared ratio between the composed Lipschitz bound and the exact synthetic supremum is:

    (39/280)^2 / (1/432) = 41067/4900

This gives a measured conservatism target for future refinement. It also motivates the next step: adaptively insert measurements or use local derivative/analytic interval certificates where the envelope is unnecessarily wide.

## 7. Failure semantics

The composition is NOT_ESTABLISHED if:
- either input direction is missing;
- either frequency endpoint is missing;
- hard measurement uncertainty semantics are unavailable;
- the derivative bound is missing, understated or tied to another model/operating point/unit;
- topology, port order or model identities disagree;
- the derivative proof does not cover the full band.

A pointwise LMI alone is not a full-band result. A derivative bound alone is not a measurement-consistency result. Both are required.

## 8. Current SOTA comparison

Relevant prior methods:
- Hindi, Seong and Boyd formulate data-consistent frequency-domain MIMO uncertainty using LMIs and SDP with bounded noise: https://web.stanford.edu/~boyd/papers/unc_fdd.html
- A 2026 preprint by Spin, Heemels, van de Wouw and Toth combines uncertain sampled FRF data with piecewise-convex inter-frequency generator sets and IQC-based robust MIMO control synthesis, with an industrial wire-bonder demonstration: https://arxiv.org/abs/2610.20248
- That paper reports orthogonal multisine excitation at 247 measured frequencies, covariance-based generator sets with approximately 95.45% confidence, and robust inter-sample analysis/controller synthesis. It is a mandatory matched benchmark, not a capability we can claim to have originated.
- SINTEF MFT-NNLS combines Vector Fitting with residue perturbation for passive multiport macromodeling: https://www.sintef.no/en/software/vector-fitting/

The AIRROAD differentiator should now be evidence composition and aerospace fault/mode semantics: shared calibration dependency, configuration/topology identity, source/storage energy, contactor state continuity, post-fault supply and mission-condition linkage. The numerical IQC/interpolation machinery itself must be compared with established work.

## 9. Benchmark targets (targets, not results)

- 100% exact Fraction agreement with each pointwise LMI and the full-band envelope.
- Zero full matrix operator bounds from rank-deficient input excitation.
- Zero continuous bound without a fully qualified derivative/regularity certificate.
- Zero gaps in the frequency domain.
- Zero statistical-to-hard-bound promotion.
- Exact full-band upper bound 39/280 for the V0.1 composition.
- Report pointwise noise radius, gamma, excitation rank, derivative L, knot spacing, final envelope and conservatism ratio independently.
- Compare global Lipschitz, local Lipschitz, analytic rational extrema, Bernstein/interval bounds and the 2026 IQC generator interpolation on matched cases.
- No robust-stability PASS while any frequency region lacks admitted nominal-plus-discrepancy evidence.

## 10. Claim ceiling

V0.1 composes synthetic pointwise deterministic data-consistency evidence with an independently derived derivative bound for one known matrix residual family. It does not prove a physical plant obeys that derivative bound, establish physical calibration, prove aircraft robust stability, or create aircraft safety/airworthiness/certification evidence.
