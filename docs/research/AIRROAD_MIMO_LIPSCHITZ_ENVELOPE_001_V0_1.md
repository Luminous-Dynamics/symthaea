# AIRROAD-MIMO-LIPSCHITZ-ENVELOPE-001 — continuous residual bounds from certified slope limits

**Status:** V0.1 synthetic exact-rational deterministic residual reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11885  
**Composes with:** #11882 exact continuous-band rational certificate, #11865 finite-knot non-identifiability, #11858 shared calibration, #11801 robust MIMO stability, ROB-DOE #4817, ROB-EXP #4800, ROB-SYSID #4801

## 1. Purpose

A rational analytic extremum proof is useful when the model residual is known exactly. Measured physical systems are different: we have bounded observations at a finite set of frequencies and need a justified way to bound what happens between them.

This reference tests a Lipschitz/derivative method. It never estimates the global Lipschitz constant from the same sparse measurements.

## 2. Continuous-envelope theorem

For a fixed residual R(omega) under a declared matrix norm, suppose a separately justified global bound L satisfies:

    ||R(omega)-R(omega_i)|| <= L*|omega-omega_i|

throughout the complete band.

If the sample observation at knot omega_i has an exact hard residual interval/ball:

    ||R(omega_i)|| <= |center_i| + epsilon_i

then:

    ||R(omega)|| <= min_i(
        |center_i| + epsilon_i + L*|omega-omega_i|
    )

for every omega in the band.

For uniform sample error epsilon, zero sample centers, sorted knots including both band endpoints, and maximum gap h, every frequency is within h/2 of a knot. Therefore:

    sup_band ||R(omega)|| <= epsilon + L*h/2

This is a continuous proof only when L is qualified over the whole band for the exact model, port mapping, operating point and units.

Without L, a finite set of residual observations does not supply a continuous-domain bound. Unknown regularity is not zero regularity.

## 3. Exact cubic counterexample

The synthetic residual is:

    r(omega)=omega*(omega-1/2)*(omega-1)

It is zero at all three observed knots:

    omega = {0,1/2,1}

The exact derivative is:

    r'(omega)=3*omega^2-3*omega+1/2

This derivative is convex. Its maximum on [0,1] occurs at either endpoint and is 1/2; its minimum occurs at omega=1/2 and is -1/4. Thus:

    |r'(omega)| <= L = 1/2

This is derived from the exact polynomial on the entire band, not inferred from its sample values.

Each observed residual is a deterministic hard interval centered at zero with radius epsilon=1/100. The maximum knot gap is h=1/2, so the maximum nearest-knot distance is 1/4:

    epsilon + L*h/2
      = 1/100 + (1/2)*(1/4)
      = 27/200

The exact residual is nonzero between the knots:

    r(1/4)=3/64
    r(3/4)=-3/64

Both values lie within the continuous cone envelope of radius 27/200. This is intentionally conservative: the certificate is a robust bound, not an exact reconstruction.

## 4. Evidence requirements for real use

A production residual-bound record must bind:
- exact source measurements and deterministic hard uncertainty semantics;
- frequency band and units;
- sorted knot identity and full endpoint/gap coverage;
- residual matrix and chosen norm/port order;
- independently qualified L value or local L_i values;
- derivative/regularity evidence source, model and operating-point identity;
- calibration/common-mode dependencies;
- method version and exact arithmetic/rounding semantics;
- unsupported gaps and claim ceiling.

A statistical confidence interval is not a deterministic hard interval. A sample-derived empirical slope is not a global derivative proof unless an independent model/theorem justifies that extrapolation.

## 5. Industry and SOTA positioning

The Lipschitz method is established mathematics, not AIRROAD novelty. Compare:
- NIST DLMF validated interval arithmetic and inclusion: https://dlmf.nist.gov/3.1.ii
- Bernstein range bounds for complex polynomials and rational functions: https://doi.org/10.1016/j.cam.2020.113377
- Data-consistent MIMO uncertainty/LMI methods: https://web.stanford.edu/~boyd/papers/unc_fdd.html
- Robust optimal experiment design for system identification: https://doi.org/10.1016/j.automatica.2006.12.013
- Robust MIMO experiment design over a continuous uncertainty set using the S-Lemma: https://research.tudelft.nl/en/publications/iterative-robust-experiment-design-for-mimo-system-identification/

The differentiator is to bind the derivative assumption, sample uncertainty, sample coverage, exact identities and residual envelope into a reproducible fail-closed engineering evidence artifact.

## 6. Future generalization

1. Matrix residuals under spectral norm.
2. Local frequency-dependent derivative bounds with complete interval cover.
3. Adaptive knot selection to reduce the cone envelope under cost constraints.
4. Shared calibration nuisance propagation through the residual balls.
5. Combination with rational/interval certificates from #11882.
6. Composition with the robust stability uncertainty model from #11801.

## 7. Benchmark targets (targets, not results)

- 100% exact Fraction agreement with the fixture.
- Zero continuous bound when L is missing.
- Zero acceptance of L below the exact derivative supremum.
- Zero uncovered frequency gap or missing endpoint.
- Zero promotion of statistical confidence intervals to deterministic hard guarantees.
- Correct enclosure of hidden between-knot residuals that are zero at all observed knots.
- Report L provenance, sample spacing, envelope width, number of knots, proof time and unsupported-domain rate separately.

## 8. Claim ceiling

V0.1 demonstrates one exact polynomial residual with an independently derived global slope bound. It does not infer real physical regularity from sparse observations, prove an aircraft converter follows the declared L, qualify calibration, or establish aircraft stability/safety/certification.
