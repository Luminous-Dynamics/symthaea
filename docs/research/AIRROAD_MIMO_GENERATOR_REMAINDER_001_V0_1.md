# AIRROAD-MIMO-GENERATOR-REMAINDER-001 — certified inter-frequency interpolation radius

**Status:** V0.1 exact-rational matrix-curve generator-cover certificate  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11918  
**Composes with:** #11882 analytic envelope, #11885 Lipschitz envelope, #11893 composed MIMO envelope, #11914 cross-frequency uncertainty and #11801 robust MIMO stability

## 1. Relation to the current IQC baseline

The October 14, 2026 preprint by Spin, Heemels, van de Wouw and Toth constructs finite-data MIMO IQC robust synthesis from uncertain sampled FRFs and piecewise-convex inter-frequency sets: https://arxiv.org/abs/2610.20248

Section 5.3.2 defines the controller-factor interpolation remainder:

    r_(k|k+1) = sup_omega ||L(phi)(omega)-phi(omega)||_F

and says the scalar can be approximated by sufficiently dense offline sampling (Eq. 60-61). The supremum is the mathematical quantity; this benchmark tests exact certification of that radius rather than replacing it with a sampled maximum.

This is a matched mathematical subproblem, not a reimplementation of the paper's IQC theorem or controller synthesis.

## 2. V0.1 exact 2x2 matrix curve

The complete normalized frequency band is [0,1]. Three measured matrix knots are at 0, 1/2 and 1; all measured knot matrices are zero.

Define:

    P = [0 1]
        [0 0]

    d(omega)=16*omega*(1-omega)*(omega-1/2)^2
    Phi(omega)=d(omega)*P

Since all knots have zero value, the piecewise linear interpolant is identically zero. P has Frobenius norm exactly one, so:

    ||Phi(omega)-L(Phi)(omega)||_F = |d(omega)|

Let t=omega-1/2 and u=t^2. Then u∈[0,1/4] and:

    d(omega)=4u-16u^2

This concave quadratic has its unique maximum at u=1/8:

    sup_[0,1] ||Phi-L(Phi)||_F = 1/4

Thus the valid generator cover is:

    Phi(omega) in L(Phi)(omega) + (1/4)*B_F

for every frequency in the interval.

## 3. Finite-grid counterexample

The three knot samples all give zero, yet the true supremum is 1/4.

Even the five-point grid {0,1/4,1/2,3/4,1} has maximum only 3/16. It underestimates the true radius by 1/16. A finite-grid maximum is a diagnostic lower bound on the supremum, not an upper bound.

## 4. Exact Lipschitz companion

With t=omega-1/2:

    d'(omega)=8t-64t^3
    |d'|^2=64u*(1-8u)^2

For u∈[0,1/4], the normalized squared expression is:

    4u*(1-8u)^2

Its only interior stationary points are u=1/8 and u=1/24; values at those points are 0 and 2/27. At endpoints the values are 0 and 1. Hence:

    sup_[0,1] |d'(omega)| = 4

The global Lipschitz radius from the nearest knot with h=1/2 is L*h/2=1, four times the exact radius 1/4. This gives a direct conservatism comparison between exact extrema and a global slope method.

## 5. Evidence and claim requirements

A production remainder artifact binds:
- adjacent frequency-knot IDs and full interval partition;
- endpoint matrices and interpolation operator;
- model, topology, operating-point, port-order and frequency-unit identities;
- matrix norm and its scaling;
- exact polynomial/rational residual or independently qualified regularity evidence;
- certified supremum radius and proof method;
- coverage of every subinterval;
- hard-set/statistical uncertainty semantics and shared calibration lineage;
- explicit claim ceiling.

The generator radius covers this declared curve only. It is not an entire IQC or robust-stability certificate.

## 6. Benchmark targets (targets, not results)

- 100% exact Fraction agreement with the 1/4 generator radius.
- Zero universal remainder bounds from sampled maxima alone.
- Zero missing frequency intervals.
- Zero radius below the exact supremum.
- Compare analytic extrema, exact interval subdivision, Bernstein bounds and Lipschitz envelopes.
- Report sampled-grid underestimation, certified radius, Lipschitz conservatism, proof time and partition complexity separately.
- Preserve full cross-frequency nuisance dependencies from #11914.

## 7. Claim ceiling

V0.1 certifies one synthetic matrix-curve inter-frequency remainder. It does not implement IQC synthesis, qualify a physical FRF or controller coprime factor, establish aircraft robust stability, or create flight-safety/airworthiness/certification evidence.
