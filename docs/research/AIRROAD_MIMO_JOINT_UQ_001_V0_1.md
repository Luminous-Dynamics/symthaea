# AIRROAD-MIMO-JOINT-UQ-001 — shared calibration uncertainty in MIMO impedance

**Status:** V0.1 exact-rational deterministic hard-set reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11858  
**Composes with:** #11764 MIMO impedance passport, #11788 independent complex enclosures, #11801 robust MIMO stability

## 1. Paradigm shift

Measurement uncertainty is not always independent per value. If the same calibration chain adds a common complex gain/phase factor to every voltage and current channel, that nuisance variable is correlated by construction.

The right mathematical object is a joint set:

    measured record = observation_model(physical signal, shared nuisance vector)

The existing independent-disc/rectangle enclosure remains a conservative outer-bound approach. It can be unnecessarily wide when the same nuisance parameter appears in both numerator and denominator. The new reference demonstrates that dependence can matter even before robust stability is computed.

A shared nuisance ID must be supported by explicit calibration/provenance coverage. It cannot be inferred from similar numeric error bounds.

## 2. Exact 2x2 reference matrix

The unperturbed matrix is:

    Z0 = [ 2+j      1+2j   ]
         [ 1/2-j    3-j    ] ohm

The current injections are I1=1+j A and I2=2 A. Base voltage observations satisfy V_ij=Z_ij*I_j:

    V11 = 1+3j V
    V21 = 3/2 - j/2 V
    V12 = 2+4j V
    V22 = 6-2j V

The exact matrix and all current/voltage records are frozen in the JSON fixture.

## 3. One shared hard calibration nuisance

One deterministic common multiplicative complex factor applies to all current and voltage records:

    q = (1+u)+j*v
    u,v in [-1/10, 1/10]

Because Re(q) >= 9/10, q cannot be zero.

The measured records are:

    I_j' = q*I_j
    V_ij' = q*V_ij

Every column and row carries the same nuisance-variable ID and the same calibration-chain identity. Therefore:

    Z_ij' = V_ij'/I_j'
          = q*V_ij/(q*I_j)
          = Z_ij

for all admissible (u,v). The exact joint impedance set is the reference matrix Z0, not an expanded independent box.

## 4. Quantified conservatism of an independent rectangle

For element Z11, exact affine propagation of the shared q yields:
- V11 real in [3/5, 7/5];
- V11 imaginary in [13/5, 17/5];
- I1 real in [4/5, 6/5];
- I1 imaginary in [4/5, 6/5].

If voltage and current rectangles are then treated independently, ordinary interval quotient arithmetic gives:

    Re(Z11) in [8/9, 9/2]
    Im(Z11) in [5/36, 45/16]

This is a valid conservative outer box, but the true correlated set is exactly Z11=2+j. The real and imaginary widths are therefore artifacts of discarding the shared calibration dependency.

This does not invalidate #11788. Independent hard-disc enclosures are still useful and conservative when shared dependence is not known or cannot be qualified. The joint method is admissible only when the same nuisance variable and complete record coverage are proven.

## 5. Evidence semantics

V0.1 admits only deterministic hard-set semantics with:
- a stable nuisance-variable ID;
- exact rational parameter domains;
- a method reference that qualifies the set as deterministic;
- complete voltage/current record coverage;
- exact port/column association;
- proof that the shared denominator factor excludes zero.

A covariance matrix, standard uncertainty, confidence ellipse or 95% interval is not automatically a deterministic hard set. Statistical-only semantics must be rejected by this API until a separate probabilistic coverage contract is declared.

## 6. Negative controls and benchmarks

The oracle checks exact matrix multiplication, shared nuisance coverage, nonzero q, exact cancellation, and the wider independent rectangle. It rejects mismatched nuisance IDs, missing channel coverage, q domains containing zero, statistical-only inputs relabelled as hard bounds, duplicate port IDs and altered row/column pairing.

Targets, not achieved results:
- 100% exact Fraction agreement with all four reference matrix elements;
- zero shared cancellation if any numerator/denominator record lacks the common nuisance ID;
- zero hard-set claims from statistical-only inputs;
- exact agreement between independent hard-set oracle and fixture;
- report independent-box width inflation separately from joint-set width;
- preserve containment when extending to affine/zono/ellipsoid representations;
- compare independent discs, rectangles, shared affine variables, zonotopes and complex-disc intersections on identical uncertainty sets.

## 7. Next extensions

1. Shared per-channel gain/phase, cross-talk and delay variables.
2. Shared synchronization/time-skew uncertainty across channels.
3. One common physical parameter across all frequency knots.
4. Deterministic affine/zonotope sets derived from calibration certificates.
5. Statistical covariance with explicit confidence coverage, kept separate from hard guarantees.
6. Bound rational state-space fitting residuals over frequency before using a measured impedance matrix as a continuous-frequency stability model.

## 8. Industry/SOTA positioning and claim ceiling

Metrology systems often preserve shared calibration and covariance dependencies in uncertainty budgets. AIRROAD's contribution is to carry such dependencies into the MIMO matrix passport and robust-decision pipeline instead of flattening every quantity into independent error bars.

V0.1 is a synthetic common-complex-calibration reference. It establishes no calibrated physical measurement, metrology capability, probabilistic coverage, robust MIMO stability, hardware performance or aircraft qualification.
