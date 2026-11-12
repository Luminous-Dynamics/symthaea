# AIRROAD-MIMO-UQ-001 — V0.1 hard complex phasor quotient enclosures

**Status:** executable conservative rectangular enclosure operator  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11788  
**MIMO evidence passport:** https://github.com/Luminous-Dynamics/symthaea/pull/11764  
**Generic hard-bound precedent:** https://github.com/Luminous-Dynamics/symthaea/pull/7989

## 1. Ownership and boundary

This is a MIMO-specific measurement operator, not another generic uncertainty system.

- MAT-UQ #6225 owns multi-fidelity uncertainty/discrepancy semantics.
- ENG-MEAS #4919 owns measurement equations and uncertainty propagation.
- MAT-METRO #6303 owns calibration and traceability.
- CAP-UQ-002 #7988 / PR #7989 owns frequency-resolved capacitor impedance/current hard intervals.
- AIRROAD MIMO passport #11764 owns complete port-matrix records and raw experiment identities.

This increment consumes deterministic complex phasor error radii and performs the MIMO-specific (V_i/I_j) transformation. It does not infer a hard bound from an instrument specification, standard uncertainty, confidence interval or covariance matrix.

## 2. Input set

Every voltage response and injected-current phasor is represented by a closed disc:

[
V in D(v_0,r_v),qquad I in D(i_0,r_i).
]

The supplied radii are accepted only under an explicit `DeterministicHardBound` method reference. Statistical-only semantics are rejected. The current disc must exclude zero.

V0.1 assumes voltage and current discs are independent sets. This is a set-theoretic worst-case assumption, not a statistical independence statement. If physical measurement errors are correlated, the independent product set remains an outer set but may be unnecessarily wide. No probability coverage or covariance-based result is claimed.

## 3. Quotient enclosure

The target is:

[
Z = rac{V}{I}.
]

First bound the current magnitude:

[
|i_0|-r_i > 0,
]
[
(|i_0|-r_i)^2 le |I|^2 le (|i_0|+r_i)^2.
]

The input discs are enclosed by axis-aligned rectangles:
[
V_r in [Re(v_0)-r_v,Re(v_0)+r_v],quad
V_i in [Im(v_0)-r_v,Im(v_0)+r_v],
]
and similarly for (I_r,I_i).

Then:
[
N_r=V_rI_r+V_iI_i,qquad
N_i=V_iI_r-V_rI_i,
]
[
Re(Z)in N_r/[D_{min},D_{max}],quad
Im(Z)in N_i/[D_{min},D_{max}],
]
where (D=|I|^2) is strictly positive.

This interval calculation intentionally ignores dependencies among the rectangle coordinates and between numerator/denominator. It yields a conservative rectangular enclosure, not the exact quotient image. A later disk/sector or affine-arithmetic method can be compared on enclosure width, cost and containment.

## 4. Numerical contract

Every arithmetic operation is expanded outwards by one representable binary64 value using bit-level next-up/next-down operations. Non-finite intermediates, overflow, underflow that collapses a positive denominator lower bound to zero, and unsupported exact-knot requests are rejected. No epsilon-based inward shrinkage occurs.

The algorithm relies on ordinary IEEE-754 binary64 round-to-nearest operations and square root. Its numerical envelope is deliberately bounded: it refuses calculations when it cannot retain a finite positive denominator enclosure.

## 5. Output

For every row/column/frequency matrix element, the output carries:
- nominal complex V/I point estimate;
- real impedance interval (ohms);
- imaginary impedance interval (ohms);
- voltage and current input error radii;
- lower/upper current magnitude and squared-magnitude bounds;
- exact observed port and driven port IDs;
- configuration, operating-point and sweep digests;
- deterministic hard-bound method reference;
- explicit `independent_discs_outer_enclosure` assumption;
- algorithm version.

A full-matrix enclosure is emitted only if every required matrix element is present and finite. No missing term becomes zero. Only exact recorded frequency knots are supported.

## 6. Frozen exact-rational reference

The machine-readable fixture `docs/research/fixtures/airroad-mimo-complex-enclosure-v0.1.json` defines one 2×2 matrix. Its current centers are axis-aligned positive real values, so the radial denominator endpoints are rational. The independent Python verifier uses `fractions.Fraction` to recompute the four exact real and imaginary intervals.

The Rust tests also generate deterministic points inside the voltage/current discs and verify every resulting quotient lies in the computed rectangle. Sampled containment supplements, but does not replace, the exact rational oracle.

## 7. Benchmarks and next tier

V0.1 targets:
- 100% exact Fraction agreement/containment for frozen rational matrix elements;
- zero sampled disc ratios outside the computed rectangles;
- zero deterministic enclosures for statistical-only inputs;
- zero finite enclosures if current uncertainty includes zero;
- zero missing matrix elements silently accepted;
- exact row/column/knot attribution and deterministic replay.

Next:
1. Add a qualified joint voltage/current uncertainty-set interface for covariance or common-mode dependencies. Do not convert covariance into hard radii without a chosen coverage/guarantee.
2. Compare rectangle, disk/sector and affine/Taylor enclosures on matched phasor sets.
3. Propagate matrix enclosures through robust singular-value / return-ratio screening, using a separately qualified stability method.
4. Validate input radii against calibrated VNA/DAQ/PHIL data and held-out repeats.

Report enclosure width inflation separately for real and imaginary components. Never optimize width by weakening containment.

## 8. Industry and metrology comparison

NIST's 2025 Metrologia paper develops an uncertainty budget for frequency-dependent four-terminal-pair capacitance measurements with a four-channel VNA. It is a metrology baseline, not a hard-radius generator: [DOI 10.1088/1681-7575/adc39e](https://doi.org/10.1088/1681-7575/adc39e).

A 2024 Sensors paper describes covariance-matrix-based uncertainty evaluation for VNA S-parameter measurements: [DOI 10.3390/s24113668](https://doi.org/10.3390/s24113668). That covariance approach is complementary; V0.1's independent hard discs are a conservative deterministic set model, not a substitute.

The differentiation opportunity is a provenance-complete MIMO matrix whose each complex element has a reproducible enclosure tied to its own excitation and observation records, with explicit numerical and dependence assumptions.

## 9. Claim ceiling

This increment proves only conservative arithmetic enclosure semantics for the admitted independent deterministic phasor discs within the stated binary64 domain. It does not establish that the source radii are physically valid or calibrated, does not establish probabilistic coverage, does not provide a general robust generalized-Nyquist solver, and does not establish measured aircraft impedance, aircraft stability, PHIL/HIL correlation, flight safety, airworthiness or certification.
