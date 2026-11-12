# AIRROAD-PDCM-IMPEDANCE-001 — V0.1 shared-source interaction fixture

**Status:** synthetic analytical reference / exact-rational negative control  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11745  
**Stacked on:** https://github.com/Luminous-Dynamics/symthaea/pull/11712

## Why this fixture matters

It demonstrates a failure mode that per-converter checks cannot see:

> Two converter input nodes can each be locally stable when supplied from a stiff bus, while their coupled network is small-signal unstable when they share a finite-impedance source.

The fixture intentionally uses only resistive feeders, lumped capacitors and ideal constant-power loads. It does not pretend to model internal converter control loops or real aircraft hardware.

## Model

Let (V_0) be the common bus voltage and (V_1,V_2) the two converter input-node voltages:

[
C_0dot V_0 = G_s(V_s-V_0)-G_1(V_0-V_1)-G_2(V_0-V_2)
]

[
C_idot V_i = G_i(V_0-V_i)-rac{P_i}{V_i}, quad iin{1,2}
]

All values are exact positive rationals.

| Parameter | Value |
|---|---:|
| Source voltage (V_s) | 2.5 V |
| Source resistance (R_s) | 1 Ω |
| Common bus capacitance (C_0) | 1 F |
| Feeder resistances (R_1,R_2) | 1 Ω each |
| Converter input capacitances (C_1,C_2) | 1 F each |
| Constant-power loads (P_1,P_2) | 0.5 W each |

The exact equilibrium is (V_0=1.5) V and (V_1=V_2=1) V. Branch currents are 0.5 A each; source current is 1 A. Power closes exactly: 2.5 W source input = 1 W load + 1 W source-resistor loss + 0.25 W + 0.25 W feeder losses.

## Per-converter versus coupled result

With (V_0) held by an ideal stiff source, each branch's local perturbation eigenvalue is:

[
lambda_{mathrm{isolated}}=rac{-G_i+P_i/V_i^2}{C_i}=-rac12 mathrm{s}^{-1}
]

Both isolated branches are locally stable.

When their common source and bus capacitor are included, the exact Jacobian is:

[
J=
egin{bmatrix}
-3 & 1 & 1\
1 & -1/2 & 0\
1 & 0 & -1/2
end{bmatrix}
]

By symmetry, the differential branch mode has eigenvalue (-1/2). The common-mode Jacobian is:

[
J_c=
egin{bmatrix}
-3 & 2\
1 & -1/2
end{bmatrix}
]

Its characteristic polynomial is:

[
lambda^2+rac72lambda-rac12=0
]

so:

[
lambda_{pm}=rac{-7pmsqrt{57}}4
]

Since (sqrt{57}>7), the larger root is positive. The complete coupled system is therefore small-signal unstable, despite each converter branch being locally stable against a stiff bus.

The full characteristic polynomial is:

[
(lambda+1/2)(lambda^2+7lambda/2-1/2)
=lambda^3+4lambda^2+rac54lambda-rac14
]

This is an exact algebraic negative control, not a claim about a physical aircraft converter.

## Frozen files and qualification

- Machine-readable known-answer values: `docs/research/fixtures/airroad-pdcm-two-converter-shared-source-v0.1.json`.
- Independent standard-library rational verifier: `tools/airroad-pdcm-impedance-qualifier/verify_two_converter_fixture.py`.
- No Symthaea runtime model is added by this fixture.
- The verifier recomputes equilibrium KCL, source/feeder/load power closure, isolated branch eigenvalues, the coupled Jacobian, its characteristic polynomial, and the unstable common-mode pole from exact rational arithmetic.

## Intended next model

The next extension is not to add PWM detail. It is to replace the static branch conductance approximation with independently measured or analytically qualified frequency-dependent source, feeder and converter port impedances. SISO and MIMO return-ratio conventions must be frozen before Nyquist classification. A two-port generalized-Nyquist or state-space eigenvalue reference must distinguish component-local stability from network-interaction stability.

## Claim ceiling

This fixture qualifies only the stated synthetic three-state RC / constant-power-load model. It does not model line inductance, converter control loops, switching, battery electrochemistry, protection, faults, thermal response, flight control, an aircraft electrical network, PHIL/HIL performance, airworthiness or certification.
