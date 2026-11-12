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

## Frequency-domain source/load view

The same linearization can be expressed as an exact source/load impedance interconnection.

Let (h_i=P_i/\bar V_i^2) denote the magnitude of the constant-power load's negative incremental conductance at its equilibrium. With the bus voltage (\hat v_0) treated as the perturbation input to branch (i), the branch equations give:

\[
\frac{\hat v_i}{\hat v_0}=\frac{G_i}{C_i s+G_i-h_i}
\]

and the current drawn from the bus is:

\[
\hat i_i=G_i(\hat v_0-\hat v_i)
\]

so the branch input admittance is:

\[
Y_i(s)=\frac{\hat i_i}{\hat v_0}
=G_i\frac{C_i s-h_i}{C_i s+G_i-h_i}
\]

The source and common bus capacitor have Thevenin impedance:

\[
Z_s(s)=\frac{1}{G_s+C_0s}
\]

With the sign convention that positive branch current is drawn from the bus, the common-mode return ratio is:

\[
L(s)=Z_s(s)\left(Y_1(s)+Y_2(s)\right)
\]

and the closed-loop characteristic condition is (1+L(s)=0).

For the frozen symmetric case, (G_s=G_1=G_2=C_0=C_1=C_2=1) and (h_1=h_2=1/2). Therefore:

\[
Z_s(s)=\frac{1}{s+1},\qquad
Y_1(s)=Y_2(s)=\frac{s-1/2}{s+1/2}
\]

\[
L(s)=\frac{2(s-1/2)}{(s+1)(s+1/2)}
\]

The closed-loop characteristic polynomial is:

\[
(s+1)(s+1/2)+2(s-1/2)
=s^2+\frac72s-\frac12
\]

which has the positive root ((-7+\sqrt{57})/4). This reproduces the common-mode instability obtained from the exact Jacobian.

This is a symmetric SISO common-mode reduction of this particular two-branch network. It does not establish a general scalar impedance criterion for asymmetric branches or a multi-port network. Those cases require a frozen port/sign convention and an independently checked MIMO return-ratio or full state-space formulation.

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

## Industry baseline and where Luminous must differentiate

The implementation target is not a replacement for mature circuit and power-electronics tools.

- **PLECS 5.0** already provides small-signal analysis blocks, AC/linearized-circuit analysis, and frequency-response analysis by applying discrete sinusoidal perturbations around a simulated steady state. Its standalone analysis also exposes loop-gain and output-impedance workflows. See [PLECS Frequency Response](https://docs.plexim.com/plecs/5.0/analysis/frequency-response/), [Small Signal Response](https://docs.plexim.com/plecs/latest/components-by-category/smallsignalresponse/) and [Analysis Tools](https://docs.plexim.com/plecs/latest/analysis/).
- **NASA NEAT** supports megawatt-scale electrical powertrain tests under simulated altitude, with AC/DC/bidirectional DC power, batteries, load banks, ground-loop paths, cooling and electric-machine testing. See [NASA NEAT](https://www.nasa.gov/eap-labs-and-testbeds/neat/).
- **NASA AREAL** is a high-power DC testbed aimed at electric-powertrain architectures and standards/verification for vertical-lift vehicles and electrified transport. See [NASA EAP Labs and Testbeds](https://www.nasa.gov/eap-labs-and-testbeds/).

Accordingly, a synthetic Rust model or a successful PLECS/ngspice comparison is not evidence that Luminous has matched these facilities. The realistic sequence is analytical fixtures, independent software-tool comparison, guarded low-energy laboratory work, PHIL with a qualified partner, then higher-power test infrastructure where authorized.

The potential product differentiation is the **evidence and decision layer around the solver**:

1. Exact as-built aircraft power-tree, port directions, motor-to-battery mapping and configuration revision.
2. Calibrated complex impedance records with frequency, temperature, bias, operating-point and uncertainty validity.
3. Explicit distinction between measured, fitted, simulated, extrapolated and unknown values.
4. Hard-bound robust stability dispositions that abstain when a required sweep or discrepancy bound is missing.
5. Cross-tool replay with solver version, netlist/model digest, exact operating point, excitation, frequency set and raw output digest.
6. Composition into mission-phase constraints, topology-aware control allocation (#11737), capacitor mass/lifetime (#11738), thermal feedback and fault evidence without giving the analysis system actuator or certification authority.
7. Decision-focused experiment selection via #7803: recommend the measurement most likely to change a real architecture choice, not merely the one expected to reduce entropy.

### Matched comparison protocol

On matched topology, operating point, model order, parameter set, port convention, frequency band and excitation, compare:
- PLECS small-signal / frequency-response results where a reproducible model is available;
- ngspice AC/transient outputs after #5633 validates the solver result parser and convergence semantics;
- independent state-space eigenvalues and generalized-Nyquist results;
- measured impedance sweeps / PHIL responses only when calibration and exact test topology are available.

Report per-frequency complex (Z) residual, magnitude and phase error, resonance frequency/Q error, stability-margin error, runtime, model-order, data requirement and unsupported-domain coverage separately. Never compare a Luminous reduced-order model to an unrelated high-fidelity model or different operating point and call the difference a win.

Proposed internal targets remain:
- 100% correct classification on the frozen analytical SISO/MIMO corpus;
- zero false-stable and false-robust negative controls;
- zero accepted extrapolated impedance points;
- exact agreement with independent state-space / Nyquist references on matched supported models;
- explicit abstention and evidence-coverage reporting;
- physical error targets fixed only after a calibrated measurement / PHIL holdout exists.

## Claim ceiling

This fixture qualifies only the stated synthetic three-state RC / constant-power-load model. It does not model line inductance, converter control loops, switching, battery electrochemistry, protection, faults, thermal response, flight control, an aircraft electrical network, PHIL/HIL performance, airworthiness or certification.
