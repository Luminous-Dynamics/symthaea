# AIRROAD-DC-DYNAMIC-STABILITY-001 — V0.1 analytical contract

**Status:** research contract / known-answer reference vectors  
**Owner:** Symthaea AIRROAD programme  
**Tracking PR:** https://github.com/Luminous-Dynamics/symthaea/pull/11712  
**Claim ceiling:** synthetic reduced-order mathematics only; no physical aircraft, converter, battery, protection or flight validation.

## 1. Purpose and boundary

V0.1 establishes a deliberately small analytical reference for the voltage-stability mechanism that is invisible to an ideal static shared-bus calculation: a source with finite output resistance feeding a constant-power load through a DC-link capacitor.

It is not a battery model, switching-converter model, protection model, DC-network solver, flight-control component or certification method. Its value is that equilibrium existence and local stability have closed-form answers, so an implementation can be tested against an independent oracle before adding model order.

Existing owners remain authoritative for their domains:

- #11640 / #11642: simultaneous ideal-common-voltage aircraft bus and branch aggregation.
- #11663 / #7805: mission-conditioned source capability and reserve.
- #11645 / #4881: aircraft protection evidence and as-built power topology.
- #6179: EMC / EMI / signal-integrity / power-integrity coupling.
- #5633: actual ngspice execution-output and convergence parsing.
- #11411 / #11428 / #11467: mission electro-thermal solver, TMS architecture and thermal operating policy.

This contract introduces none of their types and does not promote any draft implementation.

## 2. Reduced-order model

For a positive Thevenin source voltage \(V_s\), source resistance \(R_s\), DC-link capacitance \(C\), and constant-power load \(P\), define:

\[
C\frac{dV}{dt}=\frac{V_s-V}{R_s}-\frac{P}{V}
\]

with:

- \(V_s > 0\), in volts;
- \(R_s > 0\), in ohms;
- \(C > 0\), in farads;
- \(P > 0\), in watts;
- \(V > 0\), in volts, over the declared model domain.

Both right-hand terms are currents. The left-hand side is also current. The load current is \(I_L=P/V\), so it rises as bus voltage falls. This is the constant-power-load negative incremental impedance mechanism.

The model assumes:
- one ideal fixed source voltage behind a scalar series resistance;
- one lumped bus capacitance;
- one ideal constant-power load over its declared voltage domain;
- no cable inductance, distributed capacitance, converter control-loop state, switching ripple, battery electrochemistry, protection dynamics, regenerative load or thermal feedback;
- no stochastic or probabilistic interpretation of the input parameters.

The source/load values are model parameters, not measured aircraft properties. A parameter interval is a set of admitted values, not a probability distribution.

## 3. Closed-form equilibrium oracle

At equilibrium, \(dV/dt=0\):

\[
\frac{V_s-V_*}{R_s}=\frac{P}{V_*}
\]

and therefore:

\[
V_*^2 - V_s V_* + R_s P=0
\]

Define:

\[
\Delta=V_s^2-4R_sP
\]

For the positive parameter domain:

- \(\Delta > 0\): two positive equilibrium voltages exist;
- \(\Delta = 0\): one repeated equilibrium exists at the fold boundary;
- \(\Delta < 0\): no real equilibrium exists; the model must not return a steady-state-feasible disposition.

For \(\Delta \geq 0\):

\[
V_{\pm}=\frac{V_s\pm\sqrt{\Delta}}{2}
\]

The local linearization of the scalar ODE at an equilibrium is:

\[
\lambda_* =
\frac{1}{C}\left(-\frac{1}{R_s}+\frac{P}{V_*^2}\right)
\]

For \(\Delta>0\), the higher-voltage root \(V_+\) has \(\lambda_+<0\) and is locally asymptotically stable in this one-state model. The lower-voltage root \(V_-\) has \(\lambda_->0\) and is unstable. At \(\Delta=0\), \(\lambda=0\): the repeated equilibrium is non-hyperbolic and must be classified as **critical / marginal**, not as a robust stable operating point.

This is a local mathematical classification. It says nothing about the stability of a real converter, aircraft network, control loop or complete aircraft.

### Phase-line and initial-voltage qualification

The scalar vector field can be factored using the two roots:

```
dV/dt = -((V - V_-)(V - V_+)) / (R_s C V)
```

For two distinct positive roots:
- (0 < V_0 < V_-): the voltage decreases toward the singular/domain boundary; the high-voltage equilibrium is not in this initial condition's basin.
- (V_0 = V_-): the trajectory remains at the unstable equilibrium in the exact mathematical model.
- (V_- < V_0 < V_+): the voltage increases toward (V_+).
- (V_0 = V_+): the trajectory remains at the stable equilibrium.
- (V_0 > V_+): the voltage decreases toward (V_+).

Thus the basin of attraction of (V_+) in this ideal scalar model is (V_0 > V_-). This basin result is a mathematical property of this exact one-state model, not an aircraft brownout-recovery guarantee.

At the critical fold, (V_-=V_+=V_s/2). The vector field is negative on either side: it approaches the fold from above, but moves away from the fold below. This makes the repeated root semistable, not robustly stable.

Define a dimensionless static distance-to-fold diagnostic:

```
mu = Delta / V_s^2 = 1 - P / P_critical
P_critical = V_s^2 / (4 R_s)
```

For this model, `mu > 0` means two equilibria, `mu = 0` is the fold, and `mu < 0` means no equilibrium. This diagnostic is not a dynamic damping margin, certification margin or probability of failure. It must not replace the actual uncertainty-bounded worst-case check.

## 4. Frozen known-answer corpus

All cases below are synthetic and use SI units. Decimal inputs are exact rational values in the independent reference. A future binary64 implementation must be checked against those exact values, not against values generated by the implementation under test.

The canonical machine-readable known-answer corpus is frozen at [`docs/research/fixtures/airroad-dc-dynamic-stability-v0.1.json`](fixtures/airroad-dc-dynamic-stability-v0.1.json). It records decimal strings, exact rational inputs, expected roots, currents, eigenvalues, classifications and fold margins. The independent standard-library verifier is `tools/airroad-dc-stability-qualifier/verify_fixture.py`; run it with `python3 tools/airroad-dc-stability-qualifier/verify_fixture.py`. It uses exact `Fraction` arithmetic and imports no production model.

Common parameters:

| Input | Value |
|---|---:|
| \(V_s\) | 100 V |
| \(R_s\) | 1 Ω |
| \(C\) | 0.01 F |
| Static collapse threshold \(V_s^2/(4R_s)\) | 2,500 W |

| Case | \(P\) | \(\Delta\) | Equilibrium roots | Local classification |
|---|---:|---:|---:|---|
| stable-and-unstable-roots | 1,875 W | 2,500 V² | 75 V, 25 V | 75 V stable; 25 V unstable |
| near-fold-two-roots | 2,499 W | 4 V² | 51 V, 49 V | 51 V stable; 49 V unstable |
| critical-fold | 2,500 W | 0 V² | 50 V (double) | Critical / marginal; eigenvalue zero |
| no-equilibrium | 2,501 W | −4 V² | none | No steady-state equilibrium |

### Scaled 400 V / 1.5 MW fixture

The fifth fixture uses case-specific inputs rather than the common 100 V parameters:

| Input | Value |
|---|---:|
| \(V_s\) | 400 V |
| \(R_s\) | 0.02 Ω |
| \(C\) | 0.005 F |
| \(P\) | 1.5 MW |
| \(P_{critical}\) | 2.0 MW |
| \(\Delta\) | 40,000 V² |
| Equilibria | 300 V (stable), 100 V (unstable) |
| Currents | 5 kA, 15 kA |
| Eigenvalues | \(-20,000/3\;s^{-1}\), \(+20,000\;s^{-1}\) |

This is a scale-diverse analytical fixture, not an aircraft bus specification.

Additional exact known answers:

### Case stable-and-unstable-roots

At 75 V:
- source current = load current = 25 A;
- \(\lambda_+=-200/3\;s^{-1}\).

At 25 V:
- source current = load current = 75 A;
- \(\lambda_-=+200\;s^{-1}\).

### Case near-fold-two-roots

At 51 V:
- source current = load current = 49 A;
- \(\lambda_+=-200/51\;s^{-1}\).

At 49 V:
- source current = load current = 51 A;
- \(\lambda_-=+200/49\;s^{-1}\).

### Case critical-fold

At 50 V:
- source current = load current = 50 A;
- \(\lambda=0\;s^{-1}\).

The repeated root is a fold boundary, not a margin-bearing stable design. In this model, the vector field is non-positive on both sides: trajectories approach the fold from above but move away from it below.

### Case no-equilibrium

The discriminant is negative. Do not clamp the requested load, reduce it silently, substitute a nominal voltage or return a large finite current as a feasible operating point.

## 5. Exact reference versus binary64 disposition

The mathematical oracle and the executable binary64 API deliberately have different responsibilities:

- The exact-rational fixture classifies the fold case as a repeated, non-hyperbolic, semistable equilibrium.
- The current Rust API returns NumericallyIndeterminate for the exact-fold and near-fold cases whenever the computed discriminant lies inside its arithmetic ambiguity band. It does not return a positive disposition or invent a basin from an unresolved root.
- Clearly separated two-root and no-equilibrium cases can be classified by the current API, subject to the explicitly documented floating-point support profile.
- Initial-voltage classifications close to either computed equilibrium are returned as near-boundary/indeterminate basin categories rather than being silently assigned to one side.

The current 16-ULP-scaled ambiguity band is a conservative heuristic, **not a proved floating-point enclosure**. The exact-rational helper validates the frozen mathematical vectors; it does not prove that every binary64 operation is outward-rounded. The formal arithmetic qualification remains with #11703 / #11706. Until that work is independently qualified, this API is an analytical research reference only.

## 6. Domain and failure semantics

The first implementation must fail closed for:

- zero, negative, NaN or infinite \(V_s\), \(R_s\), \(C\) or \(P\);
- malformed units or dimensionally inconsistent input;
- arithmetic overflow or non-finite intermediate values;
- unsupported parameter magnitudes outside the frozen reference profile;
- any future transient trajectory reaching or crossing a declared \(V_{min}>0\) domain boundary;
- a negative discriminant when the caller requests a feasible steady-state equilibrium;
- an attempted classification of the repeated-root case as robustly stable.

The ideal constant-power expression \(P/V\) is singular at zero voltage. Any future transient API must preserve this domain boundary and report a typed non-PASS / domain-exit result. It must not clamp \(V\) to an arbitrary epsilon and continue while retaining a feasibility claim.

If a later converter model current-limits or folds back at undervoltage, that is a different, explicitly versioned load law with its own state transitions and independent reference cases. It must not be silently substituted into this V0.1 model.

## 7. Tier-0 acceptance

- Exact-rational oracle agreement with all five frozen equilibrium cases and their phase-line sign probes; the binary64 API must classify clearly separated cases and abstain (non-PASS) on the fold/near-fold ambiguity band.
- Correct root ordering and source/load current closure.
- Correct local stability sign for each distinct root.
- Critical-fold classification is marginal / non-hyperbolic, never robust PASS.
- No-equilibrium classification is non-PASS.
- Zero accepted invalid, non-finite or unsupported parameter paths.
- Deterministic canonical input/output encoding and replay.
- Independent rational oracle does not import or reuse the production Rust equations or intermediate values.
- Every frozen phase-line probe is re-evaluated from the exact ODE and its sign must match the declared increasing/decreasing/equilibrium result.
- Floating-point results report absolute error and ULP distance separately from classification correctness.
- Tests include sign mutations, unit mutations, exact-fold and adjacent representable values, overflow and domain-exit controls.

No numerical tolerance may be widened after observing a failing implementation. Freeze the corpus, expected exact values and comparison method before qualification.

## 8. Future transient extension — not implemented by this contract

A later version may integrate the scalar ODE for declared source/load steps, but only after the equilibrium oracle passes. Such a transient implementation must preregister its integration method, event handling, error estimator, tolerance and step-size sweep.

Report independently:
- bus voltage nadir and peak;
- time below declared operating-voltage bounds;
- source and load current peak;
- recovery / settling time under a declared criterion;
- energy residual and integration error;
- step-size / tolerance convergence;
- numerical failure versus physical-model non-PASS.

The next model-order increase should be justified by a specific observed discrepancy, such as line inductance or converter-control dynamics. It should not add PWM detail by default.

## 9. External comparison protocol

When available and properly qualified, compare the same equations, parameter values, initial condition and load event against:
1. the exact equilibrium oracle;
2. the Rust reduced-order implementation;
3. ngspice after #5633 proves actual solver convergence and parses the requested trace;
4. a second independently configured circuit / power-electronics tool.

Tool agreement is not physical correlation when the tools share model equations or parameter sources. A physical campaign requires calibrated synchronized voltage/current traces, exact as-tested topology, source/load identity and controlled test authority.

## 10. Research and industry comparators

### Published 2026 eVTOL propulsion-microgrid result

Zhao et al., *Small-Signal Stability Analysis and Control Design for eVTOL PDCM*, Aerospace Science and Technology, article 114083, online 16 October 2026, DOI [10.1016/j.ast.2026.114083](https://doi.org/10.1016/j.ast.2026.114083), reports a configuration-specific hardware-in-the-loop study. The article reports:
- a 75.4% increase in its calculated stability margin when the studied DC-bus voltage is raised from 800 V to 1,200 V;
- a 53.3% reduction in its reported bus-voltage oscillation value for VDCM droop relative to conventional droop;
- relative steady-state bus-voltage error reduced from 7.42% to 0.83% in its studied single-point-failure case when VDCM droop was combined with keeping the opposite rotor pair operating.

These are study-specific results for the authors' model, controller and HIL setup. They are not universal eVTOL design constants or Luminous performance results.

### Hardware-in-the-loop / PHIL reference points

Zhao et al. report eVTOL-specific HIL evidence in the 16 October 2026 paper above. A separate March 2026 preprint by Syed et al., [*Distributed Adaptive Control for DC Power Distribution in Hybrid-Electric Aircraft: Design and Experimental Validation*](https://arxiv.org/abs/2603.17518), describes distributed adaptive/passivity-based DC-bus voltage regulation and proportional source load-sharing under uncertain line resistance and inductance, evaluated against other controllers in a power-hardware-in-the-loop environment. It is a hybrid-electric aircraft comparator and a preprint, not an eVTOL certification result.

NASA's [Electric Aircraft Testbed (NEAT)](https://www.nasa.gov/eap-labs-and-testbeds/neat/) demonstrates the physical-infrastructure level required for meaningful comparison: reconfigurable full-scale electrical powertrain testing, AC/DC/bidirectional DC power, batteries, load banks, ground-loop paths, altitude simulation and cooling support. Luminous should not claim to replace such facilities. The nearer-term path is a guarded low-energy rig and matched PHIL evidence, followed by collaboration with a qualified laboratory for higher-power tests.

### Additional electrical-system comparator

*A General Adaptive Stabilization of a Brushless Generator-Based DC Electrical Power System*, IEEE Transactions on Industry Applications 62(3), May–June 2026, DOI [10.1109/TIA.2026.3661372](https://doi.org/10.1109/TIA.2026.3661372), reports an impedance model and experimental validation of an adaptive stabilization method for a brushless-generator DC system with constant-power loads. This is a useful source/load impedance and experimental-method comparator, not a directly matched eVTOL airworthiness result.

### Matched benchmarking protocol

The Luminous comparison must use identical plant parameters, initial states, load events, model fidelity, uncertainty assumptions and compute budgets. Compare at least:
1. passive source/capacitance sizing baseline;
2. conventional droop baseline;
3. virtual-DC-machine or virtual-inertia/damping droop where the plant topology permits;
4. an impedance/adaptive stabilizer baseline with its model and controller assumptions stated;
5. time-domain circuit simulation with independently checked solver convergence;
6. the Luminous evidence-linked analysis and decision pipeline.

Report separately:
- small-signal stability margin and eigenvalue/damping estimates;
- frequency-domain source/load impedance interaction and any declared impedance-ratio criterion;
- large-signal domain-of-attraction / basin estimate where the model supports it;
- voltage nadir, overshoot, settling and recovery after matched events;
- source, branch and converter peak current/power;
- fault-case bus continuity and post-event mission capability;
- added mass/volume/energy and thermal cost of capacitors, filters or stabilizers;
- robustness to source impedance, line parameters, battery SOC/SOH/temperature and model mismatch;
- false-stable / false-feasible decisions and abstention rate;
- simulation runtime, model order, physical data requirement and evidence coverage.

Do not use the 75.4% or 53.3% published figures as direct Luminous acceptance targets. First reproduce the paper's conditions or define a matched open benchmark. A fair comparison reports differences in architecture and evidence, not just headline percentages.

### Where Luminous can differentiate

The strongest opportunity is not to claim a better droop controller from a one-state model. It is to connect:
- as-built power-tree and source/feeder impedance;
- synchronized propulsor and auxiliary load events;
- state-conditioned battery/source capability;
- the exact model and controller revision;
- uncertainty-qualified stability and basin margins;
- failure-specific branch/rotor consequences;
- mission reserve, thermal reset and next-best experiment.

The offline architecture problem can then jointly screen bus voltage, source partitioning, DC-link capacitance, feeder impedance, converter-control family, propulsor load allocation and thermal policy. Hard safety constraints remain constraints, not soft objective penalties. A candidate that cannot support a robust stability claim must remain unresolved rather than being awarded a high average score.

### Regulatory and certification context

EASA's official SC-VTOL page records the final Fifth Publication of MOC-5 SC-VTOL Issue 2 on 5 November 2026, with Electrical Systems identified as the primary technical domain: [EASA Special Condition for VTOL and Means of Compliance](https://www.easa.europa.eu/en/document-library/product-certification-consultations/special-condition-vtol). The actual aircraft category, certification basis and authority determine applicability.

14 CFR §25.1310 is a Part 25 airplane requirement addressing essential-load supply in probable combinations and specified failures; it is not a universal powered-lift/eVTOL threshold. See [14 CFR §25.1310](https://www.law.cornell.edu/cfr/text/14/25.1310). No regulatory voltage margin or certification compliance is inferred by this research contract.

## 11. Claim ceiling

A passing Tier-0 suite establishes only the mathematical and executable correctness of this scalar Thevenin / constant-power-load contract over the frozen synthetic corpus. It does not establish a real battery's impedance, a motor inverter's input behavior, aircraft DC-network dynamic stability, fault interruption, continued safe flight, airworthiness or certification.
