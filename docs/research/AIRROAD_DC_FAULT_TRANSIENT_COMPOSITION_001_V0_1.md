# AIRROAD-DC-FAULT-TRANSIENT-COMPOSITION-001 — state-consistent source/capacitor reference

**Status:** V0.1 synthetic exact-rational linear-state reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11825  
**Composition:** #11818 source-fed RL subreference, #11822 capacitor RLC subreference, #11645 protection evidence, #11766 sustained supply, #11801 MIMO stability

## 1. Objective

The separate source-fed and local-capacitor references are useful, but cannot be naively combined as two independently computed current magnitudes. Their equivalent circuits, shared branch, state boundary and event time must match.

This benchmark defines one small common topology and computes:
1. the complete forced-plus-initial-state trajectory;
2. the source-forced zero-initial-state response;
3. the zero-source initial-state response;
4. the KCL-reconstructed branch current;
5. an instantaneous power-balance identity.

It is a linear time-invariant two-state reference before any protection event. It does not simulate a protection device or an aircraft system.

## 2. Common circuit and equations

One constant ideal voltage source feeds a series source resistance and inductor. The bus node has a shunt fault resistance and a DC-link capacitor to return.

State x=[i_L, v]^T with positive inductor current source-to-bus and positive capacitor current into the capacitor:

    L * di_L/dt = V_s - R_s*i_L - v
    C * dv/dt   = i_L - v/R_f
    i_fault     = v/R_f
    i_cap       = C*dv/dt

For V_s=600, R_s=L=R_f=C=1 in SI units:

    dx/dt = A*x + b
    A = [ -1  -1 ]
        [  1  -1 ]
    b = [ 600 ]
        [   0 ]

The equilibrium is x_eq=[300 A, 300 V]. The matrix exponential is:

    exp(A*t) = exp(-t) * [ cos(t)  -sin(t) ]
                         [ sin(t)   cos(t) ]

The exact initial state is x0=[0 A, 600 V]. Evaluation time is T=1/4 s. For that point, all exponential/sine/cosine arguments are within the frozen Taylor enclosure domain.

## 3. Three distinct but exactly composable responses

**Source-forced response**: zero initial state, source enabled:

    x_source(t) = x_eq - exp(A*t)*x_eq

    i_source = 300 * [1 - exp(-t)*(cos(t)-sin(t))]
    v_source = 300 * [1 - exp(-t)*(sin(t)+cos(t))]

**Initial-state response**: zero source, actual initial energy retained:

    x_initial(t) = exp(A*t)*x0

    i_initial = -600 * exp(-t)*sin(t)
    v_initial =  600 * exp(-t)*cos(t)

**Complete trajectory**:

    x_total = x_source + x_initial

    i_total = 300 * [1 - exp(-t)*(cos(t)+sin(t))]
    v_total = 300 * [1 + exp(-t)*(cos(t)-sin(t))]

A direct closed-form evaluation of x_total is computed independently and compared with the sum of the two component responses.

Important: i_initial is negative in the declared inductor-current direction at T. Its sign is physically meaningful. Summing absolute current magnitudes would destroy linear superposition and overstate the source-branch current. The fault-resistor current and capacitor current are distinct branch currents; KCL is:

    i_L = v/R_f + C*dv/dt

At T, the approximate values are:
- source-forced inductor-current contribution: +131.4266 A;
- initial-state contribution to the same inductor-current state: -115.6070 A;
- complete source-branch current: +15.8196 A;
- bus/fault voltage: 468.5734 V;
- capacitor current: -452.7539 A;
- fault-resistor current: +468.5734 A.

The exact rational enclosures in the JSON fixture are authoritative.

## 4. Instantaneous energy/power identity

The state equations imply:

    V_s*i_L = R_s*i_L^2 + v^2/R_f
              + d/dt [ (1/2)L*i_L^2 + (1/2)C*v^2 ]

The independent oracle constructs interval enclosures for i, v, di/dt and dv/dt, then verifies that input power overlaps resistive loss plus stored-energy rate. This is an internal consistency invariant, not a measured energy result.

## 5. Independent Fraction oracle

The verifier reads exact rational component parameters from the fixture, derives A and the equilibrium, computes exp(-T), sin(T) and cos(T) using alternating Taylor series with exact Fraction arithmetic, and checks:
- every component and total-state enclosure against the frozen fixture;
- direct-total vs source-forced-plus-initial-state composition;
- KCL-reconstructed current vs inductor current;
- the instantaneous power identity;
- deterministic exact output.

No float, math.exp, math.sin or math.cos is used in the oracle.

## 6. Negative controls and limits

The verifier rejects inconsistent state matrix or topology declarations, unsupported time-domain inputs and invalid Taylor domains. It must detect a response computed with changed component parameters, incorrect current signs, or different initial conditions.

This fixture tests linear superposition only because all three trajectories use the same A matrix and time-invariant topology. It does not justify summing separate RL/RLC simulations if their shared paths or boundaries differ. It does not apply through current limiting, saturation, diode commutation, contactor opening, arcing, protection transitions or any other mode change without a separately validated hybrid model.

## 7. Benchmark targets

Targets, not results:
- 100% exact-rational agreement with the frozen fixture;
- zero direct-vs-decomposed state mismatches;
- zero KCL residual outside the admitted enclosures;
- zero instantaneous power-balance failures;
- zero acceptance of absolute-magnitude component addition as signed superposition;
- zero composition across mismatched topology/state/timebase identities;
- 100% rejection of state reset without an explicit physical reset map;
- report interval widths and Taylor orders separately from classification;
- later compare with a matched ngspice/PLECS transient using the exact same source, R/L/C, initial state, time origin and tolerance.

## 8. Industry/SOTA positioning

The 2026 hybrid-electric-aircraft transient study is a stronger full-model baseline: it includes source, battery, supercapacitor, generator/fan-motor, DC load and control effects, with semi-physical comparison: https://www.mdpi.com/2226-4310/13/3/261. The DC-breaker review explicitly separates capacitor discharge, converter/cable transients and source-fed fault current: https://www.mdpi.com/2076-3417/12/1/15. NASA NEAT is the longer-term physical test/correlation context for DC and bidirectional supplies, batteries, multiple current paths and fault-management: https://www.nasa.gov/eap-labs-and-testbeds/neat/

AIRROAD should not claim that this two-state reference beats those system models. Its contribution is an executable contract for state consistency, signed decomposition, conservation and evidence identity, so more complex backends can be checked against independently derived subcases.

## 9. Claim ceiling

V0.1 is one fixed linear two-state circuit. It does not model source dynamics, converter control, multiple sources, distributed cable, protection switching, DC arcs, transient recovery voltage, thermal damage, aircraft mission capability, flight safety, airworthiness or certification. No energized fault-injection is authorized.
