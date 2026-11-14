# AIRROAD-DC-CAP-DISCHARGE-001 — exact DC-link capacitor RLC discharge reference

**Status:** V0.1 synthetic exact-rational analytical reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11822  
**Composes with:** #11645 protection evidence, #11766 post-fault supply, #11818 source-fed RL transient and #11801 robust MIMO stability

## 1. Objective and separation

Create an analytical reference for the local DC-link capacitor discharge transient. A DC fault can draw local stored energy before upstream source controls or protection devices respond. This contribution is not represented by the constant-source RL model in #11818.

The 2026 hybrid-electric-aircraft short-circuit paper models converter/load control effects and compares against a semi-physical test platform: https://www.mdpi.com/2226-4310/13/3/261. The DC breaker review distinguishes capacitor-discharge transients from converter/cable and source-fed response: https://www.mdpi.com/2076-3417/12/1/15.

## 2. V0.1 isolated underdamped series-RLC model

Assumptions: initial capacitor voltage V0 > 0; zero initial series current; constant R, L and C; no continuing source contribution; no converter control/current limit, distributed line, arc, switching or breaker recovery model.

    L di/dt + R i = vC
    dvC/dt = -i/C
    i(0) = 0
    vC(0) = V0

For the underdamped regime:

    alpha = R/(2L)
    omega0 = 1/sqrt(L*C)
    omegaD = sqrt(omega0^2 - alpha^2)
    K = V0/(L*omegaD)

    i(t) = K * exp(-alpha*t) * sin(omegaD*t)
    vC(t) = V0 * exp(-alpha*t) *
            [cos(omegaD*t) + (alpha/omegaD)*sin(omegaD*t)]

The current-squared integral is:

    I2t(T) = K^2/2 * [A(T) - B(T)]
    A(T) = (1 - exp(-2*alpha*T))/(2*alpha)
    B(T) = [2*alpha + exp(-2*alpha*T) *
            (-2*alpha*cos(2*omegaD*T)
             + 2*omegaD*sin(2*omegaD*T))]
           / [(2*alpha)^2 + (2*omegaD)^2]

Energy accounting:

    E_R(T) = R * I2t(T)
    E_L(T) = (1/2) * L * i(T)^2
    E_C(T) = (1/2) * C * vC(T)^2

    (1/2) * C * V0^2 = E_R(T) + E_L(T) + E_C(T)

## 3. Frozen exact-rational fixture

    V0 = 600 V
    R  = 3/50 ohm
    L  = 1/1,000,000 H
    C  = 1/2500 F
    alpha = 30,000 s^-1
    omega0 = 50,000 rad/s
    omegaD = 40,000 rad/s
    T = 1/160,000 s
    alpha*T = 3/16
    omegaD*T = 1/4
    E_initial = 72 J

The fixture deliberately gives rational alpha, omega0 and omegaD. The arguments for exp, sin and cos are within [0,1]. The independent oracle uses exact Fraction arithmetic and alternating Taylor series:
- exp partial indices 13/12;
- sine and cosine partial indices 7/6.

No floating libm function or binary-float endpoint is used in the oracle. Exact rational enclosures are frozen in the JSON fixture.

Expected approximate display values:
- i(T): 3076.576292626752 A;
- vC(T): 574.251255117690 V;
- I2t: 21.907305949584 to 21.907305949585 A²s;
- resistor energy: 1.314438356975 J;
- inductor energy: 4.732660842176 J;
- remaining capacitor energy: 65.952900800848 J;
- initial capacitor energy: exactly 72 J.

## 4. Energy invariant and domain

The verifier must check that the source-free energy at t=0 is enclosed by the sum of resistor loss, magnetic energy and residual capacitor energy at T. Failure of the energy intervals to overlap yields NOT_ESTABLISHED.

V0.1 requires positive finite R,L,C,V0,T; strictly underdamped parameters; alpha*T <= 1/2; omegaD*T <= 1/4; and zero initial current. Critically damped, overdamped, multi-mode and parameter-varying cases are not approximated.

This is ANALYTICAL_REFERENCE_ONLY. No device trip time, interrupting capacity, arc energy, thermal withstand or aircraft protection PASS is produced.

## 5. Evidence composition

The local capacitor-discharge model is not to be silently combined with source-fed RL current from #11818. A future joint model must share one topology, event timebase, fault location, component state and configuration identity, and prove no double counting of common paths or stored energy. #11645 owns device-specific clearing evidence and protection topology. #11766 owns the downstream sustained supply envelope.

## 6. Negative controls and benchmark targets

Negative controls reject non-positive R/L/C/V0/T, non-underdamped parameter sets, nonzero initial current, unsupported Taylor-domain arguments, invalid I2t intervals, failed energy balance, source/arc/breaker claims from the isolated model, and protection joins missing topology identity.

Targets, not achievements:
- 100% exact Fraction agreement on frozen fixtures;
- 100% containment of independent high-precision numerical trajectories;
- zero false energy-balance acceptances;
- zero protection qualification without external clearing evidence;
- report interval width vs series order and runtime separately;
- compare against ngspice/PLECS only with matched initial state, topology, source boundary, integration tolerance and observation time;
- add critically damped, overdamped, converter-controlled and multi-source stages only as separate model families.

## 7. Industry/SOTA and claim ceiling

The 2026 HEA transient work is a stronger system-level baseline because it includes control behavior and semi-physical comparison. AIRROAD's role is to produce exact analytical sub-oracles before composing a joint multi-source/control model. NASA NEAT offers later physical-correlation capabilities including DC and bidirectional-DC supplies, batteries, load banks, multiple paths and fault-management infrastructure: https://www.nasa.gov/eap-labs-and-testbeds/neat/

V0.1 is an isolated, source-free, constant-parameter, underdamped series-RLC analytical reference. It does not qualify actual capacitor banks, converters, cable networks, multiple sources, DC arcs, breakers, device protection, thermal damage, aircraft stability, flight safety, airworthiness or certification. No energized hardware fault experiment is authorized.
