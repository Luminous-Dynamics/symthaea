# AIRROAD-DC-RL-TRANSIENT-001 — exact RL feeder fault-current reference

**Status:** V0.1 synthetic exact-rational analytical reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11818  
**Composition:** #11645 fault containment, #11766 post-fault supply, #11801 robust MIMO stability, #4881 canonical power topology

## 1. Objective

Freeze a small independent mathematical reference for one source-fed DC fault path. Its purpose is to qualify the arithmetic and evidence interface for fault-current rise, clearing-time peak current, branch I²t and elementary energy balance. It is not a general DC short-circuit solver or a model of aircraft protection hardware.

## 2. Declared physical simplification

Assume a constant positive source Vs, series source/feeder resistance Rs, constant fault resistance Rf, total R = Rs + Rf > 0, series inductance L > 0, zero initial current, and constant parameters until an externally supplied clearing instant tc.

    L di/dt + R i = Vs
    i(0) = 0
    tau = L/R
    I_inf = Vs/R
    i(t) = I_inf * (1 - exp(-t/tau))

The current is monotonic over this domain. Thus i(tc) is the current immediately before interruption for this particular reference.

    I_prospective = Vs/R
    I_clear = (Vs/R) * (1 - exp(-tc/tau))

The branch let-through integral is:

    I2t(tc) = integral(0..tc) i(t)^2 dt
            = I_inf^2 * [
                tc - 2*tau*(1-exp(-x))
                + (tau/2)*(1-exp(-2*x))
              ]
    x = tc/tau

Resistive and magnetic energy terms are:

    E_fault_resistor = Rf * I2t
    E_source_feeder_resistor = Rs * I2t
    E_inductor = (1/2) * L * I_clear^2
    E_source_delivered = Vs * integral(0..tc) i(t) dt
                      = Vs * I_inf * [tc - tau*(1-exp(-x))]

The reference must verify the energy identity:

    E_source_delivered = E_fault_resistor
                       + E_source_feeder_resistor
                       + E_inductor

The identity follows from multiplying the differential equation by current and integrating. Interval outputs need only overlap consistently; the independent rational oracle must preserve containment and must not demand impossible endpoint equality when the same exponential term is conservatively bounded through separate expressions.

## 3. Exact rational fixture

    Vs = 600 V
    Rs = 1/20 ohm
    Rf = 1/20 ohm
    R  = 1/10 ohm
    L  = 1/1000 H
    tau = 1/100 s
    tc = 1/200 s
    x = tc/tau = 1/2
    I_prospective = 6000 A

For 0 <= x <= 1, the alternating Taylor partial sum

    S_n(x) = sum(k=0..n) [(-x)^k/k!]

is a lower bound for odd n and an upper bound for even n. The independent oracle uses exact Fraction arithmetic:
- exp(-1/2): order 13 lower, order 12 upper;
- exp(-1): order 13 lower, order 14 upper.

The fixture stores exact rational endpoints. Decimal displays are illustrative only.

Expected enclosures:
- I_clear: approximately 2360.816041724086 to 2360.816041724203 A;
- I2t: approximately 10483.775582106435 to 10483.775584185283 A²s;
- fault-resistor energy: approximately 524.188779105322 to 524.188779209264 J;
- source/feeder resistor energy: same, because Rs = Rf;
- inductor energy: approximately 2786.726191430891 to 2786.726191431168 J;
- source energy: approximately 3835.103749654779 to 3835.103749655485 J.

The exact fractions in the JSON fixture are authoritative.

## 4. Clearing-time evidence is an input, not an output

tc is an externally supplied hard upper bound from the exact device-specific protection evidence record. It must bind device identity/model/configuration, trip curve or test/certificate digest, current/voltage/ambient/installation applicability, tolerance, fault class, polarity, topology/configuration and scenario identity.

The RL reference does not infer a trip curve or clearing time. Missing or stale protection evidence is NOT_EVALUABLE. A nominal or average trip delay is not silently substituted for a hard upper bound.

## 5. Result and claim semantics

The output exposes prospective current, current at clearing, I²t, separate resistor energies, magnetic stored energy, source-delivered energy, exact input fractions, exponential-bound orders and all applicability identities.

This V0.1 fixture is ANALYTICAL_REFERENCE_ONLY. It does not produce a protection PASS, interrupting-capacity PASS, thermal PASS or aircraft safety PASS. A later composition may compare its result against separately admitted lower-bound device withstand and interrupt limits.

## 6. Negative controls and invariant tests

- Reject non-positive Vs, Rs, Rf, R, L or tc; reject any mismatch R != Rs + Rf.
- Reject unsupported x > 1/2 in this frozen V0.1 numerical domain.
- Reject any nonzero or undeclared initial current.
- Reject missing clearing-evidence identity for protection composition.
- Reject non-finite, overflow, invalid or negative I²t intervals.
- Reject a claim that capacitor-discharge, converter-controlled, multi-source, arc or breaker-recovery behavior was computed by this RL model.
- Verify exact rational enclosure against an independent Fraction oracle.
- Verify energy-balance interval overlap.
- Verify monotonic current and I²t under increasing Vs and increasing tc at fixed positive R,L within the admitted x domain.
- Verify exact order-independent canonical replay of the input/result representation.

## 7. Industry/SOTA position

A 2026 hybrid-electric-aircraft paper already describes fault-transient modeling that includes converter/load control behavior and compares against a semi-physical platform: https://www.mdpi.com/2226-4310/13/3/261. DC microgrid protection literature distinguishes capacitor-discharge transients from source-fed current and discusses the need for DC arc interruption: https://www.mdpi.com/2076-3417/12/1/15.

Therefore, the scalar RL reference is a verification foothold, not an attempt to outperform those system models. Future stages must explicitly add DC-link capacitor discharge, converter current limiting/control state, multiple sources, cable distribution, arc and breaker recovery voltage, and device/thermal withstand models. NASA NEAT supplies a future physical-correlation context with DC and bidirectional DC supplies, batteries, load banks, multiple power paths and fault-management capability: https://www.nasa.gov/eap-labs-and-testbeds/neat/

The differentiator is a replayable chain linking the exact fault scenario, topology, model validity, external clearing evidence, current/I²t enclosures, withstand evidence and post-fault supply identity.

## 8. Claim ceiling

V0.1 is a synthetic constant-source, constant-parameter, zero-initial-current, single-branch RL analytical reference. It does not model distributed DC-link capacitors, controlled converters, multiple feeding sources, cable wave propagation, DC arcs, breaker transient recovery voltage, selective coordination, actual device operation, thermal damage, aircraft stability, flight safety, airworthiness or certification. No energized fault-injection experiment is authorized.
