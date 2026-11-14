# AIRROAD-DC-FAULT-EVENT-STATE-001 — state-continuous contactor event reference

**Status:** V0.1 synthetic two-mode exact-rational analytical reference  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/11849  
**Composition:** #11825 coupled pre-event circuit, #11645 protection evidence, #11766 post-fault supply

## 1. Objective

A protection event changes topology; it does not erase stored physical state. This reference tests one declared ideal source-contactor opening with an explicit inductor commutation path and a retained capacitor/fault discharge path.

This is a two-mode analytical reference. It is not a hardware contactor model or a general switched circuit solver.

## 2. Pre-event state

The pre-event reference is the exact coupled source/capacitor model from #11825:
- source 600 V;
- source series R=1 ohm, L=1 H;
- bus fault Rf=1 ohm, C=1 F;
- initial inductor current 0 A and capacitor voltage 600 V;
- event time te=1/4 s.

The fixture pins the exact-rational pre-event inductor-current and bus-voltage enclosures from the source/capacitor oracle.

## 3. Event and state map

At the synthetic event, an ideal source contactor opens. The source is disconnected. The inductor current must continue through a separately declared local freewheel loop containing Rcomm=1 ohm. The capacitor remains on its own Rf=1 ohm bus-discharge path.

The state transition is identity:

    iL(te+) = iL(te-)
    vC(te+) = vC(te-)

The event must carry the exact pre/post topology digests, configuration digest, event identity, timebase and path IDs. The event time is a synthetic mathematical fixture, not a device clearing-time or contactor-delay claim.

## 4. Post-event mode

With L=C=Rcomm=Rf=1 in SI units and Delta t=1/4 s:

    L diL/dt = -Rcomm*iL
    C dvC/dt = -vC/Rf
    i_source_terminal = 0 after source isolation

Therefore:

    iL(te+Delta t) = iL(te)*exp(-Delta t)
    vC(te+Delta t) = vC(te)*exp(-Delta t)

The current in the freewheel path is not source current. The capacitor fault-resistor current is a separate branch.

## 5. Energy closure

At the event:

    EL,0 = (1/2)*L*iL(te)^2
    EC,0 = (1/2)*C*vC(te)^2

During the post-event interval:

    Ecomm = (1/2)*L*iL(te)^2 * (1-exp(-2*Delta t))
    Efault = (1/2)*C*vC(te)^2 * (1-exp(-2*Delta t))

At the end:

    EL,1 = (1/2)*L*iL(te+Delta t)^2
    EC,1 = (1/2)*C*vC(te+Delta t)^2

The required invariant is:

    EL,0 + EC,0 = Ecomm + Efault + EL,1 + EC,1

All values are bounded using exact Fraction arithmetic and alternating Taylor enclosures for exp(-1/4) and exp(-1/2).

## 6. Negative controls

The independent oracle rejects:
- silently resetting inductor current to zero;
- silently resetting capacitor voltage to zero;
- an absent commutation path;
- a post-event topology that still draws from the opened source;
- mismatched configuration or event identities;
- promotion of a synthetic event time to device-specific clearing evidence;
- unaligned event timebases;
- a failed energy-balance enclosure.

If no commutation topology is supplied, current is NOT_EVALUABLE. It is not assumed to vanish instantaneously.

## 7. Benchmark targets

Targets, not achieved results:
- exact agreement with the #11825 pre-event state;
- exact identity-map continuity for inductor current and capacitor voltage;
- zero silent state resets;
- zero source current after ideal source isolation;
- energy closure for all post-event storage and dissipation;
- zero protection disposition without a real device-specific clearing record;
- deterministic replay and exact event/topology/configuration identity.

## 8. Industry and SOTA context

The 2026 hybrid-electric-aircraft transient study includes controlled sources/loads and semi-physical comparison; it distinguishes capacitor-discharge attenuation from control-influenced current: https://www.mdpi.com/2226-4310/13/3/261. The DC-breaker review discusses capacitor-discharge and source-fed responses separately: https://www.mdpi.com/2076-3417/12/1/15. NASA NEAT is the future physical-correlation context for multiple DC paths, bidirectional supplies, batteries and fault-management infrastructure: https://www.nasa.gov/eap-labs-and-testbeds/neat/

AIRROAD's contribution is the exact, independently replayable event-state/energy contract, not a claim to replace those richer models.

## 9. Claim ceiling

Synthetic ideal two-mode analytical reference only. No actual contactor, arc, interruption, current chopping, transient recovery voltage, protection-device timing/rating, aircraft safety, airworthiness or certification is qualified. No energized fault-injection or hardware-operation authority is created.
