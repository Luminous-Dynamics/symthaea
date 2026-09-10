# Symthaea Space Exosuit — Rescue Propulsion v0.1

Status: research/simulation architecture. No flight-readiness or human-rating claim.

## Purpose

Add an optional personal self-rescue propulsion layer for regimes where the wearer cannot rely on useful ground reaction force. The intended baseline is bounded cold-gas rescue/translation, not routine powered flight.

## Environment roles

- Orbital EVA: self-rescue, detumble, attitude hold, arrest unintended separation, manual bounded translation.
- Small-body/microgravity operations: bounded translation may become a routine mobility option if later evidence supports it.
- Lunar/Mars surface: propulsion is a contingency impulse option only; exoskeleton, tether/winch, rover, and conventional transport remain preferred mobility layers.

## Hard authority boundary

Symthaea may estimate state, predict trajectories, recommend a rescue mode, and optimize resource use. It cannot directly energize thrusters.

The deterministic rescue kernel independently checks manual kill, watchdog health, isolation-valve state, propulsion power, IMU/nav validity, mass properties, command duration, acceleration ceilings, total delta-v, propellant reserve, enabled thrusters, and no-fire geometry.

The first automatic-action scope is deliberately narrow:

1. Detumble when independent attitude sensing is valid.
2. Arrest uncommanded relative drift when fresh, high-quality relative navigation confirms separation.

Automatic return-to-target guidance is out of scope for v0.1. Manual translation may be admitted inside the same hard envelope.

## Independent resources

PLSS oxygen is never a propulsion propellant. Rescue propulsion has an independent inert cold-gas propellant model and requires an independently available propulsion-power allocation. Losing rescue propulsion must not remove pressure, oxygen, ventilation, or thermal-control authority.

## No-fire geometry

Every nozzle is checked against protected geometry before allocation. The v0.1 code uses conservative protected spheres as a simple simulation primitive. Future hardware studies should replace/augment this with validated body meshes, plume cones, impingement models, and environment-specific keep-out zones while preserving fail-closed semantics.

Protected volumes should include, as appropriate:

- the wearer and pressure garment;
- visor/optics;
- PLSS seals and vulnerable surfaces;
- another crew member;
- fragile spacecraft/habitat surfaces;
- science samples and loose material zones;
- any region where plume impingement creates an unacceptable hazard.

## Posture-aware allocation

Thruster torque is computed about the supplied body center of mass, so the same nozzle layout can respond to changing wearer posture/load configuration. The current allocator is intentionally simple and simulation-only; a later tranche should couple full-frame pose into continuously estimated COM/inertia and compare deterministic constrained allocators.

## Relative navigation

`rescue_navigation` carries target-relative position/velocity, independent angular rate, tether tension, navigation quality, and state age. Stale/low-quality/invalid navigation fails closed. Tether loss by itself does not authorize translation; tether loss plus validated outward drift may recommend bounded `ArrestDrift`.

## Evidence program

SX-011: six-DOF rescue propulsion state, propellant and delta-v accounting.

SX-012: posture/COM-aware allocation and hard no-fire geometry.

SX-013: independent rescue authority, manual kill, watchdog, isolation, command/delta-v ceilings.

SX-014: relative navigation and bounded separation-arrest triggers.

SX-015: common-protocol benchmark across orbital EVA, lunar surface, Mars surface, and small-body microgravity, comparing exoskeleton, tether/winch, cold-gas, and rover strategies without hiding trade-offs in a weighted scalar score.

## Verification rules

- Simulation performance is not flight qualification.
- A qualification-level comparison cannot consume simulation-only measurements.
- Rescue propulsion is optional and non-survival-critical.
- A stuck-on thruster or stuck-open isolation path demands immediate propulsion isolation.
- Invalid relative navigation removes automatic translation authority.
- No optimizer may trade away hard safety constraints for time, energy, or mission productivity.
- No learned model, consciousness metric, or LLM bypasses the deterministic rescue kernel.

## Next evidence steps

After CI-clean software: couple the 20-DOF full-frame model to COM/inertia estimation; implement six-DOF rigid-body propagation for rescue pulses; introduce plume/impingement models; run seeded fault campaigns; then progress through cold-gas bench testing, air-bearing/suspension tests, vacuum/thermal testing, human-in-loop tests, and only much later a flight demonstration if the evidence justifies it.
