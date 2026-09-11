# Symthaea Space Exosuit — Environment Fidelity v0.1

Status: simulation/reference architecture only. This document does not establish human rating, flight qualification, medical validity, or superiority over an existing EVA suit.

## SX-016 — Full-frame environment coupling

The existing 20-DOF Symtropy full-frame exoskeleton previously constructed its `PhysicsWorld` with Earth gravity while the Space Exosuit reduced-gravity work existed beside it as a reference model.

`full_frame_environment` closes that gap without forking Symtropy physics:

- Earth/Moon/Mars/custom `GravityEnvironment` values drive the actual `PhysicsWorld.gravity` field;
- the existing full-frame articulated chains remain the physical state source;
- pressure-garment resistance is evaluated from those chains' actual joint angle/velocity readbacks;
- per-joint resistance is injected as a simulation torque before the world step;
- the resulting total pressure-resistance mechanical power is observable for later metabolic/assist trade studies.

The torque injection is a research approximation. It is not a validated pressure-garment dynamics model and does not create actuator authority.

## SX-017 — Zoned dust durability

Dust is represented by separate functional zones rather than a single suit-health scalar:

1. visor;
2. joints;
3. seals;
4. radiator;
5. outer textile;
6. suitport interface.

The twin tracks retained deposition, friction, seal risk, optical transmission, radiator heat-rejection fraction, outer-textile optical/thermal factor, and generalized abrasive wear.

Electrodynamic dust-shield availability, removal fraction, power draw, and evidence level are modeled separately. EDS mitigates declared new deposition; it does not erase accumulated contamination by assumption.

Suitport cycles expose an explicit decontamination and contamination-transfer model so future habitat/rover studies can account for dust carried across an ingress boundary.

## Evidence boundary

All reference coefficients currently carry `Simulation` evidence. They are placeholders for trade studies and deterministic regression tests. Measured coupon, regolith, thermal-vacuum, bend-cycle, seal, radiator, and suitport data should replace those coefficients without changing the state model.

Simulation evidence must not be promoted to qualification evidence merely because the software passes tests.

## Authority boundary

Neither module commands life-support hardware or certified actuators. The full-frame coupler changes only simulation physics, and the dust twin is observational/predictive. The independent Space Exosuit assist safety kernel, PLSS authority, rescue-propulsion authority, and protected survival-power rules remain unchanged.

## Next coupling targets

The next evidence-bearing integration should connect zoned dust degradation to:

- pressure/joint mechanical work;
- radiator heat-rejection capability;
- EDS electrical demand;
- visor/operator constraints;
- maintenance and remaining-useful-life estimates;
- suitport/rover/habitat decontamination services.

That will let the mission harness discover second-order effects such as dust increasing joint work or degrading thermal rejection enough that an otherwise attractive assist policy becomes resource-inferior.
