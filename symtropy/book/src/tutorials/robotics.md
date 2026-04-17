# Consciousness-gated robotics

> **Status:** stub — full tutorial in Phase 1 of the [roadmap](../roadmap.md).

This chapter will walk through:

1. Spawning a Symthaea robot (humanoid, quadruped, manipulator) via `symtropy-robotics-bridge`.
2. Wiring `EmbodimentBridge::step(thought_hv, dt, phi)` to the cognitive loop.
3. Observing Φ-gated motor authority at joint resolution.
4. Opt-in: switching to the `symtropy-rapier3d-bridge` backend for high-fidelity 3D physics.

## Planned platforms

- Humanoid (72D state / 21D cmd) — DMC-style contact dynamics
- Quadruped (4-leg × 3-joint)
- Manipulator (21D state / 8D cmd) — 7-DOF arm
- Quadrotor (13D / 4D) — flight dynamics
- Vehicle (20D / 3D) — bicycle + Pacejka tires
- AUV, helicopter, exoskeleton, surgical, orbital

All 10 Symthaea platforms implement `EmbodimentBridge`. Today's `symtropy-robotics-bridge` wires 3 of them at full state/command fidelity; the rest are skeletal — Phase 1 finishes the wiring.
