# symthaea-homeops-core

Pure domain contracts for the NestPort / HomeOps residential receiving and internal material-flow research programme.

This crate does not contain robot drivers, actuators, Holochain calls, flight control, or a storage controller. It models package manifests, tote capacity, compartment compatibility, handoff receipts, and delivery-state transitions so adapters can be independently tested.

## Safety boundary

- Unknown or contradictory data must be held or rejected.
- Package weight and dimensions are not inferred from an AI prediction.
- Cold-chain acceptance requires a fresh environmental observation.
- Damaged or unknown seals are not silently admitted to ordinary storage.
- Restricted and human-review packages route only to quarantine.
- A state transition requires a legal predecessor and strictly increasing event time.
- No type in this crate grants physical actuator authority.

The v0.1 tote fit check is a conservative axis-aligned dimensional bound; it is not a 3D packing optimizer.
