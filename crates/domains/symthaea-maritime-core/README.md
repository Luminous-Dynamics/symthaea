# Symthaea Maritime Core

`sythaea-maritime-core` is a small, mission-neutral substrate for maritime autonomy.
It is intended to be shared by AUVs, USVs, research vessels, logistics craft,
remote sensors and larger maritime platforms without importing application-specific
mission policy.

## Design invariants

1. **Local safety is local.** Fleet connectivity, remote operators and external
   positioning may improve capability, but loss of those dependencies must not
   disable platform-local safety behavior.
2. **Degrade smoothly, never optimistically.** Missing trust, navigation or health
   inputs restrict the operating envelope. Platform-specific code may further
   restrict an envelope, but should not silently broaden it.
3. **Never average away an unsafe member.** Fleet assurance is per-platform and
   per-generation; one quarantined, stale or unsafe platform remains visible.
4. **Authority is bounded.** Capabilities are explicit, epoch-bound, time-bound and
   evidence-bound.
5. **No weapon semantics in the substrate.** Maritime core exposes sensing,
   navigation, communications, cargo, inspection and maintenance capabilities.
   Target engagement and lethal-force decisions are intentionally outside its type
   vocabulary.
6. **Bridges over duplication.** Existing AUV hydrodynamics, navigation, HAL rollout
   assurance, Xenia secure sessions and Mycelix governance/evidence should be
   integrated through narrow adapters rather than copied into this crate.

## Initial modules

- `state`: shared platform identity, kinematics, energy and navigation quality.
- `health`: component health and worst-case platform health aggregation.
- `degraded`: deterministic baseline operating envelopes under dependency loss.
- `authority`: mission-neutral capability leases with epoch/time/evidence bounds.
- `fleet`: per-member fleet admission and degraded-member reporting.

## Intended next adapters

- `symthaea-auv` → maritime state/health adapter.
- `symthaea-hal` → fleet rollout/rollback assurance adapter.
- Xenia → authenticated machine-session/evidence binding adapter.
- Mycelix → distributed identity, governance, provenance and logistics bridge.

This crate is not a physical safety controller, collision-avoidance implementation,
or regulatory compliance claim. Hardware and operational validation remain required
for any real vessel.
