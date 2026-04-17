# PhysicsCallback trait

`PhysicsCallback<D>` is the extension point where any per-body metric enters the physics loop. It lives in `symtropy-physics` (Apache-2.0 OR MIT), so implementing it in a proprietary project has no licensing cost.

## Signature

```rust
pub trait PhysicsCallback<const D: usize> {
    /// Called when computing net force on a body, before integration.
    fn modulate_force(&self, body: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D>;

    /// Called during collision resolution for each contact.
    fn modulate_impulse(&self, impulse: f64, contact_point: &SVector<f64, D>) -> f64;

    /// Called during friction resolution for each contact.
    fn friction_multiplier(&self, contact_point: &SVector<f64, D>, body: BodyHandle) -> f64;

    /// Called after a collision event fires (prediction error feedback).
    fn on_collision(&mut self, event: &CollisionEvent<D>);

    /// Called when energy is dissipated (friction, damping).
    fn record_dissipation(&mut self, energy: f64);
}
```

## When each method is called

```
step(dt):
  for body in bodies:
    force := collect_forces(body)
    force := callback.modulate_force(body, &force)      // ← channel 1
    body.velocity += (force / mass) * dt

  broadphase()
  narrowphase()

  for contact in contacts:
    impulse := compute_normal_impulse(contact)
    impulse := callback.modulate_impulse(impulse, &contact.point)  // ← channel 3

    μ := body.friction_coefficient
    μ := μ * callback.friction_multiplier(&contact.point, body)   // ← channel 4
    apply_friction(contact, μ)

    callback.on_collision(&event)   // ← channel 5
    callback.record_dissipation(dissipated_energy)  // ← channel 2 (indirect)
```

## Implementation examples

- **`SimpleCoupledField<D>`** — generic metric, 4-tier motor authority. Simplest starting point.
- **`ConsciousnessField<D>`** — Φ-based coupling with full 5-channel coupling, harmony fields, thermodynamic ledger.

Both in `symtropy-consciousness-physics` (AGPL). If you want to implement your own permissively-licensed callback, see the [generic state coupling](../core-concepts/generic-state-coupling.md) tutorial.

## Performance considerations

`PhysicsCallback` is called in the hot path — hundreds of times per physics step at scale. Rules:

- **No allocation.** Pre-allocate all storage; use `ArrayVec` or stack arrays.
- **O(1) or O(log n) per call.** Lookups via `HashMap` are fine (the non-determinism concern is only for *iteration*, not lookup).
- **No locks.** Physics stepping is single-threaded; your callback state is exclusively owned during step.

## Custom implementation

See [Generic state coupling § Implementing custom couplings](../core-concepts/generic-state-coupling.md#implementing-custom-couplings).
