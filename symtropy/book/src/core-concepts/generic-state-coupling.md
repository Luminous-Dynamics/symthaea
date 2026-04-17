# Generic state coupling

Symtropy's coupling framework is generic. The default metric is Φ (integrated information) because that's the research contribution, but the machinery accepts any scalar metric in `[0, 1]`. This page shows how to use it for health, trust, skill, wealth, or any custom quantity.

## The interface

Two types do the work:

- **[`PhysicsCallback<D>`](../reference/physics-callback.md)** — the trait physics-world calls during each solver step. Five methods: `modulate_force`, `modulate_impulse`, `friction_multiplier`, `on_collision`, `record_dissipation`.
- **`SimpleCoupledField<D>`** — a ready-made implementation that treats a per-body `[0, 1]` metric as motor authority, with the same 4-tier safety system used for Φ.

You can use either. `SimpleCoupledField` is the easy path; implementing `PhysicsCallback` yourself is for custom couplings.

## The four-tier motor authority model

Every metric value maps to one of four tiers, lifted from the Nuclear Regulatory Commission's safety framework:

| Tier | Metric range | Motor authority | Meaning |
|---|---|---|---|
| **Green** | `≥ 0.6` | 100 % | Normal operation. Full motor gain. |
| **Yellow** | `0.3 – 0.6` | 50 % | Degraded. Gravity still applies; commands half-strength. |
| **Orange** | `0.1 – 0.3` | 20 % | Severely limited. Physics applies; motor almost silent. |
| **Red** | `< 0.1` | 0 % | Inert. Physics applies (gravity, collision); zero command response. |

Tier thresholds are configurable. The default values come from IIT's phase-transition literature.

## Worked example — trust modulates friction

Suppose you're building a social simulation where high trust reduces friction in collaborative zones (resonance) and distrust increases it (dissonance).

```rust
use symtropy_physics::{PhysicsWorld, BodyHandle};
use symtropy_consciousness_physics::SimpleCoupledField;
use symtropy_math::Point;
use nalgebra::SVector;

let mut world = PhysicsWorld::<2>::new(SVector::from([0.0, -9.81]));
let mut trust_field = SimpleCoupledField::<2>::new();

// Three agents, different trust levels
let alice = world.add_sphere(Point::new([-2.0, 5.0]), 0.5, 1.0);
let bob   = world.add_sphere(Point::new([ 0.0, 5.0]), 0.5, 1.0);
let eve   = world.add_sphere(Point::new([ 2.0, 5.0]), 0.5, 1.0);

trust_field.register(alice, 100.0, 10.0);
trust_field.register(bob,   100.0, 10.0);
trust_field.register(eve,   100.0, 10.0);

// Alice and Bob trust each other; Eve is viewed with suspicion
trust_field.set_metric(alice, 0.9);
trust_field.set_metric(bob,   0.9);
trust_field.set_metric(eve,   0.2);

// Alice and Bob interact with low friction; Eve slides harder against the floor
for _ in 0..600 {
    world.step_with_callback(1.0 / 60.0, &mut trust_field);
}
```

Over ten seconds, Alice and Bob settle into stable contact with minimal heat dissipation. Eve's high-friction interaction bleeds energy — if you log `trust_field.energy(eve)`, you'll see it depleting faster than the others' budgets.

## Worked example — health as motor authority

For a game where low HP reduces movement precision:

```rust
let hp = 0.35;                          // 35% health
health_field.set_metric(player, hp);    // Yellow tier → 50% motor authority

// Applying a jump force
let desired_jump = SVector::from([0.0, 10.0]);
let actual_force = health_field.modulate_force(player, &desired_jump);
// actual_force ≈ [0.0, 5.0]  — scaled by tier
```

No code in your game needs special-casing for low health; the physics layer enforces it.

## Implementing custom couplings

If you need custom behaviour (e.g. metric affects only friction, not force), implement `PhysicsCallback` yourself:

```rust
use symtropy_physics::{PhysicsCallback, BodyHandle, CollisionEvent};
use nalgebra::SVector;

struct WealthField {
    wealth: std::collections::HashMap<BodyHandle, f64>,
}

impl<const D: usize> PhysicsCallback<D> for WealthField {
    fn modulate_force(&self, _body: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D> {
        *force  // unchanged
    }

    fn friction_multiplier(&self, _point: &SVector<f64, D>, body: BodyHandle) -> f64 {
        // Wealthier bodies slide easier — lower friction
        let w = self.wealth.get(&body).copied().unwrap_or(0.5);
        1.0 - 0.5 * w
    }

    fn modulate_impulse(&self, impulse: f64, _point: &SVector<f64, D>) -> f64 { impulse }
    fn on_collision(&mut self, _event: &CollisionEvent<D>) {}
    fn record_dissipation(&mut self, _energy: f64) {}
}
```

Now drop it into the same `step_with_callback`:

```rust
let mut wealth = WealthField { wealth: Default::default() };
world.step_with_callback(dt, &mut wealth);
```

No changes needed to `PhysicsWorld`.

## Licensing for custom couplings

`PhysicsCallback` lives in `symtropy-physics` (Apache-2.0 OR MIT). You can implement it in a proprietary closed-source project without AGPL obligations. Only `symtropy-consciousness-physics` (which provides `SimpleCoupledField`, `ConsciousnessField`, `HarmonyField`, `ThermodynamicLedger`) is AGPL-licensed.

## Further reading

- [The five coupling channels](./five-channels.md) — force, energy, impulse, friction, prediction-error feedback.
- [Φ-coupling](./phi-coupling.md) — how integrated information plugs into this same framework.
- [`PhysicsCallback` reference](../reference/physics-callback.md) — full trait documentation.
