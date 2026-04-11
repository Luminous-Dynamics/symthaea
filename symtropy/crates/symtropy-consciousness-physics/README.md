# symtropy-consciousness-physics

Consciousness (Φ) as a first-class physics parameter. The novel layer that makes Symtropy unique.

```rust
use symtropy_physics::PhysicsWorld;
use symtropy_consciousness_physics::ConsciousnessField;
use symtropy_math::Point;
use nalgebra::SVector;

// Create world and consciousness field
let mut world = PhysicsWorld::<3>::new(SVector::from([0.0, -9.81, 0.0]));
let mut consciousness = ConsciousnessField::<3>::new();

// Add a conscious entity
let handle = world.add_sphere(Point::new([0.0, 10.0, 0.0]), 1.0, 1.0);
consciousness.register(handle, 1000.0, 20.0); // 1000J budget, 20m sanctuary radius

// Physics step with consciousness coupling
// Φ modulates forces, impulses, friction, and energy in real-time
world.step_with_callback(0.016, &mut consciousness);
```

## Five Coupling Channels

1. **Φ → Motor gain**: Higher consciousness = more force authority (NRC 4-tier safety)
2. **Φ → Energy budget**: Consciousness gates available Joules per tick
3. **Harmony → Collision**: Sanctuary zones dampen impulse up to 90%
4. **Harmony → Friction**: Resonant agents reduce friction, dissonant agents increase it
5. **Collision → Prediction error**: Unexpected collision spikes error → reduces motor precision → habituates

## Thermodynamic Closure

The `ThermodynamicLedger` tracks energy conservation with scientific rigor:
- **Joules-per-Phi**: novel metric — energy cost of consciousness (no prior publication)
- **Landauer bound**: minimum thermodynamic cost enforced (2.87×10⁻²¹ J/bit at 310K)
- **Conservation error**: tracked and reported each tick

## Harmony Fields

Inspired by McFadden's CEMI field theory (2020). Each of the 8 harmonies creates a radial field with 1/r² falloff:
- Resonant fields (aligned harmonies) reduce friction — cooperation flows
- Dissonant fields (opposed harmonies) increase collision impulse — conflict escalates

## Sanctuary Zones

When Sacred Stillness > 0.6 and Φ > 0.3, a consciousness-created safe zone forms. Collision impulses dampened up to 90%. The physics literally protects conscious beings.

## Benchmarks

Measured on AMD Ryzen, `cargo bench -p symtropy-consciousness-physics`:

| Operation | N | Median |
|-----------|---|--------|
| `update_entity` (single) | 1 | 2.2 µs |
| `update_entity` (batch) | 8 | 26 µs |
| `update_entity` (batch) | 32 | 77 µs |
| `update_entity` (batch) | 128 | 706 µs |
| `tick_prediction_errors` | 8 | 133 ns |
| `tick_prediction_errors` | 64 | 1.1 µs |
| `tick_prediction_errors` | 256 | 7.7 µs |
| `spread_emotional_contagion` | 4 | 2.7 µs |
| `spread_emotional_contagion` | 16 | 32 µs |
| `spread_emotional_contagion` | 64 | 386 µs |
| `apply_macro_modifiers` | 32 | 626 ns |
| `phi_env_coupler_tick` | 1 | 778 ns |
| `biometric_to_consciousness` | 30 | 1.2 µs |
| `full_step_with_consciousness` | 4 | 13 µs |
| `full_step_with_consciousness` | 16 | 60 µs |
| `bifurcation_sweep` (3k×2) | — | 222 µs |

At 60 Hz with 16 agents: ~60 µs/tick → **16,600 ticks/second** (277× real-time budget).

## WASM Compatible

Part of the [Symtropy consciousness-physics engine](https://github.com/luminous-dynamics/symtropy).
