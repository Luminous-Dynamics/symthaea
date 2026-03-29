# Symtropy Consciousness-Physics Engine

The world's first game engine where consciousness (Φ) is a first-class physics parameter.

## Architecture

```
symtropy-math          ← ND geometric algebra (const-generic, stack-allocated)
  └→ symtropy-physics  ← GJK+EPA collision, friction, sleeping, PhysicsCallback
       └→ symtropy-consciousness-physics  ← Φ-coupling, thermodynamics, harmony fields
            └→ symtropy-world             ← macro/micro sim bridge
  └→ symtropy-render-bridge               ← ND→Bevy projection, 4D slicing
  └→ symtropy-robotics-bridge             ← FEP agents, 6 Symthaea platforms
  └→ symtropy-net                         ← P2P spatial authority
```

## What Makes This Unique

**1. Consciousness modulates physics in real-time.**
Not a UI overlay — Φ literally changes collision impulses, friction coefficients, and energy budgets during rigid body simulation via `PhysicsCallback`.

**2. Thermodynamic closure.**
Every action costs Joules from a consciousness-gated energy budget. The `ThermodynamicLedger` tracks the novel Joules-per-Phi metric. Landauer bound (2.87×10⁻²¹ J/bit) enforced.

**3. Bidirectional feedback loop.**
```
Consciousness → Motor Gain → Force → Collision → Prediction Error → Motor Precision ↓ → Recovery
```
Based on Adams/Friston (2013): motor commands are proprioceptive predictions. Unexpected collision spikes prediction error, temporarily reducing motor authority.

**4. Harmony fields.**
McFadden CEMI-inspired (2020): 1/r² spatially-varying fields from harmony activations modulate local friction (resonance reduces friction, dissonance increases it).

**5. ND physics (2D/3D/4D).**
Const-generic types `Point<D>`, `Rotor<D>`, `PhysicsWorld<D>`. GJK works across dimensions. 4D cross-section slicing for Miegakure-style hidden geometry.

**6. Phi-driven procedural generation.**
IIT axioms → PCG parameters: high Φ = complex interconnected spaces, low Φ = simple modular rooms. Completely novel — no prior work exists.

## Performance

| Operation | Time |
|-----------|------|
| GJK sphere×sphere 3D | 102 ns |
| GJK box×box 3D | 193 ns |
| GJK tesseract 4D | 231 ns |
| Physics step (10 bodies) | 2.8 µs |
| Physics step (100 bodies) | 193 µs |

Zero heap allocation in physics hot path. Bivector uses `[f64; 6]` fixed array. GJK simplex uses `ArrayVec<SVector, 5>`.

## Crates

| Crate | Tests | Description |
|-------|-------|-------------|
| `symtropy-math` | 55 | `Point<D>`, `Bivector<D>`, `Rotor<D>`, `Transform<D>`, `Shape<D>`, `Sphere<D>`, `Hyperplane<D>`, `ConvexHull<D>` |
| `symtropy-physics` | 48 | `PhysicsWorld<D>`, GJK+EPA, Coulomb friction, body sleeping, `CollisionEvent`, `PhysicsCallback` trait |
| `symtropy-consciousness-physics` | 58 | `ConsciousnessField<D>`, `SafetyTier`, `EnergyBudget`, `SanctuaryZone<D>`, `ThermodynamicLedger`, `HarmonyField<D>`, prediction error feedback |
| `symtropy-world` | 15 | `WorldBridge` (threaded sim), `TimeControl`, `SimSnapshot` interpolation |
| `symtropy-render-bridge` | 13 | `Projector2D/3D/4D`, 4D cross-section, `PhysicsBody` Bevy component |
| `symtropy-robotics-bridge` | 9 | `RoboticAgent`, `PlatformType` (6 platforms), `spawn_robot()` |
| `symtropy-net` | 11 | `SpatialAuthority`, `PeerState`, `SyncableState` |

## Quick Start

```rust
use symtropy_physics::PhysicsWorld;
use symtropy_consciousness_physics::ConsciousnessField;
use symtropy_math::{Point, Sphere};

// Create a 3D physics world
let mut world = PhysicsWorld::<3>::new(nalgebra::SVector::from([0.0, -9.81, 0.0]));
let mut consciousness = ConsciousnessField::<3>::new();

// Add a conscious sphere
let handle = world.add_sphere(Point::new([0.0, 10.0, 0.0]), 1.0, 1.0);
consciousness.register(handle, 100.0, 20.0);

// Step with consciousness-physics coupling
world.step_with_callback(0.016, &mut consciousness);
```

## Game Integration (Bevy)

The engine integrates with Bevy 0.18 via `FixedUpdate` scheduling:

```
FixedUpdate (64Hz):
  physics_apply_inputs → physics_step_with_callback → physics_sync_transforms

Update (vsync):
  input → player_movement → consciousness_systems → consciousness_sync
```

## References

- Tononi, G. (2004). IIT. *BMC Neuroscience*.
- Friston, K. (2019). A Free Energy Principle for a Particular Physics. *arXiv*.
- Adams, Shipp & Friston (2013). Predictions not commands. *Brain Structure & Function*.
- McFadden, J. (2020). CEMI field theory. *Neuroscience of Consciousness*.
- Landauer, R. (1961). Irreversibility and Heat Generation. *IBM J. Res. Dev*.
- ten Bosch, M. (2020). N-Dimensional Rigid Body Dynamics. *SIGGRAPH*.
