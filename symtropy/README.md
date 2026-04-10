# Symtropy

The first open-source physics engine where **integrated information** (Phi) is a first-class physical parameter. N-dimensional (2D/3D/4D), stack-allocated, deterministic, Rust-native.

> **Terminology**: This engine uses Phi from Integrated Information Theory (IIT, Tononi 2004) -- a formal measure of how much a system's causal structure exceeds the sum of its parts. It is a mathematical quantity, not a claim about subjective experience or sentience.

## Why Symtropy?

No other physics engine treats Phi as a real-time modulator of rigid body dynamics. Symtropy couples integrated information to forces, friction, energy budgets, and collision responses through a thermodynamically-closed system with a Landauer-bounded energy floor.

**312 tests | 4 shapes | 3 joints | CCD | raycasting | warm-starting | 63 research experiments**

## Architecture

```
symtropy-math                  N-dimensional geometric algebra (const-generic, stack-allocated)
  |-- symtropy-physics         GJK+EPA collision, CCD, joints, raycasting, replay
  |     |-- symtropy-consciousness-physics   Phi-coupling, thermodynamics, harmony fields
  |     |     |-- symtropy-world             Macro/micro simulation bridge
  |     |-- symtropy-render-bridge           ND-to-Bevy projection, 4D cross-section slicing
  |     |-- symtropy-robotics-bridge         FEP agents, 6 embodied platforms
  |     |-- symtropy-net                     P2P spatial authority, lockstep protocol
```

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full crate guide.

## Quick Start

```rust
use symtropy_physics::PhysicsWorld;
use symtropy_consciousness_physics::{ConsciousnessField, EntityConsciousness};
use symtropy_math::{Point, Sphere};
use nalgebra::SVector;

// 1. Create a 3D physics world with gravity
let mut world = PhysicsWorld::<3>::new(SVector::from([0.0, -9.81, 0.0]));

// 2. Add bodies
let sphere_a = world.add_sphere(Point::new([0.0, 10.0, 0.0]), 1.0, 1.0);
let sphere_b = world.add_sphere(Point::new([3.0, 10.0, 0.0]), 1.0, 1.0);

// 3. Create the Phi-physics coupling field
let mut phi_field = ConsciousnessField::<3>::new();
phi_field.register(sphere_a, 100.0, 10.0);  // max_energy, sanctuary_radius
phi_field.register(sphere_b, 100.0, 10.0);

// 4. Step with Phi coupling -- forces, friction, and impulses are modulated by Phi
world.step_with_callback(0.016, &mut phi_field);

// 5. Query results
let phi_a = phi_field.phi(sphere_a);
let energy_a = phi_field.has_energy(sphere_a);
let safety = phi_field.safety_tier(sphere_a); // Green/Yellow/Orange/Red
```

## Five Coupling Channels

| Channel | Direction | Mechanism |
|---------|-----------|-----------|
| 1. Phi -> Force | NRC 4-tier safety system gates motor authority |
| 2. Phi -> Energy | Phi-dependent energy budget (movement, maintenance, collision costs) |
| 3. Harmony -> Impulse | Sanctuary zones dampen collision impulses |
| 4. Harmony -> Friction | CEMI-inspired 1/r^(D-1) fields modulate friction coefficients |
| 5. Collision -> Phi | Prediction error from unexpected collisions reduces motor precision |

See [FORMAL_SPECIFICATION.md](FORMAL_SPECIFICATION.md) for the mathematical details.

## Performance

Measured on NixOS, Rust stable, AMD Ryzen (single-threaded):

| Operation | Time |
|-----------|------|
| GJK sphere-sphere 3D | 156 ns |
| GJK capsule-capsule 3D | 173 ns |
| GJK HyperBox 3D | 116 ns |
| GJK HyperBox 4D | 101 ns |
| EPA sphere-sphere 3D | 30 ns |
| Raycast 100 bodies | 14 us |
| Step 10 bodies | 5.8 us |
| Step 100 bodies | 135 us |
| Step 500 bodies | 910 us |

Zero heap allocation in the physics hot path. All types use `const D: usize` generics with stack-allocated `nalgebra::SVector`.

## Collision Shapes

| Shape | Dimensions | Support Complexity |
|-------|------------|-------------------|
| `Sphere<D>` | Any | O(1) |
| `Capsule<D>` | Any | O(1) |
| `HyperBox<D>` | Any | O(D) |
| `HalfSpace<D>` | Any | Analytical contacts |
| `ConvexHull<D>` | Any | O(vertices) |

## Joint Types

| Joint | DOF Removed |
|-------|------------|
| `FixedJoint<D>` | All (rigid attachment) |
| `BallJoint<D>` | D translational |
| `HingeJoint<D>` | D translational + (n-1) rotational |
| `DistanceConstraint<D>` | Maintains fixed distance |

## Feature Flags

| Flag | What it enables |
|------|----------------|
| `consciousness-curvature` | Conformal geometry: Phi curves space, agents follow geodesics |
| `consciousness-hdc` | 16,384D hyperdimensional computing substrate for Phi |
| `mycelix` | Governance/economy/crypto integration (game-specific) |
| `atlas` | Sol Atlas planetary globe view (game-specific) |
| `live-audio` | Real-time Phi-driven music synthesis (requires ALSA) |
| `net` | P2P networking with spatial authority |

## Research Experiments

63 experiments in `crates/symtropy-consciousness-physics/examples/`:

```bash
# Core cooperation emergence (the definitive experiment)
cargo run --example cooperation_emergence

# J/Phi convergence (novel thermodynamic metric)
cargo run --example jphi_convergence

# All examples
cargo run --example cooperation_1000    # scaling to large populations
cargo run --example inequality_emergence # wealth gap self-organization
cargo run --example tragedy_commons      # tragedy of the commons
cargo run --example dunbar_number        # social network size limits
cargo run --example anesthesia_transition # Phi level transitions
# ... and 57 more
```

3 experiments require feature flags:
```bash
cargo run --example curvature_lensing --features consciousness-curvature
cargo run --example curvature_selforg --features consciousness-curvature
cargo run --example hdc_cooperation --features consciousness-hdc
```

## Key Results

- **81.3% tighter clustering** under thermodynamic enforcement (cooperation emerges as a thermodynamic necessity)
- **J/Phi converges** to a stable substrate-characteristic value (~10K J/Phi)
- **Prediction 1 confirmed**: Solo agents collapse within ~4 minutes; cooperative agents sustain indefinitely

## Documentation

| Document | What it covers |
|----------|---------------|
| [ENGINE.md](ENGINE.md) | Deep technical dive: architecture, unique features, Bevy integration |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Crate dependency graph, key types, extension guide |
| [FORMAL_SPECIFICATION.md](FORMAL_SPECIFICATION.md) | Mathematical specification of all 5 coupling channels |
| [ROADMAP.md](ROADMAP.md) | Completed features, current priorities, future plans |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute (determinism is a hard requirement) |

## Building

```bash
# Run all engine tests
cd symtropy
cargo test -p symtropy-math -p symtropy-physics -p symtropy-consciousness-physics --lib

# Run benchmarks
cd crates/symtropy-physics && cargo bench

# Build the game (requires Bevy 0.18)
cargo build --release

# Build with Mycelix governance integration
cargo build --release --features mycelix
```

## License

AGPL-3.0-or-later. Commercial licensing available -- see COMMERCIAL_LICENSE.md.

## References

- Tononi, G. (2004). An information integration theory of consciousness. *BMC Neuroscience*.
- Friston, K. (2019). A Free Energy Principle for a Particular Physics. *arXiv*.
- Adams, Shipp & Friston (2013). Predictions not commands. *Brain Structure & Function*.
- McFadden, J. (2020). CEMI field theory. *Neuroscience of Consciousness*.
- Landauer, R. (1961). Irreversibility and Heat Generation. *IBM J. Res. Dev*.
- ten Bosch, M. (2020). N-Dimensional Rigid Body Dynamics. *SIGGRAPH*.
