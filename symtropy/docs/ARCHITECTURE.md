# Symtropy Architecture

## Crate Dependency Graph

```
symtropy-math (no deps)
    Sphere<D>, Capsule<D>, HyperBox<D>, HalfSpace<D>, ConvexHull<D>
    Point<D>, Bivector<D>, Rotor<D>, Transform<D>
    Shape<D> trait (the universal collider interface)
        |
        v
symtropy-physics (depends on: symtropy-math, nalgebra, arrayvec)
    PhysicsWorld<D>          -- owns bodies, steps simulation
    RigidBody<D>             -- position, velocity, mass, collider
    GJK + EPA                -- collision detection (works in any D)
    CCD                      -- continuous collision for fast bodies
    ContactManifold<D>       -- multi-point contacts with warm-starting
    BallJoint, FixedJoint, HingeJoint  -- articulation constraints
    Raycast                  -- analytical ray-sphere intersection
    Replay + NetId           -- deterministic record/replay
    PhysicsCallback<D> trait -- THE coupling point for Phi
        |
        v
symtropy-consciousness-physics (depends on: symtropy-physics, symthaea-consciousness-equation)
    ConsciousnessField<D>    -- implements PhysicsCallback<D>
    EntityConsciousness      -- per-entity Phi computation + energy budget
    HarmonyField<D>          -- CEMI-inspired 1/r^(D-1) spatial fields
    EnergyBudget             -- Helmholtz free energy, 2nd Law enforcement
    ThermodynamicLedger      -- J/Phi metric, conservation tracking
    SafetyTier               -- NRC 4-tier motor authority
    FEP gradient             -- free energy principle agent behavior
    Active Inference         -- Bayesian belief updating
    DimensionalLeakage       -- 4D-to-3D energy transfer
    Convergence tools        -- Mann-Whitney U, Cohen's d, Holm-Bonferroni
```

## Which Crate Do I Need?

| I want to... | Use this crate |
|--------------|----------------|
| Do ND collision detection only | `symtropy-math` + `symtropy-physics` |
| Build a game with Phi-coupled physics | All three core crates + Bevy |
| Run consciousness-physics experiments | `symtropy-consciousness-physics` examples |
| Add a new collision shape | Implement `Shape<D>` in `symtropy-math` |
| Add a new joint type | Implement `Constraint<D>` in `symtropy-physics` |
| Couple custom metrics to physics | Implement `PhysicsCallback<D>` |

## The PhysicsCallback Trait

This is how integrated information enters the physics loop. The trait has 5 methods, called during collision resolution:

```rust
pub trait PhysicsCallback<const D: usize> {
    // Called when computing force on a body
    fn modulate_force(&self, body: BodyHandle, force: &SVector<f64, D>) -> SVector<f64, D>;

    // Called when resolving collision impulse
    fn modulate_impulse(&self, impulse: f64, contact_point: &SVector<f64, D>) -> f64;

    // Called when computing friction at a contact
    fn friction_multiplier(&self, contact_point: &SVector<f64, D>, body: BodyHandle) -> f64;

    // Called after a collision is resolved (for prediction error feedback)
    fn on_collision(&mut self, event: &CollisionEvent<D>);

    // Called when energy is dissipated (friction, damping)
    fn record_dissipation(&mut self, energy: f64);
}
```

`ConsciousnessField<D>` implements this trait. You can also implement it yourself for custom coupling metrics.

## Physics Step Data Flow

```
1. Clear events
2. Swap warm-starting caches
3. Integrate (semi-implicit Euler: v += a*dt, x += v*dt)
4. Broadphase (LBVH with Morton codes, or brute-force for <50 bodies)
   - Collision group/mask filtering
   - Skip static-static pairs
5. Narrowphase (GJK for intersection, EPA for penetration depth)
   - Sensor detection (emit SensorEvent, skip resolution)
   - Multi-point contact manifold generation
6. Resolve contacts (solver_iterations times):
   - Apply warm-started impulse (80% of previous frame)
   - Baumgarte position correction
   - Normal impulse (restitution)
   - Friction impulse (Coulomb, modulated by harmony fields)
   - Cache impulse for next frame
   - Emit CollisionEvent, call PhysicsCallback::on_collision()
7. Solve constraints (position + velocity level)
8. Body sleeping (velocity threshold, tick counter)
```

## Determinism Guarantees

The engine is designed for deterministic replay:

- **Integer Morton codes** in broadphase (no floating-point heuristics)
- **BTreeMap** for all entity maps (sorted iteration order)
- **Sorted collision pairs** by (BodyHandle, BodyHandle)
- **NetId** for stable body identity across machines
- **Record/replay** via `ReplayTape<D>` and `WorldSnapshot<D>`

What is NOT guaranteed:
- Cross-platform bitwise equality (different CPUs may give different float results)
- Determinism under multithreading (single-threaded physics step is deterministic)

## Extending the Engine

### Adding a New Shape

Implement `Shape<D>` in `symtropy-math`:

```rust
pub struct MyShape<const D: usize> { /* fields */ }

impl<const D: usize> Shape<D> for MyShape<D> {
    fn support(&self, direction: &SVector<f64, D>) -> SVector<f64, D> {
        // Return the point on the boundary furthest in `direction`
    }

    fn bounding_sphere(&self) -> (Point<D>, f64) {
        // Return (center, radius) of the smallest enclosing sphere
    }
}
```

GJK and EPA will automatically work with your shape.

### Adding a New Joint

Implement `Constraint<D>` in `symtropy-physics`:

```rust
pub struct MyJoint<const D: usize> { /* fields */ }

impl<const D: usize> Constraint<D> for MyJoint<D> {
    fn bodies(&self) -> (BodyHandle, BodyHandle) { /* ... */ }

    fn solve(&self, body_a: &mut RigidBody<D>, body_b: &mut RigidBody<D>, dt: f64) {
        // Apply positional corrections (called solver_iterations times)
    }

    fn solve_velocity(&self, body_a: &mut RigidBody<D>, body_b: &mut RigidBody<D>, dt: f64) {
        // Apply velocity corrections (optional, called after position solve)
    }
}
```

### Feature Flags

| Flag | Crate | What it enables |
|------|-------|----------------|
| `consciousness-curvature` | consciousness-physics | Conformal geometry: `ConformalMetric<D>`, geodesic corrections, Ricci scalar |
| `consciousness-hdc` | consciousness-physics | 16,384D HDC substrate: `HdcConsciousnessContext`, LTC neural network |
| `mycelix` | symtropy (game) | Governance, economy, factions, DKG, federated learning |
| `atlas` | symtropy (game) | Sol Atlas planetary globe view |
| `live-audio` | symtropy (game) | Real-time Phi-driven music synthesis |
| `net` | symtropy (game) | P2P networking (requires `world` feature) |
