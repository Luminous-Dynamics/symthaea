# Shape and Constraint extension

> **Status:** stub — draws from existing `../ARCHITECTURE.md` "Extending the Engine" section. Full examples coming in Phase 0.

Symtropy is extensible at two structural points:

## Adding a new shape

Implement `Shape<D>` in `symtropy-math`:

```rust
pub struct MyShape<const D: usize> { /* fields */ }

impl<const D: usize> Shape<D> for MyShape<D> {
    fn support(&self, direction: &SVector<f64, D>) -> SVector<f64, D> {
        // Return the point on the boundary furthest in `direction`.
    }

    fn bounding_sphere(&self) -> (Point<D>, f64) {
        // (center, radius) of the smallest enclosing sphere.
    }
}
```

GJK and EPA work automatically.

## Adding a new joint

Implement `Constraint<D>` in `symtropy-physics`:

```rust
pub struct MyJoint<const D: usize> { /* fields */ }

impl<const D: usize> Constraint<D> for MyJoint<D> {
    fn bodies(&self) -> (BodyHandle, BodyHandle) { /* ... */ }

    fn solve(&self, body_a: &mut RigidBody<D>, body_b: &mut RigidBody<D>, dt: f64) {
        // Positional corrections (called `solver_iterations` times).
    }

    fn solve_velocity(&self, body_a: &mut RigidBody<D>, body_b: &mut RigidBody<D>, dt: f64) {
        // Optional velocity-level corrections.
    }
}
```

## Optional ecosystem crates

Out-of-core extensions live as community crates:

- `symtropy-physics-gpu` — GPU broadphase
- `symtropy-soft` — soft body / cloth / XPBD
- `symtropy-mesh` — triangle mesh collider
- `symtropy-terrain` — heightfield + LOD streaming
- `symtropy-fluid` — SPH / FLIP

See the [roadmap](../roadmap.md) Phase 5 for the policy on ecosystem crates.
