# Symtropy Roadmap

## North Star

The best **N-dimensional Phi-coupled simulation engine** with **deterministic replay** as a first-class physical law. Not trying to out-Rapier Rapier on general physics -- the niche is physics that can be meaningfully modulated by integration metrics (Phi), and that can be proven replayable.

---

## Completed

### Physics Foundation
- GJK collision detection (works in any D, stack-allocated simplex)
- EPA penetration depth (2D edge-based, 3D face-based, ND-generic facet-based)
- Semi-implicit Euler integrator with bivector angular dynamics
- LBVH broadphase with Morton encoding (integer quantization for determinism)
- Collision groups/masks (bitmask filtering) and sensor/trigger bodies
- Body sleeping (velocity threshold + tick counter)
- O(1) body handle index (HashMap lookup replacing O(n) scan)

### Collision Shapes (all ND, const-generic)
- `Sphere<D>` -- O(1) support
- `Capsule<D>` -- O(1) support, aligned to any axis
- `HyperBox<D>` -- O(D) support (3.5x faster than ConvexHull for 4D)
- `HalfSpace<D>` -- analytical contacts for sphere/capsule/box
- `ConvexHull<D>` -- O(vertices) support, quickhull construction

### Joint Types (all ND)
- `DistanceConstraint<D>` -- fixed distance, PBD + impulse hybrid
- `BallJoint<D>` -- removes D translational DOF
- `FixedJoint<D>` -- rigid attachment (zero DOF)
- `HingeJoint<D>` -- constrains rotation to one bivector plane

### Contact System
- Multi-point contact manifold (ArrayVec, zero-heap)
- Warm-starting with proximity-based contact matching (BTreeMap cache)
- Baumgarte position correction with configurable slop

### Advanced Features
- Continuous collision detection (swept sphere-sphere, sphere-halfspace)
- Ray casting (analytical ray-sphere, ND-generic)
- Deterministic replay (NetId, ReplayTape, WorldSnapshot, replay-cli binary)

### Phi-Physics Coupling (5 channels)
- Channel 1: Phi -> Force (NRC 4-tier safety: Green/Yellow/Orange/Red)
- Channel 2: Phi -> Energy (thermodynamic budget, Landauer bound, J/Phi metric)
- Channel 3: Harmony -> Impulse (sanctuary zones dampen collisions)
- Channel 4: Harmony -> Friction (CEMI-inspired 1/r^(D-1) fields)
- Channel 5: Collision -> Phi (prediction error reduces motor precision)
- Phi-gravity (integration-weighted attraction between entities)
- Temperature -> Phi feedback (heat stress reduces integration inputs)
- Dimensional leakage coupled to entity EnergyBudget

### Research & Validation
- 63 experiments (all compiling, covering cooperation, scaling, phase transitions, economics)
- Criterion benchmark suite (GJK per shape, EPA, raycast, step at 10/100/500 bodies)
- Key result: 81.3% tighter clustering under thermodynamic enforcement
- Key result: J/Phi converges to stable substrate-characteristic value

### Infrastructure
- Cargo workspace for core engine crates
- Terminology precision pass (Phi = integrated information, not a claim about experience)
- README.md, ARCHITECTURE.md, FORMAL_SPECIFICATION.md

---

## In Progress

- HalfSpace analytical contact dispatch in world.rs narrowphase (bypass GJK for common cases)
- Multi-contact generation from EPA (perpendicular direction sampling)

---

## Planned (by priority)

### High Priority
- **Prismatic joint** -- slide along one axis (suspension, sliders)
- **Motor/drive support** -- PD controllers on hinge + prismatic joints
- **Island formation** -- Union-Find over contact/constraint graphs, skip sleeping islands
- **Composite shapes** -- union of shapes on one body (e.g., capsule torso + sphere head)

### Medium Priority
- **Incremental broadphase** -- don't rebuild LBVH every frame (insert/remove/update)
- **Curvature feedback** -- entities source conformal curvature proportional to Phi
- **ND-generic ActiveInference** -- generalize from 2D to const-generic D
- **Proper variational free energy** -- replace FEP heuristic with real VFE computation
- **Cross-platform determinism** -- float quantization or constrained libm strategy

### Lower Priority
- **crates.io publication** -- publish core engine crates as independent packages
- **Bevy integration tutorial** -- step-by-step guide for external game developers
- **Cone/Cylinder shapes** -- specialized support functions
- **Triangle mesh collider** -- only if there's demand (Rapier territory)

---

## What We Will NOT Do

- GPU-accelerated broadphase (Jolt's territory; we target <1000 bodies)
- Soft body / cloth simulation (Bevy XPBD's direction)
- Jacobian-based solver (current PBD+impulse works at target scale)
- HeightField terrain (game feature, not engine fundamental)

---

## Contributions That Move the Needle

- Anything that strengthens determinism guarantees
- ND-first features (debug rendering, ergonomics for D != 3)
- New experiments that test the formal specification's predictions
- Tests that lock down invariants (ordering, floating-point edge cases)
- Documentation improvements and getting-started guides
