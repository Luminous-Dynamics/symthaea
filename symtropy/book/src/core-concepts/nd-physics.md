# N-dimensional physics

> **Status:** stub — full chapter in Phase 0 of the [roadmap](../roadmap.md).

Symtropy's physics works in any spatial dimension. `PhysicsWorld<D>`, `Sphere<D>`, `HyperBox<D>`, `Rotor<D>`, `HingeJoint<D>` are all const-generic. 2D, 3D, 4D all compile with identical code.

## The types

| Type | Purpose | Dimensions |
|---|---|---|
| `Point<D>` | Position in D-space | Any D |
| `Bivector<D>` | Angular quantity (generalises 3D axis-angle to any D) | Any D |
| `Rotor<D>` | Rotation in D-space (generalises quaternions) | Any D |
| `Transform<D>` | Position + rotation | Any D |
| `PhysicsWorld<D>` | Simulation world | Any D |
| `RigidBody<D>` | Body with inertia, collider, transform | Any D |

## Why bivectors

In 3D, rotation is commonly represented as axis-angle (3 components) or quaternion (4 components). In 4D, rotation has **6 degrees of freedom** (the number of pairs of axes: xy, xz, xw, yz, yw, zw) — quaternions don't generalise. **Bivectors** are the natural representation: `Bivector<D>` has `D*(D-1)/2` components and works for any D.

Reference: ten Bosch, M. (2020). *N-Dimensional Rigid Body Dynamics*. SIGGRAPH.

## Performance across dimensions

GJK and EPA remain fast as D grows:

| Operation | 2D | 3D | 4D |
|---|---|---|---|
| GJK Sphere-Sphere | — | 156 ns | — |
| GJK HyperBox | — | 116 ns | 101 ns |
| GJK ConvexHull tesseract | — | — | 359 ns |

HyperBox uses O(D) support functions and is 3.5× faster than ConvexHull for 4D shapes.

## 4D cross-section visualisation

See [4D cross-section tutorial](../tutorials/4d-cross-sections.md).

## What stays 3D

The Bevy renderer itself is 3D. `symtropy-render-bridge` projects ND physics to 3D/4D Bevy meshes. For 4D, you slice a cross-section at a chosen hyperplane — Miegakure-style.
