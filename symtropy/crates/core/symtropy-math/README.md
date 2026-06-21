# symtropy-math

N-dimensional geometric algebra for game physics. Stack-allocated, zero-heap, const-generic.

```rust
use symtropy_math::{Point, Bivector, Rotor};

// 3D rotation in the xy plane by 90°
let plane = Bivector::<3>::unit_plane(0, 1);
let r = Rotor::from_plane_angle(&plane, std::f64::consts::FRAC_PI_2);
let p = Point::<3>::new([1.0, 0.0, 0.0]);
let rotated = r.rotate_point(&p); // → (0, 1, 0)

// Works in 4D too
let tesseract_rotation = Bivector::<4>::unit_plane(0, 3);
let r4d = Rotor::from_plane_angle(&tesseract_rotation, 0.5);
```

## Features

- `Point<D>`, `Bivector<D>`, `Rotor<D>`, `Transform<D>` — all const-generic
- `Shape<D>` trait with GJK-compatible support function
- `Sphere<D>`, `ConvexHull<D>`, `Hyperplane<D>` colliders
- Stack-allocated via `nalgebra::SVector` — zero heap in hot paths
- Works in 2D, 3D, 4D, or any dimension
- WASM compatible

## Why Bivectors Instead of Quaternions?

Quaternions only work in 3D. Bivectors (oriented planes) generalize rotations to any dimension. A rotation happens *in a plane*, not *around an axis*. In 4D there are 6 rotation planes — bivectors handle this naturally.

Part of the [Symtropy consciousness-physics engine](https://github.com/luminous-dynamics/symtropy).
