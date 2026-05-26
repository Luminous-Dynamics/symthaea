// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! N-dimensional geometric algebra for the Symtropy consciousness-physics engine.
//!
//! All types are parameterized by `const D: usize` — the spatial dimension.
//! This forces stack allocation via `nalgebra::SVector<f64, D>`, giving zero-allocation
//! physics ticks with full SIMD optimization for the common 2D/3D/4D cases.
//!
//! # Usage
//! ```
//! use symtropy_math::{Point, Bivector, Rotor, Transform};
//!
//! // 3D rotation in the xy plane by 90°
//! let plane = Bivector::<3>::unit_plane(0, 1);
//! let r = Rotor::from_plane_angle(&plane, std::f64::consts::FRAC_PI_2);
//! let p = Point::<3>::new([1.0, 0.0, 0.0]);
//! let rotated = r.rotate_point(&p);
//! ```
//!
//! The bivector component count is computed at compile time:
//! - 2D: 1 component (xy)
//! - 3D: 3 components (xy, xz, yz)
//! - 4D: 6 components (xy, xz, xw, yz, yw, zw)

pub mod bivector;
pub mod capsule;
pub mod compound;
pub mod convex_hull;
pub mod halfspace;
pub mod hyperbox;
pub mod hyperplane;
pub mod point;
pub mod rotor;
pub mod shape;
pub mod sphere;
pub mod sphere_utils;
pub mod transform;

pub use bivector::Bivector;
pub use capsule::Capsule;
pub use compound::CompoundShape;
pub use convex_hull::ConvexHull;
pub use halfspace::HalfSpace;
pub use hyperbox::HyperBox;
pub use hyperplane::Hyperplane;
pub use point::Point;
pub use rotor::Rotor;
pub use shape::Shape;
pub use sphere::Sphere;
pub use sphere_utils::SphereUtils;
pub use transform::Transform;

/// Type alias for common dimension specializations.
pub type Point2 = Point<2>;
pub type Point3 = Point<3>;
pub type Point4 = Point<4>;

pub type Rotor2 = Rotor<2>;
pub type Rotor3 = Rotor<3>;
pub type Rotor4 = Rotor<4>;

pub type Transform2 = Transform<2>;
pub type Transform3 = Transform<3>;
pub type Transform4 = Transform<4>;

pub type Bivector2 = Bivector<2>;
pub type Bivector3 = Bivector<3>;
pub type Bivector4 = Bivector<4>;
