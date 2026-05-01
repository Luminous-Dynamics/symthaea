// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Compound shape: a rigid body composed of multiple convex sub-shapes.
//!
//! Each child is placed at a fixed offset (Transform) relative to the
//! compound origin. The support function reduces to an argmax over all
//! children's individual support points — O(n_children) per GJK query,
//! which is acceptable since compound bodies typically have ≤16 parts.
//!
//! # Usage
//! ```
//! use symtropy_math::{CompoundShape, Point, Sphere, Transform};
//!
//! // A dumbbell: two spheres connected along the x axis.
//! let mut compound = CompoundShape::<3>::new();
//! compound.add_child(
//!     Transform::from_translation(Point::new([-2.0, 0.0, 0.0])),
//!     Box::new(Sphere::unit()),
//! );
//! compound.add_child(
//!     Transform::from_translation(Point::new([2.0, 0.0, 0.0])),
//!     Box::new(Sphere::unit()),
//! );
//! ```

use nalgebra::SVector;

use crate::point::Point;
use crate::shape::Shape;
use crate::transform::Transform;

/// A convex rigid body composed of multiple child shapes.
///
/// The overall support function is the argmax of all children's support
/// points transformed into the compound's local frame. This makes compound
/// shapes first-class `Shape<D>` objects compatible with GJK.
pub struct CompoundShape<const D: usize> {
    /// Children: (local-frame transform, shape).
    children: Vec<(Transform<D>, Box<dyn Shape<D>>)>,
    /// Cached bounding sphere: (center-in-local-frame, radius).
    cached_center: Point<D>,
    cached_radius: f64,
}

impl<const D: usize> CompoundShape<D> {
    /// Create an empty compound shape.
    pub fn new() -> Self {
        Self {
            children: Vec::new(),
            cached_center: Point::origin(),
            cached_radius: 0.0,
        }
    }

    /// Add a child shape at the given local-frame transform.
    ///
    /// Recomputes the cached bounding sphere after each addition.
    pub fn add_child(&mut self, transform: Transform<D>, shape: Box<dyn Shape<D>>) {
        self.children.push((transform, shape));
        self.recompute_bounding();
    }

    /// Number of child shapes.
    pub fn child_count(&self) -> usize {
        self.children.len()
    }

    /// Recompute the enclosing bounding sphere over all children.
    ///
    /// Algorithm:
    /// 1. Centroid = mean of all child bounding-sphere centers in local frame.
    /// 2. Radius = max(dist(centroid, child_center_world) + child_radius).
    fn recompute_bounding(&mut self) {
        if self.children.is_empty() {
            self.cached_center = Point::origin();
            self.cached_radius = 0.0;
            return;
        }

        // Step 1: centroid of all child sphere centers (in compound local frame)
        let mut center = SVector::<f64, D>::zeros();
        for (tf, child) in &self.children {
            let (local_c, _) = child.bounding_sphere();
            let world_c = tf.transform_point(&local_c).0;
            center += world_c;
        }
        center /= self.children.len() as f64;

        // Step 2: tightest enclosing sphere around centroid
        let mut radius = 0.0f64;
        for (tf, child) in &self.children {
            let (local_c, child_r) = child.bounding_sphere();
            let world_c = tf.transform_point(&local_c).0;
            let d = (world_c - center).norm() + child_r;
            if d > radius {
                radius = d;
            }
        }

        self.cached_center = Point(center);
        self.cached_radius = radius;
    }
}

impl<const D: usize> Default for CompoundShape<D> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const D: usize> Shape<D> for CompoundShape<D> {
    /// Support function: furthest point on any child in the given direction.
    ///
    /// For each child:
    ///   1. Rotate `direction` into child's local frame (inverse rotation).
    ///   2. Query child's support in that local direction.
    ///   3. Transform the result back to compound-local frame.
    ///   4. Return the child whose result has the maximum dot with `direction`.
    fn support(&self, direction: &SVector<f64, D>) -> SVector<f64, D> {
        self.children
            .iter()
            .map(|(tf, child)| {
                // Rotate direction into child's local frame
                let local_dir = tf.rotation.reverse().rotate_vector(direction);
                // Support in child's local frame
                let local_pt = child.support(&local_dir);
                // Transform back to compound's frame (rotate + translate)
                tf.transform_point(&Point(local_pt)).0
            })
            .max_by(|a, b| {
                let da = a.dot(direction);
                let db = b.dot(direction);
                da.total_cmp(&db)
            })
            .unwrap_or_else(SVector::zeros)
    }

    fn bounding_sphere(&self) -> (Point<D>, f64) {
        (self.cached_center, self.cached_radius)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bivector::Bivector;
    use crate::rotor::Rotor;
    use crate::sphere::Sphere;

    fn vec3(x: f64, y: f64, z: f64) -> SVector<f64, 3> {
        SVector::from([x, y, z])
    }

    /// Dumbbell: two unit spheres placed ±2 along X.
    fn dumbbell() -> CompoundShape<3> {
        let mut c = CompoundShape::new();
        c.add_child(
            Transform::from_translation(Point::new([-2.0, 0.0, 0.0])),
            Box::new(Sphere::unit()),
        );
        c.add_child(
            Transform::from_translation(Point::new([2.0, 0.0, 0.0])),
            Box::new(Sphere::unit()),
        );
        c
    }

    #[test]
    fn support_along_positive_x_reaches_far_child() {
        let db = dumbbell();
        let s = db.support(&vec3(1.0, 0.0, 0.0));
        // Right sphere center at x=2, radius=1 → support at x=3
        assert!((s[0] - 3.0).abs() < 1e-10, "support x = {}", s[0]);
    }

    #[test]
    fn support_along_negative_x_reaches_near_child() {
        let db = dumbbell();
        let s = db.support(&vec3(-1.0, 0.0, 0.0));
        // Left sphere center at x=-2, radius=1 → support at x=-3
        assert!((s[0] - (-3.0)).abs() < 1e-10, "support x = {}", s[0]);
    }

    #[test]
    fn support_along_y_is_symmetric() {
        let db = dumbbell();
        let s_pos = db.support(&vec3(0.0, 1.0, 0.0));
        let s_neg = db.support(&vec3(0.0, -1.0, 0.0));
        // Both spheres have radius 1; y component should be ±1
        assert!((s_pos[1] - 1.0).abs() < 1e-10);
        assert!((s_neg[1] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn bounding_sphere_covers_all_children() {
        let db = dumbbell();
        let (center, radius) = db.bounding_sphere();
        // Centroid is at origin (symmetric dumbbell)
        assert!(center.0.norm() < 1e-10, "centroid should be at origin");
        // Farthest point: sphere at x=2, radius=1 → dist 2 + 1 = 3
        assert!(radius >= 3.0 - 1e-10, "radius = {}", radius);
    }

    #[test]
    fn single_child_matches_original_shape() {
        let r2 = Sphere::new(Point::new([0.0, 0.0, 0.0]), 2.0);
        let mut c = CompoundShape::<3>::new();
        c.add_child(Transform::identity(), Box::new(r2));

        let dir = vec3(1.0, 0.0, 0.0);
        let s_compound = c.support(&dir);
        let s_direct = Sphere::new(Point::new([0.0, 0.0, 0.0]), 2.0).support(&dir);
        assert!((s_compound - s_direct).norm() < 1e-10);
    }

    #[test]
    fn child_with_rotation_transforms_direction_correctly() {
        // Sphere at origin with 90° XY rotation applied.
        // A sphere is isotropic, so rotation shouldn't matter — but transform_point
        // must not produce garbage for a rotated child offset.
        let plane = Bivector::<3>::unit_plane(0, 1);
        let rot = Rotor::from_plane_angle(&plane, std::f64::consts::FRAC_PI_2);
        let tf = Transform {
            translation: Point::new([1.0, 0.0, 0.0]),
            rotation: rot,
        };
        let mut c = CompoundShape::<3>::new();
        c.add_child(tf, Box::new(Sphere::unit()));

        // Support along +X: sphere center at x=1, radius=1 → x=2
        let s = c.support(&vec3(1.0, 0.0, 0.0));
        assert!((s[0] - 2.0).abs() < 1e-10, "s[0] = {}", s[0]);
    }

    #[test]
    fn empty_compound_returns_zero_support() {
        let c = CompoundShape::<3>::new();
        let s = c.support(&vec3(1.0, 0.0, 0.0));
        assert!(s.norm() < 1e-10);
    }

    #[test]
    fn bounding_sphere_empty_is_zero() {
        let c = CompoundShape::<3>::new();
        let (center, radius) = c.bounding_sphere();
        assert!(center.0.norm() < 1e-10);
        assert!(radius < 1e-10);
    }

    #[test]
    fn works_in_2d() {
        let mut c = CompoundShape::<2>::new();
        c.add_child(
            Transform::from_translation(Point::new([-1.0, 0.0])),
            Box::new(Sphere::new(Point::new([0.0, 0.0]), 0.5)),
        );
        c.add_child(
            Transform::from_translation(Point::new([1.0, 0.0])),
            Box::new(Sphere::new(Point::new([0.0, 0.0]), 0.5)),
        );
        let s = c.support(&SVector::<f64, 2>::from([1.0, 0.0]));
        assert!((s[0] - 1.5).abs() < 1e-10);
    }

    #[test]
    fn works_in_4d() {
        let mut c = CompoundShape::<4>::new();
        c.add_child(
            Transform::from_translation(Point::new([0.0, 0.0, 0.0, 3.0])),
            Box::new(Sphere::unit()),
        );
        let s = c.support(&SVector::<f64, 4>::from([0.0, 0.0, 0.0, 1.0]));
        assert!((s[3] - 4.0).abs() < 1e-10);
    }
}
