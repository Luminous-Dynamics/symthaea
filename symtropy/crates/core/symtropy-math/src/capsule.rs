// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! D-dimensional capsule (cylinder with hemispherical caps).
//!
//! A capsule is the Minkowski sum of a line segment and a sphere. This makes
//! it ideal for character controllers, robot limbs, and elongated bodies.
//! The support function is O(1) — no iteration over vertices.

use crate::point::Point;
use crate::shape::Shape;
use nalgebra::SVector;

/// D-dimensional capsule: a line segment of length `2 * half_height` along
/// `axis`, swept by a sphere of `radius`.
///
/// In local space, the two hemisphere centers are at:
/// - `+half_height * e_axis`
/// - `-half_height * e_axis`
///
/// where `e_axis` is the unit vector along the chosen axis.
#[derive(Clone, Copy, Debug)]
pub struct Capsule<const D: usize> {
    /// Half the distance between hemisphere centers.
    pub half_height: f64,
    /// Radius of the hemispherical caps (and the cylinder).
    pub radius: f64,
    /// Which coordinate axis the capsule is aligned to (0=X, 1=Y, 2=Z, ...).
    pub axis: usize,
}

impl<const D: usize> Capsule<D> {
    /// Create a capsule along the given axis.
    ///
    /// `half_height` is half the distance between hemisphere centers.
    /// Total length = `2 * half_height + 2 * radius`.
    pub fn new(half_height: f64, radius: f64, axis: usize) -> Self {
        debug_assert!(axis < D, "axis {axis} out of range for D={D}");
        Self {
            half_height,
            radius,
            axis,
        }
    }

    /// Create a Y-aligned capsule (the most common orientation).
    pub fn y_aligned(half_height: f64, radius: f64) -> Self {
        assert!(D >= 2, "Y-axis requires D >= 2");
        Self::new(half_height, radius, 1)
    }

    /// Total length along the axis (including caps).
    pub fn total_length(&self) -> f64 {
        2.0 * self.half_height + 2.0 * self.radius
    }
}

impl<const D: usize> Shape<D> for Capsule<D> {
    /// Support function: pick the hemisphere center that dots highest with
    /// the direction, then offset by `radius * normalize(direction)`.
    ///
    /// `support(d) = sign(d[axis]) * half_height * e_axis + radius * normalize(d)`
    fn support(&self, direction: &SVector<f64, D>) -> SVector<f64, D> {
        let norm = direction.norm();
        if norm < 1e-15 {
            // Degenerate direction — return the +axis hemisphere center
            let mut result = SVector::zeros();
            result[self.axis] = self.half_height;
            return result;
        }

        // Pick the hemisphere center in the direction of the projection
        let mut center = SVector::zeros();
        center[self.axis] = if direction[self.axis] >= 0.0 {
            self.half_height
        } else {
            -self.half_height
        };

        // Offset by radius in the support direction
        center + direction * (self.radius / norm)
    }

    fn bounding_sphere(&self) -> (Point<D>, f64) {
        (Point::origin(), self.half_height + self.radius)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn clone_box(&self) -> Box<dyn Shape<D>> {
        Box::new(*self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn support_along_axis() {
        let cap = Capsule::<3>::y_aligned(2.0, 0.5);
        let dir = SVector::from([0.0, 1.0, 0.0]);
        let sp = cap.support(&dir);
        // Should be at +half_height + radius along Y
        assert!((sp[1] - 2.5).abs() < 1e-12, "support Y+ = {}", sp[1]);
    }

    #[test]
    fn support_negative_axis() {
        let cap = Capsule::<3>::y_aligned(2.0, 0.5);
        let dir = SVector::from([0.0, -1.0, 0.0]);
        let sp = cap.support(&dir);
        assert!((sp[1] - (-2.5)).abs() < 1e-12, "support Y- = {}", sp[1]);
    }

    #[test]
    fn support_perpendicular() {
        let cap = Capsule::<3>::y_aligned(2.0, 0.5);
        let dir = SVector::from([1.0, 0.0, 0.0]);
        let sp = cap.support(&dir);
        // Perpendicular: picks +Y hemisphere, offsets by radius in X
        assert!((sp[0] - 0.5).abs() < 1e-12, "support X = {}", sp[0]);
    }

    #[test]
    fn support_diagonal() {
        let cap = Capsule::<3>::y_aligned(2.0, 0.5);
        let dir = SVector::from([1.0, 1.0, 0.0]);
        let sp = cap.support(&dir);
        // +Y hemisphere at (0, 2, 0), offset by 0.5 in direction (1,1,0)/sqrt(2)
        let expected_y = 2.0 + 0.5 / 2.0f64.sqrt();
        let expected_x = 0.5 / 2.0f64.sqrt();
        assert!((sp[1] - expected_y).abs() < 1e-10, "diag Y = {}", sp[1]);
        assert!((sp[0] - expected_x).abs() < 1e-10, "diag X = {}", sp[0]);
    }

    #[test]
    fn bounding_sphere_contains_capsule() {
        let cap = Capsule::<3>::y_aligned(3.0, 1.0);
        let (center, radius) = cap.bounding_sphere();
        assert!((center.coord(0)).abs() < 1e-12);
        assert!((radius - 4.0).abs() < 1e-12);
    }

    #[test]
    fn capsule_x_aligned() {
        let cap = Capsule::<3>::new(1.5, 0.3, 0);
        let dir = SVector::from([1.0, 0.0, 0.0]);
        let sp = cap.support(&dir);
        assert!((sp[0] - 1.8).abs() < 1e-12, "X-capsule support = {}", sp[0]);
    }

    #[test]
    fn capsule_4d() {
        let cap = Capsule::<4>::new(2.0, 1.0, 3); // W-axis aligned
        let dir = SVector::from([0.0, 0.0, 0.0, 1.0]);
        let sp = cap.support(&dir);
        assert!((sp[3] - 3.0).abs() < 1e-12, "4D capsule W+ = {}", sp[3]);
    }

    #[test]
    fn capsule_2d() {
        let cap = Capsule::<2>::new(1.0, 0.5, 0); // X-axis aligned
        let dir = SVector::from([0.0, 1.0]);
        let sp = cap.support(&dir);
        // Perpendicular: picks +X hemisphere, offsets by radius in Y
        assert!((sp[1] - 0.5).abs() < 1e-12, "2D capsule Y = {}", sp[1]);
    }

    #[test]
    fn degenerate_direction() {
        let cap = Capsule::<3>::y_aligned(2.0, 0.5);
        let dir = SVector::from([0.0, 0.0, 0.0]);
        let sp = cap.support(&dir);
        // Returns +axis hemisphere center
        assert!((sp[1] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn total_length() {
        let cap = Capsule::<3>::y_aligned(2.0, 0.5);
        assert!((cap.total_length() - 5.0).abs() < 1e-12);
    }
}
