// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! D-dimensional axis-aligned box (hyperbox / tesseract).
//!
//! The support function is O(D) — simply `sign(d[i]) * half_extents[i]` per axis.
//! This is dramatically faster than `ConvexHull::hyperbox()` which is O(2^D)
//! because it iterates over all vertices. For D=4, that's O(4) vs O(16).

use crate::point::Point;
use crate::shape::Shape;
use nalgebra::SVector;

/// D-dimensional axis-aligned box centered at the origin.
///
/// `half_extents[i]` is the half-width along axis `i`.
/// Total size along axis `i` = `2 * half_extents[i]`.
#[derive(Clone, Copy, Debug)]
pub struct HyperBox<const D: usize> {
    pub half_extents: [f64; D],
}

impl<const D: usize> HyperBox<D> {
    /// Create a box with the given half-extents per axis.
    pub fn new(half_extents: [f64; D]) -> Self {
        Self { half_extents }
    }

    /// Create a uniform cube (all axes equal).
    pub fn cube(half_extent: f64) -> Self {
        Self {
            half_extents: [half_extent; D],
        }
    }

    /// Volume of the hyperbox: product of (2 * half_extent_i) for all axes.
    pub fn volume(&self) -> f64 {
        self.half_extents.iter().map(|h| 2.0 * h).product()
    }

    /// Whether a local-space point is inside the box.
    pub fn contains_local(&self, point: &Point<D>) -> bool {
        (0..D).all(|i| point.coord(i).abs() <= self.half_extents[i])
    }
}

impl<const D: usize> Shape<D> for HyperBox<D> {
    /// Support function: `support(d)[i] = sign(d[i]) * half_extents[i]`.
    ///
    /// O(D) — no vertex enumeration. For a 4D tesseract, this is 4 operations
    /// instead of 16 (the vertex count of a tesseract).
    fn support(&self, direction: &SVector<f64, D>) -> SVector<f64, D> {
        SVector::from_fn(|i, _| {
            if direction[i] >= 0.0 {
                self.half_extents[i]
            } else {
                -self.half_extents[i]
            }
        })
    }

    fn bounding_sphere(&self) -> (Point<D>, f64) {
        let radius = self.half_extents.iter().map(|h| h * h).sum::<f64>().sqrt();
        (Point::origin(), radius)
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
    fn support_axis_aligned() {
        let b = HyperBox::<3>::new([2.0, 3.0, 1.0]);
        let dir = SVector::from([1.0, 0.0, 0.0]);
        let sp = b.support(&dir);
        assert!((sp[0] - 2.0).abs() < 1e-12);
        assert!((sp[1] - 3.0).abs() < 1e-12); // tie goes to positive
        assert!((sp[2] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn support_negative() {
        let b = HyperBox::<3>::cube(1.0);
        let dir = SVector::from([-1.0, -1.0, -1.0]);
        let sp = b.support(&dir);
        assert!((sp[0] - (-1.0)).abs() < 1e-12);
        assert!((sp[1] - (-1.0)).abs() < 1e-12);
        assert!((sp[2] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn support_dot_matches_convex_hull() {
        use crate::convex_hull::ConvexHull;

        let hb = HyperBox::<3>::new([2.0, 1.0, 3.0]);
        let ch = ConvexHull::<3>::hyperbox([2.0, 1.0, 3.0]);

        // Support points may differ when d[i]==0 (any sign is valid).
        // What matters: dot(support, direction) must be equal (both maximize it).
        let dirs = [
            SVector::from([1.0, 0.0, 0.0]),
            SVector::from([0.0, -1.0, 0.0]),
            SVector::from([1.0, 1.0, 1.0]),
            SVector::from([-0.5, 0.3, 0.8]),
            SVector::from([0.7, -0.7, 0.1]),
        ];
        for dir in &dirs {
            let dot_hb = hb.support(dir).dot(dir);
            let dot_ch = ch.support(dir).dot(dir);
            assert!(
                (dot_hb - dot_ch).abs() < 1e-10,
                "support·dir differs for {:?}: hb={dot_hb}, ch={dot_ch}",
                dir
            );
        }
    }

    #[test]
    fn bounding_sphere_radius() {
        let b = HyperBox::<3>::new([3.0, 4.0, 0.0]);
        let (_, radius) = b.bounding_sphere();
        assert!((radius - 5.0).abs() < 1e-12); // sqrt(9+16+0) = 5
    }

    #[test]
    fn volume() {
        let b = HyperBox::<3>::new([1.0, 2.0, 3.0]);
        assert!((b.volume() - 48.0).abs() < 1e-12); // 2*4*6
    }

    #[test]
    fn contains_local() {
        let b = HyperBox::<3>::cube(1.0);
        assert!(b.contains_local(&Point::origin()));
        assert!(b.contains_local(&Point::new([0.5, 0.5, 0.5])));
        assert!(!b.contains_local(&Point::new([1.5, 0.0, 0.0])));
    }

    #[test]
    fn tesseract_4d() {
        let b = HyperBox::<4>::cube(1.0);
        let dir = SVector::from([1.0, 1.0, 1.0, 1.0]);
        let sp = b.support(&dir);
        for i in 0..4 {
            assert!((sp[i] - 1.0).abs() < 1e-12);
        }
        let (_, radius) = b.bounding_sphere();
        assert!((radius - 2.0).abs() < 1e-12); // sqrt(4) = 2
    }

    #[test]
    fn hyperbox_2d() {
        let b = HyperBox::<2>::new([3.0, 1.0]);
        let dir = SVector::from([1.0, -1.0]);
        let sp = b.support(&dir);
        assert!((sp[0] - 3.0).abs() < 1e-12);
        assert!((sp[1] - (-1.0)).abs() < 1e-12);
    }

    #[test]
    fn cube_is_uniform() {
        let b = HyperBox::<3>::cube(2.5);
        assert!((b.half_extents[0] - 2.5).abs() < 1e-12);
        assert!((b.half_extents[1] - 2.5).abs() < 1e-12);
        assert!((b.half_extents[2] - 2.5).abs() < 1e-12);
    }
}
