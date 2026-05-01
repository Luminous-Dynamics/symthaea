// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! D-dimensional half-space (infinite plane with one solid side).
//!
//! Defined by a normal and offset: all points `p` where `normal · p <= offset`
//! are "inside" the solid region. The boundary plane is `normal · p = offset`.
//!
//! # GJK Compatibility
//!
//! Half-spaces are unbounded, so the support function returns a large finite
//! extent (`HALFSPACE_EXTENT = 1e6`) in the query direction. This works for
//! GJK but is imprecise. For sphere, capsule, and box shapes, use the
//! analytical contact functions instead — they're exact and faster.

use crate::point::Point;
use crate::shape::Shape;
use nalgebra::SVector;

/// Large finite extent for GJK compatibility (half-spaces are unbounded).
const HALFSPACE_EXTENT: f64 = 1e6;

/// D-dimensional half-space: the set of points where `normal · p <= offset`.
///
/// The boundary plane has equation `normal · p = offset`.
/// `normal` must be a unit vector.
#[derive(Clone, Copy, Debug)]
pub struct HalfSpace<const D: usize> {
    /// Outward-pointing unit normal of the boundary plane.
    pub normal: SVector<f64, D>,
    /// Signed distance from the origin to the boundary plane along the normal.
    /// Positive = plane is on the positive side of the origin.
    pub offset: f64,
}

impl<const D: usize> HalfSpace<D> {
    /// Create a half-space from a normal and offset.
    /// The normal should be a unit vector (not enforced, but expected).
    pub fn new(normal: SVector<f64, D>, offset: f64) -> Self {
        Self { normal, offset }
    }

    /// Create a ground plane at the given height along the given axis.
    /// Normal points in the positive axis direction (upward for Y-up).
    pub fn ground(axis: usize, height: f64) -> Self {
        let mut normal = SVector::zeros();
        normal[axis] = 1.0;
        Self {
            normal,
            offset: height,
        }
    }

    /// Signed distance from a point to the boundary plane.
    /// Positive = point is on the "outside" (normal side).
    /// Negative = point is inside the solid half-space.
    #[inline]
    pub fn signed_distance(&self, point: &SVector<f64, D>) -> f64 {
        self.normal.dot(point) - self.offset
    }

    /// Project a point onto the boundary plane.
    pub fn project(&self, point: &SVector<f64, D>) -> SVector<f64, D> {
        point - self.normal * self.signed_distance(point)
    }

    // ═══════════════════════════════════════════════════════════════════
    // Analytical contact functions — bypass GJK for exact, fast contacts
    // ═══════════════════════════════════════════════════════════════════

    /// Analytical contact between a sphere and this half-space.
    ///
    /// Returns `Some((contact_point, depth))` if overlapping, `None` otherwise.
    /// `sphere_center` is in world space.
    pub fn contact_sphere(
        &self,
        sphere_center: &SVector<f64, D>,
        sphere_radius: f64,
    ) -> Option<(SVector<f64, D>, f64)> {
        let dist = self.signed_distance(sphere_center);
        let depth = sphere_radius - dist;
        if depth <= 0.0 {
            return None;
        }
        // Contact point: closest point on sphere to the plane
        let contact = sphere_center - self.normal * dist;
        Some((contact, depth))
    }

    /// Analytical contact between a capsule and this half-space.
    ///
    /// Returns contacts for both hemisphere centers (0, 1, or 2 contacts).
    /// `capsule_pos` is the capsule's world-space center.
    pub fn contact_capsule(
        &self,
        capsule_pos: &SVector<f64, D>,
        half_height: f64,
        radius: f64,
        axis: usize,
    ) -> Vec<(SVector<f64, D>, f64)> {
        let mut contacts = Vec::new();

        // Two hemisphere centers
        let mut axis_vec = SVector::zeros();
        axis_vec[axis] = 1.0;

        let center_a = capsule_pos + axis_vec * half_height;
        let center_b = capsule_pos - axis_vec * half_height;

        if let Some(c) = self.contact_sphere(&center_a, radius) {
            contacts.push(c);
        }
        if let Some(c) = self.contact_sphere(&center_b, radius) {
            contacts.push(c);
        }

        contacts
    }

    /// Analytical contact between an axis-aligned box and this half-space.
    ///
    /// Returns contacts for all box vertices that penetrate the plane.
    /// `box_pos` is the box's world-space center.
    pub fn contact_box(
        &self,
        box_pos: &SVector<f64, D>,
        half_extents: &[f64; D],
    ) -> Vec<(SVector<f64, D>, f64)> {
        let mut contacts = Vec::new();

        // Enumerate all 2^D vertices (acceptable for D <= 4)
        let num_vertices = 1usize << D;
        for bits in 0..num_vertices {
            let mut vertex = *box_pos;
            for axis in 0..D {
                if bits & (1 << axis) != 0 {
                    vertex[axis] += half_extents[axis];
                } else {
                    vertex[axis] -= half_extents[axis];
                }
            }

            let dist = self.signed_distance(&vertex);
            if dist < 0.0 {
                let contact = self.project(&vertex);
                contacts.push((contact, -dist));
            }
        }

        contacts
    }
}

impl<const D: usize> Shape<D> for HalfSpace<D> {
    /// Support function for GJK compatibility.
    ///
    /// Returns a point at `HALFSPACE_EXTENT` in the query direction (projected
    /// onto the plane) or at `-HALFSPACE_EXTENT * normal` if direction points
    /// into the solid. This is an approximation — use analytical contacts
    /// for sphere/capsule/box instead.
    fn support(&self, direction: &SVector<f64, D>) -> SVector<f64, D> {
        let dot = direction.dot(&self.normal);
        if dot >= 0.0 {
            // Direction faces the normal — support is at the plane surface,
            // offset in the tangential direction
            let tangent = direction - self.normal * dot;
            let t_norm = tangent.norm();
            if t_norm > 1e-15 {
                self.normal * self.offset + tangent / t_norm * HALFSPACE_EXTENT
            } else {
                self.normal * self.offset
            }
        } else {
            // Direction faces into the solid — support is deep inside
            self.normal * (self.offset - HALFSPACE_EXTENT)
        }
    }

    fn bounding_sphere(&self) -> (Point<D>, f64) {
        (Point::origin(), HALFSPACE_EXTENT)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn signed_distance_above() {
        let plane = HalfSpace::<3>::ground(1, 0.0); // Y=0 plane
        let point = SVector::from([0.0, 5.0, 0.0]);
        assert!((plane.signed_distance(&point) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn signed_distance_below() {
        let plane = HalfSpace::<3>::ground(1, 0.0);
        let point = SVector::from([0.0, -3.0, 0.0]);
        assert!((plane.signed_distance(&point) - (-3.0)).abs() < 1e-12);
    }

    #[test]
    fn project_onto_plane() {
        let plane = HalfSpace::<3>::ground(1, 0.0);
        let point = SVector::from([3.0, 5.0, 7.0]);
        let proj = plane.project(&point);
        assert!((proj[0] - 3.0).abs() < 1e-12);
        assert!((proj[1] - 0.0).abs() < 1e-12);
        assert!((proj[2] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn sphere_contact_resting() {
        let plane = HalfSpace::<3>::ground(1, 0.0);
        let center = SVector::from([0.0, 0.5, 0.0]); // half-embedded
        let (contact, depth) = plane.contact_sphere(&center, 1.0).unwrap();
        assert!((depth - 0.5).abs() < 1e-12, "depth = {depth}");
        assert!(
            (contact[1] - 0.0).abs() < 1e-12,
            "contact Y = {}",
            contact[1]
        );
    }

    #[test]
    fn sphere_no_contact() {
        let plane = HalfSpace::<3>::ground(1, 0.0);
        let center = SVector::from([0.0, 2.0, 0.0]);
        assert!(plane.contact_sphere(&center, 1.0).is_none());
    }

    #[test]
    fn capsule_two_contacts() {
        let plane = HalfSpace::<3>::ground(1, 0.0);
        // Capsule lying on its side (X-aligned), center at Y=0.3
        let pos = SVector::from([0.0, 0.3, 0.0]);
        let contacts = plane.contact_capsule(&pos, 2.0, 0.5, 0);
        // Both hemisphere centers at Y=0.3, radius 0.5 → both penetrate by 0.2
        assert_eq!(contacts.len(), 2);
        for (_, depth) in &contacts {
            assert!(
                (depth - 0.2).abs() < 1e-10,
                "capsule contact depth = {depth}"
            );
        }
    }

    #[test]
    fn box_contact_on_plane() {
        let plane = HalfSpace::<3>::ground(1, 0.0);
        // Unit box centered at Y=0.5 → bottom face at Y=-0.5 → 4 vertices penetrate
        let pos = SVector::from([0.0, 0.5, 0.0]);
        let contacts = plane.contact_box(&pos, &[1.0, 1.0, 1.0]);
        // Bottom 4 vertices (Y = 0.5-1.0 = -0.5) penetrate by 0.5
        assert_eq!(contacts.len(), 4, "expected 4 bottom vertices to penetrate");
        for (_, depth) in &contacts {
            assert!((depth - 0.5).abs() < 1e-10, "box vertex depth = {depth}");
        }
    }

    #[test]
    fn box_no_contact() {
        let plane = HalfSpace::<3>::ground(1, 0.0);
        let pos = SVector::from([0.0, 5.0, 0.0]);
        let contacts = plane.contact_box(&pos, &[1.0, 1.0, 1.0]);
        assert!(contacts.is_empty());
    }

    #[test]
    fn halfspace_4d() {
        let plane = HalfSpace::<4>::ground(3, 0.0); // W=0 plane
        let center = SVector::from([0.0, 0.0, 0.0, 0.8]);
        let contact = plane.contact_sphere(&center, 1.0);
        assert!(contact.is_some());
        let (_, depth) = contact.unwrap();
        assert!((depth - 0.2).abs() < 1e-12, "4D depth = {depth}");
    }

    #[test]
    fn halfspace_offset() {
        let plane = HalfSpace::<3>::ground(1, 2.0); // Y=2 plane
        let center = SVector::from([0.0, 2.5, 0.0]);
        let contact = plane.contact_sphere(&center, 1.0);
        assert!(contact.is_some());
        let (_, depth) = contact.unwrap();
        assert!((depth - 0.5).abs() < 1e-12, "offset plane depth = {depth}");
    }

    #[test]
    fn halfspace_2d() {
        let plane = HalfSpace::<2>::ground(1, 0.0);
        let center = SVector::from([3.0, 0.5]);
        let (contact, depth) = plane.contact_sphere(&center, 1.0).unwrap();
        assert!((depth - 0.5).abs() < 1e-12);
        assert!((contact[0] - 3.0).abs() < 1e-12);
    }
}
