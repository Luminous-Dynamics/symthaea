// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: Apache-2.0 OR MIT
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Projection from ND physics space to screen/render coordinates.

use nalgebra::SVector;
use symtropy_math::{Hyperplane, Point};

/// Trait for projecting ND points to Bevy-compatible 2D/3D coordinates.
pub trait Projector<const D: usize>: Send + Sync {
    /// Project an ND point to a 3D position (x, y, z) for Bevy rendering.
    /// Returns (x, y, z) — for 2D rendering, z is used for sprite ordering.
    fn project(&self, point: &Point<D>) -> [f32; 3];

    /// Whether an ND point is visible (within the projection's view).
    fn is_visible(&self, point: &Point<D>) -> bool;

    /// Scale factor: how many render pixels per physics unit.
    fn scale(&self) -> f32;
}

/// 2D projector: direct 1:1 mapping from physics (x, y) to Bevy (x, y, 0).
pub struct Projector2D {
    /// Pixels per physics unit.
    pub pixels_per_unit: f32,
    /// Camera center in physics space.
    pub camera_x: f64,
    pub camera_y: f64,
}

impl Projector2D {
    pub fn new(pixels_per_unit: f32) -> Self {
        Self {
            pixels_per_unit,
            camera_x: 0.0,
            camera_y: 0.0,
        }
    }
}

impl Projector<2> for Projector2D {
    fn project(&self, point: &Point<2>) -> [f32; 3] {
        let x = (point.coord(0) - self.camera_x) as f32 * self.pixels_per_unit;
        let y = (point.coord(1) - self.camera_y) as f32 * self.pixels_per_unit;
        [x, y, 0.0]
    }

    fn is_visible(&self, _point: &Point<2>) -> bool {
        true // 2D is always visible (camera culling handled by Bevy)
    }

    fn scale(&self) -> f32 {
        self.pixels_per_unit
    }
}

/// 3D projector: direct mapping from physics (x, y, z) to Bevy (x, y, z).
pub struct Projector3D {
    pub scale_factor: f32,
}

impl Projector3D {
    pub fn new(scale_factor: f32) -> Self {
        Self { scale_factor }
    }
}

impl Projector<3> for Projector3D {
    fn project(&self, point: &Point<3>) -> [f32; 3] {
        [
            point.coord(0) as f32 * self.scale_factor,
            point.coord(1) as f32 * self.scale_factor,
            point.coord(2) as f32 * self.scale_factor,
        ]
    }

    fn is_visible(&self, _point: &Point<3>) -> bool {
        true // Bevy handles 3D frustum culling
    }

    fn scale(&self) -> f32 {
        self.scale_factor
    }
}

/// 4D projector: cross-section slicing.
///
/// Slices the 4D world with a 3D hyperplane perpendicular to the W axis.
/// Objects that intersect the slice appear as their 3D cross-sections.
/// Moving the slice position along W reveals hidden geometry.
pub struct Projector4D {
    /// Position of the slicing hyperplane along the W axis.
    pub w_slice: f64,
    /// Thickness of the slice (objects within this distance of the hyperplane are visible).
    pub slice_thickness: f64,
    /// Scale factor for rendering.
    pub scale_factor: f32,
}

impl Projector4D {
    pub fn new(w_slice: f64, slice_thickness: f64, scale_factor: f32) -> Self {
        Self {
            w_slice,
            slice_thickness,
            scale_factor,
        }
    }

    /// The slicing hyperplane: { (x,y,z,w) : w = w_slice }.
    pub fn slice_plane(&self) -> Hyperplane<4> {
        Hyperplane::new(SVector::from([0.0, 0.0, 0.0, 1.0]), self.w_slice)
    }

    /// Distance of a 4D point from the slicing hyperplane (along W).
    pub fn w_distance(&self, point: &Point<4>) -> f64 {
        (point.coord(3) - self.w_slice).abs()
    }

    /// Alpha (opacity) based on distance from slice plane.
    /// Objects exactly on the plane are fully opaque; objects at the edge
    /// of the slice thickness fade to transparent.
    pub fn alpha(&self, point: &Point<4>) -> f32 {
        let dist = self.w_distance(point);
        if dist >= self.slice_thickness {
            return 0.0;
        }
        (1.0 - dist / self.slice_thickness) as f32
    }
}

impl Projector<4> for Projector4D {
    fn project(&self, point: &Point<4>) -> [f32; 3] {
        // Project (x, y, z, w) → (x, y, z), discarding w
        [
            point.coord(0) as f32 * self.scale_factor,
            point.coord(1) as f32 * self.scale_factor,
            point.coord(2) as f32 * self.scale_factor,
        ]
    }

    fn is_visible(&self, point: &Point<4>) -> bool {
        self.w_distance(point) < self.slice_thickness
    }

    fn scale(&self) -> f32 {
        self.scale_factor
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projector_2d_basic() {
        let proj = Projector2D::new(32.0);
        let p = Point::new([1.0, 2.0]);
        let [x, y, z] = proj.project(&p);
        assert!((x - 32.0).abs() < 1e-5);
        assert!((y - 64.0).abs() < 1e-5);
        assert!((z - 0.0).abs() < 1e-5);
    }

    #[test]
    fn projector_2d_with_camera_offset() {
        let mut proj = Projector2D::new(32.0);
        proj.camera_x = 5.0;
        proj.camera_y = 10.0;
        let p = Point::new([6.0, 12.0]);
        let [x, y, _] = proj.project(&p);
        assert!((x - 32.0).abs() < 1e-5); // (6-5) * 32
        assert!((y - 64.0).abs() < 1e-5); // (12-10) * 32
    }

    #[test]
    fn projector_3d_basic() {
        let proj = Projector3D::new(1.0);
        let p = Point::new([1.0, 2.0, 3.0]);
        let [x, y, z] = proj.project(&p);
        assert!((x - 1.0).abs() < 1e-5);
        assert!((y - 2.0).abs() < 1e-5);
        assert!((z - 3.0).abs() < 1e-5);
    }

    #[test]
    fn projector_4d_on_slice_visible() {
        let proj = Projector4D::new(5.0, 1.0, 1.0);
        let p = Point::new([1.0, 2.0, 3.0, 5.0]); // w = 5.0, exactly on slice
        assert!(proj.is_visible(&p));
        assert!((proj.alpha(&p) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn projector_4d_off_slice_invisible() {
        let proj = Projector4D::new(5.0, 1.0, 1.0);
        let p = Point::new([1.0, 2.0, 3.0, 10.0]); // w = 10.0, far from slice
        assert!(!proj.is_visible(&p));
        assert!((proj.alpha(&p) - 0.0).abs() < 1e-5);
    }

    #[test]
    fn projector_4d_edge_fades() {
        let proj = Projector4D::new(5.0, 2.0, 1.0);
        let p = Point::new([1.0, 2.0, 3.0, 6.0]); // w = 6.0, distance 1.0 from slice
        assert!(proj.is_visible(&p));
        // Alpha = 1.0 - 1.0/2.0 = 0.5
        assert!((proj.alpha(&p) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn projector_4d_projects_xyz() {
        let proj = Projector4D::new(0.0, 1.0, 2.0);
        let p = Point::new([1.0, 2.0, 3.0, 0.5]);
        let [x, y, z] = proj.project(&p);
        assert!((x - 2.0).abs() < 1e-5); // 1.0 * 2.0
        assert!((y - 4.0).abs() < 1e-5); // 2.0 * 2.0
        assert!((z - 6.0).abs() < 1e-5); // 3.0 * 2.0
    }

    #[test]
    fn projector_4d_moving_slice_reveals() {
        let mut proj = Projector4D::new(0.0, 0.5, 1.0);

        // Hidden room at w=5.0
        let hidden = Point::new([1.0, 2.0, 3.0, 5.0]);

        // Not visible from w=0.0
        assert!(!proj.is_visible(&hidden));

        // Move slice to w=5.0 — now visible
        proj.w_slice = 5.0;
        assert!(proj.is_visible(&hidden));
    }

    #[test]
    fn slice_plane_correct() {
        let proj = Projector4D::new(3.0, 1.0, 1.0);
        let plane = proj.slice_plane();
        // Point at w=3.0 should be on the plane
        let on = Point::new([0.0, 0.0, 0.0, 3.0]);
        assert!(plane.signed_distance(&on).abs() < 1e-10);
    }
}
