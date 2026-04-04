// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Thin wrappers over terra-atlas-core picking,
//! converting local Vec3/Mat4 ↔ [f32; N] at the boundary.

use super::math::{Mat4, Vec3};

/// Unproject screen coordinates to a world-space ray.
/// Returns (ray_origin, ray_direction).
pub fn screen_to_ray(
    screen_x: f32,
    screen_y: f32,
    viewport_width: f32,
    viewport_height: f32,
    projection: &Mat4,
    view: &Mat4,
) -> Option<(Vec3, Vec3)> {
    let (origin, dir) = sol_atlas_core::picking::screen_to_ray(
        screen_x,
        screen_y,
        viewport_width,
        viewport_height,
        &projection.m,
        &view.m,
    )?;
    Some((Vec3::from(origin), Vec3::from(dir)))
}

/// Ray-sphere intersection. Returns the nearest hit point on the sphere, if any.
pub fn ray_sphere_intersect(
    ray_origin: Vec3,
    ray_dir: Vec3,
    sphere_center: Vec3,
    sphere_radius: f32,
) -> Option<Vec3> {
    sol_atlas_core::picking::ray_sphere_intersect(
        ray_origin.into(),
        ray_dir.into(),
        sphere_center.into(),
        sphere_radius,
    )
    .map(Vec3::from)
}

/// Find the nearest marker to a hit point on the sphere surface.
/// Returns the index of the nearest marker within `threshold` distance.
pub fn find_nearest_marker(
    hit_point: Vec3,
    marker_positions: &[Vec3],
    threshold: f32,
) -> Option<usize> {
    let positions: Vec<[f32; 3]> = marker_positions.iter().map(|v| (*v).into()).collect();
    sol_atlas_core::picking::find_nearest_marker(hit_point.into(), &positions, threshold)
}
