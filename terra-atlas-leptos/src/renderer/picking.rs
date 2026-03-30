// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::math::{Mat4, Vec3, Vec4};

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
    // Normalized device coordinates
    let ndc_x = (2.0 * screen_x / viewport_width) - 1.0;
    let ndc_y = 1.0 - (2.0 * screen_y / viewport_height); // flip Y

    let inv_proj = projection.inverse()?;
    let inv_view = view.inverse()?;

    // Near point in clip space → eye space → world space
    let near_clip = Vec4::new(ndc_x, ndc_y, -1.0, 1.0);
    let near_eye = inv_proj.transform_vec4(near_clip);
    let near_eye = Vec4::new(near_eye.x / near_eye.w, near_eye.y / near_eye.w, near_eye.z / near_eye.w, 1.0);
    let near_world = inv_view.transform_vec4(near_eye);

    let far_clip = Vec4::new(ndc_x, ndc_y, 1.0, 1.0);
    let far_eye = inv_proj.transform_vec4(far_clip);
    let far_eye = Vec4::new(far_eye.x / far_eye.w, far_eye.y / far_eye.w, far_eye.z / far_eye.w, 1.0);
    let far_world = inv_view.transform_vec4(far_eye);

    let origin = near_world.xyz();
    let direction = (far_world.xyz() - origin).normalize();

    Some((origin, direction))
}

/// Ray-sphere intersection. Returns the nearest hit point on the sphere, if any.
pub fn ray_sphere_intersect(
    ray_origin: Vec3,
    ray_dir: Vec3,
    sphere_center: Vec3,
    sphere_radius: f32,
) -> Option<Vec3> {
    let oc = ray_origin - sphere_center;
    let a = ray_dir.dot(ray_dir);
    let b = 2.0 * oc.dot(ray_dir);
    let c = oc.dot(oc) - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return None;
    }

    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);

    let t = if t1 > 0.0 { t1 } else if t2 > 0.0 { t2 } else { return None };

    Some(ray_origin + ray_dir * t)
}

/// Find the nearest marker to a hit point on the sphere surface.
/// Returns the index of the nearest marker within `threshold` distance.
pub fn find_nearest_marker(
    hit_point: Vec3,
    marker_positions: &[Vec3],
    threshold: f32,
) -> Option<usize> {
    let mut best_idx = None;
    let mut best_dist_sq = threshold * threshold;

    for (i, pos) in marker_positions.iter().enumerate() {
        let dist_sq = (*pos - hit_point).length_sq();
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_idx = Some(i);
        }
    }

    best_idx
}
