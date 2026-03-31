// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ray casting and hit-testing for globe interaction.

use crate::math::*;

/// Unproject screen coordinates to a world-space ray.
/// Returns `(ray_origin, ray_direction)`.
pub fn screen_to_ray(
    screen_x: f32,
    screen_y: f32,
    viewport_width: f32,
    viewport_height: f32,
    projection: &[[f32; 4]; 4],
    view: &[[f32; 4]; 4],
) -> Option<([f32; 3], [f32; 3])> {
    let ndc_x = (2.0 * screen_x / viewport_width) - 1.0;
    let ndc_y = 1.0 - (2.0 * screen_y / viewport_height);

    let inv_proj = mat4_inverse(*projection)?;
    let inv_view = mat4_inverse(*view)?;

    let near_clip = [ndc_x, ndc_y, -1.0, 1.0];
    let near_eye = mat4_transform_vec4(inv_proj, near_clip);
    let near_eye = [
        near_eye[0] / near_eye[3],
        near_eye[1] / near_eye[3],
        near_eye[2] / near_eye[3],
        1.0,
    ];
    let near_world = mat4_transform_vec4(inv_view, near_eye);

    let far_clip = [ndc_x, ndc_y, 1.0, 1.0];
    let far_eye = mat4_transform_vec4(inv_proj, far_clip);
    let far_eye = [
        far_eye[0] / far_eye[3],
        far_eye[1] / far_eye[3],
        far_eye[2] / far_eye[3],
        1.0,
    ];
    let far_world = mat4_transform_vec4(inv_view, far_eye);

    let origin = vec4_xyz(near_world);
    let direction = vec3_normalize(vec3_sub(vec4_xyz(far_world), origin));

    Some((origin, direction))
}

/// Ray-sphere intersection. Returns the nearest hit point on the sphere, if any.
pub fn ray_sphere_intersect(
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    sphere_center: [f32; 3],
    sphere_radius: f32,
) -> Option<[f32; 3]> {
    let oc = vec3_sub(ray_origin, sphere_center);
    let a = vec3_dot(ray_dir, ray_dir);
    let b = 2.0 * vec3_dot(oc, ray_dir);
    let c = vec3_dot(oc, oc) - sphere_radius * sphere_radius;
    let discriminant = b * b - 4.0 * a * c;

    if discriminant < 0.0 {
        return None;
    }

    let sqrt_disc = discriminant.sqrt();
    let t1 = (-b - sqrt_disc) / (2.0 * a);
    let t2 = (-b + sqrt_disc) / (2.0 * a);

    let t = if t1 > 0.0 {
        t1
    } else if t2 > 0.0 {
        t2
    } else {
        return None;
    };

    Some(vec3_add(ray_origin, vec3_scale(ray_dir, t)))
}

/// Find the nearest marker to a hit point on the sphere surface.
/// Returns the index of the nearest marker within `threshold` distance.
pub fn find_nearest_marker(
    hit_point: [f32; 3],
    marker_positions: &[[f32; 3]],
    threshold: f32,
) -> Option<usize> {
    let mut best_idx = None;
    let mut best_dist_sq = threshold * threshold;

    for (i, pos) in marker_positions.iter().enumerate() {
        let dist_sq = vec3_length_sq(vec3_sub(*pos, hit_point));
        if dist_sq < best_dist_sq {
            best_dist_sq = dist_sq;
            best_idx = Some(i);
        }
    }

    best_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ray_sphere_hit() {
        let origin = [0.0, 0.0, 5.0];
        let dir = [0.0, 0.0, -1.0];
        let hit = ray_sphere_intersect(origin, dir, [0.0; 3], 1.0);
        assert!(hit.is_some());
        let p = hit.unwrap();
        assert!((p[2] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_ray_sphere_miss() {
        let origin = [0.0, 5.0, 5.0];
        let dir = [0.0, 0.0, -1.0];
        let hit = ray_sphere_intersect(origin, dir, [0.0; 3], 1.0);
        assert!(hit.is_none());
    }

    #[test]
    fn test_find_nearest_marker() {
        let markers = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let hit = [0.1, 0.9, 0.0];
        let idx = find_nearest_marker(hit, &markers, 0.5);
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_find_nearest_marker_none() {
        let markers = [[1.0, 0.0, 0.0]];
        let hit = [0.0, 0.0, 1.0];
        let idx = find_nearest_marker(hit, &markers, 0.1);
        assert_eq!(idx, None);
    }
}
