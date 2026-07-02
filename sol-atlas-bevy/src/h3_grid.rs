// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! H3 hexagonal grid integration — Step 1 of the "H3 hex Earth" plan
//! (H3 for Earth, telemetry for the solar system, procedural for the rest
//! of the galaxy). This module only covers cell indexing, picking, and
//! boundary rendering; scale-transition into a walkable ground-level scene
//! is a separate, later step.

use crate::camera::OrbitalCamera;
use crate::globe::Globe;
use bevy::prelude::*;
use h3o::{CellIndex, LatLng, Resolution};
use sol_atlas_core::constants::GLOBE_RADIUS;
use sol_atlas_core::geo;

/// Default H3 resolution for cell picking — coarse enough (~5,882 cells
/// globally) that hexagons read as visible chunks of the globe rather than
/// vanishing at planet-view zoom levels. Finer resolutions make sense once
/// there's a zoom-dependent LOD story; not needed for this first pass.
pub const DEFAULT_RESOLUTION: Resolution = Resolution::Two;

/// The H3 cell currently under the cursor (if any), plus the resolution
/// picking operates at. A plain hover-highlight for now — click-to-enter
/// (the "small test level inside a cell" step) is future work.
#[derive(Resource)]
pub struct HoveredCell {
    pub cell: Option<CellIndex>,
    pub resolution: Resolution,
}

impl Default for HoveredCell {
    fn default() -> Self {
        Self {
            cell: None,
            resolution: DEFAULT_RESOLUTION,
        }
    }
}

/// Convert lat/lon (degrees) to the containing H3 cell at the given resolution.
/// Returns `None` only for non-finite input (h3o's own validity check).
pub fn latlon_to_cell(lat: f64, lon: f64, resolution: Resolution) -> Option<CellIndex> {
    LatLng::new(lat, lon).ok().map(|ll| ll.to_cell(resolution))
}

/// The cell's boundary vertices projected onto a sphere of the given radius,
/// in the same coordinate convention as `sol_atlas_core::geo::lat_lon_to_xyz`
/// (Y-up, matching the existing globe/marker code).
pub fn cell_boundary_xyz(cell: CellIndex, radius: f64) -> Vec<Vec3> {
    cell.boundary()
        .iter()
        .map(|ll| {
            let [x, y, z] = geo::lat_lon_to_xyz(ll.lat(), ll.lng(), radius);
            Vec3::new(x, y, z)
        })
        .collect()
}

/// Ray/unit-sphere intersection. Returns the nearest hit point in front of
/// the ray origin, or `None` if the ray misses the sphere entirely.
fn ray_sphere_hit(ray: Ray3d, radius: f32) -> Option<Vec3> {
    let origin = ray.origin;
    let dir = *ray.direction;
    let b = 2.0 * origin.dot(dir);
    let c = origin.dot(origin) - radius * radius;
    let discriminant = b * b - 4.0 * c;
    if discriminant < 0.0 {
        return None;
    }
    let sqrt_d = discriminant.sqrt();
    let t_near = (-b - sqrt_d) / 2.0;
    let t_far = (-b + sqrt_d) / 2.0;
    let t = if t_near >= 0.0 {
        t_near
    } else if t_far >= 0.0 {
        t_far
    } else {
        return None;
    };
    Some(origin + dir * t)
}

/// Raycasts from the cursor to the globe surface (radius `GLOBE_RADIUS`,
/// matching `globe::spawn_globe`'s earth mesh) each frame and updates
/// `HoveredCell` with the H3 cell under the cursor. Mirrors the
/// manual-raycast pattern already used in `selection.rs`'s
/// `click_select_system`.
pub fn hover_cell_system(
    windows: Query<&Window>,
    camera_q: Query<(&Camera, &GlobalTransform), With<OrbitalCamera>>,
    globe_q: Query<&GlobalTransform, With<Globe>>,
    mut hovered: ResMut<HoveredCell>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let Some(cursor_pos) = window.cursor_position() else {
        hovered.cell = None;
        return;
    };
    let Ok((camera, camera_tf)) = camera_q.single() else {
        return;
    };
    let Ok(ray) = camera.viewport_to_world(camera_tf, cursor_pos) else {
        return;
    };

    // Globe is spawned at the origin at unit scale (see globe::spawn_globe);
    // ray_sphere_hit assumes a unit sphere at the origin, so this only needs
    // to confirm a Globe entity exists rather than transform the ray by it.
    if globe_q.single().is_err() {
        return;
    }

    let Some(hit) = ray_sphere_hit(ray, GLOBE_RADIUS) else {
        hovered.cell = None;
        return;
    };

    let (lat, lon) = geo::xyz_to_lat_lon([hit.x, hit.y, hit.z], GLOBE_RADIUS as f64);
    hovered.cell = latlon_to_cell(lat, lon, hovered.resolution);
}

/// Draws the hovered cell's hexagon boundary on the globe via gizmos —
/// cheap, no mesh/material needed, matches the existing shipping-lane
/// gizmo-line rendering pattern in `atlas.rs`.
pub fn draw_hovered_cell_system(hovered: Res<HoveredCell>, mut gizmos: Gizmos) {
    let Some(cell) = hovered.cell else {
        return;
    };
    // Slightly above the globe surface so the outline doesn't z-fight with it.
    let boundary = cell_boundary_xyz(cell, (GLOBE_RADIUS * 1.01) as f64);

    for i in 0..boundary.len() {
        let a = boundary[i];
        let b = boundary[(i + 1) % boundary.len()];
        gizmos.line(a, b, Color::srgb(1.0, 0.85, 0.2));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_point_resolves_to_a_cell() {
        // San Francisco, roughly.
        let cell = latlon_to_cell(37.7749, -122.4194, Resolution::Two);
        assert!(cell.is_some());
    }

    #[test]
    fn invalid_coordinates_return_none() {
        assert!(latlon_to_cell(f64::NAN, 0.0, Resolution::Two).is_none());
    }

    #[test]
    fn boundary_has_five_or_six_vertices() {
        let cell = latlon_to_cell(37.7749, -122.4194, Resolution::Two).unwrap();
        let boundary = cell_boundary_xyz(cell, 1.0);
        // H3 cells are hexagons (6 vertices) except 12 pentagons per resolution.
        assert!(boundary.len() == 5 || boundary.len() == 6);
    }

    #[test]
    fn boundary_vertices_lie_on_the_sphere() {
        let cell = latlon_to_cell(37.7749, -122.4194, Resolution::Two).unwrap();
        let boundary = cell_boundary_xyz(cell, 1.0);
        for v in boundary {
            assert!(
                (v.length() - 1.0).abs() < 1e-4,
                "vertex not on unit sphere: {v:?}"
            );
        }
    }

    #[test]
    fn ray_hits_unit_sphere_head_on() {
        let ray = Ray3d::new(Vec3::new(0.0, 0.0, 5.0), Dir3::NEG_Z);
        let hit = ray_sphere_hit(ray, 1.0);
        assert!(hit.is_some());
        let hit = hit.unwrap();
        assert!((hit.length() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn ray_misses_unit_sphere() {
        let ray = Ray3d::new(Vec3::new(5.0, 5.0, 5.0), Dir3::NEG_Z);
        assert!(ray_sphere_hit(ray, 1.0).is_none());
    }
}
