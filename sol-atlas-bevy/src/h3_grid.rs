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

/// Default H3 resolution for cell picking, used only before the camera
/// distance is known (first frame). `resolution_for_distance` takes over
/// from there — resolution now tracks zoom level continuously rather than
/// being fixed, so hexagons always read as a sensible-sized selection area
/// relative to what's on screen instead of spanning most of the visible
/// globe at close zoom or vanishing to a speck at orbital distance.
pub const DEFAULT_RESOLUTION: Resolution = Resolution::Two;

/// Camera distance (globe radii) at/beyond which hex resolution bottoms
/// out at its coarsest — matches `CAMERA_ZOOM_MAX`.
const RESOLUTION_FAR_DISTANCE: f32 = 8.0;
/// Camera distance at/within which hex resolution caps at its finest —
/// matches `cell_entry`'s close-in zoom target (kept as a plain constant
/// here rather than importing from `cell_entry`, to avoid a dependency
/// between the two — both just need to agree on roughly the same number).
const RESOLUTION_NEAR_DISTANCE: f32 = 1.05;
/// Coarsest resolution used at `RESOLUTION_FAR_DISTANCE` and beyond.
const RESOLUTION_MIN: u8 = 0;
/// Finest resolution used at `RESOLUTION_NEAR_DISTANCE` and closer. Capped
/// well below H3's max (15) — past this a hex is smaller than a single
/// screen pixel would resolve at any distance this globe is ever rendered
/// from, so finer levels would just waste picking precision for nothing
/// visible.
const RESOLUTION_MAX: u8 = 9;

/// Maps camera distance to H3 resolution — coarse (whole-region hexes)
/// from orbit, progressively finer as the camera approaches the surface.
/// Continuous in camera distance, not tied to a separate discrete "zoom
/// level" event, so drilling in via repeated clicks (which move the camera
/// closer) naturally reveals finer hexes with no resolution "pop"
/// independent of the zoom motion itself.
pub fn resolution_for_distance(distance: f32) -> Resolution {
    let t = ((RESOLUTION_FAR_DISTANCE - distance)
        / (RESOLUTION_FAR_DISTANCE - RESOLUTION_NEAR_DISTANCE))
        .clamp(0.0, 1.0);
    let level = RESOLUTION_MIN as f32 + t * (RESOLUTION_MAX - RESOLUTION_MIN) as f32;
    Resolution::try_from(level.round() as u8).unwrap_or(Resolution::Zero)
}

/// The H3 cell currently under the cursor (if any), plus the resolution
/// picking currently operates at (recomputed from camera distance every
/// frame by `hover_cell_system`).
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

    // Read the camera's actual rendered distance (not a config resource)
    // so this stays correct even mid-transition, when cell_entry's zoom
    // system is writing the Transform directly.
    hovered.resolution = resolution_for_distance(camera_tf.translation().length());

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
    fn resolution_coarsens_with_distance() {
        let far = resolution_for_distance(RESOLUTION_FAR_DISTANCE);
        let mid =
            resolution_for_distance((RESOLUTION_FAR_DISTANCE + RESOLUTION_NEAR_DISTANCE) / 2.0);
        let near = resolution_for_distance(RESOLUTION_NEAR_DISTANCE);
        assert!(u8::from(far) < u8::from(mid));
        assert!(u8::from(mid) < u8::from(near));
    }

    #[test]
    fn resolution_clamps_beyond_the_defined_range() {
        assert_eq!(
            resolution_for_distance(RESOLUTION_FAR_DISTANCE * 10.0),
            Resolution::try_from(RESOLUTION_MIN).unwrap()
        );
        assert_eq!(
            resolution_for_distance(0.0),
            Resolution::try_from(RESOLUTION_MAX).unwrap()
        );
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
