// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Camera zoom transition into a selected H3 cell — Step 2 of the "H3 Earth
//! / telemetry solar system / procedural galaxy" plan. Double-click a
//! hovered cell to smoothly zoom the orbital camera toward it.
//!
//! Loading a walkable ground-level scene once the transition completes
//! (Step 3) is a separate, later step — not yet designed.

use crate::camera::{OrbitalCamera, OrbitalCameraConfig};
use crate::h3_grid::HoveredCell;
use bevy::input::mouse::AccumulatedMouseScroll;
use bevy::prelude::*;
use h3o::{CellIndex, LatLng};
use std::f32::consts::{PI, TAU};

/// How close the camera gets when fully zoomed into a cell, as a multiple
/// of GLOBE_RADIUS. Deliberately below the normal scroll-zoom minimum
/// (`CAMERA_ZOOM_MIN` = 1.8) — meant to feel like approaching ground level.
const CELL_ZOOM_DISTANCE: f32 = 1.05;

/// Seconds for a full hover-to-zoomed-in transition.
const ZOOM_DURATION_SECS: f32 = 1.2;

/// Max gap (seconds) between clicks to count as a double-click.
const DOUBLE_CLICK_WINDOW_SECS: f32 = 0.35;

#[derive(Resource, Default)]
pub struct CellZoomTransition {
    pub target_cell: Option<CellIndex>,
    pub active: bool,
    pub progress: f32,
    start_theta: f32,
    start_phi: f32,
    start_distance: f32,
    target_theta: f32,
    target_phi: f32,
}

/// Converts a target lat/lon into the (theta, phi) the orbital camera needs
/// to look toward that point — inverse of `camera.rs`'s spherical eye
/// formula (`eye = r * (cos(phi)*sin(theta), sin(phi), cos(phi)*cos(theta))`).
fn latlng_to_camera_angles(ll: LatLng) -> (f32, f32) {
    let [x, y, z] = sol_atlas_core::geo::lat_lon_to_xyz(ll.lat(), ll.lng(), 1.0);
    let phi = y.clamp(-1.0, 1.0).asin();
    let theta = x.atan2(z);
    (theta, phi)
}

/// Shortest angular delta from `from` to `to`, avoiding the "long way
/// around" a periodic angle when the two are close to opposite sides of 0.
fn shortest_angle_delta(from: f32, to: f32) -> f32 {
    let diff = (to - from) % TAU;
    if diff > PI {
        diff - TAU
    } else if diff < -PI {
        diff + TAU
    } else {
        diff
    }
}

/// Double-click while hovering a cell starts (or restarts) the zoom-in
/// transition. Single-click is already used by
/// `selection::click_select_system` for markers, so double-click keeps the
/// two unambiguous.
pub fn trigger_cell_zoom_system(
    mouse_button: Res<ButtonInput<MouseButton>>,
    time: Res<Time>,
    hovered: Res<HoveredCell>,
    config: Res<OrbitalCameraConfig>,
    mut transition: ResMut<CellZoomTransition>,
    mut last_click: Local<f32>,
) {
    if !mouse_button.just_pressed(MouseButton::Left) {
        return;
    }
    let now = time.elapsed_secs();
    let is_double_click = now - *last_click < DOUBLE_CLICK_WINDOW_SECS;
    *last_click = now;

    if !is_double_click {
        return;
    }
    let Some(cell) = hovered.cell else {
        return;
    };

    let (target_theta, target_phi) = latlng_to_camera_angles(cell.into());
    transition.target_cell = Some(cell);
    transition.active = true;
    transition.progress = 0.0;
    transition.start_theta = config.theta;
    transition.start_phi = config.phi;
    transition.start_distance = config.distance;
    // Store as an absolute target reachable via the shortest angular path
    // from the current theta, so the per-frame lerp never sweeps the wrong
    // way around when start/target straddle the +-PI wraparound.
    transition.target_theta = config.theta + shortest_angle_delta(config.theta, target_theta);
    transition.target_phi = target_phi;
    info!("[atlas] Zooming into H3 cell {cell}");
}

/// Cancels an in-progress transition on manual drag/scroll input, so it
/// doesn't fight the user.
pub fn cancel_zoom_on_manual_input(
    mouse_button: Res<ButtonInput<MouseButton>>,
    accumulated_scroll: Res<AccumulatedMouseScroll>,
    mut transition: ResMut<CellZoomTransition>,
) {
    if !transition.active {
        return;
    }
    if mouse_button.pressed(MouseButton::Left) || accumulated_scroll.delta.y.abs() > 0.001 {
        transition.active = false;
    }
}

/// Advances the zoom transition and, while active, takes over the orbital
/// camera's theta/phi/distance. Registered after `orbital_camera_system` in
/// the schedule so it's the last writer each frame.
pub fn cell_zoom_transition_system(
    time: Res<Time>,
    mut transition: ResMut<CellZoomTransition>,
    mut config: ResMut<OrbitalCameraConfig>,
    mut query: Query<&mut Transform, With<OrbitalCamera>>,
) {
    if !transition.active {
        return;
    }
    let Some(cell) = transition.target_cell else {
        transition.active = false;
        return;
    };

    transition.progress = (transition.progress + time.delta_secs() / ZOOM_DURATION_SECS).min(1.0);
    let t = transition.progress;
    let eased = t * t * (3.0 - 2.0 * t); // smoothstep

    config.theta =
        transition.start_theta + (transition.target_theta - transition.start_theta) * eased;
    config.phi = transition.start_phi + (transition.target_phi - transition.start_phi) * eased;
    config.distance =
        transition.start_distance + (CELL_ZOOM_DISTANCE - transition.start_distance) * eased;

    let r = config.distance;
    let eye = Vec3::new(
        r * config.phi.cos() * config.theta.sin(),
        r * config.phi.sin(),
        r * config.phi.cos() * config.theta.cos(),
    );
    for mut tf in &mut query {
        *tf = Transform::from_translation(eye).looking_at(Vec3::ZERO, Vec3::Y);
    }

    if transition.progress >= 1.0 {
        transition.active = false;
        info!(
            "[atlas] Zoom transition complete — cell {cell} centered \
             (scale-transition to a walkable scene is a separate, later step)"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortest_delta_prefers_short_way_around() {
        // From just past +PI to just past -PI should be a tiny step, not a
        // near-full-circle sweep.
        let delta = shortest_angle_delta(3.13, -3.13);
        assert!(
            delta.abs() < 0.1,
            "delta was {delta}, expected a small step"
        );
    }

    #[test]
    fn shortest_delta_matches_direct_diff_for_nearby_angles() {
        let delta = shortest_angle_delta(0.2, 0.5);
        assert!((delta - 0.3).abs() < 1e-5);
    }

    #[test]
    fn latlng_angles_roundtrip_through_geo_xyz() {
        let ll = LatLng::new(10.0, 20.0).unwrap();
        let (theta, phi) = latlng_to_camera_angles(ll);
        // Reconstruct the orbital-camera eye direction and compare against
        // geo::lat_lon_to_xyz directly — they must describe the same point.
        let direction = Vec3::new(phi.cos() * theta.sin(), phi.sin(), phi.cos() * theta.cos());
        let [x, y, z] = sol_atlas_core::geo::lat_lon_to_xyz(10.0, 20.0, 1.0);
        let expected = Vec3::new(x, y, z);
        assert!(
            direction.distance(expected) < 1e-4,
            "direction {direction:?} != expected {expected:?}"
        );
    }
}
