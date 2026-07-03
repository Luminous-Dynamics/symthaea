// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Camera zoom transition into a selected H3 cell — Step 2 of the "H3 Earth
//! / telemetry solar system / procedural galaxy" plan. Click a hovered cell
//! to smoothly zoom the orbital camera toward it; each click drills closer
//! (progressively shrinking the target distance), and since H3 resolution
//! is derived continuously from camera distance (`h3_grid::resolution_for_distance`),
//! finer hexes appear naturally as you approach rather than popping at a
//! separate, disconnected trigger.
//!
//! Loading a walkable ground-level scene once you're fully zoomed in
//! (Step 3) is a separate, later step — not yet designed.

use crate::camera::{OrbitalCamera, OrbitalCameraConfig};
use crate::h3_grid::HoveredCell;
use bevy::input::mouse::AccumulatedMouseScroll;
use bevy::prelude::*;
use h3o::{CellIndex, LatLng};
use sol_atlas_core::constants::GLOBE_RADIUS;
use std::f32::consts::{PI, TAU};

/// Floor distance a drill-down click can reach, as a multiple of
/// GLOBE_RADIUS. Deliberately below the normal scroll-zoom minimum
/// (`CAMERA_ZOOM_MIN` = 1.8) — meant to feel like approaching ground level.
const CELL_ZOOM_FLOOR: f32 = GLOBE_RADIUS * 1.05;

/// Screen-pixel movement threshold between mouse-down and mouse-up to
/// still count as a click rather than a camera-rotate drag.
const CLICK_MOVEMENT_THRESHOLD_PX: f32 = 6.0;

/// Each click zooms to this fraction of the *current* distance to the
/// target cell, not straight to the floor — so repeated clicks drill in
/// progressively (each one noticeably closer, and via the resolution
/// mapping, each one revealing finer hexes) rather than a single jump.
const ZOOM_STEP_FACTOR: f32 = 0.4;

/// Seconds for a full hover-to-zoomed-in transition.
const ZOOM_DURATION_SECS: f32 = 1.1;

#[derive(Resource, Default)]
pub struct CellZoomTransition {
    pub target_cell: Option<CellIndex>,
    pub active: bool,
    pub progress: f32,
    /// Set once, for one frame, when a transition completes at
    /// `CELL_ZOOM_FLOOR` (i.e. you've drilled all the way in, not just one
    /// step closer). `sol-atlas-bevy` can't reference symtropy's
    /// `GamePhase` (separate crate, no dependency between them) — this is
    /// the hand-off point: a symtropy-side system watches this field and
    /// decides what "arriving" means (e.g. entering a walkable view).
    /// Consumers should `.take()` it so it only fires once per arrival.
    pub arrived_at_floor: Option<CellIndex>,
    start_theta: f32,
    start_phi: f32,
    start_distance: f32,
    target_theta: f32,
    target_phi: f32,
    target_distance: f32,
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

/// Next drill-down target distance: a fixed fraction of the current
/// distance, floored so repeated clicks converge toward — but never
/// below — `CELL_ZOOM_FLOOR` instead of overshooting through the surface.
fn next_zoom_distance(current: f32) -> f32 {
    (current * ZOOM_STEP_FACTOR).max(CELL_ZOOM_FLOOR)
}

/// Click a hovered cell to drill into it: zooms the camera to
/// `ZOOM_STEP_FACTOR` of the current distance (floored at
/// `CELL_ZOOM_FLOOR`), which — via `h3_grid::resolution_for_distance` —
/// naturally reveals finer hexes on arrival. Click again on the newly
/// (smaller) hovered cell to drill further.
///
/// Distinguishes a click from a camera-rotate drag by cursor movement
/// between press and release, since both use the same left mouse button.
pub fn trigger_cell_zoom_system(
    mouse_button: Res<ButtonInput<MouseButton>>,
    windows: Query<&Window>,
    hovered: Res<HoveredCell>,
    config: Res<OrbitalCameraConfig>,
    mut transition: ResMut<CellZoomTransition>,
    mut press_pos: Local<Option<Vec2>>,
) {
    let Ok(window) = windows.single() else {
        return;
    };

    if mouse_button.just_pressed(MouseButton::Left) {
        *press_pos = window.cursor_position();
        return;
    }

    if !mouse_button.just_released(MouseButton::Left) {
        return;
    }

    let Some(start) = press_pos.take() else {
        return;
    };
    let Some(end) = window.cursor_position() else {
        return;
    };
    if start.distance(end) > CLICK_MOVEMENT_THRESHOLD_PX {
        return; // was a drag, not a click
    }

    let Some(cell) = hovered.cell else {
        return;
    };

    let (target_theta, target_phi) = latlng_to_camera_angles(cell.into());
    let target_distance = next_zoom_distance(config.distance);

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
    transition.target_distance = target_distance;
    info!(
        "[atlas] Drilling into H3 cell {cell} ({:.2} -> {:.2} globe radii)",
        config.distance, target_distance
    );
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
    config.distance = transition.start_distance
        + (transition.target_distance - transition.start_distance) * eased;

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
        let at_floor = (transition.target_distance - CELL_ZOOM_FLOOR).abs() < 1e-3;
        if at_floor {
            transition.arrived_at_floor = Some(cell);
            info!("[atlas] Arrived at cell {cell} (fully drilled in)");
        } else {
            info!(
                "[atlas] Zoom transition complete — cell {cell} centered, drill further to continue"
            );
        }
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
    fn zoom_distance_shrinks_progressively_toward_the_floor() {
        let d1 = next_zoom_distance(8.0);
        let d2 = next_zoom_distance(d1);
        let d3 = next_zoom_distance(d2);
        assert!(d1 < 8.0);
        assert!(d2 < d1);
        assert!(d3 < d2);
        assert!(d3 >= CELL_ZOOM_FLOOR);
    }

    #[test]
    fn zoom_distance_never_undershoots_the_floor() {
        // Repeatedly drilling in should converge to the floor, not below it.
        let mut d = 8.0;
        for _ in 0..50 {
            d = next_zoom_distance(d);
        }
        assert!((d - CELL_ZOOM_FLOOR).abs() < 1e-4);
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
