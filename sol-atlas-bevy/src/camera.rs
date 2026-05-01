// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Orbital camera for the globe — mouse drag to rotate, scroll to zoom,
//! golden-ratio Lissajous drift when idle.

use bevy::input::mouse::{AccumulatedMouseMotion, AccumulatedMouseScroll};
use bevy::prelude::*;
use sol_atlas_core::constants::*;

/// Marker component for the orbital camera.
#[derive(Component)]
pub struct OrbitalCamera;

/// Configuration resource for the orbital camera.
#[derive(Resource)]
pub struct OrbitalCameraConfig {
    pub theta: f32,
    pub phi: f32,
    pub distance: f32,
    pub vel_theta: f32,
    pub vel_phi: f32,
    pub dragging: bool,
    pub drift_time: f32,
    pub auto_rotate: bool,
}

impl Default for OrbitalCameraConfig {
    fn default() -> Self {
        Self {
            theta: CAMERA_INITIAL_THETA,
            phi: 0.0,
            distance: CAMERA_INITIAL_DISTANCE,
            vel_theta: 0.0,
            vel_phi: 0.0,
            dragging: false,
            drift_time: 0.0,
            auto_rotate: true,
        }
    }
}

/// Spawn a 3D perspective camera looking at the globe origin.
pub fn spawn_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, 0.0, CAMERA_INITIAL_DISTANCE).looking_at(Vec3::ZERO, Vec3::Y),
        OrbitalCamera,
    ));
}

/// Update the orbital camera from accumulated mouse input and inertia.
/// Uses `AccumulatedMouseMotion` and `AccumulatedMouseScroll` (Bevy 0.18 API).
pub fn orbital_camera_system(
    time: Res<Time>,
    mouse_button: Res<ButtonInput<MouseButton>>,
    accumulated_motion: Res<AccumulatedMouseMotion>,
    accumulated_scroll: Res<AccumulatedMouseScroll>,
    mut config: ResMut<OrbitalCameraConfig>,
    mut query: Query<&mut Transform, With<OrbitalCamera>>,
) {
    let dt = time.delta_secs();
    let phi_clamp = CAMERA_PHI_CLAMP_DEG.to_radians();

    // Dragging
    if mouse_button.pressed(MouseButton::Left) {
        config.dragging = true;
        let delta = accumulated_motion.delta;
        if delta != Vec2::ZERO {
            config.vel_theta = -delta.x * 0.005;
            config.vel_phi = -delta.y * 0.005;
            config.theta += config.vel_theta;
            config.phi = (config.phi + config.vel_phi).clamp(-phi_clamp, phi_clamp);
        }
    } else {
        config.dragging = false;
    }

    // Zoom
    let scroll = accumulated_scroll.delta.y;
    if scroll.abs() > 0.001 {
        config.distance = (config.distance - scroll * 0.3).clamp(CAMERA_ZOOM_MIN, CAMERA_ZOOM_MAX);
    }

    // Inertia decay
    if !config.dragging {
        config.theta += config.vel_theta;
        config.phi = (config.phi + config.vel_phi).clamp(-phi_clamp, phi_clamp);
        config.vel_theta *= CAMERA_INERTIA_DECAY;
        config.vel_phi *= CAMERA_INERTIA_DECAY;

        // Auto-rotate when idle
        if config.auto_rotate && config.vel_theta.abs() < 0.0001 {
            config.theta += CAMERA_AUTO_ROTATE_SPEED;
        }
    }

    // Golden-ratio Lissajous drift
    config.drift_time += dt;
    let drift_x = (config.drift_time * 0.1 * GOLDEN_RATIO).sin() * CAMERA_DRIFT_AMPLITUDE;
    let drift_y = (config.drift_time * 0.1).cos() * CAMERA_DRIFT_AMPLITUDE;

    // Compute camera position from spherical coordinates
    let r = config.distance;
    let eye = Vec3::new(
        r * config.phi.cos() * config.theta.sin() + drift_x,
        r * config.phi.sin() + drift_y,
        r * config.phi.cos() * config.theta.cos(),
    );

    for mut transform in query.iter_mut() {
        *transform = Transform::from_translation(eye).looking_at(Vec3::ZERO, Vec3::Y);
    }
}
