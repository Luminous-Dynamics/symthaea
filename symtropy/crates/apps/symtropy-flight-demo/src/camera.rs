// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Orbital camera that tracks the quadrotor.

use bevy::prelude::*;

/// Marker for the primary 3D camera.
#[derive(Component)]
pub struct DemoCamera;

/// Spawn a 3D perspective camera looking at the starting hover point.
pub fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(6.0, -6.0, 4.0).looking_at(Vec3::new(0.0, 0.0, 1.5), Vec3::Z),
        DemoCamera,
    ));
}
