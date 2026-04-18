// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chase camera for the vehicle demo.

use bevy::prelude::*;

#[derive(Component)]
pub struct DemoCamera;

/// Spawn a 3D camera looking down at the track from behind/above.
pub fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(25.0, -40.0, 22.0).looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Z),
        DemoCamera,
    ));
}
