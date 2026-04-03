// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Camera setup for the manipulator demo scene.

use bevy::prelude::*;

/// Marker for the primary 3D camera.
#[derive(Component)]
pub struct DemoCamera;

/// Spawn a 3D perspective camera looking at the workspace.
pub fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(1.5, -1.5, 1.2).looking_at(Vec3::new(0.3, 0.0, 0.3), Vec3::Z),
        DemoCamera,
    ));
}
