// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Surgical-field overhead-angle camera.

use bevy::prelude::*;

#[derive(Component)]
pub struct DemoCamera;

pub fn setup_camera(mut commands: Commands) {
    // Positions in millimeters (simulator units).
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(220.0, -220.0, 120.0).looking_at(Vec3::new(0.0, 0.0, -40.0), Vec3::Z),
        DemoCamera,
    ));
}
