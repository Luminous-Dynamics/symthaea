// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Camera framing the spacecraft.

use bevy::prelude::*;

#[derive(Component)]
pub struct DemoCamera;

pub fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(6.0, -6.0, 4.5).looking_at(Vec3::new(0.0, 0.0, 1.5), Vec3::Z),
        DemoCamera,
    ));
}
