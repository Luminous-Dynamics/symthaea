// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Camera looking at the human-exo from the side.

use bevy::prelude::*;

#[derive(Component)]
pub struct DemoCamera;

pub fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(0.0, -3.8, 1.4).looking_at(Vec3::new(0.0, 0.0, 0.9), Vec3::Z),
        DemoCamera,
    ));
}
