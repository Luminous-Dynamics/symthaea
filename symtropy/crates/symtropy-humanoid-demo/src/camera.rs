// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Camera looking at the humanoid from a three-quarter angle.

use bevy::prelude::*;

#[derive(Component)]
pub struct DemoCamera;

pub fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(2.5, -3.5, 1.6).looking_at(Vec3::new(0.0, 0.0, 1.0), Vec3::Z),
        DemoCamera,
    ));
}
