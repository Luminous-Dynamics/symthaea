// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Camera following the quadruped as it walks forward.

use bevy::prelude::*;

#[derive(Component)]
pub struct DemoCamera;

pub fn setup_camera(mut commands: Commands) {
    // The quadruped walks forward along +X; camera looks from forward-right.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(3.0, -3.5, 1.6).looking_at(Vec3::new(1.5, 0.0, 0.3), Vec3::Z),
        DemoCamera,
    ));
}
