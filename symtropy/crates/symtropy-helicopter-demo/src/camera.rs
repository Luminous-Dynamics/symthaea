// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ground-truck camera looking at the hover station.

use bevy::prelude::*;

#[derive(Component)]
pub struct DemoCamera;

pub fn setup_camera(mut commands: Commands) {
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(35.0, -35.0, 28.0).looking_at(Vec3::new(0.0, 0.0, 18.0), Vec3::Z),
        DemoCamera,
    ));
}
