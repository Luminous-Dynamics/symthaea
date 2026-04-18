// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Underwater camera — looks down and across the waypoint field.

use bevy::prelude::*;

#[derive(Component)]
pub struct DemoCamera;

pub fn setup_camera(mut commands: Commands) {
    // AUV world: surface at z=0, descending into negative z. Camera sits
    // above the surface tilting into the water column.
    commands.spawn((
        Camera3d::default(),
        Transform::from_xyz(18.0, -18.0, 10.0).looking_at(Vec3::new(0.0, 0.0, -8.0), Vec3::Z),
        DemoCamera,
    ));
}
