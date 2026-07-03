// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! System to spawn infrastructure entities for Firstlight Basin.

use crate::components::{Fabricator, InteractionTarget, PowerJunction, WaterPump};
use crate::systems::rendering::TILE_SIZE;
use bevy::prelude::*;

pub fn spawn_infrastructure_system(
    mut commands: Commands,
    layout: Res<crate::resources::SiteLayout>,
    registry: Res<crate::experience::ExperienceRegistry>,
) {
    let exp = &registry.experiences[registry.selected];
    if exp.id == "waterworks-3d" {
        info!("Skipping 2D infrastructure spawning for 3D experience");
        return;
    }

    let room_centers = &layout.room_centers;
    if room_centers.is_empty() {
        warn!("No room centers generated for infrastructure spawning!");
        return;
    }

    info!(
        "Spawning settlement infrastructure at {} room centers",
        room_centers.len()
    );

    let cols = layout.width as f32;
    let rows = layout.height as f32;

    // Helper to map grid coordinates to world coordinates
    let grid_to_world = |col: usize, row: usize| -> Vec2 {
        let x = (col as f32 - cols / 2.0) * TILE_SIZE;
        let y = (rows / 2.0 - row as f32) * TILE_SIZE;
        Vec2::new(x, y)
    };

    // 1. Power Junction (Room Center 1 or fallback)
    let idx_power = 1 % room_centers.len();
    let pos_power = room_centers[idx_power];
    let world_power = grid_to_world(pos_power.0, pos_power.1);
    commands
        .spawn((
            Sprite::from_color(Color::srgb(1.0, 0.6, 0.1), Vec2::splat(18.0)),
            Transform::from_xyz(world_power.x, world_power.y, 1.0),
            PowerJunction {
                output: 0.2, // damaged state starts low
                is_damaged: true,
                uptime_secs: 0.0,
            },
            InteractionTarget {
                radius: 40.0,
                label: "Repair Power Junction".to_string(),
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                Text2d::new("Power Junction (Offline)"),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::srgb(1.0, 0.8, 0.6)),
                Transform::from_xyz(0.0, 20.0, 1.0),
            ));
        });

    // 2. Water Pump (Room Center 2 or fallback)
    let idx_pump = if room_centers.len() > 2 { 2 } else { 0 };
    let pos_pump = room_centers[idx_pump];
    let world_pump = grid_to_world(pos_pump.0, pos_pump.1);
    commands
        .spawn((
            Sprite::from_color(Color::srgb(0.1, 0.6, 1.0), Vec2::splat(18.0)),
            Transform::from_xyz(world_pump.x, world_pump.y, 1.0),
            WaterPump {
                efficiency: 1.0,
                is_running: false,
                is_sabotaged: true,
            },
            InteractionTarget {
                radius: 40.0,
                label: "Restore Water Pump".to_string(),
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                Text2d::new("Water Pump (Sabotaged)"),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::srgb(0.6, 0.8, 1.0)),
                Transform::from_xyz(0.0, 20.0, 1.0),
            ));
        });

    // 3. Fabricator (Room Center 3 or fallback)
    let idx_fab = if room_centers.len() > 3 { 3 } else { 0 };
    let pos_fab = room_centers[idx_fab];
    let world_fab = grid_to_world(pos_fab.0, pos_fab.1);
    commands
        .spawn((
            Sprite::from_color(Color::srgb(0.7, 0.2, 0.9), Vec2::splat(18.0)),
            Transform::from_xyz(world_fab.x, world_fab.y, 1.0),
            Fabricator {
                material_reserve: 0.5,
                power_draw: 0.1,
                is_active: false,
            },
            InteractionTarget {
                radius: 40.0,
                label: "Access Fabricator".to_string(),
            },
        ))
        .with_children(|parent| {
            parent.spawn((
                Text2d::new("Fabricator"),
                TextFont {
                    font_size: FontSize::Px(12.0),
                    ..default()
                },
                TextColor(Color::srgb(0.9, 0.7, 1.0)),
                Transform::from_xyz(0.0, 20.0, 1.0),
            ));
        });
}
