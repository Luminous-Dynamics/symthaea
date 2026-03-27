// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rendering setup: camera, sprites, HUD, visual stress effects.

use bevy::prelude::*;

use crate::components::*;
use crate::resources::*;

/// Tile size in pixels.
pub const TILE_SIZE: f32 = 32.0;

/// Map dimensions (tiles).
pub const MAP_WIDTH: i32 = 24;
pub const MAP_HEIGHT: i32 = 18;

/// Spawn the camera, tile map, player, NPCs, and fusion core.
pub fn setup_world(mut commands: Commands) {
    // Camera
    commands.spawn(Camera2d);

    // Dark tile floor
    for y in -MAP_HEIGHT / 2..MAP_HEIGHT / 2 {
        for x in -MAP_WIDTH / 2..MAP_WIDTH / 2 {
            let walkable = !(x == -MAP_WIDTH / 2
                || x == MAP_WIDTH / 2 - 1
                || y == -MAP_HEIGHT / 2
                || y == MAP_HEIGHT / 2 - 1);

            let color = if walkable {
                Color::srgb(0.08, 0.08, 0.12) // dark floor
            } else {
                Color::srgb(0.15, 0.12, 0.10) // wall
            };

            commands.spawn((
                Sprite {
                    color,
                    custom_size: Some(Vec2::splat(TILE_SIZE - 1.0)),
                    ..default()
                },
                Transform::from_xyz(x as f32 * TILE_SIZE, y as f32 * TILE_SIZE, 0.0),
                Tile {
                    grid_x: x,
                    grid_y: y,
                    walkable,
                },
            ));
        }
    }

    // Player — cyan square
    commands.spawn((
        Sprite {
            color: Color::srgb(0.2, 0.8, 0.9),
            custom_size: Some(Vec2::splat(20.0)),
            ..default()
        },
        Transform::from_xyz(0.0, -100.0, 1.0),
        Player,
        Flashlight::default(),
        NoiseEmitter::default(),
    ));

    // Crew NPCs — green squares
    let npc_configs = [
        ("Kael", -60.0, -80.0),
        ("Mira", 40.0, -120.0),
        ("Soren", -20.0, -60.0),
    ];
    for (i, (name, x, y)) in npc_configs.iter().enumerate() {
        commands.spawn((
            Sprite {
                color: Color::srgb(0.3, 0.8, 0.3),
                custom_size: Some(Vec2::splat(16.0)),
                ..default()
            },
            Transform::from_xyz(*x, *y, 1.0),
            CrewNpc::new(name, i as u64 + 100),
            MoveTarget {
                target: None,
                speed: 60.0,
            },
            NoiseEmitter::default(),
        ));
    }

    // Fusion core — yellow diamond at the far end
    commands.spawn((
        Sprite {
            color: Color::srgb(1.0, 0.9, 0.2),
            custom_size: Some(Vec2::splat(24.0)),
            ..default()
        },
        Transform::from_xyz(0.0, (MAP_HEIGHT as f32 / 2.0 - 3.0) * TILE_SIZE, 1.0),
        FusionCore {
            being_extracted: false,
            extraction_progress: 0.0,
        },
    ));
}

/// HUD: stress meter, Leviathan phase indicator, extraction progress.
pub fn hud_system(
    biometrics: Res<BiometricsCtx>,
    leviathan: Res<LeviathanState>,
    cores: Query<&FusionCore>,
    game_phase: Res<State<GamePhase>>,
    // We'd need a Text entity to update — for now, print to console at low frequency
) {
    // This will be replaced with proper Bevy UI text entities
    // For the prototype, the audio IS the feedback
}

/// Visual stress effect: vignette darkening at screen edges.
/// (Placeholder — full implementation needs a post-processing shader)
pub fn visual_stress_system(
    biometrics: Res<BiometricsCtx>,
    mut camera: Query<&mut Camera>,
) {
    let _load = biometrics.model.allostatic_load;
    // In full implementation: drive post-processing shader uniforms
    // For prototype: the audio feedback carries the tension
}

/// Camera follows player.
pub fn camera_follow_system(
    player: Query<&Transform, With<Player>>,
    mut camera: Query<&mut Transform, (With<Camera2d>, Without<Player>)>,
) {
    let Ok(player_tf) = player.single() else {
        return;
    };
    let Ok(mut cam_tf) = camera.single_mut() else {
        return;
    };
    // Smooth follow
    let target = player_tf.translation.truncate();
    let current = cam_tf.translation.truncate();
    let smoothed = current.lerp(target, 0.1);
    cam_tf.translation.x = smoothed.x;
    cam_tf.translation.y = smoothed.y;
}
