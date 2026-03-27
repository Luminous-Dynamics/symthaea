// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rendering setup: camera, sprites, HUD, visual stress effects.

use bevy::prelude::*;

use crate::components::*;
use crate::resources::{BiometricsCtx, GamePhase, LeviathanState, SleepPhase};

/// Tile size in pixels.
pub const TILE_SIZE: f32 = 32.0;

/// Marker for the HUD text entity.
#[derive(Component)]
pub struct HudText;

/// Map dimensions (tiles).
pub const MAP_WIDTH: i32 = 24;
pub const MAP_HEIGHT: i32 = 18;

/// Spawn the camera, tile map, player, NPCs, and fusion core.
pub fn setup_world(mut commands: Commands) {
    // Camera
    commands.spawn(Camera2d);

    // Dark tile floor — use Sprite::from_color for solid-color rendering
    for y in -MAP_HEIGHT / 2..MAP_HEIGHT / 2 {
        for x in -MAP_WIDTH / 2..MAP_WIDTH / 2 {
            let walkable = !(x == -MAP_WIDTH / 2
                || x == MAP_WIDTH / 2 - 1
                || y == -MAP_HEIGHT / 2
                || y == MAP_HEIGHT / 2 - 1);

            let color = if walkable {
                Color::srgb(0.15, 0.15, 0.22) // floor — visible dark blue-gray
            } else {
                Color::srgb(0.4, 0.3, 0.25) // wall — clearly visible brown
            };

            commands.spawn((
                Sprite::from_color(color, Vec2::splat(TILE_SIZE - 1.0)),
                Transform::from_xyz(x as f32 * TILE_SIZE, y as f32 * TILE_SIZE, 0.0),
                Tile { grid_x: x, grid_y: y, walkable },
            ));
        }
    }

    // Player — bright cyan square
    commands.spawn((
        Sprite::from_color(Color::srgb(0.2, 0.9, 1.0), Vec2::splat(22.0)),
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
            Sprite::from_color(Color::srgb(0.3, 0.9, 0.3), Vec2::splat(16.0)),
            Transform::from_xyz(*x, *y, 1.0),
            CrewNpc::new(name, i as u64 + 100),
            MoveTarget { target: None, speed: 60.0 },
            NoiseEmitter::default(),
        ));
    }

    // Fusion core — bright yellow at the far end
    commands.spawn((
        Sprite::from_color(Color::srgb(1.0, 0.9, 0.1), Vec2::splat(26.0)),
        Transform::from_xyz(0.0, (MAP_HEIGHT as f32 / 2.0 - 3.0) * TILE_SIZE, 1.0),
        FusionCore { being_extracted: false, extraction_progress: 0.0 },
    ));

    // HUD text overlay (top-left)
    commands.spawn((
        Text::new("WASD: move | E: extract | Esc: quit"),
        TextFont { font_size: 18.0, ..default() },
        TextColor(Color::srgb(0.7, 0.9, 0.7)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(10.0),
            left: Val::Px(10.0),
            ..default()
        },
        HudText,
    ));

    info!("World spawned: {}x{} tiles, 1 player, 3 NPCs, 1 fusion core", MAP_WIDTH, MAP_HEIGHT);
}

/// Telemetry logging resource to throttle output.
#[derive(Resource)]
pub struct TelemetryTimer(pub f32);

impl Default for TelemetryTimer {
    fn default() -> Self {
        Self(0.0)
    }
}

/// HUD: update on-screen text with live telemetry.
pub fn hud_system(
    biometrics: Res<BiometricsCtx>,
    leviathan: Res<LeviathanState>,
    cores: Query<&FusionCore>,
    player: Query<&Transform, With<Player>>,
    mut hud: Query<(&mut Text, &mut TextColor), With<HudText>>,
    time: Res<Time>,
    mut timer: ResMut<TelemetryTimer>,
) {
    timer.0 += time.delta_secs();
    if timer.0 < 0.25 {
        return; // 4Hz update
    }
    timer.0 = 0.0;

    let stress = biometrics.encoder.compute_stress_vector();
    let extraction = cores.iter().next().map(|c| c.extraction_progress).unwrap_or(0.0);
    let player_pos = player.iter().next().map(|t| t.translation.truncate()).unwrap_or_default();

    let phase_str = match leviathan.phase {
        SleepPhase::Dormant => "DORMANT  ",
        SleepPhase::Stirring => "STIRRING!",
        SleepPhase::Awake => "!! AWAKE !!",
        SleepPhase::Hunting => "HUNTING!!!",
    };

    let hud_text = format!(
        "WASD: move | E: extract core | Esc: quit\n\
         Stress: {:.0}%  Load: {:.0}%  Leviathan: {}\n\
         Noise: {:.1}/{:.1}  Extract: {:.0}%  Pos: ({:.0},{:.0})",
        stress.arousal * 100.0,
        biometrics.model.allostatic_load * 100.0,
        phase_str,
        leviathan.noise_accumulator,
        leviathan.threshold,
        extraction * 100.0,
        player_pos.x,
        player_pos.y,
    );

    for (mut text, mut color) in &mut hud {
        **text = hud_text.clone();
        // Color shifts with danger
        let r = match leviathan.phase {
            SleepPhase::Dormant => 0.6,
            SleepPhase::Stirring => 0.9,
            SleepPhase::Awake | SleepPhase::Hunting => 1.0,
        };
        let g = match leviathan.phase {
            SleepPhase::Dormant => 0.9,
            SleepPhase::Stirring => 0.7,
            _ => 0.2,
        };
        *color = TextColor(Color::srgb(r, g, 0.3));
    }
}

/// Visual stress effect: placeholder.
pub fn visual_stress_system(
    _biometrics: Res<BiometricsCtx>,
    _camera: Query<&Camera>,
) {
    // Phase T2.1: post-processing shader for vignette + chromatic aberration
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
