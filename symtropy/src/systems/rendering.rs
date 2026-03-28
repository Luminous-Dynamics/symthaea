// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rendering setup: camera, level design, sprites, HUD, visual stress effects.

use bevy::prelude::*;

use crate::components::*;
use crate::resources::{BiometricsCtx, GamePhase, GovernanceLog, LeviathanState, SleepPhase, TileGrid};
use symtropy_sim_bridge::{ActiveProposal, GovernanceState};

/// Tile size in pixels.
pub const TILE_SIZE: f32 = 32.0;

/// Map dimensions (tiles).
pub const MAP_WIDTH: i32 = 30;
pub const MAP_HEIGHT: i32 = 22;

/// Marker for the HUD text entity.
#[derive(Component)]
pub struct HudText;

/// Marker for the Leviathan sprite.
#[derive(Component)]
pub struct LeviathanSprite;

/// Generate level layout from a seed.
fn level_map(seed: u64) -> Vec<Vec<u8>> {
    let dungeon = super::procgen::generate_dungeon(MAP_WIDTH as usize, MAP_HEIGHT as usize, seed);
    eprintln!("[symtropy] Generated dungeon with seed {seed}");
    return dungeon.tiles;

    // Legacy hand-designed level kept as reference:
    #[allow(unreachable_code)]
    let raw = [
        "##############################",
        "#....##########...###########",
        "#....##........#..#.........#",
        "#....##.######.#..#.##.####.#",
        "#....##.#....#.#..#.##.#..#.#",
        "##.####.#....#.#..#....#..#.#",
        "#......##....#.#..######..#.#",
        "#.####.##.####....#.......#.#",
        "#.#..#.......#.####.#######.#",
        "#.#..#.####..#.#......#.....#",
        "#.#....#..#..#.#.####.#.###.#",
        "#.######..#..#.#.#..#.#.#...#",
        "#.........#....#.#..#...#.#.#",
        "#.####.####.####.#..#####.#.#",
        "#.#..#.#.........#........#.#",
        "#.#..#.#.#########.######.#.#",
        "#.#....#.#...CC...........#.#",
        "#.######.#...CC...#########.#",
        "#........#........#.........#",
        "#.################.########.#",
        "#..............P............#",
        "##############################",
    ];

    raw.iter()
        .map(|row| {
            row.chars()
                .map(|c| match c {
                    '#' => 1,
                    'C' => 2, // core room
                    'P' => 3, // player start (treated as floor)
                    _ => 0,
                })
                .collect()
        })
        .collect()
}

/// Spawn the camera, level, player, NPCs, fusion core, and Leviathan.
pub fn setup_world(mut commands: Commands, seed: Res<crate::resources::DungeonSeed>) {
    // Camera with consciousness-driven post-processing
    commands.spawn((
        Camera2d,
        Transform::from_xyz(0.0, 0.0, 999.0),
        bevy::post_process::bloom::Bloom {
            intensity: 0.15,
            ..default()
        },
    ));

    let map = level_map(seed.0);
    let rows = map.len() as i32;
    let cols = if map.is_empty() { 0 } else { map[0].len() as i32 };

    let mut player_pos = Vec2::new(0.0, 0.0);
    let mut core_pos = Vec2::new(0.0, 0.0);

    // Build tile grid for O(1) collision lookups
    let mut tile_grid = TileGrid {
        tile_size: TILE_SIZE,
        origin_col: cols / 2,
        origin_row: rows / 2,
        cols,
        rows,
        ..default()
    };

    // Spawn tiles
    for (row_idx, row) in map.iter().enumerate() {
        for (col_idx, &cell) in row.iter().enumerate() {
            let x = (col_idx as f32 - cols as f32 / 2.0) * TILE_SIZE;
            let y = (rows as f32 / 2.0 - row_idx as f32) * TILE_SIZE;

            let (color, walkable) = match cell {
                1 => (Color::srgb(0.45, 0.35, 0.25), false), // wall — brown
                2 => {
                    core_pos = Vec2::new(x, y);
                    (Color::srgb(0.2, 0.25, 0.35), true) // core room — darker blue
                }
                3 => {
                    player_pos = Vec2::new(x, y);
                    (Color::srgb(0.25, 0.25, 0.32), true) // player start
                }
                _ => (Color::srgb(0.22, 0.22, 0.30), true), // floor — dark blue-gray
            };

            // Register in spatial lookup grid
            tile_grid.cells.insert((col_idx as i32, row_idx as i32), walkable);

            commands.spawn((
                Sprite::from_color(color, Vec2::splat(TILE_SIZE - 1.0)),
                Transform::from_xyz(x, y, 0.0),
                Tile {
                    grid_x: col_idx as i32,
                    grid_y: row_idx as i32,
                    walkable,
                },
            ));
        }
    }

    // Insert tile grid as resource for collision checks
    commands.insert_resource(tile_grid);

    // Player — bright cyan
    commands.spawn((
        Sprite::from_color(Color::srgb(0.2, 0.9, 1.0), Vec2::splat(20.0)),
        Transform::from_xyz(player_pos.x, player_pos.y, 2.0),
        Player,
        Flashlight::default(),
        NoiseEmitter::default(),
        ConsciousnessProfile::default(),
        TendBalance::new(40), // ±40 TEND credit limit
        FactionAffiliation::default(),
    ));

    // Crew NPCs — each a different green shade, spread near player
    let npc_configs = [
        ("Kael", player_pos.x - 32.0, player_pos.y, Color::srgb(0.3, 0.9, 0.3)),
        ("Mira", player_pos.x + 32.0, player_pos.y, Color::srgb(0.4, 0.85, 0.5)),
        ("Soren", player_pos.x, player_pos.y + 32.0, Color::srgb(0.25, 0.8, 0.4)),
    ];
    // NPC consciousness varies — creates natural tier distribution for governance
    let npc_consciousness = [
        [0.6, 0.5, 0.5, 0.7, 0.6, 0.5], // Kael: high care, Steward-tier
        [0.4, 0.3, 0.6, 0.5, 0.4, 0.4], // Mira: moderate, Contributor-tier
        [0.8, 0.7, 0.7, 0.4, 0.8, 0.6], // Soren: high level, near-Guardian
    ];
    for (i, (name, x, y, color)) in npc_configs.iter().enumerate() {
        let mut cp = ConsciousnessProfile {
            phi: 0.0,
            tier: 0,
            dimensions: npc_consciousness[i],
        };
        cp.compute_phi();

        commands.spawn((
            Sprite::from_color(*color, Vec2::splat(16.0)),
            Transform::from_xyz(*x, *y, 2.0),
            CrewNpc::new(name, i as u64 + 100),
            MoveTarget {
                target: None,
                speed: 60.0,
            },
            NoiseEmitter::default(),
            cp,
            TendBalance::new(40),
            FactionAffiliation {
                faction_id: None,
                ideology: [
                    npc_consciousness[i][0], // economic
                    npc_consciousness[i][1], // authority
                    npc_consciousness[i][3], // tradition
                    npc_consciousness[i][4], // individual
                ],
            },
            NpcTrust::default(),
        ));
    }

    // Fusion core — pulsing yellow in the core room
    commands.spawn((
        Sprite::from_color(Color::srgb(1.0, 0.9, 0.1), Vec2::splat(28.0)),
        Transform::from_xyz(core_pos.x, core_pos.y, 2.0),
        FusionCore {
            being_extracted: false,
            extraction_progress: 0.0,
        },
    ));

    // Leviathan — large red entity, initially invisible (alpha=0), appears on AWAKE
    commands.spawn((
        Sprite::from_color(
            Color::srgba(0.9, 0.1, 0.1, 0.0), // invisible until awake
            Vec2::new(48.0, 48.0),
        ),
        Transform::from_xyz(core_pos.x, core_pos.y + 64.0, 3.0),
        LeviathanSprite,
    ));

    // HUD
    commands.spawn((
        Text::new("WASD: move | E: extract core | Esc: quit"),
        TextFont {
            font_size: 20.0,
            ..default()
        },
        TextColor(Color::srgb(0.7, 0.9, 0.7)),
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(8.0),
            left: Val::Px(12.0),
            ..default()
        },
        HudText,
    ));

    info!(
        "World spawned: {}x{} level, player at ({:.0},{:.0}), core at ({:.0},{:.0})",
        cols, rows, player_pos.x, player_pos.y, core_pos.x, core_pos.y
    );
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
    player: Query<(&Transform, Option<&ConsciousnessProfile>, Option<&TendBalance>), With<Player>>,
    mut hud: Query<(&mut Text, &mut TextColor), With<HudText>>,
    time: Res<Time>,
    mut timer: ResMut<TelemetryTimer>,
    gov: Res<GovernanceState>,
    proposal: Res<ActiveProposal>,
    gov_log: Res<GovernanceLog>,
    explored: Res<crate::systems::minimap::ExploredTiles>,
) {
    timer.0 += time.delta_secs();
    if timer.0 < 0.25 {
        return;
    }
    timer.0 = 0.0;

    let stress = biometrics.encoder.compute_stress_vector();
    let extraction = cores
        .iter()
        .next()
        .map(|c| c.extraction_progress)
        .unwrap_or(0.0);
    let (player_pos, player_phi, player_tend) = player
        .iter()
        .next()
        .map(|(t, cp, tb)| (
            t.translation.truncate(),
            cp.map(|c| c.phi).unwrap_or(0.0),
            tb.map(|t| t.balance).unwrap_or(0),
        ))
        .unwrap_or_default();

    let phase_str = match leviathan.phase {
        SleepPhase::Dormant => "DORMANT",
        SleepPhase::Stirring => "!! STIRRING !!",
        SleepPhase::Awake => "!!! AWAKE !!!",
        SleepPhase::Hunting => ">>> HUNTING <<<",
    };

    // Oppression indicator
    let oppression = gov.oppression_index();
    let oppression_str = if oppression > 0.5 {
        "CRISIS"
    } else if oppression > 0.3 {
        "WARNING"
    } else {
        "stable"
    };

    // Active proposal line
    let proposal_str = if proposal.active {
        if proposal.vetoed {
            format!("VETOED: \"{}\" (override: {:.0}%)", proposal.description,
                if proposal.override_votes.is_empty() { 0.0 } else {
                    let total: f64 = proposal.override_votes.iter().map(|(_, w, _)| w).sum();
                    let approve: f64 = proposal.override_votes.iter().filter(|(_, _, a)| *a).map(|(_, w, _)| w).sum();
                    if total > 0.0 { approve / total * 100.0 } else { 0.0 }
                })
        } else {
            format!("VOTE: \"{}\" ({:.0}% approve, {}t left)",
                proposal.description, proposal.approval_ratio() * 100.0, proposal.ticks_remaining)
        }
    } else {
        String::new()
    };

    // Last governance event
    let last_event = gov_log.messages.last()
        .map(|m| format!("[GOV] {}", m.text))
        .unwrap_or_default();

    let hud_text = format!(
        "WASD: move | E: extract | T: TEND exchange | Esc: quit\n\
         Stress: {:.0}%  Load: {:.0}%  Leviathan: {}\n\
         Phi: {:.2}  TEND: {}  Oppression: {} ({:.0}%)  Stability: {:.0}%\n\
         Noise: {:.1}/{:.1}  Extract: {:.0}%  Explored: {:.0}%  Pos: ({:.0},{:.0})\n\
         {}\n\
         {}",
        stress.arousal * 100.0,
        biometrics.model.allostatic_load * 100.0,
        phase_str,
        player_phi,
        player_tend,
        oppression_str,
        oppression * 100.0,
        gov.stability() * 100.0,
        leviathan.noise_accumulator,
        leviathan.threshold,
        extraction * 100.0,
        explored.explore_percent(),
        player_pos.x,
        player_pos.y,
        proposal_str,
        last_event,
    );

    for (mut text, mut color) in &mut hud {
        **text = hud_text.clone();
        let (r, g) = match leviathan.phase {
            SleepPhase::Dormant => (0.6, 0.9),
            SleepPhase::Stirring => (1.0, 0.8),
            SleepPhase::Awake => (1.0, 0.3),
            SleepPhase::Hunting => (1.0, 0.1),
        };
        *color = TextColor(Color::srgb(r, g, 0.3));
    }
}

/// Update Leviathan sprite visibility based on phase.
pub fn leviathan_visual_system(
    leviathan: Res<LeviathanState>,
    mut sprites: Query<&mut Sprite, With<LeviathanSprite>>,
    player: Query<&Transform, With<Player>>,
    mut lev_transform: Query<&mut Transform, (With<LeviathanSprite>, Without<Player>)>,
    time: Res<Time>,
) {
    for mut sprite in &mut sprites {
        let alpha = match leviathan.phase {
            SleepPhase::Dormant => 0.0,
            SleepPhase::Stirring => 0.3 + (time.elapsed_secs() * 2.0).sin().abs() * 0.2,
            SleepPhase::Awake => 0.7,
            SleepPhase::Hunting => 1.0,
        };
        sprite.color = Color::srgba(0.9, 0.15, 0.1, alpha);

        // Grow when hunting
        if leviathan.phase == SleepPhase::Hunting {
            let pulse = 48.0 + (time.elapsed_secs() * 4.0).sin() * 8.0;
            sprite.custom_size = Some(Vec2::splat(pulse));
        }
    }

    // Chase player when hunting
    if leviathan.phase == SleepPhase::Hunting {
        if let Ok(player_tf) = player.single() {
            for mut tf in &mut lev_transform {
                let dir = player_tf.translation.truncate() - tf.translation.truncate();
                if dir.length() > 5.0 {
                    let move_vec = dir.normalize() * 80.0 * time.delta_secs();
                    tf.translation.x += move_vec.x;
                    tf.translation.y += move_vec.y;
                }
            }
        }
    }
}

/// Visual stress: vignette effect via darkening the background color.
pub fn visual_stress_system(
    biometrics: Res<BiometricsCtx>,
    leviathan: Res<LeviathanState>,
    mut clear_color: ResMut<ClearColor>,
) {
    let load = biometrics.model.allostatic_load;
    let danger = match leviathan.phase {
        SleepPhase::Dormant => 0.0,
        SleepPhase::Stirring => 0.2,
        SleepPhase::Awake => 0.5,
        SleepPhase::Hunting => 0.8,
    };

    // Background shifts from dark blue → dark red with stress/danger
    let stress_red = (0.02 + load * 0.1 + danger * 0.15).min(0.3);
    let base_blue = (0.04 - danger * 0.03).max(0.01);
    clear_color.0 = Color::srgb(stress_red, 0.02, base_blue);
}

/// Camera follows player smoothly.
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
    let target = player_tf.translation.truncate();
    let current = cam_tf.translation.truncate();
    let smoothed = current.lerp(target, 0.08);
    cam_tf.translation.x = smoothed.x;
    cam_tf.translation.y = smoothed.y;
}
