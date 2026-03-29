// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rendering setup: camera, level design, sprites, HUD, visual stress effects.

use bevy::prelude::*;

use crate::components::*;
use crate::resources::{BiometricsCtx, GamePhase, GovernanceLog, LeviathanState, SleepPhase, TileGrid};
// TODO: re-enable when Mycelix integration stabilizes
// use symtropy_sim_bridge::{ActiveProposal, GovernanceState};

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
pub fn setup_world(
    mut commands: Commands,
    seed: Res<crate::resources::DungeonSeed>,
    mut physics_world: ResMut<crate::resources::PhysicsWorldRes>,
) {
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

    // Player — bright cyan, with physics body
    let player_physics_handle = physics_world.world.add_sphere(
        symtropy_math::Point::new([player_pos.x as f64, player_pos.y as f64]),
        10.0, // collision radius (half the sprite size)
        1.0,  // mass
    );
    // Disable gravity for the player (top-down game)
    if let Some(body) = physics_world.world.body_mut(player_physics_handle) {
        body.linear_damping = 0.5; // high damping for responsive controls
    }
    // Register player with consciousness field
    physics_world.consciousness.register(player_physics_handle, 100.0, 50.0);
    commands.spawn((
        Sprite::from_color(Color::srgb(0.2, 0.9, 1.0), Vec2::splat(20.0)),
        Transform::from_xyz(player_pos.x, player_pos.y, 2.0),
        Player,
        Flashlight::default(),
        NoiseEmitter::default(),
        ConsciousnessComp::default(),
        TendBalance::new(40), // ±40 TEND credit limit
        FactionAffiliation::default(),
        symtropy_render_bridge::PhysicsBody::new(player_physics_handle, 10.0),
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
        let cp = ConsciousnessComp {
            sim_dimensions: npc_consciousness[i],
            ..Default::default()
        };

        // Register NPC physics body
        let npc_physics_handle = physics_world.world.add_sphere(
            symtropy_math::Point::new([*x as f64, *y as f64]),
            8.0, // collision radius
            1.0, // mass
        );
        if let Some(body) = physics_world.world.body_mut(npc_physics_handle) {
            body.linear_damping = 0.5;
        }
        // Register NPC with consciousness field
        physics_world.consciousness.register(npc_physics_handle, 80.0, 30.0);

        commands.spawn((
            Sprite::from_color(*color, Vec2::splat(16.0)),
            Transform::from_xyz(*x, *y, 2.0),
            CrewNpc::new(name, i as u64 + 100),
            MoveTarget { target: None, speed: 60.0 },
            NoiseEmitter::default(),
            cp,
            TendBalance::new(40),
            FactionAffiliation {
                faction_id: None,
                ideology: [
                    npc_consciousness[i][0],
                    npc_consciousness[i][1],
                    npc_consciousness[i][3],
                    npc_consciousness[i][4],
                ],
            },
            NpcTrust::default(),
            symtropy_render_bridge::PhysicsBody::new(npc_physics_handle, 8.0),
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
    player: Query<&Transform, With<Player>>,
    mut hud: Query<(&mut Text, &mut TextColor), With<HudText>>,
    time: Res<Time>,
    mut timer: ResMut<TelemetryTimer>,
    gov_log: Res<GovernanceLog>,
    explored: Res<crate::systems::minimap::ExploredTiles>,
    harmony: Res<crate::systems::harmonies::LocalHarmonyState>,
    collected: Res<crate::systems::scavenge::CollectedPrimitives>,
    player_consciousness: Res<crate::systems::consciousness::PlayerConsciousness>,
) {
    timer.0 += time.delta_secs();
    if timer.0 < 0.25 {
        return;
    }
    timer.0 = 0.0;

    let stress = biometrics.encoder.compute_stress_vector();
    let extraction = cores.iter().next().map(|c| c.extraction_progress).unwrap_or(0.0);
    let player_pos = player.iter().next().map(|t| t.translation.truncate()).unwrap_or_default();

    let phase_str = match leviathan.phase {
        SleepPhase::Dormant => "DORMANT",
        SleepPhase::Stirring => "!! STIRRING !!",
        SleepPhase::Awake => "!!! AWAKE !!!",
        SleepPhase::Hunting => ">>> HUNTING <<<",
    };

    let harmony_str = harmony.dominant
        .map(|h| h.name())
        .unwrap_or("none");

    let sanctuary_str = if harmony.is_sanctuary { " [SANCTUARY]" } else { "" };

    let consciousness_str = format!(
        "C={:.0}%  bottleneck: {}  stability: {:.0}%",
        player_consciousness.level * 100.0,
        player_consciousness.bottleneck,
        player_consciousness.stability * 100.0,
    );

    let hud_text = format!(
        "WASD: move | Shift: sprint | E: extract | Esc: quit\n\
         Stress: {:.0}%  Load: {:.0}%  Leviathan: {phase}{sanctuary}\n\
         {consciousness}  Harmony: {harm}  Fragments: {frags}/48\n\
         Noise: {noise:.1}/{thresh:.1}  Extract: {ext:.0}%  Explored: {exp:.0}%",
        stress.arousal * 100.0,
        biometrics.model.allostatic_load * 100.0,
        phase = phase_str,
        sanctuary = sanctuary_str,
        consciousness = consciousness_str,
        harm = harmony_str,
        frags = collected.total(),
        noise = leviathan.noise_accumulator,
        thresh = leviathan.threshold,
        ext = extraction * 100.0,
        exp = explored.explore_percent(),
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
