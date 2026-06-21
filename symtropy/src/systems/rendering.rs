// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Rendering setup: camera, level design, sprites, HUD, visual stress effects.

use crate::resources::SettlementMetrics;
use bevy::prelude::*;

use crate::components::*;
use crate::resources::{BiometricsCtx, GovernanceLog, LeviathanState, PhysicsWorldRes, SleepPhase};
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

/// Spawn the camera, level, player, NPCs, fusion core, and Leviathan.
pub fn setup_world(
    mut commands: Commands,
    layout: Res<crate::resources::SiteLayout>,
    mut physics_world: ResMut<crate::resources::PhysicsWorldRes>,
    registry: Res<crate::experience::ExperienceRegistry>,
) {
    let exp = &registry.experiences[registry.selected];
    let is_3d = exp.id == "waterworks-3d";

    if !is_3d {
        // Camera with consciousness-driven post-processing
        commands.spawn((
            Camera2d,
            Transform::from_xyz(0.0, 0.0, 999.0),
            bevy::post_process::bloom::Bloom {
                intensity: 0.15,
                ..default()
            },
        ));
    }

    let map = &layout.tiles;
    let rows = layout.height as f32;
    let cols = layout.width as f32;

    let player_pos = layout.player_start;
    let core_pos = layout.core_pos;

    // Spawn tiles
    for (row_idx, row) in map.iter().enumerate() {
        for (col_idx, &cell) in row.iter().enumerate() {
            let x = (col_idx as f32 - cols / 2.0) * TILE_SIZE;
            let y = (rows / 2.0 - row_idx as f32) * TILE_SIZE;

            let color = match cell {
                1 => Color::srgb(0.45, 0.35, 0.25), // wall — brown
                2 => Color::srgb(0.2, 0.25, 0.35),  // core room — darker blue
                3 => Color::srgb(0.25, 0.25, 0.32), // player start
                _ => Color::srgb(0.22, 0.22, 0.30), // floor — dark blue-gray
            };

            if !is_3d {
                commands.spawn((
                    Sprite::from_color(color, Vec2::splat(TILE_SIZE - 1.0)),
                    Transform::from_xyz(x, y, 0.0),
                    Tile {
                        grid_x: col_idx as i32,
                        grid_y: row_idx as i32,
                        walkable: cell != 1,
                    },
                ));
            }
        }
    }

    if is_3d {
        return;
    }

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
    physics_world
        .consciousness
        .register(player_physics_handle, 100.0, 50.0);
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

    // Crew NPCs — each archetype has a different color and caution/governance settings
    let npc_configs = [
        (
            "Engineer (Kael)",
            player_pos.x - 32.0,
            player_pos.y,
            Color::srgb(0.9, 0.5, 0.1),
            0.4, // caution
        ),
        (
            "Medic (Mira)",
            player_pos.x + 32.0,
            player_pos.y,
            Color::srgb(0.2, 0.8, 0.6),
            0.6, // caution
        ),
        (
            "Archivist (Soren)",
            player_pos.x,
            player_pos.y + 32.0,
            Color::srgb(0.3, 0.3, 0.8),
            0.7, // caution
        ),
        (
            "Convoy Lead (Jack)",
            player_pos.x - 32.0,
            player_pos.y - 32.0,
            Color::srgb(0.8, 0.3, 0.3),
            0.3, // caution
        ),
        (
            "Friendly Robot (PR-4)",
            player_pos.x + 32.0,
            player_pos.y - 32.0,
            Color::srgb(0.5, 0.8, 0.8),
            0.1, // caution
        ),
        (
            "Industrial Liaison (Nadia)",
            player_pos.x - 64.0,
            player_pos.y,
            Color::srgb(0.8, 0.8, 0.2),
            0.5, // caution
        ),
        (
            "Young Tech (Leo)",
            player_pos.x + 64.0,
            player_pos.y,
            Color::srgb(0.6, 0.9, 0.2),
            0.3, // caution
        ),
    ];
    // NPC consciousness varies — creates natural tier distribution for governance
    let npc_consciousness = [
        [0.6, 0.5, 0.5, 0.7, 0.6, 0.5], // Engineer: high care
        [0.4, 0.3, 0.6, 0.5, 0.4, 0.4], // Medic: community/social
        [0.8, 0.7, 0.7, 0.4, 0.8, 0.6], // Archivist: deep knowledge
        [0.5, 0.6, 0.4, 0.7, 0.5, 0.5], // Convoy Lead: tactical
        [0.2, 0.1, 0.8, 0.9, 0.2, 0.1], // Friendly Robot: low self-identity, high engagement
        [0.7, 0.8, 0.5, 0.5, 0.7, 0.6], // Industrial Liaison: negotiation
        [0.3, 0.4, 0.4, 0.6, 0.3, 0.3], // Young Tech: junior
    ];
    for (i, (name, x, y, color, caution)) in npc_configs.iter().enumerate() {
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
        physics_world
            .consciousness
            .register(npc_physics_handle, 80.0, 30.0);

        let mut crew_npc = CrewNpc::new(name, i as u64 + 100);
        crew_npc.caution = *caution;

        commands.spawn((
            Sprite::from_color(*color, Vec2::splat(16.0)),
            Transform::from_xyz(*x, *y, 2.0),
            crew_npc,
            MoveTarget {
                target: None,
                speed: 60.0,
            },
            NoiseEmitter::default(),
            cp,
            super::consciousness::NpcConsciousness::default(),
            super::psychology::PsychologicalNeeds {
                allostatic_load: match i {
                    0 => 0.20, // Kael
                    1 => 0.40, // Mira
                    2 => 0.10, // Soren
                    3 => 0.30, // Jack
                    4 => 0.05, // PR-4
                    5 => 0.25, // Nadia
                    _ => 0.50, // Leo
                },
                social_satiation: match i {
                    0 => 0.40,
                    1 => 0.80,
                    2 => 0.30,
                    3 => 0.50,
                    4 => 0.90,
                    5 => 0.60,
                    _ => 0.50,
                },
                engagement: match i {
                    0 => 0.90,
                    1 => 0.70,
                    2 => 0.80,
                    3 => 0.60,
                    4 => 0.95,
                    5 => 0.75,
                    _ => 0.80,
                },
                burnout_ticks: 0,
            },
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

    // Energy Wells — spatial life sources at room centers
    let well_constants = &physics_world.consciousness.constants;
    let mut well_count = 0u32;

    // Guarantee one well at player spawn — survival lifeline
    commands.spawn((
        Sprite::from_color(Color::srgba(0.1, 0.8, 0.6, 0.35), Vec2::splat(40.0)),
        Transform::from_xyz(player_pos.x, player_pos.y, 1.0),
        crate::resources::EnergyWell::new(well_constants.energy_well_regen_rate, 64.0, 5000.0),
    ));
    well_count += 1;

    for (i, &(cx, cy)) in layout.room_centers.iter().enumerate() {
        // Place wells at every other room, skip core room
        if i % 2 != 0 {
            continue;
        }
        let wx = (cx as f32 - cols as f32 / 2.0) * TILE_SIZE;
        let wy = (rows as f32 / 2.0 - cy as f32) * TILE_SIZE;

        // Skip if too close to player (already have spawn well) or core
        let near_player = (wx - player_pos.x).abs() < TILE_SIZE * 3.0
            && (wy - player_pos.y).abs() < TILE_SIZE * 3.0;
        let near_core =
            (wx - core_pos.x).abs() < TILE_SIZE * 2.0 && (wy - core_pos.y).abs() < TILE_SIZE * 2.0;
        if near_player || near_core {
            continue;
        }

        commands.spawn((
            Sprite::from_color(Color::srgba(0.1, 0.8, 0.6, 0.35), Vec2::splat(40.0)),
            Transform::from_xyz(wx, wy, 1.0),
            crate::resources::EnergyWell::new(well_constants.energy_well_regen_rate, 64.0, 5000.0),
        ));
        well_count += 1;
    }
    eprintln!("[symtropy] Energy wells placed: {well_count}");

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
    player: Query<(&Transform, &symtropy_render_bridge::PhysicsBody), With<Player>>,
    mut hud: Query<(&mut Text, &mut TextColor), With<HudText>>,
    time: Res<Time>,
    mut timer: ResMut<TelemetryTimer>,
    gov_log: Res<GovernanceLog>,
    explored: Res<crate::systems::minimap::ExploredTiles>,
    harmony: Res<crate::systems::harmonies::LocalHarmonyState>,
    collected: Res<crate::systems::scavenge::CollectedPrimitives>,
    player_consciousness: Res<crate::systems::consciousness::PlayerConsciousness>,
    physics: Res<PhysicsWorldRes>,
    thermo_hud: Res<crate::systems::thermodynamic::ThermodynamicHudState>,
    dim_state: Res<crate::systems::dimension_transition::DimensionTransition>,
    settlement: Res<SettlementMetrics>,
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
    let (player_pos, player_handle) = player
        .iter()
        .next()
        .map(|(t, pb)| (t.translation.truncate(), Some(pb.handle)))
        .unwrap_or_default();

    let phase_str = match leviathan.phase {
        SleepPhase::Dormant => "DORMANT",
        SleepPhase::Stirring => "!! STIRRING !!",
        SleepPhase::Awake => "!!! AWAKE !!!",
        SleepPhase::Hunting => ">>> HUNTING <<<",
    };

    let harmony_str = harmony.dominant.map(|h| h.name()).unwrap_or("none");

    let sanctuary_str = if harmony.is_sanctuary {
        " [SANCTUARY]"
    } else {
        ""
    };

    let consciousness_str = format!(
        "C={:.0}%  bottleneck: {}  stability: {:.0}%",
        player_consciousness.level * 100.0,
        player_consciousness.bottleneck,
        player_consciousness.stability * 100.0,
    );

    // Thermodynamic display
    let (energy_str, energy_bar) = if let Some(handle) = player_handle {
        if let Some(entity) = physics.consciousness.entities.get(&handle) {
            let e = &entity.energy;
            let frac = e.fraction_remaining();
            let bar_filled = (frac * 15.0) as usize;
            let bar = format!(
                "[{}{}]",
                "=".repeat(bar_filled),
                "-".repeat(15 - bar_filled)
            );
            let net = thermo_hud.regenerated_per_sec - thermo_hud.consumed_per_sec;
            let net_str = if net >= 0.0 {
                format!("+{:.1}", net)
            } else {
                format!("{:.1}", net)
            };
            let collapse_str = if e.is_collapsed() {
                " !! COLLAPSED !!".to_string()
            } else if net < 0.0 && e.available > 0.0 {
                format!(
                    "  collapse in: {:.0}s",
                    e.available / (-net / 64.0).max(0.001)
                )
            } else {
                String::new()
            };
            (
                format!(
                    "Energy: {:.0}/{:.0} J {} {net_str} J/s{collapse_str}",
                    e.available, e.max_energy, bar
                ),
                frac,
            )
        } else {
            ("Energy: --".to_string(), 1.0)
        }
    } else {
        ("Energy: --".to_string(), 1.0)
    };

    let dim_str = if dim_state.transitioning() {
        format!(
            "{} → {} ({:.0}%)",
            dim_state.current.name(),
            dim_state.target.name(),
            dim_state.progress * 100.0
        )
    } else if dim_state.current == crate::systems::dimension_transition::DimensionMode::D4 {
        format!("4D [F1-F4] W={:.0} [/]", dim_state.w_position)
    } else {
        format!("{} [F1-F4]", dim_state.current.name())
    };

    // Contextual hint — guide new players based on current state
    let hint = if leviathan.grace_timer > 0.0 {
        format!(
            ">> Safe for {:.0}s — explore and find energy wells (teal glow)",
            leviathan.grace_timer
        )
    } else if energy_bar < 0.4 {
        ">> LOW ENERGY — find a teal well to restore energy".to_string()
    } else if leviathan.noise_accumulator > 5.0 {
        ">> Stay quiet — the Leviathan is listening".to_string()
    } else if extraction > 0.0 && extraction < 1.0 {
        ">> Hold E near the core — stay calm to extract faster".to_string()
    } else {
        String::new()
    };

    let hud_text = format!(
        "WASD: move | Shift: sprint | E: extract | F1-F4: dimension | []: W-slide | Esc: quit\n\
         Stress: {:.0}%  Load: {:.0}%  Leviathan: {phase}{sanctuary}\n\
         {consciousness}  Harmony: {harm}  Fragments: {frags}/48\n\
         {energy}\n\
         SETTLEMENT: Power={pow:.0}% Water={wat:.0}% Trust={trust:.0}% Entropy={ent:.0}%\n\
         Dim: {dim}  Noise: {noise:.1}/{thresh:.1}  Extract: {ext:.0}%  Explored: {exp:.0}%\n\
         {hint}",
        stress.arousal * 100.0,
        biometrics.model.allostatic_load * 100.0,
        phase = phase_str,
        sanctuary = sanctuary_str,
        consciousness = consciousness_str,
        harm = harmony_str,
        frags = collected.total(),
        energy = energy_str,
        pow = settlement.power * 100.0,
        wat = settlement.water * 100.0,
        trust = settlement.trust * 100.0,
        ent = settlement.entropy * 100.0,
        dim = dim_str,
        noise = leviathan.noise_accumulator,
        thresh = leviathan.threshold,
        ext = extraction * 100.0,
        exp = explored.explore_percent(),
        hint = hint,
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
    let stress_red = (0.02f64 + load as f64 * 0.1f64 + danger as f64 * 0.15f64).min(0.3f64);
    let base_blue = (0.04f64 - danger as f64 * 0.03f64).max(0.01f64);
    clear_color.0 = Color::srgb(stress_red as f32, 0.02f32, base_blue as f32);
}

/// Camera follows player smoothly.
///
/// IMPORTANT: We use Camera2d (orthographic). Rotating a 2D camera makes sprites
/// edge-on = invisible (black screen). Instead we differentiate dimension modes via:
/// - 2D: tight zoom, top-down (default)
/// - 2.5D: zoomed out, slight Y offset to simulate depth
/// - 3D: further zoom, wider view reveals more dungeon
/// - 4D: same as 3D + W-slider reveals hidden rooms (via four_d_rendering.rs)
///
/// True 3D rendering with perspective projection requires Camera3d + bevy_pbr.
/// The consciousness-physics engine runs ND regardless of rendering mode.
pub fn camera_follow_system(
    player: Query<&Transform, With<Player>>,
    mut camera: Query<&mut Transform, (With<Camera2d>, Without<Player>)>,
    dim: Res<crate::systems::dimension_transition::DimensionTransition>,
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

    let distance = dim.effective_distance();

    // Camera zoom via Z height (higher = sees more)
    // 2D: 999 (tight), 2.5D: 1200, 3D/4D: 1600
    let base_z = match dim.target {
        crate::systems::dimension_transition::DimensionMode::D2 => 999.0,
        crate::systems::dimension_transition::DimensionMode::D2Half => 1200.0,
        crate::systems::dimension_transition::DimensionMode::D3 => 1600.0,
        crate::systems::dimension_transition::DimensionMode::D4 => 1600.0,
    };
    let target_z = base_z * distance;

    // Slight Y offset in 2.5D+ to simulate "looking from above and behind"
    let y_offset = match dim.target {
        crate::systems::dimension_transition::DimensionMode::D2 => 0.0,
        crate::systems::dimension_transition::DimensionMode::D2Half => -60.0,
        crate::systems::dimension_transition::DimensionMode::D3 => -120.0,
        crate::systems::dimension_transition::DimensionMode::D4 => -120.0,
    };

    cam_tf.translation.x = smoothed.x;
    cam_tf.translation.y = smoothed.y + y_offset;
    // Smooth Z transition
    cam_tf.translation.z += (target_z - cam_tf.translation.z) * 0.05;
    // No rotation — camera always looks straight down (sprites stay visible)
    cam_tf.rotation = Quat::IDENTITY;
}

/// Helper to draw a 2D arrow using gizmos
fn draw_arrow_2d(gizmos: &mut Gizmos, start: Vec2, end: Vec2, color: Color) {
    gizmos.line_2d(start, end, color);
    let dir = (start - end).normalize_or_zero();
    if dir == Vec2::ZERO {
        return;
    }
    let length = 8.0;
    let angle = 0.52_f32; // ~30 degrees
    let left = Vec2::new(
        dir.x * angle.cos() - dir.y * angle.sin(),
        dir.x * angle.sin() + dir.y * angle.cos(),
    ) * length;
    let right = Vec2::new(
        dir.x * angle.cos() + dir.y * angle.sin(),
        -dir.x * angle.sin() + dir.y * angle.cos(),
    ) * length;
    gizmos.line_2d(end, end + left, color);
    gizmos.line_2d(end, end + right, color);
}

/// Helper to draw a 3D arrow using gizmos
fn draw_arrow_3d(gizmos: &mut Gizmos, start: Vec3, end: Vec3, color: Color) {
    gizmos.line(start, end, color);
    let dir = (start - end).normalize_or_zero();
    if dir == Vec3::ZERO {
        return;
    }
    let length = 8.0;
    let up = Vec3::Y;
    let right_vec = dir.cross(up).normalize_or_zero();
    let up_vec = if right_vec == Vec3::ZERO {
        Vec3::X
    } else {
        right_vec.cross(dir).normalize_or_zero()
    };

    let left = (dir * 0.866 + right_vec * 0.5) * length;
    let right = (dir * 0.866 - right_vec * 0.5) * length;
    let up_p = (dir * 0.866 + up_vec * 0.5) * length;
    let down_p = (dir * 0.866 - up_vec * 0.5) * length;

    gizmos.line(end, end + left, color);
    gizmos.line(end, end + right, color);
    gizmos.line(end, end + up_p, color);
    gizmos.line(end, end + down_p, color);
}

/// Configuration for the Field Deck diagnostic overlay
#[derive(Resource)]
pub struct FieldDeckConfig {
    pub overlay_enabled: bool,
}

impl Default for FieldDeckConfig {
    fn default() -> Self {
        Self {
            overlay_enabled: true,
        }
    }
}

/// Toggles Field Deck Diagnostic Overlay via F3 or KeyF
pub fn field_deck_toggle_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut config: ResMut<FieldDeckConfig>,
) {
    if keyboard.just_pressed(KeyCode::F3) || keyboard.just_pressed(KeyCode::KeyF) {
        config.overlay_enabled = !config.overlay_enabled;
    }
}

/// Spawns floating world-space completion barks when WorldFeedbackEvents are emitted
pub fn world_feedback_listener_system(
    mut commands: Commands,
    mut events: MessageReader<crate::components::WorldFeedbackEvent>,
) {
    for event in events.read() {
        commands.spawn((
            Text2d::new(event.message.clone()),
            TextFont {
                font_size: 10.0,
                ..default()
            },
            TextColor(event.color),
            Transform::from_xyz(event.position.x, event.position.y + 15.0, 15.0),
            crate::components::WorldFeedbackLabel {
                timer: Timer::from_seconds(2.0, TimerMode::Once),
            },
        ));
    }
}

/// Floats world feedback text upwards and fades it out
pub fn update_feedback_labels_system(
    mut commands: Commands,
    mut query: Query<(
        Entity,
        &mut Transform,
        &mut TextColor,
        &mut crate::components::WorldFeedbackLabel,
    )>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    for (entity, mut tf, mut color, mut label) in &mut query {
        label.timer.tick(time.delta());
        if label.timer.is_finished() {
            commands.entity(entity).despawn();
        } else {
            tf.translation.y += 20.0 * dt;
            let progress = label.timer.fraction_remaining();
            color.0.set_alpha(progress);
        }
    }
}

/// Field Deck Diagnostic Visualizer drawing active inference indicators in world space
pub fn gizmo_telemetry_debug_system(
    mut gizmos: Gizmos,
    npcs: Query<(
        &CrewNpc,
        &Transform,
        &MoveTarget,
        &symtropy_render_bridge::PhysicsBody,
    )>,
    player_query: Query<&Transform, With<Player>>,
    physics: Res<PhysicsWorldRes>,
    phase_state: Res<State<crate::resources::GamePhase>>,
    field_deck: Res<FieldDeckConfig>,
    power_junctions: Query<(&Transform, &PowerJunction)>,
    water_pumps: Query<(&Transform, &WaterPump)>,
) {
    if !field_deck.overlay_enabled {
        return;
    }

    let is_3d = *phase_state.get() == crate::resources::GamePhase::Playing3D;

    let Some(player_tf) = player_query.iter().next() else {
        return;
    };
    let player_pos = player_tf.translation;

    // Draw crew diagnostic overlays
    for (_npc, npc_tf, move_target, body) in &npcs {
        let npc_pos = npc_tf.translation;

        let (prediction_error, npc_phi) = physics
            .consciousness
            .entities
            .get(&body.handle)
            .map(|e| (e.prediction_error, e.phi()))
            .unwrap_or((0.0, 0.5));

        // 1. Surprise Gradient (Red, pointing upwards)
        let height = (prediction_error as f32 * 50.0).clamp(5.0, 150.0);
        let ray_color = Color::srgb(1.0, 0.1, 0.1);
        if is_3d {
            gizmos.line(npc_pos, npc_pos + Vec3::new(0.0, 0.0, height), ray_color);
        } else {
            gizmos.line_2d(
                npc_pos.truncate(),
                npc_pos.truncate() + Vec2::new(0.0, height),
                ray_color,
            );
        }

        // 2. Phi Cohesion Thread (Thin ice-blue lines connecting player to crew, fading as phi drops)
        let phi_color = Color::linear_rgba(0.4, 0.8, 1.0, npc_phi as f32);
        if is_3d {
            gizmos.line(
                player_pos + Vec3::new(0.0, 0.0, 2.0),
                npc_pos + Vec3::new(0.0, 0.0, 2.0),
                phi_color,
            );
        } else {
            gizmos.line_2d(player_pos.truncate(), npc_pos.truncate(), phi_color);
        }

        // 3. Action Intention Vector (Green arrows pointing to move target)
        if let Some(target) = move_target.target {
            let target_pos = if is_3d {
                Vec3::new(target.x, target.y, npc_pos.z)
            } else {
                Vec3::new(target.x, target.y, 0.0)
            };
            let dir_color = Color::srgb(0.2, 0.9, 0.3);
            if is_3d {
                draw_arrow_3d(&mut gizmos, npc_pos, target_pos, dir_color);
            } else {
                draw_arrow_2d(
                    &mut gizmos,
                    npc_pos.truncate(),
                    target_pos.truncate(),
                    dir_color,
                );
            }
        }
    }

    // 4. Infrastructure Distress Ray (Amber/Orange vertical lines from damaged/sabotaged machines)
    let distress_color = Color::srgb(0.9, 0.5, 0.1);
    for (j_tf, junction) in &power_junctions {
        if junction.is_damaged {
            let j_pos = j_tf.translation;
            if is_3d {
                gizmos.line(j_pos, j_pos + Vec3::new(0.0, 0.0, 40.0), distress_color);
            } else {
                gizmos.line_2d(
                    j_pos.truncate(),
                    j_pos.truncate() + Vec2::new(0.0, 40.0),
                    distress_color,
                );
            }
        }
    }
    for (p_tf, pump) in &water_pumps {
        if pump.is_sabotaged {
            let p_pos = p_tf.translation;
            if is_3d {
                gizmos.line(p_pos, p_pos + Vec3::new(0.0, 0.0, 40.0), distress_color);
            } else {
                gizmos.line_2d(
                    p_pos.truncate(),
                    p_pos.truncate() + Vec2::new(0.0, 40.0),
                    distress_color,
                );
            }
        }
    }
}
