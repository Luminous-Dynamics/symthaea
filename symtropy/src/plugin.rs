// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symtropy game plugin: wires all systems together.

use bevy::prelude::*;

use crate::resources::{BiometricsCtx, GamePhase, GovernanceLog, LeviathanState};
use crate::systems;
// TODO: re-enable when multiworld-sim compiles
// use symtropy_sim_bridge::SimBridgePlugin;

/// Main game plugin. Add this to a Bevy `App` to get the full Symtropy experience.
pub struct SymtropyPlugin;

impl Plugin for SymtropyPlugin {
    fn build(&self, app: &mut App) {
        app
            // State
            .init_state::<GamePhase>()
            // TODO: re-enable sim-bridge when multiworld-sim compiles
            // .add_plugins(SimBridgePlugin)
            // Resources
            .init_resource::<BiometricsCtx>()
            .init_resource::<LeviathanState>()
            .init_resource::<GovernanceLog>()
            .init_resource::<systems::rendering::TelemetryTimer>()
            // Dark background — the void outside the Markov blanket
            .insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.04)))
            .init_resource::<FrameCounter>()
            // Startup systems
            .add_systems(Startup, (
                systems::rendering::setup_world,
                systems::audio::setup_audio,
            ))
            // Gameplay systems (run during Playing state)
            .add_systems(
                Update,
                (
                    // Input capture (must run first)
                    systems::input::input_system,
                    // Player systems
                    systems::player::player_movement_system,
                    systems::player::flashlight_system,
                    systems::player::extraction_system,
                    // NPC AI
                    systems::fep_behavior::fep_behavior_system,
                    systems::fep_behavior::npc_movement_system,
                    // TODO: governance sim tick (re-enable with sim-bridge)
                    // symtropy_sim_bridge::governance_tick_system,
                    // Leviathan
                    systems::leviathan::leviathan_system,
                    systems::leviathan::victory_check_system,
                    // Audio (reads stress + leviathan)
                    systems::audio::audio_system,
                    // Rendering + visuals
                    systems::rendering::camera_follow_system,
                    systems::rendering::visual_stress_system,
                    systems::rendering::leviathan_visual_system,
                    systems::rendering::hud_system,
                )
                    .chain()
                    .run_if(in_state(GamePhase::Playing)),
            )
            // Frame counter for startup diagnostics
            .add_systems(Update, frame_counter_system)
            // Auto-transition Loading → Playing after one frame
            .add_systems(
                Update,
                auto_start.run_if(in_state(GamePhase::Loading)),
            )
            // GameOver: log and allow restart with R
            .add_systems(
                Update,
                game_over_system.run_if(in_state(GamePhase::GameOver)),
            )
            // Victory: log and allow restart with R
            .add_systems(
                Update,
                victory_system.run_if(in_state(GamePhase::Victory)),
            );
    }
}

#[derive(Resource, Default)]
struct FrameCounter(u64);

fn frame_counter_system(
    mut counter: ResMut<FrameCounter>,
    time: Res<Time>,
    sprites: Query<(&Transform, &Sprite)>,
    cameras: Query<&Transform, With<Camera2d>>,
) {
    counter.0 += 1;
    if counter.0 <= 10 || counter.0 % 60 == 0 {
        eprintln!(
            "[symtropy] frame={} dt={:.1}ms elapsed={:.1}s",
            counter.0,
            time.delta_secs() * 1000.0,
            time.elapsed_secs(),
        );
    }
    // Dump scene info on frame 5 — exactly once
    if counter.0 == 5 {
        eprintln!("[symtropy] === SCENE DIAGNOSTIC ===");
        eprintln!("[symtropy] Total sprites: {}", sprites.iter().count());
        for tf in cameras.iter() {
            eprintln!(
                "[symtropy] Camera2d at ({:.0},{:.0},{:.0})",
                tf.translation.x, tf.translation.y, tf.translation.z,
            );
        }
        // Sample sprites — first 3, last 2, plus any large ones (player/core)
        let all: Vec<_> = sprites.iter().collect();
        for (i, (tf, sprite)) in all.iter().enumerate() {
            if i < 3 || i >= all.len() - 3 || sprite.custom_size.map_or(false, |s| s.x > 20.0) {
                eprintln!(
                    "[symtropy]   [{}] pos=({:.0},{:.0},{:.0}) size={:?} color={:?}",
                    i, tf.translation.x, tf.translation.y, tf.translation.z,
                    sprite.custom_size, sprite.color,
                );
            }
        }
        eprintln!("[symtropy] === END DIAGNOSTIC ===");
    }
}

fn auto_start(mut next_state: ResMut<NextState<GamePhase>>) {
    eprintln!("[symtropy] Loading → Playing");
    next_state.set(GamePhase::Playing);
}

fn game_over_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<GamePhase>>,
    mut leviathan: ResMut<LeviathanState>,
    mut biometrics: ResMut<BiometricsCtx>,
    mut clear_color: ResMut<ClearColor>,
    mut hud: Query<(&mut Text, &mut TextColor), With<crate::systems::rendering::HudText>>,
    mut logged: Local<bool>,
) {
    if !*logged {
        warn!("=== THE LEVIATHAN HAS CAUGHT YOU ===");
        warn!("Press R to restart, Escape to quit");
        // Flash screen red
        clear_color.0 = Color::srgb(0.3, 0.02, 0.02);
        for (mut text, mut color) in &mut hud {
            **text = "THE LEVIATHAN HAS CAUGHT YOU\n\nPress R to restart | Escape to quit".into();
            *color = TextColor(Color::srgb(1.0, 0.3, 0.2));
        }
        *logged = true;
    }
    if keyboard.just_pressed(KeyCode::KeyR) {
        *leviathan = LeviathanState::default();
        biometrics.encoder.reset();
        biometrics.model.reset();
        clear_color.0 = Color::srgb(0.02, 0.02, 0.04);
        *logged = false;
        next_state.set(GamePhase::Playing);
        info!("Restarting...");
    }
    if keyboard.just_pressed(KeyCode::Escape) {
        std::process::exit(0);
    }
}

fn victory_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<GamePhase>>,
    mut leviathan: ResMut<LeviathanState>,
    mut biometrics: ResMut<BiometricsCtx>,
    mut clear_color: ResMut<ClearColor>,
    mut hud: Query<(&mut Text, &mut TextColor), With<crate::systems::rendering::HudText>>,
    mut logged: Local<bool>,
) {
    if !*logged {
        info!("=== FUSION CORE EXTRACTED — YOU SURVIVED ===");
        clear_color.0 = Color::srgb(0.02, 0.08, 0.02);
        for (mut text, mut color) in &mut hud {
            **text = "FUSION CORE EXTRACTED — YOU SURVIVED!\n\nPress R to play again | Escape to quit".into();
            *color = TextColor(Color::srgb(0.3, 1.0, 0.4));
        }
        *logged = true;
    }
    if keyboard.just_pressed(KeyCode::KeyR) {
        *leviathan = LeviathanState::default();
        biometrics.encoder.reset();
        biometrics.model.reset();
        clear_color.0 = Color::srgb(0.02, 0.02, 0.04);
        *logged = false;
        next_state.set(GamePhase::Playing);
    }
    if keyboard.just_pressed(KeyCode::Escape) {
        std::process::exit(0);
    }
}
