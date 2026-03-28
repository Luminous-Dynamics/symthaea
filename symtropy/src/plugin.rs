// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symtropy game plugin: wires all systems together.

use bevy::prelude::*;
use crate::resources::{BiometricsCtx, GamePhase, GovernanceLog, LeviathanState};
use crate::systems;

pub struct SymtropyPlugin;

impl Plugin for SymtropyPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_state::<GamePhase>()
            .init_resource::<BiometricsCtx>()
            .init_resource::<LeviathanState>()
            .init_resource::<GovernanceLog>()
            .init_resource::<systems::consciousness::PlayerConsciousness>()
            .init_resource::<systems::rendering::TelemetryTimer>()
            .init_resource::<systems::postprocess::ConsciousnessVisuals>()
            .init_resource::<systems::postprocess::CameraTrauma>()
            .insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.04)))
            .init_resource::<crate::resources::DungeonSeed>()
            .init_resource::<systems::minimap::ExploredTiles>()
            .init_resource::<systems::harmonies::LocalHarmonyState>()
            .init_resource::<systems::scavenge::CollectedPrimitives>()
            .init_resource::<systems::audio::AudioState>()
            .add_systems(Startup, systems::audio::setup_audio)
            .add_systems(OnEnter(GamePhase::MainMenu), systems::menu::setup_menu)
            .add_systems(Update, systems::menu::menu_input_system.run_if(in_state(GamePhase::MainMenu)))
            .add_systems(OnExit(GamePhase::MainMenu), systems::menu::cleanup_menu)
            .add_systems(OnEnter(GamePhase::Loading), (
                systems::menu::setup_loading, systems::rendering::setup_world,
                systems::minimap::setup_minimap, systems::scavenge::spawn_scavenge_items,
            ).chain())
            .add_systems(Update, auto_start.run_if(in_state(GamePhase::Loading)))
            .add_systems(OnExit(GamePhase::Loading), systems::menu::cleanup_loading)
            .add_systems(Update, (
                systems::input::input_system, systems::player::player_movement_system,
                systems::player::flashlight_system, systems::player::extraction_system,
                systems::fep_behavior::fep_behavior_system, systems::fep_behavior::npc_movement_system,
            ).chain().run_if(in_state(GamePhase::Playing)))
            .add_systems(Update, (
                systems::leviathan::leviathan_system, systems::leviathan::victory_check_system,
                systems::audio::audio_system,
                systems::postprocess::update_consciousness_visuals,
                systems::postprocess::trauma_feed_system, systems::postprocess::camera_shake_system,
                systems::rendering::camera_follow_system, systems::rendering::visual_stress_system,
                systems::rendering::leviathan_visual_system,
                systems::harmonies::harmony_update_system, systems::harmonies::harmony_visual_system,
                systems::harmonies::sanctuary_system, systems::scavenge::scavenge_pickup_system,
                systems::consciousness::player_consciousness_system,
                systems::consciousness::npc_consciousness_system,
                systems::rendering::hud_system, systems::minimap::update_minimap,
            ).chain().run_if(in_state(GamePhase::Playing)))
            .add_systems(Update, game_over.run_if(in_state(GamePhase::GameOver)))
            .add_systems(Update, victory.run_if(in_state(GamePhase::Victory)));
    }
}

fn auto_start(mut s: ResMut<NextState<GamePhase>>) { eprintln!("[symtropy] Loading → Playing"); s.set(GamePhase::Playing); }

fn game_over(kb: Res<ButtonInput<KeyCode>>, mut s: ResMut<NextState<GamePhase>>, mut l: ResMut<LeviathanState>, mut b: ResMut<BiometricsCtx>, mut cc: ResMut<ClearColor>, mut hud: Query<(&mut Text, &mut TextColor), With<systems::rendering::HudText>>, mut logged: Local<bool>) {
    if !*logged { warn!("THE LEVIATHAN HAS CAUGHT YOU"); cc.0 = Color::srgb(0.3,0.02,0.02); for (mut t,mut c) in &mut hud { **t = "THE LEVIATHAN HAS CAUGHT YOU\n\nR: restart | Esc: quit".into(); *c = TextColor(Color::srgb(1.0,0.3,0.2)); } *logged = true; }
    if kb.just_pressed(KeyCode::KeyR) { *l = LeviathanState::default(); b.encoder.reset(); b.model.reset(); cc.0 = Color::srgb(0.02,0.02,0.04); *logged = false; s.set(GamePhase::MainMenu); }
    if kb.just_pressed(KeyCode::Escape) { std::process::exit(0); }
}

fn victory(kb: Res<ButtonInput<KeyCode>>, mut s: ResMut<NextState<GamePhase>>, mut l: ResMut<LeviathanState>, mut b: ResMut<BiometricsCtx>, mut cc: ResMut<ClearColor>, mut hud: Query<(&mut Text, &mut TextColor), With<systems::rendering::HudText>>, mut logged: Local<bool>) {
    if !*logged { info!("FUSION CORE EXTRACTED — YOU SURVIVED"); cc.0 = Color::srgb(0.02,0.08,0.02); for (mut t,mut c) in &mut hud { **t = "FUSION CORE EXTRACTED!\n\nR: play again | Esc: quit".into(); *c = TextColor(Color::srgb(0.3,1.0,0.4)); } *logged = true; }
    if kb.just_pressed(KeyCode::KeyR) { *l = LeviathanState::default(); b.encoder.reset(); b.model.reset(); cc.0 = Color::srgb(0.02,0.02,0.04); *logged = false; s.set(GamePhase::MainMenu); }
    if kb.just_pressed(KeyCode::Escape) { std::process::exit(0); }
}
