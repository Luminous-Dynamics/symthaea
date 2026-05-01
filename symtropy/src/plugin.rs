// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symtropy game plugin: wires all systems together.

use crate::resources::{
    BiometricsCtx, GamePhase, GovernanceLog, LeviathanState, PhysicsWorldRes, PlayerInput,
};
use crate::systems;
use bevy::prelude::*;

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
            .init_resource::<crate::experience::ExperienceRegistry>()
            .init_resource::<systems::harmonies::LocalHarmonyState>()
            .init_resource::<systems::scavenge::CollectedPrimitives>()
            .init_resource::<systems::audio::AudioState>()
            .init_resource::<systems::room_memory::RoomMemory>()
            .init_resource::<systems::dialogue::DialogueTimer>()
            // Physics engine
            .init_resource::<PhysicsWorldRes>()
            .init_resource::<PlayerInput>()
            .init_resource::<systems::thermodynamic::ThermodynamicHudState>()
            .init_resource::<systems::dimension_transition::DimensionTransition>()
            .init_resource::<systems::four_d_rendering::FourDProjector>()
            .init_resource::<systems::living_dungeon::DungeonBreathTimer>()
            .init_resource::<systems::psychology::PsychologyTimer>()
            .init_resource::<systems::consciousness_aura::ResonanceWaveTimer>()
            .init_resource::<systems::dimensional_leakage::LeakageTimer>()
            // FixedUpdate: physics + thermodynamic enforcement at consistent 64Hz
            .add_systems(FixedUpdate, (
                systems::engine_physics::physics_apply_inputs,
                systems::thermodynamic::thermodynamic_enforcement_system,
                systems::engine_physics::physics_step,
                systems::engine_physics::physics_sync_transforms,
            ).chain().run_if(in_state(GamePhase::Playing)))
            .add_systems(Startup, systems::audio::setup_audio)
            .add_systems(OnEnter(GamePhase::MainMenu), systems::menu::setup_menu)
            .add_systems(Update, (
                systems::menu::menu_input_system,
                systems::menu::nexus_background_system,
            ).run_if(in_state(GamePhase::MainMenu)))
            .add_systems(OnExit(GamePhase::MainMenu), systems::menu::cleanup_menu)
            .add_systems(OnEnter(GamePhase::Loading), (
                systems::menu::setup_loading, systems::rendering::setup_world,
                systems::minimap::setup_minimap, systems::scavenge::spawn_scavenge_items,
                systems::four_d_rendering::spawn_four_d_secrets,
                systems::four_d_rendering::assign_player_four_d,
                systems::dimensional_leakage::spawn_leakage_points,
            ).chain())
            .add_systems(Update, auto_start.run_if(in_state(GamePhase::Loading)))
            .add_systems(OnExit(GamePhase::Loading), systems::menu::cleanup_loading)
            .add_systems(Update, (
                systems::dimension_transition::dimension_input_system,
                systems::dimension_transition::dimension_transition_system,
                systems::four_d_rendering::four_d_projector_sync_system,
                systems::four_d_rendering::four_d_visibility_system,
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
            ).chain().run_if(in_state(GamePhase::Playing)))
            .add_systems(Update, (
                systems::psychology::psychology_tick_system,
                systems::consciousness::player_consciousness_system,
                systems::consciousness::npc_consciousness_system,
                systems::engine_physics::consciousness_sync_system,
                systems::thermodynamic::collapse_visual_system,
                systems::rendering::hud_system, systems::minimap::update_minimap,
                systems::room_memory::room_memory_update_system,
                systems::dialogue::dialogue_system,
                systems::living_dungeon::living_dungeon_system,
            ).chain().run_if(in_state(GamePhase::Playing)))
            // Consciousness aura + resonance wave visual manifold
            .add_systems(Update, (
                systems::consciousness_aura::spawn_auras,
                systems::consciousness_aura::aura_update_system,
                systems::consciousness_aura::resonance_wave_system,
                systems::consciousness_aura::resonance_wave_animate_system,
                systems::consciousness_aura::consciousness_perception_gate_system,
                systems::dimensional_leakage::dimensional_leakage_system,
                systems::dimensional_leakage::leakage_visual_system,
            ).chain().run_if(in_state(GamePhase::Playing)))
            // Mycelix physicalized cryptography (only when --features mycelix)
            ;
        // Sol Atlas globe view (only when --features atlas)
        #[cfg(feature = "atlas")]
        {
            app.add_plugins(bevy::pbr::MaterialPlugin::<
                sol_atlas_bevy::holographic_material::HolographicMaterial,
            >::default())
                .init_resource::<sol_atlas_bevy::camera::OrbitalCameraConfig>()
                .init_resource::<sol_atlas_bevy::timeline::TimelineState>()
                .init_resource::<sol_atlas_bevy::selection::SelectedMarker>()
                .init_resource::<systems::atlas::CurrentAesthetic>()
                .init_resource::<systems::atlas::DataView>()
                .init_resource::<systems::atlas::OverlayManager>()
                .init_resource::<sol_atlas_bevy::frame_capture::FrameCaptureConfig>()
                .init_resource::<systems::demo_director::DemoDirector>()
                .add_systems(
                    Update,
                    systems::atlas::atlas_toggle_system.run_if(in_state(GamePhase::Playing)),
                )
                .add_systems(
                    OnEnter(GamePhase::GlobeView),
                    (
                        systems::atlas::setup_globe_view,
                        sol_atlas_bevy::selection::setup_selection_ui,
                    ),
                )
                // Visibility pipeline: timeline → temporal → filter (includes LOD)
                .add_systems(
                    Update,
                    (
                        systems::atlas::timeline_visibility_system,
                        systems::atlas::temporal_4d_system,
                        systems::atlas::data_view_filter_system, // combined LOD + view filter
                    )
                        .chain()
                        .run_if(in_state(GamePhase::GlobeView)),
                )
                // Globe systems — split into two groups (Bevy tuple size limit)
                .add_systems(
                    Update,
                    (
                        systems::atlas::globe_input_system,
                        systems::atlas::draw_arcs_system,
                        sol_atlas_bevy::camera::orbital_camera_system,
                        sol_atlas_bevy::timeline::timeline_input_system,
                        sol_atlas_bevy::timeline::timeline_autoplay_system,
                        systems::atlas::celestial_orbit_system,
                        systems::atlas::holographic_pulse_system,
                        systems::audio::globe_audio_system,
                        systems::atlas::cloud_rotation_system,
                        systems::atlas::marker_pulse_system,
                        systems::atlas::consciousness_shader_system,
                        systems::atlas::planet_focus_system,
                        systems::atlas::overlay_toggle_system,
                    )
                        .run_if(in_state(GamePhase::GlobeView)),
                )
                .add_systems(
                    Update,
                    (
                        systems::atlas::city_stress_evolution_system,
                        systems::atlas::timeline_hud_system,
                        systems::atlas::data_view_switch_system,
                        systems::atlas::panel_metrics_system,
                        systems::atlas::timeline_scrubber_system,
                        systems::atlas::aesthetic_switch_system,
                        systems::atlas::aesthetic_apply_system,
                        sol_atlas_bevy::selection::click_select_system,
                        sol_atlas_bevy::selection::update_selection_text,
                        sol_atlas_bevy::holographic_material::update_holographic_time,
                        sol_atlas_bevy::frame_capture::frame_capture_system,
                        systems::demo_director::demo_director_system,
                    )
                        .run_if(in_state(GamePhase::GlobeView)),
                )
                .add_systems(
                    OnExit(GamePhase::GlobeView),
                    systems::atlas::cleanup_globe_view,
                )
                // Cinematic director (global — runs across ALL game phases)
                .add_systems(
                    Startup,
                    systems::cinematic_director::setup_cinematic_overlay
                        .run_if(resource_exists::<systems::cinematic_director::CinematicDirector>),
                )
                .add_systems(
                    Update,
                    (
                        systems::cinematic_director::cinematic_director_system,
                        systems::cinematic_director::screen_fade_system,
                        systems::cinematic_director::narration_system,
                        systems::cinematic_director::voice_narration_system,
                        systems::cinematic_director::orbital_tracks_system,
                    )
                        .run_if(resource_exists::<systems::cinematic_director::CinematicDirector>),
                );
        }

        #[cfg(feature = "mycelix")]
        {
            use symtropy_sim_bridge::SimBridgePlugin;
            app.add_plugins(SimBridgePlugin)
                .init_resource::<systems::fl_simulation::FlPool>()
                .init_resource::<systems::epistemics::PlayerEpistemicState>()
                .init_resource::<systems::dkg_ceremony::DkgCeremonyState>()
                .init_resource::<systems::medical_commons::MedicalBayState>()
                .add_systems(
                    Update,
                    (
                        symtropy_sim_bridge::governance_tick_system,
                        systems::fl_simulation::fl_aggregation_system,
                        systems::fl_simulation::byzantine_effect_system,
                        systems::epistemics::epistemic_flashlight_system,
                        systems::epistemics::epistemic_advancement_system,
                        systems::dkg_ceremony::dkg_ceremony_system,
                        systems::medical_commons::medical_commons_system,
                        systems::medical_commons::data_dividend_system,
                        systems::medical_commons::coercion_detection_system,
                    )
                        .run_if(in_state(GamePhase::Playing)),
                )
                .add_systems(
                    Update,
                    (
                        systems::governance::governance_proposal_system,
                        systems::governance::governance_voting_system,
                        systems::governance::veto_override_system,
                        systems::governance::oppression_detection_system,
                        systems::governance::consciousness_evolution_system,
                        systems::economy::tend_exchange_system,
                        systems::economy::demurrage_system,
                        systems::economy::player_tend_interaction_system,
                        systems::faction::faction_emergence_system,
                        systems::faction::faction_recruitment_system,
                        systems::faction::faction_conflict_system,
                    )
                        .run_if(in_state(GamePhase::Playing)),
                );
        }
        app.add_systems(Update, game_over.run_if(in_state(GamePhase::GameOver)))
            .add_systems(Update, victory.run_if(in_state(GamePhase::Victory)));
    }
}

fn auto_start(mut s: ResMut<NextState<GamePhase>>) {
    eprintln!("[symtropy] Loading → Playing");
    s.set(GamePhase::Playing);
}

fn game_over(
    kb: Res<ButtonInput<KeyCode>>,
    mut s: ResMut<NextState<GamePhase>>,
    mut l: ResMut<LeviathanState>,
    mut b: ResMut<BiometricsCtx>,
    mut cc: ResMut<ClearColor>,
    mut hud: Query<(&mut Text, &mut TextColor), With<systems::rendering::HudText>>,
    mut logged: Local<bool>,
) {
    if !*logged {
        warn!("THE LEVIATHAN HAS CAUGHT YOU");
        cc.0 = Color::srgb(0.3, 0.02, 0.02);
        for (mut t, mut c) in &mut hud {
            **t = "THE LEVIATHAN HAS CAUGHT YOU\n\nR: restart | Esc: quit".into();
            *c = TextColor(Color::srgb(1.0, 0.3, 0.2));
        }
        *logged = true;
    }
    if kb.just_pressed(KeyCode::KeyR) {
        *l = LeviathanState::default();
        b.encoder.reset();
        b.model.reset();
        cc.0 = Color::srgb(0.02, 0.02, 0.04);
        *logged = false;
        s.set(GamePhase::MainMenu);
    }
    if kb.just_pressed(KeyCode::Escape) {
        std::process::exit(0);
    }
}

fn victory(
    kb: Res<ButtonInput<KeyCode>>,
    mut s: ResMut<NextState<GamePhase>>,
    mut l: ResMut<LeviathanState>,
    mut b: ResMut<BiometricsCtx>,
    mut cc: ResMut<ClearColor>,
    mut hud: Query<(&mut Text, &mut TextColor), With<systems::rendering::HudText>>,
    mut logged: Local<bool>,
) {
    if !*logged {
        info!("FUSION CORE EXTRACTED — YOU SURVIVED");
        cc.0 = Color::srgb(0.02, 0.08, 0.02);
        for (mut t, mut c) in &mut hud {
            **t = "FUSION CORE EXTRACTED!\n\nR: play again | Esc: quit".into();
            *c = TextColor(Color::srgb(0.3, 1.0, 0.4));
        }
        *logged = true;
    }
    if kb.just_pressed(KeyCode::KeyR) {
        *l = LeviathanState::default();
        b.encoder.reset();
        b.model.reset();
        cc.0 = Color::srgb(0.02, 0.02, 0.04);
        *logged = false;
        s.set(GamePhase::MainMenu);
    }
    if kb.just_pressed(KeyCode::Escape) {
        std::process::exit(0);
    }
}
