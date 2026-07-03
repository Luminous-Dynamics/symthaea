// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symtropy game plugin: wires all systems together.

use crate::components;
use crate::resources::{
    BiometricsCtx, GamePhase, GovernanceLog, LeviathanState, PhysicsWorldRes, PlayerInput,
};
use crate::systems;
use bevy::app::AppExit;
use bevy::prelude::*;

pub struct SymtropyPlugin;

impl Plugin for SymtropyPlugin {
    fn build(&self, app: &mut App) {
        app
            .init_state::<GamePhase>()
            .init_resource::<BiometricsCtx>()
            .init_resource::<LeviathanState>()
            .init_resource::<crate::resources::SettlementMetrics>()
            .init_resource::<crate::resources::GovernanceVote>()
            .init_resource::<GovernanceLog>()
            .insert_resource(crate::ports::ActivePorts::local_mock())
            .init_resource::<systems::consciousness::PlayerConsciousness>()
            .init_resource::<systems::rendering::TelemetryTimer>()
            .init_resource::<systems::rendering::FieldDeckConfig>()
            .init_resource::<systems::postprocess::ConsciousnessVisuals>()
            .init_resource::<systems::postprocess::CameraTrauma>()
            .insert_resource(ClearColor(Color::srgb(0.02, 0.02, 0.04)))
            .init_resource::<crate::resources::DungeonSeed>()
            .init_resource::<crate::resources::SiteLayout>()
            .init_resource::<systems::minimap::ExploredTiles>()
            .init_resource::<crate::experience::ExperienceRegistry>()
            .add_plugins(systems::muse::MusePlugin)
            .add_plugins(symtropy_mycelix_village::city_scale_logic::CityScalePlugin { state: GamePhase::CityScale })
            .init_resource::<systems::harmonies::LocalHarmonyState>()
            .init_resource::<systems::scavenge::CollectedPrimitives>()
            .init_resource::<systems::audio::AudioState>()
            .init_resource::<systems::room_memory::RoomMemory>()
            .init_resource::<systems::dialogue::DialogueTimer>()
            // Physics engine
            .init_resource::<PhysicsWorldRes>()
            .init_resource::<PlayerInput>()
            .init_resource::<crate::resources::TutorialScenarioRes>()
            .init_resource::<systems::thermodynamic::ThermodynamicHudState>()
            .init_resource::<systems::dimension_transition::DimensionTransition>()
            .init_resource::<systems::four_d_rendering::FourDProjector>()
            .init_resource::<systems::living_dungeon::DungeonBreathTimer>()
            .init_resource::<systems::psychology::PsychologyTimer>()
            .init_resource::<systems::psychology::CrewSavedCount>()
            .init_resource::<systems::consciousness_aura::ResonanceWaveTimer>()
            .init_resource::<systems::dimensional_leakage::LeakageTimer>()
            .init_resource::<symtropy_render_bridge::TelemetryBufferResource>()
            .add_plugins(symtropy_render_bridge::TelemetryMaterialPlugin)
            .add_plugins(symtropy_render_bridge::NdSlicingPlugin)
            .add_message::<components::NpcActionEvent>()
            .add_message::<components::WorldFeedbackEvent>()
            // FixedUpdate: physics + thermodynamic enforcement at consistent 64Hz.
            // physics_apply_inputs/physics_step were empty {} bodies wired in
            // here as if they did real stepping — actual movement happens via
            // direct kinematic Transform writes in each movement system
            // (player_movement_system_3d etc.), not through these. Removed
            // rather than left as misleading no-op placeholders.
            .add_systems(FixedUpdate, (
                systems::thermodynamic::thermodynamic_enforcement_system,
                systems::engine_physics::physics_sync_transforms,
            ).chain().run_if(in_playing_or_3d))
            .add_systems(Startup, (
                systems::audio::setup_audio,
                systems::telemetry::setup_telemetry_gpu_buffer,
            ))
            .add_systems(OnEnter(GamePhase::MainMenu), systems::menu::setup_menu)
            .add_systems(Update, (
                systems::menu::menu_input_system,
                systems::menu::nexus_background_system,
            ).run_if(in_state(GamePhase::MainMenu)))
            .add_systems(OnExit(GamePhase::MainMenu), systems::menu::cleanup_menu)
            .add_systems(OnEnter(GamePhase::Loading), (
                systems::menu::setup_loading,
                systems::procgen::generate_site_layout_system,
                systems::rendering::setup_world,
                systems::settlement_spawn::spawn_infrastructure_system,
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
                systems::four_d_rendering::four_d_material_sync_system,
                systems::player::player_movement_system,
            ).chain().run_if(in_state(GamePhase::Playing)))
            .add_systems(Update, (
                systems::input::input_system,
                systems::player::flashlight_system, systems::player::extraction_system,
                systems::fep_behavior::fep_behavior_system,
                systems::fep_behavior::npc_action_system,
                systems::fep_behavior::npc_movement_system,
                systems::rendering::gizmo_telemetry_debug_system,
                systems::rendering::update_feedback_labels_system,
                systems::rendering::field_deck_toggle_system,
                systems::rendering::world_feedback_listener_system,
                systems::tutorial_scenario::tutorial_scenario_system,
            ).chain().run_if(in_playing_or_3d))
            .add_systems(Update, (
                systems::rendering::camera_follow_system,
                systems::rendering::leviathan_visual_system,
            ).chain().run_if(in_state(GamePhase::Playing)))
            .add_systems(Update, (
                systems::leviathan::leviathan_system, systems::leviathan::victory_check_system,
                systems::audio::audio_system,
                systems::postprocess::update_consciousness_visuals,
                systems::postprocess::trauma_feed_system, systems::postprocess::camera_shake_system,
                systems::rendering::visual_stress_system,
                systems::harmonies::harmony_update_system, systems::harmonies::harmony_visual_system,
                systems::harmonies::sanctuary_system, systems::scavenge::scavenge_pickup_system,
            ).chain().run_if(in_playing_or_3d))
            .add_systems(Update, (
                systems::telemetry::hydrate_gpu_telemetry_system,
                systems::telemetry::sync_telemetry_to_gpu_system,
            ).chain().run_if(in_playing_or_3d))
            .add_systems(Update, (
                systems::psychology::psychology_tick_system,
                systems::psychology::npc_visual_state_system,
                systems::psychology::npc_collapse_recovery_system,
                systems::consciousness::player_consciousness_system,
                systems::consciousness::npc_consciousness_system,
                systems::thermodynamic::collapse_visual_system,
                systems::rendering::hud_system, systems::minimap::update_minimap,
                systems::room_memory::room_memory_update_system,
                systems::dialogue::dialogue_system,
                systems::dialogue::dialogue_action_bark_system,
                systems::living_dungeon::living_dungeon_system,
                systems::settlement::settlement_metric_update_system,
                systems::settlement::settlement_interaction_system,
                systems::settlement::npc_settlement_reaction_system,
                systems::settlement::settlement_governance_trigger_system,
                systems::null_ecology::null_drone_ai_system,
                systems::null_ecology::drone_combat_system,
                systems::null_ecology::drone_spawning_system,
            ).chain().run_if(in_playing_or_3d))
            .add_systems(Update, systems::settlement::council_ui_system.run_if(in_state(GamePhase::Council)))
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
            // Embodied 3D Layer systems (Milestone H1.5)
            .init_resource::<systems::rendering_3d::FirstPersonLook>()
            .add_systems(OnEnter(GamePhase::Playing3D), (
                systems::rendering_3d::setup_world_3d,
                systems::rendering_3d::fps_cursor_grab_system,
            ))
            .add_systems(OnExit(GamePhase::Playing3D), systems::rendering_3d::fps_cursor_release_system)
            .add_systems(Update, (
                systems::rendering_3d::fps_mouse_look_system,
                systems::rendering_3d::player_movement_system_3d,
                systems::rendering_3d::camera_follow_system_3d,
                systems::rendering_3d::leviathan_visual_system_3d,
            ).chain().run_if(in_state(GamePhase::Playing3D)))
            .add_systems(Update, fps_escape_to_menu.run_if(in_state(GamePhase::Playing3D)))
            // Mycelix physicalized cryptography (only when --features mycelix)
            ;
        // Sol Atlas globe view (only when --features atlas)
        #[cfg(feature = "atlas")]
        {
            app.add_plugins(bevy::pbr::MaterialPlugin::<
                sol_atlas_bevy::holographic_material::HolographicMaterial,
            >::default())
                .add_plugins(bevy::pbr::MaterialPlugin::<
                    sol_atlas_bevy::clouds::CloudMaterial,
                >::default())
                .init_resource::<sol_atlas_bevy::camera::OrbitalCameraConfig>()
                .init_resource::<sol_atlas_bevy::timeline::TimelineState>()
                .init_resource::<sol_atlas_bevy::selection::SelectedMarker>()
                .init_resource::<systems::atlas::CurrentAesthetic>()
                .init_resource::<systems::atlas::DataView>()
                .init_resource::<systems::atlas::OverlayManager>()
                .init_resource::<sol_atlas_bevy::frame_capture::FrameCaptureConfig>()
                .init_resource::<systems::demo_director::DemoDirector>()
                .init_resource::<sol_atlas_bevy::h3_grid::HoveredCell>()
                .init_resource::<sol_atlas_bevy::cell_entry::CellZoomTransition>()
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
                        systems::atlas::draw_gravity_grid_system,
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
                        sol_atlas_bevy::clouds::update_cloud_time,
                        sol_atlas_bevy::frame_capture::frame_capture_system,
                        systems::demo_director::demo_director_system,
                    )
                        .run_if(in_state(GamePhase::GlobeView)),
                )
                // H3 hex grid — Step 1 (cell indexing + hover picking +
                // boundary rendering) and Step 2 (click-driven drill-down
                // zoom, finer hexes as you approach) of the H3 Earth /
                // telemetry solar system / procedural galaxy plan. Step 3
                // (walkable view once fully drilled in) is wired below via
                // watch_for_cell_arrival_system + the CellWalk state.
                .add_systems(
                    Update,
                    (
                        sol_atlas_bevy::h3_grid::hover_cell_system,
                        sol_atlas_bevy::h3_grid::draw_hovered_cell_system,
                        sol_atlas_bevy::cell_entry::trigger_cell_zoom_system,
                        sol_atlas_bevy::cell_entry::cancel_zoom_on_manual_input,
                        sol_atlas_bevy::cell_entry::cell_zoom_transition_system,
                        systems::atlas::watch_for_cell_arrival_system,
                    )
                        .chain()
                        .after(sol_atlas_bevy::camera::orbital_camera_system)
                        .run_if(in_state(GamePhase::GlobeView)),
                )
                // Step 3: walkable ground-level view, only reachable by
                // drilling all the way into a cell (never at intermediate
                // zoom levels — those stay the orbital GlobeView).
                .init_resource::<sol_atlas_bevy::cell_walk::CellWalkLook>()
                .add_systems(
                    OnEnter(GamePhase::CellWalk),
                    (
                        sol_atlas_bevy::cell_walk::setup_cell_walk,
                        sol_atlas_bevy::cell_walk::cell_walk_cursor_grab_system,
                    ),
                )
                .add_systems(
                    OnExit(GamePhase::CellWalk),
                    (
                        sol_atlas_bevy::cell_walk::cleanup_cell_walk,
                        sol_atlas_bevy::cell_walk::cell_walk_cursor_release_system,
                    ),
                )
                .add_systems(
                    Update,
                    (
                        sol_atlas_bevy::cell_walk::cell_walk_mouse_look_system,
                        sol_atlas_bevy::cell_walk::cell_walk_movement_system,
                        sol_atlas_bevy::cell_walk::cell_walk_camera_sync_system,
                        systems::atlas::cell_walk_escape_system,
                    )
                        .chain()
                        .run_if(in_state(GamePhase::CellWalk)),
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
                        .run_if(in_playing_or_3d),
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
                        .run_if(in_playing_or_3d),
                );
        }
        app.add_systems(Update, game_over.run_if(in_state(GamePhase::GameOver)))
            .add_systems(Update, victory.run_if(in_state(GamePhase::Victory)));
    }
}

fn auto_start(
    mut s: ResMut<NextState<GamePhase>>,
    registry: Res<crate::experience::ExperienceRegistry>,
) {
    let exp = &registry.experiences[registry.selected];
    if exp.id == "waterworks-3d" {
        eprintln!("[symtropy] Loading → Playing3D");
        s.set(GamePhase::Playing3D);
    } else {
        eprintln!("[symtropy] Loading → Playing");
        s.set(GamePhase::Playing);
    }
}

fn game_over(
    kb: Res<ButtonInput<KeyCode>>,
    mut s: ResMut<NextState<GamePhase>>,
    mut l: ResMut<LeviathanState>,
    mut b: ResMut<BiometricsCtx>,
    mut cc: ResMut<ClearColor>,
    mut hud: Query<(&mut Text, &mut TextColor), With<systems::rendering::HudText>>,
    mut logged: Local<bool>,
    mut app_exit: MessageWriter<AppExit>,
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
        app_exit.write(AppExit::Success);
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
    mut app_exit: MessageWriter<AppExit>,
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
        app_exit.write(AppExit::Success);
    }
}

fn in_playing_or_3d(state: Res<State<GamePhase>>) -> bool {
    *state.get() == GamePhase::Playing || *state.get() == GamePhase::Playing3D
}

/// Escape returns to the main menu from the 3D FPS layer — otherwise, once
/// `fps_cursor_grab_system` locks the mouse, there's no way to leave.
fn fps_escape_to_menu(kb: Res<ButtonInput<KeyCode>>, mut s: ResMut<NextState<GamePhase>>) {
    if kb.just_pressed(KeyCode::Escape) {
        s.set(GamePhase::MainMenu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::resources::GamePhase;

    #[test]
    fn test_headless_gameplay_3d_loop() {
        let mut app = App::new();
        // Add MinimalPlugins
        app.add_plugins(MinimalPlugins);
        // Add StatesPlugin so init_state can be called
        app.add_plugins(bevy::state::app::StatesPlugin);
        // Add InputPlugin so input resources are initialized
        app.add_plugins(bevy::input::InputPlugin);
        // Add AssetPlugin so Assets<Mesh>/Assets<StandardMaterial> exist
        app.add_plugins(bevy::asset::AssetPlugin::default());
        app.add_plugins(bevy::gizmos::GizmoPlugin);
        app.init_asset::<Mesh>();
        app.init_asset::<StandardMaterial>();
        // setup_world_3d loads a texture via AssetServer; needs Assets<Image>
        // registered even though we never add the full ImagePlugin here.
        app.init_asset::<bevy::image::Image>();
        // Bevy 0.19's GizmoPlugin added draw_skinned_mesh_bounds, which
        // unconditionally reads Assets<SkinnedMeshInverseBindposes> — normally
        // registered by the full mesh/PBR plugin stack we don't add here.
        app.init_asset::<bevy::mesh::skinning::SkinnedMeshInverseBindposes>();

        // Add SymtropyPlugin
        app.add_plugins(SymtropyPlugin);

        // Selection experience setup: select "waterworks-3d"
        {
            let mut registry = app
                .world_mut()
                .resource_mut::<crate::experience::ExperienceRegistry>();
            if let Some(idx) = registry
                .experiences
                .iter()
                .position(|e| e.id == "waterworks-3d")
            {
                registry.selected = idx;
            }
        }

        // Move to loading state
        app.world_mut()
            .resource_mut::<NextState<GamePhase>>()
            .set(GamePhase::Loading);

        // Tick a few times to trigger loading -> Playing3D transition
        for _ in 0..10 {
            app.update();
        }

        // Verify the transition has completed
        let current_state = app.world().resource::<State<GamePhase>>().get();
        assert_eq!(
            *current_state,
            GamePhase::Playing3D,
            "Should transition to Playing3D state"
        );

        // Verify player is spawned
        let player_count = app
            .world_mut()
            .query_filtered::<&Transform, With<crate::components::Player>>()
            .iter(app.world())
            .count();
        assert!(player_count > 0, "Player should be spawned");

        // Verify NPCs are spawned
        let npc_count = app
            .world_mut()
            .query_filtered::<&Transform, With<crate::components::CrewNpc>>()
            .iter(app.world())
            .count();
        assert_eq!(npc_count, 7, "All 7 NPCs should be spawned");

        // Tick further to simulate active gameplay
        for _ in 0..100 {
            app.update();
        }
    }
}
