// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Symtropy game plugin: wires all systems together.

use bevy::prelude::*;

use crate::resources::{BiometricsCtx, GamePhase, LeviathanState};
use crate::systems;

/// Main game plugin. Add this to a Bevy `App` to get the full Symtropy experience.
pub struct SymtropyPlugin;

impl Plugin for SymtropyPlugin {
    fn build(&self, app: &mut App) {
        app
            // State
            .init_state::<GamePhase>()
            // Resources
            .init_resource::<BiometricsCtx>()
            .init_resource::<LeviathanState>()
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
                    // Leviathan
                    systems::leviathan::leviathan_system,
                    systems::leviathan::victory_check_system,
                    // Audio (reads stress + leviathan)
                    systems::audio::audio_system,
                    // Rendering
                    systems::rendering::camera_follow_system,
                    systems::rendering::visual_stress_system,
                    systems::rendering::hud_system,
                )
                    .chain()
                    .run_if(in_state(GamePhase::Playing)),
            )
            // Auto-transition Loading → Playing after one frame
            .add_systems(
                Update,
                auto_start.run_if(in_state(GamePhase::Loading)),
            );
    }
}

fn auto_start(mut next_state: ResMut<NextState<GamePhase>>) {
    next_state.set(GamePhase::Playing);
}
