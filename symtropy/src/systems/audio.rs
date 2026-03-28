// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Audio system: computes consciousness-driven musical state.
//!
//! When symthaea-muse compiles with muse-live, this wires to cpal speakers.
//! Currently runs in "composition mode" — state computed but not output.

use bevy::prelude::*;
use symthaea_biometrics::muse_bridge::stress_to_musical_state;
use symthaea_muse::MusicalState;

use crate::resources::{BiometricsCtx, LeviathanState, SleepPhase};

/// Cached audio state for the current frame.
#[derive(Resource, Default)]
pub struct AudioState {
    pub current: Option<MusicalState>,
}

/// Initialize audio (composition mode).
pub fn setup_audio(mut commands: Commands) {
    commands.insert_resource(AudioState::default());
    info!("Audio system initialized (composition mode)");
}

/// Update audio synthesis state from stress and Leviathan phase.
pub fn audio_system(
    biometrics: Res<BiometricsCtx>,
    leviathan: Res<LeviathanState>,
    mut audio: ResMut<AudioState>,
) {
    let stress = biometrics.encoder.compute_stress_vector();
    let mut state = stress_to_musical_state(&stress, &biometrics.model);

    match leviathan.phase {
        SleepPhase::Dormant => {
            state.harmony_activations[7] = (state.harmony_activations[7] + 0.3).min(1.0);
            state.harmony_activations[3] *= 0.3;
            state.dopamine *= 0.5;
            state.noradrenaline *= 0.3;
        }
        SleepPhase::Stirring => {
            let blend = (leviathan.stirring_duration / 3.0).min(1.0);
            state.dopamine = state.dopamine * (1.0 - blend) + 0.7 * blend;
            state.prediction_error = state.prediction_error.max(blend * 0.4);
        }
        SleepPhase::Awake | SleepPhase::Hunting => {
            state.dopamine = state.dopamine.max(0.8);
            state.noradrenaline = state.noradrenaline.max(0.8);
            state.serotonin = state.serotonin.min(0.15);
            state.harmony_activations[3] = 0.9;
            state.harmony_activations[7] = 0.0;
            state.prediction_error = 0.8;
        }
    }

    audio.current = Some(state);
}
