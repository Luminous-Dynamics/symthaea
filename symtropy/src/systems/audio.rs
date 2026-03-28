// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Audio system: stress + Leviathan state → symthaea-muse.
//!
//! Currently generates compositions per-cycle and logs state.
//! Real-time cpal output will be restored when StreamingSynth
//! is re-added to symthaea-muse.

use bevy::prelude::*;
use symthaea_biometrics::muse_bridge::stress_to_musical_state;
use symthaea_muse::MusicalState;

use crate::resources::{BiometricsCtx, LeviathanState, SleepPhase};

/// Cached audio state for the current frame.
#[derive(Resource, Default)]
pub struct AudioState {
    pub current_musical_state: Option<MusicalState>,
}

/// Initialize audio resources.
pub fn setup_audio(mut commands: Commands) {
    commands.insert_resource(AudioState::default());
    info!("Audio system initialized (composition mode — cpal output pending)");
}

/// Update audio synthesis state from stress and Leviathan phase.
///
/// When cpal live output is restored, this will call output.update_state().
/// For now it computes and caches the MusicalState for telemetry/debug.
pub fn audio_system(
    biometrics: Res<BiometricsCtx>,
    leviathan: Res<LeviathanState>,
    mut audio: ResMut<AudioState>,
) {
    let stress = biometrics.encoder.compute_stress_vector();
    let mut state = stress_to_musical_state(&stress, &biometrics.model);

    // Blend Leviathan danger into audio
    match leviathan.phase {
        SleepPhase::Dormant => {
            // Calm ambient — boost SacredStillness, reduce tension
            state.harmony_activations[7] =
                (state.harmony_activations[7] + 0.3).min(1.0);
            state.harmony_activations[3] *= 0.3;
            state.dopamine *= 0.5;
            state.noradrenaline *= 0.3;
        }
        SleepPhase::Stirring => {
            // Warning — rising tension
            let blend = (leviathan.stirring_duration / 3.0).min(1.0);
            state.dopamine = state.dopamine * (1.0 - blend) + 0.7 * blend;
            state.prediction_error = state.prediction_error.max(blend * 0.4);
        }
        SleepPhase::Awake | SleepPhase::Hunting => {
            // Full horror — max tension
            state.dopamine = state.dopamine.max(0.8);
            state.noradrenaline = state.noradrenaline.max(0.8);
            state.serotonin = state.serotonin.min(0.15);
            state.harmony_activations[3] = 0.9;
            state.harmony_activations[7] = 0.0;
            state.prediction_error = 0.8;
        }
    }

    audio.current_musical_state = Some(state);
}
