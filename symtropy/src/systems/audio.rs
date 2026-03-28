// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Audio system: stress + Leviathan state → live generative music.

use bevy::prelude::*;
use symthaea_biometrics::muse_bridge::stress_to_musical_state;
use symthaea_muse::live_output::LiveMuseOutput;
use symthaea_muse::MuseConfig;

use crate::resources::{AudioOutput, BiometricsCtx, LeviathanState, SleepPhase};

/// Initialize the live audio output device.
pub fn setup_audio(mut commands: Commands) {
    let config = MuseConfig {
        num_partials: 10,
        ..MuseConfig::horror()
    };
    match LiveMuseOutput::new(config) {
        Ok(output) => {
            info!("Live audio: device opened at {}Hz", output.sample_rate());
            commands.insert_resource(AudioOutput(Some(output)));
        }
        Err(e) => {
            warn!("Audio device unavailable: {e} — running silent");
            commands.insert_resource(AudioOutput(None));
        }
    }
}

/// Update audio synthesis from stress state and Leviathan phase.
pub fn audio_system(
    biometrics: Res<BiometricsCtx>,
    leviathan: Res<LeviathanState>,
    audio: Res<AudioOutput>,
) {
    let Some(ref output) = audio.0 else { return };

    let stress = biometrics.encoder.compute_stress_vector();
    let mut state = stress_to_musical_state(&stress, &biometrics.model);

    // Blend Leviathan danger into audio
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

    output.update_state(&state);
}
