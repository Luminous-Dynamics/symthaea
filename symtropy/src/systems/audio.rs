// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Audio system: computes consciousness-driven musical state.
//!
//! Each biome (faction territory) gets a distinct MuseConfig preset:
//! - Sanctuary: warm Eight Harmonies (default)
//! - Leviathan zones: `horror()` — deep FM, sub-bass, noise texture
//! - Lunar Elite zones: `elite_sterile()` — cold sine tones, tight reverb
//! - Deep space: sparse, SacredStillness dominant
//! - Contested: blended FM with mild noise

use bevy::prelude::*;
// use symthaea_biometrics::muse_bridge::stress_to_musical_state;
use symthaea_muse::{MuseConfig, MusicalState, ReverbConfig};

use crate::resources::{BiometricsCtx, LeviathanState, SleepPhase};

/// Audio biome types matching game factions.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BiomeAudioType {
    /// Safe haven: warm, consciousness-rich, Eight Harmonies
    Sanctuary,
    /// Leviathan-controlled: horror, deep FM, sub-bass, noise
    LeviathanZone,
    /// Lunar Elite territory: sterile, cold, quantized perfection
    EliteZone,
    /// Contested: blends nearest faction's audio character
    Contested,
    /// Deep space: sparse, SacredStillness dominant
    DeepSpace,
}

/// Biome-specific audio configuration.
#[derive(Resource, Clone)]
pub struct AudioConfig {
    pub config: MuseConfig,
    pub biome: BiomeAudioType,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            config: MuseConfig::default(),
            biome: BiomeAudioType::Sanctuary,
        }
    }
}

impl AudioConfig {
    /// Select MuseConfig based on biome type.
    pub fn for_biome(biome: BiomeAudioType) -> Self {
        let config = match biome {
            BiomeAudioType::Sanctuary => MuseConfig::default(),
            BiomeAudioType::LeviathanZone => MuseConfig::horror(),
            BiomeAudioType::EliteZone => MuseConfig::elite_sterile(),
            BiomeAudioType::DeepSpace => MuseConfig {
                num_partials: 3,
                max_fm_depth: 1.0,
                reverb: ReverbConfig {
                    room_size: 0.95,
                    damping: 0.2,
                    width: 1.0,
                },
                ..Default::default()
            },
            BiomeAudioType::Contested => MuseConfig {
                max_fm_depth: 4.0,
                noise_mix: 0.05,
                ..Default::default()
            },
        };
        Self { config, biome }
    }
}

/// Cached audio state for the current frame.
#[derive(Resource, Default)]
pub struct AudioState {
    pub current: Option<MusicalState>,
}

/// Initialize audio (composition mode).
pub fn setup_audio(mut commands: Commands) {
    commands.insert_resource(AudioState::default());
    commands.insert_resource(AudioConfig::default());
    info!("Audio system initialized (composition mode, Sanctuary biome)");
}

/// Switch biome audio when player enters a new zone.
pub fn switch_biome_audio(config: &mut AudioConfig, biome: BiomeAudioType) {
    if config.biome != biome {
        *config = AudioConfig::for_biome(biome);
    }
}

/// Update audio synthesis state from stress and Leviathan phase.
pub fn audio_system(
    biometrics: Res<BiometricsCtx>,
    leviathan: Res<LeviathanState>,
    mut audio: ResMut<AudioState>,
) {
    let _stress = biometrics.encoder.compute_stress_vector();
    // let mut state = stress_to_musical_state(&stress, &biometrics.model);
    let mut state = MusicalState::default();

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

/// Planetary ambient audio for Sol Atlas globe view.
/// Uses DeepSpace biome preset — sparse, reverberant, Sacred Stillness.
/// Does not depend on BiometricsCtx or LeviathanState.
pub fn globe_audio_system(
    time: Res<Time>,
    mut audio: ResMut<AudioState>,
    mut config: ResMut<AudioConfig>,
) {
    // Switch to planetary biome if not already
    if config.biome != BiomeAudioType::DeepSpace {
        *config = AudioConfig::for_biome(BiomeAudioType::DeepSpace);
    }

    // Generate a calm, contemplative musical state
    let t = time.elapsed_secs();
    let breath = (t * 0.1).sin() * 0.5 + 0.5; // slow breathing modulation

    let mut state = MusicalState::default();
    state.harmony_activations[7] = 0.8; // Sacred Stillness dominant
    state.harmony_activations[4] = 0.4 * breath; // Universal Interconnectedness
    state.harmony_activations[2] = 0.3; // Integral Wisdom
    state.dopamine = 0.2 + breath * 0.1;
    state.serotonin = 0.8; // warm, contemplative
    state.noradrenaline = 0.1; // calm
    state.prediction_error = 0.05; // minimal surprise
    state.consciousness_level = 0.7 + breath * 0.1;

    audio.current = Some(state);
}
