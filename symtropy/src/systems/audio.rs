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

use crate::components::CrewNpc;
use crate::resources::{BiometricsCtx, LeviathanState, PhysicsWorldRes, SleepPhase};

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

/// Smoothed audio variables for cinematic lerping.
#[derive(Resource)]
pub struct SmoothedAudioState {
    pub smoothed_stress: f32,
    pub smoothed_phi: f32,
}

impl Default for SmoothedAudioState {
    fn default() -> Self {
        Self {
            smoothed_stress: 0.0,
            smoothed_phi: 0.5,
        }
    }
}

/// Initialize audio (composition mode).
pub fn setup_audio(mut commands: Commands) {
    commands.insert_resource(AudioState::default());
    commands.insert_resource(AudioConfig::default());
    commands.insert_resource(SmoothedAudioState::default());
    info!("Audio system initialized (composition mode, Sanctuary biome)");
}

/// Switch biome audio when player enters a new zone.
pub fn switch_biome_audio(config: &mut AudioConfig, biome: BiomeAudioType) {
    if config.biome != biome {
        *config = AudioConfig::for_biome(biome);
    }
}

/// Update audio synthesis state from stress, crew biometrics, and Leviathan phase.
pub fn audio_system(
    biometrics: Res<BiometricsCtx>,
    leviathan: Res<LeviathanState>,
    mut audio: ResMut<AudioState>,
    npcs: Query<
        (
            &Transform,
            &symtropy_render_bridge::PhysicsBody,
            Option<&crate::systems::psychology::PsychologicalNeeds>,
        ),
        With<CrewNpc>,
    >,
    physics: Res<PhysicsWorldRes>,
    mut config: ResMut<AudioConfig>,
    mut smoothed: ResMut<SmoothedAudioState>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();
    let _stress = biometrics.encoder.compute_stress_vector();
    let mut state = MusicalState::default();

    // Calculate average allostatic load and sync/phi across spawned crew members
    let mut total_stress = 0.0f32;
    let mut total_phi = 0.0f32;
    let mut count = 0;

    for (_, body, psych) in &npcs {
        let load = psych.map(|p| p.allostatic_load).unwrap_or(0.0);
        let phi = physics
            .consciousness
            .entities
            .get(&body.handle)
            .map(|e| e.phi())
            .unwrap_or(0.5) as f32;

        total_stress += load;
        total_phi += phi;
        count += 1;
    }

    let (avg_stress, avg_phi) = if count > 0 {
        (total_stress / count as f32, total_phi / count as f32)
    } else {
        (0.0, 0.5)
    };

    // Apply asymmetric smoothing to average stress (anxiety):
    // stress rise: fast, 0.3–0.6s (we select 0.45s)
    // stress recovery: slow, 1.5–3.0s (we select 2.25s)
    let stress_diff = avg_stress - smoothed.smoothed_stress;
    let stress_tc = if stress_diff > 0.0 { 0.45 } else { 2.25 };
    smoothed.smoothed_stress += stress_diff * (dt / stress_tc).min(1.0);

    // Smooth average phi/sync over 1.5s time constant
    let phi_diff = avg_phi - smoothed.smoothed_phi;
    smoothed.smoothed_phi += phi_diff * (dt / 1.5).min(1.0);

    let smooth_stress = smoothed.smoothed_stress;
    let smooth_phi = smoothed.smoothed_phi;

    // Modulate MusicalState parameters using smoothed values
    state.valence = 0.5 - smooth_stress * 1.5; // goes negative/dissonant under stress
    state.valence += smooth_phi * 0.5; // harmonized under high sync
    state.valence = state.valence.clamp(-1.0, 1.0);

    state.arousal = (0.3 + smooth_stress * 0.7).clamp(0.0, 1.0);
    state.noradrenaline = (0.2 + smooth_stress * 0.8).clamp(0.0, 1.0);
    state.dopamine = (0.3 + smooth_phi * 0.7).clamp(0.0, 1.0);
    state.serotonin = (0.3 + smooth_phi * 0.7).clamp(0.0, 1.0);

    // Apply baseline activations
    if smooth_stress > 0.5 {
        state.harmony_activations[3] = (state.harmony_activations[3] + smooth_stress).min(1.0);
    }
    if smooth_phi > 0.6 {
        state.harmony_activations[7] = (state.harmony_activations[7] + smooth_phi).min(1.0);
        state.harmony_activations[4] = (state.harmony_activations[4] + smooth_phi * 0.5).min(1.0);
    }

    // Modulate MuseConfig parameters using smoothed values
    config.config.base_tempo_bpm = 80.0 + smooth_phi * 40.0; // accelerate under high sync
    config.config.max_fm_depth = 3.0 + smooth_stress * 5.0; // increase FM depth under stress
    config.config.unison_detune = smooth_stress * 0.005; // unison detune under stress
    config.config.noise_mix = smooth_stress * 0.1; // scale up noise under stress

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
            // Also force horror config on awake/hunting phase
            config.config.max_fm_depth = 8.0;
            config.config.unison_detune = 0.005;
            config.config.noise_mix = 0.1;
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
