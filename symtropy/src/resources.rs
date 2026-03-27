// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Global ECS resources for Symtropy.

use bevy::prelude::*;
use symthaea_biometrics::input_telemetry::InputTelemetryEncoder;
use symthaea_biometrics::stress_model::PlayerStressModel;
// MuseConfig and LiveMuseOutput will be used when cpal audio is wired in.

/// Player behavioral biometrics state.
#[derive(Resource)]
pub struct BiometricsCtx {
    pub encoder: InputTelemetryEncoder,
    pub model: PlayerStressModel,
}

impl Default for BiometricsCtx {
    fn default() -> Self {
        Self {
            encoder: InputTelemetryEncoder::new(),
            model: PlayerStressModel::new(),
        }
    }
}

/// Leviathan sleep phases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SleepPhase {
    /// Safe — no audio effect, free exploration.
    Dormant,
    /// Warning — sub-bass hum, lights flicker.
    Stirring,
    /// Danger — full horror audio, doors seal.
    Awake,
    /// Lethal — Leviathan hunts the player.
    Hunting,
}

/// Global Leviathan state.
#[derive(Resource)]
pub struct LeviathanState {
    /// Current sleep phase.
    pub phase: SleepPhase,
    /// Accumulated noise from all sources [0, ∞).
    pub noise_accumulator: f32,
    /// Noise threshold to transition Dormant → Stirring.
    pub threshold: f32,
    /// Seconds the Leviathan has been in Stirring phase.
    pub stirring_duration: f32,
    /// Seconds of quiet since last noise.
    pub quiet_duration: f32,
}

impl Default for LeviathanState {
    fn default() -> Self {
        Self {
            phase: SleepPhase::Dormant,
            noise_accumulator: 0.0,
            threshold: 1.0,
            stirring_duration: 0.0,
            quiet_duration: 0.0,
        }
    }
}

/// Game state for phase management.
#[derive(Debug, Clone, Copy, Default, Eq, PartialEq, Hash, States)]
pub enum GamePhase {
    /// Setting up the level.
    #[default]
    Loading,
    /// Active gameplay.
    Playing,
    /// Leviathan caught the player.
    GameOver,
    /// Player escaped with the core.
    Victory,
}
