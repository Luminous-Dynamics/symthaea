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
            threshold: 10.0, // high threshold — player must actively make noise to wake it
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
    /// Governance council session — gameplay paused for voting.
    Council,
    /// Leviathan caught the player.
    GameOver,
    /// Player escaped with the core.
    Victory,
}

// ============================================================================
// Governance / Economy event log (Phase A0)
// ============================================================================

/// Log of governance events for the HUD and post-game analysis.
#[derive(Resource, Default)]
pub struct GovernanceLog {
    /// Recent governance event messages (ring buffer, max 20).
    pub messages: Vec<GovernanceMessage>,
}

/// A timestamped governance message.
pub struct GovernanceMessage {
    /// Game time when the event occurred.
    pub time_secs: f32,
    /// Human-readable event description.
    pub text: String,
    /// Severity: 0=info, 1=warning (oppression), 2=critical (crisis).
    pub severity: u8,
}

impl GovernanceLog {
    /// Push a message, keeping max 20 entries.
    pub fn push(&mut self, time_secs: f32, text: String, severity: u8) {
        self.messages.push(GovernanceMessage {
            time_secs,
            text,
            severity,
        });
        if self.messages.len() > 20 {
            self.messages.remove(0);
        }
    }
}
