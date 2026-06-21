// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Leviathan state machine: Dormant → Stirring → Awake → Hunting.

use bevy::prelude::*;

use crate::components::NoiseEmitter;
use crate::resources::{GamePhase, LeviathanState, SleepPhase};

/// Noise decay rate per second.
const NOISE_DECAY_RATE: f32 = 0.2;

/// Seconds of sustained noise to transition Stirring → Awake.
const STIRRING_TO_AWAKE_SECS: f32 = 3.0;

/// Seconds of quiet to transition Stirring → Dormant.
const QUIET_RESET_SECS: f32 = 10.0;

/// Noise floor — below this counts as "quiet."
const NOISE_FLOOR: f32 = 0.05;

/// Accumulate noise from all emitters and drive the Leviathan state machine.
pub fn leviathan_system(
    noise_sources: Query<&NoiseEmitter>,
    mut leviathan: ResMut<LeviathanState>,
    mut next_state: ResMut<NextState<GamePhase>>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();

    // Grace period — Leviathan ignores all noise at game start
    if leviathan.grace_timer > 0.0 {
        leviathan.grace_timer -= dt;
        return;
    }

    // Sum all noise sources
    let total_noise: f32 = noise_sources.iter().map(|n| n.level).sum();

    // Accumulate with decay
    leviathan.noise_accumulator += total_noise * dt;
    leviathan.noise_accumulator -= NOISE_DECAY_RATE * dt;
    leviathan.noise_accumulator = leviathan.noise_accumulator.max(0.0);

    // Track quiet duration
    if total_noise < NOISE_FLOOR {
        leviathan.quiet_duration += dt;
    } else {
        leviathan.quiet_duration = 0.0;
    }

    // State machine transitions
    match leviathan.phase {
        SleepPhase::Dormant => {
            if leviathan.noise_accumulator > leviathan.threshold {
                leviathan.phase = SleepPhase::Stirring;
                leviathan.stirring_duration = 0.0;
                warn!("Leviathan STIRRING — noise exceeded threshold");
            }
        }
        SleepPhase::Stirring => {
            if total_noise > NOISE_FLOOR {
                leviathan.stirring_duration += dt;
            }
            if leviathan.stirring_duration > STIRRING_TO_AWAKE_SECS {
                leviathan.phase = SleepPhase::Awake;
                warn!(
                    "Leviathan AWAKE — sustained noise for {}s",
                    STIRRING_TO_AWAKE_SECS
                );
            }
            if leviathan.quiet_duration > QUIET_RESET_SECS {
                leviathan.phase = SleepPhase::Dormant;
                leviathan.noise_accumulator = 0.0;
                leviathan.stirring_duration = 0.0;
            }
        }
        SleepPhase::Awake => {
            // Brief grace period then hunt
            leviathan.stirring_duration += dt;
            if leviathan.stirring_duration > STIRRING_TO_AWAKE_SECS + 2.0 {
                leviathan.phase = SleepPhase::Hunting;
            }
        }
        SleepPhase::Hunting => {
            warn!("Leviathan HUNTING — GAME OVER");
            next_state.set(GamePhase::GameOver);
        }
    }
}

/// Check if player has extracted the fusion core → victory.
pub fn victory_check_system(
    cores: Query<&crate::components::FusionCore>,
    mut next_state: ResMut<NextState<GamePhase>>,
    mut triggered: Local<bool>,
) {
    for core in &cores {
        if core.extraction_progress >= 1.0 {
            if !*triggered {
                crate::systems::chronicle_recorder::record_chronicle_event(
                    "FusionCoreExtractionSuccess",
                    serde_json::json!({
                        "progress": core.extraction_progress,
                    }),
                );
                *triggered = true;
            }
            next_state.set(GamePhase::Victory);
        }
    }
}
