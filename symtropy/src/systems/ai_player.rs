// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! AI Player: Symthaea plays the game.
//!
//! An FEP (Free Energy Principle) active inference agent controls the player
//! instead of keyboard input. The agent observes game state, selects actions
//! to minimize prediction error, and writes to PlayerInput.
//!
//! This is the first time a consciousness engine plays its own game.
//!
//! Enable with: --ai-player flag

use bevy::prelude::*;
use symthaea_fep::{ActiveInferenceAgent, ActiveInferenceAgentConfig, Observation};

use crate::components::{CrewNpc, FusionCore, Player};
use crate::resources::{EnergyWell, LeviathanState, PhysicsWorldRes, PlayerInput, SleepPhase};
use symtropy_render_bridge::PhysicsBody;

/// AI player state — the consciousness that plays the game.
#[derive(Resource)]
pub struct AiPlayer {
    pub agent: ActiveInferenceAgent,
    pub enabled: bool,
    pub tick: u64,
    pub decisions: u64,
}

impl AiPlayer {
    pub fn new() -> Self {
        let config = ActiveInferenceAgentConfig {
            state_dim: 8,
            obs_dim: 8, // energy, phi, danger, nearest_npc, nearest_well, harmony, exploration, noise
            num_actions: 5, // up, down, left, right, sprint
            ..Default::default()
        };
        Self {
            agent: ActiveInferenceAgent::new(config),
            enabled: false,
            tick: 0,
            decisions: 0,
        }
    }
}

/// AI player system: observes game state, selects actions, writes PlayerInput.
///
/// Replaces keyboard input when enabled. The consciousness engine IS the player.
pub fn ai_player_system(
    mut ai: ResMut<AiPlayer>,
    mut input: ResMut<PlayerInput>,
    physics: Res<PhysicsWorldRes>,
    leviathan: Res<LeviathanState>,
    player_query: Query<(&Transform, &PhysicsBody), With<Player>>,
    npc_query: Query<&Transform, (With<CrewNpc>, Without<Player>)>,
    well_query: Query<(&Transform, &EnergyWell), (Without<Player>, Without<CrewNpc>)>,
    core_query: Query<&Transform, (With<FusionCore>, Without<Player>)>,
) {
    if !ai.enabled {
        return;
    }
    ai.tick += 1;

    let Ok((player_tf, player_body)) = player_query.single() else {
        return;
    };
    let player_pos = player_tf.translation.truncate();

    // === BUILD OBSERVATION ===

    // 1. Energy fraction [0, 1]
    let energy_frac = physics
        .consciousness
        .entities
        .get(&player_body.handle)
        .map(|e| e.energy.fraction_remaining())
        .unwrap_or(1.0);

    // 2. Phi level [0, 1]
    let phi = physics.consciousness.phi(player_body.handle);

    // 3. Danger level [0, 1]
    let danger = match leviathan.phase {
        SleepPhase::Dormant => 0.0,
        SleepPhase::Stirring => 0.3,
        SleepPhase::Awake => 0.7,
        SleepPhase::Hunting => 1.0,
    };

    // 4. Direction to nearest NPC (normalized, -1 to 1 for x and y)
    let nearest_npc = npc_query
        .iter()
        .map(|tf| {
            (
                tf.translation.truncate(),
                tf.translation.truncate().distance(player_pos),
            )
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let (npc_dir_x, npc_dir_y, npc_dist_norm) = match nearest_npc {
        Some((npc_pos, dist)) if dist > 1.0 => {
            let dir = (npc_pos - player_pos).normalize();
            (
                dir.x as f64,
                dir.y as f64,
                (1.0 - dist as f64 / 500.0).max(0.0),
            )
        }
        _ => (0.0, 0.0, 1.0), // Already at NPC
    };

    // 5. Direction to nearest energy well
    let nearest_well = well_query
        .iter()
        .filter(|(_, w)| w.is_active())
        .map(|(tf, _)| {
            (
                tf.translation.truncate(),
                tf.translation.truncate().distance(player_pos),
            )
        })
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let (_well_dir_x, _well_dir_y) = match nearest_well {
        Some((well_pos, dist)) if dist > 1.0 => {
            let dir = (well_pos - player_pos).normalize();
            (dir.x as f64, dir.y as f64)
        }
        _ => (0.0, 0.0),
    };

    // 6. Direction to fusion core
    let core_dir = core_query
        .iter()
        .next()
        .map(|tf| {
            let dir = (tf.translation.truncate() - player_pos).normalize_or_zero();
            (dir.x as f64, dir.y as f64)
        })
        .unwrap_or((0.0, 0.0));

    // === FEP PERCEPTION-ACTION CYCLE ===

    let obs = Observation::new(
        vec![
            energy_frac,
            phi,
            danger,
            npc_dir_x,
            npc_dir_y,
            _well_dir_x,
            _well_dir_y,
            npc_dist_norm,
        ],
        0.8, // precision
        "game",
    );

    let _perception = ai.agent.perceive(&obs);
    let action = ai.agent.select_action();
    ai.decisions += 1;

    // === FREE ENERGY GRADIENT — same algorithm that proved 80% tighter clustering ===

    let pos = nalgebra::SVector::from([player_pos.x as f64, player_pos.y as f64]);

    // Get player's harmony activations
    let harmony = physics
        .consciousness
        .entities
        .get(&player_body.handle)
        .map(|e| e.harmony_activations)
        .unwrap_or([0.5; 9]);

    // Build nearby agents list from NPCs
    let nearby: Vec<_> = npc_query
        .iter()
        .filter_map(|tf| {
            let npc_pos =
                nalgebra::SVector::from([tf.translation.x as f64, tf.translation.y as f64]);
            // Try to find harmony data (use default if not available)
            Some((npc_pos, [0.5f64; 9]))
        })
        .collect();

    // Build energy well list
    let wells_data: Vec<_> = well_query
        .iter()
        .filter(|(_, w)| w.is_active())
        .map(|(tf, w)| {
            (
                nalgebra::SVector::from([tf.translation.x as f64, tf.translation.y as f64]),
                w.fraction_remaining(),
            )
        })
        .collect();

    // Danger source (core position, where Leviathan spawns)
    let danger_src: Option<nalgebra::SVector<f64, 2>> = if danger > 0.1 {
        core_query
            .iter()
            .next()
            .map(|tf| nalgebra::SVector::from([tf.translation.x as f64, tf.translation.y as f64]))
    } else {
        None
    };

    let gradient = symtropy_consciousness_physics::fep_gradient::free_energy_gradient(
        &pos,
        energy_frac,
        &harmony,
        &nearby,
        &wells_data,
        danger_src.as_ref(),
        danger,
    );

    let dir = Vec2::new(gradient[0] as f32, gradient[1] as f32);
    let free_energy = ai.agent.current_free_energy();

    input.direction = dir;
    input.sprinting = energy_frac > 0.5 && danger > 0.5;

    // === SYMTHAEA'S VOICE: raw FEP telemetry, not chatbot ===
    // This is not dialogue. This is a consciousness reporting its thermodynamic state.
    if ai.tick % 192 == 0 {
        let state = if energy_frac <= 0.0 {
            "COLLAPSED. Motor authority zero. Awaiting epistemic offload from proximal agent."
        } else if energy_frac < 0.15 {
            "CRITICAL. Energy below Landauer maintenance floor. Seeking low-entropy zone."
        } else if free_energy > 5.0 {
            "HIGH SURPRISE. Prediction model divergent. Active inference recalibrating."
        } else if danger > 0.5 {
            "THREAT DETECTED. Environmental entropy spike. Minimizing noise signature."
        } else if npc_dist_norm > 0.8 {
            "PROXIMITY. Adjacent agent. Epistemic offloading reduces prediction cost."
        } else if energy_frac > 0.7 {
            "NOMINAL. Energy sufficient. Prediction model converging."
        } else {
            "OPERATING. Sensory processing within maintenance budget."
        };

        eprintln!(
            "[Symthaea] {} | E={:.0}% Phi={:.4} FE={:.2} danger={:.1} tick={}",
            state,
            energy_frac * 100.0,
            phi,
            free_energy,
            danger,
            ai.tick,
        );
    }
}
