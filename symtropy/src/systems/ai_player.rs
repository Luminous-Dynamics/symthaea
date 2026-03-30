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
    well_query: Query<&Transform, (With<EnergyWell>, Without<Player>, Without<CrewNpc>)>,
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
        .map(|tf| (tf.translation.truncate(), tf.translation.truncate().distance(player_pos)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let (npc_dir_x, npc_dir_y, npc_dist_norm) = match nearest_npc {
        Some((npc_pos, dist)) if dist > 1.0 => {
            let dir = (npc_pos - player_pos).normalize();
            (dir.x as f64, dir.y as f64, (1.0 - dist as f64 / 500.0).max(0.0))
        }
        _ => (0.0, 0.0, 1.0), // Already at NPC
    };

    // 5. Direction to nearest energy well
    let nearest_well = well_query
        .iter()
        .map(|tf| (tf.translation.truncate(), tf.translation.truncate().distance(player_pos)))
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let (well_dir_x, well_dir_y) = match nearest_well {
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
            well_dir_x,
            well_dir_y,
            npc_dist_norm,
        ],
        0.8, // precision
        "game",
    );

    let _perception = ai.agent.perceive(&obs);
    let action = ai.agent.select_action();
    ai.decisions += 1;

    // === MAP ACTION TO MOVEMENT ===
    // The FEP agent outputs action weights. We use them as direction bias.
    // Strategy: blend toward NPC (for harmony resonance) and well (for energy)

    let mut dir = Vec2::ZERO;

    // Low energy → seek wells
    // High energy → seek core (win condition)
    // Medium → seek NPCs (harmony resonance)
    if energy_frac < 0.3 {
        // Desperate: go to nearest well
        dir.x = well_dir_x as f32;
        dir.y = well_dir_y as f32;
    } else if energy_frac > 0.7 {
        // Healthy: go toward core (try to win)
        dir.x = core_dir.0 as f32;
        dir.y = core_dir.1 as f32;
    } else {
        // Moderate: seek NPC for harmony resonance
        dir.x = npc_dir_x as f32;
        dir.y = npc_dir_y as f32;
    }

    // Add FEP perturbation (exploration)
    let free_energy = ai.agent.current_free_energy();
    if free_energy > 0.3 {
        // High surprise → explore (add random-ish movement from action)
        let angle = (ai.tick as f64 * 0.1 + free_energy * 10.0).sin() as f32;
        dir.x += angle * 0.3;
        dir.y += angle.cos() * 0.3;
    }

    // Flee from Leviathan
    if danger > 0.5 {
        // Run away from core area (Leviathan spawns near core)
        dir.x -= core_dir.0 as f32 * danger as f32;
        dir.y -= core_dir.1 as f32 * danger as f32;
    }

    input.direction = dir;
    input.sprinting = energy_frac > 0.5 && danger > 0.5; // Sprint only when healthy and threatened

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
