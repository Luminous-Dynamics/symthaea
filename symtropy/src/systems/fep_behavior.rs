// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! FEP-driven NPC behavior: free energy gradient minimization.

use bevy::prelude::*;
use symthaea_fep::Observation;

use crate::components::{CrewNpc, MoveTarget, NoiseEmitter, Player};
use crate::resources::{
    BiometricsCtx, EnergyWell, LeviathanState, PhysicsWorldRes, SettlementMetrics, SleepPhase,
};
use symtropy_render_bridge::PhysicsBody;

/// Run the FEP perception-action cycle for each crew NPC.
pub fn fep_behavior_system(
    mut npcs: Query<
        (
            &mut CrewNpc,
            &Transform,
            &mut MoveTarget,
            &mut NoiseEmitter,
            &PhysicsBody,
            Option<&crate::systems::psychology::PsychologicalNeeds>,
        ),
        Without<Player>,
    >,
    player_query: Query<(&Transform, &PhysicsBody), With<Player>>,
    other_npcs: Query<(&Transform, &PhysicsBody), (With<CrewNpc>, Without<Player>)>,
    wells: Query<(&Transform, &EnergyWell)>,
    biometrics: Res<BiometricsCtx>,
    leviathan: Res<LeviathanState>,
    physics: Res<PhysicsWorldRes>,
    settlement: Res<SettlementMetrics>,
    core_query: Query<
        &Transform,
        (
            With<crate::components::FusionCore>,
            Without<Player>,
            Without<CrewNpc>,
        ),
    >,
) {
    let Some((player_tf, player_body)) = player_query.iter().next() else {
        return;
    };
    let player_pos = player_tf.translation.truncate();

    let danger = match leviathan.phase {
        SleepPhase::Dormant => 0.0f64,
        SleepPhase::Stirring => 0.3,
        SleepPhase::Awake => 0.7,
        SleepPhase::Hunting => 1.0,
    };

    let danger_source: Option<nalgebra::SVector<f64, 2>> = if danger > 0.1 {
        core_query
            .iter()
            .next()
            .map(|tf| nalgebra::SVector::from([tf.translation.x as f64, tf.translation.y as f64]))
    } else {
        None
    };

    let well_data: Vec<(nalgebra::SVector<f64, 2>, f64)> = wells
        .iter()
        .filter(|(_, w)| w.is_active())
        .map(|(tf, w)| {
            (
                nalgebra::SVector::from([tf.translation.x as f64, tf.translation.y as f64]),
                w.fraction_remaining(),
            )
        })
        .collect();

    let mut all_agents: Vec<(nalgebra::SVector<f64, 2>, [f64; 9])> = Vec::new();

    if let Some(entity) = physics.consciousness.entities.get(&player_body.handle) {
        all_agents.push((
            nalgebra::SVector::from([player_pos.x as f64, player_pos.y as f64]),
            entity.harmony_activations,
        ));
    }
    for (tf, body) in &other_npcs {
        if let Some(entity) = physics.consciousness.entities.get(&body.handle) {
            all_agents.push((
                nalgebra::SVector::from([tf.translation.x as f64, tf.translation.y as f64]),
                entity.harmony_activations,
            ));
        }
    }

    for (mut npc, npc_tf, mut target, mut noise, body, psych) in &mut npcs {
        let npc_pos = npc_tf.translation.truncate();
        let pos = nalgebra::SVector::from([npc_pos.x as f64, npc_pos.y as f64]);

        let (energy_frac, harmony, _prediction_error, npc_phi) = physics
            .consciousness
            .entities
            .get(&body.handle)
            .map(|e| {
                (
                    e.energy.fraction_remaining(),
                    e.harmony_activations,
                    e.prediction_error,
                    e.phi(),
                )
            })
            .unwrap_or((1.0, [0.5; 9], 0.0, 0.5));

        // FEP perception with Settlement Metrics
        let obs = Observation::new(
            vec![
                energy_frac,
                biometrics.encoder.compute_stress_vector().arousal as f64,
                danger,
                npc.caution as f64,
                settlement.water as f64,
                settlement.power as f64,
            ],
            0.8,
            "game",
        );
        let _perception = npc.fep.perceive(&obs);

        let nearby: Vec<_> = all_agents
            .iter()
            .filter(|(agent_pos, _)| {
                let d = (agent_pos - pos).norm();
                d > 2.0
            })
            .cloned()
            .collect();

        let mut direction = symtropy_consciousness_physics::fep_gradient::free_energy_gradient_phi(
            &pos,
            energy_frac,
            Some(npc_phi),
            &harmony,
            &nearby,
            &well_data,
            danger_source.as_ref(),
            danger,
        );

        // EMERGENT CRISIS MODIFIERS:
        if npc_phi > 0.6 {
            if settlement.water < 0.3 {
                // Seek the Water Pump (assumed to be at the last room center, near the end)
                if let Some((wx, wy)) = well_data.last().map(|(p, _)| (p[0], p[1])) {
                    let to_pump = nalgebra::SVector::from([wx - pos[0], wy - pos[1]]).normalize();
                    direction = direction * 0.5 + to_pump * 0.5;
                }
            }
        }

        let dir_vec = Vec2::new(direction[0] as f32, direction[1] as f32);
        let load = psych.map(|p| p.allostatic_load).unwrap_or(0.0);
        let engagement = psych.map(|p| p.engagement).unwrap_or(1.0);

        let effective_engagement = if danger > 0.5 {
            engagement.max(0.5)
        } else {
            engagement
        };

        if dir_vec.length_squared() > 0.01 && effective_engagement > 0.15 {
            let speed = if energy_frac < 0.2 {
                90.0
            } else if danger > 0.5 {
                100.0
            } else {
                50.0
            };

            let psych_factor = (1.0 - load * 0.4) * effective_engagement;
            target.target = Some(npc_pos + dir_vec * 100.0);
            target.speed = speed * (1.0 - npc.caution * 0.3) * psych_factor;
            noise.level = if speed > 80.0 { 0.1 } else { 0.03 };
        } else {
            target.target = None;
            target.speed = 0.0;
            noise.level = 0.0;
        }

        let load_caution_boost = if load > 0.6 { 0.02 } else { 0.0 };
        if danger > 0.5 {
            npc.caution = (npc.caution + 0.05 + load_caution_boost).min(1.0);
        } else {
            npc.caution = (npc.caution - 0.02).max(0.0);
        }
    }
}

/// Apply NPC movement intent to physics bodies.
pub fn npc_movement_system(
    mut query: Query<
        (
            &Transform,
            &MoveTarget,
            &symtropy_render_bridge::PhysicsBody,
        ),
        With<CrewNpc>,
    >,
    mut physics: ResMut<PhysicsWorldRes>,
    tile_grid: Option<Res<crate::resources::TileGrid>>,
) {
    for (tf, target, body_comp) in &query {
        if let Some(body) = physics.world.body_mut(body_comp.handle) {
            if let Some(dest) = target.target {
                let pos = tf.translation.truncate();
                let dir = dest - pos;
                let dist = dir.length();
                if dist > 2.0 && target.speed > 0.0 {
                    let norm = dir.normalize();
                    let mut vx = norm.x as f64 * target.speed as f64;
                    let mut vy = norm.y as f64 * target.speed as f64;

                    if let Some(ref grid) = tile_grid {
                        let dt = 1.0 / 64.0_f32;
                        let new_x = tf.translation.x + vx as f32 * dt;
                        let new_y = tf.translation.y + vy as f32 * dt;
                        if !grid.is_walkable(new_x, tf.translation.y) {
                            vx = 0.0;
                        }
                        if !grid.is_walkable(tf.translation.x, new_y) {
                            vy = 0.0;
                        }
                    }

                    body.linear_velocity = nalgebra::SVector::from([vx, vy]);
                } else {
                    body.linear_velocity = nalgebra::SVector::from([0.0, 0.0]);
                }
            } else {
                body.linear_velocity = nalgebra::SVector::from([0.0, 0.0]);
            }
        }
    }
}
