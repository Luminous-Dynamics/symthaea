// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! FEP-driven NPC behavior: free energy gradient minimization.

use bevy::prelude::*;
use symthaea_fep::Observation;

use crate::components::{
    CrewNpc, MoveTarget, NoiseEmitter, NpcActionEvent, NpcActionKind, NullDrone, Player,
    PowerJunction, WaterPump, WorldFeedbackEvent,
};
use crate::resources::{
    BiometricsCtx, EnergyWell, LeviathanState, PhysicsWorldRes, SettlementMetrics, SleepPhase,
};
use symtropy_render_bridge::PhysicsBody;

/// Run the FEP perception-action cycle for each crew NPC.
pub fn fep_behavior_system(
    mut npcs: Query<
        (
            Entity,
            &mut CrewNpc,
            &Transform,
            &mut MoveTarget,
            &mut NoiseEmitter,
            &PhysicsBody,
            Option<&mut crate::systems::psychology::PsychologicalNeeds>,
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
    power_junctions: Query<(&Transform, &PowerJunction)>,
    water_pumps: Query<(&Transform, &WaterPump)>,
    drones: Query<(&Transform, &NullDrone)>,
    time: Res<Time>,
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

    // Pre-gather all NPC basic info to avoid query/borrow conflicts
    struct NpcInfo {
        entity: Entity,
        name: String,
        pos: Vec2,
        allostatic_load: f32,
    }
    let npc_infos: Vec<NpcInfo> = npcs
        .iter()
        .map(|(entity, npc, tf, _, _, _, psych)| NpcInfo {
            entity,
            name: npc.name.clone(),
            pos: tf.translation.truncate(),
            allostatic_load: psych.as_ref().map_or(0.0, |p| p.allostatic_load),
        })
        .collect();

    for (entity, mut npc, npc_tf, mut target, mut noise, body, mut psych) in &mut npcs {
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

        // ARCHETYPE-SPECIFIC ACTIVE INFERENCE ACTIONS & GOALS

        // 1. Kael (Engineer) & Leo (Young Tech)
        if npc.name.contains("Kael") || npc.name.contains("Leo") {
            let mut closest_junction: Option<(Vec2, f32)> = None;
            for (junction_tf, junction) in &power_junctions {
                if junction.is_damaged {
                    let j_pos = junction_tf.translation.truncate();
                    let dist = npc_pos.distance(j_pos);
                    if closest_junction.map_or(true, |(_, d)| dist < d) {
                        closest_junction = Some((j_pos, dist));
                    }
                }
            }
            if let Some((j_pos, _)) = closest_junction {
                let to_junction = nalgebra::SVector::from([
                    (j_pos.x - npc_pos.x) as f64,
                    (j_pos.y - npc_pos.y) as f64,
                ])
                .normalize();
                direction = direction * 0.4 + to_junction * 0.6;
            }

            // If Leo is close to Kael, scale down Leo's anxiety/stress over time
            if npc.name.contains("Leo") {
                let kael_pos = npc_infos
                    .iter()
                    .find(|info| info.name.contains("Kael"))
                    .map(|info| info.pos);
                if let Some(kael_p) = kael_pos {
                    if npc_pos.distance(kael_p) < 100.0 {
                        if let Some(ref mut p) = psych {
                            p.allostatic_load =
                                (p.allostatic_load - 0.05 * time.delta_secs()).max(0.0);
                        }
                    }
                }
            }
        }

        // 2. Mira (Medic)
        if npc.name.contains("Mira") {
            let mut closest_stressed: Option<(Vec2, f32)> = None;
            for info in &npc_infos {
                if info.entity != entity && info.allostatic_load > 0.4 {
                    let dist = npc_pos.distance(info.pos);
                    if closest_stressed.map_or(true, |(_, d)| dist < d) {
                        closest_stressed = Some((info.pos, dist));
                    }
                }
            }
            if let Some((c_pos, _)) = closest_stressed {
                let to_crew = nalgebra::SVector::from([
                    (c_pos.x - npc_pos.x) as f64,
                    (c_pos.y - npc_pos.y) as f64,
                ])
                .normalize();
                direction = direction * 0.3 + to_crew * 0.7;
            }
        }

        // 3. Jack (Convoy Lead)
        if npc.name.contains("Jack") {
            let mut closest_drone: Option<(Vec2, f32)> = None;
            for (drone_tf, _) in &drones {
                let d_pos = drone_tf.translation.truncate();
                let dist = npc_pos.distance(d_pos);
                if dist <= 250.0 {
                    if closest_drone.map_or(true, |(_, d)| dist < d) {
                        closest_drone = Some((d_pos, dist));
                    }
                }
            }
            if let Some((d_pos, _)) = closest_drone {
                let to_drone = nalgebra::SVector::from([
                    (d_pos.x - npc_pos.x) as f64,
                    (d_pos.y - npc_pos.y) as f64,
                ])
                .normalize();
                direction = direction * 0.2 + to_drone * 0.8;
            }
        }

        // 4. PR-4 (Robot)
        if npc.name.contains("PR-4") {
            let mut closest_pump: Option<(Vec2, f32)> = None;
            for (pump_tf, pump) in &water_pumps {
                if pump.is_sabotaged {
                    let p_pos = pump_tf.translation.truncate();
                    let dist = npc_pos.distance(p_pos);
                    if closest_pump.map_or(true, |(_, d)| dist < d) {
                        closest_pump = Some((p_pos, dist));
                    }
                }
            }
            if let Some((p_pos, _)) = closest_pump {
                let to_pump = nalgebra::SVector::from([
                    (p_pos.x - npc_pos.x) as f64,
                    (p_pos.y - npc_pos.y) as f64,
                ])
                .normalize();
                direction = direction * 0.4 + to_pump * 0.6;
            }
        }

        let dir_vec = Vec2::new(direction[0] as f32, direction[1] as f32);
        let load = psych.as_ref().map(|p| p.allostatic_load).unwrap_or(0.0);
        let engagement = psych.as_ref().map(|p| p.engagement).unwrap_or(1.0);

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

/// System applying repairs, healing, and drone neutralization when adjacent to targets.
/// System applying repairs, healing, and drone neutralization when adjacent to targets.
pub fn npc_action_system(
    actors: Query<(Entity, &CrewNpc, &Transform)>,
    mut needs_query: Query<&mut crate::systems::psychology::PsychologicalNeeds>,
    mut power_junctions: Query<(&Transform, &mut PowerJunction)>,
    mut water_pumps: Query<(&Transform, &mut WaterPump)>,
    mut drones: Query<(Entity, &Transform, &mut NullDrone)>,
    mut commands: Commands,
    time: Res<Time>,
    mut action_writer: MessageWriter<NpcActionEvent>,
    mut feedback_writer: MessageWriter<WorldFeedbackEvent>,
) {
    let dt = time.delta_secs();

    for (actor_entity, npc, tf) in &actors {
        let npc_pos = tf.translation.truncate();

        // 1. Kael (Engineer) repairs PowerJunction
        if npc.name.contains("Kael") {
            for (j_tf, mut junction) in &mut power_junctions {
                if junction.is_damaged {
                    let j_pos = j_tf.translation.truncate();
                    if npc_pos.distance(j_pos) < 30.0 {
                        let is_pr4_adjacent = actors.iter().any(|(_, other_npc, other_tf)| {
                            other_npc.name.contains("PR-4")
                                && other_tf.translation.truncate().distance(j_pos) < 30.0
                        });

                        if is_pr4_adjacent {
                            junction.is_damaged = false;
                            junction.output = 1.0;
                            action_writer.write(NpcActionEvent {
                                actor: actor_entity,
                                actor_name: npc.name.clone(),
                                target: None,
                                target_name: "Power Junction".to_string(),
                                action_kind: NpcActionKind::RepairJunction,
                                intensity: 1.0,
                                success_delta: 1.0,
                                settlement_metric_delta: 0.2,
                            });
                            feedback_writer.write(WorldFeedbackEvent {
                                position: j_pos,
                                message: "JUNCTION STABILIZED (100%)".to_string(),
                                color: Color::srgb(0.2, 0.9, 0.4),
                            });
                        } else if junction.output < 0.8 {
                            let old_out = junction.output;
                            junction.output = (junction.output + 0.3 * dt).min(0.8);
                            if old_out < 0.8 && junction.output >= 0.8 {
                                action_writer.write(NpcActionEvent {
                                    actor: actor_entity,
                                    actor_name: npc.name.clone(),
                                    target: None,
                                    target_name: "Power Junction".to_string(),
                                    action_kind: NpcActionKind::RepairJunction,
                                    intensity: 0.5,
                                    success_delta: 0.8,
                                    settlement_metric_delta: 0.1,
                                });
                                feedback_writer.write(WorldFeedbackEvent {
                                    position: j_pos,
                                    message: "JUNCTION RESTORED TO 80% (NEEDS PR-4)".to_string(),
                                    color: Color::srgb(0.9, 0.6, 0.2),
                                });
                            }
                        }
                    }
                }
            }
        }

        // 2. PR-4 (Robot) repairs WaterPump
        if npc.name.contains("PR-4") {
            for (p_tf, mut pump) in &mut water_pumps {
                if pump.is_sabotaged {
                    let p_pos = p_tf.translation.truncate();
                    if npc_pos.distance(p_pos) < 30.0 {
                        let is_assistant_adjacent =
                            actors.iter().any(|(_, other_npc, other_tf)| {
                                (other_npc.name.contains("Nadia")
                                    || other_npc.name.contains("Soren"))
                                    && other_tf.translation.truncate().distance(p_pos) < 30.0
                            });

                        if is_assistant_adjacent {
                            pump.is_sabotaged = false;
                            pump.efficiency = 1.0;
                            pump.is_running = true;
                            action_writer.write(NpcActionEvent {
                                actor: actor_entity,
                                actor_name: npc.name.clone(),
                                target: None,
                                target_name: "Water Pump".to_string(),
                                action_kind: NpcActionKind::RepairPump,
                                intensity: 1.0,
                                success_delta: 1.0,
                                settlement_metric_delta: 0.2,
                            });
                            feedback_writer.write(WorldFeedbackEvent {
                                position: p_pos,
                                message: "WATER PUMP PURIFIED (100%)".to_string(),
                                color: Color::srgb(0.2, 0.9, 0.4),
                            });
                        } else if pump.efficiency < 0.7 {
                            let old_eff = pump.efficiency;
                            pump.efficiency = (pump.efficiency + 0.3 * dt).min(0.7);
                            pump.is_running = true;
                            if old_eff < 0.7 && pump.efficiency >= 0.7 {
                                pump.is_sabotaged = false; // Online but partial
                                action_writer.write(NpcActionEvent {
                                    actor: actor_entity,
                                    actor_name: npc.name.clone(),
                                    target: None,
                                    target_name: "Water Pump".to_string(),
                                    action_kind: NpcActionKind::RepairPump,
                                    intensity: 0.5,
                                    success_delta: 0.7,
                                    settlement_metric_delta: 0.1,
                                });
                                feedback_writer.write(WorldFeedbackEvent {
                                    position: p_pos,
                                    message: "WATER PUMP ONLINE (CONTAMINATED)".to_string(),
                                    color: Color::srgb(0.9, 0.6, 0.2),
                                });
                            }
                        }
                    }
                }
            }
        }

        // 3. Mira (Medic) heals stressed crew member
        if npc.name.contains("Mira") {
            for (other_entity, other_npc, other_tf) in &actors {
                if other_entity != actor_entity {
                    let other_pos = other_tf.translation.truncate();
                    if npc_pos.distance(other_pos) < 30.0 {
                        if let Ok(mut other_psych) = needs_query.get_mut(other_entity) {
                            if other_psych.allostatic_load > 0.4 {
                                let kael_pos = actors
                                    .iter()
                                    .find(|(_, o_npc, _)| o_npc.name.contains("Kael"))
                                    .map(|(_, _, o_tf)| o_tf.translation.truncate());
                                let kael_far =
                                    kael_pos.map_or(true, |kp| other_pos.distance(kp) > 120.0);

                                if kael_far && other_npc.name.contains("Leo") {
                                    // Relapse state
                                    other_psych.allostatic_load =
                                        (other_psych.allostatic_load + 0.05 * dt).min(1.0);

                                    // Trigger relapse warning event and label slowly
                                    if rand::random::<f32>() < 0.01 {
                                        action_writer.write(NpcActionEvent {
                                            actor: actor_entity,
                                            actor_name: npc.name.clone(),
                                            target: Some(other_entity),
                                            target_name: other_npc.name.clone(),
                                            action_kind: NpcActionKind::HealStress,
                                            intensity: 0.0,
                                            success_delta: -0.1,
                                            settlement_metric_delta: -0.05,
                                        });
                                        feedback_writer.write(WorldFeedbackEvent {
                                            position: other_pos,
                                            message: "LEO RELAPSING (KAEL FAR)".to_string(),
                                            color: Color::srgb(0.9, 0.2, 0.2),
                                        });
                                    }
                                } else {
                                    // Successful healing
                                    let old_load = other_psych.allostatic_load;
                                    other_psych.allostatic_load =
                                        (other_psych.allostatic_load - 0.15 * dt).max(0.0);
                                    other_psych.social_satiation =
                                        (other_psych.social_satiation + 0.1 * dt).min(1.0);

                                    // Emit healer success event
                                    if old_load > 0.4 && other_psych.allostatic_load <= 0.4 {
                                        action_writer.write(NpcActionEvent {
                                            actor: actor_entity,
                                            actor_name: npc.name.clone(),
                                            target: Some(other_entity),
                                            target_name: other_npc.name.clone(),
                                            action_kind: NpcActionKind::HealStress,
                                            intensity: 1.0,
                                            success_delta: 0.15,
                                            settlement_metric_delta: 0.1,
                                        });
                                        feedback_writer.write(WorldFeedbackEvent {
                                            position: other_pos,
                                            message: "CREW STRESS STABILIZED (-15%)".to_string(),
                                            color: Color::srgb(0.2, 0.8, 0.9),
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // 4. Jack (Convoy Lead) deals damage to / destroys NullDrone
        if npc.name.contains("Jack") {
            for (drone_entity, d_tf, mut drone) in &mut drones {
                let d_pos = d_tf.translation.truncate();
                if npc_pos.distance(d_pos) < 30.0 {
                    drone.integrity -= 0.5 * dt;

                    // Gunfire stress: increase stress of nearby crew members
                    for (other_ent, other_npc, other_tf) in &actors {
                        if other_ent != actor_entity {
                            let dist = other_tf.translation.truncate().distance(npc_pos);
                            if dist < 150.0 {
                                if let Ok(mut other_psych) = needs_query.get_mut(other_ent) {
                                    other_psych.allostatic_load =
                                        (other_psych.allostatic_load + 0.02 * dt).min(1.0);
                                }
                            }
                        }
                    }

                    if drone.integrity <= 0.0 {
                        commands.entity(drone_entity).despawn();

                        action_writer.write(NpcActionEvent {
                            actor: actor_entity,
                            actor_name: npc.name.clone(),
                            target: None,
                            target_name: "NullDrone".to_string(),
                            action_kind: NpcActionKind::CombatDrone,
                            intensity: 1.0,
                            success_delta: 1.0,
                            settlement_metric_delta: 0.15,
                        });
                        feedback_writer.write(WorldFeedbackEvent {
                            position: d_pos,
                            message: "NULL DRONE DISABLED".to_string(),
                            color: Color::srgb(0.9, 0.2, 0.2),
                        });
                    }
                }
            }
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
