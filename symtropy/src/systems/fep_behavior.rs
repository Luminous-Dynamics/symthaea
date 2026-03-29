// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! FEP-driven NPC behavior: perception-action cycle for crew members.

use bevy::prelude::*;
use symthaea_fep::Observation;

use crate::components::{CrewNpc, MoveTarget, NoiseEmitter, Player};
use crate::resources::{BiometricsCtx, LeviathanState, SleepPhase};

const ACTION_ATTENTION_SHIFT: usize = 0;
const ACTION_LEARNING_RATE: usize = 1;
const ACTION_EXPLORATION: usize = 2;
const ACTION_REFLECTION: usize = 3;
#[allow(dead_code)]
const ACTION_MEMORY: usize = 4;
const ACTION_EXPECTATION_RESET: usize = 5;
const ACTION_MOTOR_OUTPUT: usize = 6;

/// Run the FEP perception-action cycle for each crew NPC.
pub fn fep_behavior_system(
    mut npcs: Query<(&mut CrewNpc, &Transform, &mut MoveTarget, &mut NoiseEmitter)>,
    player_query: Query<&Transform, With<Player>>,
    biometrics: Res<BiometricsCtx>,
    leviathan: Res<LeviathanState>,
    time: Res<Time>,
) {
    let Ok(player_tf) = player_query.single() else { return };
    let player_pos = player_tf.translation.truncate();

    for (mut npc, npc_tf, mut target, mut noise) in &mut npcs {
        let npc_pos = npc_tf.translation.truncate();
        let dist_to_player = npc_pos.distance(player_pos);

        let danger = match leviathan.phase {
            SleepPhase::Dormant => 0.0,
            SleepPhase::Stirring => 0.3,
            SleepPhase::Awake => 0.7,
            SleepPhase::Hunting => 1.0,
        };

        let obs = Observation::new(
            vec![
                (dist_to_player / 500.0).min(1.0) as f64,
                biometrics.encoder.compute_stress_vector().arousal as f64,
                danger as f64,
                npc.caution as f64,
            ],
            0.8,
            "game",
        );

        let _perception = npc.fep.perceive(&obs);
        let action = npc.fep.select_action();

        match action.action {
            ACTION_EXPLORATION => {
                if dist_to_player > 100.0 {
                    target.target = Some(player_pos);
                    target.speed = 80.0 * (1.0 - npc.caution * 0.5);
                } else {
                    let angle = time.elapsed_secs() * 0.5 + npc.caution * 3.14;
                    target.target = Some(player_pos + Vec2::new(angle.cos() * 80.0, angle.sin() * 80.0));
                    target.speed = 60.0;
                }
                noise.level = 0.05;
            }
            ACTION_ATTENTION_SHIFT => {
                target.target = Some(player_pos);
                target.speed = 0.0;
                noise.level = 0.0;
            }
            ACTION_REFLECTION => {
                target.target = None;
                target.speed = 0.0;
                noise.level = 0.0;
            }
            ACTION_EXPECTATION_RESET => {
                target.target = None;
                target.speed = 0.0;
                noise.level = 0.6;
            }
            ACTION_LEARNING_RATE => {
                if danger > 0.5 {
                    npc.caution = (npc.caution + 0.1).min(1.0);
                } else {
                    npc.caution = (npc.caution - 0.05).max(0.0);
                }
                noise.level = 0.0;
            }
            ACTION_MOTOR_OUTPUT => {
                target.target = Some(player_pos);
                target.speed = 70.0;
                noise.level = 0.03;
            }
            _ => {
                target.target = None;
                noise.level = 0.0;
            }
        }
    }
}

/// Apply NPC movement intent to physics bodies.
///
/// Reads MoveTarget (set by FEP behavior) and sets velocity on the NPC's
/// physics body. Wall collision handled by the physics system via TileGrid.
pub fn npc_movement_system(
    mut query: Query<(&Transform, &MoveTarget, &symtropy_render_bridge::PhysicsBody), With<CrewNpc>>,
    mut physics: ResMut<crate::resources::PhysicsWorldRes>,
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

                    // TileGrid wall collision filter
                    if let Some(ref grid) = tile_grid {
                        let dt = 1.0 / 64.0_f32;
                        let new_x = tf.translation.x + vx as f32 * dt;
                        let new_y = tf.translation.y + vy as f32 * dt;
                        if !grid.is_walkable(new_x, tf.translation.y) { vx = 0.0; }
                        if !grid.is_walkable(tf.translation.x, new_y) { vy = 0.0; }
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
