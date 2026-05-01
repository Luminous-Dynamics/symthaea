// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Physics engine systems — bridge between symtropy-physics and Bevy ECS.
//!
//! Three systems running in FixedUpdate:
//! 1. `physics_apply_inputs` — reads PlayerInput, sets velocity on player body
//! 2. `physics_step` — steps the PhysicsWorld forward by fixed dt
//! 3. `physics_sync_transforms` — reads body positions, writes to Bevy Transform

use bevy::prelude::*;

use crate::components::{CrewNpc, Player};
use crate::resources::{PhysicsWorldRes, PlayerInput, TileGrid};
use crate::systems::consciousness::{NpcConsciousness, PlayerConsciousness};
use symtropy_render_bridge::PhysicsBody;

/// Walk speed in physics units/second.
const WALK_SPEED: f64 = 100.0;
/// Sprint speed in physics units/second.
const SPRINT_SPEED: f64 = 200.0;

/// Read PlayerInput and set velocity on the player's physics body.
///
/// Also applies TileGrid wall collision as a velocity filter:
/// if the target position would be inside a wall, zero that velocity axis.
pub fn physics_apply_inputs(
    input: Res<PlayerInput>,
    mut physics: ResMut<PhysicsWorldRes>,
    tile_grid: Option<Res<TileGrid>>,
    players: Query<(&PhysicsBody, &Transform), With<Player>>,
) {
    for (body_comp, transform) in &players {
        let speed = if input.sprinting {
            SPRINT_SPEED
        } else {
            WALK_SPEED
        };
        let dir = input.direction;

        if dir.length_squared() < 1e-6 {
            if let Some(body) = physics.world.body_mut(body_comp.handle) {
                body.linear_velocity = nalgebra::SVector::from([0.0, 0.0]);
            }
            continue;
        }

        // Rule 1: Movement costs energy
        let dt = 1.0 / 64.0;
        let displacement = speed * dt;
        let sprint_mult = if input.sprinting {
            physics.consciousness.constants.sprint_cost_multiplier
        } else {
            1.0
        };
        let move_cost =
            displacement * physics.consciousness.constants.movement_cost_per_unit * sprint_mult;
        let energy_consumed = physics
            .consciousness
            .consume_energy(body_comp.handle, move_cost);

        // Gradual slowdown: speed scales with energy fraction consumed
        let energy_fraction = if move_cost > 1e-10 {
            energy_consumed / move_cost
        } else {
            1.0
        };
        if energy_fraction < 0.01 {
            if let Some(body) = physics.world.body_mut(body_comp.handle) {
                body.linear_velocity = nalgebra::SVector::from([0.0, 0.0]);
            }
            continue;
        }
        let speed = speed * energy_fraction; // Smooth deceleration as energy depletes

        if let Some(body) = physics.world.body_mut(body_comp.handle) {
            let norm_dir = dir.normalize();
            let mut vx = norm_dir.x as f64 * speed;
            let mut vy = norm_dir.y as f64 * speed;

            // TileGrid wall collision filter — check if target position is walkable.
            // This preserves the existing wall-sliding behavior from the tile-based system.
            if let Some(ref grid) = tile_grid {
                let dt = 1.0 / 64.0; // FixedUpdate default rate
                let new_x = transform.translation.x + vx as f32 * dt as f32;
                let new_y = transform.translation.y + vy as f32 * dt as f32;

                if !grid.is_walkable(new_x, transform.translation.y) {
                    vx = 0.0;
                }
                if !grid.is_walkable(transform.translation.x, new_y) {
                    vy = 0.0;
                }
            }

            body.linear_velocity = nalgebra::SVector::from([vx, vy]);
        }
    }
}

/// Step the physics world forward by the fixed timestep.
///
/// Uses `step_with_callback` to wire consciousness INTO the physics loop:
/// - Sanctuary zones dampen collision impulses
/// - Harmony fields modulate friction
/// - Collisions spike prediction error → reduce motor precision
/// - Energy dissipation tracked in thermodynamic ledger
pub fn physics_step(mut physics: ResMut<PhysicsWorldRes>, time: Res<Time<Fixed>>) {
    let dt = time.delta_secs_f64();
    let PhysicsWorldRes {
        ref mut world,
        ref mut consciousness,
    } = *physics;
    // Tick prediction error decay (habituation)
    consciousness.tick_prediction_errors();
    // Step physics with consciousness callback
    world.step_with_callback(dt, consciousness);
}

/// Sync consciousness state from the game's consciousness systems into the
/// physics consciousness field. This is what makes consciousness AFFECT physics.
///
/// Runs after the consciousness systems compute Φ and before the physics step.
pub fn consciousness_sync_system(
    mut physics: ResMut<PhysicsWorldRes>,
    player_c: Res<PlayerConsciousness>,
    biometrics: Res<crate::resources::BiometricsCtx>,
    harmony: Res<crate::systems::harmonies::LocalHarmonyState>,
    players: Query<(&PhysicsBody, &Transform), With<Player>>,
    npcs: Query<(&PhysicsBody, &Transform, &NpcConsciousness), With<CrewNpc>>,
) {
    // Sync player consciousness — use REAL values from the biometrics system
    let biometrics_load = biometrics.model.allostatic_load as f64;
    for (body_comp, transform) in &players {
        let inputs = symthaea_consciousness_equation::ConsciousnessInputs {
            phi: (1.0 - biometrics_load) * 0.8 + 0.2,
            broadcast: (1.0 - biometrics_load * 0.5).max(0.1),
            working_memory: (1.0 - biometrics_load * 0.6).max(0.1),
            attention: (biometrics_load * 0.3 + 0.5).min(1.0),
            recurrence: (harmony.total_energy as f64 / 8.0).min(1.0),
            embodiment: 0.7,
            knowledge: (harmony.activations.iter().filter(|&&a| a > 0.3).count() as f64 / 8.0)
                .min(1.0),
            synchrony: if harmony.is_sanctuary {
                0.9
            } else {
                0.4 + harmony.total_energy as f64 * 0.05
            },
        };
        let pos = symtropy_math::Point::new([
            transform.translation.x as f64,
            transform.translation.y as f64,
        ]);
        physics
            .consciousness
            .update_entity(body_comp.handle, &inputs, pos);

        // Sync harmony activations
        if let Some(entity) = physics.consciousness.entities.get_mut(&body_comp.handle) {
            entity.harmony_activations = [
                harmony.activations[0] as f64,
                harmony.activations[1] as f64,
                harmony.activations[2] as f64,
                harmony.activations[3] as f64,
                harmony.activations[4] as f64,
                harmony.activations[5] as f64,
                harmony.activations[6] as f64,
                harmony.activations[7] as f64,
            ];
        }
    }

    // Sync NPC consciousness — use real values from NPC consciousness system
    for (body_comp, transform, npc_c) in &npcs {
        let inputs = symthaea_consciousness_equation::ConsciousnessInputs {
            phi: npc_c.level.max(0.1),
            broadcast: (1.0 - npc_c.level * 0.3).max(0.2),
            working_memory: 0.5 + npc_c.stability * 0.3,
            attention: 0.5 + npc_c.level * 0.3,
            recurrence: (harmony.total_energy as f64 / 8.0).min(1.0),
            embodiment: (0.5 + npc_c.level * 0.3).min(1.0),
            knowledge: 0.4 + npc_c.stability * 0.2,
            synchrony: if harmony.is_sanctuary {
                0.8
            } else {
                0.3 + harmony.total_energy as f64 * 0.04
            },
        };
        let pos = symtropy_math::Point::new([
            transform.translation.x as f64,
            transform.translation.y as f64,
        ]);
        physics
            .consciousness
            .update_entity(body_comp.handle, &inputs, pos);
    }

    // Rebuild harmony field from all entity positions for Channel 4 (Harmony → Friction)
    let positions: Vec<_> = players
        .iter()
        .chain(npcs.iter().map(|(b, t, _)| (b, t)))
        .map(|(body_comp, transform)| {
            (
                body_comp.handle,
                symtropy_math::Point::new([
                    transform.translation.x as f64,
                    transform.translation.y as f64,
                ]),
            )
        })
        .collect();
    physics.consciousness.rebuild_harmony_field(&positions);
}

/// Sync physics body positions back to Bevy Transforms.
pub fn physics_sync_transforms(
    physics: Res<PhysicsWorldRes>,
    mut query: Query<(&PhysicsBody, &mut Transform)>,
) {
    for (body_comp, mut transform) in &mut query {
        if let Some(body) = physics.world.body(body_comp.handle) {
            let pos = body.position();
            transform.translation.x = pos.coord(0) as f32;
            transform.translation.y = pos.coord(1) as f32;
        }
    }
}
