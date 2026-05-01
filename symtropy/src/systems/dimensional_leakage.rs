// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dimensional energy leakage in the game world.
//!
//! Energy sinks drain agent energy via the 4th dimension (1/r² falloff).
//! Energy sources inject energy from the bulk. Wormholes link the two.
//!
//! High-Phi players can detect leakage points via the consciousness
//! perception gate (see consciousness_aura.rs). Low-Phi players just
//! experience mysterious energy drain with no explanation.

use bevy::prelude::*;

use crate::components::{CrewNpc, Player};
use crate::resources::PhysicsWorldRes;
use symtropy_render_bridge::PhysicsBody;

/// Marker component for a dimensional leakage point in the game world.
#[derive(Component)]
pub struct LeakageMarker {
    /// Is this a sink (drains energy) or source (injects energy)?
    pub is_sink: bool,
    /// Energy flow rate (J/sec, positive = source, negative = sink).
    pub flow_rate: f32,
    /// Effect radius in pixels.
    pub radius: f32,
    /// W-coordinate (how deep in the 4th dimension).
    pub w_depth: f32,
    /// Total energy transferred through this point.
    pub total_transferred: f32,
}

/// Timer for leakage effects (runs every 0.5 seconds to avoid per-frame cost).
#[derive(Resource)]
pub struct LeakageTimer(pub f32);

impl Default for LeakageTimer {
    fn default() -> Self {
        Self(0.0)
    }
}

/// Spawn dimensional leakage points in the dungeon.
/// Called during level setup (Loading phase). Cleans up previous leakage on restart.
pub fn spawn_leakage_points(
    mut commands: Commands,
    tiles: Query<(&Transform, &crate::components::Tile)>,
    existing: Query<Entity, With<LeakageMarker>>,
    seed: Res<crate::resources::DungeonSeed>,
) {
    // Clean up previous leakage points on restart
    for entity in &existing {
        commands.entity(entity).despawn();
    }

    let walkable: Vec<Vec2> = tiles
        .iter()
        .filter(|(_, t)| t.walkable)
        .map(|(tf, _)| tf.translation.truncate())
        .collect();

    if walkable.len() < 10 {
        return;
    }

    // Place 2 sink/source pairs (wormholes), seeded from dungeon
    let mut rng_seed = seed.0.wrapping_mul(7919);
    let nr = |s: &mut u64| -> f64 {
        *s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (*s >> 11) as f64 / (1u64 << 53) as f64
    };

    let num_pairs = 2;
    for pair in 0..num_pairs {
        let sink_idx = (nr(&mut rng_seed) * walkable.len() as f64) as usize % walkable.len();
        let source_idx = (nr(&mut rng_seed) * walkable.len() as f64) as usize % walkable.len();

        let sink_pos = walkable[sink_idx];
        let source_pos = walkable[source_idx];
        let w_depth = 5.0 + pair as f32 * 10.0;

        // Energy sink — drains nearby agents
        commands.spawn((
            Sprite::from_color(
                Color::srgba(0.4, 0.1, 0.8, 0.0), // starts invisible — revealed by Phi
                Vec2::splat(20.0),
            ),
            Transform::from_xyz(sink_pos.x, sink_pos.y, 1.8),
            LeakageMarker {
                is_sink: true,
                flow_rate: -0.8, // J/sec drain
                radius: 80.0,
                w_depth,
                total_transferred: 0.0,
            },
            crate::systems::four_d_rendering::FourDBody::new(
                w_depth as f64,
                Color::srgba(0.4, 0.1, 0.8, 0.3),
            ),
            crate::systems::four_d_rendering::FourDOnly,
        ));

        // Energy source — injects energy
        commands.spawn((
            Sprite::from_color(
                Color::srgba(0.1, 0.8, 0.4, 0.0), // starts invisible
                Vec2::splat(20.0),
            ),
            Transform::from_xyz(source_pos.x, source_pos.y, 1.8),
            LeakageMarker {
                is_sink: false,
                flow_rate: 0.6, // J/sec injection (less than drain — net loss to 4D)
                radius: 80.0,
                w_depth,
                total_transferred: 0.0,
            },
            crate::systems::four_d_rendering::FourDBody::new(
                w_depth as f64,
                Color::srgba(0.1, 0.8, 0.4, 0.3),
            ),
            crate::systems::four_d_rendering::FourDOnly,
        ));
    }

    info!(
        "Spawned {} dimensional wormholes ({} leakage points)",
        num_pairs,
        num_pairs * 2
    );
}

/// Apply dimensional leakage effects to nearby agents.
///
/// Sinks drain energy (mysterious loss). Sources inject energy (mysterious gain).
/// Agents near leakage points accumulate prediction errors — their 3D world model
/// cannot explain the energy flow.
pub fn dimensional_leakage_system(
    mut leakage_points: Query<
        (&Transform, &mut LeakageMarker),
        (Without<Player>, Without<CrewNpc>),
    >,
    agents: Query<
        (&Transform, &PhysicsBody),
        (Or<(With<Player>, With<CrewNpc>)>, Without<LeakageMarker>),
    >,
    mut physics: ResMut<PhysicsWorldRes>,
    mut timer: ResMut<LeakageTimer>,
    time: Res<Time>,
) {
    timer.0 += time.delta_secs();
    if timer.0 < 0.5 {
        return;
    } // every 0.5 seconds
    timer.0 = 0.0;

    for (leak_tf, mut marker) in &mut leakage_points {
        let leak_pos = leak_tf.translation.truncate();

        for (agent_tf, body) in &agents {
            let agent_pos = agent_tf.translation.truncate();
            let dist = leak_pos.distance(agent_pos);

            if dist >= marker.radius {
                continue;
            }

            // 1/r² falloff, clamped at r=8 to avoid singularity
            let r = dist.max(8.0);
            let effect = marker.flow_rate * (64.0 / (r * r)); // normalize to radius

            if let Some(entity) = physics.consciousness.entities.get_mut(&body.handle) {
                if effect < 0.0 {
                    // Sink: drain energy (mysterious loss)
                    entity.energy.consume((-effect * 0.5) as f64);
                    // Prediction error spike — agent can't explain the drain
                    entity.prediction_error += (-effect * 0.1) as f64;
                    entity.motor_precision = 1.0 / (1.0 + entity.prediction_error.abs().min(100.0));
                } else {
                    // Source: inject energy (mysterious gain)
                    entity.energy.regenerate((effect * 0.5) as f64);
                }
                marker.total_transferred += effect.abs() * 0.5;
            }
        }
    }
}

/// Visual shimmer for active leakage points — pulses proportional to flow.
pub fn leakage_visual_system(
    mut leakage_points: Query<
        (&mut Sprite, &LeakageMarker),
        Without<crate::systems::consciousness_aura::AuraSprite>,
    >,
    time: Res<Time>,
    player_c: Res<crate::systems::consciousness::PlayerConsciousness>,
) {
    let t = time.elapsed_secs();
    let phi = player_c.level as f32;

    // Visibility scales with player consciousness
    let perception = ((phi - 0.4) / 0.3).clamp(0.0, 1.0);

    for (mut sprite, marker) in &mut leakage_points {
        if perception <= 0.0 {
            sprite.color = Color::srgba(0.0, 0.0, 0.0, 0.0);
            continue;
        }

        let pulse = (t * 2.0 + marker.w_depth).sin() * 0.3 + 0.7;
        let alpha = perception * pulse * 0.4;

        if marker.is_sink {
            // Purple shimmer for sinks (energy draining into 4D)
            sprite.color = Color::srgba(0.5, 0.1, 0.9, alpha);
        } else {
            // Green shimmer for sources (energy flowing from 4D)
            sprite.color = Color::srgba(0.1, 0.9, 0.4, alpha);
        }
    }
}
