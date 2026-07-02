// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness Aura System: making the math visible.
//!
//! Every agent emits a visual aura whose geometry IS their consciousness state:
//!
//! - **Radius** = energy fraction (depleted agents dim, healthy agents glow)
//! - **Shape** = Phi level:
//!   - High Phi (integrated): smooth sine wave boundary (resonant, coherent)
//!   - Low Phi (fragmented): jagged noise boundary (static, chaotic)
//! - **Color** = dominant harmony color, blended with energy state
//! - **Opacity** = prediction error inverse (high error = flickering)
//!
//! The aura is rendered as a ring of small sprites around each agent,
//! with their positions modulated by the agent's consciousness state.
//! The player can read an NPC's biological state from the geometry of their light.
//!
//! Harmony resonance waves: when two agents achieve high resonance,
//! an animated wave propagates outward, tinting tiles and reducing friction.

use bevy::prelude::*;

use crate::components::{CrewNpc, Player};
use crate::resources::PhysicsWorldRes;
use crate::systems::consciousness::NpcConsciousness;
use crate::systems::psychology::PsychologicalNeeds;
use symtropy_render_bridge::PhysicsBody;

/// Number of sprites in each entity's aura ring.
const AURA_SEGMENTS: usize = 24;

/// Maximum aura radius (pixels) at full energy.
const MAX_AURA_RADIUS: f32 = 48.0;

/// Minimum aura radius (always visible if alive).
const MIN_AURA_RADIUS: f32 = 12.0;

/// Aura sprite size.
const AURA_SPRITE_SIZE: f32 = 4.0;

/// Marker for aura ring sprites.
#[derive(Component)]
pub struct AuraSprite {
    /// Which entity this aura belongs to.
    pub owner: Entity,
    /// Angle in the ring [0, 2pi).
    pub angle: f32,
    /// Segment index for noise computation.
    pub segment: usize,
}

/// Marker for harmony resonance wave sprites.
#[derive(Component)]
pub struct ResonanceWave {
    /// Center position.
    pub origin: Vec2,
    /// Current radius (expanding).
    pub radius: f32,
    /// Expansion speed (px/sec).
    pub speed: f32,
    /// Remaining lifetime [0, 1].
    pub life: f32,
    /// Wave color.
    pub color: Color,
}

/// Timer for resonance wave emission.
#[derive(Resource)]
pub struct ResonanceWaveTimer(pub f32);

impl Default for ResonanceWaveTimer {
    fn default() -> Self {
        Self(0.0)
    }
}

/// Spawn aura rings around player and NPCs when they enter the game.
/// Cleans up any existing auras first to prevent memory leaks on restart.
pub fn spawn_auras(
    mut commands: Commands,
    new_entities: Query<Entity, Or<(Added<Player>, Added<CrewNpc>)>>,
    existing_auras: Query<Entity, With<AuraSprite>>,
) {
    // Clean up orphaned auras from previous game sessions
    if !new_entities.is_empty() {
        for entity in &existing_auras {
            commands.entity(entity).despawn();
        }
    }
    for entity in &new_entities {
        for seg in 0..AURA_SEGMENTS {
            let angle = (seg as f32 / AURA_SEGMENTS as f32) * std::f32::consts::TAU;
            commands.spawn((
                Sprite::from_color(
                    Color::srgba(0.5, 0.8, 0.6, 0.0), // starts invisible
                    Vec2::splat(AURA_SPRITE_SIZE),
                ),
                Transform::from_xyz(0.0, 0.0, 2.5), // above floor, below entity
                AuraSprite {
                    owner: entity,
                    angle,
                    segment: seg,
                },
            ));
        }
    }
}

/// Update aura geometry every frame — the shape IS the consciousness state.
pub fn aura_update_system(
    entities: Query<
        (
            Entity,
            &Transform,
            Option<&PhysicsBody>,
            Option<&NpcConsciousness>,
            Option<&PsychologicalNeeds>,
        ),
        (Or<(With<Player>, With<CrewNpc>)>, Without<AuraSprite>),
    >,
    mut auras: Query<
        (&mut Transform, &mut Sprite, &AuraSprite),
        (Without<Player>, Without<CrewNpc>),
    >,
    physics: Res<PhysicsWorldRes>,
    harmony: Res<crate::systems::harmonies::LocalHarmonyState>,
    player_consciousness: Res<crate::systems::consciousness::PlayerConsciousness>,
    time: Res<Time>,
) {
    let t = time.elapsed_secs();

    for (mut aura_tf, mut aura_sprite, aura) in &mut auras {
        // Find owner entity
        let Ok((_, owner_tf, body, npc_c, psych)) = entities.get(aura.owner) else {
            // Owner despawned — hide
            aura_sprite.color = Color::srgba(0.0, 0.0, 0.0, 0.0);
            continue;
        };

        // Get consciousness metrics from physics engine or NPC consciousness
        let (energy_frac, _phi, prediction_error) = if let Some(body) = body {
            physics
                .consciousness
                .entities
                .get(&body.handle)
                .map(|e| {
                    (
                        e.energy.fraction_remaining() as f32,
                        e.phi() as f32,
                        e.prediction_error as f32,
                    )
                })
                .unwrap_or((1.0, 0.5, 0.0))
        } else {
            (1.0, player_consciousness.level as f32, 0.0)
        };

        // Use NPC consciousness level if available (more accurate than raw phi)
        let consciousness_level = npc_c
            .map(|c| c.level as f32)
            .unwrap_or(player_consciousness.level as f32);

        let allostatic_load = psych.map(|p| p.allostatic_load).unwrap_or(0.0);

        // === RADIUS: energy fraction maps to aura size ===
        let base_radius = MIN_AURA_RADIUS + energy_frac * (MAX_AURA_RADIUS - MIN_AURA_RADIUS);

        // === SHAPE: Phi determines boundary smoothness ===
        // High Phi: smooth sine wave (integrated, coherent)
        // Low Phi: jagged noise (fragmented, chaotic)
        let smooth_component = (aura.angle * 3.0 + t * 2.0).sin() * 0.15;

        // Noise component: pseudo-random based on segment + time
        // More noise = lower Phi = more chaos in the boundary
        let noise_freq = 7.0 + prediction_error * 5.0; // higher PE = higher frequency noise
        let noise_phase = aura.segment as f32 * 1.618 + t * (3.0 + allostatic_load * 4.0);
        let noise_component =
            (noise_phase * noise_freq).sin() * (noise_phase * noise_freq * 1.3).cos() * 0.4; // raw noise amplitude

        // Blend: high consciousness = smooth, low consciousness = noisy
        let boundary_modulation =
            smooth_component * consciousness_level + noise_component * (1.0 - consciousness_level);

        let radius = base_radius * (1.0 + boundary_modulation);

        // === POSITION: place aura sprite on ring ===
        let owner_pos = owner_tf.translation;
        aura_tf.translation.x = owner_pos.x + aura.angle.cos() * radius;
        aura_tf.translation.y = owner_pos.y + aura.angle.sin() * radius;
        aura_tf.translation.z = 2.5;

        // === COLOR: dominant harmony + energy state ===
        let harmony_color = harmony
            .dominant
            .map(|h| h.color())
            .unwrap_or(Color::srgb(0.5, 0.8, 0.6));
        let Srgba {
            red: hr,
            green: hg,
            blue: hb,
            ..
        } = harmony_color.to_srgba();

        // Energy depletion: desaturate toward gray
        let energy_blend = energy_frac.sqrt(); // sqrt for perceptual linearity
        let r = hr * energy_blend + 0.3 * (1.0 - energy_blend);
        let g = hg * energy_blend + 0.3 * (1.0 - energy_blend);
        let b = hb * energy_blend + 0.3 * (1.0 - energy_blend);

        // Burnout: shift toward red
        let burnout_blend = (allostatic_load - 0.6).max(0.0) * 2.5; // 0 below 0.6, 1.0 at 1.0
        let r = r * (1.0 - burnout_blend) + 0.9 * burnout_blend;
        let g = g * (1.0 - burnout_blend) + 0.2 * burnout_blend;
        let b = b * (1.0 - burnout_blend) + 0.1 * burnout_blend;

        // === OPACITY: prediction error causes flickering ===
        let base_alpha = energy_frac * 0.5; // max 50% opacity
        let flicker = if prediction_error > 0.3 {
            // High PE: rapid flickering (the math is uncertain)
            let flicker_speed = 8.0 + prediction_error * 12.0;
            ((t * flicker_speed + aura.segment as f32).sin() * 0.5 + 0.5)
                * (1.0 - prediction_error.min(1.0) * 0.5)
        } else {
            1.0
        };

        aura_sprite.color = Color::srgba(
            r.clamp(0.0, 1.0),
            g.clamp(0.0, 1.0),
            b.clamp(0.0, 1.0),
            (base_alpha * flicker).clamp(0.0, 0.6),
        );
    }
}

/// Emit harmony resonance waves when agents achieve high resonance nearby.
pub fn resonance_wave_system(
    mut commands: Commands,
    entities: Query<(&Transform, &PhysicsBody), With<CrewNpc>>,
    physics: Res<PhysicsWorldRes>,
    harmony: Res<crate::systems::harmonies::LocalHarmonyState>,
    mut timer: ResMut<ResonanceWaveTimer>,
    time: Res<Time>,
) {
    timer.0 += time.delta_secs();
    if timer.0 < 2.0 {
        return;
    } // Check every 2 seconds
    timer.0 = 0.0;

    // Find agents with high harmony resonance
    if harmony.total_energy < 2.0 {
        return;
    }

    let Some(dominant) = harmony.dominant else {
        return;
    };
    let wave_color = dominant.color();

    for (tf, body) in &entities {
        let phi = physics
            .consciousness
            .entities
            .get(&body.handle)
            .map(|e| e.phi() as f32)
            .unwrap_or(0.0);

        // Only emit waves from high-consciousness agents
        if phi > 0.4 && harmony.activations[dominant.index()] > 0.5 {
            let pos = tf.translation.truncate();
            commands.spawn((
                Sprite::from_color(
                    Color::srgba(0.0, 0.0, 0.0, 0.0), // starts invisible, updated by wave system
                    Vec2::splat(6.0),
                ),
                Transform::from_xyz(pos.x, pos.y, 1.5),
                ResonanceWave {
                    origin: pos,
                    radius: 0.0,
                    speed: 80.0, // px/sec expansion
                    life: 1.0,
                    color: wave_color,
                },
            ));
            break; // One wave per cycle
        }
    }
}

/// Animate resonance waves: expand outward, fade, tint tiles.
pub fn resonance_wave_animate_system(
    mut commands: Commands,
    mut waves: Query<
        (Entity, &mut ResonanceWave, &mut Transform, &mut Sprite),
        Without<crate::components::Tile>,
    >,
    mut tiles: Query<(&Transform, &mut Sprite, &crate::components::Tile), Without<ResonanceWave>>,
    time: Res<Time>,
) {
    let dt = time.delta_secs();

    for (entity, mut wave, mut tf, mut sprite) in &mut waves {
        wave.radius += wave.speed * dt;
        wave.life -= dt * 0.4; // ~2.5 second lifetime

        if wave.life <= 0.0 {
            commands.entity(entity).despawn();
            continue;
        }

        // Move wave sprite to ring edge (visual anchor point)
        tf.translation.x = wave.origin.x + wave.radius;
        tf.translation.y = wave.origin.y;

        // Wave ring visibility
        let Srgba {
            red: wr,
            green: wg,
            blue: wb,
            ..
        } = wave.color.to_srgba();
        sprite.color = Color::srgba(wr, wg, wb, wave.life * 0.3);

        // Tint tiles the wave passes through
        let wave_width = 20.0; // ring thickness
        for (tile_tf, mut tile_sprite, tile) in &mut tiles {
            if !tile.walkable {
                continue;
            }

            let tile_pos = tile_tf.translation.truncate();
            let dist = tile_pos.distance(wave.origin);

            // Is this tile in the wave ring?
            if (dist - wave.radius).abs() < wave_width {
                let ring_falloff = 1.0 - (dist - wave.radius).abs() / wave_width;
                let blend = ring_falloff * wave.life * 0.2;

                let Srgba {
                    red: tr,
                    green: tg,
                    blue: tb,
                    ..
                } = tile_sprite.color.to_srgba();
                tile_sprite.color = Color::srgb(
                    (tr + wr * blend).clamp(0.0, 1.0),
                    (tg + wg * blend).clamp(0.0, 1.0),
                    (tb + wb * blend).clamp(0.0, 1.0),
                );
            }
        }
    }
}

/// 4D Perception Gate: dimensional leakage points become visible
/// only when player Phi exceeds threshold.
///
/// This ties consciousness to perception — literally seeing things
/// that unconscious players cannot. High-Phi players detect "shimmer"
/// at dimensional leakage points, revealing the hidden 4D economy.
pub fn consciousness_perception_gate_system(
    mut four_d_secrets: Query<
        (&mut Sprite, &crate::systems::four_d_rendering::FourDBody),
        (
            With<crate::systems::four_d_rendering::FourDOnly>,
            Without<AuraSprite>,
        ),
    >,
    player_consciousness: Res<crate::systems::consciousness::PlayerConsciousness>,
    dim: Res<crate::systems::dimension_transition::DimensionTransition>,
    time: Res<Time>,
) {
    let phi = player_consciousness.level as f32;
    let t = time.elapsed_secs();

    // Phi threshold for 4D perception
    // Below 0.5: completely blind to 4D
    // 0.5-0.7: faint shimmer (almost subliminal)
    // 0.7+: clear visibility
    let perception = ((phi - 0.5) / 0.2).clamp(0.0, 1.0);

    if perception <= 0.0 {
        return;
    }

    // Only active in 4D mode (or when consciousness is high enough to "pierce the veil")
    let in_4d = dim.target == crate::systems::dimension_transition::DimensionMode::D4;
    let phi_piercing = phi > 0.7; // high consciousness sees through dimensions regardless

    if !in_4d && !phi_piercing {
        return;
    }

    for (mut sprite, body) in &mut four_d_secrets {
        // Base visibility from 4D projection system
        let w_dist = (body.w as f32 - dim.w_position).abs();

        // Consciousness-gated shimmer
        let shimmer = (t * 3.0 + body.w as f32 * 0.5).sin() * 0.3 + 0.7;
        let consciousness_alpha = perception * shimmer * 0.6;

        // If in 4D mode: normal W-distance visibility + consciousness bonus
        // If not in 4D mode but high Phi: faint shimmer only
        let alpha = if in_4d {
            let w_alpha = (1.0 - w_dist / 20.0).clamp(0.0, 1.0);
            (w_alpha + consciousness_alpha * 0.3).min(1.0)
        } else {
            // Piercing the veil: see 4D secrets from 2D/3D
            consciousness_alpha * 0.4 // very faint — a reward for high consciousness
        };

        let Srgba {
            red, green, blue, ..
        } = sprite.color.to_srgba();
        sprite.color = Color::srgba(red, green, blue, alpha);
    }
}
