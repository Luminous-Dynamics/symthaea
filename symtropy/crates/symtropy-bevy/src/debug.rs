// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Debug visualization: draw colliders, contacts, safety tiers, and energy as Bevy gizmos.
//!
//! Enabled by default via the `debug-gizmos` feature flag.
//! Disable with `default-features = false` in your Cargo.toml.

use crate::plugin::SymtropyPhysics;
use bevy::prelude::*;
use symtropy_bevy_core::PhysicsBody;
use symtropy_consciousness_physics::safety::SafetyTier;

/// Safety tier colors (NRC-inspired).
fn tier_color(tier: SafetyTier) -> Color {
    match tier {
        SafetyTier::Green => Color::srgba(0.2, 1.0, 0.3, 0.6),
        SafetyTier::Yellow => Color::srgba(1.0, 0.9, 0.2, 0.6),
        SafetyTier::Orange => Color::srgba(1.0, 0.5, 0.1, 0.6),
        SafetyTier::Red => Color::srgba(1.0, 0.1, 0.1, 0.6),
    }
}

/// Energy fraction to color (green = full, red = empty).
fn energy_color(fraction: f64) -> Color {
    let f = fraction.clamp(0.0, 1.0) as f32;
    Color::srgba(1.0 - f, f, 0.1, 0.4)
}

/// Draw debug gizmos for all physics bodies.
pub fn draw_debug_gizmos<const D: usize>(
    physics: Res<SymtropyPhysics<D>>,
    bodies: Query<(&PhysicsBody, &Transform)>,
    mut gizmos: Gizmos,
) {
    // Draw collider outlines colored by safety tier
    for (body_comp, transform) in &bodies {
        let handle = body_comp.handle;
        let tier = physics.field.safety_tier(handle);
        let color = tier_color(tier);
        let radius = body_comp.visual_radius;
        let pos = transform.translation;

        // Check if sensor
        let is_sensor = physics
            .world
            .body(handle)
            .map(|b| b.is_sensor)
            .unwrap_or(false);

        if is_sensor {
            gizmos
                .circle(
                    Isometry3d::from_translation(pos),
                    radius,
                    Color::srgba(0.5, 0.5, 1.0, 0.3),
                )
                .resolution(16);
        } else {
            if D >= 3 {
                gizmos
                    .sphere(Isometry3d::from_translation(pos), radius, color)
                    .resolution(16);
            } else {
                gizmos
                    .circle(Isometry3d::from_translation(pos), radius, color)
                    .resolution(32);
            }
        }

        // Energy bar
        if let Some(entity) = physics.field.entities.get(&handle) {
            let fraction = entity.energy.fraction_remaining();
            let bar_width = radius * 2.0;
            let bar_y = pos.y + radius + 0.1;
            let filled = bar_width * fraction as f32;

            gizmos.line(
                Vec3::new(pos.x - bar_width * 0.5, bar_y, pos.z),
                Vec3::new(pos.x + bar_width * 0.5, bar_y, pos.z),
                Color::srgba(0.2, 0.2, 0.2, 0.3),
            );
            if filled > 0.001 {
                gizmos.line(
                    Vec3::new(pos.x - bar_width * 0.5, bar_y, pos.z),
                    Vec3::new(pos.x - bar_width * 0.5 + filled, bar_y, pos.z),
                    energy_color(fraction),
                );
            }
        }
    }

    // Contacts
    for contact in &physics.world.contacts {
        let p = contact.point();
        let normal = contact.normal;
        let pos = Vec3::new(
            p[0] as f32,
            if D >= 2 { p[1] as f32 } else { 0.0 },
            if D >= 3 { p[2] as f32 } else { 0.0 },
        );
        let normal_v3 = Vec3::new(
            normal[0] as f32,
            if D >= 2 { normal[1] as f32 } else { 0.0 },
            if D >= 3 { normal[2] as f32 } else { 0.0 },
        );

        let cross_size = 0.05;
        gizmos.line(
            pos - Vec3::X * cross_size,
            pos + Vec3::X * cross_size,
            Color::srgba(1.0, 0.3, 0.3, 0.8),
        );
        gizmos.line(
            pos - Vec3::Y * cross_size,
            pos + Vec3::Y * cross_size,
            Color::srgba(1.0, 0.3, 0.3, 0.8),
        );
        if D >= 3 {
            gizmos.line(
                pos - Vec3::Z * cross_size,
                pos + Vec3::Z * cross_size,
                Color::srgba(1.0, 0.3, 0.3, 0.8),
            );
        }
        gizmos.line(pos, pos + normal_v3 * 0.2, Color::srgba(1.0, 1.0, 0.0, 0.5));
    }

    // Joints
    for constraint in &physics.world.constraints {
        let (ha, hb) = constraint.bodies();
        let body_a = physics.world.body(ha);
        let body_b = physics.world.body(hb);
        if let (Some(ba), Some(bb)) = (body_a, body_b) {
            let pa = ba.position();
            let pb = bb.position();
            let va = Vec3::new(
                pa[0] as f32,
                if D >= 2 { pa[1] as f32 } else { 0.0 },
                if D >= 3 { pa[2] as f32 } else { 0.0 },
            );
            let vb = Vec3::new(
                pb[0] as f32,
                if D >= 2 { pb[1] as f32 } else { 0.0 },
                if D >= 3 { pb[2] as f32 } else { 0.0 },
            );
            gizmos.line(va, vb, Color::srgba(0.7, 0.7, 0.8, 0.4));
        }
    }
}

fn harmony_color(idx: usize) -> Color {
    match idx {
        0 => Color::srgb(0.0, 1.0, 0.0), // Unity
        1 => Color::srgb(0.0, 1.0, 1.0), // Truth
        2 => Color::srgb(1.0, 0.0, 0.0), // Sacrifice
        3 => Color::srgb(1.0, 1.0, 0.0), // Purpose
        4 => Color::srgb(0.0, 0.0, 1.0), // Wisdom
        5 => Color::srgb(1.0, 0.0, 1.0), // Harmony
        6 => Color::srgb(1.0, 0.5, 0.0), // Courage
        7 => Color::srgb(1.0, 1.0, 1.0), // Stillness
        8 => Color::srgb(1.0, 0.2, 0.5), // Contagion
        _ => Color::BLACK,
    }
}

pub fn draw_harmony_fields<const D: usize>(physics: Res<SymtropyPhysics<D>>, mut gizmos: Gizmos) {
    use symtropy_consciousness_physics::harmony_field::NUM_HARMONIES;
    for source in &physics.field.harmony_field.sources {
        let p = source.position;
        let pos = Vec3::new(
            p[0] as f32,
            if D >= 2 { p[1] as f32 } else { 0.0 },
            if D >= 3 { p[2] as f32 } else { 0.0 },
        );
        for i in 0..NUM_HARMONIES {
            let act = source.activations[i];
            if act < 0.05 {
                continue;
            }
            let color = harmony_color(i).with_alpha(0.05);
            let base_radius = (source.radius as f32) * 0.5;
            for step in 1..=3 {
                let r = base_radius * (step as f32 / 3.0) * act as f32;
                if r < 0.01 {
                    continue;
                }
                if D >= 3 {
                    gizmos
                        .sphere(Isometry3d::from_translation(pos), r, color)
                        .resolution(8);
                } else {
                    gizmos
                        .circle(Isometry3d::from_translation(pos), r, color)
                        .resolution(16);
                }
            }
        }
    }
}

pub fn draw_phi_heatmap<const D: usize>(physics: Res<SymtropyPhysics<D>>, mut gizmos: Gizmos) {
    for (handle, entity) in &physics.field.entities {
        let phi = entity.phi();
        if phi < 0.01 {
            continue;
        }
        let Some(body) = physics.world.body(*handle) else {
            continue;
        };
        let p = body.position();
        let pos = Vec3::new(
            p[0] as f32,
            if D >= 2 { p[1] as f32 } else { 0.0 },
            if D >= 3 { p[2] as f32 } else { 0.0 },
        );
        let phi_norm = (phi / 0.3).clamp(0.0, 1.0) as f32;
        let color = Color::srgba(0.2 + phi_norm * 0.8, 0.5 + phi_norm * 0.5, 1.0, 0.15);
        let aura_radius = (0.2 + phi_norm * 1.5) as f32;
        if D >= 3 {
            gizmos
                .sphere(Isometry3d::from_translation(pos), aura_radius, color)
                .resolution(8);
        } else {
            gizmos
                .circle(Isometry3d::from_translation(pos), aura_radius, color)
                .resolution(16);
        }
    }
}
