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
///
/// Renders:
/// - Collider outlines (circles in 2D/3D) colored by safety tier
/// - Energy bar as a small line above each body
/// - Contact points as small crosses
/// - Sensor bodies with dashed outlines
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
            // Dashed circle for sensors (approximate with fewer segments)
            let segments = 8;
            for i in 0..segments {
                let a = (i as f32 / segments as f32) * std::f32::consts::TAU;
                let b_angle = ((i as f32 + 0.5) / segments as f32) * std::f32::consts::TAU;
                let start = Vec3::new(pos.x + a.cos() * radius, pos.y + a.sin() * radius, pos.z);
                let end = Vec3::new(
                    pos.x + b_angle.cos() * radius,
                    pos.y + b_angle.sin() * radius,
                    pos.z,
                );
                gizmos.line(start, end, Color::srgba(0.5, 0.5, 1.0, 0.3));
            }
        } else {
            // Solid circle for regular bodies
            gizmos.circle(Isometry3d::from_translation(pos), radius, color);
        }

        // Energy bar above the body
        if let Some(entity) = physics.field.entities.get(&handle) {
            let fraction = entity.energy.fraction_remaining();
            let bar_width = radius * 2.0;
            let bar_y = pos.y + radius + 2.0;
            let filled = bar_width * fraction as f32;

            // Background (dark)
            gizmos.line(
                Vec3::new(pos.x - bar_width * 0.5, bar_y, pos.z),
                Vec3::new(pos.x + bar_width * 0.5, bar_y, pos.z),
                Color::srgba(0.2, 0.2, 0.2, 0.3),
            );
            // Filled portion
            if filled > 0.01 {
                gizmos.line(
                    Vec3::new(pos.x - bar_width * 0.5, bar_y, pos.z),
                    Vec3::new(pos.x - bar_width * 0.5 + filled, bar_y, pos.z),
                    energy_color(fraction),
                );
            }
        }
    }

    // Draw contact points
    for contact in &physics.world.contacts {
        let p = contact.point();
        let cross_size = 1.5;
        let pos = Vec3::new(p[0] as f32, if D >= 2 { p[1] as f32 } else { 0.0 }, 0.0);
        gizmos.line(
            pos + Vec3::new(-cross_size, 0.0, 0.0),
            pos + Vec3::new(cross_size, 0.0, 0.0),
            Color::srgba(1.0, 0.3, 0.3, 0.8),
        );
        gizmos.line(
            pos + Vec3::new(0.0, -cross_size, 0.0),
            pos + Vec3::new(0.0, cross_size, 0.0),
            Color::srgba(1.0, 0.3, 0.3, 0.8),
        );
    }
}
