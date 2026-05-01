// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! 4D cross-section rendering: projects entities with a W coordinate through
//! a 3D hyperplane slice, modulating visibility and alpha.
//!
//! When `DimensionMode::D4` is active, entities with a `FourDBody` component
//! are rendered based on their distance from the current W slice position.
//! Entities near the slice are opaque; those far from it fade to invisible.
//!
//! This uses `symtropy_render_bridge::Projector4D` for the math — this module
//! just wires it into Bevy's ECS.

use bevy::prelude::*;
use symtropy_render_bridge::Projector4D;

use crate::systems::dimension_transition::{DimensionMode, DimensionTransition};

/// Component that gives an entity a position in the 4th dimension (W axis).
///
/// Entities without this component are always visible regardless of dimension mode.
/// Entities with this component fade in/out based on the W slice position in 4D mode.
#[derive(Component, Debug)]
pub struct FourDBody {
    /// Position along the W axis.
    pub w: f64,
    /// Base color to restore when leaving 4D mode (so alpha modulation is reversible).
    pub base_color: Color,
}

impl FourDBody {
    pub fn new(w: f64, base_color: Color) -> Self {
        Self { w, base_color }
    }
}

/// Marker for entities that only exist in 4D — completely hidden in 2D/2.5D/3D.
/// These are "4D secrets" that reward dimension exploration.
#[derive(Component)]
pub struct FourDOnly;

/// Resource holding the active 4D projector (created lazily when entering 4D mode).
#[derive(Resource)]
pub struct FourDProjector {
    pub projector: Projector4D,
}

impl Default for FourDProjector {
    fn default() -> Self {
        Self {
            projector: Projector4D::new(
                0.0,  // w_slice: start at origin
                20.0, // slice_thickness: entities within 20 units of the slice are visible
                1.0,  // scale_factor
            ),
        }
    }
}

/// Sync the `Projector4D` slice position from the `DimensionTransition` W slider.
///
/// Runs every frame when the game is in 4D mode (or transitioning to it).
pub fn four_d_projector_sync_system(
    dim: Res<DimensionTransition>,
    mut projector: ResMut<FourDProjector>,
) {
    // Update the projector's slice position from the dimension transition W slider
    projector.projector.w_slice = dim.w_position as f64;
}

/// Modulate sprite alpha for entities with `FourDBody` based on the W slice.
///
/// In 4D mode: alpha is set by distance from the slice hyperplane.
/// In other modes: `FourDOnly` entities are hidden; regular `FourDBody` entities
/// have their base color restored.
pub fn four_d_visibility_system(
    dim: Res<DimensionTransition>,
    projector: Res<FourDProjector>,
    mut query: Query<(&FourDBody, &mut Sprite, Option<&FourDOnly>)>,
) {
    let in_4d =
        dim.target == DimensionMode::D4 && dim.progress >= 0.5 || dim.current == DimensionMode::D4;

    for (body_4d, mut sprite, is_4d_only) in &mut query {
        if in_4d {
            // Compute alpha from W distance to slice
            let point = symtropy_math::Point::new([0.0, 0.0, 0.0, body_4d.w]);
            let alpha = projector.projector.alpha(&point);

            // Apply alpha to sprite color, preserving base RGB
            let base = body_4d.base_color.to_srgba();
            sprite.color = Color::srgba(base.red, base.green, base.blue, alpha);
        } else {
            // Not in 4D mode
            if is_4d_only.is_some() {
                // 4D-only entities are completely hidden outside 4D mode
                let base = body_4d.base_color.to_srgba();
                sprite.color = Color::srgba(base.red, base.green, base.blue, 0.0);
            } else {
                // Regular 4D bodies restore their base color
                sprite.color = body_4d.base_color;
            }
        }
    }
}

/// Spawn 4D-hidden items in the dungeon during world setup.
///
/// Called from the Loading phase. Places secret items at various W positions
/// that players can only discover by entering 4D mode and sliding the W axis.
pub fn spawn_four_d_secrets(mut commands: Commands, seed: Res<crate::resources::DungeonSeed>) {
    // Deterministic jitter from seed — no rand needed for fixed placements
    let jitter = |seed_val: u64, i: usize| -> (f32, f32) {
        let h = seed_val
            .wrapping_mul(2654435761)
            .wrapping_add(i as u64 * 131);
        let jx = ((h % 3200) as f32 / 100.0) - 16.0;
        let jy = (((h >> 16) % 3200) as f32 / 100.0) - 16.0;
        (jx, jy)
    };

    // Place 4D secrets at room-scale positions with varying W values.
    // These are "dimensional rifts" — glowing objects only visible in 4D.
    let secrets: [(f32, f32, f64, Color); 5] = [
        // (x, y, w, color)
        (64.0, -32.0, 15.0, Color::srgba(0.8, 0.2, 1.0, 0.9)),
        (-96.0, 64.0, -10.0, Color::srgba(1.0, 0.5, 0.1, 0.9)),
        (128.0, 96.0, 25.0, Color::srgba(0.1, 1.0, 0.8, 0.9)),
        (-48.0, -128.0, -20.0, Color::srgba(1.0, 1.0, 0.2, 0.9)),
        (0.0, 0.0, 5.0, Color::srgba(0.6, 0.3, 1.0, 0.9)),
    ];

    for (i, (x, y, w, color)) in secrets.iter().enumerate() {
        let (jx, jy) = jitter(seed.0, i);

        commands.spawn((
            Sprite::from_color(Color::srgba(0.0, 0.0, 0.0, 0.0), Vec2::splat(14.0)),
            Transform::from_xyz(x + jx, y + jy, 2.5), // z=2.5: above tiles, below player
            FourDBody::new(*w, *color),
            FourDOnly,
        ));
    }

    eprintln!("[4D] Spawned {} dimensional secrets", secrets.len());
}

/// Give the player a 4D body component so they are visible from all W positions.
/// This runs once during setup — the player is always at W=0 (the "home" slice).
pub fn assign_player_four_d(
    mut commands: Commands,
    players: Query<(Entity, &Sprite), (With<crate::components::Player>, Without<FourDBody>)>,
) {
    for (entity, sprite) in &players {
        commands
            .entity(entity)
            .insert(FourDBody::new(0.0, sprite.color));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_d_body_creation() {
        let body = FourDBody::new(5.0, Color::srgb(1.0, 0.0, 0.0));
        assert!((body.w - 5.0).abs() < 1e-10);
    }

    #[test]
    fn projector_default_centered_at_origin() {
        let proj = FourDProjector::default();
        assert!((proj.projector.w_slice - 0.0).abs() < 1e-10);
        assert!((proj.projector.slice_thickness - 20.0).abs() < 1e-10);
    }

    #[test]
    fn secret_at_w0_visible_at_default_slice() {
        let proj = FourDProjector::default();
        let point = symtropy_math::Point::new([0.0, 0.0, 0.0, 0.0]);
        let alpha = proj.projector.alpha(&point);
        assert!(
            (alpha - 1.0).abs() < 1e-5,
            "Entity at w=0 should be fully visible at default slice"
        );
    }

    #[test]
    fn secret_at_w30_invisible_at_default_slice() {
        let proj = FourDProjector::default();
        let point = symtropy_math::Point::new([0.0, 0.0, 0.0, 30.0]);
        let alpha = proj.projector.alpha(&point);
        assert!(
            (alpha - 0.0).abs() < 1e-5,
            "Entity at w=30 should be invisible (thickness=20)"
        );
    }

    #[test]
    fn secret_at_w10_half_visible() {
        let proj = FourDProjector::default();
        let point = symtropy_math::Point::new([0.0, 0.0, 0.0, 10.0]);
        let alpha = proj.projector.alpha(&point);
        assert!(
            (alpha - 0.5).abs() < 1e-5,
            "Entity at w=10 should be half-visible (thickness=20)"
        );
    }
}
