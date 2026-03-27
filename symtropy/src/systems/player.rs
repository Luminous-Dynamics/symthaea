// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Player movement and interaction systems.

use bevy::prelude::*;

use crate::components::{Flashlight, FusionCore, NoiseEmitter, Player};
use crate::resources::BiometricsCtx;

/// Player movement speed in pixels/second.
const PLAYER_SPEED: f32 = 120.0;

/// Move player with WASD keys.
pub fn player_movement_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut query: Query<(&mut Transform, &mut NoiseEmitter), With<Player>>,
    time: Res<Time>,
) {
    let Ok((mut transform, mut noise)) = query.single_mut() else {
        return;
    };

    let mut direction = Vec2::ZERO;
    if keyboard.pressed(KeyCode::KeyW) || keyboard.pressed(KeyCode::ArrowUp) {
        direction.y += 1.0;
    }
    if keyboard.pressed(KeyCode::KeyS) || keyboard.pressed(KeyCode::ArrowDown) {
        direction.y -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyA) || keyboard.pressed(KeyCode::ArrowLeft) {
        direction.x -= 1.0;
    }
    if keyboard.pressed(KeyCode::KeyD) || keyboard.pressed(KeyCode::ArrowRight) {
        direction.x += 1.0;
    }

    if direction != Vec2::ZERO {
        direction = direction.normalize();
        transform.translation.x += direction.x * PLAYER_SPEED * time.delta_secs();
        transform.translation.y += direction.y * PLAYER_SPEED * time.delta_secs();
        // Movement generates noise proportional to speed
        noise.level = 0.3;
    } else {
        noise.level *= 0.9; // decay when still
    }
}

/// Update flashlight flicker based on player stress.
pub fn flashlight_system(
    biometrics: Res<BiometricsCtx>,
    mut query: Query<&mut Flashlight, With<Player>>,
) {
    let Ok(mut flashlight) = query.single_mut() else {
        return;
    };
    // Flicker intensity driven by velocity surprise
    flashlight.flicker = biometrics.encoder.velocity_surprise() * 0.4
        + biometrics.model.allostatic_load * 0.3;
}

/// Fusion core extraction: hold E near the core.
pub fn extraction_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    player_query: Query<&Transform, With<Player>>,
    mut core_query: Query<(&Transform, &mut FusionCore)>,
    biometrics: Res<BiometricsCtx>,
    mut noise_query: Query<&mut NoiseEmitter, With<Player>>,
    time: Res<Time>,
) {
    let Ok(player_tf) = player_query.single() else {
        return;
    };

    for (core_tf, mut core) in &mut core_query {
        let dist = player_tf
            .translation
            .truncate()
            .distance(core_tf.translation.truncate());

        if dist < 40.0 && keyboard.pressed(KeyCode::KeyE) {
            core.being_extracted = true;

            // Extraction speed reduced by stress (shaky hands)
            let stress_penalty = 1.0 - biometrics.model.allostatic_load * 0.6;
            let base_rate = 0.15; // ~7 seconds to extract at zero stress
            core.extraction_progress += base_rate * stress_penalty * time.delta_secs();

            // Extraction generates noise — more with higher stress
            if let Ok(mut noise) = noise_query.single_mut() {
                noise.level = 0.1 + biometrics.encoder.velocity_surprise() * 0.5;
            }
        } else {
            core.being_extracted = false;
        }
    }
}
