// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Player movement, sprinting, flashlight, and extraction systems.

use bevy::prelude::*;

use crate::components::{Flashlight, FusionCore, NoiseEmitter, Player};
use crate::resources::BiometricsCtx;

/// Normal movement speed (pixels/second).
const WALK_SPEED: f32 = 100.0;
/// Sprint speed (pixels/second) — much noisier.
const SPRINT_SPEED: f32 = 200.0;
/// Walk noise level.
const WALK_NOISE: f32 = 0.15;
/// Sprint noise level.
const SPRINT_NOISE: f32 = 0.6;
/// Standing noise decay factor.
const NOISE_DECAY: f32 = 0.85;

/// Move player with WASD, sprint with Shift.
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
        let sprinting = keyboard.pressed(KeyCode::ShiftLeft) || keyboard.pressed(KeyCode::ShiftRight);
        let speed = if sprinting { SPRINT_SPEED } else { WALK_SPEED };

        transform.translation.x += direction.x * speed * time.delta_secs();
        transform.translation.y += direction.y * speed * time.delta_secs();

        noise.level = if sprinting { SPRINT_NOISE } else { WALK_NOISE };
    } else {
        noise.level *= NOISE_DECAY;
    }
}

/// Update flashlight flicker based on player stress.
pub fn flashlight_system(
    biometrics: Res<BiometricsCtx>,
    mut query: Query<(&mut Flashlight, &mut Sprite), With<Player>>,
) {
    let Ok((mut flashlight, mut sprite)) = query.single_mut() else {
        return;
    };
    flashlight.flicker = biometrics.encoder.velocity_surprise() * 0.4
        + biometrics.model.allostatic_load * 0.3;

    // Player sprite dims/brightens with stress (visual feedback)
    let stress_dim = 1.0 - biometrics.model.allostatic_load * 0.3;
    sprite.color = Color::srgba(
        0.2 * stress_dim,
        0.9 * stress_dim,
        1.0 * stress_dim,
        1.0,
    );
}

/// Fusion core extraction: hold E near the core. Mouse steadiness matters.
pub fn extraction_system(
    keyboard: Res<ButtonInput<KeyCode>>,
    player_query: Query<&Transform, With<Player>>,
    mut core_query: Query<(&Transform, &mut FusionCore, &mut Sprite), Without<Player>>,
    biometrics: Res<BiometricsCtx>,
    mut noise_query: Query<&mut NoiseEmitter, With<Player>>,
    time: Res<Time>,
) {
    let Ok(player_tf) = player_query.single() else {
        return;
    };

    for (core_tf, mut core, mut core_sprite) in &mut core_query {
        let dist = player_tf
            .translation
            .truncate()
            .distance(core_tf.translation.truncate());

        if dist < 50.0 && keyboard.pressed(KeyCode::KeyE) {
            core.being_extracted = true;

            // Extraction speed penalized by mouse jitter (the biometric mechanic!)
            // Calm hands = fast extraction. Shaky hands = slow and noisy.
            let surprise = biometrics.encoder.velocity_surprise();
            let steadiness = (1.0 - surprise).max(0.1);
            let stress_penalty = 1.0 - biometrics.model.allostatic_load * 0.5;
            let rate = 0.12 * steadiness * stress_penalty; // ~8s calm, ~25s panicked

            core.extraction_progress += rate * time.delta_secs();

            // Extraction generates noise — more with jittery hands
            if let Ok(mut noise) = noise_query.single_mut() {
                noise.level = 0.2 + surprise * 0.8;
            }

            // Core pulses faster as extraction progresses
            let pulse = (time.elapsed_secs() * (4.0 + core.extraction_progress * 12.0)).sin();
            let brightness = 0.7 + pulse * 0.3;
            core_sprite.color = Color::srgb(brightness, brightness * 0.9, 0.1);
        } else {
            core.being_extracted = false;
            // Core gentle pulse when not being extracted
            if dist < 120.0 {
                let pulse = (time.elapsed_secs() * 1.5).sin();
                core_sprite.color = Color::srgb(0.9 + pulse * 0.1, 0.8 + pulse * 0.1, 0.1);
            }
        }
    }
}
