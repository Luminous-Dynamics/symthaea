// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CRT + Consciousness post-processing shader.
//!
//! Drives visual distortion from biometric stress state:
//! - Chromatic aberration scales with allostatic_load
//! - Scanlines intensify with arousal
//! - Vignette deepens with stress + Leviathan danger
//! - Color shifts from gold (calm) → teal (stressed) → red (danger)

use bevy::prelude::*;

use crate::resources::{BiometricsCtx, LeviathanState, SleepPhase};

/// Trauma-based camera shake. Trauma decays over time, produces
/// random offset proportional to trauma level.
#[derive(Resource)]
pub struct CameraTrauma {
    /// Current trauma [0, 1]. Decays automatically.
    pub trauma: f32,
}

impl Default for CameraTrauma {
    fn default() -> Self {
        Self { trauma: 0.0 }
    }
}

impl CameraTrauma {
    /// Add trauma (clamped to 1.0).
    pub fn add(&mut self, amount: f32) {
        self.trauma = (self.trauma + amount).min(1.0);
    }
}

/// Resource holding the consciousness-driven visual parameters.
/// Updated each frame from biometrics + Leviathan state.
#[derive(Resource, Default)]
pub struct ConsciousnessVisuals {
    pub allostatic_load: f32,
    pub arousal: f32,
    pub danger: f32,
}

/// Update consciousness visuals from game state (runs every frame).
pub fn update_consciousness_visuals(
    biometrics: Res<BiometricsCtx>,
    leviathan: Res<LeviathanState>,
    mut visuals: ResMut<ConsciousnessVisuals>,
) {
    let stress = biometrics.encoder.compute_stress_vector();
    visuals.allostatic_load = biometrics.model.allostatic_load;
    visuals.arousal = stress.arousal;
    visuals.danger = match leviathan.phase {
        SleepPhase::Dormant => 0.0,
        SleepPhase::Stirring => 0.3 + leviathan.stirring_duration * 0.05,
        SleepPhase::Awake => 0.7,
        SleepPhase::Hunting => 1.0,
    };
}

/// Feed trauma from Leviathan state transitions and stress events.
pub fn trauma_feed_system(
    leviathan: Res<LeviathanState>,
    biometrics: Res<BiometricsCtx>,
    mut trauma: ResMut<CameraTrauma>,
) {
    // Leviathan phase transitions cause trauma spikes
    match leviathan.phase {
        SleepPhase::Stirring => trauma.add(0.002), // gentle rumble
        SleepPhase::Awake => trauma.add(0.01),     // strong shake
        SleepPhase::Hunting => trauma.add(0.02),   // violent shake
        _ => {}
    }

    // High velocity surprise (panic) causes micro-trauma
    let surprise = biometrics.encoder.velocity_surprise();
    if surprise > 0.5 {
        trauma.add(surprise * 0.005);
    }
}

/// Apply camera shake based on trauma level.
pub fn camera_shake_system(
    mut trauma: ResMut<CameraTrauma>,
    mut camera: Query<&mut Transform, With<Camera2d>>,
    time: Res<Time>,
) {
    // Decay trauma
    trauma.trauma *= (1.0 - 2.0 * time.delta_secs()).max(0.0);

    let Ok(mut cam_tf) = camera.single_mut() else {
        return;
    };

    if trauma.trauma > 0.001 {
        // Shake intensity = trauma^2 (feels more natural)
        let intensity = trauma.trauma * trauma.trauma;
        let max_offset = 8.0 * intensity; // max 8 pixels at full trauma

        // Use time-based pseudo-random for smooth shake
        let t = time.elapsed_secs();
        let offset_x = (t * 37.0).sin() * (t * 53.0).cos() * max_offset;
        let offset_y = (t * 43.0).cos() * (t * 61.0).sin() * max_offset;

        // Apply as offset (camera_follow_system sets the base position)
        cam_tf.translation.x += offset_x;
        cam_tf.translation.y += offset_y;
    }
}
