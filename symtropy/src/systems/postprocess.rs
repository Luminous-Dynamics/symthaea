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

// NOTE: The full custom render node integration requires Bevy's render app
// architecture (RenderApp, ViewNode, etc.) which is complex to set up.
// For now, we use the built-in ChromaticAberration + Bloom components
// driven by our consciousness data. The custom WGSL shader is ready
// for integration when we add bevy_post_process to the render graph.

// TODO: apply_builtin_postprocess — requires bevy_post_process::bloom::Bloom
// and ChromaticAberration which aren't available with current Bevy 0.18 feature set.
// Re-enable when post-process features are added to Cargo.toml.
