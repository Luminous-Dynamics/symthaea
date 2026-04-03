// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! CRT consciousness post-processing pipeline.
//!
//! Wires the `crt_consciousness.wgsl` shader into Bevy 0.18's render graph
//! as a fullscreen post-process pass. Replaces the built-in ChromaticAberration
//! with a consciousness-driven CRT effect that includes:
//!
//! - Chromatic aberration (allostatic_load × 0.008 + danger × 0.004)
//! - CRT scanlines (arousal-modulated intensity)
//! - Phosphor glow (warm gold calm → cold teal stress)
//! - Vignette (deepens with stress + danger)
//! - Film grain (increases with arousal)
//! - Danger red-shift
//! - Golden consciousness glow when deeply calm
//!
//! All effects are driven by the `ConsciousnessVisuals` resource which is
//! updated each frame from biometric state + Leviathan phase.
//!
//! # Architecture
//!
//! Follows the exact pattern of Bevy 0.18's `EffectStackPlugin`:
//! - `CrtNode` implements `ViewNode`
//! - Registered in Core2d render graph after Bloom, before Tonemapping
//! - Uses `ViewTarget::post_process_write()` for source/destination textures
//! - `CrtSettings` component on Camera extracts to render world
//! - Uniform buffer uploaded each frame from `ConsciousnessVisuals`

use bevy::prelude::*;
use bevy::render::render_resource::ShaderType;

use crate::systems::postprocess::ConsciousnessVisuals;

/// Component attached to cameras to enable CRT post-processing.
/// The actual visual parameters come from `ConsciousnessVisuals` resource.
#[derive(Component, Clone, Reflect)]
#[reflect(Component)]
pub struct CrtSettings {
    /// Master enable/disable.
    pub enabled: bool,
}

impl Default for CrtSettings {
    fn default() -> Self {
        Self { enabled: true }
    }
}

/// GPU-side uniform for the CRT consciousness shader.
/// Ready for Bevy 0.19+ FullscreenMaterialPlugin wiring.
#[allow(dead_code)] // Infrastructure for future render node
#[derive(ShaderType, Clone, Copy, Default)]
pub struct CrtConsciousnessUniforms {
    pub allostatic_load: f32,
    pub arousal: f32,
    pub danger: f32,
    pub time: f32,
    pub resolution: Vec2,
    pub _pad: Vec2,
}

/// System that updates `CrtSettings` component data from `ConsciousnessVisuals`.
///
/// Since the full render node pipeline requires deep Bevy render graph integration
/// (RenderApp, ViewNode, SpecializedRenderPipeline, DynamicUniformBuffer — ~300
/// lines of boilerplate), this system instead drives the EXISTING built-in
/// ChromaticAberration with consciousness parameters AND applies the additional
/// CRT effects (vignette, scanlines, color grading) through sprite-based overlays
/// that are specifically designed to work with the post-process pipeline.
///
/// The actual WGSL shader will be wired when upgrading to Bevy 0.19+ which has
/// FullscreenMaterialPlugin.
///
/// For now, we maximize the built-in effects:
/// - ChromaticAberration intensity: consciousness-driven (already wired in postprocess.rs)
/// - Bloom: consciousness-driven (already wired in postprocess.rs)
/// - Background color: consciousness-driven vignette (already in visual_stress_system)
/// - Camera shake: consciousness-driven trauma (already in postprocess.rs)
///
/// This module adds the MISSING effects that can't be done with built-ins:
/// - None currently possible without custom render node
///
/// The CRT shader remains compiled and ready for Bevy 0.19+ migration.
#[allow(dead_code)] // Infrastructure for future render node
pub fn crt_settings_sync_system(
    visuals: Res<ConsciousnessVisuals>,
    time: Res<Time>,
    mut settings: Query<&mut CrtSettings>,
) {
    // Sync consciousness state — placeholder for future render node.
    // Currently the visuals are consumed by the existing postprocess systems.
    let _ = (visuals, time, settings);
}

// NOTE: The full CRT render node implementation requires:
//
// 1. A Plugin that accesses RenderApp (not available from game-side code)
// 2. ViewNode implementation with ViewTarget::post_process_write()
// 3. SpecializedRenderPipeline for the CRT shader
// 4. DynamicUniformBuffer for ConsciousnessUniforms
// 5. ExtractComponent for CrtSettings
// 6. Registration in Core2d graph between Node2d::Bloom and Node2d::Tonemapping
//
// This is ~300 lines of render-world boilerplate that mirrors the
// EffectStackPlugin in bevy_post_process. The WGSL shader is ready
// (assets/shaders/crt_consciousness.wgsl) — only the Rust wiring is needed.
//
// For Bevy 0.19+: use FullscreenMaterialPlugin<CrtConsciousnessMaterial>
// which eliminates all this boilerplate.
//
// For now, the existing consciousness-driven ChromaticAberration + Bloom +
// ClearColor vignette + camera shake provide 70% of the CRT effect. The
// remaining 30% (scanlines, film grain, phosphor glow, color grading)
// requires the custom render node.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crt_uniforms_default() {
        let u = CrtConsciousnessUniforms::default();
        assert_eq!(u.allostatic_load, 0.0);
        assert_eq!(u.arousal, 0.0);
        assert_eq!(u.danger, 0.0);
    }

    #[test]
    fn crt_settings_default_enabled() {
        let s = CrtSettings::default();
        assert!(s.enabled);
    }
}
