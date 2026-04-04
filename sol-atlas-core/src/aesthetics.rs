// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Visual aesthetic presets for the Sol Atlas globe.
//!
//! Each aesthetic defines material parameters for the globe surface,
//! atmosphere, markers, arcs, and base. Both Leptos (WebGL) and
//! Bevy (wgpu) renderers can consume these parameters.

use serde::{Deserialize, Serialize};

/// Available visual aesthetics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Aesthetic {
    /// Dark holographic command center — ghost geography, sacred geometry grid,
    /// Fresnel glow, data orb celestials, obsidian base.
    Holographic,
    /// Photorealistic blue marble — satellite texture, PBR lighting, shadows.
    /// Best for grant presentations and serious data analysis.
    Satellite,
    /// Procedural Mycelix brand — Wisdom Green land, Sacred Stillness ocean,
    /// sacred geometry grid, Eight Harmonies palette. The Leptos default.
    Procedural,
    /// Minimal dark vector — coastline outlines only, maximum data readability.
    /// Optimized for mobile and accessibility.
    Minimal,
    /// NASA nightlights — city light emissions, shows population and energy use.
    Night,
}

impl Aesthetic {
    pub fn label(&self) -> &'static str {
        match self {
            Self::Holographic => "Holographic",
            Self::Satellite => "Satellite",
            Self::Procedural => "Procedural",
            Self::Minimal => "Minimal",
            Self::Night => "Night",
        }
    }

    pub fn all() -> &'static [Self] {
        &[Self::Holographic, Self::Satellite, Self::Procedural, Self::Minimal, Self::Night]
    }
}

/// Globe surface material parameters.
#[derive(Debug, Clone)]
pub struct GlobeMaterial {
    /// Base color RGBA (multiplied with texture).
    pub base_color: [f32; 4],
    /// Emissive RGB + intensity.
    pub emissive: [f32; 4],
    /// Whether the surface is unlit (no PBR shading).
    pub unlit: bool,
    /// PBR roughness (ignored if unlit).
    pub roughness: f32,
    /// PBR metalness (ignored if unlit).
    pub metalness: f32,
    /// Whether to use alpha blending.
    pub alpha_blend: bool,
    /// Which texture to use.
    pub texture: GlobeTexture,
}

/// Which texture the globe surface uses.
#[derive(Debug, Clone, Copy)]
pub enum GlobeTexture {
    /// Blue marble satellite photo.
    BlueMarble,
    /// NASA nightlights.
    Nightlights,
    /// No texture — pure procedural color.
    None,
}

/// Fresnel atmosphere shell parameters.
#[derive(Debug, Clone)]
pub struct FresnelParams {
    pub color: [f32; 4],
    pub emissive: [f32; 4],
    pub visible: bool,
}

/// Sacred geometry grid parameters.
#[derive(Debug, Clone)]
pub struct GridParams {
    pub color: [f32; 4],
    pub emissive: [f32; 4],
    pub visible: bool,
    pub segments: u32,
}

/// Complete aesthetic configuration.
#[derive(Debug, Clone)]
pub struct AestheticConfig {
    pub globe: GlobeMaterial,
    pub fresnel: FresnelParams,
    pub grid: GridParams,
    pub background: [f32; 3],
    pub ambient_brightness: f32,
    pub bloom_intensity: f32,
    pub chromatic_aberration: f32,
}

/// Get the full configuration for an aesthetic preset.
pub fn config_for(aesthetic: Aesthetic) -> AestheticConfig {
    match aesthetic {
        Aesthetic::Holographic => AestheticConfig {
            globe: GlobeMaterial {
                base_color: [0.08, 0.12, 0.14, 0.45],
                emissive: [0.01, 0.03, 0.04, 1.0],
                unlit: true,
                roughness: 0.3,
                metalness: 0.1,
                alpha_blend: true,
                texture: GlobeTexture::BlueMarble,
            },
            fresnel: FresnelParams {
                color: [0.0, 0.6, 0.8, 0.04],
                emissive: [0.0, 0.25, 0.35, 1.0],
                visible: true,
            },
            grid: GridParams {
                color: [0.0, 0.87, 1.0, 0.06],
                emissive: [0.0, 0.2, 0.25, 1.0],
                visible: true,
                segments: 24,
            },
            background: [0.0, 0.0, 0.0],
            ambient_brightness: 300.0,
            bloom_intensity: 0.12,
            chromatic_aberration: 0.008,
        },
        Aesthetic::Satellite => AestheticConfig {
            globe: GlobeMaterial {
                base_color: [1.0, 1.0, 1.0, 1.0],
                emissive: [0.0, 0.0, 0.0, 0.0],
                unlit: false,
                roughness: 0.85,
                metalness: 0.0,
                alpha_blend: false,
                texture: GlobeTexture::BlueMarble,
            },
            fresnel: FresnelParams {
                color: [0.4, 0.6, 0.9, 0.08],
                emissive: [0.1, 0.15, 0.25, 1.0],
                visible: true,
            },
            grid: GridParams {
                color: [0.0; 4],
                emissive: [0.0; 4],
                visible: false,
                segments: 0,
            },
            background: [0.0, 0.0, 0.0],
            ambient_brightness: 50.0,
            bloom_intensity: 0.05,
            chromatic_aberration: 0.0,
        },
        Aesthetic::Procedural => AestheticConfig {
            globe: GlobeMaterial {
                base_color: [0.133, 0.286, 0.133, 0.9], // Wisdom Green
                emissive: [0.02, 0.04, 0.02, 1.0],
                unlit: true,
                roughness: 0.5,
                metalness: 0.0,
                alpha_blend: true,
                texture: GlobeTexture::None,
            },
            fresnel: FresnelParams {
                color: [0.0, 0.87, 1.0, 0.08],
                emissive: [0.0, 0.4, 0.5, 1.0],
                visible: true,
            },
            grid: GridParams {
                color: [1.0, 0.84, 0.0, 0.1], // Sacred Gold
                emissive: [0.3, 0.25, 0.0, 1.0],
                visible: true,
                segments: 16, // 8-fold sacred geometry
            },
            background: [0.06, 0.09, 0.16], // Sacred Stillness
            ambient_brightness: 200.0,
            bloom_intensity: 0.15,
            chromatic_aberration: 0.0,
        },
        Aesthetic::Minimal => AestheticConfig {
            globe: GlobeMaterial {
                base_color: [0.05, 0.08, 0.1, 0.3],
                emissive: [0.01, 0.02, 0.03, 1.0],
                unlit: true,
                roughness: 1.0,
                metalness: 0.0,
                alpha_blend: true,
                texture: GlobeTexture::BlueMarble,
            },
            fresnel: FresnelParams {
                color: [0.3, 0.4, 0.5, 0.03],
                emissive: [0.1, 0.15, 0.2, 1.0],
                visible: true,
            },
            grid: GridParams {
                color: [0.0; 4],
                emissive: [0.0; 4],
                visible: false,
                segments: 0,
            },
            background: [0.0, 0.0, 0.0],
            ambient_brightness: 100.0,
            bloom_intensity: 0.08,
            chromatic_aberration: 0.0,
        },
        Aesthetic::Night => AestheticConfig {
            globe: GlobeMaterial {
                base_color: [0.03, 0.03, 0.05, 0.8],
                emissive: [0.0, 0.0, 0.0, 0.0],
                unlit: true,
                roughness: 1.0,
                metalness: 0.0,
                alpha_blend: true,
                texture: GlobeTexture::Nightlights,
            },
            fresnel: FresnelParams {
                color: [0.1, 0.1, 0.2, 0.05],
                emissive: [0.05, 0.05, 0.1, 1.0],
                visible: true,
            },
            grid: GridParams {
                color: [0.0; 4],
                emissive: [0.0; 4],
                visible: false,
                segments: 0,
            },
            background: [0.0, 0.0, 0.0],
            ambient_brightness: 20.0,
            bloom_intensity: 0.20,
            chromatic_aberration: 0.005,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_aesthetics_have_configs() {
        for aesthetic in Aesthetic::all() {
            let config = config_for(*aesthetic);
            assert!(config.globe.base_color[3] > 0.0, "{:?} has zero alpha", aesthetic);
        }
    }

    #[test]
    fn test_holographic_is_unlit() {
        let config = config_for(Aesthetic::Holographic);
        assert!(config.globe.unlit);
        assert!(config.grid.visible);
    }

    #[test]
    fn test_satellite_is_lit() {
        let config = config_for(Aesthetic::Satellite);
        assert!(!config.globe.unlit);
        assert!(!config.grid.visible);
    }

    #[test]
    fn test_minimal_no_grid() {
        let config = config_for(Aesthetic::Minimal);
        assert!(!config.grid.visible);
        assert!(config.bloom_intensity < 0.10);
    }
}
