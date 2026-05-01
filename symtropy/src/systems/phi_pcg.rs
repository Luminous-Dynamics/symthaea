// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Consciousness-driven procedural generation (Phi-PCG).
//!
//! Modulates dungeon generation based on consciousness state. This is a
//! **completely novel** contribution — no existing work uses IIT integration
//! measures to drive procedural content generation.
//!
//! # IIT-Grounded Design Principles
//!
//! **High Φ** = high integration → complex, interconnected spaces:
//! - Deeper BSP recursion (more rooms)
//! - More corridor connections (higher connectivity)
//! - Heterogeneous room sizes (high differentiation)
//! - Feedback loops in layout (circular paths)
//!
//! **Low Φ** = low integration → modular, repetitive spaces:
//! - Shallow BSP (fewer, larger rooms)
//! - Minimal corridors (linear paths)
//! - Uniform room sizes (low differentiation)
//! - Tree-structured layout (no loops)
//!
//! **Bottleneck-responsive**:
//! - "attention" bottleneck → fewer visual distractors (simpler rooms)
//! - "embodiment" bottleneck → more physical puzzles (narrower corridors)
//! - "knowledge" bottleneck → more scavenge items (learning opportunities)
//!
//! **Harmony-resonant**:
//! - Sacred Stillness → symmetrical, calm spaces
//! - Evolutionary Progression → branching corridors (exploration)

/// Parameters for consciousness-driven generation.
#[derive(Debug, Clone)]
pub struct PhiPcgParams {
    /// Current Φ level [0, 1]. Controls overall complexity.
    pub phi: f64,
    /// Current consciousness bottleneck (from Master Equation).
    pub bottleneck: String,
    /// Dominant harmony (index 0-7, or None).
    pub dominant_harmony: Option<usize>,
    /// Sacred Stillness activation [0, 1].
    pub stillness: f64,
}

impl Default for PhiPcgParams {
    fn default() -> Self {
        Self {
            phi: 0.5,
            bottleneck: "none".to_string(),
            dominant_harmony: None,
            stillness: 0.0,
        }
    }
}

/// Consciousness-modulated dungeon generation parameters.
///
/// These modify the BSP algorithm's behavior based on Φ.
#[derive(Debug, Clone)]
pub struct PhiDungeonConfig {
    /// BSP recursion depth: higher Φ → deeper tree → more rooms.
    pub bsp_depth: usize,
    /// Minimum room size: lower Φ → larger minimum (fewer, bigger rooms).
    pub min_room_size: usize,
    /// Minimum partition size for splitting.
    pub min_split_size: usize,
    /// Corridor width: bottleneck "embodiment" → narrower (physical challenge).
    pub corridor_width: usize,
    /// Extra connections: high Φ → more cross-links (integration).
    pub extra_connections: usize,
    /// Room size variance: high Φ → more varied (differentiation).
    pub size_variance: f64,
    /// Symmetry bias: Sacred Stillness → more symmetrical layouts.
    pub symmetry_bias: f64,
}

impl PhiDungeonConfig {
    /// Generate dungeon config from consciousness parameters.
    ///
    /// This is the core mapping from IIT theory to PCG parameters.
    pub fn from_phi(params: &PhiPcgParams) -> Self {
        let phi = params.phi.clamp(0.0, 1.0);

        // BSP depth: phi=0 → 3 (few rooms), phi=1 → 7 (many rooms)
        let bsp_depth = 3 + (phi * 4.0) as usize;

        // Min room size: phi=0 → 7 (big rooms), phi=1 → 4 (small varied rooms)
        let min_room_size = 7 - (phi * 3.0) as usize;

        // Split size: scales with room size
        let min_split_size = min_room_size * 2 + 2;

        // Corridor width: default 2, "embodiment" bottleneck → 1 (tighter)
        let corridor_width = if params.bottleneck == "embodiment" {
            1
        } else {
            2
        };

        // Extra connections: phi=0 → 0 (tree), phi=1 → 3 (loops/integration)
        let extra_connections = (phi * 3.0) as usize;

        // Size variance: phi=0 → 0.1 (uniform), phi=1 → 0.8 (differentiated)
        let size_variance = 0.1 + phi * 0.7;

        // Symmetry: Sacred Stillness → more symmetrical
        let symmetry_bias = params.stillness * 0.8;

        Self {
            bsp_depth,
            min_room_size,
            min_split_size,
            corridor_width,
            extra_connections,
            size_variance,
            symmetry_bias,
        }
    }

    /// Complexity score [0, 1]: how complex is this generated dungeon config?
    ///
    /// Used for testing the correlation between Φ and complexity.
    pub fn complexity_score(&self) -> f64 {
        let depth_score = (self.bsp_depth as f64 - 3.0) / 4.0; // [0, 1]
        let connection_score = self.extra_connections as f64 / 3.0; // [0, 1]
        let variance_score = self.size_variance; // [0.1, 0.8] → normalize
        let room_density = 1.0 - (self.min_room_size as f64 - 4.0) / 3.0; // [0, 1]

        (depth_score + connection_score + variance_score + room_density) / 4.0
    }
}

/// Scavenge item density modifier based on consciousness bottleneck.
///
/// When "knowledge" is the bottleneck, more learning items spawn.
/// When "attention" is the bottleneck, fewer items (less clutter).
pub fn scavenge_density_modifier(bottleneck: &str) -> f64 {
    match bottleneck {
        "knowledge" => 1.5,  // More items to learn from
        "attention" => 0.6,  // Fewer distractors
        "embodiment" => 0.8, // Moderate (physical focus)
        _ => 1.0,
    }
}

/// Room color temperature modifier based on dominant harmony.
///
/// Returns a (warmth, saturation) pair that the renderer can use
/// to tint rooms based on the prevailing harmony.
pub fn harmony_color_modulation(dominant_harmony: Option<usize>) -> (f32, f32) {
    match dominant_harmony {
        Some(0) => (0.8, 0.6), // Resonant Coherence → warm amber
        Some(1) => (0.4, 0.7), // Pan-Sentient Flourishing → cool green
        Some(2) => (0.6, 0.5), // Integral Wisdom → balanced
        Some(3) => (0.7, 0.8), // Infinite Play → warm, vivid
        Some(4) => (0.3, 0.6), // Universal Interconnectedness → cool blue
        Some(5) => (0.5, 0.7), // Sacred Reciprocity → balanced warm
        Some(6) => (0.6, 0.9), // Evolutionary Progression → vivid
        Some(7) => (0.2, 0.3), // Sacred Stillness → cool, muted
        _ => (0.5, 0.5),       // No dominant → neutral
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn high_phi_more_complex() {
        let low = PhiDungeonConfig::from_phi(&PhiPcgParams {
            phi: 0.1,
            ..Default::default()
        });
        let high = PhiDungeonConfig::from_phi(&PhiPcgParams {
            phi: 0.9,
            ..Default::default()
        });

        assert!(
            high.bsp_depth > low.bsp_depth,
            "high phi depth {} should > low phi depth {}",
            high.bsp_depth,
            low.bsp_depth
        );
        assert!(
            high.extra_connections > low.extra_connections,
            "high phi connections {} should > low {}",
            high.extra_connections,
            low.extra_connections
        );
        assert!(
            high.size_variance > low.size_variance,
            "high phi variance {} should > low {}",
            high.size_variance,
            low.size_variance
        );
    }

    #[test]
    fn complexity_correlates_with_phi() {
        let phis = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let mut prev_complexity = 0.0;

        for &phi in &phis {
            let config = PhiDungeonConfig::from_phi(&PhiPcgParams {
                phi,
                ..Default::default()
            });
            let complexity = config.complexity_score();
            assert!(
                complexity >= prev_complexity - 0.01, // allow tiny float rounding
                "phi={} complexity={} should be >= prev={}",
                phi,
                complexity,
                prev_complexity
            );
            prev_complexity = complexity;
        }
    }

    #[test]
    fn embodiment_bottleneck_narrows_corridors() {
        let normal = PhiDungeonConfig::from_phi(&PhiPcgParams {
            phi: 0.5,
            bottleneck: "none".to_string(),
            ..Default::default()
        });
        let embodiment = PhiDungeonConfig::from_phi(&PhiPcgParams {
            phi: 0.5,
            bottleneck: "embodiment".to_string(),
            ..Default::default()
        });

        assert!(
            embodiment.corridor_width < normal.corridor_width,
            "embodiment corridors {} should be < normal {}",
            embodiment.corridor_width,
            normal.corridor_width
        );
    }

    #[test]
    fn stillness_increases_symmetry() {
        let calm = PhiDungeonConfig::from_phi(&PhiPcgParams {
            phi: 0.5,
            stillness: 0.9,
            ..Default::default()
        });
        let active = PhiDungeonConfig::from_phi(&PhiPcgParams {
            phi: 0.5,
            stillness: 0.1,
            ..Default::default()
        });

        assert!(
            calm.symmetry_bias > active.symmetry_bias,
            "stillness symmetry {} should > active {}",
            calm.symmetry_bias,
            active.symmetry_bias
        );
    }

    #[test]
    fn scavenge_density_knowledge_bottleneck() {
        assert!(scavenge_density_modifier("knowledge") > 1.0);
        assert!(scavenge_density_modifier("attention") < 1.0);
        assert!((scavenge_density_modifier("other") - 1.0).abs() < 1e-10);
    }

    #[test]
    fn harmony_colors_all_defined() {
        for i in 0..8 {
            let (warmth, sat) = harmony_color_modulation(Some(i));
            assert!(warmth >= 0.0 && warmth <= 1.0);
            assert!(sat >= 0.0 && sat <= 1.0);
        }
        let (w, s) = harmony_color_modulation(None);
        assert!((w - 0.5).abs() < 1e-5);
    }

    #[test]
    fn extreme_phi_values_safe() {
        // Should not panic at extremes
        let _ = PhiDungeonConfig::from_phi(&PhiPcgParams {
            phi: 0.0,
            ..Default::default()
        });
        let _ = PhiDungeonConfig::from_phi(&PhiPcgParams {
            phi: 1.0,
            ..Default::default()
        });
        let _ = PhiDungeonConfig::from_phi(&PhiPcgParams {
            phi: -0.5, // clamped
            ..Default::default()
        });
        let _ = PhiDungeonConfig::from_phi(&PhiPcgParams {
            phi: 2.0, // clamped
            ..Default::default()
        });
    }

    #[test]
    fn config_values_in_bounds() {
        for phi_10 in 0..=10 {
            let phi = phi_10 as f64 / 10.0;
            let config = PhiDungeonConfig::from_phi(&PhiPcgParams {
                phi,
                ..Default::default()
            });
            assert!(config.bsp_depth >= 3 && config.bsp_depth <= 7);
            assert!(config.min_room_size >= 4 && config.min_room_size <= 7);
            assert!(config.extra_connections <= 3);
            assert!(config.corridor_width >= 1 && config.corridor_width <= 2);
        }
    }
}
