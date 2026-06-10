// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Substrate-dependent timbre: how different physical substrates sound.
//!
//! Maps substrate properties (speed, energy, integration capacity) to
//! synthesis parameters. Each substrate type produces a distinct sonic
//! character reflecting its computational nature:
//!
//! | Substrate | Character | Key Trait |
//! |-----------|-----------|-----------|
//! | Biological | Warm, imprecise, breathy | Natural vibrato, noise |
//! | Silicon | Crystalline, precise, clean | Pure partials, tight ADSR |
//! | Quantum | Entangled stereo, shimmering | Correlated L/R, interference |
//! | Photonic | Bright, ultra-fast transients | Many partials, sharp attack |
//! | Neuromorphic | Spike-like, stochastic | Irregular onsets, jitter |
//! | Biochemical | Slow, evolving, organic | Long morphs, gentle |
//! | Hybrid | Blended characteristics | Weighted mix |
//! | Exotic | Unpredictable, alien | Random modulation |

use crate::{MuseConfig, MusicalState};

/// Substrate type for timbre selection.
///
/// Mirrors `symthaea_core::hdc::substrate_independence::SubstrateType`
/// without requiring the core dependency.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubstrateTimbreType {
    Biological,
    Silicon,
    Quantum,
    Photonic,
    Neuromorphic,
    Biochemical,
    Hybrid,
    Exotic,
}

impl Default for SubstrateTimbreType {
    fn default() -> Self {
        Self::Silicon
    }
}

/// Substrate timbre modifier applied to synthesis parameters.
#[derive(Debug, Clone)]
pub struct SubstrateTimbreModifier {
    pub substrate: SubstrateTimbreType,
    /// Vibrato depth multiplier (biological = 2.0, silicon = 0.1).
    pub vibrato_scale: f32,
    /// Noise mix addition (biological = 0.03, silicon = 0.0).
    pub noise_addition: f32,
    /// Partial count preference (photonic = 16, silicon = 4).
    pub preferred_partials: usize,
    /// Attack time multiplier (photonic = 0.3, biochemical = 3.0).
    pub attack_scale: f32,
    /// FM depth multiplier (exotic = 2.0, biological = 0.5).
    pub fm_scale: f32,
    /// Stereo correlation [-1, 1] (quantum = oscillating, others = 1.0).
    pub stereo_correlation: f32,
    /// Morph rate multiplier (biochemical = 0.1, photonic = 5.0).
    pub morph_rate: f32,
    /// Jitter amount (neuromorphic = high, silicon = 0).
    pub jitter: f32,
}

impl SubstrateTimbreModifier {
    /// Create a timbre modifier for the given substrate type.
    pub fn for_substrate(substrate: SubstrateTimbreType) -> Self {
        match substrate {
            SubstrateTimbreType::Biological => Self {
                substrate,
                vibrato_scale: 2.0,      // natural vibrato
                noise_addition: 0.03,    // breath noise
                preferred_partials: 8,   // moderate richness
                attack_scale: 1.0,       // natural
                fm_scale: 0.5,           // mild modulation
                stereo_correlation: 1.0, // normal stereo
                morph_rate: 0.5,         // moderate evolution
                jitter: 0.02,            // slight imprecision
            },
            SubstrateTimbreType::Silicon => Self {
                substrate,
                vibrato_scale: 0.1,    // minimal vibrato
                noise_addition: 0.0,   // clean
                preferred_partials: 4, // sparse, pure
                attack_scale: 0.5,     // snappy
                fm_scale: 1.0,         // precise FM
                stereo_correlation: 1.0,
                morph_rate: 1.0,
                jitter: 0.0, // perfectly precise
            },
            SubstrateTimbreType::Quantum => Self {
                substrate,
                vibrato_scale: 0.3,
                noise_addition: 0.01,
                preferred_partials: 12,
                attack_scale: 0.8,
                fm_scale: 1.5,           // rich interference patterns
                stereo_correlation: 0.0, // entangled: anti-correlated channels
                morph_rate: 3.0,         // rapid superposition collapse
                jitter: 0.005,
            },
            SubstrateTimbreType::Photonic => Self {
                substrate,
                vibrato_scale: 0.05, // almost none
                noise_addition: 0.005,
                preferred_partials: 16, // maximum brightness
                attack_scale: 0.1,      // ultra-fast transients
                fm_scale: 0.8,
                stereo_correlation: 1.0,
                morph_rate: 5.0, // speed of light
                jitter: 0.0,
            },
            SubstrateTimbreType::Neuromorphic => Self {
                substrate,
                vibrato_scale: 0.8,
                noise_addition: 0.02,
                preferred_partials: 6,
                attack_scale: 1.2,
                fm_scale: 0.7,
                stereo_correlation: 0.8,
                morph_rate: 0.8,
                jitter: 0.05, // spike-train irregularity
            },
            SubstrateTimbreType::Biochemical => Self {
                substrate,
                vibrato_scale: 1.5,
                noise_addition: 0.04, // molecular noise
                preferred_partials: 6,
                attack_scale: 3.0, // very slow
                fm_scale: 0.3,
                stereo_correlation: 1.0,
                morph_rate: 0.1, // glacial evolution
                jitter: 0.03,
            },
            SubstrateTimbreType::Hybrid => Self {
                substrate,
                vibrato_scale: 1.0,
                noise_addition: 0.015,
                preferred_partials: 8,
                attack_scale: 1.0,
                fm_scale: 1.0,
                stereo_correlation: 0.9,
                morph_rate: 1.0,
                jitter: 0.01,
            },
            SubstrateTimbreType::Exotic => Self {
                substrate,
                vibrato_scale: 3.0,   // alien vibrato
                noise_addition: 0.06, // high noise
                preferred_partials: 10,
                attack_scale: 0.7,
                fm_scale: 2.5,            // extreme modulation
                stereo_correlation: -0.5, // anti-phase channels
                morph_rate: 8.0,          // chaotic
                jitter: 0.08,             // very irregular
            },
        }
    }

    /// Apply substrate timbre modifications to a MuseConfig.
    ///
    /// Modifies synthesis parameters to reflect the substrate's sonic character.
    pub fn apply_to_config(&self, config: &mut MuseConfig) {
        config.num_partials = self.preferred_partials;
        config.noise_mix = (config.noise_mix + self.noise_addition).clamp(0.0, 0.3);
        config.max_fm_depth *= self.fm_scale;
    }

    /// Apply substrate timbre modifications to a MusicalState.
    ///
    /// Modulates neuromodulators to shift timbre character.
    pub fn apply_to_state(&self, state: &mut MusicalState) {
        // Jitter modulates noradrenaline (edge/tension)
        state.noradrenaline = (state.noradrenaline + self.jitter).clamp(0.0, 1.0);
    }

    /// Get per-sample stereo decorrelation for quantum/exotic substrates.
    ///
    /// Returns (left_scale, right_scale) that create phase relationships
    /// between channels reflecting the substrate's information integration.
    /// - correlation=1.0: normal stereo (identical signal)
    /// - correlation=0.0: independent channels (quantum entanglement)
    /// - correlation=-0.5: anti-phase (exotic interference)
    pub fn stereo_decorrelation(&self, sample_idx: usize) -> (f32, f32) {
        if (self.stereo_correlation - 1.0).abs() < 0.01 {
            return (1.0, 1.0); // normal
        }

        // Create oscillating correlation for quantum substrates
        let phase = sample_idx as f32 * 0.001; // slow oscillation
        let corr =
            self.stereo_correlation + (1.0 - self.stereo_correlation.abs()) * 0.5 * phase.sin();

        let mid = 0.5 * (1.0 + corr);
        let side = 0.5 * (1.0 - corr);

        (mid + side, mid - side)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn biological_warm_character() {
        let bio = SubstrateTimbreModifier::for_substrate(SubstrateTimbreType::Biological);
        assert!(
            bio.vibrato_scale > 1.0,
            "biological should have natural vibrato"
        );
        assert!(
            bio.noise_addition > 0.0,
            "biological should have breath noise"
        );
        assert!(bio.jitter > 0.0, "biological should have imprecision");
    }

    #[test]
    fn silicon_crystalline_character() {
        let si = SubstrateTimbreModifier::for_substrate(SubstrateTimbreType::Silicon);
        assert!(
            si.vibrato_scale < 0.2,
            "silicon should have minimal vibrato"
        );
        assert_eq!(si.noise_addition, 0.0, "silicon should be clean");
        assert_eq!(si.jitter, 0.0, "silicon should be precise");
        assert!(
            si.preferred_partials <= 6,
            "silicon should have sparse spectrum"
        );
    }

    #[test]
    fn quantum_entangled_stereo() {
        let q = SubstrateTimbreModifier::for_substrate(SubstrateTimbreType::Quantum);
        assert!(
            q.stereo_correlation < 0.5,
            "quantum should have low stereo correlation"
        );

        let (l, r) = q.stereo_decorrelation(0);
        assert!((l - r).abs() > 0.01, "quantum channels should differ");
    }

    #[test]
    fn photonic_fast_bright() {
        let ph = SubstrateTimbreModifier::for_substrate(SubstrateTimbreType::Photonic);
        assert!(
            ph.attack_scale < 0.2,
            "photonic should have ultra-fast attack"
        );
        assert_eq!(
            ph.preferred_partials, 16,
            "photonic should be maximally bright"
        );
        assert!(ph.morph_rate > 3.0, "photonic should morph rapidly");
    }

    #[test]
    fn apply_to_config_modifies() {
        let exotic = SubstrateTimbreModifier::for_substrate(SubstrateTimbreType::Exotic);
        let mut config = MuseConfig::default();
        let original_fm = config.max_fm_depth;
        exotic.apply_to_config(&mut config);

        assert_eq!(config.num_partials, 10);
        assert!(config.noise_mix > 0.0);
        assert!(
            config.max_fm_depth > original_fm,
            "exotic should increase FM depth"
        );
    }

    #[test]
    fn all_substrates_produce_valid_modifiers() {
        let substrates = [
            SubstrateTimbreType::Biological,
            SubstrateTimbreType::Silicon,
            SubstrateTimbreType::Quantum,
            SubstrateTimbreType::Photonic,
            SubstrateTimbreType::Neuromorphic,
            SubstrateTimbreType::Biochemical,
            SubstrateTimbreType::Hybrid,
            SubstrateTimbreType::Exotic,
        ];

        for s in &substrates {
            let m = SubstrateTimbreModifier::for_substrate(*s);
            assert!(m.vibrato_scale >= 0.0);
            assert!(m.noise_addition >= 0.0);
            assert!(m.preferred_partials >= 1 && m.preferred_partials <= 16);
            assert!(m.attack_scale > 0.0);
        }
    }

    #[test]
    fn normal_stereo_correlation_unity() {
        let si = SubstrateTimbreModifier::for_substrate(SubstrateTimbreType::Silicon);
        let (l, r) = si.stereo_decorrelation(42);
        assert!((l - 1.0).abs() < 0.01);
        assert!((r - 1.0).abs() < 0.01);
    }
}
