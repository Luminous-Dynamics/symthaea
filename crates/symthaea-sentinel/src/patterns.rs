// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Audio pattern definitions and similarity structures.
//!
//! This module provides:
//! - `AudioPattern`: Learned pattern with prototypes and features
//! - `AudioCategory`: Category classification enum
//! - `PatternSimilarity`: Similarity scores structure

use crate::features::MEL_BANDS;
use crate::hdc::HV;

/// Number of LTC timescale levels for multi-scale prototypes
pub const NUM_LTC_LEVELS: usize = 5;

/// Maximum number of exemplars to store per pattern
pub const MAX_EXEMPLARS: usize = 5;

// =============================================================================
// Audio Category
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AudioCategory {
    Music,
    Speech,
    Nature,
    Urban,
    Mechanical,
    Silence,
    Unknown,
}

// =============================================================================
// Pattern Similarity
// =============================================================================

#[derive(Clone)]
pub struct PatternSimilarity {
    pub timbre: f32,
    pub rhythm: f32,
    pub envelope: f32,
    pub rhythm_freq: f32,
    pub timbre_freq: f32,
    pub spectral: f32,
    pub multi_scale: f32,
    pub combined: f32,
}

// =============================================================================
// Audio Pattern
// =============================================================================

#[derive(Clone)]
pub struct AudioPattern {
    pub name: String,
    pub category: AudioCategory,

    // Stored prototypes (averaged representation)
    pub timbre_prototype: HV,
    pub rhythm_prototype: HV,
    pub envelope_prototype: HV,
    pub context_prototype: HV,

    // Multi-prototype exemplars
    pub timbre_exemplars: Vec<HV>,
    pub rhythm_exemplars: Vec<HV>,
    pub trajectory_exemplars: Vec<HV>,

    // Trajectory encoding
    pub trajectory_prototype: HV,

    // Multi-scale prototypes
    pub timbre_scale_prototypes: [HV; NUM_LTC_LEVELS],
    pub rhythm_scale_prototypes: [HV; NUM_LTC_LEVELS],

    // Temporal signatures
    pub rhythm_freq_signature: [f32; 8],
    pub timbre_freq_signature: [f32; 8],

    // Aggregate spectral features
    pub mean_spectral_centroid: f32,
    pub mean_spectral_flatness: f32,
    pub mean_onset_strength: f32,
    pub mean_temporal_regularity: f32,
    pub mean_mel_bands: Vec<f32>,
    pub mean_envelope_delta: f32,
    pub mean_envelope_variance: f32,
    pub mean_spectral_flux: f32,
    pub mean_harmonic_ratio: f32,
    pub mean_cfc_theta_gamma: f32,
    pub mean_cfc_delta_beta: f32,
    pub mean_attack_sharpness: f32,
    pub mean_decay_roughness: f32,
    pub mean_silence_ratio: f32,
    pub mean_burst_density: f32,

    // Accumulators (used during learning)
    pub(crate) centroid_sum: f32,
    pub(crate) flatness_sum: f32,
    pub(crate) onset_sum: f32,
    pub(crate) regularity_sum: f32,
    pub(crate) mel_bands_sum: Vec<f32>,
    pub(crate) envelope_delta_sum: f32,
    pub(crate) envelope_variance_sum: f32,
    pub(crate) spectral_flux_sum: f32,
    pub(crate) harmonic_ratio_sum: f32,
    pub(crate) cfc_theta_gamma_sum: f32,
    pub(crate) cfc_delta_beta_sum: f32,
    pub(crate) attack_sharpness_sum: f32,
    pub(crate) decay_roughness_sum: f32,
    pub(crate) silence_ratio_sum: f32,
    pub(crate) burst_density_sum: f32,

    // Statistics
    pub frame_count: usize,
}

impl AudioPattern {
    pub fn new(name: &str, category: AudioCategory) -> Self {
        Self {
            name: name.to_string(),
            category,
            timbre_prototype: HV::zero(),
            rhythm_prototype: HV::zero(),
            envelope_prototype: HV::zero(),
            context_prototype: HV::zero(),
            timbre_exemplars: Vec::new(),
            rhythm_exemplars: Vec::new(),
            trajectory_exemplars: Vec::new(),
            trajectory_prototype: HV::zero(),
            timbre_scale_prototypes: [HV::zero(), HV::zero(), HV::zero(), HV::zero(), HV::zero()],
            rhythm_scale_prototypes: [HV::zero(), HV::zero(), HV::zero(), HV::zero(), HV::zero()],
            rhythm_freq_signature: [0.0; 8],
            timbre_freq_signature: [0.0; 8],
            mean_spectral_centroid: 0.0,
            mean_spectral_flatness: 0.0,
            mean_onset_strength: 0.0,
            mean_temporal_regularity: 0.0,
            mean_mel_bands: vec![0.0; MEL_BANDS],
            mean_envelope_delta: 0.0,
            mean_envelope_variance: 0.0,
            mean_spectral_flux: 0.0,
            mean_harmonic_ratio: 0.0,
            mean_cfc_theta_gamma: 0.0,
            mean_cfc_delta_beta: 0.0,
            mean_attack_sharpness: 0.0,
            mean_decay_roughness: 0.0,
            mean_silence_ratio: 0.0,
            mean_burst_density: 0.0,
            centroid_sum: 0.0,
            flatness_sum: 0.0,
            onset_sum: 0.0,
            regularity_sum: 0.0,
            mel_bands_sum: vec![0.0; MEL_BANDS],
            envelope_delta_sum: 0.0,
            envelope_variance_sum: 0.0,
            spectral_flux_sum: 0.0,
            harmonic_ratio_sum: 0.0,
            cfc_theta_gamma_sum: 0.0,
            cfc_delta_beta_sum: 0.0,
            attack_sharpness_sum: 0.0,
            decay_roughness_sum: 0.0,
            silence_ratio_sum: 0.0,
            burst_density_sum: 0.0,
            frame_count: 0,
        }
    }
}

// =============================================================================
// Ambient Contexts
// =============================================================================

/// Pre-defined ambient context profiles
pub struct AmbientContexts {
    pub office: AudioPattern,
    pub traffic: AudioPattern,
    pub nature: AudioPattern,
    pub home_quiet: AudioPattern,
    pub cafe: AudioPattern,
}

impl AmbientContexts {
    /// Create baseline context patterns
    pub fn baseline() -> Self {
        let mut office = AudioPattern::new("Office", AudioCategory::Urban);
        office.rhythm_freq_signature = [0.1, 0.05, 0.02, 0.01, 0.01, 0.0, 0.0, 0.0];
        office.timbre_freq_signature = [0.2, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0];

        let mut traffic = AudioPattern::new("Traffic", AudioCategory::Urban);
        traffic.rhythm_freq_signature = [0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0];
        traffic.timbre_freq_signature = [0.1, 0.1, 0.1, 0.1, 0.05, 0.02, 0.01, 0.0];

        let mut nature = AudioPattern::new("Nature", AudioCategory::Nature);
        nature.rhythm_freq_signature = [0.15, 0.1, 0.05, 0.02, 0.01, 0.01, 0.0, 0.0];
        nature.timbre_freq_signature = [0.1, 0.15, 0.1, 0.05, 0.02, 0.01, 0.0, 0.0];

        let mut home_quiet = AudioPattern::new("HomeQuiet", AudioCategory::Silence);
        home_quiet.rhythm_freq_signature = [0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0];
        home_quiet.timbre_freq_signature = [0.1, 0.05, 0.02, 0.01, 0.0, 0.0, 0.0, 0.0];

        let mut cafe = AudioPattern::new("Cafe", AudioCategory::Urban);
        cafe.rhythm_freq_signature = [0.1, 0.15, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01];
        cafe.timbre_freq_signature = [0.1, 0.1, 0.15, 0.1, 0.05, 0.02, 0.01, 0.0];

        Self {
            office,
            traffic,
            nature,
            home_quiet,
            cafe,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_audio_category_variants() {
        assert_ne!(AudioCategory::Music, AudioCategory::Speech);
        assert_eq!(AudioCategory::Unknown, AudioCategory::Unknown);
        assert_ne!(AudioCategory::Nature, AudioCategory::Urban);
    }

    #[test]
    fn test_audio_pattern_new() {
        let pattern = AudioPattern::new("test_pattern", AudioCategory::Music);
        assert_eq!(pattern.name, "test_pattern");
        assert_eq!(pattern.category, AudioCategory::Music);
        assert_eq!(pattern.frame_count, 0);
        assert_eq!(pattern.mean_spectral_centroid, 0.0);
        assert!(pattern.timbre_exemplars.is_empty());
        assert!(pattern.rhythm_exemplars.is_empty());
    }

    #[test]
    fn test_audio_pattern_prototypes_are_zero() {
        let pattern = AudioPattern::new("test", AudioCategory::Speech);
        assert!(pattern.timbre_prototype.values.iter().all(|v| *v == 0.0));
        assert!(pattern.rhythm_prototype.values.iter().all(|v| *v == 0.0));
        assert!(pattern.envelope_prototype.values.iter().all(|v| *v == 0.0));
        assert!(pattern.context_prototype.values.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_audio_pattern_freq_signatures_zero() {
        let pattern = AudioPattern::new("test", AudioCategory::Nature);
        assert!(pattern.rhythm_freq_signature.iter().all(|v| *v == 0.0));
        assert!(pattern.timbre_freq_signature.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_audio_pattern_mel_bands_correct_size() {
        let pattern = AudioPattern::new("test", AudioCategory::Urban);
        assert_eq!(pattern.mean_mel_bands.len(), MEL_BANDS);
        assert_eq!(pattern.mel_bands_sum.len(), MEL_BANDS);
    }

    #[test]
    fn test_audio_pattern_clone() {
        let mut pattern = AudioPattern::new("original", AudioCategory::Mechanical);
        pattern.mean_spectral_centroid = 3000.0;
        pattern.frame_count = 42;
        let cloned = pattern.clone();
        assert_eq!(cloned.name, "original");
        assert_eq!(cloned.category, AudioCategory::Mechanical);
        assert_eq!(cloned.mean_spectral_centroid, 3000.0);
        assert_eq!(cloned.frame_count, 42);
    }

    #[test]
    fn test_pattern_similarity_clone() {
        let sim = PatternSimilarity {
            timbre: 0.8,
            rhythm: 0.6,
            envelope: 0.7,
            rhythm_freq: 0.5,
            timbre_freq: 0.4,
            spectral: 0.9,
            multi_scale: 0.3,
            combined: 0.65,
        };
        let cloned = sim.clone();
        assert_eq!(cloned.combined, 0.65);
        assert_eq!(cloned.timbre, 0.8);
    }

    // =========================================================================
    // AmbientContexts tests
    // =========================================================================

    #[test]
    fn test_ambient_contexts_baseline() {
        let contexts = AmbientContexts::baseline();
        assert_eq!(contexts.office.name, "Office");
        assert_eq!(contexts.office.category, AudioCategory::Urban);
        assert_eq!(contexts.traffic.name, "Traffic");
        assert_eq!(contexts.nature.name, "Nature");
        assert_eq!(contexts.nature.category, AudioCategory::Nature);
        assert_eq!(contexts.home_quiet.name, "HomeQuiet");
        assert_eq!(contexts.home_quiet.category, AudioCategory::Silence);
        assert_eq!(contexts.cafe.name, "Cafe");
    }

    #[test]
    fn test_ambient_contexts_have_nonzero_signatures() {
        let contexts = AmbientContexts::baseline();
        // All contexts should have some nonzero frequency signature entries
        assert!(
            contexts
                .office
                .rhythm_freq_signature
                .iter()
                .any(|v| *v > 0.0)
        );
        assert!(
            contexts
                .traffic
                .rhythm_freq_signature
                .iter()
                .any(|v| *v > 0.0)
        );
        assert!(
            contexts
                .nature
                .timbre_freq_signature
                .iter()
                .any(|v| *v > 0.0)
        );
        assert!(contexts.cafe.rhythm_freq_signature.iter().any(|v| *v > 0.0));
    }

    #[test]
    fn test_ambient_contexts_home_quiet_lowest_energy() {
        let contexts = AmbientContexts::baseline();
        let home_max: f32 = contexts
            .home_quiet
            .rhythm_freq_signature
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        let traffic_max: f32 = contexts
            .traffic
            .rhythm_freq_signature
            .iter()
            .copied()
            .fold(0.0f32, f32::max);
        assert!(
            home_max < traffic_max,
            "Home quiet should have lower rhythm energy than traffic"
        );
    }
}
