// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Epigenetic profiling: imprinting validation, methylation encoding, and similarity.
//!
//! Uses 16,384D HDC vectors to represent methylation patterns for
//! high-dimensional similarity comparison between epigenetic profiles.

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::types::EpigeneticProfile;

// Seed constants for deterministic HV generation
const SEED_GLOBAL_METH: u64 = 0xEF16_0001;
const SEED_H19_IGF2: u64 = 0xEF16_0002;
const SEED_SNRPN: u64 = 0xEF16_0003;
const SEED_X_INACT: u64 = 0xEF16_0004;
const SEED_TELOMERE: u64 = 0xEF16_0005;

/// Validate imprinting status of an epigenetic profile.
///
/// Checks H19/IGF2, SNRPN, and X-inactivation status. Returns (valid, issues)
/// where issues lists human-readable descriptions of any problems found.
pub fn validate_imprinting(profile: &EpigeneticProfile) -> (bool, Vec<String>) {
    let mut issues = Vec::new();

    if profile.h19_igf2_imprinting < 0.8 {
        issues.push(format!(
            "H19/IGF2 imprinting incomplete ({:.2}, threshold 0.8)",
            profile.h19_igf2_imprinting
        ));
    }

    if profile.snrpn_imprinting < 0.8 {
        issues.push(format!(
            "SNRPN imprinting incomplete ({:.2}, threshold 0.8)",
            profile.snrpn_imprinting
        ));
    }

    if !profile.x_inactivation_status {
        issues.push("X-inactivation not established".into());
    }

    if profile.telomere_length < 0.5 {
        issues.push(format!(
            "Telomere length critically short ({:.2}, threshold 0.5)",
            profile.telomere_length
        ));
    }

    if profile.global_methylation < 0.3 {
        issues.push(format!(
            "Global methylation critically low ({:.2}, threshold 0.3)",
            profile.global_methylation
        ));
    }

    (issues.is_empty(), issues)
}

/// Encode an epigenetic profile as a 16,384D holographic hypervector.
///
/// Each epigenetic feature is encoded as a thermometer-coded random HV,
/// then all features are bundled into a single representation.
pub fn encode_methylation_pattern(profile: &EpigeneticProfile) -> ContinuousHV {
    let global_meth = ContinuousHV::random(HDC_DIMENSION, SEED_GLOBAL_METH)
        .scale(profile.global_methylation as f32);
    let h19 = ContinuousHV::random(HDC_DIMENSION, SEED_H19_IGF2)
        .scale(profile.h19_igf2_imprinting as f32);
    let snrpn =
        ContinuousHV::random(HDC_DIMENSION, SEED_SNRPN).scale(profile.snrpn_imprinting as f32);
    let x_inact =
        ContinuousHV::random(HDC_DIMENSION, SEED_X_INACT).scale(if profile.x_inactivation_status {
            1.0
        } else {
            0.0
        });
    let telomere =
        ContinuousHV::random(HDC_DIMENSION, SEED_TELOMERE).scale(profile.telomere_length as f32);

    ContinuousHV::bundle(&[&global_meth, &h19, &snrpn, &x_inact, &telomere])
}

/// Compute cosine similarity between two epigenetic profiles in HDC space.
///
/// Returns a value in [-1, 1] where 1 means identical profiles.
pub fn methylation_similarity(a: &EpigeneticProfile, b: &EpigeneticProfile) -> f64 {
    let hv_a = encode_methylation_pattern(a);
    let hv_b = encode_methylation_pattern(b);
    hv_a.similarity(&hv_b) as f64
}

/// Compute an overall epigenetic quality score (0-1).
///
/// Combines imprinting completeness, methylation level, and telomere length.
pub fn epigenetic_quality_score(profile: &EpigeneticProfile) -> f64 {
    let imprint_score = (profile.h19_igf2_imprinting + profile.snrpn_imprinting) / 2.0;
    let telomere_score = profile.telomere_length.clamp(0.0, 1.5) / 1.5;
    let meth_score = profile.global_methylation.clamp(0.0, 1.0);
    let x_score = if profile.x_inactivation_status {
        1.0
    } else {
        0.5
    };

    (0.3 * imprint_score + 0.2 * telomere_score + 0.3 * meth_score + 0.2 * x_score).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_healthy_newborn() {
        let profile = EpigeneticProfile::healthy_newborn();
        let (valid, issues) = validate_imprinting(&profile);
        assert!(valid, "Healthy newborn should pass: {:?}", issues);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_validate_partially_erased() {
        let profile = EpigeneticProfile::partially_erased();
        let (valid, issues) = validate_imprinting(&profile);
        assert!(!valid, "Partially erased should fail validation");
        assert!(!issues.is_empty());
    }

    #[test]
    fn test_validate_low_h19() {
        let mut profile = EpigeneticProfile::healthy_newborn();
        profile.h19_igf2_imprinting = 0.5;
        let (valid, issues) = validate_imprinting(&profile);
        assert!(!valid);
        assert!(issues.iter().any(|s| s.contains("H19/IGF2")));
    }

    #[test]
    fn test_validate_low_snrpn() {
        let mut profile = EpigeneticProfile::healthy_newborn();
        profile.snrpn_imprinting = 0.3;
        let (valid, issues) = validate_imprinting(&profile);
        assert!(!valid);
        assert!(issues.iter().any(|s| s.contains("SNRPN")));
    }

    #[test]
    fn test_validate_x_inactivation() {
        let mut profile = EpigeneticProfile::healthy_newborn();
        profile.x_inactivation_status = false;
        let (valid, issues) = validate_imprinting(&profile);
        assert!(!valid);
        assert!(issues.iter().any(|s| s.contains("X-inactivation")));
    }

    #[test]
    fn test_encode_deterministic() {
        let profile = EpigeneticProfile::healthy_newborn();
        let hv1 = encode_methylation_pattern(&profile);
        let hv2 = encode_methylation_pattern(&profile);
        let sim = hv1.similarity(&hv2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Same profile should produce identical HV"
        );
    }

    #[test]
    fn test_encode_dimension() {
        let profile = EpigeneticProfile::healthy_newborn();
        let hv = encode_methylation_pattern(&profile);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_similarity_identical() {
        let profile = EpigeneticProfile::healthy_newborn();
        let sim = methylation_similarity(&profile, &profile);
        assert!(
            (sim - 1.0).abs() < 1e-4,
            "Identical profiles should have similarity ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_similarity_different() {
        let healthy = EpigeneticProfile::healthy_newborn();
        let erased = EpigeneticProfile::partially_erased();
        let sim = methylation_similarity(&healthy, &erased);
        assert!(
            sim < 0.95,
            "Different profiles should have lower similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_quality_score_healthy() {
        let profile = EpigeneticProfile::healthy_newborn();
        let score = epigenetic_quality_score(&profile);
        assert!(
            score > 0.7,
            "Healthy profile should have high quality score: {}",
            score
        );
    }

    #[test]
    fn test_quality_score_erased() {
        let profile = EpigeneticProfile::partially_erased();
        let score = epigenetic_quality_score(&profile);
        assert!(
            score < epigenetic_quality_score(&EpigeneticProfile::healthy_newborn()),
            "Erased profile should have lower quality than healthy"
        );
    }

    #[test]
    fn test_quality_score_bounded() {
        let profile = EpigeneticProfile::healthy_newborn();
        let score = epigenetic_quality_score(&profile);
        assert!(score >= 0.0 && score <= 1.0);
    }
}
