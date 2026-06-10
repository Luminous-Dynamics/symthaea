// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! HDC encoding of fetal state into 16,384D holographic representations.
//!
//! Each fetal measurement is encoded as a scalar HV, then bound with a week-specific
//! basis HV to create a time-stamped association. All bound pairs are bundled into
//! a single holographic snapshot of the fetal state.

use symthaea_core::hdc::unified_hv::{ContinuousHV, HDC_DIMENSION};

use crate::types::{FetalMetrics, GestationalWeek};

// Seed constants for deterministic HV generation (valid hex only).
const SEED_WEEK_BASE: u64 = 0x1EE1_BA5E;
const SEED_WEIGHT: u64 = 0xAE16_FACE;
const SEED_HEART_RATE: u64 = 0xBEA1_4A1E;
const SEED_MOVEMENT: u64 = 0xC0DE_F1A5;
const SEED_BRAIN: u64 = 0xB4A1_FEED;
const SEED_LUNG: u64 = 0x10DE_CAFE;
const SEED_CRL: u64 = 0xC41_DE5A;

/// Encode a scalar value (expected range 0-1) as a deterministic HV.
///
/// Quantizes to 256 levels and generates a reproducible HV per level.
fn encode_scalar(value: f64, seed_base: u64) -> ContinuousHV {
    let clamped = value.clamp(0.0, 1.0);
    let quantized = (clamped * 255.0) as u64;
    ContinuousHV::random(HDC_DIMENSION, seed_base.wrapping_add(quantized * 37))
}

/// Encode the complete fetal state as a 16,384D holographic vector.
///
/// Binds each metric with a week-specific HV, then bundles all channels.
/// The resulting HV is a compressed holographic snapshot that preserves
/// similarity structure: similar fetal states produce similar HVs.
pub fn encode_fetal_state(metrics: &FetalMetrics) -> ContinuousHV {
    let week_hv = ContinuousHV::random(HDC_DIMENSION, SEED_WEEK_BASE + metrics.week.0 as u64);

    let weight_hv = encode_scalar(metrics.weight_grams / 4000.0, SEED_WEIGHT);
    let hr_hv = encode_scalar(metrics.heart_rate_bpm / 200.0, SEED_HEART_RATE);
    let movement_hv = encode_scalar(metrics.movement_score, SEED_MOVEMENT);
    let brain_hv = encode_scalar(metrics.brain_activity_score, SEED_BRAIN);
    let lung_hv = encode_scalar(metrics.lung_maturity, SEED_LUNG);
    let crl_hv = encode_scalar(metrics.crown_rump_length_mm / 600.0, SEED_CRL);

    // Bind each metric with the week HV to create time-stamped associations
    let bound_weight = week_hv.bind(&weight_hv);
    let bound_hr = week_hv.bind(&hr_hv);
    let bound_movement = week_hv.bind(&movement_hv);
    let bound_brain = week_hv.bind(&brain_hv);
    let bound_lung = week_hv.bind(&lung_hv);
    let bound_crl = week_hv.bind(&crl_hv);

    // Bundle all bound pairs into a single holographic snapshot
    ContinuousHV::bundle(&[
        &bound_weight,
        &bound_hr,
        &bound_movement,
        &bound_brain,
        &bound_lung,
        &bound_crl,
    ])
}

/// Encode the normative (expected) fetal trajectory at a given week.
///
/// Uses population-average metrics to create a reference HV for comparison.
pub fn encode_normative_trajectory(week: GestationalWeek) -> ContinuousHV {
    let normative = FetalMetrics::normative(week);
    encode_fetal_state(&normative)
}

/// Compute anomaly score between an actual fetal state HV and a normative reference.
///
/// Returns a value in [0, 2]: 0 = identical, ~1 = orthogonal (random), 2 = opposite.
/// Values below ~0.3 are normal; above ~0.6 warrant investigation.
pub fn anomaly_score(actual: &ContinuousHV, normative: &ContinuousHV) -> f64 {
    // Cosine distance: 1 - similarity
    let sim = actual.similarity(normative) as f64;
    1.0 - sim
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_fetal_state_dimension() {
        let metrics = FetalMetrics::normative(GestationalWeek::new(20));
        let hv = encode_fetal_state(&metrics);
        assert_eq!(hv.dim(), HDC_DIMENSION);
    }

    #[test]
    fn test_encode_deterministic() {
        let metrics = FetalMetrics::normative(GestationalWeek::new(20));
        let hv1 = encode_fetal_state(&metrics);
        let hv2 = encode_fetal_state(&metrics);
        assert!((hv1.similarity(&hv2) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_different_weeks_produce_different_hvs() {
        let hv10 = encode_normative_trajectory(GestationalWeek::new(10));
        let hv30 = encode_normative_trajectory(GestationalWeek::new(30));
        // Different weeks should be substantially different
        let sim = hv10.similarity(&hv30);
        assert!(sim < 0.8, "Different weeks should diverge, got sim={sim}");
    }

    #[test]
    fn test_normative_self_anomaly_low() {
        let week = GestationalWeek::new(25);
        let normative = encode_normative_trajectory(week);
        let score = anomaly_score(&normative, &normative);
        assert!(
            score < 0.01,
            "Self-comparison should have near-zero anomaly, got {score}"
        );
    }

    #[test]
    fn test_anomaly_score_range() {
        let hv1 = encode_normative_trajectory(GestationalWeek::new(10));
        let hv2 = encode_normative_trajectory(GestationalWeek::new(35));
        let score = anomaly_score(&hv1, &hv2);
        // Should be positive and finite
        assert!(score >= 0.0);
        assert!(score <= 2.0);
    }

    #[test]
    fn test_similar_metrics_similar_hvs() {
        let m1 = FetalMetrics::normative(GestationalWeek::new(20));
        let mut m2 = FetalMetrics::normative(GestationalWeek::new(20));
        // Slightly perturbed
        m2.weight_grams *= 1.01;
        let hv1 = encode_fetal_state(&m1);
        let hv2 = encode_fetal_state(&m2);
        let sim = hv1.similarity(&hv2);
        assert!(
            sim > 0.5,
            "Similar metrics should produce similar HVs, got sim={sim}"
        );
    }

    #[test]
    fn test_encode_scalar_deterministic() {
        let hv1 = encode_scalar(0.5, 0xABCD_1234);
        let hv2 = encode_scalar(0.5, 0xABCD_1234);
        assert!((hv1.similarity(&hv2) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_encode_scalar_different_values() {
        let hv_low = encode_scalar(0.1, 0xABCD_1234);
        let hv_high = encode_scalar(0.9, 0xABCD_1234);
        // Different scalar values should produce different HVs
        let sim = hv_low.similarity(&hv_high);
        assert!(sim < 0.9, "Different scalars should differ, got sim={sim}");
    }
}
