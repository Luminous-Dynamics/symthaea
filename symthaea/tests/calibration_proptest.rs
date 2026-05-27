// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/*!
Property-Based Tests for Calibration System Invariants

Verifies that the calibration pipeline maintains safety invariants under
arbitrary z-score inputs: bounded factors, no panics, finite outputs,
merge stability, sleep quality monotonicity, and JSON roundtrip fidelity.
*/

use proptest::prelude::*;
use symthaea::cognitive_loop::calibration::{NeuromodCalibration, SharedCalibrationProfile};

// ═══════════════════════════════════════════════════════════════════════════════
// Strategies
// ═══════════════════════════════════════════════════════════════════════════════

/// Finite z-score in [-10, 10].
fn z_score() -> impl Strategy<Value = f64> {
    (-10.0..=10.0_f64).prop_filter("finite", |z| z.is_finite())
}

/// One of the 15 recognized benchmark names.
fn benchmark() -> impl Strategy<Value = &'static str> {
    prop_oneof![
        Just("Executive::Stroop"),
        Just("Executive::Flanker"),
        Just("WorM::N-back"),
        Just("WorM::DigitSpan"),
        Just("Inhibition::StopSignal"),
        Just("Inhibition::GoNoGo"),
        Just("SustainedAttention::CPT"),
        Just("SustainedAttention::SART"),
        Just("SustainedAttention::PVT"),
        Just("Executive::DualTask"),
        Just("Social::UltimatumGame"),
        Just("Social::RME"),
        Just("PVT::lapse_rate"),
        Just("Allostatic::Endocannabinoid"),
        Just("Metacognition::FeelingOfKnowing"),
    ]
}

/// Vector of 1..15 (benchmark, z-score) pairs.
fn z_pairs(min: usize, max: usize) -> impl Strategy<Value = Vec<(&'static str, f64)>> {
    prop::collection::vec((benchmark(), z_score()), min..=max)
}

// ═══════════════════════════════════════════════════════════════════════════════
// Properties
// ═══════════════════════════════════════════════════════════════════════════════

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// All sensitivity factors must lie within [0.85, 1.15] for any z-score input.
    #[test]
    fn prop_factors_always_bounded(pairs in z_pairs(1, 15)) {
        let cal = NeuromodCalibration::from_z_scores(&pairs);
        for adj in &cal.adjustments {
            prop_assert!(
                adj.sensitivity_factor >= 0.85 && adj.sensitivity_factor <= 1.15,
                "{} factor {} not in [0.85, 1.15]",
                adj.transmitter, adj.sensitivity_factor
            );
        }
    }

    /// Confidence delta must lie within [-0.015, 0.015] for any z-score input.
    #[test]
    fn prop_confidence_delta_bounded(pairs in z_pairs(1, 15)) {
        let cal = NeuromodCalibration::from_z_scores(&pairs);
        prop_assert!(
            cal.confidence_delta >= -0.015 && cal.confidence_delta <= 0.015,
            "confidence_delta {} not in [-0.015, 0.015]",
            cal.confidence_delta
        );
    }

    /// from_z_scores never panics and all outputs are finite.
    #[test]
    fn prop_from_z_scores_never_panics(pairs in z_pairs(1, 15)) {
        let cal = NeuromodCalibration::from_z_scores(&pairs);
        for adj in &cal.adjustments {
            prop_assert!(adj.sensitivity_factor.is_finite(),
                "{} factor is not finite", adj.transmitter);
            prop_assert!(adj.z_score.is_finite(),
                "{} z_score is not finite", adj.transmitter);
        }
        prop_assert!(cal.confidence_delta.is_finite());
        prop_assert!(cal.coverage.is_finite());
    }

    /// Merging peer calibrations keeps factors finite and within a reasonable range.
    #[test]
    fn prop_merge_preserves_reasonable_range(
        pairs in z_pairs(1, 10),
        peer_pairs in z_pairs(1, 10),
        weight in 0.0..=1.0_f32,
    ) {
        let mut cal = NeuromodCalibration::from_z_scores(&pairs);
        let peer_cal = NeuromodCalibration::from_z_scores(&peer_pairs);
        let peer_profile = peer_cal.to_shareable(Some("peer".into()));
        cal.merge_peer_calibration(&peer_profile, weight);
        for adj in &cal.adjustments {
            prop_assert!(adj.sensitivity_factor.is_finite(),
                "{} factor not finite after merge", adj.transmitter);
            prop_assert!(
                adj.sensitivity_factor >= 0.5 && adj.sensitivity_factor <= 2.0,
                "{} factor {} outside [0.5, 2.0] after merge",
                adj.transmitter, adj.sensitivity_factor
            );
        }
        prop_assert!(cal.confidence_delta.is_finite());
    }

    /// Scaling by sleep quality moves factors toward neutral (1.0).
    #[test]
    fn prop_sleep_quality_moves_toward_neutral(
        pairs in z_pairs(1, 10),
        quality in 0.0..=1.0_f32,
    ) {
        let cal_full = NeuromodCalibration::from_z_scores(&pairs);
        let mut cal_scaled = cal_full.clone();
        cal_scaled.scale_by_sleep_quality(quality);
        for (full, scaled) in cal_full.adjustments.iter().zip(cal_scaled.adjustments.iter()) {
            let full_dev = (full.sensitivity_factor - 1.0).abs();
            let scaled_dev = (scaled.sensitivity_factor - 1.0).abs();
            prop_assert!(
                scaled_dev <= full_dev + 0.0001,
                "{}: scaled deviation {} > full deviation {} (quality={})",
                full.transmitter, scaled_dev, full_dev, quality
            );
        }
    }

    /// to_shareable → serialize → deserialize produces finite factors.
    #[test]
    fn prop_json_roundtrip_finite(pairs in z_pairs(1, 15)) {
        let cal = NeuromodCalibration::from_z_scores(&pairs);
        let shareable = cal.to_shareable(Some("rt".into()));
        let json = serde_json::to_string(&shareable).expect("serialize");
        let deser: SharedCalibrationProfile = serde_json::from_str(&json).expect("deserialize");
        for (_, &factor) in &deser.sensitivity_factors {
            prop_assert!(factor.is_finite(), "deserialized factor not finite: {factor}");
        }
        prop_assert!(deser.confidence_delta.is_finite());
        prop_assert!(deser.coverage.is_finite());
    }
}