// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Reasoning Manager — Reasoning Reliability → Learning Rate & Confidence
//!
//! Consolidates reasoning feedback into a single [`CognitiveSubsystem`] that reads
//! from an immutable [`CycleSnapshot`] and produces [`SubsystemOutput`] proposals.
//!
//! ## Reasoning Processes Modeled
//!
//! 1. **Reliability tracking**: Prediction confidence EMA as reasoning quality (Stanovich 2011)
//! 2. **Epistemic gating**: High epistemic confidence → LR boost (Koriat 2007)
//! 3. **Consciousness gating**: Low Psi → confidence penalty (Baars 2002)
//! 4. **Prediction trend**: Rising confidence → positive affect (Carver & Scheier 1998)
//!
//! ## Design
//!
//! Does NOT replace inline reasoning at `cycle_phase_dynamics.rs:2459-2624`.
//! Proposes changes via SubsystemOutput which the OutputCollector integrates.

use super::super::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput, output_flags,
};

/// Reasoning Manager — consolidates reasoning reliability feedback into LR modulation,
/// confidence gating, and affective trend detection.
///
/// Implements `CognitiveSubsystem` at interval 73 (co-prime).
pub struct ReasoningManager {
    /// EMA of prediction confidence (reasoning reliability proxy)
    reliability_ema: f64,
    /// Previous prediction confidence (for trend detection)
    prev_confidence: f64,
    /// Consecutive cycles of rising confidence
    rising_streak: u32,
    /// Consecutive cycles of falling confidence
    falling_streak: u32,
    /// Cumulative reasoning quality signal
    cumulative_quality: f64,
}

impl Default for ReasoningManager {
    fn default() -> Self {
        Self {
            reliability_ema: 0.5,
            prev_confidence: 0.5,
            rising_streak: 0,
            falling_streak: 0,
            cumulative_quality: 0.0,
        }
    }
}

impl ReasoningManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 73;

    /// EMA smoothing factor for reliability tracking.
    /// Basis: Stanovich (2011) — slow integration of reasoning quality evidence.
    const RELIABILITY_EMA_ALPHA: f64 = 0.08;

    /// Epistemic confidence above this triggers LR boost.
    /// Basis: Koriat (2007) — high feeling-of-knowing supports learning acceleration.
    const HIGH_EPISTEMIC_THRESHOLD: f32 = 0.65;

    /// Unified Psi below this triggers confidence penalty.
    /// Basis: Baars (2002) — reasoning without conscious access is unreliable.
    const PSI_REASONING_THRESHOLD: f64 = 0.15;

    /// Maximum LR boost from reliable reasoning.
    const LR_BOOST_SCALE: f64 = 0.15;

    /// Confidence penalty scale for low-Psi reasoning.
    const CONFIDENCE_PENALTY_SCALE: f64 = 0.02;

    /// Valence shift per trend detection.
    /// Basis: Carver & Scheier (1998) — affect from rate of goal progress.
    const TREND_VALENCE_SCALE: f32 = 0.02;

    /// Decay rate for cumulative quality signal.
    const QUALITY_DECAY: f64 = 0.95;

    /// Current reasoning reliability EMA [0, 1].
    pub fn reliability_ema(&self) -> f64 {
        self.reliability_ema
    }

    /// Cumulative reasoning quality signal.
    pub fn cumulative_quality(&self) -> f64 {
        self.cumulative_quality
    }

    /// Consecutive cycles of rising prediction confidence.
    pub fn rising_streak(&self) -> u32 {
        self.rising_streak
    }

    /// Consecutive cycles of falling prediction confidence.
    pub fn falling_streak(&self) -> u32 {
        self.falling_streak
    }
}

impl CognitiveSubsystem for ReasoningManager {
    fn name(&self) -> &'static str {
        "reasoning_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // ── 1. Update reliability EMA ──────────────────────────────────────
        self.reliability_ema = self.reliability_ema * (1.0 - Self::RELIABILITY_EMA_ALPHA)
            + snapshot.prediction_confidence * Self::RELIABILITY_EMA_ALPHA;

        // ── 2. Cumulative quality with decay ───────────────────────────────
        self.cumulative_quality = self.cumulative_quality * Self::QUALITY_DECAY
            + snapshot.epistemic_confidence as f64 * 0.05;

        // ── 3. Epistemic-gated LR modulation ───────────────────────────────
        // High epistemic confidence → reliable reasoning → boost learning
        if snapshot.epistemic_confidence > Self::HIGH_EPISTEMIC_THRESHOLD {
            let boost = (snapshot.epistemic_confidence - Self::HIGH_EPISTEMIC_THRESHOLD) as f64
                / (1.0 - Self::HIGH_EPISTEMIC_THRESHOLD as f64);
            output.lr_modulation = 1.0 + boost * Self::LR_BOOST_SCALE;
        }

        // ── 4. Psi-gated confidence ────────────────────────────────────────
        // Low consciousness → reasoning is unreliable → penalize confidence
        if snapshot.unified_psi < Self::PSI_REASONING_THRESHOLD {
            let penalty = (Self::PSI_REASONING_THRESHOLD - snapshot.unified_psi)
                / Self::PSI_REASONING_THRESHOLD;
            output.confidence_delta -= penalty * Self::CONFIDENCE_PENALTY_SCALE;
            output.flags |= output_flags::ANOMALY_DETECTED;
        }

        // ── 5. Prediction trend → valence ──────────────────────────────────
        // Rising prediction confidence → positive affect (progress signal)
        let delta = snapshot.prediction_confidence - self.prev_confidence;
        if delta > 0.01 {
            self.rising_streak = self.rising_streak.saturating_add(1);
            self.falling_streak = 0;
            if self.rising_streak >= 3 {
                output.valence_delta += Self::TREND_VALENCE_SCALE;
            }
        } else if delta < -0.01 {
            self.falling_streak = self.falling_streak.saturating_add(1);
            self.rising_streak = 0;
            if self.falling_streak >= 3 {
                output.valence_delta -= Self::TREND_VALENCE_SCALE;
            }
        } else {
            // Stable — reset both streaks
            self.rising_streak = 0;
            self.falling_streak = 0;
        }

        self.prev_confidence = snapshot.prediction_confidence;

        // ── 6. High reliability → suppress exploration ─────────────────────
        // When reasoning is working well, exploit rather than explore
        if self.reliability_ema > 0.7 {
            output.exploration_delta -= (self.reliability_ema - 0.7) * 0.1;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(32);
        data.extend_from_slice(&self.reliability_ema.to_le_bytes());
        data.extend_from_slice(&self.prev_confidence.to_le_bytes());
        data.extend_from_slice(&self.rising_streak.to_le_bytes());
        data.extend_from_slice(&self.falling_streak.to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 24 {
            return Err(format!(
                "ReasoningManager checkpoint too short: {} < 24",
                data.len()
            ));
        }
        self.reliability_ema = f64::from_le_bytes(
            data[0..8]
                .try_into()
                .map_err(|_| "ReasoningManager: corrupt checkpoint bytes [0..8]".to_string())?,
        );
        self.prev_confidence = f64::from_le_bytes(
            data[8..16]
                .try_into()
                .map_err(|_| "ReasoningManager: corrupt checkpoint bytes [8..16]".to_string())?,
        );
        self.rising_streak = u32::from_le_bytes(
            data[16..20]
                .try_into()
                .map_err(|_| "ReasoningManager: corrupt checkpoint bytes [16..20]".to_string())?,
        );
        self.falling_streak = u32::from_le_bytes(
            data[20..24]
                .try_into()
                .map_err(|_| "ReasoningManager: corrupt checkpoint bytes [20..24]".to_string())?,
        );
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::subsystem_trait::CycleSnapshot;

    fn snapshot_with_reasoning(
        prediction_confidence: f64,
        epistemic_confidence: f32,
        unified_psi: f64,
    ) -> CycleSnapshot {
        CycleSnapshot {
            prediction_confidence,
            epistemic_confidence,
            unified_psi,
            ..Default::default()
        }
    }

    #[test]
    fn test_defaults() {
        let rm = ReasoningManager::default();
        assert!((rm.reliability_ema - 0.5).abs() < 1e-10);
        assert!((rm.prev_confidence - 0.5).abs() < 1e-10);
        assert_eq!(rm.rising_streak, 0);
        assert_eq!(rm.falling_streak, 0);
        assert!((rm.cumulative_quality - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_name_and_interval() {
        let rm = ReasoningManager::default();
        assert_eq!(rm.name(), "reasoning_manager");
        assert_eq!(rm.interval(), 73);
    }

    #[test]
    fn test_high_epistemic_boosts_lr() {
        let mut rm = ReasoningManager::default();
        // epistemic_confidence = 0.9 (well above 0.65 threshold)
        let snapshot = snapshot_with_reasoning(0.5, 0.9, 0.5);

        let output = rm.process(&snapshot);
        assert!(
            output.lr_modulation > 1.0,
            "High epistemic confidence should boost LR: {}",
            output.lr_modulation
        );
        // Verify the boost magnitude is reasonable
        assert!(
            output.lr_modulation < 1.0 + ReasoningManager::LR_BOOST_SCALE + 0.01,
            "LR boost should not exceed scale: {}",
            output.lr_modulation
        );
    }

    #[test]
    fn test_low_epistemic_no_lr_boost() {
        let mut rm = ReasoningManager::default();
        // epistemic_confidence = 0.3 (below 0.65 threshold)
        let snapshot = snapshot_with_reasoning(0.5, 0.3, 0.5);

        let output = rm.process(&snapshot);
        assert!(
            (output.lr_modulation - 1.0).abs() < 1e-10,
            "Low epistemic confidence should not boost LR: {}",
            output.lr_modulation
        );
    }

    #[test]
    fn test_low_psi_penalizes_confidence() {
        let mut rm = ReasoningManager::default();
        // unified_psi = 0.05 (well below 0.15 threshold)
        let snapshot = snapshot_with_reasoning(0.5, 0.5, 0.05);

        let output = rm.process(&snapshot);
        assert!(
            output.confidence_delta < 0.0,
            "Low Psi should penalize confidence: {}",
            output.confidence_delta
        );
        assert!(
            output.flags & output_flags::ANOMALY_DETECTED != 0,
            "Low Psi should set ANOMALY_DETECTED flag"
        );
    }

    #[test]
    fn test_adequate_psi_no_penalty() {
        let mut rm = ReasoningManager::default();
        // unified_psi = 0.5 (well above 0.15 threshold)
        let snapshot = snapshot_with_reasoning(0.5, 0.5, 0.5);

        let output = rm.process(&snapshot);
        assert!(
            output.confidence_delta >= 0.0,
            "Adequate Psi should not penalize confidence: {}",
            output.confidence_delta
        );
        assert!(
            output.flags & output_flags::ANOMALY_DETECTED == 0,
            "Adequate Psi should not set ANOMALY_DETECTED flag"
        );
    }

    #[test]
    fn test_rising_trend_positive_valence() {
        let mut rm = ReasoningManager::default();

        // Feed rising prediction confidence over several cycles
        for i in 0..5 {
            let conf = 0.3 + (i as f64) * 0.05;
            let snapshot = snapshot_with_reasoning(conf, 0.5, 0.5);
            rm.process(&snapshot);
        }

        // After 4 rising cycles (streak >= 3), should get positive valence
        let snapshot = snapshot_with_reasoning(0.55, 0.5, 0.5);
        let output = rm.process(&snapshot);
        assert!(
            output.valence_delta > 0.0,
            "Rising confidence trend should produce positive valence: {}",
            output.valence_delta
        );
    }

    #[test]
    fn test_falling_trend_negative_valence() {
        let mut rm = ReasoningManager::default();

        // Feed falling prediction confidence over several cycles
        for i in 0..5 {
            let conf = 0.8 - (i as f64) * 0.05;
            let snapshot = snapshot_with_reasoning(conf, 0.5, 0.5);
            rm.process(&snapshot);
        }

        // After 4 falling cycles (streak >= 3), should get negative valence
        let snapshot = snapshot_with_reasoning(0.50, 0.5, 0.5);
        let output = rm.process(&snapshot);
        assert!(
            output.valence_delta < 0.0,
            "Falling confidence trend should produce negative valence: {}",
            output.valence_delta
        );
    }

    #[test]
    fn test_high_reliability_suppresses_exploration() {
        let mut rm = ReasoningManager::default();
        // Set reliability EMA high by processing many high-confidence snapshots
        rm.reliability_ema = 0.85;

        let snapshot = snapshot_with_reasoning(0.85, 0.5, 0.5);
        let output = rm.process(&snapshot);
        assert!(
            output.exploration_delta < 0.0,
            "High reliability should suppress exploration: {}",
            output.exploration_delta
        );
    }

    #[test]
    fn test_low_reliability_no_exploration_suppression() {
        let mut rm = ReasoningManager::default();
        rm.reliability_ema = 0.4;

        let snapshot = snapshot_with_reasoning(0.4, 0.5, 0.5);
        let output = rm.process(&snapshot);
        // exploration_delta should be >= 0 (no suppression from reliability)
        // It could be negative from other factors, but reliability path should not fire
        assert!(
            output.exploration_delta >= -1e-10,
            "Low reliability should not suppress exploration: {}",
            output.exploration_delta
        );
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let rm = ReasoningManager {
            reliability_ema: 0.73,
            prev_confidence: 0.42,
            rising_streak: 5,
            falling_streak: 0,
            cumulative_quality: 1.23,
        };

        let checkpoint = rm.checkpoint();
        let mut rm2 = ReasoningManager::default();
        rm2.restore(&checkpoint).unwrap();

        assert!((rm2.reliability_ema - 0.73).abs() < 1e-10);
        assert!((rm2.prev_confidence - 0.42).abs() < 1e-10);
        assert_eq!(rm2.rising_streak, 5);
        assert_eq!(rm2.falling_streak, 0);
    }

    #[test]
    fn test_restore_rejects_short_data() {
        let mut rm = ReasoningManager::default();
        let result = rm.restore(&[0u8; 10]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("too short"));
    }

    #[test]
    fn test_reliability_ema_converges() {
        let mut rm = ReasoningManager::default();

        // Feed constant high prediction confidence — EMA should converge toward it
        for _ in 0..200 {
            let snapshot = snapshot_with_reasoning(0.9, 0.5, 0.5);
            rm.process(&snapshot);
        }

        assert!(
            (rm.reliability_ema - 0.9).abs() < 0.05,
            "Reliability EMA should converge toward 0.9: {}",
            rm.reliability_ema
        );
    }

    #[test]
    fn test_cumulative_quality_decays() {
        let mut rm = ReasoningManager::default();

        // Build up quality
        for _ in 0..50 {
            let snapshot = snapshot_with_reasoning(0.5, 0.8, 0.5);
            rm.process(&snapshot);
        }
        let peak = rm.cumulative_quality;
        assert!(peak > 0.0);

        // Feed zero epistemic confidence — quality should decay
        for _ in 0..100 {
            let snapshot = snapshot_with_reasoning(0.5, 0.0, 0.5);
            rm.process(&snapshot);
        }

        assert!(
            rm.cumulative_quality < peak,
            "Cumulative quality should decay: {} < {}",
            rm.cumulative_quality,
            peak
        );
    }

    // ── Property Tests ────────────────────────────────────────────────────
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_reasoning_reliability_converges(
            conf in 0.0f64..1.0
        ) {
            let mut rm = ReasoningManager::default();
            let snap = snapshot_with_reasoning(conf, 0.5, 0.5);
            for _ in 0..200 {
                rm.process(&snap);
            }
            // EMA should converge near the input value
            let diff = (rm.reliability_ema() - conf).abs();
            prop_assert!(diff < 0.15,
                "reliability_ema {} didn't converge toward {}", rm.reliability_ema(), conf);
        }

        #[test]
        fn prop_reasoning_output_bounded(
            conf in 0.0f64..1.0,
            epistemic in 0.0f32..1.0,
            psi in 0.0f64..1.0
        ) {
            let mut rm = ReasoningManager::default();
            let snap = snapshot_with_reasoning(conf, epistemic, psi);
            let output = rm.process(&snap);
            prop_assert!(output.lr_modulation >= 0.5,
                "lr_modulation below 0.5: {}", output.lr_modulation);
            prop_assert!(output.lr_modulation <= 1.5,
                "lr_modulation above 1.5: {}", output.lr_modulation);
            prop_assert!(output.confidence_delta.abs() < 0.1,
                "confidence_delta out of bounds: {}", output.confidence_delta);
            prop_assert!(output.valence_delta.abs() < 0.1,
                "valence_delta out of bounds: {}", output.valence_delta);
        }

        #[test]
        fn prop_reasoning_cumulative_quality_non_negative(
            epistemic in 0.0f32..1.0
        ) {
            let mut rm = ReasoningManager::default();
            let snap = snapshot_with_reasoning(0.5, epistemic, 0.5);
            for _ in 0..100 {
                rm.process(&snap);
            }
            prop_assert!(rm.cumulative_quality() >= 0.0,
                "cumulative_quality negative: {}", rm.cumulative_quality());
        }
    }
}
