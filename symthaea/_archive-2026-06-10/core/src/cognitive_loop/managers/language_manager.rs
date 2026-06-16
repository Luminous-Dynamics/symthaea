// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Language Manager — Broca Quality Feedback → Confidence & Learning
//!
//! Consolidates language generation feedback into a single [`CognitiveSubsystem`]
//! that reads from an immutable [`CycleSnapshot`] and produces [`SubsystemOutput`] proposals.
//!
//! ## Language Processing Modeled
//!
//! 1. **Generation quality**: Epistemic confidence EMA as quality signal (Clark 2013)
//! 2. **Coherence gating**: High coherence + quality → confidence boost (Hagoort 2005)
//! 3. **Consciousness cadence**: Low Psi → dampen LR for language (Dehaene 2014)
//! 4. **Fluency monitoring**: Sustained low coherence → consolidation request (Levelt 1989)
//!
//! ## Design
//!
//! Does NOT call BrocaManager.generate() directly (that stays inline in cycle).
//! Proposes changes via SubsystemOutput which the OutputCollector integrates.

use super::super::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput, output_flags,
};

/// Language Manager — consolidates Broca quality feedback into confidence and LR proposals.
///
/// Implements `CognitiveSubsystem` at interval 61 (co-prime).
pub struct LanguageManager {
    /// EMA of epistemic confidence (generation quality proxy)
    quality_ema: f32,
    /// EMA of coherence (fluency proxy)
    coherence_ema: f32,
    /// Consecutive cycles of low coherence
    low_coherence_streak: u32,
    /// Consecutive cycles of high quality generation
    high_quality_streak: u32,
}

impl Default for LanguageManager {
    fn default() -> Self {
        Self {
            quality_ema: 0.5,
            coherence_ema: 0.5,
            low_coherence_streak: 0,
            high_quality_streak: 0,
        }
    }
}

impl LanguageManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 61;

    /// EMA smoothing factor for generation quality tracking.
    /// Basis: Clark (2013) — predictive processing temporal integration.
    const QUALITY_EMA_ALPHA: f32 = 0.1;

    /// EMA smoothing factor for coherence (fluency) tracking.
    const COHERENCE_EMA_ALPHA: f32 = 0.12;

    /// Coherence below this indicates fluency degradation.
    /// Basis: Levelt (1989) — speech production monitoring threshold.
    const LOW_COHERENCE_THRESHOLD: f32 = 0.3;

    /// Quality EMA above this counts as high-quality generation.
    const HIGH_QUALITY_THRESHOLD: f32 = 0.7;

    /// Unified Psi below this dampens language learning rate.
    /// Basis: Dehaene (2014) — consciousness as global workspace access gate.
    const PSI_CADENCE_THRESHOLD: f64 = 0.2;

    /// Per-cycle confidence boost scale from high quality generation.
    /// Basis: Hagoort (2005) — MUC (Memory, Unification, Control) confidence.
    const QUALITY_CONFIDENCE_SCALE: f64 = 0.03;

    /// Cycles of sustained low coherence before requesting consolidation.
    /// Basis: Levelt (1989) — speech monitoring error accumulation.
    const FLUENCY_DEGRADATION_CYCLES: u32 = 10;

    /// Minimum LR multiplier when Psi is below cadence threshold.
    const CONSCIOUSNESS_LR_FLOOR: f64 = 0.7;

    /// Current generation quality EMA [0, 1].
    pub fn quality_ema(&self) -> f32 {
        self.quality_ema
    }

    /// Current coherence (fluency) EMA [0, 1].
    pub fn coherence_ema(&self) -> f32 {
        self.coherence_ema
    }

    /// Consecutive low-coherence cycles (fluency degradation indicator).
    pub fn low_coherence_streak(&self) -> u32 {
        self.low_coherence_streak
    }
}

impl CognitiveSubsystem for LanguageManager {
    fn name(&self) -> &'static str {
        "language_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // ── Update EMAs ─────────────────────────────────────────────────
        self.quality_ema = self.quality_ema * (1.0 - Self::QUALITY_EMA_ALPHA)
            + snapshot.epistemic_confidence * Self::QUALITY_EMA_ALPHA;
        self.coherence_ema = self.coherence_ema * (1.0 - Self::COHERENCE_EMA_ALPHA)
            + snapshot.coherence * Self::COHERENCE_EMA_ALPHA;

        // ── 1. Generation quality → confidence ──────────────────────────
        if self.quality_ema > Self::HIGH_QUALITY_THRESHOLD {
            self.high_quality_streak = self.high_quality_streak.saturating_add(1);
            output.confidence_delta += self.quality_ema as f64 * Self::QUALITY_CONFIDENCE_SCALE;
        } else {
            self.high_quality_streak = 0;
            // Below-average quality → mild negative
            if self.quality_ema < 0.4 {
                output.confidence_delta -= 0.01;
            }
        }

        // ── 2. Coherence gating (fluency monitoring) ────────────────────
        if self.coherence_ema < Self::LOW_COHERENCE_THRESHOLD {
            self.low_coherence_streak = self.low_coherence_streak.saturating_add(1);
            if self.low_coherence_streak >= Self::FLUENCY_DEGRADATION_CYCLES {
                output.flags |= output_flags::REQUEST_CONSOLIDATION;
            }
        } else {
            self.low_coherence_streak = 0;
        }

        // ── 3. Consciousness cadence: low Psi → dampen LR ──────────────
        if snapshot.unified_psi < Self::PSI_CADENCE_THRESHOLD {
            let dampen = Self::CONSCIOUSNESS_LR_FLOOR
                + (1.0 - Self::CONSCIOUSNESS_LR_FLOOR)
                    * (snapshot.unified_psi / Self::PSI_CADENCE_THRESHOLD);
            output.lr_modulation = dampen;
        }

        // ── 4. High quality + high coherence → positive valence ─────────
        if self.quality_ema > 0.6 && self.coherence_ema > 0.6 {
            output.valence_delta += 0.01;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        // Format: [quality_ema:f32][coherence_ema:f32][low_coherence_streak:u32][high_quality_streak:u32]
        let mut data = Vec::with_capacity(16);
        data.extend_from_slice(&self.quality_ema.to_le_bytes());
        data.extend_from_slice(&self.coherence_ema.to_le_bytes());
        data.extend_from_slice(&self.low_coherence_streak.to_le_bytes());
        data.extend_from_slice(&self.high_quality_streak.to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 16 {
            return Err(format!(
                "LanguageManager checkpoint too short: {} < 16",
                data.len()
            ));
        }
        self.quality_ema = f32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| "LanguageManager: corrupt checkpoint bytes [0..4]".to_string())?,
        );
        self.coherence_ema = f32::from_le_bytes(
            data[4..8]
                .try_into()
                .map_err(|_| "LanguageManager: corrupt checkpoint bytes [4..8]".to_string())?,
        );
        self.low_coherence_streak = u32::from_le_bytes(
            data[8..12]
                .try_into()
                .map_err(|_| "LanguageManager: corrupt checkpoint bytes [8..12]".to_string())?,
        );
        self.high_quality_streak = u32::from_le_bytes(
            data[12..16]
                .try_into()
                .map_err(|_| "LanguageManager: corrupt checkpoint bytes [12..16]".to_string())?,
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

    fn snapshot_with(epistemic: f32, coherence: f32, psi: f64) -> CycleSnapshot {
        CycleSnapshot {
            epistemic_confidence: epistemic,
            coherence,
            unified_psi: psi,
            ..Default::default()
        }
    }

    #[test]
    fn test_defaults() {
        let lm = LanguageManager::default();
        assert_eq!(lm.quality_ema, 0.5);
        assert_eq!(lm.coherence_ema, 0.5);
        assert_eq!(lm.low_coherence_streak, 0);
        assert_eq!(lm.high_quality_streak, 0);
    }

    #[test]
    fn test_name_and_interval() {
        let lm = LanguageManager::default();
        assert_eq!(lm.name(), "language_manager");
        assert_eq!(lm.interval(), 61);
    }

    #[test]
    fn test_high_quality_boosts_confidence() {
        let mut lm = LanguageManager::default();
        // Drive quality EMA above HIGH_QUALITY_THRESHOLD (0.7)
        let high_quality = snapshot_with(0.95, 0.8, 0.5);
        for _ in 0..50 {
            lm.process(&high_quality);
        }
        assert!(
            lm.quality_ema > LanguageManager::HIGH_QUALITY_THRESHOLD,
            "Quality EMA should exceed threshold: {}",
            lm.quality_ema
        );

        let output = lm.process(&high_quality);
        assert!(
            output.confidence_delta > 0.0,
            "High quality should boost confidence: {}",
            output.confidence_delta
        );
    }

    #[test]
    fn test_low_coherence_requests_consolidation() {
        let mut lm = LanguageManager::default();
        let low_coherence = snapshot_with(0.5, 0.1, 0.5);

        // Need FLUENCY_DEGRADATION_CYCLES (10) to trigger consolidation,
        // plus enough cycles for EMA to drop below LOW_COHERENCE_THRESHOLD (0.3)
        for _ in 0..30 {
            lm.process(&low_coherence);
        }

        assert!(
            lm.coherence_ema < LanguageManager::LOW_COHERENCE_THRESHOLD,
            "Coherence EMA should be below threshold: {}",
            lm.coherence_ema
        );

        let output = lm.process(&low_coherence);
        assert!(
            output.flags & output_flags::REQUEST_CONSOLIDATION != 0,
            "Sustained low coherence should request consolidation"
        );
    }

    #[test]
    fn test_low_psi_dampens_lr() {
        let mut lm = LanguageManager::default();
        let low_psi = snapshot_with(0.5, 0.5, 0.05);

        let output = lm.process(&low_psi);
        assert!(
            output.lr_modulation < 1.0,
            "Low Psi should dampen LR: {}",
            output.lr_modulation
        );
        assert!(
            output.lr_modulation >= LanguageManager::CONSCIOUSNESS_LR_FLOOR,
            "LR should not go below floor: {} >= {}",
            output.lr_modulation,
            LanguageManager::CONSCIOUSNESS_LR_FLOOR
        );
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let lm = LanguageManager {
            quality_ema: 0.73,
            coherence_ema: 0.42,
            low_coherence_streak: 5,
            high_quality_streak: 12,
        };

        let checkpoint = lm.checkpoint();
        let mut lm2 = LanguageManager::default();
        lm2.restore(&checkpoint).unwrap();

        assert!((lm2.quality_ema - 0.73).abs() < 1e-6);
        assert!((lm2.coherence_ema - 0.42).abs() < 1e-6);
        assert_eq!(lm2.low_coherence_streak, 5);
        assert_eq!(lm2.high_quality_streak, 12);
    }

    #[test]
    fn test_restore_rejects_short_data() {
        let mut lm = LanguageManager::default();
        let result = lm.restore(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_quality_coherence_positive_valence() {
        let mut lm = LanguageManager::default();
        // Push both EMAs above 0.6
        let good = snapshot_with(0.9, 0.9, 0.5);
        for _ in 0..50 {
            lm.process(&good);
        }

        let output = lm.process(&good);
        assert!(
            output.valence_delta > 0.0,
            "High quality + coherence should produce positive valence: {}",
            output.valence_delta
        );
    }

    #[test]
    fn test_low_quality_widens_cadence() {
        use crate::cognitive_loop::thresholds::BROCA_QUALITY_CADENCE_THRESHOLD;

        let mut lm = LanguageManager::default();
        // Drive quality EMA well below cadence threshold (0.4)
        let low_quality = snapshot_with(0.1, 0.5, 0.5);
        for _ in 0..80 {
            lm.process(&low_quality);
        }
        assert!(
            lm.quality_ema() < BROCA_QUALITY_CADENCE_THRESHOLD,
            "quality_ema should be below cadence threshold: {} < {}",
            lm.quality_ema(),
            BROCA_QUALITY_CADENCE_THRESHOLD
        );

        // Recover quality above threshold
        let high_quality = snapshot_with(0.9, 0.5, 0.5);
        for _ in 0..80 {
            lm.process(&high_quality);
        }
        assert!(
            lm.quality_ema() >= BROCA_QUALITY_CADENCE_THRESHOLD,
            "quality_ema should recover above cadence threshold: {} >= {}",
            lm.quality_ema(),
            BROCA_QUALITY_CADENCE_THRESHOLD
        );
    }

    #[test]
    fn test_coherence_streak_resets_on_recovery() {
        let mut lm = LanguageManager::default();
        // Build low coherence streak
        let low = snapshot_with(0.5, 0.1, 0.5);
        for _ in 0..20 {
            lm.process(&low);
        }
        assert!(lm.low_coherence_streak > 0);

        // Recover coherence — enough cycles to push EMA above threshold
        let high = snapshot_with(0.5, 0.9, 0.5);
        for _ in 0..50 {
            lm.process(&high);
        }
        assert_eq!(
            lm.low_coherence_streak, 0,
            "Streak should reset when coherence recovers"
        );
    }

    // ── Property Tests ────────────────────────────────────────────────────
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn prop_language_lr_floor_holds(
            psi in 0.0f64..0.2
        ) {
            let mut lm = LanguageManager::default();
            let snap = snapshot_with(0.5, 0.5, psi);
            let output = lm.process(&snap);
            prop_assert!(output.lr_modulation >= LanguageManager::CONSCIOUSNESS_LR_FLOOR,
                "LR floor violated: {} for psi={}", output.lr_modulation, psi);
        }

        #[test]
        fn prop_language_quality_ema_bounded(
            epistemic in 0.0f32..1.0,
            coherence in 0.0f32..1.0
        ) {
            let mut lm = LanguageManager::default();
            let snap = snapshot_with(epistemic, coherence, 0.5);
            for _ in 0..100 {
                lm.process(&snap);
            }
            prop_assert!(lm.quality_ema() >= 0.0,
                "quality_ema below 0: {}", lm.quality_ema());
            prop_assert!(lm.quality_ema() <= 1.0,
                "quality_ema above 1: {}", lm.quality_ema());
            prop_assert!(lm.coherence_ema() >= 0.0,
                "coherence_ema below 0: {}", lm.coherence_ema());
            prop_assert!(lm.coherence_ema() <= 1.0,
                "coherence_ema above 1: {}", lm.coherence_ema());
        }

        #[test]
        fn prop_language_output_bounded(
            epistemic in 0.0f32..1.0,
            coherence in 0.0f32..1.0,
            psi in 0.0f64..1.0
        ) {
            let mut lm = LanguageManager::default();
            let snap = snapshot_with(epistemic, coherence, psi);
            let output = lm.process(&snap);
            prop_assert!(output.confidence_delta.abs() < 0.1,
                "confidence_delta out of bounds: {}", output.confidence_delta);
            prop_assert!(output.lr_modulation >= 0.5,
                "lr_modulation below 0.5: {}", output.lr_modulation);
            prop_assert!(output.lr_modulation <= 1.5,
                "lr_modulation above 1.5: {}", output.lr_modulation);
        }
    }
}
