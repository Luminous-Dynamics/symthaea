// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Learning Manager — FEP, Dream, School, Evolution
//!
//! Consolidates learning-related logic into a single [`CognitiveSubsystem`] that reads
//! from an immutable [`CycleSnapshot`] and produces [`SubsystemOutput`] proposals.
//!
//! ## Learning Systems Modeled
//!
//! 1. **FEP surprise-driven plasticity**: High surprise → boost LR (Friston 2010)
//! 2. **Dream consolidation**: Low arousal periods → offline replay (Walker 2017)
//! 3. **Curriculum progression**: Skill mastery signals → advance difficulty
//! 4. **Stability-plasticity dilemma**: Balance between new learning and retention
//!
//! ## Design
//!
//! Tracks learning dynamics meta-state (surprise history, plasticity windows).
//! Does NOT own actual learning algorithms — models the meta-level gating.

use super::super::subsystem_trait::{
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput, output_flags,
};
use super::super::thresholds;

/// Learning Manager — meta-level learning rate gating and plasticity control.
///
/// Implements `CognitiveSubsystem` at interval 13 (co-prime).
pub struct LearningManager {
    /// Rolling surprise history (ring buffer, capacity 16)
    surprise_window: [f32; 16],
    surprise_idx: usize,
    surprise_count: usize,
    /// Current plasticity level [0, 1] — how open the system is to learning
    plasticity: f32,
    /// Whether in a "dream" consolidation phase (low arousal sustained)
    in_dream_phase: bool,
    /// Consecutive low-arousal cycles
    low_arousal_streak: u32,
    /// Error trend direction: positive = errors increasing (need more learning)
    error_trend: f32,
    /// Reputation-modulated learning rate modifier (0.5-2.0x).
    /// Updated from ReputationBridge when Mycelix peer reputation data arrives.
    #[cfg(feature = "epistemic")]
    reputation_lr_modifier: f64,
    last_peer_wisdom_hash: [u8; 32],
    federated_wisdom_acc: f32,
}

impl Default for LearningManager {
    fn default() -> Self {
        Self {
            surprise_window: [0.0; 16],
            surprise_idx: 0,
            surprise_count: 0,
            plasticity: 0.5, // moderate initial plasticity
            in_dream_phase: false,
            low_arousal_streak: 0,
            error_trend: 0.0,
            #[cfg(feature = "epistemic")]
            reputation_lr_modifier: 1.0,
            last_peer_wisdom_hash: [0u8; 32],
            federated_wisdom_acc: 0.0, // neutral default
        }
    }
}

impl LearningManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 13;

    /// Arousal threshold below which dream-like consolidation may occur.
    /// Basis: Walker (2017) — NREM sleep facilitates memory integration.
    const DREAM_AROUSAL_THRESHOLD: f32 = 0.2;

    /// Cycles of low arousal before entering dream phase.
    const DREAM_ONSET_CYCLES: u32 = 15;

    /// Plasticity adaptation rate.
    /// Basis: Abraham & Bear (1996) — metaplasticity BCM rule.
    const PLASTICITY_RATE: f32 = 0.05;

    /// Maximum plasticity boost from surprise.
    const MAX_PLASTICITY_VAL: f32 = 0.95;

    /// Minimum plasticity (always retain some learning capacity).
    const MIN_PLASTICITY: f32 = 0.1;

    /// Current plasticity level [0.1, 0.95].
    pub fn plasticity(&self) -> f32 {
        self.plasticity
    }

    /// Whether in dream consolidation phase (low arousal sustained).
    pub fn in_dream_phase(&self) -> bool {
        self.in_dream_phase
    }

    /// Error trend direction: positive = errors increasing.
    pub fn error_trend(&self) -> f32 {
        self.error_trend
    }

    /// Apply a plasticity boost from dream consolidation quality.
    ///
    /// High consolidation reliability signals that offline replay was effective,
    /// priming waking plasticity for enhanced encoding of new experiences.
    ///
    /// Science: Diekelmann & Born (2010) — effective consolidation enhances subsequent
    /// Apply a plasticity boost from dream consolidation quality.
    pub fn apply_dream_consolidation_boost(&mut self, consolidation_reliability: f32) {
        use super::super::thresholds::{
            DREAM_CONSOLIDATION_LR_BOOST, DREAM_CONSOLIDATION_LR_THRESHOLD,
        };
        if consolidation_reliability.is_finite()
            && consolidation_reliability > DREAM_CONSOLIDATION_LR_THRESHOLD as f32
        {
            let boost = (consolidation_reliability - DREAM_CONSOLIDATION_LR_THRESHOLD as f32)
                * DREAM_CONSOLIDATION_LR_BOOST as f32;
            self.plasticity =
                (self.plasticity + boost).clamp(Self::MIN_PLASTICITY, Self::MAX_PLASTICITY_VAL);
        }
    }

    /// Inject peer-shared wisdom into the learning system with epistemic humility.
    ///
    /// Returns true if the wisdom was newly integrated.
    pub fn inject_federated_wisdom(&mut self, peer_id: String, hash: [u8; 32], value: f32) -> bool {
        if hash == self.last_peer_wisdom_hash {
            return false;
        }
        self.last_peer_wisdom_hash = hash;

        // Epistemic Humility: Don't trust peer wisdom blindly.
        // Integrate only a fraction based on trust (here hardcoded humility factor 0.3).
        let humility_factor = 0.3;
        let integrated_value = value * humility_factor;

        self.federated_wisdom_acc += integrated_value;

        // High-value federated wisdom boosts local plasticity to facilitate integration
        let boost = (integrated_value * 0.05).min(0.1);
        self.plasticity = (self.plasticity + boost).min(Self::MAX_PLASTICITY_VAL);

        tracing::debug!(
            "🧪 Epistemic Humility: Integrated {:.2} from {} (raw={:.2})",
            integrated_value,
            peer_id,
            value
        );
        true
    }

    fn mean_surprise(&self) -> f32 {
        if self.surprise_count == 0 {
            return 0.0;
        }
        let sum: f32 = self.surprise_window[..self.surprise_count].iter().sum();
        sum / self.surprise_count as f32
    }

    fn push_surprise(&mut self, surprise: f32) {
        let old = if self.surprise_count == 16 {
            self.surprise_window[self.surprise_idx]
        } else {
            0.0
        };
        self.surprise_window[self.surprise_idx] = surprise;
        self.surprise_idx = (self.surprise_idx + 1) % 16;
        if self.surprise_count < 16 {
            self.surprise_count += 1;
        }
        // Update error trend (online linear regression proxy)
        self.error_trend = self.error_trend * 0.9 + (surprise - old) * 0.1;
    }
}

impl CognitiveSubsystem for LearningManager {
    fn name(&self) -> &'static str {
        "learning_manager"
    }

    fn interval(&self) -> u32 {
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // ── 1. Surprise tracking ──────────────────────────────────────────
        let surprise = snapshot.prediction_error;
        self.push_surprise(surprise);
        let mean_surprise = self.mean_surprise();

        // ── 2. Plasticity dynamics (BCM metaplasticity) ───────────────────
        // High surprise → increase plasticity (learn more)
        // Low surprise → decrease plasticity (stabilize)
        let plasticity_drive = if mean_surprise > thresholds::LEARNING_PLASTICITY_HIGH_SURPRISE {
            Self::PLASTICITY_RATE // increase plasticity
        } else if mean_surprise < thresholds::LEARNING_PLASTICITY_LOW_SURPRISE {
            -Self::PLASTICITY_RATE * 0.5 // decrease plasticity (slower)
        } else {
            0.0 // neutral zone
        };
        self.plasticity = (self.plasticity + plasticity_drive)
            .clamp(Self::MIN_PLASTICITY, Self::MAX_PLASTICITY_VAL);

        // Plasticity → LR modulation
        // High plasticity = higher learning rate
        output.lr_modulation = thresholds::LEARNING_LR_FLOOR
            + self.plasticity as f64 * thresholds::LEARNING_LR_PLASTICITY_SCALE;

        // ── Epistemic: reputation-modulated learning rate ──
        // Blend Mycelix peer reputation into LR: high-reputation peers'
        // knowledge is learned from faster (0.5x-2.0x modifier).
        #[cfg(feature = "epistemic")]
        {
            output.lr_modulation *= self.reputation_lr_modifier;
        }

        // ── 3. Dream consolidation phase ──────────────────────────────────
        if snapshot.arousal < Self::DREAM_AROUSAL_THRESHOLD {
            self.low_arousal_streak += 1;
            if self.low_arousal_streak >= Self::DREAM_ONSET_CYCLES {
                self.in_dream_phase = true;
                // During dream: boost consolidation, suppress exploration
                output.exploration_delta -= thresholds::LEARNING_DREAM_EXPLORATION_DAMPEN;
                output.lr_modulation *= thresholds::LEARNING_DREAM_LR_BOOST;
                output.flags |= output_flags::REQUEST_CONSOLIDATION;
                output.flags |= output_flags::REQUEST_REST;
            }
        } else {
            self.low_arousal_streak = 0;
            self.in_dream_phase = false;
        }

        // ── 4. Error trend → adaptive response ───────────────────────────
        if self.error_trend > 0.05 {
            // Errors increasing → need more exploration
            output.exploration_delta +=
                self.error_trend as f64 * thresholds::LEARNING_ERROR_TREND_EXPLORATION;
            // Also boost plasticity to learn from errors
            output.lr_modulation *= thresholds::LEARNING_ERROR_TREND_LR_BOOST;
        } else if self.error_trend < -0.05 {
            // Errors decreasing → stabilize (things are working)
            output.confidence_delta +=
                (-self.error_trend) as f64 * thresholds::LEARNING_ERROR_TREND_CONFIDENCE;
        }

        // ── 5. Dissipative health → learning gate ─────────────────────────
        // Low dissipative health means the system is stressed, reduce learning
        if snapshot.dissipative_health < thresholds::LEARNING_DISSIPATIVE_HEALTH_THRESHOLD {
            let dampen = 1.0
                - (thresholds::LEARNING_DISSIPATIVE_HEALTH_THRESHOLD - snapshot.dissipative_health)
                    * thresholds::LEARNING_DISSIPATIVE_HEALTH_SENSITIVITY;
            output.lr_modulation *= dampen.max(0.7);
        }

        // ── 6. Somatic stress → plasticity dampening ──────────────────────
        if snapshot.somatic_stress > thresholds::LEARNING_SOMATIC_STRESS_THRESHOLD {
            output.lr_modulation *= 1.0
                - (snapshot.somatic_stress - thresholds::LEARNING_SOMATIC_STRESS_THRESHOLD)
                    * thresholds::LEARNING_SOMATIC_STRESS_SENSITIVITY;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(32);
        data.extend_from_slice(&self.plasticity.to_le_bytes());
        data.extend_from_slice(&self.error_trend.to_le_bytes());
        data.extend_from_slice(&self.low_arousal_streak.to_le_bytes());
        data.push(self.in_dream_phase as u8);
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 13 {
            return Err(format!(
                "LearningManager checkpoint too short: {} < 13",
                data.len()
            ));
        }
        self.plasticity = f32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| "LearningManager: corrupt checkpoint bytes [0..4]".to_string())?,
        );
        self.error_trend = f32::from_le_bytes(
            data[4..8]
                .try_into()
                .map_err(|_| "LearningManager: corrupt checkpoint bytes [4..8]".to_string())?,
        );
        self.low_arousal_streak = u32::from_le_bytes(
            data[8..12]
                .try_into()
                .map_err(|_| "LearningManager: corrupt checkpoint bytes [8..12]".to_string())?,
        );
        self.in_dream_phase = data[12] != 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::subsystem_trait::CycleSnapshot;

    fn snapshot(pe: f32, arousal: f32, diss_health: f32) -> CycleSnapshot {
        CycleSnapshot {
            prediction_error: pe,
            arousal,
            dissipative_health: diss_health as f64,
            prediction_confidence: 0.5,
            coherence: 0.5,
            ..Default::default()
        }
    }

    #[test]
    fn test_defaults() {
        let lm = LearningManager::default();
        assert_eq!(lm.plasticity, 0.5);
        assert!(!lm.in_dream_phase);
    }

    #[test]
    fn test_name_and_interval() {
        let lm = LearningManager::default();
        assert_eq!(lm.name(), "learning_manager");
        assert_eq!(lm.interval(), 13);
    }

    #[test]
    fn test_plasticity_increases_on_surprise() {
        let mut lm = LearningManager::default();
        let surprising = snapshot(0.7, 0.5, 0.8);

        for _ in 0..20 {
            lm.process(&surprising);
        }

        assert!(
            lm.plasticity > 0.5,
            "Plasticity should increase on surprise: {}",
            lm.plasticity
        );
    }

    #[test]
    fn test_plasticity_decreases_on_low_error() {
        let mut lm = LearningManager::default();
        let boring = snapshot(0.02, 0.5, 0.8);

        for _ in 0..30 {
            lm.process(&boring);
        }

        assert!(
            lm.plasticity < 0.5,
            "Plasticity should decrease on low error: {}",
            lm.plasticity
        );
    }

    #[test]
    fn test_dream_phase_on_low_arousal() {
        let mut lm = LearningManager::default();
        let drowsy = snapshot(0.1, 0.1, 0.8);

        for _ in 0..20 {
            lm.process(&drowsy);
        }

        assert!(
            lm.in_dream_phase,
            "Should enter dream phase on sustained low arousal"
        );
    }

    #[test]
    fn test_dream_requests_rest() {
        let mut lm = LearningManager::default();
        let drowsy = snapshot(0.1, 0.1, 0.8);

        for _ in 0..20 {
            lm.process(&drowsy);
        }

        let output = lm.process(&drowsy);
        assert!(output.flags & output_flags::REQUEST_REST != 0);
        assert!(output.flags & output_flags::REQUEST_CONSOLIDATION != 0);
    }

    #[test]
    fn test_low_health_dampens_learning() {
        let mut lm = LearningManager::default();
        let stressed = snapshot(0.3, 0.5, 0.2);

        let output = lm.process(&stressed);
        assert!(
            output.lr_modulation < 1.0,
            "Low health should dampen learning: {}",
            output.lr_modulation
        );
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let lm = LearningManager {
            plasticity: 0.73,
            error_trend: -0.12,
            low_arousal_streak: 8,
            in_dream_phase: true,
            ..Default::default()
        };

        let data = lm.checkpoint();
        let mut lm2 = LearningManager::default();
        lm2.restore(&data).unwrap();

        assert!((lm2.plasticity - 0.73).abs() < 1e-6);
        assert!((lm2.error_trend - (-0.12)).abs() < 1e-6);
        assert_eq!(lm2.low_arousal_streak, 8);
        assert!(lm2.in_dream_phase);
    }
}
