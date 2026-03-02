//! # Perception Manager — Attention, Multi-Modal, Social
//!
//! Consolidates perception-related logic into a single [`CognitiveSubsystem`] that reads
//! from an immutable [`CycleSnapshot`] and produces [`SubsystemOutput`] proposals.
//!
//! ## Perception Systems Modeled
//!
//! 1. **Attention allocation**: Budget-aware processing (Posner 1980)
//! 2. **Multi-modal integration**: Cross-modal binding strength (Damasio 1994)
//! 3. **Social perception**: Theory-of-mind priming (Baron-Cohen 1995)
//! 4. **Perceptual vigilance**: Arousal-dependent sensitivity (Yerkes-Dodson 1908)
//!
//! ## Design
//!
//! Tracks attention budget utilization and perceptual sensitivity as internal state.
//! Proposes arousal/confidence/exploration adjustments based on perceptual quality.

use super::super::subsystem_trait::{
    output_flags, CognitiveSubsystem, CycleSnapshot, SubsystemOutput,
};

/// Perception Manager — attention gating and perceptual quality assessment.
///
/// Implements `CognitiveSubsystem` at interval 19 (co-prime).
pub struct PerceptionManager {
    /// Attention sensitivity [0.5, 2.0] — modulates perceptual thresholds
    attention_sensitivity: f32,
    /// Rolling attention budget utilization (EMA) [0, 1]
    budget_utilization: f32,
    /// Consecutive cycles where budget was exceeded
    budget_exceeded_streak: u32,
    /// Perceptual coherence history (ring buffer, capacity 8)
    coherence_history: [f32; 8],
    coherence_idx: usize,
    coherence_count: usize,
    /// Whether currently in vigilant mode (high attention)
    vigilant: bool,
}

impl Default for PerceptionManager {
    fn default() -> Self {
        Self {
            attention_sensitivity: 1.0,
            budget_utilization: 0.0,
            budget_exceeded_streak: 0,
            coherence_history: [0.0; 8],
            coherence_idx: 0,
            coherence_count: 0,
            vigilant: false,
        }
    }
}

impl PerceptionManager {
    /// Budget utilization threshold for pre-emptive gating.
    /// Basis: Lavie (2005) — perceptual load theory.
    const BUDGET_WARNING_THRESHOLD: f32 = 0.8;

    /// Consecutive budget-exceeded cycles before forced gating.
    const BUDGET_EXCEEDED_LIMIT: u32 = 3;

    /// Arousal level for Yerkes-Dodson optimal performance.
    /// Basis: Yerkes & Dodson (1908) — inverted-U performance curve.
    const OPTIMAL_AROUSAL: f32 = 0.5;

    /// Width of optimal arousal zone (±).
    const AROUSAL_ZONE_WIDTH: f32 = 0.2;

    fn mean_coherence(&self) -> f32 {
        if self.coherence_count == 0 {
            return 0.5;
        }
        let sum: f32 = self.coherence_history[..self.coherence_count].iter().sum();
        sum / self.coherence_count as f32
    }

    fn push_coherence(&mut self, coherence: f32) {
        self.coherence_history[self.coherence_idx] = coherence;
        self.coherence_idx = (self.coherence_idx + 1) % 8;
        if self.coherence_count < 8 {
            self.coherence_count += 1;
        }
    }
}

impl CognitiveSubsystem for PerceptionManager {
    fn name(&self) -> &'static str {
        "perception_manager"
    }

    fn interval(&self) -> u32 {
        19 // co-prime
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // ── 1. Attention budget management ────────────────────────────────
        let budget_frac = if snapshot.attention_budget_exceeded != 0 {
            1.0
        } else {
            0.5
        };
        self.budget_utilization = self.budget_utilization * 0.8 + budget_frac * 0.2;

        if snapshot.attention_budget_exceeded != 0 {
            self.budget_exceeded_streak += 1;
        } else {
            self.budget_exceeded_streak = 0;
        }

        // Pre-emptive gating when budget consistently exceeded
        if self.budget_exceeded_streak >= Self::BUDGET_EXCEEDED_LIMIT {
            // Signal need to shed processing load
            output.exploration_delta -= 0.02; // reduce exploration (expensive)
            output.flags |= output_flags::ESCALATE_URGENCY;
        }

        // Budget warning
        if self.budget_utilization > Self::BUDGET_WARNING_THRESHOLD {
            // Approaching limit — dampen learning rate to reduce computation
            output.lr_modulation *= 0.95;
        }

        // ── 2. Perceptual coherence tracking ──────────────────────────────
        let coherence = snapshot.coherence;
        self.push_coherence(coherence);
        let mean_coh = self.mean_coherence();

        // High coherence → perceptual confidence boost
        if mean_coh > 0.7 {
            output.confidence_delta += (mean_coh - 0.7) as f64 * 0.015;
        }

        // Low coherence → signal exploration (might need different perspective)
        if mean_coh < 0.3 {
            output.exploration_delta += (0.3 - mean_coh) as f64 * 0.02;
            output.flags |= output_flags::REQUEST_EXPLORATION;
        }

        // ── 3. Yerkes-Dodson arousal regulation ───────────────────────────
        // Pull arousal toward optimal zone for peak perceptual performance
        let arousal_distance = snapshot.arousal - Self::OPTIMAL_AROUSAL;
        if arousal_distance.abs() > Self::AROUSAL_ZONE_WIDTH {
            // Outside optimal zone — apply corrective pressure
            let correction = -arousal_distance * 0.03;
            output.arousal_delta += correction;
        }

        // ── 4. Vigilance mode ─────────────────────────────────────────────
        // Enter vigilant mode on low coherence + high prediction error
        let should_be_vigilant = coherence < 0.4 && snapshot.prediction_error > 0.4;
        if should_be_vigilant && !self.vigilant {
            self.vigilant = true;
            self.attention_sensitivity = (self.attention_sensitivity * 1.2).min(2.0);
        } else if !should_be_vigilant && self.vigilant {
            self.vigilant = false;
            self.attention_sensitivity = (self.attention_sensitivity * 0.9).max(0.5);
        }

        // ── 5. Phenomenal binding → confidence modulation ─────────────────
        let binding = snapshot.phenomenal_binding;
        if binding > 0.7 {
            output.confidence_delta += (binding - 0.7) * 0.01;
        } else if binding < 0.3 {
            output.confidence_delta -= (0.3 - binding) * 0.01;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(16);
        data.extend_from_slice(&self.attention_sensitivity.to_le_bytes());
        data.extend_from_slice(&self.budget_utilization.to_le_bytes());
        data.extend_from_slice(&self.budget_exceeded_streak.to_le_bytes());
        data.push(self.vigilant as u8);
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() < 13 {
            return Err(format!(
                "PerceptionManager checkpoint too short: {} < 13",
                data.len()
            ));
        }
        self.attention_sensitivity = f32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| "PerceptionManager: corrupt checkpoint bytes [0..4]".to_string())?,
        );
        self.budget_utilization = f32::from_le_bytes(
            data[4..8]
                .try_into()
                .map_err(|_| "PerceptionManager: corrupt checkpoint bytes [4..8]".to_string())?,
        );
        self.budget_exceeded_streak = u32::from_le_bytes(
            data[8..12]
                .try_into()
                .map_err(|_| "PerceptionManager: corrupt checkpoint bytes [8..12]".to_string())?,
        );
        self.vigilant = data[12] != 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_loop::subsystem_trait::CycleSnapshot;

    fn snapshot(coherence: f32, pe: f32, arousal: f32, budget_exceeded: bool) -> CycleSnapshot {
        CycleSnapshot {
            coherence,
            prediction_error: pe,
            arousal,
            attention_budget_exceeded: u8::from(budget_exceeded),
            prediction_confidence: 0.5,
            ..Default::default()
        }
    }

    #[test]
    fn test_defaults() {
        let pm = PerceptionManager::default();
        assert_eq!(pm.attention_sensitivity, 1.0);
        assert!(!pm.vigilant);
    }

    #[test]
    fn test_name_and_interval() {
        let pm = PerceptionManager::default();
        assert_eq!(pm.name(), "perception_manager");
        assert_eq!(pm.interval(), 19);
    }

    #[test]
    fn test_budget_exceeded_escalates() {
        let mut pm = PerceptionManager::default();
        let exceeded = snapshot(0.5, 0.3, 0.5, true);

        for _ in 0..5 {
            pm.process(&exceeded);
        }

        let output = pm.process(&exceeded);
        assert!(
            output.flags & output_flags::ESCALATE_URGENCY != 0,
            "Should escalate urgency on sustained budget exceeded"
        );
    }

    #[test]
    fn test_high_coherence_boosts_confidence() {
        let mut pm = PerceptionManager::default();
        let coherent = snapshot(0.9, 0.1, 0.5, false);

        for _ in 0..10 {
            pm.process(&coherent);
        }

        let output = pm.process(&coherent);
        assert!(
            output.confidence_delta > 0.0,
            "High coherence should boost confidence: {}",
            output.confidence_delta
        );
    }

    #[test]
    fn test_low_coherence_drives_exploration() {
        let mut pm = PerceptionManager::default();
        let incoherent = snapshot(0.1, 0.3, 0.5, false);

        for _ in 0..10 {
            pm.process(&incoherent);
        }

        let output = pm.process(&incoherent);
        assert!(
            output.exploration_delta > 0.0,
            "Low coherence should drive exploration: {}",
            output.exploration_delta
        );
    }

    #[test]
    fn test_yerkes_dodson_high_arousal() {
        let mut pm = PerceptionManager::default();
        let high_arousal = snapshot(0.5, 0.3, 0.9, false);

        let output = pm.process(&high_arousal);
        assert!(
            output.arousal_delta < 0.0,
            "High arousal should be pulled down: {}",
            output.arousal_delta
        );
    }

    #[test]
    fn test_yerkes_dodson_low_arousal() {
        let mut pm = PerceptionManager::default();
        let low_arousal = snapshot(0.5, 0.3, 0.1, false);

        let output = pm.process(&low_arousal);
        assert!(
            output.arousal_delta > 0.0,
            "Low arousal should be pulled up: {}",
            output.arousal_delta
        );
    }

    #[test]
    fn test_vigilance_activates() {
        let mut pm = PerceptionManager::default();
        let alarming = snapshot(0.2, 0.7, 0.5, false);

        pm.process(&alarming);
        assert!(
            pm.vigilant,
            "Should enter vigilant mode on low coherence + high error"
        );
        assert!(
            pm.attention_sensitivity > 1.0,
            "Sensitivity should increase in vigilance: {}",
            pm.attention_sensitivity
        );
    }

    #[test]
    fn test_checkpoint_roundtrip() {
        let pm = PerceptionManager {
            attention_sensitivity: 1.5,
            budget_utilization: 0.7,
            budget_exceeded_streak: 3,
            vigilant: true,
            ..Default::default()
        };

        let data = pm.checkpoint();
        let mut pm2 = PerceptionManager::default();
        pm2.restore(&data).unwrap();

        assert!((pm2.attention_sensitivity - 1.5).abs() < 1e-6);
        assert!((pm2.budget_utilization - 0.7).abs() < 1e-6);
        assert_eq!(pm2.budget_exceeded_streak, 3);
        assert!(pm2.vigilant);
    }
}
