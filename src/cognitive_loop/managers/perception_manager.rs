// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
    CognitiveSubsystem, CycleSnapshot, SubsystemOutput, output_flags,
};
use super::super::thresholds::{
    self, PERCEPTION_BINDING_HIGH, PERCEPTION_BINDING_LOW, PERCEPTION_COHERENCE_HIGH,
    PERCEPTION_COHERENCE_LOW, PERCEPTION_SENSITIVITY_MAX, PERCEPTION_SENSITIVITY_MIN,
    PERCEPTION_VIGILANCE_COHERENCE, PERCEPTION_VIGILANCE_PE,
};
use crate::cognitive_loop::types::PredictionEvidenceCursor;

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
    /// Prevents a repeated or stale PE sample from toggling vigilance twice.
    prediction_error_cursor: PredictionEvidenceCursor,
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
            prediction_error_cursor: PredictionEvidenceCursor::default(),
        }
    }
}

impl PerceptionManager {
    /// Co-prime scheduling interval (cycles).
    pub const INTERVAL: u32 = 19;

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

    /// Current attention sensitivity level [0.5, 2.0].
    pub fn attention_sensitivity(&self) -> f32 {
        self.attention_sensitivity
    }

    /// Rolling attention budget utilization (EMA) [0, 1].
    pub fn budget_utilization(&self) -> f32 {
        self.budget_utilization
    }

    /// Whether currently in vigilant mode (high attention).
    pub fn is_vigilant(&self) -> bool {
        self.vigilant
    }

    /// Mean perceptual coherence from rolling history.
    pub fn mean_coherence_score(&self) -> f32 {
        self.mean_coherence()
    }

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
        Self::INTERVAL
    }

    fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        let mut output = SubsystemOutput::NEUTRAL;

        // ── 1. Attention budget management ────────────────────────────────
        let budget_frac = if snapshot.attention_budget_exceeded != 0 {
            1.0
        } else {
            0.5
        };
        self.budget_utilization = self.budget_utilization * thresholds::PERCEPTION_BUDGET_EMA_DECAY
            + budget_frac * thresholds::PERCEPTION_BUDGET_EMA_NEW;

        if snapshot.attention_budget_exceeded != 0 {
            self.budget_exceeded_streak += 1;
        } else {
            self.budget_exceeded_streak = 0;
        }

        // Pre-emptive gating when budget consistently exceeded
        if self.budget_exceeded_streak >= Self::BUDGET_EXCEEDED_LIMIT {
            // Signal need to shed processing load
            output.exploration_delta -= thresholds::PERCEPTION_BUDGET_EXPLORATION_DAMPEN;
            output.flags |= output_flags::ESCALATE_URGENCY;
        }

        // Budget warning
        if self.budget_utilization > Self::BUDGET_WARNING_THRESHOLD {
            // Approaching limit — dampen learning rate to reduce computation
            output.lr_modulation *= thresholds::PERCEPTION_BUDGET_LR_DAMPEN;
        }

        // ── 2. Perceptual coherence tracking ──────────────────────────────
        let coherence = snapshot.coherence;
        self.push_coherence(coherence);
        let mean_coh = self.mean_coherence();

        // High coherence → perceptual confidence boost
        if mean_coh > PERCEPTION_COHERENCE_HIGH {
            output.confidence_delta += f64::from(mean_coh - PERCEPTION_COHERENCE_HIGH)
                * thresholds::PERCEPTION_COHERENCE_CONFIDENCE_SCALE;
        }

        // Low coherence → signal exploration (might need different perspective)
        if mean_coh < PERCEPTION_COHERENCE_LOW {
            output.exploration_delta += f64::from(PERCEPTION_COHERENCE_LOW - mean_coh)
                * thresholds::PERCEPTION_COHERENCE_EXPLORATION_SCALE;
            output.flags |= output_flags::REQUEST_EXPLORATION;
        }

        // ── 3. Yerkes-Dodson arousal regulation ───────────────────────────
        // Pull arousal toward optimal zone for peak perceptual performance
        let arousal_distance = snapshot.arousal - Self::OPTIMAL_AROUSAL;
        if arousal_distance.abs() > Self::AROUSAL_ZONE_WIDTH {
            // Outside optimal zone — apply corrective pressure
            let correction = -arousal_distance * thresholds::PERCEPTION_AROUSAL_CORRECTION_GAIN;
            output.arousal_delta += correction;
        }

        // ── 4. Vigilance mode ─────────────────────────────────────────────
        // Enter vigilant mode on low coherence + high prediction error
        let prediction_error_sample = self
            .prediction_error_cursor
            .admit_evidence(&snapshot.prediction_error_evidence(), snapshot.cycle_number);
        if let Some(prediction_error) = prediction_error_sample {
            let should_be_vigilant = coherence < PERCEPTION_VIGILANCE_COHERENCE
                && prediction_error > PERCEPTION_VIGILANCE_PE;
            if should_be_vigilant && !self.vigilant {
                self.vigilant = true;
                self.attention_sensitivity = (self.attention_sensitivity
                    * thresholds::PERCEPTION_VIGILANCE_AMPLIFY)
                    .min(PERCEPTION_SENSITIVITY_MAX);
            } else if !should_be_vigilant && self.vigilant {
                self.vigilant = false;
                self.attention_sensitivity = (self.attention_sensitivity
                    * thresholds::PERCEPTION_VIGILANCE_RECOVERY)
                    .max(PERCEPTION_SENSITIVITY_MIN);
            }
        }

        // ── 5. Phenomenal binding → confidence modulation ─────────────────
        let binding = snapshot.phenomenal_binding;
        let high = f64::from(PERCEPTION_BINDING_HIGH);
        let low = f64::from(PERCEPTION_BINDING_LOW);
        if binding > high {
            output.confidence_delta +=
                (binding - high) * thresholds::PERCEPTION_BINDING_CONFIDENCE_SCALE;
        } else if binding < low {
            output.confidence_delta -=
                (low - binding) * thresholds::PERCEPTION_BINDING_CONFIDENCE_SCALE;
        }

        output
    }

    fn checkpoint(&self) -> Vec<u8> {
        // Preserve the legacy 13-byte state prefix and append PEVC v2 watermarks.
        let mut data = Vec::with_capacity(36);
        data.extend_from_slice(&self.attention_sensitivity.to_le_bytes());
        data.extend_from_slice(&self.budget_utilization.to_le_bytes());
        data.extend_from_slice(&self.budget_exceeded_streak.to_le_bytes());
        data.push(self.vigilant as u8);
        data.extend_from_slice(b"PEVC");
        data.push(2); // extension version
        let observed_cycle = self
            .prediction_error_cursor
            .last_observed_observation_cycle();
        data.push(u8::from(observed_cycle.is_some()));
        data.extend_from_slice(&observed_cycle.unwrap_or_default().to_le_bytes());
        let admitted_cycle = self
            .prediction_error_cursor
            .last_admitted_observation_cycle();
        data.push(u8::from(admitted_cycle.is_some()));
        data.extend_from_slice(&admitted_cycle.unwrap_or_default().to_le_bytes());
        data
    }

    fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        if data.len() != 13 && data.len() != 36 {
            return Err(format!(
                "PerceptionManager checkpoint has unsupported length: {}",
                data.len()
            ));
        }

        // Decode/validate cursor extension before mutating manager state.
        // Legacy 13-byte checkpoints restore with an empty cursor.
        let (restored_observed_cycle, restored_admitted_cycle) = if data.len() == 13 {
            (None, None)
        } else {
            if &data[13..17] != b"PEVC" {
                return Err("PerceptionManager: invalid PE cursor checkpoint magic".to_string());
            }
            if data[17] != 2 {
                return Err(format!(
                    "PerceptionManager: unsupported PE cursor checkpoint version {}",
                    data[17]
                ));
            }
            let has_observed = data[18];
            let observed_value = u64::from_le_bytes(
                data[19..27]
                    .try_into()
                    .map_err(|_| "PerceptionManager: corrupt observed-cycle bytes".to_string())?,
            );
            let has_admitted = data[27];
            let admitted_value = u64::from_le_bytes(
                data[28..36]
                    .try_into()
                    .map_err(|_| "PerceptionManager: corrupt admitted-cycle bytes".to_string())?,
            );
            if has_observed > 1 || has_admitted > 1 {
                return Err("PerceptionManager: invalid PE cursor presence flag".to_string());
            }
            let observed = match has_observed {
                1 => Some(observed_value),
                0 if observed_value == 0 => None,
                0 => {
                    return Err(
                        "PerceptionManager: absent observed cycle must encode zero".to_string()
                    );
                }
                _ => unreachable!("presence flag was checked above"),
            };
            let admitted = match has_admitted {
                1 => Some(admitted_value),
                0 if admitted_value == 0 => None,
                0 => {
                    return Err(
                        "PerceptionManager: absent admitted cycle must encode zero".to_string()
                    );
                }
                _ => unreachable!("presence flag was checked above"),
            };
            if admitted.is_some_and(|cycle| observed.is_none_or(|observed_cycle| cycle > observed_cycle)) {
                return Err(
                    "PerceptionManager: admitted PE cycle exceeds observed high-water mark"
                        .to_string(),
                );
            }
            (observed, admitted)
        };

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
        self.prediction_error_cursor
            .restore_cycles(restored_observed_cycle, restored_admitted_cycle);
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
            prediction_error_valid: 1,
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
    fn test_invalid_pe_does_not_clear_existing_vigilance() {
        let mut manager = PerceptionManager::default();
        manager.vigilant = true;
        let initial_sensitivity = manager.attention_sensitivity;
        let mut snapshot = snapshot(0.9, 0.0, 0.5, false);
        snapshot.prediction_error_valid = 0;

        let _ = manager.process(&snapshot);

        assert!(manager.vigilant);
        assert_eq!(manager.attention_sensitivity, initial_sensitivity);
    }

    #[test]
    fn duplicate_pe_cannot_toggle_vigilance_twice_in_one_cycle() {
        let mut manager = PerceptionManager::default();
        let mut alarming = snapshot(0.2, 0.7, 0.5, false);
        alarming.cycle_number = 10;
        manager.process(&alarming);
        assert!(manager.vigilant);
        let sensitivity_after_alarm = manager.attention_sensitivity;

        // Contradictory PE/coherence payload with the same cycle is not a
        // second admissible observation and must not clear vigilance.
        alarming.coherence = 0.9;
        alarming.prediction_error = 0.0;
        manager.process(&alarming);
        assert!(manager.vigilant);
        assert_eq!(manager.attention_sensitivity, sensitivity_after_alarm);
    }

    #[test]
    fn invalid_newer_pe_blocks_delayed_older_but_allows_same_cycle_recovery() {
        let mut manager = PerceptionManager::default();
        let mut invalid = snapshot(0.2, 0.9, 0.5, false);
        invalid.cycle_number = 12;
        invalid.prediction_error_valid = 0;
        manager.process(&invalid);
        assert!(!manager.vigilant);

        let mut delayed = snapshot(0.2, 0.9, 0.5, false);
        delayed.cycle_number = 11;
        manager.process(&delayed);
        assert!(!manager.vigilant);

        let mut retry = snapshot(0.2, 0.9, 0.5, false);
        retry.cycle_number = 12;
        manager.process(&retry);
        assert!(manager.vigilant);
        assert_eq!(
            manager.prediction_error_cursor.last_admitted_observation_cycle(),
            Some(12)
        );
    }

    #[test]
    fn checkpoint_roundtrip_preserves_vigilance_pe_watermarks() {
        let mut manager = PerceptionManager::default();
        let mut valid = snapshot(0.2, 0.9, 0.5, false);
        valid.cycle_number = 10;
        manager.process(&valid);
        let mut invalid = valid;
        invalid.cycle_number = 12;
        invalid.prediction_error_valid = 0;
        manager.process(&invalid);

        let checkpoint = manager.checkpoint();
        assert_eq!(checkpoint.len(), 36);
        let mut restored = PerceptionManager::default();
        restored.restore(&checkpoint).unwrap();
        assert_eq!(
            restored.prediction_error_cursor.last_observed_observation_cycle(),
            Some(12)
        );
        assert_eq!(
            restored.prediction_error_cursor.last_admitted_observation_cycle(),
            Some(10)
        );

        let mut delayed = valid;
        delayed.cycle_number = 11;
        restored.process(&delayed);
        assert_eq!(
            restored.prediction_error_cursor.last_admitted_observation_cycle(),
            Some(10)
        );

        let mut retry = valid;
        retry.cycle_number = 12;
        restored.process(&retry);
        assert_eq!(
            restored.prediction_error_cursor.last_admitted_observation_cycle(),
            Some(12)
        );
    }

    #[test]
    fn legacy_perception_checkpoint_clears_pe_cursor() {
        let mut manager = PerceptionManager::default();
        let mut prior = snapshot(0.2, 0.9, 0.5, false);
        prior.cycle_number = 80;
        manager.process(&prior);
        let legacy = manager.checkpoint()[..13].to_vec();

        manager.restore(&legacy).unwrap();
        assert_eq!(
            manager.prediction_error_cursor.last_observed_observation_cycle(),
            None
        );
        let mut first_after_restore = prior;
        first_after_restore.cycle_number = 1;
        manager.process(&first_after_restore);
        assert_eq!(
            manager.prediction_error_cursor.last_admitted_observation_cycle(),
            Some(1)
        );
    }

    #[test]
    fn malformed_perception_pe_checkpoint_is_rejected_atomically() {
        let mut manager = PerceptionManager::default();
        let mut sample = snapshot(0.2, 0.9, 0.5, false);
        sample.cycle_number = 12;
        manager.process(&sample);
        let sensitivity_before = manager.attention_sensitivity;
        let mut checkpoint = manager.checkpoint();
        checkpoint[0..4].copy_from_slice(&0.5_f32.to_le_bytes());
        checkpoint[28..36].copy_from_slice(&13_u64.to_le_bytes());
        assert!(manager.restore(&checkpoint).is_err());
        assert_eq!(manager.attention_sensitivity, sensitivity_before);
        assert_eq!(
            manager.prediction_error_cursor.last_observed_observation_cycle(),
            Some(12)
        );
        assert_eq!(
            manager.prediction_error_cursor.last_admitted_observation_cycle(),
            Some(12)
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
