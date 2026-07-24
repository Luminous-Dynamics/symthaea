// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Contrastive evidence for learning long-term Eight Harmony preferences.
//!
//! Raw `activation * quality` confounds prevalence with value: a harmony that is
//! always active accumulates a large bias even when it has no predictive power.
//! This ledger separately estimates quality while a harmony is active and while
//! it is inactive, then learns from the difference.

use serde::{Deserialize, Serialize};

/// Provenance channel for harmony-quality evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarmonyEvidenceSource {
    SelfEvaluation,
    Human,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HarmonyChannel {
    active_quality_sum: [f32; 8],
    active_weight: [f32; 8],
    inactive_quality_sum: [f32; 8],
    inactive_weight: [f32; 8],
}

impl HarmonyChannel {
    fn new() -> Self {
        Self {
            active_quality_sum: [0.0; 8],
            active_weight: [0.0; 8],
            inactive_quality_sum: [0.0; 8],
            inactive_weight: [0.0; 8],
        }
    }

    fn observe(&mut self, activations: &[f32; 8], quality: f32, confidence: f32) {
        for index in 0..8 {
            let activation = finite_unit(activations[index]);
            let active_weight = confidence * activation;
            let inactive_weight = confidence * (1.0 - activation);
            self.active_quality_sum[index] += active_weight * quality;
            self.active_weight[index] += active_weight;
            self.inactive_quality_sum[index] += inactive_weight * quality;
            self.inactive_weight[index] += inactive_weight;
        }
    }

    fn advantage(&self, index: usize) -> Option<f32> {
        // Require meaningful evidence on both sides of the contrast. This avoids
        // declaring an always-on harmony good merely because all observed works
        // happened to score well.
        if self.active_weight[index] < 1.0 || self.inactive_weight[index] < 1.0 {
            return None;
        }
        let active = self.active_quality_sum[index] / self.active_weight[index];
        let inactive = self.inactive_quality_sum[index] / self.inactive_weight[index];
        Some((active - inactive).clamp(-1.0, 1.0))
    }

    fn validate(&self) -> bool {
        self.active_quality_sum
            .iter()
            .chain(self.active_weight.iter())
            .chain(self.inactive_quality_sum.iter())
            .chain(self.inactive_weight.iter())
            .all(|value| value.is_finite() && *value >= 0.0)
    }
}

impl Default for HarmonyChannel {
    fn default() -> Self {
        Self::new()
    }
}

/// Persistable contrastive ledger with human and self-evaluation channels kept
/// separate for auditability.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HarmonyEvidenceLedger {
    self_evaluation: HarmonyChannel,
    human: HarmonyChannel,
}

impl HarmonyEvidenceLedger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Observe one artifact. `quality` and `confidence` are bounded to [0, 1].
    pub fn observe(
        &mut self,
        source: HarmonyEvidenceSource,
        activations: &[f32; 8],
        quality: f32,
        confidence: f32,
    ) {
        let quality = finite_unit(quality);
        let confidence = finite_unit(confidence);
        match source {
            HarmonyEvidenceSource::SelfEvaluation => {
                self.self_evaluation
                    .observe(activations, quality, confidence);
            }
            HarmonyEvidenceSource::Human => {
                self.human.observe(activations, quality, confidence);
            }
        }
    }

    /// Estimated active-versus-inactive quality advantage in [-1, 1].
    pub fn advantage(&self, source: HarmonyEvidenceSource, index: usize) -> Option<f32> {
        if index >= 8 {
            return None;
        }
        match source {
            HarmonyEvidenceSource::SelfEvaluation => self.self_evaluation.advantage(index),
            HarmonyEvidenceSource::Human => self.human.advantage(index),
        }
    }

    /// Convert available contrastive evidence to the existing [0, 1] preference
    /// scale. Human evidence receives 80% weight when both channels are present.
    pub fn preference(&self, index: usize, fallback: f32) -> f32 {
        let self_advantage = self.advantage(HarmonyEvidenceSource::SelfEvaluation, index);
        let human_advantage = self.advantage(HarmonyEvidenceSource::Human, index);
        let advantage = match (self_advantage, human_advantage) {
            (Some(self_value), Some(human_value)) => 0.2 * self_value + 0.8 * human_value,
            (Some(value), None) | (None, Some(value)) => value,
            (None, None) => return finite_unit(fallback),
        };
        (0.5 + 0.5 * advantage).clamp(0.0, 1.0)
    }

    pub(crate) fn validate(&self) -> bool {
        self.self_evaluation.validate() && self.human.validate()
    }
}

fn finite_unit(value: f32) -> f32 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prevalence_alone_does_not_create_preference() {
        let mut ledger = HarmonyEvidenceLedger::new();
        for _ in 0..20 {
            ledger.observe(HarmonyEvidenceSource::SelfEvaluation, &[1.0; 8], 0.9, 1.0);
        }
        assert_eq!(
            ledger.advantage(HarmonyEvidenceSource::SelfEvaluation, 0),
            None
        );
        assert_eq!(ledger.preference(0, 0.2), 0.2);
    }

    #[test]
    fn active_high_inactive_low_creates_positive_advantage() {
        let mut ledger = HarmonyEvidenceLedger::new();
        let mut active = [0.0; 8];
        active[2] = 1.0;
        let inactive = [0.0; 8];
        for _ in 0..10 {
            ledger.observe(HarmonyEvidenceSource::Human, &active, 0.9, 1.0);
            ledger.observe(HarmonyEvidenceSource::Human, &inactive, 0.1, 1.0);
        }
        assert!(ledger.preference(2, 0.5) > 0.8);
    }

    #[test]
    fn human_evidence_dominates_conflicting_self_evidence() {
        let mut ledger = HarmonyEvidenceLedger::new();
        let mut active = [0.0; 8];
        active[4] = 1.0;
        let inactive = [0.0; 8];
        for _ in 0..10 {
            ledger.observe(HarmonyEvidenceSource::SelfEvaluation, &active, 0.1, 1.0);
            ledger.observe(HarmonyEvidenceSource::SelfEvaluation, &inactive, 0.9, 1.0);
            ledger.observe(HarmonyEvidenceSource::Human, &active, 0.9, 1.0);
            ledger.observe(HarmonyEvidenceSource::Human, &inactive, 0.1, 1.0);
        }
        assert!(ledger.preference(4, 0.5) > 0.65);
    }
}
