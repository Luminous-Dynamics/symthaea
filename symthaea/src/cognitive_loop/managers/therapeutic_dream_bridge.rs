// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Therapeutic Dream Bridge — counterfactual therapeutic reasoning.
//!
//! Implements `DreamableAction` for therapeutic interventions, enabling
//! the dream engine to generate counterfactual therapeutic scenarios:
//! "What if the client used cognitive reappraisal instead of avoidance?"
//!
//! Feature gate: `therapeutic`

use serde::{Deserialize, Serialize};
use symthaea_clinical::rdoc::RDocDomain;
use symthaea_clinical::therapeutic_modalities::{TherapeuticIntervention, TherapeuticModality};
use symthaea_dream::DreamableAction;

/// A dreamable wrapper around a therapeutic intervention.
///
/// Encodes the intervention as a float vector for the dream engine's
/// counterfactual reasoning, while preserving the clinical semantics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DreamableTherapeuticAction {
    /// The intervention modality
    pub modality: TherapeuticModality,
    /// Name of the technique
    pub technique: String,
    /// Target RDoC domains (encoded as indices)
    pub target_domains: Vec<u8>,
    /// Intervention intensity (0.0-1.0)
    pub intensity: f32,
    /// Evidence level (0.0-1.0)
    pub evidence_level: f32,
    /// Minimum alliance required
    pub min_alliance: f32,
}

impl DreamableTherapeuticAction {
    /// Create from a `TherapeuticIntervention`.
    pub fn from_intervention(intervention: &TherapeuticIntervention, intensity: f32) -> Self {
        let target_domains: Vec<u8> = intervention
            .target_domains
            .iter()
            .map(|d| match d {
                RDocDomain::NegativeValence => 0,
                RDocDomain::PositiveValence => 1,
                RDocDomain::CognitiveSystems => 2,
                RDocDomain::SocialProcesses => 3,
                RDocDomain::ArousalRegulatory => 4,
                RDocDomain::Sensorimotor => 5,
            })
            .collect();

        Self {
            modality: intervention.modality,
            technique: intervention.technique.clone(),
            target_domains,
            intensity: intensity.clamp(0.0, 1.0),
            evidence_level: intervention.evidence_level,
            min_alliance: intervention.min_alliance,
        }
    }

    /// Generate an alternative intervention (different modality, same targets).
    fn alternative_modality(&self, seed: u64) -> TherapeuticModality {
        let modalities = [
            TherapeuticModality::Cbt,
            TherapeuticModality::Act,
            TherapeuticModality::Dbt,
            TherapeuticModality::Narrative,
            TherapeuticModality::Somatic,
            TherapeuticModality::Psychodynamic,
            TherapeuticModality::Motivational,
            TherapeuticModality::Emdr,
            TherapeuticModality::Ifs,
        ];
        let idx = (seed as usize) % modalities.len();
        let alt = modalities[idx];
        if alt == self.modality {
            modalities[(idx + 1) % modalities.len()]
        } else {
            alt
        }
    }
}

/// Record of a predicted outcome and optional actual outcome for closed-loop validation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionRecord {
    /// Predicted RDoC outcome state.
    pub predicted_outcome: Vec<f32>,
    /// Actual RDoC outcome (filled in after the fact).
    pub actual_outcome: Option<Vec<f32>>,
    /// L2 prediction error (computed when actual is recorded).
    pub prediction_error: Option<f32>,
    /// Cycle when prediction was made.
    pub cycle: u64,
}

/// Tracker for dream counterfactual prediction accuracy.
///
/// Maintains a rolling history of predicted vs actual RDoC outcomes,
/// enabling closed-loop validation of therapeutic counterfactual reasoning.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TherapeuticDreamTracker {
    /// Prediction history (ring buffer, max 64).
    pub predictions: Vec<PredictionRecord>,
}

impl TherapeuticDreamTracker {
    /// Maximum prediction history size.
    const MAX_HISTORY: usize = 64;

    pub fn new() -> Self {
        Self::default()
    }

    /// Record a predicted outcome.
    pub fn record_prediction(&mut self, predicted: Vec<f32>, cycle: u64) {
        if self.predictions.len() >= Self::MAX_HISTORY {
            self.predictions.remove(0);
        }
        self.predictions.push(PredictionRecord {
            predicted_outcome: predicted,
            actual_outcome: None,
            prediction_error: None,
            cycle,
        });
    }

    /// Record the actual outcome for the most recent unresolved prediction.
    ///
    /// Computes L2 prediction error between predicted and actual RDoC states.
    pub fn record_actual_outcome(&mut self, actual: &[f32]) {
        if let Some(record) = self
            .predictions
            .iter_mut()
            .rev()
            .find(|r| r.actual_outcome.is_none())
        {
            let error: f32 = record
                .predicted_outcome
                .iter()
                .zip(actual.iter())
                .map(|(p, a)| (p - a).powi(2))
                .sum::<f32>()
                .sqrt();
            record.actual_outcome = Some(actual.to_vec());
            record.prediction_error = Some(error);
        }
    }

    /// Mean prediction error across resolved predictions (lower = better).
    ///
    /// Returns 1.0 if no resolved predictions exist.
    pub fn prediction_accuracy(&self) -> f32 {
        let errors: Vec<f32> = self
            .predictions
            .iter()
            .filter_map(|r| r.prediction_error)
            .collect();
        if errors.is_empty() {
            return 1.0;
        }
        errors.iter().sum::<f32>() / errors.len() as f32
    }

    /// Serialize tracker state to bytes for process restart resilience.
    ///
    /// Layout: [prediction_count: u32 = 4][resolved_count: u32 = 4]
    ///         [prediction_accuracy: f32 = 4]
    /// Total: 12 bytes
    ///
    /// Note: Individual PredictionRecord history is not checkpointed here
    /// (it uses Serialize/Deserialize for full persistence). This captures
    /// the summary statistics needed for fast restart.
    pub fn checkpoint(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(12);
        data.extend_from_slice(&(self.predictions.len() as u32).to_le_bytes());
        let resolved = self
            .predictions
            .iter()
            .filter(|r| r.actual_outcome.is_some())
            .count() as u32;
        data.extend_from_slice(&resolved.to_le_bytes());
        data.extend_from_slice(&self.prediction_accuracy().to_le_bytes());
        data
    }

    /// Restore tracker state from a checkpoint.
    ///
    /// Since individual predictions cannot be reconstructed from summary
    /// statistics, this is a best-effort restore that validates the data
    /// format. The prediction history will rebuild naturally from subsequent
    /// `record_prediction` / `record_actual_outcome` calls.
    pub fn restore(&mut self, data: &[u8]) -> Result<(), String> {
        const MIN_SIZE: usize = 4 + 4 + 4; // 12
        if data.len() < MIN_SIZE {
            return Err(format!(
                "TherapeuticDreamTracker checkpoint too short: {} < {}",
                data.len(),
                MIN_SIZE
            ));
        }
        // Validation only — we cannot reconstruct individual predictions
        // from summary statistics. The counts are informational.
        let _prediction_count = u32::from_le_bytes(
            data[0..4]
                .try_into()
                .map_err(|_| "TherapeuticDreamTracker: corrupt prediction_count bytes")?,
        );
        let _resolved_count = u32::from_le_bytes(
            data[4..8]
                .try_into()
                .map_err(|_| "TherapeuticDreamTracker: corrupt resolved_count bytes")?,
        );
        let _accuracy = f32::from_le_bytes(
            data[8..12]
                .try_into()
                .map_err(|_| "TherapeuticDreamTracker: corrupt accuracy bytes")?,
        );
        Ok(())
    }

    /// Fraction of predictions that have been resolved with actual outcomes.
    pub fn resolution_rate(&self) -> f32 {
        if self.predictions.is_empty() {
            return 0.0;
        }
        let resolved = self
            .predictions
            .iter()
            .filter(|r| r.actual_outcome.is_some())
            .count();
        resolved as f32 / self.predictions.len() as f32
    }
}

impl DreamableTherapeuticAction {
    /// Generate a set of counterfactual therapeutic actions with unique modalities.
    ///
    /// Each seed produces a perturbation with a different modality, enabling
    /// the dream engine to compare multiple alternative strategies.
    pub fn generate_counterfactual_set(&self, seeds: &[u64]) -> Vec<DreamableTherapeuticAction> {
        let mut seen_modalities = std::collections::HashSet::new();
        seen_modalities.insert(self.modality);
        let mut set = Vec::new();

        for &seed in seeds {
            let perturbed = self.perturb(seed);
            if !seen_modalities.contains(&perturbed.modality) {
                seen_modalities.insert(perturbed.modality);
                set.push(perturbed);
            }
        }
        set
    }

    /// Rank counterfactual actions by predicted RDoC improvement.
    ///
    /// Returns indices sorted by total improvement magnitude (best first).
    pub fn rank_counterfactuals(
        actions: &[DreamableTherapeuticAction],
        state: &[f32],
    ) -> Vec<(usize, f32)> {
        let mut ranked: Vec<(usize, f32)> = actions
            .iter()
            .enumerate()
            .map(|(i, action)| {
                let outcome = action.predict_outcome(state);
                let improvement: f32 = outcome
                    .iter()
                    .zip(state.iter())
                    .map(|(o, s)| {
                        // Positive for neg_valence reduction, positive for other improvements
                        o - s
                    })
                    .sum();
                (i, improvement)
            })
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked
    }
}

impl DreamableAction for DreamableTherapeuticAction {
    fn perturb(&self, seed: u64) -> Self {
        let alt_modality = self.alternative_modality(seed);

        // Perturb intensity slightly
        let hash = blake3::hash(&seed.to_le_bytes());
        let noise_byte = hash.as_bytes()[0] as f32 / 255.0;
        let perturbed_intensity = (self.intensity + (noise_byte - 0.5) * 0.3).clamp(0.0, 1.0);

        Self {
            modality: alt_modality,
            technique: format!("{} (counterfactual)", self.technique),
            target_domains: self.target_domains.clone(),
            intensity: perturbed_intensity,
            evidence_level: self.evidence_level,
            min_alliance: self.min_alliance,
        }
    }

    fn predict_outcome(&self, state: &[f32]) -> Vec<f32> {
        // Predict therapeutic outcome as delta on RDoC dimensions.
        // State is assumed to be [neg_valence, pos_valence, cognitive, social, arousal, sensorimotor].
        let mut outcome = state.to_vec();
        if outcome.len() < 6 {
            outcome.resize(6, 0.0);
        }

        let effect = self.intensity * self.evidence_level;

        for &domain_idx in &self.target_domains {
            let idx = domain_idx as usize;
            if idx < outcome.len() {
                match idx {
                    0 => outcome[idx] -= effect * 0.3,  // Reduce negative valence
                    1 => outcome[idx] += effect * 0.2,  // Increase positive valence
                    2 => outcome[idx] += effect * 0.15, // Improve cognitive function
                    3 => outcome[idx] += effect * 0.1,  // Improve social processing
                    4 => outcome[idx] -= effect * 0.2,  // Reduce arousal dysregulation
                    5 => outcome[idx] += effect * 0.05, // Minor sensorimotor improvement
                    _ => {}
                }
                outcome[idx] = outcome[idx].clamp(0.0, 1.0);
            }
        }

        outcome
    }

    fn magnitude(&self) -> f32 {
        // Intervention magnitude based on intensity and number of targeted domains
        self.intensity * (self.target_domains.len() as f32 / 6.0).clamp(0.1, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_action() -> DreamableTherapeuticAction {
        DreamableTherapeuticAction {
            modality: TherapeuticModality::Cbt,
            technique: "cognitive_reappraisal".to_string(),
            target_domains: vec![0, 2], // NegativeValence, CognitiveSystems
            intensity: 0.7,
            evidence_level: 0.85,
            min_alliance: 0.4,
        }
    }

    #[test]
    fn test_perturb_changes_modality() {
        let action = sample_action();
        let perturbed = action.perturb(42);
        assert_ne!(perturbed.modality, action.modality);
        assert!(perturbed.technique.contains("counterfactual"));
    }

    #[test]
    fn test_perturb_deterministic() {
        let action = sample_action();
        let p1 = action.perturb(42);
        let p2 = action.perturb(42);
        assert_eq!(p1.modality, p2.modality);
        assert_eq!(p1.intensity, p2.intensity);
    }

    #[test]
    fn test_predict_outcome_reduces_negative_valence() {
        let action = sample_action();
        let state = vec![0.8, 0.3, 0.5, 0.5, 0.6, 0.5];
        let outcome = action.predict_outcome(&state);
        assert!(outcome[0] < state[0], "negative valence should decrease");
        assert!(outcome[2] > state[2], "cognitive should improve");
    }

    #[test]
    fn test_predict_outcome_clamps() {
        let action = DreamableTherapeuticAction {
            modality: TherapeuticModality::Dbt,
            technique: "distress_tolerance".to_string(),
            target_domains: vec![0, 4],
            intensity: 1.0,
            evidence_level: 1.0,
            min_alliance: 0.2,
        };
        let state = vec![0.05, 0.95, 0.5, 0.5, 0.05, 0.5];
        let outcome = action.predict_outcome(&state);
        for &v in &outcome {
            assert!(v >= 0.0 && v <= 1.0);
        }
    }

    #[test]
    fn test_magnitude_scales_with_domains() {
        let single = DreamableTherapeuticAction {
            modality: TherapeuticModality::Act,
            technique: "defusion".to_string(),
            target_domains: vec![0],
            intensity: 0.5,
            evidence_level: 0.8,
            min_alliance: 0.3,
        };
        let multi = DreamableTherapeuticAction {
            target_domains: vec![0, 1, 2, 3],
            ..single.clone()
        };
        assert!(multi.magnitude() > single.magnitude());
    }

    #[test]
    fn test_from_intervention() {
        use symthaea_clinical::therapeutic_modalities::InterventionLibrary;
        let lib = InterventionLibrary::bootstrap();
        let interventions = lib.by_modality(TherapeuticModality::Cbt);
        assert!(!interventions.is_empty());
        let dreamable = DreamableTherapeuticAction::from_intervention(&interventions[0], 0.6);
        assert_eq!(dreamable.modality, TherapeuticModality::Cbt);
        assert_eq!(dreamable.intensity, 0.6);
    }

    #[test]
    fn test_predict_outcome_short_state() {
        let action = sample_action();
        let state = vec![0.5, 0.5]; // Only 2 elements
        let outcome = action.predict_outcome(&state);
        assert_eq!(outcome.len(), 6); // Padded to 6
    }

    // ── Multi-strategy counterfactual tests ──

    #[test]
    fn test_counterfactual_set_unique_modalities() {
        let action = sample_action();
        let seeds: Vec<u64> = (0..20).collect();
        let set = action.generate_counterfactual_set(&seeds);
        // Should have unique modalities (up to 8 alternatives since 9 total minus original)
        let modalities: std::collections::HashSet<_> = set.iter().map(|a| a.modality).collect();
        assert_eq!(
            set.len(),
            modalities.len(),
            "All counterfactuals should have unique modalities",
        );
        assert!(
            set.len() >= 3,
            "Should generate at least 3 unique alternatives, got {}",
            set.len(),
        );
    }

    #[test]
    fn test_rank_counterfactuals_best_first() {
        let action = sample_action();
        let seeds: Vec<u64> = (0..10).collect();
        let set = action.generate_counterfactual_set(&seeds);
        let state = vec![0.8, 0.3, 0.4, 0.4, 0.7, 0.4];
        let ranked = DreamableTherapeuticAction::rank_counterfactuals(&set, &state);
        assert!(!ranked.is_empty());
        // Verify sorted descending by improvement
        for w in ranked.windows(2) {
            assert!(
                w[0].1 >= w[1].1,
                "Rankings should be sorted: {} >= {}",
                w[0].1,
                w[1].1,
            );
        }
    }

    // ── Prediction tracking tests ──

    #[test]
    fn test_prediction_tracker_record_and_resolve() {
        let mut tracker = TherapeuticDreamTracker::new();
        tracker.record_prediction(vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 1);
        assert_eq!(tracker.predictions.len(), 1);
        assert_eq!(tracker.resolution_rate(), 0.0);

        tracker.record_actual_outcome(&[0.4, 0.6, 0.5, 0.5, 0.4, 0.5]);
        assert_eq!(tracker.resolution_rate(), 1.0);
        assert!(tracker.prediction_accuracy() < 1.0);
        assert!(tracker.prediction_accuracy() > 0.0);
    }

    #[test]
    fn test_prediction_tracker_accuracy_decreases_with_better_predictions() {
        let mut tracker = TherapeuticDreamTracker::new();
        // Good prediction
        tracker.record_prediction(vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 1);
        tracker.record_actual_outcome(&[0.5, 0.5, 0.5, 0.5, 0.5, 0.5]); // perfect
        let good_accuracy = tracker.prediction_accuracy();

        // Bad prediction
        let mut tracker2 = TherapeuticDreamTracker::new();
        tracker2.record_prediction(vec![0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 1);
        tracker2.record_actual_outcome(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0]); // way off
        let bad_accuracy = tracker2.prediction_accuracy();

        assert!(
            good_accuracy < bad_accuracy,
            "Better predictions should have lower error: {} vs {}",
            good_accuracy,
            bad_accuracy,
        );
    }

    #[test]
    fn test_prediction_tracker_history_bounded() {
        let mut tracker = TherapeuticDreamTracker::new();
        for i in 0..100 {
            tracker.record_prediction(vec![0.5; 6], i);
        }
        assert!(
            tracker.predictions.len() <= TherapeuticDreamTracker::MAX_HISTORY,
            "History should be bounded at {}",
            TherapeuticDreamTracker::MAX_HISTORY,
        );
    }

    #[test]
    fn test_closed_loop_10_rounds() {
        let mut tracker = TherapeuticDreamTracker::new();
        let action = sample_action();
        let initial_state = vec![0.7, 0.3, 0.5, 0.5, 0.6, 0.5];

        for i in 0..10 {
            let predicted = action.predict_outcome(&initial_state);
            tracker.record_prediction(predicted, i);
            // Simulate actual outcome close to predicted (with noise)
            let mut actual = action.predict_outcome(&initial_state);
            actual[0] += 0.05; // small deviation
            tracker.record_actual_outcome(&actual);
        }

        assert_eq!(tracker.predictions.len(), 10);
        assert_eq!(tracker.resolution_rate(), 1.0);
        assert!(
            tracker.prediction_accuracy().is_finite(),
            "Prediction accuracy should be finite",
        );
    }
}
