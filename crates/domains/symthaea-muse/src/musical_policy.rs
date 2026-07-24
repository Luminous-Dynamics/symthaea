// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Explicit policy evaluation, separate from the learned outcome model.
//!
//! The outcome model answers "what will this intervention do?". This module
//! answers "which predicted valid outcome best serves the artist and formal
//! obligation?". Prediction accuracy is retained as evidence after measurement
//! but is never rewarded as musical value during candidate selection.

use crate::adaptive_prediction::{InterventionCalibrationEvidence, OutcomeUncertainty};
use crate::cognitive_bridge::{
    MusicalOutcomeError, ObservedMusicalOutcome, PredictedMusicalOutcome, SymbolicAction,
};
use serde::{Deserialize, Serialize};
use symthaea_music_theory::TheoryValidationReport;

pub const MUSICAL_POLICY_VERSION: &str = "musical-policy-v1";

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OutcomeChannelWeights {
    pub tension: f32,
    pub density: f32,
    pub familiarity: f32,
    pub tonal_displacement: f32,
}

impl OutcomeChannelWeights {
    fn normalized(self) -> Self {
        let sum = self.tension.max(0.0)
            + self.density.max(0.0)
            + self.familiarity.max(0.0)
            + self.tonal_displacement.max(0.0);
        if sum <= f32::EPSILON {
            return Self {
                tension: 0.25,
                density: 0.25,
                familiarity: 0.25,
                tonal_displacement: 0.25,
            };
        }
        Self {
            tension: self.tension.max(0.0) / sum,
            density: self.density.max(0.0) / sum,
            familiarity: self.familiarity.max(0.0) / sum,
            tonal_displacement: self.tonal_displacement.max(0.0) / sum,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalPolicyPreference {
    pub policy_version: String,
    pub action: SymbolicAction,
    pub desired_outcome: PredictedMusicalOutcome,
    pub channel_weights: OutcomeChannelWeights,
    pub uncertainty_penalty_weight: f32,
    pub motif_identity_weight: f32,
}

impl MusicalPolicyPreference {
    /// Frozen initial utility targets. These are preferences, not predictions;
    /// learning an outcome model never mutates them.
    pub fn for_action(action: SymbolicAction) -> Self {
        let (desired_outcome, channel_weights, motif_identity_weight) = match action {
            SymbolicAction::ReturnOpeningMaterial => (
                PredictedMusicalOutcome {
                    tension_delta: -0.2,
                    density_delta: 0.0,
                    familiarity_delta: 0.5,
                    tonal_displacement_delta: -0.35,
                },
                OutcomeChannelWeights {
                    tension: 0.20,
                    density: 0.10,
                    familiarity: 0.40,
                    tonal_displacement: 0.30,
                },
                0.30,
            ),
            SymbolicAction::StrengthenCadence => (
                PredictedMusicalOutcome {
                    tension_delta: -0.35,
                    density_delta: -0.05,
                    familiarity_delta: 0.2,
                    tonal_displacement_delta: -0.2,
                },
                OutcomeChannelWeights {
                    tension: 0.40,
                    density: 0.10,
                    familiarity: 0.15,
                    tonal_displacement: 0.35,
                },
                0.10,
            ),
            _ => (
                crate::cognitive_bridge::default_predicted_outcome(action),
                OutcomeChannelWeights {
                    tension: 0.25,
                    density: 0.25,
                    familiarity: 0.25,
                    tonal_displacement: 0.25,
                },
                0.10,
            ),
        };
        Self {
            policy_version: MUSICAL_POLICY_VERSION.into(),
            action,
            desired_outcome,
            channel_weights,
            uncertainty_penalty_weight: 0.15,
            motif_identity_weight,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyCandidateEvidence {
    pub alternative_id: String,
    pub theory_validation: TheoryValidationReport,
    pub preserved_invariants: bool,
    pub overdue_obligations_remaining: usize,
    pub unresolved_obligations_remaining: usize,
    pub obligation_pressure_remaining: f32,
    pub target_obligation_verified: Option<bool>,
    pub motif_return_similarity: Option<f32>,
    pub prediction: InterventionCalibrationEvidence,
    /// Measured outcome is retained only to audit the world model. It is not
    /// consulted by the policy ordering.
    pub observed_outcome: Option<ObservedMusicalOutcome>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PolicyCandidateAssessment {
    pub alternative_id: String,
    pub eligible: bool,
    pub theory_validation: TheoryValidationReport,
    pub preserved_invariants: bool,
    pub overdue_obligations_remaining: usize,
    pub unresolved_obligations_remaining: usize,
    pub obligation_pressure_remaining: f32,
    pub target_obligation_verified: Option<bool>,
    pub motif_return_similarity: Option<f32>,
    pub predicted_outcome: PredictedMusicalOutcome,
    pub outcome_utility: f32,
    pub uncertainty_penalty: f32,
    pub prediction_error: Option<MusicalOutcomeError>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalPolicySelection {
    pub policy: MusicalPolicyPreference,
    pub recommended_id: Option<String>,
    pub rationale: Vec<String>,
    pub assessments: Vec<PolicyCandidateAssessment>,
}

pub fn select_by_musical_policy(
    policy: MusicalPolicyPreference,
    candidates: &[PolicyCandidateEvidence],
) -> MusicalPolicySelection {
    let weights = policy.channel_weights.normalized();
    let mut assessments: Vec<_> = candidates
        .iter()
        .map(|candidate| {
            let prediction = candidate.prediction.calibrated;
            let distance = weighted_distance(prediction, policy.desired_outcome, weights);
            let uncertainty_penalty =
                weighted_uncertainty(candidate.prediction.uncertainty, weights)
                    * policy.uncertainty_penalty_weight.max(0.0);
            let motif_bonus = candidate
                .motif_return_similarity
                .unwrap_or(0.0)
                .clamp(0.0, 1.0)
                * policy.motif_identity_weight.max(0.0);
            let outcome_utility = motif_bonus - distance - uncertainty_penalty;
            PolicyCandidateAssessment {
                alternative_id: candidate.alternative_id.clone(),
                eligible: candidate.theory_validation.valid && candidate.preserved_invariants,
                theory_validation: candidate.theory_validation.clone(),
                preserved_invariants: candidate.preserved_invariants,
                overdue_obligations_remaining: candidate.overdue_obligations_remaining,
                unresolved_obligations_remaining: candidate.unresolved_obligations_remaining,
                obligation_pressure_remaining: candidate
                    .obligation_pressure_remaining
                    .clamp(0.0, 1.0),
                target_obligation_verified: candidate.target_obligation_verified,
                motif_return_similarity: candidate
                    .motif_return_similarity
                    .map(|value| value.clamp(0.0, 1.0)),
                predicted_outcome: prediction,
                outcome_utility,
                uncertainty_penalty,
                prediction_error: candidate
                    .observed_outcome
                    .map(|observed| prediction.error(observed)),
            }
        })
        .collect();

    assessments.sort_by(|left, right| {
        right
            .eligible
            .cmp(&left.eligible)
            .then_with(|| {
                target_rank(left.target_obligation_verified)
                    .cmp(&target_rank(right.target_obligation_verified))
            })
            .then_with(|| {
                left.overdue_obligations_remaining
                    .cmp(&right.overdue_obligations_remaining)
            })
            .then_with(|| right.outcome_utility.total_cmp(&left.outcome_utility))
            .then_with(|| {
                left.obligation_pressure_remaining
                    .total_cmp(&right.obligation_pressure_remaining)
            })
            .then_with(|| {
                left.unresolved_obligations_remaining
                    .cmp(&right.unresolved_obligations_remaining)
            })
            .then_with(|| left.alternative_id.cmp(&right.alternative_id))
    });

    let recommended = assessments.iter().find(|assessment| assessment.eligible);
    let rationale = if let Some(selected) = recommended {
        vec![
            format!(
                "selected {} after canonical theory and Preserve-contract checks",
                selected.alternative_id
            ),
            format!(
                "policy utility {:.3} from predicted musical effects; uncertainty penalty {:.3}",
                selected.outcome_utility, selected.uncertainty_penalty
            ),
            "prediction accuracy is retained separately and did not participate in selection"
                .to_owned(),
        ]
    } else {
        vec!["no candidate passed canonical theory and Preserve-contract checks".into()]
    };

    MusicalPolicySelection {
        policy,
        recommended_id: recommended.map(|assessment| assessment.alternative_id.clone()),
        rationale,
        assessments,
    }
}

fn weighted_distance(
    actual: PredictedMusicalOutcome,
    desired: PredictedMusicalOutcome,
    weights: OutcomeChannelWeights,
) -> f32 {
    (actual.tension_delta - desired.tension_delta).abs() * weights.tension
        + (actual.density_delta - desired.density_delta).abs() * weights.density
        + (actual.familiarity_delta - desired.familiarity_delta).abs() * weights.familiarity
        + (actual.tonal_displacement_delta - desired.tonal_displacement_delta).abs()
            * weights.tonal_displacement
}

fn weighted_uncertainty(uncertainty: OutcomeUncertainty, weights: OutcomeChannelWeights) -> f32 {
    uncertainty.tension.abs() * weights.tension
        + uncertainty.density.abs() * weights.density
        + uncertainty.familiarity.abs() * weights.familiarity
        + uncertainty.tonal_displacement.abs() * weights.tonal_displacement
}

fn target_rank(value: Option<bool>) -> u8 {
    match value {
        Some(true) => 0,
        None => 1,
        Some(false) => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::adaptive_prediction::{
        InterventionPredictionContext, PredictionContext, PredictionEvidenceSource, TextureBand,
    };
    use crate::cognitive_bridge::CognitiveSection;
    use crate::intervention::{InterventionDescriptor, InterventionStrategy, ObligationClass};
    use symthaea_music_theory::{ScoreValidationConfig, validate_score};

    fn prediction(
        id: &str,
        predicted: PredictedMusicalOutcome,
        observed: ObservedMusicalOutcome,
    ) -> PolicyCandidateEvidence {
        let base = PredictionContext::new(
            SymbolicAction::ReturnOpeningMaterial,
            CognitiveSection::Recapitulation,
            "Sonata",
            "Sonata",
            4,
            TextureBand::Chamber,
        );
        let descriptor = InterventionDescriptor::new(
            SymbolicAction::ReturnOpeningMaterial,
            InterventionStrategy::Literal,
            CognitiveSection::Exposition,
            CognitiveSection::Recapitulation,
            ObligationClass::ReturnMotif,
            0,
            1.0,
            0.5,
            0.5,
            0.5,
            0.5,
            8,
            80,
        );
        let empty = symthaea_music_theory::Score::new(
            symthaea_music_theory::Key::major(symthaea_music_theory::PitchClass::C),
            120.0,
            4,
        );
        let mut report = validate_score(&empty, &ScoreValidationConfig::default());
        report.valid = true;
        report.issues.clear();
        PolicyCandidateEvidence {
            alternative_id: id.into(),
            theory_validation: report,
            preserved_invariants: true,
            overdue_obligations_remaining: 0,
            unresolved_obligations_remaining: 0,
            obligation_pressure_remaining: 0.0,
            target_obligation_verified: Some(true),
            motif_return_similarity: Some(1.0),
            prediction: InterventionCalibrationEvidence {
                model_version: "test".into(),
                context: InterventionPredictionContext::new(base, descriptor),
                source: PredictionEvidenceSource::HandAuthoredPrior,
                intervention_context_samples: 0,
                strategy_fallback_samples: 0,
                action_fallback_samples: 0,
                intervention_context_moments: None,
                strategy_fallback_moments: None,
                action_fallback_moments: None,
                prior: predicted,
                calibrated: predicted,
                uncertainty: OutcomeUncertainty::default(),
            },
            observed_outcome: Some(observed),
        }
    }

    #[test]
    fn prediction_accuracy_is_not_mistaken_for_musical_utility() {
        let desirable = PredictedMusicalOutcome {
            tension_delta: -0.2,
            density_delta: 0.0,
            familiarity_delta: 0.5,
            tonal_displacement_delta: -0.35,
        };
        let undesirable = PredictedMusicalOutcome {
            tension_delta: 0.4,
            density_delta: 0.4,
            familiarity_delta: -0.4,
            tonal_displacement_delta: 0.5,
        };
        let candidates = vec![
            prediction(
                "desirable-but-imperfectly-predicted",
                desirable,
                ObservedMusicalOutcome {
                    tension_delta: 0.2,
                    density_delta: 0.2,
                    familiarity_delta: 0.1,
                    tonal_displacement_delta: 0.1,
                },
            ),
            prediction(
                "undesirable-but-perfectly-predicted",
                undesirable,
                ObservedMusicalOutcome {
                    tension_delta: 0.4,
                    density_delta: 0.4,
                    familiarity_delta: -0.4,
                    tonal_displacement_delta: 0.5,
                },
            ),
        ];
        let selection = select_by_musical_policy(
            MusicalPolicyPreference::for_action(SymbolicAction::ReturnOpeningMaterial),
            &candidates,
        );
        assert_eq!(
            selection.recommended_id.as_deref(),
            Some("desirable-but-imperfectly-predicted")
        );
        let accurate = selection
            .assessments
            .iter()
            .find(|item| item.alternative_id == "undesirable-but-perfectly-predicted")
            .unwrap();
        assert_eq!(accurate.prediction_error.unwrap().mean_absolute_error, 0.0);
    }
}
