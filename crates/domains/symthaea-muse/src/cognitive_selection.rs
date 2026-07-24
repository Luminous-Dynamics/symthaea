// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transparent selection among theory-valid symbolic alternatives.
//!
//! Symthaea does not write notes here. The theory layer supplies alternatives
//! and evidence; this module orders them with an explicit lexicographic policy
//! rather than hiding musical judgment inside one opaque quality score.

use crate::cognitive_bridge::{
    CognitiveDecisionTrace, MusicalOutcomeError, SymbolicMeasurementEvidence,
};
use serde::{Deserialize, Serialize};

/// Evidence supplied for one theory-generated alternative.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolicAlternativeEvidence {
    pub alternative_id: String,
    pub measurement: SymbolicMeasurementEvidence,
    /// Hard theory invariants and validation passed.
    pub hard_constraints_valid: bool,
    /// The Studio Preserve side of the edit contract was respected.
    pub preserved_invariants: bool,
    /// Formal promises already due after this alternative.
    pub overdue_obligations_remaining: usize,
    /// All still-pending formal promises after this alternative.
    pub unresolved_obligations_remaining: usize,
    /// Priority-weighted remaining deadline pressure in [0, 1].
    pub obligation_pressure_remaining: f32,
    /// Whether the proposal's driving promise was independently verified.
    /// `None` means the proposal was not driven by one specific obligation.
    #[serde(default)]
    pub target_obligation_verified: Option<bool>,
    /// Transformation-aware thematic identity when the candidate returns a motif.
    #[serde(default)]
    pub motif_return_similarity: Option<f32>,
}

/// Interpretable assessment retained for every alternative, selected or not.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolicAlternativeAssessment {
    pub alternative_id: String,
    pub eligible: bool,
    pub hard_constraints_valid: bool,
    pub preserved_invariants: bool,
    pub overdue_obligations_remaining: usize,
    pub unresolved_obligations_remaining: usize,
    pub obligation_pressure_remaining: f32,
    pub target_obligation_verified: Option<bool>,
    pub motif_return_similarity: Option<f32>,
    pub prediction_error: MusicalOutcomeError,
}

/// Deterministic recommendation plus complete competing evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolicAlternativeSelection {
    pub recommended_id: Option<String>,
    pub rationale: Vec<String>,
    pub assessments: Vec<SymbolicAlternativeAssessment>,
}

/// Select among already-valid musical alternatives.
///
/// Ordering is deliberately lexicographic:
///
/// 1. reject hard-theory failures;
/// 2. reject Preserve-contract failures;
/// 3. minimize overdue obligations;
/// 4. minimize remaining deadline pressure;
/// 5. minimize mean prediction error;
/// 6. minimize unresolved obligations;
/// 7. use the stable alternative ID as a deterministic tie-break.
pub fn select_symbolic_alternative(
    trace: &CognitiveDecisionTrace,
    alternatives: &[SymbolicAlternativeEvidence],
) -> SymbolicAlternativeSelection {
    let mut assessments: Vec<SymbolicAlternativeAssessment> = alternatives
        .iter()
        .map(|alternative| SymbolicAlternativeAssessment {
            alternative_id: alternative.alternative_id.clone(),
            eligible: alternative.hard_constraints_valid && alternative.preserved_invariants,
            hard_constraints_valid: alternative.hard_constraints_valid,
            preserved_invariants: alternative.preserved_invariants,
            overdue_obligations_remaining: alternative.overdue_obligations_remaining,
            unresolved_obligations_remaining: alternative.unresolved_obligations_remaining,
            obligation_pressure_remaining: alternative
                .obligation_pressure_remaining
                .clamp(0.0, 1.0),
            target_obligation_verified: alternative.target_obligation_verified,
            motif_return_similarity: alternative
                .motif_return_similarity
                .map(|value| value.clamp(0.0, 1.0)),
            prediction_error: trace
                .predicted_outcome
                .error(alternative.measurement.observed_outcome),
        })
        .collect();

    assessments.sort_by(|left, right| {
        right
            .eligible
            .cmp(&left.eligible)
            .then_with(|| {
                target_verification_rank(left.target_obligation_verified)
                    .cmp(&target_verification_rank(right.target_obligation_verified))
            })
            .then_with(|| {
                left.overdue_obligations_remaining
                    .cmp(&right.overdue_obligations_remaining)
            })
            .then_with(|| {
                left.obligation_pressure_remaining
                    .total_cmp(&right.obligation_pressure_remaining)
            })
            .then_with(|| {
                left.prediction_error
                    .mean_absolute_error
                    .total_cmp(&right.prediction_error.mean_absolute_error)
            })
            .then_with(|| {
                left.unresolved_obligations_remaining
                    .cmp(&right.unresolved_obligations_remaining)
            })
            .then_with(|| {
                right
                    .motif_return_similarity
                    .unwrap_or(0.0)
                    .total_cmp(&left.motif_return_similarity.unwrap_or(0.0))
            })
            .then_with(|| left.alternative_id.cmp(&right.alternative_id))
    });

    let recommended = assessments.iter().find(|item| item.eligible);
    let mut rationale = Vec::new();
    if let Some(selected) = recommended {
        rationale.push(format!(
            "selected {} after hard theory and Preserve-contract checks",
            selected.alternative_id
        ));
        if let Some(verified) = selected.target_obligation_verified {
            rationale.push(format!(
                "driving obligation score-side verification: {}",
                if verified { "passed" } else { "failed" }
            ));
        }
        rationale.push(format!(
            "{} overdue obligation(s), remaining pressure {:.3}",
            selected.overdue_obligations_remaining, selected.obligation_pressure_remaining
        ));
        rationale.push(format!(
            "mean absolute prediction error {:.3}; channel errors retained separately",
            selected.prediction_error.mean_absolute_error
        ));
    } else {
        rationale.push(
            "no alternative satisfied both hard theory and Preserve-contract constraints".into(),
        );
    }

    SymbolicAlternativeSelection {
        recommended_id: recommended.map(|item| item.alternative_id.clone()),
        rationale,
        assessments,
    }
}

fn target_verification_rank(value: Option<bool>) -> u8 {
    match value {
        Some(true) => 0,
        None => 1,
        Some(false) => 2,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cognitive_bridge::{
        ActionScope, CognitiveSection, InferenceEvidence, PredictedMusicalOutcome, SymbolicAction,
        SymbolicActionProposal, SymbolicMusicObservation,
    };
    use crate::musical_inference::MusicAction;
    use symthaea_music_theory::ScoreCognitiveProfile;

    fn trace() -> CognitiveDecisionTrace {
        CognitiveDecisionTrace {
            observation: SymbolicMusicObservation {
                section: CognitiveSection::Development,
                active_goal: None,
                goal_urgency: 0.5,
                valence: 0.0,
                arousal: 0.5,
                prediction_error: 0.2,
                consciousness_level: 0.5,
                dominant_harmony: 0,
                dominant_harmony_activation: 0.7,
                pending_obligations: 0,
                overdue_obligations: Vec::new(),
                obligation_demands: Vec::new(),
                obligation_pressure: 0.0,
            },
            inference: InferenceEvidence {
                source_action: MusicAction::IncreaseComplexity,
                free_energy: 0.2,
                prediction_error: 0.2,
                surprise: 0.1,
                sensory_precision: 1.0,
                prior_precision: 1.0,
            },
            proposal: SymbolicActionProposal {
                action: SymbolicAction::IncreaseDensity,
                driving_obligation_id: None,
                supporting_obligation_ids: Vec::new(),
                deferred_obligation_ids: Vec::new(),
                scope: ActionScope::CurrentPhrase,
                preserve: Vec::new(),
                urgency: 0.5,
                confidence: 0.5,
                rationale: Vec::new(),
            },
            predicted_outcome: PredictedMusicalOutcome {
                tension_delta: 0.1,
                density_delta: 0.3,
                familiarity_delta: 0.0,
                tonal_displacement_delta: 0.0,
            },
        }
    }

    fn alternative(
        id: &str,
        density_delta: f32,
        valid: bool,
        preserved: bool,
        overdue: usize,
        pressure: f32,
    ) -> SymbolicAlternativeEvidence {
        let baseline = ScoreCognitiveProfile::default();
        let candidate = ScoreCognitiveProfile {
            tension: 0.1,
            density: density_delta,
            ..ScoreCognitiveProfile::default()
        };
        SymbolicAlternativeEvidence {
            alternative_id: id.into(),
            measurement: SymbolicMeasurementEvidence::new(baseline, candidate),
            hard_constraints_valid: valid,
            preserved_invariants: preserved,
            overdue_obligations_remaining: overdue,
            unresolved_obligations_remaining: overdue,
            obligation_pressure_remaining: pressure,
            target_obligation_verified: None,
            motif_return_similarity: None,
        }
    }

    #[test]
    fn invalid_perfect_prediction_loses_to_valid_music() {
        let selection = select_symbolic_alternative(
            &trace(),
            &[
                alternative("invalid", 0.3, false, true, 0, 0.0),
                alternative("valid", 0.2, true, true, 0, 0.0),
            ],
        );
        assert_eq!(selection.recommended_id.as_deref(), Some("valid"));
    }

    #[test]
    fn overdue_formal_promises_outrank_a_smaller_prediction_error() {
        let selection = select_symbolic_alternative(
            &trace(),
            &[
                alternative("perfect-but-overdue", 0.3, true, true, 1, 1.0),
                alternative("formally-responsible", 0.2, true, true, 0, 0.2),
            ],
        );
        assert_eq!(
            selection.recommended_id.as_deref(),
            Some("formally-responsible")
        );
    }

    #[test]
    fn no_eligible_alternative_produces_no_recommendation() {
        let selection = select_symbolic_alternative(
            &trace(),
            &[alternative("broken", 0.3, true, false, 0, 0.0)],
        );
        assert_eq!(selection.recommended_id, None);
    }

    #[test]
    fn verified_driving_promise_outranks_better_prediction_fit() {
        let mut failed = alternative("prediction-perfect", 0.3, true, true, 0, 0.0);
        failed.target_obligation_verified = Some(false);
        let mut verified = alternative("promise-kept", 0.1, true, true, 0, 0.0);
        verified.target_obligation_verified = Some(true);
        let selection = select_symbolic_alternative(&trace(), &[failed, verified]);
        assert_eq!(selection.recommended_id.as_deref(), Some("promise-kept"));
    }
}
