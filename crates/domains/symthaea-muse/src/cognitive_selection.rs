// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transparent selection among theory-valid symbolic alternatives.
//!
//! Symthaea does not write notes here. The theory layer supplies alternatives
//! and evidence; this module orders them with an explicit lexicographic policy
//! rather than hiding musical judgment inside one opaque quality score.

use crate::cognitive_bridge::{
    CognitiveDecisionTrace, MusicalOutcomeError, PredictedMusicalOutcome, SymbolicAction,
    SymbolicMeasurementEvidence, default_predicted_outcome,
};
use crate::evidence_digest::canonical_json_sha256;
use crate::musical_inference::MusicAction;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_music_theory::Score;

pub const COGNITIVE_SELECTION_POLICY_VERSION: &str = "cognitive-alternative-selection-v1";
pub const COGNITIVE_SCORE_COMMIT_PLAN_VERSION: &str = "cognitive-score-commit-plan-v1";

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

/// Explicit evidence for the opt-in cognitive influence boundary.
///
/// The formal proposal remains authoritative about *what obligation is being
/// served*. This record only allows the FEP source action to express a desired
/// effect while choosing among alternatives that already survived theory,
/// Preserve-contract, and formal-obligation gates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveAlternativeSelectionV1 {
    pub policy_version: String,
    pub source_action: MusicAction,
    pub formal_proposal_action: SymbolicAction,
    pub desired_outcome: PredictedMusicalOutcome,
    pub selection: SymbolicAlternativeSelection,
}

/// A theory-generated score kept inseparable from the evidence used to rank it.
///
/// This closes a subtle provenance hole: callers cannot pass one slice of
/// alternative evidence and a separate map of scores that merely happen to use
/// the same labels. The score and evidence enter the commit planner as one bound
/// object and receive independent canonical digests.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveScoreAlternativeV1 {
    pub evidence: SymbolicAlternativeEvidence,
    pub score: Score,
}

/// Immutable identity record retained for every candidate in one commit plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CognitiveCandidateCommitmentV1 {
    pub alternative_id: String,
    pub score_sha256: String,
    pub evidence_sha256: String,
}

/// Auditable shadow-vs-cognitive score selection without mutating product state.
///
/// `shadow_selection` uses the existing formal prediction target. The cognitive
/// arm uses the real FEP source action through [`select_symbolic_alternative_from_inference`].
/// Both arms see the exact same bound candidate set. A later Studio integration
/// may commit `cognitive_selected_id`, but this type itself is evidence-only.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CognitiveScoreCommitPlanV1 {
    pub plan_version: String,
    pub candidate_set_sha256: String,
    pub candidates: Vec<CognitiveCandidateCommitmentV1>,
    pub shadow_selection: SymbolicAlternativeSelection,
    pub cognitive_selection: CognitiveAlternativeSelectionV1,
    pub shadow_selected_id: String,
    pub shadow_score_sha256: String,
    pub cognitive_selected_id: String,
    pub cognitive_score_sha256: String,
    pub score_identity_diverged: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CognitiveScoreCommitPlanError {
    EmptyAlternativeId,
    NonCanonicalAlternativeId(String),
    DuplicateAlternativeId(String),
    NoShadowRecommendation,
    NoCognitiveRecommendation,
    RecommendedAlternativeMissing(String),
    CanonicalizationFailed(String),
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

/// Select with a preference derived from the *actual* FEP source action.
///
/// This is deliberately opt-in. `trace.proposal.action` remains the formal
/// action selected after obligation/goal arbitration; cognition cannot use
/// this function to bypass that action, make an invalid candidate eligible,
/// or outrank overdue formal promises. It can only decide which eligible
/// alternative best matches the effect implied by its own source action.
pub fn select_symbolic_alternative_from_inference(
    trace: &CognitiveDecisionTrace,
    alternatives: &[SymbolicAlternativeEvidence],
) -> CognitiveAlternativeSelectionV1 {
    let desired_outcome = cognitive_target_for_source_action(trace.inference.source_action);
    let mut cognitive_trace = trace.clone();
    cognitive_trace.predicted_outcome = desired_outcome;
    let mut selection = select_symbolic_alternative(&cognitive_trace, alternatives);
    selection.rationale.insert(
        0,
        format!(
            "cognitive preference derived from {:?}; formal proposal {:?} remains authoritative",
            trace.inference.source_action, trace.proposal.action
        ),
    );

    CognitiveAlternativeSelectionV1 {
        policy_version: COGNITIVE_SELECTION_POLICY_VERSION.into(),
        source_action: trace.inference.source_action,
        formal_proposal_action: trace.proposal.action,
        desired_outcome,
        selection,
    }
}

/// Build a non-mutating score commit plan over one exact bound candidate set.
///
/// The function deliberately computes the shadow and cognitive arms in the
/// same call so they cannot silently see different candidate sets. Candidate
/// commitments bind both the concrete score and the ranking evidence. The
/// resulting boolean means only that the selected symbolic score identity
/// changed; it says nothing about musical quality or listener preference.
pub fn plan_cognitive_score_commit(
    trace: &CognitiveDecisionTrace,
    alternatives: &[CognitiveScoreAlternativeV1],
) -> Result<CognitiveScoreCommitPlanV1, CognitiveScoreCommitPlanError> {
    let mut seen = BTreeSet::new();
    let mut commitments = Vec::with_capacity(alternatives.len());
    let mut evidence = Vec::with_capacity(alternatives.len());

    for alternative in alternatives {
        let raw_id = &alternative.evidence.alternative_id;
        let id = raw_id.trim();
        if id.is_empty() {
            return Err(CognitiveScoreCommitPlanError::EmptyAlternativeId);
        }
        if id != raw_id {
            return Err(CognitiveScoreCommitPlanError::NonCanonicalAlternativeId(
                raw_id.clone(),
            ));
        }
        if !seen.insert(id.to_owned()) {
            return Err(CognitiveScoreCommitPlanError::DuplicateAlternativeId(
                id.to_owned(),
            ));
        }
        let score_sha256 = canonical_json_sha256(&alternative.score).map_err(|error| {
            CognitiveScoreCommitPlanError::CanonicalizationFailed(error.to_string())
        })?;
        let evidence_sha256 = canonical_json_sha256(&alternative.evidence).map_err(|error| {
            CognitiveScoreCommitPlanError::CanonicalizationFailed(error.to_string())
        })?;
        commitments.push(CognitiveCandidateCommitmentV1 {
            alternative_id: id.to_owned(),
            score_sha256,
            evidence_sha256,
        });
        evidence.push(alternative.evidence.clone());
    }

    commitments.sort_by(|left, right| left.alternative_id.cmp(&right.alternative_id));
    let candidate_set_sha256 = canonical_json_sha256(&commitments).map_err(|error| {
        CognitiveScoreCommitPlanError::CanonicalizationFailed(error.to_string())
    })?;

    let shadow_selection = select_symbolic_alternative(trace, &evidence);
    let cognitive_selection = select_symbolic_alternative_from_inference(trace, &evidence);
    let shadow_selected_id = shadow_selection
        .recommended_id
        .clone()
        .ok_or(CognitiveScoreCommitPlanError::NoShadowRecommendation)?;
    let cognitive_selected_id = cognitive_selection
        .selection
        .recommended_id
        .clone()
        .ok_or(CognitiveScoreCommitPlanError::NoCognitiveRecommendation)?;

    let score_hash_for = |id: &str| {
        commitments
            .iter()
            .find(|candidate| candidate.alternative_id == id)
            .map(|candidate| candidate.score_sha256.clone())
            .ok_or_else(|| CognitiveScoreCommitPlanError::RecommendedAlternativeMissing(id.into()))
    };
    let shadow_score_sha256 = score_hash_for(&shadow_selected_id)?;
    let cognitive_score_sha256 = score_hash_for(&cognitive_selected_id)?;
    let score_identity_diverged = shadow_score_sha256 != cognitive_score_sha256;

    Ok(CognitiveScoreCommitPlanV1 {
        plan_version: COGNITIVE_SCORE_COMMIT_PLAN_VERSION.into(),
        candidate_set_sha256,
        candidates: commitments,
        shadow_selection,
        cognitive_selection,
        shadow_selected_id,
        shadow_score_sha256,
        cognitive_selected_id,
        cognitive_score_sha256,
        score_identity_diverged,
    })
}

/// Effect target corresponding to the FEP action before formal goal/obligation
/// arbitration. The mapping intentionally mirrors the bridge's no-goal,
/// no-obligation fallback and is regression-tested against that bridge so the
/// two semantics cannot silently drift.
pub fn cognitive_target_for_source_action(source_action: MusicAction) -> PredictedMusicalOutcome {
    default_predicted_outcome(symbolic_action_for_source(source_action))
}

fn symbolic_action_for_source(source_action: MusicAction) -> SymbolicAction {
    match source_action {
        MusicAction::FollowHarmony => SymbolicAction::Maintain,
        MusicAction::ChromaticExplore => SymbolicAction::IncreaseHarmonicInstability,
        MusicAction::RepeatMotif => SymbolicAction::DevelopMotif,
        MusicAction::ModulateKey => SymbolicAction::ModulateToRelatedKey,
        MusicAction::IncreaseComplexity => SymbolicAction::IncreaseDensity,
        MusicAction::ResolveTension => SymbolicAction::StrengthenCadence,
        MusicAction::AddCountermelody => SymbolicAction::AddCounterline,
        MusicAction::Maintain => SymbolicAction::Maintain,
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
        SymbolicActionProposal, SymbolicMusicObservation, propose_symbolic_action,
    };
    use crate::musical_inference::{MusicAction, MusicInferenceResult};
    use symthaea_music_theory::{Key, PitchClass, ScoreCognitiveProfile};

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

    fn alternative_for_outcome(
        id: &str,
        outcome: PredictedMusicalOutcome,
        valid: bool,
        preserved: bool,
        overdue: usize,
        pressure: f32,
    ) -> SymbolicAlternativeEvidence {
        let baseline = ScoreCognitiveProfile {
            tension: 0.5,
            density: 0.5,
            familiarity: 0.5,
            tonal_displacement: 0.5,
            ..ScoreCognitiveProfile::default()
        };
        let candidate = ScoreCognitiveProfile {
            tension: baseline.tension + outcome.tension_delta,
            density: baseline.density + outcome.density_delta,
            familiarity: baseline.familiarity + outcome.familiarity_delta,
            tonal_displacement: baseline.tonal_displacement + outcome.tonal_displacement_delta,
            ..baseline
        };
        SymbolicAlternativeEvidence {
            alternative_id: id.into(),
            measurement: SymbolicMeasurementEvidence::new(baseline, candidate),
            hard_constraints_valid: valid,
            preserved_invariants: preserved,
            overdue_obligations_remaining: overdue,
            unresolved_obligations_remaining: overdue,
            obligation_pressure_remaining: pressure,
            target_obligation_verified: Some(true),
            motif_return_similarity: Some(1.0),
        }
    }

    fn return_trace(source_action: MusicAction) -> CognitiveDecisionTrace {
        let mut value = trace();
        value.inference.source_action = source_action;
        value.proposal.action = SymbolicAction::ReturnOpeningMaterial;
        value.predicted_outcome = default_predicted_outcome(SymbolicAction::ReturnOpeningMaterial);
        value
    }

    fn inference(action: MusicAction) -> MusicInferenceResult {
        MusicInferenceResult {
            action,
            free_energy: 0.2,
            prediction_error: 0.2,
            surprise: 0.1,
            is_surprised: false,
            learning_rate_mod: 1.0,
            sensory_precision: 1.0,
            prior_precision: 1.0,
        }
    }

    fn score(tempo_bpm: f32) -> Score {
        Score::new(Key::major(PitchClass::C), tempo_bpm, 4)
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

    #[test]
    fn source_cognition_can_change_choice_without_changing_formal_action() {
        let trace = return_trace(MusicAction::IncreaseComplexity);
        let formal_return = default_predicted_outcome(SymbolicAction::ReturnOpeningMaterial);
        let denser = default_predicted_outcome(SymbolicAction::IncreaseDensity);
        let alternatives = [
            alternative_for_outcome("formal-return", formal_return, true, true, 0, 0.0),
            alternative_for_outcome("cognition-dense", denser, true, true, 0, 0.0),
        ];

        let formal_selection = select_symbolic_alternative(&trace, &alternatives);
        assert_eq!(
            formal_selection.recommended_id.as_deref(),
            Some("formal-return")
        );

        let cognitive = select_symbolic_alternative_from_inference(&trace, &alternatives);
        assert_eq!(
            cognitive.formal_proposal_action,
            SymbolicAction::ReturnOpeningMaterial
        );
        assert_eq!(cognitive.source_action, MusicAction::IncreaseComplexity);
        assert_eq!(
            cognitive.selection.recommended_id.as_deref(),
            Some("cognition-dense")
        );
    }

    #[test]
    fn cognition_cannot_make_invalid_candidate_eligible() {
        let trace = return_trace(MusicAction::IncreaseComplexity);
        let desired = cognitive_target_for_source_action(MusicAction::IncreaseComplexity);
        let fallback = default_predicted_outcome(SymbolicAction::ReturnOpeningMaterial);
        let alternatives = [
            alternative_for_outcome("invalid-perfect", desired, false, true, 0, 0.0),
            alternative_for_outcome("valid-fallback", fallback, true, true, 0, 0.0),
        ];

        let cognitive = select_symbolic_alternative_from_inference(&trace, &alternatives);
        assert_eq!(
            cognitive.selection.recommended_id.as_deref(),
            Some("valid-fallback")
        );
    }

    #[test]
    fn overdue_formal_promise_still_outranks_cognitive_fit() {
        let trace = return_trace(MusicAction::IncreaseComplexity);
        let desired = cognitive_target_for_source_action(MusicAction::IncreaseComplexity);
        let fallback = default_predicted_outcome(SymbolicAction::ReturnOpeningMaterial);
        let alternatives = [
            alternative_for_outcome("cognitive-fit-overdue", desired, true, true, 1, 1.0),
            alternative_for_outcome("formal-first", fallback, true, true, 0, 0.0),
        ];

        let cognitive = select_symbolic_alternative_from_inference(&trace, &alternatives);
        assert_eq!(
            cognitive.selection.recommended_id.as_deref(),
            Some("formal-first")
        );
    }

    #[test]
    fn score_commit_plan_binds_shadow_and_cognitive_choices_to_content() {
        let trace = return_trace(MusicAction::IncreaseComplexity);
        let formal_return = default_predicted_outcome(SymbolicAction::ReturnOpeningMaterial);
        let denser = default_predicted_outcome(SymbolicAction::IncreaseDensity);
        let candidates = [
            CognitiveScoreAlternativeV1 {
                evidence: alternative_for_outcome(
                    "formal-return",
                    formal_return,
                    true,
                    true,
                    0,
                    0.0,
                ),
                score: score(120.0),
            },
            CognitiveScoreAlternativeV1 {
                evidence: alternative_for_outcome(
                    "cognition-dense",
                    denser,
                    true,
                    true,
                    0,
                    0.0,
                ),
                score: score(121.0),
            },
        ];

        let plan = plan_cognitive_score_commit(&trace, &candidates).unwrap();
        assert_eq!(plan.plan_version, COGNITIVE_SCORE_COMMIT_PLAN_VERSION);
        assert_eq!(plan.shadow_selected_id, "formal-return");
        assert_eq!(plan.cognitive_selected_id, "cognition-dense");
        assert_ne!(plan.shadow_score_sha256, plan.cognitive_score_sha256);
        assert!(plan.score_identity_diverged);
        assert_eq!(plan.candidates.len(), 2);
        assert_eq!(plan.candidate_set_sha256.len(), 64);
    }

    #[test]
    fn different_selected_labels_do_not_imply_score_divergence() {
        let trace = return_trace(MusicAction::IncreaseComplexity);
        let shared_score = score(120.0);
        let candidates = [
            CognitiveScoreAlternativeV1 {
                evidence: alternative_for_outcome(
                    "formal-return",
                    default_predicted_outcome(SymbolicAction::ReturnOpeningMaterial),
                    true,
                    true,
                    0,
                    0.0,
                ),
                score: shared_score.clone(),
            },
            CognitiveScoreAlternativeV1 {
                evidence: alternative_for_outcome(
                    "cognition-dense",
                    default_predicted_outcome(SymbolicAction::IncreaseDensity),
                    true,
                    true,
                    0,
                    0.0,
                ),
                score: shared_score,
            },
        ];

        let plan = plan_cognitive_score_commit(&trace, &candidates).unwrap();
        assert_ne!(plan.shadow_selected_id, plan.cognitive_selected_id);
        assert_eq!(plan.shadow_score_sha256, plan.cognitive_score_sha256);
        assert!(!plan.score_identity_diverged);
    }

    #[test]
    fn candidate_set_commitment_is_order_independent() {
        let trace = return_trace(MusicAction::IncreaseComplexity);
        let formal_return = default_predicted_outcome(SymbolicAction::ReturnOpeningMaterial);
        let denser = default_predicted_outcome(SymbolicAction::IncreaseDensity);
        let left = CognitiveScoreAlternativeV1 {
            evidence: alternative_for_outcome("a", formal_return, true, true, 0, 0.0),
            score: score(120.0),
        };
        let right = CognitiveScoreAlternativeV1 {
            evidence: alternative_for_outcome("b", denser, true, true, 0, 0.0),
            score: score(121.0),
        };

        let first = plan_cognitive_score_commit(&trace, &[left.clone(), right.clone()]).unwrap();
        let second = plan_cognitive_score_commit(&trace, &[right, left]).unwrap();
        assert_eq!(first.candidate_set_sha256, second.candidate_set_sha256);
    }

    #[test]
    fn noncanonical_alternative_ids_are_rejected() {
        let trace = return_trace(MusicAction::IncreaseComplexity);
        let error = plan_cognitive_score_commit(
            &trace,
            &[CognitiveScoreAlternativeV1 {
                evidence: alternative_for_outcome(
                    " padded ",
                    default_predicted_outcome(SymbolicAction::ReturnOpeningMaterial),
                    true,
                    true,
                    0,
                    0.0,
                ),
                score: score(120.0),
            }],
        )
        .unwrap_err();
        assert_eq!(
            error,
            CognitiveScoreCommitPlanError::NonCanonicalAlternativeId(" padded ".into())
        );
    }

    #[test]
    fn duplicate_alternative_ids_are_rejected_before_selection() {
        let trace = return_trace(MusicAction::IncreaseComplexity);
        let evidence = alternative_for_outcome(
            "duplicate",
            default_predicted_outcome(SymbolicAction::ReturnOpeningMaterial),
            true,
            true,
            0,
            0.0,
        );
        let error = plan_cognitive_score_commit(
            &trace,
            &[
                CognitiveScoreAlternativeV1 {
                    evidence: evidence.clone(),
                    score: score(120.0),
                },
                CognitiveScoreAlternativeV1 {
                    evidence,
                    score: score(121.0),
                },
            ],
        )
        .unwrap_err();
        assert_eq!(
            error,
            CognitiveScoreCommitPlanError::DuplicateAlternativeId("duplicate".into())
        );
    }

    #[test]
    fn source_action_mapping_matches_bridge_fallback_semantics() {
        for action in [
            MusicAction::FollowHarmony,
            MusicAction::ChromaticExplore,
            MusicAction::RepeatMotif,
            MusicAction::ModulateKey,
            MusicAction::IncreaseComplexity,
            MusicAction::ResolveTension,
            MusicAction::AddCountermelody,
            MusicAction::Maintain,
        ] {
            let observation = SymbolicMusicObservation {
                section: CognitiveSection::Development,
                active_goal: None,
                goal_urgency: 0.0,
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
            };
            let bridge = propose_symbolic_action(&inference(action), observation);
            assert_eq!(bridge.proposal.action, symbolic_action_for_source(action));
            assert_eq!(
                bridge.predicted_outcome,
                cognitive_target_for_source_action(action)
            );
        }
    }
}
