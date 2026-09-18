// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003B0: prove that cognitive effect targets can drive the same musical
//! policy engine used by the live Sonata intervention path.
//!
//! This is intentionally a qualification-only tranche. It changes no Studio
//! endpoint and commits no score. The theorem is narrow: clone the exact formal
//! Sonata policy and replace only its desired effect with the FEP-derived target.
//! The production selector's action identity, channel weights, motif weighting,
//! uncertainty penalty, theory/Preserve gates, obligation ordering, and stable
//! tie-break behavior remain unchanged.

use symthaea_muse::adaptive_prediction::{
    InterventionCalibrationEvidence, InterventionPredictionContext, OutcomeUncertainty,
    PredictionContext, PredictionEvidenceSource, TextureBand,
};
use symthaea_muse::cognitive_bridge::{
    CognitiveSection, ObservedMusicalOutcome, PredictedMusicalOutcome, SymbolicAction,
};
use symthaea_muse::cognitive_selection::cognitive_target_for_source_action;
use symthaea_muse::intervention::{
    InterventionDescriptor, InterventionStrategy, ObligationClass,
};
use symthaea_muse::musical_inference::MusicAction;
use symthaea_muse::musical_policy::{
    MusicalPolicyPreference, PolicyCandidateEvidence, select_by_musical_policy,
};
use symthaea_music_theory::{Key, PitchClass, Score, ScoreValidationConfig, validate_score};

fn cognitive_overlay(
    formal: &MusicalPolicyPreference,
    source_action: MusicAction,
) -> MusicalPolicyPreference {
    let mut cognitive = formal.clone();
    cognitive.desired_outcome = cognitive_target_for_source_action(source_action);
    cognitive
}

fn candidate(
    id: &str,
    predicted: PredictedMusicalOutcome,
    valid: bool,
    overdue: usize,
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
    let empty = Score::new(Key::major(PitchClass::C), 120.0, 4);
    let mut report = validate_score(&empty, &ScoreValidationConfig::default());
    report.valid = valid;
    if valid {
        report.issues.clear();
    }

    PolicyCandidateEvidence {
        alternative_id: id.into(),
        theory_validation: report,
        preserved_invariants: true,
        overdue_obligations_remaining: overdue,
        unresolved_obligations_remaining: overdue,
        obligation_pressure_remaining: if overdue == 0 { 0.0 } else { 1.0 },
        target_obligation_verified: Some(true),
        motif_return_similarity: Some(1.0),
        prediction: InterventionCalibrationEvidence {
            model_version: "mel-003b0-test".into(),
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
        observed_outcome: Some(ObservedMusicalOutcome {
            tension_delta: predicted.tension_delta,
            density_delta: predicted.density_delta,
            familiarity_delta: predicted.familiarity_delta,
            tonal_displacement_delta: predicted.tonal_displacement_delta,
        }),
    }
}

#[test]
fn cognitive_overlay_changes_only_the_desired_effect() {
    let formal = MusicalPolicyPreference::for_action(SymbolicAction::ReturnOpeningMaterial);
    let cognitive = cognitive_overlay(&formal, MusicAction::IncreaseComplexity);

    assert_eq!(cognitive.action, formal.action);
    assert_eq!(cognitive.policy_version, formal.policy_version);
    assert_eq!(cognitive.channel_weights, formal.channel_weights);
    assert_eq!(
        cognitive.uncertainty_penalty_weight,
        formal.uncertainty_penalty_weight
    );
    assert_eq!(cognitive.motif_identity_weight, formal.motif_identity_weight);
    assert_ne!(cognitive.desired_outcome, formal.desired_outcome);
    assert_eq!(
        cognitive.desired_outcome,
        cognitive_target_for_source_action(MusicAction::IncreaseComplexity)
    );
}

#[test]
fn production_policy_can_change_choice_when_only_the_cognitive_target_changes() {
    let formal_policy = MusicalPolicyPreference::for_action(SymbolicAction::ReturnOpeningMaterial);
    let cognitive_policy = cognitive_overlay(&formal_policy, MusicAction::IncreaseComplexity);
    let candidates = [
        candidate("formal-return", formal_policy.desired_outcome, true, 0),
        candidate("cognitive-density", cognitive_policy.desired_outcome, true, 0),
    ];

    let shadow = select_by_musical_policy(formal_policy, &candidates);
    let cognitive = select_by_musical_policy(cognitive_policy, &candidates);

    assert_eq!(shadow.recommended_id.as_deref(), Some("formal-return"));
    assert_eq!(
        cognitive.recommended_id.as_deref(),
        Some("cognitive-density")
    );
}

#[test]
fn production_policy_still_rejects_an_invalid_cognitive_match() {
    let formal_policy = MusicalPolicyPreference::for_action(SymbolicAction::ReturnOpeningMaterial);
    let cognitive_policy = cognitive_overlay(&formal_policy, MusicAction::IncreaseComplexity);
    let candidates = [
        candidate("formal-valid", formal_policy.desired_outcome, true, 0),
        candidate(
            "cognitive-invalid",
            cognitive_policy.desired_outcome,
            false,
            0,
        ),
    ];

    let cognitive = select_by_musical_policy(cognitive_policy, &candidates);
    assert_eq!(cognitive.recommended_id.as_deref(), Some("formal-valid"));
}

#[test]
fn overdue_formal_promises_still_outrank_cognitive_utility() {
    let formal_policy = MusicalPolicyPreference::for_action(SymbolicAction::ReturnOpeningMaterial);
    let cognitive_policy = cognitive_overlay(&formal_policy, MusicAction::IncreaseComplexity);
    let candidates = [
        candidate("formal-responsible", formal_policy.desired_outcome, true, 0),
        candidate(
            "cognitive-fit-overdue",
            cognitive_policy.desired_outcome,
            true,
            1,
        ),
    ];

    let cognitive = select_by_musical_policy(cognitive_policy, &candidates);
    assert_eq!(
        cognitive.recommended_id.as_deref(),
        Some("formal-responsible")
    );
}
