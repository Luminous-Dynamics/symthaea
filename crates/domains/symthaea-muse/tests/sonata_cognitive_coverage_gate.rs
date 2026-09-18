// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003B4: bind the generic measured effect-coverage gate to a real Sonata
//! intervention batch and require cognition to remain shadow-only when the
//! requested dominant effect is absent from the formal candidate frontier.

use symthaea_muse::cognitive_bridge::{
    ActionScope, CognitiveDecisionTrace, CognitiveObligationDemand, CognitiveSection,
    InferenceEvidence, PredictedMusicalOutcome, PreserveInvariant, SymbolicAction,
    SymbolicActionProposal, SymbolicMusicObservation,
};
use symthaea_muse::cognitive_selection::cognitive_target_for_source_action;
use symthaea_muse::musical_inference::MusicAction;
use symthaea_muse::musical_policy::{
    EffectCoverageDisposition, MusicalPolicySelection, OutcomeChannel, PolicyCandidateEvidence,
    assess_effect_coverage, select_by_musical_policy,
};
use symthaea_muse::sonata_intervention::{
    SonataInterventionBatch, generate_and_rank_sonata_return,
};
use symthaea_music_theory::{
    MusicalIntent, ObligationKind, SonataRealization, Style, compose_sonata_with_plan,
};

fn return_trace(realization: &SonataRealization) -> CognitiveDecisionTrace {
    let obligation = realization
        .plan
        .obligations
        .items()
        .iter()
        .find(|item| {
            matches!(
                &item.kind,
                ObligationKind::ReturnMotif { motif_id, .. } if motif_id == "sonata.primary"
            )
        })
        .unwrap();

    CognitiveDecisionTrace {
        observation: SymbolicMusicObservation {
            section: CognitiveSection::Recapitulation,
            active_goal: None,
            goal_urgency: 1.0,
            valence: 0.0,
            arousal: 0.5,
            prediction_error: 0.3,
            consciousness_level: 0.5,
            dominant_harmony: 0,
            dominant_harmony_activation: 0.8,
            pending_obligations: 1,
            overdue_obligations: vec![obligation.id],
            obligation_demands: vec![CognitiveObligationDemand {
                id: obligation.id,
                priority: obligation.priority,
                due_by: obligation.due_by,
                overdue: true,
                kind: obligation.kind.clone(),
            }],
            obligation_pressure: 1.0,
        },
        inference: InferenceEvidence {
            source_action: MusicAction::IncreaseComplexity,
            free_energy: 0.2,
            prediction_error: 0.3,
            surprise: 0.2,
            sensory_precision: 1.0,
            prior_precision: 1.0,
        },
        proposal: SymbolicActionProposal {
            action: SymbolicAction::ReturnOpeningMaterial,
            driving_obligation_id: Some(obligation.id),
            supporting_obligation_ids: vec![obligation.id],
            deferred_obligation_ids: Vec::new(),
            scope: ActionScope::CurrentSection,
            preserve: vec![
                PreserveInvariant::MotifIdentity,
                PreserveInvariant::Meter,
                PreserveInvariant::FormLength,
                PreserveInvariant::Ending,
            ],
            urgency: 1.0,
            confidence: 0.7,
            rationale: vec!["restore the primary subject".into()],
        },
        predicted_outcome: PredictedMusicalOutcome {
            tension_delta: -0.2,
            density_delta: 0.0,
            familiarity_delta: 0.5,
            tonal_displacement_delta: -0.35,
        },
    }
}

fn replay_evidence(batch: &SonataInterventionBatch) -> Vec<PolicyCandidateEvidence> {
    batch
        .candidates
        .iter()
        .map(|candidate| {
            let assessment = batch
                .selection
                .assessments
                .iter()
                .find(|assessment| assessment.alternative_id == candidate.alternative_id)
                .expect("every generated Sonata candidate has a policy assessment");
            PolicyCandidateEvidence {
                alternative_id: candidate.alternative_id.clone(),
                theory_validation: candidate.theory_validation.clone(),
                preserved_invariants: candidate.preserved_invariants,
                overdue_obligations_remaining: assessment.overdue_obligations_remaining,
                unresolved_obligations_remaining: assessment.unresolved_obligations_remaining,
                obligation_pressure_remaining: assessment.obligation_pressure_remaining,
                target_obligation_verified: assessment.target_obligation_verified,
                motif_return_similarity: assessment.motif_return_similarity,
                prediction: candidate.prediction.clone(),
                observed_outcome: Some(candidate.measurement.observed_outcome),
            }
        })
        .collect()
}

fn gated_selection(
    shadow: &MusicalPolicySelection,
    cognitive: MusicalPolicySelection,
    coverage: EffectCoverageDisposition,
) -> MusicalPolicySelection {
    match coverage {
        EffectCoverageDisposition::CognitiveEligible => cognitive,
        EffectCoverageDisposition::NoRequestedEffect
        | EffectCoverageDisposition::ShadowFallbackInvalidRequest
        | EffectCoverageDisposition::ShadowFallbackInsufficientAlternatives
        | EffectCoverageDisposition::ShadowFallbackUnsupportedDominant => shadow.clone(),
    }
}

#[test]
fn real_sonata_increase_complexity_remains_shadow_only_without_density_coverage() {
    let intent = MusicalIntent {
        seed: 71,
        bars: 4,
        ..MusicalIntent::default()
    };
    let spec = Style::Sonata.spec();
    let realization = compose_sonata_with_plan(&intent, &spec).unwrap();
    let trace = return_trace(&realization);
    let batch = generate_and_rank_sonata_return(&realization, &trace).unwrap();
    let evidence = replay_evidence(&batch);

    let mut cognitive_policy = batch.selection.policy.clone();
    cognitive_policy.desired_outcome =
        cognitive_target_for_source_action(trace.inference.source_action);

    let coverage = assess_effect_coverage(&cognitive_policy, &evidence, 0.01);
    assert_eq!(coverage.dominant_channel, Some(OutcomeChannel::Density));
    assert_ne!(
        coverage.disposition,
        EffectCoverageDisposition::CognitiveEligible,
        "pitch-only Sonata returns must not receive authority for an unsupported density target"
    );

    let density = coverage
        .channels
        .iter()
        .find(|channel| channel.channel == OutcomeChannel::Density)
        .unwrap();
    assert_eq!(density.observed_span, Some(0.0));
    assert_eq!(density.directional_reach, Some(0.0));
    assert!(!density.supported);

    let cognitive = select_by_musical_policy(cognitive_policy, &evidence);
    let committed = gated_selection(&batch.selection, cognitive, coverage.disposition);
    assert_eq!(
        committed, batch.selection,
        "unsupported cognitive targets must preserve the exact production shadow selection"
    );
}

#[test]
fn effect_coverage_is_computed_from_the_same_replayed_formal_frontier() {
    let intent = MusicalIntent {
        seed: 73,
        bars: 4,
        ..MusicalIntent::default()
    };
    let spec = Style::Sonata.spec();
    let realization = compose_sonata_with_plan(&intent, &spec).unwrap();
    let trace = return_trace(&realization);
    let batch = generate_and_rank_sonata_return(&realization, &trace).unwrap();
    let evidence = replay_evidence(&batch);

    let replayed = select_by_musical_policy(batch.selection.policy.clone(), &evidence);
    assert_eq!(replayed, batch.selection);

    let mut cognitive_policy = batch.selection.policy.clone();
    cognitive_policy.desired_outcome =
        cognitive_target_for_source_action(trace.inference.source_action);
    let coverage = assess_effect_coverage(&cognitive_policy, &evidence, 0.01);

    for id in &coverage.formal_frontier_ids {
        let assessment = batch
            .selection
            .assessments
            .iter()
            .find(|assessment| &assessment.alternative_id == id)
            .expect("coverage frontier ids must come from the exact production batch");
        assert!(assessment.eligible);
    }
    assert!(coverage.measured_frontier_alternatives > 0);
}
