// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003B2: characterize whether the current Sonata-return candidate family
//! can actually express the primary effect requested by an FEP source action.
//!
//! This is a limitation ratchet, not a quality test. A cognitive target should
//! not be treated as product control merely because its policy ordering changes
//! if the generated candidate family has no meaningful variation on that
//! target's dominant symbolic channel.

use symthaea_muse::cognitive_bridge::{
    ActionScope, CognitiveDecisionTrace, CognitiveObligationDemand, CognitiveSection,
    InferenceEvidence, PredictedMusicalOutcome, PreserveInvariant, SymbolicAction,
    SymbolicActionProposal, SymbolicMusicObservation,
};
use symthaea_muse::cognitive_selection::cognitive_target_for_source_action;
use symthaea_muse::musical_inference::MusicAction;
use symthaea_muse::sonata_intervention::generate_and_rank_sonata_return;
use symthaea_music_theory::{
    MusicalIntent, ObligationKind, SonataRealization, Style, compose_sonata_with_plan,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EffectChannel {
    Tension,
    Density,
    Familiarity,
    TonalDisplacement,
}

fn dominant_channel(target: PredictedMusicalOutcome) -> EffectChannel {
    let channels = [
        (EffectChannel::Tension, target.tension_delta.abs()),
        (EffectChannel::Density, target.density_delta.abs()),
        (EffectChannel::Familiarity, target.familiarity_delta.abs()),
        (
            EffectChannel::TonalDisplacement,
            target.tonal_displacement_delta.abs(),
        ),
    ];
    channels
        .into_iter()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .unwrap()
        .0
}

fn channel_value(
    outcome: symthaea_muse::cognitive_bridge::ObservedMusicalOutcome,
    channel: EffectChannel,
) -> f32 {
    match channel {
        EffectChannel::Tension => outcome.tension_delta,
        EffectChannel::Density => outcome.density_delta,
        EffectChannel::Familiarity => outcome.familiarity_delta,
        EffectChannel::TonalDisplacement => outcome.tonal_displacement_delta,
    }
}

fn return_trace(realization: &SonataRealization, source_action: MusicAction) -> CognitiveDecisionTrace {
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
            source_action,
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

#[test]
fn increase_complexity_targets_density_but_current_return_candidates_have_zero_density_span() {
    let intent = MusicalIntent {
        seed: 59,
        bars: 4,
        ..MusicalIntent::default()
    };
    let spec = Style::Sonata.spec();
    let realization = compose_sonata_with_plan(&intent, &spec).unwrap();
    let trace = return_trace(&realization, MusicAction::IncreaseComplexity);
    let target = cognitive_target_for_source_action(trace.inference.source_action);
    assert_eq!(dominant_channel(target), EffectChannel::Density);

    let batch = generate_and_rank_sonata_return(&realization, &trace).unwrap();
    let density_values: Vec<f32> = batch
        .candidates
        .iter()
        .map(|candidate| candidate.measurement.observed_outcome.density_delta)
        .collect();
    let min = density_values
        .iter()
        .copied()
        .min_by(f32::total_cmp)
        .unwrap();
    let max = density_values
        .iter()
        .copied()
        .max_by(f32::total_cmp)
        .unwrap();

    assert_eq!(min, 0.0);
    assert_eq!(max, 0.0);
    assert_eq!(max - min, 0.0);
}

#[test]
fn pitch_only_return_strategies_preserve_onset_density_across_the_candidate_set() {
    let intent = MusicalIntent {
        seed: 61,
        bars: 4,
        ..MusicalIntent::default()
    };
    let spec = Style::Sonata.spec();
    let realization = compose_sonata_with_plan(&intent, &spec).unwrap();
    let trace = return_trace(&realization, MusicAction::IncreaseComplexity);
    let batch = generate_and_rank_sonata_return(&realization, &trace).unwrap();

    let baseline = &batch.candidates[0].measurement.baseline;
    for candidate in &batch.candidates {
        assert_eq!(candidate.measurement.candidate.onset_count, baseline.onset_count);
        assert_eq!(
            candidate.measurement.candidate.notes_per_beat,
            baseline.notes_per_beat
        );
        assert_eq!(candidate.measurement.observed_outcome.density_delta, 0.0);
    }
}

#[test]
fn candidate_effect_coverage_must_be_measured_on_content_not_candidate_labels() {
    let intent = MusicalIntent {
        seed: 67,
        bars: 4,
        ..MusicalIntent::default()
    };
    let spec = Style::Sonata.spec();
    let realization = compose_sonata_with_plan(&intent, &spec).unwrap();
    let trace = return_trace(&realization, MusicAction::IncreaseComplexity);
    let batch = generate_and_rank_sonata_return(&realization, &trace).unwrap();
    assert!(batch.candidates.len() > 1);

    let channel = dominant_channel(cognitive_target_for_source_action(
        trace.inference.source_action,
    ));
    let values: Vec<f32> = batch
        .candidates
        .iter()
        .map(|candidate| channel_value(candidate.measurement.observed_outcome, channel))
        .collect();

    assert!(
        values.windows(2).all(|pair| pair[0] == pair[1]),
        "multiple named alternatives do not imply coverage of the requested effect channel"
    );
}
