// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003B5: test-only probe for the smallest bounded Sonata transformation
//! that can genuinely increase the measured density channel without touching
//! the returning melody or changing harmonic pitch content.

use std::collections::BTreeMap;
use symthaea_muse::cognitive_bridge::{
    ActionScope, CognitiveDecisionTrace, CognitiveObligationDemand, CognitiveSection,
    InferenceEvidence, PredictedMusicalOutcome, PreserveInvariant, SymbolicAction,
    SymbolicActionProposal, SymbolicMeasurementEvidence, SymbolicMusicObservation,
};
use symthaea_muse::cognitive_selection::cognitive_target_for_source_action;
use symthaea_muse::musical_inference::MusicAction;
use symthaea_muse::musical_policy::{
    EffectCoverageDisposition, OutcomeChannel, PolicyCandidateEvidence, assess_effect_coverage,
};
use symthaea_muse::sonata_intervention::{
    SonataInterventionBatch, generate_and_rank_sonata_return,
};
use symthaea_music_theory::{
    Duration, MusicalIntent, ObligationKind, Score, ScoreValidationConfig, SonataRealization,
    SonataSectionKind, Style, VoiceRole, compose_sonata_with_plan, profile_score_region,
    validate_score, verify_sonata_obligations,
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

fn should_rearticulate(
    note: &symthaea_music_theory::ScoreNote,
    start: Duration,
    end: Duration,
) -> bool {
    let in_target = note.onset.beats() >= start.beats() && note.onset.beats() < end.beats();
    let accompaniment = matches!(note.role, VoiceRole::Harmony | VoiceRole::Bass);
    if !in_target || !accompaniment || note.duration.beats() < 1.0 {
        return false;
    }
    let split_onset = note.onset + note.duration.scale(1, 2);
    split_onset.beats() < end.beats()
}

/// Re-articulate accompaniment events inside the target section without
/// changing pitch content or total sounding duration. A note long enough to
/// subdivide becomes two consecutive same-pitch notes whose exact rational
/// durations sum to the original duration. The second attack must itself land
/// inside the target, so every counted transformation can affect target density.
fn rearticulate_accompaniment(score: &Score, start: Duration, end: Duration) -> Score {
    let mut candidate = Score::new(score.key, score.tempo_bpm, score.meter);
    for note in &score.notes {
        if should_rearticulate(note, start, end) {
            let first_duration = note.duration.scale(1, 2);
            let second_duration = note.duration.saturating_sub(first_duration);
            let mut first = *note;
            first.duration = first_duration;
            let mut second = *note;
            second.onset = note.onset + first_duration;
            second.duration = second_duration;
            candidate.push(first);
            candidate.push(second);
        } else {
            candidate.push(*note);
        }
    }
    candidate
}

fn preserves_bounded_region_contract(
    baseline: &Score,
    candidate: &Score,
    start: Duration,
    end: Duration,
) -> bool {
    if baseline.voice(VoiceRole::Melody) != candidate.voice(VoiceRole::Melody) {
        return false;
    }
    // Construction walks the baseline event vector in order and only expands
    // accompaniment notes inside the target, so outside-target order is itself
    // part of the preservation contract; no post-hoc identity inference is needed.
    let outside = |score: &Score| {
        score
            .notes
            .iter()
            .copied()
            .filter(|note| {
                note.onset.beats() < start.beats() || note.onset.beats() >= end.beats()
            })
            .collect::<Vec<_>>()
    };
    baseline.key == candidate.key
        && baseline.tempo_bpm == candidate.tempo_bpm
        && baseline.meter == candidate.meter
        && baseline.total_beats == candidate.total_beats
        && outside(baseline) == outside(candidate)
}

/// Exact accompaniment pitch-duration mass for notes whose attacks lie inside
/// the target. Splitting a note must preserve this map exactly: same role, same
/// symbolic pitch, same total rational duration.
fn accompaniment_pitch_duration_mass(
    score: &Score,
    start: Duration,
    end: Duration,
) -> BTreeMap<(u8, u8), Duration> {
    let mut mass = BTreeMap::new();
    for note in &score.notes {
        if note.onset.beats() < start.beats() || note.onset.beats() >= end.beats() {
            continue;
        }
        let role = match note.role {
            VoiceRole::Harmony => 0u8,
            VoiceRole::Bass => 1u8,
            _ => continue,
        };
        mass.entry((role, note.pitch.midi()))
            .and_modify(|duration| *duration = *duration + note.duration)
            .or_insert(note.duration);
    }
    mass
}

fn baseline_policy_evidence(batch: &SonataInterventionBatch) -> PolicyCandidateEvidence {
    let candidate = batch
        .candidates
        .iter()
        .find(|candidate| candidate.alternative_id == "current-score")
        .expect("Sonata batch must retain current-score baseline");
    let assessment = batch
        .selection
        .assessments
        .iter()
        .find(|assessment| assessment.alternative_id == candidate.alternative_id)
        .unwrap();
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
}

fn validation_config() -> ScoreValidationConfig {
    ScoreValidationConfig {
        max_melodic_leap_semitones: 24,
        check_strong_beat_consonance: false,
        check_parallel_perfect_motion: false,
        ..ScoreValidationConfig::default()
    }
}

#[test]
fn accompaniment_rearticulation_can_open_real_density_coverage_without_touching_the_melody() {
    let intent = MusicalIntent {
        seed: 79,
        bars: 4,
        ..MusicalIntent::default()
    };
    let spec = Style::Sonata.spec();
    let realization = compose_sonata_with_plan(&intent, &spec).unwrap();
    let trace = return_trace(&realization);
    let batch = generate_and_rank_sonata_return(&realization, &trace).unwrap();
    let target = realization
        .plan
        .sections
        .iter()
        .find(|section| section.kind == SonataSectionKind::RecapitulationPrimary)
        .unwrap();

    let split_count = realization
        .score
        .notes
        .iter()
        .filter(|note| should_rearticulate(note, target.start, target.end))
        .count();
    assert!(split_count > 0, "the probe needs at least one bounded accompaniment split");

    let enriched = rearticulate_accompaniment(&realization.score, target.start, target.end);
    assert_eq!(
        enriched.notes.len(),
        realization.score.notes.len() + split_count,
        "each bounded re-articulation may add exactly one attack and no more"
    );
    assert!(
        preserves_bounded_region_contract(
            &realization.score,
            &enriched,
            target.start,
            target.end
        ),
        "the density probe may alter accompaniment articulation only inside the target section"
    );
    assert_eq!(
        accompaniment_pitch_duration_mass(&realization.score, target.start, target.end),
        accompaniment_pitch_duration_mass(&enriched, target.start, target.end),
        "re-articulation must preserve exact accompaniment pitch-duration mass"
    );

    let theory_validation = validate_score(&enriched, &validation_config());
    assert!(
        theory_validation.valid,
        "bounded accompaniment rearticulation must remain canonically theory-valid: {:?}",
        theory_validation.issues
    );

    let (resolution, verification) = verify_sonata_obligations(&enriched, &realization.plan);
    let obligation_id = trace.proposal.driving_obligation_id.unwrap();
    let target_verified = verification
        .iter()
        .find(|evidence| evidence.obligation_id == obligation_id)
        .is_some_and(|evidence| evidence.verified);
    assert!(target_verified, "the primary-return obligation must remain verified");

    let baseline_profile =
        profile_score_region(&realization.score, target.start, target.end).unwrap();
    let enriched_profile = profile_score_region(&enriched, target.start, target.end).unwrap();
    assert_eq!(
        enriched_profile.onset_count,
        baseline_profile.onset_count + split_count,
        "every bounded split must contribute exactly one new target-region onset"
    );
    assert!(enriched_profile.notes_per_beat > baseline_profile.notes_per_beat);
    assert_eq!(enriched_profile.active_voice_count, baseline_profile.active_voice_count);

    let measurement = SymbolicMeasurementEvidence::new(baseline_profile, enriched_profile);
    assert!(
        measurement.observed_outcome.density_delta > 0.0,
        "re-articulation must create a real measured density increase"
    );
    // These channels are analytically invariant under this transformation:
    // familiarity observes Melody/CounterMelody interval grammar, while tonal
    // displacement is duration-weighted pitch-class mass. Neither changes.
    assert_eq!(measurement.observed_outcome.familiarity_delta, 0.0);
    assert!(measurement.observed_outcome.tonal_displacement_delta.abs() <= f32::EPSILON);
    // Tension is intentionally not claimed invariant: its current proxy samples
    // vertical onset events and note-count-normalized structural emphasis, so
    // additional accompaniment attacks may cause measurable leakage.
    assert!(measurement.observed_outcome.tension_delta.is_finite());
    assert_eq!(
        realization.score.voice(VoiceRole::Melody),
        enriched.voice(VoiceRole::Melody),
        "the returning melody itself must remain byte-identical"
    );

    let pressure = resolution.pressure_at(enriched.total_beats);
    let baseline = baseline_policy_evidence(&batch);
    // Capability must exist on the same formal tier the production selector
    // can actually choose from; a denser but formally worse candidate is not
    // evidence of controllability.
    assert_eq!(baseline.target_obligation_verified, Some(target_verified));
    assert_eq!(baseline.overdue_obligations_remaining, pressure.overdue_count);
    assert_eq!(
        baseline.unresolved_obligations_remaining,
        resolution.unresolved().len()
    );

    let enriched_evidence = PolicyCandidateEvidence {
        alternative_id: "accompaniment-rearticulation".into(),
        theory_validation,
        preserved_invariants: true,
        overdue_obligations_remaining: pressure.overdue_count,
        unresolved_obligations_remaining: resolution.unresolved().len(),
        obligation_pressure_remaining: pressure.weighted_pressure,
        target_obligation_verified: Some(target_verified),
        motif_return_similarity: baseline.motif_return_similarity,
        // Prediction is intentionally held at the current baseline model here.
        // This probe establishes action-space capability, not prediction quality.
        prediction: baseline.prediction.clone(),
        observed_outcome: Some(measurement.observed_outcome),
    };

    let mut cognitive_policy = batch.selection.policy.clone();
    cognitive_policy.desired_outcome =
        cognitive_target_for_source_action(trace.inference.source_action);
    let coverage = assess_effect_coverage(
        &cognitive_policy,
        &[baseline, enriched_evidence],
        0.01,
    );

    assert_eq!(coverage.dominant_channel, Some(OutcomeChannel::Density));
    assert_eq!(
        coverage.disposition,
        EffectCoverageDisposition::CognitiveEligible,
        "a formally admissible, measured density alternative should open the capability gate"
    );
    let density = coverage
        .channels
        .iter()
        .find(|channel| channel.channel == OutcomeChannel::Density)
        .unwrap();
    assert!(density.observed_span.unwrap_or_default() >= 0.01);
    assert!(density.directional_reach.unwrap_or_default() >= 0.01);
}

#[test]
fn bounded_density_probe_replicates_across_a_fixed_sonata_seed_panel() {
    for seed in [3u64, 11, 23, 41, 59, 79, 97, 127] {
        let intent = MusicalIntent {
            seed,
            bars: 4,
            ..MusicalIntent::default()
        };
        let spec = Style::Sonata.spec();
        let realization = compose_sonata_with_plan(&intent, &spec).unwrap();
        let target = realization
            .plan
            .sections
            .iter()
            .find(|section| section.kind == SonataSectionKind::RecapitulationPrimary)
            .unwrap();
        let split_count = realization
            .score
            .notes
            .iter()
            .filter(|note| should_rearticulate(note, target.start, target.end))
            .count();
        assert!(split_count > 0, "seed {seed}: no bounded accompaniment split available");

        let enriched = rearticulate_accompaniment(&realization.score, target.start, target.end);
        assert!(
            preserves_bounded_region_contract(
                &realization.score,
                &enriched,
                target.start,
                target.end
            ),
            "seed {seed}: preservation contract failed"
        );
        assert_eq!(
            accompaniment_pitch_duration_mass(&realization.score, target.start, target.end),
            accompaniment_pitch_duration_mass(&enriched, target.start, target.end),
            "seed {seed}: pitch-duration mass changed"
        );

        let report = validate_score(&enriched, &validation_config());
        assert!(report.valid, "seed {seed}: theory validation failed: {:?}", report.issues);

        let baseline_profile =
            profile_score_region(&realization.score, target.start, target.end).unwrap();
        let enriched_profile = profile_score_region(&enriched, target.start, target.end).unwrap();
        let measurement = SymbolicMeasurementEvidence::new(baseline_profile, enriched_profile);
        assert!(
            measurement.observed_outcome.density_delta > 0.0,
            "seed {seed}: density did not increase"
        );
        assert_eq!(
            measurement.observed_outcome.familiarity_delta, 0.0,
            "seed {seed}: familiarity leaked"
        );
        assert!(
            measurement.observed_outcome.tonal_displacement_delta.abs() <= f32::EPSILON,
            "seed {seed}: tonal displacement leaked"
        );
        assert!(
            measurement.observed_outcome.tension_delta.is_finite(),
            "seed {seed}: tension leakage was non-finite"
        );
    }
}
