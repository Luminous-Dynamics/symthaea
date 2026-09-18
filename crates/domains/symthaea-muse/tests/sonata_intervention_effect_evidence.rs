// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003B7: bind InterventionEffectEvidenceV1 to a real bounded Sonata
//! accompaniment re-articulation rather than synthetic outcome fixtures.

use symthaea_muse::cognitive_bridge::SymbolicMeasurementEvidence;
use symthaea_muse::cognitive_selection::cognitive_target_for_source_action;
use symthaea_muse::evidence_digest::intervention_effect_evidence::{
    EffectRelationV1, InterventionEffectEvidenceDisposition, RequestRoleV1,
    account_intervention_effects,
};
use symthaea_muse::musical_inference::MusicAction;
use symthaea_muse::musical_policy::OutcomeChannel;
use symthaea_music_theory::{
    Duration, MusicalIntent, Score, SonataSectionKind, Style, VoiceRole, compose_sonata_with_plan,
    profile_score_region,
};

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

fn measured_effect(seed: u64) -> symthaea_muse::cognitive_bridge::ObservedMusicalOutcome {
    let intent = MusicalIntent {
        seed,
        bars: 4,
        ..MusicalIntent::default()
    };
    let realization = compose_sonata_with_plan(&intent, &Style::Sonata.spec()).unwrap();
    let target = realization
        .plan
        .sections
        .iter()
        .find(|section| section.kind == SonataSectionKind::RecapitulationPrimary)
        .unwrap();
    assert!(
        realization
            .score
            .notes
            .iter()
            .any(|note| should_rearticulate(note, target.start, target.end)),
        "seed {seed}: probe requires at least one bounded accompaniment split"
    );

    let enriched = rearticulate_accompaniment(&realization.score, target.start, target.end);
    let baseline = profile_score_region(&realization.score, target.start, target.end).unwrap();
    let candidate = profile_score_region(&enriched, target.start, target.end).unwrap();
    SymbolicMeasurementEvidence::new(baseline, candidate).observed_outcome
}

fn channel(
    evidence: &symthaea_muse::evidence_digest::intervention_effect_evidence::InterventionEffectEvidenceV1,
    channel: OutcomeChannel,
) -> &symthaea_muse::evidence_digest::intervention_effect_evidence::ChannelInterventionEffectV1 {
    evidence
        .channels
        .iter()
        .find(|entry| entry.channel == channel)
        .expect("all four symbolic outcome channels must be retained")
}

#[test]
fn real_sonata_density_probe_records_primary_secondary_and_preserved_channels_separately() {
    let observed = measured_effect(79);
    let requested = cognitive_target_for_source_action(MusicAction::IncreaseComplexity);
    let evidence = account_intervention_effects(requested, observed, 0.01);

    assert_eq!(
        evidence.disposition,
        InterventionEffectEvidenceDisposition::Valid
    );
    assert_eq!(
        evidence.dominant_requested_channel,
        Some(OutcomeChannel::Density)
    );

    // IncreaseComplexity -> IncreaseDensity currently asks for density +0.35
    // and secondary tension +0.15. The evidence must carry those roles directly.
    let density = channel(&evidence, OutcomeChannel::Density);
    assert_eq!(density.request_role, RequestRoleV1::Dominant);
    assert_eq!(density.relation, EffectRelationV1::RequestedAligned);

    let tension = channel(&evidence, OutcomeChannel::Tension);
    assert_eq!(tension.request_role, RequestRoleV1::Secondary);
    let expected_tension_relation = if observed.tension_delta.abs() <= 0.01 {
        EffectRelationV1::RequestedUnchanged
    } else if observed.tension_delta.is_sign_positive() {
        EffectRelationV1::RequestedAligned
    } else {
        EffectRelationV1::RequestedOpposed
    };
    assert_eq!(tension.relation, expected_tension_relation);

    // These channels are not requested by IncreaseDensity and the bounded
    // re-articulation preserves both in the score-side measurement.
    let familiarity = channel(&evidence, OutcomeChannel::Familiarity);
    assert_eq!(familiarity.request_role, RequestRoleV1::Unrequested);
    assert_eq!(
        familiarity.relation,
        EffectRelationV1::UnrequestedPreserved
    );
    let tonal = channel(&evidence, OutcomeChannel::TonalDisplacement);
    assert_eq!(tonal.request_role, RequestRoleV1::Unrequested);
    assert_eq!(tonal.relation, EffectRelationV1::UnrequestedPreserved);
    assert_eq!(evidence.changed_unrequested_channels, 0);
}

#[test]
fn real_effect_accounting_replicates_across_the_density_probe_seed_panel() {
    let requested = cognitive_target_for_source_action(MusicAction::IncreaseComplexity);
    for seed in [3u64, 11, 23, 41, 59, 79, 97, 127] {
        let observed = measured_effect(seed);
        let evidence = account_intervention_effects(requested, observed, 0.01);
        assert_eq!(
            evidence.disposition,
            InterventionEffectEvidenceDisposition::Valid,
            "seed {seed}: invalid effect accounting"
        );
        assert_eq!(
            evidence.dominant_requested_channel,
            Some(OutcomeChannel::Density),
            "seed {seed}: dominant request drifted"
        );

        let density = channel(&evidence, OutcomeChannel::Density);
        assert_eq!(
            density.request_role,
            RequestRoleV1::Dominant,
            "seed {seed}: density request role drifted"
        );
        assert_eq!(
            density.relation,
            EffectRelationV1::RequestedAligned,
            "seed {seed}: bounded intervention did not align with density request"
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::Tension).request_role,
            RequestRoleV1::Secondary,
            "seed {seed}: tension must remain a secondary request"
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::Familiarity).request_role,
            RequestRoleV1::Unrequested,
            "seed {seed}: familiarity request role drifted"
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::Familiarity).relation,
            EffectRelationV1::UnrequestedPreserved,
            "seed {seed}: familiarity was not preserved"
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::TonalDisplacement).request_role,
            RequestRoleV1::Unrequested,
            "seed {seed}: tonal-displacement request role drifted"
        );
        assert_eq!(
            channel(&evidence, OutcomeChannel::TonalDisplacement).relation,
            EffectRelationV1::UnrequestedPreserved,
            "seed {seed}: tonal displacement was not preserved"
        );
        assert_eq!(
            evidence.changed_unrequested_channels, 0,
            "seed {seed}: an unrequested symbolic channel changed"
        );
    }
}
