// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003B7/B9: bind intervention-effect evidence to a real bounded Sonata
//! accompaniment re-articulation, then summarize the fixed replication panel.

use symthaea_muse::cognitive_bridge::SymbolicMeasurementEvidence;
use symthaea_muse::cognitive_selection::cognitive_target_for_source_action;
use symthaea_muse::evidence_digest::intervention_effect_evidence::{
    EffectRelationV1, InterventionEffectEvidenceDisposition, InterventionEffectEvidenceV1,
    RequestRoleV1, account_intervention_effects,
};
use symthaea_muse::evidence_digest::intervention_effect_panel::{
    ChannelEffectPanelV1, InterventionEffectPanelV1, summarize_intervention_effect_panel,
};
use symthaea_muse::musical_inference::MusicAction;
use symthaea_muse::musical_policy::OutcomeChannel;
use symthaea_music_theory::{
    Duration, MusicalIntent, Score, SonataSectionKind, Style, VoiceRole, compose_sonata_with_plan,
    profile_score_region,
};

const SONATA_EFFECT_PANEL_SEEDS: [u64; 8] = [3, 11, 23, 41, 59, 79, 97, 127];
const EFFECT_EPSILON: f32 = 0.01;

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
    evidence: &InterventionEffectEvidenceV1,
    channel: OutcomeChannel,
) -> &symthaea_muse::evidence_digest::intervention_effect_evidence::ChannelInterventionEffectV1 {
    evidence
        .channels
        .iter()
        .find(|entry| entry.channel == channel)
        .expect("all four symbolic outcome channels must be retained")
}

fn panel_channel(
    panel: &InterventionEffectPanelV1,
    channel: OutcomeChannel,
) -> &ChannelEffectPanelV1 {
    panel
        .channels
        .iter()
        .find(|entry| entry.channel == channel)
        .expect("all four symbolic outcome channels must be retained in the panel")
}

fn panel_records() -> Vec<InterventionEffectEvidenceV1> {
    let requested = cognitive_target_for_source_action(MusicAction::IncreaseComplexity);
    SONATA_EFFECT_PANEL_SEEDS
        .into_iter()
        .map(|seed| {
            let observed = measured_effect(seed);
            account_intervention_effects(requested, observed, EFFECT_EPSILON)
        })
        .collect()
}

#[test]
fn real_sonata_density_probe_records_primary_secondary_and_preserved_channels_separately() {
    let observed = measured_effect(79);
    let requested = cognitive_target_for_source_action(MusicAction::IncreaseComplexity);
    let evidence = account_intervention_effects(requested, observed, EFFECT_EPSILON);

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
    let expected_tension_relation = if observed.tension_delta.abs() <= EFFECT_EPSILON {
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
    for (seed, evidence) in SONATA_EFFECT_PANEL_SEEDS.into_iter().zip(panel_records()) {
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

#[test]
fn sonata_density_effect_panel_quantifies_secondary_tension_without_hiding_it() {
    let records = panel_records();
    let panel = summarize_intervention_effect_panel(&records).unwrap();

    assert_eq!(panel.sample_count, SONATA_EFFECT_PANEL_SEEDS.len());
    assert_eq!(panel.movement_epsilon, EFFECT_EPSILON);
    assert_eq!(
        panel.dominant_requested_channel,
        Some(OutcomeChannel::Density)
    );

    let density = panel_channel(&panel, OutcomeChannel::Density);
    assert_eq!(density.request_role, RequestRoleV1::Dominant);
    assert_eq!(density.samples, SONATA_EFFECT_PANEL_SEEDS.len());
    assert_eq!(
        density.relation_counts.requested_aligned,
        SONATA_EFFECT_PANEL_SEEDS.len()
    );
    assert_eq!(density.relation_counts.total(), SONATA_EFFECT_PANEL_SEEDS.len());
    assert!(density.observed_min > EFFECT_EPSILON);
    assert!(density.observed_min.is_finite());
    assert!(density.observed_max.is_finite());
    assert!(density.observed_mean.is_finite());
    assert!(density.observed_min <= density.observed_mean);
    assert!(density.observed_mean <= density.observed_max);

    // Tension is a secondary request, so retain the complete quantitative
    // response and categorical counts without manufacturing a pass/fail verdict.
    let tension = panel_channel(&panel, OutcomeChannel::Tension);
    assert_eq!(tension.request_role, RequestRoleV1::Secondary);
    assert_eq!(tension.samples, SONATA_EFFECT_PANEL_SEEDS.len());
    assert_eq!(tension.relation_counts.total(), SONATA_EFFECT_PANEL_SEEDS.len());
    assert!(tension.observed_min.is_finite());
    assert!(tension.observed_max.is_finite());
    assert!(tension.observed_mean.is_finite());
    assert!(tension.observed_min <= tension.observed_mean);
    assert!(tension.observed_mean <= tension.observed_max);

    // The two unrequested symbolic channels must remain preserved across every
    // member of the exact fixed panel.
    for channel_id in [OutcomeChannel::Familiarity, OutcomeChannel::TonalDisplacement] {
        let channel = panel_channel(&panel, channel_id);
        assert_eq!(channel.request_role, RequestRoleV1::Unrequested);
        assert_eq!(
            channel.relation_counts.unrequested_preserved,
            SONATA_EFFECT_PANEL_SEEDS.len()
        );
        assert_eq!(channel.relation_counts.unrequested_changed, 0);
        assert_eq!(channel.relation_counts.total(), SONATA_EFFECT_PANEL_SEEDS.len());
        assert!(channel.observed_min.abs() <= f32::EPSILON);
        assert!(channel.observed_max.abs() <= f32::EPSILON);
        assert!(channel.observed_mean.abs() <= f32::EPSILON);
    }
}
