// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003C5D: observational eight-seed acoustic cross-check panel.
//!
//! This test keeps raw performed-note observations, two independent acoustic
//! proxy outcomes, and direct matched-window overlap topology separate. It does
//! not require every note or overlap cluster to satisfy either acoustic gate.

use symthaea_muse::evidence_digest::{
    acoustic_window_dependency::{
        AcousticWindowDependencySampleV1, summarize_acoustic_window_dependencies,
    },
    canonical_json_sha256,
    rendered_attack_crosscheck::{
        RenderedAttackCrosscheckClassV1, evaluate_rendered_attack_crosscheck,
    },
    rendered_attack_evidence::measure_rendered_attack,
    rendered_attack_gate::RenderedAttackGateConfigV1,
    rendered_transient_contrast::RenderedTransientContrastConfigV1,
};
use symthaea_muse::theory_realize::{PerformedVoice, perform_with_spec, realize_with_spec};
use symthaea_muse::{AudioData, MusicalState};
use symthaea_music_theory::{
    Duration, MusicalIntent, Score, SonataSectionKind, Style, VoiceRole, compose_sonata_with_plan,
};

const SAMPLE_RATE: u32 = 44_100;
const SEEDS: [u64; 8] = [3, 11, 23, 41, 59, 79, 97, 127];
const START_MATCH_EPSILON_SECS: f32 = 0.000_1;
const FREQUENCY_MATCH_EPSILON_HZ: f32 = 0.001;
const PRE_WINDOW_SECS: f32 = 0.005;
const POST_WINDOW_SECS: f32 = 0.050;

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

fn stereo_frames(audio: &AudioData) -> &[[f32; 2]] {
    match audio {
        AudioData::StereoF32(frames) => frames,
        other => panic!("native Sonata renderer must emit StereoF32, got {other:?}"),
    }
}

#[derive(Debug, Clone)]
struct PerformedAttack {
    voice: String,
    frequency_hz: f32,
    start_time: f32,
}

fn accompaniment_attacks(voices: &[PerformedVoice]) -> Vec<PerformedAttack> {
    voices
        .iter()
        .filter(|voice| matches!(voice.name.as_str(), "Harmony" | "Bass"))
        .flat_map(|voice| {
            voice.notes.iter().map(|note| PerformedAttack {
                voice: voice.name.clone(),
                frequency_hz: note.frequency,
                start_time: note.start_time,
            })
        })
        .collect()
}

fn newly_introduced_attacks(
    baseline: &[PerformedAttack],
    candidate: &[PerformedAttack],
) -> Vec<PerformedAttack> {
    let mut used = vec![false; baseline.len()];
    let mut introduced = Vec::new();

    for attack in candidate {
        let matching = baseline.iter().enumerate().position(|(index, baseline)| {
            !used[index]
                && baseline.voice == attack.voice
                && (baseline.frequency_hz - attack.frequency_hz).abs()
                    <= FREQUENCY_MATCH_EPSILON_HZ
                && (baseline.start_time - attack.start_time).abs() <= START_MATCH_EPSILON_SECS
        });
        if let Some(index) = matching {
            used[index] = true;
        } else {
            introduced.push(attack.clone());
        }
    }

    introduced.sort_by(|left, right| {
        left.start_time
            .total_cmp(&right.start_time)
            .then_with(|| left.voice.cmp(&right.voice))
            .then_with(|| left.frequency_hz.total_cmp(&right.frequency_hz))
    });
    introduced
}

#[test]
fn fixed_sonata_panel_records_joint_acoustic_classes_and_window_dependence() {
    let difference_config = RenderedAttackGateConfigV1::default();
    let transient_config = RenderedTransientContrastConfigV1::default();

    // Lock both independently calibrated operating points. This application
    // tranche observes outcomes; it does not retune either acoustic proxy.
    assert_eq!(difference_config.ratio_floor_rms, 1.0e-6);
    assert_eq!(difference_config.minimum_post_difference_rms, 1.0e-4);
    assert_eq!(difference_config.minimum_post_to_pre_ratio, 2.0);
    assert_eq!(transient_config.activity_floor, 1.0e-6);
    assert_eq!(transient_config.minimum_candidate_post_activity, 1.0e-4);
    assert_eq!(transient_config.minimum_candidate_growth_ratio, 2.0);
    assert_eq!(transient_config.minimum_growth_excess_over_baseline, 2.0);

    let spec = Style::Sonata.spec();
    let state = MusicalState::default();
    let mut samples = Vec::new();
    let mut expected_attack_count = 0usize;

    for seed in SEEDS {
        let intent = MusicalIntent {
            seed,
            bars: 4,
            ..MusicalIntent::default()
        };
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
        assert!(split_count > 0, "seed {seed}: no bounded symbolic split available");

        let candidate_score =
            rearticulate_accompaniment(&realization.score, target.start, target.end);
        let baseline_performed = perform_with_spec(&realization.score, &spec, seed, &state);
        let candidate_performed = perform_with_spec(&candidate_score, &spec, seed, &state);
        let baseline_attacks = accompaniment_attacks(&baseline_performed);
        let candidate_attacks = accompaniment_attacks(&candidate_performed);
        let introduced = newly_introduced_attacks(&baseline_attacks, &candidate_attacks);
        assert_eq!(
            introduced.len(),
            split_count,
            "seed {seed}: symbolic splits must survive as the same number of introduced performed accompaniment attacks"
        );
        expected_attack_count += split_count;

        let baseline_render =
            realize_with_spec(&realization.score, &spec, seed, &state, SAMPLE_RATE);
        let candidate_render =
            realize_with_spec(&candidate_score, &spec, seed, &state, SAMPLE_RATE);
        let baseline_frames = stereo_frames(&baseline_render.audio);
        let candidate_frames = stereo_frames(&candidate_render.audio);
        assert_eq!(
            baseline_frames.len(),
            candidate_frames.len(),
            "seed {seed}: bounded intervention must preserve render length"
        );

        let subject_id = format!("sonata-seed-{seed}");
        for (attack_ordinal, attack) in introduced.iter().enumerate() {
            let evidence = measure_rendered_attack(
                baseline_frames,
                candidate_frames,
                SAMPLE_RATE,
                attack.start_time,
                PRE_WINDOW_SECS,
                POST_WINDOW_SECS,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "seed {seed} attack {attack_ordinal}: malformed rendered-attack evidence: {error:?}"
                )
            });
            let crosscheck = evaluate_rendered_attack_crosscheck(
                &evidence,
                difference_config,
                transient_config,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "seed {seed} attack {attack_ordinal}: acoustic crosscheck failed on valid evidence: {error:?}"
                )
            });
            samples.push(AcousticWindowDependencySampleV1 {
                subject_id: subject_id.clone(),
                attack_ordinal,
                crosscheck,
            });
        }
    }

    let panel = summarize_acoustic_window_dependencies(&samples).unwrap();
    assert_eq!(panel.subject_count, SEEDS.len());
    assert_eq!(panel.attack_count, expected_attack_count);
    assert_eq!(panel.samples.len(), expected_attack_count);
    assert!(panel.window_overlap_cluster_count >= SEEDS.len());
    assert!(panel.window_overlap_cluster_count <= panel.attack_count);
    assert_eq!(
        panel.singleton_cluster_count + panel.multi_member_cluster_count,
        panel.window_overlap_cluster_count
    );
    assert!(panel.subjects.iter().all(|subject| {
        subject.attack_count > 0
            && subject.window_overlap_cluster_count > 0
            && subject.window_overlap_cluster_count <= subject.attack_count
    }));

    let mut both = 0usize;
    let mut localized_only = 0usize;
    let mut transient_only = 0usize;
    let mut neither = 0usize;
    for sample in &panel.samples {
        match sample.crosscheck.class {
            RenderedAttackCrosscheckClassV1::LocalizedDifferenceAndCandidateTransient => both += 1,
            RenderedAttackCrosscheckClassV1::LocalizedDifferenceOnly => localized_only += 1,
            RenderedAttackCrosscheckClassV1::CandidateTransientOnly => transient_only += 1,
            RenderedAttackCrosscheckClassV1::Neither => neither += 1,
        }
    }
    assert_eq!(both + localized_only + transient_only + neither, panel.attack_count);

    let homogeneous_cluster_count = panel
        .clusters
        .iter()
        .filter(|cluster| {
            cluster.members.first().is_some_and(|first| {
                cluster.members.iter().all(|member| member.class == first.class)
            })
        })
        .count();
    let mixed_class_cluster_count = panel.window_overlap_cluster_count - homogeneous_cluster_count;

    let digest = canonical_json_sha256(&panel).unwrap();
    assert_eq!(digest.len(), 64);
    assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));

    // Observational output only. None of these counts is a pass criterion or
    // an independent-sample count; overlap topology is printed explicitly.
    println!(
        "MEL003C5D subjects={} raw_attacks={} overlap_clusters={} singleton_clusters={} multi_member_clusters={} attacks_in_overlap={} both={} localized_only={} transient_only={} neither={} homogeneous_clusters={} mixed_class_clusters={} sha256={}",
        panel.subject_count,
        panel.attack_count,
        panel.window_overlap_cluster_count,
        panel.singleton_cluster_count,
        panel.multi_member_cluster_count,
        panel.attacks_in_overlapping_clusters,
        both,
        localized_only,
        transient_only,
        neither,
        homogeneous_cluster_count,
        mixed_class_cluster_count,
        digest
    );
}
