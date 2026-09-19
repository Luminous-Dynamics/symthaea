// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003C6C: observational eight-seed Sonata frequency-domain panel.
//!
//! This applies the frozen C6A log-mel representation to the same bounded
//! Sonata re-articulation used by C3-C5. It records continuous spectral
//! descriptors and direct spectral-window overlap topology without introducing
//! a spectral pass/fail threshold.

use symthaea_muse::evidence_digest::{
    canonical_json_sha256,
    rendered_spectral_window_evidence::{
        RenderedSpectralWindowConfigV1, measure_rendered_spectral_window,
    },
    rendered_spectral_window_panel::{
        RenderedSpectralMetricSummaryV1, RenderedSpectralPanelSampleV1,
        summarize_rendered_spectral_window_panel,
    },
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

fn assert_finite_ordered(summary: &RenderedSpectralMetricSummaryV1) {
    assert!(summary.minimum.is_finite());
    assert!(summary.maximum.is_finite());
    assert!(summary.mean.is_finite());
    assert!(summary.minimum <= summary.mean);
    assert!(summary.mean <= summary.maximum);
}

#[test]
fn fixed_sonata_panel_records_spectral_distribution_without_thresholding() {
    let spectral_config = RenderedSpectralWindowConfigV1::default();
    // Freeze the C6A representation here. This application is observational;
    // no parameter may be tuned from the Sonata outcomes.
    assert_eq!(spectral_config.fft_size, 1_024);
    assert_eq!(spectral_config.mel_bands, 64);
    assert_eq!(spectral_config.f_min_hz, 20.0);
    assert_eq!(spectral_config.f_max_hz, 16_000.0);

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
            let evidence = measure_rendered_spectral_window(
                baseline_frames,
                candidate_frames,
                SAMPLE_RATE,
                attack.start_time,
                spectral_config,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "seed {seed} attack {attack_ordinal}: malformed spectral-window evidence: {error:?}"
                )
            });
            samples.push(RenderedSpectralPanelSampleV1 {
                subject_id: subject_id.clone(),
                attack_ordinal,
                evidence,
            });
        }
    }

    let panel = summarize_rendered_spectral_window_panel(&samples).unwrap();
    assert_eq!(panel.subject_count, SEEDS.len());
    assert_eq!(panel.attack_count, expected_attack_count);
    assert_eq!(panel.samples.len(), expected_attack_count);
    assert!(panel.window_overlap_cluster_count >= SEEDS.len());
    assert!(panel.window_overlap_cluster_count <= panel.attack_count);
    assert_eq!(
        panel.singleton_cluster_count + panel.multi_member_cluster_count,
        panel.window_overlap_cluster_count
    );
    assert!(panel.attacks_in_overlapping_clusters <= panel.attack_count);
    assert!(panel.subjects.iter().all(|subject| {
        subject.attack_count > 0
            && subject.window_overlap_cluster_count > 0
            && subject.window_overlap_cluster_count <= subject.attack_count
    }));

    for summary in [
        &panel.baseline_positive_logmel_flux,
        &panel.candidate_positive_logmel_flux,
        &panel.candidate_flux_excess,
        &panel.pre_between_arm_rms_distance,
        &panel.post_between_arm_rms_distance,
    ] {
        assert_finite_ordered(summary);
    }

    let digest = canonical_json_sha256(&panel).unwrap();
    assert_eq!(digest.len(), 64);
    assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));

    // Observational output only. Metric signs/magnitudes and overlap counts are
    // retained for later inspection; none is encoded as a success criterion.
    println!(
        "MEL003C6C subjects={} raw_attacks={} spectral_overlap_clusters={} singleton_clusters={} multi_member_clusters={} attacks_in_overlap={} baseline_flux=[{:.8},{:.8}] mean={:.8} candidate_flux=[{:.8},{:.8}] mean={:.8} flux_excess=[{:.8},{:.8}] mean={:.8} pre_arm_distance=[{:.8},{:.8}] mean={:.8} post_arm_distance=[{:.8},{:.8}] mean={:.8} sha256={}",
        panel.subject_count,
        panel.attack_count,
        panel.window_overlap_cluster_count,
        panel.singleton_cluster_count,
        panel.multi_member_cluster_count,
        panel.attacks_in_overlapping_clusters,
        panel.baseline_positive_logmel_flux.minimum,
        panel.baseline_positive_logmel_flux.maximum,
        panel.baseline_positive_logmel_flux.mean,
        panel.candidate_positive_logmel_flux.minimum,
        panel.candidate_positive_logmel_flux.maximum,
        panel.candidate_positive_logmel_flux.mean,
        panel.candidate_flux_excess.minimum,
        panel.candidate_flux_excess.maximum,
        panel.candidate_flux_excess.mean,
        panel.pre_between_arm_rms_distance.minimum,
        panel.pre_between_arm_rms_distance.maximum,
        panel.pre_between_arm_rms_distance.mean,
        panel.post_between_arm_rms_distance.minimum,
        panel.post_between_arm_rms_distance.maximum,
        panel.post_between_arm_rms_distance.mean,
        digest
    );
}
