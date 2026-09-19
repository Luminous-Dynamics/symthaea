// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003C6F: canonical eight-seed pre-perceptual acoustic evidence bundle.
//!
//! Reconstructs C5 time-domain, C6 frequency-domain, and C6D/C6E multimodal
//! evidence from the same performed attacks, then cross-checks identities and
//! representation-specific direct-sample-overlap topology without introducing
//! a perceptual or artistic verdict.

use serde::Serialize;
use symthaea_muse::evidence_digest::{
    acoustic_window_dependency::{
        AcousticWindowDependencyPanelV1, AcousticWindowDependencySampleV1,
        summarize_acoustic_window_dependencies,
    },
    canonical_json_sha256,
    rendered_attack_crosscheck::{
        RenderedAttackCrosscheckClassV1, evaluate_rendered_attack_crosscheck,
    },
    rendered_attack_evidence::measure_rendered_attack,
    rendered_attack_gate::RenderedAttackGateConfigV1,
    rendered_multimodal_attack_evidence::bind_rendered_multimodal_attack_evidence,
    rendered_multimodal_attack_panel::{
        RenderedMultimodalAttackPanelV1, RenderedMultimodalPanelSampleV1,
        summarize_rendered_multimodal_attack_panel,
    },
    rendered_spectral_window_evidence::{
        RenderedSpectralWindowConfigV1, measure_rendered_spectral_window,
    },
    rendered_spectral_window_panel::{
        RenderedSpectralPanelSampleV1, RenderedSpectralWindowPanelV1,
        summarize_rendered_spectral_window_panel,
    },
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
const TIME_PRE_WINDOW_SECS: f32 = 0.005;
const TIME_POST_WINDOW_SECS: f32 = 0.050;
const BUNDLE_VERSION: &str = "mel003c6f-sonata-multimodal-bundle-v1";

#[derive(Serialize)]
struct SonataMultimodalBundleV1<'a> {
    bundle_version: &'static str,
    seeds: [u64; 8],
    time_domain: &'a AcousticWindowDependencyPanelV1,
    spectral: &'a RenderedSpectralWindowPanelV1,
    multimodal: &'a RenderedMultimodalAttackPanelV1,
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
fn fixed_sonata_panel_binds_time_spectral_and_shared_waveform_provenance() {
    let difference_config = RenderedAttackGateConfigV1::default();
    let transient_config = RenderedTransientContrastConfigV1::default();
    let spectral_config = RenderedSpectralWindowConfigV1::default();

    // Freeze every pre-perceptual operating point/representation. This tranche
    // cross-binds evidence; it does not retune any time-domain or spectral rule.
    assert_eq!(difference_config.ratio_floor_rms, 1.0e-6);
    assert_eq!(difference_config.minimum_post_difference_rms, 1.0e-4);
    assert_eq!(difference_config.minimum_post_to_pre_ratio, 2.0);
    assert_eq!(transient_config.activity_floor, 1.0e-6);
    assert_eq!(transient_config.minimum_candidate_post_activity, 1.0e-4);
    assert_eq!(transient_config.minimum_candidate_growth_ratio, 2.0);
    assert_eq!(transient_config.minimum_growth_excess_over_baseline, 2.0);
    assert_eq!(spectral_config.fft_size, 1_024);
    assert_eq!(spectral_config.mel_bands, 64);
    assert_eq!(spectral_config.f_min_hz, 20.0);
    assert_eq!(spectral_config.f_max_hz, 16_000.0);

    let spec = Style::Sonata.spec();
    let state = MusicalState::default();
    let mut time_samples = Vec::new();
    let mut spectral_samples = Vec::new();
    let mut multimodal_samples = Vec::new();
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
            let time_evidence = measure_rendered_attack(
                baseline_frames,
                candidate_frames,
                SAMPLE_RATE,
                attack.start_time,
                TIME_PRE_WINDOW_SECS,
                TIME_POST_WINDOW_SECS,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "seed {seed} attack {attack_ordinal}: malformed time-domain evidence: {error:?}"
                )
            });
            let crosscheck = evaluate_rendered_attack_crosscheck(
                &time_evidence,
                difference_config,
                transient_config,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "seed {seed} attack {attack_ordinal}: C5 crosscheck failed on valid evidence: {error:?}"
                )
            });
            let spectral = measure_rendered_spectral_window(
                baseline_frames,
                candidate_frames,
                SAMPLE_RATE,
                attack.start_time,
                spectral_config,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "seed {seed} attack {attack_ordinal}: malformed spectral evidence: {error:?}"
                )
            });
            let multimodal = bind_rendered_multimodal_attack_evidence(&crosscheck, &spectral)
                .unwrap_or_else(|error| {
                    panic!(
                        "seed {seed} attack {attack_ordinal}: same-attack multimodal binding failed: {error:?}"
                    )
                });

            time_samples.push(AcousticWindowDependencySampleV1 {
                subject_id: subject_id.clone(),
                attack_ordinal,
                crosscheck: crosscheck.clone(),
            });
            spectral_samples.push(RenderedSpectralPanelSampleV1 {
                subject_id: subject_id.clone(),
                attack_ordinal,
                evidence: spectral.clone(),
            });
            multimodal_samples.push(RenderedMultimodalPanelSampleV1 {
                subject_id: subject_id.clone(),
                attack_ordinal,
                evidence: multimodal,
            });
        }
    }

    let time_panel = summarize_acoustic_window_dependencies(&time_samples).unwrap();
    let spectral_panel = summarize_rendered_spectral_window_panel(&spectral_samples).unwrap();
    let multimodal_panel = summarize_rendered_multimodal_attack_panel(&multimodal_samples).unwrap();

    for subject_count in [
        time_panel.subject_count,
        spectral_panel.subject_count,
        multimodal_panel.subject_count,
    ] {
        assert_eq!(subject_count, SEEDS.len());
    }
    for attack_count in [
        time_panel.attack_count,
        spectral_panel.attack_count,
        multimodal_panel.attack_count,
    ] {
        assert_eq!(attack_count, expected_attack_count);
    }

    // C6E must reproduce the representation-specific direct-sample topology
    // already derived by the dedicated C5 and C6 panels.
    assert_eq!(
        multimodal_panel.time_domain_window_cluster_count,
        time_panel.window_overlap_cluster_count
    );
    assert_eq!(
        multimodal_panel.spectral_window_cluster_count,
        spectral_panel.window_overlap_cluster_count
    );

    let mut both = 0usize;
    let mut localized_only = 0usize;
    let mut transient_only = 0usize;
    let mut neither = 0usize;
    for sample in &time_panel.samples {
        match sample.crosscheck.class {
            RenderedAttackCrosscheckClassV1::LocalizedDifferenceAndCandidateTransient => both += 1,
            RenderedAttackCrosscheckClassV1::LocalizedDifferenceOnly => localized_only += 1,
            RenderedAttackCrosscheckClassV1::CandidateTransientOnly => transient_only += 1,
            RenderedAttackCrosscheckClassV1::Neither => neither += 1,
        }
    }
    assert_eq!(both + localized_only + transient_only + neither, expected_attack_count);
    assert_eq!(
        multimodal_panel.c5_class_counts.localized_difference_and_candidate_transient,
        both
    );
    assert_eq!(multimodal_panel.c5_class_counts.localized_difference_only, localized_only);
    assert_eq!(multimodal_panel.c5_class_counts.candidate_transient_only, transient_only);
    assert_eq!(multimodal_panel.c5_class_counts.neither, neither);

    // Under the frozen production geometries both transforms reuse some source
    // waveform samples but neither window wholly contains the other.
    assert!(multimodal_panel.shared_sample_count.minimum > 0);
    assert!(multimodal_panel.time_domain_only_sample_count.minimum > 0);
    assert!(multimodal_panel.spectral_only_sample_count.minimum > 0);

    let time_digest = canonical_json_sha256(&time_panel).unwrap();
    let spectral_digest = canonical_json_sha256(&spectral_panel).unwrap();
    let multimodal_digest = canonical_json_sha256(&multimodal_panel).unwrap();
    let bundle = SonataMultimodalBundleV1 {
        bundle_version: BUNDLE_VERSION,
        seeds: SEEDS,
        time_domain: &time_panel,
        spectral: &spectral_panel,
        multimodal: &multimodal_panel,
    };
    let bundle_digest = canonical_json_sha256(&bundle).unwrap();
    for digest in [&time_digest, &spectral_digest, &multimodal_digest, &bundle_digest] {
        assert_eq!(digest.len(), 64);
        assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    // Observational output only. Time-domain classes remain C5 proxy outcomes;
    // C6 spectral metrics remain continuous; no perceptual claim is introduced.
    println!(
        "MEL003C6F subjects={} raw_attacks={} c5_clusters={} c6_clusters={} both={} localized_only={} transient_only={} neither={} shared_samples=[{},{}] mean={:.2} c5_only=[{},{}] mean={:.2} c6_only=[{},{}] mean={:.2} spectral_flux_excess_mean={:.8} spectral_post_arm_distance_mean={:.8} c5_sha256={} c6_sha256={} multimodal_sha256={} bundle_sha256={}",
        multimodal_panel.subject_count,
        multimodal_panel.attack_count,
        multimodal_panel.time_domain_window_cluster_count,
        multimodal_panel.spectral_window_cluster_count,
        both,
        localized_only,
        transient_only,
        neither,
        multimodal_panel.shared_sample_count.minimum,
        multimodal_panel.shared_sample_count.maximum,
        multimodal_panel.shared_sample_count.mean,
        multimodal_panel.time_domain_only_sample_count.minimum,
        multimodal_panel.time_domain_only_sample_count.maximum,
        multimodal_panel.time_domain_only_sample_count.mean,
        multimodal_panel.spectral_only_sample_count.minimum,
        multimodal_panel.spectral_only_sample_count.maximum,
        multimodal_panel.spectral_only_sample_count.mean,
        spectral_panel.candidate_flux_excess.mean,
        spectral_panel.post_between_arm_rms_distance.mean,
        time_digest,
        spectral_digest,
        multimodal_digest,
        bundle_digest
    );
}
