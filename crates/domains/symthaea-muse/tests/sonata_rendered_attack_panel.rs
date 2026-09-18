// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003C4D: bind the observational eight-seed rendered-attack panel to
//! exact per-seed score and native-waveform identities. Gate rejections remain
//! data; provenance mismatch or malformed evidence fails closed.

use symthaea_muse::evidence_digest::{
    canonical_json_sha256,
    rendered_attack_evidence::measure_rendered_attack,
    rendered_attack_gate::{RenderedAttackGateConfigV1, evaluate_rendered_attack_gate},
    rendered_attack_panel::{RenderedAttackPanelSampleV1, summarize_rendered_attack_panel},
    rendered_attack_provenance::{
        RENDERED_ATTACK_PROVENANCE_VERSION, RenderedAttackPanelProvenanceV1,
        RenderedAttackSubjectProvenanceV1, bind_rendered_attack_panel,
    },
    sha256_hex,
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
const RENDERER_ID: &str = "symthaea-muse::theory_realize::realize_with_spec/native";

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

fn stereo_sha256(frames: &[[f32; 2]]) -> String {
    let mut bytes = Vec::with_capacity(frames.len() * 2 * std::mem::size_of::<f32>());
    for frame in frames {
        for sample in frame {
            assert!(sample.is_finite(), "audio provenance cannot hash NaN/Inf");
            bytes.extend_from_slice(&sample.to_le_bytes());
        }
    }
    sha256_hex(&bytes)
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
fn fixed_sonata_seed_panel_binds_gate_results_to_exact_score_and_audio_identities() {
    let config = RenderedAttackGateConfigV1::default();
    // Lock the exact C3B operating point. C4 is replication, not retuning.
    assert_eq!(config.ratio_floor_rms, 1.0e-6);
    assert_eq!(config.minimum_post_difference_rms, 1.0e-4);
    assert_eq!(config.minimum_post_to_pre_ratio, 2.0);

    let spec = Style::Sonata.spec();
    let state = MusicalState::default();
    let mut samples = Vec::new();
    let mut provenance_subjects = Vec::new();
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
        let baseline_score_sha256 = canonical_json_sha256(&realization.score).unwrap();
        let candidate_score_sha256 = canonical_json_sha256(&candidate_score).unwrap();
        assert_ne!(
            baseline_score_sha256, candidate_score_sha256,
            "seed {seed}: the bounded intervention must change concrete score identity"
        );

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
        provenance_subjects.push(RenderedAttackSubjectProvenanceV1 {
            subject_id: subject_id.clone(),
            seed,
            baseline_score_sha256,
            candidate_score_sha256,
            baseline_audio_sha256: stereo_sha256(baseline_frames),
            candidate_audio_sha256: stereo_sha256(candidate_frames),
        });

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
            let gate = evaluate_rendered_attack_gate(&evidence, config).unwrap_or_else(|error| {
                panic!(
                    "seed {seed} attack {attack_ordinal}: frozen gate could not evaluate valid evidence: {error:?}"
                )
            });
            samples.push(RenderedAttackPanelSampleV1 {
                subject_id: subject_id.clone(),
                attack_ordinal,
                attack_time_secs: attack.start_time,
                gate,
            });
        }
    }

    let panel = summarize_rendered_attack_panel(&samples).unwrap();
    assert_eq!(panel.subject_count, SEEDS.len());
    assert_eq!(panel.attack_count, expected_attack_count);
    assert_eq!(panel.samples.len(), expected_attack_count);
    assert_eq!(
        panel.localized_change_count + panel.rejected_count,
        panel.attack_count
    );
    assert_eq!(panel.subjects.len(), SEEDS.len());
    assert!(panel.subjects.iter().all(|subject| subject.attack_count > 0));
    assert!(panel.pre_difference_rms.minimum.is_finite());
    assert!(panel.pre_difference_rms.maximum.is_finite());
    assert!(panel.pre_difference_rms.mean.is_finite());
    assert!(panel.post_difference_rms.minimum.is_finite());
    assert!(panel.post_difference_rms.maximum.is_finite());
    assert!(panel.post_difference_rms.mean.is_finite());
    assert!(panel.post_to_pre_ratio.minimum.is_finite());
    assert!(panel.post_to_pre_ratio.maximum.is_finite());
    assert!(panel.post_to_pre_ratio.mean.is_finite());

    let provenance = RenderedAttackPanelProvenanceV1 {
        provenance_version: RENDERED_ATTACK_PROVENANCE_VERSION.into(),
        renderer_id: RENDERER_ID.into(),
        sample_rate: SAMPLE_RATE,
        pre_window_seconds: PRE_WINDOW_SECS,
        post_window_seconds: POST_WINDOW_SECS,
        subjects: provenance_subjects,
    };
    let bound = bind_rendered_attack_panel(&panel, &provenance).unwrap();
    let panel_digest = canonical_json_sha256(&panel).unwrap();
    assert_eq!(bound.panel_sha256, panel_digest);
    assert_eq!(bound.provenance.subjects.len(), SEEDS.len());
    for digest in [
        &bound.panel_sha256,
        &bound.provenance_sha256,
        &bound.binding_sha256,
    ] {
        assert_eq!(digest.len(), 64);
        assert!(digest.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

    // Useful under --nocapture / targeted qualification runs. Gate rejections
    // remain observations; the binding digest commits the complete panel plus
    // exact score/audio provenance without turning it into a success score.
    println!(
        "MEL003C4D subjects={} attacks={} localized={} rejected={} post_energy={} growth={} pre_rms=[{:.8},{:.8}] mean={:.8} post_rms=[{:.8},{:.8}] mean={:.8} ratio=[{:.3},{:.3}] mean={:.3} panel_sha256={} provenance_sha256={} binding_sha256={}",
        panel.subject_count,
        panel.attack_count,
        panel.localized_change_count,
        panel.rejected_count,
        panel.post_energy_requirement_met_count,
        panel.growth_requirement_met_count,
        panel.pre_difference_rms.minimum,
        panel.pre_difference_rms.maximum,
        panel.pre_difference_rms.mean,
        panel.post_difference_rms.minimum,
        panel.post_difference_rms.maximum,
        panel.post_difference_rms.mean,
        panel.post_to_pre_ratio.minimum,
        panel.post_to_pre_ratio.maximum,
        panel.post_to_pre_ratio.mean,
        bound.panel_sha256,
        bound.provenance_sha256,
        bound.binding_sha256
    );
}
