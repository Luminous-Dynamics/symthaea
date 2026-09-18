// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003C3C: apply the already-frozen rendered-attack gate to the bounded
//! Sonata accompaniment re-articulation without changing its thresholds.

use symthaea_muse::evidence_digest::rendered_attack_evidence::measure_rendered_attack;
use symthaea_muse::evidence_digest::rendered_attack_gate::{
    RenderedAttackGateConfigV1, evaluate_rendered_attack_gate,
};
use symthaea_muse::theory_realize::{PerformedVoice, perform_with_spec, realize_with_spec};
use symthaea_muse::{AudioData, MusicalState};
use symthaea_music_theory::{
    Duration, MusicalIntent, Score, SonataSectionKind, Style, VoiceRole, compose_sonata_with_plan,
};

const SAMPLE_RATE: u32 = 44_100;
const SEED: u64 = 79;
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

    introduced
}

#[test]
fn frozen_localized_change_gate_accepts_each_new_sonata_performed_attack() {
    let intent = MusicalIntent {
        seed: SEED,
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
    assert!(split_count > 0, "gate application requires a symbolic split");

    let candidate_score =
        rearticulate_accompaniment(&realization.score, target.start, target.end);
    let state = MusicalState::default();

    let baseline_performed = perform_with_spec(&realization.score, &spec, SEED, &state);
    let candidate_performed = perform_with_spec(&candidate_score, &spec, SEED, &state);
    let baseline_attacks = accompaniment_attacks(&baseline_performed);
    let candidate_attacks = accompaniment_attacks(&candidate_performed);
    let introduced = newly_introduced_attacks(&baseline_attacks, &candidate_attacks);
    assert_eq!(
        introduced.len(), split_count,
        "the gate may only evaluate attacks already proven to survive performance realization"
    );

    let baseline_render = realize_with_spec(&realization.score, &spec, SEED, &state, SAMPLE_RATE);
    let candidate_render = realize_with_spec(&candidate_score, &spec, SEED, &state, SAMPLE_RATE);
    let baseline_frames = stereo_frames(&baseline_render.audio);
    let candidate_frames = stereo_frames(&candidate_render.audio);
    assert_eq!(baseline_frames.len(), candidate_frames.len());

    // These values are frozen in MEL-003C3B and are asserted here so this
    // application tranche cannot silently tune them to the Sonata result.
    let config = RenderedAttackGateConfigV1::default();
    assert_eq!(config.ratio_floor_rms, 1.0e-6);
    assert_eq!(config.minimum_post_difference_rms, 1.0e-4);
    assert_eq!(config.minimum_post_to_pre_ratio, 2.0);

    for attack in &introduced {
        let evidence = measure_rendered_attack(
            baseline_frames,
            candidate_frames,
            SAMPLE_RATE,
            attack.start_time,
            PRE_WINDOW_SECS,
            POST_WINDOW_SECS,
        )
        .unwrap();
        let result = evaluate_rendered_attack_gate(&evidence, config).unwrap();
        assert!(
            result.localized_change_detected,
            "frozen gate rejected new performed {} attack at {:.6}s: pre_rms={:.8} post_rms={:.8} ratio={:.3}",
            attack.voice,
            attack.start_time,
            result.pre_difference_rms,
            result.post_difference_rms,
            result.post_to_pre_ratio
        );
    }
}
