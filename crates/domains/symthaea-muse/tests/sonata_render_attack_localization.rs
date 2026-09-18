// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003C2: bind the bounded Sonata density intervention to the renderer's
//! own performed-note timeline, then require each newly introduced performed
//! accompaniment attack to coincide with a local native-waveform A/B change.
//!
//! This establishes performed-event and local waveform survival only. It does
//! not claim that an acoustic onset detector, a listener, or a musical-quality
//! judgment would distinguish the intervention.

use symthaea_muse::theory_realize::{PerformedVoice, perform_with_spec, realize_with_spec};
use symthaea_muse::{AudioData, MusicalState};
use symthaea_music_theory::{
    Duration, MusicalIntent, Score, SonataSectionKind, Style, VoiceRole, compose_sonata_with_plan,
};

const SAMPLE_RATE: u32 = 44_100;
const SEED: u64 = 79;
const START_MATCH_EPSILON_SECS: f32 = 0.000_1;
const FREQUENCY_MATCH_EPSILON_HZ: f32 = 0.001;
const WINDOW_PRE_SECS: f32 = 0.005;
const WINDOW_POST_SECS: f32 = 0.050;

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

#[derive(Debug, Clone, Copy)]
struct LocalWaveformDelta {
    changed_frames: usize,
    rms_delta: f64,
    peak_absolute_delta: f32,
}

fn local_waveform_delta(
    baseline: &[[f32; 2]],
    candidate: &[[f32; 2]],
    attack_time: f32,
) -> LocalWaveformDelta {
    assert_eq!(baseline.len(), candidate.len());
    assert!(attack_time.is_finite() && attack_time >= 0.0);

    let start_time = (attack_time - WINDOW_PRE_SECS).max(0.0);
    let end_time = attack_time + WINDOW_POST_SECS;
    let start = (start_time * SAMPLE_RATE as f32).floor() as usize;
    let end = ((end_time * SAMPLE_RATE as f32).ceil() as usize).min(baseline.len());
    assert!(start < end, "performed attack must map inside rendered audio");

    let mut changed_frames = 0usize;
    let mut squared_sum = 0.0_f64;
    let mut peak_absolute_delta = 0.0_f32;
    for (left, right) in baseline[start..end].iter().zip(&candidate[start..end]) {
        let dl = left[0] - right[0];
        let dr = left[1] - right[1];
        if dl != 0.0 || dr != 0.0 {
            changed_frames += 1;
        }
        for delta in [dl, dr] {
            squared_sum += (delta as f64) * (delta as f64);
            peak_absolute_delta = peak_absolute_delta.max(delta.abs());
        }
    }

    let sample_count = ((end - start) * 2) as f64;
    LocalWaveformDelta {
        changed_frames,
        rms_delta: (squared_sum / sample_count).sqrt(),
        peak_absolute_delta,
    }
}

#[test]
fn symbolic_rearticulation_survives_as_new_performed_attacks_with_local_waveform_change() {
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
    assert!(split_count > 0, "localization probe requires a symbolic split");

    let candidate_score =
        rearticulate_accompaniment(&realization.score, target.start, target.end);
    let state = MusicalState::default();

    // The renderer's public performed-note path contains the exact note timing
    // after swing/rubato/expression/humanization. We compare that path before
    // consulting waveform windows so symbolic beat positions are not mistaken
    // for actual rendered attack times.
    let baseline_performed = perform_with_spec(&realization.score, &spec, SEED, &state);
    let candidate_performed = perform_with_spec(&candidate_score, &spec, SEED, &state);
    let baseline_attacks = accompaniment_attacks(&baseline_performed);
    let candidate_attacks = accompaniment_attacks(&candidate_performed);

    assert_eq!(
        candidate_attacks.len(),
        baseline_attacks.len() + split_count,
        "every bounded symbolic split must survive as exactly one additional performed accompaniment attack"
    );

    let introduced = newly_introduced_attacks(&baseline_attacks, &candidate_attacks);
    assert_eq!(
        introduced.len(),
        split_count,
        "performed-note matching must identify exactly the attacks introduced by symbolic re-articulation"
    );
    assert!(
        introduced.iter().all(|attack| {
            attack.start_time.is_finite()
                && attack.start_time >= 0.0
                && attack.frequency_hz.is_finite()
                && attack.frequency_hz > 0.0
        }),
        "introduced performed attacks must have finite physical timing/frequency"
    );

    // Hold the same renderer inputs fixed as MEL-003C1. The only score change
    // is the already-bounded accompaniment re-articulation.
    let baseline_render = realize_with_spec(&realization.score, &spec, SEED, &state, SAMPLE_RATE);
    let candidate_render = realize_with_spec(&candidate_score, &spec, SEED, &state, SAMPLE_RATE);
    let baseline_frames = stereo_frames(&baseline_render.audio);
    let candidate_frames = stereo_frames(&candidate_render.audio);
    assert_eq!(baseline_frames.len(), candidate_frames.len());

    for attack in &introduced {
        let local = local_waveform_delta(baseline_frames, candidate_frames, attack.start_time);
        assert!(
            local.changed_frames > 0,
            "new performed {} attack at {:.6}s must coincide with at least one changed waveform frame",
            attack.voice,
            attack.start_time
        );
        assert!(
            local.rms_delta.is_finite() && local.rms_delta > 0.0,
            "new performed {} attack at {:.6}s must have finite non-zero local RMS A/B delta",
            attack.voice,
            attack.start_time
        );
        assert!(
            local.peak_absolute_delta.is_finite() && local.peak_absolute_delta > 0.0,
            "new performed {} attack at {:.6}s must have finite non-zero local peak A/B delta",
            attack.voice,
            attack.start_time
        );
    }
}
