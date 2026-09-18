// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003C1: establish only that the bounded Sonata accompaniment
//! re-articulation survives the deterministic native score renderer as a
//! measurable waveform change. This is not yet an onset, perceptual, or
//! listener-quality claim.

use symthaea_muse::{AudioData, MusicalState};
use symthaea_muse::theory_realize::realize_with_spec;
use symthaea_music_theory::{
    Duration, MusicalIntent, Score, SonataSectionKind, Style, VoiceRole, compose_sonata_with_plan,
};

const SAMPLE_RATE: u32 = 44_100;
const SEED: u64 = 79;

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

#[derive(Debug, Clone, Copy)]
struct WaveformDelta {
    changed_frames: usize,
    mean_absolute_delta: f64,
    rms_delta: f64,
    peak_absolute_delta: f32,
}

fn compare_waveforms(left: &[[f32; 2]], right: &[[f32; 2]]) -> WaveformDelta {
    assert_eq!(left.len(), right.len(), "comparison requires equal frame counts");
    assert!(!left.is_empty(), "comparison requires rendered audio");

    let mut changed_frames = 0usize;
    let mut absolute_sum = 0.0_f64;
    let mut squared_sum = 0.0_f64;
    let mut peak_absolute_delta = 0.0_f32;

    for (left, right) in left.iter().zip(right) {
        let dl = left[0] - right[0];
        let dr = left[1] - right[1];
        if dl != 0.0 || dr != 0.0 {
            changed_frames += 1;
        }
        for delta in [dl, dr] {
            let magnitude = delta.abs();
            absolute_sum += magnitude as f64;
            squared_sum += (delta as f64) * (delta as f64);
            peak_absolute_delta = peak_absolute_delta.max(magnitude);
        }
    }

    let samples = (left.len() * 2) as f64;
    WaveformDelta {
        changed_frames,
        mean_absolute_delta: absolute_sum / samples,
        rms_delta: (squared_sum / samples).sqrt(),
        peak_absolute_delta,
    }
}

fn assert_finite(frames: &[[f32; 2]], label: &str) {
    assert!(
        frames
            .iter()
            .all(|frame| frame[0].is_finite() && frame[1].is_finite()),
        "{label} render contains NaN/Inf"
    );
}

#[test]
fn bounded_sonata_density_intervention_survives_native_render_as_waveform_change() {
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
    assert!(split_count > 0, "render probe requires a real symbolic intervention");

    let candidate = rearticulate_accompaniment(&realization.score, target.start, target.end);
    let state = MusicalState::default();

    // Render each arm twice with every renderer input held fixed. Exact repeat
    // equality is a prerequisite before treating an A/B waveform difference as
    // evidence that the symbolic intervention survived realization.
    let baseline_a = realize_with_spec(&realization.score, &spec, SEED, &state, SAMPLE_RATE);
    let baseline_b = realize_with_spec(&realization.score, &spec, SEED, &state, SAMPLE_RATE);
    let candidate_a = realize_with_spec(&candidate, &spec, SEED, &state, SAMPLE_RATE);
    let candidate_b = realize_with_spec(&candidate, &spec, SEED, &state, SAMPLE_RATE);

    let baseline_a = stereo_frames(&baseline_a.audio);
    let baseline_b = stereo_frames(&baseline_b.audio);
    let candidate_a = stereo_frames(&candidate_a.audio);
    let candidate_b = stereo_frames(&candidate_b.audio);

    assert_finite(baseline_a, "baseline");
    assert_finite(candidate_a, "candidate");
    assert_eq!(baseline_a, baseline_b, "baseline native render must be exactly reproducible");
    assert_eq!(candidate_a, candidate_b, "candidate native render must be exactly reproducible");
    assert_eq!(
        baseline_a.len(),
        candidate_a.len(),
        "bounded re-articulation preserves symbolic duration, so compared renders must share frame count"
    );

    let delta = compare_waveforms(baseline_a, candidate_a);
    assert!(
        delta.changed_frames > 0,
        "the intervention must survive realization into at least one changed audio frame"
    );
    assert!(delta.mean_absolute_delta.is_finite() && delta.mean_absolute_delta > 0.0);
    assert!(delta.rms_delta.is_finite() && delta.rms_delta > 0.0);
    assert!(
        delta.peak_absolute_delta.is_finite() && delta.peak_absolute_delta > 0.0,
        "rendered waveform difference must have non-zero finite amplitude"
    );
}
