// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! External-comparable symbolic pitch metrics for Melothaea benchmarks.
//!
//! V1 freezes the unambiguous note-count-based pitch metrics whose definitions
//! are documented by MusPy and overlap with common MGEval-style descriptive
//! analysis. It deliberately does not yet claim executable parity with either
//! Python implementation; cross-language reference fixtures belong to a
//! separate qualification tranche.
//!
//! These are descriptive statistics, never a musical-quality score.

use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use symthaea_music_theory::Score;

pub const SYMBOLIC_PITCH_METRICS_VERSION: &str = "melothaea-symbolic-pitch-metrics-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum BenchmarkScaleModeV1 {
    Major,
    Minor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BenchmarkScaleV1 {
    /// Chromatic root, C=0 through B=11.
    pub root: u8,
    pub mode: BenchmarkScaleModeV1,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SymbolicPitchMetricsV1 {
    pub metrics_version: String,
    pub note_count: usize,
    /// MusPy-compatible: number of distinct MIDI pitches used.
    pub n_pitches_used: usize,
    /// MusPy-compatible: number of distinct chromatic pitch classes used.
    pub n_pitch_classes_used: usize,
    /// Highest MIDI pitch minus lowest MIDI pitch, in semitones.
    pub pitch_range_semitones: u8,
    /// Shannon entropy, base 2, over the note-count-normalized MIDI histogram.
    pub pitch_entropy_bits: f64,
    /// Shannon entropy, base 2, over the note-count-normalized pitch-class histogram.
    pub pitch_class_entropy_bits: f64,
    /// Maximum note-count pitch-in-scale rate over all 12 major + 12 minor scales.
    pub scale_consistency: f64,
    /// Deterministic first maximum in root-major/minor order. This field is
    /// additional provenance; external compatibility is claimed only for the
    /// scalar `scale_consistency` definition.
    pub best_scale: BenchmarkScaleV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SymbolicPitchMetricErrorV1 {
    EmptyScore,
    RootOutsideChromaticRange { root: u8 },
    Serialization,
}

impl SymbolicPitchMetricsV1 {
    pub fn canonical_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }
}

/// Compute note-count-based pitch statistics without rasterizing time.
///
/// Empty scores are represented as an explicit error rather than manufacturing
/// zero values. This corresponds to the same undefined-input boundary where
/// several external metric implementations return NaN, while keeping serialized
/// benchmark evidence free of NaN.
pub fn measure_symbolic_pitch_metrics(
    score: &Score,
) -> Result<SymbolicPitchMetricsV1, SymbolicPitchMetricErrorV1> {
    if score.notes.is_empty() {
        return Err(SymbolicPitchMetricErrorV1::EmptyScore);
    }

    let mut pitches = [0u64; 128];
    let mut pitch_classes = [0u64; 12];
    let mut min_pitch = u8::MAX;
    let mut max_pitch = u8::MIN;

    for note in &score.notes {
        let pitch = note.pitch.midi();
        pitches[usize::from(pitch)] += 1;
        pitch_classes[usize::from(pitch % 12)] += 1;
        min_pitch = min_pitch.min(pitch);
        max_pitch = max_pitch.max(pitch);
    }

    let note_count = score.notes.len();
    let n_pitches_used = pitches.iter().filter(|count| **count > 0).count();
    let n_pitch_classes_used = pitch_classes.iter().filter(|count| **count > 0).count();
    let pitch_entropy_bits = shannon_entropy_bits(&pitches, note_count);
    let pitch_class_entropy_bits = shannon_entropy_bits(&pitch_classes, note_count);
    let (best_scale, scale_consistency) = best_scale_consistency(&pitch_classes, note_count);

    Ok(SymbolicPitchMetricsV1 {
        metrics_version: SYMBOLIC_PITCH_METRICS_VERSION.into(),
        note_count,
        n_pitches_used,
        n_pitch_classes_used,
        pitch_range_semitones: max_pitch - min_pitch,
        pitch_entropy_bits,
        pitch_class_entropy_bits,
        scale_consistency,
        best_scale,
    })
}

pub fn pitch_in_scale_rate(
    score: &Score,
    scale: BenchmarkScaleV1,
) -> Result<f64, SymbolicPitchMetricErrorV1> {
    if score.notes.is_empty() {
        return Err(SymbolicPitchMetricErrorV1::EmptyScore);
    }
    if scale.root > 11 {
        return Err(SymbolicPitchMetricErrorV1::RootOutsideChromaticRange { root: scale.root });
    }

    let in_scale = score
        .notes
        .iter()
        .filter(|note| scale_contains_pitch_class(scale, note.pitch.midi() % 12))
        .count();
    Ok(in_scale as f64 / score.notes.len() as f64)
}

fn shannon_entropy_bits<const N: usize>(counts: &[u64; N], total: usize) -> f64 {
    counts
        .iter()
        .copied()
        .filter(|count| *count > 0)
        .map(|count| {
            let probability = count as f64 / total as f64;
            -probability * probability.log2()
        })
        .sum()
}

fn best_scale_consistency(
    pitch_classes: &[u64; 12],
    note_count: usize,
) -> (BenchmarkScaleV1, f64) {
    let mut best = BenchmarkScaleV1 {
        root: 0,
        mode: BenchmarkScaleModeV1::Major,
    };
    let mut best_rate = f64::NEG_INFINITY;

    for root in 0..12u8 {
        for mode in [BenchmarkScaleModeV1::Major, BenchmarkScaleModeV1::Minor] {
            let scale = BenchmarkScaleV1 { root, mode };
            let matched: u64 = pitch_classes
                .iter()
                .enumerate()
                .filter(|(pitch_class, _)| {
                    scale_contains_pitch_class(scale, *pitch_class as u8)
                })
                .map(|(_, count)| *count)
                .sum();
            let rate = matched as f64 / note_count as f64;
            if rate > best_rate {
                best = scale;
                best_rate = rate;
            }
        }
    }

    (best, best_rate)
}

fn scale_contains_pitch_class(scale: BenchmarkScaleV1, pitch_class: u8) -> bool {
    let interval = (pitch_class + 12 - scale.root) % 12;
    match scale.mode {
        BenchmarkScaleModeV1::Major => matches!(interval, 0 | 2 | 4 | 5 | 7 | 9 | 11),
        BenchmarkScaleModeV1::Minor => matches!(interval, 0 | 2 | 3 | 5 | 7 | 8 | 10),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_music_theory::{
        Duration, Emphasis, Key, PartId, Pitch, PitchClass, ScoreNote, VoiceRole,
    };

    fn score_from_midis(midis: &[u8]) -> Score {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for (index, midi) in midis.iter().copied().enumerate() {
            score.push(ScoreNote {
                part: PartId(1),
                pitch: Pitch::from_midi(midi),
                onset: Duration::new(index as i64, 1),
                duration: Duration::quarter(),
                velocity: 0.7,
                role: VoiceRole::Melody,
                emphasis: Emphasis::Normal,
                section_intensity: 1.0,
            });
        }
        score
    }

    #[test]
    fn empty_score_is_explicitly_undefined() {
        let score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        assert_eq!(
            measure_symbolic_pitch_metrics(&score),
            Err(SymbolicPitchMetricErrorV1::EmptyScore)
        );
    }

    #[test]
    fn simple_c_major_fixture_matches_closed_form_pitch_statistics() {
        let metrics = measure_symbolic_pitch_metrics(&score_from_midis(&[60, 64, 67, 72])).unwrap();
        assert_eq!(metrics.note_count, 4);
        assert_eq!(metrics.n_pitches_used, 4);
        assert_eq!(metrics.n_pitch_classes_used, 3);
        assert_eq!(metrics.pitch_range_semitones, 12);
        assert!((metrics.pitch_entropy_bits - 2.0).abs() < 1e-12);
        assert!((metrics.pitch_class_entropy_bits - 1.5).abs() < 1e-12);
        assert!((metrics.scale_consistency - 1.0).abs() < 1e-12);
        assert_eq!(
            metrics.best_scale,
            BenchmarkScaleV1 {
                root: 0,
                mode: BenchmarkScaleModeV1::Major,
            }
        );
    }

    #[test]
    fn chromatic_fixture_has_expected_entropy_and_scale_ceiling() {
        let midis: Vec<u8> = (60..72).collect();
        let metrics = measure_symbolic_pitch_metrics(&score_from_midis(&midis)).unwrap();
        assert_eq!(metrics.n_pitch_classes_used, 12);
        assert!((metrics.pitch_class_entropy_bits - 12f64.log2()).abs() < 1e-12);
        assert!((metrics.scale_consistency - (7.0 / 12.0)).abs() < 1e-12);
    }

    #[test]
    fn pitch_in_scale_rate_is_note_count_weighted() {
        let score = score_from_midis(&[60, 60, 61, 64]);
        let rate = pitch_in_scale_rate(
            &score,
            BenchmarkScaleV1 {
                root: 0,
                mode: BenchmarkScaleModeV1::Major,
            },
        )
        .unwrap();
        assert!((rate - 0.75).abs() < 1e-12);
    }

    #[test]
    fn octave_duplicates_affect_pitch_entropy_but_share_pitch_class() {
        let metrics = measure_symbolic_pitch_metrics(&score_from_midis(&[60, 72])).unwrap();
        assert_eq!(metrics.n_pitches_used, 2);
        assert_eq!(metrics.n_pitch_classes_used, 1);
        assert!((metrics.pitch_entropy_bits - 1.0).abs() < 1e-12);
        assert!(metrics.pitch_class_entropy_bits.abs() < 1e-12);
    }

    #[test]
    fn metric_commitment_is_stable_and_loads_no_quality_authority() {
        let metrics = measure_symbolic_pitch_metrics(&score_from_midis(&[60, 64, 67])).unwrap();
        let digest = metrics.canonical_sha256().unwrap();
        assert_eq!(digest.len(), 64);
        assert_eq!(digest, metrics.canonical_sha256().unwrap());
    }
}
