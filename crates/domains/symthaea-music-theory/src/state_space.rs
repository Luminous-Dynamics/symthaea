// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Temporal musical state space for Melothaea.
//!
//! The music-theory crate already provides static fingerprints and
//! transformation-aware motif evidence. This module adds the missing temporal
//! substrate: a score becomes a sequence of auditable structural states.
//!
//! This is not a perceptual-quality oracle. The state contains symbolic
//! descriptors only. Recurrence and novelty are derived from those descriptors,
//! so every value is reproducible from the Score.
//!
//! The same canonical frame vector can later be encoded into HDC/VSA without
//! changing the musical semantics.

use crate::score::{Score, ScoreNote, VoiceRole};
use serde::{Deserialize, Serialize};

/// Number of dimensions in a frame's canonical structural vector.
///
/// Layout: onset density, pitch entropy, rhythm entropy, mean interval,
/// contour asymmetry, structural intensity, 12 pitch-class bins, 5 rhythm
/// bins, 12 interval bins, and 3 contour bins.
pub const STATE_DIMS: usize = 38;

const PITCH_CLASS_BINS: usize = 12;
const RHYTHM_BINS: usize = 5;
const INTERVAL_BINS: usize = 12;
const CONTOUR_BINS: usize = 3;
const RHYTHM_THRESHOLDS: [f64; RHYTHM_BINS - 1] = [0.25, 0.5, 1.0, 2.0];

/// One temporal slice of a symbolic score.
///
/// Novelty is relative to the best-matching earlier frame in the same
/// trajectory. It is not a claim about novelty relative to a training corpus
/// or to human music.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalStateFrame {
    pub start_beat: f64,
    pub end_beat: f64,
    pub event_count: usize,
    pub onset_density: f64,
    pub pitch_class_hist: [f64; PITCH_CLASS_BINS],
    pub rhythm_hist: [f64; RHYTHM_BINS],
    pub interval_hist: [f64; INTERVAL_BINS],
    pub contour_hist: [f64; CONTOUR_BINS],
    pub pitch_entropy: f64,
    pub rhythm_entropy: f64,
    pub mean_interval: f64,
    pub contour_asymmetry: f64,
    pub structural_intensity: f64,
    pub nearest_prior_similarity: Option<f64>,
    pub novelty: Option<f64>,
}

impl MusicalStateFrame {
    /// Canonical fixed-length structural representation for distance metrics
    /// or later HDC/VSA encoding.
    pub fn vector(&self) -> [f64; STATE_DIMS] {
        let mut v = [0.0; STATE_DIMS];
        v[0] = self.onset_density;
        v[1] = self.pitch_entropy;
        v[2] = self.rhythm_entropy;
        v[3] = self.mean_interval;
        v[4] = self.contour_asymmetry;
        v[5] = self.structural_intensity;

        let mut offset = 6;
        v[offset..offset + PITCH_CLASS_BINS].copy_from_slice(&self.pitch_class_hist);
        offset += PITCH_CLASS_BINS;
        v[offset..offset + RHYTHM_BINS].copy_from_slice(&self.rhythm_hist);
        offset += RHYTHM_BINS;
        v[offset..offset + INTERVAL_BINS].copy_from_slice(&self.interval_hist);
        offset += INTERVAL_BINS;
        v[offset..offset + CONTOUR_BINS].copy_from_slice(&self.contour_hist);
        v
    }

    /// Cosine similarity between two canonical frame states.
    pub fn similarity(&self, other: &Self) -> f64 {
        cosine_similarity(&self.vector(), &other.vector())
    }
}

/// A time-ordered sequence of MusicalStateFrame values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalStateTrajectory {
    pub window_beats: f64,
    pub hop_beats: f64,
    pub frames: Vec<MusicalStateFrame>,
}

impl MusicalStateTrajectory {
    /// Build a trajectory using fixed-size windows and hops.
    ///
    /// The final partial window is included when it contains at least one
    /// onset. Empty trailing windows are omitted so silence outside a score
    /// does not create artificial recurrence.
    pub fn from_score(
        score: &Score,
        window_beats: f64,
        hop_beats: f64,
    ) -> Result<Self, String> {
        if !window_beats.is_finite() || window_beats <= 0.0 {
            return Err("window_beats must be finite and > 0".into());
        }
        if !hop_beats.is_finite() || hop_beats <= 0.0 {
            return Err("hop_beats must be finite and > 0".into());
        }

        let total = score.total_beats.beats();
        if total <= 0.0 {
            return Ok(Self {
                window_beats,
                hop_beats,
                frames: Vec::new(),
            });
        }

        let mut frames = Vec::new();
        let mut start = 0.0;
        while start < total {
            let end = (start + window_beats).min(total);
            let notes = notes_in_window(score, start, end);
            if !notes.is_empty() {
                frames.push(frame_from_notes(score, start, end, &notes));
            }
            if end >= total {
                break;
            }
            start += hop_beats;
        }

        // Only earlier frames are allowed to influence recurrence/novelty.
        for i in 1..frames.len() {
            let best = (0..i)
                .filter(|&j| frames[j].event_count > 0)
                .map(|j| frames[i].similarity(&frames[j]))
                .max_by(f64::total_cmp);
            frames[i].nearest_prior_similarity = best;
            frames[i].novelty = best.map(|s| (1.0 - s).clamp(0.0, 1.0));
        }

        Ok(Self {
            window_beats,
            hop_beats,
            frames,
        })
    }

    /// Return frame pairs whose structural similarity clears threshold.
    ///
    /// This is a compact recurrence topology. Callers can render it as a
    /// recurrence plot or inspect it as a graph without storing an N x N
    /// matrix.
    pub fn recurrence_pairs(&self, threshold: f64) -> Vec<(usize, usize, f64)> {
        let threshold = threshold.clamp(-1.0, 1.0);
        let mut pairs = Vec::new();
        for i in 0..self.frames.len() {
            for j in 0..i {
                let similarity = self.frames[i].similarity(&self.frames[j]);
                if similarity >= threshold {
                    pairs.push((j, i, similarity));
                }
            }
        }
        pairs
    }

    /// Fraction of possible frame pairs that are recurrent at threshold.
    pub fn recurrence_rate(&self, threshold: f64) -> f64 {
        let n = self.frames.len();
        if n < 2 {
            return 0.0;
        }
        let possible = (n * (n - 1) / 2) as f64;
        self.recurrence_pairs(threshold).len() as f64 / possible
    }

    /// Return frame indices whose best-prior novelty exceeds the threshold.
    pub fn novelty_peaks(&self, minimum_novelty: f64) -> Vec<usize> {
        self.frames
            .iter()
            .enumerate()
            .filter_map(|(i, frame)| {
                frame
                    .novelty
                    .filter(|&n| n >= minimum_novelty)
                    .map(|_| i)
            })
            .collect()
    }
}

fn notes_in_window(score: &Score, start: f64, end: f64) -> Vec<ScoreNote> {
    let mut notes: Vec<_> = score
        .notes
        .iter()
        .copied()
        .filter(|note| {
            let onset = note.onset.beats();
            onset >= start - 1e-9 && onset < end - 1e-9
        })
        .collect();
    notes.sort_by(|a, b| {
        a.onset
            .beats()
            .total_cmp(&b.onset.beats())
            .then_with(|| a.pitch.midi().cmp(&b.pitch.midi()))
    });
    notes
}

fn frame_from_notes(score: &Score, start: f64, end: f64, notes: &[ScoreNote]) -> MusicalStateFrame {
    let mut pitch_class_hist = [0.0; PITCH_CLASS_BINS];
    let mut rhythm_hist = [0.0; RHYTHM_BINS];
    let mut interval_hist = [0.0; INTERVAL_BINS];
    let mut contour_hist = [0.0; CONTOUR_BINS];

    let tonic = score.key.tonic.value() as i32;
    let melody: Vec<_> = notes
        .iter()
        .copied()
        .filter(|note| note.role == VoiceRole::Melody)
        .collect();

    let mut intensity_sum = 0.0;
    for note in notes {
        let duration = note.duration.beats().max(0.0);
        intensity_sum += note.section_intensity as f64;

        let pc = note.pitch.pitch_class().value() as i32;
        let relative_pc = (pc - tonic).rem_euclid(12) as usize;
        pitch_class_hist[relative_pc] += duration;

        let duration_bin = RHYTHM_THRESHOLDS
            .iter()
            .position(|&threshold| duration <= threshold + 1e-12)
            .unwrap_or(RHYTHM_BINS - 1);
        rhythm_hist[duration_bin] += 1.0;
    }

    let mut interval_sum = 0.0;
    for pair in melody.windows(2) {
        let delta = pair[1].pitch.midi() as i32 - pair[0].pitch.midi() as i32;
        let magnitude = delta.unsigned_abs() as usize;
        interval_hist[magnitude.min(INTERVAL_BINS - 1)] += 1.0;
        interval_sum += magnitude as f64;

        let contour_bin = match delta.cmp(&0) {
            std::cmp::Ordering::Greater => 0,
            std::cmp::Ordering::Less => 1,
            std::cmp::Ordering::Equal => 2,
        };
        contour_hist[contour_bin] += 1.0;
    }

    normalize(&mut pitch_class_hist);
    normalize(&mut rhythm_hist);
    normalize(&mut interval_hist);
    normalize(&mut contour_hist);

    let directional = contour_hist[0] + contour_hist[1];
    let contour_asymmetry = if directional <= f64::EPSILON {
        0.0
    } else {
        (contour_hist[0] - contour_hist[1]).abs().min(directional) / directional
    };

    MusicalStateFrame {
        start_beat: start,
        end_beat: end,
        event_count: notes.len(),
        onset_density: notes.len() as f64 / (end - start).max(1e-9),
        pitch_class_hist,
        rhythm_hist,
        interval_hist,
        contour_hist,
        pitch_entropy: normalized_entropy(&pitch_class_hist),
        rhythm_entropy: normalized_entropy(&rhythm_hist),
        mean_interval: if melody.len() < 2 {
            0.0
        } else {
            (interval_sum / (melody.len() - 1) as f64 / 11.0).clamp(0.0, 1.0)
        },
        contour_asymmetry,
        structural_intensity: (intensity_sum / notes.len().max(1) as f64).clamp(0.0, 1.0),
        nearest_prior_similarity: None,
        novelty: None,
    }
}

fn normalize<const N: usize>(hist: &mut [f64; N]) {
    let total = hist.iter().sum::<f64>();
    if total <= f64::EPSILON {
        return;
    }
    for value in hist.iter_mut() {
        *value /= total;
    }
}

fn normalized_entropy<const N: usize>(hist: &[f64; N]) -> f64 {
    let max_entropy = (N as f64).ln();
    if max_entropy <= f64::EPSILON {
        return 0.0;
    }
    let entropy = hist
        .iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.ln())
        .sum::<f64>();
    (entropy / max_entropy).clamp(0.0, 1.0)
}

fn cosine_similarity<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    let dot = a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a <= f64::EPSILON || norm_b <= f64::EPSILON {
        0.0
    } else {
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::harmony::Key;
    use crate::pitch::{Pitch, PitchClass};
    use crate::rhythm::Duration;
    use crate::score::{Emphasis, PartId};

    fn note(pc: u8, octave: i32, onset: i64) -> ScoreNote {
        ScoreNote {
            part: PartId::UNASSIGNED,
            pitch: Pitch::new(PitchClass::new(pc), octave),
            onset: Duration::new(onset, 1),
            duration: Duration::quarter(),
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn score(notes: &[ScoreNote], tonic: u8) -> Score {
        let mut score = Score::new(Key::major(PitchClass::new(tonic)), 120.0, 4);
        for n in notes.iter().copied() {
            score.push(n);
        }
        score
    }

    #[test]
    fn vector_is_fixed_width_and_transposition_invariant() {
        let a = score(
            &[
                note(0, 4, 0),
                note(2, 4, 1),
                note(4, 4, 2),
                note(7, 4, 3),
            ],
            0,
        );
        let b = score(
            &[
                note(2, 4, 0),
                note(4, 4, 1),
                note(6, 4, 2),
                note(9, 4, 3),
            ],
            2,
        );

        let ta = MusicalStateTrajectory::from_score(&a, 4.0, 4.0).unwrap();
        let tb = MusicalStateTrajectory::from_score(&b, 4.0, 4.0).unwrap();

        assert_eq!(ta.frames.len(), 1);
        assert_eq!(tb.frames.len(), 1);
        assert_eq!(ta.frames[0].vector().len(), STATE_DIMS);
        assert!((ta.frames[0].similarity(&tb.frames[0]) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn repeated_windows_form_recurrence() {
        let notes = [
            note(0, 4, 0),
            note(2, 4, 1),
            note(4, 4, 2),
            note(0, 4, 3),
            note(2, 4, 4),
            note(4, 4, 5),
        ];
        let trajectory =
            MusicalStateTrajectory::from_score(&score(&notes, 0), 3.0, 3.0).unwrap();

        assert_eq!(trajectory.frames.len(), 2);
        assert!(
            trajectory.frames[1]
                .nearest_prior_similarity
                .expect("second frame has a prior")
                > 0.99
        );
        assert!(
            trajectory.frames[1].novelty.expect("second frame has a prior") < 0.01
        );
        assert!(trajectory.recurrence_rate(0.99) > 0.9);
    }

    #[test]
    fn distinct_windows_are_not_forced_to_recur() {
        let notes = [
            note(0, 4, 0),
            note(2, 4, 1),
            note(4, 4, 2),
            note(0, 5, 3),
            note(11, 4, 4),
            note(10, 4, 5),
        ];
        let trajectory =
            MusicalStateTrajectory::from_score(&score(&notes, 0), 3.0, 3.0).unwrap();

        assert_eq!(trajectory.frames.len(), 2);
        assert!(trajectory.frames[1].novelty.unwrap() > 0.01);
        assert!(trajectory.recurrence_pairs(0.99).is_empty());
    }

    #[test]
    fn invalid_window_sizes_fail_closed() {
        let s = score(&[], 0);
        assert!(MusicalStateTrajectory::from_score(&s, 0.0, 1.0).is_err());
        assert!(MusicalStateTrajectory::from_score(&s, 1.0, 0.0).is_err());
        assert!(MusicalStateTrajectory::from_score(&s, f64::NAN, 1.0).is_err());
    }
}
