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

use crate::rhythm::Duration;
use crate::score::{PartId, Score, ScoreNote};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Number of dimensions in a frame's canonical structural vector.
///
/// Layout: six bounded scalars, a part-identity availability bit, 12 pitch-
/// class bins, 5 rhythm bins, 12 verified line-interval bins, 3 verified
/// line-contour bins, and 8 register bins.
pub const MUSICAL_STATE_SPACE_V1: &str = "musical-state-space-v1";
pub const STATE_DIMS: usize = 47;

const PITCH_CLASS_BINS: usize = 12;
const RHYTHM_BINS: usize = 5;
const INTERVAL_BINS: usize = 12;
const CONTOUR_BINS: usize = 3;
const REGISTER_BINS: usize = 8;
const RHYTHM_THRESHOLDS: [f64; RHYTHM_BINS - 1] = [0.25, 0.5, 1.0, 2.0];
/// Bounds quadratic recurrence analysis and prevents tiny-hop resource exhaustion.
const MAX_TRAJECTORY_WINDOWS: usize = 4096;

/// One temporal slice of a symbolic score.
///
/// Novelty is relative to the best-matching earlier frame in the same
/// trajectory. It is not a claim about novelty relative to a training corpus
/// or to human music.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalStateFrame {
    pub schema_version: String,
    pub start_beat: f64,
    pub end_beat: f64,
    pub event_count: usize,
    pub onset_density: f64,
    /// Duration-weighted pitch-class occupancy relative to the score-level tonic.
    pub pitch_class_hist: [f64; PITCH_CLASS_BINS],
    pub rhythm_hist: [f64; RHYTHM_BINS],
    /// Per-part interval magnitudes; zeroed when persistent part identity is unavailable.
    pub line_interval_hist: [f64; INTERVAL_BINS],
    /// Per-part contour distribution; zeroed when persistent part identity is unavailable.
    pub line_contour_hist: [f64; CONTOUR_BINS],
    /// Duration-weighted MIDI register occupancy within the frame in eight bins.
    pub register_hist: [f64; REGISTER_BINS],
    /// True only when every sounding note contributing to the frame has a real PartId.
    pub part_identity_available: bool,
    /// Number of verified transitions between adjacent singleton onsets within a part.
    pub line_transition_count: usize,
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
        // Bound scalar magnitudes so density cannot overwhelm distributional
        // features merely because its natural unit is unbounded attacks/beat.
        v[0] = (self.onset_density / 4.0).clamp(0.0, 1.0);
        v[1] = self.pitch_entropy;
        v[2] = self.rhythm_entropy;
        v[3] = self.mean_interval;
        v[4] = self.contour_asymmetry;
        v[5] = self.structural_intensity;
        v[6] = if self.part_identity_available { 1.0 } else { 0.0 };

        let mut offset = 7;
        v[offset..offset + PITCH_CLASS_BINS].copy_from_slice(&self.pitch_class_hist);
        offset += PITCH_CLASS_BINS;
        v[offset..offset + RHYTHM_BINS].copy_from_slice(&self.rhythm_hist);
        offset += RHYTHM_BINS;
        v[offset..offset + INTERVAL_BINS].copy_from_slice(&self.line_interval_hist);
        offset += INTERVAL_BINS;
        v[offset..offset + CONTOUR_BINS].copy_from_slice(&self.line_contour_hist);
        offset += CONTOUR_BINS;
        v[offset..offset + REGISTER_BINS].copy_from_slice(&self.register_hist);
        v
    }

    /// Build one deterministic frame for a caller-selected region.
    /// Returns None for empty, reversed, negative, out-of-score, or truly silent regions.
    /// A region containing only carried-in sustains is still a valid frame; its
    /// event_count is zero while occupancy features describe the sounding notes.
    pub fn from_region(score: &Score, start: Duration, end: Duration) -> Option<Self> {
        let start_beat = start.beats();
        let end_beat = end.beats();
        let total = score.total_beats.beats();
        if start_beat < 0.0 || end_beat <= start_beat || end_beat > total + 1e-9 {
            return None;
        }
        let notes = notes_in_window(score, start_beat, end_beat);
        let active_notes = overlapping_notes_in_window(score, start_beat, end_beat);
        if active_notes.is_empty() {
            return None;
        }
        Some(frame_from_notes(
            score,
            start_beat,
            end_beat,
            &notes,
            &active_notes,
        ))
    }

    /// Cosine similarity between two canonical frame states.
    pub fn similarity(&self, other: &Self) -> f64 {
        cosine_similarity(&self.vector(), &other.vector())
    }
}

/// A time-ordered sequence of MusicalStateFrame values.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MusicalStateTrajectory {
    pub schema_version: String,
    pub window_beats: f64,
    pub hop_beats: f64,
    pub frames: Vec<MusicalStateFrame>,
}

impl MusicalStateTrajectory {
    /// Build a trajectory using fixed-size windows and hops.
    ///
    /// Windows are emitted whenever at least one note sounds, including
    /// sustain-only windows with no new attacks. Truly silent windows are
    /// omitted so silence outside a score does not create artificial recurrence.
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
                schema_version: MUSICAL_STATE_SPACE_V1.into(),
                window_beats,
                hop_beats,
                frames: Vec::new(),
            });
        }

        // Fast-reject clearly pathological hops before repeatedly scanning the
        // score. Keep a one-window rounding margin; the bounded index loop below
        // is authoritative at the exact limit and prevents false rejection.
        let hop_limited = (total / hop_beats).ceil();
        let window_limited = ((total - window_beats).max(0.0) / hop_beats).ceil() + 1.0;
        let estimated_windows = hop_limited.min(window_limited);
        if !estimated_windows.is_finite()
            || estimated_windows > MAX_TRAJECTORY_WINDOWS as f64 + 1.0
        {
            return Err(format!(
                "hop_beats would produce more than {MAX_TRAJECTORY_WINDOWS} trajectory windows"
            ));
        }

        // Index-derived starts avoid cumulative drift from repeated hop addition.
        // This bounded loop is authoritative when the estimate is near the limit.
        let mut frames = Vec::new();
        let mut frame_index = 0usize;
        loop {
            let start = frame_index as f64 * hop_beats;
            if start >= total {
                break;
            }
            if frame_index >= MAX_TRAJECTORY_WINDOWS {
                return Err(format!(
                    "hop_beats would produce more than {MAX_TRAJECTORY_WINDOWS} trajectory windows"
                ));
            }
            let end = (start + window_beats).min(total);
            let notes = notes_in_window(score, start, end);
            let active_notes = overlapping_notes_in_window(score, start, end);
            if !active_notes.is_empty() {
                frames.push(frame_from_notes(
                    score,
                    start,
                    end,
                    &notes,
                    &active_notes,
                ));
            }
            if end >= total {
                break;
            }
            frame_index += 1;
        }

        // Only earlier frames are allowed to influence recurrence/novelty.
        for i in 1..frames.len() {
            // Every emitted frame contains at least one sounding note, even if
            // it has no new attacks. Such sustain-only states are valid prior
            // context for recurrence and novelty.
            let best = (0..i)
                .map(|j| frames[i].similarity(&frames[j]))
                .max_by(f64::total_cmp);
            frames[i].nearest_prior_similarity = best;
            frames[i].novelty = best.map(|s| (1.0 - s).clamp(0.0, 1.0));
        }

        Ok(Self {
            schema_version: MUSICAL_STATE_SPACE_V1.into(),
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

    /// Return local maxima of best-prior novelty above the threshold.
    ///
    /// Consecutive equal-valued maxima form one plateau and produce its
    /// lower-middle index. Missing novelty values split runs; non-finite values
    /// are ignored. A plateau with no strictly lower neighbor is not a peak.
    pub fn novelty_peaks(&self, minimum_novelty: f64) -> Vec<usize> {
        let values: Vec<Option<f64>> = self
            .frames
            .iter()
            .map(|frame| frame.novelty.filter(|value| value.is_finite()))
            .collect();
        let mut peaks = Vec::new();
        let mut run_start = 0usize;

        while run_start < values.len() {
            while run_start < values.len() && values[run_start].is_none() {
                run_start += 1;
            }
            if run_start == values.len() {
                break;
            }

            let mut run_end = run_start;
            while run_end + 1 < values.len() && values[run_end + 1].is_some() {
                run_end += 1;
            }

            let mut group_start = run_start;
            while group_start <= run_end {
                let value = values[group_start].expect("run contains only finite values");
                let mut group_end = group_start;
                while group_end < run_end && values[group_end + 1] == Some(value) {
                    group_end += 1;
                }

                let left = (group_start > run_start)
                    .then(|| values[group_start - 1].expect("within finite run"));
                let right = (group_end < run_end)
                    .then(|| values[group_end + 1].expect("within finite run"));
                let not_below_neighbor =
                    left.is_none_or(|neighbor| value >= neighbor)
                        && right.is_none_or(|neighbor| value >= neighbor);
                let strictly_above_neighbor =
                    left.is_some_and(|neighbor| value > neighbor)
                        || right.is_some_and(|neighbor| value > neighbor);

                if value >= minimum_novelty && not_below_neighbor && strictly_above_neighbor {
                    peaks.push((group_start + group_end) / 2);
                }
                group_start = group_end + 1;
            }

            run_start = run_end + 1;
        }

        peaks
    }
}

fn overlapping_notes_in_window(score: &Score, start: f64, end: f64) -> Vec<ScoreNote> {
    score
        .notes
        .iter()
        .copied()
        .filter(|note| {
            let onset = note.onset.beats();
            let note_end = (note.onset + note.duration).beats();
            (note_end.min(end) - onset.max(start)) > 0.0
        })
        .collect()
}

fn notes_in_window(score: &Score, start: f64, end: f64) -> Vec<ScoreNote> {
    let mut notes: Vec<_> = score
        .notes
        .iter()
        .copied()
        .filter(|note| {
            let onset = note.onset.beats();
            onset >= start && onset < end
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

fn frame_from_notes(
    score: &Score,
    start: f64,
    end: f64,
    notes: &[ScoreNote],
    active_notes: &[ScoreNote],
) -> MusicalStateFrame {
    let mut pitch_class_hist = [0.0; PITCH_CLASS_BINS];
    let mut rhythm_hist = [0.0; RHYTHM_BINS];
    let mut line_interval_hist = [0.0; INTERVAL_BINS];
    let mut line_contour_hist = [0.0; CONTOUR_BINS];
    let mut register_hist = [0.0; REGISTER_BINS];

    let tonic = score.key.tonic.value() as i32;
    let part_identity_available = !active_notes.is_empty()
        && active_notes.iter().all(|note| note.part.is_assigned());

    for note in notes {
        let duration = note.duration.beats().max(0.0);

        let duration_bin = RHYTHM_THRESHOLDS
            .iter()
            .position(|&threshold| duration <= threshold + 1e-12)
            .unwrap_or(RHYTHM_BINS - 1);
        rhythm_hist[duration_bin] += 1.0;
    }

    // Structural intensity follows the notes actually sounding in this window,
    // including carry-ins, rather than dropping to zero in a sustain-only frame.
    let intensity_sum = active_notes
        .iter()
        .map(|note| note.section_intensity as f64)
        .sum::<f64>();

    // Pitch-class and register features describe occupancy *inside* the frame,
    // not the full notated duration of attacks. Include notes carried in from
    // earlier windows and clip every contribution to the current interval.
    for note in active_notes {
        let onset = note.onset.beats();
        let note_end = (note.onset + note.duration).beats();
        let overlap = (note_end.min(end) - onset.max(start)).max(0.0);
        if overlap <= 0.0 {
            continue;
        }

        let pc = note.pitch.pitch_class().value() as i32;
        let relative_pc = (pc - tonic).rem_euclid(12) as usize;
        pitch_class_hist[relative_pc] += overlap;

        let midi = note.pitch.midi().clamp(0, 127) as usize;
        register_hist[(midi / 16).min(REGISTER_BINS - 1)] += overlap;
    }

    // PartId—not VoiceRole—is the identity of a continuing line. If any event
    // lacks part identity, do not infer a line by sorting functional roles.
    let mut line_transition_count = 0usize;
    let mut interval_sum = 0.0;
    if part_identity_available {
        let mut lines: BTreeMap<PartId, Vec<ScoreNote>> = BTreeMap::new();
        for note in notes {
            lines.entry(note.part).or_default().push(*note);
        }
        for line in lines.values_mut() {
            line.sort_by(|a, b| a.onset.beats().total_cmp(&b.onset.beats()));
            // Same-onset groups with multiple notes are ambiguous as melodic
            // transitions. Preserve the gap instead of making a false edge.
            let mut unique_onsets: Vec<Option<ScoreNote>> = Vec::new();
            let mut i = 0;
            while i < line.len() {
                let mut j = i + 1;
                while j < line.len()
                    && (line[j].onset.beats() - line[i].onset.beats()).abs() <= 1e-9
                {
                    j += 1;
                }
                unique_onsets.push(if j - i == 1 { Some(line[i]) } else { None });
                i = j;
            }
            for pair in unique_onsets.windows(2) {
                let (Some(left), Some(right)) = (pair[0], pair[1]) else {
                    continue;
                };
                let delta = right.pitch.midi() as i32 - left.pitch.midi() as i32;
                let magnitude = delta.unsigned_abs() as usize;
                line_interval_hist[magnitude.min(INTERVAL_BINS - 1)] += 1.0;
                interval_sum += magnitude as f64;
                line_transition_count += 1;

                let contour_bin = match delta.cmp(&0) {
                    std::cmp::Ordering::Greater => 0,
                    std::cmp::Ordering::Less => 1,
                    std::cmp::Ordering::Equal => 2,
                };
                line_contour_hist[contour_bin] += 1.0;
            }
        }
    }

    normalize(&mut pitch_class_hist);
    normalize(&mut rhythm_hist);
    normalize(&mut line_interval_hist);
    normalize(&mut line_contour_hist);
    normalize(&mut register_hist);

    let directional = line_contour_hist[0] + line_contour_hist[1];
    let contour_asymmetry = if directional <= f64::EPSILON {
        0.0
    } else {
        (line_contour_hist[0] - line_contour_hist[1]).abs() / directional
    };

    MusicalStateFrame {
        schema_version: MUSICAL_STATE_SPACE_V1.into(),
        start_beat: start,
        end_beat: end,
        event_count: notes.len(),
        onset_density: notes.len() as f64 / (end - start).max(1e-9),
        pitch_class_hist,
        rhythm_hist,
        line_interval_hist,
        line_contour_hist,
        register_hist,
        part_identity_available,
        line_transition_count,
        pitch_entropy: normalized_entropy(&pitch_class_hist),
        rhythm_entropy: normalized_entropy(&rhythm_hist),
        mean_interval: if line_transition_count == 0 {
            0.0
        } else {
            (interval_sum / line_transition_count as f64 / 11.0).clamp(0.0, 1.0)
        },
        contour_asymmetry,
        structural_intensity: (intensity_sum / active_notes.len().max(1) as f64).clamp(0.0, 1.0),
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
    use crate::score::{Emphasis, PartId, VoiceRole};

    fn note(pc: i32, octave: i32, onset: i64) -> ScoreNote {
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

    fn score(notes: &[ScoreNote], tonic: i32) -> Score {
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
        assert_eq!(ta.schema_version, MUSICAL_STATE_SPACE_V1);
        assert_eq!(ta.frames[0].schema_version, MUSICAL_STATE_SPACE_V1);
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
    fn line_motion_requires_explicit_part_identity() {
        let unassigned = score(&[note(0, 4, 0), note(2, 4, 1), note(4, 4, 2)], 0);
        let frame = MusicalStateTrajectory::from_score(&unassigned, 3.0, 3.0)
            .unwrap()
            .frames
            .remove(0);
        assert!(!frame.part_identity_available);
        assert_eq!(frame.line_transition_count, 0);
        assert!(frame.line_interval_hist.iter().all(|value| *value == 0.0));

        let mut assigned = Score::new(Key::major(PitchClass::new(0)), 120.0, 4);
        for (pc, onset) in [(0, 0), (2, 1), (4, 2)] {
            let mut n = note(pc, 4, onset);
            n.part = PartId(7);
            assigned.push(n);
        }
        let frame = MusicalStateTrajectory::from_score(&assigned, 3.0, 3.0)
            .unwrap()
            .frames
            .remove(0);
        assert!(frame.part_identity_available);
        assert_eq!(frame.line_transition_count, 2);
    }

    #[test]
    fn duration_weighted_occupancy_includes_carry_ins_and_clips_to_window() {
        let mut held_c = note(0, 4, 0);
        held_c.duration = Duration::new(4, 1);
        let mut attack_g = note(7, 5, 2);
        attack_g.duration = Duration::new(2, 1);
        let piece = score(&[held_c, attack_g], 0);

        // Only the G attack starts in [2, 2.5), but both notes sound for half
        // a beat inside that region. Their out-of-window tails must not count.
        let frame = MusicalStateFrame::from_region(
            &piece,
            Duration::new(2, 1),
            Duration::new(5, 2),
        )
        .expect("the region contains an attack");

        assert_eq!(frame.event_count, 1);
        assert!((frame.pitch_class_hist[0] - 0.5).abs() < 1e-9);
        assert!((frame.pitch_class_hist[7] - 0.5).abs() < 1e-9);
        assert!((frame.register_hist[3] - 0.5).abs() < 1e-9);
        assert!((frame.register_hist[4] - 0.5).abs() < 1e-9);
    }

    #[test]
    fn sustain_only_windows_are_kept_with_zero_attacks() {
        let mut held = note(0, 4, 0);
        held.duration = Duration::new(4, 1);
        held.part = PartId(7);
        held.section_intensity = 0.6;
        let piece = score(&[held], 0);

        let sustain = MusicalStateFrame::from_region(
            &piece,
            Duration::new(1, 1),
            Duration::new(2, 1),
        )
        .expect("the sustained note sounds inside the frame");
        assert_eq!(sustain.event_count, 0);
        assert_eq!(sustain.onset_density, 0.0);
        assert!(sustain.part_identity_available);
        assert!((sustain.pitch_class_hist[0] - 1.0).abs() < 1e-9);
        assert!((sustain.register_hist[3] - 1.0).abs() < 1e-9);
        assert!((sustain.structural_intensity - 0.6).abs() < 1e-9);

        let trajectory = MusicalStateTrajectory::from_score(&piece, 1.0, 1.0).unwrap();
        assert_eq!(trajectory.frames.len(), 4);
        assert_eq!(trajectory.frames[0].event_count, 1);
        assert!(trajectory.frames[1..].iter().all(|frame| frame.event_count == 0));
        assert!(trajectory.frames[1..]
            .iter()
            .all(|frame| frame.pitch_class_hist[0] == 1.0));
        assert!((trajectory.frames[2].nearest_prior_similarity.unwrap() - 1.0).abs() < 1e-9);
        assert!(trajectory.frames[2].novelty.unwrap() < 1e-9);
    }

    #[test]
    fn register_profile_is_transposition_sensitive() {
        let low = score(&[note(0, 2, 0), note(2, 2, 1)], 0);
        let high = score(&[note(0, 6, 0), note(2, 6, 1)], 0);
        let low_frame = MusicalStateTrajectory::from_score(&low, 2.0, 2.0)
            .unwrap()
            .frames
            .remove(0);
        let high_frame = MusicalStateTrajectory::from_score(&high, 2.0, 2.0)
            .unwrap()
            .frames
            .remove(0);
        assert_ne!(low_frame.register_hist, high_frame.register_hist);
        assert!(low_frame.similarity(&high_frame) < 0.999);
    }

    #[test]
    fn tiny_hops_fail_closed_before_quadratic_analysis() {
        let s = score(&[note(0, 4, 0), note(2, 4, 1), note(4, 4, 2)], 0);
        assert!(MusicalStateTrajectory::from_score(&s, 1.0, 1e-9).is_err());
        let full_window = MusicalStateTrajectory::from_score(&s, 3.0, 1e-9).unwrap();
        assert_eq!(full_window.frames.len(), 1);
    }

    #[test]
    fn frame_budget_allows_exact_limit_with_short_windows() {
        let s = score(&[note(0, 4, 0), note(7, 4, 4095)], 0);
        let trajectory = MusicalStateTrajectory::from_score(&s, 0.1, 1.0)
            .expect("4096 possible window starts are within the fixed frame budget");
        assert_eq!(trajectory.frames.len(), 2);
    }

    #[test]
    fn novelty_peaks_select_local_maxima_and_collapse_plateaus() {
        let notes: Vec<_> = (0..6).map(|index| note(index, 4, index as i64)).collect();
        let mut trajectory =
            MusicalStateTrajectory::from_score(&score(&notes, 0), 1.0, 1.0).unwrap();
        for (frame, novelty) in trajectory.frames.iter_mut().zip([
            None,
            Some(0.2),
            Some(0.8),
            Some(0.8),
            Some(0.3),
            Some(0.9),
        ]) {
            frame.novelty = novelty;
        }

        assert_eq!(trajectory.novelty_peaks(0.5), vec![2, 5]);
        assert_eq!(trajectory.novelty_peaks(0.95), Vec::<usize>::new());
    }

    #[test]
    fn flat_novelty_run_is_not_reported_as_a_peak() {
        let notes: Vec<_> = (0..4).map(|index| note(index, 4, index as i64)).collect();
        let mut trajectory =
            MusicalStateTrajectory::from_score(&score(&notes, 0), 1.0, 1.0).unwrap();
        for (frame, novelty) in trajectory.frames.iter_mut().zip([
            None,
            Some(0.7),
            Some(0.7),
            Some(0.7),
        ]) {
            frame.novelty = novelty;
        }

        assert!(trajectory.novelty_peaks(0.5).is_empty());
    }

    #[test]
    fn invalid_window_sizes_fail_closed() {
        let s = score(&[], 0);
        assert!(MusicalStateTrajectory::from_score(&s, 0.0, 1.0).is_err());
        assert!(MusicalStateTrajectory::from_score(&s, 1.0, 0.0).is_err());
        assert!(MusicalStateTrajectory::from_score(&s, f64::NAN, 1.0).is_err());
    }
}
