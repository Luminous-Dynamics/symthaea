// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Source-exact MusPy-style raster metrics over MEL-BENCH's exact time grid.
//!
//! Compatibility in this module is intentionally narrow: one non-drum/piano
//! symbolic population. MusPy's empty-beat/measure and groove functions include
//! every track, while Melothaea's current benchmark `Score` does not encode a
//! drum-track flag. This module therefore does not generalize its parity claim to
//! arbitrary multitrack MIDI.

use crate::evidence_digest::benchmark_symbolic_parity::{
    MUSPY_METRICS_SOURCE_BLOB, MUSPY_REFERENCE_REVISION,
};
use crate::evidence_digest::benchmark_time_grid::{
    ExactSymbolicRasterV1, ExternalTimeGridFamilyV1, MuspyPolyphonyRateComparatorV1,
    SYMBOLIC_TIME_GRID_VERSION,
};
use crate::evidence_digest::canonical_json_sha256;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const MUSPY_RASTER_METRICS_VERSION: &str = "melothaea-muspy-raster-metrics-v1";
pub const MUSPY_DEFAULT_POLYPHONY_RATE_THRESHOLD: u16 = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MuspyRasterTrackProfileV1 {
    SingleNonDrumPiano,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MuspyRasterMetricsV1 {
    pub metrics_version: String,
    pub source_revision: String,
    pub metrics_source_blob: String,
    pub track_profile: MuspyRasterTrackProfileV1,
    pub raster_sha256: String,
    pub polyphony_rate_threshold: u16,
    /// `None` represents the exact MusPy undefined/NaN boundary without
    /// introducing non-finite values into canonical benchmark evidence.
    pub polyphony: Option<f64>,
    pub polyphony_rate: Option<f64>,
    pub empty_beat_rate: Option<f64>,
    pub empty_measure_rate: Option<f64>,
    pub groove_consistency: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MuspyRasterMetricErrorV1 {
    WrongGridVersion { found: String },
    WrongGridFamily,
    WrongPolyphonyComparator,
    WrongSourceRevision { found: String },
    WrongSourceBlob { found: String },
    ZeroResolution,
    InvalidMeasureResolution,
    InvalidSourceScoreDigest,
    RasterNotesNotCanonical,
    PitchOutsideMidiRange { note_index: usize, pitch: u8 },
    NonPositiveDuration { note_index: usize },
    NoteEndOverflow { note_index: usize },
    NoteOutsideRaster {
        note_index: usize,
        end_step: u64,
        total_steps: u64,
    },
    TotalStepsMismatch { expected: u64, found: u64 },
    BucketCountOverflow,
    Serialization,
}

impl MuspyRasterMetricsV1 {
    pub fn canonical_sha256(&self) -> Result<String, serde_json::Error> {
        canonical_json_sha256(self)
    }
}

/// Compute source-exact metrics from a previously exact-rasterized piano score.
///
/// Important external semantics retained literally:
/// - piano-roll occupancy is binary per pitch, so duplicate same-pitch notes do
///   not increase polyphony;
/// - `polyphony_rate` uses **strictly greater than** `threshold`;
/// - empty beat/measure accounting uses `note.end // resolution` inclusively,
///   including a bucket at an exact note-end boundary;
/// - groove compares neighboring binary onset vectors by Hamming distance.
pub fn measure_muspy_piano_raster_metrics(
    raster: &ExactSymbolicRasterV1,
    polyphony_rate_threshold: u16,
) -> Result<MuspyRasterMetricsV1, MuspyRasterMetricErrorV1> {
    validate_raster(raster)?;
    let raster_sha256 = raster
        .canonical_sha256()
        .map_err(|_| MuspyRasterMetricErrorV1::Serialization)?;

    let active = active_pitch_summary(raster)?;
    let polyphony = if active.active_steps == 0 {
        None
    } else {
        Some(active.pitch_step_sum as f64 / active.active_steps as f64)
    };
    let polyphony_rate = if raster.total_steps == 0 {
        None
    } else {
        Some(
            active.steps_strictly_above(polyphony_rate_threshold) as f64
                / raster.total_steps as f64,
        )
    };

    let empty_beat_rate = empty_bucket_rate(raster, u64::from(raster.policy.steps_per_quarter))?;
    let empty_measure_rate =
        empty_bucket_rate(raster, u64::from(raster.policy.steps_per_measure))?;
    let groove_consistency = groove_consistency(raster)?;

    Ok(MuspyRasterMetricsV1 {
        metrics_version: MUSPY_RASTER_METRICS_VERSION.into(),
        source_revision: MUSPY_REFERENCE_REVISION.into(),
        metrics_source_blob: MUSPY_METRICS_SOURCE_BLOB.into(),
        track_profile: MuspyRasterTrackProfileV1::SingleNonDrumPiano,
        raster_sha256,
        polyphony_rate_threshold,
        polyphony,
        polyphony_rate,
        empty_beat_rate,
        empty_measure_rate,
        groove_consistency,
    })
}

fn validate_raster(raster: &ExactSymbolicRasterV1) -> Result<(), MuspyRasterMetricErrorV1> {
    let policy = &raster.policy;
    if policy.version != SYMBOLIC_TIME_GRID_VERSION {
        return Err(MuspyRasterMetricErrorV1::WrongGridVersion {
            found: policy.version.clone(),
        });
    }
    if policy.family != ExternalTimeGridFamilyV1::MusPyResolution {
        return Err(MuspyRasterMetricErrorV1::WrongGridFamily);
    }
    if policy.muspy_polyphony_rate_comparator
        != MuspyPolyphonyRateComparatorV1::StrictlyGreaterThanThreshold
    {
        return Err(MuspyRasterMetricErrorV1::WrongPolyphonyComparator);
    }
    if policy.muspy_reference_revision != MUSPY_REFERENCE_REVISION {
        return Err(MuspyRasterMetricErrorV1::WrongSourceRevision {
            found: policy.muspy_reference_revision.clone(),
        });
    }
    if policy.muspy_metrics_source_blob != MUSPY_METRICS_SOURCE_BLOB {
        return Err(MuspyRasterMetricErrorV1::WrongSourceBlob {
            found: policy.muspy_metrics_source_blob.clone(),
        });
    }
    if policy.steps_per_quarter == 0 {
        return Err(MuspyRasterMetricErrorV1::ZeroResolution);
    }
    if policy.steps_per_measure == 0
        || policy.steps_per_measure % policy.steps_per_quarter != 0
    {
        return Err(MuspyRasterMetricErrorV1::InvalidMeasureResolution);
    }
    if !is_sha256(&raster.source_score_sha256) {
        return Err(MuspyRasterMetricErrorV1::InvalidSourceScoreDigest);
    }
    if !raster.notes.windows(2).all(|pair| pair[0] <= pair[1]) {
        return Err(MuspyRasterMetricErrorV1::RasterNotesNotCanonical);
    }

    let mut expected_total = 0u64;
    for (note_index, note) in raster.notes.iter().enumerate() {
        if note.pitch > 127 {
            return Err(MuspyRasterMetricErrorV1::PitchOutsideMidiRange {
                note_index,
                pitch: note.pitch,
            });
        }
        if note.duration_steps == 0 {
            return Err(MuspyRasterMetricErrorV1::NonPositiveDuration { note_index });
        }
        let end_step = note
            .start_step
            .checked_add(note.duration_steps)
            .ok_or(MuspyRasterMetricErrorV1::NoteEndOverflow { note_index })?;
        if end_step > raster.total_steps {
            return Err(MuspyRasterMetricErrorV1::NoteOutsideRaster {
                note_index,
                end_step,
                total_steps: raster.total_steps,
            });
        }
        expected_total = expected_total.max(end_step);
    }
    if raster.total_steps != expected_total {
        return Err(MuspyRasterMetricErrorV1::TotalStepsMismatch {
            expected: expected_total,
            found: raster.total_steps,
        });
    }
    Ok(())
}

struct ActivePitchSummary {
    active_steps: u128,
    pitch_step_sum: u128,
    spans: Vec<(u64, u64, u16)>,
}

impl ActivePitchSummary {
    fn steps_strictly_above(&self, threshold: u16) -> u128 {
        self.spans
            .iter()
            .filter(|(_, _, active)| *active > threshold)
            .map(|(start, end, _)| u128::from(end - start))
            .sum()
    }
}

fn active_pitch_summary(
    raster: &ExactSymbolicRasterV1,
) -> Result<ActivePitchSummary, MuspyRasterMetricErrorV1> {
    let mut per_pitch: [Vec<(u64, u64)>; 128] = std::array::from_fn(|_| Vec::new());
    for (note_index, note) in raster.notes.iter().enumerate() {
        let end = note
            .start_step
            .checked_add(note.duration_steps)
            .ok_or(MuspyRasterMetricErrorV1::NoteEndOverflow { note_index })?;
        per_pitch[usize::from(note.pitch)].push((note.start_step, end));
    }

    let mut events: BTreeMap<u64, i32> = BTreeMap::new();
    for intervals in &mut per_pitch {
        if intervals.is_empty() {
            continue;
        }
        intervals.sort_unstable();
        let mut current = intervals[0];
        for &(start, end) in &intervals[1..] {
            if start <= current.1 {
                current.1 = current.1.max(end);
            } else {
                add_interval_events(&mut events, current);
                current = (start, end);
            }
        }
        add_interval_events(&mut events, current);
    }

    let mut active = 0i32;
    let mut previous = 0u64;
    let mut active_steps = 0u128;
    let mut pitch_step_sum = 0u128;
    let mut spans = Vec::new();
    for (position, delta) in events {
        if position > previous && active > 0 {
            let span = position - previous;
            active_steps += u128::from(span);
            pitch_step_sum += u128::from(span) * active as u128;
            spans.push((previous, position, active as u16));
        }
        active += delta;
        previous = position;
    }

    Ok(ActivePitchSummary {
        active_steps,
        pitch_step_sum,
        spans,
    })
}

fn add_interval_events(events: &mut BTreeMap<u64, i32>, interval: (u64, u64)) {
    *events.entry(interval.0).or_insert(0) += 1;
    *events.entry(interval.1).or_insert(0) -= 1;
}

fn empty_bucket_rate(
    raster: &ExactSymbolicRasterV1,
    bucket_resolution: u64,
) -> Result<Option<f64>, MuspyRasterMetricErrorV1> {
    if raster.total_steps < 1 {
        return Ok(None);
    }
    let bucket_count = raster
        .total_steps
        .checked_div(bucket_resolution)
        .and_then(|count| count.checked_add(1))
        .ok_or(MuspyRasterMetricErrorV1::BucketCountOverflow)?;

    let mut occupied = Vec::with_capacity(raster.notes.len());
    for (note_index, note) in raster.notes.iter().enumerate() {
        let end = note
            .start_step
            .checked_add(note.duration_steps)
            .ok_or(MuspyRasterMetricErrorV1::NoteEndOverflow { note_index })?;
        occupied.push((note.start_step / bucket_resolution, end / bucket_resolution));
    }
    let occupied_count = union_inclusive_count(&mut occupied);
    Ok(Some(1.0 - occupied_count as f64 / bucket_count as f64))
}

fn union_inclusive_count(intervals: &mut [(u64, u64)]) -> u128 {
    if intervals.is_empty() {
        return 0;
    }
    intervals.sort_unstable();
    let mut total = 0u128;
    let mut current = intervals[0];
    for &(start, end) in &intervals[1..] {
        if start <= current.1.saturating_add(1) {
            current.1 = current.1.max(end);
        } else {
            total += u128::from(current.1 - current.0) + 1;
            current = (start, end);
        }
    }
    total + u128::from(current.1 - current.0) + 1
}

fn groove_consistency(
    raster: &ExactSymbolicRasterV1,
) -> Result<Option<f64>, MuspyRasterMetricErrorV1> {
    let measure_resolution = u64::from(raster.policy.steps_per_measure);
    let measure_count = raster
        .total_steps
        .checked_div(measure_resolution)
        .and_then(|count| count.checked_add(1))
        .ok_or(MuspyRasterMetricErrorV1::BucketCountOverflow)?;
    if measure_count < 2 {
        return Ok(None);
    }

    let mut patterns: BTreeMap<u64, BTreeSet<u64>> = BTreeMap::new();
    for note in &raster.notes {
        let measure = note.start_step / measure_resolution;
        let position = note.start_step % measure_resolution;
        patterns.entry(measure).or_default().insert(position);
    }

    let mut affected_pairs = BTreeSet::new();
    for &measure in patterns.keys() {
        if measure > 0 {
            affected_pairs.insert(measure - 1);
        }
        if measure + 1 < measure_count {
            affected_pairs.insert(measure);
        }
    }

    let empty = BTreeSet::new();
    let mut hamming = 0u128;
    for pair in affected_pairs {
        let left = patterns.get(&pair).unwrap_or(&empty);
        let right = patterns.get(&(pair + 1)).unwrap_or(&empty);
        hamming += left.symmetric_difference(right).count() as u128;
    }
    let denominator = measure_resolution as f64 * (measure_count - 1) as f64;
    Ok(Some(1.0 - hamming as f64 / denominator))
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence_digest::benchmark_time_grid::{
        SymbolicTimeGridPolicyV1, rasterize_score_exact,
    };
    use symthaea_music_theory::{
        Duration, Emphasis, Key, PartId, Pitch, PitchClass, Score, ScoreNote, VoiceRole,
    };

    fn note(onset: Duration, duration: Duration, pitch: u8) -> ScoreNote {
        ScoreNote {
            part: PartId(1),
            pitch: Pitch::from_midi(pitch),
            onset,
            duration,
            velocity: 0.7,
            role: VoiceRole::Melody,
            emphasis: Emphasis::Normal,
            section_intensity: 1.0,
        }
    }

    fn raster(notes: Vec<ScoreNote>, steps_per_quarter: u32) -> ExactSymbolicRasterV1 {
        let mut score = Score::new(Key::major(PitchClass::C), 120.0, 4);
        for note in notes {
            score.push(note);
        }
        let policy = SymbolicTimeGridPolicyV1::muspy_source_exact(4, steps_per_quarter).unwrap();
        rasterize_score_exact(&score, policy).unwrap()
    }

    #[test]
    fn two_pitch_chord_does_not_cross_muspy_default_threshold() {
        let two = raster(
            vec![
                note(Duration::zero(), Duration::quarter(), 60),
                note(Duration::zero(), Duration::quarter(), 64),
            ],
            4,
        );
        let metrics = measure_muspy_piano_raster_metrics(
            &two,
            MUSPY_DEFAULT_POLYPHONY_RATE_THRESHOLD,
        )
        .unwrap();
        assert_eq!(metrics.polyphony, Some(2.0));
        assert_eq!(metrics.polyphony_rate, Some(0.0));

        let three = raster(
            vec![
                note(Duration::zero(), Duration::quarter(), 60),
                note(Duration::zero(), Duration::quarter(), 64),
                note(Duration::zero(), Duration::quarter(), 67),
            ],
            4,
        );
        let metrics = measure_muspy_piano_raster_metrics(
            &three,
            MUSPY_DEFAULT_POLYPHONY_RATE_THRESHOLD,
        )
        .unwrap();
        assert_eq!(metrics.polyphony, Some(3.0));
        assert_eq!(metrics.polyphony_rate, Some(1.0));
    }

    #[test]
    fn duplicate_same_pitch_is_binary_pianoroll_occupancy() {
        let duplicate = raster(
            vec![
                note(Duration::zero(), Duration::quarter(), 60),
                note(Duration::zero(), Duration::quarter(), 60),
            ],
            4,
        );
        let metrics = measure_muspy_piano_raster_metrics(&duplicate, 2).unwrap();
        assert_eq!(metrics.polyphony, Some(1.0));
    }

    #[test]
    fn empty_score_maps_external_nan_boundaries_to_none() {
        let empty = raster(Vec::new(), 4);
        let metrics = measure_muspy_piano_raster_metrics(&empty, 2).unwrap();
        assert_eq!(metrics.polyphony, None);
        assert_eq!(metrics.polyphony_rate, None);
        assert_eq!(metrics.empty_beat_rate, None);
        assert_eq!(metrics.empty_measure_rate, None);
        assert_eq!(metrics.groove_consistency, None);
    }

    #[test]
    fn empty_bucket_metrics_preserve_muspy_inclusive_note_end_quirk() {
        let one_beat = raster(
            vec![note(Duration::zero(), Duration::quarter(), 60)],
            4,
        );
        let metrics = measure_muspy_piano_raster_metrics(&one_beat, 2).unwrap();
        // MusPy computes end_bucket = note.end // resolution and includes that
        // bucket, so an exact beat-boundary note marks both beat buckets used.
        assert_eq!(metrics.empty_beat_rate, Some(0.0));

        let one_measure = raster(
            vec![note(Duration::zero(), Duration::whole(), 60)],
            4,
        );
        let metrics = measure_muspy_piano_raster_metrics(&one_measure, 2).unwrap();
        assert_eq!(metrics.empty_measure_rate, Some(0.0));
    }

    #[test]
    fn groove_uses_neighboring_binary_onset_hamming_distance() {
        let repeated = raster(
            vec![
                note(Duration::zero(), Duration::quarter(), 60),
                note(Duration::whole(), Duration::quarter(), 64),
            ],
            4,
        );
        let metrics = measure_muspy_piano_raster_metrics(&repeated, 2).unwrap();
        assert_eq!(metrics.groove_consistency, Some(1.0));

        let shifted = raster(
            vec![
                note(Duration::zero(), Duration::quarter(), 60),
                note(Duration::new(17, 4), Duration::quarter(), 64),
            ],
            4,
        );
        let metrics = measure_muspy_piano_raster_metrics(&shifted, 2).unwrap();
        assert!((metrics.groove_consistency.unwrap() - 0.875).abs() < 1.0e-12);
    }

    #[test]
    fn raster_length_and_external_identity_are_load_bearing() {
        let mut subject = raster(
            vec![note(Duration::zero(), Duration::quarter(), 60)],
            4,
        );
        subject.total_steps += 1;
        assert!(matches!(
            measure_muspy_piano_raster_metrics(&subject, 2),
            Err(MuspyRasterMetricErrorV1::TotalStepsMismatch { .. })
        ));

        let mut subject = raster(
            vec![note(Duration::zero(), Duration::quarter(), 60)],
            4,
        );
        subject.policy.muspy_metrics_source_blob = "0".repeat(40);
        assert!(matches!(
            measure_muspy_piano_raster_metrics(&subject, 2),
            Err(MuspyRasterMetricErrorV1::WrongSourceBlob { .. })
        ));
    }

    #[test]
    fn metric_artifact_has_stable_canonical_identity() {
        let subject = raster(
            vec![note(Duration::zero(), Duration::quarter(), 60)],
            4,
        );
        let metrics = measure_muspy_piano_raster_metrics(&subject, 2).unwrap();
        assert_eq!(
            metrics.canonical_sha256().unwrap(),
            metrics.canonical_sha256().unwrap()
        );
    }
}
