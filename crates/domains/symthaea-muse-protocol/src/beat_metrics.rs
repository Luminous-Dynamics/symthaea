// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MIREX 2026 beat metric semantics, pinned to the public evaluator and mir_eval.
//!
//! This module scores source-local event-frame arrays. It does not load audio,
//! infer beats, select a dataset split, or authorize a benchmark claim.
//!
//! Reference pins:
//! - MIREX evaluator: ismir-mirex/mirex-evaluation
//!   b9fa0b0b32e2145af31f35830f78fc9d09a4301b
//! - mir_eval 0.8.2 tag commit:
//!   8db0b3812e2032544c1fc00d02d4256cab043f3d
//!
//! The eight public MIREX columns are represented separately. Cemgil's
//! best-of-metrical-level diagnostic is kept as an additional field and is
//! never substituted for the original-level Cemgil column.

use serde::{Deserialize, Serialize};

/// Schema version for the portable score result.
pub const MIREX_BEAT_SCORE_SCHEMA_VERSION: u16 = 1;

/// Exact source revision of the public MIREX evaluation harness.
pub const MIREX_EVALUATOR_SOURCE_COMMIT: &str =
    "b9fa0b0b32e2145af31f35830f78fc9d09a4301b";

/// Reference mir_eval package version used for known-answer parity.
pub const MIR_EVAL_REFERENCE_VERSION: &str = "0.8.2";

/// Exact source revision for the mir_eval 0.8.2 tag.
pub const MIR_EVAL_REFERENCE_COMMIT: &str =
    "8db0b3812e2032544c1fc00d02d4256cab043f3d";

/// mir_eval.beat.evaluate's default initial trim, in seconds.
pub const MIREX_INITIAL_TRIM_SECONDS: u64 = 5;

/// mir_eval F-measure matching tolerance, in milliseconds.
pub const MIREX_BEAT_F_MEASURE_TOLERANCE_MS: u64 = 70;

/// mir_eval Cemgil Gaussian sigma, in milliseconds.
pub const MIREX_CEMGIL_SIGMA_MS: u64 = 40;

/// mir_eval continuity phase threshold, in parts per million.
pub const MIREX_CONTINUITY_PHASE_THRESHOLD_PPM: u32 = 175_000;

/// mir_eval continuity period threshold, in parts per million.
pub const MIREX_CONTINUITY_PERIOD_THRESHOLD_PPM: u32 = 175_000;

/// mir_eval Goto normalized beat-error threshold.
pub const MIREX_GOTO_ERROR_THRESHOLD: f64 = 0.35;

/// mir_eval Goto correct-region absolute-error mean threshold.
pub const MIREX_GOTO_MEAN_THRESHOLD: f64 = 0.2;

/// mir_eval Goto correct-region sample-standard-deviation threshold.
pub const MIREX_GOTO_STD_THRESHOLD: f64 = 0.2;

/// mir_eval P-score window as a fraction of median reference IOI.
pub const MIREX_P_SCORE_WINDOW_RATIO: f64 = 0.2;

/// One MIREX beat-track score vector.
///
/// The first eight fields are the MIREX public table metrics, in public-view
/// order. `cemgil_best_metric_level` is retained as a separate diagnostic:
/// the public MIREX view reports only original-level `cemgil`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirexBeatMetricScores {
    pub f_measure: f64,
    pub cemgil: f64,
    pub goto: f64,
    pub p_score: f64,
    pub cmlc: f64,
    pub cmlt: f64,
    pub amlc: f64,
    pub amlt: f64,
    pub cemgil_best_metric_level: f64,
}

/// Per-track report, including counts before and after mir_eval's five-second trim.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirexBeatTrackReportV1 {
    pub schema_version: u16,
    pub scores: MirexBeatMetricScores,
    pub reference_beats_input: u64,
    pub estimated_beats_input: u64,
    pub reference_beats_scored: u64,
    pub estimated_beats_scored: u64,
    pub sample_rate_hz: u32,
    pub initial_trim_seconds: u64,
}

/// Errors from invalid event arrays or unsupported source clocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MirexBeatMetricError {
    ZeroSampleRate,
    EmptySourceClock,
    SourceClockExceedsJsonSafeInteger,
    SourceDurationExceedsMirEvalLimit,
    ReferenceFrameOutOfBounds,
    EstimatedFrameOutOfBounds,
    ReferenceEventsNotStrictlyIncreasing,
    EstimatedEventsNotStrictlyIncreasing,
}

/// Score one beat track using the pinned MIREX/mir_eval semantics.
///
/// Input event frames must be strictly increasing and source-local. The
/// function applies mir_eval's default five-second initial trim to both
/// sequences before computing all eight public columns.
pub fn score_mirex_beat_track(
    reference_frames: &[u64],
    estimated_frames: &[u64],
    sample_rate_hz: u32,
    total_frames: u64,
) -> Result<MirexBeatTrackReportV1, MirexBeatMetricError> {
    validate_event_inputs(
        reference_frames,
        estimated_frames,
        sample_rate_hz,
        total_frames,
    )?;

    let trim_frame = u64::from(sample_rate_hz) * MIREX_INITIAL_TRIM_SECONDS;
    let reference: Vec<u64> = reference_frames
        .iter()
        .copied()
        .filter(|frame| *frame >= trim_frame)
        .collect();
    let estimated: Vec<u64> = estimated_frames
        .iter()
        .copied()
        .filter(|frame| *frame >= trim_frame)
        .collect();

    let scores = score_trimmed_beat_frames(&reference, &estimated, sample_rate_hz);

    Ok(MirexBeatTrackReportV1 {
        schema_version: MIREX_BEAT_SCORE_SCHEMA_VERSION,
        scores,
        reference_beats_input: reference_frames.len() as u64,
        estimated_beats_input: estimated_frames.len() as u64,
        reference_beats_scored: reference.len() as u64,
        estimated_beats_scored: estimated.len() as u64,
        sample_rate_hz,
        initial_trim_seconds: MIREX_INITIAL_TRIM_SECONDS,
    })
}

/// Score downbeats with the MIREX ±70 ms event F-measure.
///
/// The downbeat task uses its own event list and does not apply the beat
/// track's five-second initial trim.
pub fn score_mirex_downbeat_f_measure(
    reference_frames: &[u64],
    estimated_frames: &[u64],
    sample_rate_hz: u32,
    total_frames: u64,
) -> Result<f64, MirexBeatMetricError> {
    validate_event_inputs(
        reference_frames,
        estimated_frames,
        sample_rate_hz,
        total_frames,
    )?;
    Ok(event_f_measure(
        reference_frames,
        estimated_frames,
        sample_rate_hz,
        MIREX_BEAT_F_MEASURE_TOLERANCE_MS,
    ))
}

/// Explicit track aggregation policy.
///
/// MIREX's public evaluator writes an arithmetic mean across per-track
/// scores. Dataset-specific protocols such as valid-segment duration-weighted
/// JSIR evaluation must select the weighted mode explicitly.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MirexBeatAggregationMode {
    ArithmeticMeanPerTrack,
    DurationWeighted,
}

/// One scored track and its declared aggregation weight.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightedMirexBeatScores {
    pub scores: MirexBeatMetricScores,
    pub weight_frames: u64,
}

/// Aggregated metric vector. Each metric remains separate.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MirexBeatAggregateScores {
    pub mode: MirexBeatAggregationMode,
    pub track_count: u64,
    pub total_weight_frames: u64,
    pub scores: MirexBeatMetricScores,
}

/// Aggregate per-track scores using the explicitly selected protocol policy.
pub fn aggregate_mirex_beat_scores(
    tracks: &[WeightedMirexBeatScores],
    mode: MirexBeatAggregationMode,
) -> Option<MirexBeatAggregateScores> {
    if tracks.is_empty()
        || (mode == MirexBeatAggregationMode::DurationWeighted
            && tracks.iter().any(|track| track.weight_frames == 0))
    {
        return None;
    }

    let total_weight: u128 = match mode {
        MirexBeatAggregationMode::ArithmeticMeanPerTrack => tracks.len() as u128,
        MirexBeatAggregationMode::DurationWeighted => tracks
            .iter()
            .map(|track| u128::from(track.weight_frames))
            .sum(),
    };
    if total_weight == 0 {
        return None;
    }

    let weight = |track: &WeightedMirexBeatScores| -> f64 {
        match mode {
            MirexBeatAggregationMode::ArithmeticMeanPerTrack => 1.0,
            MirexBeatAggregationMode::DurationWeighted => track.weight_frames as f64,
        }
    };
    let weighted = |select: fn(&MirexBeatMetricScores) -> f64| -> f64 {
        let numerator: f64 = tracks
            .iter()
            .map(|track| select(&track.scores) * weight(track))
            .sum();
        let denominator: f64 = tracks.iter().map(weight).sum();
        numerator / denominator
    };

    let scores = MirexBeatMetricScores {
        f_measure: weighted(|scores| scores.f_measure),
        cemgil: weighted(|scores| scores.cemgil),
        goto: weighted(|scores| scores.goto),
        p_score: weighted(|scores| scores.p_score),
        cmlc: weighted(|scores| scores.cmlc),
        cmlt: weighted(|scores| scores.cmlt),
        amlc: weighted(|scores| scores.amlc),
        amlt: weighted(|scores| scores.amlt),
        cemgil_best_metric_level: weighted(|scores| scores.cemgil_best_metric_level),
    };

    Some(MirexBeatAggregateScores {
        mode,
        track_count: tracks.len() as u64,
        total_weight_frames: tracks.iter().map(|track| track.weight_frames).sum(),
        scores,
    })
}

fn validate_event_inputs(
    reference_frames: &[u64],
    estimated_frames: &[u64],
    sample_rate_hz: u32,
    total_frames: u64,
) -> Result<(), MirexBeatMetricError> {
    if sample_rate_hz == 0 {
        return Err(MirexBeatMetricError::ZeroSampleRate);
    }
    if total_frames == 0 {
        return Err(MirexBeatMetricError::EmptySourceClock);
    }
    if total_frames > crate::transition::MAX_SAFE_JSON_INTEGER {
        return Err(MirexBeatMetricError::SourceClockExceedsJsonSafeInteger);
    }
    if u128::from(total_frames)
        > u128::from(sample_rate_hz) * 30_000u128
    {
        return Err(MirexBeatMetricError::SourceDurationExceedsMirEvalLimit);
    }
    if reference_frames.iter().any(|frame| *frame >= total_frames) {
        return Err(MirexBeatMetricError::ReferenceFrameOutOfBounds);
    }
    if estimated_frames.iter().any(|frame| *frame >= total_frames) {
        return Err(MirexBeatMetricError::EstimatedFrameOutOfBounds);
    }
    if reference_frames.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(MirexBeatMetricError::ReferenceEventsNotStrictlyIncreasing);
    }
    if estimated_frames.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(MirexBeatMetricError::EstimatedEventsNotStrictlyIncreasing);
    }
    Ok(())
}

fn score_trimmed_beat_frames(
    reference: &[u64],
    estimated: &[u64],
    sample_rate_hz: u32,
) -> MirexBeatMetricScores {
    if reference.is_empty() || estimated.is_empty() {
        return MirexBeatMetricScores {
            f_measure: 0.0,
            cemgil: 0.0,
            goto: 0.0,
            p_score: 0.0,
            cmlc: 0.0,
            cmlt: 0.0,
            amlc: 0.0,
            amlt: 0.0,
            cemgil_best_metric_level: 0.0,
        };
    }

    let f_measure = event_f_measure(
        reference,
        estimated,
        sample_rate_hz,
        MIREX_BEAT_F_MEASURE_TOLERANCE_MS,
    );
    let (cemgil, cemgil_best_metric_level) =
        cemgil_scores(reference, estimated, sample_rate_hz);
    let reference_seconds = frames_to_seconds(reference, sample_rate_hz);
    let estimated_seconds = frames_to_seconds(estimated, sample_rate_hz);
    let goto = goto_score(&reference_seconds, &estimated_seconds);
    let p_score = p_score(reference, estimated, sample_rate_hz);
    let (cmlc, cmlt, amlc, amlt) =
        continuity_scores(&reference_seconds, &estimated_seconds);

    MirexBeatMetricScores {
        f_measure,
        cemgil,
        goto,
        p_score,
        cmlc,
        cmlt,
        amlc,
        amlt,
        cemgil_best_metric_level,
    }
}

fn event_f_measure(
    reference: &[u64],
    estimated: &[u64],
    sample_rate_hz: u32,
    tolerance_ms: u64,
) -> f64 {
    if reference.is_empty() || estimated.is_empty() {
        return 0.0;
    }
    let tolerance_left = u128::from(sample_rate_hz) * u128::from(tolerance_ms);
    let mut reference_index = 0usize;
    let mut estimated_index = 0usize;
    let mut matching_count = 0usize;

    while reference_index < reference.len() && estimated_index < estimated.len() {
        let reference_frame = reference[reference_index];
        let estimated_frame = estimated[estimated_index];
        let delta = reference_frame.abs_diff(estimated_frame);
        let within = u128::from(delta) * 1_000
            <= tolerance_left;
        if within {
            matching_count += 1;
            reference_index += 1;
            estimated_index += 1;
        } else if estimated_frame < reference_frame {
            estimated_index += 1;
        } else {
            reference_index += 1;
        }
    }

    let precision = matching_count as f64 / estimated.len() as f64;
    let recall = matching_count as f64 / reference.len() as f64;
    if precision + recall == 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

fn frames_to_seconds(frames: &[u64], sample_rate_hz: u32) -> Vec<f64> {
    frames
        .iter()
        .map(|frame| *frame as f64 / f64::from(sample_rate_hz))
        .collect()
}

fn nearest_distance(reference: &[f64], estimate: f64) -> f64 {
    let index = reference.partition_point(|value| *value < estimate);
    match index {
        0 => (reference[0] - estimate).abs(),
        n if n == reference.len() => (reference[n - 1] - estimate).abs(),
        n => {
            let before = (reference[n - 1] - estimate).abs();
            let after = (reference[n] - estimate).abs();
            before.min(after)
        }
    }
}

fn cemgil_scores(
    reference_frames: &[u64],
    estimated_frames: &[u64],
    sample_rate_hz: u32,
) -> (f64, f64) {
    if reference_frames.is_empty() || estimated_frames.is_empty() {
        return (0.0, 0.0);
    }
    let reference = frames_to_seconds(reference_frames, sample_rate_hz);
    let estimated = frames_to_seconds(estimated_frames, sample_rate_hz);
    let variations = reference_beat_variations(&reference);
    let sigma = 0.040_f64;
    let mut scores = Vec::with_capacity(variations.len());

    for variation in variations {
        let denominator = 0.5 * (estimated.len() + variation.len()) as f64;
        if denominator == 0.0 {
            scores.push(0.0);
            continue;
        }
        let total: f64 = variation
            .iter()
            .map(|beat| {
                let error = nearest_distance(&estimated, *beat);
                (-(error * error) / (2.0 * sigma * sigma)).exp()
            })
            .sum();
        scores.push(total / denominator);
    }
    (scores.first().copied().unwrap_or(0.0), scores.into_iter().fold(0.0, f64::max))
}

fn reference_beat_variations(reference: &[f64]) -> Vec<Vec<f64>> {
    let mut doubled = Vec::with_capacity(reference.len().saturating_mul(2).saturating_sub(1));
    for pair in reference.windows(2) {
        doubled.push(pair[0]);
        doubled.push((pair[0] + pair[1]) * 0.5);
    }
    if let Some(last) = reference.last() {
        doubled.push(*last);
    }

    let offbeat = doubled.iter().skip(1).step_by(2).copied().collect();
    let half_even = reference.iter().step_by(2).copied().collect();
    let half_odd = reference.iter().skip(1).step_by(2).copied().collect();
    vec![reference.to_vec(), offbeat, doubled, half_even, half_odd]
}

fn goto_score(reference: &[f64], estimated: &[f64]) -> f64 {
    if reference.is_empty() || estimated.is_empty() {
        return 0.0;
    }
    let mut beat_error = vec![1.0_f64; reference.len()];

    for index in 1..reference.len().saturating_sub(1) {
        let previous_interval = 0.5 * (reference[index] - reference[index - 1]);
        let next_interval = 0.5 * (reference[index + 1] - reference[index]);
        let window_min = reference[index] - previous_interval;
        let window_max = reference[index] + next_interval;
        let first = estimated.partition_point(|value| *value < window_min);
        let after = estimated.partition_point(|value| *value < window_max);
        let count = after.saturating_sub(first);

        if count == 1 {
            let offset = estimated[first] - reference[index];
            beat_error[index] = if offset < 0.0 {
                offset / previous_interval
            } else {
                offset / next_interval
            };
        } else {
            beat_error[index] = 1.0;
        }
    }

    let incorrect: Vec<usize> = beat_error
        .iter()
        .enumerate()
        .filter_map(|(index, error)| {
            (error.abs() > MIREX_GOTO_ERROR_THRESHOLD).then_some(index)
        })
        .collect();

    if incorrect.is_empty() {
        return 0.0;
    }
    let track: Vec<f64>;
    if incorrect.len() < 3 {
        let start = incorrect[0] + 1;
        let end = incorrect[incorrect.len() - 1].saturating_sub(1);
        if start >= end {
            return 0.0;
        }
        track = beat_error[start..end].to_vec();
    } else {
        let mut best_gap = 0usize;
        let mut best_pair = (0usize, 0usize);
        for pair_index in 0..incorrect.len() - 1 {
            let gap = incorrect[pair_index + 1] - incorrect[pair_index];
            if gap > best_gap {
                best_gap = gap;
                best_pair = (incorrect[pair_index], incorrect[pair_index + 1]);
            }
        }
        if best_gap.saturating_sub(1) as f64
            <= 0.25 * reference.len().saturating_sub(2) as f64
        {
            return 0.0;
        }
        track = beat_error[best_pair.0..=best_pair.1].to_vec();
    }

    if track.len() < 2 {
        return 0.0;
    }
    let mean_abs = track.iter().map(|value| value.abs()).sum::<f64>() / track.len() as f64;
    let mean = track.iter().sum::<f64>() / track.len() as f64;
    let variance = track
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (track.len() - 1) as f64;
    let standard_deviation = variance.sqrt();
    if mean_abs < MIREX_GOTO_MEAN_THRESHOLD
        && standard_deviation < MIREX_GOTO_STD_THRESHOLD
    {
        1.0
    } else {
        0.0
    }
}

fn p_score(
    reference_frames: &[u64],
    estimated_frames: &[u64],
    sample_rate_hz: u32,
) -> f64 {
    if reference_frames.len() <= 1 || estimated_frames.len() <= 1 {
        return 0.0;
    }

    let offset = reference_frames[0].min(estimated_frames[0]);
    let reference_bins = quantize_10ms_bins(reference_frames, offset, sample_rate_hz);
    let estimated_bins = quantize_10ms_bins(estimated_frames, offset, sample_rate_hz);
    if reference_bins.len() <= 1 || estimated_bins.is_empty() {
        return 0.0;
    }

    let intervals: Vec<u64> = reference_bins
        .windows(2)
        .map(|pair| pair[1] - pair[0])
        .collect();
    let median = median_f64(&intervals);
    let window = round_ties_even(MIREX_P_SCORE_WINDOW_RATIO * median) as u64;

    let mut total_pairs = 0u128;
    let mut left = 0usize;
    let mut right = 0usize;
    for reference_bin in &reference_bins {
        let low = reference_bin.saturating_sub(window);
        let high = reference_bin.saturating_add(window);
        while left < estimated_bins.len() && estimated_bins[left] < low {
            left += 1;
        }
        if right < left {
            right = left;
        }
        while right < estimated_bins.len() && estimated_bins[right] <= high {
            right += 1;
        }
        total_pairs += (right - left) as u128;
    }

    let denominator = reference_frames.len().max(estimated_frames.len()) as f64;
    total_pairs as f64 / denominator
}

fn quantize_10ms_bins(frames: &[u64], offset: u64, sample_rate_hz: u32) -> Vec<u64> {
    let mut bins = Vec::with_capacity(frames.len());
    for frame in frames {
        let numerator = u128::from(*frame - offset) * 100;
        let denominator = u128::from(sample_rate_hz);
        let bin = numerator.div_ceil(denominator) as u64;
        if bins.last().copied() != Some(bin) {
            bins.push(bin);
        }
    }
    bins
}

fn median_f64(values: &[u64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 1 {
        sorted[mid] as f64
    } else {
        (sorted[mid - 1] as f64 + sorted[mid] as f64) * 0.5
    }
}

fn round_ties_even(value: f64) -> f64 {
    let floor = value.floor();
    let fraction = value - floor;
    if fraction < 0.5 {
        floor
    } else if fraction > 0.5 {
        floor + 1.0
    } else if (floor as i64) % 2 == 0 {
        floor
    } else {
        floor + 1.0
    }
}

fn continuity_scores(
    reference: &[f64],
    estimated: &[f64],
) -> (f64, f64, f64, f64) {
    if reference.len() <= 1 || estimated.len() <= 1 {
        return (0.0, 0.0, 0.0, 0.0);
    }
    let variations = reference_beat_variations(reference);
    let mut continuous = Vec::with_capacity(variations.len());
    let mut total = Vec::with_capacity(variations.len());

    for variation in variations {
        let annotation_count = reference.len().max(estimated.len());
        let mut used = vec![false; variation.len()];
        let mut successes = vec![false; annotation_count];

        for (estimate_index, estimate) in estimated.iter().enumerate() {
            if variation.is_empty() {
                continue;
            }
            let nearest = nearest_index(&variation, *estimate);
            if used[nearest] {
                continue;
            }
            let minimum_difference = (variation[nearest] - *estimate).abs();
            let reference_interval = if estimate_index == 0 || nearest == 0 {
                if nearest + 1 < variation.len() {
                    variation[nearest + 1] - variation[nearest]
                } else {
                    variation[nearest] - variation[nearest - 1]
                }
            } else {
                variation[nearest] - variation[nearest - 1]
            };
            let estimated_interval = if estimate_index + 1 < estimated.len() {
                estimated[estimate_index + 1] - estimated[estimate_index]
            } else {
                estimated[estimate_index] - estimated[estimate_index - 1]
            };

            let (phase, period) = if reference_interval == 0.0 {
                let phase = if minimum_difference == 0.0 { 1.0 } else { f64::INFINITY };
                let period = if estimated_interval == 0.0 { 0.0 } else { f64::INFINITY };
                (phase, period)
            } else {
                (
                    (minimum_difference / reference_interval).abs(),
                    (1.0 - estimated_interval / reference_interval).abs(),
                )
            };

            if phase < 0.175 && period < 0.175 {
                used[nearest] = true;
                successes[estimate_index] = true;
            }
        }

        let mut longest = 0usize;
        let mut current = 0usize;
        for success in &successes {
            if *success {
                current += 1;
                longest = longest.max(current);
            } else {
                current = 0;
            }
        }
        let matched = successes.iter().filter(|success| **success).count();
        continuous.push(longest as f64 / annotation_count as f64);
        total.push(matched as f64 / annotation_count as f64);
    }

    let cmlc = continuous.first().copied().unwrap_or(0.0);
    let cmlt = total.first().copied().unwrap_or(0.0);
    let amlc = continuous.into_iter().fold(0.0, f64::max);
    let amlt = total.into_iter().fold(0.0, f64::max);
    (cmlc, cmlt, amlc, amlt)
}

fn nearest_index(sorted: &[f64], value: f64) -> usize {
    let index = sorted.partition_point(|candidate| *candidate < value);
    match index {
        0 => 0,
        n if n == sorted.len() => n - 1,
        n => {
            let before = (sorted[n - 1] - value).abs();
            let after = (sorted[n] - value).abs();
            if before <= after { n - 1 } else { n }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames_from_seconds(seconds: &[f64], sample_rate_hz: u32) -> Vec<u64> {
        seconds
            .iter()
            .map(|seconds| (*seconds * f64::from(sample_rate_hz)).round() as u64)
            .collect()
    }

    fn score_seconds(reference: &[f64], estimated: &[f64]) -> MirexBeatMetricScores {
        let sample_rate = 1_000;
        let reference_frames = frames_from_seconds(reference, sample_rate);
        let estimated_frames = frames_from_seconds(estimated, sample_rate);
        let total_frames = 30_000 * u64::from(sample_rate);
        score_mirex_beat_track(
            &reference_frames,
            &estimated_frames,
            sample_rate,
            total_frames,
        )
        .unwrap()
        .scores
    }

    fn reference() -> Vec<f64> {
        (0..=12).map(|value| value as f64).collect()
    }

    #[test]
    fn perfect_grid_matches_mireval_082_golden_vector() {
        let scores = score_seconds(&reference(), &reference());
        assert!((scores.f_measure - 1.0).abs() < 1e-12);
        assert!((scores.cemgil - 1.0).abs() < 1e-12);
        assert!((scores.goto - 1.0).abs() < 1e-12);
        assert!((scores.p_score - 1.0).abs() < 1e-12);
        assert!((scores.cmlc - 1.0).abs() < 1e-12);
        assert!((scores.cmlt - 1.0).abs() < 1e-12);
        assert!((scores.amlc - 1.0).abs() < 1e-12);
        assert!((scores.amlt - 1.0).abs() < 1e-12);
        assert!((scores.cemgil_best_metric_level - 1.0).abs() < 1e-12);
    }

    #[test]
    fn f_measure_treats_seventy_ms_as_inclusive_and_seventy_one_as_outside() {
        let reference = reference();
        let shift_70: Vec<f64> = reference.iter().map(|beat| beat + 0.070).collect();
        let shift_71: Vec<f64> = reference.iter().map(|beat| beat + 0.071).collect();
        assert!((score_seconds(&reference, &shift_70).f_measure - 1.0).abs() < 1e-12);
        assert!(score_seconds(&reference, &shift_71).f_measure.abs() < 1e-12);
    }

    #[test]
    fn half_time_preserves_any_metric_level_but_not_correct_metric_level() {
        let scores = score_seconds(
            &reference(),
            &(0..=6).map(|index| (index * 2) as f64).collect::<Vec<_>>(),
        );
        assert!((scores.f_measure - 2.0 / 3.0).abs() < 1e-6);
        assert!((scores.cemgil - 2.0 / 3.0).abs() < 1e-6);
        assert!((scores.cemgil_best_metric_level - 1.0).abs() < 1e-6);
        assert!(scores.cmlc.abs() < 1e-12);
        assert!(scores.cmlt.abs() < 1e-12);
        assert!((scores.amlc - 1.0).abs() < 1e-12);
        assert!((scores.amlt - 1.0).abs() < 1e-12);
    }

    #[test]
    fn double_time_is_not_mistaken_for_correct_metric_level() {
        let estimated: Vec<f64> = (0..=25).map(|index| index as f64 * 0.5).collect();
        let scores = score_seconds(&reference(), &estimated);
        assert!((scores.f_measure - 2.0 / 3.0).abs() < 1e-6);
        assert!((scores.cemgil - 2.0 / 3.0).abs() < 1e-6);
        assert!(scores.cmlc.abs() < 1e-12);
        assert!(scores.cmlt.abs() < 1e-12);
    }

    #[test]
    fn one_missing_beat_reduces_continuity_without_hiding_f1() {
        let reference = reference();
        let estimated: Vec<f64> = reference.iter().copied().filter(|beat| *beat != 8.0).collect();
        let scores = score_seconds(&reference, &estimated);
        assert!((scores.f_measure - 14.0 / 15.0).abs() < 1e-6);
        assert!((scores.cmlc - 0.375).abs() < 1e-6);
        assert!((scores.cmlt - 0.75).abs() < 1e-6);
    }

    #[test]
    fn initial_five_second_trim_is_inclusive_at_exactly_five_seconds() {
        let sample_rate = 1_000;
        let reference = [4_999, 5_000, 5_001, 6_000];
        let estimated = [4_999, 5_000, 5_001, 6_000];
        let report = score_mirex_beat_track(
            &reference,
            &estimated,
            sample_rate,
            10_000,
        )
        .unwrap();
        assert_eq!(report.reference_beats_input, 4);
        assert_eq!(report.estimated_beats_input, 4);
        assert_eq!(report.reference_beats_scored, 3);
        assert_eq!(report.estimated_beats_scored, 3);
    }

    #[test]
    fn downbeat_f_measure_does_not_apply_the_beat_initial_trim() {
        let reference = [1_000, 2_000, 3_000];
        let estimated = [1_000, 2_000, 3_000];
        let score = score_mirex_downbeat_f_measure(&reference, &estimated, 1_000, 10_000).unwrap();
        assert_eq!(score, 1.0);
    }

    #[test]
    fn empty_predictions_produce_finite_zero_scores() {
        let scores = score_seconds(&reference(), &[]);
        assert_eq!(scores.f_measure, 0.0);
        assert_eq!(scores.cemgil, 0.0);
        assert_eq!(scores.goto, 0.0);
        assert_eq!(scores.p_score, 0.0);
        assert_eq!(scores.cmlc, 0.0);
        assert_eq!(scores.cmlt, 0.0);
        assert_eq!(scores.amlc, 0.0);
        assert_eq!(scores.amlt, 0.0);
    }

    #[test]
    fn arithmetic_and_duration_weighted_aggregation_are_explicit() {
        let a = MirexBeatMetricScores {
            f_measure: 1.0,
            cemgil: 1.0,
            goto: 1.0,
            p_score: 1.0,
            cmlc: 1.0,
            cmlt: 1.0,
            amlc: 1.0,
            amlt: 1.0,
            cemgil_best_metric_level: 1.0,
        };
        let b = MirexBeatMetricScores {
            f_measure: 0.0,
            cemgil: 0.0,
            goto: 0.0,
            p_score: 0.0,
            cmlc: 0.0,
            cmlt: 0.0,
            amlc: 0.0,
            amlt: 0.0,
            cemgil_best_metric_level: 0.0,
        };
        let tracks = [
            WeightedMirexBeatScores {
                scores: a,
                weight_frames: 90,
            },
            WeightedMirexBeatScores {
                scores: b,
                weight_frames: 10,
            },
        ];
        let arithmetic = aggregate_mirex_beat_scores(
            &tracks,
            MirexBeatAggregationMode::ArithmeticMeanPerTrack,
        )
        .unwrap();
        let weighted = aggregate_mirex_beat_scores(
            &tracks,
            MirexBeatAggregationMode::DurationWeighted,
        )
        .unwrap();
        assert!((arithmetic.scores.f_measure - 0.5).abs() < 1e-12);
        assert!((weighted.scores.f_measure - 0.9).abs() < 1e-12);
    }

    #[test]
    fn rejects_non_monotonic_events_and_invalid_source_clocks() {
        assert_eq!(
            score_mirex_beat_track(&[6, 5], &[6, 7], 1_000, 20_000).unwrap_err(),
            MirexBeatMetricError::ReferenceEventsNotStrictlyIncreasing
        );
        assert_eq!(
            score_mirex_beat_track(&[6], &[7], 0, 20_000).unwrap_err(),
            MirexBeatMetricError::ZeroSampleRate
        );
    }
}
