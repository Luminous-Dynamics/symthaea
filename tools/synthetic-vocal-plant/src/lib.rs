// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! SYNTH-VOCAL-001: deterministic, synthetic-only acoustic plant.
//!
//! The renderer under test is the legacy formant source-filter implementation
//! exposed by symthaea-vocal-tract. It is deliberately not described as the
//! recovered Series 23 physical articulatory renderer or as a human singer.
//! Each segment is rendered independently; the only cross-segment influence
//! is the declared, bounded overlap used to join adjacent segments.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ops::Range;
use std::path::{Path, PathBuf};
use symthaea_vocal_tract::formant_extraction::{self, ExtractionConfig};
use symthaea_vocal_tract::speech::vocoder;
use symthaea_vocal_tract::types::{FormantFrame, SourceType};

pub const SCHEMA_VERSION: &str = "synthetic-vocal-plant-1.0.0";
pub const RENDERER_BACKEND_ID: &str =
    "symthaea_vocal_tract::speech::vocoder::synthesize";
pub const RENDERER_CLASS: &str = "legacy_formant_source_filter";
pub const FRAME_RATE_HZ: u32 = 200;
pub const DEFAULT_SAMPLE_RATE_HZ: u32 = 24_000;
pub const DEFAULT_CROSSFADE_MS: u32 = 5;
pub const MIN_F0_AUTOCORRELATION: f64 = 0.5;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SyntheticSegmentSpec {
    /// Stable synthetic event identity; not a speaker, singer, or real phone.
    pub segment_id: String,
    /// Number of 5 ms control frames.
    pub frame_count: usize,
    pub f0_hz: f32,
    pub f1_hz: f32,
    pub f2_hz: f32,
    pub f3_hz: f32,
    pub b1_hz: f32,
    pub b2_hz: f32,
    pub b3_hz: f32,
    pub energy: f32,
    pub vibrato_rate_hz: f32,
    pub vibrato_depth_cents: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EditPatch {
    NoOp,
    PitchShiftCents { delta_cents: f32 },
    FormantScale { scale: f32 },
    EnergyScale { scale: f32 },
    UnsupportedControl { control_id: String },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct SampleRange {
    pub start: usize,
    pub end: usize,
}

impl SampleRange {
    pub fn as_range(self) -> Range<usize> {
        self.start..self.end
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct LocalityReport {
    pub allowed_dependency_cone: SampleRange,
    pub compared_protected_samples: usize,
    pub changed_protected_samples: usize,
    pub protected_exact_sample_fraction: f64,
    pub target_rms_delta: f64,
    pub target_max_abs_delta: f64,
    pub passes_exact_locality: bool,
}

#[derive(Debug, Clone)]
pub struct RenderedPhrase {
    pub samples: Vec<f32>,
    pub segment_ranges: Vec<(String, SampleRange)>,
    pub seam_guard_samples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioRecord {
    pub case_id: String,
    pub target_segment_id: Option<String>,
    pub patch: Option<EditPatch>,
    pub expected_response: String,
    pub audio_file: String,
    pub pcm_sha256: String,
    pub sample_count: usize,
    pub target_f0_median_hz: Option<f32>,
    pub baseline_target_f0_median_hz: Option<f32>,
    pub target_formant_observation: Option<FormantObservationSummary>,
    pub baseline_target_formant_observation: Option<FormantObservationSummary>,
    pub locality: Option<LocalityReport>,
}

/// PCM-only formant observations produced by the existing LPC analysis path.
/// These are analyzer estimates, not calibrated phonetic ground truth.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FormantObservationSummary {
    pub observer_id: String,
    pub analysis_sample_rate_hz: u32,
    pub frame_rate_hz: f32,
    pub window_ms: f32,
    pub lpc_order: usize,
    pub preemphasis: f32,
    pub f0_min_hz: f32,
    pub f0_max_hz: f32,
    pub voicing_threshold: f32,
    pub analysis_frame_count: usize,
    pub voiced_frame_count: usize,
    /// Frames whose LPC output equals the extractor's documented fallback
    /// frequency/bandwidth pair for the corresponding formant.
    pub f1_fallback_sentinel_frames: usize,
    pub f2_fallback_sentinel_frames: usize,
    pub f3_fallback_sentinel_frames: usize,
    /// Fully ordered frames with no fallback sentinel detected.
    pub qualified_formant_frame_count: usize,
    pub median_f1_hz: Option<f32>,
    pub median_f2_hz: Option<f32>,
    pub median_f3_hz: Option<f32>,
    pub mad_f1_hz: Option<f32>,
    pub mad_f2_hz: Option<f32>,
    pub mad_f3_hz: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RejectedControlRecord {
    pub case_id: String,
    pub target_segment_id: String,
    pub patch: EditPatch,
    pub disposition: String,
    pub rejection_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuiteManifest {
    pub schema_version: String,
    pub content_class: String,
    pub human_recordings_ingested: bool,
    pub external_corpus_assets_used: bool,
    pub voice_identity_targeting: bool,
    pub renderer_backend_id: String,
    pub renderer_class: String,
    pub renderer_claim_boundary: String,
    pub tool_source_revision: String,
    pub sample_rate_hz: u32,
    pub frame_rate_hz: u32,
    pub crossfade_ms: u32,
    pub parameter_manifest_sha256: String,
    /// Full source parameters are retained so the digest is independently auditable.
    pub segment_parameters: Vec<SyntheticSegmentSpec>,
    pub baseline_pcm_sha256: String,
    pub scenarios: Vec<ScenarioRecord>,
    pub unsupported_control_rejections: Vec<RejectedControlRecord>,
}

fn validate_segment(spec: &SyntheticSegmentSpec, sample_rate_hz: u32) -> Result<(), String> {
    if spec.segment_id.trim().is_empty() {
        return Err("segment_id must be non-empty".into());
    }
    if !(8..=400).contains(&spec.frame_count) {
        return Err(format!(
            "segment {} frame_count must be in 8..=400",
            spec.segment_id
        ));
    }
    let finite = [
        spec.f0_hz,
        spec.f1_hz,
        spec.f2_hz,
        spec.f3_hz,
        spec.b1_hz,
        spec.b2_hz,
        spec.b3_hz,
        spec.energy,
        spec.vibrato_rate_hz,
        spec.vibrato_depth_cents,
    ];
    if finite.iter().any(|value| !value.is_finite()) {
        return Err(format!(
            "segment {} contains a non-finite parameter",
            spec.segment_id
        ));
    }
    if !(60.0..=700.0).contains(&spec.f0_hz) {
        return Err(format!("segment {} F0 is outside 60..=700 Hz", spec.segment_id));
    }
    let nyquist = sample_rate_hz as f32 / 2.0;
    if !(100.0..nyquist).contains(&spec.f1_hz)
        || spec.f1_hz >= spec.f2_hz
        || spec.f2_hz >= spec.f3_hz
        || spec.f3_hz >= nyquist
    {
        return Err(format!(
            "segment {} requires 100 Hz <= F1 < F2 < F3 < Nyquist",
            spec.segment_id
        ));
    }
    if [spec.b1_hz, spec.b2_hz, spec.b3_hz]
        .iter()
        .any(|bandwidth| !(20.0..=1_000.0).contains(bandwidth))
    {
        return Err(format!(
            "segment {} bandwidths must be in 20..=1000 Hz",
            spec.segment_id
        ));
    }
    if !(0.0..=1.0).contains(&spec.energy)
        || !(0.0..=12.0).contains(&spec.vibrato_rate_hz)
        || !(0.0..=100.0).contains(&spec.vibrato_depth_cents)
    {
        return Err(format!(
            "segment {} energy/vibrato parameter is outside its allowed range",
            spec.segment_id
        ));
    }
    Ok(())
}

fn validate_phrase(
    segments: &[SyntheticSegmentSpec],
    sample_rate_hz: u32,
) -> Result<(), String> {
    if sample_rate_hz < 8_000 || sample_rate_hz > 96_000 {
        return Err("sample rate must be in 8000..=96000 Hz".into());
    }
    if segments.is_empty() {
        return Err("phrase must contain at least one segment".into());
    }
    let mut ids = std::collections::BTreeSet::new();
    for segment in segments {
        validate_segment(segment, sample_rate_hz)?;
        if !ids.insert(segment.segment_id.as_str()) {
            return Err(format!("duplicate segment_id: {}", segment.segment_id));
        }
    }
    Ok(())
}

/// Apply a single bounded edit to one stable synthetic segment.
pub fn apply_edit(
    segments: &[SyntheticSegmentSpec],
    target_segment_id: &str,
    patch: &EditPatch,
    sample_rate_hz: u32,
) -> Result<Vec<SyntheticSegmentSpec>, String> {
    validate_phrase(segments, sample_rate_hz)?;
    let mut updated = segments.to_vec();
    let target = updated
        .iter_mut()
        .find(|segment| segment.segment_id == target_segment_id)
        .ok_or_else(|| format!("unknown target segment: {target_segment_id}"))?;

    match patch {
        EditPatch::NoOp => {}
        EditPatch::PitchShiftCents { delta_cents } => {
            if !delta_cents.is_finite() || delta_cents.abs() > 1_200.0 {
                return Err("pitch shift must be finite and within +/-1200 cents".into());
            }
            target.f0_hz *= 2.0_f32.powf(*delta_cents / 1_200.0);
        }
        EditPatch::FormantScale { scale } => {
            if !scale.is_finite() || !(0.8..=1.2).contains(scale) {
                return Err("formant scale must be finite and within 0.8..=1.2".into());
            }
            target.f1_hz *= *scale;
            target.f2_hz *= *scale;
            target.f3_hz *= *scale;
        }
        EditPatch::EnergyScale { scale } => {
            if !scale.is_finite() || !(0.0..=2.0).contains(scale) {
                return Err("energy scale must be finite and within 0.0..=2.0".into());
            }
            target.energy = (target.energy * *scale).clamp(0.0, 1.0);
        }
        EditPatch::UnsupportedControl { control_id } => {
            return Err(format!("unsupported renderer control: {control_id}"));
        }
    }

    validate_phrase(&updated, sample_rate_hz)?;
    Ok(updated)
}

fn render_segment(
    spec: &SyntheticSegmentSpec,
    sample_rate_hz: u32,
    absolute_start_frame: usize,
) -> Vec<f32> {
    let frames: Vec<FormantFrame> = (0..spec.frame_count)
        .map(|index| {
            let absolute_time = (absolute_start_frame + index) as f32 / FRAME_RATE_HZ as f32;
            let vibrato_phase =
                std::f32::consts::TAU * spec.vibrato_rate_hz * absolute_time;
            let vibrato_cents = spec.vibrato_depth_cents * vibrato_phase.sin();
            let f0 = spec.f0_hz * 2.0_f32.powf(vibrato_cents / 1_200.0);
            FormantFrame {
                f1: spec.f1_hz,
                f2: spec.f2_hz,
                f3: spec.f3_hz,
                b1: spec.b1_hz,
                b2: spec.b2_hz,
                b3: spec.b3_hz,
                f0,
                energy: spec.energy,
                voicing: 1.0,
                time: absolute_time,
                source_type: SourceType::Vowel,
                nasal_zero_freq: 0.0,
                nasal_zero_bw: 0.0,
            }
        })
        .collect();

    vocoder::synthesize(&frames, sample_rate_hz)
}

/// Render each synthetic event independently and join with an explicit
/// overlap. Independent block state makes the allowed locality cone auditable.
pub fn render_phrase(
    segments: &[SyntheticSegmentSpec],
    sample_rate_hz: u32,
    crossfade_ms: u32,
) -> Result<RenderedPhrase, String> {
    validate_phrase(segments, sample_rate_hz)?;
    if crossfade_ms > 20 {
        return Err("crossfade must be <= 20 ms".into());
    }
    let fade_samples = (sample_rate_hz as usize * crossfade_ms as usize) / 1_000;
    let mut output = Vec::new();
    let mut ranges = Vec::with_capacity(segments.len());
    let mut absolute_frame = 0usize;

    for segment in segments {
        let block = render_segment(segment, sample_rate_hz, absolute_frame);
        if block.is_empty() {
            return Err(format!("renderer returned an empty block for {}", segment.segment_id));
        }
        if !output.is_empty() && fade_samples >= block.len() {
            return Err("crossfade must be shorter than every rendered segment".into());
        }

        let start = if output.is_empty() {
            output.extend_from_slice(&block);
            0
        } else {
            let overlap = fade_samples.min(output.len());
            let start = output.len() - overlap;
            for index in 0..overlap {
                let alpha = (index + 1) as f32 / (overlap + 1) as f32;
                output[start + index] =
                    output[start + index] * (1.0 - alpha) + block[index] * alpha;
            }
            output.extend_from_slice(&block[overlap..]);
            start
        };
        let end = start + block.len();
        ranges.push((
            segment.segment_id.clone(),
            SampleRange { start, end },
        ));
        absolute_frame += segment.frame_count;
    }

    if output.iter().any(|sample| !sample.is_finite()) {
        return Err("renderer produced non-finite PCM".into());
    }
    Ok(RenderedPhrase {
        samples: output,
        segment_ranges: ranges,
        seam_guard_samples: fade_samples,
    })
}

fn find_range(
    ranges: &[(String, SampleRange)],
    segment_id: &str,
) -> Result<SampleRange, String> {
    ranges
        .iter()
        .find(|(id, _)| id == segment_id)
        .map(|(_, range)| *range)
        .ok_or_else(|| format!("rendered segment range not found: {segment_id}"))
}

/// Compare baseline and candidate PCM. The allowed range must include the
/// target block and its declared seam/dependency guard.
pub fn measure_locality(
    baseline: &[f32],
    candidate: &[f32],
    allowed_dependency_cone: SampleRange,
) -> Result<LocalityReport, String> {
    if baseline.len() != candidate.len() {
        return Err("baseline and candidate sample counts differ".into());
    }
    if allowed_dependency_cone.start >= allowed_dependency_cone.end
        || allowed_dependency_cone.end > baseline.len()
    {
        return Err("allowed dependency cone is empty or outside the PCM range".into());
    }

    let mut compared = 0usize;
    let mut changed = 0usize;
    for (index, (before, after)) in baseline.iter().zip(candidate.iter()).enumerate() {
        if index < allowed_dependency_cone.start || index >= allowed_dependency_cone.end {
            compared += 1;
            if before.to_bits() != after.to_bits() {
                changed += 1;
            }
        }
    }

    let mut square_sum = 0.0f64;
    let mut max_abs = 0.0f64;
    let target_count = allowed_dependency_cone.end - allowed_dependency_cone.start;
    for (before, after) in baseline
        .iter()
        .zip(candidate.iter())
        .skip(allowed_dependency_cone.start)
        .take(target_count)
    {
        let delta = (*after - *before).abs() as f64;
        square_sum += delta * delta;
        max_abs = max_abs.max(delta);
    }
    let exact_fraction = if compared == 0 {
        1.0
    } else {
        (compared - changed) as f64 / compared as f64
    };
    let target_rms_delta = if target_count == 0 {
        0.0
    } else {
        (square_sum / target_count as f64).sqrt()
    };

    Ok(LocalityReport {
        allowed_dependency_cone,
        compared_protected_samples: compared,
        changed_protected_samples: changed,
        protected_exact_sample_fraction: exact_fraction,
        target_rms_delta,
        target_max_abs_delta: max_abs,
        passes_exact_locality: changed == 0,
    })
}

/// Estimate F0 from PCM using normalized autocorrelation and parabolic
/// interpolation. This intentionally does not inspect the renderer's F0
/// control frames, so a control value cannot certify itself.
pub fn estimate_f0_hz(
    window: &[f32],
    sample_rate_hz: u32,
    min_f0_hz: f32,
    max_f0_hz: f32,
) -> Option<f32> {
    if sample_rate_hz == 0
        || window.len() < 64
        || !min_f0_hz.is_finite()
        || !max_f0_hz.is_finite()
        || min_f0_hz <= 0.0
        || max_f0_hz <= min_f0_hz
    {
        return None;
    }
    if window.iter().any(|sample| !sample.is_finite()) {
        return None;
    }

    let mean = window.iter().map(|value| *value as f64).sum::<f64>() / window.len() as f64;
    let centered: Vec<f64> = window.iter().map(|value| *value as f64 - mean).collect();
    let min_lag = ((sample_rate_hz as f32 / max_f0_hz).floor() as usize).max(1);
    let max_lag = ((sample_rate_hz as f32 / min_f0_hz).ceil() as usize)
        .min(centered.len().saturating_sub(2));
    if min_lag >= max_lag {
        return None;
    }

    let correlation = |lag: usize| -> f64 {
        let count = centered.len() - lag;
        let mut cross = 0.0;
        let mut left_energy = 0.0;
        let mut right_energy = 0.0;
        for index in 0..count {
            let left = centered[index];
            let right = centered[index + lag];
            cross += left * right;
            left_energy += left * left;
            right_energy += right * right;
        }
        let denominator = (left_energy * right_energy).sqrt();
        if denominator <= 1e-18 {
            0.0
        } else {
            cross / denominator
        }
    };

    let scores: Vec<f64> = (min_lag..=max_lag).map(correlation).collect();
    let first_peak = (1..scores.len().saturating_sub(1)).find(|index| {
        scores[*index] >= scores[*index - 1]
            && scores[*index] >= scores[*index + 1]
            && scores[*index] >= MIN_F0_AUTOCORRELATION
    });
    let best_index = first_peak.unwrap_or_else(|| {
        scores
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap_or(0)
    });
    let best_lag = min_lag + best_index;
    let best_score = scores.get(best_index).copied().unwrap_or(f64::NEG_INFINITY);
    if best_lag == 0 || best_score < MIN_F0_AUTOCORRELATION {
        return None;
    }

    let refined_lag = if best_lag > min_lag && best_lag < max_lag {
        let left = correlation(best_lag - 1);
        let center = correlation(best_lag);
        let right = correlation(best_lag + 1);
        let denominator = left - 2.0 * center + right;
        if denominator.abs() > 1e-12 {
            let offset = (0.5 * (left - right) / denominator).clamp(-0.5, 0.5);
            best_lag as f64 + offset
        } else {
            best_lag as f64
        }
    } else {
        best_lag as f64
    };
    let frequency = sample_rate_hz as f64 / refined_lag;
    (frequency.is_finite() && frequency >= min_f0_hz as f64 && frequency <= max_f0_hz as f64)
        .then_some(frequency as f32)
}

/// Median of independently estimated F0 values in 40 ms windows / 5 ms hops.
pub fn measure_segment_f0(
    rendered: &RenderedPhrase,
    segment_id: &str,
    sample_rate_hz: u32,
) -> Result<Option<f32>, String> {
    let range = find_range(&rendered.segment_ranges, segment_id)?;
    let guard = rendered.seam_guard_samples;
    let start = range.start.saturating_add(guard);
    let end = range.end.saturating_sub(guard);
    if end <= start {
        return Ok(None);
    }
    let samples = &rendered.samples[start..end];
    let window_len = ((sample_rate_hz as usize) * 40 / 1_000).max(64);
    let hop = ((sample_rate_hz as usize) * 5 / 1_000).max(1);
    if samples.len() < window_len {
        return Ok(estimate_f0_hz(samples, sample_rate_hz, 60.0, 700.0));
    }

    let mut estimates = Vec::new();
    for offset in (0..=samples.len() - window_len).step_by(hop) {
        if let Some(f0) =
            estimate_f0_hz(&samples[offset..offset + window_len], sample_rate_hz, 60.0, 700.0)
        {
            estimates.push(f0);
        }
    }
    estimates.sort_by(f32::total_cmp);
    let median = match estimates.len() {
        0 => None,
        n if n % 2 == 1 => Some(estimates[n / 2]),
        n => Some((estimates[n / 2 - 1] + estimates[n / 2]) * 0.5),
    };
    Ok(median)
}

fn median_f32(mut values: Vec<f32>) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 0 {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    })
}

fn median_and_mad(values: Vec<f32>) -> (Option<f32>, Option<f32>) {
    let center = median_f32(values.clone());
    let deviations = match center {
        Some(median) => values
            .into_iter()
            .map(|value| (value - median).abs())
            .collect(),
        None => Vec::new(),
    };
    (center, median_f32(deviations))
}

/// Summarize F1-F3 estimates from PCM only. This invokes the LPC analysis
/// path independently of the renderer's FormantFrame inputs; it does not
/// receive or inspect the synthetic target formant parameters.
pub fn measure_segment_formants(
    rendered: &RenderedPhrase,
    segment_id: &str,
    sample_rate_hz: u32,
) -> Result<FormantObservationSummary, String> {
    let range = find_range(&rendered.segment_ranges, segment_id)?;
    let guard = rendered.seam_guard_samples;
    let start = range.start.saturating_add(guard);
    let end = range.end.saturating_sub(guard);
    let samples = if end > start {
        &rendered.samples[start..end]
    } else {
        &[]
    };

    let mut config = ExtractionConfig::default();
    // Align the observer's voiced pitch gate to the synthetic plant's full
    // supported range rather than silently dropping the 400–700 Hz strata.
    config.f0_max = 700.0;
    let frames = formant_extraction::extract_formant_frames(samples, sample_rate_hz, &config);
    let voiced: Vec<&FormantFrame> = frames
        .iter()
        .filter(|frame| frame.source_type == SourceType::Vowel && frame.energy > 0.02)
        .collect();
    // The existing extractor substitutes these frequency/bandwidth defaults
    // when a spectral peak is absent. Count those sentinels and do not treat a
    // fallback triplet as an observed three-formant result.
    let f1_fallback_sentinel_frames = voiced
        .iter()
        .filter(|frame| frame.f1 == 500.0 && frame.b1 == 90.0)
        .count();
    let f2_fallback_sentinel_frames = voiced
        .iter()
        .filter(|frame| frame.f2 == 1_500.0 && frame.b2 == 110.0)
        .count();
    let f3_fallback_sentinel_frames = voiced
        .iter()
        .filter(|frame| frame.f3 == 2_500.0 && frame.b3 == 150.0)
        .count();
    let max_formant_hz = config.analysis_sr as f32 * 0.45;
    let ordered: Vec<&FormantFrame> = voiced
        .iter()
        .copied()
        .filter(|frame| {
            let has_fallback_sentinel = (frame.f1 == 500.0 && frame.b1 == 90.0)
                || (frame.f2 == 1_500.0 && frame.b2 == 110.0)
                || (frame.f3 == 2_500.0 && frame.b3 == 150.0);
            !has_fallback_sentinel
                && [frame.f1, frame.f2, frame.f3]
                    .iter()
                    .all(|value| value.is_finite() && *value >= 150.0 && *value < max_formant_hz)
                && frame.f1 < frame.f2
                && frame.f2 < frame.f3
        })
        .collect();

    let (median_f1_hz, mad_f1_hz) =
        median_and_mad(ordered.iter().map(|frame| frame.f1).collect());
    let (median_f2_hz, mad_f2_hz) =
        median_and_mad(ordered.iter().map(|frame| frame.f2).collect());
    let (median_f3_hz, mad_f3_hz) =
        median_and_mad(ordered.iter().map(|frame| frame.f3).collect());

    Ok(FormantObservationSummary {
        observer_id: "symthaea_vocal_tract::formant_extraction::extract_formant_frames".into(),
        analysis_sample_rate_hz: config.analysis_sr,
        frame_rate_hz: config.frame_rate,
        window_ms: config.window_secs * 1_000.0,
        lpc_order: config.lpc_order,
        preemphasis: config.preemphasis,
        f0_min_hz: config.f0_min,
        f0_max_hz: config.f0_max,
        voicing_threshold: config.voicing_threshold,
        analysis_frame_count: frames.len(),
        voiced_frame_count: voiced.len(),
        f1_fallback_sentinel_frames,
        f2_fallback_sentinel_frames,
        f3_fallback_sentinel_frames,
        qualified_formant_frame_count: ordered.len(),
        median_f1_hz: median_f1_hz,
        median_f2_hz: median_f2_hz,
        median_f3_hz: median_f3_hz,
        mad_f1_hz,
        mad_f2_hz,
        mad_f3_hz,
    })
}
        median_f1_hz,
        median_f2_hz,
        median_f3_hz,
        mad_f1_hz,
        mad_f2_hz,
        mad_f3_hz,
    })
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn sha256_pcm(samples: &[f32]) -> String {
    let mut hasher = Sha256::new();
    for sample in samples {
        hasher.update(sample.to_le_bytes());
    }
    let digest = hasher.finalize();
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub fn write_float_wav(path: &Path, samples: &[f32], sample_rate_hz: u32) -> Result<(), String> {
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err("refusing to write non-finite PCM".into());
    }
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: sample_rate_hz,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|error| format!("failed to create WAV: {error}"))?;
    for sample in samples {
        writer
            .write_sample(*sample)
            .map_err(|error| format!("failed to write WAV sample: {error}"))?;
    }
    writer
        .finalize()
        .map_err(|error| format!("failed to finalize WAV: {error}"))?;
    Ok(())
}

fn default_segments() -> Vec<SyntheticSegmentSpec> {
    let parameters = [
        ("proxy-vowel-a", 220.00, 420.0, 1_450.0, 2_700.0, 0.62),
        ("proxy-vowel-b", 246.94, 520.0, 1_250.0, 2_500.0, 0.66),
        ("proxy-vowel-c", 261.63, 650.0, 1_100.0, 2_450.0, 0.70),
        ("proxy-vowel-d", 293.66, 400.0, 1_900.0, 3_000.0, 0.64),
        ("proxy-vowel-e", 329.63, 500.0, 1_550.0, 2_800.0, 0.68),
    ];
    parameters
        .into_iter()
        .map(
            |(id, f0, f1, f2, f3, energy)| SyntheticSegmentSpec {
                segment_id: id.into(),
                frame_count: 64,
                f0_hz: f0,
                f1_hz: f1,
                f2_hz: f2,
                f3_hz: f3,
                b1_hz: 70.0,
                b2_hz: 95.0,
                b3_hz: 140.0,
                energy,
                vibrato_rate_hz: 5.5,
                vibrato_depth_cents: 12.0,
            },
        )
        .collect()
}

fn range_by_id(
    rendered: &RenderedPhrase,
    segment_id: &str,
) -> Result<SampleRange, String> {
    find_range(&rendered.segment_ranges, segment_id)
}

// Keep the provenance-bearing fields explicit at this serialization boundary.
#[allow(clippy::too_many_arguments)]
fn save_case(
    output_dir: &Path,
    case_id: &str,
    rendered: &RenderedPhrase,
    target_segment_id: Option<&str>,
    patch: Option<EditPatch>,
    expected_response: &str,
    baseline: &RenderedPhrase,
    baseline_target_f0: Option<f32>,
    allowed_range: Option<SampleRange>,
    sample_rate_hz: u32,
) -> Result<ScenarioRecord, String> {
    let file_name = format!("{case_id}.wav");
    write_float_wav(&output_dir.join(&file_name), &rendered.samples, sample_rate_hz)?;
    let target_f0 = match target_segment_id {
        Some(segment_id) => measure_segment_f0(rendered, segment_id, sample_rate_hz)?,
        None => None,
    };
    let target_formants = match target_segment_id {
        Some(segment_id) => Some(measure_segment_formants(rendered, segment_id, sample_rate_hz)?),
        None => None,
    };
    let baseline_target_formants = match target_segment_id {
        Some(segment_id) => {
            Some(measure_segment_formants(baseline, segment_id, sample_rate_hz)?)
        }
        None => None,
    };
    let locality = match allowed_range {
        Some(range) => Some(measure_locality(
            &baseline.samples,
            &rendered.samples,
            range,
        )?),
        None => None,
    };
    Ok(ScenarioRecord {
        case_id: case_id.into(),
        target_segment_id: target_segment_id.map(str::to_owned),
        patch,
        expected_response: expected_response.into(),
        audio_file: file_name,
        pcm_sha256: sha256_pcm(&rendered.samples),
        sample_count: rendered.samples.len(),
        target_f0_median_hz: target_f0,
        baseline_target_f0_median_hz: baseline_target_f0,
        target_formant_observation: target_formants,
        baseline_target_formant_observation: baseline_target_formants,
        locality,
    })
}

/// Generate a permission-free fixture suite. Output paths and all input
/// parameters are deterministic; no external assets or recordings are read.
pub fn generate_suite(output_dir: &Path) -> Result<SuiteManifest, String> {
    std::fs::create_dir_all(output_dir)
        .map_err(|error| format!("failed to create output directory: {error}"))?;
    let sample_rate_hz = DEFAULT_SAMPLE_RATE_HZ;
    let crossfade_ms = DEFAULT_CROSSFADE_MS;
    let segments = default_segments();
    let baseline = render_phrase(&segments, sample_rate_hz, crossfade_ms)?;
    let target_id = "proxy-vowel-c";
    let target_range = range_by_id(&baseline, target_id)?;
    let baseline_target_f0 = measure_segment_f0(&baseline, target_id, sample_rate_hz)?;

    let mut scenarios = Vec::new();
    scenarios.push(save_case(
        output_dir,
        "baseline",
        &baseline,
        None,
        None,
        "reference",
        &baseline,
        baseline_target_f0,
        None,
        sample_rate_hz,
    )?);

    let no_op_specs = apply_edit(
        &segments,
        target_id,
        &EditPatch::NoOp,
        sample_rate_hz,
    )?;
    let no_op = render_phrase(&no_op_specs, sample_rate_hz, crossfade_ms)?;
    scenarios.push(save_case(
        output_dir,
        "no_op",
        &no_op,
        Some(target_id),
        Some(EditPatch::NoOp),
        "byte_identical_to_baseline",
        &baseline,
        baseline_target_f0,
        Some(target_range),
        sample_rate_hz,
    )?);

    let pitch_up_patch = EditPatch::PitchShiftCents { delta_cents: 100.0 };
    let pitch_up_specs = apply_edit(&segments, target_id, &pitch_up_patch, sample_rate_hz)?;
    let pitch_up = render_phrase(&pitch_up_specs, sample_rate_hz, crossfade_ms)?;
    scenarios.push(save_case(
        output_dir,
        "pitch_up_100c",
        &pitch_up,
        Some(target_id),
        Some(pitch_up_patch),
        "independent_target_f0_increases; protected_segments_exact",
        &baseline,
        baseline_target_f0,
        Some(target_range),
        sample_rate_hz,
    )?);

    let wrong_direction_patch = EditPatch::PitchShiftCents { delta_cents: -100.0 };
    let wrong_direction_specs =
        apply_edit(&segments, target_id, &wrong_direction_patch, sample_rate_hz)?;
    let wrong_direction = render_phrase(&wrong_direction_specs, sample_rate_hz, crossfade_ms)?;
    scenarios.push(save_case(
        output_dir,
        "wrong_direction_pitch_down_100c",
        &wrong_direction,
        Some(target_id),
        Some(wrong_direction_patch),
        "negative_control; target_f0_moves_opposite_to_requested_pitch_up",
        &baseline,
        baseline_target_f0,
        Some(target_range),
        sample_rate_hz,
    )?);

    let formant_patch = EditPatch::FormantScale { scale: 1.10 };
    let formant_specs = apply_edit(&segments, target_id, &formant_patch, sample_rate_hz)?;
    let formant_only = render_phrase(&formant_specs, sample_rate_hz, crossfade_ms)?;
    scenarios.push(save_case(
        output_dir,
        "formant_scale_110pct",
        &formant_only,
        Some(target_id),
        Some(formant_patch),
        "formant_parameters_change; commanded_f0_is_preserved",
        &baseline,
        baseline_target_f0,
        Some(target_range),
        sample_rate_hz,
    )?);

    let first_collateral_edit =
        apply_edit(&segments, target_id, &pitch_up_patch, sample_rate_hz)?;
    let collateral_specs = apply_edit(
        &first_collateral_edit,
        "proxy-vowel-e",
        &pitch_up_patch,
        sample_rate_hz,
    )?;
    let collateral = render_phrase(&collateral_specs, sample_rate_hz, crossfade_ms)?;
    scenarios.push(save_case(
        output_dir,
        "collateral_corruption",
        &collateral,
        Some(target_id),
        Some(pitch_up_patch),
        "negative_control; second_protected_segment_is_modified_and_must_fail_locality",
        &baseline,
        baseline_target_f0,
        Some(target_range),
        sample_rate_hz,
    )?);

    let parameter_bytes = serde_json::to_vec(&segments)
        .map_err(|error| format!("failed to serialize parameters: {error}"))?;
    let unsupported_patch = EditPatch::UnsupportedControl {
        control_id: "vibrato_phase_offset".into(),
    };
    let unsupported_reason = apply_edit(
        &segments,
        target_id,
        &unsupported_patch,
        sample_rate_hz,
    )
    .expect_err("unsupported control must be rejected before rendering");
    let unsupported_control_rejections = vec![RejectedControlRecord {
        case_id: "unsupported_control".into(),
        target_segment_id: target_id.into(),
        patch: unsupported_patch,
        disposition: "REJECTED_UNSUPPORTED".into(),
        rejection_reason: unsupported_reason,
    }];

    let manifest = SuiteManifest {
        schema_version: SCHEMA_VERSION.into(),
        content_class: "synthetic_only".into(),
        human_recordings_ingested: false,
        external_corpus_assets_used: false,
        voice_identity_targeting: false,
        renderer_backend_id: RENDERER_BACKEND_ID.into(),
        renderer_class: RENDERER_CLASS.into(),
        renderer_claim_boundary: "legacy formant source-filter test plant; not Series 23 physical rendering and not human-naturalness evidence".into(),
        tool_source_revision: std::env::var("SYNTH_VOCAL_PLANT_SOURCE_REVISION")
            .unwrap_or_else(|_| "unknown-at-build".into()),
        sample_rate_hz,
        frame_rate_hz: FRAME_RATE_HZ,
        crossfade_ms,
        parameter_manifest_sha256: sha256_bytes(&parameter_bytes),
        segment_parameters: segments.clone(),
        baseline_pcm_sha256: sha256_pcm(&baseline.samples),
        scenarios,
        unsupported_control_rejections,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("failed to serialize suite manifest: {error}"))?;
    std::fs::write(output_dir.join("manifest.json"), &manifest_bytes)
        .map_err(|error| format!("failed to write manifest: {error}"))?;
    std::fs::write(
        output_dir.join("manifest.sha256"),
        format!("{}  manifest.json\n", sha256_bytes(&manifest_bytes)),
    )
    .map_err(|error| format!("failed to write manifest digest: {error}"))?;
    Ok(manifest)
}

pub fn default_output_dir() -> PathBuf {
    PathBuf::from("synthetic-vocal-plant-output")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> Vec<SyntheticSegmentSpec> {
        default_segments()
    }

    fn render(segments: &[SyntheticSegmentSpec]) -> RenderedPhrase {
        render_phrase(
            segments,
            DEFAULT_SAMPLE_RATE_HZ,
            DEFAULT_CROSSFADE_MS,
        )
        .expect("synthetic phrase should render")
    }

    #[test]
    fn renderer_is_deterministic_for_identical_inputs() {
        let first = render(&base());
        let second = render(&base());
        assert_eq!(sha256_pcm(&first.samples), sha256_pcm(&second.samples));
        assert_eq!(first.samples, second.samples);
    }

    #[test]
    fn no_op_patch_is_byte_identical() {
        let segments = base();
        let baseline = render(&segments);
        let edited = apply_edit(
            &segments,
            "proxy-vowel-c",
            &EditPatch::NoOp,
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("no-op should be valid");
        let candidate = render(&edited);
        assert_eq!(baseline.samples, candidate.samples);
    }

    #[test]
    fn pitch_edit_changes_only_its_declared_dependency_cone() {
        let segments = base();
        let baseline = render(&segments);
        let edited = apply_edit(
            &segments,
            "proxy-vowel-c",
            &EditPatch::PitchShiftCents { delta_cents: 100.0 },
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("bounded pitch edit should be valid");
        let candidate = render(&edited);
        let allowed = range_by_id(&baseline, "proxy-vowel-c").expect("target range");
        let report =
            measure_locality(&baseline.samples, &candidate.samples, allowed).expect("locality");
        assert!(report.passes_exact_locality);
        assert_eq!(report.changed_protected_samples, 0);
        assert!(report.target_rms_delta > 0.0);
    }

    #[test]
    fn collateral_edit_is_detected_outside_target_cone() {
        let segments = base();
        let baseline = render(&segments);
        let once = apply_edit(
            &segments,
            "proxy-vowel-c",
            &EditPatch::PitchShiftCents { delta_cents: 100.0 },
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("target edit");
        let twice = apply_edit(
            &once,
            "proxy-vowel-e",
            &EditPatch::PitchShiftCents { delta_cents: 100.0 },
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("deliberate collateral edit");
        let candidate = render(&twice);
        let allowed = range_by_id(&baseline, "proxy-vowel-c").expect("target range");
        let report =
            measure_locality(&baseline.samples, &candidate.samples, allowed).expect("locality");
        assert!(!report.passes_exact_locality);
        assert!(report.changed_protected_samples > 0);
    }

    #[test]
    fn wrong_direction_edit_has_negative_parameter_delta() {
        let edited = apply_edit(
            &base(),
            "proxy-vowel-c",
            &EditPatch::PitchShiftCents { delta_cents: -100.0 },
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("bounded pitch edit");
        let original = base()
            .into_iter()
            .find(|segment| segment.segment_id == "proxy-vowel-c")
            .expect("target");
        let candidate = edited
            .into_iter()
            .find(|segment| segment.segment_id == "proxy-vowel-c")
            .expect("edited target");
        assert!(candidate.f0_hz < original.f0_hz);
    }

    #[test]
    fn formant_scale_does_not_change_declared_f0() {
        let original = base();
        let edited = apply_edit(
            &original,
            "proxy-vowel-c",
            &EditPatch::FormantScale { scale: 1.10 },
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("formant scale");
        let before = original
            .iter()
            .find(|segment| segment.segment_id == "proxy-vowel-c")
            .expect("target");
        let after = edited
            .iter()
            .find(|segment| segment.segment_id == "proxy-vowel-c")
            .expect("edited target");
        assert_eq!(before.f0_hz, after.f0_hz);
        assert!(after.f1_hz > before.f1_hz);
        assert!(after.f2_hz > before.f2_hz);
        assert!(after.f3_hz > before.f3_hz);
    }


    #[test]
    fn independent_pcm_measurement_tracks_controlled_pitch_shift() {
        let segments = base();
        let baseline = render(&segments);
        let edited_specs = apply_edit(
            &segments,
            "proxy-vowel-c",
            &EditPatch::PitchShiftCents { delta_cents: 100.0 },
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("bounded pitch edit");
        let edited = render(&edited_specs);
        let before = measure_segment_f0(&baseline, "proxy-vowel-c", DEFAULT_SAMPLE_RATE_HZ)
            .expect("baseline measurement")
            .expect("baseline F0");
        let after = measure_segment_f0(&edited, "proxy-vowel-c", DEFAULT_SAMPLE_RATE_HZ)
            .expect("edited measurement")
            .expect("edited F0");
        assert!(after > before * 1.04, "expected a positive pitch shift: {before} -> {after}");
    }

    #[test]
    fn formant_observer_analyzes_pcm_and_records_its_configuration() {
        let baseline = render(&base());
        let observed =
            measure_segment_formants(&baseline, "proxy-vowel-c", DEFAULT_SAMPLE_RATE_HZ)
                .expect("formant observer should process the target PCM");
        assert!(observed.analysis_frame_count > 0);
        assert_eq!(observed.analysis_sample_rate_hz, 10_000);
        assert_eq!(observed.lpc_order, 16);
        assert_eq!(observed.window_ms, 25.0);
        assert!(!observed.observer_id.is_empty());
    }

    #[test]
    fn formant_observer_does_not_report_voiced_frames_for_silenced_pcm() {
        let mut rendered = render(&base());
        let range = range_by_id(&rendered, "proxy-vowel-c").expect("target range");
        for sample in &mut rendered.samples[range.start..range.end] {
            *sample = 0.0;
        }
        let observed =
            measure_segment_formants(&rendered, "proxy-vowel-c", DEFAULT_SAMPLE_RATE_HZ)
                .expect("silenced PCM is a valid analyzer input");
        assert_eq!(observed.voiced_frame_count, 0);
        assert_eq!(observed.ordered_formant_frame_count, 0);
        assert_eq!(observed.median_f1_hz, None);
        assert_eq!(observed.median_f2_hz, None);
        assert_eq!(observed.median_f3_hz, None);
    }

    #[test]
    fn independent_f0_estimator_recovers_a_clean_sine() {
        let sample_rate = 24_000_u32;
        let frequency = 220.0_f32;
        let samples: Vec<f32> = (0..4_800)
            .map(|index| {
                (std::f32::consts::TAU * frequency * index as f32 / sample_rate as f32).sin()
            })
            .collect();
        let estimated =
            estimate_f0_hz(&samples, sample_rate, 80.0, 500.0).expect("estimate");
        assert!((estimated - frequency).abs() < 0.5, "estimated {estimated}");
    }

    #[test]
    fn independent_f0_rejects_silence_and_invalid_inputs() {
        assert!(estimate_f0_hz(&[0.0; 2_000], 24_000, 80.0, 500.0).is_none());
        assert!(estimate_f0_hz(&[0.0; 32], 24_000, 80.0, 500.0).is_none());
        assert!(estimate_f0_hz(&[0.0; 256], 0, 80.0, 500.0).is_none());
    }

    #[test]
    fn duplicate_segment_ids_are_rejected() {
        let mut segments = base();
        segments[1].segment_id = segments[0].segment_id.clone();
        assert!(render_phrase(&segments, DEFAULT_SAMPLE_RATE_HZ, 5).is_err());
    }

    #[test]
    fn invalid_formant_order_is_rejected() {
        let mut segments = base();
        segments[0].f2_hz = segments[0].f1_hz;
        assert!(render_phrase(&segments, DEFAULT_SAMPLE_RATE_HZ, 5).is_err());
    }

    #[test]
    fn unsupported_renderer_control_is_rejected_without_fallback() {
        let result = apply_edit(
            &base(),
            "proxy-vowel-c",
            &EditPatch::UnsupportedControl {
                control_id: "vibrato_phase_offset".into(),
            },
            DEFAULT_SAMPLE_RATE_HZ,
        );
        assert!(result.is_err());
        assert!(result.expect_err("must reject").contains("unsupported renderer control"));
    }

    #[test]
    fn unknown_target_and_unbounded_patch_are_rejected() {
        let segments = base();
        assert!(apply_edit(
            &segments,
            "not-a-segment",
            &EditPatch::NoOp,
            DEFAULT_SAMPLE_RATE_HZ
        )
        .is_err());
        assert!(apply_edit(
            &segments,
            "proxy-vowel-c",
            &EditPatch::PitchShiftCents { delta_cents: 2_000.0 },
            DEFAULT_SAMPLE_RATE_HZ
        )
        .is_err());
    }

    #[test]
    fn sample_ranges_and_crossfade_are_deterministic() {
        let rendered = render(&base());
        assert_eq!(rendered.segment_ranges.len(), 5);
        assert_eq!(rendered.seam_guard_samples, 120);
        assert!(rendered.segment_ranges.windows(2).all(|pair| {
            pair[0].1.start < pair[1].1.start && pair[0].1.end <= pair[1].1.end
        }));
        assert_eq!(rendered.samples.len(), 5 * 64 * 120 - 4 * 120);
    }

    #[test]
    fn locality_rejects_length_mismatch_and_invalid_cones() {
        assert!(measure_locality(&[0.0; 10], &[0.0; 9], SampleRange { start: 1, end: 4 })
            .is_err());
        assert!(measure_locality(&[0.0; 10], &[0.0; 10], SampleRange { start: 4, end: 4 })
            .is_err());
        assert!(measure_locality(&[0.0; 10], &[0.0; 10], SampleRange { start: 4, end: 12 })
            .is_err());
    }

    #[test]
    fn suite_manifest_is_synthetic_only_and_contains_negative_controls() {
        let dir = std::env::temp_dir().join(format!(
            "synthetic-vocal-plant-test-{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let manifest = generate_suite(&dir).expect("suite should generate");
        assert_eq!(manifest.content_class, "synthetic_only");
        assert!(!manifest.human_recordings_ingested);
        assert!(!manifest.external_corpus_assets_used);
        assert!(!manifest.voice_identity_targeting);
        assert!(manifest.scenarios.iter().any(|case| case.case_id == "no_op"));
        assert!(manifest
            .scenarios
            .iter()
            .any(|case| case.case_id == "wrong_direction_pitch_down_100c"));
        assert!(manifest
            .scenarios
            .iter()
            .any(|case| case.case_id == "collateral_corruption"));
        assert_eq!(manifest.unsupported_control_rejections.len(), 1);
        assert_eq!(
            manifest.unsupported_control_rejections[0].disposition,
            "REJECTED_UNSUPPORTED"
        );
        let no_op = manifest
            .scenarios
            .iter()
            .find(|case| case.case_id == "no_op")
            .expect("no-op case");
        assert_eq!(no_op.pcm_sha256, manifest.baseline_pcm_sha256);
        let collateral = manifest
            .scenarios
            .iter()
            .find(|case| case.case_id == "collateral_corruption")
            .expect("collateral case");
        assert!(!collateral
            .locality
            .as_ref()
            .expect("locality report")
            .passes_exact_locality);
        assert!(dir.join("manifest.json").is_file());
        assert!(dir.join("manifest.sha256").is_file());
        let _ = std::fs::remove_dir_all(dir);
    }
}
