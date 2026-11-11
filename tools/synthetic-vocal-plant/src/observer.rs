// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! SYNTH-OBS-001: independent, PCM-only acoustic observations.
//!
//! The renderer supplies no control values to this module. F0 is measured
//! from PCM through two independently implemented paths: this crate's
//! normalized-autocorrelation observer and the vocal-tract crate's LPC
//! extraction path. Disagreement is preserved as a diagnostic.

use serde::{Deserialize, Serialize};
use std::path::Path;
use symthaea_vocal_tract::formant_extraction::{
    ExtractionConfig, extract_formant_frames,
};

use crate::{
    DEFAULT_CROSSFADE_MS, DEFAULT_SAMPLE_RATE_HZ, EditPatch, RenderedPhrase, SampleRange,
    SyntheticSegmentSpec, apply_edit, default_segments, estimate_f0_hz, render_phrase,
    sha256_bytes, sha256_pcm,
};

pub const OBSERVER_SCHEMA_VERSION: &str = "synthetic-acoustic-observations-1.0.0";
pub const AUTOCORRELATION_ANALYZER_ID: &str = "synthetic-vocal-plant.autocorrelation-f0";
pub const AUTOCORRELATION_ANALYZER_VERSION: &str = "1.1.0";
pub const LPC_ANALYZER_ID: &str =
    "symthaea_vocal_tract::formant_extraction::extract_formant_frames";
pub const LPC_ANALYZER_VERSION: &str = "symthaea-vocal-tract-0.1.0/lpc16-10khz-200hz-v1";
pub const F0_CROSSCHECK_LIMIT_CENTS: f32 = 50.0;
pub const MIN_ELIGIBLE_FRAMES: usize = 5;
pub const MIN_VOICED_FRACTION: f32 = 0.50;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservationDisposition {
    Observed,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum F0CrosscheckStatus {
    Agree,
    Disagree,
    OneObserverUnavailable,
    BothUnavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MetricObservationRecord {
    pub metric_id: String,
    pub unit: String,
    pub value: Option<f64>,
    pub disposition: ObservationDisposition,
    pub analyzer_id: String,
    pub analyzer_version: String,
    pub pcm_sha256: String,
    pub window: SampleRange,
    pub eligible_frame_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SegmentObservationRecord {
    pub source_case_id: String,
    pub case_id: String,
    pub segment_id: String,
    pub pcm_sha256: String,
    pub window: SampleRange,
    pub window_start_secs: f64,
    pub window_end_secs: f64,
    pub total_analysis_frames: usize,
    pub voiced_analysis_frames: usize,
    pub f0_crosscheck_status: F0CrosscheckStatus,
    pub f0_observer_difference_cents: Option<f64>,
    pub metrics: Vec<MetricObservationRecord>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VcrbObservationAdapter {
    pub source_case_id: String,
    pub pair_id: String,
    pub calibration_id: String,
    pub metric_id: String,
    pub unit: String,
    pub value: f64,
    pub analyzer_id: String,
    pub analyzer_version: String,
    pub pcm_sha256: String,
    pub window: SampleRange,
}

/// Narrow observation shape for the existing VCRB-001 evaluator.
/// This intentionally does not duplicate its ControlSpec/ControlTrial logic.
pub fn to_vcrb_observations(
    records: &[SegmentObservationRecord],
    pair_id: &str,
    calibration_id: &str,
) -> Result<Vec<VcrbObservationAdapter>, String> {
    if pair_id.trim().is_empty() || calibration_id.trim().is_empty() {
        return Err("pair_id and calibration_id must be non-empty".into());
    }
    let mut output = Vec::new();
    for record in records {
        for metric in &record.metrics {
            if metric.disposition != ObservationDisposition::Observed {
                continue;
            }
            let value = metric.value.ok_or_else(|| {
                format!(
                    "metric {} is marked observed without a numeric value",
                    metric.metric_id
                )
            })?;
            if !value.is_finite() {
                return Err(format!("metric {} is non-finite", metric.metric_id));
            }
            output.push(VcrbObservationAdapter {
                source_case_id: record.source_case_id.clone(),
                pair_id: pair_id.into(),
                calibration_id: calibration_id.into(),
                metric_id: metric.metric_id.clone(),
                unit: metric.unit.clone(),
                value,
                analyzer_id: metric.analyzer_id.clone(),
                analyzer_version: metric.analyzer_version.clone(),
                pcm_sha256: metric.pcm_sha256.clone(),
                window: metric.window,
            });
        }
    }
    Ok(output)
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationMetric {
    pub metric_id: String,
    pub unit: String,
    pub repeat_max_abs_delta: Option<f64>,
    pub disposition: ObservationDisposition,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct NoOpCalibrationReport {
    pub calibration_id: String,
    pub calibration_source_case_id: String,
    pub confirmatory_source_case_id: String,
    pub calibration_parameter_sha256: String,
    pub confirmatory_parameter_sha256: String,
    pub calibration_pcm_sha256: String,
    pub repeat_count: usize,
    pub exact_pcm_repeatability: bool,
    pub split_identity_disjoint: bool,
    pub maximum_observed_metric_delta: Option<f64>,
    pub metrics: Vec<CalibrationMetric>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObserverBundleManifest {
    pub schema_version: String,
    pub content_class: String,
    pub human_recordings_ingested: bool,
    pub external_corpus_assets_used: bool,
    pub voice_identity_targeting: bool,
    pub source_revision: String,
    pub sample_rate_hz: u32,
    pub observer_configuration_sha256: String,
    pub lpc_analyzer_id: String,
    pub lpc_analyzer_version: String,
    pub autocorrelation_analyzer_id: String,
    pub autocorrelation_analyzer_version: String,
    pub calibration_file: String,
    pub calibration_file_sha256: String,
    pub observations_file: String,
    pub observations_file_sha256: String,
    pub calibration: NoOpCalibrationReport,
    pub observation_record_count: usize,
}

fn lpc_config() -> ExtractionConfig {
    ExtractionConfig {
        frame_rate: 200.0,
        window_secs: 0.025,
        analysis_sr: 10_000,
        lpc_order: 16,
        preemphasis: 0.97,
        f0_min: 60.0,
        f0_max: 700.0,
        voicing_threshold: 0.30,
        silence_rel_energy: 0.02,
    }
}

fn config_digest(config: &ExtractionConfig) -> String {
    let canonical = format!(
        "frame_rate={:.6};window_secs={:.6};analysis_sr={};lpc_order={};preemphasis={:.6};f0_min={:.6};f0_max={:.6};voicing_threshold={:.6};silence_rel_energy={:.6}",
        config.frame_rate,
        config.window_secs,
        config.analysis_sr,
        config.lpc_order,
        config.preemphasis,
        config.f0_min,
        config.f0_max,
        config.voicing_threshold,
        config.silence_rel_energy
    );
    sha256_bytes(canonical.as_bytes())
}

fn median(values: &mut [f32]) -> Option<f32> {
    values.retain(|value| value.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) * 0.5
    })
}

fn measure_autocorrelation_f0(
    samples: &[f32],
    sample_rate_hz: u32,
    window: SampleRange,
    edge_guard_samples: usize,
) -> Option<f32> {
    if window.end > samples.len() || window.start >= window.end {
        return None;
    }
    let start = window.start.saturating_add(edge_guard_samples);
    let end = window.end.saturating_sub(edge_guard_samples);
    if end <= start {
        return None;
    }
    let usable = &samples[start..end];
    let window_len = ((sample_rate_hz as usize) * 40 / 1_000).max(64);
    let hop = ((sample_rate_hz as usize) * 5 / 1_000).max(1);
    if usable.len() < window_len {
        return estimate_f0_hz(usable, sample_rate_hz, 60.0, 700.0);
    }
    let mut estimates = Vec::new();
    for offset in (0..=usable.len() - window_len).step_by(hop) {
        if let Some(value) =
            estimate_f0_hz(&usable[offset..offset + window_len], sample_rate_hz, 60.0, 700.0)
        {
            estimates.push(value);
        }
    }
    median(&mut estimates)
}

fn option_record(
    metric_id: &str,
    unit: &str,
    value: Option<f32>,
    analyzer_id: &str,
    analyzer_version: &str,
    pcm_sha256: &str,
    window: SampleRange,
    eligible_frame_count: usize,
    minimum_frames: usize,
) -> MetricObservationRecord {
    let accepted = value.filter(|value| value.is_finite()).filter(|_| {
        eligible_frame_count >= minimum_frames
    });
    MetricObservationRecord {
        metric_id: metric_id.into(),
        unit: unit.into(),
        value: accepted.map(f64::from),
        disposition: if accepted.is_some() {
            ObservationDisposition::Observed
        } else {
            ObservationDisposition::Indeterminate
        },
        analyzer_id: analyzer_id.into(),
        analyzer_version: analyzer_version.into(),
        pcm_sha256: pcm_sha256.into(),
        window,
        eligible_frame_count,
    }
}

/// Extract independent acoustic measurements for each stable segment ID.
///
/// LPC frames are accepted only when their complete analysis windows are
/// inside the segment after excluding the segment edge/seam guard. The
/// autocorrelation observer receives PCM only and applies its own edge guard.
pub fn observe_phrase(
    source_case_id: &str,
    case_id: &str,
    rendered: &RenderedPhrase,
    sample_rate_hz: u32,
) -> Result<Vec<SegmentObservationRecord>, String> {
    if source_case_id.trim().is_empty() || case_id.trim().is_empty() {
        return Err("source_case_id and case_id must be non-empty".into());
    }
    if sample_rate_hz < 8_000 || sample_rate_hz > 96_000 {
        return Err("sample rate must be in 8000..=96000 Hz".into());
    }
    if rendered.samples.is_empty()
        || rendered.samples.iter().any(|sample| !sample.is_finite())
    {
        return Err("PCM is empty or contains non-finite samples".into());
    }

    let config = lpc_config();
    let lpc_frames = extract_formant_frames(&rendered.samples, sample_rate_hz, &config);
    let pcm_digest = sha256_pcm(&rendered.samples);
    let analysis_window_samples =
        (config.window_secs * sample_rate_hz as f32).ceil() as usize;
    let edge_guard = rendered.seam_guard_samples.max(analysis_window_samples);
    let mut output = Vec::with_capacity(rendered.segment_ranges.len());

    for (segment_id, segment_range) in &rendered.segment_ranges {
        if segment_range.start >= segment_range.end
            || segment_range.end > rendered.samples.len()
        {
            return Err(format!("invalid range for segment {segment_id}"));
        }
        let trim = edge_guard;
        let start_sample = segment_range.start.saturating_add(trim);
        let end_sample = segment_range.end.saturating_sub(trim);
        let safe_window = SampleRange {
            start: start_sample.min(end_sample),
            end: end_sample,
        };
        let start_secs = safe_window.start as f64 / sample_rate_hz as f64;
        let end_secs = safe_window.end as f64 / sample_rate_hz as f64;
        let window_secs = config.window_secs as f64;

        let eligible: Vec<_> = lpc_frames
            .iter()
            .filter(|frame| {
                frame.time as f64 >= start_secs
                    && frame.time as f64 + window_secs <= end_secs
            })
            .collect();
        let voiced: Vec<_> = eligible
            .iter()
            .copied()
            .filter(|frame| {
                frame.voicing >= config.voicing_threshold
                    && frame.f0.is_finite()
                    && frame.f0 > 0.0
            })
            .collect();

        let mut lpc_f0_values: Vec<f32> = voiced.iter().map(|frame| frame.f0).collect();
        let mut f1_values: Vec<f32> = voiced.iter().map(|frame| frame.f1).collect();
        let mut f2_values: Vec<f32> = voiced.iter().map(|frame| frame.f2).collect();
        let mut f3_values: Vec<f32> = voiced.iter().map(|frame| frame.f3).collect();
        let mut energy_values: Vec<f32> = eligible.iter().map(|frame| frame.energy).collect();

        let lpc_f0 = median(&mut lpc_f0_values);
        let f1 = median(&mut f1_values);
        let f2 = median(&mut f2_values);
        let f3 = median(&mut f3_values);
        let energy = median(&mut energy_values);
        let voiced_fraction = if eligible.is_empty() {
            None
        } else {
            Some(voiced.len() as f32 / eligible.len() as f32)
        };

        let custom_f0 = measure_autocorrelation_f0(
            &rendered.samples,
            sample_rate_hz,
            *segment_range,
            edge_guard,
        );
        let crosscheck_cents = match (custom_f0, lpc_f0) {
            (Some(left), Some(right)) if left > 0.0 && right > 0.0 => {
                Some(1200.0 * (right / left).log2())
            }
            _ => None,
        };
        let crosscheck_status = match crosscheck_cents {
            Some(value) if value.is_finite() && value.abs() <= F0_CROSSCHECK_LIMIT_CENTS => {
                F0CrosscheckStatus::Agree
            }
            Some(_) => F0CrosscheckStatus::Disagree,
            None if custom_f0.is_none() && lpc_f0.is_none() => {
                F0CrosscheckStatus::BothUnavailable
            }
            None => F0CrosscheckStatus::OneObserverUnavailable,
        };

        let metric_window = safe_window;
        let metrics = vec![
            option_record(
                "pitch.f0.median.autocorrelation",
                "Hz",
                custom_f0,
                AUTOCORRELATION_ANALYZER_ID,
                AUTOCORRELATION_ANALYZER_VERSION,
                &pcm_digest,
                metric_window,
                eligible.len(),
                MIN_ELIGIBLE_FRAMES,
            ),
            option_record(
                "pitch.f0.median.lpc_autocorrelation",
                "Hz",
                lpc_f0,
                LPC_ANALYZER_ID,
                LPC_ANALYZER_VERSION,
                &pcm_digest,
                metric_window,
                voiced.len(),
                MIN_ELIGIBLE_FRAMES,
            ),
            option_record(
                "formant.f1.median",
                "Hz",
                f1,
                LPC_ANALYZER_ID,
                LPC_ANALYZER_VERSION,
                &pcm_digest,
                metric_window,
                voiced.len(),
                MIN_ELIGIBLE_FRAMES,
            ),
            option_record(
                "formant.f2.median",
                "Hz",
                f2,
                LPC_ANALYZER_ID,
                LPC_ANALYZER_VERSION,
                &pcm_digest,
                metric_window,
                voiced.len(),
                MIN_ELIGIBLE_FRAMES,
            ),
            option_record(
                "formant.f3.median",
                "Hz",
                f3,
                LPC_ANALYZER_ID,
                LPC_ANALYZER_VERSION,
                &pcm_digest,
                metric_window,
                voiced.len(),
                MIN_ELIGIBLE_FRAMES,
            ),
            option_record(
                "vocal.energy.median",
                "relative_peak_rms",
                energy,
                LPC_ANALYZER_ID,
                LPC_ANALYZER_VERSION,
                &pcm_digest,
                metric_window,
                eligible.len(),
                MIN_ELIGIBLE_FRAMES,
            ),
            option_record(
                "vocal.voicing.coverage",
                "fraction",
                voiced_fraction,
                LPC_ANALYZER_ID,
                LPC_ANALYZER_VERSION,
                &pcm_digest,
                metric_window,
                eligible.len(),
                MIN_ELIGIBLE_FRAMES,
            ),
        ];

        output.push(SegmentObservationRecord {
            source_case_id: source_case_id.into(),
            case_id: case_id.into(),
            segment_id: segment_id.clone(),
            pcm_sha256: pcm_digest.clone(),
            window: metric_window,
            window_start_secs: start_secs,
            window_end_secs: end_secs,
            total_analysis_frames: eligible.len(),
            voiced_analysis_frames: voiced.len(),
            f0_crosscheck_status: crosscheck_status,
            f0_observer_difference_cents: crosscheck_cents.map(f64::from),
            metrics,
        });
    }
    Ok(output)
}

fn calibration_segments() -> Vec<SyntheticSegmentSpec> {
    vec![SyntheticSegmentSpec {
        segment_id: "calibration-proxy-vowel".into(),
        frame_count: 64,
        f0_hz: 196.0,
        f1_hz: 460.0,
        f2_hz: 1_360.0,
        f3_hz: 2_620.0,
        b1_hz: 72.0,
        b2_hz: 96.0,
        b3_hz: 145.0,
        energy: 0.61,
        vibrato_rate_hz: 5.2,
        vibrato_depth_cents: 8.0,
    }]
}

fn calibration_parameter_digest(segments: &[SyntheticSegmentSpec]) -> Result<String, String> {
    let bytes = serde_json::to_vec(segments)
        .map_err(|error| format!("failed to serialize calibration parameters: {error}"))?;
    Ok(sha256_bytes(&bytes))
}

fn confirmatory_parameter_digest() -> Result<String, String> {
    let bytes = serde_json::to_vec(&default_segments())
        .map_err(|error| format!("failed to serialize confirmatory parameters: {error}"))?;
    Ok(sha256_bytes(&bytes))
}

pub fn calibrate_no_op_repeatability(
    calibration_source_case_id: &str,
    confirmatory_source_case_id: &str,
    segments: &[SyntheticSegmentSpec],
    sample_rate_hz: u32,
) -> Result<NoOpCalibrationReport, String> {
    if calibration_source_case_id.trim().is_empty()
        || confirmatory_source_case_id.trim().is_empty()
        || calibration_source_case_id == confirmatory_source_case_id
    {
        return Err("calibration and confirmatory source identities must be distinct".into());
    }
    let calibration_parameter_sha256 = calibration_parameter_digest(segments)?;
    let confirmatory_parameter_sha256 = confirmatory_parameter_digest()?;
    let rendered_a = render_phrase(segments, sample_rate_hz, DEFAULT_CROSSFADE_MS)?;
    let rendered_b = render_phrase(segments, sample_rate_hz, DEFAULT_CROSSFADE_MS)?;
    let observations_a = observe_phrase(
        calibration_source_case_id,
        "calibration_repeat_a",
        &rendered_a,
        sample_rate_hz,
    )?;
    let observations_b = observe_phrase(
        calibration_source_case_id,
        "calibration_repeat_b",
        &rendered_b,
        sample_rate_hz,
    )?;
    if observations_a.len() != observations_b.len() {
        return Err("calibration repeat observation coverage differs".into());
    }

    let exact_pcm_repeatability = rendered_a.samples == rendered_b.samples;
    let mut metrics = Vec::new();
    let mut maximum_observed_metric_delta: Option<f64> = None;
    for (left, right) in observations_a.iter().zip(observations_b.iter()) {
        for left_metric in &left.metrics {
            let right_metric = right
                .metrics
                .iter()
                .find(|metric| metric.metric_id == left_metric.metric_id)
                .ok_or_else(|| format!("missing repeat metric {}", left_metric.metric_id))?;
            let delta = match (left_metric.value, right_metric.value) {
                (Some(a), Some(b)) => {
                    let difference = (a - b).abs();
                    maximum_observed_metric_delta = Some(
                        maximum_observed_metric_delta
                            .map_or(difference, |current| current.max(difference)),
                    );
                    Some(difference)
                }
                _ => None,
            };
            metrics.push(CalibrationMetric {
                metric_id: left_metric.metric_id.clone(),
                unit: left_metric.unit.clone(),
                repeat_max_abs_delta: delta,
                disposition: if delta.is_some() {
                    ObservationDisposition::Observed
                } else {
                    ObservationDisposition::Indeterminate
                },
            });
        }
    }
    let split_identity_disjoint = calibration_parameter_sha256 != confirmatory_parameter_sha256
        && calibration_source_case_id != confirmatory_source_case_id;
    let calibration_pcm_sha256 = sha256_pcm(&rendered_a.samples);
    let calibration_id = sha256_bytes(
        format!(
            "{}:{}:{}:{}",
            calibration_source_case_id,
            calibration_parameter_sha256,
            calibration_pcm_sha256,
            config_digest(&lpc_config())
        )
        .as_bytes(),
    );

    Ok(NoOpCalibrationReport {
        calibration_id,
        calibration_source_case_id: calibration_source_case_id.into(),
        confirmatory_source_case_id: confirmatory_source_case_id.into(),
        calibration_parameter_sha256,
        confirmatory_parameter_sha256,
        calibration_pcm_sha256,
        repeat_count: 2,
        exact_pcm_repeatability,
        split_identity_disjoint,
        maximum_observed_metric_delta,
        metrics,
    })
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ObserverJsonlRecord {
    pub schema_version: String,
    pub record: SegmentObservationRecord,
}

pub fn generate_observer_bundle(output_dir: &Path) -> Result<ObserverBundleManifest, String> {
    std::fs::create_dir_all(output_dir)
        .map_err(|error| format!("failed to create observer output directory: {error}"))?;
    let sample_rate_hz = DEFAULT_SAMPLE_RATE_HZ;
    let source_case_id = "SYNTH_VOCAL_CONFIRMATORY_V1";
    let confirmatory = default_segments();
    let calibration = calibrate_no_op_repeatability(
        "SYNTH_VOCAL_CALIBRATION_V1",
        source_case_id,
        &calibration_segments(),
        sample_rate_hz,
    )?;
    let baseline = render_phrase(
        &confirmatory,
        sample_rate_hz,
        DEFAULT_CROSSFADE_MS,
    )?;
    let pitch_up_specs = apply_edit(
        &confirmatory,
        "proxy-vowel-c",
        &EditPatch::PitchShiftCents { delta_cents: 100.0 },
        sample_rate_hz,
    )?;
    let pitch_up = render_phrase(
        &pitch_up_specs,
        sample_rate_hz,
        DEFAULT_CROSSFADE_MS,
    )?;
    let formant_specs = apply_edit(
        &confirmatory,
        "proxy-vowel-c",
        &EditPatch::FormantScale { scale: 1.10 },
        sample_rate_hz,
    )?;
    let formant_shift = render_phrase(
        &formant_specs,
        sample_rate_hz,
        DEFAULT_CROSSFADE_MS,
    )?;

    let mut records = Vec::new();
    for (case_id, rendered) in [
        ("confirmatory_baseline", &baseline),
        ("confirmatory_pitch_up", &pitch_up),
        ("confirmatory_formant_scale", &formant_shift),
    ] {
        records.extend(observe_phrase(
            source_case_id,
            case_id,
            rendered,
            sample_rate_hz,
        )?);
    }

    let mut jsonl = String::new();
    for record in &records {
        let wrapped = ObserverJsonlRecord {
            schema_version: OBSERVER_SCHEMA_VERSION.into(),
            record: record.clone(),
        };
        let line = serde_json::to_string(&wrapped)
            .map_err(|error| format!("failed to serialize observation record: {error}"))?;
        jsonl.push_str(&line);
        jsonl.push('\n');
    }
    let observations_file = "observations.jsonl";
    let observations_path = output_dir.join(observations_file);
    std::fs::write(&observations_path, &jsonl)
        .map_err(|error| format!("failed to write observations: {error}"))?;
    let observations_file_sha256 = sha256_bytes(jsonl.as_bytes());

    let calibration_file = "calibration.json";
    let calibration_bytes = serde_json::to_vec_pretty(&calibration)
        .map_err(|error| format!("failed to serialize calibration report: {error}"))?;
    std::fs::write(output_dir.join(calibration_file), &calibration_bytes)
        .map_err(|error| format!("failed to write calibration report: {error}"))?;
    let calibration_file_sha256 = sha256_bytes(&calibration_bytes);

    let config = lpc_config();
    let manifest = ObserverBundleManifest {
        schema_version: OBSERVER_SCHEMA_VERSION.into(),
        content_class: "synthetic_only".into(),
        human_recordings_ingested: false,
        external_corpus_assets_used: false,
        voice_identity_targeting: false,
        source_revision: std::env::var("SYNTH_VOCAL_PLANT_SOURCE_REVISION")
            .unwrap_or_else(|_| "unknown-at-build".into()),
        sample_rate_hz,
        observer_configuration_sha256: config_digest(&config),
        lpc_analyzer_id: LPC_ANALYZER_ID.into(),
        lpc_analyzer_version: LPC_ANALYZER_VERSION.into(),
        autocorrelation_analyzer_id: AUTOCORRELATION_ANALYZER_ID.into(),
        autocorrelation_analyzer_version: AUTOCORRELATION_ANALYZER_VERSION.into(),
        calibration_file: calibration_file.into(),
        calibration_file_sha256,
        observations_file: observations_file.into(),
        observations_file_sha256,
        calibration,
        observation_record_count: records.len(),
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)
        .map_err(|error| format!("failed to serialize observer manifest: {error}"))?;
    std::fs::write(output_dir.join("observer-manifest.json"), &manifest_bytes)
        .map_err(|error| format!("failed to write observer manifest: {error}"))?;
    std::fs::write(
        output_dir.join("observer-manifest.sha256"),
        format!("{}  observer-manifest.json\n", sha256_bytes(&manifest_bytes)),
    )
    .map_err(|error| format!("failed to write observer manifest digest: {error}"))?;
    Ok(manifest)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn phrase() -> RenderedPhrase {
        render_phrase(
            &default_segments(),
            DEFAULT_SAMPLE_RATE_HZ,
            DEFAULT_CROSSFADE_MS,
        )
        .expect("synthetic phrase should render")
    }

    #[test]
    fn observer_produces_pcm_bound_f0_and_formant_records() {
        let rendered = phrase();
        let records = observe_phrase(
            "CONFIRMATORY_TEST_CASE",
            "baseline",
            &rendered,
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("PCM observer should run");
        let target = records
            .iter()
            .find(|record| record.segment_id == "proxy-vowel-c")
            .expect("target segment");
        assert_eq!(target.pcm_sha256, sha256_pcm(&rendered.samples));
        assert_eq!(target.metrics.len(), 7);
        assert!(target.window.start < target.window.end);
        assert!(target.total_analysis_frames >= MIN_ELIGIBLE_FRAMES);
        assert!(target
            .metrics
            .iter()
            .any(|metric| metric.metric_id == "formant.f1.median"));
        assert!(target
            .metrics
            .iter()
            .filter(|metric| metric.disposition == ObservationDisposition::Observed)
            .all(|metric| metric.value.is_some_and(f64::is_finite)));
    }

    #[test]
    fn silent_pcm_never_receives_a_fabricated_f0_or_formant_pass() {
        let silent = RenderedPhrase {
            samples: vec![0.0; DEFAULT_SAMPLE_RATE_HZ as usize],
            segment_ranges: vec![(
                "silent-segment".into(),
                SampleRange {
                    start: 0,
                    end: DEFAULT_SAMPLE_RATE_HZ as usize,
                },
            )],
            seam_guard_samples: 120,
        };
        let records = observe_phrase(
            "SYNTHETIC_SILENCE",
            "silence",
            &silent,
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("silence remains analyzable");
        let record = &records[0];
        assert_eq!(record.f0_crosscheck_status, F0CrosscheckStatus::BothUnavailable);
        assert!(record.metrics.iter().any(|metric| {
            metric.metric_id == "pitch.f0.median.autocorrelation"
                && metric.disposition == ObservationDisposition::Indeterminate
                && metric.value.is_none()
        }));
        assert!(record.metrics.iter().any(|metric| {
            metric.metric_id == "formant.f1.median"
                && metric.disposition == ObservationDisposition::Indeterminate
                && metric.value.is_none()
        }));
    }

    #[test]
    fn vcrb_adapter_excludes_indeterminate_metrics_and_preserves_units() {
        let records = observe_phrase(
            "CONFIRMATORY_TEST_CASE",
            "baseline",
            &phrase(),
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("observer");
        let adapted = to_vcrb_observations(&records, "pair-001", "calibration-001")
            .expect("adapter");
        assert!(!adapted.is_empty());
        assert!(adapted.iter().all(|row| {
            row.pair_id == "pair-001"
                && row.calibration_id == "calibration-001"
                && !row.unit.is_empty()
                && row.value.is_finite()
        }));
        assert!(to_vcrb_observations(&records, "", "calibration-001").is_err());
    }

    #[test]
    fn calibration_uses_a_distinct_parameter_family_and_is_byte_repeatable() {
        let calibration = calibrate_no_op_repeatability(
            "CALIBRATION_FAMILY_A",
            "CONFIRMATORY_FAMILY_B",
            &calibration_segments(),
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("repeatability calibration");
        assert!(calibration.exact_pcm_repeatability);
        assert!(calibration.split_identity_disjoint);
        assert_ne!(
            calibration.calibration_parameter_sha256,
            calibration.confirmatory_parameter_sha256
        );
        assert_eq!(calibration.repeat_count, 2);
    }

    #[test]
    fn observer_outputs_are_deterministic_for_identical_pcm() {
        let rendered = phrase();
        let first = observe_phrase(
            "CONFIRMATORY_TEST_CASE",
            "baseline",
            &rendered,
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("first observation");
        let second = observe_phrase(
            "CONFIRMATORY_TEST_CASE",
            "baseline",
            &rendered,
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("second observation");
        assert_eq!(first, second);
    }

    #[test]
    fn empty_or_non_finite_pcm_is_rejected() {
        let empty = RenderedPhrase {
            samples: Vec::new(),
            segment_ranges: Vec::new(),
            seam_guard_samples: 0,
        };
        assert!(observe_phrase("CASE", "empty", &empty, DEFAULT_SAMPLE_RATE_HZ).is_err());
        let non_finite = RenderedPhrase {
            samples: vec![f32::NAN; 512],
            segment_ranges: vec![(
                "bad".into(),
                SampleRange { start: 0, end: 512 },
            )],
            seam_guard_samples: 0,
        };
        assert!(observe_phrase("CASE", "bad", &non_finite, DEFAULT_SAMPLE_RATE_HZ).is_err());
    }

    #[test]
    fn split_identity_collision_is_rejected() {
        assert!(calibrate_no_op_repeatability(
            "same-family",
            "same-family",
            &calibration_segments(),
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .is_err());
    }

    #[test]
    fn f0_disagreement_is_a_diagnostic_not_an_averaged_result() {
        let rendered = phrase();
        let records = observe_phrase(
            "CONFIRMATORY_TEST_CASE",
            "baseline",
            &rendered,
            DEFAULT_SAMPLE_RATE_HZ,
        )
        .expect("observer");
        let target = records
            .iter()
            .find(|record| record.segment_id == "proxy-vowel-c")
            .expect("target");
        if target.f0_crosscheck_status == F0CrosscheckStatus::Disagree {
            assert!(target.f0_observer_difference_cents.is_some());
        }
    }
}
