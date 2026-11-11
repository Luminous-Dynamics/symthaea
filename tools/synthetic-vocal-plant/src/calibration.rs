// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! SYNTH-CAL-001: fit a local response profile from measured synthetic PCM.
//!
//! This module compiles independent observer deltas into a VCCA-shaped
//! sensitivity profile. It does not reimplement VCCA allocation or authorize
//! renderer actions. All fitted cells are tied to the named legacy renderer,
//! LPC/autocorrelation analyzers, source families and calibration receipt.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

use crate::observer::{
    AUTOCORRELATION_ANALYZER_ID, AUTOCORRELATION_ANALYZER_VERSION, LPC_ANALYZER_ID,
    LPC_ANALYZER_VERSION, NoOpCalibrationReport, calibrate_no_op_repeatability,
    observe_phrase,
};
use crate::{
    DEFAULT_CROSSFADE_MS, DEFAULT_SAMPLE_RATE_HZ, EditPatch, RENDERER_BACKEND_ID,
    SyntheticSegmentSpec, apply_edit, default_segments, render_phrase,
    sha256_bytes, sha256_pcm,
};

pub const RESPONSE_PROFILE_SCHEMA: &str = "synthetic-vocal-response-profile-1.0.0";
pub const RESPONSE_FIT_METHOD: &str = "through_origin_least_squares_with_worst_case_ratio_bound-v1";
pub const MIN_CALIBRATION_PAIRS_PER_CELL: usize = 8;
pub const HOLDOUT_NUMERIC_EPSILON: f64 = 1e-8;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlProbe {
    pub control_id: String,
    pub control_unit: String,
    pub input_delta: f64,
    pub patch: EditPatch,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationPairRecord {
    pub source_case_id: String,
    pub family_parameter_sha256: String,
    pub control_id: String,
    pub control_unit: String,
    pub input_delta: f64,
    pub target_segment_id: String,
    pub baseline_pcm_sha256: String,
    pub intervention_pcm_sha256: String,
    pub metric_deltas: Vec<MeasuredMetricDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MeasuredMetricDelta {
    pub metric_id: String,
    pub unit: String,
    pub baseline_value: f64,
    pub intervention_value: f64,
    pub delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VccaSensitivityCellAdapter {
    pub metric_id: String,
    pub metric_unit: String,
    pub control_id: String,
    pub control_unit: String,
    pub effect_per_control_unit: f64,
    pub absolute_uncertainty_per_control_unit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VccaControlBoundAdapter {
    pub control_id: String,
    pub unit: String,
    pub supported: bool,
    pub minimum_delta: f64,
    pub maximum_delta: f64,
    pub trust_radius: f64,
    pub effort_scale: f64,
    pub effort_weight: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VccaMetricResidualAdapter {
    pub metric_id: String,
    pub unit: String,
    pub residual_uncertainty: f64,
    pub repeatability_receipt_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct VccaResponseProfileAdapter {
    pub response_profile_sha256: String,
    pub response_matrix_sha256: String,
    pub frozen_context_sha256: String,
    pub calibration_receipt_sha256: String,
    pub controls: Vec<VccaControlBoundAdapter>,
    pub metrics: Vec<VccaMetricResidualAdapter>,
    pub sensitivities: Vec<VccaSensitivityCellAdapter>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FittedSensitivity {
    pub metric_id: String,
    pub metric_unit: String,
    pub control_id: String,
    pub control_unit: String,
    pub effect_per_control_unit: f64,
    pub absolute_uncertainty_per_control_unit: f64,
    pub observed_pair_count: usize,
    pub minimum_calibration_input: f64,
    pub maximum_calibration_input: f64,
    pub maximum_absolute_fit_residual: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HoldoutPredictionCheck {
    pub source_case_id: String,
    pub control_id: String,
    pub control_delta: f64,
    pub metric_id: String,
    pub metric_unit: String,
    pub measured_delta: f64,
    pub predicted_delta: f64,
    pub uncertainty_radius: f64,
    pub absolute_prediction_error: f64,
    pub covered: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CalibrationSplitIdentity {
    pub source_case_id: String,
    pub parameter_sha256: String,
    pub baseline_pcm_sha256: String,
    pub intervention_pair_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ControlSupportCheck {
    pub control_id: String,
    pub target_metric_id: String,
    pub effect_per_control_unit: Option<f64>,
    pub expected_response_sign: String,
    pub supported: bool,
    pub disposition_reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SyntheticResponseProfile {
    pub schema_version: String,
    pub content_class: String,
    pub human_recordings_ingested: bool,
    pub external_corpus_assets_used: bool,
    pub voice_identity_targeting: bool,
    pub renderer_backend_id: String,
    pub renderer_class: String,
    pub analyzer_id: String,
    pub analyzer_version: String,
    pub source_revision: String,
    pub fit_method: String,
    pub sample_rate_hz: u32,
    pub target_segment_role: String,
    pub calibration_splits: Vec<CalibrationSplitIdentity>,
    pub calibration_pairs: Vec<CalibrationPairRecord>,
    pub confirmatory_source_case_id: String,
    pub confirmatory_parameter_sha256: String,
    pub no_op_calibration: NoOpCalibrationReport,
    pub fitted_sensitivities: Vec<FittedSensitivity>,
    pub holdout_checks: Vec<HoldoutPredictionCheck>,
    pub holdout_coverage_fraction: f64,
    pub all_required_cells_present: bool,
    pub control_support_checks: Vec<ControlSupportCheck>,
    pub all_controls_supported: bool,
    pub all_holdout_checks_covered: bool,
    pub exact_no_op_repeatability: bool,
    pub split_identities_disjoint: bool,
    pub promotable: bool,
    pub response_matrix_sha256: String,
    pub response_profile_sha256: String,
    pub calibration_receipt_sha256: String,
    pub vcca_adapter: VccaResponseProfileAdapter,
}

fn median_metric_map(
    records: &[crate::observer::SegmentObservationRecord],
    segment_id: &str,
) -> Result<BTreeMap<String, (String, f64)>, String> {
    let record = records
        .iter()
        .find(|record| record.segment_id == segment_id)
        .ok_or_else(|| format!("observer output missing target segment {segment_id}"))?;
    let mut map = BTreeMap::new();
    for metric in &record.metrics {
        if metric.disposition != crate::observer::ObservationDisposition::Observed {
            return Err(format!(
                "required metric {} is indeterminate on segment {}",
                metric.metric_id, segment_id
            ));
        }
        let value = metric.value.ok_or_else(|| {
            format!(
                "metric {} has observed disposition but no value",
                metric.metric_id
            )
        })?;
        if !value.is_finite() {
            return Err(format!("metric {} is non-finite", metric.metric_id));
        }
        if map
            .insert(metric.metric_id.clone(), (metric.unit.clone(), value))
            .is_some()
        {
            return Err(format!("duplicate metric observation {}", metric.metric_id));
        }
    }
    if map.is_empty() {
        return Err("observer returned no required metric observations".into());
    }
    Ok(map)
}

fn target_id_for_family(segments: &[SyntheticSegmentSpec]) -> Result<&str, String> {
    segments
        .iter()
        .find(|segment| segment.segment_id.ends_with("-target"))
        .or_else(|| segments.iter().find(|segment| segment.segment_id == "proxy-vowel-c"))
        .map(|segment| segment.segment_id.as_str())
        .ok_or_else(|| "calibration family does not declare a target segment".into())
}

fn family(
    prefix: &str,
    f0_scale: f32,
    formant_scale: f32,
    energy_scale: f32,
) -> Vec<SyntheticSegmentSpec> {
    let mut segments = default_segments();
    for (index, segment) in segments.iter_mut().enumerate() {
        let old_id = segment.segment_id.clone();
        segment.segment_id = if index == 2 {
            format!("{prefix}-target")
        } else {
            format!("{prefix}-{old_id}")
        };
        segment.f0_hz *= f0_scale;
        segment.f1_hz *= formant_scale;
        segment.f2_hz *= formant_scale;
        segment.f3_hz *= formant_scale;
        segment.energy = (segment.energy * energy_scale).clamp(0.0, 1.0);
    }
    segments
}

fn family_digest(segments: &[SyntheticSegmentSpec]) -> Result<String, String> {
    let bytes = serde_json::to_vec(segments)
        .map_err(|error| format!("failed to serialize calibration family: {error}"))?;
    Ok(sha256_bytes(&bytes))
}

fn probe_matrix() -> Vec<ControlProbe> {
    let mut probes = Vec::new();
    for cents in [-50.0_f32, -25.0, 25.0, 50.0] {
        probes.push(ControlProbe {
            control_id: "pitch_shift_cents".into(),
            control_unit: "cents".into(),
            input_delta: f64::from(cents),
            patch: EditPatch::PitchShiftCents {
                delta_cents: cents,
            },
        });
    }
    for delta in [-0.10_f32, -0.05, 0.05, 0.10] {
        probes.push(ControlProbe {
            control_id: "formant_scale_delta".into(),
            control_unit: "unitless_scale_delta".into(),
            input_delta: f64::from(delta),
            patch: EditPatch::FormantScale { scale: 1.0 + delta },
        });
    }
    for delta in [-0.20_f32, -0.10, 0.10, 0.20] {
        probes.push(ControlProbe {
            control_id: "energy_scale_delta".into(),
            control_unit: "unitless_scale_delta".into(),
            input_delta: f64::from(delta),
            patch: EditPatch::EnergyScale { scale: 1.0 + delta },
        });
    }
    probes
}

fn render_and_measure(
    source_case_id: &str,
    case_id: &str,
    segments: &[SyntheticSegmentSpec],
    sample_rate_hz: u32,
) -> Result<(crate::RenderedPhrase, BTreeMap<String, (String, f64)>), String> {
    let rendered = render_phrase(segments, sample_rate_hz, DEFAULT_CROSSFADE_MS)?;
    let target_id = target_id_for_family(segments)?;
    let observations =
        observe_phrase(source_case_id, case_id, &rendered, sample_rate_hz)?;
    let metrics = median_metric_map(&observations, target_id)?;
    Ok((rendered, metrics))
}

fn fit_through_origin(samples: &[(f64, f64)]) -> Result<(f64, f64, f64), String> {
    if samples.len() < MIN_CALIBRATION_PAIRS_PER_CELL {
        return Err(format!(
            "response cell requires at least {MIN_CALIBRATION_PAIRS_PER_CELL} pairs; got {}",
            samples.len()
        ));
    }
    if samples
        .iter()
        .any(|(input, output)| !input.is_finite() || !output.is_finite() || *input == 0.0)
    {
        return Err("response cell contains an invalid control/metric delta".into());
    }
    let denominator = samples.iter().map(|(input, _)| input * input).sum::<f64>();
    if !denominator.is_finite() || denominator <= f64::EPSILON {
        return Err("response fit has zero or non-finite control excitation".into());
    }
    let numerator = samples
        .iter()
        .map(|(input, output)| input * output)
        .sum::<f64>();
    let slope = numerator / denominator;
    if !slope.is_finite() {
        return Err("response fit produced a non-finite coefficient".into());
    }
    let mut max_abs_residual = 0.0_f64;
    let mut max_ratio = 0.0_f64;
    for (input, output) in samples {
        let residual = (output - slope * input).abs();
        max_abs_residual = max_abs_residual.max(residual);
        max_ratio = max_ratio.max(residual / input.abs());
    }
    if !max_abs_residual.is_finite() || !max_ratio.is_finite() {
        return Err("response fit produced a non-finite uncertainty bound".into());
    }
    Ok((slope, max_ratio, max_abs_residual))
}

fn control_bounds(probes: &[ControlProbe]) -> Result<Vec<VccaControlBoundAdapter>, String> {
    let mut grouped: BTreeMap<String, (String, f64, f64)> = BTreeMap::new();
    for probe in probes {
        let entry = grouped
            .entry(probe.control_id.clone())
            .or_insert((probe.control_unit.clone(), probe.input_delta, probe.input_delta));
        if entry.0 != probe.control_unit {
            return Err(format!("control {} has inconsistent units", probe.control_id));
        }
        entry.1 = entry.1.min(probe.input_delta);
        entry.2 = entry.2.max(probe.input_delta);
    }
    grouped
        .into_iter()
        .map(|(control_id, (unit, minimum_delta, maximum_delta))| {
            if minimum_delta >= 0.0 || maximum_delta <= 0.0 {
                return Err(format!(
                    "control {control_id} requires measured positive and negative interventions"
                ));
            }
            let trust_radius = minimum_delta.abs().max(maximum_delta.abs());
            Ok(VccaControlBoundAdapter {
                control_id,
                unit,
                supported: true,
                minimum_delta,
                maximum_delta,
                trust_radius,
                effort_scale: trust_radius,
                effort_weight: 1.0,
            })
        })
        .collect()
}

fn holdout_probe() -> Vec<ControlProbe> {
    vec![
        ControlProbe {
            control_id: "pitch_shift_cents".into(),
            control_unit: "cents".into(),
            input_delta: 35.0,
            patch: EditPatch::PitchShiftCents {
                delta_cents: 35.0,
            },
        },
        ControlProbe {
            control_id: "formant_scale_delta".into(),
            control_unit: "unitless_scale_delta".into(),
            input_delta: 0.07,
            patch: EditPatch::FormantScale { scale: 1.07 },
        },
        ControlProbe {
            control_id: "energy_scale_delta".into(),
            control_unit: "unitless_scale_delta".into(),
            input_delta: -0.15,
            patch: EditPatch::EnergyScale { scale: 0.85 },
        },
    ]
}

fn metric_repeatability_map(
    calibration: &NoOpCalibrationReport,
) -> Result<BTreeMap<String, (String, f64)>, String> {
    let mut map = BTreeMap::new();
    for metric in &calibration.metrics {
        if metric.disposition != crate::observer::ObservationDisposition::Observed {
            return Err(format!(
                "no-op calibration metric {} is indeterminate",
                metric.metric_id
            ));
        }
        let delta = metric.repeat_max_abs_delta.ok_or_else(|| {
            format!(
                "no-op calibration metric {} lacks a repeat delta",
                metric.metric_id
            )
        })?;
        if !delta.is_finite() || delta < 0.0 {
            return Err(format!(
                "no-op calibration metric {} has invalid repeat delta",
                metric.metric_id
            ));
        }
        match map.get_mut(&metric.metric_id) {
            Some((existing_unit, existing_delta)) => {
                if existing_unit.as_str() != metric.unit.as_str() {
                    return Err(format!(
                        "no-op calibration metric {} has inconsistent units",
                        metric.metric_id
                    ));
                }
                *existing_delta = (*existing_delta).max(delta);
            }
            None => {
                map.insert(metric.metric_id.clone(), (metric.unit.clone(), delta));
            }
        }
    }
    Ok(map)
}

fn vcca_matrix_digest(
    sensitivities: &[FittedSensitivity],
    controls: &[VccaControlBoundAdapter],
    metrics: &[VccaMetricResidualAdapter],
) -> Result<String, String> {
    let bytes = serde_json::to_vec(&(sensitivities, controls, metrics))
        .map_err(|error| format!("failed to serialize response matrix: {error}"))?;
    Ok(sha256_bytes(&bytes))
}

fn build_profile() -> Result<SyntheticResponseProfile, String> {
    let sample_rate_hz = DEFAULT_SAMPLE_RATE_HZ;
    let probes = probe_matrix();
    let mut samples_by_cell: BTreeMap<(String, String), Vec<(f64, f64)>> = BTreeMap::new();
    let mut units_by_metric = BTreeMap::new();
    let mut calibration_pairs = Vec::new();
    let mut calibration_splits = Vec::new();
    let calibration_families = [
        (
            "SYNTH_CAL_FAMILY_A",
            family("cal-a", 0.90, 0.95, 0.88),
        ),
        (
            "SYNTH_CAL_FAMILY_B",
            family("cal-b", 1.08, 1.05, 0.96),
        ),
    ];

    for (source_case_id, segments) in calibration_families.iter() {
        let parameter_sha = family_digest(segments)?;
        let target_id = target_id_for_family(segments)?.to_owned();
        let (baseline_rendered, baseline_metrics) = render_and_measure(
            source_case_id,
            "calibration_baseline",
            segments,
            sample_rate_hz,
        )?;
        for (metric_id, (unit, _)) in &baseline_metrics {
            if let Some(existing) = units_by_metric.insert(metric_id.clone(), unit.clone()) {
                if existing != unit.clone() {
                    return Err(format!("metric {metric_id} has inconsistent units"));
                }
            }
        }

        let mut family_pair_count = 0usize;
        for probe in &probes {
            let edited = apply_edit(
                segments,
                &target_id,
                &probe.patch,
                sample_rate_hz,
            )?;
            let (intervention_rendered, intervention_metrics) = render_and_measure(
                source_case_id,
                &format!("{}_{}", probe.control_id, probe.input_delta),
                &edited,
                sample_rate_hz,
            )?;
            if baseline_metrics.len() != intervention_metrics.len() {
                return Err("baseline/intervention metric coverage differs".into());
            }
            let mut metric_deltas = Vec::new();
            for (metric_id, (unit, baseline_value)) in &baseline_metrics {
                let (intervention_unit, intervention_value) = intervention_metrics
                    .get(metric_id)
                    .ok_or_else(|| format!("intervention missing metric {metric_id}"))?;
                if unit != intervention_unit {
                    return Err(format!("metric {metric_id} unit changed across intervention"));
                }
                let delta = intervention_value - baseline_value;
                if !delta.is_finite() {
                    return Err(format!("metric {metric_id} delta is non-finite"));
                }
                samples_by_cell
                    .entry((metric_id.clone(), probe.control_id.clone()))
                    .or_default()
                    .push((probe.input_delta, delta));
                metric_deltas.push(MeasuredMetricDelta {
                    metric_id: metric_id.clone(),
                    unit: unit.clone(),
                    baseline_value: *baseline_value,
                    intervention_value: *intervention_value,
                    delta,
                });
            }
            calibration_pairs.push(CalibrationPairRecord {
                source_case_id: (*source_case_id).into(),
                family_parameter_sha256: parameter_sha.clone(),
                control_id: probe.control_id.clone(),
                control_unit: probe.control_unit.clone(),
                input_delta: probe.input_delta,
                target_segment_id: target_id.clone(),
                baseline_pcm_sha256: sha256_pcm(&baseline_rendered.samples),
                intervention_pcm_sha256: sha256_pcm(&intervention_rendered.samples),
                metric_deltas,
            });
            family_pair_count += 1;
        }
        calibration_splits.push(CalibrationSplitIdentity {
            source_case_id: (*source_case_id).into(),
            parameter_sha256: parameter_sha,
            baseline_pcm_sha256: sha256_pcm(&baseline_rendered.samples),
            intervention_pair_count: family_pair_count,
        });
    }

    let confirmatory_segments = default_segments();
    let confirmatory_source_case_id = "SYNTH_CONTROL_CONFIRMATORY_V1";
    let confirmatory_parameter_sha256 = family_digest(&confirmatory_segments)?;
    let calibration_parameter_hashes: BTreeSet<String> = calibration_splits
        .iter()
        .map(|split| split.parameter_sha256.clone())
        .collect();
    if calibration_parameter_hashes.contains(&confirmatory_parameter_sha256) {
        return Err("confirmatory parameter family overlaps calibration split".into());
    }

    let no_op_segments = family("cal-noop", 0.99, 0.97, 0.92);
    let no_op_calibration = calibrate_no_op_repeatability(
        "SYNTH_CONTROL_NOOP_CALIBRATION_V1",
        confirmatory_source_case_id,
        &no_op_segments,
        sample_rate_hz,
    )?;
    let repeatability = metric_repeatability_map(&no_op_calibration)?;

    let mut fitted_sensitivities = Vec::new();
    for ((metric_id, control_id), samples) in &samples_by_cell {
        let metric_unit = units_by_metric
            .get(metric_id)
            .ok_or_else(|| format!("metric {metric_id} missing calibration unit"))?;
        let (repeat_unit, _) = repeatability
            .get(metric_id)
            .ok_or_else(|| format!("metric {metric_id} missing repeatability observation"))?;
        if metric_unit != repeat_unit {
            return Err(format!("metric {metric_id} calibration/repeatability units differ"));
        }
        let (slope, coefficient_uncertainty, max_residual) =
            fit_through_origin(samples)?;
        fitted_sensitivities.push(FittedSensitivity {
            metric_id: metric_id.clone(),
            metric_unit: metric_unit.clone(),
            control_id: control_id.clone(),
            control_unit: probes
                .iter()
                .find(|probe| probe.control_id == *control_id)
                .ok_or_else(|| format!("missing control metadata {control_id}"))?
                .control_unit
                .clone(),
            effect_per_control_unit: slope,
            absolute_uncertainty_per_control_unit: coefficient_uncertainty
                + f64::EPSILON * (1.0 + slope.abs()),
            observed_pair_count: samples.len(),
            minimum_calibration_input: samples
                .iter()
                .map(|(input, _)| *input)
                .fold(f64::INFINITY, f64::min),
            maximum_calibration_input: samples
                .iter()
                .map(|(input, _)| *input)
                .fold(f64::NEG_INFINITY, f64::max),
            maximum_absolute_fit_residual: max_residual,
        });
    }

    let expected_cell_count = units_by_metric.len()
        * probes
            .iter()
            .map(|probe| probe.control_id.as_str())
            .collect::<BTreeSet<_>>()
            .len();
    let all_required_cells_present = fitted_sensitivities.len() == expected_cell_count
        && fitted_sensitivities.iter().all(|cell| {
            cell.observed_pair_count >= MIN_CALIBRATION_PAIRS_PER_CELL
                && cell.effect_per_control_unit.is_finite()
                && cell.absolute_uncertainty_per_control_unit.is_finite()
        });
    if !all_required_cells_present {
        return Err("response matrix is incomplete or contains unqualified cells".into());
    }

    let holdout_baseline = confirmatory_segments.clone();
    let holdout_target_id = target_id_for_family(&holdout_baseline)?.to_owned();
    let (_, holdout_baseline_metrics) = render_and_measure(
        confirmatory_source_case_id,
        "confirmatory_baseline",
        &holdout_baseline,
        sample_rate_hz,
    )?;
    let mut holdout_checks = Vec::new();
    for probe in holdout_probe() {
        let edited = apply_edit(
            &holdout_baseline,
            &holdout_target_id,
            &probe.patch,
            sample_rate_hz,
        )?;
        let (_, measured_metrics) = render_and_measure(
            confirmatory_source_case_id,
            &format!("holdout_{}", probe.control_id),
            &edited,
            sample_rate_hz,
        )?;
        for (metric_id, (unit, baseline_value)) in &holdout_baseline_metrics {
            let (measured_unit, measured_value) = measured_metrics
                .get(metric_id)
                .ok_or_else(|| format!("holdout intervention missing metric {metric_id}"))?;
            if unit != measured_unit {
                return Err(format!("holdout metric {metric_id} changed units"));
            }
            let cell = fitted_sensitivities
                .iter()
                .find(|cell| {
                    cell.metric_id == *metric_id && cell.control_id == probe.control_id
                })
                .ok_or_else(|| format!("response profile missing cell {metric_id}/{}", probe.control_id))?;
            let residual_uncertainty = repeatability
                .get(metric_id)
                .filter(|(repeat_unit, _)| repeat_unit == unit)
                .map(|(_, value)| *value)
                .ok_or_else(|| format!("metric {metric_id} missing no-op uncertainty"))?;
            let predicted_delta = cell.effect_per_control_unit * probe.input_delta;
            let uncertainty_radius = residual_uncertainty
                + cell.absolute_uncertainty_per_control_unit * probe.input_delta.abs();
            let actual_delta = measured_value - baseline_value;
            let error = (actual_delta - predicted_delta).abs();
            let covered = error <= uncertainty_radius + HOLDOUT_NUMERIC_EPSILON;
            holdout_checks.push(HoldoutPredictionCheck {
                source_case_id: confirmatory_source_case_id.into(),
                control_id: probe.control_id.clone(),
                control_delta: probe.input_delta,
                metric_id: metric_id.clone(),
                metric_unit: unit.clone(),
                measured_delta: actual_delta,
                predicted_delta,
                uncertainty_radius,
                absolute_prediction_error: error,
                covered,
            });
        }
    }

    let holdout_coverage_fraction = if holdout_checks.is_empty() {
        0.0
    } else {
        holdout_checks.iter().filter(|check| check.covered).count() as f64
            / holdout_checks.len() as f64
    };
    let all_holdout_checks_covered = !holdout_checks.is_empty()
        && holdout_checks.iter().all(|check| check.covered);

    let mut bounds = control_bounds(&probes)?;
    let required_target_metrics = [
        ("pitch_shift_cents", "pitch.f0.median.autocorrelation"),
        ("formant_scale_delta", "formant.f1.median"),
        ("energy_scale_delta", "vocal.energy.median"),
    ];
    let mut control_support_checks = Vec::new();
    for (control_id, target_metric_id) in required_target_metrics {
        let effect = fitted_sensitivities
            .iter()
            .find(|cell| {
                cell.control_id == control_id && cell.metric_id == target_metric_id
            })
            .map(|cell| cell.effect_per_control_unit);
        let supported = effect.is_some_and(|value| value.is_finite() && value > 1e-6);
        control_support_checks.push(ControlSupportCheck {
            control_id: control_id.into(),
            target_metric_id: target_metric_id.into(),
            effect_per_control_unit: effect,
            expected_response_sign: "positive".into(),
            supported,
            disposition_reason: if supported {
                "measured_positive_target_response".into()
            } else {
                "requested_control_has_no_qualified_positive_target_response".into()
            },
        });
        if !supported {
            if let Some(bound) = bounds.iter_mut().find(|bound| bound.control_id == control_id) {
                bound.supported = false;
                bound.minimum_delta = 0.0;
                bound.maximum_delta = 0.0;
                bound.trust_radius = 0.0;
                bound.effort_scale = 1.0;
            }
        }
    }
    let all_controls_supported = control_support_checks.iter().all(|check| check.supported);

    let vcca_cells: Vec<VccaSensitivityCellAdapter> = fitted_sensitivities
        .iter()
        .map(|cell| VccaSensitivityCellAdapter {
            metric_id: cell.metric_id.clone(),
            metric_unit: cell.metric_unit.clone(),
            control_id: cell.control_id.clone(),
            control_unit: cell.control_unit.clone(),
            effect_per_control_unit: cell.effect_per_control_unit,
            absolute_uncertainty_per_control_unit: cell.absolute_uncertainty_per_control_unit,
        })
        .collect();
    let vcca_metrics: Vec<VccaMetricResidualAdapter> = units_by_metric
        .iter()
        .map(|(metric_id, unit)| {
            let residual = repeatability
                .get(metric_id)
                .filter(|(repeat_unit, _)| repeat_unit == unit)
                .map(|(_, value)| *value)
                .ok_or_else(|| format!("missing repeatability for metric {metric_id}"))?;
            Ok(VccaMetricResidualAdapter {
                metric_id: metric_id.clone(),
                unit: unit.clone(),
                residual_uncertainty: residual,
                repeatability_receipt_id: no_op_calibration.calibration_id.clone(),
            })
        })
        .collect::<Result<_, String>>()?;

    let calibration_receipt_bytes = serde_json::to_vec(&(
        &calibration_splits,
        &calibration_pairs,
        &no_op_calibration,
        &fitted_sensitivities,
        &control_support_checks,
        &holdout_checks,
    ))
    .map_err(|error| format!("failed to serialize calibration receipt: {error}"))?;
    let calibration_receipt_sha256 = sha256_bytes(&calibration_receipt_bytes);
    let frozen_context_sha256 = sha256_bytes(
        format!(
            "{}:{}:{}:{}:{}:{}:{}",
            RENDERER_BACKEND_ID,
            LPC_ANALYZER_ID,
            LPC_ANALYZER_VERSION,
            AUTOCORRELATION_ANALYZER_ID,
            AUTOCORRELATION_ANALYZER_VERSION,
            RESPONSE_FIT_METHOD,
            sample_rate_hz
        )
        .as_bytes(),
    );
    let response_matrix_sha256 =
        vcca_matrix_digest(&vcca_cells, &bounds, &vcca_metrics)?;
    let split_identities_disjoint = calibration_splits.iter().all(|split| {
        split.source_case_id != confirmatory_source_case_id
            && split.parameter_sha256 != confirmatory_parameter_sha256
    }) && no_op_calibration.split_identity_disjoint;
    let exact_no_op_repeatability = no_op_calibration.exact_pcm_repeatability;
    let promotable = all_required_cells_present
        && all_controls_supported
        && all_holdout_checks_covered
        && exact_no_op_repeatability
        && split_identities_disjoint;

    let mut profile = SyntheticResponseProfile {
        schema_version: RESPONSE_PROFILE_SCHEMA.into(),
        content_class: "synthetic_only".into(),
        human_recordings_ingested: false,
        external_corpus_assets_used: false,
        voice_identity_targeting: false,
        renderer_backend_id: RENDERER_BACKEND_ID.into(),
        renderer_class: "legacy_formant_source_filter".into(),
        analyzer_id: LPC_ANALYZER_ID.into(),
        analyzer_version: LPC_ANALYZER_VERSION.into(),
        source_revision: std::env::var("SYNTH_VOCAL_PLANT_SOURCE_REVISION")
            .unwrap_or_else(|_| "unknown-at-build".into()),
        fit_method: RESPONSE_FIT_METHOD.into(),
        sample_rate_hz,
        target_segment_role: "synthetic_proxy_event".into(),
        calibration_splits,
        calibration_pairs,
        confirmatory_source_case_id: confirmatory_source_case_id.into(),
        confirmatory_parameter_sha256,
        no_op_calibration,
        fitted_sensitivities,
        holdout_checks,
        holdout_coverage_fraction,
        all_required_cells_present,
        control_support_checks,
        all_controls_supported,
        all_holdout_checks_covered,
        exact_no_op_repeatability,
        split_identities_disjoint,
        promotable,
        response_matrix_sha256,
        response_profile_sha256: String::new(),
        calibration_receipt_sha256,
        vcca_adapter: VccaResponseProfileAdapter {
            response_profile_sha256: String::new(),
            response_matrix_sha256: String::new(),
            frozen_context_sha256,
            calibration_receipt_sha256: String::new(),
            controls: bounds,
            metrics: vcca_metrics,
            sensitivities: vcca_cells,
        },
    };

    let profile_core = serde_json::to_vec(&(
        &profile.schema_version,
        &profile.content_class,
        &profile.renderer_backend_id,
        &profile.analyzer_id,
        &profile.analyzer_version,
        &profile.source_revision,
        &profile.calibration_splits,
        &profile.calibration_pairs,
        &profile.confirmatory_source_case_id,
        &profile.confirmatory_parameter_sha256,
        &profile.fitted_sensitivities,
        &profile.control_support_checks,
        &profile.holdout_checks,
        &profile.response_matrix_sha256,
        &profile.calibration_receipt_sha256,
    ))
    .map_err(|error| format!("failed to serialize response profile core: {error}"))?;
    profile.response_profile_sha256 = sha256_bytes(&profile_core);
    profile.vcca_adapter.response_profile_sha256 = profile.response_profile_sha256.clone();
    profile.vcca_adapter.response_matrix_sha256 = profile.response_matrix_sha256.clone();
    profile.vcca_adapter.calibration_receipt_sha256 =
        profile.calibration_receipt_sha256.clone();
    Ok(profile)
}

pub fn generate_response_profile(output_dir: &Path) -> Result<SyntheticResponseProfile, String> {
    std::fs::create_dir_all(output_dir)
        .map_err(|error| format!("failed to create response profile directory: {error}"))?;
    let profile = build_profile()?;
    let profile_bytes = serde_json::to_vec_pretty(&profile)
        .map_err(|error| format!("failed to serialize response profile: {error}"))?;
    std::fs::write(output_dir.join("response-profile.json"), &profile_bytes)
        .map_err(|error| format!("failed to write response profile: {error}"))?;
    std::fs::write(
        output_dir.join("response-profile.sha256"),
        format!("{}  response-profile.json\n", sha256_bytes(&profile_bytes)),
    )
    .map_err(|error| format!("failed to write profile digest: {error}"))?;

    let pair_bytes = serde_json::to_vec_pretty(&profile.calibration_pairs)
        .map_err(|error| format!("failed to serialize calibration pairs: {error}"))?;
    std::fs::write(output_dir.join("response-calibration-pairs.json"), &pair_bytes)
        .map_err(|error| format!("failed to write calibration pairs: {error}"))?;
    std::fs::write(
        output_dir.join("response-calibration-pairs.sha256"),
        format!(
            "{}  response-calibration-pairs.json\n",
            sha256_bytes(&pair_bytes)
        ),
    )
    .map_err(|error| format!("failed to write calibration pair digest: {error}"))?;

    let adapter_bytes = serde_json::to_vec_pretty(&profile.vcca_adapter)
        .map_err(|error| format!("failed to serialize VCCA adapter: {error}"))?;
    std::fs::write(output_dir.join("vcca-response-profile.json"), &adapter_bytes)
        .map_err(|error| format!("failed to write VCCA adapter: {error}"))?;
    Ok(profile)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn through_origin_fit_recovers_linear_gain_and_uncertainty() {
        let samples = vec![
            (-2.0, -6.0),
            (-1.0, -3.0),
            (1.0, 3.0),
            (2.0, 6.0),
            (-3.0, -9.0),
            (3.0, 9.0),
            (-4.0, -12.0),
            (4.0, 12.0),
        ];
        let (slope, uncertainty, residual) =
            fit_through_origin(&samples).expect("linear response");
        assert!((slope - 3.0).abs() < 1e-12);
        assert!(uncertainty < 1e-12);
        assert!(residual < 1e-12);
    }

    #[test]
    fn fit_rejects_zero_or_insufficient_control_excitation() {
        assert!(fit_through_origin(&[(1.0, 2.0); 3]).is_err());
        assert!(fit_through_origin(&[(0.0, 0.0); 8]).is_err());
    }

    #[test]
    fn calibration_probe_matrix_contains_positive_and_negative_controls() {
        let probes = probe_matrix();
        let controls = control_bounds(&probes).expect("symmetric control ranges");
        assert_eq!(controls.len(), 3);
        assert!(controls
            .iter()
            .all(|control| control.minimum_delta < 0.0 && control.maximum_delta > 0.0));
    }

    #[test]
    fn family_parameter_variants_are_distinct_and_have_stable_target_ids() {
        let left = family("family-a", 0.90, 0.95, 0.88);
        let right = family("family-b", 1.08, 1.05, 0.96);
        assert_ne!(
            family_digest(&left).expect("left digest"),
            family_digest(&right).expect("right digest")
        );
        assert_ne!(
            target_id_for_family(&left).expect("left target"),
            target_id_for_family(&right).expect("right target")
        );
    }

    #[test]
    fn profile_builder_requires_independent_calibration_and_holdout() {
        let profile = build_profile().expect("response profile generation");
        assert!(profile.all_required_cells_present);
        assert!(profile.split_identities_disjoint);
        assert!(profile.exact_no_op_repeatability);
        assert!(!profile.fitted_sensitivities.is_empty());
        assert!(profile.holdout_checks.len() >= 3);
        // Held-out coverage is reported exactly; no success is fabricated.
        assert_eq!(
            profile.all_holdout_checks_covered,
            profile
                .holdout_checks
                .iter()
                .all(|check| check.covered)
        );
        assert_eq!(
            profile.promotable,
            profile.all_required_cells_present
                && profile.all_controls_supported
                && profile.all_holdout_checks_covered
                && profile.exact_no_op_repeatability
                && profile.split_identities_disjoint
        );
    }

    #[test]
    fn vcca_adapter_is_complete_and_uses_calibrated_trust_radii() {
        let profile = build_profile().expect("response profile");
        let adapter = profile.vcca_adapter;
        assert!(!adapter.response_profile_sha256.is_empty());
        assert!(!adapter.response_matrix_sha256.is_empty());
        assert!(!adapter.calibration_receipt_sha256.is_empty());
        assert_eq!(adapter.controls.len(), 3);
        assert!(adapter.controls.iter().all(|control| {
            control.effort_scale > 0.0
                && (control.supported == (control.trust_radius > 0.0))
                && (control.supported
                    || (control.minimum_delta == 0.0 && control.maximum_delta == 0.0))
        }));
        assert_eq!(
            profile.all_controls_supported,
            profile.control_support_checks.iter().all(|check| check.supported)
        );
        assert!(!adapter.metrics.is_empty());
        assert_eq!(
            adapter.sensitivities.len(),
            adapter.controls.len() * adapter.metrics.len()
        );
    }
}
