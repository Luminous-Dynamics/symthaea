// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic, report-only PLSS telemetry trend analysis (SX-026).
//!
//! Trends are computed separately for each metric, provenance class, and source.
//! They are not safety limits, calibrated confidence intervals, inspired-CO2
//! estimates, alarms, or commands to physical life-support hardware.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::plss_telemetry::{
    PlssTelemetryMetric, PlssTelemetryObservation, PlssTelemetrySourceKind,
    PlssTelemetryTolerances, PlssTelemetryUnit, PLSS_TELEMETRY_METRIC_COUNT,
};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlssTelemetryTrendConfig {
    pub window_start_s: f64,
    pub window_end_s: f64,
    pub maximum_gap_s: f64,
    pub minimum_samples: usize,
    pub minimum_confidence: f64,
    pub minimum_coverage_fraction: f64,
    /// Used only to flag disagreement between separate sources' final readings.
    /// These are software tolerances, not human atmosphere limits.
    pub cross_source_tolerances: PlssTelemetryTolerances,
}

impl PlssTelemetryTrendConfig {
    pub fn simulation_reference() -> Self {
        Self {
            window_start_s: 0.0,
            window_end_s: 60.0,
            maximum_gap_s: 30.0,
            minimum_samples: 3,
            minimum_confidence: 0.70,
            minimum_coverage_fraction: 0.50,
            cross_source_tolerances: PlssTelemetryTolerances {
                co2_partial_pressure_pa: 250.0,
                oxygen_partial_pressure_pa: 1_000.0,
                circuit_pressure_pa: 2_000.0,
                gas_temperature_k: 2.0,
                ventilation_flow_l_min: 15.0,
                relative_humidity_percent: 5.0,
            },
        }
    }

    pub fn is_valid(&self) -> bool {
        self.window_start_s.is_finite()
            && self.window_start_s >= 0.0
            && self.window_end_s.is_finite()
            && self.window_end_s > self.window_start_s
            && self.maximum_gap_s.is_finite()
            && self.maximum_gap_s > 0.0
            && self.minimum_samples >= 2
            && self.minimum_samples <= 1_000_000
            && self.minimum_confidence.is_finite()
            && (0.0..=1.0).contains(&self.minimum_confidence)
            && self.minimum_coverage_fraction.is_finite()
            && (0.0..=1.0).contains(&self.minimum_coverage_fraction)
            && [
                self.cross_source_tolerances.co2_partial_pressure_pa,
                self.cross_source_tolerances.oxygen_partial_pressure_pa,
                self.cross_source_tolerances.circuit_pressure_pa,
                self.cross_source_tolerances.gas_temperature_k,
                self.cross_source_tolerances.ventilation_flow_l_min,
                self.cross_source_tolerances.relative_humidity_percent,
            ]
            .into_iter()
            .all(|value| value.is_finite() && value >= 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlssTelemetryTrendStatus {
    Fresh,
    Conflicted,
    InsufficientEvidence,
    Missing,
    LowConfidence,
    Invalid,
    FutureDated,
    OutOfWindow,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlssTelemetrySourceTrend {
    pub source_id: String,
    pub status: PlssTelemetryTrendStatus,
    pub unit: PlssTelemetryUnit,
    /// Number of valid unique timestamps after identical timestamp duplicates are collapsed.
    pub sample_count: usize,
    pub first_timestamp_s: Option<f64>,
    pub last_timestamp_s: Option<f64>,
    pub first_value: Option<f64>,
    pub last_value: Option<f64>,
    pub minimum_observed_value: Option<f64>,
    pub maximum_observed_value: Option<f64>,
    /// Largest absolute deviation from the straight line joining the two endpoints.
    /// This is a descriptive excursion metric, not an alarm threshold.
    pub maximum_deviation_from_endpoint_secant: Option<f64>,
    pub first_uncertainty_abs: Option<f64>,
    pub last_uncertainty_abs: Option<f64>,
    pub maximum_observed_gap_s: Option<f64>,
    /// Observed first-to-last span divided by the requested window duration.
    /// This is a coverage proxy, not a statement that the sensor ran continuously.
    pub temporal_coverage_fraction: Option<f64>,
    pub change_over_observed_span: Option<f64>,
    /// Endpoint secant rate per second, not a fitted regression slope.
    pub rate_per_s: Option<f64>,
    /// Bounds obtained from endpoint absolute uncertainty envelopes; not statistical CI.
    pub rate_lower_bound_per_s: Option<f64>,
    pub rate_upper_bound_per_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlssTelemetryTrendLane {
    pub metric: PlssTelemetryMetric,
    pub source_kind: PlssTelemetrySourceKind,
    pub status: PlssTelemetryTrendStatus,
    pub source_trends: Vec<PlssTelemetrySourceTrend>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlssTelemetryTrendReport {
    pub evidence_level: TrendEvidenceLevel,
    pub config: PlssTelemetryTrendConfig,
    /// Fixed metric/provenance order; sources inside each lane are sorted by source_id.
    pub lanes: Vec<PlssTelemetryTrendLane>,
    pub input_observation_count: usize,
    pub accepted_observation_count: usize,
    pub rejected_invalid_observations: usize,
    pub rejected_future_dated_observations: usize,
    pub excluded_out_of_window_observations: usize,
    pub excluded_low_confidence_observations: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendEvidenceLevel {
    SimulationOnly,
}

impl PlssTelemetryTrendReport {
    pub fn lane(
        &self,
        metric: PlssTelemetryMetric,
        source_kind: PlssTelemetrySourceKind,
    ) -> Option<&PlssTelemetryTrendLane> {
        self.lanes
            .iter()
            .find(|lane| lane.metric == metric && lane.source_kind == source_kind)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlssTelemetryTrendReceipt {
    pub source_revision: String,
    pub input_observation_count: usize,
    pub accepted_observation_count: usize,
    pub rejected_invalid_observations: usize,
    pub rejected_future_dated_observations: usize,
    pub excluded_out_of_window_observations: usize,
    pub excluded_low_confidence_observations: usize,
    pub input_digest_sha256: String,
    pub result_digest_sha256: String,
    /// Binds the revision and both fingerprints to all counts; not a signature.
    pub receipt_digest_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlssTelemetryTrendError {
    InvalidConfig,
    InvalidSourceRevision,
}

pub fn analyze_plss_telemetry_trends(
    observations: &[PlssTelemetryObservation],
    config: PlssTelemetryTrendConfig,
) -> Result<PlssTelemetryTrendReport, PlssTelemetryTrendError> {
    if !config.is_valid() {
        return Err(PlssTelemetryTrendError::InvalidConfig);
    }

    let mut rejected_invalid = 0;
    let mut rejected_future = 0;
    let mut excluded_window = 0;
    let mut excluded_low_confidence = 0;
    let mut accepted_count = 0;
    let lane_capacity = PLSS_TELEMETRY_METRIC_COUNT * PlssTelemetrySourceKind::ALL.len();
    let mut lanes = Vec::with_capacity(lane_capacity);

    for metric in PlssTelemetryMetric::ALL {
        for source_kind in PlssTelemetrySourceKind::ALL {
            let mut by_source: Vec<(String, Vec<PlssTelemetryObservation>)> = Vec::new();
            let mut saw_any = false;
            let mut saw_invalid = false;
            let mut saw_future = false;
            let mut saw_outside = false;
            let mut saw_low_confidence = false;

            for observation in observations
                .iter()
                .filter(|observation| {
                    observation.metric == metric && observation.source_kind == source_kind
                })
            {
                saw_any = true;
                if !observation.is_valid() {
                    rejected_invalid += 1;
                    saw_invalid = true;
                    continue;
                }
                if observation.observed_at_s > config.window_end_s {
                    rejected_future += 1;
                    saw_future = true;
                    continue;
                }
                if observation.observed_at_s < config.window_start_s {
                    excluded_window += 1;
                    saw_outside = true;
                    continue;
                }
                if observation.confidence < config.minimum_confidence
                    || observation.confidence <= 0.0
                {
                    excluded_low_confidence += 1;
                    saw_low_confidence = true;
                    continue;
                }
                accepted_count += 1;
                if let Some((_, samples)) = by_source
                    .iter_mut()
                    .find(|(id, _)| id == &observation.source_id)
                {
                    samples.push(observation.clone());
                } else {
                    by_source.push((observation.source_id.clone(), vec![observation.clone()]));
                }
            }

            by_source.sort_by(|left, right| left.0.cmp(&right.0));
            let mut source_trends = Vec::with_capacity(by_source.len());
            for (source_id, mut samples) in by_source {
                samples.sort_by(|left, right| left.observed_at_s.total_cmp(&right.observed_at_s));
                source_trends.push(analyze_source_trend(source_id, samples, config));
            }

            let status = if source_trends.is_empty() {
                match (saw_invalid, saw_future, saw_outside, saw_low_confidence, saw_any) {
                    (true, false, false, false, true) => PlssTelemetryTrendStatus::Invalid,
                    (false, true, false, false, true) => PlssTelemetryTrendStatus::FutureDated,
                    (false, false, true, false, true) => PlssTelemetryTrendStatus::OutOfWindow,
                    (false, false, false, true, true) => PlssTelemetryTrendStatus::LowConfidence,
                    (false, false, false, false, false) => PlssTelemetryTrendStatus::Missing,
                    _ => PlssTelemetryTrendStatus::InsufficientEvidence,
                }
            } else if source_trends
                .iter()
                .any(|trend| trend.status == PlssTelemetryTrendStatus::Conflicted)
                || sources_disagree(&source_trends, metric, config.cross_source_tolerances)
            {
                PlssTelemetryTrendStatus::Conflicted
            } else if source_trends
                .iter()
                .all(|trend| trend.status == PlssTelemetryTrendStatus::Fresh)
            {
                PlssTelemetryTrendStatus::Fresh
            } else {
                PlssTelemetryTrendStatus::InsufficientEvidence
            };

            lanes.push(PlssTelemetryTrendLane { metric, source_kind, status, source_trends });
        }
    }

    Ok(PlssTelemetryTrendReport {
        evidence_level: TrendEvidenceLevel::SimulationOnly,
        config,
        lanes,
        input_observation_count: observations.len(),
        accepted_observation_count: accepted_count,
        rejected_invalid_observations: rejected_invalid,
        rejected_future_dated_observations: rejected_future,
        excluded_out_of_window_observations: excluded_window,
        excluded_low_confidence_observations: excluded_low_confidence,
    })
}

fn analyze_source_trend(
    source_id: String,
    samples: Vec<PlssTelemetryObservation>,
    config: PlssTelemetryTrendConfig,
) -> PlssTelemetrySourceTrend {
    let metric = samples[0].metric;
    let unit = metric.unit();
    let mut unique: Vec<PlssTelemetryObservation> = Vec::with_capacity(samples.len());
    let mut duplicate_conflict = false;
    for sample in samples {
        if let Some(previous) = unique.last() {
            if sample.observed_at_s == previous.observed_at_s {
                if same_sample(previous, &sample) {
                    continue;
                }
                duplicate_conflict = true;
                continue;
            }
        }
        unique.push(sample);
    }

    let empty = |status, sample_count| PlssTelemetrySourceTrend {
        source_id: source_id.clone(),
        status,
        unit,
        sample_count,
        first_timestamp_s: unique.first().map(|sample| sample.observed_at_s),
        last_timestamp_s: unique.last().map(|sample| sample.observed_at_s),
        first_value: unique.first().map(|sample| sample.value),
        last_value: unique.last().map(|sample| sample.value),
        minimum_observed_value: unique.iter().map(|sample| sample.value).min_by(f64::total_cmp),
        maximum_observed_value: unique.iter().map(|sample| sample.value).max_by(f64::total_cmp),
        maximum_deviation_from_endpoint_secant: None,
        first_uncertainty_abs: unique.first().map(|sample| sample.uncertainty_abs),
        last_uncertainty_abs: unique.last().map(|sample| sample.uncertainty_abs),
        maximum_observed_gap_s: None,
        temporal_coverage_fraction: None,
        change_over_observed_span: None,
        rate_per_s: None,
        rate_lower_bound_per_s: None,
        rate_upper_bound_per_s: None,
    };

    if unique.len() < 2 {
        let status = if duplicate_conflict {
            PlssTelemetryTrendStatus::Conflicted
        } else {
            PlssTelemetryTrendStatus::InsufficientEvidence
        };
        return empty(status, unique.len());
    }

    let first = &unique[0];
    let last = unique.last().expect("at least two unique samples checked above");
    let elapsed = last.observed_at_s - first.observed_at_s;
    if !elapsed.is_finite() || elapsed <= 0.0 {
        return empty(PlssTelemetryTrendStatus::InsufficientEvidence, unique.len());
    }
    let window_duration = config.window_end_s - config.window_start_s;
    let coverage = elapsed / window_duration;
    let mut maximum_gap: f64 = 0.0;
    for pair in unique.windows(2) {
        maximum_gap = maximum_gap.max(pair[1].observed_at_s - pair[0].observed_at_s);
    }

    let change = last.value - first.value;
    let rate = change / elapsed;
    if !change.is_finite() || !rate.is_finite() {
        return empty(PlssTelemetryTrendStatus::Invalid, unique.len());
    }
    let mut maximum_deviation: f64 = 0.0;
    for sample in &unique {
        let endpoint_secant_value =
            first.value + rate * (sample.observed_at_s - first.observed_at_s);
        let deviation = (sample.value - endpoint_secant_value).abs();
        if !endpoint_secant_value.is_finite() || !deviation.is_finite() {
            return empty(PlssTelemetryTrendStatus::Invalid, unique.len());
        }
        maximum_deviation = maximum_deviation.max(deviation);
    }

    if duplicate_conflict
        || unique.len() < config.minimum_samples
        || !coverage.is_finite()
        || coverage < config.minimum_coverage_fraction
        || maximum_gap > config.maximum_gap_s
    {
        let status = if duplicate_conflict {
            PlssTelemetryTrendStatus::Conflicted
        } else {
            PlssTelemetryTrendStatus::InsufficientEvidence
        };
        let mut result = empty(status, unique.len());
        result.maximum_observed_gap_s = Some(maximum_gap);
        result.temporal_coverage_fraction = Some(coverage);
        result.maximum_deviation_from_endpoint_secant = Some(maximum_deviation);
        return result;
    }

    let rate_uncertainty = (first.uncertainty_abs + last.uncertainty_abs) / elapsed;
    let lower = (change - first.uncertainty_abs - last.uncertainty_abs) / elapsed;
    let upper = (change + first.uncertainty_abs + last.uncertainty_abs) / elapsed;
    if !rate_uncertainty.is_finite() || !lower.is_finite() || !upper.is_finite() {
        return empty(PlssTelemetryTrendStatus::Invalid, unique.len());
    }

    PlssTelemetrySourceTrend {
        source_id,
        status: PlssTelemetryTrendStatus::Fresh,
        unit,
        sample_count: unique.len(),
        first_timestamp_s: Some(first.observed_at_s),
        last_timestamp_s: Some(last.observed_at_s),
        first_value: Some(first.value),
        last_value: Some(last.value),
        minimum_observed_value: unique.iter().map(|sample| sample.value).min_by(f64::total_cmp),
        maximum_observed_value: unique.iter().map(|sample| sample.value).max_by(f64::total_cmp),
        maximum_deviation_from_endpoint_secant: Some(maximum_deviation),
        first_uncertainty_abs: Some(first.uncertainty_abs),
        last_uncertainty_abs: Some(last.uncertainty_abs),
        maximum_observed_gap_s: Some(maximum_gap),
        temporal_coverage_fraction: Some(coverage),
        change_over_observed_span: Some(change),
        rate_per_s: Some(rate),
        rate_lower_bound_per_s: Some(lower),
        rate_upper_bound_per_s: Some(upper),
    }
}

fn sources_disagree(
    trends: &[PlssTelemetrySourceTrend],
    metric: PlssTelemetryMetric,
    tolerances: PlssTelemetryTolerances,
) -> bool {
    let eligible: Vec<&PlssTelemetrySourceTrend> = trends
        .iter()
        .filter(|trend| trend.status == PlssTelemetryTrendStatus::Fresh)
        .collect();
    let tolerance = match metric {
        PlssTelemetryMetric::CircuitCo2PartialPressurePa => tolerances.co2_partial_pressure_pa,
        PlssTelemetryMetric::OxygenPartialPressurePa => tolerances.oxygen_partial_pressure_pa,
        PlssTelemetryMetric::CircuitPressurePa => tolerances.circuit_pressure_pa,
        PlssTelemetryMetric::GasTemperatureK => tolerances.gas_temperature_k,
        PlssTelemetryMetric::VentilationFlowLMin => tolerances.ventilation_flow_l_min,
        PlssTelemetryMetric::RelativeHumidityPercent => tolerances.relative_humidity_percent,
    };
    eligible.iter().enumerate().any(|(index, left)| {
        eligible.iter().skip(index + 1).any(|right| {
            (left.last_value.unwrap_or(0.0) - right.last_value.unwrap_or(0.0)).abs()
                > tolerance
                    + left.last_uncertainty_abs.unwrap_or(0.0)
                    + right.last_uncertainty_abs.unwrap_or(0.0)
        })
    })
}

fn same_sample(left: &PlssTelemetryObservation, right: &PlssTelemetryObservation) -> bool {
    left.metric == right.metric
        && left.unit == right.unit
        && left.value == right.value
        && left.uncertainty_abs == right.uncertainty_abs
        && left.observed_at_s == right.observed_at_s
        && left.confidence == right.confidence
        && left.source_id == right.source_id
        && left.source_kind == right.source_kind
}

pub fn analyze_plss_telemetry_trends_with_receipt(
    observations: &[PlssTelemetryObservation],
    config: PlssTelemetryTrendConfig,
    source_revision: &str,
) -> Result<(PlssTelemetryTrendReport, PlssTelemetryTrendReceipt), PlssTelemetryTrendError> {
    let valid = (source_revision.len() == 40 || source_revision.len() == 64)
        && source_revision.bytes().all(|byte| byte.is_ascii_hexdigit());
    if !valid {
        return Err(PlssTelemetryTrendError::InvalidSourceRevision);
    }
    let report = analyze_plss_telemetry_trends(observations, config)?;
    let mut receipt = PlssTelemetryTrendReceipt {
        source_revision: source_revision.to_ascii_lowercase(),
        input_observation_count: report.input_observation_count,
        accepted_observation_count: report.accepted_observation_count,
        rejected_invalid_observations: report.rejected_invalid_observations,
        rejected_future_dated_observations: report.rejected_future_dated_observations,
        excluded_out_of_window_observations: report.excluded_out_of_window_observations,
        excluded_low_confidence_observations: report.excluded_low_confidence_observations,
        input_digest_sha256: digest_trend_input(observations),
        result_digest_sha256: digest_trend_report(&report),
        receipt_digest_sha256: String::new(),
    };
    receipt.receipt_digest_sha256 = digest_trend_receipt(&receipt);
    Ok((report, receipt))
}

struct HashWriter(Sha256);

impl HashWriter {
    fn new(domain: &str) -> Self {
        let mut writer = Self(Sha256::new());
        writer.string(domain);
        writer
    }
    fn raw(&mut self, bytes: &[u8]) { self.0.update(bytes); }
    fn u64(&mut self, value: u64) { self.raw(&value.to_le_bytes()); }
    fn usize(&mut self, value: usize) { self.u64(value as u64); }
    fn f64(&mut self, value: f64) { self.u64(value.to_bits()); }
    fn string(&mut self, value: &str) { self.usize(value.len()); self.raw(value.as_bytes()); }
    fn option_f64(&mut self, value: Option<f64>) {
        match value {
            Some(number) => {
                self.raw(&[1]);
                self.f64(number);
            }
            None => self.raw(&[0]),
        }
    }
    fn finish(self) -> String {
        let digest = self.0.finalize();
        format!("sha256:{}", digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>())
    }
}

fn digest_trend_input(observations: &[PlssTelemetryObservation]) -> String {
    let mut sorted = observations.to_vec();
    sorted.sort_by(|a,b| a.metric.cmp(&b.metric)
        .then(a.source_kind.cmp(&b.source_kind))
        .then(a.source_id.cmp(&b.source_id))
        .then(a.observed_at_s.total_cmp(&b.observed_at_s))
        .then(a.unit.cmp(&b.unit))
        .then(a.value.total_cmp(&b.value))
        .then(a.uncertainty_abs.total_cmp(&b.uncertainty_abs))
        .then(a.confidence.total_cmp(&b.confidence)));
    let mut h = HashWriter::new("symthaea.plss.telemetry.trend.input.v1");
    h.usize(sorted.len());
    for item in &sorted {
        h.string(&format!("{:?}", item.metric));
        h.string(&format!("{:?}", item.unit));
        h.string(&format!("{:?}", item.source_kind));
        h.string(&item.source_id);
        h.f64(item.value);
        h.f64(item.uncertainty_abs);
        h.f64(item.observed_at_s);
        h.f64(item.confidence);
    }
    h.finish()
}

fn digest_trend_report(report: &PlssTelemetryTrendReport) -> String {
    let mut h = HashWriter::new("symthaea.plss.telemetry.trend.result.v1");
    h.string(&format!("{:?}", report.evidence_level));
    h.usize(report.lanes.len());
    h.f64(report.config.window_start_s);
    h.f64(report.config.window_end_s);
    h.f64(report.config.maximum_gap_s);
    h.usize(report.config.minimum_samples);
    h.f64(report.config.minimum_confidence);
    h.f64(report.config.minimum_coverage_fraction);
    h.f64(report.config.cross_source_tolerances.co2_partial_pressure_pa);
    h.f64(report.config.cross_source_tolerances.oxygen_partial_pressure_pa);
    h.f64(report.config.cross_source_tolerances.circuit_pressure_pa);
    h.f64(report.config.cross_source_tolerances.gas_temperature_k);
    h.f64(report.config.cross_source_tolerances.ventilation_flow_l_min);
    h.f64(report.config.cross_source_tolerances.relative_humidity_percent);
    h.usize(report.input_observation_count);
    h.usize(report.accepted_observation_count);
    h.usize(report.rejected_invalid_observations);
    h.usize(report.rejected_future_dated_observations);
    h.usize(report.excluded_out_of_window_observations);
    h.usize(report.excluded_low_confidence_observations);
    for lane in &report.lanes {
        h.string(&format!("{:?}", lane.metric));
        h.string(&format!("{:?}", lane.source_kind));
        h.string(&format!("{:?}", lane.status));
        h.usize(lane.source_trends.len());
        for source in &lane.source_trends {
            h.string(&source.source_id);
            h.string(&format!("{:?}", source.status));
            h.string(&format!("{:?}", source.unit));
            h.usize(source.sample_count);
            h.option_f64(source.first_timestamp_s);
            h.option_f64(source.last_timestamp_s);
            h.option_f64(source.first_value);
            h.option_f64(source.last_value);
            h.option_f64(source.minimum_observed_value);
            h.option_f64(source.maximum_observed_value);
            h.option_f64(source.maximum_deviation_from_endpoint_secant);
            h.option_f64(source.first_uncertainty_abs);
            h.option_f64(source.last_uncertainty_abs);
            h.option_f64(source.maximum_observed_gap_s);
            h.option_f64(source.temporal_coverage_fraction);
            h.option_f64(source.change_over_observed_span);
            h.option_f64(source.rate_per_s);
            h.option_f64(source.rate_lower_bound_per_s);
            h.option_f64(source.rate_upper_bound_per_s);
        }
    }
    h.finish()
}

fn digest_trend_receipt(receipt: &PlssTelemetryTrendReceipt) -> String {
    let mut h = HashWriter::new("symthaea.plss.telemetry.trend.receipt.v1");
    h.string(&receipt.source_revision);
    h.usize(receipt.input_observation_count);
    h.usize(receipt.accepted_observation_count);
    h.usize(receipt.rejected_invalid_observations);
    h.usize(receipt.rejected_future_dated_observations);
    h.usize(receipt.excluded_out_of_window_observations);
    h.usize(receipt.excluded_low_confidence_observations);
    h.string(&receipt.input_digest_sha256);
    h.string(&receipt.result_digest_sha256);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(
        metric: PlssTelemetryMetric,
        value: f64,
        time: f64,
        id: &str,
    ) -> PlssTelemetryObservation {
        PlssTelemetryObservation {
            metric,
            unit: metric.unit(),
            value,
            uncertainty_abs: 1.0,
            observed_at_s: time,
            confidence: 0.95,
            source_id: id.to_string(),
            source_kind: PlssTelemetrySourceKind::Instrument,
        }
    }

    fn config() -> PlssTelemetryTrendConfig {
        PlssTelemetryTrendConfig::simulation_reference()
    }

    #[test]
    fn monotone_series_produces_endpoint_rate_with_explicit_units() {
        let report = analyze_plss_telemetry_trends(&[
            obs(PlssTelemetryMetric::CircuitCo2PartialPressurePa, 200.0, 0.0, "a"),
            obs(PlssTelemetryMetric::CircuitCo2PartialPressurePa, 210.0, 30.0, "a"),
            obs(PlssTelemetryMetric::CircuitCo2PartialPressurePa, 230.0, 60.0, "a"),
        ], config()).unwrap();
        let lane = report
            .lane(
                PlssTelemetryMetric::CircuitCo2PartialPressurePa,
                PlssTelemetrySourceKind::Instrument,
            )
            .unwrap();
        assert_eq!(lane.status, PlssTelemetryTrendStatus::Fresh);
        assert_eq!(lane.source_trends[0].rate_per_s, Some(0.5));
        assert_eq!(lane.source_trends[0].unit, PlssTelemetryUnit::Pascal);
    }

    #[test]
    fn sparse_windows_and_excessive_gaps_do_not_manufacture_zero_rate() {
        let mut cfg = config();
        cfg.minimum_coverage_fraction = 0.9;
        let report = analyze_plss_telemetry_trends(&[
            obs(PlssTelemetryMetric::GasTemperatureK, 290.0, 10.0, "a"),
            obs(PlssTelemetryMetric::GasTemperatureK, 290.0, 20.0, "a"),
            obs(PlssTelemetryMetric::GasTemperatureK, 290.0, 30.0, "a"),
        ], cfg).unwrap();
        let lane = report
            .lane(
                PlssTelemetryMetric::GasTemperatureK,
                PlssTelemetrySourceKind::Instrument,
            )
            .unwrap();
        assert_eq!(lane.status, PlssTelemetryTrendStatus::InsufficientEvidence);
        assert_eq!(lane.source_trends[0].rate_per_s, None);
    }

    #[test]
    fn intermediate_spike_remains_visible_despite_benign_endpoint_rate() {
        let report = analyze_plss_telemetry_trends(
            &[
                obs(PlssTelemetryMetric::CircuitCo2PartialPressurePa, 200.0, 0.0, "a"),
                obs(PlssTelemetryMetric::CircuitCo2PartialPressurePa, 1_000.0, 30.0, "a"),
                obs(PlssTelemetryMetric::CircuitCo2PartialPressurePa, 230.0, 60.0, "a"),
            ],
            config(),
        )
        .unwrap();
        let lane = report
            .lane(
                PlssTelemetryMetric::CircuitCo2PartialPressurePa,
                PlssTelemetrySourceKind::Instrument,
            )
            .unwrap();
        assert_eq!(lane.status, PlssTelemetryTrendStatus::Fresh);
        let trend = &lane.source_trends[0];
        assert_eq!(trend.rate_per_s, Some(0.5));
        assert_eq!(trend.maximum_observed_value, Some(1_000.0));
        assert_eq!(trend.maximum_deviation_from_endpoint_secant, Some(785.0));
    }

    #[test]
    fn excessive_internal_gap_prevents_slope_even_with_full_window_coverage() {
        let mut cfg = config();
        cfg.maximum_gap_s = 20.0;
        let report = analyze_plss_telemetry_trends(
            &[
                obs(PlssTelemetryMetric::GasTemperatureK, 290.0, 0.0, "a"),
                obs(PlssTelemetryMetric::GasTemperatureK, 291.0, 10.0, "a"),
                obs(PlssTelemetryMetric::GasTemperatureK, 292.0, 60.0, "a"),
            ],
            cfg,
        )
        .unwrap();
        let source = &report
            .lane(
                PlssTelemetryMetric::GasTemperatureK,
                PlssTelemetrySourceKind::Instrument,
            )
            .unwrap()
            .source_trends[0];
        assert_eq!(source.status, PlssTelemetryTrendStatus::InsufficientEvidence);
        assert_eq!(source.maximum_observed_gap_s, Some(50.0));
        assert_eq!(source.temporal_coverage_fraction, Some(1.0));
        assert_eq!(source.rate_per_s, None);
    }

    #[test]
    fn contradictory_equal_time_samples_conflict_and_have_no_rate() {
        let report = analyze_plss_telemetry_trends(&[
            obs(PlssTelemetryMetric::CircuitPressurePa, 50_000.0, 0.0, "a"),
            obs(PlssTelemetryMetric::CircuitPressurePa, 55_000.0, 0.0, "a"),
            obs(PlssTelemetryMetric::CircuitPressurePa, 54_000.0, 30.0, "a"),
        ], config()).unwrap();
        let source = &report
            .lane(
                PlssTelemetryMetric::CircuitPressurePa,
                PlssTelemetrySourceKind::Instrument,
            )
            .unwrap()
            .source_trends[0];
        assert_eq!(source.status, PlssTelemetryTrendStatus::Conflicted);
        assert_eq!(source.rate_per_s, None);
        assert_eq!(source.maximum_observed_gap_s, Some(30.0));
        assert_eq!(source.temporal_coverage_fraction, Some(0.5));
    }

    #[test]
    fn distinct_sources_remain_visible_and_disagreement_is_not_averaged() {
        let report = analyze_plss_telemetry_trends(&[
            obs(PlssTelemetryMetric::CircuitCo2PartialPressurePa, 200.0, 0.0, "a"),
            obs(PlssTelemetryMetric::CircuitCo2PartialPressurePa, 250.0, 30.0, "a"),
            obs(PlssTelemetryMetric::CircuitCo2PartialPressurePa, 300.0, 60.0, "a"),
            obs(PlssTelemetryMetric::CircuitCo2PartialPressurePa, 300.0, 0.0, "b"),
            obs(PlssTelemetryMetric::CircuitCo2PartialPressurePa, 350.0, 30.0, "b"),
            obs(PlssTelemetryMetric::CircuitCo2PartialPressurePa, 1_000.0, 60.0, "b"),
        ], config()).unwrap();
        let lane = report
            .lane(
                PlssTelemetryMetric::CircuitCo2PartialPressurePa,
                PlssTelemetrySourceKind::Instrument,
            )
            .unwrap();
        assert_eq!(lane.status, PlssTelemetryTrendStatus::Conflicted);
        assert_eq!(lane.source_trends.len(), 2);
        assert_eq!(lane.source_trends[0].source_id, "a");
        assert_eq!(lane.source_trends[1].source_id, "b");
        assert_eq!(lane.source_trends[0].rate_per_s, Some(100.0 / 60.0));
        assert_eq!(lane.source_trends[1].rate_per_s, Some(700.0 / 60.0));
    }

    #[test]
    fn future_invalid_outside_and_missing_evidence_remain_distinct() {
        let mut invalid = obs(PlssTelemetryMetric::GasTemperatureK, 290.0, 30.0, "bad");
        invalid.value = f64::NAN;
        let mut cfg = config();
        cfg.window_start_s = 10.0;
        cfg.window_end_s = 70.0;
        let report = analyze_plss_telemetry_trends(
            &[
                invalid,
                obs(PlssTelemetryMetric::VentilationFlowLMin, 150.0, 80.0, "future"),
                obs(PlssTelemetryMetric::RelativeHumidityPercent, 45.0, 5.0, "outside"),
            ],
            cfg,
        )
        .unwrap();
        assert_eq!(
            report
                .lane(
                    PlssTelemetryMetric::GasTemperatureK,
                    PlssTelemetrySourceKind::Instrument,
                )
                .unwrap()
                .status,
            PlssTelemetryTrendStatus::Invalid
        );
        assert_eq!(
            report
                .lane(
                    PlssTelemetryMetric::VentilationFlowLMin,
                    PlssTelemetrySourceKind::Instrument,
                )
                .unwrap()
                .status,
            PlssTelemetryTrendStatus::FutureDated
        );
        assert_eq!(
            report
                .lane(
                    PlssTelemetryMetric::RelativeHumidityPercent,
                    PlssTelemetrySourceKind::Instrument,
                )
                .unwrap()
                .status,
            PlssTelemetryTrendStatus::OutOfWindow
        );
        assert_eq!(
            report
                .lane(
                    PlssTelemetryMetric::OxygenPartialPressurePa,
                    PlssTelemetrySourceKind::Instrument,
                )
                .unwrap()
                .status,
            PlssTelemetryTrendStatus::Missing
        );
    }

    #[test]
    fn input_order_does_not_change_report_or_receipt_digests() {
        let a = obs(PlssTelemetryMetric::GasTemperatureK, 290.0, 0.0, "a");
        let b = obs(PlssTelemetryMetric::GasTemperatureK, 292.0, 30.0, "b");
        let c = obs(PlssTelemetryMetric::GasTemperatureK, 291.0, 60.0, "a");
        let rev = "d".repeat(40);
        let (report_a, receipt_a) = analyze_plss_telemetry_trends_with_receipt(
            &[a.clone(), b.clone(), c.clone()],
            config(),
            &rev,
        )
        .unwrap();
        let (report_b, receipt_b) =
            analyze_plss_telemetry_trends_with_receipt(&[c, a, b], config(), &rev).unwrap();
        assert_eq!(report_a, report_b);
        assert_eq!(receipt_a.input_digest_sha256, receipt_b.input_digest_sha256);
        assert_eq!(receipt_a.result_digest_sha256, receipt_b.result_digest_sha256);
        assert_eq!(receipt_a.receipt_digest_sha256, receipt_b.receipt_digest_sha256);
    }

    #[test]
    fn changed_configuration_changes_result_fingerprint() {
        let rev = "e".repeat(40);
        let (_, a) = analyze_plss_telemetry_trends_with_receipt(&[], config(), &rev).unwrap();
        let mut changed = config();
        changed.maximum_gap_s += 1.0;
        let (_, b) = analyze_plss_telemetry_trends_with_receipt(&[], changed, &rev).unwrap();
        assert_eq!(a.input_digest_sha256, b.input_digest_sha256);
        assert_ne!(a.result_digest_sha256, b.result_digest_sha256);
        assert_ne!(a.receipt_digest_sha256, b.receipt_digest_sha256);
    }

    #[test]
    fn low_confidence_samples_do_not_influence_trend() {
        let mut cfg = config();
        cfg.minimum_confidence = 0.7;
        let mut low = obs(PlssTelemetryMetric::GasTemperatureK, 900.0, 60.0, "a");
        low.confidence = 0.2;
        let report = analyze_plss_telemetry_trends(&[
            obs(PlssTelemetryMetric::GasTemperatureK, 290.0, 0.0, "a"),
            obs(PlssTelemetryMetric::GasTemperatureK, 291.0, 30.0, "a"),
            low,
        ], cfg).unwrap();
        assert_eq!(report.excluded_low_confidence_observations, 1);
        assert_eq!(
            report
                .lane(
                    PlssTelemetryMetric::GasTemperatureK,
                    PlssTelemetrySourceKind::Instrument,
                )
                .unwrap()
                .source_trends[0]
                .last_value,
            Some(291.0)
        );
        assert_eq!(
            report
                .lane(
                    PlssTelemetryMetric::GasTemperatureK,
                    PlssTelemetrySourceKind::Instrument,
                )
                .unwrap()
                .status,
            PlssTelemetryTrendStatus::InsufficientEvidence
        );
    }

    #[test]
    fn changing_a_sample_value_changes_input_and_result_fingerprints() {
        let revision = "f".repeat(40);
        let first = [
            obs(PlssTelemetryMetric::GasTemperatureK, 290.0, 0.0, "a"),
            obs(PlssTelemetryMetric::GasTemperatureK, 291.0, 30.0, "a"),
            obs(PlssTelemetryMetric::GasTemperatureK, 292.0, 60.0, "a"),
        ];
        let mut changed = first.clone();
        changed[2].value = 292.5;
        let (_, first_receipt) =
            analyze_plss_telemetry_trends_with_receipt(&first, config(), &revision).unwrap();
        let (_, changed_receipt) =
            analyze_plss_telemetry_trends_with_receipt(&changed, config(), &revision).unwrap();
        assert_ne!(
            first_receipt.input_digest_sha256,
            changed_receipt.input_digest_sha256
        );
        assert_ne!(
            first_receipt.result_digest_sha256,
            changed_receipt.result_digest_sha256
        );
        assert_ne!(
            first_receipt.receipt_digest_sha256,
            changed_receipt.receipt_digest_sha256
        );
    }

    #[test]
    fn invalid_config_and_revision_fail_closed() {
        let mut changed = config();
        changed.minimum_samples = 1;
        assert_eq!(
            analyze_plss_telemetry_trends(&[], changed),
            Err(PlssTelemetryTrendError::InvalidConfig)
        );
        assert_eq!(
            analyze_plss_telemetry_trends_with_receipt(&[], config(), "bad"),
            Err(PlssTelemetryTrendError::InvalidSourceRevision)
        );
    }
}
