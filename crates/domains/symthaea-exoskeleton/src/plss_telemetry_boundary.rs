// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Report-only, uncertainty-aware PLSS telemetry boundary projections (SX-027).
//!
//! This module extrapolates SX-026 endpoint-rate envelopes toward explicit,
//! caller-supplied thresholds. It does not define or qualify thresholds, predict
//! physiological exposure, infer inspired CO2, issue alarms, or control hardware.
//! A candidate crossing interval is a local linear projection, not a guarantee.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::plss_telemetry::{
    PlssTelemetryMetric, PlssTelemetrySourceKind, PlssTelemetryUnit,
};
use crate::plss_telemetry_trend::{
    PlssTelemetryTrendReport, PlssTelemetryTrendReceipt,
    PlssTelemetryTrendStatus, TrendEvidenceLevel, verify_plss_telemetry_trend_receipt,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PlssBoundaryDirection {
    Upper,
    Lower,
}

/// An explicitly versioned engineering boundary. This is caller-supplied policy
/// data, not an endorsed human-exposure limit or suit operating requirement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlssTelemetryBoundary {
    pub metric: PlssTelemetryMetric,
    pub source_kind: PlssTelemetrySourceKind,
    pub source_id: String,
    pub direction: PlssBoundaryDirection,
    pub threshold_value: f64,
    /// SHA-like identifier for the external threshold-set revision (40/64 hex).
    /// It provides provenance linkage, not qualification or authenticity.
    pub threshold_revision: String,
    /// Maximum projection horizon in seconds from the last accepted sample.
    /// Candidate intervals that exceed it are marked CandidateBeyondHorizon.
    pub maximum_projection_horizon_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlssBoundaryProjectionStatus {
    /// The complete rate/value envelope reaches the boundary within the horizon.
    CandidateWithinHorizon,
    /// At least part of the calculated crossing interval exceeds the horizon.
    CandidateBeyondHorizon,
    /// The complete current value-uncertainty envelope is already beyond the boundary.
    AlreadyBeyondBoundary,
    /// The boundary intersects the current value-uncertainty envelope.
    BoundaryWithinCurrentUncertainty,
    /// The complete rate envelope points away from the selected boundary.
    MovingAway,
    /// No admissible endpoint rate points toward the selected boundary, but the
    /// interval includes a flat (zero) rate.
    NotApproaching,
    /// Some admissible endpoint rates point toward the boundary while others do not.
    DirectionUncertain,
    /// Upstream trend evidence is not Fresh or required numeric fields are absent.
    InsufficientEvidence,
    /// SX-026 marks the selected lane or source as conflicted.
    Conflicted,
    /// No source with this exact metric/provenance/source ID exists in the report.
    MissingSource,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlssTelemetryBoundaryProjection {
    pub metric: PlssTelemetryMetric,
    pub source_kind: PlssTelemetrySourceKind,
    pub source_id: String,
    pub unit: PlssTelemetryUnit,
    pub direction: PlssBoundaryDirection,
    pub threshold_value: f64,
    pub threshold_revision: String,
    pub maximum_projection_horizon_s: f64,
    pub lane_status: Option<PlssTelemetryTrendStatus>,
    pub source_trend_status: Option<PlssTelemetryTrendStatus>,
    pub status: PlssBoundaryProjectionStatus,
    pub last_timestamp_s: Option<f64>,
    pub last_value: Option<f64>,
    pub last_uncertainty_abs: Option<f64>,
    pub endpoint_rate_per_s: Option<f64>,
    pub endpoint_rate_lower_bound_per_s: Option<f64>,
    pub endpoint_rate_upper_bound_per_s: Option<f64>,
    /// Earliest candidate crossing from the near side of the current value
    /// envelope, assuming the fastest admissible rate toward the threshold.
    pub earliest_candidate_crossing_after_last_sample_s: Option<f64>,
    /// Latest candidate crossing from the far side of the current value
    /// envelope, assuming the slowest admissible rate toward the threshold.
    /// None means the rate envelope does not establish a finite latest crossing.
    pub latest_candidate_crossing_after_last_sample_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlssTelemetryBoundaryReport {
    pub evidence_level: TrendEvidenceLevel,
    pub source_revision: String,
    pub source_trend_result_digest_sha256: String,
    pub projections: Vec<PlssTelemetryBoundaryProjection>,
    pub input_boundary_count: usize,
    pub candidate_count: usize,
    pub direction_uncertain_count: usize,
    pub blocked_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlssTelemetryBoundaryReceipt {
    pub source_revision: String,
    pub source_trend_result_digest_sha256: String,
    pub input_boundary_count: usize,
    pub candidate_count: usize,
    pub direction_uncertain_count: usize,
    pub blocked_count: usize,
    pub input_digest_sha256: String,
    pub result_digest_sha256: String,
    /// Detects changes in report metadata; it is not a digital signature.
    pub receipt_digest_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlssTelemetryBoundaryError {
    InvalidBoundary,
    InvalidThresholdRevision,
    InvalidTrendReceipt,
}

/// Project a sorted set of caller-supplied boundaries against one SX-026 report.
///
/// This fails closed if the trend report cannot be verified against its SX-026
/// receipt. A boundary is only numerically projected when both the aggregate
/// metric/provenance lane and the selected source trend are Fresh. This means
/// cross-source disagreement suppresses projections rather than selecting the
/// most convenient sensor. The output remains SimulationOnly.
pub fn project_plss_telemetry_boundaries(
    trend_report: &PlssTelemetryTrendReport,
    trend_receipt: &PlssTelemetryTrendReceipt,
    boundaries: &[PlssTelemetryBoundary],
) -> Result<
    (PlssTelemetryBoundaryReport, PlssTelemetryBoundaryReceipt),
    PlssTelemetryBoundaryError,
> {
    if !verify_plss_telemetry_trend_receipt(trend_report, trend_receipt) {
        return Err(PlssTelemetryBoundaryError::InvalidTrendReceipt);
    }

    let mut canonical_boundaries = boundaries.to_vec();
    for boundary in &mut canonical_boundaries {
        validate_boundary(boundary)?;
        boundary.threshold_revision = boundary.threshold_revision.to_ascii_lowercase();
    }
    canonical_boundaries.sort_by(compare_boundaries);

    let projections: Vec<_> = canonical_boundaries
        .iter()
        .map(|boundary| project_one(trend_report, boundary))
        .collect();

    let candidate_count = projections
        .iter()
        .filter(|projection| matches!(
            projection.status,
            PlssBoundaryProjectionStatus::CandidateWithinHorizon
                | PlssBoundaryProjectionStatus::CandidateBeyondHorizon
        ))
        .count();
    let direction_uncertain_count = projections
        .iter()
        .filter(|projection| projection.status == PlssBoundaryProjectionStatus::DirectionUncertain)
        .count();
    let blocked_count = projections.len() - candidate_count - direction_uncertain_count;

    let report = PlssTelemetryBoundaryReport {
        evidence_level: trend_report.evidence_level,
        source_revision: trend_receipt.source_revision.to_ascii_lowercase(),
        source_trend_result_digest_sha256: trend_receipt.result_digest_sha256.clone(),
        input_boundary_count: projections.len(),
        candidate_count,
        direction_uncertain_count,
        blocked_count,
        projections,
    };

    let mut receipt = PlssTelemetryBoundaryReceipt {
        source_revision: report.source_revision.clone(),
        source_trend_result_digest_sha256: report.source_trend_result_digest_sha256.clone(),
        input_boundary_count: report.input_boundary_count,
        candidate_count: report.candidate_count,
        direction_uncertain_count: report.direction_uncertain_count,
        blocked_count: report.blocked_count,
        input_digest_sha256: digest_boundary_inputs(
            &report.source_revision,
            &report.source_trend_result_digest_sha256,
            &report.projections,
        ),
        result_digest_sha256: digest_boundary_report(&report),
        receipt_digest_sha256: String::new(),
    };
    receipt.receipt_digest_sha256 = digest_boundary_receipt(&receipt);
    Ok((report, receipt))
}

/// Verify the boundary-report receipt independently of the boundary inputs.
/// Input digest is recomputed from the report's retained threshold definitions.
pub fn verify_plss_telemetry_boundary_receipt(
    report: &PlssTelemetryBoundaryReport,
    receipt: &PlssTelemetryBoundaryReceipt,
) -> bool {
    if !is_revision(&receipt.source_revision)
        || !is_prefixed_digest(&receipt.source_trend_result_digest_sha256)
        || !is_prefixed_digest(&receipt.input_digest_sha256)
        || !is_prefixed_digest(&receipt.result_digest_sha256)
        || !is_prefixed_digest(&receipt.receipt_digest_sha256)
    {
        return false;
    }
    report.source_revision == receipt.source_revision.to_ascii_lowercase()
        && report.source_trend_result_digest_sha256 == receipt.source_trend_result_digest_sha256
        && report.input_boundary_count == report.projections.len()
        && report.input_boundary_count == receipt.input_boundary_count
        && report.candidate_count == receipt.candidate_count
        && report.direction_uncertain_count == receipt.direction_uncertain_count
        && report.blocked_count == receipt.blocked_count
        && report.candidate_count + report.direction_uncertain_count + report.blocked_count
            == report.input_boundary_count
        && receipt.input_digest_sha256
            == digest_boundary_inputs(
                &report.source_revision,
                &report.source_trend_result_digest_sha256,
                &report.projections,
            )
        && receipt.result_digest_sha256 == digest_boundary_report(report)
        && receipt.receipt_digest_sha256 == digest_boundary_receipt(receipt)
}

fn validate_boundary(boundary: &PlssTelemetryBoundary) -> Result<(), PlssTelemetryBoundaryError> {
    if boundary.source_id.trim().is_empty()
        || boundary.source_id.trim() != boundary.source_id
        || !boundary.threshold_value.is_finite()
        || !boundary.metric.accepts_value(boundary.threshold_value)
        || !boundary.maximum_projection_horizon_s.is_finite()
        || boundary.maximum_projection_horizon_s <= 0.0
    {
        return Err(PlssTelemetryBoundaryError::InvalidBoundary);
    }
    if !is_revision(&boundary.threshold_revision) {
        return Err(PlssTelemetryBoundaryError::InvalidThresholdRevision);
    }
    Ok(())
}

fn is_revision(revision: &str) -> bool {
    (revision.len() == 40 || revision.len() == 64)
        && revision.bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn is_prefixed_digest(digest: &str) -> bool {
    digest.len() == 71
        && digest.starts_with("sha256:")
        && digest[7..].bytes().all(|byte| byte.is_ascii_hexdigit())
}

fn compare_boundaries(
    left: &PlssTelemetryBoundary,
    right: &PlssTelemetryBoundary,
) -> std::cmp::Ordering {
    left.metric
        .cmp(&right.metric)
        .then(left.source_kind.cmp(&right.source_kind))
        .then(left.source_id.cmp(&right.source_id))
        .then(left.direction.cmp(&right.direction))
        .then(left.threshold_value.total_cmp(&right.threshold_value))
        .then(left.threshold_revision.cmp(&right.threshold_revision))
        .then(left.maximum_projection_horizon_s.total_cmp(&right.maximum_projection_horizon_s))
}

fn project_one(
    report: &PlssTelemetryTrendReport,
    boundary: &PlssTelemetryBoundary,
) -> PlssTelemetryBoundaryProjection {
    let mut projection = PlssTelemetryBoundaryProjection {
        metric: boundary.metric,
        source_kind: boundary.source_kind,
        source_id: boundary.source_id.clone(),
        unit: boundary.metric.unit(),
        direction: boundary.direction,
        threshold_value: boundary.threshold_value,
        threshold_revision: boundary.threshold_revision.to_ascii_lowercase(),
        maximum_projection_horizon_s: boundary.maximum_projection_horizon_s,
        lane_status: None,
        source_trend_status: None,
        status: PlssBoundaryProjectionStatus::MissingSource,
        last_timestamp_s: None,
        last_value: None,
        last_uncertainty_abs: None,
        endpoint_rate_per_s: None,
        endpoint_rate_lower_bound_per_s: None,
        endpoint_rate_upper_bound_per_s: None,
        earliest_candidate_crossing_after_last_sample_s: None,
        latest_candidate_crossing_after_last_sample_s: None,
    };

    let Some(lane) = report.lane(boundary.metric, boundary.source_kind) else {
        return projection;
    };
    projection.lane_status = Some(lane.status);
    let Some(source) = lane
        .source_trends
        .iter()
        .find(|source| source.source_id == boundary.source_id)
    else {
        return projection;
    };
    projection.source_trend_status = Some(source.status);
    projection.last_timestamp_s = source.last_timestamp_s;
    projection.last_value = source.last_value;
    projection.last_uncertainty_abs = source.last_uncertainty_abs;
    projection.endpoint_rate_per_s = source.rate_per_s;
    projection.endpoint_rate_lower_bound_per_s = source.rate_lower_bound_per_s;
    projection.endpoint_rate_upper_bound_per_s = source.rate_upper_bound_per_s;

    if lane.status == PlssTelemetryTrendStatus::Conflicted
        || source.status == PlssTelemetryTrendStatus::Conflicted
    {
        projection.status = PlssBoundaryProjectionStatus::Conflicted;
        return projection;
    }
    if lane.status != PlssTelemetryTrendStatus::Fresh
        || source.status != PlssTelemetryTrendStatus::Fresh
    {
        projection.status = PlssBoundaryProjectionStatus::InsufficientEvidence;
        return projection;
    }

    let (
        Some(last_value),
        Some(last_uncertainty),
        Some(rate),
        Some(rate_lower),
        Some(rate_upper),
    ) = (
        source.last_value,
        source.last_uncertainty_abs,
        source.rate_per_s,
        source.rate_lower_bound_per_s,
        source.rate_upper_bound_per_s,
    ) else {
        projection.status = PlssBoundaryProjectionStatus::InsufficientEvidence;
        return projection;
    };
    if !last_value.is_finite()
        || !last_uncertainty.is_finite()
        || last_uncertainty < 0.0
        || !rate.is_finite()
        || !rate_lower.is_finite()
        || !rate_upper.is_finite()
        || rate_lower > rate_upper
        || rate < rate_lower
        || rate > rate_upper
    {
        projection.status = PlssBoundaryProjectionStatus::InsufficientEvidence;
        return projection;
    }

    // Orient the metric so an upper and lower boundary both become an increasing crossing.
    let (oriented_value, oriented_threshold, oriented_rate_lower, oriented_rate_upper) =
        match boundary.direction {
            PlssBoundaryDirection::Upper => (
                last_value,
                boundary.threshold_value,
                rate_lower,
                rate_upper,
            ),
            PlssBoundaryDirection::Lower => (
                -last_value,
                -boundary.threshold_value,
                -rate_upper,
                -rate_lower,
            ),
        };
    let near_edge = oriented_value + last_uncertainty;
    let far_edge = oriented_value - last_uncertainty;

    if far_edge >= oriented_threshold {
        projection.status = PlssBoundaryProjectionStatus::AlreadyBeyondBoundary;
        return projection;
    }
    if near_edge >= oriented_threshold {
        projection.status = PlssBoundaryProjectionStatus::BoundaryWithinCurrentUncertainty;
        return projection;
    }

    if oriented_rate_upper < 0.0 {
        projection.status = PlssBoundaryProjectionStatus::MovingAway;
        return projection;
    }
    if oriented_rate_upper == 0.0 {
        projection.status = PlssBoundaryProjectionStatus::NotApproaching;
        return projection;
    }

    let nearest_distance = oriented_threshold - near_edge;
    let farthest_distance = oriented_threshold - far_edge;
    let earliest = nearest_distance / oriented_rate_upper;
    if !earliest.is_finite() || earliest < 0.0 {
        projection.status = PlssBoundaryProjectionStatus::InsufficientEvidence;
        return projection;
    }
    projection.earliest_candidate_crossing_after_last_sample_s = Some(earliest);

    if oriented_rate_lower <= 0.0 {
        projection.status = PlssBoundaryProjectionStatus::DirectionUncertain;
        return projection;
    }

    let latest = farthest_distance / oriented_rate_lower;
    if !latest.is_finite() || latest < earliest {
        projection.status = PlssBoundaryProjectionStatus::CandidateBeyondHorizon;
        return projection;
    }
    projection.latest_candidate_crossing_after_last_sample_s = Some(latest);
    if latest <= boundary.maximum_projection_horizon_s {
        projection.status = PlssBoundaryProjectionStatus::CandidateWithinHorizon;
    } else {
        projection.status = PlssBoundaryProjectionStatus::CandidateBeyondHorizon;
    }
    projection
}

struct HashWriter(Sha256);

impl HashWriter {
    fn new(domain: &str) -> Self {
        let mut writer = Self(Sha256::new());
        writer.string(domain);
        writer
    }

    fn raw(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn u64(&mut self, value: u64) {
        self.raw(&value.to_le_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.u64(value as u64);
    }

    fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    fn string(&mut self, value: &str) {
        self.usize(value.len());
        self.raw(value.as_bytes());
    }

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
        format!(
            "sha256:{}",
            digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>()
        )
    }
}

fn digest_boundary_inputs(
    source_revision: &str,
    trend_result_digest: &str,
    projections: &[PlssTelemetryBoundaryProjection],
) -> String {
    let mut sorted: Vec<_> = projections.iter().collect();
    sorted.sort_by(|left, right| {
        left.metric.cmp(&right.metric)
            .then(left.source_kind.cmp(&right.source_kind))
            .then(left.source_id.cmp(&right.source_id))
            .then(left.direction.cmp(&right.direction))
            .then(left.threshold_value.total_cmp(&right.threshold_value))
            .then(left.threshold_revision.cmp(&right.threshold_revision))
            .then(left.maximum_projection_horizon_s.total_cmp(&right.maximum_projection_horizon_s))
    });
    let mut h = HashWriter::new("symthaea.plss.telemetry.boundary.input.v1");
    h.string(source_revision);
    h.string(trend_result_digest);
    h.usize(sorted.len());
    for projection in sorted {
        h.string(metric_tag(projection.metric));
        h.string(source_kind_tag(projection.source_kind));
        h.string(&projection.source_id);
        h.string(direction_tag(projection.direction));
        h.f64(projection.threshold_value);
        h.string(&projection.threshold_revision.to_ascii_lowercase());
        h.f64(projection.maximum_projection_horizon_s);
    }
    h.finish()
}

fn digest_boundary_report(report: &PlssTelemetryBoundaryReport) -> String {
    let mut h = HashWriter::new("symthaea.plss.telemetry.boundary.result.v1");
    h.string(&report.source_revision);
    h.string(&report.source_trend_result_digest_sha256);
    h.string(match report.evidence_level {
        TrendEvidenceLevel::SimulationOnly => "simulation_only",
    });
    h.usize(report.input_boundary_count);
    h.usize(report.candidate_count);
    h.usize(report.direction_uncertain_count);
    h.usize(report.blocked_count);
    h.usize(report.projections.len());
    for projection in &report.projections {
        h.string(metric_tag(projection.metric));
        h.string(source_kind_tag(projection.source_kind));
        h.string(&projection.source_id);
        h.string(unit_tag(projection.unit));
        h.string(direction_tag(projection.direction));
        h.f64(projection.threshold_value);
        h.string(&projection.threshold_revision);
        h.f64(projection.maximum_projection_horizon_s);
        h.string(option_trend_status_tag(projection.lane_status));
        h.string(option_trend_status_tag(projection.source_trend_status));
        h.string(projection_status_tag(projection.status));
        h.option_f64(projection.last_timestamp_s);
        h.option_f64(projection.last_value);
        h.option_f64(projection.last_uncertainty_abs);
        h.option_f64(projection.endpoint_rate_per_s);
        h.option_f64(projection.endpoint_rate_lower_bound_per_s);
        h.option_f64(projection.endpoint_rate_upper_bound_per_s);
        h.option_f64(projection.earliest_candidate_crossing_after_last_sample_s);
        h.option_f64(projection.latest_candidate_crossing_after_last_sample_s);
    }
    h.finish()
}

fn digest_boundary_receipt(receipt: &PlssTelemetryBoundaryReceipt) -> String {
    let mut h = HashWriter::new("symthaea.plss.telemetry.boundary.receipt.v1");
    h.string(&receipt.source_revision);
    h.string(&receipt.source_trend_result_digest_sha256);
    h.usize(receipt.input_boundary_count);
    h.usize(receipt.candidate_count);
    h.usize(receipt.direction_uncertain_count);
    h.usize(receipt.blocked_count);
    h.string(&receipt.input_digest_sha256);
    h.string(&receipt.result_digest_sha256);
    h.finish()
}

fn metric_tag(metric: PlssTelemetryMetric) -> &'static str {
    match metric {
        PlssTelemetryMetric::CircuitCo2PartialPressurePa => "circuit_co2_partial_pressure_pa",
        PlssTelemetryMetric::OxygenPartialPressurePa => "oxygen_partial_pressure_pa",
        PlssTelemetryMetric::CircuitPressurePa => "circuit_pressure_pa",
        PlssTelemetryMetric::GasTemperatureK => "gas_temperature_k",
        PlssTelemetryMetric::VentilationFlowLMin => "ventilation_flow_l_min",
        PlssTelemetryMetric::RelativeHumidityPercent => "relative_humidity_percent",
    }
}

fn source_kind_tag(kind: PlssTelemetrySourceKind) -> &'static str {
    match kind {
        PlssTelemetrySourceKind::Instrument => "instrument",
        PlssTelemetrySourceKind::ModelEstimate => "model_estimate",
        PlssTelemetrySourceKind::ModelPrediction => "model_prediction",
    }
}

fn unit_tag(unit: PlssTelemetryUnit) -> &'static str {
    match unit {
        PlssTelemetryUnit::Pascal => "pascal",
        PlssTelemetryUnit::Kelvin => "kelvin",
        PlssTelemetryUnit::LiterPerMinute => "liter_per_minute",
        PlssTelemetryUnit::Percent => "percent",
    }
}

fn direction_tag(direction: PlssBoundaryDirection) -> &'static str {
    match direction {
        PlssBoundaryDirection::Upper => "upper",
        PlssBoundaryDirection::Lower => "lower",
    }
}

fn projection_status_tag(status: PlssBoundaryProjectionStatus) -> &'static str {
    match status {
        PlssBoundaryProjectionStatus::CandidateWithinHorizon => "candidate_within_horizon",
        PlssBoundaryProjectionStatus::CandidateBeyondHorizon => "candidate_beyond_horizon",
        PlssBoundaryProjectionStatus::AlreadyBeyondBoundary => "already_beyond_boundary",
        PlssBoundaryProjectionStatus::BoundaryWithinCurrentUncertainty => {
            "boundary_within_current_uncertainty"
        },
        PlssBoundaryProjectionStatus::MovingAway => "moving_away",
        PlssBoundaryProjectionStatus::NotApproaching => "not_approaching",
        PlssBoundaryProjectionStatus::DirectionUncertain => "direction_uncertain",
        PlssBoundaryProjectionStatus::InsufficientEvidence => "insufficient_evidence",
        PlssBoundaryProjectionStatus::Conflicted => "conflicted",
        PlssBoundaryProjectionStatus::MissingSource => "missing_source",
    }
}

fn trend_status_tag(status: PlssTelemetryTrendStatus) -> &'static str {
    match status {
        PlssTelemetryTrendStatus::Fresh => "fresh",
        PlssTelemetryTrendStatus::Conflicted => "conflicted",
        PlssTelemetryTrendStatus::InsufficientEvidence => "insufficient_evidence",
        PlssTelemetryTrendStatus::Missing => "missing",
        PlssTelemetryTrendStatus::LowConfidence => "low_confidence",
        PlssTelemetryTrendStatus::Invalid => "invalid",
        PlssTelemetryTrendStatus::FutureDated => "future_dated",
        PlssTelemetryTrendStatus::OutOfWindow => "out_of_window",
    }
}

fn option_trend_status_tag(status: Option<PlssTelemetryTrendStatus>) -> &'static str {
    match status {
        Some(value) => trend_status_tag(value),
        None => "none",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plss_telemetry::PlssTelemetryObservation;
    use crate::plss_telemetry_trend::{
        PlssTelemetryTrendConfig, analyze_plss_telemetry_trends_with_receipt,
    };

    fn obs(value: f64, time: f64, source_id: &str) -> PlssTelemetryObservation {
        PlssTelemetryObservation {
            metric: PlssTelemetryMetric::CircuitCo2PartialPressurePa,
            unit: PlssTelemetryUnit::Pascal,
            value,
            uncertainty_abs: 1.0,
            observed_at_s: time,
            confidence: 0.95,
            source_id: source_id.to_string(),
            source_kind: PlssTelemetrySourceKind::Instrument,
        }
    }

    fn trends(
        observations: &[PlssTelemetryObservation],
    ) -> (PlssTelemetryTrendReport, PlssTelemetryTrendReceipt) {
        analyze_plss_telemetry_trends_with_receipt(
            observations,
            PlssTelemetryTrendConfig::simulation_reference(),
            &"a".repeat(40),
        ).unwrap()
    }

    fn boundary(
        direction: PlssBoundaryDirection,
        threshold: f64,
        horizon: f64,
    ) -> PlssTelemetryBoundary {
        PlssTelemetryBoundary {
            metric: PlssTelemetryMetric::CircuitCo2PartialPressurePa,
            source_kind: PlssTelemetrySourceKind::Instrument,
            source_id: "sensor-a".to_string(),
            direction,
            threshold_value: threshold,
            threshold_revision: "b".repeat(40),
            maximum_projection_horizon_s: horizon,
        }
    }

    fn project(
        observations: &[PlssTelemetryObservation],
        target: PlssTelemetryBoundary,
    ) -> PlssTelemetryBoundaryProjection {
        let (report, receipt) = trends(observations);
        project_plss_telemetry_boundaries(&report, &receipt, &[target])
            .unwrap().0.projections.remove(0)
    }

    #[test]
    fn rising_upper_boundary_uses_value_and_rate_envelopes() {
        let p = project(&[
            obs(200.0, 0.0, "sensor-a"),
            obs(300.0, 30.0, "sensor-a"),
            obs(400.0, 60.0, "sensor-a"),
        ], boundary(PlssBoundaryDirection::Upper, 500.0, 60.0));
        assert_eq!(p.status, PlssBoundaryProjectionStatus::CandidateWithinHorizon);
        let early = p.earliest_candidate_crossing_after_last_sample_s.unwrap();
        let late = p.latest_candidate_crossing_after_last_sample_s.unwrap();
        assert!(early > 0.0 && late >= early && late <= 60.0);
    }

    #[test]
    fn falling_lower_boundary_is_oriented_as_an_increasing_crossing() {
        let p = project(&[
            obs(400.0, 0.0, "sensor-a"),
            obs(300.0, 30.0, "sensor-a"),
            obs(200.0, 60.0, "sensor-a"),
        ], boundary(PlssBoundaryDirection::Lower, 100.0, 60.0));
        assert_eq!(p.status, PlssBoundaryProjectionStatus::CandidateWithinHorizon);
        assert!(p.earliest_candidate_crossing_after_last_sample_s.unwrap() > 0.0);
        assert!(p.latest_candidate_crossing_after_last_sample_s.unwrap()
            >= p.earliest_candidate_crossing_after_last_sample_s.unwrap());
    }

    #[test]
    fn weak_rate_interval_with_zero_endpoint_is_direction_uncertain() {
        let p = project(&[
            obs(200.0, 0.0, "sensor-a"),
            obs(201.0, 30.0, "sensor-a"),
            obs(202.0, 60.0, "sensor-a"),
        ], boundary(PlssBoundaryDirection::Upper, 250.0, 60.0));
        assert_eq!(p.status, PlssBoundaryProjectionStatus::DirectionUncertain);
        assert!(p.earliest_candidate_crossing_after_last_sample_s.is_some());
        assert_eq!(p.latest_candidate_crossing_after_last_sample_s, None);
    }

    #[test]
    fn rising_series_toward_lower_boundary_is_moving_away() {
        let p = project(&[
            obs(200.0, 0.0, "sensor-a"),
            obs(300.0, 30.0, "sensor-a"),
            obs(400.0, 60.0, "sensor-a"),
        ], boundary(PlssBoundaryDirection::Lower, 100.0, 60.0));
        assert_eq!(p.status, PlssBoundaryProjectionStatus::MovingAway);
        assert_eq!(p.earliest_candidate_crossing_after_last_sample_s, None);
    }

    #[test]
    fn flat_rate_envelope_is_not_mislabelled_as_moving_away() {
        let mut flat = obs(300.0, 0.0, "sensor-a");
        flat.uncertainty_abs = 0.0;
        let mut middle = obs(300.0, 30.0, "sensor-a");
        middle.uncertainty_abs = 0.0;
        let mut last = obs(300.0, 60.0, "sensor-a");
        last.uncertainty_abs = 0.0;
        let p = project(
            &[flat, middle, last],
            boundary(PlssBoundaryDirection::Upper, 500.0, 60.0),
        );
        assert_eq!(p.status, PlssBoundaryProjectionStatus::NotApproaching);
        assert_eq!(p.earliest_candidate_crossing_after_last_sample_s, None);
    }

    #[test]
    fn uncertainty_overlap_suppresses_a_future_crossing_estimate() {
        let p = project(&[
            obs(200.0, 0.0, "sensor-a"),
            obs(300.0, 30.0, "sensor-a"),
            obs(400.0, 60.0, "sensor-a"),
        ], boundary(PlssBoundaryDirection::Upper, 400.5, 60.0));
        assert_eq!(p.status, PlssBoundaryProjectionStatus::BoundaryWithinCurrentUncertainty);
        assert_eq!(p.earliest_candidate_crossing_after_last_sample_s, None);
    }

    #[test]
    fn fully_beyond_boundary_is_reported_without_extrapolation() {
        let p = project(&[
            obs(200.0, 0.0, "sensor-a"),
            obs(300.0, 30.0, "sensor-a"),
            obs(400.0, 60.0, "sensor-a"),
        ], boundary(PlssBoundaryDirection::Upper, 399.0, 60.0));
        assert_eq!(p.status, PlssBoundaryProjectionStatus::AlreadyBeyondBoundary);
        assert_eq!(p.earliest_candidate_crossing_after_last_sample_s, None);
    }

    #[test]
    fn candidate_interval_that_exceeds_horizon_is_not_marked_within_horizon() {
        let p = project(&[
            obs(200.0, 0.0, "sensor-a"),
            obs(300.0, 30.0, "sensor-a"),
            obs(400.0, 60.0, "sensor-a"),
        ], boundary(PlssBoundaryDirection::Upper, 500.0, 10.0));
        assert_eq!(p.status, PlssBoundaryProjectionStatus::CandidateBeyondHorizon);
        assert!(p.latest_candidate_crossing_after_last_sample_s.unwrap() > 10.0);
    }

    #[test]
    fn aggregate_source_conflict_blocks_projection() {
        let p = project(&[
            obs(200.0, 0.0, "sensor-a"),
            obs(250.0, 30.0, "sensor-a"),
            obs(300.0, 60.0, "sensor-a"),
            obs(800.0, 0.0, "sensor-b"),
            obs(850.0, 30.0, "sensor-b"),
            obs(900.0, 60.0, "sensor-b"),
        ], boundary(PlssBoundaryDirection::Upper, 1_000.0, 60.0));
        assert_eq!(p.status, PlssBoundaryProjectionStatus::Conflicted);
        assert_eq!(p.earliest_candidate_crossing_after_last_sample_s, None);
    }

    #[test]
    fn unknown_source_stays_missing() {
        let (report, receipt) = trends(&[
            obs(200.0, 0.0, "sensor-a"),
            obs(300.0, 30.0, "sensor-a"),
            obs(400.0, 60.0, "sensor-a"),
        ]);
        let mut b = boundary(PlssBoundaryDirection::Upper, 500.0, 60.0);
        b.source_id = "unwired-sensor".into();
        let (out, _) = project_plss_telemetry_boundaries(&report, &receipt, &[b]).unwrap();
        assert_eq!(out.projections[0].status, PlssBoundaryProjectionStatus::MissingSource);
    }

    #[test]
    fn altered_trend_report_is_rejected_against_prior_receipt() {
        let (mut report, receipt) = trends(&[
            obs(200.0, 0.0, "sensor-a"),
            obs(300.0, 30.0, "sensor-a"),
            obs(400.0, 60.0, "sensor-a"),
        ]);
        report.lanes[0].status = PlssTelemetryTrendStatus::Conflicted;
        assert_eq!(
            project_plss_telemetry_boundaries(
                &report,
                &receipt,
                &[boundary(PlssBoundaryDirection::Upper, 500.0, 60.0)],
            ),
            Err(PlssTelemetryBoundaryError::InvalidTrendReceipt)
        );
    }

    #[test]
    fn invalid_threshold_revisions_and_domains_fail_closed() {
        let (report, receipt) = trends(&[
            obs(200.0, 0.0, "sensor-a"),
            obs(300.0, 30.0, "sensor-a"),
            obs(400.0, 60.0, "sensor-a"),
        ]);
        let mut bad_revision = boundary(PlssBoundaryDirection::Upper, 500.0, 60.0);
        bad_revision.threshold_revision = "not-a-revision".into();
        assert_eq!(
            project_plss_telemetry_boundaries(&report, &receipt, &[bad_revision]),
            Err(PlssTelemetryBoundaryError::InvalidThresholdRevision)
        );
        let mut bad_value = boundary(PlssBoundaryDirection::Upper, f64::NAN, 60.0);
        assert_eq!(
            project_plss_telemetry_boundaries(&report, &receipt, &[bad_value.clone()]),
            Err(PlssTelemetryBoundaryError::InvalidBoundary)
        );
        bad_value.threshold_value = 101.0;
        bad_value.metric = PlssTelemetryMetric::RelativeHumidityPercent;
        assert_eq!(
            project_plss_telemetry_boundaries(&report, &receipt, &[bad_value]),
            Err(PlssTelemetryBoundaryError::InvalidBoundary)
        );
    }

    #[test]
    fn boundary_order_is_canonical_and_receipt_is_tamper_evident() {
        let (report, receipt) = trends(&[
            obs(200.0, 0.0, "sensor-a"),
            obs(300.0, 30.0, "sensor-a"),
            obs(400.0, 60.0, "sensor-a"),
        ]);
        let a = boundary(PlssBoundaryDirection::Upper, 500.0, 60.0);
        let b = boundary(PlssBoundaryDirection::Upper, 550.0, 60.0);
        let (out_a, receipt_a) =
            project_plss_telemetry_boundaries(&report, &receipt, &[a.clone(), b.clone()])
                .unwrap();
        let (out_b, receipt_b) =
            project_plss_telemetry_boundaries(&report, &receipt, &[b, a]).unwrap();
        assert_eq!(out_a, out_b);
        assert_eq!(receipt_a, receipt_b);
        assert!(verify_plss_telemetry_boundary_receipt(&out_a, &receipt_a));
        let mut tampered = receipt_a.clone();
        tampered.result_digest_sha256 = "sha256:".to_string() + &"0".repeat(64);
        assert!(!verify_plss_telemetry_boundary_receipt(&out_a, &tampered));
    }

    #[test]
    fn empty_boundary_set_has_a_verifiable_receipt() {
        let (report, receipt) = trends(&[]);
        let (out, receipt_out) = project_plss_telemetry_boundaries(&report, &receipt, &[]).unwrap();
        assert_eq!(out.input_boundary_count, 0);
        assert_eq!(out.blocked_count, 0);
        assert!(verify_plss_telemetry_boundary_receipt(&out, &receipt_out));
    }
}
