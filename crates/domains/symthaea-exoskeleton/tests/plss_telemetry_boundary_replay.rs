// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent replay for SX-027's versioned boundary corpus.
//! This test intentionally reimplements the canonical hash encoding instead
//! of using production receipt-verification helpers to validate the receipt.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use symthaea_exoskeleton::plss_telemetry::{
    PlssTelemetryMetric, PlssTelemetryObservation, PlssTelemetrySourceKind, PlssTelemetryUnit,
};
use symthaea_exoskeleton::plss_telemetry_boundary::{
    project_plss_telemetry_boundaries, PlssBoundaryDirection, PlssBoundaryProjectionStatus,
    PlssTelemetryBoundary, PlssTelemetryBoundaryReceipt,
    PlssTelemetryBoundaryReport,
};
use symthaea_exoskeleton::plss_telemetry_trend::{
    analyze_plss_telemetry_trends_with_receipt, PlssTelemetryTrendConfig,
    PlssTelemetryTrendReceipt, PlssTelemetryTrendStatus, TrendEvidenceLevel,
};

const FIXTURE: &str = include_str!("fixtures/plss_telemetry_boundary_v1.json");

#[derive(Debug, Deserialize)]
struct Corpus {
    schema: String,
    source_revision: String,
    observations: Vec<PlssTelemetryObservation>,
    boundaries: Vec<PlssTelemetryBoundary>,
    expected: Vec<ExpectedProjection>,
    expected_counts: ExpectedCounts,
}

#[derive(Debug, Deserialize)]
struct ExpectedProjection {
    source_id: String,
    direction: PlssBoundaryDirection,
    threshold_value: f64,
    maximum_projection_horizon_s: f64,
    status: PlssBoundaryProjectionStatus,
    earliest_candidate_crossing_after_last_sample_s: Option<f64>,
    latest_candidate_crossing_after_last_sample_s: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct ExpectedCounts {
    boundary_count: usize,
    candidate_count: usize,
    direction_uncertain_count: usize,
    blocked_count: usize,
}

#[test]
fn replays_versioned_boundary_corpus_and_independently_verifies_receipts() {
    let corpus: Corpus = serde_json::from_str(FIXTURE).expect("boundary fixture must parse");
    assert_eq!(corpus.schema, "symthaea.plss.telemetry.boundary.v1");
    assert_eq!(corpus.observations.len(), 3);
    assert_eq!(corpus.boundaries.len(), 6);
    assert_eq!(corpus.expected.len(), 6);

    let (trend_report, trend_receipt) = analyze_plss_telemetry_trends_with_receipt(
        &corpus.observations,
        PlssTelemetryTrendConfig::simulation_reference(),
        &corpus.source_revision,
    ).expect("source trend report should be generated deterministically");

    let (report, receipt) = project_plss_telemetry_boundaries(
        &trend_report, &trend_receipt, &corpus.boundaries,
    ).expect("valid source receipt and boundaries should project");
    assert_eq!(report.evidence_level, TrendEvidenceLevel::SimulationOnly);
    assert_eq!(report.input_boundary_count, corpus.expected_counts.boundary_count);
    assert_eq!(report.candidate_count, corpus.expected_counts.candidate_count);
    assert_eq!(report.direction_uncertain_count, corpus.expected_counts.direction_uncertain_count);
    assert_eq!(report.blocked_count, corpus.expected_counts.blocked_count);

    for expected in &corpus.expected {
        let actual = report.projections.iter().find(|projection| {
            projection.source_id == expected.source_id
                && projection.direction == expected.direction
                && projection.threshold_value == expected.threshold_value
                && projection.maximum_projection_horizon_s == expected.maximum_projection_horizon_s
        }).expect("expected case should appear exactly in canonical output");
        assert_eq!(actual.status, expected.status, "case for {}", expected.source_id);
        assert_option_close(
            actual.earliest_candidate_crossing_after_last_sample_s,
            expected.earliest_candidate_crossing_after_last_sample_s,
        );
        assert_option_close(
            actual.latest_candidate_crossing_after_last_sample_s,
            expected.latest_candidate_crossing_after_last_sample_s,
        );
    }

    assert_eq!(receipt.input_digest_sha256, independent_input_digest(&report));
    assert_eq!(receipt.result_digest_sha256, independent_report_digest(&report));
    assert_eq!(receipt.receipt_digest_sha256, independent_receipt_digest(&receipt));
}

#[test]
fn input_permutation_preserves_report_and_receipt_and_tampering_is_detected() {
    let corpus: Corpus = serde_json::from_str(FIXTURE).expect("boundary fixture must parse");
    let (trend_a, source_receipt_a) = analyze_plss_telemetry_trends_with_receipt(
        &corpus.observations,
        PlssTelemetryTrendConfig::simulation_reference(),
        &corpus.source_revision,
    ).unwrap();
    let (report_a, receipt_a) =
        project_plss_telemetry_boundaries(&trend_a, &source_receipt_a, &corpus.boundaries).unwrap();

    let mut reversed_observations = corpus.observations.clone();
    reversed_observations.reverse();
    let mut reversed_boundaries = corpus.boundaries.clone();
    reversed_boundaries.reverse();
    let (trend_b, source_receipt_b) = analyze_plss_telemetry_trends_with_receipt(
        &reversed_observations,
        PlssTelemetryTrendConfig::simulation_reference(),
        &corpus.source_revision,
    ).unwrap();
    let (report_b, receipt_b) = project_plss_telemetry_boundaries(
        &trend_b,
        &source_receipt_b,
        &reversed_boundaries,
    )
    .unwrap();

    assert_eq!(trend_a, trend_b);
    assert_eq!(source_receipt_a, source_receipt_b);
    assert_eq!(report_a, report_b);
    assert_eq!(receipt_a, receipt_b);

    let mut altered_source_receipt = source_receipt_a.clone();
    altered_source_receipt.result_digest_sha256 = format!("sha256:{}", "0".repeat(64));
    assert!(project_plss_telemetry_boundaries(
        &trend_a, &altered_source_receipt, &corpus.boundaries,
    ).is_err());

    let mut altered_projection_report = report_a.clone();
    altered_projection_report.projections[0].status = PlssBoundaryProjectionStatus::MovingAway;
    assert_ne!(
        receipt_a.result_digest_sha256,
        independent_report_digest(&altered_projection_report),
    );
}

fn assert_option_close(actual: Option<f64>, expected: Option<f64>) {
    match (actual, expected) {
        (Some(actual), Some(expected)) => {
            assert!(
                (actual - expected).abs() < 1.0e-9,
                "{actual} != {expected}"
            );
        },
        (None, None) => {}
        (actual, expected) => panic!("Option mismatch: {actual:?} != {expected:?}"),
    }
}

struct IndependentHashWriter(Sha256);

impl IndependentHashWriter {
    fn new(domain: &str) -> Self {
        let mut value = Self(Sha256::new());
        value.string(domain);
        value
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

fn independent_input_digest(report: &PlssTelemetryBoundaryReport) -> String {
    let mut sorted: Vec<_> = report.projections.iter().collect();
    sorted.sort_by(|left, right| {
        left.metric.cmp(&right.metric)
            .then(left.source_kind.cmp(&right.source_kind))
            .then(left.source_id.cmp(&right.source_id))
            .then(left.direction.cmp(&right.direction))
            .then(left.threshold_value.total_cmp(&right.threshold_value))
            .then(left.threshold_revision.cmp(&right.threshold_revision))
            .then(left.maximum_projection_horizon_s.total_cmp(&right.maximum_projection_horizon_s))
    });
    let mut h = IndependentHashWriter::new("symthaea.plss.telemetry.boundary.input.v1");
    h.string(&report.source_revision);
    h.string(&report.source_trend_result_digest_sha256);
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

fn independent_report_digest(report: &PlssTelemetryBoundaryReport) -> String {
    let mut h = IndependentHashWriter::new("symthaea.plss.telemetry.boundary.result.v1");
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
    for p in &report.projections {
        h.string(metric_tag(p.metric));
        h.string(source_kind_tag(p.source_kind));
        h.string(&p.source_id);
        h.string(unit_tag(p.unit));
        h.string(direction_tag(p.direction));
        h.f64(p.threshold_value);
        h.string(&p.threshold_revision);
        h.f64(p.maximum_projection_horizon_s);
        h.string(option_trend_status_tag(p.lane_status));
        h.string(option_trend_status_tag(p.source_trend_status));
        h.string(projection_status_tag(p.status));
        h.option_f64(p.last_timestamp_s);
        h.option_f64(p.last_value);
        h.option_f64(p.last_uncertainty_abs);
        h.option_f64(p.endpoint_rate_per_s);
        h.option_f64(p.endpoint_rate_lower_bound_per_s);
        h.option_f64(p.endpoint_rate_upper_bound_per_s);
        h.option_f64(p.earliest_candidate_crossing_after_last_sample_s);
        h.option_f64(p.latest_candidate_crossing_after_last_sample_s);
    }
    h.finish()
}

fn independent_receipt_digest(receipt: &PlssTelemetryBoundaryReceipt) -> String {
    let mut h = IndependentHashWriter::new("symthaea.plss.telemetry.boundary.receipt.v1");
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
