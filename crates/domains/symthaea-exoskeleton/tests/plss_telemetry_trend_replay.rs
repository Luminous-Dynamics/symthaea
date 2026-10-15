// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent replay for SX-026's versioned trend corpus.
//! This verifier duplicates the canonical SHA-256 byte encoding so it does not
//! validate production digests by calling the production digest helpers.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use symthaea_exoskeleton::plss_telemetry::{
    PlssTelemetryMetric, PlssTelemetryObservation, PlssTelemetrySourceKind,
};
use symthaea_exoskeleton::plss_telemetry_trend::{
    analyze_plss_telemetry_trends_with_receipt, PlssTelemetryTrendConfig,
    PlssTelemetryTrendLane, PlssTelemetryTrendReceipt, PlssTelemetryTrendReport,
    PlssTelemetryTrendStatus, TrendEvidenceLevel,
};

const FIXTURE: &str = include_str!("fixtures/plss_telemetry_trend_v1.json");
const SYNTHETIC_REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

#[derive(Debug, Deserialize)]
struct Corpus {
    schema_version: u32,
    config: PlssTelemetryTrendConfig,
    observations: Vec<PlssTelemetryObservation>,
    expected: Vec<ExpectedLane>,
    expected_counts: ExpectedCounts,
}

#[derive(Debug, Deserialize)]
struct ExpectedLane {
    metric: PlssTelemetryMetric,
    source_kind: PlssTelemetrySourceKind,
    status: PlssTelemetryTrendStatus,
    source_ids: Vec<String>,
    rates_per_s: Vec<Option<f64>>,
}

#[derive(Debug, Deserialize)]
struct ExpectedCounts {
    input_observation_count: usize,
    accepted_observation_count: usize,
    rejected_invalid_observations: usize,
    rejected_future_dated_observations: usize,
    excluded_out_of_window_observations: usize,
    excluded_low_confidence_observations: usize,
}

#[test]
fn replays_versioned_trend_corpus_and_independently_verifies_sha256_receipt() {
    let corpus: Corpus = serde_json::from_str(FIXTURE).expect("trend fixture JSON must parse");
    assert_eq!(corpus.schema_version, 1);
    assert_eq!(corpus.observations.len(), 21);
    assert_eq!(corpus.expected.len(), 12);

    let (report, receipt) = analyze_plss_telemetry_trends_with_receipt(
        &corpus.observations,
        corpus.config,
        SYNTHETIC_REVISION,
    )
    .expect("fixture has a valid trend policy and syntactically valid test revision");

    assert_eq!(report.evidence_level, TrendEvidenceLevel::SimulationOnly);
    assert_eq!(report.lanes.len(), 18);
    assert_eq!(report.input_observation_count, corpus.expected_counts.input_observation_count);
    assert_eq!(
        report.accepted_observation_count,
        corpus.expected_counts.accepted_observation_count
    );
    assert_eq!(
        report.rejected_invalid_observations,
        corpus.expected_counts.rejected_invalid_observations
    );
    assert_eq!(
        report.rejected_future_dated_observations,
        corpus.expected_counts.rejected_future_dated_observations
    );
    assert_eq!(
        report.excluded_out_of_window_observations,
        corpus.expected_counts.excluded_out_of_window_observations
    );
    assert_eq!(
        report.excluded_low_confidence_observations,
        corpus.expected_counts.excluded_low_confidence_observations
    );

    for expected in &corpus.expected {
        let lane = report
            .lane(expected.metric, expected.source_kind)
            .unwrap_or_else(|| {
                panic!(
                    "missing trend lane {:?}/{:?}",
                    expected.metric, expected.source_kind
                )
            });
        assert_eq!(lane.status, expected.status);
        assert_eq!(
            lane.source_trends.iter().map(|trend| trend.source_id.clone()).collect::<Vec<_>>(),
            expected.source_ids
        );
        assert_eq!(
            lane.source_trends.iter().map(|trend| trend.rate_per_s).collect::<Vec<_>>(),
            expected.rates_per_s
        );
        if lane.status == PlssTelemetryTrendStatus::Conflicted {
            for trend in &lane.source_trends {
                if trend.status == PlssTelemetryTrendStatus::Conflicted {
                    assert_eq!(trend.rate_per_s, None);
                    assert_eq!(trend.rate_lower_bound_per_s, None);
                    assert_eq!(trend.rate_upper_bound_per_s, None);
                }
            }
        }
    }

    assert_eq!(receipt.source_revision, SYNTHETIC_REVISION);
    assert_eq!(receipt.input_observation_count, 21);
    assert_eq!(receipt.input_digest_sha256, independently_hash_input(&corpus.observations));
    assert_eq!(receipt.result_digest_sha256, independently_hash_result(&report));
    assert_eq!(receipt.receipt_digest_sha256, independently_hash_receipt(&receipt));

    let mut reversed = corpus.observations.clone();
    reversed.reverse();
    let (replayed, replay_receipt) = analyze_plss_telemetry_trends_with_receipt(
        &reversed,
        corpus.config,
        SYNTHETIC_REVISION,
    )
    .unwrap();
    assert_eq!(replayed, report);
    assert_eq!(replay_receipt.input_digest_sha256, receipt.input_digest_sha256);
    assert_eq!(replay_receipt.result_digest_sha256, receipt.result_digest_sha256);
    assert_eq!(replay_receipt.receipt_digest_sha256, receipt.receipt_digest_sha256);
}

#[test]
fn empty_sparse_and_conflicted_series_have_no_invented_rate() {
    let corpus: Corpus = serde_json::from_str(FIXTURE).unwrap();
    let (report, _) = analyze_plss_telemetry_trends_with_receipt(
        &corpus.observations,
        corpus.config,
        SYNTHETIC_REVISION,
    )
    .unwrap();
    let empty = report.lane(
        PlssTelemetryMetric::OxygenPartialPressurePa,
        PlssTelemetrySourceKind::ModelEstimate,
    ).unwrap();
    assert_eq!(empty.status, PlssTelemetryTrendStatus::Missing);
    assert!(empty.source_trends.is_empty());

    let sparse = report.lane(
        PlssTelemetryMetric::OxygenPartialPressurePa,
        PlssTelemetrySourceKind::Instrument,
    ).unwrap();
    assert_eq!(sparse.status, PlssTelemetryTrendStatus::InsufficientEvidence);
    assert_eq!(sparse.source_trends[0].rate_per_s, None);

    let conflict = report.lane(
        PlssTelemetryMetric::CircuitPressurePa,
        PlssTelemetrySourceKind::Instrument,
    ).unwrap();
    assert_eq!(conflict.status, PlssTelemetryTrendStatus::Conflicted);
    assert_eq!(conflict.source_trends[0].rate_per_s, None);
}

struct VerifierHash(Sha256);

impl VerifierHash {
    fn new(domain: &str) -> Self {
        let mut h = Self(Sha256::new());
        h.string(domain);
        h
    }
    fn raw(&mut self, bytes: &[u8]) { self.0.update(bytes); }
    fn u64(&mut self, value: u64) { self.raw(&value.to_le_bytes()); }
    fn usize(&mut self, value: usize) { self.u64(value as u64); }
    fn f64(&mut self, value: f64) { self.u64(value.to_bits()); }
    fn string(&mut self, value: &str) { self.usize(value.len()); self.raw(value.as_bytes()); }
    fn option_f64(&mut self, value: Option<f64>) {
        match value {
            Some(number) => { self.raw(&[1]); self.f64(number); }
            None => self.raw(&[0]),
        }
    }
    fn finish(self) -> String {
        let digest = self.0.finalize();
        format!("sha256:{}", digest.iter().map(|byte| format!("{byte:02x}")).collect::<String>())
    }
}

fn independently_hash_input(observations: &[PlssTelemetryObservation]) -> String {
    let mut sorted = observations.to_vec();
    sorted.sort_by(|a, b| a.metric.cmp(&b.metric)
        .then(a.source_kind.cmp(&b.source_kind))
        .then(a.source_id.cmp(&b.source_id))
        .then(a.observed_at_s.total_cmp(&b.observed_at_s))
        .then(a.unit.cmp(&b.unit))
        .then(a.value.total_cmp(&b.value))
        .then(a.uncertainty_abs.total_cmp(&b.uncertainty_abs))
        .then(a.confidence.total_cmp(&b.confidence)));
    let mut h = VerifierHash::new("symthaea.plss.telemetry.trend.input.v1");
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

fn independently_hash_result(report: &PlssTelemetryTrendReport) -> String {
    let mut h = VerifierHash::new("symthaea.plss.telemetry.trend.result.v1");
    h.string(&format!("{:?}", report.evidence_level));
    h.usize(report.lanes.len());
    h.f64(report.config.window_start_s);
    h.f64(report.config.window_end_s);
    h.f64(report.config.maximum_gap_s);
    h.usize(report.config.minimum_samples);
    h.f64(report.config.minimum_confidence);
    h.f64(report.config.minimum_coverage_fraction);
    let tol = report.config.cross_source_tolerances;
    h.f64(tol.co2_partial_pressure_pa);
    h.f64(tol.oxygen_partial_pressure_pa);
    h.f64(tol.circuit_pressure_pa);
    h.f64(tol.gas_temperature_k);
    h.f64(tol.ventilation_flow_l_min);
    h.f64(tol.relative_humidity_percent);
    h.usize(report.input_observation_count);
    h.usize(report.accepted_observation_count);
    h.usize(report.rejected_invalid_observations);
    h.usize(report.rejected_future_dated_observations);
    h.usize(report.excluded_out_of_window_observations);
    h.usize(report.excluded_low_confidence_observations);
    for lane in &report.lanes { hash_lane(&mut h, lane); }
    h.finish()
}

fn hash_lane(h: &mut VerifierHash, lane: &PlssTelemetryTrendLane) {
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
        h.option_f64(source.first_uncertainty_abs);
        h.option_f64(source.last_uncertainty_abs);
        h.option_f64(source.maximum_observed_gap_s);
        h.option_f64(source.temporal_coverage_fraction);
        h.option_f64(source.change_over_window);
        h.option_f64(source.rate_per_s);
        h.option_f64(source.rate_lower_bound_per_s);
        h.option_f64(source.rate_upper_bound_per_s);
    }
}

fn independently_hash_receipt(receipt: &PlssTelemetryTrendReceipt) -> String {
    let mut h = VerifierHash::new("symthaea.plss.telemetry.trend.receipt.v1");
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
