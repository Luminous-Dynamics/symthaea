// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent replay of the versioned PLSS telemetry adversarial corpus.
//!
//! The receipt fingerprint logic here is deliberately reimplemented separately
//! from the production module so a shared helper cannot make the check tautological.
//! The synthetic source revision below checks binding/format semantics only; it
//! is not represented as the revision of this test binary or as provenance.

use serde::Deserialize;
use sha2::{Digest, Sha256};
use symthaea_exoskeleton::plss_telemetry::{
    evaluate_plss_telemetry_with_receipt, fuse_plss_telemetry, FusedPlssTelemetryMetric,
    PlssTelemetryConfig, PlssTelemetryError, PlssTelemetryEvidenceLevel, PlssTelemetryFusionReport,
    PlssTelemetryMetric, PlssTelemetryObservation, PlssTelemetryReceipt, PlssTelemetrySourceKind,
    PlssTelemetryStatus, PLSS_TELEMETRY_METRIC_COUNT,
};

const CORPUS_JSON: &str = include_str!("fixtures/plss_telemetry_v1.json");
const SYNTHETIC_REVISION: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";

#[derive(Debug, Deserialize)]
struct Corpus {
    schema_version: u32,
    evaluated_at_s: f64,
    observations: Vec<PlssTelemetryObservation>,
    expected: Vec<ExpectedMetric>,
}

#[derive(Debug, Deserialize)]
struct ExpectedMetric {
    metric: PlssTelemetryMetric,
    source_kind: PlssTelemetrySourceKind,
    status: PlssTelemetryStatus,
    accepted_sources: usize,
    #[serde(default)]
    requires_no_point_estimate: bool,
}

#[test]
fn replay_versioned_corpus_and_independently_verify_receipt() {
    let corpus: Corpus = serde_json::from_str(CORPUS_JSON).expect("fixture JSON must parse");
    assert_eq!(corpus.schema_version, 1);
    assert_eq!(corpus.observations.len(), 12);
    assert_eq!(corpus.expected.len(), 11);

    let config = PlssTelemetryConfig::simulation_reference();
    let report = fuse_plss_telemetry(&corpus.observations, corpus.evaluated_at_s, config)
        .expect("fixed corpus has a valid fusion config and evaluation time");

    assert_eq!(report.evidence_level, PlssTelemetryEvidenceLevel::SimulationOnly);
    assert_eq!(
        report.metrics.len(),
        PLSS_TELEMETRY_METRIC_COUNT * PlssTelemetrySourceKind::ALL.len()
    );
    for metric in PlssTelemetryMetric::ALL {
        for source_kind in PlssTelemetrySourceKind::ALL {
            assert!(
                report.metric(metric, source_kind).is_some(),
                "missing explicit output lane for {metric:?}/{source_kind:?}"
            );
        }
    }
    for expected in &corpus.expected {
        let actual = report
            .metric(expected.metric, expected.source_kind)
            .unwrap_or_else(|| {
                panic!(
                    "missing result for {:?}/{:?}",
                    expected.metric, expected.source_kind
                )
            });
        assert_eq!(
            actual.status, expected.status,
            "unexpected status for {:?}/{:?}",
            expected.metric, expected.source_kind
        );
        assert_eq!(
            actual.accepted_source_count, expected.accepted_sources,
            "accepted source count mismatch for {:?}/{:?}",
            expected.metric, expected.source_kind
        );
        if expected.requires_no_point_estimate {
            assert_eq!(actual.value, None, "conflicts must not produce a point estimate");
            assert_eq!(actual.lower_bound, None);
            assert_eq!(actual.upper_bound, None);
        }
    }

    let (report_from_receipt_api, receipt) = evaluate_plss_telemetry_with_receipt(
        &corpus.observations,
        corpus.evaluated_at_s,
        config,
        SYNTHETIC_REVISION,
    )
    .expect("synthetic revision is only a test fixture for revision-binding semantics");
    assert_eq!(report, report_from_receipt_api);
    assert_eq!(receipt.source_revision, SYNTHETIC_REVISION);
    assert_eq!(receipt.input_observation_count, 12);
    // Locked expectations make accidental corpus drift visible in review.
    assert_eq!(receipt.accepted_source_metric_pairs, 7);
    assert_eq!(receipt.rejected_invalid_observations, 1);
    assert_eq!(receipt.rejected_future_dated_observations, 1);
    assert_eq!(receipt.excluded_stale_observations, 1);
    assert_eq!(receipt.excluded_low_confidence_observations, 1);

    // Independent verifier: don't trust the production digest helper to verify
    // its own outputs. Recompute the defined canonical byte stream here.
    assert_eq!(
        receipt.input_digest_sha256,
        independent_input_digest(&corpus.observations)
    );
    assert_eq!(receipt.result_digest_sha256, independent_result_digest(&report));
    assert_eq!(receipt.receipt_digest_sha256, independent_receipt_digest(&receipt));

    // The corpus fingerprint must not depend on JSON/collector order.
    let mut reversed = corpus.observations.clone();
    reversed.reverse();
    let (replayed, replay_receipt) = evaluate_plss_telemetry_with_receipt(
        &reversed,
        corpus.evaluated_at_s,
        config,
        SYNTHETIC_REVISION,
    )
    .unwrap();
    assert_eq!(replayed, report);
    assert_eq!(replay_receipt.input_digest_sha256, receipt.input_digest_sha256);
    assert_eq!(replay_receipt.result_digest_sha256, receipt.result_digest_sha256);
    assert_eq!(replay_receipt.receipt_digest_sha256, receipt.receipt_digest_sha256);
}

#[test]
fn corpus_has_negative_controls_for_every_non_nominal_status() {
    let corpus: Corpus = serde_json::from_str(CORPUS_JSON).unwrap();
    let statuses: Vec<PlssTelemetryStatus> =
        corpus.expected.iter().map(|expected| expected.status).collect();

    for required in [
        PlssTelemetryStatus::Missing,
        PlssTelemetryStatus::Conflicted,
        PlssTelemetryStatus::FutureDated,
        PlssTelemetryStatus::Stale,
        PlssTelemetryStatus::LowConfidence,
        PlssTelemetryStatus::Invalid,
    ] {
        assert!(statuses.contains(&required), "corpus lacks {required:?} control");
    }
}

#[test]
fn malformed_revision_is_rejected_before_a_receipt_is_emitted() {
    assert_eq!(
        evaluate_plss_telemetry_with_receipt(
            &[],
            100.0,
            PlssTelemetryConfig::simulation_reference(),
            "not-a-revision",
        ),
        Err(PlssTelemetryError::InvalidSourceRevision)
    );
}

// Minimal independent SHA-256 byte-stream verifier. All integers and float bit
// patterns are little-endian u64; strings are UTF-8 with a u64 byte-length prefix.
struct VerifierHash(Sha256);

impl VerifierHash {
    fn new(domain: &str) -> Self {
        let mut this = Self(Sha256::new());
        this.write_str(domain);
        this
    }

    fn write_raw(&mut self, bytes: &[u8]) {
        self.0.update(bytes);
    }

    fn write_u64(&mut self, value: u64) {
        self.write_raw(&value.to_le_bytes());
    }

    fn write_usize(&mut self, value: usize) {
        self.write_u64(value as u64);
    }

    fn write_f64(&mut self, value: f64) {
        self.write_u64(value.to_bits());
    }

    fn write_str(&mut self, value: &str) {
        self.write_usize(value.len());
        self.write_raw(value.as_bytes());
    }

    fn write_option_f64(&mut self, value: Option<f64>) {
        match value {
            Some(number) => {
                self.write_raw(&[1]);
                self.write_f64(number);
            }
            None => self.write_raw(&[0]),
        }
    }

    fn finish(self) -> String {
        let digest = self.0.finalize();
        let encoded: String = digest.iter().map(|byte| format!("{byte:02x}")).collect();
        format!("sha256:{encoded}")
    }
}

fn independent_input_digest(observations: &[PlssTelemetryObservation]) -> String {
    let mut sorted = observations.to_vec();
    sorted.sort_by(|left, right| {
        left.metric
            .cmp(&right.metric)
            .then(left.source_kind.cmp(&right.source_kind))
            .then(left.source_id.cmp(&right.source_id))
            .then(left.observed_at_s.total_cmp(&right.observed_at_s))
            .then(left.unit.cmp(&right.unit))
            .then(left.value.total_cmp(&right.value))
            .then(left.uncertainty_abs.total_cmp(&right.uncertainty_abs))
            .then(left.confidence.total_cmp(&right.confidence))
    });

    let mut hash = VerifierHash::new("symthaea.plss.telemetry.input.v1");
    hash.write_usize(sorted.len());
    for observation in &sorted {
        hash.write_str(&format!("{:?}", observation.metric));
        hash.write_str(&format!("{:?}", observation.unit));
        hash.write_f64(observation.value);
        hash.write_f64(observation.uncertainty_abs);
        hash.write_f64(observation.observed_at_s);
        hash.write_f64(observation.confidence);
        hash.write_str(&observation.source_id);
        hash.write_str(&format!("{:?}", observation.source_kind));
    }
    hash.finish()
}

fn independent_result_digest(report: &PlssTelemetryFusionReport) -> String {
    let mut hash = VerifierHash::new("symthaea.plss.telemetry.result.v1");
    hash.write_f64(report.evaluated_at_s);
    hash.write_str(&format!("{:?}", report.evidence_level));
    hash.write_f64(report.config.maximum_observation_age_s);
    hash.write_f64(report.config.minimum_confidence);
    let tolerances = report.config.disagreement_tolerances;
    hash.write_f64(tolerances.co2_partial_pressure_pa);
    hash.write_f64(tolerances.oxygen_partial_pressure_pa);
    hash.write_f64(tolerances.circuit_pressure_pa);
    hash.write_f64(tolerances.gas_temperature_k);
    hash.write_f64(tolerances.ventilation_flow_l_min);
    hash.write_f64(tolerances.relative_humidity_percent);
    hash.write_usize(report.rejected_invalid_observations);
    hash.write_usize(report.rejected_future_dated_observations);
    hash.write_usize(report.excluded_stale_observations);
    hash.write_usize(report.excluded_low_confidence_observations);
    hash.write_usize(report.metrics.len());
    for metric in &report.metrics {
        hash_metric(&mut hash, metric);
    }
    hash.finish()
}

fn hash_metric(hash: &mut VerifierHash, metric: &FusedPlssTelemetryMetric) {
    hash.write_str(&format!("{:?}", metric.metric));
    hash.write_str(&format!("{:?}", metric.unit));
    hash.write_str(&format!("{:?}", metric.source_kind));
    hash.write_str(&format!("{:?}", metric.status));
    hash.write_option_f64(metric.value);
    hash.write_option_f64(metric.lower_bound);
    hash.write_option_f64(metric.upper_bound);
    hash.write_option_f64(metric.confidence);
    hash.write_usize(metric.accepted_source_count);
    hash.write_usize(metric.source_ids.len());
    for source_id in &metric.source_ids {
        hash.write_str(source_id);
    }
}

fn independent_receipt_digest(receipt: &PlssTelemetryReceipt) -> String {
    let mut hash = VerifierHash::new("symthaea.plss.telemetry.receipt.v1");
    hash.write_str(&receipt.source_revision);
    hash.write_usize(receipt.input_observation_count);
    hash.write_usize(receipt.accepted_source_metric_pairs);
    hash.write_usize(receipt.rejected_invalid_observations);
    hash.write_usize(receipt.rejected_future_dated_observations);
    hash.write_usize(receipt.excluded_stale_observations);
    hash.write_usize(receipt.excluded_low_confidence_observations);
    hash.write_str(&receipt.input_digest_sha256);
    hash.write_str(&receipt.result_digest_sha256);
    hash.finish()
}
