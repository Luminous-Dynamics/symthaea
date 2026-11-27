#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_semiconductor::benchmark::{
    CoverageBacktestInputV1, CoverageBacktestStatusV1, CoverageEstimandV1,
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
    REALIZED_COVERAGE_BACKTEST_SCHEMA_V1, REALIZED_COVERAGE_OBSERVATION_SCHEMA_V1,
    REALIZED_COVERAGE_POLICY_SCHEMA_V1, RealizedCoverageObservationV1, RealizedCoveragePolicyV1,
    StreamingPredictionReceiptV1, evaluate_realized_coverage_backtest_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn json_sha<T: Serialize + ?Sized>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn prediction(sequence: u64, event: &str) -> StreamingPredictionReceiptV1 {
    let mut receipt = StreamingPredictionReceiptV1 {
        schema: "luminous.semi-bench.streaming-prediction-receipt.v1".into(),
        sequence,
        sample_id: format!("sample-{sequence}"),
        measurement_event_id: event.into(),
        event_time_unix_seconds: sequence as i64 * 10,
        feature_available_at_unix_seconds: sequence as i64 * 10 + 1,
        prediction_at_unix_seconds: sequence as i64 * 10 + 2,
        clock_domain_id: "fab-clock".into(),
        global_model_artifact_sha256: digest('a'),
        feature_schema_sha256: digest('b'),
        policy_sha256: digest('c'),
        context_key: Some("tool-a/ch-1".into()),
        feature_vector_sha256: digest('d'),
        selected_feature_indices: vec![0],
        selected_feature_vector: vec![1.0],
        global_baseline_prediction: 0.5,
        feature_adapter_prediction: 0.0,
        context_correction: 0.0,
        temporal_correction: 0.0,
        base_prediction: 0.5,
        prediction_value: 0.5,
        adapter_state_before_sha256: digest('e'),
        adapter_state_after_sha256: digest('e'),
        receipt_sha256: String::new(),
    };
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    receipt.receipt_sha256 = json_sha(&canonical);
    receipt
}

fn eligibility(
    sequence: u64,
    event: &str,
    outcome: f64,
) -> LabelConsumptionEligibilityReportV1 {
    let mut report = LabelConsumptionEligibilityReportV1 {
        schema: "luminous.semi-bench.label-consumption-report.v1".into(),
        label_id: format!("label-{sequence}"),
        label_revision: 1,
        measurement_event_id: event.into(),
        calibration_certificate_id: "certificate-1".into(),
        frontier: LabelEvidenceFrontierV1::CurrentAtK1,
        eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        downstream_consumption_allowed: true,
        reasons: Vec::new(),
        label_sha256: digest('f'),
        measurement_sha256: digest('a'),
        qualified_corrected_value: outcome,
        qualified_standard_uncertainty_upper_bound: Some(0.1),
        qualification_report_sha256: digest('b'),
        requalification_report_sha256: digest('c'),
        requalification_frontier_sha256: digest('d'),
        source_authenticity_proven: false,
        training_snapshot_mutated: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    report.report_sha256 = json_sha(&canonical);
    report
}

fn observation(sequence: u64, outcome: f64) -> RealizedCoverageObservationV1 {
    let event = format!("event-{sequence}");
    let mut row = RealizedCoverageObservationV1 {
        schema: REALIZED_COVERAGE_OBSERVATION_SCHEMA_V1.into(),
        prediction_receipt: prediction(sequence, &event),
        interval_lower: 0.0,
        interval_upper: 1.0,
        label_id: format!("label-{sequence}"),
        label_revision: 1,
        label_payload_sha256: digest('f'),
        outcome_value: outcome,
        label_available_at_unix_seconds: sequence as i64 * 10 + 3,
        eligibility_report_available_at_unix_seconds: sequence as i64 * 10 + 4,
        delivered_at_unix_seconds: sequence as i64 * 10 + 5,
        clock_domain_id: "fab-clock".into(),
        eligibility_report: eligibility(sequence, &event, outcome),
        observation_sha256: String::new(),
    };
    let mut canonical = row.clone();
    canonical.observation_sha256.clear();
    row.observation_sha256 = json_sha(&canonical);
    row
}

fn input(rows: Vec<RealizedCoverageObservationV1>, lag: usize) -> CoverageBacktestInputV1 {
    CoverageBacktestInputV1 {
        schema: REALIZED_COVERAGE_BACKTEST_SCHEMA_V1.into(),
        run_id: "hac-reference-run".into(),
        cutoff_at_unix_seconds: 10_000,
        source_stream_sha256: digest('e'),
        policy: RealizedCoveragePolicyV1 {
            schema: REALIZED_COVERAGE_POLICY_SCHEMA_V1.into(),
            policy_id: "hac-reference-policy".into(),
            revision: 1,
            target_coverage: 0.95,
            hac_max_lag: lag,
            confidence_z: 1.959963984540054,
            minimum_qualified_observations: 2,
            short_memory_assumption_declared: true,
        },
        observations: rows,
    }
}

#[test]
fn independent_hac_formula_matches_reported_bartlett_standard_error() {
    let misses = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
    let rows = misses
        .iter()
        .enumerate()
        .map(|(i, miss)| observation(i as u64 + 1, if *miss == 1.0 { 1.5 } else { 0.5 }))
        .collect::<Vec<_>>();
    let report = evaluate_realized_coverage_backtest_v1(&input(rows, 1)).unwrap();

    let mean = misses.iter().sum::<f64>() / misses.len() as f64;
    let centered = misses.map(|value| value - mean);
    let gamma_zero = centered.iter().map(|value| value * value).sum::<f64>()
        / centered.len() as f64;
    let gamma_one = centered
        .iter()
        .skip(1)
        .zip(centered.iter())
        .map(|(left, right)| left * right)
        .sum::<f64>()
        / centered.len() as f64;
    let independent_oracle_se = ((gamma_zero + gamma_one) / centered.len() as f64).sqrt();

    assert_eq!(report.status, CoverageBacktestStatusV1::Complete);
    assert_eq!(report.estimand, CoverageEstimandV1::QualifiedObservedStream);
    assert!(
        (report.hac_standard_error.unwrap() - independent_oracle_se).abs() < 1.0e-12
    );
    assert!(!report.iid_inference_claimed);
    assert!(!report.population_claim_authorized_by_qualified_subset_alone);
}

#[test]
fn positive_clustered_misses_produce_larger_hac_than_iid_standard_error() {
    let outcomes = [0.5, 0.5, 1.5, 1.5, 0.5, 0.5, 1.5, 1.5, 0.5, 0.5];
    let rows = outcomes
        .iter()
        .enumerate()
        .map(|(i, value)| observation(i as u64 + 1, *value))
        .collect::<Vec<_>>();
    let report = evaluate_realized_coverage_backtest_v1(&input(rows, 1)).unwrap();
    assert!(
        report.hac_standard_error.unwrap()
            > report.iid_standard_error_diagnostic.unwrap()
    );
}
