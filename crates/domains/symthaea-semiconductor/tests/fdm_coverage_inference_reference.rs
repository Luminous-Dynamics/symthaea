#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_semiconductor::benchmark::{
    FDM_COVERAGE_ASSUMPTION_SCHEMA_V1, FDM_COVERAGE_POLICY_SCHEMA_V1,
    FDM_COVERAGE_SCHEMA_V1, FDM_COVERAGE_SCORE_SCHEMA_V1,
    FdmCoverageAssumptionEvidenceV1, FdmCoverageInputV1, FdmCoveragePolicyV1,
    FdmCoverageStatusV1, FdmQualifiedScoreObservationV1,
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
    LabelEvidenceFrontierV1, StreamingPredictionReceiptV1,
    evaluate_fdm_coverage_inference_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn json_sha<T: Serialize + ?Sized>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn absolute_error_score_digest() -> String {
    format!(
        "{:x}",
        Sha256::digest(b"abs(qualified_corrected_value - frozen_prediction_receipt.prediction_value)")
    )
}

fn prediction(sequence: u64, model: &str) -> StreamingPredictionReceiptV1 {
    let mut receipt = StreamingPredictionReceiptV1 {
        schema: "luminous.semi-bench.streaming-prediction-receipt.v1".into(),
        sequence,
        sample_id: format!("sample-{sequence}"),
        measurement_event_id: format!("event-{sequence}"),
        event_time_unix_seconds: sequence as i64 * 10,
        feature_available_at_unix_seconds: sequence as i64 * 10 + 1,
        prediction_at_unix_seconds: sequence as i64 * 10 + 2,
        clock_domain_id: "fab-clock".into(),
        global_model_artifact_sha256: model.into(),
        feature_schema_sha256: digest('c'),
        policy_sha256: digest('d'),
        context_key: Some("tool-a/ch-1".into()),
        feature_vector_sha256: digest('e'),
        selected_feature_indices: vec![0],
        selected_feature_vector: vec![1.0],
        global_baseline_prediction: 0.0,
        feature_adapter_prediction: 0.0,
        context_correction: 0.0,
        temporal_correction: 0.0,
        base_prediction: 0.0,
        prediction_value: 0.0,
        adapter_state_before_sha256: digest('f'),
        adapter_state_after_sha256: digest('f'),
        receipt_sha256: String::new(),
    };
    let mut canonical = receipt.clone();
    canonical.receipt_sha256.clear();
    receipt.receipt_sha256 = json_sha(&canonical);
    receipt
}

fn eligibility(
    sequence: u64,
    outcome: f64,
    payload: &str,
    status: LabelConsumptionEligibilityV1,
) -> LabelConsumptionEligibilityReportV1 {
    let mut report = LabelConsumptionEligibilityReportV1 {
        schema: "luminous.semi-bench.label-consumption-report.v1".into(),
        label_id: format!("label-{sequence}"),
        label_revision: 1,
        measurement_event_id: format!("event-{sequence}"),
        calibration_certificate_id: "certificate-1".into(),
        frontier: LabelEvidenceFrontierV1::CurrentAtK1,
        eligibility: status,
        downstream_consumption_allowed: status
            == LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        reasons: Vec::new(),
        label_sha256: payload.into(),
        measurement_sha256: digest('1'),
        qualified_corrected_value: outcome,
        qualified_standard_uncertainty_upper_bound: Some(0.1),
        qualification_report_sha256: digest('2'),
        requalification_report_sha256: digest('3'),
        requalification_frontier_sha256: digest('4'),
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

fn observation(sequence: u64, value: f64, model: &str) -> FdmQualifiedScoreObservationV1 {
    let payload = digest('5');
    let mut row = FdmQualifiedScoreObservationV1 {
        schema: FDM_COVERAGE_SCORE_SCHEMA_V1.into(),
        prediction_receipt: prediction(sequence, model),
        label_id: format!("label-{sequence}"),
        label_revision: 1,
        label_payload_sha256: payload.clone(),
        label_available_at_unix_seconds: sequence as i64 * 10 + 3,
        eligibility_report_available_at_unix_seconds: sequence as i64 * 10 + 4,
        delivered_at_unix_seconds: sequence as i64 * 10 + 5,
        clock_domain_id: "fab-clock".into(),
        eligibility_report: eligibility(
            sequence,
            value,
            &payload,
            LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        ),
        observation_sha256: String::new(),
    };
    let mut canonical = row.clone();
    canonical.observation_sha256.clear();
    row.observation_sha256 = json_sha(&canonical);
    row
}

fn assumptions(model: &str) -> FdmCoverageAssumptionEvidenceV1 {
    FdmCoverageAssumptionEvidenceV1 {
        schema: FDM_COVERAGE_ASSUMPTION_SCHEMA_V1.into(),
        evidence_id: "independent-process-assumption-record".into(),
        source_process_model_sha256: digest('a'),
        predictor_artifact_sha256: model.into(),
        score_definition_sha256: absolute_error_score_digest(),
        assumption_evidence_sha256: digest('b'),
        stationary_causal_innovation_representation_declared: true,
        iid_innovations_declared: true,
        fixed_predictor_independent_of_calibration_and_test_declared: true,
        continuous_score_cdf_declared: true,
        quantile_density_regularity_declared: true,
        no_pairwise_score_ties_declared: true,
        indicator_theta_upper_bounds: vec![1.0, 0.5, 0.25, 0.125],
        tail_power_constant: 0.01,
        tail_power_exponent: 4.0,
        quantile_density_lower_bound: Some(0.05),
        quantile_density_upper_bound: Some(1.0),
    }
}

fn input(
    calibration: Vec<FdmQualifiedScoreObservationV1>,
    test: Vec<FdmQualifiedScoreObservationV1>,
    block_length: usize,
) -> FdmCoverageInputV1 {
    FdmCoverageInputV1 {
        schema: FDM_COVERAGE_SCHEMA_V1.into(),
        run_id: "independent-fdm-reference-run".into(),
        cutoff_at_unix_seconds: 100_000,
        source_stream_sha256: digest('6'),
        calibration_manifest_sha256: digest('7'),
        test_manifest_sha256: digest('8'),
        policy: FdmCoveragePolicyV1 {
            schema: FDM_COVERAGE_POLICY_SCHEMA_V1.into(),
            policy_id: "paper-block-variance-reference".into(),
            revision: 1,
            target_coverage: 0.90,
            calibration_block_length: block_length,
            confidence_z: 1.959963984540054,
            minimum_calibration_scores: 8,
            minimum_test_scores: 4,
        },
        assumptions: assumptions(&digest('9')),
        calibration,
        test,
    }
}

#[test]
fn independent_block_means_oracle_reconstructs_paper_standard_error() {
    let model = digest('9');
    let all = (1..=40)
        .map(|sequence| {
            let value = if sequence % 13 == 0 { 1.5 } else { 0.5 };
            observation(sequence, value, &model)
        })
        .collect::<Vec<_>>();
    let input = input(all[..28].to_vec(), all[28..].to_vec(), 4);
    let report = evaluate_fdm_coverage_inference_v1(&input).unwrap();

    let n = input.calibration.len();
    let m = input.test.len();
    let mut scores = input
        .calibration
        .iter()
        .map(|row| {
            (row.eligibility_report.qualified_corrected_value
                - row.prediction_receipt.prediction_value)
                .abs()
        })
        .collect::<Vec<_>>();
    scores.sort_by(f64::total_cmp);
    let rank = ((n as f64 + 1.0) * input.policy.target_coverage).ceil() as usize;
    let q_hat = scores[rank - 1];

    let indicators = input
        .calibration
        .iter()
        .map(|row| {
            let score = (row.eligibility_report.qualified_corrected_value
                - row.prediction_receipt.prediction_value)
                .abs();
            if score <= q_hat { 1.0 } else { 0.0 }
        })
        .collect::<Vec<_>>();
    let ell = input.policy.calibration_block_length;
    let blocks = n / ell;
    let means = (0..blocks)
        .map(|block| {
            let slice = &indicators[block * ell..(block + 1) * ell];
            slice.iter().sum::<f64>() / ell as f64
        })
        .collect::<Vec<_>>();
    let mean = means.iter().sum::<f64>() / blocks as f64;
    let sigma_squared = ell as f64 / blocks as f64
        * means.iter().map(|value| (value - mean).powi(2)).sum::<f64>();
    let expected_se = sigma_squared.sqrt() * (1.0 / n as f64 + 1.0 / m as f64).sqrt();

    let test_covered = input
        .test
        .iter()
        .filter(|row| {
            (row.eligibility_report.qualified_corrected_value
                - row.prediction_receipt.prediction_value)
                .abs()
                <= q_hat
        })
        .count();

    assert_eq!(report.status, FdmCoverageStatusV1::Complete);
    assert_eq!(report.cutoff_rank, rank);
    assert!((report.calibration_cutoff.unwrap() - q_hat).abs() < 1.0e-12);
    assert_eq!(report.covered_test_count, test_covered);
    assert!((report.estimated_long_run_variance.unwrap() - sigma_squared).abs() < 1.0e-12);
    assert!((report.realized_coverage_standard_error.unwrap() - expected_se).abs() < 1.0e-12);
    assert_eq!(
        report.complete_calibration_block_count,
        blocks
    );
    assert_eq!(
        report.discarded_calibration_remainder,
        n % ell
    );
    assert!(!report.long_memory_supported);
    assert!(report.assumptions_declared_but_not_proven);
}

#[test]
fn incomplete_or_unqualified_stream_cannot_emit_paper_inference() {
    let model = digest('9');
    let all = (1..=20)
        .map(|sequence| observation(sequence, 0.5, &model))
        .collect::<Vec<_>>();
    let mut value = input(all[..12].to_vec(), all[12..].to_vec(), 3);
    value.test[1].eligibility_report.eligibility = LabelConsumptionEligibilityV1::Unknown;
    value.test[1].eligibility_report.downstream_consumption_allowed = false;
    let mut canonical = value.test[1].eligibility_report.clone();
    canonical.report_sha256.clear();
    value.test[1].eligibility_report.report_sha256 = json_sha(&canonical);
    let mut canonical_observation = value.test[1].clone();
    canonical_observation.observation_sha256.clear();
    value.test[1].observation_sha256 = json_sha(&canonical_observation);
    assert!(
        evaluate_fdm_coverage_inference_v1(&value).is_err(),
        "K1-unqualified outcomes must not be silently removed from the FDM stream"
    );
}
