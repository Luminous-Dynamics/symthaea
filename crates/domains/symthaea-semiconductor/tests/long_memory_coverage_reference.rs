#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_semiconductor::benchmark::{
    FdmQualifiedScoreObservationV1, GaussianLinearLongMemoryEvidenceV1,
    LONG_MEMORY_ASSUMPTION_SCHEMA_V1, LONG_MEMORY_COVERAGE_POLICY_SCHEMA_V1,
    LONG_MEMORY_COVERAGE_SCHEMA_V1, LongMemoryCoverageInputV1,
    LongMemoryCoveragePolicyV1, LongMemoryCoverageStatusV1,
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
    LabelEvidenceFrontierV1, StreamingPredictionReceiptV1,
    evaluate_long_memory_coverage_inference_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn json_sha<T: Serialize + ?Sized>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn score_definition_sha() -> String {
    format!(
        "{:x}",
        Sha256::digest(b"abs(qualified_corrected_value - frozen_prediction_receipt.prediction_value)")
    )
}

fn xorshift64star(seed: u64, count: usize) -> Vec<u64> {
    let mut state = seed;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        state ^= state >> 12;
        state ^= state << 25;
        state ^= state >> 27;
        state = state.wrapping_mul(2_682_582_165_773_633_871);
        values.push(state);
    }
    values
}

fn gaussian_innovations(seed: u64, count: usize) -> Vec<f64> {
    let raw = xorshift64star(seed, 2 * count.div_ceil(2));
    let mut values = Vec::with_capacity(count);
    for pair in raw.chunks_exact(2) {
        let u1 = (pair[0] as f64 + 0.5) / (u64::MAX as f64 + 1.0);
        let u2 = (pair[1] as f64 + 0.5) / (u64::MAX as f64 + 1.0);
        let radius = (-2.0 * u1.ln()).sqrt();
        values.push(radius * (2.0 * std::f64::consts::PI * u2).cos());
        values.push(radius * (2.0 * std::f64::consts::PI * u2).sin());
    }
    values.truncate(count);
    values
}

fn synthetic_power_law_residuals(seed: u64, count: usize, beta: f64) -> Vec<f64> {
    let memory = 1_024_usize;
    let innovations = gaussian_innovations(seed, count + memory + 1);
    let coefficients = (0..=memory)
        .map(|lag| (lag as f64 + 1.0).powf(-beta))
        .collect::<Vec<_>>();
    (0..count)
        .map(|time| {
            coefficients
                .iter()
                .enumerate()
                .map(|(lag, coefficient)| coefficient * innovations[time + memory - lag])
                .sum::<f64>()
        })
        .collect()
}

fn prediction(sequence: u64, model: &str) -> StreamingPredictionReceiptV1 {
    let event_time = sequence as i64 * 10;
    let mut row = StreamingPredictionReceiptV1 {
        schema: "luminous.semi-bench.streaming-prediction-receipt.v1".into(),
        sequence,
        sample_id: format!("sample-{sequence}"),
        measurement_event_id: format!("event-{sequence}"),
        event_time_unix_seconds: event_time,
        feature_available_at_unix_seconds: event_time + 1,
        prediction_at_unix_seconds: event_time + 2,
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
    let mut canonical = row.clone();
    canonical.receipt_sha256.clear();
    row.receipt_sha256 = json_sha(&canonical);
    row
}

fn observation(
    sequence: u64,
    residual: f64,
    model: &str,
) -> FdmQualifiedScoreObservationV1 {
    let event_time = sequence as i64 * 10;
    let payload = digest('5');
    let mut report = LabelConsumptionEligibilityReportV1 {
        schema: "luminous.semi-bench.label-consumption-report.v1".into(),
        label_id: format!("label-{sequence}"),
        label_revision: 1,
        measurement_event_id: format!("event-{sequence}"),
        calibration_certificate_id: "certificate-1".into(),
        frontier: LabelEvidenceFrontierV1::CurrentAtK1,
        eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        downstream_consumption_allowed: true,
        reasons: Vec::new(),
        label_sha256: payload.clone(),
        measurement_sha256: digest('6'),
        qualified_corrected_value: residual,
        qualified_standard_uncertainty_upper_bound: Some(0.1),
        qualification_report_sha256: digest('7'),
        requalification_report_sha256: digest('8'),
        requalification_frontier_sha256: digest('9'),
        source_authenticity_proven: false,
        training_snapshot_mutated: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    let mut canonical_report = report.clone();
    canonical_report.report_sha256.clear();
    report.report_sha256 = json_sha(&canonical_report);

    let mut row = FdmQualifiedScoreObservationV1 {
        schema: "luminous.semi-bench.fdm-coverage-score.v1".into(),
        prediction_receipt: prediction(sequence, model),
        label_id: format!("label-{sequence}"),
        label_revision: 1,
        label_payload_sha256: payload,
        label_available_at_unix_seconds: event_time + 3,
        eligibility_report_available_at_unix_seconds: event_time + 4,
        delivered_at_unix_seconds: event_time + 5,
        clock_domain_id: "fab-clock".into(),
        eligibility_report: report,
        observation_sha256: String::new(),
    };
    let mut canonical = row.clone();
    canonical.observation_sha256.clear();
    row.observation_sha256 = json_sha(&canonical);
    row
}

fn assumptions(model: &str) -> GaussianLinearLongMemoryEvidenceV1 {
    GaussianLinearLongMemoryEvidenceV1 {
        schema: LONG_MEMORY_ASSUMPTION_SCHEMA_V1.into(),
        evidence_id: "synthetic-gaussian-linear-residual-model".into(),
        source_process_model_sha256: digest('a'),
        predictor_artifact_sha256: model.into(),
        absolute_error_score_definition_sha256: score_definition_sha(),
        assumption_evidence_sha256: digest('b'),
        gaussian_iid_innovations_declared: true,
        zero_mean_residuals_declared: true,
        regularly_varying_coefficients_declared: true,
        fixed_predictor_independent_of_calibration_and_test_declared: true,
        leading_coefficient_abs_lower_bound: 0.1,
        coefficient_scale_lower_bound: 0.5,
        coefficient_refinement_exponent: 0.5,
    }
}

fn input(
    calibration: Vec<FdmQualifiedScoreObservationV1>,
    test: Vec<FdmQualifiedScoreObservationV1>,
    model: &str,
) -> LongMemoryCoverageInputV1 {
    LongMemoryCoverageInputV1 {
        schema: LONG_MEMORY_COVERAGE_SCHEMA_V1.into(),
        run_id: "independent-long-memory-reference".into(),
        cutoff_at_unix_seconds: 1_000_000,
        source_stream_sha256: digest('1'),
        calibration_manifest_sha256: digest('2'),
        test_manifest_sha256: digest('3'),
        policy: LongMemoryCoveragePolicyV1 {
            schema: LONG_MEMORY_COVERAGE_POLICY_SCHEMA_V1.into(),
            policy_id: "appendix-c-reference-policy".into(),
            revision: 1,
            target_coverage: 0.90,
            block_exponent_zeta: 2.0 / 3.0,
            significance_level: 0.05,
            minimum_calibration_count: 64,
            minimum_test_count: 64,
            minimum_paired_block_count: 20,
        },
        assumptions: assumptions(model),
        calibration,
        test,
    }
}

fn independent_quantile(values: &[f64], p: f64) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(f64::total_cmp);
    let rank = ((p * ordered.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(ordered.len() - 1);
    ordered[rank]
}

fn independent_normalization(n: usize, beta: f64) -> f64 {
    let log_n = (n as f64).ln();
    let d = 3.0 - 4.0 * beta;
    let integral = if d.abs() < 1.0e-8 {
        log_n
    } else {
        (d * log_n).exp_m1() / d
    };
    ((1.0 + integral) / n as f64).sqrt()
}

#[test]
fn unqualified_k1_label_cannot_enter_the_long_memory_stream() {
    let n = 256;
    let m = 256;
    let model = digest('9');
    let residuals = synthetic_power_law_residuals(31, n + m, 0.85);
    let rows = residuals
        .iter()
        .enumerate()
        .map(|(index, value)| observation(index as u64 + 1, *value, &model))
        .collect::<Vec<_>>();
    let mut value = input(rows[..n].to_vec(), rows[n..].to_vec(), &model);
    value.test[2].eligibility_report.eligibility = LabelConsumptionEligibilityV1::Unknown;
    value.test[2].eligibility_report.downstream_consumption_allowed = false;
    let mut canonical_report = value.test[2].eligibility_report.clone();
    canonical_report.report_sha256.clear();
    value.test[2].eligibility_report.report_sha256 = json_sha(&canonical_report);
    let mut canonical_observation = value.test[2].clone();
    canonical_observation.observation_sha256.clear();
    value.test[2].observation_sha256 = json_sha(&canonical_observation);
    assert!(evaluate_long_memory_coverage_inference_v1(&value).is_err());
}

#[test]
fn non_adjacent_calibration_and_test_windows_are_rejected() {
    let n = 256;
    let m = 256;
    let model = digest('9');
    let residuals = synthetic_power_law_residuals(37, n + m, 0.85);
    let rows = residuals
        .iter()
        .enumerate()
        .map(|(index, value)| observation(index as u64 + 1, *value, &model))
        .collect::<Vec<_>>();
    let mut value = input(rows[..n].to_vec(), rows[n..].to_vec(), &model);
    value.test[0].prediction_receipt.sequence += 2;
    let mut canonical_prediction = value.test[0].prediction_receipt.clone();
    canonical_prediction.receipt_sha256.clear();
    value.test[0].prediction_receipt.receipt_sha256 = json_sha(&canonical_prediction);
    let mut canonical_observation = value.test[0].clone();
    canonical_observation.observation_sha256.clear();
    value.test[0].observation_sha256 = json_sha(&canonical_observation);
    assert!(evaluate_long_memory_coverage_inference_v1(&value).is_err());
}

#[test]
fn insufficient_paired_block_count_returns_no_distributional_claim() {
    let n = 256;
    let model = digest('9');
    let residuals = synthetic_power_law_residuals(41, 2 * n, 0.85);
    let rows = residuals
        .iter()
        .enumerate()
        .map(|(index, value)| observation(index as u64 + 1, *value, &model))
        .collect::<Vec<_>>();
    let mut value = input(rows[..n].to_vec(), rows[n..].to_vec(), &model);
    value.policy.minimum_paired_block_count = 200;
    let report = evaluate_long_memory_coverage_inference_v1(&value).unwrap();
    assert_eq!(report.status, LongMemoryCoverageStatusV1::InsufficientInput);
    assert!(!report.long_memory_supported);
    assert_eq!(report.block_distribution_lower_quantile, None);
}

#[test]
fn permuted_identical_frozen_stream_replays_to_the_same_block_distribution_receipt() {
    let n = 256;
    let m = 256;
    let model = digest('9');
    let residuals = synthetic_power_law_residuals(19, n + m, 0.85);
    let rows = residuals
        .iter()
        .enumerate()
        .map(|(index, value)| observation(index as u64 + 1, *value, &model))
        .collect::<Vec<_>>();
    let mut value = input(rows[..n].to_vec(), rows[n..].to_vec(), &model);
    let first = evaluate_long_memory_coverage_inference_v1(&value).unwrap();
    value.calibration.reverse();
    value.test.reverse();
    let second = evaluate_long_memory_coverage_inference_v1(&value).unwrap();
    assert_eq!(first.report_sha256, second.report_sha256);
    assert_eq!(
        first.normalized_block_distribution_sha256,
        second.normalized_block_distribution_sha256
    );
}

#[test]
fn estimated_beta_outside_the_stationary_long_memory_range_is_unknown_not_clipped() {
    let n = 256;
    let model = digest('9');
    let calibration = (1..=n as u64)
        .map(|sequence| observation(sequence, 1.0, &model))
        .collect::<Vec<_>>();
    let test = ((n + 1)..=(2 * n) as u64)
        .map(|sequence| observation(sequence, 1.0, &model))
        .collect::<Vec<_>>();
    let input = input(calibration, test, &model);
    let report = evaluate_long_memory_coverage_inference_v1(&input).unwrap();
    assert_eq!(report.beta_estimate, Some(0.5));
    assert_eq!(report.status, LongMemoryCoverageStatusV1::AssumptionsNotSupported);
    assert_eq!(report.normalization_r_n, None);
    assert!(!report.long_memory_supported);
}

#[test]
fn appendix_c_block_cdf_matches_independent_two_scale_and_paired_block_oracle() {
    let n = 256;
    let m = 256;
    let model = digest('9');
    let residuals = synthetic_power_law_residuals(3, n + m, 0.85);
    let rows = residuals
        .iter()
        .enumerate()
        .map(|(index, value)| observation(index as u64 + 1, *value, &model))
        .collect::<Vec<_>>();
    let input = input(rows[..n].to_vec(), rows[n..].to_vec(), &model);
    let report = evaluate_long_memory_coverage_inference_v1(&input).unwrap();
    assert_eq!(report.status, LongMemoryCoverageStatusV1::Complete);

    let h = (n as f64).sqrt().floor() as usize;
    let q = |length: usize| {
        (0..=n - length)
            .map(|start| residuals[start..start + length].iter().sum::<f64>().powi(2))
            .sum::<f64>()
            / (n - length + 1) as f64
    };
    let q_h = q(h);
    let q_2h = q(2 * h);
    let beta = 1.5 - (q_2h / q_h).ln() / (2.0 * std::f64::consts::LN_2);
    assert!((report.beta_estimate.unwrap() - beta).abs() < 1.0e-12);

    let b = (n as f64).powf(2.0 / 3.0).floor() as usize;
    let ell_b = b * m / n;
    let rank = ((b as f64 + 1.0) * input.policy.target_coverage).ceil() as usize;
    let r_b = independent_normalization(b, beta);
    let mut normalized = Vec::new();

    let calibration_scores = residuals[..n].iter().map(|x| x.abs()).collect::<Vec<_>>();
    for start in 0..=n - b - ell_b {
        let mut block_cal = calibration_scores[start..start + b].to_vec();
        block_cal.sort_by(f64::total_cmp);
        let cutoff = block_cal[rank - 1];
        let block_test = &calibration_scores[start + b..start + b + ell_b];
        let covered = block_test.iter().filter(|score| **score <= cutoff).count();
        let coverage = covered as f64 / ell_b as f64;
        normalized.push((coverage - input.policy.target_coverage) / r_b);
    }

    let low = independent_quantile(&normalized, 0.025);
    let high = independent_quantile(&normalized, 0.975);
    let r_n = independent_normalization(n, beta);
    let mut full_cal = calibration_scores.clone();
    full_cal.sort_by(f64::total_cmp);
    let full_rank = ((n as f64 + 1.0) * input.policy.target_coverage).ceil() as usize;
    let full_cutoff = full_cal[full_rank - 1];
    let test_scores = residuals[n..].iter().map(|x| x.abs()).collect::<Vec<_>>();
    let covered = test_scores.iter().filter(|score| **score <= full_cutoff).count();
    let observed = covered as f64 / m as f64;
    let statistic = (observed - input.policy.target_coverage) / r_n;

    assert_eq!(report.beta_estimate.unwrap(), beta);
    assert!((report.normalization_r_b.unwrap() - r_b).abs() < 1.0e-12);
    assert!((report.normalization_r_n.unwrap() - r_n).abs() < 1.0e-12);
    assert!((report.block_distribution_lower_quantile.unwrap() - low).abs() < 1.0e-12);
    assert!((report.block_distribution_upper_quantile.unwrap() - high).abs() < 1.0e-12);
    assert!((report.normalized_full_window_statistic.unwrap() - statistic).abs() < 1.0e-12);
    assert_eq!(report.paired_block_count, normalized.len());
    assert!(!report.iid_inference_claimed);
    assert!(!report.population_claim_authorized_by_qualified_stream_alone);
    assert!(report.assumptions_declared_but_not_proven);
}
