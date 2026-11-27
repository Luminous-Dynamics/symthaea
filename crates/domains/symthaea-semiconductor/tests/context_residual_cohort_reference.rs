#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_semiconductor::benchmark::{
    CONTEXT_RESIDUAL_COHORT_SCHEMA_V1, CONTEXT_RESIDUAL_OBSERVATION_SCHEMA_V1,
    ContextResidualCohortInputV1, ContextResidualCohortStatusV1,
    ContextResidualObservationV1, LabelConsumptionEligibilityReportV1,
    LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1, StreamingPredictionReceiptV1,
    evaluate_context_residual_cohort_v1, seal_context_residual_cohort_v1,
    seal_context_residual_observation_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn json_digest<T: Serialize + ?Sized>(value: &T) -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(value).expect("test fixtures serialize"))
    )
}

fn make_observation(
    id: &str,
    context: &str,
    common_cause_group: &str,
    residual: f64,
) -> ContextResidualObservationV1 {
    let sequence = id.parse::<u64>().unwrap_or(0);
    let sample_id = format!("sample-{id}");
    let event_id = format!("event-{id}");
    let label_id = format!("label-{id}");
    let payload = digest('e');
    let prediction_value = 10.0;
    let label_value = prediction_value + residual;

    let mut prediction = StreamingPredictionReceiptV1 {
        schema: "luminous.semi-bench.streaming-prediction-receipt.v1".into(),
        sequence,
        sample_id,
        measurement_event_id: event_id.clone(),
        event_time_unix_seconds: 100,
        feature_available_at_unix_seconds: 101,
        prediction_at_unix_seconds: 102,
        clock_domain_id: "fab-clock".into(),
        global_model_artifact_sha256: digest('a'),
        feature_schema_sha256: digest('b'),
        policy_sha256: digest('c'),
        context_key: Some(context.into()),
        feature_vector_sha256: digest('d'),
        selected_feature_indices: vec![0],
        selected_feature_vector: vec![1.0],
        global_baseline_prediction: 10.0,
        feature_adapter_prediction: 0.0,
        context_correction: 0.0,
        temporal_correction: 0.0,
        base_prediction: prediction_value,
        prediction_value,
        adapter_state_before_sha256: digest('f'),
        adapter_state_after_sha256: digest('f'),
        receipt_sha256: String::new(),
    };
    let mut canonical_prediction = prediction.clone();
    canonical_prediction.receipt_sha256.clear();
    prediction.receipt_sha256 = json_digest(&canonical_prediction);

    let mut eligibility = LabelConsumptionEligibilityReportV1 {
        schema: "luminous.semi-bench.label-consumption-report.v1".into(),
        label_id: label_id.clone(),
        label_revision: 1,
        measurement_event_id: event_id,
        calibration_certificate_id: "cal-epoch-1-cert".into(),
        frontier: LabelEvidenceFrontierV1::CurrentAtK1,
        eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        downstream_consumption_allowed: true,
        reasons: Vec::new(),
        label_sha256: payload.clone(),
        measurement_sha256: digest('1'),
        qualified_corrected_value: label_value,
        qualified_standard_uncertainty_upper_bound: Some(0.05),
        qualification_report_sha256: digest('2'),
        requalification_report_sha256: digest('3'),
        requalification_frontier_sha256: digest('4'),
        source_authenticity_proven: false,
        training_snapshot_mutated: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    let mut canonical_eligibility = eligibility.clone();
    canonical_eligibility.report_sha256.clear();
    eligibility.report_sha256 = json_digest(&canonical_eligibility);

    let mut observation = ContextResidualObservationV1 {
        schema: CONTEXT_RESIDUAL_OBSERVATION_SCHEMA_V1.into(),
        observation_id: id.into(),
        prediction_receipt: prediction,
        label_id,
        label_revision: 1,
        label_payload_sha256: payload,
        label_value,
        label_available_at_unix_seconds: 110,
        eligibility_report_available_at_unix_seconds: 111,
        delivered_at_unix_seconds: 112,
        common_cause_group_id: common_cause_group.into(),
        common_cause_membership_receipt_sha256: digest('5'),
        calibration_epoch_id: "cal-epoch-1".into(),
        eligibility_report: eligibility,
        observation_sha256: String::new(),
    };
    observation.observation_sha256 =
        seal_context_residual_observation_v1(&observation).unwrap();
    observation
}

fn fixture(observations: Vec<ContextResidualObservationV1>) -> ContextResidualCohortInputV1 {
    let mut cohort = ContextResidualCohortInputV1 {
        schema: CONTEXT_RESIDUAL_COHORT_SCHEMA_V1.into(),
        cohort_id: "independent-oracle-cohort".into(),
        clock_domain_id: "fab-clock".into(),
        calibration_epoch_id: "cal-epoch-1".into(),
        window_start_unix_seconds: 90,
        window_end_unix_seconds: 105,
        closed_at_unix_seconds: 120,
        minimum_independent_groups: 2,
        inventory_complete: true,
        cohort_closed: true,
        common_cause_registry_sha256: digest('6'),
        observations,
        cohort_sha256: String::new(),
    };
    cohort.cohort_sha256 = seal_context_residual_cohort_v1(&cohort).unwrap();
    cohort
}

fn independent_median(values: &mut [f64]) -> f64 {
    values.sort_by(f64::total_cmp);
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        values[mid - 1] / 2.0 + values[mid] / 2.0
    } else {
        values[mid]
    }
}

#[test]
fn estimator_matches_independent_two_stage_group_median_oracle() {
    let observations = vec![
        make_observation("1", "tool-a/ch-1", "reference-a", 2.0),
        make_observation("2", "tool-a/ch-2", "reference-a", 2.2),
        make_observation("3", "tool-b/ch-1", "reference-b", 4.0),
        make_observation("4", "tool-c/ch-1", "reference-c", 2.1),
    ];
    let mut oracle_groups: std::collections::BTreeMap<String, Vec<f64>> =
        std::collections::BTreeMap::new();
    for observation in &observations {
        oracle_groups
            .entry(observation.common_cause_group_id.clone())
            .or_default()
            .push(
                observation.label_value
                    - observation.prediction_receipt.prediction_value,
            );
    }
    let mut group_medians = oracle_groups
        .values_mut()
        .map(|values| independent_median(values))
        .collect::<Vec<_>>();
    let expected_global = independent_median(&mut group_medians);

    let report = evaluate_context_residual_cohort_v1(&fixture(observations)).unwrap();
    assert_eq!(report.status, ContextResidualCohortStatusV1::Complete);
    assert!(
        (report.global_common_mode_residual.unwrap() - expected_global).abs() < 1.0e-12
    );
}

#[test]
fn observation_volume_inside_one_group_does_not_increase_its_global_weight() {
    let mut observations = Vec::new();
    for index in 0.. 9 {
        observations.push(make_observation(
            &format!("a-{index}"),
            &format!("tool-a/ch-{index}"),
            "reference-a",
            10.0,
        ));
    }
    observations.push(make_observation(
        "b-1",
        "tool-b/ch-1",
        "reference-b",
        0.0,
    ));
    let report = evaluate_context_residual_cohort_v1(&fixture(observations)).unwrap();
    assert_eq!(report.independent_common_cause_group_count, 2);
    assert!((report.global_common_mode_residual.unwrap() - 5.0).abs() < 1.0e-12);
}
