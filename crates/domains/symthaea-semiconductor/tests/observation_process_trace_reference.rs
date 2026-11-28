#![forbid(unsafe_code)]

use std::collections::BTreeSet;

use symthaea_semiconductor::benchmark::{
    OBSERVATION_TRACE_SCHEMA_V1, ObservationControllerKindV1,
    ObservationProcessTraceInputV1, ObservationRoundTraceV1,
    OperationalAcquisitionEvidenceV1, OperationalAcquisitionStateV1,
    K1QualificationEvidenceV1, K1QualificationStateV1,
    ControllerFeedbackEvidenceV1, ControllerTrackingStateV1,
    IndependentAuditEvidenceV1, IndependentAuditStateV1, AuditRetentionStateV1,
    evaluate_observation_process_trace_v1,
};

fn digest(byte: char) -> String {
    std::iter::repeat_n(byte, 64).collect()
}

fn row(round: u64) -> ObservationRoundTraceV1 {
    ObservationRoundTraceV1 {
        round_id: round,
        sample_id: format!("sample-{round}"),
        measurement_event_id: format!("event-{round}"),
        event_time_unix_seconds: 10_000 + round as i64,
        prediction_at_unix_seconds: 10_001 + round as i64,
        round_time_unix_seconds: 10_001 + round as i64,
        clock_domain_id: "oracle-clock".into(),
        prediction_receipt_sha256: digest(char::from(b'a' + (round % 20) as u8)),
        model_artifact_sha256: digest('b'),
        interval_method_sha256: digest('c'),
        operational: OperationalAcquisitionEvidenceV1 {
            state: OperationalAcquisitionStateV1::Unselected,
            design_sha256: Some(digest('d')),
            inclusion_probability: Some(0.2),
            measured_at_unix_seconds: None,
            measurement_system_sha256: None,
            physical_measurement_cost: None,
            physical_measurement_latency_seconds: None,
        },
        k1: K1QualificationEvidenceV1 {
            state: K1QualificationStateV1::NotApplicable,
            label_id: None,
            label_revision: None,
            label_payload_sha256: None,
            qualification_report_sha256: None,
            measurement_event_id: None,
            corrected_value: None,
            standard_uncertainty_upper_bound: None,
            label_available_at_unix_seconds: None,
            report_available_at_unix_seconds: None,
            calibration_certificate_sha256: None,
            requalification_report_sha256: None,
            qualified_loss: None,
            no_promotion_reason: None,
        },
        controller: ControllerFeedbackEvidenceV1 {
            tracking_state: ControllerTrackingStateV1::NotSubmitted,
            policy_sha256: None,
            delay_contract_sha256: None,
            tracking_capacity: None,
            tracking_occupancy_after: None,
            feedback_delay_contract_steps: None,
            actual_feedback_delay_steps: None,
            feedback_round: None,
            feedback_received_at_unix_seconds: None,
            proxy_uniform_draw: None,
            proxy_delay_steps: None,
            proxy_survival_probability: None,
            prediction_receipt_sha256: None,
            feedback_receipt_sha256: None,
            update_receipt_sha256: None,
            controller_state_changed: false,
            theorem_assumptions_intact: false,
            immediate_trivial_loss: None,
        },
        independent_audit: IndependentAuditEvidenceV1 {
            state: IndependentAuditStateV1::NotSelected,
            retention: AuditRetentionStateV1::Neither,
            inclusion_probability: None,
            audit_selected_at_unix_seconds: None,
            audit_measured_at_unix_seconds: None,
            audit_measurement_cost: None,
            audit_measurement_receipt_sha256: None,
            audit_loss: None,
            unavailable_reason: None,
        },
    }
}

fn input(rounds: Vec<ObservationRoundTraceV1>) -> ObservationProcessTraceInputV1 {
    ObservationProcessTraceInputV1 {
        schema: OBSERVATION_TRACE_SCHEMA_V1.into(),
        trace_id: "independent-observation-oracle-v1".into(),
        population_manifest_sha256: digest('1'),
        source_ledger_sha256: digest('2'),
        clock_domain_id: "oracle-clock".into(),
        controller_kind: ObservationControllerKindV1::None,
        controller_policy_sha256: None,
        controller_delay_contract_sha256: None,
        controller_delay_steps: None,
        controller_tracking_capacity: None,
        operational_measurement_budget: 10_000,
        independent_audit_budget: 10_000,
        expected_population_round_ids: rounds.iter().map(|r| r.round_id).collect(),
        audit_design: None,
        rounds,
    }
}

/// Oracle computes the four-way population partition without calling the
/// production selection-risk estimator.
fn independent_partition_oracle(
    input: &ObservationProcessTraceInputV1,
) -> (BTreeSet<u64>, BTreeSet<u64>, BTreeSet<u64>, BTreeSet<u64>) {
    let mut eligible = BTreeSet::new();
    let mut unknown = BTreeSet::new();
    let mut blocked = BTreeSet::new();
    let mut unmeasured = BTreeSet::new();

    for row in &input.rounds {
        match row.k1.state {
            K1QualificationStateV1::Eligible => {
                assert_eq!(row.operational.state, OperationalAcquisitionStateV1::Measured);
                eligible.insert(row.round_id);
            }
            K1QualificationStateV1::Unknown => {
                assert_eq!(row.operational.state, OperationalAcquisitionStateV1::Measured);
                unknown.insert(row.round_id);
            }
            K1QualificationStateV1::Blocked => {
                assert_eq!(row.operational.state, OperationalAcquisitionStateV1::Measured);
                blocked.insert(row.round_id);
            }
            K1QualificationStateV1::NotApplicable => {
                assert_ne!(row.operational.state, OperationalAcquisitionStateV1::Measured);
                unmeasured.insert(row.round_id);
            }
        }
    }
    (eligible, unknown, blocked, unmeasured)
}

#[test]
fn independent_partition_and_partial_identification_oracle_match() {
    let mut qualified = row(1);
    qualified.operational.state = OperationalAcquisitionStateV1::Measured;
    qualified.operational.design_sha256 = Some(digest('3'));
    qualified.operational.measured_at_unix_seconds = Some(10_010);
    qualified.operational.measurement_system_sha256 = Some(digest('4'));
    qualified.operational.physical_measurement_cost = Some(1.0);
    qualified.k1 = K1QualificationEvidenceV1 {
        state: K1QualificationStateV1::Eligible,
        label_id: Some("label-1".into()),
        label_revision: Some(1),
        label_payload_sha256: Some(digest('5')),
        qualification_report_sha256: Some(digest('6')),
        measurement_event_id: Some("event-1".into()),
        corrected_value: Some(4.2),
        standard_uncertainty_upper_bound: Some(0.05),
        label_available_at_unix_seconds: Some(10_011),
        report_available_at_unix_seconds: Some(10_012),
        calibration_certificate_sha256: Some(digest('7')),
        requalification_report_sha256: Some(digest('8')),
        qualified_loss: Some(0.2),
        no_promotion_reason: None,
    };
    let mut unknown = qualified.clone();
    unknown.round_id = 2;
    unknown.sample_id = "sample-2".into();
    unknown.measurement_event_id = "event-2".into();
    unknown.prediction_receipt_sha256 = digest('9');
    unknown.operational.measured_at_unix_seconds = Some(10_011);
    unknown.k1.label_id = Some("label-2".into());
    unknown.k1.measurement_event_id = Some("event-2".into());
    unknown.k1.state = K1QualificationStateV1::Unknown;
    unknown.k1.corrected_value = None;
    unknown.k1.standard_uncertainty_upper_bound = None;
    unknown.k1.calibration_certificate_sha256 = None;
    unknown.k1.requalification_report_sha256 = None;
    unknown.k1.qualified_loss = None;

    let mut blocked = unknown.clone();
    blocked.round_id = 3;
    blocked.sample_id = "sample-3".into();
    blocked.measurement_event_id = "event-3".into();
    blocked.prediction_receipt_sha256 = digest('a');
    blocked.k1.label_id = Some("label-3".into());
    blocked.k1.measurement_event_id = Some("event-3".into());
    blocked.k1.state = K1QualificationStateV1::Blocked;

    let unmeasured = row(4);
    let input = input(vec![qualified, unknown, blocked, unmeasured]);
    let (eligible, unknown, blocked, unmeasured) = independent_partition_oracle(&input);
    assert_eq!(eligible.len(), 1);
    assert_eq!(unknown.len(), 1);
    assert_eq!(blocked.len(), 1);
    assert_eq!(unmeasured.len(), 1);

    let report = evaluate_observation_process_trace_v1(&input).unwrap();
    assert_eq!(report.population_count, 4);
    assert_eq!(report.k1_eligible_count, eligible.len());
    assert_eq!(report.k1_unknown_count, unknown.len());
    assert_eq!(report.k1_blocked_count, blocked.len());
    assert_eq!(report.selection_risk_report.qualified_count, eligible.len());
    assert_eq!(report.selection_risk_report.missing_count, 3);
    assert!((report.selection_risk_report.partial_identification_lower - 0.05).abs() < 1e-12);
    assert!((report.selection_risk_report.partial_identification_upper - 0.80).abs() < 1e-12);
    assert_eq!(report.selection_risk_report.population_point_estimate, None);
    assert!(!report.combined_ipw_estimate_emitted);
    assert!(!report.combined_coverage_theorem_emitted);
}

#[test]
fn independent_oracle_detects_inventory_omission() {
    let mut input = input(vec![row(1), row(2), row(3)]);
    input.expected_population_round_ids = vec![1, 2, 3, 4];
    assert!(evaluate_observation_process_trace_v1(&input).is_err());
}

#[test]
fn trace_order_permutation_does_not_change_population_oracle_result() {
    let first = input(vec![row(1), row(2), row(3)]);
    let first_report = evaluate_observation_process_trace_v1(&first).unwrap();
    let mut second = input(vec![row(3), row(1), row(2)]);
    second.expected_population_round_ids = vec![1, 2, 3];
    let second_report = evaluate_observation_process_trace_v1(&second).unwrap();
    assert_eq!(first_report.input_sha256, second_report.input_sha256);
    assert_eq!(first_report.report_sha256, second_report.report_sha256);
}
