#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};
use symthaea_semiconductor::benchmark::{
    FeedbackTrackingRecordV1, FeedbackTrackingStateV1, K1QualificationRecordV1,
    K1QualificationStateV1, ObservationAssumptionStatusV1, ObservationProcessAssumptionsV1,
    OBSERVATION_PROCESS_SCHEMA_V1, ObservationProcessInputV1, ObservationProcessRoundV1,
    ObservationRiskControllerV1, PhysicalAcquisitionRecordV1,
    PhysicalAcquisitionStateV1, PhysicalSelectionDesignKindV1, evaluate_observation_process_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn sha256_json<T: Serialize + ?Sized>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn row(
    round_id: u64,
    acquisition: PhysicalAcquisitionStateV1,
    qualification: K1QualificationStateV1,
    loss: Option<f64>,
) -> ObservationProcessRoundV1 {
    let measured = acquisition == PhysicalAcquisitionStateV1::Measured;
    let has_label = measured && qualification != K1QualificationStateV1::NotApplicable;
    ObservationProcessRoundV1 {
        round_id,
        sample_id: format!("oracle-sample-{round_id}"),
        measurement_event_id: format!("oracle-event-{round_id}"),
        prediction_receipt_sha256: digest('a'),
        interval_receipt_sha256: digest('b'),
        model_artifact_sha256: digest('c'),
        event_time_unix_seconds: 10_000 + round_id as i64 * 10,
        prediction_time_unix_seconds: 10_001 + round_id as i64 * 10,
        clock_domain_id: "oracle-clock".into(),
        context_stratum_id: Some(format!("oracle-stratum-{}", round_id % 2)),
        acquisition: PhysicalAcquisitionRecordV1 {
            state: acquisition,
            selection_design_kind: PhysicalSelectionDesignKindV1::DeterministicOrAdaptive,
            selection_design_sha256: None,
            inclusion_probability: None,
            acquisition_cost_units: measured.then_some(1.0),
            selected_at_round: measured.then_some(round_id),
            measured_at_unix_seconds: measured.then_some(10_002 + round_id as i64 * 10),
            operational_measurement_method_sha256: measured.then(|| digest('0')),
        },
        qualification: K1QualificationRecordV1 {
            state: qualification,
            label_id: has_label.then(|| format!("oracle-label-{round_id}")),
            label_revision: has_label.then_some(1),
            label_payload_sha256: has_label.then(|| digest('d')),
            measurement_event_id: has_label.then(|| format!("oracle-event-{round_id}")),
            label_available_at_unix_seconds: has_label.then_some(10_003 + round_id as i64 * 10),
            report_sha256: matches!(
                qualification,
                K1QualificationStateV1::Eligible
                    | K1QualificationStateV1::Unknown
                    | K1QualificationStateV1::Blocked
                    | K1QualificationStateV1::Revoked
            )
            .then(|| digest('e')),
            report_available_at_unix_seconds: matches!(
                qualification,
                K1QualificationStateV1::Eligible
                    | K1QualificationStateV1::Unknown
                    | K1QualificationStateV1::Blocked
            )
            .then_some(10_004 + round_id as i64 * 10),
            qualified_loss: loss,
            qualified_loss_evidence_sha256: loss.map(|_| digest('f')),
        },
        tracking: FeedbackTrackingRecordV1 {
            state: FeedbackTrackingStateV1::NotApplicable,
            proxy_delay_rounds: None,
            actual_feedback_delay_rounds: None,
            controller_feedback_received_at_unix_seconds: None,
            proxy_survival_probability: None,
            tracking_slot_id: None,
            tracking_end_round: None,
            action_receipt_sha256: None,
            controller_update_applied: false,
        },
        independent_audit: None,
        physical_delivery_probability_given_selection: None,
        qualification_probability_given_delivery: None,
        online_tracking_probability_given_selection: None,
        frontier_availability_probability_given_qualified: None,
        update_frontier_at_unix_seconds: None,
    }
}

fn input(mut rounds: Vec<ObservationProcessRoundV1>) -> ObservationProcessInputV1 {
    rounds.sort_by_key(|item| item.round_id);
    ObservationProcessInputV1 {
        schema: OBSERVATION_PROCESS_SCHEMA_V1.into(),
        run_id: "independent-observation-oracle-001".into(),
        population_manifest_sha256: digest('1'),
        source_ledger_sha256: sha256_json(&rounds),
        controller: ObservationRiskControllerV1::None,
        controller_policy_sha256: None,
        delay_contract_sha256: None,
        delay_contract_declared_and_enforced: false,
        fixed_delay_rounds: None,
        maximum_feedback_delay_rounds: None,
        tracking_capacity: None,
        closeout_round: 4,
        independent_audit_design: None,
        qualified_observation_assumptions: ObservationProcessAssumptionsV1 {
            conditional_probability_factorization: ObservationAssumptionStatusV1::Unknown,
            conditional_ignorability: ObservationAssumptionStatusV1::Unknown,
            positivity: ObservationAssumptionStatusV1::Unknown,
            evidence_receipt_sha256: digest('9'),
        },
        rounds,
    }
}

#[test]
fn population_bounds_match_independent_manual_partition_oracle() {
    let rows = vec![
        row(
            1,
            PhysicalAcquisitionStateV1::NotSelected,
            K1QualificationStateV1::NotApplicable,
            None,
        ),
        row(
            2,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            Some(0.25),
        ),
        row(
            3,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Unknown,
            None,
        ),
        row(
            4,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Blocked,
            None,
        ),
    ];
    let report = evaluate_observation_process_v1(&input(rows)).unwrap();

    // Independent arithmetic: N=4, observed qualified loss sum=0.25,
    // three non-qualified/unmeasured outcomes have losses in [0,1].
    let n = 4.0_f64;
    let observed_loss_sum = 0.25_f64;
    let missing_count = 3.0_f64;
    let expected_lower = observed_loss_sum / n;
    let expected_upper = (observed_loss_sum + missing_count) / n;

    assert!((report.population_risk.partial_identification_lower - expected_lower).abs() < 1e-12);
    assert!((report.population_risk.partial_identification_upper - expected_upper).abs() < 1e-12);
    assert!(
        (report.population_risk.identification_interval_width - (expected_upper - expected_lower))
            .abs()
            < 1e-12
    );
    assert_eq!(report.funnel.not_selected, 1);
    assert_eq!(report.funnel.k1_eligible, 1);
    assert_eq!(report.funnel.k1_unknown, 1);
    assert_eq!(report.funnel.k1_blocked, 1);
    assert_eq!(report.funnel.measured, 3);
    assert_eq!(report.funnel.population_rounds, 4);
    assert!(
        !report
            .population_risk
            .population_claim_authorized_by_qualified_subset_alone
    );
}

#[test]
fn qualified_observation_risk_is_separate_and_unknown_assumptions_suppress_ipw() {
    let rows = vec![
        row(
            1,
            PhysicalAcquisitionStateV1::NotSelected,
            K1QualificationStateV1::NotApplicable,
            None,
        ),
        row(
            2,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            Some(0.25),
        ),
        row(
            3,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Pending,
            None,
        ),
        row(
            4,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Revoked,
            None,
        ),
    ];
    let report = evaluate_observation_process_v1(&input(rows)).unwrap();

    assert_eq!(
        report.qualified_observation_risk.identification,
        symthaea_semiconductor::benchmark::PopulationRiskIdentificationV1::WorstCaseBoundsOnly
    );
    assert!(report.qualified_observation_risk.physical_observation_ipw.is_none());
    assert!(report.qualified_observation_risk.online_update_ipw.is_none());
    assert!(report.qualified_observation_risk.mnar_or_propensity_warning);
    assert_eq!(report.qualified_observation_risk.pending_qualification_fraction, 0.25);
    assert_eq!(report.qualified_observation_risk.blocked_or_revoked_fraction, 0.25);

    // The older selection/audit estimator remains a separate report and its
    // partial-identification interval is not replaced by the staged risk bound.
    assert!(report.population_risk.identification_interval_width >= 0.0);
}

#[test]
fn canonical_observation_report_is_invariant_to_input_row_order() {
    let rows = vec![
        row(
            1,
            PhysicalAcquisitionStateV1::NotSelected,
            K1QualificationStateV1::NotApplicable,
            None,
        ),
        row(
            2,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Eligible,
            Some(0.25),
        ),
        row(
            3,
            PhysicalAcquisitionStateV1::Measured,
            K1QualificationStateV1::Unknown,
            None,
        ),
    ];
    let first = input(rows.clone());
    let mut second = input(rows);
    second.rounds.reverse();
    second.source_ledger_sha256 = sha256_json(&{
        let mut canonical = second.rounds.clone();
        canonical.sort_by_key(|item| item.round_id);
        canonical
    });

    let first_report = evaluate_observation_process_v1(&first).unwrap();
    let second_report = evaluate_observation_process_v1(&second).unwrap();
    assert_eq!(first_report.report_sha256, second_report.report_sha256);
    assert_eq!(first_report.input_sha256, second_report.input_sha256);
}
