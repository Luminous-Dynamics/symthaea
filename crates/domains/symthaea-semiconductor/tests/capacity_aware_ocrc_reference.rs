#![forbid(unsafe_code)]

use symthaea_semiconductor::benchmark::{
    CA_OCRC_FEEDBACK_SCHEMA_V1, CA_OCRC_POLICY_SCHEMA_V1, CA_OCRC_ROUND_INPUT_SCHEMA_V1,
    CaoOcrcAdmissionV1, CaoOcrcEngineV1, CaoOcrcFeedbackInputV1,
    CaoOcrcGuaranteeStatusV1, CaoOcrcPolicyV1, CaoOcrcPopulationScopeV1,
    CaoOcrcRoundInputV1, CaoOcrcTrivialSetV1, LabelConsumptionEligibilityReportV1,
    LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1, advance_ca_ocrc_round_v1,
    create_ca_ocrc_engine_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn policy(capacity: usize) -> CaoOcrcPolicyV1 {
    CaoOcrcPolicyV1 {
        schema: CA_OCRC_POLICY_SCHEMA_V1.into(),
        policy_id: "ca-ocrc-independent-reference-v1".into(),
        revision: 1,
        target_risk: 0.10,
        learning_rate: 0.05,
        score_upper_bound: 100.0,
        initial_threshold: 0.5,
        tracking_capacity: capacity,
        maximum_feedback_delay_rounds: 2,
        population_scope: CaoOcrcPopulationScopeV1::QualifiedMeasuredStreamOnly,
        proxy_survival: vec![1.0, 0.5, 0.25],
    }
}

fn engine(capacity: usize) -> CaoOcrcEngineV1 {
    create_ca_ocrc_engine_v1(
        "independent-ca-ocrc",
        "fab-clock",
        policy(capacity),
    )
    .unwrap()
}

fn standalone_policy(capacity: usize) -> CaoOcrcPolicyV1 {
    policy(capacity)
}

fn round(round_index: u64, draw: f64, threshold: f64) -> CaoOcrcRoundInputV1 {
    let centre = 10.0;
    CaoOcrcRoundInputV1 {
        schema: CA_OCRC_ROUND_INPUT_SCHEMA_V1.into(),
        round_index,
        sample_id: format!("sample-{round_index}"),
        measurement_event_id: format!("measurement-{round_index}"),
        event_time_unix_seconds: round_index as i64 * 10,
        prediction_at_unix_seconds: round_index as i64 * 10 + 1,
        clock_domain_id: "fab-clock".into(),
        global_model_artifact_sha256: digest('a'),
        interval_policy_sha256: digest('b'),
        upstream_prediction_receipt_sha256: digest('c'),
        trivial_set: None::<CaoOcrcTrivialSetV1>,
        threshold_used: threshold,
        score_location: centre,
        interval_lower: Some(centre - threshold),
        interval_upper: Some(centre + threshold),
        proxy_uniform_draw: Some(draw),
    }
}

fn feedback(
    origin: u64,
    delivery: u64,
    value: f64,
    label_id: &str,
) -> CaoOcrcFeedbackInputV1 {
    let payload = digest('d');
    let event_id = format!("measurement-{origin}");
    let mut report = LabelConsumptionEligibilityReportV1 {
        schema: "luminous.semi-bench.label-consumption-report.v1".into(),
        label_id: label_id.into(),
        label_revision: 1,
        measurement_event_id: event_id.clone(),
        calibration_certificate_id: "cert-001".into(),
        frontier: LabelEvidenceFrontierV1::CurrentAtK1,
        eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        downstream_consumption_allowed: true,
        reasons: Vec::new(),
        label_sha256: payload.clone(),
        measurement_sha256: digest('e'),
        qualified_corrected_value: value,
        qualified_standard_uncertainty_upper_bound: Some(0.1),
        qualification_report_sha256: digest('f'),
        requalification_report_sha256: digest('1'),
        requalification_frontier_sha256: digest('2'),
        source_authenticity_proven: false,
        training_snapshot_mutated: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    report.report_sha256 =
        format!("{:x}", sha2::Sha256::digest(serde_json::to_vec(&canonical).unwrap()));

    CaoOcrcFeedbackInputV1 {
        schema: CA_OCRC_FEEDBACK_SCHEMA_V1.into(),
        delivery_round: delivery,
        originating_round: origin,
        sample_id: format!("sample-{origin}"),
        label_id: label_id.into(),
        label_revision: 1,
        measurement_event_id: event_id,
        label_payload_sha256: payload,
        corrected_measurement_value: value,
        label_available_at_unix_seconds: origin as i64 * 10 + 2,
        eligibility_report_available_at_unix_seconds: delivery as i64 * 10 + 2,
        delivered_at_unix_seconds: delivery as i64 * 10 + 3,
        clock_domain_id: "fab-clock".into(),
        idempotency_key: format!("{label_id}:1"),
        eligibility_report: report,
    }
}

/// This test oracle is deliberately scalar: it does not call the production
/// proxy sampler, proxy-survival helper, or threshold-update helper.
fn oracle_proxy_delay(uniform: f64) -> u64 {
    // For F=[1.0, 0.5, 0.25], the discrete PMF is:
    // P(delay=0)=0.5, P(delay=1)=0.25, P(delay=2)=0.25.
    if uniform < 0.5 {
        0
    } else if uniform < 0.75 {
        1
    } else {
        2
    }
}

fn oracle_loss(value: f64, centre: f64, half_width: f64) -> f64 {
    if value < centre - half_width || value > centre + half_width {
        1.0
    } else {
        0.0
    }
}

fn oracle_threshold_update(
    threshold: f64,
    eta: f64,
    alpha: f64,
    outcomes: &[(u64, f64)],
) -> f64 {
    outcomes.iter().fold(threshold, |current, (delay, loss)| {
        let survival = match delay {
            0 => 1.0,
            1 => 0.5,
            2 => 0.25,
            _ => unreachable!("fixture delay is in the declared support"),
        };
        current + eta * (loss - alpha) / survival
    })
}

#[test]
fn two_feedback_arrivals_match_independent_proxy_and_ipw_oracle() {
    let mut controller = engine(2);

    let draws = [0.80, 0.80, 0.10];
    let expected_delays = draws.map(oracle_proxy_delay);
    assert_eq!(expected_delays, [2, 2, 0]);

    let first_threshold = controller.threshold;
    let r1 = advance_ca_ocrc_round_v1(
        &mut controller,
        &round(1, draws[0], first_threshold),
        &[],
    )
    .unwrap();
    assert_eq!(r1.selected_proxy_delay_rounds, expected_delays[0] as i64);
    assert_eq!(r1.admission, CaoOcrcAdmissionV1::CapacityAdmitted);

    let threshold_r2 = controller.threshold;
    let r2 = advance_ca_ocrc_round_v1(
        &mut controller,
        &round(2, draws[1], threshold_r2),
        &[],
    )
    .unwrap();
    assert_eq!(r2.selected_proxy_delay_rounds, expected_delays[1] as i64);
    assert_eq!(r2.admission, CaoOcrcAdmissionV1::CapacityAdmitted);

    // Round 3 attempts admission before it consumes both feedback events.
    // Capacity is full, so the current round is denied; then rounds 1 and 2
    // arrive at actual delays 2 and 1, respectively.
    let threshold_r3 = controller.threshold;
    let r3 = advance_ca_ocrc_round_v1(
        &mut controller,
        &round(3, draws[2], threshold_r3),
        &[
            feedback(1, 3, 11.0, "label-1"),
            feedback(2, 3, 10.0, "label-2"),
        ],
    )
    .unwrap();
    assert_eq!(r3.selected_proxy_delay_rounds, expected_delays[2] as i64);
    assert_eq!(r3.admission, CaoOcrcAdmissionV1::CapacitySaturated);
    assert_eq!(r3.tracking_count_after_admission, 2);
    assert_eq!(r3.received_feedback_receipts.len(), 2);
    assert_eq!(r3.received_feedback_receipts[0].actual_delay_rounds, Some(2));
    assert_eq!(r3.received_feedback_receipts[1].actual_delay_rounds, Some(1));
    assert_eq!(
        r3.received_feedback_receipts[0].inverse_proxy_survival_weight,
        Some(4.0)
    );
    assert_eq!(
        r3.received_feedback_receipts[1].inverse_proxy_survival_weight,
        Some(2.0)
    );

    let losses = [
        (2, oracle_loss(11.0, 10.0, 0.5)),
        (1, oracle_loss(10.0, 10.0, 0.5)),
    ];
    let expected_threshold = oracle_threshold_update(0.5, 0.05, 0.10, &losses);
    assert!((r3.threshold_after - expected_threshold).abs() < 1.0e-12);
    assert!((controller.threshold - expected_threshold).abs() < 1.0e-12);
    assert!(controller.tracked_rounds.is_empty());
}

#[test]
fn proxy_deadline_inclusive_eviction_order_matches_independent_queue_oracle() {
    let mut controller = engine(1);
    let first_draw = 0.80; // proxy delay 2, expires at round 3.
    let first = advance_ca_ocrc_round_v1(
        &mut controller,
        &round(1, first_draw, controller.threshold),
        &[],
    )
    .unwrap();
    assert_eq!(first.selected_proxy_delay_rounds, 2);

    let second = advance_ca_ocrc_round_v1(
        &mut controller,
        &round(2, 0.80, controller.threshold),
        &[],
    )
    .unwrap();
    assert_eq!(second.admission, CaoOcrcAdmissionV1::CapacitySaturated);

    // Equality with the proxy deadline is still tracked; the round is not
    // evicted before round-3 feedback reception.
    let third = advance_ca_ocrc_round_v1(
        &mut controller,
        &round(3, 0.80, controller.threshold),
        &[feedback(1, 3, 11.0, "label-deadline")],
    )
    .unwrap();
    assert!(third.evicted_round_ids.is_empty());
    assert_eq!(
        third.received_feedback_receipts[0].disposition,
        symthaea_semiconductor::benchmark::CaoOcrcFeedbackDispositionV1::ReceivedAndApplied
    );
    assert!(controller.tracked_rounds.is_empty());
}

#[test]
fn risk_bound_oracle_does_not_confuse_spare_capacity_with_full_tracking() {
    let mut policy = standalone_policy(4);
    policy.proxy_survival = vec![1.0, 1.0, 0.8];
    assert!(symthaea_semiconductor::benchmark::compute_ca_ocrc_risk_bound_v1(&policy, 100).is_err());

    policy.proxy_survival = vec![1.0, 1.0, 1.0];
    let bound =
        symthaea_semiconductor::benchmark::compute_ca_ocrc_risk_bound_v1(&policy, 100).unwrap();
    assert_eq!(
        bound.regime,
        symthaea_semiconductor::benchmark::CaoOcrcBoundRegimeV1::UnlimitedTrackingOcrcLemmaTwo
    );
}

#[test]
fn evidence_filtering_marks_the_run_as_not_inheriting_the_paper_guarantee() {
    let mut controller = engine(1);
    let first_threshold = controller.threshold;
    advance_ca_ocrc_round_v1(
        &mut controller,
        &round(1, 0.80, first_threshold),
        &[],
    )
    .unwrap();

    let mut blocked = feedback(1, 2, 11.0, "blocked-label");
    blocked.eligibility_report.eligibility = LabelConsumptionEligibilityV1::Blocked;
    blocked.eligibility_report.downstream_consumption_allowed = false;
    let mut canonical = blocked.eligibility_report.clone();
    canonical.report_sha256.clear();
    blocked.eligibility_report.report_sha256 =
        format!("{:x}", sha2::Sha256::digest(serde_json::to_vec(&canonical).unwrap()));

    let second_threshold = controller.threshold;
    let receipt = advance_ca_ocrc_round_v1(
        &mut controller,
        &round(2, 0.80, second_threshold),
        &[blocked],
    )
    .unwrap();
    assert_eq!(
        receipt.guarantee_status,
        CaoOcrcGuaranteeStatusV1::InvalidatedByK1QualificationFiltering
    );
    assert_eq!(
        controller.guarantee_status,
        CaoOcrcGuaranteeStatusV1::InvalidatedByK1QualificationFiltering
    );
    assert_eq!(controller.threshold, 0.5);
}
