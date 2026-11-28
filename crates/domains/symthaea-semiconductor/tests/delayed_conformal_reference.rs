#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};

use symthaea_semiconductor::benchmark::{
    CA_OCRC_FEEDBACK_SCHEMA_V1, CA_OCRC_POLICY_SCHEMA_V1, CA_OCRC_ROUND_SCHEMA_V1,
    DELAYED_ACI_DELIVERY_SCHEMA_V1, DELAYED_ACI_POLICY_SCHEMA_V1,
    DELAYED_ACI_PREDICTION_SCHEMA_V1, CaOcrcFeedbackV1, CaOcrcPolicyV1,
    CaOcrcPredictionKindV1, CaOcrcRoundInputV1, CaOcrcTrackingDispositionV1,
    DelayedAciLabelDeliveryV1, DelayedAciPolicyV1, DelayedAciPredictionInputV1,
    DelayedAciDispositionV1, LabelConsumptionEligibilityReportV1,
    LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
    ca_ocrc_proxy_survival_v1, ca_ocrc_risk_gap_bound_v1,
    create_ca_ocrc_engine_v1, create_delayed_aci_engine_v1,
    deliver_delayed_aci_label_v1,
    register_delayed_aci_prediction_v1, start_ca_ocrc_round_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn sha256_json<T: Serialize + ?Sized>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn eligibility_report(
    label_id: &str,
    revision: u64,
    event_id: &str,
    value: f64,
    payload: &str,
) -> LabelConsumptionEligibilityReportV1 {
    let mut report = LabelConsumptionEligibilityReportV1 {
        schema: "luminous.semi-bench.label-consumption-report.v1".into(),
        label_id: label_id.into(),
        label_revision: revision,
        measurement_event_id: event_id.into(),
        calibration_certificate_id: "reference-cal-1".into(),
        frontier: LabelEvidenceFrontierV1::CurrentAtK1,
        eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        downstream_consumption_allowed: true,
        reasons: Vec::new(),
        label_sha256: payload.into(),
        measurement_sha256: digest('a'),
        qualified_corrected_value: value,
        qualified_standard_uncertainty_upper_bound: Some(0.02),
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
    report.report_sha256 = sha256_json(&canonical);
    report
}

fn ca_policy() -> CaOcrcPolicyV1 {
    let mut pmf = vec![0.0; 6];
    pmf[0] = 0.5; // proxy delay -1 (do not track)
    pmf[5] = 0.5; // proxy delay d_max = 4
    CaOcrcPolicyV1 {
        schema: CA_OCRC_POLICY_SCHEMA_V1.into(),
        policy_id: "ca-ocrc-independent-oracle".into(),
        revision: 1,
        target_risk: 0.05,
        learning_rate: 0.10,
        score_bound_b: 1.0,
        score_contract_sha256: digest('f'),
        initial_threshold: 0.5,
        tracking_capacity: 2,
        maximum_feedback_delay: 4,
        proxy_delay_probabilities: pmf,
        require_source_authenticity_assertion: false,
        maximum_inverse_probability_weight: 100.0,
    }
}

fn ca_round(index: u64, draw: f64) -> CaOcrcRoundInputV1 {
    CaOcrcRoundInputV1 {
        schema: CA_OCRC_ROUND_SCHEMA_V1.into(),
        round_index: index,
        prediction_id: format!("prediction-{index}"),
        sample_id: format!("sample-{index}"),
        measurement_event_id: format!("measurement-event-{index}"),
        prediction_kind: CaOcrcPredictionKindV1::Nontrivial,
        lower_bound: Some(0.0),
        upper_bound: Some(1.0),
        threshold_used: 0.5,
        score_contract_sha256: digest('f'),
        prediction_set_sha256: digest('g'),
        event_time_unix_seconds: 1_800_000_000 + index as i64 * 60,
        prediction_at_unix_seconds: 1_800_000_002 + index as i64 * 60,
        clock_domain_id: "fab-clock".into(),
        uniform_draw: draw,
    }
}

fn ca_feedback(origin: u64, current: u64, value: f64) -> CaOcrcFeedbackV1 {
    let payload = digest('h');
    CaOcrcFeedbackV1 {
        schema: CA_OCRC_FEEDBACK_SCHEMA_V1.into(),
        current_round: current,
        origin_round: origin,
        actual_delay: current - origin,
        prediction_set_sha256: digest('g'),
        prediction_id: format!("prediction-{origin}"),
        sample_id: format!("sample-{origin}"),
        measurement_event_id: format!("measurement-event-{origin}"),
        label_id: format!("label-{origin}"),
        label_revision: 1,
        label_payload_sha256: payload.clone(),
        label_value: value,
        label_available_at_unix_seconds: 1_800_000_120 + origin as i64 * 60,
        eligibility_report_available_at_unix_seconds: 1_800_000_121 + origin as i64 * 60,
        delivered_at_unix_seconds: 1_800_000_122 + current as i64 * 60,
        clock_domain_id: "fab-clock".into(),
        eligibility_report: eligibility_report(
            &format!("label-{origin}"),
            1,
            &format!("measurement-event-{origin}"),
            value,
            &payload,
        ),
    }
}

#[test]
fn ca_ocrc_ipw_update_matches_independent_closed_form() {
    let policy = ca_policy();
    assert_eq!(ca_ocrc_proxy_survival_v1(&policy, 2).unwrap(), 0.5);
    let mut engine = create_ca_ocrc_engine_v1("oracle-engine", "fab-clock", policy).unwrap();

    let first = start_ca_ocrc_round_v1(&mut engine, &ca_round(0, 0.9)).unwrap();
    assert_eq!(first.sampled_proxy_delay, Some(4));
    assert!(first.admitted);
    assert_eq!(first.prediction_set_sha256, digest('g'));

    // These rounds advance the logical clock; the fixed-delay proxy lease for
    // prediction 0 remains active through round 4.
    start_ca_ocrc_round_v1(&mut engine, &ca_round(1, 0.1)).unwrap();
    start_ca_ocrc_round_v1(&mut engine, &ca_round(2, 0.1)).unwrap();

    let receipt = symthaea_semiconductor::benchmark::receive_ca_ocrc_feedback_v1(
        &mut engine,
        &ca_feedback(0, 2, 2.0),
    )
    .unwrap();
    assert_eq!(receipt.disposition, CaOcrcTrackingDispositionV1::FeedbackUpdated);
    assert_eq!(receipt.prediction_set_sha256, Some(digest('g')));

    // L = 1; alpha = 0.05; F(2) = 0.5; eta = 0.1.
    // r' = r + eta * (L-alpha)/F(2) = 0.5 + .1*(.95)/.5 = .69.
    let independently_computed = 0.5 + 0.10 * (1.0 - 0.05) / 0.5;
    assert!((receipt.threshold_after - independently_computed).abs() < 1e-12);
}

#[test]
fn ca_ocrc_risk_gap_bound_matches_independent_formula() {
    let policy = ca_policy();
    let horizon = 10_000_u64;
    let bound = ca_ocrc_risk_gap_bound_v1(&policy, horizon).unwrap();

    let ell_f = 4.0 * 0.5;
    let f_at_max = 0.5;
    let h = |x: f64| x - 1.0 - x.ln();
    let boundary = policy.score_bound_b / (policy.learning_rate * horizon as f64);
    let feedback = (policy.tracking_capacity as f64 + 1.0) / (f_at_max * horizon as f64);
    let capacity = (-((policy.tracking_capacity as f64) * h(ell_f / policy.tracking_capacity as f64))).exp();
    let independently_computed = boundary + feedback + capacity;

    assert!((bound.tracking_load_ell_f - ell_f).abs() < 1e-12);
    assert!((bound.total_upper_bound - independently_computed).abs() < 1e-12);
}

#[test]
fn delayed_aci_phase_recurrence_matches_independent_equation() {
    let policy = DelayedAciPolicyV1 {
        schema: DELAYED_ACI_POLICY_SCHEMA_V1.into(),
        policy_id: "tau-daci-independent-oracle".into(),
        revision: 1,
        target_miscoverage: 0.05,
        learning_rate: 0.10,
        fixed_delay_rounds: 2,
        require_source_authenticity_assertion: false,
    };
    let mut engine = create_delayed_aci_engine_v1("aci-oracle", "fab-clock", 0.10, policy).unwrap();

    for round in 0..2 {
        let input = DelayedAciPredictionInputV1 {
            schema: DELAYED_ACI_PREDICTION_SCHEMA_V1.into(),
            prediction_id: format!("prediction-{round}"),
            sample_id: format!("sample-{round}"),
            measurement_event_id: format!("event-{round}"),
            round_index: round,
            event_time_unix_seconds: 1_800_000_000 + round as i64 * 60,
            feature_available_at_unix_seconds: 1_800_000_001 + round as i64 * 60,
            prediction_at_unix_seconds: 1_800_000_002 + round as i64 * 60,
            clock_domain_id: "fab-clock".into(),
            global_model_artifact_sha256: digest('a'),
            prediction_set_sha256: digest(if round == 0 { 'b' } else { 'c' }),
            lower_bound: 0.0,
            upper_bound: 1.0,
        };
        register_delayed_aci_prediction_v1(&mut engine, &input).unwrap();
    }

    let payload = digest('e');
    let report = eligibility_report("label-0", 1, "event-0", 2.0, &payload);
    let delivery = DelayedAciLabelDeliveryV1 {
        schema: DELAYED_ACI_DELIVERY_SCHEMA_V1.into(),
        prediction_id: "prediction-0".into(),
        sample_id: "sample-0".into(),
        label_id: "label-0".into(),
        label_revision: 1,
        measurement_event_id: "event-0".into(),
        label_payload_sha256: payload,
        label_value: 2.0,
        prediction_round: 0,
        delivery_round: 2,
        label_available_at_unix_seconds: 1_800_000_120,
        eligibility_report_available_at_unix_seconds: 1_800_000_121,
        delivered_at_unix_seconds: 1_800_000_122,
        clock_domain_id: "fab-clock".into(),
        eligibility_report: report,
    };
    let receipt = deliver_delayed_aci_label_v1(&mut engine, &delivery).unwrap();
    assert_eq!(receipt.disposition, DelayedAciDispositionV1::UpdatedMissed);

    // tau=2 creates independent phases. Round 0 only mutates phase 0:
    // alpha' = 0.10 + 0.10 * (0.05 - 1) = 0.005.
    let independent_alpha = 0.10 + 0.10 * (0.05 - 1.0);
    assert!((engine.phase_alpha[0] - independent_alpha).abs() < 1e-12);
    assert_eq!(engine.phase_alpha[1], 0.10);
}
