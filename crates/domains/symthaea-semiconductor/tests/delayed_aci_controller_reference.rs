#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};

use symthaea_semiconductor::benchmark::{
    DELAYED_ACI_DELIVERY_SCHEMA_V1, DELAYED_ACI_POLICY_SCHEMA_V1, DELAYED_ACI_PREDICTION_SCHEMA_V1,
    DelayedAciLabelDeliveryV1, DelayedAciPolicyV1, DelayedAciPredictionInputV1,
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1,
    create_delayed_aci_engine_v1, deliver_delayed_aci_feedback_v1, issue_delayed_aci_prediction_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn hash_json<T: Serialize>(value: &T) -> String {
    format!("{:x}", Sha256::digest(serde_json::to_vec(value).unwrap()))
}

fn eligible_report(
    label_id: &str,
    event_id: &str,
    value: f64,
    payload: &str,
) -> LabelConsumptionEligibilityReportV1 {
    let mut report = LabelConsumptionEligibilityReportV1 {
        schema: "luminous.semi-bench.label-consumption-report.v1".into(),
        label_id: label_id.into(),
        label_revision: 1,
        measurement_event_id: event_id.into(),
        calibration_certificate_id: "certificate-aci-oracle".into(),
        frontier: LabelEvidenceFrontierV1::CurrentAtK1,
        eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        downstream_consumption_allowed: true,
        reasons: Vec::new(),
        label_sha256: payload.into(),
        measurement_sha256: digest('b'),
        qualified_corrected_value: value,
        qualified_standard_uncertainty_upper_bound: Some(0.01),
        qualification_report_sha256: digest('c'),
        requalification_report_sha256: digest('d'),
        requalification_frontier_sha256: digest('e'),
        source_authenticity_proven: false,
        training_snapshot_mutated: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    report.report_sha256 = hash_json(&canonical);
    report
}

fn label_delivery(
    step: u64,
    sample: &str,
    event: &str,
    label: &str,
    value: f64,
) -> DelayedAciLabelDeliveryV1 {
    let payload = digest('f');
    DelayedAciLabelDeliveryV1 {
        schema: DELAYED_ACI_DELIVERY_SCHEMA_V1.into(),
        sample_id: sample.into(),
        label_id: label.into(),
        label_revision: 1,
        measurement_event_id: event.into(),
        label_payload_sha256: payload.clone(),
        label_value: value,
        label_available_at_unix_seconds: 10_000 + step as i64,
        eligibility_report_available_at_unix_seconds: 10_001 + step as i64,
        delivered_at_unix_seconds: 10_002 + step as i64,
        feedback_step: step,
        clock_domain_id: "aci-clock".into(),
        idempotency_key: format!("{label}:1"),
        eligibility_report: eligible_report(label, event, value, &payload),
    }
}

#[test]
fn interleaved_controller_matches_independent_long_stream_recursion() {
    let delay = 3_u64;
    let sample_count = ninety_six();
    let target = 0.20;
    let gamma = 0.10;
    let policy = DelayedAciPolicyV1 {
        schema: DELAYED_ACI_POLICY_SCHEMA_V1.into(),
        policy_id: "independent-fixed-delay-oracle".into(),
        revision: 1,
        target_miscoverage: target,
        adaptation_rate: gamma,
        minimum_miscoverage: 0.01,
        maximum_miscoverage: 0.40,
        feedback_delay_steps: delay,
    };
    let mut engine = create_delayed_aci_engine_v1(
        "aci-independent-oracle",
        "aci-clock",
        &digest('a'),
        policy.clone(),
    )
    .unwrap();

    let mut oracle_lanes = vec![target; delay as usize];
    let mut issued: Vec<DelayedAciPredictionReceiptV1> = Vec::new();
    let mut accepted = 0_u64;

    for stream_step in 0..(sample_count as u64 + delay) {
        // Fixed-delay feedback is delivered before the prediction at the same
        // stream step, so the corresponding residue-class lane has the correct
        // updated alpha for its next prediction.
        if stream_step >= delay {
            let prediction_index = (stream_step - delay) as usize;
            let prediction = &issued[prediction_index];
            let value = outcome(prediction_index);
            let update = deliver_delayed_aci_feedback_v1(
                &mut engine,
                &label_delivery(
                    stream_step,
                    &prediction.sample_id,
                    &prediction.measurement_event_id,
                    &format!("label-{prediction_index}"),
                    value,
                ),
            )
            .unwrap();
            assert_eq!(update.disposition, DelayedAciDispositionV1::Updated);

            let lane = prediction.lane as usize;
            let missed = value < prediction.lower_bound || value > prediction.upper_bound;
            let next = (oracle_lanes[lane] + gamma * (target - if missed { 1.0 } else { 0.0 }))
                .clamp(0.01, 0.40);
            oracle_lanes[lane] = next;
            accepted += 1;
        }

        if stream_step < sample_count as u64 {
            let lane = (stream_step % delay) as usize;
            let alpha = oracle_lanes[lane];
            let radius = 1.40 - alpha;
            let sample_id = format!("sample-{stream_step}");
            let event_id = format!("event-{stream_step}");
            let input = DelayedAciPredictionInputV1 {
                schema: DELAYED_ACI_PREDICTION_SCHEMA_V1.into(),
                sample_id,
                measurement_event_id: event_id,
                event_time_unix_seconds: 10_000 + stream_step as i64,
                prediction_at_unix_seconds: 10_001 + stream_step as i64,
                clock_domain_id: "aci-clock".into(),
                interval_method_sha256: digest('a'),
                constructed_for_alpha: alpha,
                lower_bound: -radius,
                upper_bound: radius,
            };
            let receipt = issue_delayed_aci_prediction_v1(&mut engine, &input).unwrap();
            issued.push(receipt);
        }
    }

    assert_eq!(accepted, sample_count as u64);
    assert_eq!(engine.state.accepted_feedback_count, sample_count as u64);
    assert!(engine.state.fixed_delay_assumptions_intact);
    assert!(engine.pending_predictions.is_empty());
    for (actual, expected) in engine.state.alpha_lanes.iter().zip(&oracle_lanes) {
        assert!((actual - expected).abs() < 1.0e-12);
    }
}

#[test]
fn k0_feedback_does_not_change_alpha_or_claim_intact_fixed_delay_assumptions() {
    let policy = DelayedAciPolicyV1 {
        schema: DELAYED_ACI_POLICY_SCHEMA_V1.into(),
        policy_id: "k0-negative-control".into(),
        revision: 1,
        target_miscoverage: 0.10,
        adaptation_rate: 0.05,
        minimum_miscoverage: 0.01,
        maximum_miscoverage: 0.40,
        feedback_delay_steps: 1,
    };
    let mut engine =
        create_delayed_aci_engine_v1("aci-k0-control", "aci-clock", &digest('a'), policy).unwrap();
    let input = DelayedAciPredictionInputV1 {
        schema: DELAYED_ACI_PREDICTION_SCHEMA_V1.into(),
        sample_id: "s0".into(),
        measurement_event_id: "e0".into(),
        event_time_unix_seconds: 10,
        prediction_at_unix_seconds: 11,
        clock_domain_id: "aci-clock".into(),
        interval_method_sha256: digest('a'),
        constructed_for_alpha: 0.10,
        lower_bound: -1.0,
        upper_bound: 1.0,
    };
    issue_delayed_aci_prediction_v1(&mut engine, &input).unwrap();
    let before = engine.state.alpha_lanes.clone();

    let payload = digest('f');
    let mut report = eligible_report("l0", "e0", 0.5, &payload);
    report.frontier = LabelEvidenceFrontierV1::AsKnownAtK0;
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    report.report_sha256 = hash_json(&canonical);
    let delivery = DelayedAciLabelDeliveryV1 {
        schema: DELAYED_ACI_DELIVERY_SCHEMA_V1.into(),
        sample_id: "s0".into(),
        label_id: "l0".into(),
        label_revision: 1,
        measurement_event_id: "e0".into(),
        label_payload_sha256: payload,
        label_value: 0.5,
        label_available_at_unix_seconds: 12,
        eligibility_report_available_at_unix_seconds: 13,
        delivered_at_unix_seconds: 14,
        feedback_step: 1,
        clock_domain_id: "aci-clock".into(),
        idempotency_key: "l0:1".into(),
        eligibility_report: report,
    };
    let receipt = deliver_delayed_aci_feedback_v1(&mut engine, &delivery).unwrap();
    assert_eq!(receipt.disposition, DelayedAciDispositionV1::Rejected);
    assert_eq!(engine.state.alpha_lanes, before);
    assert!(!engine.state.fixed_delay_assumptions_intact);
}

fn ninety_six() -> usize {
    96
}

fn outcome(index: usize) -> f64 {
    const VALUES: [f64; 8] = [0.1, 0.2, 1.5, -1.6, 0.4, 1.1, -0.3, 0.8];
    VALUES[index % VALUES.len()]
}
