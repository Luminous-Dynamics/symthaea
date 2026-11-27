#![forbid(unsafe_code)]

use serde::Serialize;
use sha2::{Digest, Sha256};

use symthaea_semiconductor::benchmark::{
    DELAYED_ACI_DELIVERY_SCHEMA_V1, DELAYED_ACI_POLICY_SCHEMA_V1, DELAYED_ACI_PREDICTION_SCHEMA_V1,
    DelayedAciFeedbackModeV1, DelayedAciLabelDeliveryV1, DelayedAciPolicyV1,
    DelayedAciPredictionInputV1, LabelConsumptionEligibilityReportV1,
    LabelConsumptionEligibilityV1, LabelEvidenceFrontierV1, create_delayed_aci_engine_v1,
    current_delayed_aci_alpha_v1, deliver_delayed_aci_feedback_v1,
    register_delayed_aci_prediction_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn json_digest<T: Serialize + ?Sized>(value: &T) -> String {
    format!(
        "{:x}",
        Sha256::digest(serde_json::to_vec(value).expect("fixture serializes"))
    )
}

fn policy(delay_steps: u64) -> DelayedAciPolicyV1 {
    DelayedAciPolicyV1 {
        schema: DELAYED_ACI_POLICY_SCHEMA_V1.into(),
        policy_id: "independent-delayed-aci-oracle".into(),
        revision: 1,
        target_miscoverage: 0.05,
        learning_rate: 0.10,
        feedback_mode: DelayedAciFeedbackModeV1::FixedDelayPhased { delay_steps },
    }
}

fn prediction(sequence: u64, alpha: f64) -> DelayedAciPredictionInputV1 {
    DelayedAciPredictionInputV1 {
        schema: DELAYED_ACI_PREDICTION_SCHEMA_V1.into(),
        sample_id: format!("sample-{sequence}"),
        measurement_event_id: format!("event-{sequence}"),
        event_time_unix_seconds: 10_000 + sequence as i64,
        feature_available_at_unix_seconds: 10_001 + sequence as i64,
        prediction_at_unix_seconds: 10_002 + sequence as i64,
        clock_domain_id: "fab-clock".into(),
        global_model_artifact_sha256: digest('a'),
        calibration_model_sha256: digest('b'),
        alpha_state_used: alpha,
        point_prediction: 10.0,
        interval_lower: 9.0,
        interval_upper: 11.0,
        residual_scale: Some(1.0),
    }
}

fn label_report(
    label_id: &str,
    revision: u64,
    event_id: &str,
    payload_digest: &str,
    value: f64,
) -> LabelConsumptionEligibilityReportV1 {
    let mut report = LabelConsumptionEligibilityReportV1 {
        schema: "luminous.semi-bench.label-consumption-report.v1".into(),
        label_id: label_id.into(),
        label_revision: revision,
        measurement_event_id: event_id.into(),
        calibration_certificate_id: "cal-oracle-1".into(),
        frontier: LabelEvidenceFrontierV1::CurrentAtK1,
        eligibility: LabelConsumptionEligibilityV1::EligibleUnderDeclaredPolicy,
        downstream_consumption_allowed: true,
        reasons: Vec::new(),
        label_sha256: payload_digest.into(),
        measurement_sha256: digest('c'),
        qualified_corrected_value: value,
        qualified_standard_uncertainty_upper_bound: Some(0.1),
        qualification_report_sha256: digest('d'),
        requalification_report_sha256: digest('e'),
        requalification_frontier_sha256: digest('f'),
        source_authenticity_proven: false,
        training_snapshot_mutated: false,
        equipment_control_authority: false,
        report_sha256: String::new(),
    };
    let mut canonical = report.clone();
    canonical.report_sha256.clear();
    report.report_sha256 = json_digest(&canonical);
    report
}

fn delivery(sequence: u64, delivered_round: u64, value: f64) -> DelayedAciLabelDeliveryV1 {
    let label_id = format!("label-{sequence}");
    let event_id = format!("event-{sequence}");
    let payload = digest('c');
    DelayedAciLabelDeliveryV1 {
        schema: DELAYED_ACI_DELIVERY_SCHEMA_V1.into(),
        sample_id: format!("sample-{sequence}"),
        label_id: label_id.clone(),
        label_revision: 1,
        measurement_event_id: event_id.clone(),
        label_payload_sha256: payload.clone(),
        label_value: value,
        label_available_at_unix_seconds: 20_000 + delivered_round as i64 * 3,
        eligibility_report_available_at_unix_seconds: 20_001 + delivered_round as i64 * 3,
        delivered_at_unix_seconds: 20_002 + delivered_round as i64 * 3,
        delivery_round_sequence: delivered_round,
        clock_domain_id: "fab-clock".into(),
        idempotency_key: format!("{label_id}:1"),
        eligibility_report: label_report(&label_id, 1, &event_id, &payload, value),
    }
}

#[test]
fn fixed_delay_phase_states_match_an_independent_interleaved_aci_oracle() {
    let tau = 3_u64;
    let mut engine = create_delayed_aci_engine_v1(
        "aci-reference-oracle",
        "fab-clock",
        &digest('a'),
        &digest('b'),
        policy(tau),
        0.10,
    )
    .unwrap();

    // Independent reference state: one alpha per modulo-tau phase.
    let mut reference_alpha = [0.10_f64; 3];
    let outcomes = [12.0, 10.5, 12.0, 10.5, 10.5, 12.0];

    for round in 0..6_u64 {
        if round >= tau {
            let prior = round - tau;
            let value = outcomes[prior as usize];
            let report =
                deliver_delayed_aci_feedback_v1(&mut engine, &delivery(prior, round, value))
                    .unwrap();
            assert_eq!(
                report.disposition,
                if value < 9.0 || value > 11.0 {
                    symthaea_semiconductor::benchmark::DelayedAciDispositionV1::UpdatedMissed
                } else {
                    symthaea_semiconductor::benchmark::DelayedAciDispositionV1::UpdatedCovered
                }
            );

            let miss = if value < 9.0 || value > 11.0 {
                1.0
            } else {
                0.0
            };
            let phase = (prior % tau) as usize;
            reference_alpha[phase] =
                (reference_alpha[phase] + 0.10 * (0.05 - miss)).clamp(0.0, 1.0);
        }

        let phase = (round % tau) as usize;
        let alpha = current_delayed_aci_alpha_v1(&engine).unwrap();
        assert!((alpha - reference_alpha[phase]).abs() < 1.0e-12);
        register_delayed_aci_prediction_v1(&mut engine, &prediction(round, alpha)).unwrap();
    }

    for phase in 0..tau as usize {
        assert!((engine.phase_alpha_states[phase] - reference_alpha[phase]).abs() < 1.0e-12);
    }
}

#[test]
fn fixed_delay_phase_updates_do_not_cross_contaminate_other_phases() {
    let tau = 4_u64;
    let mut engine = create_delayed_aci_engine_v1(
        "aci-phase-isolation",
        "fab-clock",
        &digest('a'),
        &digest('b'),
        policy(tau),
        0.10,
    )
    .unwrap();

    for sequence in 0..tau {
        let alpha = current_delayed_aci_alpha_v1(&engine).unwrap();
        register_delayed_aci_prediction_v1(&mut engine, &prediction(sequence, alpha)).unwrap();
    }

    let before = engine.phase_alpha_states.clone();
    let outcome = delivery(2, tau + 2, 12.0);
    let receipt = deliver_delayed_aci_feedback_v1(&mut engine, &outcome).unwrap();
    assert_eq!(
        receipt.disposition,
        symthaea_semiconductor::benchmark::DelayedAciDispositionV1::UpdatedMissed
    );

    for phase in 0..tau as usize {
        if phase == 2 {
            assert!(
                (engine.phase_alpha_states[phase] - (before[phase] + 0.10 * (0.05 - 1.0))).abs()
                    < 1.0e-12
            );
        } else {
            assert_eq!(engine.phase_alpha_states[phase], before[phase]);
        }
    }
}
