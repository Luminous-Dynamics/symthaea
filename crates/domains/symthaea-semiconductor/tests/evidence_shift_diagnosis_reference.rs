#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use symthaea_semiconductor::benchmark::{
    EVIDENCE_SHIFT_OBSERVATION_SCHEMA_V1, EVIDENCE_SHIFT_POLICY_SCHEMA_V1,
    EvidenceShiftMonitorV1, EvidenceShiftObservationV1, EvidenceShiftPolicyV1,
    ShiftChannelPolicyV1, ShiftDiagnosticStateV1, ShiftSignalChannelV1,
    create_evidence_shift_monitor_v1, observe_evidence_shift_v1,
};

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn policy() -> EvidenceShiftPolicyV1 {
    let config = ShiftChannelPolicyV1 {
        baseline_mean: 0.0,
        baseline_standard_deviation: 1.0,
        cusum_allowance: 0.5,
        cusum_threshold: 4.0,
        ewma_alpha: 0.25,
        ewma_threshold: 3.0,
        minimum_observations: 2,
    };
    EvidenceShiftPolicyV1 {
        schema: EVIDENCE_SHIFT_POLICY_SCHEMA_V1.into(),
        policy_id: "independent-shift-oracle".into(),
        revision: 1,
        clock_domain_id: "fab-clock".into(),
        nominal_baseline_manifest_sha256: digest('a'),
        process_event_context_window_seconds: 100,
        maximum_absolute_standardized_value: 50.0,
        channels: BTreeMap::from([
            (ShiftSignalChannelV1::VmResidual, config.clone()),
            (ShiftSignalChannelV1::ReferenceStandardResidual, config.clone()),
            (ShiftSignalChannelV1::SensorObservabilityScore, config),
        ]),
    }
}

fn observation(
    sequence: u64,
    value: f64,
) -> EvidenceShiftObservationV1 {
    EvidenceShiftObservationV1 {
        schema: EVIDENCE_SHIFT_OBSERVATION_SCHEMA_V1.into(),
        observation_id: format!("reference-{sequence}"),
        stream_sequence: sequence,
        scope_id: "tool-a/chamber-1".into(),
        event_time_unix_seconds: sequence as i64,
        available_at_unix_seconds: sequence as i64 + 1,
        measurement_event_id: None,
        clock_domain_id: "fab-clock".into(),
        nominal_baseline_manifest_sha256: digest('a'),
        channel: ShiftSignalChannelV1::ReferenceStandardResidual,
        observation_value: Some(value),
        prediction_receipt: None,
        label_report: None,
        reference_standard_id: Some("transfer-standard-1".into()),
        reference_evidence_sha256: Some(digest('c')),
        reference_evidence_qualified: true,
        sensor_projection_sha256: None,
        process_event_marker: None,
        source_evidence_sha256: digest('b'),
    }
}

#[derive(Clone, Copy, Debug)]
struct OracleState {
    positive: f64,
    negative: f64,
    ewma: f64,
    observations: u64,
}

fn independent_reference_recurrence(values: &[f64]) -> Vec<OracleState> {
    let allowance = 0.5;
    let alpha = 0.25;
    let mut state = OracleState {
        positive: 0.0,
        negative: 0.0,
        ewma: 0.0,
        observations: 0,
    };
    let mut outputs = Vec::new();
    for value in values {
        let standardized = *value;
        let positive_innovation = standardized - allowance;
        let negative_innovation = -standardized - allowance;
        state.positive = (state.positive + positive_innovation).max(0.0);
        state.negative = (state.negative + negative_innovation).max(0.0);
        state.ewma = alpha.mul_add(standardized, (1.0 - alpha) * state.ewma);
        state.observations += 1;
        outputs.push(state);
    }
    outputs
}

fn production_replay(values: &[f64]) -> (EvidenceShiftMonitorV1, Vec<String>) {
    let mut monitor =
        create_evidence_shift_monitor_v1("shift-oracle-engine", policy()).unwrap();
    let mut digests = Vec::new();
    for (index, value) in values.iter().enumerate() {
        let result = observe_evidence_shift_v1(
            &mut monitor,
            &observation(index as u64 + 1, *value),
        )
        .unwrap();
        digests.push(result.assessment_sha256);
    }
    (monitor, digests)
}

#[test]
fn production_cusum_ewma_matches_independent_known_answer_recurrence() {
    let values = [0.1, -0.2, 0.3, 2.0, 2.1, 2.2, 2.3, 2.4];
    let expected = independent_reference_recurrence(&values);
    let (monitor, _) = production_replay(&values);
    let actual = monitor
        .channel_states
        .get(&(
            "tool-a/chamber-1".to_string(),
            ShiftSignalChannelV1::ReferenceStandardResidual,
        ))
        .unwrap();

    let final_expected = expected.last().unwrap();
    assert_eq!(actual.observations, final_expected.observations);
    assert!((actual.positive_cusum - final_expected.positive).abs() < 1.0e-12);
    assert!((actual.negative_cusum - final_expected.negative).abs() < 1.0e-12);
    assert!((actual.ewma - final_expected.ewma).abs() < 1.0e-12);
}

#[test]
fn positive_and_negative_shift_recurrences_are_symmetric() {
    let positive = [2.0, 2.0, 2.0, 2.0, 2.0];
    let negative = [-2.0, -2.0, -2.0, -2.0, -2.0];
    let (positive_monitor, _) = production_replay(&positive);
    let (negative_monitor, _) = production_replay(&negative);
    let positive_state = positive_monitor
        .channel_states
        .get(&(
            "tool-a/chamber-1".to_string(),
            ShiftSignalChannelV1::ReferenceStandardResidual,
        ))
        .unwrap();
    let negative_state = negative_monitor
        .channel_states
        .get(&(
            "tool-a/chamber-1".to_string(),
            ShiftSignalChannelV1::ReferenceStandardResidual,
        ))
        .unwrap();

    assert!((positive_state.positive_cusum - negative_state.negative_cusum).abs() < 1.0e-12);
    assert!((positive_state.negative_cusum - negative_state.positive_cusum).abs() < 1.0e-12);
    assert!((positive_state.ewma + negative_state.ewma).abs() < 1.0e-12);
}

#[test]
fn frozen_stream_replay_produces_identical_assessment_receipts() {
    let values = [0.0, 0.1, -0.1, 0.2, 1.8, 2.1, 2.0];
    let (_, first) = production_replay(&values);
    let (_, second) = production_replay(&values);
    assert_eq!(first, second);
}

#[test]
fn reference_only_alarm_does_not_claim_process_shift() {
    let values = [3.0, 3.0, 3.0];
    let (monitor, receipts) = production_replay(&values);
    assert_eq!(receipts.len(), values.len());
    let latest = monitor
        .assessments_by_observation
        .values()
        .max_by_key(|assessment| assessment.stream_sequence)
        .unwrap();
    assert_eq!(
        latest.diagnostic_state,
        ShiftDiagnosticStateV1::MeasurementSystemShiftSuspected
    );
}
