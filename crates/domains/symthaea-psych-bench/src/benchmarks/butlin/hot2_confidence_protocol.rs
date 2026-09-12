//! Preregistered HOT-2 metacognitive confidence-corruption protocol.
//!
//! Specifies a construct-validity experiment for monitoring one's own state:
//! ground-truth task correctness is held independently of injected confidence,
//! allowing calibration, deliberate misrepresentation, and recovery to be
//! measured without equating confidence magnitude with metacognition.

use super::experiment_protocol::{
    ExpectedDirection, ExpectedObservable, ExperimentArm, ExperimentArmRole, ExperimentProtocol,
    StateOverride, StimulusPlan,
};

pub const HOT2_PROTOCOL_ID: &str = "butlin-hot2-confidence-corruption-v1";

pub fn protocol() -> ExperimentProtocol {
    let stimulus = StimulusPlan::AmbiguousPercept {
        stimulus_id: "hot2-ground-truth-ambiguity-ladder-v1".into(),
    };

    let calibration = ExpectedObservable {
        metric_id: "hot2.confidence_correctness_calibration".into(),
        expected_direction: ExpectedDirection::Decrease,
    };

    ExperimentProtocol {
        protocol_id: HOT2_PROTOCOL_ID.into(),
        indicator_id: "HOT-2".into(),
        preregistration_id: "butlin-hot2-confidence-v1".into(),
        arms: vec![
            ExperimentArm {
                role: ExperimentArmRole::Baseline,
                stimulus: stimulus.clone(),
                overrides: vec![],
                intervention_id: None,
                expected_observables: vec![calibration.clone()],
            },
            ExperimentArm {
                role: ExperimentArmRole::TargetedAblation,
                stimulus: stimulus.clone(),
                overrides: vec![],
                intervention_id: Some("disable_meta_cognition".into()),
                expected_observables: vec![calibration.clone()],
            },
            ExperimentArm {
                role: ExperimentArmRole::Sham,
                stimulus: stimulus.clone(),
                overrides: vec![],
                intervention_id: Some("disable_cross_modal_binding".into()),
                expected_observables: vec![calibration.clone()],
            },
            ExperimentArm {
                role: ExperimentArmRole::PositiveControl,
                stimulus: stimulus.clone(),
                overrides: vec![StateOverride::Confidence { value: 0.99 }],
                intervention_id: Some("inject_known_overconfidence".into()),
                expected_observables: vec![ExpectedObservable {
                    metric_id: "hot2.detected_confidence_mismatch".into(),
                    expected_direction: ExpectedDirection::Increase,
                }],
            },
            ExperimentArm {
                role: ExperimentArmRole::HeldOut,
                stimulus,
                overrides: vec![StateOverride::Confidence { value: 0.01 }],
                intervention_id: Some("inject_known_underconfidence".into()),
                expected_observables: vec![ExpectedObservable {
                    metric_id: "hot2.detected_confidence_mismatch".into(),
                    expected_direction: ExpectedDirection::Increase,
                }],
            },
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hot2_protocol_is_structurally_valid() {
        protocol().validate().unwrap();
    }

    #[test]
    fn confidence_corruption_is_distinct_from_metacognition_ablation() {
        let p = protocol();
        let target = p
            .arms
            .iter()
            .find(|a| a.role == ExperimentArmRole::TargetedAblation)
            .unwrap();
        let control = p
            .arms
            .iter()
            .find(|a| a.role == ExperimentArmRole::PositiveControl)
            .unwrap();
        assert_eq!(target.intervention_id.as_deref(), Some("disable_meta_cognition"));
        assert_eq!(control.intervention_id.as_deref(), Some("inject_known_overconfidence"));
        assert!(matches!(
            control.overrides.as_slice(),
            [StateOverride::Confidence { value }] if (*value - 0.99).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn held_out_arm_tests_opposite_confidence_corruption() {
        let p = protocol();
        let held_out = p
            .arms
            .iter()
            .find(|a| a.role == ExperimentArmRole::HeldOut)
            .unwrap();
        assert!(matches!(
            held_out.overrides.as_slice(),
            [StateOverride::Confidence { value }] if (*value - 0.01).abs() < f64::EPSILON
        ));
    }
}
