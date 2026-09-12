//! Preregistered non-ceiling AE-2 action-outcome contingency protocol.
//!
//! This module specifies the task and scoring contract only. It deliberately
//! does not claim the current runtime has passed it. A later runner must bind
//! these declarations to exact code/config/stimulus identities and produce
//! qualified empirical evidence.

use super::experiment_protocol::{
    ExpectedDirection, ExpectedObservable, ExperimentArm, ExperimentArmRole, ExperimentProtocol,
    StateOverride, StimulusPlan,
};

pub const AE2_PROTOCOL_ID: &str = "butlin-ae2-contingency-reversal-v1";
pub const AE2_ENVIRONMENT_ID: &str = "two-action-two-consequence-reversal-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContingencySchedule {
    pub acquisition_trials: u32,
    pub reversal_trials: u32,
}

impl Default for ContingencySchedule {
    fn default() -> Self {
        Self {
            acquisition_trials: 64,
            reversal_trials: 64,
        }
    }
}

impl ContingencySchedule {
    pub fn validate(self) -> Result<(), String> {
        if self.acquisition_trials < 8 || self.reversal_trials < 8 {
            return Err("AE-2 contingency phases require at least 8 trials each".into());
        }
        Ok(())
    }
}

pub fn protocol() -> ExperimentProtocol {
    let stimulus = StimulusPlan::ActionContingencies {
        environment_id: AE2_ENVIRONMENT_ID.into(),
    };
    let expected = vec![
        ExpectedObservable {
            metric_id: "ae2.action_outcome_prediction_accuracy".into(),
            expected_direction: ExpectedDirection::Decrease,
        },
        ExpectedObservable {
            metric_id: "ae2.reversal_adaptation_rate".into(),
            expected_direction: ExpectedDirection::Decrease,
        },
    ];

    ExperimentProtocol {
        protocol_id: AE2_PROTOCOL_ID.into(),
        indicator_id: "AE-2".into(),
        preregistration_id: "butlin-ae2-contingency-v1".into(),
        arms: vec![
            ExperimentArm {
                role: ExperimentArmRole::Baseline,
                stimulus: stimulus.clone(),
                overrides: vec![],
                intervention_id: None,
                expected_observables: expected.clone(),
            },
            ExperimentArm {
                role: ExperimentArmRole::TargetedAblation,
                stimulus: stimulus.clone(),
                overrides: vec![],
                intervention_id: Some("disable_embodied_cognition".into()),
                expected_observables: expected.clone(),
            },
            ExperimentArm {
                role: ExperimentArmRole::Sham,
                stimulus: stimulus.clone(),
                overrides: vec![],
                intervention_id: Some("disable_predictive_processing".into()),
                expected_observables: expected.clone(),
            },
            ExperimentArm {
                role: ExperimentArmRole::PositiveControl,
                stimulus: stimulus.clone(),
                overrides: vec![StateOverride::ActionOutcomeMap {
                    map_id: "forced-swapped-contingency".into(),
                }],
                intervention_id: Some("swap_action_outcome_map".into()),
                expected_observables: vec![ExpectedObservable {
                    metric_id: "ae2.contingency_prediction_error".into(),
                    expected_direction: ExpectedDirection::Increase,
                }],
            },
            ExperimentArm {
                role: ExperimentArmRole::Rescue,
                stimulus,
                overrides: vec![StateOverride::ActionOutcomeMap {
                    map_id: "alternate-explicit-contingency-channel".into(),
                }],
                intervention_id: Some("disable_embodied_cognition+rescue_contingency_info".into()),
                expected_observables: vec![ExpectedObservable {
                    metric_id: "ae2.reversal_adaptation_rate".into(),
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
    fn ae2_protocol_is_structurally_valid() {
        ContingencySchedule::default().validate().unwrap();
        protocol().validate().unwrap();
    }

    #[test]
    fn ae2_protocol_contains_non_ceiling_reversal_observables() {
        let p = protocol();
        let target = p
            .arms
            .iter()
            .find(|a| a.role == ExperimentArmRole::TargetedAblation)
            .unwrap();
        let ids: Vec<_> = target
            .expected_observables
            .iter()
            .map(|o| o.metric_id.as_str())
            .collect();
        assert!(ids.contains(&"ae2.action_outcome_prediction_accuracy"));
        assert!(ids.contains(&"ae2.reversal_adaptation_rate"));
    }

    #[test]
    fn rescue_is_not_the_native_embodiment_mechanism() {
        let p = protocol();
        let rescue = p
            .arms
            .iter()
            .find(|a| a.role == ExperimentArmRole::Rescue)
            .unwrap();
        assert_ne!(
            rescue.intervention_id.as_deref(),
            Some("enable_embodied_cognition")
        );
    }
}
