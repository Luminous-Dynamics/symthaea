//! Typed experimental-input contract for Butlin indicator qualification.
//!
//! This is intentionally data-only. It standardizes how baseline, targeted
//! ablation, sham, positive-control, rescue, and held-out arms describe their
//! stimuli and controlled state manipulations. It does not execute cognitive
//! experiments and grants no evidence authority by itself.

use std::collections::BTreeSet;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExperimentArmRole {
    Baseline,
    TargetedAblation,
    Sham,
    PositiveControl,
    Rescue,
    HeldOut,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StimulusPlan {
    ExactReplay { sequence_id: String },
    PredictableVsSurprising { schedule_id: String },
    LabelledCategories { schedule_id: String, categories: Vec<String> },
    AmbiguousPercept { stimulus_id: String },
    CrossModalPairs { schedule_id: String },
    ActionContingencies { environment_id: String },
}

#[derive(Debug, Clone, PartialEq)]
pub enum StateOverride {
    AttentionTarget { target_id: String },
    Confidence { value: f64 },
    RecurrentReset,
    WorkspaceInjection { payload_id: String },
    ActionOutcomeMap { map_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedObservable {
    pub metric_id: String,
    pub expected_direction: ExpectedDirection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpectedDirection {
    Increase,
    Decrease,
    Invariant,
    DivergeFromBaseline,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentArm {
    pub role: ExperimentArmRole,
    pub stimulus: StimulusPlan,
    pub overrides: Vec<StateOverride>,
    pub intervention_id: Option<String>,
    pub expected_observables: Vec<ExpectedObservable>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExperimentProtocol {
    pub protocol_id: String,
    pub indicator_id: String,
    pub preregistration_id: String,
    pub arms: Vec<ExperimentArm>,
}

impl ExperimentProtocol {
    /// Structural validation only. Runtime achievability/specificity belongs
    /// to the existing qualification runtime and evidence-plane layers.
    pub fn validate(&self) -> Result<(), String> {
        if self.protocol_id.trim().is_empty() {
            return Err("protocol_id must be non-empty".into());
        }
        if self.indicator_id.trim().is_empty() {
            return Err("indicator_id must be non-empty".into());
        }
        if self.preregistration_id.trim().is_empty() {
            return Err("preregistration_id must be non-empty".into());
        }
        if self.arms.is_empty() {
            return Err("protocol requires at least one arm".into());
        }

        let roles: BTreeSet<_> = self.arms.iter().map(|arm| arm.role).collect();
        if !roles.contains(&ExperimentArmRole::Baseline) {
            return Err("protocol requires a baseline arm".into());
        }
        if !roles.contains(&ExperimentArmRole::TargetedAblation) {
            return Err("protocol requires a targeted-ablation arm".into());
        }
        if !roles.contains(&ExperimentArmRole::Sham) {
            return Err("protocol requires a sham arm".into());
        }

        for arm in &self.arms {
            if matches!(arm.role, ExperimentArmRole::TargetedAblation | ExperimentArmRole::Sham)
                && arm
                    .intervention_id
                    .as_deref()
                    .map(str::trim)
                    .unwrap_or_default()
                    .is_empty()
            {
                return Err(format!("{:?} arm requires intervention_id", arm.role));
            }

            for state in &arm.overrides {
                if let StateOverride::Confidence { value } = state {
                    if !value.is_finite() || !(0.0..=1.0).contains(value) {
                        return Err("confidence override must be finite and inside [0,1]".into());
                    }
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay() -> StimulusPlan {
        StimulusPlan::ExactReplay {
            sequence_id: "stimulus-v1".into(),
        }
    }

    fn minimal_protocol() -> ExperimentProtocol {
        ExperimentProtocol {
            protocol_id: "butlin-test".into(),
            indicator_id: "AE-2".into(),
            preregistration_id: "pre-reg-001".into(),
            arms: vec![
                ExperimentArm {
                    role: ExperimentArmRole::Baseline,
                    stimulus: replay(),
                    overrides: vec![],
                    intervention_id: None,
                    expected_observables: vec![],
                },
                ExperimentArm {
                    role: ExperimentArmRole::TargetedAblation,
                    stimulus: replay(),
                    overrides: vec![],
                    intervention_id: Some("disable_embodied_cognition".into()),
                    expected_observables: vec![],
                },
                ExperimentArm {
                    role: ExperimentArmRole::Sham,
                    stimulus: replay(),
                    overrides: vec![],
                    intervention_id: Some("disable_predictive_processing".into()),
                    expected_observables: vec![],
                },
            ],
        }
    }

    #[test]
    fn minimal_three_arm_protocol_is_valid() {
        minimal_protocol().validate().unwrap();
    }

    #[test]
    fn baseline_target_and_sham_are_mandatory() {
        let mut p = minimal_protocol();
        p.arms.retain(|a| a.role != ExperimentArmRole::Sham);
        assert!(p.validate().unwrap_err().contains("sham"));
    }

    #[test]
    fn targeted_and_sham_arms_require_named_interventions() {
        let mut p = minimal_protocol();
        p.arms
            .iter_mut()
            .find(|a| a.role == ExperimentArmRole::TargetedAblation)
            .unwrap()
            .intervention_id = None;
        assert!(p.validate().is_err());
    }

    #[test]
    fn confidence_override_fails_closed_outside_probability_range() {
        let mut p = minimal_protocol();
        p.arms[0]
            .overrides
            .push(StateOverride::Confidence { value: 1.2 });
        assert!(p.validate().is_err());
    }
}
