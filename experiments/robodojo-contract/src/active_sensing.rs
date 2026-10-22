//! Cost-sensitive active sensing for embodied-policy experiments.
//!
//! This is a pure decision-support rule, not a sensor driver or safety controller.
//! Probabilities must be calibrated on held-out data; downstream robot/simulator
//! adapters still enforce collision, workspace, interlock, and actuator limits.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SensingState {
    pub predicted_success_without_sensing: f64,
    pub unsafe_event_probability_without_sensing: f64,
    pub remaining_energy_j: f64,
    pub remaining_latency_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct SensingOption {
    pub sensor_id: String,
    pub available: bool,
    /// Expected task-success probability after acquiring this observation.
    /// None means the value model cannot currently justify a forecast.
    pub predicted_success_after_sensing: Option<f64>,
    /// Expected unsafe-event probability after sensing and replanning.
    pub unsafe_event_probability_after_sensing: Option<f64>,
    pub energy_cost_j: f64,
    pub latency_cost_ms: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActiveSensingPolicy {
    /// Utility for successful task completion, in one declared utility unit.
    pub success_value: f64,
    /// Utility penalty for task non-completion, in the same unit.
    pub failure_cost: f64,
    /// Additional penalty for an unsafe event, in the same unit.
    pub unsafe_event_cost: f64,
    /// Conversion factors that price sensing resources in the declared utility.
    pub utility_per_joule: f64,
    pub utility_per_millisecond: f64,
    pub minimum_net_benefit: f64,
    pub minimum_continue_probability: f64,
    pub maximum_unsafe_event_probability: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensingError {
    EmptySensorId,
    InvalidProbability,
    InvalidBudget,
    InvalidCost,
    InvalidPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensingStopReason {
    RequiredSensorUnavailable,
    RequiredSensorOverBudget,
    PosteriorForecastUnavailable,
    RiskRemainsAboveLimit,
    ConfidenceRemainsBelowThreshold,
    NetBenefitBelowThreshold,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProceedReason {
    CurrentEstimateMeetsDeclaredGates,
    ExpectedBenefitBelowThreshold,
    OptionalSensorUnavailable,
    OptionalSensorOverBudget,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SensingDecision {
    ProceedWithoutSensing {
        reason: ProceedReason,
        estimated_net_benefit: Option<f64>,
    },
    AcquireObservation {
        sensor_id: String,
        estimated_net_benefit: f64,
        required_for_risk_gate: bool,
    },
    StopAndEscalate {
        reason: SensingStopReason,
        estimated_net_benefit: Option<f64>,
    },
}

impl ActiveSensingPolicy {
    pub fn validate(self) -> Result<(), SensingError> {
        let non_negative = [
            self.success_value,
            self.failure_cost,
            self.unsafe_event_cost,
            self.utility_per_joule,
            self.utility_per_millisecond,
            self.minimum_net_benefit,
        ];
        if non_negative.iter().any(|x| !x.is_finite() || *x < 0.0) {
            return Err(SensingError::InvalidPolicy);
        }
        if !valid_probability(self.minimum_continue_probability)
            || !valid_probability(self.maximum_unsafe_event_probability)
        {
            return Err(SensingError::InvalidPolicy);
        }
        Ok(())
    }
}

impl SensingState {
    pub fn validate(self) -> Result<(), SensingError> {
        if !valid_probability(self.predicted_success_without_sensing)
            || !valid_probability(self.unsafe_event_probability_without_sensing)
        {
            return Err(SensingError::InvalidProbability);
        }
        if !self.remaining_energy_j.is_finite()
            || !self.remaining_latency_ms.is_finite()
            || self.remaining_energy_j < 0.0
            || self.remaining_latency_ms < 0.0
        {
            return Err(SensingError::InvalidBudget);
        }
        Ok(())
    }
}

impl SensingOption {
    pub fn validate(&self) -> Result<(), SensingError> {
        if self.sensor_id.trim().is_empty() {
            return Err(SensingError::EmptySensorId);
        }
        if !self.energy_cost_j.is_finite()
            || !self.latency_cost_ms.is_finite()
            || self.energy_cost_j < 0.0
            || self.latency_cost_ms < 0.0
        {
            return Err(SensingError::InvalidCost);
        }
        if self.predicted_success_after_sensing.is_some_and(|p| !valid_probability(p))
            || self
                .unsafe_event_probability_after_sensing
                .is_some_and(|p| !valid_probability(p))
        {
            return Err(SensingError::InvalidProbability);
        }
        Ok(())
    }
}

fn valid_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn expected_utility(success_probability: f64, unsafe_probability: f64, policy: ActiveSensingPolicy) -> f64 {
    success_probability * policy.success_value
        - (1.0 - success_probability) * policy.failure_cost
        - unsafe_probability * policy.unsafe_event_cost
}

fn sensing_cost(option: &SensingOption, policy: ActiveSensingPolicy) -> f64 {
    option.energy_cost_j * policy.utility_per_joule
        + option.latency_cost_ms * policy.utility_per_millisecond
}

/// Choose among proceeding, taking another observation, and escalating.
///
/// Safety gates dominate expected utility: if the current risk estimate exceeds
/// the declared limit, the function only recommends sensing when the option has
/// a forecast below that limit and its resource budget is available. It does not
/// authorize the eventual movement; the caller must re-evaluate after the
/// observation and run the platform's independent safety constraints.
pub fn choose_active_sensing(
    state: SensingState,
    option: &SensingOption,
    policy: ActiveSensingPolicy,
) -> Result<SensingDecision, SensingError> {
    state.validate()?;
    option.validate()?;
    policy.validate()?;

    let must_sense_for_risk =
        state.unsafe_event_probability_without_sensing > policy.maximum_unsafe_event_probability;
    let must_improve_confidence =
        state.predicted_success_without_sensing < policy.minimum_continue_probability;

    if !option.available {
        if must_sense_for_risk || must_improve_confidence {
            return Ok(SensingDecision::StopAndEscalate {
                reason: SensingStopReason::RequiredSensorUnavailable,
                estimated_net_benefit: None,
            });
        }
        return Ok(SensingDecision::ProceedWithoutSensing {
            reason: ProceedReason::OptionalSensorUnavailable,
            estimated_net_benefit: None,
        });
    }

    let (Some(after_success), Some(after_unsafe)) = (
        option.predicted_success_after_sensing,
        option.unsafe_event_probability_after_sensing,
    ) else {
        if must_sense_for_risk || must_improve_confidence {
            return Ok(SensingDecision::StopAndEscalate {
                reason: SensingStopReason::PosteriorForecastUnavailable,
                estimated_net_benefit: None,
            });
        }
        return Ok(SensingDecision::ProceedWithoutSensing {
            reason: ProceedReason::ExpectedBenefitBelowThreshold,
            estimated_net_benefit: None,
        });
    };

    if must_sense_for_risk && after_unsafe > policy.maximum_unsafe_event_probability {
        return Ok(SensingDecision::StopAndEscalate {
            reason: SensingStopReason::RiskRemainsAboveLimit,
            estimated_net_benefit: None,
        });
    }
    if must_improve_confidence && after_success < policy.minimum_continue_probability {
        return Ok(SensingDecision::StopAndEscalate {
            reason: SensingStopReason::ConfidenceRemainsBelowThreshold,
            estimated_net_benefit: None,
        });
    }

    let within_budget = option.energy_cost_j <= state.remaining_energy_j
        && option.latency_cost_ms <= state.remaining_latency_ms;
    let utility_before = expected_utility(
        state.predicted_success_without_sensing,
        state.unsafe_event_probability_without_sensing,
        policy,
    );
    let utility_after = expected_utility(after_success, after_unsafe, policy);
    let net_benefit = utility_after - utility_before - sensing_cost(option, policy);
    if !net_benefit.is_finite() {
        return Err(SensingError::InvalidCost);
    }

    if !within_budget {
        if must_sense_for_risk || must_improve_confidence {
            return Ok(SensingDecision::StopAndEscalate {
                reason: SensingStopReason::RequiredSensorOverBudget,
                estimated_net_benefit: Some(net_benefit),
            });
        }
        return Ok(SensingDecision::ProceedWithoutSensing {
            reason: ProceedReason::OptionalSensorOverBudget,
            estimated_net_benefit: Some(net_benefit),
        });
    }

    if must_sense_for_risk {
        return Ok(SensingDecision::AcquireObservation {
            sensor_id: option.sensor_id.clone(),
            estimated_net_benefit: net_benefit,
            required_for_risk_gate: true,
        });
    }

    if must_improve_confidence {
        if net_benefit >= policy.minimum_net_benefit
            && after_unsafe <= policy.maximum_unsafe_event_probability
        {
            return Ok(SensingDecision::AcquireObservation {
                sensor_id: option.sensor_id.clone(),
                estimated_net_benefit: net_benefit,
                required_for_risk_gate: false,
            });
        }
        return Ok(SensingDecision::StopAndEscalate {
            reason: if after_unsafe > policy.maximum_unsafe_event_probability {
                SensingStopReason::RiskRemainsAboveLimit
            } else {
                SensingStopReason::NetBenefitBelowThreshold
            },
            estimated_net_benefit: Some(net_benefit),
        });
    }

    if after_unsafe > policy.maximum_unsafe_event_probability {
        return Ok(SensingDecision::StopAndEscalate {
            reason: SensingStopReason::RiskRemainsAboveLimit,
            estimated_net_benefit: Some(net_benefit),
        });
    }

    if net_benefit >= policy.minimum_net_benefit {
        Ok(SensingDecision::AcquireObservation {
            sensor_id: option.sensor_id.clone(),
            estimated_net_benefit: net_benefit,
            required_for_risk_gate: false,
        })
    } else {
        Ok(SensingDecision::ProceedWithoutSensing {
            reason: ProceedReason::ExpectedBenefitBelowThreshold,
            estimated_net_benefit: Some(net_benefit),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy() -> ActiveSensingPolicy {
        ActiveSensingPolicy {
            success_value: 100.0,
            failure_cost: 100.0,
            unsafe_event_cost: 1_000.0,
            utility_per_joule: 0.01,
            utility_per_millisecond: 0.001,
            minimum_net_benefit: 1.0,
            minimum_continue_probability: 0.8,
            maximum_unsafe_event_probability: 0.05,
        }
    }

    fn state() -> SensingState {
        SensingState {
            predicted_success_without_sensing: 0.6,
            unsafe_event_probability_without_sensing: 0.02,
            remaining_energy_j: 1_000.0,
            remaining_latency_ms: 5_000.0,
        }
    }

    fn option() -> SensingOption {
        SensingOption {
            sensor_id: "depth-reobserve".to_owned(),
            available: true,
            predicted_success_after_sensing: Some(0.9),
            unsafe_event_probability_after_sensing: Some(0.01),
            energy_cost_j: 10.0,
            latency_cost_ms: 100.0,
        }
    }

    #[test]
    fn acquires_sensor_when_expected_decision_value_exceeds_cost() {
        let decision = choose_active_sensing(state(), &option(), policy()).unwrap();
        assert!(matches!(
            decision,
            SensingDecision::AcquireObservation {
                required_for_risk_gate: false,
                ..
            }
        ));
    }

    #[test]
    fn low_value_sensing_does_not_spend_resources() {
        let mut current = state();
        current.predicted_success_without_sensing = 0.9;
        let mut candidate = option();
        candidate.predicted_success_after_sensing = Some(0.91);
        candidate.energy_cost_j = 1_000.0;
        let decision = choose_active_sensing(current, &candidate, policy()).unwrap();
        assert!(matches!(
            decision,
            SensingDecision::ProceedWithoutSensing {
                reason: ProceedReason::ExpectedBenefitBelowThreshold,
                ..
            }
        ));
    }

    #[test]
    fn risk_gate_forces_sensing_when_the_observation_can_restore_the_bound() {
        let mut current = state();
        current.predicted_success_without_sensing = 0.95;
        current.unsafe_event_probability_without_sensing = 0.2;
        let mut candidate = option();
        candidate.predicted_success_after_sensing = Some(0.85);
        candidate.unsafe_event_probability_after_sensing = Some(0.01);
        candidate.energy_cost_j = 900.0;
        let decision = choose_active_sensing(current, &candidate, policy()).unwrap();
        assert!(matches!(
            decision,
            SensingDecision::AcquireObservation {
                required_for_risk_gate: true,
                ..
            }
        ));
    }

    #[test]
    fn stops_when_sensing_cannot_restore_the_risk_bound() {
        let mut current = state();
        current.unsafe_event_probability_without_sensing = 0.3;
        let mut candidate = option();
        candidate.unsafe_event_probability_after_sensing = Some(0.1);
        assert!(matches!(
            choose_active_sensing(current, &candidate, policy()).unwrap(),
            SensingDecision::StopAndEscalate {
                reason: SensingStopReason::RiskRemainsAboveLimit,
                ..
            }
        ));
    }

    #[test]
    fn required_sensing_over_budget_stops_instead_of_proceeding() {
        let mut current = state();
        current.unsafe_event_probability_without_sensing = 0.3;
        current.remaining_energy_j = 2.0;
        assert!(matches!(
            choose_active_sensing(current, &option(), policy()).unwrap(),
            SensingDecision::StopAndEscalate {
                reason: SensingStopReason::RequiredSensorOverBudget,
                ..
            }
        ));
    }

    #[test]
    fn missing_posterior_forecast_does_not_become_zero_cost_value() {
        let mut candidate = option();
        candidate.predicted_success_after_sensing = None;
        assert!(matches!(
            choose_active_sensing(state(), &candidate, policy()).unwrap(),
            SensingDecision::StopAndEscalate {
                reason: SensingStopReason::PosteriorForecastUnavailable,
                ..
            }
        ));
    }

    #[test]
    fn unavailable_required_sensor_escalates() {
        let mut current = state();
        current.predicted_success_without_sensing = 0.3;
        let mut candidate = option();
        candidate.available = false;
        assert!(matches!(
            choose_active_sensing(current, &candidate, policy()).unwrap(),
            SensingDecision::StopAndEscalate {
                reason: SensingStopReason::RequiredSensorUnavailable,
                ..
            }
        ));
    }

    #[test]
    fn utility_overflow_is_rejected_instead_of_becoming_a_decision() {
        let mut current_policy = policy();
        current_policy.utility_per_joule = f64::MAX;
        let mut candidate = option();
        candidate.energy_cost_j = 10.0;
        assert_eq!(
            choose_active_sensing(state(), &candidate, current_policy),
            Err(SensingError::InvalidCost)
        );
    }

    #[test]
    fn risk_failure_is_reported_when_confidence_is_high_but_risk_is_not_reduced() {
        let mut current = state();
        current.predicted_success_without_sensing = 0.3;
        current.unsafe_event_probability_without_sensing = 0.01;
        let mut candidate = option();
        candidate.predicted_success_after_sensing = Some(0.95);
        candidate.unsafe_event_probability_after_sensing = Some(0.2);
        let decision = choose_active_sensing(current, &candidate, policy()).unwrap();
        assert!(matches!(
            decision,
            SensingDecision::StopAndEscalate {
                reason: SensingStopReason::RiskRemainsAboveLimit,
                ..
            }
        ));
    }

    #[test]
    fn invalid_probabilities_and_costs_are_rejected() {
        let mut candidate = option();
        candidate.unsafe_event_probability_after_sensing = Some(f64::NAN);
        assert_eq!(
            choose_active_sensing(state(), &candidate, policy()),
            Err(SensingError::InvalidProbability)
        );
        let mut candidate = option();
        candidate.energy_cost_j = -1.0;
        assert_eq!(
            choose_active_sensing(state(), &candidate, policy()),
            Err(SensingError::InvalidCost)
        );
    }

    #[test]
    fn cannot_continue_below_confidence_threshold_without_a_sufficient_observation() {
        let current = state();
        let mut candidate = option();
        candidate.predicted_success_after_sensing = Some(0.7);
        assert!(matches!(
            choose_active_sensing(current, &candidate, policy()).unwrap(),
            SensingDecision::StopAndEscalate {
                reason: SensingStopReason::ConfidenceRemainsBelowThreshold,
                ..
            }
        ));
    }
}
