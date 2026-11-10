//! Calibration-bounded, distributionally robust active sensing.
//!
//! This is an additive extension to the point-estimate v0 active-sensing rule.
//! It uses probability intervals and upper-bound sensing costs to compute a
//! conservative lower bound on the value of an observation. It is decision
//! support only: it is not a sensor driver, safety case, collision checker,
//! runtime authority, or command/actuation path.
//!
//! Intervals are only as credible as the calibration protocol and the
//! deployment-domain coverage named by their binding. Structural validation
//! does not prove empirical coverage.

use std::collections::BTreeSet;

use super::active_sensing::ActiveSensingPolicy;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProbabilityInterval {
    pub lower: f64,
    pub upper: f64,
}

impl ProbabilityInterval {
    pub fn new(lower: f64, upper: f64) -> Result<Self, RobustSensingError> {
        let interval = Self { lower, upper };
        interval.validate()?;
        Ok(interval)
    }

    pub fn validate(self) -> Result<(), RobustSensingError> {
        if !self.lower.is_finite()
            || !self.upper.is_finite()
            || self.lower < 0.0
            || self.upper > 1.0
            || self.lower > self.upper
        {
            return Err(RobustSensingError::InvalidProbabilityInterval);
        }
        Ok(())
    }

    pub fn width(self) -> f64 {
        self.upper - self.lower
    }
}

/// Calibration and evaluation identity for the model that emitted an interval.
///
/// A different model, task protocol, deployment domain or split creates a
/// different binding. A calibration split must never equal its evaluation
/// split. Non-empty names are identity checks, not proof that the underlying
/// artifacts are correctly hashed or empirically calibrated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationBinding {
    pub profile_id: String,
    pub forecast_model_id: String,
    pub task_protocol_id: String,
    pub evaluation_domain_id: String,
    pub calibration_split_id: String,
    pub evaluation_split_id: String,
}

impl CalibrationBinding {
    pub fn validate(&self) -> Result<(), RobustSensingError> {
        for value in [
            &self.profile_id,
            &self.forecast_model_id,
            &self.task_protocol_id,
            &self.evaluation_domain_id,
            &self.calibration_split_id,
            &self.evaluation_split_id,
        ] {
            if value.trim().is_empty() {
                return Err(RobustSensingError::MissingCalibrationIdentity);
            }
        }
        if self.calibration_split_id == self.evaluation_split_id {
            return Err(RobustSensingError::CalibrationEvaluationSplitOverlap);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustSensingState {
    pub predicted_success: ProbabilityInterval,
    pub unsafe_event_probability: ProbabilityInterval,
    pub remaining_energy_j: f64,
    pub remaining_latency_ms: f64,
    pub calibration: CalibrationBinding,
}

impl RobustSensingState {
    pub fn validate(&self) -> Result<(), RobustSensingError> {
        self.predicted_success.validate()?;
        self.unsafe_event_probability.validate()?;
        if !self.remaining_energy_j.is_finite()
            || !self.remaining_latency_ms.is_finite()
            || self.remaining_energy_j < 0.0
            || self.remaining_latency_ms < 0.0
        {
            return Err(RobustSensingError::InvalidBudget);
        }
        self.calibration.validate()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustSensingOption {
    pub sensor_id: String,
    pub available: bool,
    pub predicted_success_after: ProbabilityInterval,
    pub unsafe_event_probability_after: ProbabilityInterval,
    /// Conservative upper bounds, not point estimates.
    pub energy_cost_upper_j: Option<f64>,
    pub latency_cost_upper_ms: Option<f64>,
    pub calibration: CalibrationBinding,
}

impl RobustSensingOption {
    pub fn validate(&self, expected: &CalibrationBinding) -> Result<(), RobustSensingError> {
        if self.sensor_id.trim().is_empty() {
            return Err(RobustSensingError::EmptySensorId);
        }
        self.predicted_success_after.validate()?;
        self.unsafe_event_probability_after.validate()?;
        self.calibration.validate()?;
        if &self.calibration != expected {
            return Err(RobustSensingError::CalibrationBindingMismatch(
                self.sensor_id.clone(),
            ));
        }
        if self.available {
            let energy = self
                .energy_cost_upper_j
                .ok_or(RobustSensingError::MissingResourceUpperBound)?;
            let latency = self
                .latency_cost_upper_ms
                .ok_or(RobustSensingError::MissingResourceUpperBound)?;
            if !energy.is_finite() || !latency.is_finite() || energy < 0.0 || latency < 0.0 {
                return Err(RobustSensingError::InvalidResourceUpperBound);
            }
        }
        if self
            .energy_cost_upper_j
            .is_some_and(|value| !value.is_finite() || value < 0.0)
            || self
                .latency_cost_upper_ms
                .is_some_and(|value| !value.is_finite() || value < 0.0)
        {
            return Err(RobustSensingError::InvalidResourceUpperBound);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustSensingError {
    InvalidProbabilityInterval,
    InvalidBudget,
    EmptySensorId,
    MissingCalibrationIdentity,
    CalibrationEvaluationSplitOverlap,
    CalibrationBindingMismatch(String),
    MissingResourceUpperBound,
    InvalidResourceUpperBound,
    DuplicateSensorId(String),
    InvalidPolicy,
    NonFiniteRobustValue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CandidateDisposition {
    Unavailable,
    OverBudget,
    CannotRestoreRiskLimit,
    CannotRestoreConfidence,
    RobustValueBelowThreshold,
    EligibleOptional,
    EligibleForRequiredSensing,
    Selected,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateEvaluation {
    pub sensor_id: String,
    pub disposition: CandidateDisposition,
    pub conservative_net_benefit: Option<f64>,
    pub post_success_lower: f64,
    pub post_risk_upper: f64,
    pub energy_cost_upper_j: Option<f64>,
    pub latency_cost_upper_ms: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustStopReason {
    RequiredObservationUnavailable,
    NoSensorMeetsRequiredGates,
    RequiredObservationOverBudget,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustProceedReason {
    CurrentBoundsMeetDeclaredGates,
    NoOptionalSensorHasPositiveRobustValue,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RobustSensingRecommendation {
    AcquireObservation {
        sensor_id: String,
        conservative_net_benefit: f64,
        required_for_risk_gate: bool,
        required_for_confidence_gate: bool,
    },
    ProceedWithoutSensing {
        reason: RobustProceedReason,
    },
    StopAndEscalate {
        reason: RobustStopReason,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct RobustSensingResult {
    pub recommendation: RobustSensingRecommendation,
    pub evaluations: Vec<CandidateEvaluation>,
    pub current_success_interval_width: f64,
    pub current_risk_interval_width: f64,
    pub uncertainty_profile_id: String,
}

/// Compute minimum utility after sensing minus maximum utility before sensing
/// minus upper-bound resource cost.
///
/// Utility is monotone increasing in success probability and monotone
/// decreasing in unsafe-event probability. Therefore:
/// - minimum after-utility uses lower success and upper risk bounds;
/// - maximum before-utility uses upper success and lower risk bounds.
fn conservative_lower_bound_value(
    before_success: ProbabilityInterval,
    before_risk: ProbabilityInterval,
    after_success: ProbabilityInterval,
    after_risk: ProbabilityInterval,
    energy_cost_upper_j: f64,
    latency_cost_upper_ms: f64,
    policy: ActiveSensingPolicy,
) -> Result<f64, RobustSensingError> {
    let utility = |success: f64, unsafe_probability: f64| {
        success * policy.success_value
            - (1.0 - success) * policy.failure_cost
            - unsafe_probability * policy.unsafe_event_cost
    };

    let minimum_after_utility = utility(after_success.lower, after_risk.upper);
    let maximum_before_utility = utility(before_success.upper, before_risk.lower);
    let maximum_sensing_cost = energy_cost_upper_j * policy.utility_per_joule
        + latency_cost_upper_ms * policy.utility_per_millisecond;
    let lower_bound =
        minimum_after_utility - maximum_before_utility - maximum_sensing_cost;

    if !lower_bound.is_finite() {
        return Err(RobustSensingError::NonFiniteRobustValue);
    }
    Ok(lower_bound)
}

/// Evaluate a predeclared set of sensors under one common calibrated forecast
/// binding, utility profile and resource budget.
///
/// If current upper risk violates the hard limit, a candidate must restore its
/// post-sensing upper risk bound before it can be selected. If current lower
/// success is below the continuation threshold, a candidate must restore the
/// post-sensing lower success bound. Required observations may be selected even
/// when their conservative economic value is negative; if no admissible
/// observation fits the budget, the result is StopAndEscalate.
///
/// Optional observations are selected only when their conservative LCB meets
/// the policy's minimum net benefit. Ties are broken by sensor ID, so candidate
/// input order cannot change the chosen sensor.
pub fn choose_robust_active_sensing(
    state: &RobustSensingState,
    options: &[RobustSensingOption],
    policy: ActiveSensingPolicy,
) -> Result<RobustSensingResult, RobustSensingError> {
    state.validate()?;
    policy
        .validate()
        .map_err(|_| RobustSensingError::InvalidPolicy)?;

    let mut ids = BTreeSet::new();
    for option in options {
        option.validate(&state.calibration)?;
        if !ids.insert(option.sensor_id.as_str()) {
            return Err(RobustSensingError::DuplicateSensorId(
                option.sensor_id.clone(),
            ));
        }
    }

    let risk_required =
        state.unsafe_event_probability.upper > policy.maximum_unsafe_event_probability;
    let confidence_required =
        state.predicted_success.lower < policy.minimum_continue_probability;

    let mut evaluations = Vec::with_capacity(options.len());
    let mut selectable: Vec<(String, f64, bool, bool)> = Vec::new();
    let mut any_available = false;
    let mut any_within_budget = false;

    for option in options {
        let mut evaluation = CandidateEvaluation {
            sensor_id: option.sensor_id.clone(),
            disposition: CandidateDisposition::Unavailable,
            conservative_net_benefit: None,
            post_success_lower: option.predicted_success_after.lower,
            post_risk_upper: option.unsafe_event_probability_after.upper,
            energy_cost_upper_j: option.energy_cost_upper_j,
            latency_cost_upper_ms: option.latency_cost_upper_ms,
        };

        if !option.available {
            evaluations.push(evaluation);
            continue;
        }
        any_available = true;

        let energy = option
            .energy_cost_upper_j
            .ok_or(RobustSensingError::MissingResourceUpperBound)?;
        let latency = option
            .latency_cost_upper_ms
            .ok_or(RobustSensingError::MissingResourceUpperBound)?;
        let within_budget =
            energy <= state.remaining_energy_j && latency <= state.remaining_latency_ms;

        if !within_budget {
            evaluation.disposition = CandidateDisposition::OverBudget;
            evaluations.push(evaluation);
            continue;
        }
        any_within_budget = true;

        if option.unsafe_event_probability_after.upper
            > policy.maximum_unsafe_event_probability
        {
            evaluation.disposition = CandidateDisposition::CannotRestoreRiskLimit;
            evaluations.push(evaluation);
            continue;
        }
        if option.predicted_success_after.lower < policy.minimum_continue_probability {
            evaluation.disposition = CandidateDisposition::CannotRestoreConfidence;
            evaluations.push(evaluation);
            continue;
        }

        let lower_bound = conservative_lower_bound_value(
            state.predicted_success,
            state.unsafe_event_probability,
            option.predicted_success_after,
            option.unsafe_event_probability_after,
            energy,
            latency,
            policy,
        )?;
        evaluation.conservative_net_benefit = Some(lower_bound);

        let required = risk_required || confidence_required;
        if required {
            evaluation.disposition = CandidateDisposition::EligibleForRequiredSensing;
            selectable.push((
                option.sensor_id.clone(),
                lower_bound,
                risk_required,
                confidence_required,
            ));
        } else if lower_bound >= policy.minimum_net_benefit {
            evaluation.disposition = CandidateDisposition::EligibleOptional;
            selectable.push((option.sensor_id.clone(), lower_bound, false, false));
        } else {
            evaluation.disposition = CandidateDisposition::RobustValueBelowThreshold;
        }
        evaluations.push(evaluation);
    }

    if let Some((sensor_id, lower_bound, required_for_risk_gate, required_for_confidence_gate)) =
        selectable.into_iter().max_by(|left, right| {
            left.1
                .total_cmp(&right.1)
                .then_with(|| right.0.cmp(&left.0))
        })
    {
        if let Some(selected) = evaluations
            .iter_mut()
            .find(|evaluation| evaluation.sensor_id == sensor_id)
        {
            selected.disposition = CandidateDisposition::Selected;
        }
        return Ok(RobustSensingResult {
            recommendation: RobustSensingRecommendation::AcquireObservation {
                sensor_id,
                conservative_net_benefit: lower_bound,
                required_for_risk_gate,
                required_for_confidence_gate,
            },
            evaluations,
            current_success_interval_width: state.predicted_success.width(),
            current_risk_interval_width: state.unsafe_event_probability.width(),
            uncertainty_profile_id: state.calibration.profile_id.clone(),
        });
    }

    let reason = if risk_required {
        if !any_available {
            RobustStopReason::RequiredObservationUnavailable
        } else if !any_within_budget {
            RobustStopReason::RequiredObservationOverBudget
        } else {
            RobustStopReason::NoSensorMeetsRequiredGates
        }
    } else if confidence_required {
        if !any_available {
            RobustStopReason::RequiredObservationUnavailable
        } else if !any_within_budget {
            RobustStopReason::RequiredObservationOverBudget
        } else {
            RobustStopReason::NoSensorMeetsRequiredGates
        }
    } else {
        return Ok(RobustSensingResult {
            recommendation: RobustSensingRecommendation::ProceedWithoutSensing {
                reason: if options.is_empty() {
                    RobustProceedReason::CurrentBoundsMeetDeclaredGates
                } else {
                    RobustProceedReason::NoOptionalSensorHasPositiveRobustValue
                },
            },
            evaluations,
            current_success_interval_width: state.predicted_success.width(),
            current_risk_interval_width: state.unsafe_event_probability.width(),
            uncertainty_profile_id: state.calibration.profile_id.clone(),
        });
    };

    Ok(RobustSensingResult {
        recommendation: RobustSensingRecommendation::StopAndEscalate { reason },
        evaluations,
        current_success_interval_width: state.predicted_success.width(),
        current_risk_interval_width: state.unsafe_event_probability.width(),
        uncertainty_profile_id: state.calibration.profile_id.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binding() -> CalibrationBinding {
        CalibrationBinding {
            profile_id: "calibration-profile-v1".into(),
            forecast_model_id: "risk-model-v4".into(),
            task_protocol_id: "robodojo-rc-v1".into(),
            evaluation_domain_id: "heldout-layout-family-b".into(),
            calibration_split_id: "calibration-seeds-001-050".into(),
            evaluation_split_id: "evaluation-seeds-051-100".into(),
        }
    }

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

    fn state() -> RobustSensingState {
        RobustSensingState {
            predicted_success: ProbabilityInterval::new(0.70, 0.80).unwrap(),
            unsafe_event_probability: ProbabilityInterval::new(0.01, 0.03).unwrap(),
            remaining_energy_j: 100.0,
            remaining_latency_ms: 1_000.0,
            calibration: binding(),
        }
    }

    fn option(
        sensor_id: &str,
        success: ProbabilityInterval,
        risk: ProbabilityInterval,
        energy: f64,
        latency: f64,
    ) -> RobustSensingOption {
        RobustSensingOption {
            sensor_id: sensor_id.into(),
            available: true,
            predicted_success_after: success,
            unsafe_event_probability_after: risk,
            energy_cost_upper_j: Some(energy),
            latency_cost_upper_ms: Some(latency),
            calibration: binding(),
        }
    }

    fn close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-9, "left={left}, right={right}");
    }

    #[test]
    fn intervals_reject_non_finite_reversed_and_out_of_range_values() {
        assert_eq!(
            ProbabilityInterval::new(f64::NAN, 0.5),
            Err(RobustSensingError::InvalidProbabilityInterval)
        );
        assert_eq!(
            ProbabilityInterval::new(0.7, 0.6),
            Err(RobustSensingError::InvalidProbabilityInterval)
        );
        assert_eq!(
            ProbabilityInterval::new(0.0, 1.1),
            Err(RobustSensingError::InvalidProbabilityInterval)
        );
    }

    #[test]
    fn calibration_and_evaluation_splits_must_be_disjoint() {
        let mut context = binding();
        context.evaluation_split_id = context.calibration_split_id.clone();
        assert_eq!(
            context.validate(),
            Err(RobustSensingError::CalibrationEvaluationSplitOverlap)
        );
    }

    #[test]
    fn optional_sensor_with_positive_point_estimate_but_negative_lcb_is_not_selected() {
        let mut current = state();
        current.predicted_success = ProbabilityInterval::new(0.80, 0.95).unwrap();
        let candidate = option(
            "depth",
            ProbabilityInterval::new(0.81, 0.98).unwrap(),
            ProbabilityInterval::new(0.01, 0.03).unwrap(),
            10.0,
            100.0,
        );
        let result = choose_robust_active_sensing(&current, &[candidate], policy()).unwrap();

        assert_eq!(
            result.recommendation,
            RobustSensingRecommendation::ProceedWithoutSensing {
                reason: RobustProceedReason::NoOptionalSensorHasPositiveRobustValue
            }
        );
        close(
            result.evaluations[0]
                .conservative_net_benefit
                .unwrap(),
            -48.2,
        );
    }

    #[test]
    fn risk_midpoint_below_limit_does_not_pass_when_upper_bound_exceeds_it() {
        let mut current = state();
        current.unsafe_event_probability = ProbabilityInterval::new(0.00, 0.08).unwrap();
        let unsafe_sensor = option(
            "depth",
            ProbabilityInterval::new(0.90, 0.95).unwrap(),
            ProbabilityInterval::new(0.01, 0.10).unwrap(),
            5.0,
            50.0,
        );
        let result = choose_robust_active_sensing(&current, &[unsafe_sensor], policy()).unwrap();

        assert_eq!(
            result.recommendation,
            RobustSensingRecommendation::StopAndEscalate {
                reason: RobustStopReason::NoSensorRestoresRiskLimit
            }
        );
    }

    #[test]
    fn selects_on_conservative_lower_bound_not_optimistic_upper_forecast() {
        let conservative = option(
            "a-conservative",
            ProbabilityInterval::new(0.94, 0.96).unwrap(),
            ProbabilityInterval::new(0.01, 0.02).unwrap(),
            2.0,
            10.0,
        );
        let optimistic = option(
            "z-optimistic",
            ProbabilityInterval::new(0.70, 1.00).unwrap(),
            ProbabilityInterval::new(0.01, 0.04).unwrap(),
            0.1,
            1.0,
        );
        let result =
            choose_robust_active_sensing(&state(), &[optimistic, conservative], policy()).unwrap();

        assert_eq!(
            result
                .evaluations
                .iter()
                .find(|item| item.sensor_id == "a-conservative")
                .unwrap()
                .disposition,
            CandidateDisposition::Selected
        );
        assert_eq!(
            result
                .evaluations
                .iter()
                .find(|item| item.sensor_id == "z-optimistic")
                .unwrap()
                .disposition,
            CandidateDisposition::RobustValueBelowThreshold
        );
        assert_eq!(
            result.recommendation,
            RobustSensingRecommendation::AcquireObservation {
                sensor_id: "a-conservative".into(),
                conservative_net_benefit: result
                    .evaluations
                    .iter()
                    .find(|item| item.sensor_id == "a-conservative")
                    .unwrap()
                    .conservative_net_benefit
                    .unwrap(),
                required_for_risk_gate: false,
                required_for_confidence_gate: false,
            }
        );
    }

    #[test]
    fn required_risk_sensing_is_selected_even_when_economic_lcb_is_negative() {
        let mut current = state();
        current.unsafe_event_probability = ProbabilityInterval::new(0.04, 0.20).unwrap();
        let sensor = option(
            "risk-camera",
            ProbabilityInterval::new(0.85, 0.95).unwrap(),
            ProbabilityInterval::new(0.01, 0.04).unwrap(),
            90.0,
            900.0,
        );
        let result = choose_robust_active_sensing(&current, &[sensor], policy()).unwrap();
        assert!(matches!(
            result.recommendation,
            RobustSensingRecommendation::AcquireObservation {
                required_for_risk_gate: true,
                ..
            }
        ));
    }

    #[test]
    fn required_risk_sensing_with_unavailable_sensor_escalates() {
        let mut current = state();
        current.unsafe_event_probability = ProbabilityInterval::new(0.06, 0.20).unwrap();
        let mut sensor = option(
            "depth",
            ProbabilityInterval::new(0.90, 0.95).unwrap(),
            ProbabilityInterval::new(0.01, 0.03).unwrap(),
            5.0,
            50.0,
        );
        sensor.available = false;
        let result = choose_robust_active_sensing(&current, &[sensor], policy()).unwrap();
        assert_eq!(
            result.recommendation,
            RobustSensingRecommendation::StopAndEscalate {
                reason: RobustStopReason::RequiredObservationUnavailable
            }
        );
    }

    #[test]
    fn required_risk_sensing_over_budget_escalates() {
        let mut current = state();
        current.unsafe_event_probability = ProbabilityInterval::new(0.06, 0.20).unwrap();
        current.remaining_energy_j = 1.0;
        let sensor = option(
            "depth",
            ProbabilityInterval::new(0.90, 0.95).unwrap(),
            ProbabilityInterval::new(0.01, 0.03).unwrap(),
            5.0,
            50.0,
        );
        let result = choose_robust_active_sensing(&current, &[sensor], policy()).unwrap();
        assert_eq!(
            result.recommendation,
            RobustSensingRecommendation::StopAndEscalate {
                reason: RobustStopReason::RequiredObservationOverBudget
            }
        );
    }

    #[test]
    fn required_risk_sensing_does_not_choose_sensor_that_leaves_risk_above_limit() {
        let mut current = state();
        current.unsafe_event_probability = ProbabilityInterval::new(0.06, 0.20).unwrap();
        let sensor = option(
            "depth",
            ProbabilityInterval::new(0.90, 0.95).unwrap(),
            ProbabilityInterval::new(0.01, 0.08).unwrap(),
            5.0,
            50.0,
        );
        let result = choose_robust_active_sensing(&current, &[sensor], policy()).unwrap();
        assert_eq!(
            result.recommendation,
            RobustSensingRecommendation::StopAndEscalate {
                reason: RobustStopReason::NoSensorRestoresRiskLimit
            }
        );
    }

    #[test]
    fn missing_cost_upper_bound_is_unknown_not_zero() {
        let mut sensor = option(
            "depth",
            ProbabilityInterval::new(0.90, 0.95).unwrap(),
            ProbabilityInterval::new(0.01, 0.03).unwrap(),
            5.0,
            50.0,
        );
        sensor.energy_cost_upper_j = None;
        assert_eq!(
            choose_robust_active_sensing(&state(), &[sensor], policy()),
            Err(RobustSensingError::MissingResourceUpperBound)
        );
    }

    #[test]
    fn optional_sensor_over_budget_is_not_selected() {
        let mut current = state();
        current.predicted_success = ProbabilityInterval::new(0.90, 0.95).unwrap();
        let sensor = option(
            "depth",
            ProbabilityInterval::new(0.99, 1.00).unwrap(),
            ProbabilityInterval::new(0.00, 0.01).unwrap(),
            200.0,
            100.0,
        );
        let result = choose_robust_active_sensing(&current, &[sensor], policy()).unwrap();
        assert!(matches!(
            result.recommendation,
            RobustSensingRecommendation::ProceedWithoutSensing { .. }
        ));
        assert_eq!(
            result.evaluations[0].disposition,
            CandidateDisposition::OverBudget
        );
    }

    #[test]
    fn rejects_forecasts_bound_to_a_different_calibration_profile() {
        let mut sensor = option(
            "depth",
            ProbabilityInterval::new(0.90, 0.95).unwrap(),
            ProbabilityInterval::new(0.01, 0.03).unwrap(),
            5.0,
            50.0,
        );
        sensor.calibration.forecast_model_id = "different-model".into();
        assert!(matches!(
            choose_robust_active_sensing(&state(), &[sensor], policy()),
            Err(RobustSensingError::CalibrationBindingMismatch(_))
        ));
    }

    #[test]
    fn duplicate_sensor_ids_reject_the_candidate_set() {
        let left = option(
            "depth",
            ProbabilityInterval::new(0.90, 0.95).unwrap(),
            ProbabilityInterval::new(0.01, 0.03).unwrap(),
            5.0,
            50.0,
        );
        let right = left.clone();
        assert_eq!(
            choose_robust_active_sensing(&state(), &[left, right], policy()),
            Err(RobustSensingError::DuplicateSensorId("depth".into()))
        );
    }

    #[test]
    fn sensor_selection_is_deterministic_under_candidate_permutation() {
        let alpha = option(
            "alpha",
            ProbabilityInterval::new(0.90, 0.95).unwrap(),
            ProbabilityInterval::new(0.01, 0.03).unwrap(),
            5.0,
            50.0,
        );
        let mut beta = alpha.clone();
        beta.sensor_id = "beta".into();

        let first = choose_robust_active_sensing(
            &state(),
            &[alpha.clone(), beta.clone()],
            policy(),
        )
        .unwrap();
        let second =
            choose_robust_active_sensing(&state(), &[beta, alpha], policy()).unwrap();

        assert_eq!(first.recommendation, second.recommendation);
        assert_eq!(
            first.recommendation,
            RobustSensingRecommendation::AcquireObservation {
                sensor_id: "alpha".into(),
                conservative_net_benefit: match first.recommendation {
                    RobustSensingRecommendation::AcquireObservation {
                        conservative_net_benefit,
                        ..
                    } => conservative_net_benefit,
                    _ => unreachable!(),
                },
                required_for_risk_gate: false,
                required_for_confidence_gate: false,
            }
        );
    }

    #[test]
    fn interval_widths_and_calibration_identity_are_preserved() {
        let result = choose_robust_active_sensing(&state(), &[], policy()).unwrap();
        close(result.current_success_interval_width, 0.10);
        close(result.current_risk_interval_width, 0.02);
        assert_eq!(result.uncertainty_profile_id, "calibration-profile-v1");
    }

    #[test]
    fn no_option_can_create_an_actuation_command_or_authority() {
        let result = choose_robust_active_sensing(&state(), &[], policy()).unwrap();
        assert_eq!(
            result.recommendation,
            RobustSensingRecommendation::ProceedWithoutSensing {
                reason: RobustProceedReason::CurrentBoundsMeetDeclaredGates
            }
        );
        // This API returns only a sensing recommendation and candidate audit,
        // not an action, controller ticket or actuation authorization.
    }
}
