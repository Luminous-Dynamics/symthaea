//! Dependency-free fail-closed contract gate for embodied-policy experiments.
//!
//! This validates a candidate-action boundary; it is not a robot controller,
//! collision checker, or proof of physical safety. Prediction error can request
//! replanning, but environment-specific limits and collision checks must be
//! enforced by an adapter with access to the robot and simulator telemetry.
#![forbid(unsafe_code)]

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisLimit { pub min: f64, pub max: f64 }

impl AxisLimit {
    pub fn new(min: f64, max: f64) -> Result<Self, ConfigError> {
        if !min.is_finite() || !max.is_finite() || min > max {
            return Err(ConfigError::InvalidAxisLimit);
        }
        Ok(Self { min, max })
    }
    fn contains(self, value: f64) -> bool {
        value.is_finite() && value >= self.min && value <= self.max
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GateConfig {
    pub state_dim: usize,
    pub action_limits: Vec<AxisLimit>,
    pub prediction_error_threshold: f64,
    pub max_consecutive_high_error: usize,
}

impl GateConfig {
    pub fn new(
        state_dim: usize,
        action_limits: Vec<AxisLimit>,
        prediction_error_threshold: f64,
        max_consecutive_high_error: usize,
    ) -> Result<Self, ConfigError> {
        if state_dim == 0 { return Err(ConfigError::EmptyState); }
        if action_limits.is_empty() { return Err(ConfigError::EmptyActionSpace); }
        if action_limits.iter().any(|x| !x.min.is_finite() || !x.max.is_finite() || x.min > x.max) {
            return Err(ConfigError::InvalidAxisLimit);
        }
        if !prediction_error_threshold.is_finite() || prediction_error_threshold < 0.0 {
            return Err(ConfigError::InvalidPredictionThreshold);
        }
        if max_consecutive_high_error == 0 { return Err(ConfigError::ZeroRecoveryBudget); }
        Ok(Self { state_dim, action_limits, prediction_error_threshold, max_consecutive_high_error })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    EmptyState, EmptyActionSpace, InvalidAxisLimit, InvalidPredictionThreshold, ZeroRecoveryBudget,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ObservationFrame {
    pub episode_id: String,
    pub step: u64,
    pub state_values: Vec<f64>,
    /// Caller-defined predictor error; must be calibrated for the chosen policy.
    pub prediction_error: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ActionProposal {
    pub episode_id: String,
    pub based_on_step: u64,
    pub values: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    NoActiveEpisode,
    EmptyEpisodeId,
    EpisodeMismatch,
    StateDimensionMismatch { expected: usize, actual: usize },
    NonFiniteObservation { index: usize },
    InvalidPredictionError,
    NonMonotonicStep { previous: u64, received: u64 },
    SustainedPredictionError { observations: usize, threshold: f64 },
    ActionDimensionMismatch { expected: usize, actual: usize },
    NonFiniteAction { index: usize },
    ActionOutOfBounds { index: usize, value: f64, min: f64, max: f64 },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObservationDecision {
    Accepted,
    RequestReplan { prediction_error: f64, consecutive_high_error: usize },
    SafeStop(StopReason),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ActionRejection {
    SafeStop(StopReason),
    NoObservation,
    StaleObservation { observed_step: u64, action_step: u64 },
    ReplanRequired { observed_step: u64 },
}

/// Episode-scoped guard. Invalid observations/actions latch a stop until an
/// explicit episode reset. Repeated high prediction errors stop the trace;
/// one high error requests replanning and cannot authorize a new action.
#[derive(Debug, Clone)]
pub struct EpisodeGuard {
    config: GateConfig,
    episode_id: Option<String>,
    last_step: Option<u64>,
    consecutive_high_error: usize,
    observation_permits_action: bool,
    fault: Option<StopReason>,
}

impl EpisodeGuard {
    pub fn new(config: GateConfig) -> Self {
        Self {
            config, episode_id: None, last_step: None, consecutive_high_error: 0,
            observation_permits_action: false, fault: None,
        }
    }

    /// Starts a new episode and is the only operation that clears a latched stop.
    pub fn start_episode(&mut self, episode_id: impl Into<String>) -> Result<(), StopReason> {
        let id = episode_id.into();
        if id.trim().is_empty() {
            self.episode_id = None;
            self.last_step = None;
            self.consecutive_high_error = 0;
            self.observation_permits_action = false;
            self.fault = Some(StopReason::EmptyEpisodeId);
            return Err(StopReason::EmptyEpisodeId);
        }
        self.episode_id = Some(id);
        self.last_step = None;
        self.consecutive_high_error = 0;
        self.observation_permits_action = false;
        self.fault = None;
        Ok(())
    }

    pub fn last_observed_step(&self) -> Option<u64> { self.last_step }
    pub fn consecutive_high_error(&self) -> usize { self.consecutive_high_error }
    pub fn stopped(&self) -> Option<&StopReason> { self.fault.as_ref() }

    pub fn observe(&mut self, frame: &ObservationFrame) -> ObservationDecision {
        if let Some(reason) = self.fault.clone() { return ObservationDecision::SafeStop(reason); }
        let Some(active) = self.episode_id.as_deref() else {
            return self.stop(StopReason::NoActiveEpisode);
        };
        if active != frame.episode_id { return self.stop(StopReason::EpisodeMismatch); }
        if frame.state_values.len() != self.config.state_dim {
            return self.stop(StopReason::StateDimensionMismatch {
                expected: self.config.state_dim, actual: frame.state_values.len(),
            });
        }
        if let Some(index) = frame.state_values.iter().position(|x| !x.is_finite()) {
            return self.stop(StopReason::NonFiniteObservation { index });
        }
        if !frame.prediction_error.is_finite() || frame.prediction_error < 0.0 {
            return self.stop(StopReason::InvalidPredictionError);
        }
        if let Some(previous) = self.last_step {
            if frame.step <= previous {
                return self.stop(StopReason::NonMonotonicStep { previous, received: frame.step });
            }
        }

        self.last_step = Some(frame.step);
        if frame.prediction_error > self.config.prediction_error_threshold {
            self.consecutive_high_error = self.consecutive_high_error.saturating_add(1);
            self.observation_permits_action = false;
            if self.consecutive_high_error >= self.config.max_consecutive_high_error {
                return self.stop(StopReason::SustainedPredictionError {
                    observations: self.consecutive_high_error,
                    threshold: self.config.prediction_error_threshold,
                });
            }
            ObservationDecision::RequestReplan {
                prediction_error: frame.prediction_error,
                consecutive_high_error: self.consecutive_high_error,
            }
        } else {
            self.consecutive_high_error = 0;
            self.observation_permits_action = true;
            ObservationDecision::Accepted
        }
    }

    /// Validates a candidate only; it never sends commands to a robot.
    /// Approval means only that this contract's checks passed.
    pub fn approve_action(&mut self, proposal: &ActionProposal) -> Result<Vec<f64>, ActionRejection> {
        if let Some(reason) = self.fault.clone() { return Err(ActionRejection::SafeStop(reason)); }
        let Some(active) = self.episode_id.as_deref() else {
            return self.stop_action(StopReason::NoActiveEpisode);
        };
        if active != proposal.episode_id { return self.stop_action(StopReason::EpisodeMismatch); }
        let Some(observed_step) = self.last_step else { return Err(ActionRejection::NoObservation); };
        if proposal.based_on_step != observed_step {
            return Err(ActionRejection::StaleObservation {
                observed_step, action_step: proposal.based_on_step,
            });
        }
        if !self.observation_permits_action {
            return Err(ActionRejection::ReplanRequired { observed_step });
        }
        if proposal.values.len() != self.config.action_limits.len() {
            return self.stop_action(StopReason::ActionDimensionMismatch {
                expected: self.config.action_limits.len(), actual: proposal.values.len(),
            });
        }
        for (index, (&value, limit)) in proposal.values.iter().zip(&self.config.action_limits).enumerate() {
            if !value.is_finite() { return self.stop_action(StopReason::NonFiniteAction { index }); }
            if !limit.contains(value) {
                return self.stop_action(StopReason::ActionOutOfBounds {
                    index, value, min: limit.min, max: limit.max,
                });
            }
        }
        Ok(proposal.values.clone())
    }

    fn stop(&mut self, reason: StopReason) -> ObservationDecision {
        self.observation_permits_action = false;
        if self.fault.is_none() { self.fault = Some(reason.clone()); }
        ObservationDecision::SafeStop(self.fault.clone().unwrap_or(reason))
    }

    fn stop_action<T>(&mut self, reason: StopReason) -> Result<T, ActionRejection> {
        self.observation_permits_action = false;
        if self.fault.is_none() { self.fault = Some(reason.clone()); }
        Err(ActionRejection::SafeStop(self.fault.clone().unwrap_or(reason)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guard() -> EpisodeGuard {
        EpisodeGuard::new(GateConfig::new(
            3, vec![AxisLimit::new(-1.0, 1.0).unwrap(), AxisLimit::new(0.0, 0.08).unwrap()],
            0.5, 3,
        ).unwrap())
    }
    fn obs(ep: &str, step: u64, error: f64) -> ObservationFrame {
        ObservationFrame { episode_id: ep.to_owned(), step, state_values: vec![0.1, 0.2, 0.3], prediction_error: error }
    }
    fn action(ep: &str, step: u64) -> ActionProposal {
        ActionProposal { episode_id: ep.to_owned(), based_on_step: step, values: vec![0.25, 0.04] }
    }

    #[test]
    fn valid_observation_and_bounded_action_pass() {
        let mut g = guard(); g.start_episode("a").unwrap();
        assert_eq!(g.observe(&obs("a", 1, 0.2)), ObservationDecision::Accepted);
        assert_eq!(g.approve_action(&action("a", 1)), Ok(vec![0.25, 0.04]));
    }
    #[test]
    fn stale_action_is_rejected() {
        let mut g = guard(); g.start_episode("a").unwrap();
        g.observe(&obs("a", 1, 0.1)); g.observe(&obs("a", 2, 0.1));
        assert_eq!(g.approve_action(&action("a", 1)),
            Err(ActionRejection::StaleObservation { observed_step: 2, action_step: 1 }));
    }
    #[test]
    fn wrong_episode_latches_stop_until_reset() {
        let mut g = guard(); g.start_episode("a").unwrap();
        assert_eq!(g.observe(&obs("b", 1, 0.1)), ObservationDecision::SafeStop(StopReason::EpisodeMismatch));
        assert!(matches!(g.approve_action(&action("a", 1)), Err(ActionRejection::SafeStop(_))));
        g.start_episode("c").unwrap();
        assert_eq!(g.observe(&obs("c", 0, 0.1)), ObservationDecision::Accepted);
    }
    #[test]
    fn non_finite_observation_stops() {
        let mut g = guard(); g.start_episode("a").unwrap();
        let mut frame = obs("a", 1, 0.1); frame.state_values[1] = f64::NAN;
        assert_eq!(g.observe(&frame), ObservationDecision::SafeStop(StopReason::NonFiniteObservation { index: 1 }));
    }
    #[test]
    fn repeated_or_out_of_order_steps_stop() {
        let mut g = guard(); g.start_episode("a").unwrap(); g.observe(&obs("a", 7, 0.1));
        assert_eq!(g.observe(&obs("a", 7, 0.1)),
            ObservationDecision::SafeStop(StopReason::NonMonotonicStep { previous: 7, received: 7 }));
    }
    #[test]
    fn high_error_requests_replan_and_blocks_action_until_recovery() {
        let mut g = guard(); g.start_episode("a").unwrap();
        assert_eq!(g.observe(&obs("a", 1, 0.9)),
            ObservationDecision::RequestReplan { prediction_error: 0.9, consecutive_high_error: 1 });
        assert_eq!(g.approve_action(&action("a", 1)), Err(ActionRejection::ReplanRequired { observed_step: 1 }));
        assert_eq!(g.observe(&obs("a", 2, 0.1)), ObservationDecision::Accepted);
        assert_eq!(g.consecutive_high_error(), 0);
        assert!(g.approve_action(&action("a", 2)).is_ok());
    }
    #[test]
    fn sustained_high_error_latches_stop() {
        let mut g = guard(); g.start_episode("a").unwrap();
        g.observe(&obs("a", 1, 0.9)); g.observe(&obs("a", 2, 0.9));
        assert_eq!(g.observe(&obs("a", 3, 0.9)),
            ObservationDecision::SafeStop(StopReason::SustainedPredictionError { observations: 3, threshold: 0.5 }));
    }
    #[test]
    fn non_finite_action_is_never_returned() {
        let mut g = guard(); g.start_episode("a").unwrap(); g.observe(&obs("a", 1, 0.1));
        let mut bad = action("a", 1); bad.values[0] = f64::INFINITY;
        assert_eq!(g.approve_action(&bad), Err(ActionRejection::SafeStop(StopReason::NonFiniteAction { index: 0 })));
    }
    #[test]
    fn out_of_range_action_is_never_returned() {
        let mut g = guard(); g.start_episode("a").unwrap(); g.observe(&obs("a", 1, 0.1));
        let mut bad = action("a", 1); bad.values[1] = 0.1;
        assert!(matches!(g.approve_action(&bad), Err(ActionRejection::SafeStop(StopReason::ActionOutOfBounds { index: 1, .. }))));
    }
    #[test]
    fn episode_reset_clears_previous_state() {
        let mut g = guard(); g.start_episode("a").unwrap(); g.observe(&obs("a", 10, 0.1));
        g.observe(&obs("a", 11, 0.9)); g.start_episode("b").unwrap();
        assert_eq!(g.last_observed_step(), None); assert_eq!(g.consecutive_high_error(), 0);
        assert_eq!(g.approve_action(&action("b", 0)), Err(ActionRejection::NoObservation));
    }
    #[test]
    fn invalid_configuration_is_rejected() {
        assert_eq!(AxisLimit::new(2.0, 1.0), Err(ConfigError::InvalidAxisLimit));
        assert_eq!(GateConfig::new(0, vec![AxisLimit { min: 0.0, max: 1.0 }], 0.5, 3), Err(ConfigError::EmptyState));
        assert_eq!(GateConfig::new(1, Vec::new(), 0.5, 3), Err(ConfigError::EmptyActionSpace));
        assert_eq!(GateConfig::new(1, vec![AxisLimit { min: 0.0, max: 1.0 }], f64::NAN, 3), Err(ConfigError::InvalidPredictionThreshold));
        assert_eq!(GateConfig::new(1, vec![AxisLimit { min: 0.0, max: 1.0 }], 0.5, 0), Err(ConfigError::ZeroRecoveryBudget));
    }
    #[test]
    fn same_trace_has_same_decisions() {
        fn trace() -> Vec<ObservationDecision> {
            let mut g = guard(); g.start_episode("repeatable").unwrap();
            vec![g.observe(&obs("repeatable", 0, 0.1)), g.observe(&obs("repeatable", 1, 0.7)), g.observe(&obs("repeatable", 2, 0.2))]
        }
        assert_eq!(trace(), trace());
    }
}
