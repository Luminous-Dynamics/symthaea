//! Bounded stateful model of adaptive photonic fusion outcomes.
//!
//! This is an abstract transition model, not a photonics/noise simulator. It keeps
//! successful fusion, recoverable state transitions, resource restart and terminal
//! failure distinct. Outcome probabilities are conditional on the current state.
//! All horizons are bounded by max_attempts, so retries cannot cause unbounded work.

use std::collections::HashSet;
use std::error::Error;
use std::fmt;

const MAX_FUSION_STATES: usize = 1024;
const MAX_FUSION_ATTEMPTS: usize = 10_000;
const PROBABILITY_SUM_TOLERANCE: f64 = 1e-12;

/// Transition taken after one mutually exclusive fusion outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionTransition {
    /// The desired resource-state/fusion operation succeeded.
    Success,
    /// The operation is unrecoverable for this run.
    TerminalFailure,
    /// A recoverable outcome leaves the process in the specified state.
    ContinueTo(usize),
    /// The current resource must be restarted at the process's declared start state.
    Restart,
}

/// One conditional outcome of a fusion operation.
///
/// Probabilities and resource increments are conditional on the parent state and
/// one attempt. Latency includes only this attempt's declared gate/feedforward
/// boundary; source setup, external queueing and unlisted overhead are excluded.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionOutcome {
    /// Stable outcome identity, unique within one state.
    pub id: String,
    /// Probability of this outcome conditional on the current state.
    pub probability: f64,
    /// Feedforward/control latency associated with this outcome, in milliseconds.
    pub feedforward_latency_ms: f64,
    /// Incremental photon/resource count consumed by this outcome.
    pub photons_consumed: u64,
    /// State-machine transition after this outcome is observed.
    pub transition: FusionTransition,
}

/// Fusion state with its attempt cost and complete set of conditional outcomes.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionState {
    /// Stable state identity.
    pub id: String,
    /// Duration of one fusion attempt in this state, in milliseconds.
    pub attempt_latency_ms: f64,
    /// Mutually exclusive and collectively exhaustive outcomes for this state.
    pub outcomes: Vec<FusionOutcome>,
}

/// Bounded finite-state fusion process representing a fixed/adaptive RUS policy.
#[derive(Debug, Clone, PartialEq)]
pub struct FusionProcess {
    /// Index of the initial state and restart destination.
    pub start_state: usize,
    /// Maximum number of attempted fusion transitions before cap exhaustion.
    pub max_attempts: usize,
    /// State definitions. Every state must be reachable from start through
    /// positive-probability transitions.
    pub states: Vec<FusionState>,
}

/// Aggregate outcome/cost estimate for one bounded process invocation.
///
/// Expected costs are unconditional per invocation. The success-conditional fields
/// are expectations among successful invocations; None means success has zero
/// probability under the declared model.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FusionProcessEstimate {
    /// Probability of reaching a successful terminal outcome within the cap.
    pub success_probability: f64,
    /// Probability of terminal unrecoverable failure within the cap.
    pub terminal_failure_probability: f64,
    /// Probability that the attempt cap is reached while another transition is required.
    pub attempt_cap_exhaustion_probability: f64,
    /// Expected fusion attempts per invocation, including unsuccessful invocations.
    pub expected_attempts: f64,
    /// Expected elapsed modeled time per invocation, in milliseconds.
    pub expected_latency_ms: f64,
    /// Expected photons/resource units consumed per invocation.
    pub expected_photons_consumed: f64,
    /// Expected fusion attempts conditional on success, if success is possible.
    pub expected_attempts_given_success: Option<f64>,
    /// Expected modeled latency conditional on success, if success is possible.
    pub expected_latency_ms_given_success: Option<f64>,
    /// Expected resource consumption conditional on success, if success is possible.
    pub expected_photons_consumed_given_success: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default)]
struct AccumulatedEstimate {
    success_probability: f64,
    terminal_failure_probability: f64,
    attempt_cap_exhaustion_probability: f64,
    expected_attempts: f64,
    expected_latency_ms: f64,
    expected_photons_consumed: f64,
    successful_attempts_moment: f64,
    successful_latency_moment: f64,
    successful_photons_moment: f64,
}

impl FusionProcess {
    /// Validate the state machine before any cost calculation.
    pub fn validate(&self) -> Result<(), FusionProcessError> {
        if self.states.is_empty() || self.states.len() > MAX_FUSION_STATES {
            return Err(FusionProcessError::InvalidStateCount(self.states.len()));
        }
        if self.start_state >= self.states.len() {
            return Err(FusionProcessError::InvalidStartState(self.start_state));
        }
        if self.max_attempts == 0 || self.max_attempts > MAX_FUSION_ATTEMPTS {
            return Err(FusionProcessError::InvalidAttemptCap(self.max_attempts));
        }

        let mut state_ids = HashSet::with_capacity(self.states.len());
        for state in &self.states {
            validate_fusion_id(&state.id, "state_id")?;
            if !state_ids.insert(state.id.as_str()) {
                return Err(FusionProcessError::DuplicateStateId(state.id.clone()));
            }
            if !state.attempt_latency_ms.is_finite() || state.attempt_latency_ms <= 0.0 {
                return Err(FusionProcessError::InvalidAttemptLatency {
                    state_id: state.id.clone(),
                    value: state.attempt_latency_ms,
                });
            }
            if state.outcomes.is_empty() {
                return Err(FusionProcessError::NoOutcomes(state.id.clone()));
            }

            let mut local_outcome_ids = HashSet::with_capacity(state.outcomes.len());
            let mut probability_sum = 0.0;
            for outcome in &state.outcomes {
                validate_fusion_id(&outcome.id, "outcome_id")?;
                if !local_outcome_ids.insert(outcome.id.as_str()) {
                    return Err(FusionProcessError::DuplicateOutcomeId {
                        state_id: state.id.clone(),
                        outcome_id: outcome.id.clone(),
                    });
                }
                if !outcome.probability.is_finite() || !(0.0..=1.0).contains(&outcome.probability) {
                    return Err(FusionProcessError::InvalidOutcomeProbability {
                        state_id: state.id.clone(),
                        outcome_id: outcome.id.clone(),
                        value: outcome.probability,
                    });
                }
                if !outcome.feedforward_latency_ms.is_finite()
                    || outcome.feedforward_latency_ms < 0.0
                {
                    return Err(FusionProcessError::InvalidFeedforwardLatency {
                        state_id: state.id.clone(),
                        outcome_id: outcome.id.clone(),
                        value: outcome.feedforward_latency_ms,
                    });
                }
                if let FusionTransition::ContinueTo(target) = outcome.transition {
                    if target >= self.states.len() {
                        return Err(FusionProcessError::InvalidTransitionTarget {
                            state_id: state.id.clone(),
                            outcome_id: outcome.id.clone(),
                            target,
                        });
                    }
                }
                probability_sum += outcome.probability;
            }
            if (probability_sum - 1.0).abs() > PROBABILITY_SUM_TOLERANCE {
                return Err(FusionProcessError::OutcomeProbabilitySum {
                    state_id: state.id.clone(),
                    sum: probability_sum,
                });
            }
        }

        // Reject dead configuration branches so every declared state participates in
        // the policy. Probability-zero transitions do not establish reachability.
        let mut reached = vec![false; self.states.len()];
        let mut pending = vec![self.start_state];
        reached[self.start_state] = true;
        let mut cursor = 0;
        while cursor < pending.len() {
            let index = pending[cursor];
            cursor += 1;
            for outcome in &self.states[index].outcomes {
                if outcome.probability <= 0.0 {
                    continue;
                }
                let next = match outcome.transition {
                    FusionTransition::ContinueTo(target) => Some(target),
                    FusionTransition::Restart => Some(self.start_state),
                    FusionTransition::Success | FusionTransition::TerminalFailure => None,
                };
                if let Some(next) = next {
                    if !reached[next] {
                        reached[next] = true;
                        pending.push(next);
                    }
                }
            }
        }
        if let Some(index) = reached.iter().position(|value| !value) {
            return Err(FusionProcessError::UnreachableState(index));
        }

        Ok(())
    }

    /// Compute exact finite-horizon Markov expectations for the declared transition model.
    ///
    /// This is dynamic programming over remaining attempt count, not Monte Carlo.
    /// Transition probabilities are assumed known and state-conditional; uncertainty,
    /// correlations, physical calibration and latency distributions are out of scope.
    pub fn estimate(&self) -> Result<FusionProcessEstimate, FusionProcessError> {
        self.validate()?;

        // With zero attempts left, a live process is exhausted. This base row makes
        // all retries and restarts bounded by max_attempts.
        let mut previous = vec![
            AccumulatedEstimate {
                attempt_cap_exhaustion_probability: 1.0,
                ..AccumulatedEstimate::default()
            };
            self.states.len()
        ];

        for _remaining in 1..=self.max_attempts {
            let mut current = Vec::with_capacity(self.states.len());
            for state in &self.states {
                let mut total = AccumulatedEstimate::default();
                for outcome in &state.outcomes {
                    let probability = outcome.probability;
                    let attempt_latency = state.attempt_latency_ms + outcome.feedforward_latency_ms;
                    if !attempt_latency.is_finite() {
                        return Err(FusionProcessError::InvalidTotalLatency {
                            state_id: state.id.clone(),
                            outcome_id: outcome.id.clone(),
                        });
                    }
                    let photons = outcome.photons_consumed as f64;

                    total.expected_attempts += probability;
                    total.expected_latency_ms += probability * attempt_latency;
                    total.expected_photons_consumed += probability * photons;

                    match outcome.transition {
                        FusionTransition::Success => {
                            total.success_probability += probability;
                            total.successful_attempts_moment += probability;
                            total.successful_latency_moment += probability * attempt_latency;
                            total.successful_photons_moment += probability * photons;
                        }
                        FusionTransition::TerminalFailure => {
                            total.terminal_failure_probability += probability;
                        }
                        FusionTransition::ContinueTo(target) => {
                            accumulate_continuation(
                                &mut total,
                                probability,
                                attempt_latency,
                                photons,
                                previous[target],
                            );
                        }
                        FusionTransition::Restart => {
                            accumulate_continuation(
                                &mut total,
                                probability,
                                attempt_latency,
                                photons,
                                previous[self.start_state],
                            );
                        }
                    }
                }
                current.push(total);
            }
            previous = current;
        }

        let result = previous[self.start_state];
        let outcome_probability_sum = result.success_probability
            + result.terminal_failure_probability
            + result.attempt_cap_exhaustion_probability;
        if !outcome_probability_sum.is_finite()
            || (outcome_probability_sum - 1.0).abs() > 1e-9
        {
            return Err(FusionProcessError::InvalidComputedProbabilitySum(outcome_probability_sum));
        }
        for (field, value) in [
            ("expected_attempts", result.expected_attempts),
            ("expected_latency_ms", result.expected_latency_ms),
            ("expected_photons_consumed", result.expected_photons_consumed),
            ("successful_attempts_moment", result.successful_attempts_moment),
            ("successful_latency_moment", result.successful_latency_moment),
            ("successful_photons_moment", result.successful_photons_moment),
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(FusionProcessError::InvalidComputedCost { field, value });
            }
        }

        let conditional = |moment: f64| {
            (result.success_probability > 0.0)
                .then(|| moment / result.success_probability)
        };
        Ok(FusionProcessEstimate {
            success_probability: result.success_probability,
            terminal_failure_probability: result.terminal_failure_probability,
            attempt_cap_exhaustion_probability: result.attempt_cap_exhaustion_probability,
            expected_attempts: result.expected_attempts,
            expected_latency_ms: result.expected_latency_ms,
            expected_photons_consumed: result.expected_photons_consumed,
            expected_attempts_given_success: conditional(result.successful_attempts_moment),
            expected_latency_ms_given_success: conditional(result.successful_latency_moment),
            expected_photons_consumed_given_success: conditional(result.successful_photons_moment),
        })
    }
}

fn accumulate_continuation(
    total: &mut AccumulatedEstimate,
    probability: f64,
    current_latency_ms: f64,
    current_photons: f64,
    future: AccumulatedEstimate,
) {
    total.success_probability += probability * future.success_probability;
    total.terminal_failure_probability += probability * future.terminal_failure_probability;
    total.attempt_cap_exhaustion_probability +=
        probability * future.attempt_cap_exhaustion_probability;
    total.expected_attempts += probability * future.expected_attempts;
    total.expected_latency_ms += probability * future.expected_latency_ms;
    total.expected_photons_consumed += probability * future.expected_photons_consumed;

    total.successful_attempts_moment += probability
        * (future.successful_attempts_moment + future.success_probability);
    total.successful_latency_moment += probability
        * (future.successful_latency_moment + future.success_probability * current_latency_ms);
    total.successful_photons_moment += probability
        * (future.successful_photons_moment + future.success_probability * current_photons);
}

fn validate_fusion_id(value: &str, field: &'static str) -> Result<(), FusionProcessError> {
    if value.is_empty()
        || !value.bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'-')
        })
    {
        return Err(FusionProcessError::InvalidIdentifier {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

/// Validation/evaluation error for a bounded fusion transition process.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionProcessError {
    /// State vector is empty or exceeds the supported state budget.
    InvalidStateCount(usize),
    /// Initial state index is outside the state vector.
    InvalidStartState(usize),
    /// Attempt cap is zero or exceeds the supported finite horizon.
    InvalidAttemptCap(usize),
    /// State or outcome ID is empty or not canonical single-token ASCII.
    InvalidIdentifier { field: &'static str, value: String },
    /// Two states share an identity.
    DuplicateStateId(String),
    /// A declared state is not reachable under any positive-probability transition.
    UnreachableState(usize),
    /// State has no outcomes.
    NoOutcomes(String),
    /// State attempt latency is non-finite or non-positive.
    InvalidAttemptLatency { state_id: String, value: f64 },
    /// Outcome ID is duplicated in one state.
    DuplicateOutcomeId { state_id: String, outcome_id: String },
    /// Outcome probability is outside [0, 1] or non-finite.
    InvalidOutcomeProbability { state_id: String, outcome_id: String, value: f64 },
    /// Outcome feedforward latency is negative or non-finite.
    InvalidFeedforwardLatency { state_id: String, outcome_id: String, value: f64 },
    /// Attempt and feedforward latency sum is non-finite.
    InvalidTotalLatency { state_id: String, outcome_id: String },
    /// Outcome probabilities do not sum to one within tolerance.
    OutcomeProbabilitySum { state_id: String, sum: f64 },
    /// State transition target is outside the state vector.
    InvalidTransitionTarget { state_id: String, outcome_id: String, target: usize },
    /// Computed terminal outcome probabilities do not sum to one.
    InvalidComputedProbabilitySum(f64),
    /// A computed expected cost or success-weighted cost is negative or non-finite.
    InvalidComputedCost { field: &'static str, value: f64 },
}

impl fmt::Display for FusionProcessError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStateCount(count) => write!(f, "invalid fusion state count: {count}"),
            Self::InvalidStartState(state) => write!(f, "invalid fusion start state: {state}"),
            Self::InvalidAttemptCap(cap) => write!(f, "invalid fusion attempt cap: {cap}"),
            Self::InvalidIdentifier { field, value } =>
                write!(f, "invalid fusion identifier for {field}: {value}"),
            Self::DuplicateStateId(id) => write!(f, "duplicate fusion state ID: {id}"),
            Self::UnreachableState(state) => write!(f, "unreachable fusion state: {state}"),
            Self::NoOutcomes(state) => write!(f, "fusion state has no outcomes: {state}"),
            Self::InvalidAttemptLatency { state_id, value } =>
                write!(f, "invalid attempt latency for state {state_id}: {value} ms"),
            Self::DuplicateOutcomeId { state_id, outcome_id } =>
                write!(f, "duplicate outcome {outcome_id} in state {state_id}"),
            Self::InvalidOutcomeProbability { state_id, outcome_id, value } => write!(
                f,
                "invalid probability {value} for outcome {outcome_id} in state {state_id}"
            ),
            Self::InvalidFeedforwardLatency { state_id, outcome_id, value } => write!(
                f,
                "invalid feedforward latency for outcome {outcome_id} in state {state_id}: {value} ms"
            ),
            Self::InvalidTotalLatency { state_id, outcome_id } => write!(
                f,
                "attempt and feedforward latency overflow for {outcome_id} in state {state_id}"
            ),
            Self::OutcomeProbabilitySum { state_id, sum } =>
                write!(f, "outcome probabilities for state {state_id} sum to {sum}, expected 1"),
            Self::InvalidTransitionTarget { state_id, outcome_id, target } => write!(
                f,
                "outcome {outcome_id} in state {state_id} targets invalid state {target}"
            ),
            Self::InvalidComputedProbabilitySum(sum) =>
                write!(f, "computed fusion terminal probabilities sum to {sum}, expected 1"),
            Self::InvalidComputedCost { field, value } =>
                write!(f, "computed fusion cost {field} is invalid: {value}"),
        }
    }
}

impl Error for FusionProcessError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn outcome(
        id: &str,
        probability: f64,
        feedforward_latency_ms: f64,
        photons_consumed: u64,
        transition: FusionTransition,
    ) -> FusionOutcome {
        FusionOutcome {
            id: id.to_owned(),
            probability,
            feedforward_latency_ms,
            photons_consumed,
            transition,
        }
    }

    fn one_state_rus(max_attempts: usize) -> FusionProcess {
        FusionProcess {
            start_state: 0,
            max_attempts,
            states: vec![FusionState {
                id: "fusion".to_owned(),
                attempt_latency_ms: 2.0,
                outcomes: vec![
                    outcome("success", 0.5, 0.2, 2, FusionTransition::Success),
                    outcome("recoverable", 0.3, 0.5, 1, FusionTransition::ContinueTo(0)),
                    outcome("destructive", 0.2, 0.1, 2, FusionTransition::TerminalFailure),
                ],
            }],
        }
    }

    #[test]
    fn bounded_rus_distinguishes_success_terminal_failure_and_cap_exhaustion() {
        let result = one_state_rus(3).estimate().unwrap();
        assert!((result.success_probability - 0.695).abs() < 1e-12);
        assert!((result.terminal_failure_probability - 0.278).abs() < 1e-12);
        assert!((result.attempt_cap_exhaustion_probability - 0.027).abs() < 1e-12);
        assert!((result.expected_attempts - 1.39).abs() < 1e-12);
        assert!((result.expected_latency_ms - 3.1553).abs() < 1e-10);
        assert!((result.expected_photons_consumed - 2.363).abs() < 1e-12);
        assert!((result.expected_attempts_given_success.unwrap() - (0.935 / 0.695)).abs() < 1e-12);
    }

    #[test]
    fn adaptive_transition_uses_outcome_dependent_next_state() {
        let process = FusionProcess {
            start_state: 0,
            max_attempts: 2,
            states: vec![
                FusionState {
                    id: "source-ready".to_owned(),
                    attempt_latency_ms: 1.0,
                    outcomes: vec![
                        outcome("direct-success", 0.2, 0.0, 2, FusionTransition::Success),
                        outcome("switch-to-recovery", 0.8, 0.1, 1, FusionTransition::ContinueTo(1)),
                    ],
                },
                FusionState {
                    id: "reconfigured-emitter".to_owned(),
                    attempt_latency_ms: 2.0,
                    outcomes: vec![
                        outcome("recovery-success", 0.9, 0.2, 2, FusionTransition::Success),
                        outcome("recovery-failed", 0.1, 0.0, 1, FusionTransition::TerminalFailure),
                    ],
                },
            ],
        };
        let result = process.estimate().unwrap();
        assert!((result.success_probability - 0.92).abs() < 1e-12);
        assert!((result.terminal_failure_probability - 0.08).abs() < 1e-12);
        assert_eq!(result.attempt_cap_exhaustion_probability, 0.0);
        assert!((result.expected_attempts - 1.8).abs() < 1e-12);
    }

    #[test]
    fn restart_returns_to_declared_start_state_and_obeys_attempt_cap() {
        let process = FusionProcess {
            start_state: 0,
            max_attempts: 2,
            states: vec![FusionState {
                id: "start".to_owned(),
                attempt_latency_ms: 1.0,
                outcomes: vec![
                    outcome("success", 0.4, 0.0, 1, FusionTransition::Success),
                    outcome("restart", 0.6, 0.0, 1, FusionTransition::Restart),
                ],
            }],
        };
        let result = process.estimate().unwrap();
        assert!((result.success_probability - 0.64).abs() < 1e-12);
        assert!((result.attempt_cap_exhaustion_probability - 0.36).abs() < 1e-12);
        assert_eq!(result.terminal_failure_probability, 0.0);
    }

    #[test]
    fn no_success_has_none_for_conditional_success_costs() {
        let process = FusionProcess {
            start_state: 0,
            max_attempts: 2,
            states: vec![FusionState {
                id: "only-failure".to_owned(),
                attempt_latency_ms: 1.0,
                outcomes: vec![outcome("fail", 1.0, 0.0, 3, FusionTransition::TerminalFailure)],
            }],
        };
        let result = process.estimate().unwrap();
        assert_eq!(result.success_probability, 0.0);
        assert_eq!(result.expected_attempts_given_success, None);
        assert_eq!(result.expected_latency_ms_given_success, None);
        assert_eq!(result.expected_photons_consumed_given_success, None);
    }

    #[test]
    fn rejects_invalid_probability_sums_duplicates_and_unreachable_states() {
        let mut process = one_state_rus(2);
        process.states[0].outcomes[0].probability = 0.4;
        assert!(matches!(
            process.estimate(),
            Err(FusionProcessError::OutcomeProbabilitySum { .. })
        ));

        process = one_state_rus(2);
        process.states[0].outcomes[1].id = "success".to_owned();
        assert!(matches!(
            process.estimate(),
            Err(FusionProcessError::DuplicateOutcomeId { .. })
        ));

        process = one_state_rus(2);
        process.states.push(FusionState {
            id: "orphan".to_owned(),
            attempt_latency_ms: 1.0,
            outcomes: vec![outcome("orphan-success", 1.0, 0.0, 1, FusionTransition::Success)],
        });
        assert!(matches!(
            process.estimate(),
            Err(FusionProcessError::UnreachableState(1))
        ));
    }

    #[test]
    fn rejects_bad_transition_targets_caps_and_latencies() {
        let mut process = one_state_rus(1);
        process.states[0].outcomes[1].transition = FusionTransition::ContinueTo(9);
        assert!(matches!(
            process.estimate(),
            Err(FusionProcessError::InvalidTransitionTarget { target: 9, .. })
        ));

        process = one_state_rus(0);
        assert!(matches!(process.estimate(), Err(FusionProcessError::InvalidAttemptCap(0))));

        process = one_state_rus(1);
        process.states[0].attempt_latency_ms = f64::NAN;
        assert!(matches!(
            process.estimate(),
            Err(FusionProcessError::InvalidAttemptLatency { .. })
        ));

        process = one_state_rus(1);
        process.states[0].outcomes[0].feedforward_latency_ms = -1.0;
        assert!(matches!(
            process.estimate(),
            Err(FusionProcessError::InvalidFeedforwardLatency { .. })
        ));
    }

    #[test]
    fn zero_probability_transition_does_not_make_a_state_reachable() {
        let process = FusionProcess {
            start_state: 0,
            max_attempts: 1,
            states: vec![
                FusionState {
                    id: "start".to_owned(),
                    attempt_latency_ms: 1.0,
                    outcomes: vec![
                        outcome("success", 1.0, 0.0, 1, FusionTransition::Success),
                        outcome("impossible-branch", 0.0, 0.0, 0, FusionTransition::ContinueTo(1)),
                    ],
                },
                FusionState {
                    id: "unreachable".to_owned(),
                    attempt_latency_ms: 1.0,
                    outcomes: vec![outcome("finish", 1.0, 0.0, 1, FusionTransition::Success)],
                },
            ],
        };
        assert!(matches!(
            process.estimate(),
            Err(FusionProcessError::UnreachableState(1))
        ));
    }

    #[test]
    fn one_state_geometric_control_matches_closed_form_across_caps() {
        let success = 0.5_f64;
        let recoverable = 0.3_f64;
        let destructive = 0.2_f64;
        let mean_latency_per_attempt = 2.27_f64;
        let mean_photons_per_attempt = 1.7_f64;

        for cap in 1usize..=8 {
            let result = one_state_rus(cap).estimate().unwrap();
            let recoverable_sum = (1.0 - recoverable.powi(cap as i32)) / (1.0 - recoverable);
            let success_probability = success * recoverable_sum;
            let terminal_failure_probability = destructive * recoverable_sum;
            let exhaustion_probability = recoverable.powi(cap as i32);

            assert!((result.success_probability - success_probability).abs() < 1e-12);
            assert!(
                (result.terminal_failure_probability - terminal_failure_probability).abs() < 1e-12
            );
            assert!(
                (result.attempt_cap_exhaustion_probability - exhaustion_probability).abs() < 1e-12
            );
            assert!((result.expected_attempts - recoverable_sum).abs() < 1e-12);
            assert!(
                (result.expected_latency_ms
                    - mean_latency_per_attempt * recoverable_sum)
                    .abs()
                    < 1e-10
            );
            assert!(
                (result.expected_photons_consumed
                    - mean_photons_per_attempt * recoverable_sum)
                    .abs()
                    < 1e-12
            );

            let mut successful_attempt_moment = 0.0;
            let mut successful_latency_moment = 0.0;
            let mut successful_photon_moment = 0.0;
            for attempt_index in 0..cap {
                let path_probability = recoverable.powi(attempt_index as i32) * success;
                successful_attempt_moment += path_probability * (attempt_index + 1) as f64;
                successful_latency_moment +=
                    path_probability * (attempt_index as f64 * 2.5 + 2.2);
                successful_photon_moment += path_probability * (attempt_index as f64 + 2.0);
            }
            assert!(
                (result.expected_attempts_given_success.unwrap()
                    - successful_attempt_moment / success_probability).abs() < 1e-12
            );
            assert!(
                (result.expected_latency_ms_given_success.unwrap()
                    - successful_latency_moment / success_probability).abs() < 1e-12
            );
            assert!(
                (result.expected_photons_consumed_given_success.unwrap()
                    - successful_photon_moment / success_probability).abs() < 1e-12
            );
        }
    }

    #[test]
    fn rejects_invalid_probabilities_and_overflowing_combined_latency() {
        let mut process = one_state_rus(2);
        process.states[0].outcomes[0].probability = f64::NAN;
        assert!(matches!(
            process.estimate(),
            Err(FusionProcessError::InvalidOutcomeProbability { .. })
        ));

        process = one_state_rus(2);
        process.states[0].outcomes[0].probability = 1.1;
        assert!(matches!(
            process.estimate(),
            Err(FusionProcessError::InvalidOutcomeProbability { .. })
        ));

        process = one_state_rus(1);
        process.states[0].attempt_latency_ms = 1.0e308;
        process.states[0].outcomes[0].feedforward_latency_ms = 1.0e308;
        assert!(matches!(
            process.estimate(),
            Err(FusionProcessError::InvalidTotalLatency { .. })
        ));
    }

    #[test]
    fn rejects_overflow_in_aggregated_expected_latency() {
        let process = FusionProcess {
            start_state: 0,
            max_attempts: 3,
            states: vec![FusionState {
                id: "zero-yield-loop".to_owned(),
                attempt_latency_ms: 1.0e308,
                outcomes: vec![outcome(
                    "continue",
                    1.0,
                    0.0,
                    1,
                    FusionTransition::ContinueTo(0),
                )],
            }],
        };
        assert!(matches!(
            process.estimate(),
            Err(FusionProcessError::InvalidComputedCost {
                field: "expected_latency_ms",
                ..
            })
        ));
    }

    #[test]
    fn estimate_is_deterministic_for_the_same_declared_process() {
        let process = one_state_rus(5);
        assert_eq!(process.estimate().unwrap(), process.estimate().unwrap());
    }
}
