pub mod paired_world;

use std::error::Error;
use std::fmt;

/// Qualification state of an independently established reference exponent.
///
/// Lyapunov-time normalization is intentionally allowed only for `Reliable`.
/// A candidate's own diagnostic estimate must never normalize its score.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EstimatorQualification {
    Reliable,
    ConditionallyReliable,
    DiagnosticOnly,
    Inconclusive,
}

/// Errors returned by the evaluator-side metric contracts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MetricError {
    EmptyForecast,
    InvalidThreshold,
    NonFiniteValue(&'static str),
    NegativeValue(&'static str),
    NonPositiveValue(&'static str),
    NonIncreasingForecastTime,
    ReferenceNotReliable,
    EmptyInformationSets,
    InvalidProbability(&'static str),
    ProbabilityMassNotOne(&'static str),
    InconsistentActionCount,
    InvalidActionIndex,
    CandidateActionRequired,
    InformationIncreasedExpectedRisk,
}

impl fmt::Display for MetricError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyForecast => write!(formatter, "forecast series is empty"),
            Self::InvalidThreshold => {
                write!(formatter, "forecast threshold must be finite and positive")
            }
            Self::NonFiniteValue(name) => write!(formatter, "{name} must be finite"),
            Self::NegativeValue(name) => write!(formatter, "{name} must be non-negative"),
            Self::NonPositiveValue(name) => write!(formatter, "{name} must be positive"),
            Self::NonIncreasingForecastTime => {
                write!(formatter, "forecast times must be strictly increasing")
            }
            Self::ReferenceNotReliable => {
                write!(formatter, "Lyapunov reference is not qualified as Reliable")
            }
            Self::EmptyInformationSets => {
                write!(formatter, "at least one information set is required")
            }
            Self::InvalidProbability(name) => {
                write!(formatter, "{name} must be finite and in (0, 1]")
            }
            Self::ProbabilityMassNotOne(name) => {
                write!(formatter, "{name} must sum to one within tolerance")
            }
            Self::InconsistentActionCount => write!(
                formatter,
                "all states must declare the same non-zero action count"
            ),
            Self::InvalidActionIndex => write!(
                formatter,
                "candidate action index is outside the declared action set"
            ),
            Self::CandidateActionRequired => write!(
                formatter,
                "a candidate action is required to calculate policy risk"
            ),
            Self::InformationIncreasedExpectedRisk => write!(
                formatter,
                "expected posterior Bayes risk cannot exceed prior Bayes risk"
            ),
        }
    }
}

impl Error for MetricError {}

/// A reference maximal Lyapunov exponent, in explicitly declared inverse-time units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QualifiedLyapunovReference {
    pub lambda_max: f64,
    pub qualification: EstimatorQualification,
}

impl QualifiedLyapunovReference {
    /// Return the Lyapunov time only for a finite, positive, reliable reference.
    pub fn lyapunov_time(self) -> Result<f64, MetricError> {
        if self.qualification != EstimatorQualification::Reliable {
            return Err(MetricError::ReferenceNotReliable);
        }
        if !self.lambda_max.is_finite() {
            return Err(MetricError::NonFiniteValue("lambda_max"));
        }
        if self.lambda_max <= 0.0 {
            return Err(MetricError::NonPositiveValue("lambda_max"));
        }
        let time = 1.0 / self.lambda_max;
        if !time.is_finite() {
            return Err(MetricError::NonFiniteValue("Lyapunov time"));
        }
        Ok(time)
    }
}

/// One point on a free-running or teacher-forced forecast error curve.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ForecastSample {
    /// Forecast horizon in the source system's declared time unit.
    pub horizon: f64,
    /// Non-negative normalized prediction error at this horizon.
    pub normalized_error: f64,
}

/// Observed forecast horizon under a predeclared error threshold.
///
/// The result is the last sampled horizon in the initial contiguous prefix
/// whose error does not exceed `threshold`. If the first point already fails,
/// the result is zero. This is an observed discrete threshold crossing, not a
/// continuous-time interpolation or confidence bound.
pub fn forecast_horizon(samples: &[ForecastSample], threshold: f64) -> Result<f64, MetricError> {
    if samples.is_empty() {
        return Err(MetricError::EmptyForecast);
    }
    if !threshold.is_finite() || threshold <= 0.0 {
        return Err(MetricError::InvalidThreshold);
    }

    let mut previous_time = 0.0;
    let mut last_passing_time = 0.0;
    let mut prefix_is_valid = true;

    for sample in samples {
        if !sample.horizon.is_finite() {
            return Err(MetricError::NonFiniteValue("forecast horizon"));
        }
        if sample.horizon <= previous_time {
            return Err(MetricError::NonIncreasingForecastTime);
        }
        if !sample.normalized_error.is_finite() {
            return Err(MetricError::NonFiniteValue("normalized forecast error"));
        }
        if sample.normalized_error < 0.0 {
            return Err(MetricError::NegativeValue("normalized forecast error"));
        }

        if prefix_is_valid && sample.normalized_error > threshold {
            prefix_is_valid = false;
        }
        if prefix_is_valid {
            last_passing_time = sample.horizon;
        }
        previous_time = sample.horizon;
    }

    Ok(last_passing_time)
}

/// Normalize a horizon using an independently qualified reference exponent.
pub fn horizon_in_lyapunov_times(
    horizon: f64,
    reference: QualifiedLyapunovReference,
) -> Result<f64, MetricError> {
    if !horizon.is_finite() {
        return Err(MetricError::NonFiniteValue("forecast horizon"));
    }
    if horizon < 0.0 {
        return Err(MetricError::NegativeValue("forecast horizon"));
    }
    let normalized = horizon / reference.lyapunov_time()?;
    if !normalized.is_finite() {
        return Err(MetricError::NonFiniteValue(
            "Lyapunov-normalized forecast horizon",
        ));
    }
    Ok(normalized)
}

/// Conditional state probabilities and losses for every available action
/// inside one candidate-observable information set.
#[derive(Clone, Debug, PartialEq)]
pub struct ConditionalStateLoss {
    /// Probability of this latent state conditional on the information set.
    pub probability: f64,
    /// Loss for each action, in the same order for every state.
    pub action_losses: Vec<f64>,
}

/// One candidate-observable information set.
///
/// `probability` is the probability mass of the information set under the
/// benchmark prior. `states` contains the evaluator's conditional latent
/// state distribution. `candidate_action` is the action selected by the
/// candidate. It may be `None` for `evaluate_bayes_risk`; policy regret via
/// `evaluate_decisions` requires an action for every information set.
#[derive(Clone, Debug, PartialEq)]
pub struct InformationSet {
    pub probability: f64,
    pub states: Vec<ConditionalStateLoss>,
    pub candidate_action: Option<usize>,
}

/// Bayes-risk lower bounds under candidate-observable information.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BayesRisk {
    /// Best expected loss available with the supplied information.
    pub bayes_risk: f64,
    /// Evaluator-only lower bound when the latent state is revealed.
    pub perfect_information_risk: f64,
}

/// Decision-risk decomposition under candidate-observable information.
///
/// - `bayes_risk`: best expected loss available with the same information.
/// - `policy_risk`: expected loss of the candidate's selected action.
/// - `perfect_information_risk`: evaluator-only lower bound with latent state
///   revealed before action selection.
/// - `regret`: policy risk minus information-set Bayes risk.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionRisk {
    pub bayes_risk: f64,
    pub policy_risk: f64,
    pub perfect_information_risk: f64,
    pub regret: f64,
}

const PROBABILITY_TOLERANCE: f64 = 1e-9;
const REGRET_ROUNDOFF_TOLERANCE: f64 = 1e-10;

fn validate_probability(value: f64, name: &'static str) -> Result<(), MetricError> {
    if !value.is_finite() || value <= 0.0 || value > 1.0 {
        return Err(MetricError::InvalidProbability(name));
    }
    Ok(())
}

fn validate_information_sets(sets: &[InformationSet]) -> Result<usize, MetricError> {
    if sets.is_empty() {
        return Err(MetricError::EmptyInformationSets);
    }

    let total_mass: f64 = sets.iter().map(|set| set.probability).sum();
    if !total_mass.is_finite() || (total_mass - 1.0).abs() > PROBABILITY_TOLERANCE {
        return Err(MetricError::ProbabilityMassNotOne(
            "information-set probability mass",
        ));
    }

    let mut action_count = None;

    for set in sets {
        validate_probability(set.probability, "information-set probability")?;
        if set.states.is_empty() {
            return Err(MetricError::ProbabilityMassNotOne(
                "conditional state probability mass",
            ));
        }

        let conditional_mass: f64 = set.states.iter().map(|state| state.probability).sum();
        if !conditional_mass.is_finite() || (conditional_mass - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(MetricError::ProbabilityMassNotOne(
                "conditional state probability mass",
            ));
        }

        for state in &set.states {
            validate_probability(state.probability, "conditional state probability")?;
            if state.action_losses.is_empty() {
                return Err(MetricError::InconsistentActionCount);
            }
            match action_count {
                Some(expected) if expected != state.action_losses.len() => {
                    return Err(MetricError::InconsistentActionCount);
                }
                None => action_count = Some(state.action_losses.len()),
                _ => {}
            }

            for loss in &state.action_losses {
                if !loss.is_finite() {
                    return Err(MetricError::NonFiniteValue("action loss"));
                }
                if *loss < 0.0 {
                    return Err(MetricError::NegativeValue("action loss"));
                }
            }
        }
    }

    let count = action_count.ok_or(MetricError::InconsistentActionCount)?;
    for set in sets {
        if let Some(action) = set.candidate_action
            && action >= count
        {
            return Err(MetricError::InvalidActionIndex);
        }
    }
    Ok(count)
}

/// Evaluate the information-set Bayes risk and perfect-information lower
/// bound without requiring a candidate policy action.
///
/// This is the appropriate evaluator for prior/posterior EVSI calculations:
/// candidate policy regret is a separate quantity and is evaluated by
/// `evaluate_decisions`.
pub fn evaluate_bayes_risk(sets: &[InformationSet]) -> Result<BayesRisk, MetricError> {
    let action_count = validate_information_sets(sets)?;
    let mut bayes_risk = 0.0;
    let mut perfect_information_risk = 0.0;

    for set in sets {
        let mut expected_losses = vec![0.0; action_count];
        let mut conditional_perfect_risk = 0.0;

        for state in &set.states {
            for (action, expected_loss) in expected_losses.iter_mut().enumerate() {
                *expected_loss += state.probability * state.action_losses[action];
            }

            let best_state_loss = state
                .action_losses
                .iter()
                .copied()
                .fold(f64::INFINITY, f64::min);
            conditional_perfect_risk += state.probability * best_state_loss;
        }

        if expected_losses.iter().any(|loss| !loss.is_finite())
            || !conditional_perfect_risk.is_finite()
        {
            return Err(MetricError::NonFiniteValue("expected decision risk"));
        }

        let best_information_set_loss = expected_losses
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        bayes_risk += set.probability * best_information_set_loss;
        perfect_information_risk += set.probability * conditional_perfect_risk;
    }

    if !bayes_risk.is_finite() || !perfect_information_risk.is_finite() {
        return Err(MetricError::NonFiniteValue("aggregate decision risk"));
    }

    Ok(BayesRisk {
        bayes_risk,
        perfect_information_risk,
    })
}

/// Evaluate policy risk and regret in addition to the Bayes-risk lower bounds.
/// Every information set must contain a candidate action.
pub fn evaluate_decisions(sets: &[InformationSet]) -> Result<DecisionRisk, MetricError> {
    let bayes = evaluate_bayes_risk(sets)?;
    let mut policy_risk = 0.0;

    for set in sets {
        let chosen_action = set
            .candidate_action
            .ok_or(MetricError::CandidateActionRequired)?;
        let conditional_policy_risk: f64 = set
            .states
            .iter()
            .map(|state| state.probability * state.action_losses[chosen_action])
            .sum();

        if !conditional_policy_risk.is_finite() {
            return Err(MetricError::NonFiniteValue("expected policy risk"));
        }
        policy_risk += set.probability * conditional_policy_risk;
    }

    if !policy_risk.is_finite() {
        return Err(MetricError::NonFiniteValue("aggregate policy risk"));
    }

    let raw_regret = policy_risk - bayes.bayes_risk;
    if !raw_regret.is_finite() {
        return Err(MetricError::NonFiniteValue("decision regret"));
    }
    if raw_regret < -REGRET_ROUNDOFF_TOLERANCE {
        return Err(MetricError::InformationIncreasedExpectedRisk);
    }

    Ok(DecisionRisk {
        bayes_risk: bayes.bayes_risk,
        policy_risk,
        perfect_information_risk: bayes.perfect_information_risk,
        regret: raw_regret.max(0.0),
    })
}

/// Expected value of an observation, given exact prior and expected posterior
/// Bayes risks and the observation's complete incremental cost in the same
/// predeclared task-loss units.
///
/// `expected_posterior_bayes_risk` is the expectation over all possible
/// observation outcomes, not the risk after one realized observation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InformationValue {
    pub gross_value: f64,
    pub acquisition_cost: f64,
    pub net_value: f64,
}

pub fn expected_value_of_information(
    prior_bayes_risk: f64,
    expected_posterior_bayes_risk: f64,
    acquisition_cost: f64,
) -> Result<InformationValue, MetricError> {
    for (name, value) in [
        ("prior Bayes risk", prior_bayes_risk),
        (
            "expected posterior Bayes risk",
            expected_posterior_bayes_risk,
        ),
        ("acquisition cost", acquisition_cost),
    ] {
        if !value.is_finite() {
            return Err(MetricError::NonFiniteValue(name));
        }
        if value < 0.0 {
            return Err(MetricError::NegativeValue(name));
        }
    }

    if expected_posterior_bayes_risk > prior_bayes_risk + REGRET_ROUNDOFF_TOLERANCE {
        return Err(MetricError::InformationIncreasedExpectedRisk);
    }

    let gross_value = (prior_bayes_risk - expected_posterior_bayes_risk).max(0.0);
    Ok(InformationValue {
        gross_value,
        acquisition_cost,
        net_value: gross_value - acquisition_cost,
    })
}

/// Resource dimensions for matched-workload Pareto comparisons.
///
/// All values are non-negative integer quantities with units encoded in field
/// names. A comparison is meaningful only when the measurement boundary,
/// workload and hardware are comparable.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResourceVector {
    pub sensor_samples: u64,
    pub sensor_wakeups: u64,
    pub acquisition_latency_us: u64,
    pub compute_time_us: u64,
    pub end_to_end_p99_latency_us: u64,
    pub peak_memory_bytes: u64,
    pub persistent_memory_bytes: u64,
    pub training_time_us: u64,
    pub update_time_us: u64,
    pub preprocessing_time_us: u64,
    pub measured_energy_uj: u64,
    pub communication_bytes: u64,
}

/// Componentwise Pareto dominance. Returns false for equal vectors.
pub fn pareto_dominates(left: ResourceVector, right: ResourceVector) -> bool {
    let no_worse = left.sensor_samples <= right.sensor_samples
        && left.sensor_wakeups <= right.sensor_wakeups
        && left.acquisition_latency_us <= right.acquisition_latency_us
        && left.compute_time_us <= right.compute_time_us
        && left.end_to_end_p99_latency_us <= right.end_to_end_p99_latency_us
        && left.peak_memory_bytes <= right.peak_memory_bytes
        && left.persistent_memory_bytes <= right.persistent_memory_bytes
        && left.training_time_us <= right.training_time_us
        && left.update_time_us <= right.update_time_us
        && left.preprocessing_time_us <= right.preprocessing_time_us
        && left.measured_energy_uj <= right.measured_energy_uj
        && left.communication_bytes <= right.communication_bytes;

    let strictly_better = left.sensor_samples < right.sensor_samples
        || left.sensor_wakeups < right.sensor_wakeups
        || left.acquisition_latency_us < right.acquisition_latency_us
        || left.compute_time_us < right.compute_time_us
        || left.end_to_end_p99_latency_us < right.end_to_end_p99_latency_us
        || left.peak_memory_bytes < right.peak_memory_bytes
        || left.persistent_memory_bytes < right.persistent_memory_bytes
        || left.training_time_us < right.training_time_us
        || left.update_time_us < right.update_time_us
        || left.preprocessing_time_us < right.preprocessing_time_us
        || left.measured_energy_uj < right.measured_energy_uj
        || left.communication_bytes < right.communication_bytes;

    no_worse && strictly_better
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(probability: f64, action_losses: &[f64]) -> ConditionalStateLoss {
        ConditionalStateLoss {
            probability,
            action_losses: action_losses.to_vec(),
        }
    }

    #[test]
    fn only_reliable_positive_reference_can_define_lyapunov_time() {
        let reference = QualifiedLyapunovReference {
            lambda_max: 0.5,
            qualification: EstimatorQualification::Reliable,
        };
        assert!((reference.lyapunov_time().unwrap() - 2.0).abs() < 1e-12);

        for qualification in [
            EstimatorQualification::ConditionallyReliable,
            EstimatorQualification::DiagnosticOnly,
            EstimatorQualification::Inconclusive,
        ] {
            let unqualified = QualifiedLyapunovReference {
                lambda_max: 0.5,
                qualification,
            };
            assert_eq!(
                unqualified.lyapunov_time(),
                Err(MetricError::ReferenceNotReliable)
            );
        }
    }

    #[test]
    fn lyapunov_time_rejects_nonpositive_and_nonfinite_exponents() {
        for lambda_max in [0.0, -0.1, f64::NAN, f64::INFINITY] {
            let reference = QualifiedLyapunovReference {
                lambda_max,
                qualification: EstimatorQualification::Reliable,
            };
            assert!(reference.lyapunov_time().is_err());
        }
    }

    #[test]
    fn forecast_horizon_is_the_last_passing_prefix_sample() {
        let samples = [
            ForecastSample {
                horizon: 1.0,
                normalized_error: 0.02,
            },
            ForecastSample {
                horizon: 2.0,
                normalized_error: 0.08,
            },
            ForecastSample {
                horizon: 3.0,
                normalized_error: 0.12,
            },
            ForecastSample {
                horizon: 4.0,
                normalized_error: 0.04,
            },
        ];
        assert_eq!(forecast_horizon(&samples, 0.1).unwrap(), 2.0);
    }

    #[test]
    fn forecast_horizon_reports_zero_if_first_sample_fails() {
        let samples = [ForecastSample {
            horizon: 1.0,
            normalized_error: 0.2,
        }];
        assert_eq!(forecast_horizon(&samples, 0.1).unwrap(), 0.0);
    }

    #[test]
    fn forecast_horizon_rejects_non_monotonic_time() {
        let samples = [
            ForecastSample {
                horizon: 2.0,
                normalized_error: 0.01,
            },
            ForecastSample {
                horizon: 1.0,
                normalized_error: 0.02,
            },
        ];
        assert_eq!(
            forecast_horizon(&samples, 0.1),
            Err(MetricError::NonIncreasingForecastTime)
        );
    }

    #[test]
    fn horizon_normalization_uses_only_the_reference_time_unit() {
        let reference = QualifiedLyapunovReference {
            lambda_max: 0.25,
            qualification: EstimatorQualification::Reliable,
        };
        assert!((horizon_in_lyapunov_times(8.0, reference).unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn bayes_risk_is_distinct_from_clairvoyant_risk_and_policy_regret() {
        // Two hidden states are indistinguishable to this candidate. Action 0
        // is safe in state A but costly in state B; action 1 has the reverse
        // trade-off. The information-set Bayes action is 0. The candidate
        // chooses 1, so regret is measured relative to the same information.
        let information_sets = [InformationSet {
            probability: 1.0,
            states: vec![state(0.5, &[0.0, 1.0]), state(0.5, &[0.25, 0.0])],
            candidate_action: Some(1),
        }];

        let result = evaluate_decisions(&information_sets).unwrap();
        assert!((result.bayes_risk - 0.125).abs() < 1e-12);
        assert!((result.policy_risk - 0.5).abs() < 1e-12);
        assert!((result.perfect_information_risk - 0.0).abs() < 1e-12);
        assert!((result.regret - 0.375).abs() < 1e-12);
    }

    #[test]
    fn a_discriminating_observation_can_make_regret_zero() {
        let information_sets = [
            InformationSet {
                probability: 0.5,
                states: vec![state(1.0, &[0.0, 1.0])],
                candidate_action: Some(0),
            },
            InformationSet {
                probability: 0.5,
                states: vec![state(1.0, &[0.25, 0.0])],
                candidate_action: Some(1),
            },
        ];

        let result = evaluate_decisions(&information_sets).unwrap();
        assert!((result.regret - 0.0).abs() < 1e-12);
        assert!((result.bayes_risk - 0.0).abs() < 1e-12);
    }

    #[test]
    fn bayes_risk_can_be_evaluated_without_a_candidate_action() {
        let information_sets = [InformationSet {
            probability: 1.0,
            states: vec![state(1.0, &[0.0, 1.0])],
            candidate_action: None,
        }];

        let result = evaluate_bayes_risk(&information_sets).unwrap();
        assert_eq!(result.bayes_risk, 0.0);
        assert_eq!(result.perfect_information_risk, 0.0);
    }

    #[test]
    fn regret_requires_a_candidate_action() {
        let information_sets = [InformationSet {
            probability: 1.0,
            states: vec![state(1.0, &[0.0, 1.0])],
            candidate_action: None,
        }];
        assert_eq!(
            evaluate_decisions(&information_sets),
            Err(MetricError::CandidateActionRequired)
        );
    }

    #[test]
    fn information_value_uses_expected_posterior_risk_and_cost() {
        let value = expected_value_of_information(1.0, 0.25, 0.25).unwrap();
        assert!((value.gross_value - 0.75).abs() < 1e-12);
        assert!((value.net_value - 0.5).abs() < 1e-12);
    }

    #[test]
    fn zero_value_observation_is_not_made_worthwhile_by_entropy_alone() {
        // If an observation leaves the optimal task risk unchanged, it has
        // zero decision value regardless of any separate information score.
        let value = expected_value_of_information(0.25, 0.25, 0.01).unwrap();
        assert_eq!(value.gross_value, 0.0);
        assert!(value.net_value < 0.0);
    }

    #[test]
    fn expected_information_cannot_increase_bayes_risk() {
        assert_eq!(
            expected_value_of_information(0.25, 0.3, 0.0),
            Err(MetricError::InformationIncreasedExpectedRisk)
        );
    }

    #[test]
    fn resource_vector_requires_componentwise_improvement() {
        let lower = ResourceVector {
            sensor_samples: 10,
            sensor_wakeups: 2,
            compute_time_us: 100,
            measured_energy_uj: 50,
            ..ResourceVector::default()
        };
        let higher = ResourceVector {
            sensor_samples: 12,
            sensor_wakeups: 3,
            compute_time_us: 110,
            measured_energy_uj: 60,
            ..ResourceVector::default()
        };
        assert!(pareto_dominates(lower, higher));
        assert!(!pareto_dominates(higher, lower));
        assert!(!pareto_dominates(lower, lower));
    }

    #[test]
    fn resource_vectors_with_tradeoffs_are_not_ordered() {
        let fewer_samples = ResourceVector {
            sensor_samples: 5,
            compute_time_us: 200,
            ..ResourceVector::default()
        };
        let more_samples_less_compute = ResourceVector {
            sensor_samples: 10,
            compute_time_us: 100,
            ..ResourceVector::default()
        };
        assert!(!pareto_dominates(fewer_samples, more_samples_less_compute));
        assert!(!pareto_dominates(more_samples_less_compute, fewer_samples));
    }
    #[test]
    fn forecast_validates_samples_after_the_first_threshold_crossing() {
        let samples = [
            ForecastSample {
                horizon: 1.0,
                normalized_error: 0.01,
            },
            ForecastSample {
                horizon: 2.0,
                normalized_error: 0.2,
            },
            ForecastSample {
                horizon: 1.5,
                normalized_error: 0.03,
            },
        ];
        assert_eq!(
            forecast_horizon(&samples, 0.1),
            Err(MetricError::NonIncreasingForecastTime)
        );
    }

    #[test]
    fn expected_information_value_can_be_negative_after_cost() {
        let value = expected_value_of_information(1.0, 0.9, 0.2).unwrap();
        assert!((value.gross_value - 0.1).abs() < 1e-12);
        assert!((value.net_value + 0.1).abs() < 1e-12);
    }

    #[test]
    fn decision_evaluator_rejects_malformed_conditional_probability_mass() {
        let information_sets = [InformationSet {
            probability: 1.0,
            states: vec![state(0.4, &[0.0, 1.0]), state(0.4, &[1.0, 0.0])],
            candidate_action: Some(0),
        }];
        assert_eq!(
            evaluate_decisions(&information_sets),
            Err(MetricError::ProbabilityMassNotOne(
                "conditional state probability mass"
            ))
        );
    }

    #[test]
    fn normalized_horizon_rejects_overflow() {
        let reference = QualifiedLyapunovReference {
            lambda_max: f64::MAX,
            qualification: EstimatorQualification::Reliable,
        };
        assert_eq!(
            horizon_in_lyapunov_times(f64::MAX, reference),
            Err(MetricError::NonFiniteValue(
                "Lyapunov-normalized forecast horizon"
            ))
        );
    }
}
