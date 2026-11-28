//! Candidate-side interaction contract and decision validation.
//!
//! A candidate receives only an observation, the action vocabulary, and
//! remaining budget. It is never passed a game, task specification, hidden
//! state, random seed, reward oracle, or exact evaluator planner.

use std::collections::HashSet;
use std::error::Error;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ForecastValue {
    Boolean(bool),
    Integer(i64),
    Category(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct ForecastMass {
    pub value: ForecastValue,
    pub probability: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FeatureForecast {
    pub feature_id: u16,
    pub outcomes: Vec<ForecastMass>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CandidateDecision<A> {
    pub action: A,
    /// Candidate-reported probability that its selected action achieves the
    /// task-relevant postcondition. None means the policy has no calibrated
    /// estimate; action-selection softmax is not a substitute for confidence.
    pub confidence: Option<f64>,
    /// Candidate forecasts for named, evaluator-defined observation features.
    pub forecasts: Vec<FeatureForecast>,
}

#[derive(Debug, Clone, Copy)]
pub struct CandidateInput<'a, O, A> {
    pub observation: &'a O,
    /// All action attempts permitted by the task contract. A blocked attempt
    /// may still be legal and consume budget.
    pub legal_actions: &'a [A],
    pub remaining_actions: u32,
    pub episode_step: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct CandidateFeedback<'a, O, A> {
    /// Previous candidate-visible observation.
    pub previous_observation: &'a O,
    pub action: &'a A,
    /// Next candidate-visible observation, never evaluator ground truth.
    pub next_observation: &'a O,
}

/// A stateful candidate policy. Internal memory belongs to the candidate.
/// The evaluator retains ownership of the environment and transition oracle.
pub trait InteractiveCandidate<O, A> {
    type Error: Error;

    /// Clear candidate-side state between episodes. No task seed or evaluator ID is passed.
    fn reset(&mut self) -> Result<(), Self::Error>;

    fn decide(
        &mut self,
        input: CandidateInput<'_, O, A>,
    ) -> Result<CandidateDecision<A>, Self::Error>;

    fn observe_transition(
        &mut self,
        feedback: CandidateFeedback<'_, O, A>,
    ) -> Result<(), Self::Error>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecisionValidationError {
    ActionNotOffered,
    InvalidConfidence,
    DuplicateFeatureId(u16),
    EmptyForecast(u16),
    DuplicateForecastValue(u16),
    InvalidProbability(u16),
    ProbabilityMassMismatch(u16),
}

impl Display for DecisionValidationError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ActionNotOffered => write!(f, "candidate selected an action not offered to it"),
            Self::InvalidConfidence => write!(f, "candidate confidence must be finite and in [0,1]"),
            Self::DuplicateFeatureId(id) => write!(f, "duplicate forecast feature ID {id}"),
            Self::EmptyForecast(id) => write!(f, "forecast feature {id} has no outcomes"),
            Self::DuplicateForecastValue(id) => {
                write!(f, "forecast feature {id} repeats an outcome value")
            }
            Self::InvalidProbability(id) => {
                write!(f, "forecast feature {id} contains an invalid probability")
            }
            Self::ProbabilityMassMismatch(id) => {
                write!(f, "forecast feature {id} probabilities do not sum to one")
            }
        }
    }
}

impl Error for DecisionValidationError {}

/// Validate the candidate output before the evaluator executes its action.
/// Invalid candidate outputs remain in the denominator and must not be repaired
/// silently by the harness.
pub fn validate_decision<A: PartialEq>(
    decision: &CandidateDecision<A>,
    legal_actions: &[A],
) -> Result<(), DecisionValidationError> {
    if !legal_actions.iter().any(|action| action == &decision.action) {
        return Err(DecisionValidationError::ActionNotOffered);
    }
    if decision
        .confidence
        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
    {
        return Err(DecisionValidationError::InvalidConfidence);
    }

    let mut seen_features = HashSet::new();
    for forecast in &decision.forecasts {
        if !seen_features.insert(forecast.feature_id) {
            return Err(DecisionValidationError::DuplicateFeatureId(
                forecast.feature_id,
            ));
        }
        if forecast.outcomes.is_empty() {
            return Err(DecisionValidationError::EmptyForecast(forecast.feature_id));
        }

        let mut seen_values = HashSet::new();
        let mut probability_sum = 0.0;
        for mass in &forecast.outcomes {
            if !seen_values.insert(&mass.value) {
                return Err(DecisionValidationError::DuplicateForecastValue(
                    forecast.feature_id,
                ));
            }
            if !mass.probability.is_finite()
                || !(0.0..=1.0).contains(&mass.probability)
            {
                return Err(DecisionValidationError::InvalidProbability(
                    forecast.feature_id,
                ));
            }
            probability_sum += mass.probability;
        }
        if (probability_sum - 1.0).abs() > 1e-6 {
            return Err(DecisionValidationError::ProbabilityMassMismatch(
                forecast.feature_id,
            ));
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_decision() -> CandidateDecision<u8> {
        CandidateDecision {
            action: 2,
            confidence: Some(0.75),
            forecasts: vec![
                FeatureForecast {
                    feature_id: 1,
                    outcomes: vec![
                        ForecastMass {
                            value: ForecastValue::Integer(4),
                            probability: 0.7,
                        },
                        ForecastMass {
                            value: ForecastValue::Integer(5),
                            probability: 0.3,
                        },
                    ],
                },
                FeatureForecast {
                    feature_id: 2,
                    outcomes: vec![ForecastMass {
                        value: ForecastValue::Boolean(true),
                        probability: 1.0,
                    }],
                },
            ],
        }
    }

    #[test]
    fn accepts_well_formed_candidate_decision() {
        assert_eq!(validate_decision(&valid_decision(), &[0, 1, 2, 3]), Ok(()));
    }

    #[test]
    fn rejects_action_not_in_offered_set() {
        let mut decision = valid_decision();
        decision.action = 4;
        assert_eq!(
            validate_decision(&decision, &[0, 1, 2, 3]),
            Err(DecisionValidationError::ActionNotOffered)
        );
    }

    #[test]
    fn rejects_non_finite_or_out_of_range_confidence() {
        for confidence in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
            let mut decision = valid_decision();
            decision.confidence = Some(confidence);
            assert_eq!(
                validate_decision(&decision, &[0, 1, 2, 3]),
                Err(DecisionValidationError::InvalidConfidence)
            );
        }
    }

    #[test]
    fn accepts_unavailable_confidence_without_fabricating_calibration() {
        let mut decision = valid_decision();
        decision.confidence = None;
        assert_eq!(validate_decision(&decision, &[0, 1, 2, 3]), Ok(()));
    }

    #[test]
    fn rejects_duplicate_features_and_values() {
        let mut duplicate_feature = valid_decision();
        duplicate_feature.forecasts.push(duplicate_feature.forecasts[0].clone());
        assert_eq!(
            validate_decision(&duplicate_feature, &[0, 1, 2, 3]),
            Err(DecisionValidationError::DuplicateFeatureId(1))
        );

        let mut duplicate_value = valid_decision();
        duplicate_value.forecasts[0].outcomes[1].value =
            duplicate_value.forecasts[0].outcomes[0].value.clone();
        assert_eq!(
            validate_decision(&duplicate_value, &[0, 1, 2, 3]),
            Err(DecisionValidationError::DuplicateForecastValue(1))
        );
    }

    #[test]
    fn rejects_invalid_probability_and_bad_total_mass() {
        let mut invalid = valid_decision();
        invalid.forecasts[0].outcomes[0].probability = f64::NAN;
        assert_eq!(
            validate_decision(&invalid, &[0, 1, 2, 3]),
            Err(DecisionValidationError::InvalidProbability(1))
        );

        let mut bad_mass = valid_decision();
        bad_mass.forecasts[0].outcomes[0].probability = 0.6;
        assert_eq!(
            validate_decision(&bad_mass, &[0, 1, 2, 3]),
            Err(DecisionValidationError::ProbabilityMassMismatch(1))
        );
    }

    #[test]
    fn rejects_empty_forecast_but_allows_no_forecast() {
        let mut empty = valid_decision();
        empty.forecasts[0].outcomes.clear();
        assert_eq!(
            validate_decision(&empty, &[0, 1, 2, 3]),
            Err(DecisionValidationError::EmptyForecast(1))
        );

        let mut none = valid_decision();
        none.forecasts.clear();
        assert_eq!(validate_decision(&none, &[0, 1, 2, 3]), Ok(()));
    }
}
