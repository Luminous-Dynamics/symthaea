//! Decision-Relevant Predictability Horizon (DRPH).
//!
//! This module consumes independently generated *simultaneous* upper bounds
//! for policy regret and critical-event risk. It does not estimate confidence
//! bounds or validate their statistical assumptions. The evaluator must
//! qualify those bounds over the complete horizon/family/comparison scope
//! before setting `BoundQualification::Qualified`.

/// Independent qualification of the statistical bound supplied at one horizon.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoundQualification {
    /// The evaluator has independently qualified the bound and its scope.
    Qualified,
    /// The value is diagnostic only; no horizon claim may rely on it.
    DiagnosticOnly,
    /// Available evidence is insufficient to qualify the bound.
    Inconclusive,
}

/// Simultaneous upper bounds at one forecast horizon.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionHorizonSample {
    /// Positive horizon in the system's declared physical-time unit.
    pub horizon: f64,
    /// Simultaneous upper bound on avoidable policy regret, in task-loss units.
    pub regret_upper_bound: f64,
    /// Simultaneous upper bound on critical-event probability, in [0, 1].
    pub critical_risk_upper_bound: f64,
    /// Family-wise confidence level for the simultaneous bounds, in (0, 1).
    pub simultaneous_confidence_level: f64,
    pub qualification: BoundQualification,
}

/// Application-declared admissibility limits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionHorizonThresholds {
    /// Maximum acceptable upper bound on policy regret, in task-loss units.
    pub maximum_regret: f64,
    /// Maximum acceptable upper bound on critical-event probability.
    pub maximum_critical_risk: f64,
    /// Minimum family-wise confidence level required for a horizon claim.
    pub minimum_simultaneous_confidence: f64,
}

/// Why the initial sequence of decision-admissible horizons ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionHorizonFailure {
    UnqualifiedBounds,
    InsufficientConfidence,
    RegretBudgetExceeded,
    CriticalRiskBudgetExceeded,
}

/// First horizon at which the decision-admissibility contract failed.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionHorizonStop {
    pub horizon: f64,
    pub reason: DecisionHorizonFailure,
}

/// Conservative, discrete decision-relevant predictability result.
///
/// `horizon` is the last sample in the initial contiguous prefix for which
/// *both* simultaneous upper bounds meet their declared thresholds and the
/// bound is independently qualified. It is not an interpolated crossing or
/// a continuous-time guarantee.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DecisionHorizonReport {
    pub horizon: f64,
    pub qualifying_sample_count: usize,
    /// Minimum confidence level among the accepted prefix; None when no
    /// horizon sample qualified.
    pub minimum_simultaneous_confidence: Option<f64>,
    pub first_failure: Option<DecisionHorizonStop>,
}

/// Input and numerical-contract failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecisionHorizonError {
    EmptySeries,
    NonFiniteValue(&'static str),
    NegativeValue(&'static str),
    ProbabilityOutOfRange(&'static str),
    NonIncreasingHorizon,
    InvalidThreshold(&'static str),
}

impl std::fmt::Display for DecisionHorizonError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptySeries => write!(formatter, "decision-horizon series is empty"),
            Self::NonFiniteValue(name) => write!(formatter, "{name} must be finite"),
            Self::NegativeValue(name) => write!(formatter, "{name} must be non-negative"),
            Self::ProbabilityOutOfRange(name) => {
                write!(formatter, "{name} must be in the required probability range")
            }
            Self::NonIncreasingHorizon => {
                write!(formatter, "decision horizons must be strictly increasing")
            }
            Self::InvalidThreshold(name) => write!(formatter, "{name} is invalid"),
        }
    }
}

impl std::error::Error for DecisionHorizonError {}

fn validate_thresholds(
    thresholds: DecisionHorizonThresholds,
) -> Result<(), DecisionHorizonError> {
    if !thresholds.maximum_regret.is_finite() {
        return Err(DecisionHorizonError::NonFiniteValue("maximum regret"));
    }
    if thresholds.maximum_regret < 0.0 {
        return Err(DecisionHorizonError::NegativeValue("maximum regret"));
    }
    if !thresholds.maximum_critical_risk.is_finite()
        || !(0.0..=1.0).contains(&thresholds.maximum_critical_risk)
    {
        return Err(DecisionHorizonError::InvalidThreshold(
            "maximum critical risk",
        ));
    }
    if !thresholds.minimum_simultaneous_confidence.is_finite()
        || !(0.0..1.0).contains(&thresholds.minimum_simultaneous_confidence)
    {
        return Err(DecisionHorizonError::InvalidThreshold(
            "minimum simultaneous confidence",
        ));
    }
    Ok(())
}

/// Compute the conservative Decision-Relevant Predictability Horizon.
///
/// The supplied regret and critical-risk values must already be valid
/// simultaneous upper confidence bounds over the full preregistered
/// horizon/family/comparison scope. Pointwise intervals, uncorrected repeated
/// looks, or candidate-self-certified bounds must not be marked Qualified.
///
/// All samples are validated, including samples after the first failing
/// horizon. The report then stops at the first failure rather than allowing a
/// later recovery to extend the initial contiguous decision-admissible prefix.
pub fn decision_relevant_predictability_horizon(
    samples: &[DecisionHorizonSample],
    thresholds: DecisionHorizonThresholds,
) -> Result<DecisionHorizonReport, DecisionHorizonError> {
    if samples.is_empty() {
        return Err(DecisionHorizonError::EmptySeries);
    }
    validate_thresholds(thresholds)?;

    let mut previous_horizon = 0.0;
    let mut accepted_horizon = 0.0;
    let mut qualifying_sample_count = 0;
    let mut minimum_confidence: Option<f64> = None;
    let mut first_failure: Option<DecisionHorizonStop> = None;

    for sample in samples {
        if !sample.horizon.is_finite() {
            return Err(DecisionHorizonError::NonFiniteValue("horizon"));
        }
        if sample.horizon <= previous_horizon {
            return Err(DecisionHorizonError::NonIncreasingHorizon);
        }
        if !sample.regret_upper_bound.is_finite() {
            return Err(DecisionHorizonError::NonFiniteValue(
                "regret upper bound",
            ));
        }
        if sample.regret_upper_bound < 0.0 {
            return Err(DecisionHorizonError::NegativeValue(
                "regret upper bound",
            ));
        }
        if !sample.critical_risk_upper_bound.is_finite()
            || !(0.0..=1.0).contains(&sample.critical_risk_upper_bound)
        {
            return Err(DecisionHorizonError::ProbabilityOutOfRange(
                "critical-risk upper bound",
            ));
        }
        if !sample.simultaneous_confidence_level.is_finite()
            || !(0.0..1.0).contains(&sample.simultaneous_confidence_level)
        {
            return Err(DecisionHorizonError::ProbabilityOutOfRange(
                "simultaneous confidence level",
            ));
        }

        if first_failure.is_none() {
            let reason = if sample.qualification != BoundQualification::Qualified {
                Some(DecisionHorizonFailure::UnqualifiedBounds)
            } else if sample.simultaneous_confidence_level
                < thresholds.minimum_simultaneous_confidence
            {
                Some(DecisionHorizonFailure::InsufficientConfidence)
            } else if sample.regret_upper_bound > thresholds.maximum_regret {
                Some(DecisionHorizonFailure::RegretBudgetExceeded)
            } else if sample.critical_risk_upper_bound > thresholds.maximum_critical_risk {
                Some(DecisionHorizonFailure::CriticalRiskBudgetExceeded)
            } else {
                None
            };

            if let Some(reason) = reason {
                first_failure = Some(DecisionHorizonStop {
                    horizon: sample.horizon,
                    reason,
                });
            } else {
                accepted_horizon = sample.horizon;
                qualifying_sample_count += 1;
                minimum_confidence = Some(
                    minimum_confidence
                        .map_or(sample.simultaneous_confidence_level, |current| {
                            current.min(sample.simultaneous_confidence_level)
                        }),
                );
            }
        }

        previous_horizon = sample.horizon;
    }

    Ok(DecisionHorizonReport {
        horizon: accepted_horizon,
        qualifying_sample_count,
        minimum_simultaneous_confidence: minimum_confidence,
        first_failure,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const LIMITS: DecisionHorizonThresholds = DecisionHorizonThresholds {
        maximum_regret: 0.1,
        maximum_critical_risk: 0.01,
        minimum_simultaneous_confidence: 0.9,
    };

    fn sample(
        horizon: f64,
        regret: f64,
        critical_risk: f64,
        confidence: f64,
        qualification: BoundQualification,
    ) -> DecisionHorizonSample {
        DecisionHorizonSample {
            horizon,
            regret_upper_bound: regret,
            critical_risk_upper_bound: critical_risk,
            simultaneous_confidence_level: confidence,
            qualification,
        }
    }

    #[test]
    fn horizon_uses_both_regret_and_critical_risk_bounds() {
        let samples = [
            sample(1.0, 0.02, 0.001, 0.95, BoundQualification::Qualified),
            sample(2.0, 0.08, 0.009, 0.95, BoundQualification::Qualified),
            sample(3.0, 0.11, 0.002, 0.95, BoundQualification::Qualified),
            sample(4.0, 0.01, 0.001, 0.95, BoundQualification::Qualified),
        ];

        let result = decision_relevant_predictability_horizon(&samples, LIMITS).unwrap();
        assert_eq!(result.horizon, 2.0);
        assert_eq!(result.qualifying_sample_count, 2);
        assert_eq!(
            result.first_failure,
            Some(DecisionHorizonStop {
                horizon: 3.0,
                reason: DecisionHorizonFailure::RegretBudgetExceeded,
            })
        );
    }

    #[test]
    fn critical_risk_can_end_horizon_even_when_regret_is_acceptable() {
        let samples = [
            sample(1.0, 0.01, 0.001, 0.95, BoundQualification::Qualified),
            sample(2.0, 0.02, 0.02, 0.95, BoundQualification::Qualified),
        ];

        let result = decision_relevant_predictability_horizon(&samples, LIMITS).unwrap();
        assert_eq!(result.horizon, 1.0);
        assert_eq!(
            result.first_failure.unwrap().reason,
            DecisionHorizonFailure::CriticalRiskBudgetExceeded
        );
    }

    #[test]
    fn unqualified_or_underconfident_bounds_do_not_extend_horizon() {
        let unqualified = [
            sample(1.0, 0.01, 0.001, 0.95, BoundQualification::Qualified),
            sample(2.0, 0.01, 0.001, 0.95, BoundQualification::DiagnosticOnly),
        ];
        let result = decision_relevant_predictability_horizon(&unqualified, LIMITS).unwrap();
        assert_eq!(result.horizon, 1.0);
        assert_eq!(
            result.first_failure.unwrap().reason,
            DecisionHorizonFailure::UnqualifiedBounds
        );

        let underconfident = [sample(
            1.0,
            0.01,
            0.001,
            0.85,
            BoundQualification::Qualified,
        )];
        let result = decision_relevant_predictability_horizon(&underconfident, LIMITS).unwrap();
        assert_eq!(result.horizon, 0.0);
        assert_eq!(result.minimum_simultaneous_confidence, None);
        assert_eq!(
            result.first_failure.unwrap().reason,
            DecisionHorizonFailure::InsufficientConfidence
        );
    }

    #[test]
    fn validates_the_entire_series_after_first_failure() {
        let samples = [
            sample(1.0, 0.01, 0.001, 0.95, BoundQualification::Qualified),
            sample(2.0, 0.2, 0.001, 0.95, BoundQualification::Qualified),
            sample(1.5, 0.01, 0.001, 0.95, BoundQualification::Qualified),
        ];
        assert_eq!(
            decision_relevant_predictability_horizon(&samples, LIMITS),
            Err(DecisionHorizonError::NonIncreasingHorizon)
        );
    }

    #[test]
    fn rejects_invalid_probability_and_loss_bounds() {
        assert_eq!(
            decision_relevant_predictability_horizon(
                &[sample(1.0, -0.1, 0.001, 0.95, BoundQualification::Qualified)],
                LIMITS
            ),
            Err(DecisionHorizonError::NegativeValue("regret upper bound"))
        );
        assert_eq!(
            decision_relevant_predictability_horizon(
                &[sample(1.0, 0.01, 1.1, 0.95, BoundQualification::Qualified)],
                LIMITS
            ),
            Err(DecisionHorizonError::ProbabilityOutOfRange(
                "critical-risk upper bound"
            ))
        );
    }

    #[test]
    fn rejects_invalid_threshold_contracts() {
        let invalid = DecisionHorizonThresholds {
            maximum_regret: 0.1,
            maximum_critical_risk: 0.01,
            minimum_simultaneous_confidence: 1.0,
        };
        assert_eq!(
            decision_relevant_predictability_horizon(
                &[sample(1.0, 0.01, 0.001, 0.95, BoundQualification::Qualified)],
                invalid
            ),
            Err(DecisionHorizonError::InvalidThreshold(
                "minimum simultaneous confidence"
            ))
        );
    }

    #[test]
    fn reports_no_horizon_when_first_sample_is_not_admissible() {
        let samples = [sample(
            1.0,
            0.2,
            0.02,
            0.95,
            BoundQualification::Qualified,
        )];
        let result = decision_relevant_predictability_horizon(&samples, LIMITS).unwrap();
        assert_eq!(result.horizon, 0.0);
        assert_eq!(result.qualifying_sample_count, 0);
        assert_eq!(
            result.first_failure.unwrap().reason,
            DecisionHorizonFailure::RegretBudgetExceeded
        );
    }

    #[test]
    fn preserves_the_confidence_floor_of_the_accepted_prefix() {
        let samples = [
            sample(1.0, 0.01, 0.001, 0.97, BoundQualification::Qualified),
            sample(2.0, 0.01, 0.001, 0.92, BoundQualification::Qualified),
        ];
        let result = decision_relevant_predictability_horizon(&samples, LIMITS).unwrap();
        assert_eq!(result.horizon, 2.0);
        assert_eq!(result.minimum_simultaneous_confidence, Some(0.92));
    }

    #[test]
    fn requires_nonempty_samples() {
        assert_eq!(
            decision_relevant_predictability_horizon(&[], LIMITS),
            Err(DecisionHorizonError::EmptySeries)
        );
    }
}
