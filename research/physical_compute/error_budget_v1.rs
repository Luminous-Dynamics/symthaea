//! Compositional error budget for analog / mixed-signal computation.
//!
//! This module turns cross-layer non-idealities into explicit output-space
//! contributions. It never assumes error sources are independent by default.
//! It propagates declared local magnitudes through caller-supplied sensitivity
//! gains; it does not infer physical distributions or task sensitivities.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    WeightPerturbation,
    MultiplicativeDistortion,
    NonlinearTransferResidual,
    AdditiveOffset,
    InputConversion,
    OutputQuantization,
    Accumulation,
    ReadNoise,
    TimingJitter,
    SpatialFault,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MagnitudeKind {
    DeterministicUpperBound,
    StandardDeviation,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorContribution {
    pub kind: ErrorKind,
    /// Local magnitude in the declared layer's norm/units.
    pub magnitude: f64,
    /// Bound on how this local error is amplified to the operator output.
    pub propagation_gain: f64,
    pub magnitude_kind: MagnitudeKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DependenceAssumption {
    /// Default: assume errors may be correlated; use the conservative L1
    /// standard-deviation bound rather than root-sum-square.
    UnknownOrCorrelated,
    /// The caller DECLARES pairwise/collective independence based on external
    /// evidence. This function checks digest syntax only; it cannot verify the
    /// evidence or establish physical independence.
    IndependentDeclared { evidence_sha256: String },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrorBudgetSummary {
    pub deterministic_output_bound: f64,
    pub stochastic_sigma_l1_upper: f64,
    pub stochastic_sigma_rss_under_independence_assumption: Option<f64>,
    pub task_deterministic_bound: f64,
    pub task_sigma_l1_upper: f64,
    pub task_sigma_rss_under_independence_assumption: Option<f64>,
    pub independence_assumption_declared: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BudgetError {
    InvalidSensitivity,
    InvalidMagnitude,
    InvalidPropagationGain,
    InvalidIndependenceDigest,
    NumericOverflow,
}

/// Compute a compositional error budget.
///
/// Deterministic norm bounds always add linearly. Standard-deviation terms
/// also add linearly unless explicit independence evidence is supplied; only
/// then an RSS value is exposed under that declared assumption. RSS is a variance-combination estimate, not
/// an upper bound without the corresponding distributional assumptions.
///
/// The task_sensitivity_bound must be independently justified for the
/// workload contract (for example, by a condition-aware perturbation theorem
/// or a validated local Jacobian bound). This function does not substitute a
/// condition number for that proof automatically.
pub fn compose_error_budget(
    contributions: &[ErrorContribution],
    task_sensitivity_bound: f64,
    dependence: &DependenceAssumption,
) -> Result<ErrorBudgetSummary, BudgetError> {
    if !task_sensitivity_bound.is_finite() || task_sensitivity_bound < 0.0 {
        return Err(BudgetError::InvalidSensitivity);
    }

    let independent = match dependence {
        DependenceAssumption::UnknownOrCorrelated => false,
        DependenceAssumption::IndependentDeclared { evidence_sha256 } => {
            let valid = evidence_sha256.len() == 71
                && evidence_sha256.starts_with("sha256:")
                && evidence_sha256[7..]
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte));
            if !valid {
                return Err(BudgetError::InvalidIndependenceDigest);
            }
            true
        }
    };

    let mut deterministic = 0.0_f64;
    let mut stochastic_l1 = 0.0_f64;
    let mut stochastic_squares = 0.0_f64;

    for contribution in contributions {
        if !contribution.magnitude.is_finite() || contribution.magnitude < 0.0 {
            return Err(BudgetError::InvalidMagnitude);
        }
        if !contribution.propagation_gain.is_finite()
            || contribution.propagation_gain < 0.0
        {
            return Err(BudgetError::InvalidPropagationGain);
        }
        let output_magnitude = contribution.magnitude * contribution.propagation_gain;
        if !output_magnitude.is_finite() {
            return Err(BudgetError::NumericOverflow);
        }
        match contribution.magnitude_kind {
            MagnitudeKind::DeterministicUpperBound => {
                deterministic += output_magnitude;
            }
            MagnitudeKind::StandardDeviation => {
                stochastic_l1 += output_magnitude;
                stochastic_squares += output_magnitude * output_magnitude;
            }
        }
        if !deterministic.is_finite()
            || !stochastic_l1.is_finite()
            || !stochastic_squares.is_finite()
        {
            return Err(BudgetError::NumericOverflow);
        }
    }

    let stochastic_rss = if independent {
        Some(stochastic_squares.sqrt())
    } else {
        None
    };
    let task_deterministic = deterministic * task_sensitivity_bound;
    let task_sigma_l1 = stochastic_l1 * task_sensitivity_bound;
    let task_sigma_rss = stochastic_rss.map(|sigma| sigma * task_sensitivity_bound);
    if !task_deterministic.is_finite()
        || !task_sigma_l1.is_finite()
        || task_sigma_rss.is_some_and(|value| !value.is_finite())
    {
        return Err(BudgetError::NumericOverflow);
    }

    Ok(ErrorBudgetSummary {
        deterministic_output_bound: deterministic,
        stochastic_sigma_l1_upper: stochastic_l1,
        stochastic_sigma_rss_under_independence_assumption: stochastic_rss,
        task_deterministic_bound: task_deterministic,
        task_sigma_l1_upper: task_sigma_l1,
        task_sigma_rss_under_independence_assumption: task_sigma_rss,
        independence_assumption_declared: independent,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_error_bounds_add_linearly_and_propagate_to_task() {
        let contributions = [
            ErrorContribution {
                kind: ErrorKind::WeightPerturbation,
                magnitude: 0.1,
                propagation_gain: 2.0,
                magnitude_kind: MagnitudeKind::DeterministicUpperBound,
            },
            ErrorContribution {
                kind: ErrorKind::OutputQuantization,
                magnitude: 0.2,
                propagation_gain: 1.0,
                magnitude_kind: MagnitudeKind::DeterministicUpperBound,
            },
        ];
        let summary = compose_error_budget(
            &contributions,
            3.0,
            &DependenceAssumption::UnknownOrCorrelated,
        ).unwrap();
        assert!((summary.deterministic_output_bound - 0.4).abs() < 1.0e-12);
        assert!((summary.task_deterministic_bound - 1.2).abs() < 1.0e-12);
        assert!(!summary.independence_assumption_declared);
    }

    #[test]
    fn unknown_correlation_uses_l1_sigma_bound_not_rss() {
        let contributions = [
            ErrorContribution {
                kind: ErrorKind::ReadNoise,
                magnitude: 0.3,
                propagation_gain: 1.0,
                magnitude_kind: MagnitudeKind::StandardDeviation,
            },
            ErrorContribution {
                kind: ErrorKind::Accumulation,
                magnitude: 0.4,
                propagation_gain: 1.0,
                magnitude_kind: MagnitudeKind::StandardDeviation,
            },
        ];
        let summary = compose_error_budget(
            &contributions,
            1.0,
            &DependenceAssumption::UnknownOrCorrelated,
        ).unwrap();
        assert!((summary.stochastic_sigma_l1_upper - 0.7).abs() < 1.0e-12);
        assert!((summary.stochastic_sigma_rss_under_independence_assumption.is_none()).abs() < 1.0e-12);
        assert!(!summary.independence_assumption_declared);
    }

    #[test]
    fn rss_is_exposed_only_with_valid_independence_evidence() {
        let contributions = [
            ErrorContribution {
                kind: ErrorKind::ReadNoise,
                magnitude: 0.3,
                propagation_gain: 1.0,
                magnitude_kind: MagnitudeKind::StandardDeviation,
            },
            ErrorContribution {
                kind: ErrorKind::Accumulation,
                magnitude: 0.4,
                propagation_gain: 1.0,
                magnitude_kind: MagnitudeKind::StandardDeviation,
            },
        ];
        let evidence = DependenceAssumption::IndependentDeclared {
            evidence_sha256: format!("sha256:{}", "a".repeat(64)),
        };
        let summary = compose_error_budget(&contributions, 2.0, &evidence).unwrap();
        assert!((summary.stochastic_sigma_rss_under_independence_assumption.unwrap() - 0.5).abs() < 1.0e-12);
        assert!((summary.task_sigma_rss_under_independence_assumption.unwrap() - 1.0).abs() < 1.0e-12);
        assert!(summary.independence_assumption_declared);
        assert!((summary.stochastic_sigma_l1_upper - 0.7).abs() < 1.0e-12);
    }

    #[test]
    fn invalid_inputs_and_missing_independence_evidence_fail_closed() {
        let negative = [ErrorContribution {
            kind: ErrorKind::AdditiveOffset,
            magnitude: -1.0,
            propagation_gain: 1.0,
            magnitude_kind: MagnitudeKind::DeterministicUpperBound,
        }];
        assert_eq!(
            compose_error_budget(&negative, 1.0, &DependenceEvidence::UnknownOrCorrelated),
            Err(BudgetError::InvalidMagnitude)
        );

        let invalid_digest = DependenceAssumption::IndependentDeclared {
            evidence_sha256: "sha256:not-a-digest".into(),
        };
        assert_eq!(
            compose_error_budget(&[], 1.0, &invalid_digest),
            Err(BudgetError::InvalidIndependenceDigest)
        );
        assert_eq!(
            compose_error_budget(&[], f64::NAN, &DependenceEvidence::UnknownOrCorrelated),
            Err(BudgetError::InvalidSensitivity)
        );
    }

    #[test]
    fn no_contributions_produces_zero_budget() {
        let summary = compose_error_budget(
            &[],
            9.0,
            &DependenceEvidence::UnknownOrCorrelated,
        ).unwrap();
        assert_eq!(summary.deterministic_output_bound, 0.0);
        assert_eq!(summary.task_deterministic_bound, 0.0);
        assert_eq!(summary.stochastic_sigma_l1_upper, 0.0);
        assert!(summary.stochastic_sigma_rss_under_independence_assumption.is_none());
    }
}
