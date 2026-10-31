//! Synthetic single-instrument profiled Anderson–Rubin (pAR) reference.
//!
//! Scope is intentionally narrow:
//! - one scalar endogenous regressor;
//! - one scalar instrument;
//! - an intercept, no other controls;
//! - fixed-design, conditionally Gaussian, homoskedastic, independent rows;
//! - a sample-normalized RMS bound on the direct outcome effect Z * gamma.
//!
//! For candidate structural effect b, regress R(b) = Y - b X on an
//! intercept and Z. Let gamma_hat(b) be the instrument slope, sigma_hat(b)^2
//! the residual mean square with n - 2 degrees of freedom, and
//! AR_exact(b) = gamma_hat(b)^2 * sum((Z - mean(Z))^2) / sigma_hat(b)^2.
//!
//! The sample-normalized one-instrument pAR closed form is:
//!
//! pAR(b; g) = (sqrt(AR_exact(b)) - sqrt(n) * g / sigma_hat(b))_+^2.
//!
//! Here g bounds the sample RMS of the direct effect:
//! abs(gamma) * sqrt(sum((Z - mean(Z))^2) / n) <= g.
//! A candidate is not rejected when pAR <= the caller-supplied F(1, n-2)
//! critical value. This function evaluates one candidate; it does not invert
//! the test into a continuous confidence set.
//!
//! This is a synthetic mathematical reference. It does not validate the
//! instrument's randomization or exclusion assumptions, estimate g, or prove
//! a biological causal effect. The current pAR theory used here is conditional
//! Gaussian and fixed-design; heteroskedastic and clustered extensions require
//! a separate validity argument and are not implemented.

use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IvObservation {
    pub instrument: f64,
    pub treatment: f64,
    pub outcome: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SingleInstrumentDiagnostics {
    pub observation_count: usize,
    pub instrument_sum_squares: f64,
    /// Slope from treatment on intercept and instrument.
    pub first_stage_slope: f64,
    /// First-stage F statistic. No universal pass/fail threshold is applied.
    pub first_stage_f: f64,
    /// Slope from outcome on intercept and instrument.
    pub reduced_form_slope: f64,
    /// Conventional Wald ratio; absent only when the estimated first stage is
    /// exactly zero. It is diagnostic and may be unreliable under weak IV.
    pub wald_ratio: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProfiledArResult {
    pub candidate_effect: f64,
    pub implied_direct_effect_coefficient: f64,
    pub implied_direct_effect_rms: f64,
    pub allowed_direct_effect_rms: f64,
    pub residual_sigma: f64,
    pub exact_exclusion_ar_f: f64,
    pub profiled_ar_f: f64,
    pub f_critical_value: f64,
    pub accepted: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum IvAuditError {
    TooFewObservations { minimum: usize, actual: usize },
    NonFiniteInput { observation_index: usize, field: &'static str },
    InvalidCandidateEffect { value: f64 },
    InvalidDirectEffectBound { value: f64 },
    InvalidFCriticalValue { value: f64 },
    NoInstrumentVariation,
    DegenerateFirstStageResidual,
    DegenerateOutcomeResidual { candidate_effect: f64 },
    NumericalOverflow { field: &'static str },
}

impl fmt::Display for IvAuditError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewObservations { minimum, actual } => {
                write!(f, "at least {minimum} observations are required; received {actual}")
            }
            Self::NonFiniteInput { observation_index, field } => {
                write!(f, "observation {observation_index} has non-finite {field}")
            }
            Self::InvalidCandidateEffect { value } => {
                write!(f, "candidate effect must be finite; received {value}")
            }
            Self::InvalidDirectEffectBound { value } => {
                write!(f, "direct-effect RMS bound must be finite and non-negative; received {value}")
            }
            Self::InvalidFCriticalValue { value } => {
                write!(f, "F critical value must be finite and positive; received {value}")
            }
            Self::NoInstrumentVariation => write!(f, "instrument has no sample variation"),
            Self::DegenerateFirstStageResidual => {
                write!(f, "first-stage residual variance is zero; F statistic is degenerate")
            }
            Self::DegenerateOutcomeResidual { candidate_effect } => write!(
                f, "outcome residual variance is zero for candidate effect {candidate_effect}"
            ),
            Self::NumericalOverflow { field } => {
                write!(f, "numerical overflow while computing {field}")
            }
        }
    }
}

impl std::error::Error for IvAuditError {}

fn compensated_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    // Kahan summation reduces false nonzero moments for balanced fixed designs.
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    sum
}

fn compensated_dot(left: &[f64], right: &[f64]) -> f64 {
    compensated_sum(left.iter().zip(right).map(|(a, b)| a * b))
}

fn checked_sum_squares(values: &[f64], field: &'static str) -> Result<f64, IvAuditError> {
    let sum = compensated_sum(values.iter().map(|value| value * value));
    if !sum.is_finite() {
        return Err(IvAuditError::NumericalOverflow { field });
    }
    Ok(sum)
}

/// Compute first-stage and reduced-form diagnostics for a fixed sample.
///
/// The conventional Wald ratio is returned for audit comparison only. It is
/// not used as the pAR inference procedure and no universal first-stage-F
/// threshold is treated as a validity rule.
pub fn diagnose_single_instrument(
    observations: &[IvObservation],
) -> Result<SingleInstrumentDiagnostics, IvAuditError> {
    const MINIMUM_OBSERVATIONS: usize = 4;
    if observations.len() < MINIMUM_OBSERVATIONS {
        return Err(IvAuditError::TooFewObservations {
            minimum: MINIMUM_OBSERVATIONS,
            actual: observations.len(),
        });
    }

    for (index, observation) in observations.iter().enumerate() {
        for (field, value) in [
            ("instrument", observation.instrument),
            ("treatment", observation.treatment),
            ("outcome", observation.outcome),
        ] {
            if !value.is_finite() {
                return Err(IvAuditError::NonFiniteInput {
                    observation_index: index,
                    field,
                });
            }
        }
    }

    let n = observations.len();
    let degrees_of_freedom = (n - 2) as f64;
    let instrument_mean =
        compensated_sum(observations.iter().map(|item| item.instrument)) / n as f64;
    let treatment_mean =
        compensated_sum(observations.iter().map(|item| item.treatment)) / n as f64;
    let outcome_mean =
        compensated_sum(observations.iter().map(|item| item.outcome)) / n as f64;

    let centered_instrument: Vec<f64> = observations
        .iter()
        .map(|item| item.instrument - instrument_mean)
        .collect();
    let instrument_sum_squares =
        checked_sum_squares(&centered_instrument, "instrument sum of squares")?;
    if instrument_sum_squares == 0.0 {
        return Err(IvAuditError::NoInstrumentVariation);
    }

    let centered_treatment: Vec<f64> = observations
        .iter()
        .map(|item| item.treatment - treatment_mean)
        .collect();
    let centered_outcome: Vec<f64> = observations
        .iter()
        .map(|item| item.outcome - outcome_mean)
        .collect();

    let first_stage_cross_product = compensated_dot(&centered_instrument, &centered_treatment);
    let reduced_form_cross_product = compensated_dot(&centered_instrument, &centered_outcome);

    let first_stage_slope = first_stage_cross_product / instrument_sum_squares;
    let reduced_form_slope = reduced_form_cross_product / instrument_sum_squares;

    let first_stage_residuals: Vec<f64> = centered_treatment
        .iter()
        .zip(&centered_instrument)
        .map(|(x, z)| x - first_stage_slope * z)
        .collect();
    let first_stage_sse =
        checked_sum_squares(&first_stage_residuals, "first-stage residual sum of squares")?;
    if first_stage_sse == 0.0 {
        return Err(IvAuditError::DegenerateFirstStageResidual);
    }
    let first_stage_sigma_squared = first_stage_sse / degrees_of_freedom;
    let first_stage_f = first_stage_slope * first_stage_slope * instrument_sum_squares
        / first_stage_sigma_squared;

    if !first_stage_slope.is_finite()
        || !reduced_form_slope.is_finite()
        || !first_stage_f.is_finite()
    {
        return Err(IvAuditError::NumericalOverflow {
            field: "first-stage / reduced-form diagnostics",
        });
    }

    let wald_ratio = if first_stage_slope == 0.0 {
        None
    } else {
        let ratio = reduced_form_slope / first_stage_slope;
        if !ratio.is_finite() {
            return Err(IvAuditError::NumericalOverflow {
                field: "Wald ratio",
            });
        }
        Some(ratio)
    };

    Ok(SingleInstrumentDiagnostics {
        observation_count: n,
        instrument_sum_squares,
        first_stage_slope,
        first_stage_f,
        reduced_form_slope,
        wald_ratio,
    })
}

/// Evaluate one candidate effect using the one-instrument, sample-normalized
/// profiled Anderson–Rubin closed form.
///
/// direct_effect_rms_bound is specified in outcome units and bounds
/// abs(gamma) * sqrt(sum((z - mean(z))^2) / n). f_critical_value must be the
/// declared upper-tail critical value for F(1, n-2), chosen before unblinding.
/// This returns the statistic and acceptance decision; it does not compute a
/// p-value or a confidence-set inversion.
pub fn profile_candidate(
    observations: &[IvObservation],
    candidate_effect: f64,
    direct_effect_rms_bound: f64,
    f_critical_value: f64,
) -> Result<(SingleInstrumentDiagnostics, ProfiledArResult), IvAuditError> {
    if !candidate_effect.is_finite() {
        return Err(IvAuditError::InvalidCandidateEffect {
            value: candidate_effect,
        });
    }
    if !direct_effect_rms_bound.is_finite() || direct_effect_rms_bound < 0.0 {
        return Err(IvAuditError::InvalidDirectEffectBound {
            value: direct_effect_rms_bound,
        });
    }
    if !f_critical_value.is_finite() || f_critical_value <= 0.0 {
        return Err(IvAuditError::InvalidFCriticalValue {
            value: f_critical_value,
        });
    }

    let diagnostics = diagnose_single_instrument(observations)?;
    let n = diagnostics.observation_count;
    let degrees_of_freedom = (n - 2) as f64;

    // Residualize Y - beta X on the intercept and instrument. The fitted
    // instrument coefficient is the direct-effect coefficient compatible
    // with this candidate structural effect.
    let residualized_outcomes: Vec<f64> = observations
        .iter()
        .map(|item| item.outcome - candidate_effect * item.treatment)
        .collect();
    if residualized_outcomes.iter().any(|value| !value.is_finite()) {
        return Err(IvAuditError::NumericalOverflow {
            field: "candidate outcome residualization",
        });
    }

    let residualized_mean = compensated_sum(residualized_outcomes.iter().copied()) / n as f64;
    let instrument_mean =
        compensated_sum(observations.iter().map(|item| item.instrument)) / n as f64;
    let centered_instrument: Vec<f64> = observations
        .iter()
        .map(|item| item.instrument - instrument_mean)
        .collect();
    let centered_candidate_residuals: Vec<f64> = residualized_outcomes
        .iter()
        .map(|value| value - residualized_mean)
        .collect();

    let cross_product = compensated_dot(&centered_instrument, &centered_candidate_residuals);
    let implied_direct_effect_coefficient =
        cross_product / diagnostics.instrument_sum_squares;

    let fitted_residuals: Vec<f64> = centered_candidate_residuals
        .iter()
        .zip(&centered_instrument)
        .map(|(residual, z)| residual - implied_direct_effect_coefficient * z)
        .collect();
    let residual_sse =
        checked_sum_squares(&fitted_residuals, "candidate residual sum of squares")?;
    if residual_sse == 0.0 {
        return Err(IvAuditError::DegenerateOutcomeResidual {
            candidate_effect,
        });
    }

    let residual_variance = residual_sse / degrees_of_freedom;
    let residual_sigma = residual_variance.sqrt();
    let exact_exclusion_ar_f = implied_direct_effect_coefficient
        * implied_direct_effect_coefficient
        * diagnostics.instrument_sum_squares
        / residual_variance;

    // Proposition 9's closed form for the sample-normalized RMS support.
    let excess_root_statistic = (exact_exclusion_ar_f.sqrt()
        - (n as f64).sqrt() * direct_effect_rms_bound / residual_sigma)
        .max(0.0);
    let profiled_ar_f = excess_root_statistic * excess_root_statistic;

    let implied_direct_effect_rms = implied_direct_effect_coefficient.abs()
        * (diagnostics.instrument_sum_squares / n as f64).sqrt();

    if !implied_direct_effect_coefficient.is_finite()
        || !residual_sigma.is_finite()
        || !exact_exclusion_ar_f.is_finite()
        || !profiled_ar_f.is_finite()
        || !implied_direct_effect_rms.is_finite()
    {
        return Err(IvAuditError::NumericalOverflow {
            field: "profiled Anderson-Rubin statistic",
        });
    }

    let result = ProfiledArResult {
        candidate_effect,
        implied_direct_effect_coefficient,
        implied_direct_effect_rms,
        allowed_direct_effect_rms: direct_effect_rms_bound,
        residual_sigma,
        exact_exclusion_ar_f,
        profiled_ar_f,
        f_critical_value,
        accepted: profiled_ar_f <= f_critical_value,
    };

    Ok((diagnostics, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    const F95_DF1_DF18: f64 = 4.413_873_419_170_566;

    fn design(first_stage: f64, direct_effect: f64) -> Vec<IvObservation> {
        let first_stage_noise = [
            0.15, -0.15, 0.12, -0.12, 0.18,
            -0.18, 0.10, -0.10, 0.14, -0.14,
        ];
        let outcome_noise = [
            0.10, -0.10, 0.08, -0.08, 0.12,
            -0.12, 0.06, -0.06, 0.09, -0.09,
        ];
        let mut observations = Vec::with_capacity(20);
        for index in 0..10 {
            let magnitude = 0.05 + 0.1 * index as f64;
            for sign in [-1.0, 1.0] {
                let instrument = sign * magnitude;
                let treatment =
                    first_stage * instrument + first_stage_noise[index];
                let outcome =
                    3.0 * treatment + direct_effect * instrument + outcome_noise[index];
                observations.push(IvObservation {
                    instrument,
                    treatment,
                    outcome,
                });
            }
        }
        observations
    }

    fn close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance,
            "actual {actual:.15}, expected {expected:.15}, tolerance {tolerance}"
        );
    }

    #[test]
    fn strong_first_stage_reports_continuous_strength_diagnostic() {
        let observations = design(2.0, 0.6);
        let diagnostics = diagnose_single_instrument(&observations).unwrap();
        close(diagnostics.first_stage_slope, 2.0, 1.0e-12);
        close(diagnostics.reduced_form_slope, 6.6, 1.0e-12);
        close(diagnostics.first_stage_f, 1210.313_447_927_199_4, 1.0e-9);
        close(diagnostics.wald_ratio.unwrap(), 3.3, 1.0e-12);
    }

    #[test]
    fn weak_first_stage_is_reported_without_an_f_ten_pass_gate() {
        let observations = design(0.05, 0.6);
        let diagnostics = diagnose_single_instrument(&observations).unwrap();
        close(diagnostics.first_stage_slope, 0.05, 1.0e-12);
        close(diagnostics.first_stage_f, 0.756_445_904_954_499_9, 1.0e-10);
        assert!(diagnostics.first_stage_f < 1.0);
        let zero_relevance = diagnose_single_instrument(&design(0.0, 0.6)).unwrap();
        close(zero_relevance.first_stage_slope, 0.0, 0.0);
        close(zero_relevance.first_stage_f, 0.0, 0.0);
        assert_eq!(zero_relevance.wald_ratio, None);
    }

    #[test]
    fn exact_exclusion_is_the_zero_radius_par_case() {
        let observations = design(2.0, 0.0);
        let (diagnostics, result) =
            profile_candidate(&observations, 3.0, 0.0, F95_DF1_DF18).unwrap();
        close(result.implied_direct_effect_coefficient, 0.0, 1.0e-12);
        close(result.profiled_ar_f, result.exact_exclusion_ar_f, 1.0e-12);
        assert!(result.accepted);
        assert_eq!(diagnostics.observation_count, 20);
    }

    #[test]
    fn admissible_direct_effect_can_make_candidate_unrejected() {
        let observations = design(2.0, 0.6);
        let (_, result) =
            profile_candidate(&observations, 3.0, 0.35, F95_DF1_DF18).unwrap();
        close(result.implied_direct_effect_coefficient, 0.6, 1.0e-12);
        close(result.implied_direct_effect_rms, 0.345_976_877_840_123_7, 1.0e-12);
        close(result.profiled_ar_f, 0.0, 1.0e-12);
        assert!(result.accepted);
    }

    #[test]
    fn p_ar_retains_sampling_uncertainty_near_direct_effect_bound() {
        let observations = design(2.0, 0.6);
        let (_, result) =
            profile_candidate(&observations, 3.0, 0.31, F95_DF1_DF18).unwrap();
        assert!(result.implied_direct_effect_rms > result.allowed_direct_effect_rms);
        close(result.profiled_ar_f, 2.740_946_271_084_397, 1.0e-9);
        assert!(result.accepted);
    }

    #[test]
    fn overly_tight_direct_effect_support_can_reject_candidate() {
        let observations = design(2.0, 0.6);
        let (_, result) =
            profile_candidate(&observations, 3.0, 0.1, F95_DF1_DF18).unwrap();
        close(result.profiled_ar_f, 128.127_439_973_594_5, 1.0e-8);
        assert!(!result.accepted);
    }

    #[test]
    fn weak_first_stage_can_leave_far_candidate_unrejected() {
        let observations = design(0.05, 0.6);
        let (diagnostics, result) =
            profile_candidate(&observations, 10.0, 0.35, F95_DF1_DF18).unwrap();
        assert!(diagnostics.first_stage_f < 1.0);
        close(result.implied_direct_effect_coefficient, 0.25, 1.0e-12);
        close(result.profiled_ar_f, 0.0, 1.0e-12);
        assert!(result.accepted);
    }

    #[test]
    fn same_far_candidate_is_rejected_with_strong_first_stage() {
        let observations = design(2.0, 0.6);
        let (diagnostics, result) =
            profile_candidate(&observations, 10.0, 0.35, F95_DF1_DF18).unwrap();
        assert!(diagnostics.first_stage_f > 1000.0);
        assert!(result.profiled_ar_f > 1000.0);
        assert!(!result.accepted);
    }

    #[test]
    fn zero_relevance_leaves_all_candidates_compatible_when_bound_covers_direct_effect() {
        let observations = design(0.0, 0.6);
        let (_, at_three) =
            profile_candidate(&observations, 3.0, 0.35, F95_DF1_DF18).unwrap();
        let (_, at_ten) =
            profile_candidate(&observations, 10.0, 0.35, F95_DF1_DF18).unwrap();
        assert!(at_three.accepted);
        assert!(at_ten.accepted);
        close(at_three.implied_direct_effect_coefficient, at_ten.implied_direct_effect_coefficient, 1.0e-12);
    }

    #[test]
    fn wald_ratio_is_biased_by_direct_effect_even_when_first_stage_is_strong() {
        let observations = design(2.0, 0.6);
        let diagnostics = diagnose_single_instrument(&observations).unwrap();
        close(diagnostics.wald_ratio.unwrap(), 3.3, 1.0e-12);
        assert!((diagnostics.wald_ratio.unwrap() - 3.0).abs() > 0.2);
    }

    #[test]
    fn p_ar_statistic_never_exceeds_exact_exclusion_statistic() {
        let observations = design(2.0, 0.6);
        let (_, exact) =
            profile_candidate(&observations, 3.0, 0.0, F95_DF1_DF18).unwrap();
        let (_, bounded) =
            profile_candidate(&observations, 3.0, 0.2, F95_DF1_DF18).unwrap();
        assert!(bounded.profiled_ar_f <= exact.profiled_ar_f);
    }

    #[test]
    fn scaling_instrument_and_keeping_rms_bound_preserves_par_statistic() {
        let observations = design(2.0, 0.6);
        let scaled: Vec<IvObservation> = observations
            .iter()
            .map(|item| IvObservation {
                instrument: 10.0 * item.instrument,
                treatment: item.treatment,
                outcome: item.outcome,
            })
            .collect();
        let (_, original) =
            profile_candidate(&observations, 3.0, 0.35, F95_DF1_DF18).unwrap();
        let (_, transformed) =
            profile_candidate(&scaled, 3.0, 0.35, F95_DF1_DF18).unwrap();
        close(original.profiled_ar_f, transformed.profiled_ar_f, 1.0e-10);
        close(original.implied_direct_effect_rms, transformed.implied_direct_effect_rms, 1.0e-12);
    }

    #[test]
    fn observation_order_does_not_change_the_result() {
        let observations = design(2.0, 0.6);
        let reversed: Vec<IvObservation> = observations.iter().copied().rev().collect();
        let (_, original) =
            profile_candidate(&observations, 3.0, 0.31, F95_DF1_DF18).unwrap();
        let (_, reversed_result) =
            profile_candidate(&reversed, 3.0, 0.31, F95_DF1_DF18).unwrap();
        close(original.profiled_ar_f, reversed_result.profiled_ar_f, 1.0e-12);
    }

    #[test]
    fn zero_instrument_variation_fails_closed() {
        let observations = vec![
            IvObservation { instrument: 1.0, treatment: 1.0, outcome: 1.0 },
            IvObservation { instrument: 1.0, treatment: 2.0, outcome: 2.0 },
            IvObservation { instrument: 1.0, treatment: 3.0, outcome: 3.0 },
            IvObservation { instrument: 1.0, treatment: 4.0, outcome: 4.0 },
        ];
        assert_eq!(
            diagnose_single_instrument(&observations),
            Err(IvAuditError::NoInstrumentVariation)
        );
    }

    #[test]
    fn non_finite_values_fail_closed() {
        let mut observations = design(2.0, 0.6);
        observations[0].outcome = f64::NAN;
        assert!(matches!(
            diagnose_single_instrument(&observations),
            Err(IvAuditError::NonFiniteInput { field: "outcome", .. })
        ));
    }

    #[test]
    fn too_few_observations_fail_closed() {
        let observations = design(2.0, 0.6);
        assert!(matches!(
            diagnose_single_instrument(&observations[..3]),
            Err(IvAuditError::TooFewObservations { .. })
        ));
    }

    #[test]
    fn invalid_bound_and_critical_value_fail_closed() {
        let observations = design(2.0, 0.6);
        assert!(matches!(
            profile_candidate(&observations, 3.0, -0.1, F95_DF1_DF18),
            Err(IvAuditError::InvalidDirectEffectBound { .. })
        ));
        assert!(matches!(
            profile_candidate(&observations, 3.0, 0.1, 0.0),
            Err(IvAuditError::InvalidFCriticalValue { .. })
        ));
    }

    #[test]
    fn zero_candidate_residual_variance_is_not_given_a_finite_statistic() {
        let mut observations = design(2.0, 0.6);
        for observation in &mut observations {
            observation.outcome = observation.instrument;
        }
        assert!(matches!(
            profile_candidate(&observations, 0.0, 0.0, F95_DF1_DF18),
            Err(IvAuditError::DegenerateOutcomeResidual { .. })
        ));
    }

    #[test]
    fn zero_bound_reduces_to_ar_and_large_bound_can_reduce_profiled_statistic_to_zero() {
        let observations = design(2.0, 0.6);
        let (_, exact) =
            profile_candidate(&observations, 3.0, 0.0, F95_DF1_DF18).unwrap();
        let (_, broad) =
            profile_candidate(&observations, 3.0, 10.0, F95_DF1_DF18).unwrap();
        close(exact.profiled_ar_f, exact.exact_exclusion_ar_f, 1.0e-12);
        close(broad.profiled_ar_f, 0.0, 1.0e-12);
        assert!(exact.profiled_ar_f > broad.profiled_ar_f);
    }
}
