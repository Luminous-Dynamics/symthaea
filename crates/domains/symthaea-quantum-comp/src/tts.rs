//! Statistically explicit time-to-solution (TTS) estimates.
//!
//! This module is unit-agnostic: the caller may supply normalized round trips,
//! simulated time steps, seconds, or measured energy as the per-run cost, but the
//! unit must be declared in the experiment manifest. It reports literature-style
//! expected TTS and whole-run operational TTS, with uncertainty-aware estimates,
//! for each fixed problem instance separately.

use core::fmt;

/// Summary of repeated Bernoulli solver trials for one fixed problem instance.
#[derive(Debug, Clone, PartialEq)]
pub struct TtsEstimate {
    /// Number of successful independent attempts on this instance.
    pub successes: usize,
    /// Total independent attempts on this instance; must be greater than zero.
    pub trials: usize,
    /// Empirical single-run success probability.
    pub empirical_success_probability: f64,
    /// Lower Wilson confidence bound for single-run success probability.
    pub success_probability_lower: f64,
    /// Upper Wilson confidence bound for single-run success probability.
    pub success_probability_upper: f64,
    /// Desired aggregate probability of at least one success.
    pub target_success_probability: f64,
    /// Cost of one attempt, in the caller-declared unit.
    pub single_run_cost: f64,
    /// Continuous expected repetitions using the literature TTS equation.
    ///
    /// None means no empirical successes were observed, so finite TTS is not
    /// supported by the point estimate.
    pub empirical_expected_repetitions: Option<f64>,
    /// Literature-style empirical expected TTS in the declared cost unit.
    pub empirical_expected_tts: Option<f64>,
    /// Minimum whole attempts needed to reach the target at the empirical probability.
    pub empirical_whole_repetitions: Option<u64>,
    /// Operational empirical TTS using whole attempts.
    pub empirical_whole_tts: Option<f64>,
    /// Continuous expected repetitions using the lower Wilson confidence bound.
    pub conservative_expected_repetitions: Option<f64>,
    /// Literature-style conservative expected TTS in the declared cost unit.
    pub conservative_expected_tts: Option<f64>,
    /// Minimum whole attempts using the lower Wilson confidence bound.
    pub conservative_whole_repetitions: Option<u64>,
    /// Operational conservative TTS using whole attempts.
    pub conservative_whole_tts: Option<f64>,
}

/// Validation errors for time-to-solution calculations.
#[derive(Debug, Clone, PartialEq)]
pub enum TtsError {
    /// At least one trial is required.
    ZeroTrials,
    /// Success count cannot exceed the trial count.
    SuccessesExceedTrials {
        /// Reported successes.
        successes: usize,
        /// Reported trials.
        trials: usize,
    },
    /// Target aggregate success probability must lie strictly between zero and one.
    InvalidTarget {
        /// Supplied target probability.
        value: f64,
    },
    /// A duration/cost or confidence parameter is invalid.
    InvalidPositiveFinite {
        /// Parameter identifier.
        name: &'static str,
        /// Supplied value.
        value: f64,
    },
    /// The computed whole-run count cannot fit in a u64.
    RepetitionCountOverflow,
    /// A computed cost is non-finite.
    NumericOverflow,
}

impl fmt::Display for TtsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroTrials => write!(f, "TTS estimation requires at least one trial"),
            Self::SuccessesExceedTrials { successes, trials } => {
                write!(f, "{successes} successes exceed {trials} trials")
            }
            Self::InvalidTarget { value } => {
                write!(f, "target success probability must be in (0, 1), got {value}")
            }
            Self::InvalidPositiveFinite { name, value } => {
                write!(f, "{name} must be positive and finite, got {value}")
            }
            Self::RepetitionCountOverflow => {
                write!(f, "required repetition count exceeds the representable range")
            }
            Self::NumericOverflow => write!(f, "non-finite result in TTS calculation"),
        }
    }
}

impl std::error::Error for TtsError {}

impl TtsEstimate {
    /// Estimate time-to-solution for one problem instance.
    ///
    /// For empirical single-run success probability p and aggregate target q,
    /// the published expected-repetition equation is ln(1-q) / ln(1-p). This
    /// implementation reports that continuous quantity for literature comparison
    /// and the ceiling as a separate operational attempt count. Both are also
    /// computed using the lower Wilson confidence bound on p.
    ///
    /// A zero observed success rate yields unavailable TTS, not zero cost. If the
    /// confidence interval includes zero, conservative TTS is unavailable.
    /// A perfect empirical rate is assigned one minimum meaningful attempt rather
    /// than the zero-valued mathematical limit, because a real solver must run.
    ///
    /// z_score sets the two-sided Wilson interval width (1.959963984540054 is
    /// the usual normal critical value for a 95% interval). Per-instance estimates
    /// should be retained before any graph-level aggregation.
    pub fn from_counts(
        successes: usize,
        trials: usize,
        single_run_cost: f64,
        target_success_probability: f64,
        z_score: f64,
    ) -> Result<Self, TtsError> {
        if trials == 0 {
            return Err(TtsError::ZeroTrials);
        }
        if successes > trials {
            return Err(TtsError::SuccessesExceedTrials { successes, trials });
        }
        if !target_success_probability.is_finite()
            || target_success_probability <= 0.0
            || target_success_probability >= 1.0
        {
            return Err(TtsError::InvalidTarget {
                value: target_success_probability,
            });
        }
        for (name, value) in [("single_run_cost", single_run_cost), ("z_score", z_score)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(TtsError::InvalidPositiveFinite { name, value });
            }
        }

        let n = trials as f64;
        let p = successes as f64 / n;
        let (lower, upper) = wilson_interval(p, n, z_score);

        let empirical_expected_repetitions =
            expected_repetitions_for_target(p, target_success_probability)?;
        let empirical_whole_repetitions = whole_repetitions_for_target(p, target_success_probability)?;
        let conservative_expected_repetitions =
            expected_repetitions_for_target(lower, target_success_probability)?;
        let conservative_whole_repetitions =
            whole_repetitions_for_target(lower, target_success_probability)?;

        let empirical_expected_tts =
            total_expected_cost(empirical_expected_repetitions, single_run_cost)?;
        let empirical_whole_tts = total_whole_cost(empirical_whole_repetitions, single_run_cost)?;
        let conservative_expected_tts =
            total_expected_cost(conservative_expected_repetitions, single_run_cost)?;
        let conservative_whole_tts =
            total_whole_cost(conservative_whole_repetitions, single_run_cost)?;

        Ok(Self {
            successes,
            trials,
            empirical_success_probability: p,
            success_probability_lower: lower,
            success_probability_upper: upper,
            target_success_probability,
            single_run_cost,
            empirical_expected_repetitions,
            empirical_expected_tts,
            empirical_whole_repetitions,
            empirical_whole_tts,
            conservative_expected_repetitions,
            conservative_expected_tts,
            conservative_whole_repetitions,
            conservative_whole_tts,
        })
    }
}

/// Continuous expected repetition count used by published TTS equations.
fn expected_repetitions_for_target(p: f64, target: f64) -> Result<Option<f64>, TtsError> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(TtsError::NumericOverflow);
    }
    if p == 0.0 {
        return Ok(None);
    }
    if p == 1.0 {
        return Ok(Some(1.0));
    }

    let repetitions = (-target).ln_1p() / (-p).ln_1p();
    if !repetitions.is_finite() || repetitions <= 0.0 {
        return Err(TtsError::RepetitionCountOverflow);
    }
    Ok(Some(repetitions))
}

/// Minimum whole-run count for the target aggregate success probability.
fn whole_repetitions_for_target(p: f64, target: f64) -> Result<Option<u64>, TtsError> {
    let Some(expected) = expected_repetitions_for_target(p, target)? else {
        return Ok(None);
    };
    let rounded = expected.ceil();
    // >= is intentional: u64::MAX rounds upward when converted to f64.
    if !rounded.is_finite() || rounded >= u64::MAX as f64 {
        return Err(TtsError::RepetitionCountOverflow);
    }
    Ok(Some(rounded.max(1.0) as u64))
}

/// Compute a two-sided Wilson score interval for a Bernoulli probability.
fn wilson_interval(p: f64, n: f64, z: f64) -> (f64, f64) {
    let z_squared = z * z;
    let denominator = 1.0 + z_squared / n;
    let center = (p + z_squared / (2.0 * n)) / denominator;
    let margin =
        z * (p * (1.0 - p) / n + z_squared / (4.0 * n * n)).sqrt() / denominator;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

fn total_expected_cost(
    repetitions: Option<f64>,
    single_run_cost: f64,
) -> Result<Option<f64>, TtsError> {
    let Some(repetitions) = repetitions else {
        return Ok(None);
    };
    let cost = repetitions * single_run_cost;
    if cost.is_finite() {
        Ok(Some(cost))
    } else {
        Err(TtsError::NumericOverflow)
    }
}

fn total_whole_cost(
    repetitions: Option<u64>,
    single_run_cost: f64,
) -> Result<Option<f64>, TtsError> {
    let Some(repetitions) = repetitions else {
        return Ok(None);
    };
    let cost = repetitions as f64 * single_run_cost;
    if cost.is_finite() {
        Ok(Some(cost))
    } else {
        Err(TtsError::NumericOverflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const Z_95: f64 = 1.959_963_984_540_054;

    #[test]
    fn literature_tts_and_operational_whole_run_tts_are_both_reported() {
        let estimate = TtsEstimate::from_counts(50, 100, 2.0, 0.99, Z_95)
            .expect("valid trial summary");
        // ln(0.01)/ln(0.5) ~= 6.643856; operational scheduling needs seven runs.
        assert_eq!(estimate.empirical_whole_repetitions, Some(7));
        assert_eq!(estimate.empirical_whole_tts, Some(14.0));
        assert!(
            (estimate.empirical_expected_repetitions.unwrap() - 6.643_856_189_774_724).abs()
                < 1e-12
        );
        assert!(
            (estimate.empirical_expected_tts.unwrap() - 13.287_712_379_549_448).abs()
                < 1e-12
        );
        assert!(estimate.success_probability_lower < 0.5);
        assert!(estimate.success_probability_upper > 0.5);
        assert!(estimate.conservative_whole_repetitions.unwrap() >= 7);
    }

    #[test]
    fn no_successes_never_becomes_zero_time_to_solution() {
        let estimate = TtsEstimate::from_counts(0, 100, 1.0, 0.99, Z_95)
            .expect("zero successes are valid data");
        assert_eq!(estimate.empirical_success_probability, 0.0);
        assert_eq!(estimate.empirical_expected_repetitions, None);
        assert_eq!(estimate.empirical_whole_repetitions, None);
        assert_eq!(estimate.empirical_expected_tts, None);
        assert_eq!(estimate.empirical_whole_tts, None);
        assert_eq!(estimate.success_probability_lower, 0.0);
        assert_eq!(estimate.conservative_expected_tts, None);
        assert_eq!(estimate.conservative_whole_tts, None);
    }

    #[test]
    fn all_successes_still_reports_statistical_uncertainty() {
        let estimate = TtsEstimate::from_counts(10, 10, 3.0, 0.99, Z_95)
            .expect("all-successes are valid data");
        assert_eq!(estimate.empirical_expected_repetitions, Some(1.0));
        assert_eq!(estimate.empirical_whole_repetitions, Some(1));
        assert_eq!(estimate.empirical_expected_tts, Some(3.0));
        assert_eq!(estimate.empirical_whole_tts, Some(3.0));
        assert!(estimate.success_probability_lower > 0.0);
        assert!(estimate.success_probability_lower < 1.0);
        assert!(estimate.conservative_whole_repetitions.unwrap() > 1);
        assert!(estimate.conservative_whole_tts.unwrap() > 3.0);
    }

    #[test]
    fn a_single_perfect_trial_does_not_claim_certainty() {
        let estimate = TtsEstimate::from_counts(1, 1, 1.0, 0.99, Z_95)
            .expect("one success is valid data");
        assert_eq!(estimate.empirical_expected_repetitions, Some(1.0));
        assert_eq!(estimate.empirical_whole_repetitions, Some(1));
        assert!(estimate.success_probability_lower < 1.0);
        assert!(estimate.conservative_whole_repetitions.unwrap() > 1);
    }

    #[test]
    fn rejects_invalid_counts_and_non_finite_settings() {
        assert!(matches!(
            TtsEstimate::from_counts(0, 0, 1.0, 0.99, Z_95),
            Err(TtsError::ZeroTrials)
        ));
        assert!(matches!(
            TtsEstimate::from_counts(2, 1, 1.0, 0.99, Z_95),
            Err(TtsError::SuccessesExceedTrials { .. })
        ));
        assert!(matches!(
            TtsEstimate::from_counts(1, 2, 1.0, 1.0, Z_95),
            Err(TtsError::InvalidTarget { .. })
        ));
        assert!(matches!(
            TtsEstimate::from_counts(1, 2, 1.0, 0.0, Z_95),
            Err(TtsError::InvalidTarget { .. })
        ));
        assert!(matches!(
            TtsEstimate::from_counts(1, 2, f64::INFINITY, 0.99, Z_95),
            Err(TtsError::InvalidPositiveFinite { .. })
        ));
        assert!(matches!(
            TtsEstimate::from_counts(1, 2, 1.0, 0.99, f64::NAN),
            Err(TtsError::InvalidPositiveFinite { .. })
        ));
    }

    #[test]
    fn tiny_success_probability_keeps_large_cost_visible() {
        let estimate = TtsEstimate::from_counts(1, 10_000, 0.25, 0.99, Z_95)
            .expect("valid low success probability");
        assert!(estimate.empirical_whole_repetitions.unwrap() > 40_000);
        assert!(estimate.empirical_whole_tts.unwrap() > 10_000.0);
        assert!(
            estimate.conservative_whole_tts.unwrap()
                >= estimate.empirical_whole_tts.unwrap()
        );
    }
}
