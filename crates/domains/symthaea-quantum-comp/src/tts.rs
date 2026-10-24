//! Statistically explicit time-to-solution (TTS) estimates.
//!
//! This module is unit-agnostic: the caller may supply normalized round trips,
//! simulated time steps, seconds, or measured energy as the per-run cost, but the
//! unit must be declared in the experiment manifest. It reports empirical and
//! conservative TTS separately and never treats repeated trials on one instance
//! as independent problem instances.

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
    /// Minimum whole number of attempts at the empirical probability.
    ///
    /// None means no empirical successes were observed, so finite TTS is not
    /// supported by the point estimate.
    pub empirical_repetitions: Option<u64>,
    /// Empirical TTS in the caller-declared cost unit.
    pub empirical_tts: Option<f64>,
    /// Minimum attempts when using the lower Wilson confidence bound.
    ///
    /// None means the confidence interval includes zero success probability.
    pub conservative_repetitions: Option<u64>,
    /// Conservative TTS based on the lower Wilson confidence bound.
    pub conservative_tts: Option<f64>,
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
                write!(f, "required whole-run count exceeds the u64 range")
            }
            Self::NumericOverflow => write!(f, "non-finite result in TTS calculation"),
        }
    }
}

impl std::error::Error for TtsError {}

impl TtsEstimate {
    /// Estimate time-to-solution for one problem instance.
    ///
    /// For empirical single-run success probability p and target q, the required
    /// number of independent attempts is:
    ///
    /// ceil(ln(1-q) / ln(1-p)).
    ///
    /// The empirical TTS is one-run cost times that whole-run count (not the
    /// fractional expectation). The conservative TTS repeats the calculation
    /// with the lower Wilson confidence bound on p. If no successes are observed,
    /// point TTS is unavailable; if the confidence interval includes zero,
    /// conservative TTS is unavailable. Neither case is replaced with zero.
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
            || !(0.0..1.0).contains(&target_success_probability)
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
        let empirical_repetitions = repetitions_for_target(p, target_success_probability)?;
        let conservative_repetitions =
            repetitions_for_target(lower, target_success_probability)?;

        let empirical_tts = total_cost(empirical_repetitions, single_run_cost)?;
        let conservative_tts = total_cost(conservative_repetitions, single_run_cost)?;

        Ok(Self {
            successes,
            trials,
            empirical_success_probability: p,
            success_probability_lower: lower,
            success_probability_upper: upper,
            target_success_probability,
            single_run_cost,
            empirical_repetitions,
            empirical_tts,
            conservative_repetitions,
            conservative_tts,
        })
    }
}

/// Return the minimum whole-run count to reach the target aggregate success.
///
/// One run with success probability p repeated r times succeeds at least once
/// with probability 1 - (1-p)^r. A success probability of exactly one requires
/// one run; zero observed probability yields None, not zero cost.
fn repetitions_for_target(p: f64, target: f64) -> Result<Option<u64>, TtsError> {
    if !p.is_finite() || !(0.0..=1.0).contains(&p) {
        return Err(TtsError::NumericOverflow);
    }
    if p == 0.0 {
        return Ok(None);
    }
    if p == 1.0 {
        return Ok(Some(1));
    }

    let numerator = (-target).ln_1p();
    let denominator = (-p).ln_1p();
    let raw = numerator / denominator;
    if !raw.is_finite() || raw <= 0.0 || raw.ceil() > u64::MAX as f64 {
        return Err(TtsError::RepetitionCountOverflow);
    }
    Ok(Some(raw.ceil() as u64))
}

/// Compute a two-sided Wilson score interval for a Bernoulli probability.
fn wilson_interval(p: f64, n: f64, z: f64) -> (f64, f64) {
    let z_squared = z * z;
    let denominator = 1.0 + z_squared / n;
    let center = (p + z_squared / (2.0 * n)) / denominator;
    let margin = z * ((p * (1.0 - p) / n + z_squared / (4.0 * n * n)).sqrt())
        / denominator;
    ((center - margin).max(0.0), (center + margin).min(1.0))
}

fn total_cost(repetitions: Option<u64>, single_run_cost: f64) -> Result<Option<f64>, TtsError> {
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
    fn empirical_tts_uses_whole_repetitions_to_hit_the_target() {
        let estimate = TtsEstimate::from_counts(50, 100, 2.0, 0.99, Z_95)
            .expect("valid trial summary");
        // ln(0.01)/ln(0.5) ~= 6.64; whole attempts require seven runs.
        assert_eq!(estimate.empirical_repetitions, Some(7));
        assert_eq!(estimate.empirical_tts, Some(14.0));
        assert!(estimate.success_probability_lower < 0.5);
        assert!(estimate.success_probability_upper > 0.5);
        assert!(estimate.conservative_repetitions.unwrap() >= 7);
    }

    #[test]
    fn no_successes_never_becomes_zero_time_to_solution() {
        let estimate = TtsEstimate::from_counts(0, 100, 1.0, 0.99, Z_95)
            .expect("zero successes are valid data");
        assert_eq!(estimate.empirical_success_probability, 0.0);
        assert_eq!(estimate.empirical_repetitions, None);
        assert_eq!(estimate.empirical_tts, None);
        assert_eq!(estimate.success_probability_lower, 0.0);
        assert_eq!(estimate.conservative_tts, None);
    }

    #[test]
    fn all_successes_still_reports_statistical_uncertainty() {
        let estimate = TtsEstimate::from_counts(10, 10, 3.0, 0.99, Z_95)
            .expect("all-successes are valid data");
        assert_eq!(estimate.empirical_repetitions, Some(1));
        assert_eq!(estimate.empirical_tts, Some(3.0));
        assert!(estimate.success_probability_lower > 0.0);
        assert!(estimate.success_probability_lower < 1.0);
        assert!(estimate.conservative_repetitions.unwrap() > 1);
        assert!(estimate.conservative_tts.unwrap() > 3.0);
    }

    #[test]
    fn a_single_perfect_trial_does_not_claim_certainty() {
        let estimate = TtsEstimate::from_counts(1, 1, 1.0, 0.99, Z_95)
            .expect("one success is valid data");
        assert_eq!(estimate.empirical_repetitions, Some(1));
        assert!(estimate.success_probability_lower < 1.0);
        assert!(estimate.conservative_repetitions.unwrap() > 1);
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
        assert!(estimate.empirical_repetitions.unwrap() > 40_000);
        assert!(estimate.empirical_tts.unwrap() > 10_000.0);
        assert!(estimate.conservative_tts.unwrap() >= estimate.empirical_tts.unwrap());
    }
}
