//! A small, dependency-free reference for held-out selective-retrieval risk gates.
//!
//! This crate estimates a one-sided Clopper-Pearson upper confidence bound for
//! a Bernoulli unsafe-result rate. The caller defines what counts as unsafe
//! before evaluation (for example, exact-ID Recall@K below a frozen threshold).
//!
//! # Statistical boundary
//!
//! The binomial bound assumes the evaluated units are independent Bernoulli
//! trials with a common success probability, or exchangeable under the declared
//! evaluation design. Repeated timing rounds of the same query are not new
//! independent units. Correlated queries from the same memory lineage should be
//! grouped/sampled at the lineage level. The bound describes the evaluated
//! population; it is not a distribution-free guarantee under arbitrary drift,
//! adversarial selection, or changing retrieval/index versions.
//!
//! This crate evaluates an observed held-out accepted set. It does not predict
//! per-query risk, fit a selective policy, prove ANN correctness, or replace
//! temporal/scope/deletion invariants.

use std::error::Error;
use std::fmt;

mod risk_coverage;
pub use risk_coverage::{
    FixedSequenceRiskReport, RiskCoverageError, RiskCoveragePoint, RiskObservation,
    ThresholdCertificationPoint, ThresholdCertificationStatus,
    certify_fixed_threshold_sequence, risk_coverage_curve,
};

/// Errors in risk-bound or policy configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskGateError {
    /// The requested confidence level must be strictly between zero and one.
    InvalidConfidenceLevel,
    /// The maximum unsafe rate must be strictly between zero and one.
    InvalidMaximumUnsafeRate,
    /// Unsafe outcomes cannot exceed the number of accepted trials.
    UnsafeCountExceedsAccepted,
    /// A binomial upper bound needs at least one evaluated trial.
    NoTrials,
    /// The minimum evidence count must be positive.
    ZeroMinimumAccepted,
    /// The requested minimum sample count exceeds the platform range.
    SampleSizeOverflow,
}

impl fmt::Display for RiskGateError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfidenceLevel => {
                write!(formatter, "confidence level must be strictly between 0 and 1")
            }
            Self::InvalidMaximumUnsafeRate => {
                write!(formatter, "maximum unsafe rate must be strictly between 0 and 1")
            }
            Self::UnsafeCountExceedsAccepted => {
                write!(formatter, "unsafe outcome count exceeds accepted trial count")
            }
            Self::NoTrials => write!(formatter, "at least one evaluated trial is required"),
            Self::ZeroMinimumAccepted => {
                write!(formatter, "minimum accepted sample count must be positive")
            }
            Self::SampleSizeOverflow => {
                write!(formatter, "required sample size exceeds the platform range")
            }
        }
    }
}

impl Error for RiskGateError {}

/// Decision from an evaluation of a frozen selective-retrieval policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RiskGateDecision {
    /// The held-out evidence meets the configured upper-bound risk budget.
    Eligible,
    /// Too few independent accepted cases exist to qualify the policy.
    InsufficientEvidence,
    /// The estimated upper confidence bound exceeds the configured budget.
    RiskBudgetExceeded,
}

/// Auditable summary for one frozen policy on a held-out set.
#[derive(Debug, Clone, PartialEq)]
pub struct RiskGateReport {
    /// Number of held-out cases the policy accepted (not all observed queries).
    pub accepted_count: usize,
    /// Number of accepted cases labelled unsafe by the preregistered scorer.
    pub unsafe_count: usize,
    /// Observed unsafe rate, or None when no cases were accepted.
    pub observed_unsafe_rate: Option<f64>,
    /// One-sided Clopper-Pearson upper confidence bound, or None without data.
    pub upper_confidence_bound: Option<f64>,
    /// Requested confidence level, e.g. 0.95.
    pub confidence_level: f64,
    /// Maximum permitted upper bound, e.g. 0.01.
    pub maximum_unsafe_rate: f64,
    /// Minimum number of accepted independent evaluation units.
    pub minimum_accepted: usize,
    /// Final gate decision; no sample is treated as a pass.
    pub decision: RiskGateDecision,
}

/// Preregistered risk budget for one query stratum and one frozen policy.
///
/// Use a separate gate per important stratum rather than pooling easy queries
/// with harder or security-sensitive ones. Tune the policy elsewhere, then
/// evaluate once on a locked test set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RiskBudget {
    /// Maximum allowed one-sided upper confidence bound on unsafe rate.
    maximum_unsafe_rate: f64,
    /// Confidence level for the one-sided exact binomial bound.
    confidence_level: f64,
    /// Minimum accepted test units required before a pass can be reported.
    minimum_accepted: usize,
}

impl RiskBudget {
    /// Construct a validated risk budget.
    pub fn new(
        maximum_unsafe_rate: f64,
        confidence_level: f64,
        minimum_accepted: usize,
    ) -> Result<Self, RiskGateError> {
        validate_rate(maximum_unsafe_rate).map_err(|_| RiskGateError::InvalidMaximumUnsafeRate)?;
        validate_confidence(confidence_level)?;
        if minimum_accepted == 0 {
            return Err(RiskGateError::ZeroMinimumAccepted);
        }
        Ok(Self {
            maximum_unsafe_rate,
            confidence_level,
            minimum_accepted,
        })
    }

    /// Maximum one-sided upper confidence bound permitted by this budget.
    pub const fn maximum_unsafe_rate(self) -> f64 {
        self.maximum_unsafe_rate
    }

    /// Confidence level requested for the one-sided bound.
    pub const fn confidence_level(self) -> f64 {
        self.confidence_level
    }

    /// Minimum accepted cases required before a threshold can be tested.
    pub const fn minimum_accepted(self) -> usize {
        self.minimum_accepted
    }

    /// Score the observed accepted cases for a frozen policy on held-out data.
    ///
    /// The caller must define the unsafe label and independent sampling unit
    /// before inspecting final-test outcomes. A zero-case evaluation has no
    /// bound and returns InsufficientEvidence, never Eligible.
    pub fn assess(
        self,
        unsafe_count: usize,
        accepted_count: usize,
    ) -> Result<RiskGateReport, RiskGateError> {
        if unsafe_count > accepted_count {
            return Err(RiskGateError::UnsafeCountExceedsAccepted);
        }

        if accepted_count == 0 {
            return Ok(RiskGateReport {
                accepted_count,
                unsafe_count,
                observed_unsafe_rate: None,
                upper_confidence_bound: None,
                confidence_level: self.confidence_level,
                maximum_unsafe_rate: self.maximum_unsafe_rate,
                minimum_accepted: self.minimum_accepted,
                decision: RiskGateDecision::InsufficientEvidence,
            });
        }

        let upper = one_sided_clopper_pearson_upper(
            unsafe_count,
            accepted_count,
            self.confidence_level,
        )?;
        let observed = unsafe_count as f64 / accepted_count as f64;
        let decision = if accepted_count < self.minimum_accepted {
            RiskGateDecision::InsufficientEvidence
        } else if upper <= self.maximum_unsafe_rate {
            RiskGateDecision::Eligible
        } else {
            RiskGateDecision::RiskBudgetExceeded
        };

        Ok(RiskGateReport {
            accepted_count,
            unsafe_count,
            observed_unsafe_rate: Some(observed),
            upper_confidence_bound: Some(upper),
            confidence_level: self.confidence_level,
            maximum_unsafe_rate: self.maximum_unsafe_rate,
            minimum_accepted: self.minimum_accepted,
            decision,
        })
    }
}

/// Compute the one-sided exact Clopper-Pearson upper confidence bound for a
/// Bernoulli unsafe rate.
///
/// unsafe_count is the number of unsafe accepted cases, and trials is the
/// number of accepted evaluation units. confidence_level must be in (0, 1).
/// The result solves P[X <= unsafe_count | X ~ Binomial(trials, p)] = alpha
/// for p, with alpha = 1 - confidence_level.
///
/// For zero observed unsafe cases this simplifies to
/// 1 - alpha^(1 / trials). For nonzero counts, a bisection solver evaluates
/// the binomial CDF in the log domain to avoid underflow for small tail
/// probabilities. This is an exact binomial interval up to floating-point
/// numerical precision; it does not make the underlying sampling assumptions
/// true.
pub fn one_sided_clopper_pearson_upper(
    unsafe_count: usize,
    trials: usize,
    confidence_level: f64,
) -> Result<f64, RiskGateError> {
    validate_confidence(confidence_level)?;
    if trials == 0 {
        return Err(RiskGateError::NoTrials);
    }
    if unsafe_count > trials {
        return Err(RiskGateError::UnsafeCountExceedsAccepted);
    }
    if unsafe_count == trials {
        return Ok(1.0);
    }

    let alpha = 1.0 - confidence_level;
    if unsafe_count == 0 {
        // -expm1(x) is more accurate than 1 - exp(x) when x is near zero.
        return Ok(-((alpha.ln() / trials as f64).exp_m1()));
    }

    let target_log_probability = alpha.ln();
    let mut lower = 0.0;
    let mut upper = 1.0;

    // CDF(X <= k) decreases monotonically with p. Maintain a bracket whose
    // upper endpoint has CDF <= alpha; returning that endpoint is conservative
    // up to floating-point precision.
    for _ in 0..80 {
        let midpoint = lower + (upper - lower) / 2.0;
        let log_cdf = log_binomial_cdf_at_most(unsafe_count, trials, midpoint);
        if log_cdf > target_log_probability {
            lower = midpoint;
        } else {
            upper = midpoint;
        }
    }
    Ok(upper)
}

/// Minimum independent accepted cases needed for a zero-failure sample to put
/// its one-sided exact upper bound at or below maximum_unsafe_rate.
///
/// This is a sample-size lower bound for the zero-failure case only. Nonzero
/// observed failures require a larger sample and must use the general bound.
pub fn minimum_zero_failure_trials(
    maximum_unsafe_rate: f64,
    confidence_level: f64,
) -> Result<usize, RiskGateError> {
    validate_rate(maximum_unsafe_rate).map_err(|_| RiskGateError::InvalidMaximumUnsafeRate)?;
    validate_confidence(confidence_level)?;

    let alpha = 1.0 - confidence_level;
    // ln_1p preserves tiny but representable risk budgets without rounding
    // (1 - maximum_unsafe_rate) to exactly 1.0.
    let required = alpha.ln() / (-maximum_unsafe_rate).ln_1p();
    if !required.is_finite() || required <= 0.0 {
        return Err(RiskGateError::InvalidMaximumUnsafeRate);
    }
    if required >= usize::MAX as f64 {
        return Err(RiskGateError::SampleSizeOverflow);
    }
    Ok(required.ceil() as usize)
}

fn validate_confidence(confidence_level: f64) -> Result<(), RiskGateError> {
    if !confidence_level.is_finite() || !(0.0 < confidence_level && confidence_level < 1.0) {
        return Err(RiskGateError::InvalidConfidenceLevel);
    }
    Ok(())
}

fn validate_rate(rate: f64) -> Result<(), RiskGateError> {
    if !rate.is_finite() || !(0.0 < rate && rate < 1.0) {
        return Err(RiskGateError::InvalidMaximumUnsafeRate);
    }
    Ok(())
}

/// Log of the binomial lower-tail probability. Summing in log space avoids
/// underflow when the CDF is far into the lower tail.
fn log_binomial_cdf_at_most(k: usize, n: usize, p: f64) -> f64 {
    if k >= n {
        return 0.0;
    }
    if p <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return f64::NEG_INFINITY;
    }

    let log_p = p.ln();
    let log_q = (-p).ln_1p();
    let mut log_term = n as f64 * log_q; // log P[X = 0]
    let mut log_sum = log_term;

    for j in 1..=k {
        log_term += ((n - j + 1) as f64).ln() - (j as f64).ln() + log_p - log_q;
        log_sum = log_add_exp(log_sum, log_term);
    }
    log_sum.min(0.0)
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    if left == f64::NEG_INFINITY {
        return right;
    }
    if right == f64::NEG_INFINITY {
        return left;
    }
    let maximum = left.max(right);
    let minimum = left.min(right);
    maximum + (minimum - maximum).exp().ln_1p()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_failure_bound_matches_closed_form() {
        let upper = one_sided_clopper_pearson_upper(0, 100, 0.95).unwrap();
        let expected = 1.0 - 0.05_f64.powf(1.0 / 100.0);
        assert!((upper - expected).abs() < 1e-12);
    }

    #[test]
    fn classic_one_failure_in_one_hundred_is_about_4_7_percent() {
        let upper = one_sided_clopper_pearson_upper(1, 100, 0.95).unwrap();
        assert!(upper > 0.04 && upper < 0.05, "upper={upper}");
    }

    #[test]
    fn all_unsafe_cases_yield_unit_upper_bound() {
        assert_eq!(one_sided_clopper_pearson_upper(10, 10, 0.95).unwrap(), 1.0);
    }

    #[test]
    fn upper_bound_increases_with_unsafe_count() {
        let zero = one_sided_clopper_pearson_upper(0, 100, 0.95).unwrap();
        let one = one_sided_clopper_pearson_upper(1, 100, 0.95).unwrap();
        let two = one_sided_clopper_pearson_upper(2, 100, 0.95).unwrap();
        assert!(zero < one && one < two);
    }

    #[test]
    fn zero_failures_need_299_trials_for_one_percent_at_95_percent_confidence() {
        assert_eq!(minimum_zero_failure_trials(0.01, 0.95).unwrap(), 299);
        let at_298 = one_sided_clopper_pearson_upper(0, 298, 0.95).unwrap();
        let at_299 = one_sided_clopper_pearson_upper(0, 299, 0.95).unwrap();
        assert!(at_298 > 0.01);
        assert!(at_299 <= 0.01);
    }

    #[test]
    fn empty_or_too_small_accepted_set_never_passes() {
        let budget = RiskBudget::new(0.01, 0.95, 299).unwrap();
        let empty = budget.assess(0, 0).unwrap();
        assert_eq!(empty.decision, RiskGateDecision::InsufficientEvidence);
        assert_eq!(empty.upper_confidence_bound, None);

        let too_small = budget.assess(0, 100).unwrap();
        assert_eq!(too_small.decision, RiskGateDecision::InsufficientEvidence);
        assert!(too_small.upper_confidence_bound.unwrap() > 0.01);
    }

    #[test]
    fn zero_failures_only_pass_after_evidence_and_risk_budgets_pass() {
        let budget = RiskBudget::new(0.01, 0.95, 299).unwrap();
        let report = budget.assess(0, 299).unwrap();
        assert_eq!(report.decision, RiskGateDecision::Eligible);
        assert!(report.upper_confidence_bound.unwrap() <= 0.01);
    }

    #[test]
    fn nonzero_failures_can_exceed_gate_after_minimum_sample_is_met() {
        let budget = RiskBudget::new(0.01, 0.95, 299).unwrap();
        let report = budget.assess(1, 299).unwrap();
        assert_eq!(report.decision, RiskGateDecision::RiskBudgetExceeded);
        assert!(report.upper_confidence_bound.unwrap() > 0.01);
    }

    #[test]
    fn invalid_configurations_and_counts_fail_closed() {
        assert_eq!(
            RiskBudget::new(0.0, 0.95, 100).unwrap_err(),
            RiskGateError::InvalidMaximumUnsafeRate
        );
        assert_eq!(
            RiskBudget::new(0.01, 1.0, 100).unwrap_err(),
            RiskGateError::InvalidConfidenceLevel
        );
        assert_eq!(
            RiskBudget::new(0.01, 0.95, 0).unwrap_err(),
            RiskGateError::ZeroMinimumAccepted
        );
        assert_eq!(
            one_sided_clopper_pearson_upper(0, 0, 0.95).unwrap_err(),
            RiskGateError::NoTrials
        );
        assert_eq!(
            one_sided_clopper_pearson_upper(3, 2, 0.95).unwrap_err(),
            RiskGateError::UnsafeCountExceedsAccepted
        );
    }

    #[test]
    fn confidence_bounds_tighten_with_more_zero_failure_trials() {
        let n100 = one_sided_clopper_pearson_upper(0, 100, 0.95).unwrap();
        let n1000 = one_sided_clopper_pearson_upper(0, 1_000, 0.95).unwrap();
        assert!(n1000 < n100);
    }

    #[test]
    fn tiny_risk_target_reports_sample_size_overflow() {
        assert_eq!(
            minimum_zero_failure_trials(1e-300, 0.95).unwrap_err(),
            RiskGateError::SampleSizeOverflow
        );
    }
}
