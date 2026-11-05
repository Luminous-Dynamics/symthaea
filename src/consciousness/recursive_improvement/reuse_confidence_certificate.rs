// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Pass/fail REUSE lower confidence certificate.
//!
//! Implements Lemma A3 of Sun et al. (arXiv:2609.33180v2), Appendix A.3:
//! the maximum of (1) a Clopper-Pearson sign/disagreement lower bound and
//! (2) a paired empirical-Bernstein lower bound.
//!
//! This module only computes the numerical certificate. It does not persist
//! the adaptive error budget, authenticate the task population, protect a
//! holdout, enforce decision-only feedback, or authorize promotion.

use symthaea_statistics::special::betai;

pub const MAX_CERTIFICATE_SAMPLE_COUNT: usize = 1_000_000;
const BETA_BISECTION_STEPS: usize = 96;
const KL_BISECTION_STEPS: usize = 96;

#[derive(Debug, Clone, PartialEq)]
pub enum ReuseCertificateError {
    InvalidErrorBudget,
    EmptySample,
    SampleCountTooLarge { actual: usize, maximum: usize },
    InvalidPairedDifference(i8),
    InvalidSuccessCount { successes: usize, trials: usize },
    NumericalFailure(&'static str),
}

impl std::fmt::Display for ReuseCertificateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidErrorBudget => {
                write!(f, "certificate error budget must be finite and strictly between 0 and 1")
            }
            Self::EmptySample => write!(f, "paired certificate requires at least one task"),
            Self::SampleCountTooLarge { actual, maximum } => {
                write!(f, "sample count {actual} exceeds reviewed maximum {maximum}")
            }
            Self::InvalidPairedDifference(value) => {
                write!(f, "pass/fail paired difference must be -1, 0 or +1, got {value}")
            }
            Self::InvalidSuccessCount { successes, trials } => {
                write!(f, "success count {successes} exceeds trial count {trials}")
            }
            Self::NumericalFailure(operation) => {
                write!(f, "non-finite or out-of-domain result in {operation}")
            }
        }
    }
}

impl std::error::Error for ReuseCertificateError {}

/// Audit-only numerical certificate for one selected pass/fail candidate.
///
/// Do not return this structure to the self-modifying proposer. REUSE requires
/// that the proposer receives only the round decision y_t; the certificate is
/// retained by the independent evaluator for auditing and cumulative progress.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReuseConfidenceCertificate {
    sample_count: usize,
    candidate_wins: usize,
    incumbent_wins: usize,
    discordant_count: usize,
    empirical_improvement: f64,
    conditional_win_lower: f64,
    discordance_rate_lower: f64,
    sign_lower_bound: f64,
    empirical_bernstein_lower_bound: f64,
    lower_bound: f64,
    cumulative_increment: f64,
}

impl ReuseConfidenceCertificate {
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    pub fn candidate_wins(&self) -> usize {
        self.candidate_wins
    }

    pub fn incumbent_wins(&self) -> usize {
        self.incumbent_wins
    }

    pub fn discordant_count(&self) -> usize {
        self.discordant_count
    }

    pub fn empirical_improvement(&self) -> f64 {
        self.empirical_improvement
    }

    pub fn conditional_win_lower(&self) -> f64 {
        self.conditional_win_lower
    }

    pub fn discordance_rate_lower(&self) -> f64 {
        self.discordance_rate_lower
    }

    pub fn sign_lower_bound(&self) -> f64 {
        self.sign_lower_bound
    }

    pub fn empirical_bernstein_lower_bound(&self) -> f64 {
        self.empirical_bernstein_lower_bound
    }

    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }

    /// Nonnegative amount that may be added to the cumulative progress ledger.
    pub fn cumulative_increment(&self) -> f64 {
        self.cumulative_increment
    }
}

/// Compute the pass/fail lower confidence certificate from matched paired
/// outcomes Z_i = candidate_success - incumbent_success in {-1, 0, +1}.
///
/// `delta_c` is the independent confidence-bound allocation (1-rho)*delta
/// for this specific candidate comparison. The two Clopper-Pearson limits each
/// use delta_c/4. The paired empirical-Bernstein bound uses L=ln(4/delta_c).
///
/// The input must contain the complete preregistered eligible task population
/// in canonical order. Missingness, exclusions and task clustering must have
/// been resolved by the caller under the benchmark protocol before calling.
pub fn pass_fail_confidence_certificate(
    paired_differences: &[i8],
    delta_c: f64,
) -> Result<ReuseConfidenceCertificate, ReuseCertificateError> {
    if !delta_c.is_finite() || delta_c <= 0.0 || delta_c >= 1.0 || delta_c / 4.0 == 0.0 {
        return Err(ReuseCertificateError::InvalidErrorBudget);
    }
    let n = paired_differences.len();
    if n == 0 {
        return Err(ReuseCertificateError::EmptySample);
    }
    if n > MAX_CERTIFICATE_SAMPLE_COUNT {
        return Err(ReuseCertificateError::SampleCountTooLarge {
            actual: n,
            maximum: MAX_CERTIFICATE_SAMPLE_COUNT,
        });
    }

    let mut candidate_wins = 0usize;
    let mut incumbent_wins = 0usize;
    for &difference in paired_differences {
        match difference {
            1 => candidate_wins += 1,
            -1 => incumbent_wins += 1,
            0 => {}
            other => return Err(ReuseCertificateError::InvalidPairedDifference(other)),
        }
    }

    let discordant_count = candidate_wins + incumbent_wins;
    let empirical_improvement =
        (candidate_wins as f64 - incumbent_wins as f64) / n as f64;
    let conditional_win_lower = clopper_pearson_lower(
        candidate_wins,
        discordant_count,
        delta_c / 4.0,
    )?;
    let discordance_rate_lower =
        clopper_pearson_lower(discordant_count, n, delta_c / 4.0)?;

    // Lemma A3: if q_lo <= 1/2 the sign-based lower bound is -infinity.
    // For M=0, both exact limits are zero, so this branch is also the
    // conservative no-disagreement behavior.
    let sign_lower_bound = if conditional_win_lower > 0.5 {
        (2.0 * conditional_win_lower - 1.0) * discordance_rate_lower
    } else {
        f64::NEG_INFINITY
    };

    let z2 = discordant_count as f64 / n as f64;
    let log_term = 4.0_f64.ln() - delta_c.ln();
    let second_moment_upper = upper_second_moment(z2, n, log_term)?;
    let linear = 2.0 * log_term / (3.0 * n as f64);
    let radicand = linear * linear + 2.0 * second_moment_upper * log_term / n as f64;
    if !linear.is_finite() || !radicand.is_finite() || radicand < 0.0 {
        return Err(ReuseCertificateError::NumericalFailure(
            "paired empirical-Bernstein deviation",
        ));
    }
    let empirical_bernstein_lower_bound =
        empirical_improvement - linear - radicand.sqrt();
    if !empirical_bernstein_lower_bound.is_finite() {
        return Err(ReuseCertificateError::NumericalFailure(
            "paired empirical-Bernstein lower bound",
        ));
    }

    let lower_bound = sign_lower_bound.max(empirical_bernstein_lower_bound);
    if lower_bound.is_nan() {
        return Err(ReuseCertificateError::NumericalFailure(
            "combined pass/fail lower bound",
        ));
    }
    let cumulative_increment = lower_bound.max(0.0);

    Ok(ReuseConfidenceCertificate {
        sample_count: n,
        candidate_wins,
        incumbent_wins,
        discordant_count,
        empirical_improvement,
        conditional_win_lower,
        discordance_rate_lower,
        sign_lower_bound,
        empirical_bernstein_lower_bound,
        lower_bound,
        cumulative_increment,
    })
}

/// One-sided Clopper-Pearson lower limit theta(k; m, alpha).
///
/// For 0 < k < m, this is the alpha-quantile of Beta(k, m-k+1), found by
/// monotone inversion of the repository's regularized incomplete-beta CDF.
/// Boundary cases use their exact closed forms.
pub fn clopper_pearson_lower(
    successes: usize,
    trials: usize,
    alpha: f64,
) -> Result<f64, ReuseCertificateError> {
    if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
        return Err(ReuseCertificateError::InvalidErrorBudget);
    }
    if successes > trials {
        return Err(ReuseCertificateError::InvalidSuccessCount { successes, trials });
    }
    if trials == 0 || successes == 0 {
        return Ok(0.0);
    }
    if trials > MAX_CERTIFICATE_SAMPLE_COUNT {
        return Err(ReuseCertificateError::SampleCountTooLarge {
            actual: trials,
            maximum: MAX_CERTIFICATE_SAMPLE_COUNT,
        });
    }
    if successes == trials {
        let value = alpha.powf(1.0 / trials as f64);
        return finite_probability(value, "Clopper-Pearson all-success boundary");
    }

    let a = successes as f64;
    let b = (trials - successes + 1) as f64;
    let mut low = 0.0_f64;
    let mut high = 1.0_f64;
    for _ in 0..BETA_BISECTION_STEPS {
        let midpoint = low + (high - low) * 0.5;
        let cdf = betai(a, b, midpoint);
        if !cdf.is_finite() || !(0.0..=1.0).contains(&cdf) {
            return Err(ReuseCertificateError::NumericalFailure(
                "regularized incomplete beta CDF",
            ));
        }
        if cdf < alpha {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }
    finite_probability(
        low + (high - low) * 0.5,
        "Clopper-Pearson beta quantile",
    )
}

/// Upper confidence limit for E[Z^2] from the Bernoulli KL inequality:
/// max q in [z2,1] such that n*kl(z2 || q) <= L.
fn upper_second_moment(
    z2: f64,
    sample_count: usize,
    log_term: f64,
) -> Result<f64, ReuseCertificateError> {
    if !z2.is_finite()
        || !(0.0..=1.0).contains(&z2)
        || sample_count == 0
        || !log_term.is_finite()
        || log_term <= 0.0
    {
        return Err(ReuseCertificateError::NumericalFailure(
            "upper second-moment input",
        ));
    }
    if z2 == 1.0 {
        return Ok(1.0);
    }
    if z2 == 0.0 {
        let value = -(-log_term / sample_count as f64).exp_m1();
        return finite_probability(value, "upper second-moment zero boundary");
    }

    let mut low = z2;
    let mut high = 1.0_f64;
    let budget = log_term / sample_count as f64;
    for _ in 0..KL_BISECTION_STEPS {
        let midpoint = low + (high - low) * 0.5;
        let divergence = bernoulli_kl(z2, midpoint);
        if divergence <= budget {
            low = midpoint;
        } else {
            high = midpoint;
        }
    }
    finite_probability(low + (high - low) * 0.5, "upper second-moment KL inverse")
}

/// Bernoulli KL(a || b), for 0 <= a <= b <= 1.
fn bernoulli_kl(a: f64, b: f64) -> f64 {
    if a == 0.0 {
        return -(-b).ln_1p();
    }
    if a == 1.0 {
        return -b.ln();
    }
    if b == 1.0 {
        return f64::INFINITY;
    }
    a * (a.ln() - b.ln()) + (1.0 - a) * ((1.0 - a).ln() - (1.0 - b).ln())
}

fn finite_probability(
    value: f64,
    operation: &'static str,
) -> Result<f64, ReuseCertificateError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(value)
    } else {
        Err(ReuseCertificateError::NumericalFailure(operation))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clopper_pearson_matches_independent_scipy_reference_vectors() {
        // Reference values generated independently with SciPy scipy.stats.beta.ppf.
        let cases = [
            (5, 10, 0.025, 0.187_086_028_447_398_55),
            (50, 100, 0.0125, 0.384_779_357_302_161_6),
            (1_000, 2_000, 0.00025, 0.460_898_721_299_683_23),
            (5_000, 10_000, 0.00025, 0.482_551_964_967_352_9),
        ];
        for (successes, trials, alpha, expected) in cases {
            let actual = clopper_pearson_lower(successes, trials, alpha).unwrap();
            assert!(
                (actual - expected).abs() < 2e-11,
                "k={successes}, m={trials}, alpha={alpha}: {actual} != {expected}"
            );
        }
    }

    #[test]
    fn clopper_pearson_handles_zero_all_success_and_invalid_counts() {
        assert_eq!(clopper_pearson_lower(0, 100, 0.025).unwrap(), 0.0);
        let expected = 0.025_f64.powf(0.01);
        assert!(
            (clopper_pearson_lower(100, 100, 0.025).unwrap() - expected).abs() < 1e-15
        );
        assert_eq!(
            clopper_pearson_lower(3, 2, 0.025),
            Err(ReuseCertificateError::InvalidSuccessCount {
                successes: 3,
                trials: 2
            })
        );
    }

    #[test]
    fn pass_fail_certificate_matches_independent_reference_fixture() {
        // 40 candidate wins, 10 incumbent wins and 50 ties, n=100.
        // The expected limits and lower bounds were computed with SciPy and
        // an independent implementation of Appendix A.1/A.3.
        let mut paired = vec![1_i8; 40];
        paired.extend(vec![-1_i8; 10]);
        paired.extend(vec![0_i8; 50]);
        let certificate = pass_fail_confidence_certificate(&paired, 0.05).unwrap();

        assert_eq!(certificate.sample_count(), 100);
        assert_eq!(certificate.candidate_wins(), 40);
        assert_eq!(certificate.incumbent_wins(), 10);
        assert_eq!(certificate.discordant_count(), 50);
        assert!((certificate.empirical_improvement() - 0.3).abs() < 1e-15);
        assert!((certificate.conditional_win_lower() - 0.643_191_882_701_759_1).abs() < 2e-10);
        assert!((certificate.discordance_rate_lower() - 0.384_779_357_302_161_6).abs() < 2e-10);
        assert!((certificate.sign_lower_bound() - 0.110_194_561_193_738_76).abs() < 2e-10);
        assert!(
            (certificate.empirical_bernstein_lower_bound() - 0.031_271_979_004_443_895).abs()
                < 2e-10
        );
        assert!((certificate.lower_bound() - 0.110_194_561_193_738_76).abs() < 2e-10);
        assert_eq!(
            certificate.cumulative_increment(),
            certificate.lower_bound()
        );
    }

    #[test]
    fn all_ties_produce_zero_cumulative_increment_without_fabricating_a_sign_bound() {
        let paired = vec![0_i8; 100];
        let certificate = pass_fail_confidence_certificate(&paired, 0.05).unwrap();
        assert_eq!(certificate.discordant_count(), 0);
        assert_eq!(certificate.conditional_win_lower(), 0.0);
        assert_eq!(certificate.discordance_rate_lower(), 0.0);
        assert!(certificate.sign_lower_bound().is_infinite());
        assert!(certificate.sign_lower_bound().is_sign_negative());
        assert!(certificate.lower_bound() < 0.0);
        assert_eq!(certificate.cumulative_increment(), 0.0);
    }

    #[test]
    fn certificate_rejects_empty_oversized_invalid_and_non_finite_inputs() {
        assert_eq!(
            pass_fail_confidence_certificate(&[], 0.05),
            Err(ReuseCertificateError::EmptySample)
        );
        assert_eq!(
            pass_fail_confidence_certificate(&[1, 2, 0], 0.05),
            Err(ReuseCertificateError::InvalidPairedDifference(2))
        );
        assert_eq!(
            pass_fail_confidence_certificate(&[1, 0], f64::NAN),
            Err(ReuseCertificateError::InvalidErrorBudget)
        );
        assert_eq!(
            pass_fail_confidence_certificate(&[1, 0], 0.0),
            Err(ReuseCertificateError::InvalidErrorBudget)
        );
        assert_eq!(
            pass_fail_confidence_certificate(&vec![0_i8; MAX_CERTIFICATE_SAMPLE_COUNT + 1], 0.05),
            Err(ReuseCertificateError::SampleCountTooLarge {
                actual: MAX_CERTIFICATE_SAMPLE_COUNT + 1,
                maximum: MAX_CERTIFICATE_SAMPLE_COUNT
            })
        );
    }
}
