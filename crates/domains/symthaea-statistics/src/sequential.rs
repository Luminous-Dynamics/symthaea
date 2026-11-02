// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Narrow sequential inference primitives with explicit model assumptions.
//!
//! This module starts with a Bernoulli SPRT for two fixed simple hypotheses.
//! It does not make arbitrary repeated fixed-sample p-value inspection valid,
//! and it does not handle clustered observations or composite hypotheses.

/// Invalid parameter or state in a sequential procedure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequentialError {
    InvalidProbability(&'static str),
    InvalidErrorBudget,
    InvalidBounds,
    ObservationOutOfBounds,
    AlreadyStopped,
    CountOverflow,
    NonFiniteStatistic,
}

/// Terminal or non-terminal decision from a Bernoulli SPRT.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SequentialDecision {
    Continue,
    AcceptAlternative,
    AcceptNull,
}

/// Bernoulli SPRT for two fixed simple hypotheses:
/// H0: P(success) = p0; H1: P(success) = p1.
///
/// Observations must be independent Bernoulli outcomes from the same declared
/// sampling process. Alpha and beta are Wald design error targets. This is not
/// a procedure for composite hypotheses, clusters, adaptively changed endpoints
/// or post-hoc choices of p0 and p1.
#[derive(Debug, Clone, PartialEq)]
pub struct BernoulliSprt {
    p0: f64,
    p1: f64,
    alpha: f64,
    beta: f64,
    upper_log_boundary: f64,
    lower_log_boundary: f64,
    log_likelihood_ratio: f64,
    successes: u64,
    failures: u64,
    decision: SequentialDecision,
}

impl BernoulliSprt {
    /// Construct an SPRT for fixed probabilities and error targets.
    pub fn new(p0: f64, p1: f64, alpha: f64, beta: f64) -> Result<Self, SequentialError> {
        for (name, value) in [
            ("p0", p0),
            ("p1", p1),
            ("alpha", alpha),
            ("beta", beta),
        ] {
            if !value.is_finite() || value <= 0.0 || value >= 1.0 {
                return Err(SequentialError::InvalidProbability(name));
            }
        }
        if p0 == p1 {
            return Err(SequentialError::InvalidProbability("distinct p0 and p1"));
        }
        if alpha + beta >= 1.0 {
            return Err(SequentialError::InvalidErrorBudget);
        }

        let upper_log_boundary = (1.0 - beta).ln() - alpha.ln();
        let lower_log_boundary = beta.ln() - (1.0 - alpha).ln();
        if !upper_log_boundary.is_finite() || !lower_log_boundary.is_finite() {
            return Err(SequentialError::NonFiniteStatistic);
        }

        Ok(Self {
            p0,
            p1,
            alpha,
            beta,
            upper_log_boundary,
            lower_log_boundary,
            log_likelihood_ratio: 0.0,
            successes: 0,
            failures: 0,
            decision: SequentialDecision::Continue,
        })
    }

    /// Add one Bernoulli observation. Further data are rejected after stopping.
    pub fn observe(&mut self, success: bool) -> Result<SequentialDecision, SequentialError> {
        if self.decision != SequentialDecision::Continue {
            return Err(SequentialError::AlreadyStopped);
        }

        self.successes
            .checked_add(self.failures)
            .and_then(|total| total.checked_add(1))
            .ok_or(SequentialError::CountOverflow)?;

        let next_count = if success {
            self.successes.checked_add(1).ok_or(SequentialError::CountOverflow)?
        } else {
            self.failures.checked_add(1).ok_or(SequentialError::CountOverflow)?
        };

        let increment = if success {
            self.p1.ln() - self.p0.ln()
        } else {
            (-self.p1).ln_1p() - (-self.p0).ln_1p()
        };
        let next_llr = self.log_likelihood_ratio + increment;
        if !next_llr.is_finite() {
            return Err(SequentialError::NonFiniteStatistic);
        }

        if success {
            self.successes = next_count;
        } else {
            self.failures = next_count;
        }
        self.log_likelihood_ratio = next_llr;
        self.decision = if next_llr >= self.upper_log_boundary {
            SequentialDecision::AcceptAlternative
        } else if next_llr <= self.lower_log_boundary {
            SequentialDecision::AcceptNull
        } else {
            SequentialDecision::Continue
        };
        Ok(self.decision)
    }

    pub fn decision(&self) -> SequentialDecision {
        self.decision
    }

    pub fn log_likelihood_ratio(&self) -> f64 {
        self.log_likelihood_ratio
    }

    pub fn successes(&self) -> u64 {
        self.successes
    }

    pub fn failures(&self) -> u64 {
        self.failures
    }

    pub fn observations(&self) -> u64 {
        self.successes + self.failures
    }

    pub fn upper_log_boundary(&self) -> f64 {
        self.upper_log_boundary
    }

    pub fn lower_log_boundary(&self) -> f64 {
        self.lower_log_boundary
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    pub fn beta(&self) -> f64 {
        self.beta
    }
}

/// A time-uniform confidence interval for a bounded sample mean.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedMeanInterval {
    pub observations: u64,
    pub mean: f64,
    pub radius: f64,
    pub lower: f64,
    pub upper: f64,
    pub confidence: f64,
}

/// Hoeffding confidence sequence for independent observations in a fixed,
/// known interval [lower_bound, upper_bound].
///
/// At time t, this uses a two-sided spending schedule
/// alpha_t = 6 * alpha / (pi^2 * t^2). Since the sum over t is alpha, a
/// union bound gives simultaneous coverage across all inspected times under
/// the independent bounded-observation model.
///
/// This is not valid for arbitrary within-participant serial dependence or
/// clustered trials. Aggregate at the true independent unit or use a
/// dependence-aware procedure.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedMeanConfidenceSequence {
    lower_bound: f64,
    upper_bound: f64,
    range: f64,
    alpha: f64,
    observations: u64,
    mean: f64,
}

impl BoundedMeanConfidenceSequence {
    /// Construct a confidence sequence for a fixed, finite support interval.
    pub fn new(
        lower_bound: f64,
        upper_bound: f64,
        alpha: f64,
    ) -> Result<Self, SequentialError> {
        if !lower_bound.is_finite()
            || !upper_bound.is_finite()
            || lower_bound >= upper_bound
            || !(upper_bound - lower_bound).is_finite()
        {
            return Err(SequentialError::InvalidBounds);
        }
        if !alpha.is_finite() || alpha <= 0.0 || alpha >= 1.0 {
            return Err(SequentialError::InvalidProbability("alpha"));
        }

        Ok(Self {
            lower_bound,
            upper_bound,
            range: upper_bound - lower_bound,
            alpha,
            observations: 0,
            mean: 0.0,
        })
    }

    /// Add an observation and return the current time-uniform interval.
    ///
    /// Invalid observations are rejected without mutating the accumulator.
    pub fn observe(&mut self, value: f64) -> Result<BoundedMeanInterval, SequentialError> {
        if !value.is_finite() || value < self.lower_bound || value > self.upper_bound {
            return Err(SequentialError::ObservationOutOfBounds);
        }

        let next_n = self.observations.checked_add(1).ok_or(SequentialError::CountOverflow)?;
        let next_mean = if self.observations == 0 {
            value
        } else {
            self.mean + (value - self.mean) / next_n as f64
        };
        if !next_mean.is_finite() {
            return Err(SequentialError::NonFiniteStatistic);
        }

        // log(pi^2 * n^2 / (3 * alpha)) is evaluated additively to avoid
        // overflow/underflow in the product and ratio.
        let n = next_n as f64;
        let log_term = 2.0 * std::f64::consts::PI.ln()
            + 2.0 * n.ln()
            - 3.0_f64.ln()
            - self.alpha.ln();
        if !log_term.is_finite() || log_term <= 0.0 {
            return Err(SequentialError::NonFiniteStatistic);
        }

        let variance_factor = log_term / (2.0 * n);
        if !variance_factor.is_finite() || variance_factor < 0.0 {
            return Err(SequentialError::NonFiniteStatistic);
        }

        // If the radius covers the complete support, clipping gives the full
        // interval. Cap before multiplying to avoid avoidable overflow.
        let radius = if variance_factor >= 1.0 {
            self.range
        } else {
            self.range * variance_factor.sqrt()
        };
        if !radius.is_finite() {
            return Err(SequentialError::NonFiniteStatistic);
        }

        let lower = (next_mean - radius).max(self.lower_bound);
        let upper = (next_mean + radius).min(self.upper_bound);
        self.observations = next_n;
        self.mean = next_mean;

        Ok(BoundedMeanInterval {
            observations: next_n,
            mean: next_mean,
            radius,
            lower,
            upper,
            confidence: 1.0 - self.alpha,
        })
    }

    pub fn observations(&self) -> u64 {
        self.observations
    }

    pub fn mean(&self) -> Option<f64> {
        (self.observations > 0).then_some(self.mean)
    }

    pub fn lower_bound(&self) -> f64 {
        self.lower_bound
    }

    pub fn upper_bound(&self) -> f64 {
        self.upper_bound
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repeated_successes_cross_alternative_boundary_and_stop() {
        let mut test = BernoulliSprt::new(0.3, 0.7, 0.05, 0.05).unwrap();
        let mut decision = SequentialDecision::Continue;
        for _ in 0..4 {
            decision = test.observe(true).unwrap();
        }
        assert_eq!(decision, SequentialDecision::AcceptAlternative);
        assert!(test.log_likelihood_ratio() >= test.upper_log_boundary());
        assert_eq!(test.observe(false), Err(SequentialError::AlreadyStopped));
    }

    #[test]
    fn repeated_failures_cross_null_boundary() {
        let mut test = BernoulliSprt::new(0.3, 0.7, 0.05, 0.05).unwrap();
        let mut decision = SequentialDecision::Continue;
        for _ in 0..4 {
            decision = test.observe(false).unwrap();
        }
        assert_eq!(decision, SequentialDecision::AcceptNull);
        assert!(test.log_likelihood_ratio() <= test.lower_log_boundary());
    }

    #[test]
    fn decreasing_alternative_is_supported() {
        let mut test = BernoulliSprt::new(0.7, 0.3, 0.05, 0.05).unwrap();
        let mut decision = SequentialDecision::Continue;
        for _ in 0..4 {
            decision = test.observe(false).unwrap();
        }
        assert_eq!(decision, SequentialDecision::AcceptAlternative);
    }

    #[test]
    fn rejects_overlapping_wald_error_budget() {
        assert_eq!(
            BernoulliSprt::new(0.3, 0.7, 0.7, 0.3),
            Err(SequentialError::InvalidErrorBudget)
        );
    }

    #[test]
    fn rejects_invalid_probabilities_and_equal_hypotheses() {
        assert!(matches!(
            BernoulliSprt::new(0.0, 0.7, 0.05, 0.05),
            Err(SequentialError::InvalidProbability("p0"))
        ));
        assert!(matches!(
            BernoulliSprt::new(0.3, 0.3, 0.05, 0.05),
            Err(SequentialError::InvalidProbability("distinct p0 and p1"))
        ));
        assert!(matches!(
            BernoulliSprt::new(0.3, 0.7, 1.0, 0.05),
            Err(SequentialError::InvalidProbability("alpha"))
        ));
    }

    #[test]
    fn bounded_mean_sequence_contains_constant_mean_and_updates_time() {
        let mut cs = BoundedMeanConfidenceSequence::new(0.0, 1.0, 0.05).unwrap();
        assert_eq!(cs.mean(), None);
        let first = cs.observe(0.75).unwrap();
        let second = cs.observe(0.75).unwrap();

        assert_eq!(first.observations, 1);
        assert_eq!(second.observations, 2);
        assert!((second.mean - 0.75).abs() < 1e-12);
        assert!(first.lower <= 0.75 && first.upper >= 0.75);
        assert_eq!(second.confidence, 0.95);
    }

    #[test]
    fn bounded_mean_sequence_rejects_out_of_support_without_mutation() {
        let mut cs = BoundedMeanConfidenceSequence::new(0.0, 1.0, 0.05).unwrap();
        assert_eq!(
            cs.observe(1.1),
            Err(SequentialError::ObservationOutOfBounds)
        );
        assert_eq!(cs.observations(), 0);
        assert_eq!(cs.mean(), None);
        assert_eq!(
            cs.observe(f64::NAN),
            Err(SequentialError::ObservationOutOfBounds)
        );
    }

    #[test]
    fn bounded_mean_sequence_matches_reference_radius_at_two_thousand() {
        let mut cs = BoundedMeanConfidenceSequence::new(0.0, 1.0, 0.05).unwrap();
        let mut interval = None;
        for _ in 0..2000 {
            interval = Some(cs.observe(0.75).unwrap());
        }
        let interval = interval.unwrap();
        assert!((interval.radius - 0.069_621_089_972_200_32).abs() < 1e-14);
        assert!((interval.lower - (0.75 - interval.radius)).abs() < 1e-14);
        assert!((interval.upper - (0.75 + interval.radius)).abs() < 1e-14);
        assert_eq!(interval.observations, 2000);
    }

    #[test]
    fn sprt_total_count_overflow_is_rejected() {
        let mut test = BernoulliSprt::new(0.3, 0.7, 0.05, 0.05).unwrap();
        test.successes = u64::MAX;
        assert_eq!(test.observe(false), Err(SequentialError::CountOverflow));
        assert_eq!(test.failures, 0);
    }

    #[test]
    fn bounded_mean_sequence_rejects_invalid_bounds_and_alpha() {
        assert_eq!(
            BoundedMeanConfidenceSequence::new(1.0, 1.0, 0.05),
            Err(SequentialError::InvalidBounds)
        );
        assert_eq!(
            BoundedMeanConfidenceSequence::new(0.0, 1.0, 0.0),
            Err(SequentialError::InvalidProbability("alpha"))
        );
    }

}
