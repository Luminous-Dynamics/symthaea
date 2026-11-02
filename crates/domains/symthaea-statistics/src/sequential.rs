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
        self.successes.saturating_add(self.failures)
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
}
