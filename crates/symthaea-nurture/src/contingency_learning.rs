// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Contingency learning: P(caregiver_responds | distress).
//!
//! Tracks the infant's learned expectation of caregiver responsiveness via
//! Bayesian updating. This forms the empirical foundation of attachment:
//! "When I signal distress, does someone come?"
//!
//! Science: Watson (1979) contingency perception; Bigelow & DeCoste (2003).

use serde::{Deserialize, Serialize};

/// Contingency learner for caregiver response prediction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContingencyLearner {
    /// Learned probability that caregiver responds to distress (0-1).
    pub response_probability: f64,
    /// Total distress signals emitted.
    pub sample_count: u64,
    /// Number of responses received.
    pub response_count: u64,
    /// Mean latency of caregiver response (seconds), EMA.
    pub response_latency_mean: f64,
    /// Variance of response latency (for prediction uncertainty).
    pub response_latency_variance: f64,
    /// Prior strength (pseudo-count for Bayesian updating).
    prior_strength: f64,
}

impl ContingencyLearner {
    /// Create a new contingency learner with uninformative prior.
    pub fn new() -> Self {
        Self {
            response_probability: 0.5,
            sample_count: 0,
            response_count: 0,
            response_latency_mean: 30.0, // Default 30 seconds
            response_latency_variance: 100.0,
            prior_strength: 2.0, // Weak prior (equivalent to 1 success + 1 failure)
        }
    }

    /// Record a distress-response event.
    ///
    /// - `responded`: whether the caregiver responded
    /// - `latency_seconds`: time until response (0 if no response)
    pub fn record_distress_response(&mut self, responded: bool, latency_seconds: f64) {
        self.sample_count += 1;
        if responded {
            self.response_count += 1;

            // Update latency EMA (only for positive responses)
            let alpha = 0.1;
            let delta = latency_seconds - self.response_latency_mean;
            self.response_latency_mean += alpha * delta;
            self.response_latency_variance =
                (1.0 - alpha) * (self.response_latency_variance + alpha * delta * delta);
        }

        // Bayesian update: P(respond) = (successes + prior) / (total + 2*prior)
        self.response_probability = (self.response_count as f64 + self.prior_strength / 2.0)
            / (self.sample_count as f64 + self.prior_strength);
    }

    /// Current predicted probability that caregiver will respond.
    pub fn predicted_response_probability(&self) -> f64 {
        self.response_probability
    }

    /// Expected wait time for response (seconds).
    pub fn expected_wait_time(&self) -> f64 {
        self.response_latency_mean
    }

    /// Uncertainty in response prediction (higher = more unpredictable).
    pub fn response_uncertainty(&self) -> f64 {
        if self.sample_count < 2 {
            return 1.0; // Maximum uncertainty with no data
        }
        // Entropy-like measure: p * (1-p), maximal at p=0.5
        let p = self.response_probability;
        4.0 * p * (1.0 - p) // Scaled to [0, 1]
    }

    /// Standard deviation of response latency.
    pub fn response_latency_std(&self) -> f64 {
        self.response_latency_variance.sqrt()
    }
}

impl Default for ContingencyLearner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_learner_uninformative() {
        let learner = ContingencyLearner::new();
        assert!((learner.response_probability - 0.5).abs() < 0.01);
        assert_eq!(learner.sample_count, 0);
    }

    #[test]
    fn test_reliable_caregiver_high_probability() {
        let mut learner = ContingencyLearner::new();
        for _ in 0..100 {
            learner.record_distress_response(true, 5.0);
        }
        assert!(
            learner.predicted_response_probability() > 0.9,
            "Should learn high probability with consistent responses, got {}",
            learner.predicted_response_probability()
        );
    }

    #[test]
    fn test_absent_caregiver_low_probability() {
        let mut learner = ContingencyLearner::new();
        for _ in 0..100 {
            learner.record_distress_response(false, 0.0);
        }
        assert!(
            learner.predicted_response_probability() < 0.1,
            "Should learn low probability with no responses, got {}",
            learner.predicted_response_probability()
        );
    }

    #[test]
    fn test_inconsistent_caregiver_moderate_probability() {
        let mut learner = ContingencyLearner::new();
        for i in 0..100 {
            let responds = i % 2 == 0; // 50% response rate
            learner.record_distress_response(responds, if responds { 10.0 } else { 0.0 });
        }
        let p = learner.predicted_response_probability();
        assert!(
            p > 0.4 && p < 0.6,
            "50% response rate should give ~0.5 probability, got {p}"
        );
    }

    #[test]
    fn test_latency_tracking() {
        let mut learner = ContingencyLearner::new();
        // All responses at 10 seconds
        for _ in 0..50 {
            learner.record_distress_response(true, 10.0);
        }
        assert!(
            (learner.expected_wait_time() - 10.0).abs() < 5.0,
            "Mean latency should approach 10s, got {}",
            learner.expected_wait_time()
        );
    }

    #[test]
    fn test_uncertainty_high_for_inconsistent() {
        let mut consistent = ContingencyLearner::new();
        let mut inconsistent = ContingencyLearner::new();

        for _ in 0..100 {
            consistent.record_distress_response(true, 5.0);
        }
        for i in 0..100 {
            inconsistent.record_distress_response(i % 2 == 0, 5.0);
        }

        assert!(
            inconsistent.response_uncertainty() > consistent.response_uncertainty(),
            "Inconsistent should have higher uncertainty: {} vs {}",
            inconsistent.response_uncertainty(),
            consistent.response_uncertainty()
        );
    }

    #[test]
    fn test_sample_count_tracks() {
        let mut learner = ContingencyLearner::new();
        for _ in 0..37 {
            learner.record_distress_response(true, 5.0);
        }
        assert_eq!(learner.sample_count, 37);
        assert_eq!(learner.response_count, 37);
    }

    #[test]
    fn test_prior_prevents_extreme_early_estimates() {
        let mut learner = ContingencyLearner::new();
        learner.record_distress_response(true, 5.0);
        // With Bayesian prior, single success shouldn't give P=1.0
        assert!(
            learner.predicted_response_probability() < 0.8,
            "Prior should temper single-sample estimate, got {}",
            learner.predicted_response_probability()
        );
    }
}
