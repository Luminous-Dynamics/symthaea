// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Privacy Budget Tracking
//!
//! Tracks cumulative privacy expenditure across multiple operations
//! using sequential composition.

use crate::{DpError, DpResult};
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

/// Privacy budget tracker for managing cumulative privacy loss
///
/// Uses sequential composition to track total (ε,δ) expenditure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyBudget {
    /// Maximum allowed epsilon
    max_epsilon: f64,
    /// Maximum allowed delta
    max_delta: f64,
    /// Current cumulative epsilon
    current_epsilon: f64,
    /// Current cumulative delta
    current_delta: f64,
    /// Number of queries performed
    query_count: usize,
    /// History of privacy expenditures
    #[serde(skip)]
    history: Vec<PrivacyExpenditure>,
}

/// Record of a single privacy expenditure
#[derive(Debug, Clone)]
pub struct PrivacyExpenditure {
    pub epsilon: f64,
    pub delta: f64,
    pub operation: String,
    pub timestamp: std::time::Instant,
}

impl PrivacyBudget {
    /// Create a new privacy budget
    ///
    /// # Arguments
    /// * `max_epsilon` - Maximum total epsilon allowed
    /// * `max_delta` - Maximum total delta allowed
    pub fn new(max_epsilon: f64, max_delta: f64) -> Self {
        info!(
            max_epsilon = max_epsilon,
            max_delta = max_delta,
            "Created privacy budget"
        );

        Self {
            max_epsilon,
            max_delta,
            current_epsilon: 0.0,
            current_delta: 0.0,
            query_count: 0,
            history: Vec::new(),
        }
    }

    /// Consume privacy budget for an operation
    ///
    /// Uses sequential composition: ε_total = Σε_i, δ_total = Σδ_i
    ///
    /// # Arguments
    /// * `epsilon` - Epsilon for this operation
    /// * `delta` - Delta for this operation
    ///
    /// # Returns
    /// Ok(()) if budget available, Err if exhausted
    pub fn consume(&mut self, epsilon: f64, delta: f64) -> DpResult<()> {
        self.consume_with_label(epsilon, delta, "anonymous")
    }

    /// Consume privacy budget with an operation label
    pub fn consume_with_label(&mut self, epsilon: f64, delta: f64, operation: &str) -> DpResult<()> {
        let new_epsilon = self.current_epsilon + epsilon;
        let new_delta = self.current_delta + delta;

        if new_epsilon > self.max_epsilon {
            warn!(
                requested = epsilon,
                remaining = self.remaining_epsilon(),
                "Privacy budget exhausted (epsilon)"
            );
            return Err(DpError::BudgetExhausted {
                requested: epsilon,
                remaining: self.remaining_epsilon(),
            });
        }

        if new_delta > self.max_delta {
            warn!(
                requested = delta,
                remaining = self.remaining_delta(),
                "Privacy budget exhausted (delta)"
            );
            return Err(DpError::BudgetExhausted {
                requested: delta,
                remaining: self.remaining_delta(),
            });
        }

        self.current_epsilon = new_epsilon;
        self.current_delta = new_delta;
        self.query_count += 1;

        self.history.push(PrivacyExpenditure {
            epsilon,
            delta,
            operation: operation.to_string(),
            timestamp: std::time::Instant::now(),
        });

        debug!(
            epsilon = epsilon,
            delta = delta,
            operation = operation,
            total_epsilon = self.current_epsilon,
            total_delta = self.current_delta,
            "Consumed privacy budget"
        );

        Ok(())
    }

    /// Check if an operation would exceed the budget
    pub fn can_afford(&self, epsilon: f64, delta: f64) -> bool {
        self.current_epsilon + epsilon <= self.max_epsilon
            && self.current_delta + delta <= self.max_delta
    }

    /// Get remaining epsilon budget
    pub fn remaining_epsilon(&self) -> f64 {
        self.max_epsilon - self.current_epsilon
    }

    /// Get remaining delta budget
    pub fn remaining_delta(&self) -> f64 {
        self.max_delta - self.current_delta
    }

    /// Get current epsilon spent
    pub fn spent_epsilon(&self) -> f64 {
        self.current_epsilon
    }

    /// Get current delta spent
    pub fn spent_delta(&self) -> f64 {
        self.current_delta
    }

    /// Get the number of queries performed
    pub fn query_count(&self) -> usize {
        self.query_count
    }

    /// Get percentage of epsilon budget used
    pub fn epsilon_usage_percent(&self) -> f64 {
        (self.current_epsilon / self.max_epsilon) * 100.0
    }

    /// Get percentage of delta budget used
    pub fn delta_usage_percent(&self) -> f64 {
        (self.current_delta / self.max_delta) * 100.0
    }

    /// Reset the budget (use with caution)
    pub fn reset(&mut self) {
        warn!("Privacy budget reset - this should only be done at epoch boundaries");
        self.current_epsilon = 0.0;
        self.current_delta = 0.0;
        self.query_count = 0;
        self.history.clear();
    }

    /// Get the history of expenditures
    pub fn history(&self) -> &[PrivacyExpenditure] {
        &self.history
    }

    /// Create a snapshot of the current state
    pub fn snapshot(&self) -> BudgetSnapshot {
        BudgetSnapshot {
            max_epsilon: self.max_epsilon,
            max_delta: self.max_delta,
            current_epsilon: self.current_epsilon,
            current_delta: self.current_delta,
            query_count: self.query_count,
        }
    }
}

/// Immutable snapshot of budget state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetSnapshot {
    pub max_epsilon: f64,
    pub max_delta: f64,
    pub current_epsilon: f64,
    pub current_delta: f64,
    pub query_count: usize,
}

impl std::fmt::Display for PrivacyBudget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PrivacyBudget(ε={:.4}/{:.4}, δ={:.2e}/{:.2e}, queries={})",
            self.current_epsilon,
            self.max_epsilon,
            self.current_delta,
            self.max_delta,
            self.query_count
        )
    }
}

/// Privacy budget for a single training round
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundBudget {
    /// Epsilon per round
    pub epsilon_per_round: f64,
    /// Delta per round
    pub delta_per_round: f64,
    /// Total number of rounds
    pub total_rounds: usize,
    /// Current round
    pub current_round: usize,
}

impl RoundBudget {
    /// Create a new round-based budget
    ///
    /// Divides total budget evenly across rounds.
    pub fn new(total_epsilon: f64, total_delta: f64, total_rounds: usize) -> Self {
        Self {
            epsilon_per_round: total_epsilon / total_rounds as f64,
            delta_per_round: total_delta / total_rounds as f64,
            total_rounds,
            current_round: 0,
        }
    }

    /// Advance to the next round
    pub fn advance_round(&mut self) -> bool {
        if self.current_round < self.total_rounds {
            self.current_round += 1;
            true
        } else {
            false
        }
    }

    /// Get total epsilon used so far
    pub fn total_epsilon_used(&self) -> f64 {
        self.epsilon_per_round * self.current_round as f64
    }

    /// Get total delta used so far
    pub fn total_delta_used(&self) -> f64 {
        self.delta_per_round * self.current_round as f64
    }

    /// Remaining rounds
    pub fn remaining_rounds(&self) -> usize {
        self.total_rounds.saturating_sub(self.current_round)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_creation() {
        let budget = PrivacyBudget::new(1.0, 1e-5);
        assert_eq!(budget.remaining_epsilon(), 1.0);
        assert_eq!(budget.remaining_delta(), 1e-5);
        assert_eq!(budget.query_count(), 0);
    }

    #[test]
    fn test_budget_consumption() {
        let mut budget = PrivacyBudget::new(1.0, 1e-5);

        budget.consume(0.3, 1e-6).unwrap();
        assert!((budget.remaining_epsilon() - 0.7).abs() < 1e-10);
        assert_eq!(budget.query_count(), 1);

        budget.consume(0.3, 1e-6).unwrap();
        assert!((budget.remaining_epsilon() - 0.4).abs() < 1e-10);
        assert_eq!(budget.query_count(), 2);
    }

    #[test]
    fn test_budget_exhaustion() {
        let mut budget = PrivacyBudget::new(1.0, 1e-5);

        budget.consume(0.9, 1e-6).unwrap();
        let result = budget.consume(0.2, 1e-6);

        assert!(result.is_err());
        match result {
            Err(DpError::BudgetExhausted { requested, remaining }) => {
                assert!((requested - 0.2).abs() < 1e-10);
                assert!((remaining - 0.1).abs() < 1e-10);
            }
            _ => panic!("Expected BudgetExhausted error"),
        }
    }

    #[test]
    fn test_can_afford() {
        let mut budget = PrivacyBudget::new(1.0, 1e-5);
        budget.consume(0.8, 1e-6).unwrap();

        assert!(budget.can_afford(0.1, 1e-7));
        assert!(!budget.can_afford(0.3, 1e-7));
    }

    #[test]
    fn test_round_budget() {
        let mut round_budget = RoundBudget::new(1.0, 1e-5, 10);

        assert!((round_budget.epsilon_per_round - 0.1).abs() < 1e-10);
        assert_eq!(round_budget.remaining_rounds(), 10);

        round_budget.advance_round();
        assert_eq!(round_budget.current_round, 1);
        assert_eq!(round_budget.remaining_rounds(), 9);
        assert!((round_budget.total_epsilon_used() - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_budget_display() {
        let budget = PrivacyBudget::new(1.0, 1e-5);
        let display = format!("{}", budget);
        assert!(display.contains("PrivacyBudget"));
    }
}
