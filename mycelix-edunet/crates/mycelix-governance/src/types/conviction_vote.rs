// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Conviction voting implementation - time-weighted commitment to prevent manipulation
//!
//! ## Revolutionary Feature: Exponential Manipulation Cost
//!
//! Traditional voting allows instant strategic voting:
//! - Vote → Immediate full weight → Withdraw → No cost
//! - Manipulation cost: Near zero
//!
//! Conviction voting changes the game theory:
//! - Conviction grows exponentially with time locked
//! - conviction(t) = 1 + (max - 1) × (1 - e^(-t/τ))
//! - Early unlock = Heavy penalty (50% reputation loss)
//!
//! **Result**: Manipulation cost increases exponentially with conviction required
//!
//! ## Mathematical Foundation
//!
//! ```text
//! Conviction Formula:
//!   C(t) = 1 + (C_max - 1) × (1 - e^(-t/τ))
//!
//! where:
//!   t = time locked (days)
//!   τ = time constant (default: 7 days)
//!   C_max = maximum conviction multiplier (default: 2.0)
//!
//! Manipulation Cost:
//!   Cost = (1 + time_factor) × (1 + opportunity_cost_rate × t)
//!
//! Example (τ = 7 days):
//!   t=1 day   → C(1) ≈ 1.13  (13% bonus)
//!   t=7 days  → C(7) ≈ 1.63  (63% bonus)
//!   t=14 days → C(14) ≈ 1.86 (86% bonus)
//!   t=∞       → C(∞) = 2.00  (100% bonus)
//! ```

use serde::{Deserialize, Serialize};

/// Parameters for conviction voting system
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConvictionParameters {
    /// Time constant (τ) in days for exponential growth (default: 7.0)
    pub time_constant_days: f64,

    /// Maximum conviction multiplier (C_max) (default: 2.0 = 2x weight)
    pub max_conviction_multiplier: f64,

    /// Minimum lock period to get any bonus (hours) (default: 24)
    pub min_lock_period_hours: i64,

    /// Penalty for breaking commitment early (% of allocated reputation) (default: 0.5 = 50%)
    pub early_unlock_penalty: f64,

    /// Whether unlocking is allowed at all (default: true)
    pub allow_early_unlock: bool,
}

impl Default for ConvictionParameters {
    fn default() -> Self {
        Self {
            time_constant_days: 7.0,
            max_conviction_multiplier: 2.0,
            min_lock_period_hours: 24,
            early_unlock_penalty: 0.5,
            allow_early_unlock: true,
        }
    }
}

impl ConvictionParameters {
    /// Validate conviction parameters
    pub fn validate(&self) -> Result<(), String> {
        if self.time_constant_days <= 0.0 {
            return Err("time_constant_days must be positive".to_string());
        }

        if self.max_conviction_multiplier < 1.0 || self.max_conviction_multiplier > 10.0 {
            return Err("max_conviction_multiplier must be between 1.0 and 10.0".to_string());
        }

        if self.min_lock_period_hours < 0 {
            return Err("min_lock_period_hours cannot be negative".to_string());
        }

        if self.early_unlock_penalty < 0.0 || self.early_unlock_penalty > 1.0 {
            return Err("early_unlock_penalty must be between 0.0 and 1.0".to_string());
        }

        Ok(())
    }

    /// Calculate conviction multiplier based on time locked
    ///
    /// ## Formula
    ///
    /// ```text
    /// C(t) = 1 + (C_max - 1) × (1 - e^(-t/τ))
    /// ```
    ///
    /// ## Example
    ///
    /// ```rust
    /// use mycelix_governance::types::conviction_vote::ConvictionParameters;
    ///
    /// let params = ConvictionParameters::default();
    /// let now = chrono::Utc::now().timestamp();
    /// let lock_start = now - (7 * 86400); // 7 days ago
    ///
    /// let conviction = params.calculate_conviction(lock_start, now);
    /// assert!((conviction - 1.63).abs() < 0.05); // ~1.63 after 7 days
    /// ```
    pub fn calculate_conviction(&self, lock_start: i64, current_time: i64) -> f64 {
        let elapsed_seconds = (current_time - lock_start).max(0) as f64;
        let elapsed_days = elapsed_seconds / 86400.0;

        // Check minimum lock period
        let min_lock_days = self.min_lock_period_hours as f64 / 24.0;
        if elapsed_days < min_lock_days {
            return 1.0; // No bonus yet
        }

        // Apply exponential formula
        let tau = self.time_constant_days;
        let max_mult = self.max_conviction_multiplier;

        1.0 + (max_mult - 1.0) * (1.0 - (-elapsed_days / tau).exp())
    }

    /// Calculate penalty amount for early unlock
    ///
    /// Returns the amount of reputation that will be lost
    pub fn calculate_unlock_penalty(&self, reputation_locked: f64) -> f64 {
        if !self.allow_early_unlock {
            return f64::INFINITY; // Cannot unlock at all
        }

        reputation_locked * self.early_unlock_penalty
    }

    /// Check if conviction has reached maximum
    pub fn has_reached_maximum(&self, lock_start: i64, current_time: i64) -> bool {
        let conviction = self.calculate_conviction(lock_start, current_time);
        (conviction - self.max_conviction_multiplier).abs() < 0.01
    }

    /// Get time remaining to reach maximum conviction
    ///
    /// Returns seconds remaining, or None if already at maximum
    pub fn time_to_maximum(&self, lock_start: i64, current_time: i64) -> Option<i64> {
        if self.has_reached_maximum(lock_start, current_time) {
            return None;
        }

        // Solve for t when C(t) ≈ 0.99 × C_max
        // t = -τ × ln(1 - 0.99 × (C_max - 1) / (C_max - 1))
        // t = -τ × ln(0.01)
        // t ≈ 4.6 × τ

        let days_to_max = 4.6 * self.time_constant_days;
        let elapsed_days = (current_time - lock_start) as f64 / 86400.0;
        let remaining_days = (days_to_max - elapsed_days).max(0.0);

        Some((remaining_days * 86400.0) as i64)
    }
}

/// A vote with conviction lock
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConvictionVote {
    /// When the conviction lock started
    pub lock_start: i64,

    /// When the lock expires (None = indefinite until proposal resolves)
    pub lock_expiry: Option<i64>,

    /// Current conviction multiplier (recalculated on each query)
    pub current_conviction: f64,

    /// Parameters used for this vote
    pub params: ConvictionParameters,

    /// Whether the vote has been unlocked early
    pub unlocked_early: bool,

    /// Penalty paid for early unlock (if any)
    pub penalty_paid: f64,
}

impl ConvictionVote {
    /// Create a new conviction vote with lock
    ///
    /// ## Arguments
    ///
    /// * `lock_duration_hours` - How long to lock (None = until proposal resolves)
    /// * `params` - Conviction parameters
    ///
    /// ## Returns
    ///
    /// A new ConvictionVote with conviction=1.0 initially
    pub fn new(lock_duration_hours: Option<i64>, params: ConvictionParameters) -> Self {
        let now = chrono::Utc::now().timestamp();
        let lock_expiry = lock_duration_hours.map(|hours| now + (hours * 3600));

        Self {
            lock_start: now,
            lock_expiry,
            current_conviction: 1.0,
            params,
            unlocked_early: false,
            penalty_paid: 0.0,
        }
    }

    /// Update conviction based on current time
    ///
    /// This should be called before using the vote weight
    pub fn update_conviction(&mut self) {
        if self.unlocked_early {
            return; // Conviction frozen at unlock time
        }

        let now = chrono::Utc::now().timestamp();
        self.current_conviction = self.params.calculate_conviction(self.lock_start, now);
    }

    /// Check if the lock has naturally expired
    pub fn is_naturally_expired(&self) -> bool {
        match self.lock_expiry {
            Some(expiry) => chrono::Utc::now().timestamp() >= expiry,
            None => false, // Indefinite lock
        }
    }

    /// Get remaining lock time in seconds
    pub fn remaining_lock_time(&self) -> Option<i64> {
        match self.lock_expiry {
            Some(expiry) => {
                let now = chrono::Utc::now().timestamp();
                Some((expiry - now).max(0))
            }
            None => None, // Indefinite
        }
    }

    /// Unlock early with penalty
    ///
    /// Returns the penalty amount that must be paid
    pub fn unlock_early(&mut self, reputation_locked: f64) -> Result<f64, String> {
        if self.unlocked_early {
            return Err("Already unlocked".to_string());
        }

        if self.is_naturally_expired() {
            // Free unlock if naturally expired
            self.unlocked_early = true;
            return Ok(0.0);
        }

        let penalty = self.params.calculate_unlock_penalty(reputation_locked);

        if penalty.is_infinite() {
            return Err("Early unlock not allowed".to_string());
        }

        self.unlocked_early = true;
        self.penalty_paid = penalty;

        Ok(penalty)
    }

    /// Get time remaining to reach maximum conviction
    pub fn time_to_max_conviction(&self) -> Option<i64> {
        let now = chrono::Utc::now().timestamp();
        self.params.time_to_maximum(self.lock_start, now)
    }

    /// Check if conviction has reached maximum
    pub fn at_max_conviction(&self) -> bool {
        let now = chrono::Utc::now().timestamp();
        self.params.has_reached_maximum(self.lock_start, now)
    }
}

/// Conviction decay model for reputation over time
///
/// This prevents "parking" conviction indefinitely
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConvictionDecay {
    /// Decay rate per day (default: 0.01 = 1% per day)
    pub daily_decay_rate: f64,

    /// Minimum conviction before decay starts (default: 1.0)
    pub decay_threshold: f64,
}

impl Default for ConvictionDecay {
    fn default() -> Self {
        Self {
            daily_decay_rate: 0.01, // 1% per day
            decay_threshold: 1.0,
        }
    }
}

impl ConvictionDecay {
    /// Apply decay to conviction
    ///
    /// Returns new conviction value after decay
    pub fn apply_decay(&self, current_conviction: f64, days_inactive: i64) -> f64 {
        if current_conviction <= self.decay_threshold {
            return current_conviction;
        }

        let decay_factor = (1.0 - self.daily_decay_rate).powi(days_inactive as i32);
        (current_conviction * decay_factor).max(self.decay_threshold)
    }

    /// Calculate days until conviction decays to threshold
    pub fn days_to_threshold(&self, current_conviction: f64) -> i64 {
        if current_conviction <= self.decay_threshold {
            return 0;
        }

        // Solve: threshold = current × (1 - rate)^days
        // days = ln(threshold/current) / ln(1 - rate)
        let ratio = self.decay_threshold / current_conviction;
        let days = ratio.ln() / (1.0 - self.daily_decay_rate).ln();

        days.ceil() as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conviction_calculation() {
        let params = ConvictionParameters::default();

        // Test at various time points
        let now = chrono::Utc::now().timestamp();

        // t=0: conviction should be 1.0
        let c0 = params.calculate_conviction(now, now);
        assert!((c0 - 1.0).abs() < 0.01);

        // t=1 day: conviction ≈ 1.13
        let c1 = params.calculate_conviction(now - 86400, now);
        assert!((c1 - 1.13).abs() < 0.05);

        // t=7 days (1 time constant): conviction ≈ 1.63
        let c7 = params.calculate_conviction(now - (7 * 86400), now);
        assert!((c7 - 1.63).abs() < 0.05);

        // t=14 days: conviction ≈ 1.86
        let c14 = params.calculate_conviction(now - (14 * 86400), now);
        assert!((c14 - 1.86).abs() < 0.05);
    }

    #[test]
    fn test_min_lock_period() {
        let params = ConvictionParameters::default();
        let now = chrono::Utc::now().timestamp();

        // Less than 24 hours = no bonus
        let lock_start = now - (12 * 3600); // 12 hours ago
        let conviction = params.calculate_conviction(lock_start, now);
        assert_eq!(conviction, 1.0);

        // Exactly 24 hours = bonus starts
        let lock_start = now - (24 * 3600);
        let conviction = params.calculate_conviction(lock_start, now);
        assert!(conviction > 1.0);
    }

    #[test]
    fn test_conviction_vote_creation() {
        let params = ConvictionParameters::default();
        let vote = ConvictionVote::new(Some(168), params); // 7 days

        assert_eq!(vote.current_conviction, 1.0);
        assert!(!vote.unlocked_early);
        assert_eq!(vote.penalty_paid, 0.0);
    }

    #[test]
    fn test_conviction_update() {
        let params = ConvictionParameters::default();
        let mut vote = ConvictionVote::new(None, params);

        // Artificially set lock_start to 7 days ago
        vote.lock_start = chrono::Utc::now().timestamp() - (7 * 86400);

        vote.update_conviction();
        assert!((vote.current_conviction - 1.63).abs() < 0.05);
    }

    #[test]
    fn test_early_unlock_penalty() {
        let params = ConvictionParameters::default();
        let mut vote = ConvictionVote::new(Some(168), params);

        // Unlock early
        let penalty = vote.unlock_early(100.0).unwrap();
        assert_eq!(penalty, 50.0); // 50% of 100

        assert!(vote.unlocked_early);
        assert_eq!(vote.penalty_paid, 50.0);

        // Cannot unlock twice
        assert!(vote.unlock_early(100.0).is_err());
    }

    #[test]
    fn test_natural_expiry() {
        let params = ConvictionParameters::default();
        let mut vote = ConvictionVote::new(Some(1), params); // 1 hour

        // Artificially expire the lock
        vote.lock_expiry = Some(chrono::Utc::now().timestamp() - 3600);

        assert!(vote.is_naturally_expired());

        // No penalty for natural expiry
        let penalty = vote.unlock_early(100.0).unwrap();
        assert_eq!(penalty, 0.0);
    }

    #[test]
    fn test_time_to_maximum() {
        let params = ConvictionParameters::default();
        let now = chrono::Utc::now().timestamp();

        // At t=0, time to max should be ~4.6 × 7 days = ~32 days
        let time_to_max = params.time_to_maximum(now, now).unwrap();
        let expected_days = 4.6 * 7.0;
        let expected_seconds = (expected_days * 86400.0) as i64;
        assert!((time_to_max - expected_seconds).abs() < 3600); // Within 1 hour

        // At t=42 days (6τ), should be within 0.01 of maximum (threshold used by has_reached_maximum)
        // C(42) = 1 + 1 × (1 - e^(-6)) ≈ 1.9975, which is < 0.01 from max of 2.0
        let lock_start = now - (42 * 86400);
        assert!(params.has_reached_maximum(lock_start, now));
        assert!(params.time_to_maximum(lock_start, now).is_none());
    }

    #[test]
    fn test_conviction_decay() {
        let decay = ConvictionDecay::default();

        // 10% conviction after 30 days of 1% daily decay
        let initial = 2.0;
        let decayed = decay.apply_decay(initial, 30);
        let expected = 2.0 * 0.99_f64.powi(30); // ≈ 1.48
        assert!((decayed - expected).abs() < 0.01);

        // Below threshold should not decay
        let low_conviction = 0.9;
        let decayed = decay.apply_decay(low_conviction, 30);
        assert_eq!(decayed, low_conviction);
    }

    #[test]
    fn test_days_to_threshold() {
        let decay = ConvictionDecay::default();

        let current = 2.0;
        let days = decay.days_to_threshold(current);

        // Verify by applying decay for that many days
        let decayed = decay.apply_decay(current, days);
        assert!((decayed - decay.decay_threshold).abs() < 0.01);
    }

    #[test]
    fn test_params_validation() {
        let mut params = ConvictionParameters::default();
        assert!(params.validate().is_ok());

        params.time_constant_days = -1.0;
        assert!(params.validate().is_err());

        params.time_constant_days = 7.0;
        params.max_conviction_multiplier = 15.0;
        assert!(params.validate().is_err());

        params.max_conviction_multiplier = 2.0;
        params.early_unlock_penalty = 1.5;
        assert!(params.validate().is_err());
    }
}
