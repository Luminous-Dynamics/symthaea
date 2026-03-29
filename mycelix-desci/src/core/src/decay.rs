// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Decay Mechanics System
//!
//! Implements time-based decay for epistemic trust and verification weights.
//! Older verifications contribute less to current trust scores,
//! encouraging ongoing verification of claims.

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

/// Decay function type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DecayFunction {
    /// Linear decay: weight = 1 - (age / half_life)
    Linear,
    /// Exponential decay: weight = exp(-lambda * age)
    Exponential,
    /// Logarithmic decay: weight = 1 / (1 + ln(1 + age / scale))
    Logarithmic,
    /// Step decay: full weight until threshold, then reduced
    Step,
    /// No decay - weight remains constant
    None,
}

impl DecayFunction {
    pub fn description(&self) -> &'static str {
        match self {
            Self::Linear => "Linear decay over time",
            Self::Exponential => "Exponential decay (half-life model)",
            Self::Logarithmic => "Slow logarithmic decay",
            Self::Step => "Step function decay after threshold",
            Self::None => "No decay applied",
        }
    }
}

/// Configuration for decay mechanics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayConfig {
    /// Type of decay function
    pub decay_function: DecayFunction,
    /// Half-life for exponential decay (in days)
    pub half_life_days: f64,
    /// Minimum weight (decay floor)
    pub min_weight: f64,
    /// Scale factor for logarithmic decay
    pub log_scale_days: f64,
    /// Threshold for step decay (days)
    pub step_threshold_days: f64,
    /// Weight after step threshold
    pub step_reduced_weight: f64,
    /// Whether to apply decay to verification counts
    pub decay_verifications: bool,
    /// Whether to apply decay to trust scores
    pub decay_trust: bool,
    /// Whether to apply decay to claim confidence
    pub decay_confidence: bool,
}

impl Default for DecayConfig {
    fn default() -> Self {
        Self {
            decay_function: DecayFunction::Exponential,
            half_life_days: 365.0, // 1 year half-life
            min_weight: 0.1,       // Never fully decay
            log_scale_days: 180.0,
            step_threshold_days: 730.0, // 2 years
            step_reduced_weight: 0.5,
            decay_verifications: true,
            decay_trust: true,
            decay_confidence: true,
        }
    }
}

/// Decay calculator
#[derive(Debug, Clone)]
pub struct DecayCalculator {
    config: DecayConfig,
}

impl DecayCalculator {
    pub fn new() -> Self {
        Self::with_config(DecayConfig::default())
    }

    pub fn with_config(config: DecayConfig) -> Self {
        Self { config }
    }

    /// Calculate decay weight for a given timestamp
    pub fn calculate_weight(&self, timestamp: DateTime<Utc>) -> f64 {
        self.calculate_weight_at(timestamp, Utc::now())
    }

    /// Calculate decay weight relative to a reference time
    pub fn calculate_weight_at(&self, timestamp: DateTime<Utc>, reference: DateTime<Utc>) -> f64 {
        if timestamp >= reference {
            return 1.0; // No decay for future timestamps
        }

        let age_days = (reference - timestamp).num_seconds() as f64 / 86400.0;

        let raw_weight = match self.config.decay_function {
            DecayFunction::None => 1.0,
            DecayFunction::Linear => self.linear_decay(age_days),
            DecayFunction::Exponential => self.exponential_decay(age_days),
            DecayFunction::Logarithmic => self.logarithmic_decay(age_days),
            DecayFunction::Step => self.step_decay(age_days),
        };

        raw_weight.max(self.config.min_weight)
    }

    fn linear_decay(&self, age_days: f64) -> f64 {
        let max_age = self.config.half_life_days * 2.0;
        1.0 - (age_days / max_age).min(1.0)
    }

    fn exponential_decay(&self, age_days: f64) -> f64 {
        let lambda = 0.693 / self.config.half_life_days; // ln(2) / half_life
        (-lambda * age_days).exp()
    }

    fn logarithmic_decay(&self, age_days: f64) -> f64 {
        1.0 / (1.0 + (1.0 + age_days / self.config.log_scale_days).ln())
    }

    fn step_decay(&self, age_days: f64) -> f64 {
        if age_days < self.config.step_threshold_days {
            1.0
        } else {
            self.config.step_reduced_weight
        }
    }

    /// Apply decay to a weighted verification
    pub fn decay_verification(&self, weight: f64, timestamp: DateTime<Utc>) -> f64 {
        if !self.config.decay_verifications {
            return weight;
        }
        weight * self.calculate_weight(timestamp)
    }

    /// Apply decay to a trust score
    pub fn decay_trust(&self, trust: f64, last_activity: DateTime<Utc>) -> f64 {
        if !self.config.decay_trust {
            return trust;
        }
        trust * self.calculate_weight(last_activity)
    }

    /// Apply decay to a confidence score
    pub fn decay_confidence(&self, confidence: f64, created_at: DateTime<Utc>) -> f64 {
        if !self.config.decay_confidence {
            return confidence;
        }
        confidence * self.calculate_weight(created_at)
    }

    /// Calculate effective verification count with decay
    pub fn effective_verification_count(
        &self,
        verifications: &[(f64, DateTime<Utc>)],
    ) -> f64 {
        verifications
            .iter()
            .map(|(weight, timestamp)| self.decay_verification(*weight, *timestamp))
            .sum()
    }

    /// Get decay statistics for a set of timestamps
    pub fn decay_stats(&self, timestamps: &[DateTime<Utc>]) -> DecayStats {
        if timestamps.is_empty() {
            return DecayStats {
                count: 0,
                raw_weight: 0.0,
                decayed_weight: 0.0,
                decay_ratio: 1.0,
                oldest_weight: 0.0,
                newest_weight: 0.0,
                average_weight: 0.0,
            };
        }

        let weights: Vec<f64> = timestamps.iter().map(|t| self.calculate_weight(*t)).collect();
        let raw_weight = timestamps.len() as f64;
        let decayed_weight: f64 = weights.iter().sum();

        DecayStats {
            count: timestamps.len(),
            raw_weight,
            decayed_weight,
            decay_ratio: decayed_weight / raw_weight,
            oldest_weight: weights.iter().cloned().fold(f64::INFINITY, f64::min),
            newest_weight: weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            average_weight: decayed_weight / timestamps.len() as f64,
        }
    }
}

impl Default for DecayCalculator {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about decay application
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayStats {
    /// Number of items
    pub count: usize,
    /// Raw (undecayed) weight
    pub raw_weight: f64,
    /// Decayed weight
    pub decayed_weight: f64,
    /// Ratio of decayed to raw
    pub decay_ratio: f64,
    /// Oldest item's weight
    pub oldest_weight: f64,
    /// Newest item's weight
    pub newest_weight: f64,
    /// Average weight
    pub average_weight: f64,
}

/// Decaying accumulator for ongoing calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayingAccumulator {
    /// Current decayed value
    value: f64,
    /// Last update time
    last_update: DateTime<Utc>,
    /// Decay calculator
    #[serde(skip)]
    calculator: Option<DecayCalculator>,
    /// Half-life for serialization
    half_life_days: f64,
}

impl DecayingAccumulator {
    pub fn new(half_life_days: f64) -> Self {
        let config = DecayConfig {
            decay_function: DecayFunction::Exponential,
            half_life_days,
            ..Default::default()
        };

        Self {
            value: 0.0,
            last_update: Utc::now(),
            calculator: Some(DecayCalculator::with_config(config)),
            half_life_days,
        }
    }

    fn get_calculator(&self) -> DecayCalculator {
        self.calculator.clone().unwrap_or_else(|| {
            let config = DecayConfig {
                decay_function: DecayFunction::Exponential,
                half_life_days: self.half_life_days,
                ..Default::default()
            };
            DecayCalculator::with_config(config)
        })
    }

    /// Add a value to the accumulator
    pub fn add(&mut self, amount: f64) {
        let now = Utc::now();
        let calc = self.get_calculator();

        // First decay the current value
        let decay_factor = calc.calculate_weight_at(self.last_update, now);
        self.value = self.value * decay_factor + amount;
        self.last_update = now;
    }

    /// Get the current decayed value
    pub fn current(&self) -> f64 {
        let now = Utc::now();
        let calc = self.get_calculator();
        let decay_factor = calc.calculate_weight_at(self.last_update, now);
        self.value * decay_factor
    }

    /// Get value at a specific time
    pub fn value_at(&self, time: DateTime<Utc>) -> f64 {
        let calc = self.get_calculator();
        if time <= self.last_update {
            // Looking at past - find what value was before decay
            let decay_since = calc.calculate_weight_at(time, self.last_update);
            self.value / decay_since.max(0.001) // Avoid division by zero
        } else {
            // Looking at future - apply more decay
            let decay_factor = calc.calculate_weight_at(self.last_update, time);
            self.value * decay_factor
        }
    }

    /// Reset the accumulator
    pub fn reset(&mut self) {
        self.value = 0.0;
        self.last_update = Utc::now();
    }
}

/// Preset decay profiles for different use cases
pub mod presets {
    use super::*;

    /// Fast decay for ephemeral claims (M0 materiality)
    pub fn ephemeral() -> DecayConfig {
        DecayConfig {
            decay_function: DecayFunction::Exponential,
            half_life_days: 7.0, // 1 week half-life
            min_weight: 0.01,
            ..Default::default()
        }
    }

    /// Medium decay for temporal claims (M1 materiality)
    pub fn temporal() -> DecayConfig {
        DecayConfig {
            decay_function: DecayFunction::Exponential,
            half_life_days: 90.0, // 3 month half-life
            min_weight: 0.1,
            ..Default::default()
        }
    }

    /// Slow decay for persistent claims (M2 materiality)
    pub fn persistent() -> DecayConfig {
        DecayConfig {
            decay_function: DecayFunction::Logarithmic,
            half_life_days: 365.0,
            log_scale_days: 365.0,
            min_weight: 0.3,
            ..Default::default()
        }
    }

    /// No decay for foundational claims (M3 materiality)
    pub fn foundational() -> DecayConfig {
        DecayConfig {
            decay_function: DecayFunction::None,
            min_weight: 1.0,
            ..Default::default()
        }
    }

    /// Decay profile matching materiality level
    pub fn for_materiality(level: u8) -> DecayConfig {
        match level {
            0 => ephemeral(),
            1 => temporal(),
            2 => persistent(),
            3 => foundational(),
            _ => DecayConfig::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_decay() {
        let config = DecayConfig {
            decay_function: DecayFunction::Exponential,
            half_life_days: 365.0,
            min_weight: 0.0,
            ..Default::default()
        };
        let calc = DecayCalculator::with_config(config);

        let now = Utc::now();

        // Just now should be ~1.0
        let weight_now = calc.calculate_weight_at(now, now);
        assert!((weight_now - 1.0).abs() < 0.001);

        // Half-life ago should be ~0.5
        let half_life_ago = now - Duration::days(365);
        let weight_half = calc.calculate_weight_at(half_life_ago, now);
        assert!((weight_half - 0.5).abs() < 0.01);

        // Two half-lives ago should be ~0.25
        let two_half_lives = now - Duration::days(730);
        let weight_quarter = calc.calculate_weight_at(two_half_lives, now);
        assert!((weight_quarter - 0.25).abs() < 0.01);
    }

    #[test]
    fn test_linear_decay() {
        let config = DecayConfig {
            decay_function: DecayFunction::Linear,
            half_life_days: 365.0,
            min_weight: 0.0,
            ..Default::default()
        };
        let calc = DecayCalculator::with_config(config);

        let now = Utc::now();
        let half_life_ago = now - Duration::days(365);

        let weight = calc.calculate_weight_at(half_life_ago, now);
        assert!((weight - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_step_decay() {
        let config = DecayConfig {
            decay_function: DecayFunction::Step,
            step_threshold_days: 365.0,
            step_reduced_weight: 0.5,
            min_weight: 0.0,
            ..Default::default()
        };
        let calc = DecayCalculator::with_config(config);

        let now = Utc::now();

        // Before threshold - full weight
        let before = now - Duration::days(100);
        assert_eq!(calc.calculate_weight_at(before, now), 1.0);

        // After threshold - reduced weight
        let after = now - Duration::days(400);
        assert_eq!(calc.calculate_weight_at(after, now), 0.5);
    }

    #[test]
    fn test_min_weight_floor() {
        let config = DecayConfig {
            decay_function: DecayFunction::Exponential,
            half_life_days: 1.0, // Very fast decay
            min_weight: 0.1,
            ..Default::default()
        };
        let calc = DecayCalculator::with_config(config);

        let now = Utc::now();
        let ancient = now - Duration::days(3650); // 10 years ago

        let weight = calc.calculate_weight_at(ancient, now);
        assert!(weight >= 0.1);
    }

    #[test]
    fn test_no_decay() {
        let config = DecayConfig {
            decay_function: DecayFunction::None,
            ..Default::default()
        };
        let calc = DecayCalculator::with_config(config);

        let now = Utc::now();
        let ancient = now - Duration::days(3650);

        assert_eq!(calc.calculate_weight_at(ancient, now), 1.0);
    }

    #[test]
    fn test_effective_verification_count() {
        let calc = DecayCalculator::new();
        let now = Utc::now();

        let verifications = vec![
            (1.0, now),                       // Recent - high weight
            (1.0, now - Duration::days(365)), // 1 year old - ~0.5 weight
            (1.0, now - Duration::days(730)), // 2 years old - ~0.25 weight
        ];

        let effective = calc.effective_verification_count(&verifications);

        // Should be less than 3.0 (raw sum)
        assert!(effective < 3.0);
        // But more than 1.0 (just the newest)
        assert!(effective > 1.0);
    }

    #[test]
    fn test_decaying_accumulator() {
        let mut acc = DecayingAccumulator::new(365.0);

        acc.add(100.0);
        let initial = acc.current();
        assert!((initial - 100.0).abs() < 0.1);

        // Value should be close to what we added
        assert!(acc.current() > 99.0);
    }

    #[test]
    fn test_presets() {
        let ephemeral = presets::ephemeral();
        assert_eq!(ephemeral.half_life_days, 7.0);

        let foundational = presets::foundational();
        assert_eq!(foundational.decay_function, DecayFunction::None);

        let by_materiality = presets::for_materiality(2);
        assert_eq!(by_materiality.decay_function, DecayFunction::Logarithmic);
    }

    #[test]
    fn test_decay_stats() {
        let calc = DecayCalculator::new();
        let now = Utc::now();

        let timestamps = vec![
            now,
            now - Duration::days(100),
            now - Duration::days(200),
        ];

        let stats = calc.decay_stats(&timestamps);

        assert_eq!(stats.count, 3);
        assert_eq!(stats.raw_weight, 3.0);
        assert!(stats.decayed_weight < 3.0);
        assert!(stats.decay_ratio < 1.0);
        assert!(stats.newest_weight > stats.oldest_weight);
    }
}
