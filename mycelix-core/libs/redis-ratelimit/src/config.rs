// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rate limiter configuration

use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Base trait for rate limit configurations
pub trait RateLimitConfig: Send + Sync {
    /// Get the maximum requests allowed
    fn max_requests(&self) -> u64;

    /// Get the time window duration
    fn window(&self) -> Duration;

    /// Get the algorithm name
    fn algorithm(&self) -> &'static str;

    /// Generate the Redis key suffix for this config
    fn key_suffix(&self) -> String {
        format!("{}:{}", self.algorithm(), self.window().as_secs())
    }
}

/// Sliding window rate limiter configuration
///
/// Provides smooth rate limiting by dividing the window into sub-windows
/// and using weighted counts from the previous window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SlidingWindowConfig {
    /// Maximum requests per window
    pub max_requests: u64,
    /// Window duration
    pub window_secs: u64,
    /// Number of sub-windows for precision (default: 10)
    pub precision: u32,
}

impl SlidingWindowConfig {
    /// Create a new sliding window config
    pub fn new(max_requests: u64, window: Duration) -> Self {
        Self {
            max_requests,
            window_secs: window.as_secs(),
            precision: 10,
        }
    }

    /// Set the precision (number of sub-windows)
    pub fn with_precision(mut self, precision: u32) -> Self {
        self.precision = precision.max(1);
        self
    }

    /// Get sub-window duration in milliseconds
    pub fn sub_window_ms(&self) -> u64 {
        (self.window_secs * 1000) / self.precision as u64
    }
}

impl RateLimitConfig for SlidingWindowConfig {
    fn max_requests(&self) -> u64 {
        self.max_requests
    }

    fn window(&self) -> Duration {
        Duration::from_secs(self.window_secs)
    }

    fn algorithm(&self) -> &'static str {
        "sliding_window"
    }
}

/// Token bucket rate limiter configuration
///
/// Allows bursts up to bucket capacity while maintaining
/// a steady average rate through token replenishment.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBucketConfig {
    /// Maximum tokens (bucket capacity)
    pub capacity: u64,
    /// Tokens added per second
    pub refill_rate: f64,
    /// Initial tokens (defaults to capacity)
    pub initial_tokens: Option<u64>,
}

impl TokenBucketConfig {
    /// Create a new token bucket config
    ///
    /// # Arguments
    /// * `capacity` - Maximum tokens in the bucket
    /// * `refill_rate` - Tokens added per second
    pub fn new(capacity: u64, refill_rate: f64) -> Self {
        Self {
            capacity,
            refill_rate,
            initial_tokens: None,
        }
    }

    /// Create from requests per window
    pub fn from_rate(max_requests: u64, window: Duration) -> Self {
        let refill_rate = max_requests as f64 / window.as_secs_f64();
        Self::new(max_requests, refill_rate)
    }

    /// Set initial tokens
    pub fn with_initial_tokens(mut self, tokens: u64) -> Self {
        self.initial_tokens = Some(tokens.min(self.capacity));
        self
    }

    /// Get initial tokens (defaults to capacity)
    pub fn get_initial_tokens(&self) -> u64 {
        self.initial_tokens.unwrap_or(self.capacity)
    }
}

impl RateLimitConfig for TokenBucketConfig {
    fn max_requests(&self) -> u64 {
        self.capacity
    }

    fn window(&self) -> Duration {
        // Effective window based on refill rate
        Duration::from_secs_f64(self.capacity as f64 / self.refill_rate)
    }

    fn algorithm(&self) -> &'static str {
        "token_bucket"
    }
}

/// Fixed window rate limiter configuration
///
/// Simple counter-based limiting that resets at window boundaries.
/// May allow up to 2x burst at window boundaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixedWindowConfig {
    /// Maximum requests per window
    pub max_requests: u64,
    /// Window duration in seconds
    pub window_secs: u64,
}

impl FixedWindowConfig {
    /// Create a new fixed window config
    pub fn new(max_requests: u64, window: Duration) -> Self {
        Self {
            max_requests,
            window_secs: window.as_secs(),
        }
    }

    /// Get the current window start timestamp
    pub fn current_window_start(&self) -> u64 {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_secs();
        (now / self.window_secs) * self.window_secs
    }
}

impl RateLimitConfig for FixedWindowConfig {
    fn max_requests(&self) -> u64 {
        self.max_requests
    }

    fn window(&self) -> Duration {
        Duration::from_secs(self.window_secs)
    }

    fn algorithm(&self) -> &'static str {
        "fixed_window"
    }
}

/// Leaky bucket rate limiter configuration
///
/// Provides constant output rate regardless of input pattern.
/// Good for protecting downstream services from bursts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeakyBucketConfig {
    /// Bucket capacity (queue size)
    pub capacity: u64,
    /// Leak rate (requests processed per second)
    pub leak_rate: f64,
}

impl LeakyBucketConfig {
    /// Create a new leaky bucket config
    pub fn new(capacity: u64, leak_rate: f64) -> Self {
        Self { capacity, leak_rate }
    }

    /// Create from max requests and window
    pub fn from_rate(max_requests: u64, window: Duration) -> Self {
        let leak_rate = max_requests as f64 / window.as_secs_f64();
        Self::new(max_requests, leak_rate)
    }

    /// Time to process one request in milliseconds
    pub fn leak_interval_ms(&self) -> u64 {
        (1000.0 / self.leak_rate) as u64
    }
}

impl RateLimitConfig for LeakyBucketConfig {
    fn max_requests(&self) -> u64 {
        self.capacity
    }

    fn window(&self) -> Duration {
        Duration::from_secs_f64(self.capacity as f64 / self.leak_rate)
    }

    fn algorithm(&self) -> &'static str {
        "leaky_bucket"
    }
}

/// Trust-based rate limit multipliers for FL clients
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustMultiplier {
    /// Trust score threshold
    pub min_trust: f64,
    /// Rate limit multiplier (1.0 = normal, 2.0 = double limit)
    pub multiplier: f64,
}

impl TrustMultiplier {
    /// Create default trust tiers
    pub fn default_tiers() -> Vec<Self> {
        vec![
            Self { min_trust: 0.9, multiplier: 2.0 },  // Highly trusted: 2x limit
            Self { min_trust: 0.7, multiplier: 1.5 },  // Trusted: 1.5x limit
            Self { min_trust: 0.5, multiplier: 1.0 },  // Normal: standard limit
            Self { min_trust: 0.3, multiplier: 0.5 },  // Suspicious: half limit
            Self { min_trust: 0.0, multiplier: 0.25 }, // Untrusted: quarter limit
        ]
    }

    /// Get multiplier for a given trust score
    pub fn get_multiplier(tiers: &[Self], trust_score: f64) -> f64 {
        for tier in tiers {
            if trust_score >= tier.min_trust {
                return tier.multiplier;
            }
        }
        0.25 // Default to minimum for unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sliding_window_config() {
        let config = SlidingWindowConfig::new(100, Duration::from_secs(60));
        assert_eq!(config.max_requests(), 100);
        assert_eq!(config.window(), Duration::from_secs(60));
        assert_eq!(config.algorithm(), "sliding_window");
        assert_eq!(config.sub_window_ms(), 6000);
    }

    #[test]
    fn test_sliding_window_precision() {
        let config = SlidingWindowConfig::new(100, Duration::from_secs(60))
            .with_precision(20);
        assert_eq!(config.precision, 20);
        assert_eq!(config.sub_window_ms(), 3000);
    }

    #[test]
    fn test_token_bucket_config() {
        let config = TokenBucketConfig::new(100, 10.0);
        assert_eq!(config.capacity, 100);
        assert_eq!(config.refill_rate, 10.0);
        assert_eq!(config.get_initial_tokens(), 100);
    }

    #[test]
    fn test_token_bucket_from_rate() {
        let config = TokenBucketConfig::from_rate(60, Duration::from_secs(60));
        assert_eq!(config.capacity, 60);
        assert!((config.refill_rate - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_fixed_window_config() {
        let config = FixedWindowConfig::new(100, Duration::from_secs(60));
        assert_eq!(config.max_requests(), 100);
        assert_eq!(config.window_secs, 60);
    }

    #[test]
    fn test_leaky_bucket_config() {
        let config = LeakyBucketConfig::new(100, 10.0);
        assert_eq!(config.capacity, 100);
        assert_eq!(config.leak_rate, 10.0);
        assert_eq!(config.leak_interval_ms(), 100);
    }

    #[test]
    fn test_trust_multiplier() {
        let tiers = TrustMultiplier::default_tiers();

        assert!((TrustMultiplier::get_multiplier(&tiers, 0.95) - 2.0).abs() < 0.001);
        assert!((TrustMultiplier::get_multiplier(&tiers, 0.75) - 1.5).abs() < 0.001);
        assert!((TrustMultiplier::get_multiplier(&tiers, 0.55) - 1.0).abs() < 0.001);
        assert!((TrustMultiplier::get_multiplier(&tiers, 0.35) - 0.5).abs() < 0.001);
        assert!((TrustMultiplier::get_multiplier(&tiers, 0.15) - 0.25).abs() < 0.001);
    }
}
