// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rate limiter implementation

use crate::{
    config::{
        FixedWindowConfig, LeakyBucketConfig, RateLimitConfig,
        SlidingWindowConfig, TokenBucketConfig, TrustMultiplier,
    },
    lua_scripts, RateLimitError, KEY_PREFIX,
};
use redis::{aio::ConnectionManager, AsyncCommands, Script};
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::{debug, instrument, warn};

/// Result of a rate limit check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitResult {
    /// Whether the request is allowed
    pub allowed: bool,
    /// Remaining requests in current window
    pub remaining: u64,
    /// Time to wait before retrying (if limited)
    pub retry_after: Duration,
    /// Current usage count
    pub current_usage: u64,
    /// The limit that was applied
    pub limit: u64,
}

impl RateLimitResult {
    /// Create an allowed result
    pub fn allowed(remaining: u64, current_usage: u64, limit: u64) -> Self {
        Self {
            allowed: true,
            remaining,
            retry_after: Duration::ZERO,
            current_usage,
            limit,
        }
    }

    /// Create a limited result
    pub fn limited(remaining: u64, retry_after: Duration, current_usage: u64, limit: u64) -> Self {
        Self {
            allowed: false,
            remaining,
            retry_after,
            current_usage,
            limit,
        }
    }
}

/// Distributed rate limiter using Redis
pub struct RateLimiter {
    conn: ConnectionManager,
    key_prefix: String,
    trust_tiers: Vec<TrustMultiplier>,
}

impl RateLimiter {
    /// Create a new rate limiter
    ///
    /// # Arguments
    /// * `redis_url` - Redis connection URL (e.g., "redis://localhost:6379")
    #[instrument(skip_all, fields(url = %redis_url))]
    pub async fn new(redis_url: &str) -> Result<Self, RateLimitError> {
        let client = redis::Client::open(redis_url)
            .map_err(|e| RateLimitError::ConnectionError(e.to_string()))?;

        let conn = ConnectionManager::new(client)
            .await
            .map_err(|e| RateLimitError::ConnectionError(e.to_string()))?;

        debug!("Connected to Redis for rate limiting");

        Ok(Self {
            conn,
            key_prefix: KEY_PREFIX.to_string(),
            trust_tiers: TrustMultiplier::default_tiers(),
        })
    }

    /// Create with custom key prefix
    pub fn with_prefix(mut self, prefix: &str) -> Self {
        self.key_prefix = prefix.to_string();
        self
    }

    /// Set custom trust tiers
    pub fn with_trust_tiers(mut self, tiers: Vec<TrustMultiplier>) -> Self {
        self.trust_tiers = tiers;
        self
    }

    /// Build the full Redis key
    fn build_key(&self, identifier: &str, config: &impl RateLimitConfig) -> String {
        format!("{}{}:{}", self.key_prefix, config.algorithm(), identifier)
    }

    /// Get current timestamp in milliseconds
    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock before UNIX epoch")
            .as_millis() as u64
    }

    /// Check rate limit using sliding window algorithm
    #[instrument(skip(self, config), fields(id = %identifier))]
    pub async fn check_sliding_window(
        &self,
        identifier: &str,
        config: &SlidingWindowConfig,
    ) -> Result<RateLimitResult, RateLimitError> {
        self.check_sliding_window_with_cost(identifier, config, 1).await
    }

    /// Check rate limit with custom cost
    pub async fn check_sliding_window_with_cost(
        &self,
        identifier: &str,
        config: &SlidingWindowConfig,
        cost: u64,
    ) -> Result<RateLimitResult, RateLimitError> {
        let key = self.build_key(identifier, config);
        let now_ms = Self::now_ms();
        let window_ms = config.window().as_millis() as u64;

        let script = Script::new(lua_scripts::SLIDING_WINDOW_COUNTER);
        let mut conn = self.conn.clone();

        let result: Vec<i64> = script
            .key(&key)
            .arg(config.max_requests)
            .arg(window_ms)
            .arg(now_ms)
            .arg(cost)
            .invoke_async(&mut conn)
            .await?;

        let allowed = result[0] == 1;
        let remaining = result[1] as u64;
        let retry_after_ms = result[2] as u64;
        let current_usage = result[3] as u64;

        debug!(
            allowed = allowed,
            remaining = remaining,
            current_usage = current_usage,
            "Sliding window check"
        );

        Ok(if allowed {
            RateLimitResult::allowed(remaining, current_usage, config.max_requests)
        } else {
            RateLimitResult::limited(
                remaining,
                Duration::from_millis(retry_after_ms),
                current_usage,
                config.max_requests,
            )
        })
    }

    /// Check rate limit using token bucket algorithm
    #[instrument(skip(self, config), fields(id = %identifier))]
    pub async fn check_token_bucket(
        &self,
        identifier: &str,
        config: &TokenBucketConfig,
    ) -> Result<RateLimitResult, RateLimitError> {
        self.check_token_bucket_with_cost(identifier, config, 1).await
    }

    /// Check token bucket with custom cost
    pub async fn check_token_bucket_with_cost(
        &self,
        identifier: &str,
        config: &TokenBucketConfig,
        cost: u64,
    ) -> Result<RateLimitResult, RateLimitError> {
        let key = self.build_key(identifier, config);
        let now_ms = Self::now_ms();

        let script = Script::new(lua_scripts::TOKEN_BUCKET);
        let mut conn = self.conn.clone();

        let result: Vec<i64> = script
            .key(&key)
            .arg(config.capacity)
            .arg(config.refill_rate)
            .arg(now_ms)
            .arg(cost)
            .invoke_async(&mut conn)
            .await?;

        let allowed = result[0] == 1;
        let remaining = result[1] as u64;
        let retry_after_ms = result[2] as u64;
        let current_usage = result[3] as u64;

        debug!(
            allowed = allowed,
            remaining = remaining,
            "Token bucket check"
        );

        Ok(if allowed {
            RateLimitResult::allowed(remaining, current_usage, config.capacity)
        } else {
            RateLimitResult::limited(
                remaining,
                Duration::from_millis(retry_after_ms),
                current_usage,
                config.capacity,
            )
        })
    }

    /// Check rate limit using fixed window algorithm
    #[instrument(skip(self, config), fields(id = %identifier))]
    pub async fn check_fixed_window(
        &self,
        identifier: &str,
        config: &FixedWindowConfig,
    ) -> Result<RateLimitResult, RateLimitError> {
        let key = self.build_key(identifier, config);
        let now_ms = Self::now_ms();
        let window_ms = config.window().as_millis() as u64;

        let script = Script::new(lua_scripts::FIXED_WINDOW);
        let mut conn = self.conn.clone();

        let result: Vec<i64> = script
            .key(&key)
            .arg(config.max_requests)
            .arg(window_ms)
            .arg(now_ms)
            .arg(1_u64)
            .invoke_async(&mut conn)
            .await?;

        let allowed = result[0] == 1;
        let remaining = result[1] as u64;
        let retry_after_ms = result[2] as u64;
        let current_usage = result[3] as u64;

        debug!(
            allowed = allowed,
            remaining = remaining,
            "Fixed window check"
        );

        Ok(if allowed {
            RateLimitResult::allowed(remaining, current_usage, config.max_requests)
        } else {
            RateLimitResult::limited(
                remaining,
                Duration::from_millis(retry_after_ms),
                current_usage,
                config.max_requests,
            )
        })
    }

    /// Check rate limit using leaky bucket algorithm
    #[instrument(skip(self, config), fields(id = %identifier))]
    pub async fn check_leaky_bucket(
        &self,
        identifier: &str,
        config: &LeakyBucketConfig,
    ) -> Result<RateLimitResult, RateLimitError> {
        let key = self.build_key(identifier, config);
        let now_ms = Self::now_ms();

        let script = Script::new(lua_scripts::LEAKY_BUCKET);
        let mut conn = self.conn.clone();

        let result: Vec<i64> = script
            .key(&key)
            .arg(config.capacity)
            .arg(config.leak_rate)
            .arg(now_ms)
            .arg(1_u64)
            .invoke_async(&mut conn)
            .await?;

        let allowed = result[0] == 1;
        let remaining = result[1] as u64;
        let retry_after_ms = result[2] as u64;
        let current_usage = result[3] as u64;

        debug!(
            allowed = allowed,
            remaining = remaining,
            "Leaky bucket check"
        );

        Ok(if allowed {
            RateLimitResult::allowed(remaining, current_usage, config.capacity)
        } else {
            RateLimitResult::limited(
                remaining,
                Duration::from_millis(retry_after_ms),
                current_usage,
                config.capacity,
            )
        })
    }

    /// Check rate limit with trust-based multiplier
    ///
    /// Adjusts the effective limit based on the client's trust score.
    pub async fn check_with_trust(
        &self,
        identifier: &str,
        config: &SlidingWindowConfig,
        trust_score: f64,
    ) -> Result<RateLimitResult, RateLimitError> {
        let multiplier = TrustMultiplier::get_multiplier(&self.trust_tiers, trust_score);
        let adjusted_limit = (config.max_requests as f64 * multiplier) as u64;

        let adjusted_config = SlidingWindowConfig {
            max_requests: adjusted_limit,
            window_secs: config.window_secs,
            precision: config.precision,
        };

        debug!(
            trust_score = trust_score,
            multiplier = multiplier,
            adjusted_limit = adjusted_limit,
            "Applying trust-based rate limit"
        );

        self.check_sliding_window(identifier, &adjusted_config).await
    }

    /// Reset rate limit for an identifier
    #[instrument(skip(self), fields(id = %identifier))]
    pub async fn reset(&self, identifier: &str, algorithm: &str) -> Result<u64, RateLimitError> {
        let key = format!("{}{}:{}", self.key_prefix, algorithm, identifier);

        let script = Script::new(lua_scripts::RESET);
        let mut conn = self.conn.clone();

        let deleted: i64 = script.key(&key).invoke_async(&mut conn).await?;

        debug!(deleted = deleted, "Reset rate limit");

        Ok(deleted as u64)
    }

    /// Get current rate limit info without consuming
    pub async fn get_info(
        &self,
        identifier: &str,
        config: &impl RateLimitConfig,
    ) -> Result<(u64, u64), RateLimitError> {
        let key = self.build_key(identifier, config);
        let now_ms = Self::now_ms();
        let window_ms = config.window().as_millis() as u64;

        let script = Script::new(lua_scripts::GET_INFO);
        let mut conn = self.conn.clone();

        let result: Vec<i64> = script
            .key(&key)
            .arg(config.algorithm())
            .arg(config.max_requests())
            .arg(window_ms)
            .arg(now_ms)
            .invoke_async(&mut conn)
            .await?;

        Ok((result[0] as u64, result[1] as u64))
    }

    /// Check if Redis connection is healthy
    pub async fn health_check(&self) -> Result<bool, RateLimitError> {
        let mut conn = self.conn.clone();
        let pong: String = redis::cmd("PING").query_async(&mut conn).await?;
        Ok(pong == "PONG")
    }
}

/// Builder for creating rate limiters with custom configuration
pub struct RateLimiterBuilder {
    redis_url: String,
    key_prefix: String,
    trust_tiers: Vec<TrustMultiplier>,
}

impl RateLimiterBuilder {
    /// Create a new builder
    pub fn new(redis_url: &str) -> Self {
        Self {
            redis_url: redis_url.to_string(),
            key_prefix: KEY_PREFIX.to_string(),
            trust_tiers: TrustMultiplier::default_tiers(),
        }
    }

    /// Set custom key prefix
    pub fn prefix(mut self, prefix: &str) -> Self {
        self.key_prefix = prefix.to_string();
        self
    }

    /// Set custom trust tiers
    pub fn trust_tiers(mut self, tiers: Vec<TrustMultiplier>) -> Self {
        self.trust_tiers = tiers;
        self
    }

    /// Build the rate limiter
    pub async fn build(self) -> Result<RateLimiter, RateLimitError> {
        let limiter = RateLimiter::new(&self.redis_url).await?;
        Ok(limiter
            .with_prefix(&self.key_prefix)
            .with_trust_tiers(self.trust_tiers))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_result_allowed() {
        let result = RateLimitResult::allowed(99, 1, 100);
        assert!(result.allowed);
        assert_eq!(result.remaining, 99);
        assert_eq!(result.current_usage, 1);
        assert_eq!(result.retry_after, Duration::ZERO);
    }

    #[test]
    fn test_rate_limit_result_limited() {
        let result = RateLimitResult::limited(0, Duration::from_secs(30), 100, 100);
        assert!(!result.allowed);
        assert_eq!(result.remaining, 0);
        assert_eq!(result.current_usage, 100);
        assert_eq!(result.retry_after, Duration::from_secs(30));
    }

    #[test]
    fn test_builder_defaults() {
        let builder = RateLimiterBuilder::new("redis://localhost:6379");
        assert_eq!(builder.key_prefix, KEY_PREFIX);
        assert!(!builder.trust_tiers.is_empty());
    }
}
