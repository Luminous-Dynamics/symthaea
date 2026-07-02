// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rate limiting middleware using token bucket algorithm
//!
//! Provides configurable per-endpoint rate limiting with burst support

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use serde_json::json;
use std::num::NonZeroU32;
use std::sync::Arc;

/// Rate limiter configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Requests per second allowed
    pub requests_per_second: NonZeroU32,
    /// Burst size (allow temporary spikes)
    pub burst_size: NonZeroU32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            // 100 requests per second
            requests_per_second: NonZeroU32::new(100).unwrap(),
            // Allow bursts of 20 requests
            burst_size: NonZeroU32::new(20).unwrap(),
        }
    }
}

impl RateLimitConfig {
    /// Create from environment variables
    pub fn from_env() -> Self {
        let rps = std::env::var("RATE_LIMIT_RPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .and_then(NonZeroU32::new)
            .unwrap_or_else(|| NonZeroU32::new(100).unwrap());

        let burst = std::env::var("RATE_LIMIT_BURST")
            .ok()
            .and_then(|s| s.parse().ok())
            .and_then(NonZeroU32::new)
            .unwrap_or_else(|| NonZeroU32::new(20).unwrap());

        Self {
            requests_per_second: rps,
            burst_size: burst,
        }
    }
}

/// Global rate limiter state
pub type GlobalRateLimiter = Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>;

/// Create a rate limiter with the given configuration
pub fn create_rate_limiter(config: RateLimitConfig) -> GlobalRateLimiter {
    let quota = Quota::per_second(config.requests_per_second)
        .allow_burst(config.burst_size);

    Arc::new(RateLimiter::direct(quota))
}

/// Rate limiting middleware
///
/// Returns 429 Too Many Requests when limit exceeded
pub async fn rate_limit_middleware(
    State(limiter): State<GlobalRateLimiter>,
    req: Request,
    next: Next,
) -> Response {
    match limiter.check() {
        Ok(_) => {
            // Request allowed, proceed
            next.run(req).await
        }
        Err(_) => {
            // Rate limit exceeded
            tracing::warn!("Rate limit exceeded");

            (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": 1
                }))
            ).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_creation() {
        let config = RateLimitConfig::default();
        let limiter = create_rate_limiter(config);

        // Should allow first request
        assert!(limiter.check().is_ok());
    }

    #[test]
    fn test_rate_limiter_burst() {
        let config = RateLimitConfig {
            requests_per_second: NonZeroU32::new(10).unwrap(),
            burst_size: NonZeroU32::new(5).unwrap(),
        };
        let limiter = create_rate_limiter(config);

        // Should allow burst
        for _ in 0..5 {
            assert!(limiter.check().is_ok());
        }
    }

    #[test]
    fn test_config_from_env() {
        std::env::set_var("RATE_LIMIT_RPS", "50");
        std::env::set_var("RATE_LIMIT_BURST", "10");

        let config = RateLimitConfig::from_env();
        assert_eq!(config.requests_per_second.get(), 50);
        assert_eq!(config.burst_size.get(), 10);

        std::env::remove_var("RATE_LIMIT_RPS");
        std::env::remove_var("RATE_LIMIT_BURST");
    }

    #[test]
    fn test_default_config() {
        let config = RateLimitConfig::default();
        assert_eq!(config.requests_per_second.get(), 100);
        assert_eq!(config.burst_size.get(), 20);
    }
}
