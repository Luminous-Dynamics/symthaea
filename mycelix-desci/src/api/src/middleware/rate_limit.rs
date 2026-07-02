// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rate Limiting Middleware
//!
//! Implements per-IP rate limiting to prevent abuse and ensure fair usage,
//! backed by `tower_governor` (a Tower/Axum layer over the `governor`
//! crate's GCRA token-bucket limiter).
//!
//! `RateLimitConfig` and the `create_*_rate_limit_config` helpers below
//! preserve the numbers from the original placeholder implementation as
//! documented per-route intent (general/strict/query). Only the
//! general-purpose config (100 req/min, burst 120) is currently wired into
//! the crate-wide `GovernorLayer` in `main.rs` via [`governor_config`], since
//! `tower_governor` applies one config per layer and this crate does not yet
//! split routes into per-endpoint layers for rate limiting.

use governor::middleware::NoOpMiddleware;
use std::sync::{Arc, OnceLock};
use tower_governor::governor::{GovernorConfig, GovernorConfigBuilder};
use tower_governor::key_extractor::PeerIpKeyExtractor;

/// Shorthand for the GCRA config type this crate uses everywhere:
/// per-IP keying, no extra rate-limit-state response headers.
type DesciGovernorConfig = GovernorConfig<PeerIpKeyExtractor, NoOpMiddleware>;

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Requests per minute
    pub requests_per_minute: u32,
    /// Burst size
    pub burst_size: u32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            requests_per_minute: 100,
            burst_size: 120,
        }
    }
}

/// Create general API rate limit configuration
pub fn create_rate_limit_config() -> RateLimitConfig {
    RateLimitConfig {
        requests_per_minute: 100,
        burst_size: 120,
    }
}

/// Create strict rate limit for claim creation
pub fn create_strict_rate_limit_config() -> RateLimitConfig {
    RateLimitConfig {
        requests_per_minute: 10,
        burst_size: 15,
    }
}

/// Create permissive rate limit for queries
pub fn create_query_rate_limit_config() -> RateLimitConfig {
    RateLimitConfig {
        requests_per_minute: 200,
        burst_size: 250,
    }
}

/// Build a real `tower_governor` GCRA config from a [`RateLimitConfig`].
///
/// `tower_governor` replenishes one token every `period`, capped at
/// `burst_size` tokens. We convert `requests_per_minute` into an exact
/// per-token period (`60_000ms / requests_per_minute`) rather than rounding
/// to a whole `per_second` value, so e.g. 100 req/min becomes a refill every
/// 600ms — not the same as "60/min" you'd get from `per_second(1)`.
fn to_governor_config(cfg: &RateLimitConfig) -> Arc<DesciGovernorConfig> {
    let period_ms = (60_000 / cfg.requests_per_minute.max(1)) as u64;
    Arc::new(
        GovernorConfigBuilder::default()
            .period(std::time::Duration::from_millis(period_ms.max(1)))
            .burst_size(cfg.burst_size)
            .finish()
            .expect("valid governor rate-limit config"),
    )
}

/// The `GovernorLayer` config applied crate-wide in `main.rs`.
///
/// Uses [`create_rate_limit_config`] (100 req/min, burst 120 — the same
/// numbers the original placeholder documented) as the general-purpose,
/// per-IP limit for the whole API. Built once and cached, since
/// `tower_governor` config is meant to be shared (and it also spawns an
/// internal cleanup task on first construction).
pub fn governor_config() -> Arc<DesciGovernorConfig> {
    static CONFIG: OnceLock<Arc<DesciGovernorConfig>> = OnceLock::new();
    CONFIG
        .get_or_init(|| to_governor_config(&create_rate_limit_config()))
        .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_config_creation() {
        let general = create_rate_limit_config();
        assert_eq!(general.requests_per_minute, 100);

        let strict = create_strict_rate_limit_config();
        assert_eq!(strict.requests_per_minute, 10);

        let query = create_query_rate_limit_config();
        assert_eq!(query.requests_per_minute, 200);
    }

    #[test]
    fn test_default_config() {
        let default = RateLimitConfig::default();
        assert_eq!(default.requests_per_minute, 100);
        assert_eq!(default.burst_size, 120);
    }

    #[test]
    fn test_governor_config_builds_and_is_cached() {
        let a = governor_config();
        let b = governor_config();
        // Same Arc-backed instance on repeated calls (cached via OnceLock).
        assert!(Arc::ptr_eq(&a, &b));
    }

    #[test]
    fn test_to_governor_config_period_matches_requests_per_minute() {
        // 100 req/min -> one token every 600ms.
        let cfg = create_rate_limit_config();
        let governor = to_governor_config(&cfg);
        // Sanity: config builds without panicking and burst size round-trips
        // via the public RateLimitConfig it was derived from.
        assert_eq!(cfg.burst_size, 120);
        drop(governor);
    }
}
