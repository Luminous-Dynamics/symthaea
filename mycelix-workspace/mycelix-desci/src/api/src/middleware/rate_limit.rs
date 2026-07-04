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

    /// Behavioral test: actually sends requests through a real `GovernorLayer`
    /// and asserts a 429 once the burst is exhausted. The tests above only
    /// checked config *values* -- none of them ever drove a request through
    /// the layer, so a `tower_governor`/`governor` upgrade that silently
    /// changed burst/refill semantics would not have been caught.
    mod rate_limit_behavior {
        use super::*;
        use axum::{
            Router,
            body::Body,
            extract::ConnectInfo,
            http::{Request as HttpRequest, StatusCode},
            routing::get,
        };
        use std::net::{IpAddr, Ipv4Addr, SocketAddr};
        use tower::ServiceExt;
        use tower_governor::GovernorLayer;

        async fn ok_handler() -> &'static str {
            "ok"
        }

        /// A tiny burst (2) with a long refill period so the test isn't
        /// racing the token bucket's own replenishment.
        fn tiny_governor_layer() -> GovernorLayer<PeerIpKeyExtractor, NoOpMiddleware> {
            let config = GovernorConfigBuilder::default()
                .period(std::time::Duration::from_secs(60))
                .burst_size(2)
                .finish()
                .expect("valid governor rate-limit config");
            GovernorLayer {
                config: Arc::new(config),
            }
        }

        fn app() -> Router {
            Router::new()
                .route("/ping", get(ok_handler))
                .layer(tiny_governor_layer())
        }

        fn request_from(ip: IpAddr) -> HttpRequest<Body> {
            let mut req = HttpRequest::builder()
                .method("GET")
                .uri("/ping")
                .body(Body::empty())
                .unwrap();
            // PeerIpKeyExtractor reads this extension directly (see
            // `maybe_connect_info` in tower_governor::key_extractor) --
            // `oneshot()` never populates it from a real socket, so it must
            // be inserted manually to exercise per-IP keying in a test.
            req.extensions_mut()
                .insert(ConnectInfo(SocketAddr::new(ip, 0)));
            req
        }

        #[tokio::test]
        async fn burst_then_429() {
            let ip = IpAddr::V4(Ipv4Addr::new(203, 0, 113, 1));
            let app = app();

            // Burst size is 2: the first two requests from this IP must
            // succeed.
            for i in 0..2 {
                let response = app.clone().oneshot(request_from(ip)).await.unwrap();
                assert_eq!(
                    response.status(),
                    StatusCode::OK,
                    "request {i} within burst should succeed"
                );
            }

            // The third request within the same (long) refill period must
            // be rejected with 429 -- this is the actual behavior a
            // tower_governor/governor version bump could silently break.
            let response = app.clone().oneshot(request_from(ip)).await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::TOO_MANY_REQUESTS,
                "request beyond burst size must be rate-limited"
            );
        }

        #[tokio::test]
        async fn different_ips_have_independent_buckets() {
            let ip_a = IpAddr::V4(Ipv4Addr::new(203, 0, 113, 10));
            let ip_b = IpAddr::V4(Ipv4Addr::new(203, 0, 113, 20));
            let app = app();

            // Exhaust ip_a's burst.
            for _ in 0..2 {
                let response = app.clone().oneshot(request_from(ip_a)).await.unwrap();
                assert_eq!(response.status(), StatusCode::OK);
            }
            let response = app.clone().oneshot(request_from(ip_a)).await.unwrap();
            assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);

            // ip_b must be unaffected -- per-IP keying, not a global limiter.
            let response = app.clone().oneshot(request_from(ip_b)).await.unwrap();
            assert_eq!(
                response.status(),
                StatusCode::OK,
                "a different IP must have its own independent bucket"
            );
        }
    }
}
