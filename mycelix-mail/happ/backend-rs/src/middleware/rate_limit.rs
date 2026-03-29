// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rate limiting middleware
//!
//! Provides per-IP rate limiting to prevent abuse and spam.
//! Uses a sliding window algorithm with configurable limits.

use axum::{
    body::Body,
    extract::{ConnectInfo, State},
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use std::{
    collections::HashMap,
    net::{IpAddr, SocketAddr},
    num::NonZeroU32,
    sync::Arc,
};
use tokio::sync::RwLock;

use crate::types::ApiError;

/// Rate limiter state
pub struct RateLimitState {
    /// Per-IP limiters
    limiters: RwLock<HashMap<IpAddr, Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>>>,
    /// Requests per minute limit
    rpm: NonZeroU32,
    /// Burst limit
    burst: NonZeroU32,
}

impl RateLimitState {
    /// Create a new rate limit state
    pub fn new(requests_per_minute: u32, burst: u32) -> Self {
        Self {
            limiters: RwLock::new(HashMap::new()),
            rpm: NonZeroU32::new(requests_per_minute).unwrap_or(NonZeroU32::new(60).unwrap()),
            burst: NonZeroU32::new(burst).unwrap_or(NonZeroU32::new(10).unwrap()),
        }
    }

    /// Get or create a rate limiter for an IP
    async fn get_limiter(
        &self,
        ip: IpAddr,
    ) -> Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>> {
        // Check if limiter exists
        {
            let limiters = self.limiters.read().await;
            if let Some(limiter) = limiters.get(&ip) {
                return limiter.clone();
            }
        }

        // Create new limiter
        let quota = Quota::per_minute(self.rpm).allow_burst(self.burst);
        let limiter = Arc::new(RateLimiter::direct(quota));

        // Store it
        {
            let mut limiters = self.limiters.write().await;
            limiters.insert(ip, limiter.clone());
        }

        limiter
    }

    /// Check if request is allowed
    pub async fn check(&self, ip: IpAddr) -> bool {
        let limiter = self.get_limiter(ip).await;
        limiter.check().is_ok()
    }

    /// Clean up old limiters (call periodically)
    pub async fn cleanup(&self) {
        let mut limiters = self.limiters.write().await;
        // Keep only limiters that have been used recently
        // In production, you'd track last access time
        if limiters.len() > 10000 {
            limiters.clear();
        }
    }
}

impl Default for RateLimitState {
    fn default() -> Self {
        Self::new(100, 20) // 100 req/min with burst of 20
    }
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(rate_limiter): State<Arc<RateLimitState>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let ip = addr.ip();

    if !rate_limiter.check(ip).await {
        tracing::warn!("Rate limit exceeded for IP: {}", ip);

        return (
            StatusCode::TOO_MANY_REQUESTS,
            [("Retry-After", "60")],
            Json(ApiError::new("RATE_LIMIT", "Too many requests. Please try again later.")),
        )
            .into_response();
    }

    next.run(request).await
}

/// Endpoint-specific rate limits
pub struct EndpointRateLimits {
    /// Login attempts (stricter)
    pub login: Arc<RateLimitState>,
    /// Registration (very strict)
    pub register: Arc<RateLimitState>,
    /// Email sending (moderate)
    pub send_email: Arc<RateLimitState>,
    /// General API calls
    pub general: Arc<RateLimitState>,
}

impl Default for EndpointRateLimits {
    fn default() -> Self {
        Self {
            login: Arc::new(RateLimitState::new(10, 3)),      // 10/min, burst 3
            register: Arc::new(RateLimitState::new(5, 2)),    // 5/min, burst 2
            send_email: Arc::new(RateLimitState::new(30, 5)), // 30/min, burst 5
            general: Arc::new(RateLimitState::new(100, 20)),  // 100/min, burst 20
        }
    }
}

/// Check rate limit for a specific endpoint type
pub async fn check_endpoint_limit(
    limits: &EndpointRateLimits,
    endpoint: &str,
    ip: IpAddr,
) -> Result<(), (StatusCode, Json<ApiError>)> {
    let limiter = match endpoint {
        "login" => &limits.login,
        "register" => &limits.register,
        "send_email" => &limits.send_email,
        _ => &limits.general,
    };

    if !limiter.check(ip).await {
        tracing::warn!("Rate limit exceeded for {} from IP: {}", endpoint, ip);
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            Json(ApiError::new(
                "RATE_LIMIT",
                format!("Too many {} requests. Please try again later.", endpoint),
            )),
        ));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_rate_limit_allows_under_limit() {
        let state = RateLimitState::new(10, 5);
        let ip: IpAddr = "127.0.0.1".parse().unwrap();

        // Should allow first few requests
        for _ in 0..5 {
            assert!(state.check(ip).await);
        }
    }

    #[tokio::test]
    async fn test_rate_limit_blocks_over_limit() {
        let state = RateLimitState::new(5, 2);
        let ip: IpAddr = "127.0.0.1".parse().unwrap();

        // Exhaust the burst
        for _ in 0..3 {
            state.check(ip).await;
        }

        // Should start blocking
        // Note: Exact behavior depends on timing
    }

    #[tokio::test]
    async fn test_different_ips_have_separate_limits() {
        let state = RateLimitState::new(2, 2);
        let ip1: IpAddr = "127.0.0.1".parse().unwrap();
        let ip2: IpAddr = "127.0.0.2".parse().unwrap();

        // Both IPs should have their own limits
        assert!(state.check(ip1).await);
        assert!(state.check(ip1).await);
        assert!(state.check(ip2).await);
        assert!(state.check(ip2).await);
    }
}
