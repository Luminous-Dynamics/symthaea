// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Security Middleware for Mycelix Mail API
//!
//! Comprehensive security layers including:
//! - Rate limiting
//! - CORS configuration
//! - Security headers (CSP, HSTS, etc.)
//! - Request validation
//! - IP blocking
//! - Audit logging

use axum::{
    body::Body,
    extract::{ConnectInfo, State},
    http::{header, HeaderMap, HeaderValue, Method, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use dashmap::DashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn};

// ============================================================================
// Rate Limiting
// ============================================================================

#[derive(Clone)]
pub struct RateLimiter {
    /// Requests per window
    limit: u32,
    /// Window duration
    window: Duration,
    /// Request counts by IP
    requests: Arc<DashMap<String, (u32, Instant)>>,
}

impl RateLimiter {
    pub fn new(limit: u32, window_secs: u64) -> Self {
        Self {
            limit,
            window: Duration::from_secs(window_secs),
            requests: Arc::new(DashMap::new()),
        }
    }

    pub fn check(&self, key: &str) -> Result<(), RateLimitError> {
        let now = Instant::now();

        let mut entry = self.requests.entry(key.to_string()).or_insert((0, now));

        // Reset if window expired
        if now.duration_since(entry.1) >= self.window {
            entry.0 = 0;
            entry.1 = now;
        }

        // Check limit
        if entry.0 >= self.limit {
            let reset_in = self.window.as_secs() - now.duration_since(entry.1).as_secs();
            return Err(RateLimitError {
                limit: self.limit,
                remaining: 0,
                reset_in,
            });
        }

        entry.0 += 1;

        Ok(())
    }

    pub fn remaining(&self, key: &str) -> u32 {
        self.requests
            .get(key)
            .map(|e| self.limit.saturating_sub(e.0))
            .unwrap_or(self.limit)
    }
}

#[derive(Debug)]
pub struct RateLimitError {
    pub limit: u32,
    pub remaining: u32,
    pub reset_in: u64,
}

/// Rate limiting middleware
pub async fn rate_limit_middleware(
    State(limiter): State<Arc<RateLimiter>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let key = addr.ip().to_string();

    match limiter.check(&key) {
        Ok(()) => {
            let remaining = limiter.remaining(&key);
            let mut response = next.run(request).await;

            // Add rate limit headers
            let headers = response.headers_mut();
            headers.insert(
                "X-RateLimit-Limit",
                HeaderValue::from_str(&limiter.limit.to_string()).unwrap(),
            );
            headers.insert(
                "X-RateLimit-Remaining",
                HeaderValue::from_str(&remaining.to_string()).unwrap(),
            );

            response
        }
        Err(err) => {
            warn!(ip = %key, "Rate limit exceeded");

            let mut response = (
                StatusCode::TOO_MANY_REQUESTS,
                format!("Rate limit exceeded. Try again in {} seconds.", err.reset_in),
            )
                .into_response();

            let headers = response.headers_mut();
            headers.insert(
                "X-RateLimit-Limit",
                HeaderValue::from_str(&err.limit.to_string()).unwrap(),
            );
            headers.insert(
                "X-RateLimit-Remaining",
                HeaderValue::from_static("0"),
            );
            headers.insert(
                "Retry-After",
                HeaderValue::from_str(&err.reset_in.to_string()).unwrap(),
            );

            response
        }
    }
}

// ============================================================================
// Security Headers
// ============================================================================

/// Add security headers to all responses
pub async fn security_headers_middleware(request: Request<Body>, next: Next) -> Response {
    let mut response = next.run(request).await;
    let headers = response.headers_mut();

    // Content Security Policy
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; \
             script-src 'self' 'unsafe-inline' 'unsafe-eval'; \
             style-src 'self' 'unsafe-inline'; \
             img-src 'self' data: https:; \
             font-src 'self' data:; \
             connect-src 'self' https://api.mycelix.mail wss://api.mycelix.mail; \
             frame-ancestors 'none'; \
             base-uri 'self'; \
             form-action 'self'",
        ),
    );

    // HTTP Strict Transport Security
    headers.insert(
        header::STRICT_TRANSPORT_SECURITY,
        HeaderValue::from_static("max-age=31536000; includeSubDomains; preload"),
    );

    // X-Content-Type-Options
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );

    // X-Frame-Options
    headers.insert(
        header::X_FRAME_OPTIONS,
        HeaderValue::from_static("DENY"),
    );

    // X-XSS-Protection
    headers.insert(
        "X-XSS-Protection",
        HeaderValue::from_static("1; mode=block"),
    );

    // Referrer-Policy
    headers.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );

    // Permissions-Policy
    headers.insert(
        "Permissions-Policy",
        HeaderValue::from_static(
            "accelerometer=(), camera=(), geolocation=(), gyroscope=(), \
             magnetometer=(), microphone=(), payment=(), usb=()",
        ),
    );

    // Cache-Control for API responses
    if !headers.contains_key(header::CACHE_CONTROL) {
        headers.insert(
            header::CACHE_CONTROL,
            HeaderValue::from_static("no-store, no-cache, must-revalidate"),
        );
    }

    response
}

// ============================================================================
// CORS Configuration
// ============================================================================

use tower_http::cors::{AllowOrigin, CorsLayer};

pub fn cors_layer(allowed_origins: Vec<String>) -> CorsLayer {
    let origins: Vec<HeaderValue> = allowed_origins
        .iter()
        .filter_map(|o| o.parse().ok())
        .collect();

    if origins.is_empty() {
        // Secure-by-default: restrict to localhost instead of allowing arbitrary websites.
        CorsLayer::new()
            .allow_origin(AllowOrigin::predicate(|origin, _| {
                let o = origin.as_bytes();
                o.starts_with(b"http://localhost")
                    || o.starts_with(b"https://localhost")
                    || o.starts_with(b"http://127.0.0.1")
                    || o.starts_with(b"https://127.0.0.1")
                    || o.starts_with(b"http://[::1]")
                    || o.starts_with(b"https://[::1]")
                    || o == b"null"
            }))
            .allow_methods([
                Method::GET,
                Method::POST,
                Method::PUT,
                Method::PATCH,
                Method::DELETE,
                Method::OPTIONS,
            ])
            .allow_headers([
                header::AUTHORIZATION,
                header::CONTENT_TYPE,
                header::ACCEPT,
                header::ORIGIN,
            ])
            .max_age(Duration::from_secs(86400))
    } else {
        CorsLayer::new()
            .allow_origin(origins)
            .allow_methods([
                Method::GET,
                Method::POST,
                Method::PUT,
                Method::PATCH,
                Method::DELETE,
                Method::OPTIONS,
            ])
            .allow_headers([
                header::AUTHORIZATION,
                header::CONTENT_TYPE,
                header::ACCEPT,
                header::ORIGIN,
            ])
            .allow_credentials(true)
            .max_age(Duration::from_secs(86400))
    }
}

// ============================================================================
// Request Validation
// ============================================================================

/// Validate incoming requests
pub async fn request_validation_middleware(
    headers: HeaderMap,
    request: Request<Body>,
    next: Next,
) -> Response {
    // Check content type for POST/PUT/PATCH
    let method = request.method();
    if matches!(method, &Method::POST | &Method::PUT | &Method::PATCH) {
        if let Some(content_type) = headers.get(header::CONTENT_TYPE) {
            let ct = content_type.to_str().unwrap_or("");
            if !ct.starts_with("application/json")
                && !ct.starts_with("multipart/form-data")
                && !ct.starts_with("application/x-www-form-urlencoded")
            {
                return (
                    StatusCode::UNSUPPORTED_MEDIA_TYPE,
                    "Unsupported content type",
                )
                    .into_response();
            }
        }
    }

    // Check for suspicious headers
    if headers.contains_key("X-Forwarded-Host") {
        // Could be host header injection attempt
        warn!("Suspicious X-Forwarded-Host header detected");
    }

    next.run(request).await
}

// ============================================================================
// IP Blocking
// ============================================================================

#[derive(Clone)]
pub struct IPBlocker {
    blocked_ips: Arc<RwLock<std::collections::HashSet<String>>>,
    blocked_ranges: Arc<RwLock<Vec<ipnetwork::IpNetwork>>>,
}

impl IPBlocker {
    pub fn new() -> Self {
        Self {
            blocked_ips: Arc::new(RwLock::new(std::collections::HashSet::new())),
            blocked_ranges: Arc::new(RwLock::new(Vec::new())),
        }
    }

    pub async fn block_ip(&self, ip: &str) {
        self.blocked_ips.write().await.insert(ip.to_string());
    }

    pub async fn unblock_ip(&self, ip: &str) {
        self.blocked_ips.write().await.remove(ip);
    }

    pub async fn is_blocked(&self, ip: &str) -> bool {
        // Check exact IP
        if self.blocked_ips.read().await.contains(ip) {
            return true;
        }

        // Check IP ranges
        if let Ok(addr) = ip.parse::<std::net::IpAddr>() {
            for range in self.blocked_ranges.read().await.iter() {
                if range.contains(addr) {
                    return true;
                }
            }
        }

        false
    }
}

impl Default for IPBlocker {
    fn default() -> Self {
        Self::new()
    }
}

/// IP blocking middleware
pub async fn ip_block_middleware(
    State(blocker): State<Arc<IPBlocker>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let ip = addr.ip().to_string();

    if blocker.is_blocked(&ip).await {
        warn!(ip = %ip, "Blocked IP attempted access");
        return (StatusCode::FORBIDDEN, "Access denied").into_response();
    }

    next.run(request).await
}

// ============================================================================
// Audit Logging
// ============================================================================

#[derive(Clone)]
pub struct AuditLogger {
    // In production, would write to database or external service
}

impl AuditLogger {
    pub fn new() -> Self {
        Self {}
    }

    pub fn log_request(&self, user_id: Option<&str>, action: &str, resource: &str, ip: &str) {
        info!(
            user_id = user_id.unwrap_or("anonymous"),
            action = action,
            resource = resource,
            ip = ip,
            "Audit log"
        );
    }

    pub fn log_auth_attempt(&self, email: &str, success: bool, ip: &str) {
        if success {
            info!(email = email, ip = ip, "Successful authentication");
        } else {
            warn!(email = email, ip = ip, "Failed authentication attempt");
        }
    }

    pub fn log_security_event(&self, event_type: &str, details: &str, ip: &str) {
        warn!(event_type = event_type, details = details, ip = ip, "Security event");
    }
}

impl Default for AuditLogger {
    fn default() -> Self {
        Self::new()
    }
}

/// Audit logging middleware
pub async fn audit_middleware(
    State(logger): State<Arc<AuditLogger>>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let method = request.method().to_string();
    let uri = request.uri().path().to_string();
    let ip = addr.ip().to_string();

    // Extract user ID from auth if available
    let user_id = request
        .extensions()
        .get::<super::auth::Claims>()
        .map(|c| c.sub.clone());

    logger.log_request(user_id.as_deref(), &method, &uri, &ip);

    next.run(request).await
}

// ============================================================================
// Request Size Limiting
// ============================================================================

use tower_http::limit::RequestBodyLimitLayer;

pub fn request_size_limit_layer(max_size: usize) -> RequestBodyLimitLayer {
    RequestBodyLimitLayer::new(max_size)
}

// ============================================================================
// Timeout Configuration
// ============================================================================

use tower_http::timeout::TimeoutLayer;

pub fn timeout_layer(timeout_secs: u64) -> TimeoutLayer {
    TimeoutLayer::new(Duration::from_secs(timeout_secs))
}

// ============================================================================
// Combined Security Stack
// ============================================================================

use axum::{middleware, Router};
use tower::ServiceBuilder;
use tower_http::compression::CompressionLayer;

pub fn apply_security_layers<S>(
    router: Router<S>,
    rate_limiter: Arc<RateLimiter>,
    ip_blocker: Arc<IPBlocker>,
    audit_logger: Arc<AuditLogger>,
    allowed_origins: Vec<String>,
) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    router
        .layer(
            ServiceBuilder::new()
                // Compression
                .layer(CompressionLayer::new())
                // Request size limit (10MB)
                .layer(request_size_limit_layer(10 * 1024 * 1024))
                // Timeout (30 seconds)
                .layer(timeout_layer(30))
                // CORS
                .layer(cors_layer(allowed_origins)),
        )
        .layer(middleware::from_fn(security_headers_middleware))
        .layer(middleware::from_fn(request_validation_middleware))
        .layer(middleware::from_fn_with_state(
            rate_limiter,
            rate_limit_middleware,
        ))
        .layer(middleware::from_fn_with_state(ip_blocker, ip_block_middleware))
        .layer(middleware::from_fn_with_state(audit_logger, audit_middleware))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter() {
        let limiter = RateLimiter::new(5, 60);

        // Should allow first 5 requests
        for _ in 0..5 {
            assert!(limiter.check("test-ip").is_ok());
        }

        // 6th request should fail
        assert!(limiter.check("test-ip").is_err());

        // Different IP should still work
        assert!(limiter.check("other-ip").is_ok());
    }

    #[tokio::test]
    async fn test_ip_blocker() {
        let blocker = IPBlocker::new();

        assert!(!blocker.is_blocked("1.2.3.4").await);

        blocker.block_ip("1.2.3.4").await;
        assert!(blocker.is_blocked("1.2.3.4").await);

        blocker.unblock_ip("1.2.3.4").await;
        assert!(!blocker.is_blocked("1.2.3.4").await);
    }
}
