// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rate Limiting Middleware
//!
//! Sliding window rate limiting with Redis backend

use axum::{
    body::Body,
    extract::State,
    http::{Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
};
use redis::AsyncCommands;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tracing::{info, warn};

/// Rate limit configuration
#[derive(Clone, Debug)]
pub struct RateLimitConfig {
    /// Maximum requests per window
    pub max_requests: u32,
    /// Window duration
    pub window: Duration,
    /// Key prefix for Redis
    pub key_prefix: String,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            max_requests: 100,
            window: Duration::from_secs(60),
            key_prefix: "ratelimit".to_string(),
        }
    }
}

/// Rate limiter state
#[derive(Clone)]
pub struct RateLimiter {
    redis: redis::Client,
    config: RateLimitConfig,
}

impl RateLimiter {
    pub fn new(redis_url: &str, config: RateLimitConfig) -> Result<Self, redis::RedisError> {
        let redis = redis::Client::open(redis_url)?;
        Ok(Self { redis, config })
    }

    /// Check if request is allowed using sliding window algorithm
    pub async fn check(&self, key: &str) -> Result<RateLimitResult, redis::RedisError> {
        let mut conn = self.redis.get_async_connection().await?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let window_ms = self.config.window.as_millis() as u64;
        let window_start = now - window_ms;

        let redis_key = format!("{}:{}", self.config.key_prefix, key);

        // Sliding window using sorted set
        // 1. Remove old entries
        // 2. Add current request
        // 3. Count requests in window
        // 4. Set expiry

        let script = redis::Script::new(
            r#"
            local key = KEYS[1]
            local now = tonumber(ARGV[1])
            local window_start = tonumber(ARGV[2])
            local max_requests = tonumber(ARGV[3])
            local window_ms = tonumber(ARGV[4])

            -- Remove old entries
            redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

            -- Count current requests
            local count = redis.call('ZCARD', key)

            if count < max_requests then
                -- Add this request
                redis.call('ZADD', key, now, now .. '-' .. math.random())
                redis.call('PEXPIRE', key, window_ms)
                return {1, max_requests - count - 1, 0}
            else
                -- Get oldest entry to calculate retry time
                local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
                local retry_after = 0
                if #oldest >= 2 then
                    retry_after = oldest[2] + window_ms - now
                end
                return {0, 0, retry_after}
            end
            "#,
        );

        let result: (i32, i32, i64) = script
            .key(&redis_key)
            .arg(now)
            .arg(window_start)
            .arg(self.config.max_requests)
            .arg(window_ms)
            .invoke_async(&mut conn)
            .await?;

        Ok(RateLimitResult {
            allowed: result.0 == 1,
            remaining: result.1 as u32,
            retry_after_ms: result.2 as u64,
            limit: self.config.max_requests,
        })
    }
}

#[derive(Debug, Clone)]
pub struct RateLimitResult {
    pub allowed: bool,
    pub remaining: u32,
    pub retry_after_ms: u64,
    pub limit: u32,
}

/// Rate limiting middleware layer
pub async fn rate_limit_middleware(
    State(limiter): State<Arc<RateLimiter>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    // Extract rate limit key (IP or user ID)
    let key = extract_rate_limit_key(&request);

    match limiter.check(&key).await {
        Ok(result) => {
            if result.allowed {
                let mut response = next.run(request).await;

                // Add rate limit headers
                let headers = response.headers_mut();
                headers.insert(
                    "X-RateLimit-Limit",
                    result.limit.to_string().parse().unwrap(),
                );
                headers.insert(
                    "X-RateLimit-Remaining",
                    result.remaining.to_string().parse().unwrap(),
                );

                response
            } else {
                warn!(key = %key, "Rate limit exceeded");

                let retry_after_secs = (result.retry_after_ms / 1000).max(1);

                Response::builder()
                    .status(StatusCode::TOO_MANY_REQUESTS)
                    .header("Retry-After", retry_after_secs.to_string())
                    .header("X-RateLimit-Limit", result.limit.to_string())
                    .header("X-RateLimit-Remaining", "0")
                    .body(Body::from("Rate limit exceeded"))
                    .unwrap()
            }
        }
        Err(e) => {
            // On Redis error, allow the request but log
            warn!(error = %e, "Rate limiter error, allowing request");
            next.run(request).await
        }
    }
}

fn extract_rate_limit_key(request: &Request<Body>) -> String {
    // Try to get user ID from auth header
    if let Some(auth) = request.headers().get("authorization") {
        if let Ok(auth_str) = auth.to_str() {
            if auth_str.starts_with("Bearer ") {
                // Extract user ID from JWT (simplified)
                let token = &auth_str[7..];
                if let Some(user_id) = extract_user_id_from_token(token) {
                    return format!("user:{}", user_id);
                }
            }
        }
    }

    // Fall back to IP address
    let ip = request
        .headers()
        .get("x-forwarded-for")
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.split(',').next())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    format!("ip:{}", ip)
}

fn extract_user_id_from_token(token: &str) -> Option<String> {
    // Simplified JWT parsing - in production use proper JWT library
    let parts: Vec<&str> = token.split('.').collect();
    if parts.len() != 3 {
        return None;
    }

    // Decode payload (middle part)
    if let Ok(payload) = base64::decode(parts[1]) {
        if let Ok(claims) = serde_json::from_slice::<serde_json::Value>(&payload) {
            return claims["sub"].as_str().map(|s| s.to_string());
        }
    }

    None
}

/// Different rate limit tiers
pub struct RateLimitTiers {
    /// Anonymous/unauthenticated requests
    pub anonymous: RateLimitConfig,
    /// Authenticated users
    pub authenticated: RateLimitConfig,
    /// Premium users
    pub premium: RateLimitConfig,
    /// API endpoints (stricter)
    pub api: RateLimitConfig,
}

impl Default for RateLimitTiers {
    fn default() -> Self {
        Self {
            anonymous: RateLimitConfig {
                max_requests: 30,
                window: Duration::from_secs(60),
                key_prefix: "rl:anon".to_string(),
            },
            authenticated: RateLimitConfig {
                max_requests: 100,
                window: Duration::from_secs(60),
                key_prefix: "rl:auth".to_string(),
            },
            premium: RateLimitConfig {
                max_requests: 500,
                window: Duration::from_secs(60),
                key_prefix: "rl:premium".to_string(),
            },
            api: RateLimitConfig {
                max_requests: 1000,
                window: Duration::from_secs(60),
                key_prefix: "rl:api".to_string(),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert_eq!(config.max_requests, 100);
        assert_eq!(config.window, Duration::from_secs(60));
    }

    #[test]
    fn test_extract_user_id_from_token() {
        // Mock JWT with sub claim
        let payload = base64::encode(r#"{"sub":"user-123","exp":9999999999}"#);
        let token = format!("header.{}.signature", payload);

        let user_id = extract_user_id_from_token(&token);
        assert_eq!(user_id, Some("user-123".to_string()));
    }
}
