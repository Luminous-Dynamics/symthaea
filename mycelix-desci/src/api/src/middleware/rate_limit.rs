// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Rate Limiting Middleware
//!
//! Implements per-IP rate limiting to prevent abuse and ensure fair usage
//!
//! NOTE: This is a placeholder implementation. For production use, integrate
//! a proper rate limiting solution like tower-governor or redis-based rate limiting.

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
}
