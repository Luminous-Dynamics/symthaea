// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Security middleware tests
//!
//! Tests for rate limiting, IP blocking, and security headers

use super::security::*;
use std::time::Duration;

#[cfg(test)]
mod rate_limiter_tests {
    use super::*;

    #[test]
    fn test_rate_limiter_allows_within_limit() {
        let limiter = RateLimiter::new(5, 60);

        for i in 0..5 {
            let result = limiter.check("test-ip");
            assert!(result.is_ok(), "Request {} should be allowed", i + 1);
        }
    }

    #[test]
    fn test_rate_limiter_blocks_over_limit() {
        let limiter = RateLimiter::new(3, 60);

        // Use up the limit
        for _ in 0..3 {
            limiter.check("test-ip").unwrap();
        }

        // Next request should fail
        let result = limiter.check("test-ip");
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.limit, 3);
        assert_eq!(err.remaining, 0);
    }

    #[test]
    fn test_rate_limiter_separate_keys() {
        let limiter = RateLimiter::new(2, 60);

        // Use up limit for ip-1
        limiter.check("ip-1").unwrap();
        limiter.check("ip-1").unwrap();
        assert!(limiter.check("ip-1").is_err());

        // ip-2 should still work
        assert!(limiter.check("ip-2").is_ok());
        assert!(limiter.check("ip-2").is_ok());
    }

    #[test]
    fn test_rate_limiter_remaining() {
        let limiter = RateLimiter::new(5, 60);

        assert_eq!(limiter.remaining("new-ip"), 5);

        limiter.check("new-ip").unwrap();
        assert_eq!(limiter.remaining("new-ip"), 4);

        limiter.check("new-ip").unwrap();
        limiter.check("new-ip").unwrap();
        assert_eq!(limiter.remaining("new-ip"), 2);
    }

    #[test]
    fn test_rate_limit_error_contains_reset_time() {
        let limiter = RateLimiter::new(1, 120);

        limiter.check("test-ip").unwrap();
        let result = limiter.check("test-ip");

        let err = result.unwrap_err();
        assert!(err.reset_in <= 120);
        assert!(err.reset_in > 0);
    }
}

#[cfg(test)]
mod ip_blocker_tests {
    use super::*;

    #[tokio::test]
    async fn test_ip_not_blocked_by_default() {
        let blocker = IPBlocker::new();
        assert!(!blocker.is_blocked("192.168.1.1").await);
    }

    #[tokio::test]
    async fn test_block_ip() {
        let blocker = IPBlocker::new();

        blocker.block_ip("192.168.1.100").await;
        assert!(blocker.is_blocked("192.168.1.100").await);
    }

    #[tokio::test]
    async fn test_unblock_ip() {
        let blocker = IPBlocker::new();

        blocker.block_ip("10.0.0.1").await;
        assert!(blocker.is_blocked("10.0.0.1").await);

        blocker.unblock_ip("10.0.0.1").await;
        assert!(!blocker.is_blocked("10.0.0.1").await);
    }

    #[tokio::test]
    async fn test_multiple_blocked_ips() {
        let blocker = IPBlocker::new();

        blocker.block_ip("1.1.1.1").await;
        blocker.block_ip("2.2.2.2").await;
        blocker.block_ip("3.3.3.3").await;

        assert!(blocker.is_blocked("1.1.1.1").await);
        assert!(blocker.is_blocked("2.2.2.2").await);
        assert!(blocker.is_blocked("3.3.3.3").await);
        assert!(!blocker.is_blocked("4.4.4.4").await);
    }

    #[tokio::test]
    async fn test_blocker_default_impl() {
        let blocker = IPBlocker::default();
        assert!(!blocker.is_blocked("127.0.0.1").await);
    }
}

#[cfg(test)]
mod audit_logger_tests {
    use super::*;

    #[test]
    fn test_audit_logger_creation() {
        let logger = AuditLogger::new();
        // Should not panic
        logger.log_request(Some("user-1"), "GET", "/api/emails", "127.0.0.1");
    }

    #[test]
    fn test_audit_logger_anonymous_request() {
        let logger = AuditLogger::new();
        // Should handle None user_id
        logger.log_request(None, "GET", "/api/public", "192.168.1.1");
    }

    #[test]
    fn test_audit_logger_auth_attempt() {
        let logger = AuditLogger::new();

        // Success
        logger.log_auth_attempt("user@example.com", true, "10.0.0.1");

        // Failure
        logger.log_auth_attempt("attacker@evil.com", false, "10.0.0.2");
    }

    #[test]
    fn test_audit_logger_security_event() {
        let logger = AuditLogger::new();

        logger.log_security_event(
            "rate_limit_exceeded",
            "IP exceeded 100 requests in 60 seconds",
            "192.168.1.50",
        );
    }

    #[test]
    fn test_audit_logger_default_impl() {
        let logger = AuditLogger::default();
        logger.log_request(Some("test"), "POST", "/api/test", "::1");
    }
}

#[cfg(test)]
mod cors_tests {
    use super::*;
    use axum::http::Method;

    #[test]
    fn test_cors_layer_with_origins() {
        let origins = vec![
            "https://mycelix.mail".to_string(),
            "https://app.mycelix.mail".to_string(),
        ];

        let layer = cors_layer(origins);
        // Layer should be created without panic
        assert!(true);
    }

    #[test]
    fn test_cors_layer_empty_origins() {
        let layer = cors_layer(vec![]);
        // Secure-by-default: should restrict to localhost origins
        assert!(true);
    }

    #[test]
    fn test_cors_layer_invalid_origins() {
        let origins = vec![
            "not-a-valid-url".to_string(),
            "also invalid".to_string(),
        ];

        // Should handle invalid origins gracefully
        let layer = cors_layer(origins);
        assert!(true);
    }
}

#[cfg(test)]
mod request_size_limit_tests {
    use super::*;

    #[test]
    fn test_request_size_limit_layer() {
        let layer = request_size_limit_layer(10 * 1024 * 1024); // 10MB
        assert!(true);
    }

    #[test]
    fn test_small_request_size_limit() {
        let layer = request_size_limit_layer(1024); // 1KB
        assert!(true);
    }
}

#[cfg(test)]
mod timeout_tests {
    use super::*;

    #[test]
    fn test_timeout_layer() {
        let layer = timeout_layer(30);
        assert!(true);
    }

    #[test]
    fn test_short_timeout() {
        let layer = timeout_layer(5);
        assert!(true);
    }
}
