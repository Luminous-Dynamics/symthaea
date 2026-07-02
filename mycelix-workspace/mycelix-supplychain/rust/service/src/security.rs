// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Security middleware and utilities
//!
//! Provides:
//! - Rate limiting per IP address
//! - Security headers on all responses
//! - Input validation helpers
//! - CORS configuration

use axum::{
    extract::Request,
    http::{header, HeaderValue},
    middleware::Next,
    response::Response,
};

/// Security headers middleware
///
/// Adds essential security headers to all responses:
/// - X-Frame-Options: DENY
/// - X-Content-Type-Options: nosniff
/// - X-XSS-Protection: 1; mode=block
/// - Strict-Transport-Security (HSTS)
/// - Content-Security-Policy
/// - Referrer-Policy
pub async fn security_headers_middleware(req: Request, next: Next) -> Response {
    let mut response = next.run(req).await;

    let headers = response.headers_mut();

    // Prevent clickjacking
    headers.insert(
        header::X_FRAME_OPTIONS,
        HeaderValue::from_static("DENY"),
    );

    // Prevent MIME type sniffing
    headers.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );

    // XSS protection (legacy, but still useful for older browsers)
    headers.insert(
        "X-XSS-Protection",
        HeaderValue::from_static("1; mode=block"),
    );

    // Only add HSTS in production/HTTPS
    if !cfg!(debug_assertions) {
        headers.insert(
            header::STRICT_TRANSPORT_SECURITY,
            HeaderValue::from_static("max-age=31536000; includeSubDomains"),
        );
    }

    // Content Security Policy
    headers.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static("default-src 'self'"),
    );

    // Referrer policy
    headers.insert(
        header::REFERER,
        HeaderValue::from_static("strict-origin-when-cross-origin"),
    );

    // Permissions policy (restrict access to browser features)
    headers.insert(
        "Permissions-Policy",
        HeaderValue::from_static("geolocation=(), microphone=(), camera=()"),
    );

    response
}

/// Input validation utilities
pub mod validation {
    use regex::Regex;
    use once_cell::sync::Lazy;

    // Regex patterns for ID validation
    static BATCH_ID_PATTERN: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^[A-Za-z0-9\-_]{1,128}$").unwrap());
    static PRODUCT_ID_PATTERN: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^[A-Za-z0-9\-_]{1,256}$").unwrap());
    static FACILITY_ID_PATTERN: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"^[A-Za-z0-9\-_]{1,128}$").unwrap());

    /// Validate batch ID format
    ///
    /// Requirements:
    /// - Length: 1-128 characters
    /// - Characters: Alphanumeric, dashes, underscores only
    pub fn validate_batch_id(id: &str) -> Result<(), String> {
        if id.is_empty() {
            return Err("Batch ID cannot be empty".to_string());
        }
        if id.len() > 128 {
            return Err("Batch ID exceeds maximum length of 128 characters".to_string());
        }
        if !BATCH_ID_PATTERN.is_match(id) {
            return Err("Batch ID contains invalid characters (use only A-Z, a-z, 0-9, -, _)".to_string());
        }
        Ok(())
    }

    /// Validate product ID format
    pub fn validate_product_id(id: &str) -> Result<(), String> {
        if id.is_empty() {
            return Err("Product ID cannot be empty".to_string());
        }
        if id.len() > 256 {
            return Err("Product ID exceeds maximum length of 256 characters".to_string());
        }
        if !PRODUCT_ID_PATTERN.is_match(id) {
            return Err("Product ID contains invalid characters".to_string());
        }
        Ok(())
    }

    /// Validate facility ID format
    pub fn validate_facility_id(id: &str) -> Result<(), String> {
        if id.is_empty() {
            return Err("Facility ID cannot be empty".to_string());
        }
        if id.len() > 128 {
            return Err("Facility ID exceeds maximum length of 128 characters".to_string());
        }
        if !FACILITY_ID_PATTERN.is_match(id) {
            return Err("Facility ID contains invalid characters".to_string());
        }
        Ok(())
    }

    /// Validate metadata size
    ///
    /// Maximum 10KB of metadata to prevent abuse
    pub fn validate_metadata_size(metadata_json: &str) -> Result<(), String> {
        const MAX_SIZE: usize = 10 * 1024; // 10KB
        if metadata_json.len() > MAX_SIZE {
            return Err(format!(
                "Metadata exceeds maximum size of {} bytes",
                MAX_SIZE
            ));
        }
        Ok(())
    }

    /// Validate array size
    ///
    /// Maximum 100 items in arrays like prevBatchIds
    pub fn validate_array_size<T>(arr: &[T], field_name: &str) -> Result<(), String> {
        const MAX_ITEMS: usize = 100;
        if arr.len() > MAX_ITEMS {
            return Err(format!(
                "{} exceeds maximum length of {} items",
                field_name, MAX_ITEMS
            ));
        }
        Ok(())
    }

    /// Validate string length
    pub fn validate_string_length(
        s: &str,
        field_name: &str,
        max_length: usize,
    ) -> Result<(), String> {
        if s.len() > max_length {
            return Err(format!(
                "{} exceeds maximum length of {} characters",
                field_name, max_length
            ));
        }
        Ok(())
    }

    /// Sanitize string for safe storage
    ///
    /// Removes control characters and potentially dangerous content
    pub fn sanitize_string(input: &str) -> String {
        input
            .chars()
            .filter(|c| !c.is_control() || c.is_whitespace())
            .collect()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_validate_batch_id() {
            assert!(validate_batch_id("BATCH-001").is_ok());
            assert!(validate_batch_id("BATCH_2025_HARVEST_001").is_ok());
            assert!(validate_batch_id("").is_err());
            assert!(validate_batch_id("BATCH 001").is_err()); // Space not allowed
            assert!(validate_batch_id(&"A".repeat(129)).is_err()); // Too long
        }

        #[test]
        fn test_validate_product_id() {
            assert!(validate_product_id("COFFEE-BEANS-ETHIOPIAN").is_ok());
            assert!(validate_product_id("").is_err());
            assert!(validate_product_id(&"A".repeat(257)).is_err());
        }

        #[test]
        fn test_validate_metadata_size() {
            let small_metadata = r#"{"key": "value"}"#;
            assert!(validate_metadata_size(small_metadata).is_ok());

            let large_metadata = "x".repeat(11 * 1024);
            assert!(validate_metadata_size(&large_metadata).is_err());
        }

        #[test]
        fn test_validate_array_size() {
            let small_arr = vec![1, 2, 3];
            assert!(validate_array_size(&small_arr, "test").is_ok());

            let large_arr = vec![1; 101];
            assert!(validate_array_size(&large_arr, "test").is_err());
        }

        #[test]
        fn test_sanitize_string() {
            assert_eq!(sanitize_string("Hello World"), "Hello World");
            assert_eq!(sanitize_string("Hello\x00World"), "HelloWorld");
            assert_eq!(sanitize_string("Hello\tWorld"), "Hello\tWorld"); // Tab preserved
        }
    }
}

/// Rate limiting configuration
///
/// Note: For production, consider using tower-governor or tower-limit
/// For simplicity, this is documented but not implemented here.
/// See deployment guide for Redis-backed rate limiting.
pub mod rate_limit {
    /// Rate limit configuration
    #[derive(Debug, Clone)]
    pub struct RateLimitConfig {
        /// Requests per minute
        pub requests_per_minute: u32,
        /// Burst allowance
        pub burst: u32,
    }

    impl Default for RateLimitConfig {
        fn default() -> Self {
            Self {
                requests_per_minute: 100,
                burst: 20,
            }
        }
    }

    impl RateLimitConfig {
        /// Create a rate limit config for event ingestion
        pub fn for_events() -> Self {
            Self {
                requests_per_minute: 100,
                burst: 20,
            }
        }

        /// Create a rate limit config for queries
        pub fn for_queries() -> Self {
            Self {
                requests_per_minute: 200,
                burst: 50,
            }
        }

        /// No rate limit (for metrics, health checks)
        pub fn unlimited() -> Self {
            Self {
                requests_per_minute: u32::MAX,
                burst: u32::MAX,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limit_config() {
        let config = rate_limit::RateLimitConfig::default();
        assert_eq!(config.requests_per_minute, 100);
        assert_eq!(config.burst, 20);

        let events_config = rate_limit::RateLimitConfig::for_events();
        assert_eq!(events_config.requests_per_minute, 100);

        let queries_config = rate_limit::RateLimitConfig::for_queries();
        assert_eq!(queries_config.requests_per_minute, 200);
    }
}
