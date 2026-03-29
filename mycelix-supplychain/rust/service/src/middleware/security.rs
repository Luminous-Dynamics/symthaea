// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
// Security headers middleware
//!
//! Adds security headers to protect against common web vulnerabilities

use axum::{
    extract::Request,
    http::{HeaderValue, header},
    middleware::Next,
    response::Response,
};

/// Add security headers to all responses
///
/// Headers added:
/// - X-Content-Type-Options: Prevent MIME sniffing
/// - X-Frame-Options: Prevent clickjacking
/// - X-XSS-Protection: Enable XSS filter
/// - Content-Security-Policy: Restrict resource loading
/// - Strict-Transport-Security: Force HTTPS
/// - Referrer-Policy: Control referrer information
/// - Permissions-Policy: Restrict browser features
pub async fn security_headers(req: Request, next: Next) -> Response {
    let mut response = next.run(req).await;
    let headers = response.headers_mut();

    // Prevent MIME type sniffing
    headers.insert(
        header::HeaderName::from_static("x-content-type-options"),
        HeaderValue::from_static("nosniff")
    );

    // Prevent clickjacking attacks
    headers.insert(
        header::HeaderName::from_static("x-frame-options"),
        HeaderValue::from_static("DENY")
    );

    // Enable XSS protection (legacy, but doesn't hurt)
    headers.insert(
        header::HeaderName::from_static("x-xss-protection"),
        HeaderValue::from_static("1; mode=block")
    );

    // Content Security Policy - restrict resource loading
    headers.insert(
        header::HeaderName::from_static("content-security-policy"),
        HeaderValue::from_static("default-src 'self'; frame-ancestors 'none'")
    );

    // Force HTTPS in production (31536000 seconds = 1 year)
    // Note: Only effective if served over HTTPS
    headers.insert(
        header::HeaderName::from_static("strict-transport-security"),
        HeaderValue::from_static("max-age=31536000; includeSubDomains; preload")
    );

    // Control referrer information leakage
    headers.insert(
        header::HeaderName::from_static("referrer-policy"),
        HeaderValue::from_static("strict-origin-when-cross-origin")
    );

    // Permissions policy (restrict browser features)
    headers.insert(
        header::HeaderName::from_static("permissions-policy"),
        HeaderValue::from_static("geolocation=(), microphone=(), camera=()")
    );

    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        body::Body,
        http::{Request, StatusCode},
        middleware,
        response::IntoResponse,
        routing::get,
        Router,
    };
    use tower::ServiceExt;

    async fn test_handler() -> impl IntoResponse {
        (StatusCode::OK, "test")
    }

    #[tokio::test]
    async fn test_security_headers_present() {
        let app = Router::new()
            .route("/test", get(test_handler))
            .layer(middleware::from_fn(security_headers));

        let response = app
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();

        let headers = response.headers();

        // Verify all security headers are present
        assert!(headers.contains_key("x-content-type-options"));
        assert_eq!(
            headers.get("x-content-type-options").unwrap(),
            "nosniff"
        );

        assert!(headers.contains_key("x-frame-options"));
        assert_eq!(
            headers.get("x-frame-options").unwrap(),
            "DENY"
        );

        assert!(headers.contains_key("x-xss-protection"));
        assert!(headers.contains_key("content-security-policy"));
        assert!(headers.contains_key("strict-transport-security"));
        assert!(headers.contains_key("referrer-policy"));
        assert!(headers.contains_key("permissions-policy"));
    }

    #[tokio::test]
    async fn test_security_headers_dont_override_existing() {
        let app = Router::new()
            .route("/test", get(test_handler))
            .layer(middleware::from_fn(security_headers));

        let response = app
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();

        // Response should still be successful
        assert_eq!(response.status(), StatusCode::OK);
    }
}
