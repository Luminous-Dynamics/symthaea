// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Request tracing middleware
//!
//! Adds request correlation IDs and performance tracking to all HTTP requests

use axum::{
    extract::Request,
    middleware::Next,
    response::Response,
};
use std::time::Instant;
use tracing::{info_span, Instrument};
use uuid::Uuid;

/// Middleware to add request tracing with correlation IDs
///
/// This middleware:
/// - Generates a unique request_id for each request
/// - Creates a tracing span with request metadata
/// - Logs request start and completion
/// - Records request duration
/// - Propagates request_id through the entire request lifecycle
pub async fn trace_request(req: Request, next: Next) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let method = req.method().clone();
    let uri = req.uri().clone();
    let path = uri.path().to_string();

    // Create a span for this request with key metadata
    let span = info_span!(
        "http_request",
        request_id = %request_id,
        method = %method,
        path = %path,
        status = tracing::field::Empty,  // Will be filled in later
        duration_ms = tracing::field::Empty,  // Will be filled in later
    );

    async move {
        tracing::info!("Request started");
        let start = Instant::now();

        // Process the request
        let response = next.run(req).await;

        // Record metrics
        let duration = start.elapsed();
        let duration_ms = duration.as_millis() as u64;
        let status = response.status().as_u16();

        // Update span with completion metadata
        tracing::Span::current().record("status", status);
        tracing::Span::current().record("duration_ms", duration_ms);

        tracing::info!(
            status = status,
            duration_ms = duration_ms,
            "Request completed"
        );

        response
    }
    .instrument(span)
    .await
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
        (StatusCode::OK, "test response")
    }

    #[tokio::test]
    async fn test_trace_request_middleware() {
        let app = Router::new()
            .route("/test", get(test_handler))
            .layer(middleware::from_fn(trace_request));

        let response = app
            .oneshot(Request::builder().uri("/test").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
    }
}
