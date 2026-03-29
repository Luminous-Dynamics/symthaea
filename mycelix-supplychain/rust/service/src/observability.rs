// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observability module for structured logging and request tracking
//!
//! Provides correlation IDs, performance timing, and structured logging

use axum::{extract::Request, middleware::Next, response::Response};
use std::time::Instant;
use tracing::{info, warn};
use uuid::Uuid;

/// Initialize tracing subscriber with structured logging
///
/// Configured with JSON formatting for production observability
pub fn init_tracing() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,provenance_service=debug"));

    // Check if JSON logging is requested (for production)
    let use_json = std::env::var("LOG_FORMAT")
        .map(|v| v.to_lowercase() == "json")
        .unwrap_or(false);

    if use_json {
        // JSON-structured logging for production
        tracing_subscriber::registry()
            .with(env_filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .json()  // Use JSON format
                    .with_current_span(true)  // Include current span context
                    .with_span_list(true)  // Include full span hierarchy
                    .with_target(true)  // Include target (module path)
                    .with_level(true)  // Include log level
            )
            .init();

        info!("Structured JSON logging initialized");
    } else {
        // Human-readable logging for development
        tracing_subscriber::registry()
            .with(env_filter)
            .with(
                tracing_subscriber::fmt::layer()
                    .with_target(true)
                    .with_line_number(true)
                    .with_thread_ids(false)
            )
            .init();

        info!("Structured logging initialized (human-readable format)");
    }
}

/// Middleware for request logging with timing and correlation IDs
///
/// Uses tracing spans for better structured logging and context propagation
pub async fn request_logging_middleware(req: Request, next: Next) -> Response {
    use tracing::{info_span, Instrument};

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
        info!("Request started");
        let start = Instant::now();

        // Process the request
        let response = next.run(req).await;

        // Record metrics
        let duration = start.elapsed();
        let duration_ms = duration.as_millis() as u64;
        let status = response.status();
        let status_code = status.as_u16();

        // Update span with completion metadata
        tracing::Span::current().record("status", status_code);
        tracing::Span::current().record("duration_ms", duration_ms);

        // Log completion with appropriate level
        if status.is_success() {
            info!(
                status = status_code,
                duration_ms = duration_ms,
                "Request completed"
            );
        } else if status.is_client_error() {
            warn!(
                status = status_code,
                duration_ms = duration_ms,
                "Request failed (client error)"
            );
        } else {
            warn!(
                status = status_code,
                duration_ms = duration_ms,
                "Request failed (server error)"
            );
        }

        response
    }
    .instrument(span)
    .await
}

/// Log database operations for debugging
#[macro_export]
macro_rules! log_db_operation {
    ($operation:expr, $duration:expr) => {
        tracing::debug!(
            operation = $operation,
            duration_ms = %$duration.as_millis(),
            "Database operation completed"
        );
    };
}

/// Log validation errors
pub fn log_validation_error(field: &str, reason: &str) {
    warn!(
        field = field,
        reason = reason,
        "Validation error"
    );
}

/// Log claim creation
pub fn log_claim_created(claim_id: &str, event_type: &str, batch_id: &str) {
    info!(
        claim_id = claim_id,
        event_type = event_type,
        batch_id = batch_id,
        "Claim created"
    );
}

/// Log lineage resolution
pub fn log_lineage_resolved(batch_id: &str, claim_count: usize, duration_ms: u64) {
    info!(
        batch_id = batch_id,
        claim_count = claim_count,
        duration_ms = duration_ms,
        "Lineage resolved"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logging_functions() {
        // Just verify they compile and don't panic
        log_validation_error("test_field", "test reason");
        log_claim_created("test-id", "PRODUCED", "BATCH-001");
        log_lineage_resolved("BATCH-001", 5, 100);
    }
}
