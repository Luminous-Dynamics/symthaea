// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metrics Middleware
//!
//! Tracks HTTP request metrics for Prometheus

use axum::{extract::Request, middleware::Next, response::Response};
use std::time::Instant;

use crate::metrics;

/// Metrics tracking middleware
pub struct MetricsMiddleware;

impl MetricsMiddleware {
    /// Track HTTP request metrics
    pub async fn track(req: Request, next: Next) -> Response {
        let method = req.method().to_string();
        let path = req.uri().path().to_string();
        let start = Instant::now();

        // Process request
        let response = next.run(req).await;

        // Track metrics
        let duration = start.elapsed();
        let status = response.status();

        metrics::track_http_request(&method, &path, status.as_u16());
        metrics::HTTP_REQUEST_DURATION
            .with_label_values(&[&method, &path])
            .observe(duration.as_secs_f64());

        response
    }
}
