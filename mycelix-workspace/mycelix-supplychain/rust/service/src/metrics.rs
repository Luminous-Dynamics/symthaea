// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prometheus metrics for monitoring and observability
//!
//! Tracks:
//! - Event ingestion counts by type
//! - API request latency by endpoint
//! - Database query duration
//! - Active database connections
//! - System health metrics

use once_cell::sync::Lazy;
use prometheus::{
    opts, register_counter_vec, register_gauge, register_histogram_vec, CounterVec, Encoder,
    Gauge, HistogramVec, TextEncoder,
};
use std::time::Instant;

/// Counter for total events ingested, labeled by event_type
pub static EVENTS_INGESTED: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        opts!("supplychain_events_ingested_total", "Total events ingested"),
        &["event_type"]
    )
    .expect("Failed to register events_ingested metric")
});

/// Histogram for API request latency, labeled by endpoint and method
pub static API_REQUEST_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "supplychain_api_request_duration_seconds",
        "API request duration in seconds",
        &["method", "endpoint", "status"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
    )
    .expect("Failed to register api_request_duration metric")
});

/// Histogram for database query duration
pub static DB_QUERY_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "supplychain_db_query_duration_seconds",
        "Database query duration in seconds",
        &["operation"],
        vec![0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1]
    )
    .expect("Failed to register db_query_duration metric")
});

/// Counter for total claims stored
pub static CLAIMS_STORED: Lazy<prometheus::Counter> = Lazy::new(|| {
    register_counter_vec!(
        opts!("supplychain_claims_stored_total", "Total claims stored"),
        &[]
    )
    .expect("Failed to register claims_stored metric")
    .with_label_values(&[])
});

/// Gauge for active database connections
pub static DB_CONNECTIONS_ACTIVE: Lazy<Gauge> = Lazy::new(|| {
    register_gauge!(
        opts!(
            "supplychain_db_connections_active",
            "Number of active database connections"
        )
    )
    .expect("Failed to register db_connections_active metric")
});

/// Histogram for lineage chain depth
pub static LINEAGE_DEPTH: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "supplychain_lineage_depth",
        "Depth of lineage chains",
        &["batch_id_prefix"],
        vec![1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    )
    .expect("Failed to register lineage_depth metric")
});

/// Counter for validation errors
pub static VALIDATION_ERRORS: Lazy<CounterVec> = Lazy::new(|| {
    register_counter_vec!(
        opts!(
            "supplychain_validation_errors_total",
            "Total validation errors"
        ),
        &["error_type"]
    )
    .expect("Failed to register validation_errors metric")
});

/// Histogram for query result counts
pub static QUERY_RESULTS_COUNT: Lazy<HistogramVec> = Lazy::new(|| {
    register_histogram_vec!(
        "supplychain_query_results_count",
        "Number of results returned by queries",
        &["query_type"],
        vec![1.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0]
    )
    .expect("Failed to register query_results_count metric")
});

/// Helper struct for timing operations
pub struct MetricTimer {
    start: Instant,
}

impl MetricTimer {
    pub fn new() -> Self {
        Self {
            start: Instant::now(),
        }
    }

    /// Record the duration to a histogram
    pub fn observe(self, histogram: &HistogramVec, labels: &[&str]) {
        let duration = self.start.elapsed().as_secs_f64();
        histogram.with_label_values(labels).observe(duration);
    }
}

/// Record an event ingestion
pub fn record_event_ingested(event_type: &str) {
    EVENTS_INGESTED.with_label_values(&[event_type]).inc();
    CLAIMS_STORED.inc();
}

/// Record API request
pub fn record_api_request(method: &str, endpoint: &str, status: u16, duration_secs: f64) {
    API_REQUEST_DURATION
        .with_label_values(&[method, endpoint, &status.to_string()])
        .observe(duration_secs);
}

/// Record database query
pub fn record_db_query(operation: &str, duration_secs: f64) {
    DB_QUERY_DURATION
        .with_label_values(&[operation])
        .observe(duration_secs);
}

/// Record lineage depth
pub fn record_lineage_depth(batch_id_prefix: &str, depth: usize) {
    LINEAGE_DEPTH
        .with_label_values(&[batch_id_prefix])
        .observe(depth as f64);
}

/// Record validation error
pub fn record_validation_error(error_type: &str) {
    VALIDATION_ERRORS.with_label_values(&[error_type]).inc();
}

/// Get metrics in Prometheus text format
pub fn get_metrics() -> Result<String, Box<dyn std::error::Error>> {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    Ok(String::from_utf8(buffer)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_recording() {
        // Test that metrics can be recorded without panicking
        record_event_ingested("PRODUCED");
        record_api_request("POST", "/v1/events", 201, 0.05);
        record_db_query("store_claim", 0.001);
        record_lineage_depth("BATCH", 5);
        record_validation_error("invalid_quantity");

        // Test metrics retrieval
        let metrics_text = get_metrics().expect("Failed to get metrics");
        assert!(metrics_text.contains("supplychain_events_ingested_total"));
    }

    #[test]
    fn test_metric_timer() {
        let timer = MetricTimer::new();
        std::thread::sleep(std::time::Duration::from_millis(10));
        // Just verify it doesn't panic
        timer.observe(&DB_QUERY_DURATION, &["test"]);
    }
}
