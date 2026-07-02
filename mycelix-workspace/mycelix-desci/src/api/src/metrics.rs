// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Prometheus Metrics
//!
//! Comprehensive metrics collection for monitoring and observability

use lazy_static::lazy_static;
use prometheus::{
    CounterVec, Encoder, HistogramVec, IntCounterVec, IntGauge, TextEncoder, register_counter_vec,
    register_histogram_vec, register_int_counter_vec, register_int_gauge,
};

lazy_static! {
    /// Total HTTP requests counter
    pub static ref HTTP_REQUESTS_TOTAL: IntCounterVec = register_int_counter_vec!(
        "mycelix_http_requests_total",
        "Total number of HTTP requests",
        &["method", "endpoint", "status"]
    )
    .expect("Failed to create HTTP_REQUESTS_TOTAL metric");

    /// HTTP request duration histogram
    pub static ref HTTP_REQUEST_DURATION: HistogramVec = register_histogram_vec!(
        "mycelix_http_request_duration_seconds",
        "HTTP request duration in seconds",
        &["method", "endpoint"],
        vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    )
    .expect("Failed to create HTTP_REQUEST_DURATION metric");

    /// Claims created counter
    pub static ref CLAIMS_CREATED: IntCounterVec = register_int_counter_vec!(
        "mycelix_claims_created_total",
        "Total number of claims created",
        &["tier"]
    )
    .expect("Failed to create CLAIMS_CREATED metric");

    /// Verifications added counter
    pub static ref VERIFICATIONS_ADDED: IntCounterVec = register_int_counter_vec!(
        "mycelix_verifications_added_total",
        "Total number of verifications added",
        &["claim_tier"]
    )
    .expect("Failed to create VERIFICATIONS_ADDED metric");

    /// Query operations counter
    pub static ref QUERY_OPERATIONS: IntCounterVec = register_int_counter_vec!(
        "mycelix_query_operations_total",
        "Total number of query operations",
        &["filter_type"]
    )
    .expect("Failed to create QUERY_OPERATIONS metric");

    /// Query duration histogram
    pub static ref QUERY_DURATION: HistogramVec = register_histogram_vec!(
        "mycelix_query_duration_seconds",
        "Query operation duration in seconds",
        &["filter_type"],
        vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    )
    .expect("Failed to create QUERY_DURATION metric");

    /// Trust score updates counter
    pub static ref TRUST_UPDATES: IntCounterVec = register_int_counter_vec!(
        "mycelix_trust_updates_total",
        "Total number of trust score updates",
        &["participant_type"]
    )
    .expect("Failed to create TRUST_UPDATES metric");

    /// Active claims gauge
    pub static ref ACTIVE_CLAIMS: IntGauge = register_int_gauge!(
        "mycelix_active_claims",
        "Current number of active claims in the system"
    )
    .expect("Failed to create ACTIVE_CLAIMS metric");

    /// Storage operations counter
    pub static ref STORAGE_OPERATIONS: IntCounterVec = register_int_counter_vec!(
        "mycelix_storage_operations_total",
        "Total number of storage operations",
        &["operation", "status"]
    )
    .expect("Failed to create STORAGE_OPERATIONS metric");

    /// API errors counter
    pub static ref API_ERRORS: CounterVec = register_counter_vec!(
        "mycelix_api_errors_total",
        "Total number of API errors",
        &["endpoint", "error_type"]
    )
    .expect("Failed to create API_ERRORS metric");
}

/// Encode all metrics to Prometheus text format
pub fn encode_metrics() -> Result<String, prometheus::Error> {
    let encoder = TextEncoder::new();
    let metric_families = prometheus::gather();
    let mut buffer = Vec::new();
    encoder.encode(&metric_families, &mut buffer)?;
    String::from_utf8(buffer)
        .map_err(|e| prometheus::Error::Msg(format!("Failed to convert metrics to UTF-8: {}", e)))
}

/// Track HTTP request
pub fn track_http_request(method: &str, endpoint: &str, status: u16) {
    HTTP_REQUESTS_TOTAL
        .with_label_values(&[method, endpoint, &status.to_string()])
        .inc();
}

/// Track claim creation
pub fn track_claim_creation(tier: &str) {
    CLAIMS_CREATED.with_label_values(&[tier]).inc();
    ACTIVE_CLAIMS.inc();
}

/// Track verification addition
pub fn track_verification(claim_tier: &str) {
    VERIFICATIONS_ADDED.with_label_values(&[claim_tier]).inc();
}

/// Track query operation
pub fn track_query(filter_type: &str, duration_secs: f64) {
    QUERY_OPERATIONS.with_label_values(&[filter_type]).inc();
    QUERY_DURATION
        .with_label_values(&[filter_type])
        .observe(duration_secs);
}

/// Track trust update
pub fn track_trust_update(participant_type: &str) {
    TRUST_UPDATES.with_label_values(&[participant_type]).inc();
}

/// Track storage operation
pub fn track_storage_operation(operation: &str, success: bool) {
    let status = if success { "success" } else { "failure" };
    STORAGE_OPERATIONS
        .with_label_values(&[operation, status])
        .inc();
}

/// Track API error
pub fn track_api_error(endpoint: &str, error_type: &str) {
    API_ERRORS.with_label_values(&[endpoint, error_type]).inc();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_encoding() {
        track_http_request("GET", "/api/v1/claims", 200);
        track_claim_creation("E0");
        track_verification("E1");

        let result = encode_metrics();
        assert!(result.is_ok());
        let metrics = result.unwrap();
        assert!(metrics.contains("mycelix_http_requests_total"));
        assert!(metrics.contains("mycelix_claims_created_total"));
    }

    #[test]
    fn test_track_operations() {
        track_query("category", 0.001);
        track_trust_update("researcher");
        track_storage_operation("store", true);
        track_api_error("/api/v1/claims", "validation_error");

        let metrics = encode_metrics().unwrap();
        assert!(metrics.contains("mycelix_query_operations_total"));
        assert!(metrics.contains("mycelix_trust_updates_total"));
        assert!(metrics.contains("mycelix_storage_operations_total"));
        assert!(metrics.contains("mycelix_api_errors_total"));
    }
}
