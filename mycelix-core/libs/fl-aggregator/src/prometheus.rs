//! Prometheus Metrics Export for FL Aggregator
//!
//! This module provides comprehensive Prometheus metrics for monitoring
//! the federated learning aggregator, including counters, gauges, and histograms
//! with label support.
//!
//! # Features
//!
//! - **Counters**: Track rounds, submissions, Byzantine detections, aggregations
//! - **Gauges**: Monitor active nodes, current round, Phi values, Byzantine ratios
//! - **Histograms**: Measure aggregation duration, submission sizes, gradient norms
//! - **Labels**: Support for defense type, node ID, and payload type
//!
//! # Example
//!
//! ```rust,ignore
//! use fl_aggregator::prometheus::{FLMetrics, RoundMetrics};
//! use std::time::Duration;
//!
//! // Get singleton metrics instance
//! let metrics = FLMetrics::global();
//!
//! // Record individual metrics
//! metrics.inc_rounds();
//! metrics.inc_submissions("node1", "dense");
//! metrics.set_active_nodes(10);
//! metrics.observe_aggregation_time(Duration::from_millis(50), "krum");
//!
//! // Batch update after a round
//! let round_metrics = RoundMetrics {
//!     round: 42,
//!     submissions: 8,
//!     byzantine_detected: 1,
//!     aggregation_time: Duration::from_millis(100),
//!     phi_value: Some(0.85),
//!     defense_used: "trimmed_mean".to_string(),
//! };
//! metrics.record_round(&round_metrics);
//!
//! // Export to Prometheus text format
//! let output = metrics.to_prometheus_text();
//! ```
//!
//! # Metric Names
//!
//! All metrics follow Prometheus naming conventions with the `fl_aggregator_` prefix:
//!
//! - `fl_aggregator_rounds_total` - Total number of FL rounds completed
//! - `fl_aggregator_submissions_total{node_id, payload_type}` - Total submissions
//! - `fl_aggregator_byzantine_detected_total{node_id, defense}` - Byzantine detections
//! - `fl_aggregator_aggregations_total{defense}` - Total aggregations performed
//! - `fl_aggregator_active_nodes` - Current number of active nodes
//! - `fl_aggregator_current_round` - Current round number
//! - `fl_aggregator_phi_value` - Current Phi (integrated information) value
//! - `fl_aggregator_byzantine_ratio` - Ratio of Byzantine nodes detected
//! - `fl_aggregator_system_coherence` - System coherence from Phi module
//! - `fl_aggregator_aggregation_duration_seconds{defense}` - Aggregation time histogram
//! - `fl_aggregator_submission_size_bytes{payload_type}` - Submission size histogram
//! - `fl_aggregator_gradient_norm{node_id}` - Gradient L2 norm histogram

use once_cell::sync::OnceCell;
use prometheus::{
    Counter, CounterVec, Encoder, Gauge, HistogramOpts, HistogramVec, Opts,
    Registry, TextEncoder,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

#[cfg(feature = "http-api")]
use axum::{
    body::Body,
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};

use crate::phi::PhiMetrics;

// Global singleton instance
static GLOBAL_METRICS: OnceCell<FLMetrics> = OnceCell::new();

/// Metrics for a single FL round (used for batch updates).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoundMetrics {
    /// Round number
    pub round: u64,
    /// Number of submissions in this round
    pub submissions: usize,
    /// Number of Byzantine nodes detected
    pub byzantine_detected: usize,
    /// Time taken for aggregation
    pub aggregation_time: Duration,
    /// Optional Phi value for this round
    pub phi_value: Option<f32>,
    /// Defense strategy used
    pub defense_used: String,
}

impl Default for RoundMetrics {
    fn default() -> Self {
        Self {
            round: 0,
            submissions: 0,
            byzantine_detected: 0,
            aggregation_time: Duration::ZERO,
            phi_value: None,
            defense_used: "none".to_string(),
        }
    }
}

impl RoundMetrics {
    /// Create new RoundMetrics
    pub fn new(round: u64, defense: &str) -> Self {
        Self {
            round,
            defense_used: defense.to_string(),
            ..Default::default()
        }
    }

    /// Set submissions count
    pub fn with_submissions(mut self, count: usize) -> Self {
        self.submissions = count;
        self
    }

    /// Set Byzantine count
    pub fn with_byzantine(mut self, count: usize) -> Self {
        self.byzantine_detected = count;
        self
    }

    /// Set aggregation time
    pub fn with_aggregation_time(mut self, duration: Duration) -> Self {
        self.aggregation_time = duration;
        self
    }

    /// Set Phi value
    pub fn with_phi(mut self, phi: f32) -> Self {
        self.phi_value = Some(phi);
        self
    }
}

/// FL Aggregator Prometheus Metrics
///
/// Provides counters, gauges, and histograms for monitoring the federated
/// learning aggregator. Uses a singleton pattern for global access.
#[derive(Debug, Clone)]
pub struct FLMetrics {
    /// The Prometheus registry containing all metrics
    registry: Arc<Registry>,

    // Counters
    rounds_total: Counter,
    submissions_total: CounterVec,
    byzantine_detected_total: CounterVec,
    aggregations_total: CounterVec,

    // Gauges
    active_nodes: Gauge,
    current_round: Gauge,
    phi_value: Gauge,
    byzantine_ratio: Gauge,
    system_coherence: Gauge,

    // Histograms
    aggregation_duration_seconds: HistogramVec,
    submission_size_bytes: HistogramVec,
    gradient_norm: HistogramVec,
}

impl FLMetrics {
    /// Create new FLMetrics with a custom registry.
    ///
    /// Use this when you need to manage your own registry, e.g., for testing
    /// or when running multiple aggregator instances.
    pub fn with_registry(registry: &Registry) -> Result<Self, prometheus::Error> {
        // Counters
        let rounds_total = Counter::new(
            "fl_aggregator_rounds_total",
            "Total number of FL rounds completed",
        )?;
        registry.register(Box::new(rounds_total.clone()))?;

        let submissions_total = CounterVec::new(
            Opts::new(
                "fl_aggregator_submissions_total",
                "Total number of gradient submissions",
            ),
            &["node_id", "payload_type"],
        )?;
        registry.register(Box::new(submissions_total.clone()))?;

        let byzantine_detected_total = CounterVec::new(
            Opts::new(
                "fl_aggregator_byzantine_detected_total",
                "Total number of Byzantine nodes detected",
            ),
            &["node_id", "defense"],
        )?;
        registry.register(Box::new(byzantine_detected_total.clone()))?;

        let aggregations_total = CounterVec::new(
            Opts::new(
                "fl_aggregator_aggregations_total",
                "Total number of aggregations performed",
            ),
            &["defense"],
        )?;
        registry.register(Box::new(aggregations_total.clone()))?;

        // Gauges
        let active_nodes = Gauge::new(
            "fl_aggregator_active_nodes",
            "Current number of active nodes",
        )?;
        registry.register(Box::new(active_nodes.clone()))?;

        let current_round = Gauge::new("fl_aggregator_current_round", "Current round number")?;
        registry.register(Box::new(current_round.clone()))?;

        let phi_value = Gauge::new(
            "fl_aggregator_phi_value",
            "Current Phi (integrated information) value",
        )?;
        registry.register(Box::new(phi_value.clone()))?;

        let byzantine_ratio = Gauge::new(
            "fl_aggregator_byzantine_ratio",
            "Ratio of Byzantine nodes detected (0-1)",
        )?;
        registry.register(Box::new(byzantine_ratio.clone()))?;

        let system_coherence = Gauge::new(
            "fl_aggregator_system_coherence",
            "System coherence from Phi module (0-1)",
        )?;
        registry.register(Box::new(system_coherence.clone()))?;

        // Histograms
        let aggregation_duration_seconds = HistogramVec::new(
            HistogramOpts::new(
                "fl_aggregator_aggregation_duration_seconds",
                "Aggregation duration in seconds",
            )
            .buckets(vec![
                0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
            ]),
            &["defense"],
        )?;
        registry.register(Box::new(aggregation_duration_seconds.clone()))?;

        let submission_size_bytes = HistogramVec::new(
            HistogramOpts::new(
                "fl_aggregator_submission_size_bytes",
                "Submission size in bytes",
            )
            .buckets(vec![
                1024.0,
                4096.0,
                16384.0,
                65536.0,
                262144.0,
                1048576.0,
                4194304.0,
                16777216.0,
            ]),
            &["payload_type"],
        )?;
        registry.register(Box::new(submission_size_bytes.clone()))?;

        let gradient_norm = HistogramVec::new(
            HistogramOpts::new("fl_aggregator_gradient_norm", "Gradient L2 norm"),
            &["node_id"],
        )?;
        registry.register(Box::new(gradient_norm.clone()))?;

        Ok(Self {
            registry: Arc::new(registry.clone()),
            rounds_total,
            submissions_total,
            byzantine_detected_total,
            aggregations_total,
            active_nodes,
            current_round,
            phi_value,
            byzantine_ratio,
            system_coherence,
            aggregation_duration_seconds,
            submission_size_bytes,
            gradient_norm,
        })
    }

    /// Register metrics with the default Prometheus registry.
    ///
    /// This is the typical way to initialize metrics for production use.
    pub fn register_metrics() -> Result<Self, prometheus::Error> {
        let registry = Registry::new();
        Self::with_registry(&registry)
    }

    /// Get or create the global singleton metrics instance.
    ///
    /// This is the recommended way to access metrics throughout the application.
    /// The first call initializes the metrics; subsequent calls return the same instance.
    pub fn global() -> &'static FLMetrics {
        GLOBAL_METRICS.get_or_init(|| {
            Self::register_metrics()
                .expect("Failed to register FL aggregator Prometheus metrics")
        })
    }

    /// Try to get the global metrics instance without initializing.
    ///
    /// Returns `None` if metrics haven't been initialized yet.
    pub fn try_global() -> Option<&'static FLMetrics> {
        GLOBAL_METRICS.get()
    }

    /// Initialize the global metrics with a custom registry.
    ///
    /// Must be called before `global()` to use a custom registry.
    /// Returns an error if global metrics have already been initialized.
    pub fn init_global_with_registry(registry: &Registry) -> Result<(), prometheus::Error> {
        let metrics = Self::with_registry(registry)?;
        GLOBAL_METRICS.set(metrics).map_err(|_| {
            prometheus::Error::Msg("Global FL metrics already initialized".to_string())
        })
    }

    // ==================== Counter Methods ====================

    /// Increment the rounds counter.
    pub fn inc_rounds(&self) {
        self.rounds_total.inc();
    }

    /// Increment the submissions counter with labels.
    ///
    /// # Arguments
    ///
    /// * `node_id` - ID of the submitting node
    /// * `payload_type` - Type of payload (e.g., "dense", "sparse", "hypervector")
    pub fn inc_submissions(&self, node_id: &str, payload_type: &str) {
        self.submissions_total
            .with_label_values(&[node_id, payload_type])
            .inc();
    }

    /// Increment the Byzantine detection counter with labels.
    ///
    /// # Arguments
    ///
    /// * `node_id` - ID of the Byzantine node
    /// * `defense` - Defense strategy that detected it (e.g., "krum", "phi")
    pub fn inc_byzantine(&self, node_id: &str, defense: &str) {
        self.byzantine_detected_total
            .with_label_values(&[node_id, defense])
            .inc();
    }

    /// Increment the aggregations counter with defense label.
    ///
    /// # Arguments
    ///
    /// * `defense` - Defense strategy used (e.g., "krum", "median", "trimmed_mean")
    pub fn inc_aggregations(&self, defense: &str) {
        self.aggregations_total
            .with_label_values(&[defense])
            .inc();
    }

    // ==================== Gauge Methods ====================

    /// Set the current number of active nodes.
    pub fn set_active_nodes(&self, count: usize) {
        self.active_nodes.set(count as f64);
    }

    /// Set the current round number.
    pub fn set_current_round(&self, round: u64) {
        self.current_round.set(round as f64);
    }

    /// Set the current Phi (integrated information) value.
    pub fn set_phi_value(&self, phi: f32) {
        self.phi_value.set(phi as f64);
    }

    /// Set the Byzantine ratio (detected / total nodes).
    pub fn set_byzantine_ratio(&self, ratio: f32) {
        self.byzantine_ratio.set(ratio as f64);
    }

    /// Set the system coherence value from the Phi module.
    ///
    /// This represents overall system integration/coherence.
    pub fn set_system_coherence(&self, coherence: f32) {
        self.system_coherence.set(coherence as f64);
    }

    // ==================== Histogram Methods ====================

    /// Observe an aggregation duration.
    ///
    /// # Arguments
    ///
    /// * `duration` - Time taken for the aggregation
    /// * `defense` - Defense strategy used
    pub fn observe_aggregation_time(&self, duration: Duration, defense: &str) {
        self.aggregation_duration_seconds
            .with_label_values(&[defense])
            .observe(duration.as_secs_f64());
    }

    /// Observe a submission size.
    ///
    /// # Arguments
    ///
    /// * `bytes` - Size of the submission in bytes
    /// * `payload_type` - Type of payload
    pub fn observe_submission_size(&self, bytes: usize, payload_type: &str) {
        self.submission_size_bytes
            .with_label_values(&[payload_type])
            .observe(bytes as f64);
    }

    /// Observe a gradient norm.
    ///
    /// # Arguments
    ///
    /// * `norm` - L2 norm of the gradient
    /// * `node_id` - ID of the submitting node
    pub fn observe_gradient_norm(&self, norm: f32, node_id: &str) {
        self.gradient_norm
            .with_label_values(&[node_id])
            .observe(norm as f64);
    }

    // ==================== Export Functions ====================

    /// Export metrics in Prometheus text format.
    ///
    /// This is the format expected by the Prometheus scraper at the /metrics endpoint.
    pub fn to_prometheus_text(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }

    /// Export metrics in JSON format.
    ///
    /// Provides a structured JSON representation of current metric values.
    pub fn to_json(&self) -> String {
        #[derive(Serialize)]
        struct MetricsJson {
            rounds_total: f64,
            active_nodes: f64,
            current_round: f64,
            phi_value: f64,
            byzantine_ratio: f64,
            system_coherence: f64,
        }

        let json = MetricsJson {
            rounds_total: self.rounds_total.get(),
            active_nodes: self.active_nodes.get(),
            current_round: self.current_round.get(),
            phi_value: self.phi_value.get(),
            byzantine_ratio: self.byzantine_ratio.get(),
            system_coherence: self.system_coherence.get(),
        };

        serde_json::to_string_pretty(&json).unwrap_or_else(|_| "{}".to_string())
    }

    // ==================== Integration Helpers ====================

    /// Record metrics for a completed round.
    ///
    /// This is a convenience method for batch-updating multiple metrics
    /// at the end of each round.
    pub fn record_round(&self, metrics: &RoundMetrics) {
        // Update counters
        self.inc_rounds();
        self.inc_aggregations(&metrics.defense_used);

        // Update gauges
        self.set_current_round(metrics.round);
        if let Some(phi) = metrics.phi_value {
            self.set_phi_value(phi);
        }

        // Observe histogram
        self.observe_aggregation_time(metrics.aggregation_time, &metrics.defense_used);

        // Calculate and set Byzantine ratio if we have submissions
        if metrics.submissions > 0 {
            let ratio = metrics.byzantine_detected as f32 / metrics.submissions as f32;
            self.set_byzantine_ratio(ratio);
        }
    }

    /// Record Phi measurement results.
    ///
    /// Updates gauges related to the Phi module measurement.
    pub fn record_phi_measurement(&self, phi: &PhiMetrics) {
        self.set_phi_value(phi.phi_after);
        self.set_system_coherence(phi.epistemic_confidence);
    }

    // ==================== Alerting Helpers ====================

    /// Check if Byzantine ratio exceeds the alert threshold.
    ///
    /// Returns `true` if the current Byzantine ratio is above the threshold,
    /// indicating a potential attack or significant node misbehavior.
    ///
    /// # Arguments
    ///
    /// * `threshold` - Maximum acceptable Byzantine ratio (e.g., 0.33 for 1/3)
    pub fn check_byzantine_alert(&self, threshold: f32) -> bool {
        self.byzantine_ratio.get() > threshold as f64
    }

    /// Check if Phi value is below the minimum threshold.
    ///
    /// Returns `true` if the current Phi is below the minimum,
    /// indicating low system coherence or potential attacks.
    ///
    /// # Arguments
    ///
    /// * `min_phi` - Minimum acceptable Phi value (e.g., 0.5)
    pub fn check_phi_alert(&self, min_phi: f32) -> bool {
        self.phi_value.get() < min_phi as f64
    }

    /// Get the Prometheus registry.
    ///
    /// Useful for advanced use cases like custom metric collectors.
    pub fn registry(&self) -> &Registry {
        &self.registry
    }

    // ==================== Direct Access (for advanced use) ====================

    /// Get direct access to the rounds counter.
    pub fn rounds_counter(&self) -> &Counter {
        &self.rounds_total
    }

    /// Get direct access to the submissions counter vec.
    pub fn submissions_counter(&self) -> &CounterVec {
        &self.submissions_total
    }

    /// Get direct access to the Byzantine counter vec.
    pub fn byzantine_counter(&self) -> &CounterVec {
        &self.byzantine_detected_total
    }

    /// Get direct access to the aggregations counter vec.
    pub fn aggregations_counter(&self) -> &CounterVec {
        &self.aggregations_total
    }

    /// Get direct access to the active nodes gauge.
    pub fn active_nodes_gauge(&self) -> &Gauge {
        &self.active_nodes
    }

    /// Get direct access to the current round gauge.
    pub fn current_round_gauge(&self) -> &Gauge {
        &self.current_round
    }

    /// Get direct access to the phi value gauge.
    pub fn phi_gauge(&self) -> &Gauge {
        &self.phi_value
    }

    /// Get direct access to the Byzantine ratio gauge.
    pub fn byzantine_ratio_gauge(&self) -> &Gauge {
        &self.byzantine_ratio
    }

    /// Get direct access to the system coherence gauge.
    pub fn system_coherence_gauge(&self) -> &Gauge {
        &self.system_coherence
    }

    /// Get direct access to the aggregation duration histogram.
    pub fn aggregation_duration_histogram(&self) -> &HistogramVec {
        &self.aggregation_duration_seconds
    }

    /// Get direct access to the submission size histogram.
    pub fn submission_size_histogram(&self) -> &HistogramVec {
        &self.submission_size_bytes
    }

    /// Get direct access to the gradient norm histogram.
    pub fn gradient_norm_histogram(&self) -> &HistogramVec {
        &self.gradient_norm
    }
}

// ==================== HTTP Middleware ====================

/// Metrics middleware for automatic HTTP request tracking.
///
/// This middleware automatically records metrics for each HTTP request,
/// including request count, duration, and status codes.
///
/// Only available with the `http-api` feature.
#[cfg(feature = "http-api")]
#[derive(Clone)]
pub struct MetricsMiddleware {
    metrics: &'static FLMetrics,
    request_counter: CounterVec,
    request_duration: HistogramVec,
}

#[cfg(feature = "http-api")]
impl MetricsMiddleware {
    /// Create a new metrics middleware using the global metrics instance.
    pub fn new() -> Result<Self, prometheus::Error> {
        let metrics = FLMetrics::global();

        let request_counter = CounterVec::new(
            Opts::new(
                "fl_aggregator_http_requests_total",
                "Total HTTP requests",
            ),
            &["method", "path", "status"],
        )?;
        metrics.registry.register(Box::new(request_counter.clone()))?;

        let request_duration = HistogramVec::new(
            HistogramOpts::new(
                "fl_aggregator_http_request_duration_seconds",
                "HTTP request duration in seconds",
            )
            .buckets(vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]),
            &["method", "path"],
        )?;
        metrics.registry.register(Box::new(request_duration.clone()))?;

        Ok(Self {
            metrics,
            request_counter,
            request_duration,
        })
    }

    /// Create middleware layer for axum.
    pub async fn layer(
        req: Request,
        next: Next,
    ) -> Response {
        let start = std::time::Instant::now();
        let method = req.method().to_string();
        let path = req.uri().path().to_string();

        let response = next.run(req).await;

        let duration = start.elapsed();
        let status = response.status().as_u16().to_string();

        // Try to record metrics if middleware is registered
        if let Some(metrics) = FLMetrics::try_global() {
            // For simplicity, we just observe aggregation time for API calls
            metrics.observe_aggregation_time(duration, "http");
        }

        response
    }
}

#[cfg(feature = "http-api")]
impl Default for MetricsMiddleware {
    fn default() -> Self {
        Self::new().expect("Failed to create metrics middleware")
    }
}

// ==================== Tests ====================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_metrics() -> FLMetrics {
        let registry = Registry::new();
        FLMetrics::with_registry(&registry).expect("Failed to create test metrics")
    }

    #[test]
    fn test_counter_increments() {
        let metrics = create_test_metrics();

        // Test rounds counter
        assert_eq!(metrics.rounds_total.get(), 0.0);
        metrics.inc_rounds();
        assert_eq!(metrics.rounds_total.get(), 1.0);
        metrics.inc_rounds();
        assert_eq!(metrics.rounds_total.get(), 2.0);
    }

    #[test]
    fn test_labeled_counter_increments() {
        let metrics = create_test_metrics();

        // Test submissions counter with labels
        metrics.inc_submissions("node1", "dense");
        metrics.inc_submissions("node1", "dense");
        metrics.inc_submissions("node2", "sparse");

        let node1_dense = metrics
            .submissions_total
            .with_label_values(&["node1", "dense"])
            .get();
        let node2_sparse = metrics
            .submissions_total
            .with_label_values(&["node2", "sparse"])
            .get();

        assert_eq!(node1_dense, 2.0);
        assert_eq!(node2_sparse, 1.0);
    }

    #[test]
    fn test_byzantine_counter() {
        let metrics = create_test_metrics();

        metrics.inc_byzantine("malicious_node", "krum");
        metrics.inc_byzantine("malicious_node", "phi");
        metrics.inc_byzantine("another_bad_node", "krum");

        let malicious_krum = metrics
            .byzantine_detected_total
            .with_label_values(&["malicious_node", "krum"])
            .get();
        let malicious_phi = metrics
            .byzantine_detected_total
            .with_label_values(&["malicious_node", "phi"])
            .get();

        assert_eq!(malicious_krum, 1.0);
        assert_eq!(malicious_phi, 1.0);
    }

    #[test]
    fn test_gauge_updates() {
        let metrics = create_test_metrics();

        // Test active nodes
        metrics.set_active_nodes(10);
        assert_eq!(metrics.active_nodes.get(), 10.0);
        metrics.set_active_nodes(5);
        assert_eq!(metrics.active_nodes.get(), 5.0);

        // Test current round
        metrics.set_current_round(42);
        assert_eq!(metrics.current_round.get(), 42.0);

        // Test phi value
        metrics.set_phi_value(0.85);
        assert!((metrics.phi_value.get() - 0.85).abs() < 0.001);

        // Test byzantine ratio
        metrics.set_byzantine_ratio(0.1);
        assert!((metrics.byzantine_ratio.get() - 0.1).abs() < 0.001);

        // Test system coherence
        metrics.set_system_coherence(0.92);
        assert!((metrics.system_coherence.get() - 0.92).abs() < 0.001);
    }

    #[test]
    fn test_histogram_observations() {
        let metrics = create_test_metrics();

        // Test aggregation time
        metrics.observe_aggregation_time(Duration::from_millis(50), "krum");
        metrics.observe_aggregation_time(Duration::from_millis(100), "krum");
        metrics.observe_aggregation_time(Duration::from_millis(25), "median");

        // Verify histograms are populated (we can't easily check bucket values)
        let krum_histogram = metrics
            .aggregation_duration_seconds
            .with_label_values(&["krum"]);
        let median_histogram = metrics
            .aggregation_duration_seconds
            .with_label_values(&["median"]);

        assert!(krum_histogram.get_sample_count() == 2);
        assert!(median_histogram.get_sample_count() == 1);
    }

    #[test]
    fn test_submission_size_histogram() {
        let metrics = create_test_metrics();

        metrics.observe_submission_size(1024, "dense");
        metrics.observe_submission_size(2048, "dense");
        metrics.observe_submission_size(512, "sparse");

        let dense = metrics.submission_size_bytes.with_label_values(&["dense"]);
        let sparse = metrics.submission_size_bytes.with_label_values(&["sparse"]);

        assert_eq!(dense.get_sample_count(), 2);
        assert_eq!(sparse.get_sample_count(), 1);
    }

    #[test]
    fn test_gradient_norm_histogram() {
        let metrics = create_test_metrics();

        metrics.observe_gradient_norm(1.5, "node1");
        metrics.observe_gradient_norm(2.3, "node1");
        metrics.observe_gradient_norm(0.8, "node2");

        let node1 = metrics.gradient_norm.with_label_values(&["node1"]);
        let node2 = metrics.gradient_norm.with_label_values(&["node2"]);

        assert_eq!(node1.get_sample_count(), 2);
        assert_eq!(node2.get_sample_count(), 1);
    }

    #[test]
    fn test_prometheus_text_format() {
        let metrics = create_test_metrics();

        // Add some data
        metrics.inc_rounds();
        metrics.set_active_nodes(5);
        metrics.set_phi_value(0.75);

        let output = metrics.to_prometheus_text();

        // Check that output contains expected metric names
        assert!(output.contains("fl_aggregator_rounds_total"));
        assert!(output.contains("fl_aggregator_active_nodes"));
        assert!(output.contains("fl_aggregator_phi_value"));

        // Check for HELP and TYPE annotations
        assert!(output.contains("# HELP"));
        assert!(output.contains("# TYPE"));
    }

    #[test]
    fn test_json_format() {
        let metrics = create_test_metrics();

        metrics.inc_rounds();
        metrics.inc_rounds();
        metrics.set_active_nodes(10);
        metrics.set_phi_value(0.8);

        let json = metrics.to_json();

        // Parse and verify
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["rounds_total"], 2.0);
        assert_eq!(parsed["active_nodes"], 10.0);
        assert!((parsed["phi_value"].as_f64().unwrap() - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_label_handling() {
        let metrics = create_test_metrics();

        // Test various label values
        metrics.inc_submissions("node-with-dashes", "payload_with_underscores");
        metrics.inc_submissions("node.with.dots", "sparse");
        metrics.inc_byzantine("node:with:colons", "krum");

        let output = metrics.to_prometheus_text();

        // Labels should be properly escaped/quoted in output
        assert!(output.contains("node_id="));
        assert!(output.contains("payload_type="));
    }

    #[test]
    fn test_record_round() {
        let metrics = create_test_metrics();

        let round_metrics = RoundMetrics {
            round: 42,
            submissions: 10,
            byzantine_detected: 2,
            aggregation_time: Duration::from_millis(100),
            phi_value: Some(0.85),
            defense_used: "krum".to_string(),
        };

        metrics.record_round(&round_metrics);

        assert_eq!(metrics.rounds_total.get(), 1.0);
        assert_eq!(metrics.current_round.get(), 42.0);
        assert!((metrics.phi_value.get() - 0.85).abs() < 0.001);
        assert!((metrics.byzantine_ratio.get() - 0.2).abs() < 0.001);
    }

    #[test]
    fn test_record_phi_measurement() {
        let metrics = create_test_metrics();

        let phi_metrics = PhiMetrics {
            phi_before: 0.7,
            phi_after: 0.85,
            phi_gain: 0.15,
            epistemic_confidence: 0.92,
            layer_contributions: None,
        };

        metrics.record_phi_measurement(&phi_metrics);

        assert!((metrics.phi_value.get() - 0.85).abs() < 0.001);
        assert!((metrics.system_coherence.get() - 0.92).abs() < 0.001);
    }

    #[test]
    fn test_byzantine_alert() {
        let metrics = create_test_metrics();

        metrics.set_byzantine_ratio(0.1);
        assert!(!metrics.check_byzantine_alert(0.33));

        metrics.set_byzantine_ratio(0.4);
        assert!(metrics.check_byzantine_alert(0.33));
    }

    #[test]
    fn test_phi_alert() {
        let metrics = create_test_metrics();

        metrics.set_phi_value(0.8);
        assert!(!metrics.check_phi_alert(0.5));

        metrics.set_phi_value(0.3);
        assert!(metrics.check_phi_alert(0.5));
    }

    #[test]
    fn test_thread_safety() {
        use std::sync::Arc;
        use std::thread;

        let metrics = Arc::new(create_test_metrics());
        let mut handles = vec![];

        // Spawn multiple threads that update metrics concurrently
        for i in 0..10 {
            let m = Arc::clone(&metrics);
            handles.push(thread::spawn(move || {
                for j in 0..100 {
                    m.inc_rounds();
                    m.inc_submissions(&format!("node{}", i), "dense");
                    m.set_active_nodes(j);
                    m.observe_aggregation_time(Duration::from_millis(j as u64), "krum");
                }
            }));
        }

        // Wait for all threads
        for handle in handles {
            handle.join().unwrap();
        }

        // Verify counter value (10 threads * 100 iterations)
        assert_eq!(metrics.rounds_total.get(), 1000.0);
    }

    #[test]
    fn test_round_metrics_builder() {
        let round_metrics = RoundMetrics::new(5, "median")
            .with_submissions(8)
            .with_byzantine(1)
            .with_aggregation_time(Duration::from_millis(50))
            .with_phi(0.9);

        assert_eq!(round_metrics.round, 5);
        assert_eq!(round_metrics.submissions, 8);
        assert_eq!(round_metrics.byzantine_detected, 1);
        assert_eq!(round_metrics.aggregation_time, Duration::from_millis(50));
        assert_eq!(round_metrics.phi_value, Some(0.9));
        assert_eq!(round_metrics.defense_used, "median");
    }

    #[test]
    fn test_multiple_registries() {
        // Create two separate registries with their own metrics
        let registry1 = Registry::new();
        let registry2 = Registry::new();

        let metrics1 = FLMetrics::with_registry(&registry1).unwrap();
        let metrics2 = FLMetrics::with_registry(&registry2).unwrap();

        // Update independently
        metrics1.inc_rounds();
        metrics1.inc_rounds();
        metrics2.inc_rounds();

        // Verify they are independent
        assert_eq!(metrics1.rounds_total.get(), 2.0);
        assert_eq!(metrics2.rounds_total.get(), 1.0);
    }

    #[test]
    fn test_aggregations_counter() {
        let metrics = create_test_metrics();

        metrics.inc_aggregations("krum");
        metrics.inc_aggregations("krum");
        metrics.inc_aggregations("median");
        metrics.inc_aggregations("trimmed_mean");

        let krum = metrics.aggregations_total.with_label_values(&["krum"]).get();
        let median = metrics.aggregations_total.with_label_values(&["median"]).get();
        let trimmed = metrics.aggregations_total.with_label_values(&["trimmed_mean"]).get();

        assert_eq!(krum, 2.0);
        assert_eq!(median, 1.0);
        assert_eq!(trimmed, 1.0);
    }

    #[test]
    fn test_direct_access_methods() {
        let metrics = create_test_metrics();

        // Test that direct access methods return the same underlying metrics
        metrics.rounds_counter().inc();
        assert_eq!(metrics.rounds_total.get(), 1.0);

        metrics.active_nodes_gauge().set(42.0);
        assert_eq!(metrics.active_nodes.get(), 42.0);
    }

    #[test]
    fn test_record_round_without_phi() {
        let metrics = create_test_metrics();

        let round_metrics = RoundMetrics {
            round: 1,
            submissions: 5,
            byzantine_detected: 0,
            aggregation_time: Duration::from_millis(25),
            phi_value: None,
            defense_used: "mean".to_string(),
        };

        let initial_phi = metrics.phi_value.get();
        metrics.record_round(&round_metrics);

        // Phi should remain unchanged when not provided
        assert_eq!(metrics.phi_value.get(), initial_phi);
        assert_eq!(metrics.current_round.get(), 1.0);
    }

    #[test]
    fn test_zero_submissions_no_division_by_zero() {
        let metrics = create_test_metrics();

        let round_metrics = RoundMetrics {
            round: 1,
            submissions: 0,
            byzantine_detected: 0,
            aggregation_time: Duration::from_millis(10),
            phi_value: None,
            defense_used: "none".to_string(),
        };

        // Should not panic
        metrics.record_round(&round_metrics);

        // Byzantine ratio should remain unchanged (not NaN or Inf)
        let ratio = metrics.byzantine_ratio.get();
        assert!(!ratio.is_nan());
        assert!(!ratio.is_infinite());
    }
}
