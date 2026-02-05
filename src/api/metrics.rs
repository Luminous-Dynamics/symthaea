//! # Lightweight Metrics Infrastructure
//!
//! Provides Prometheus-compatible metrics without external dependencies.
//! Avoids OpenSSL/hyper complexity by using a simple in-memory registry.
//!
//! ## Usage
//!
//! ```rust
//! use symthaea::api::metrics::{MetricsRegistry, MetricType};
//!
//! let registry = MetricsRegistry::new();
//!
//! // Register metrics
//! registry.register("phi_calculations_total", MetricType::Counter, "Total Φ calculations");
//! registry.register("phi_value", MetricType::Gauge, "Current integrated information");
//!
//! // Update metrics
//! registry.increment("phi_calculations_total");
//! registry.set_gauge("phi_value", 0.847);
//!
//! // Export as Prometheus text format
//! let output = registry.to_prometheus_text();
//! ```

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Type of metric
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MetricType {
    /// Monotonically increasing counter
    Counter,
    /// Point-in-time value that can go up or down
    Gauge,
    /// Distribution of values (stores count, sum, buckets)
    Histogram,
}

/// A single metric with its metadata and value
#[derive(Debug)]
struct Metric {
    name: String,
    metric_type: MetricType,
    help: String,
    /// For counters and gauges, stored as f64 bits
    value: AtomicU64,
    /// For histograms: (count, sum, bucket_counts)
    histogram_data: Option<RwLock<HistogramData>>,
}

/// Histogram bucket data
#[derive(Debug, Default)]
struct HistogramData {
    count: u64,
    sum: f64,
    /// (upper_bound, count) pairs
    buckets: Vec<(f64, u64)>,
}

impl Metric {
    fn new_counter(name: &str, help: &str) -> Self {
        Self {
            name: name.to_string(),
            metric_type: MetricType::Counter,
            help: help.to_string(),
            value: AtomicU64::new(0),
            histogram_data: None,
        }
    }

    fn new_gauge(name: &str, help: &str) -> Self {
        Self {
            name: name.to_string(),
            metric_type: MetricType::Gauge,
            help: help.to_string(),
            value: AtomicU64::new(0),
            histogram_data: None,
        }
    }

    fn new_histogram(name: &str, help: &str, buckets: Vec<f64>) -> Self {
        let bucket_counts: Vec<(f64, u64)> = buckets.into_iter().map(|b| (b, 0)).collect();
        Self {
            name: name.to_string(),
            metric_type: MetricType::Histogram,
            help: help.to_string(),
            value: AtomicU64::new(0),
            histogram_data: Some(RwLock::new(HistogramData {
                count: 0,
                sum: 0.0,
                buckets: bucket_counts,
            })),
        }
    }

    fn increment(&self) {
        self.value.fetch_add(1, Ordering::Relaxed);
    }

    fn add(&self, delta: u64) {
        self.value.fetch_add(delta, Ordering::Relaxed);
    }

    fn set_gauge(&self, value: f64) {
        self.value.store(value.to_bits(), Ordering::Relaxed);
    }

    fn get_value(&self) -> f64 {
        match self.metric_type {
            MetricType::Counter => self.value.load(Ordering::Relaxed) as f64,
            MetricType::Gauge => f64::from_bits(self.value.load(Ordering::Relaxed)),
            MetricType::Histogram => self
                .histogram_data
                .as_ref()
                .map(|h| h.read().sum)
                .unwrap_or(0.0),
        }
    }

    fn observe_histogram(&self, value: f64) {
        if let Some(ref data) = self.histogram_data {
            let mut h = data.write();
            h.count += 1;
            h.sum += value;
            for (bound, count) in h.buckets.iter_mut() {
                if value <= *bound {
                    *count += 1;
                }
            }
        }
    }

    /// Format as Prometheus text exposition format
    fn to_prometheus_text(&self) -> String {
        let mut output = String::new();

        // HELP line
        output.push_str(&format!("# HELP {} {}\n", self.name, self.help));

        // TYPE line
        let type_str = match self.metric_type {
            MetricType::Counter => "counter",
            MetricType::Gauge => "gauge",
            MetricType::Histogram => "histogram",
        };
        output.push_str(&format!("# TYPE {} {}\n", self.name, type_str));

        // Value line(s)
        match self.metric_type {
            MetricType::Counter | MetricType::Gauge => {
                let value = self.get_value();
                // Use integer format for counters, float for gauges
                if self.metric_type == MetricType::Counter && value.fract() == 0.0 {
                    output.push_str(&format!("{} {}\n", self.name, value as u64));
                } else {
                    output.push_str(&format!("{} {}\n", self.name, value));
                }
            }
            MetricType::Histogram => {
                if let Some(ref data) = self.histogram_data {
                    let h = data.read();
                    for (bound, count) in &h.buckets {
                        output.push_str(&format!(
                            "{}_bucket{{le=\"{}\"}} {}\n",
                            self.name, bound, count
                        ));
                    }
                    output.push_str(&format!(
                        "{}_bucket{{le=\"+Inf\"}} {}\n",
                        self.name, h.count
                    ));
                    output.push_str(&format!("{}_sum {}\n", self.name, h.sum));
                    output.push_str(&format!("{}_count {}\n", self.name, h.count));
                }
            }
        }

        output
    }
}

/// Thread-safe metrics registry
#[derive(Debug, Clone)]
pub struct MetricsRegistry {
    metrics: Arc<RwLock<HashMap<String, Arc<Metric>>>>,
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl MetricsRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Create a registry with standard Symthaea metrics pre-registered
    pub fn with_standard_metrics() -> Self {
        let registry = Self::new();

        // Consciousness metrics
        registry.register_counter(
            "phi_calculations_total",
            "Total integrated information calculations",
        );
        registry.register_gauge("phi_current", "Current Φ (integrated information) value");
        registry.register_gauge("coherence_field", "Current coherence field strength");
        registry.register_gauge(
            "global_workspace_broadcast_count",
            "Number of GWT broadcasts",
        );

        // HDC metrics
        registry.register_counter("hdc_bind_ops_total", "Total HDC bind operations");
        registry.register_counter("hdc_bundle_ops_total", "Total HDC bundle operations");
        registry.register_counter(
            "hdc_similarity_ops_total",
            "Total HDC similarity computations",
        );
        registry.register_histogram(
            "hdc_similarity_seconds",
            "HDC similarity computation duration",
            vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        );

        // LTC metrics
        registry.register_counter("ltc_steps_total", "Total LTC network steps");
        registry.register_gauge("ltc_tau_mean", "Mean time constant across LTC network");

        // Voice metrics
        registry.register_counter(
            "voice_utterances_total",
            "Total voice utterances synthesized",
        );
        registry.register_gauge("voice_audio_seconds", "Total audio seconds generated");

        // API metrics
        registry.register_counter("api_requests_total", "Total API requests");
        registry.register_counter("api_errors_total", "Total API errors");
        registry.register_histogram(
            "api_request_duration_seconds",
            "API request duration",
            vec![0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
        );

        // Swarm metrics
        registry.register_gauge("swarm_peers_connected", "Number of connected swarm peers");
        registry.register_counter("swarm_messages_sent_total", "Total swarm messages sent");
        registry.register_counter(
            "swarm_messages_received_total",
            "Total swarm messages received",
        );

        // Memory metrics
        registry.register_gauge("memory_entries_total", "Total entries in memory store");
        registry.register_gauge("memory_cache_hit_ratio", "Memory cache hit ratio");

        registry
    }

    /// Register a counter metric
    pub fn register_counter(&self, name: &str, help: &str) {
        let metric = Arc::new(Metric::new_counter(name, help));
        self.metrics.write().insert(name.to_string(), metric);
    }

    /// Register a gauge metric
    pub fn register_gauge(&self, name: &str, help: &str) {
        let metric = Arc::new(Metric::new_gauge(name, help));
        self.metrics.write().insert(name.to_string(), metric);
    }

    /// Register a histogram metric with custom buckets
    pub fn register_histogram(&self, name: &str, help: &str, buckets: Vec<f64>) {
        let metric = Arc::new(Metric::new_histogram(name, help, buckets));
        self.metrics.write().insert(name.to_string(), metric);
    }

    /// Register a metric with explicit type
    pub fn register(&self, name: &str, metric_type: MetricType, help: &str) {
        match metric_type {
            MetricType::Counter => self.register_counter(name, help),
            MetricType::Gauge => self.register_gauge(name, help),
            MetricType::Histogram => {
                // Use default buckets for histogram
                self.register_histogram(
                    name,
                    help,
                    vec![
                        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0,
                    ],
                );
            }
        }
    }

    /// Increment a counter by 1
    pub fn increment(&self, name: &str) {
        if let Some(metric) = self.metrics.read().get(name) {
            metric.increment();
        }
    }

    /// Add a value to a counter
    pub fn add(&self, name: &str, delta: u64) {
        if let Some(metric) = self.metrics.read().get(name) {
            metric.add(delta);
        }
    }

    /// Set a gauge value
    pub fn set_gauge(&self, name: &str, value: f64) {
        if let Some(metric) = self.metrics.read().get(name) {
            metric.set_gauge(value);
        }
    }

    /// Observe a value for a histogram
    pub fn observe(&self, name: &str, value: f64) {
        if let Some(metric) = self.metrics.read().get(name) {
            metric.observe_histogram(value);
        }
    }

    /// Get a metric's current value
    pub fn get(&self, name: &str) -> Option<f64> {
        self.metrics.read().get(name).map(|m| m.get_value())
    }

    /// Export all metrics in Prometheus text exposition format
    pub fn to_prometheus_text(&self) -> String {
        let metrics = self.metrics.read();
        let mut output = String::new();

        // Sort by name for consistent output
        let mut names: Vec<_> = metrics.keys().collect();
        names.sort();

        for name in names {
            if let Some(metric) = metrics.get(name) {
                output.push_str(&metric.to_prometheus_text());
            }
        }

        output
    }

    /// Export all metrics as JSON
    pub fn to_json(&self) -> MetricsSnapshot {
        let metrics = self.metrics.read();
        let mut entries = Vec::new();

        for (name, metric) in metrics.iter() {
            entries.push(MetricEntry {
                name: name.clone(),
                metric_type: metric.metric_type,
                help: metric.help.clone(),
                value: metric.get_value(),
            });
        }

        // Sort for consistent output
        entries.sort_by(|a, b| a.name.cmp(&b.name));

        MetricsSnapshot {
            timestamp: chrono::Utc::now().timestamp(),
            metrics: entries,
        }
    }

    /// Get the number of registered metrics
    pub fn len(&self) -> usize {
        self.metrics.read().len()
    }

    /// Check if registry is empty
    pub fn is_empty(&self) -> bool {
        self.metrics.read().is_empty()
    }
}

/// JSON-serializable metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    /// Unix timestamp when snapshot was taken
    pub timestamp: i64,
    /// All metrics
    pub metrics: Vec<MetricEntry>,
}

/// A single metric entry for JSON serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricEntry {
    /// Metric name
    pub name: String,
    /// Metric type
    #[serde(rename = "type")]
    pub metric_type: MetricType,
    /// Help text
    pub help: String,
    /// Current value
    pub value: f64,
}

/// Global metrics registry instance
static GLOBAL_REGISTRY: once_cell::sync::Lazy<MetricsRegistry> =
    once_cell::sync::Lazy::new(MetricsRegistry::with_standard_metrics);

/// Get the global metrics registry
pub fn global() -> &'static MetricsRegistry {
    &GLOBAL_REGISTRY
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_counter() {
        let registry = MetricsRegistry::new();
        registry.register_counter("test_counter", "A test counter");

        assert_eq!(registry.get("test_counter"), Some(0.0));

        registry.increment("test_counter");
        assert_eq!(registry.get("test_counter"), Some(1.0));

        registry.add("test_counter", 5);
        assert_eq!(registry.get("test_counter"), Some(6.0));
    }

    #[test]
    fn test_gauge() {
        let registry = MetricsRegistry::new();
        registry.register_gauge("test_gauge", "A test gauge");

        registry.set_gauge("test_gauge", 42.5);
        assert_eq!(registry.get("test_gauge"), Some(42.5));

        registry.set_gauge("test_gauge", -10.0);
        assert_eq!(registry.get("test_gauge"), Some(-10.0));
    }

    #[test]
    fn test_histogram() {
        let registry = MetricsRegistry::new();
        registry.register_histogram("test_hist", "A test histogram", vec![0.1, 0.5, 1.0]);

        registry.observe("test_hist", 0.05);
        registry.observe("test_hist", 0.3);
        registry.observe("test_hist", 0.8);

        // Sum should be 0.05 + 0.3 + 0.8 = 1.15
        let sum = registry.get("test_hist").unwrap();
        assert!((sum - 1.15).abs() < 0.001);
    }

    #[test]
    fn test_prometheus_format() {
        let registry = MetricsRegistry::new();
        registry.register_counter("requests_total", "Total requests");
        registry.register_gauge("temperature", "Current temperature");

        registry.add("requests_total", 100);
        registry.set_gauge("temperature", 23.5);

        let output = registry.to_prometheus_text();

        assert!(output.contains("# HELP requests_total Total requests"));
        assert!(output.contains("# TYPE requests_total counter"));
        assert!(output.contains("requests_total 100"));
        assert!(output.contains("# TYPE temperature gauge"));
        assert!(output.contains("temperature 23.5"));
    }

    #[test]
    fn test_json_export() {
        let registry = MetricsRegistry::new();
        registry.register_counter("test_metric", "Test");
        registry.increment("test_metric");

        let snapshot = registry.to_json();
        assert_eq!(snapshot.metrics.len(), 1);
        assert_eq!(snapshot.metrics[0].name, "test_metric");
        assert_eq!(snapshot.metrics[0].value, 1.0);
    }

    #[test]
    fn test_standard_metrics() {
        let registry = MetricsRegistry::with_standard_metrics();

        // Should have standard consciousness metrics
        assert!(registry.get("phi_current").is_some());
        assert!(registry.get("phi_calculations_total").is_some());
        assert!(registry.get("coherence_field").is_some());

        // Should have HDC metrics
        assert!(registry.get("hdc_bind_ops_total").is_some());

        // Should have API metrics
        assert!(registry.get("api_requests_total").is_some());
    }

    #[test]
    fn test_global_registry() {
        let registry = global();
        assert!(registry.get("phi_current").is_some());
    }
}
