// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
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
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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

        // Networking subsystem metrics (P2P / federation / mesh / governance)
        registry.register_counter(
            "iroh_handshakes_failed_total",
            "Iroh inbound QUIC handshake failures (transient errors, with 100ms backoff)",
        );
        registry.register_counter(
            "federation_rounds_total",
            "Successful federated learning synchronisation rounds",
        );
        registry.register_counter(
            "mesh_packets_sent_total",
            "Wisdom packets flushed to the DualLayerMesh bridge actor",
        );
        registry.register_counter(
            "crisis_dispatches_total",
            "Civic crisis events dispatched to the Mycelix governance channel",
        );

        // Memory metrics
        registry.register_gauge("memory_entries_total", "Total entries in memory store");
        registry.register_gauge("memory_cache_hit_ratio", "Memory cache hit ratio");

        // Neuromodulator bath metrics
        register_bath_metrics(&registry);

        // Module timing metrics (microseconds per cycle)
        register_timing_metrics(&registry);

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

/// Register bath-related gauge metrics in the given registry.
pub fn register_bath_metrics(registry: &MetricsRegistry) {
    registry.register_gauge("bath_da_effective", "Dopamine effective level");
    registry.register_gauge("bath_ne_effective", "Noradrenaline effective level");
    registry.register_gauge("bath_sht_effective", "Serotonin effective level");
    registry.register_gauge("bath_ach_effective", "Acetylcholine effective level");
    registry.register_gauge("bath_gaba_effective", "GABA effective level");
    registry.register_gauge("bath_oxy_effective", "Oxytocin effective level");
    registry.register_gauge("bath_glut_effective", "Glutamate effective level");
    registry.register_gauge("bath_aden_effective", "Adenosine effective level");
    registry.register_gauge("bath_ecb_effective", "Endocannabinoid effective level");
    registry.register_gauge("bath_allostatic_load", "Allostatic stress load");
    registry.register_gauge("bath_ei_ratio", "Excitatory/inhibitory balance ratio");
    registry.register_gauge("bath_sleep_pressure", "Adenosine-based sleep pressure");
    registry.register_gauge(
        "bath_active_injection_count",
        "Active pharmacological injections",
    );
    registry.register_gauge("bath_entropy", "Bath phase space entropy");
}

/// Update bath gauge values from current state.
pub fn update_bath_metrics(
    registry: &MetricsRegistry,
    state: &[f32; 9],
    allostatic_load: f32,
    ei_ratio: f32,
    sleep_pressure: f32,
    injection_count: usize,
) {
    registry.set_gauge("bath_da_effective", state[0] as f64);
    registry.set_gauge("bath_ne_effective", state[1] as f64);
    registry.set_gauge("bath_sht_effective", state[2] as f64);
    registry.set_gauge("bath_ach_effective", state[3] as f64);
    registry.set_gauge("bath_gaba_effective", state[4] as f64);
    registry.set_gauge("bath_oxy_effective", state[5] as f64);
    registry.set_gauge("bath_glut_effective", state[6] as f64);
    registry.set_gauge("bath_aden_effective", state[7] as f64);
    registry.set_gauge("bath_ecb_effective", state[8] as f64);
    registry.set_gauge("bath_allostatic_load", allostatic_load as f64);
    registry.set_gauge("bath_ei_ratio", ei_ratio as f64);
    registry.set_gauge("bath_sleep_pressure", sleep_pressure as f64);
    registry.set_gauge("bath_active_injection_count", injection_count as f64);
    registry.set_gauge("bath_entropy", 0.0);
}

/// Register module timing gauge metrics in the given registry.
pub fn register_timing_metrics(registry: &MetricsRegistry) {
    registry.register_gauge(
        "symthaea_timing_core_hdc_encode_us",
        "HDC encoding time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_core_cfc_step_us",
        "CfC temporal step time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_core_training_us",
        "Training step time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_consciousness_engine_us",
        "Consciousness engine total time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_ethics_engine_us",
        "Ethics engine total time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_fep_inference_us",
        "FEP predictive processing time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_resonator_us",
        "Resonator recall time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_gwt_us",
        "Global workspace broadcast time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_spectral_mip_us",
        "Spectral MIP finder time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_moral_topology_us",
        "Moral topology analysis time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_homeostasis_us",
        "Homeostasis clamp time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_metadata_assembly_us",
        "Metadata assembly time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_parallel_postprocess_us",
        "Parallel post-processing time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_dream_cycle_us",
        "Dream cycle time (microseconds)",
    );
    registry.register_gauge(
        "symthaea_timing_total_cycle_us",
        "Total cognitive cycle time (microseconds)",
    );
}

/// Update timing gauge values from a completed cycle's `ModuleTimings`.
pub fn update_timing_metrics(
    registry: &MetricsRegistry,
    timings: &crate::cognitive_loop::ModuleTimings,
    cycle_duration_us: u64,
) {
    registry.set_gauge(
        "symthaea_timing_core_hdc_encode_us",
        timings.core_hdc_encode as f64,
    );
    registry.set_gauge(
        "symthaea_timing_core_cfc_step_us",
        timings.core_cfc_step as f64,
    );
    registry.set_gauge(
        "symthaea_timing_core_training_us",
        timings.core_training as f64,
    );
    // Consciousness engine: sum of sub-components if engine total is 0
    let cons_engine = if timings.consciousness_engine > 0 {
        timings.consciousness_engine
    } else {
        timings.consciousness_engine_equation_v2
            + timings.consciousness_engine_pipeline
            + timings.consciousness_engine_multimodal
    };
    registry.set_gauge(
        "symthaea_timing_consciousness_engine_us",
        cons_engine as f64,
    );
    // Ethics engine: sum of sub-components if engine total is 0
    let ethics_engine = if timings.ethics_engine > 0 {
        timings.ethics_engine
    } else {
        timings.ethics_engine_moral + timings.ethics_engine_value + timings.ethics_engine_harmonies
    };
    registry.set_gauge("symthaea_timing_ethics_engine_us", ethics_engine as f64);
    registry.set_gauge(
        "symthaea_timing_fep_inference_us",
        timings.predictive_processing as f64,
    );
    registry.set_gauge(
        "symthaea_timing_resonator_us",
        timings.resonator_recall as f64,
    );
    registry.set_gauge("symthaea_timing_gwt_us", timings.gwt as f64);
    registry.set_gauge(
        "symthaea_timing_spectral_mip_us",
        timings.spectral_mip as f64,
    );
    registry.set_gauge(
        "symthaea_timing_moral_topology_us",
        timings.moral_topology as f64,
    );
    registry.set_gauge("symthaea_timing_homeostasis_us", timings.homeostasis as f64);
    registry.set_gauge(
        "symthaea_timing_metadata_assembly_us",
        timings.metadata_assembly as f64,
    );
    registry.set_gauge(
        "symthaea_timing_parallel_postprocess_us",
        timings.core_parallel_postprocess as f64,
    );
    registry.set_gauge("symthaea_timing_dream_cycle_us", timings.dream_cycle as f64);
    registry.set_gauge("symthaea_timing_total_cycle_us", cycle_duration_us as f64);
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
    fn test_networking_metrics_registered() {
        let registry = MetricsRegistry::with_standard_metrics();
        assert!(registry.get("iroh_handshakes_failed_total").is_some());
        assert!(registry.get("federation_rounds_total").is_some());
        assert!(registry.get("mesh_packets_sent_total").is_some());
        assert!(registry.get("crisis_dispatches_total").is_some());
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

    #[test]
    fn test_bath_metrics_registered_in_standard() {
        let registry = MetricsRegistry::with_standard_metrics();
        assert!(registry.get("bath_da_effective").is_some());
        assert!(registry.get("bath_allostatic_load").is_some());
        assert!(registry.get("bath_ei_ratio").is_some());
        assert!(registry.get("bath_entropy").is_some());
    }

    #[test]
    fn test_bath_metrics_values_update() {
        let registry = MetricsRegistry::with_standard_metrics();
        let state = [0.5, 0.6, 0.7, 0.4, 0.3, 0.2, 0.3, 0.1, 0.15];
        update_bath_metrics(&registry, &state, 0.25, 1.1, 0.3, 2);
        assert_eq!(registry.get("bath_da_effective"), Some(0.5));
        assert_eq!(registry.get("bath_allostatic_load"), Some(0.25));
        assert_eq!(registry.get("bath_active_injection_count"), Some(2.0));
    }

    #[test]
    fn test_bath_metrics_prometheus_output() {
        let registry = MetricsRegistry::with_standard_metrics();
        let state = [0.5; 9];
        update_bath_metrics(&registry, &state, 0.1, 1.0, 0.2, 0);
        let output = registry.to_prometheus_text();
        assert!(output.contains("bath_da_effective"));
    }

    #[test]
    fn test_bath_metrics_json_includes_bath() {
        let registry = MetricsRegistry::with_standard_metrics();
        let state = [0.5; 9];
        update_bath_metrics(&registry, &state, 0.1, 1.0, 0.2, 0);
        let snapshot = registry.to_json();
        assert!(snapshot.metrics.iter().any(|m| m.name.starts_with("bath_")));
    }

    #[test]
    fn test_timing_metrics_registered_in_standard() {
        let registry = MetricsRegistry::with_standard_metrics();
        assert!(registry.get("symthaea_timing_core_hdc_encode_us").is_some());
        assert!(registry.get("symthaea_timing_core_cfc_step_us").is_some());
        assert!(registry.get("symthaea_timing_core_training_us").is_some());
        assert!(
            registry
                .get("symthaea_timing_consciousness_engine_us")
                .is_some()
        );
        assert!(registry.get("symthaea_timing_ethics_engine_us").is_some());
        assert!(registry.get("symthaea_timing_total_cycle_us").is_some());
    }

    #[test]
    fn test_timing_metrics_values_update() {
        let registry = MetricsRegistry::with_standard_metrics();
        let mut timings = crate::cognitive_loop::ModuleTimings::default();
        timings.core_hdc_encode = 150;
        timings.core_cfc_step = 300;
        timings.gwt = 50;
        timings.ethics_engine_moral = 100;
        timings.ethics_engine_value = 80;
        timings.ethics_engine_harmonies = 70;
        update_timing_metrics(&registry, &timings, 5000);
        assert_eq!(
            registry.get("symthaea_timing_core_hdc_encode_us"),
            Some(150.0)
        );
        assert_eq!(
            registry.get("symthaea_timing_core_cfc_step_us"),
            Some(300.0)
        );
        assert_eq!(registry.get("symthaea_timing_gwt_us"), Some(50.0));
        // Ethics engine sums sub-components when engine total is 0
        assert_eq!(
            registry.get("symthaea_timing_ethics_engine_us"),
            Some(250.0)
        );
        assert_eq!(registry.get("symthaea_timing_total_cycle_us"), Some(5000.0));
    }

    #[test]
    fn test_timing_metrics_prometheus_output() {
        let registry = MetricsRegistry::with_standard_metrics();
        let timings = crate::cognitive_loop::ModuleTimings::default();
        update_timing_metrics(&registry, &timings, 1000);
        let output = registry.to_prometheus_text();
        assert!(output.contains("symthaea_timing_total_cycle_us"));
        assert!(output.contains("# TYPE symthaea_timing_total_cycle_us gauge"));
    }
}
