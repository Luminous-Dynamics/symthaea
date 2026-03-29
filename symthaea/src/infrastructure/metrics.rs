// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! J1: Prometheus Metrics
//!
//! Exposes consciousness and performance metrics in Prometheus format
//! for monitoring and alerting.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// A metric value
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// Counter (monotonically increasing)
    Counter(u64),
    /// Gauge (can go up or down)
    Gauge(f64),
    /// Histogram bucket counts
    Histogram {
        buckets: Vec<(f64, u64)>,
        sum: f64,
        count: u64,
    },
    /// Summary quantiles
    Summary {
        quantiles: Vec<(f64, f64)>,
        sum: f64,
        count: u64,
    },
}

/// Metric labels
pub type Labels = Vec<(String, String)>;

/// A complete metric with metadata
#[derive(Debug, Clone)]
pub struct Metric {
    /// Metric name
    pub name: String,
    /// Help text
    pub help: String,
    /// Metric type
    pub metric_type: MetricType,
    /// Values with labels
    pub values: Vec<(Labels, MetricValue)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    Counter,
    Gauge,
    Histogram,
    Summary,
    Untyped,
}

impl std::fmt::Display for MetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricType::Counter => write!(f, "counter"),
            MetricType::Gauge => write!(f, "gauge"),
            MetricType::Histogram => write!(f, "histogram"),
            MetricType::Summary => write!(f, "summary"),
            MetricType::Untyped => write!(f, "untyped"),
        }
    }
}

/// Metrics collector for Symthaea
pub struct MetricsCollector {
    /// Consciousness metrics
    phi: AtomicF64,
    coherence: AtomicF64,
    consciousness_level: AtomicF64,

    /// Request counters
    requests_total: AtomicU64,
    #[allow(dead_code)] // RESERVED(future): per-type request metrics
    requests_by_type: HashMap<String, AtomicU64>,

    /// Execution metrics
    executions_total: AtomicU64,
    executions_vetoed: AtomicU64,
    executions_gated: AtomicU64,

    /// Performance metrics
    request_duration_sum: AtomicF64,
    request_duration_count: AtomicU64,
    intellisense_latency_sum: AtomicF64,
    intellisense_count: AtomicU64,

    /// Safety metrics
    safety_scans_total: AtomicU64,
    threats_detected: AtomicU64,

    /// Cache metrics
    cache_hits: AtomicU64,
    cache_misses: AtomicU64,

    /// Service start time
    start_time: Instant,
    start_timestamp: u64,

    /// Active connections
    active_connections: AtomicU64,
}

// Wrapper for atomic f64 operations
struct AtomicF64(AtomicU64);

impl AtomicF64 {
    fn new(val: f64) -> Self {
        Self(AtomicU64::new(val.to_bits()))
    }

    fn load(&self) -> f64 {
        f64::from_bits(self.0.load(Ordering::Relaxed))
    }

    fn store(&self, val: f64) {
        self.0.store(val.to_bits(), Ordering::Relaxed);
    }

    fn add(&self, val: f64) {
        loop {
            let current = self.0.load(Ordering::Relaxed);
            let new = f64::from_bits(current) + val;
            if self
                .0
                .compare_exchange_weak(current, new.to_bits(), Ordering::Relaxed, Ordering::Relaxed)
                .is_ok()
            {
                break;
            }
        }
    }
}

impl MetricsCollector {
    /// Create a new metrics collector
    pub fn new() -> Self {
        let start_timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            phi: AtomicF64::new(0.0),
            coherence: AtomicF64::new(0.0),
            consciousness_level: AtomicF64::new(0.0),
            requests_total: AtomicU64::new(0),
            requests_by_type: HashMap::new(),
            executions_total: AtomicU64::new(0),
            executions_vetoed: AtomicU64::new(0),
            executions_gated: AtomicU64::new(0),
            request_duration_sum: AtomicF64::new(0.0),
            request_duration_count: AtomicU64::new(0),
            intellisense_latency_sum: AtomicF64::new(0.0),
            intellisense_count: AtomicU64::new(0),
            safety_scans_total: AtomicU64::new(0),
            threats_detected: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            start_time: Instant::now(),
            start_timestamp,
            active_connections: AtomicU64::new(0),
        }
    }

    // Consciousness metrics updates

    /// Update Phi value
    pub fn set_phi(&self, phi: f64) {
        self.phi.store(phi);
    }

    /// Update coherence
    pub fn set_coherence(&self, coherence: f64) {
        self.coherence.store(coherence);
    }

    /// Update consciousness level
    pub fn set_consciousness_level(&self, level: f64) {
        self.consciousness_level.store(level);
    }

    // Request tracking

    /// Increment request counter
    pub fn inc_requests(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Track request by type
    pub fn track_request(&self, _request_type: &str) {
        self.inc_requests();
        // Note: For thread-safe type tracking, would need concurrent map
    }

    /// Record request duration
    pub fn record_request_duration(&self, duration: Duration) {
        self.request_duration_sum.add(duration.as_secs_f64());
        self.request_duration_count.fetch_add(1, Ordering::Relaxed);
    }

    // Execution tracking

    /// Track successful execution
    pub fn track_execution(&self, vetoed: bool, gated: bool) {
        self.executions_total.fetch_add(1, Ordering::Relaxed);
        if vetoed {
            self.executions_vetoed.fetch_add(1, Ordering::Relaxed);
        }
        if gated {
            self.executions_gated.fetch_add(1, Ordering::Relaxed);
        }
    }

    // IntelliSense tracking

    /// Record intellisense latency
    pub fn record_intellisense_latency(&self, latency: Duration) {
        self.intellisense_latency_sum.add(latency.as_secs_f64());
        self.intellisense_count.fetch_add(1, Ordering::Relaxed);
    }

    // Safety tracking

    /// Track safety scan
    pub fn track_safety_scan(&self, threat_detected: bool) {
        self.safety_scans_total.fetch_add(1, Ordering::Relaxed);
        if threat_detected {
            self.threats_detected.fetch_add(1, Ordering::Relaxed);
        }
    }

    // Cache tracking

    /// Track cache access
    pub fn track_cache(&self, hit: bool) {
        if hit {
            self.cache_hits.fetch_add(1, Ordering::Relaxed);
        } else {
            self.cache_misses.fetch_add(1, Ordering::Relaxed);
        }
    }

    // Connection tracking

    /// Increment active connections
    pub fn inc_connections(&self) {
        self.active_connections.fetch_add(1, Ordering::Relaxed);
    }

    /// Decrement active connections
    pub fn dec_connections(&self) {
        self.active_connections.fetch_sub(1, Ordering::Relaxed);
    }

    /// Get uptime in seconds
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Export all metrics in Prometheus format
    pub fn export(&self) -> String {
        let mut output = String::new();

        // Service info
        self.write_metric(
            &mut output,
            "symthaea_info",
            "Symthaea service information",
            MetricType::Gauge,
            &[(vec![("version".into(), "0.1.0".into())], 1.0)],
        );

        // Uptime
        self.write_metric(
            &mut output,
            "symthaea_uptime_seconds",
            "Service uptime in seconds",
            MetricType::Gauge,
            &[(vec![], self.uptime_secs() as f64)],
        );

        self.write_metric(
            &mut output,
            "symthaea_start_time_seconds",
            "Service start time as Unix timestamp",
            MetricType::Gauge,
            &[(vec![], self.start_timestamp as f64)],
        );

        // Consciousness metrics
        self.write_metric(
            &mut output,
            "symthaea_phi",
            "Current integrated information (Phi) value",
            MetricType::Gauge,
            &[(vec![], self.phi.load())],
        );

        self.write_metric(
            &mut output,
            "symthaea_coherence",
            "Current system coherence (0-1)",
            MetricType::Gauge,
            &[(vec![], self.coherence.load())],
        );

        self.write_metric(
            &mut output,
            "symthaea_consciousness_level",
            "Current consciousness level (0-1)",
            MetricType::Gauge,
            &[(vec![], self.consciousness_level.load())],
        );

        // Request metrics
        self.write_metric(
            &mut output,
            "symthaea_requests_total",
            "Total number of IPC requests",
            MetricType::Counter,
            &[(vec![], self.requests_total.load(Ordering::Relaxed) as f64)],
        );

        let req_count = self.request_duration_count.load(Ordering::Relaxed);
        if req_count > 0 {
            let avg_duration = self.request_duration_sum.load() / req_count as f64;
            self.write_metric(
                &mut output,
                "symthaea_request_duration_seconds_avg",
                "Average request duration in seconds",
                MetricType::Gauge,
                &[(vec![], avg_duration)],
            );
        }

        // Execution metrics
        self.write_metric(
            &mut output,
            "symthaea_executions_total",
            "Total command execution attempts",
            MetricType::Counter,
            &[(vec![], self.executions_total.load(Ordering::Relaxed) as f64)],
        );

        self.write_metric(
            &mut output,
            "symthaea_executions_vetoed_total",
            "Commands vetoed by safety system",
            MetricType::Counter,
            &[(
                vec![],
                self.executions_vetoed.load(Ordering::Relaxed) as f64,
            )],
        );

        self.write_metric(
            &mut output,
            "symthaea_executions_gated_total",
            "Commands gated by Phi threshold",
            MetricType::Counter,
            &[(vec![], self.executions_gated.load(Ordering::Relaxed) as f64)],
        );

        // IntelliSense metrics
        let is_count = self.intellisense_count.load(Ordering::Relaxed);
        if is_count > 0 {
            let avg_latency = self.intellisense_latency_sum.load() / is_count as f64;
            self.write_metric(
                &mut output,
                "symthaea_intellisense_latency_seconds_avg",
                "Average IntelliSense latency in seconds",
                MetricType::Gauge,
                &[(vec![], avg_latency)],
            );
        }

        self.write_metric(
            &mut output,
            "symthaea_intellisense_requests_total",
            "Total IntelliSense requests",
            MetricType::Counter,
            &[(vec![], is_count as f64)],
        );

        // Safety metrics
        self.write_metric(
            &mut output,
            "symthaea_safety_scans_total",
            "Total safety scans performed",
            MetricType::Counter,
            &[(
                vec![],
                self.safety_scans_total.load(Ordering::Relaxed) as f64,
            )],
        );

        self.write_metric(
            &mut output,
            "symthaea_threats_detected_total",
            "Total threats detected",
            MetricType::Counter,
            &[(vec![], self.threats_detected.load(Ordering::Relaxed) as f64)],
        );

        // Cache metrics
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        self.write_metric(
            &mut output,
            "symthaea_cache_hits_total",
            "Total cache hits",
            MetricType::Counter,
            &[(vec![], hits as f64)],
        );

        self.write_metric(
            &mut output,
            "symthaea_cache_misses_total",
            "Total cache misses",
            MetricType::Counter,
            &[(vec![], misses as f64)],
        );

        if hits + misses > 0 {
            let hit_rate = hits as f64 / (hits + misses) as f64;
            self.write_metric(
                &mut output,
                "symthaea_cache_hit_ratio",
                "Cache hit ratio (0-1)",
                MetricType::Gauge,
                &[(vec![], hit_rate)],
            );
        }

        // Connection metrics
        self.write_metric(
            &mut output,
            "symthaea_active_connections",
            "Number of active client connections",
            MetricType::Gauge,
            &[(
                vec![],
                self.active_connections.load(Ordering::Relaxed) as f64,
            )],
        );

        output
    }

    fn write_metric(
        &self,
        output: &mut String,
        name: &str,
        help: &str,
        metric_type: MetricType,
        values: &[(Vec<(String, String)>, f64)],
    ) {
        // HELP line
        output.push_str(&format!("# HELP {name} {help}\n"));
        // TYPE line
        output.push_str(&format!("# TYPE {name} {metric_type}\n"));
        // Value lines
        for (labels, value) in values {
            if labels.is_empty() {
                output.push_str(&format!("{} {}\n", name, format_value(*value)));
            } else {
                let label_str: Vec<String> = labels
                    .iter()
                    .map(|(k, v)| format!("{}=\"{}\"", k, escape_label_value(v)))
                    .collect();
                output.push_str(&format!(
                    "{}{{{}}} {}\n",
                    name,
                    label_str.join(","),
                    format_value(*value)
                ));
            }
        }
    }
}

impl Default for MetricsCollector {
    fn default() -> Self {
        Self::new()
    }
}

fn format_value(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v.is_sign_positive() {
            "+Inf".to_string()
        } else {
            "-Inf".to_string()
        }
    } else if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

fn escape_label_value(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

/// HTTP handler for /metrics endpoint
pub fn metrics_handler(collector: &MetricsCollector) -> (String, String) {
    let body = collector.export();
    let content_type = "text/plain; version=0.0.4; charset=utf-8".to_string();
    (content_type, body)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_collector() {
        let collector = MetricsCollector::new();

        collector.set_phi(0.85);
        collector.set_coherence(0.92);
        collector.inc_requests();
        collector.track_execution(false, false);

        let output = collector.export();

        assert!(output.contains("symthaea_phi"));
        assert!(output.contains("0.85"));
        assert!(output.contains("symthaea_requests_total"));
        assert!(output.contains("symthaea_executions_total"));
    }

    #[test]
    fn test_prometheus_format() {
        let collector = MetricsCollector::new();
        collector.set_phi(0.5);

        let output = collector.export();

        assert!(output.contains("# HELP symthaea_phi"));
        assert!(output.contains("# TYPE symthaea_phi gauge"));
        assert!(output.contains("symthaea_phi 0.5"));
    }

    #[test]
    fn test_cache_metrics() {
        let collector = MetricsCollector::new();

        for _ in 0..8 {
            collector.track_cache(true);
        }
        for _ in 0..2 {
            collector.track_cache(false);
        }

        let output = collector.export();
        assert!(output.contains("symthaea_cache_hits_total 8"));
        assert!(output.contains("symthaea_cache_misses_total 2"));
        assert!(output.contains("symthaea_cache_hit_ratio 0.8"));
    }

    #[test]
    fn test_label_escaping() {
        assert_eq!(escape_label_value("test"), "test");
        assert_eq!(escape_label_value("te\"st"), "te\\\"st");
        assert_eq!(escape_label_value("te\\st"), "te\\\\st");
        assert_eq!(escape_label_value("te\nst"), "te\\nst");
    }
}
