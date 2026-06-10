// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Telemetry Export Module
//!
//! Provides sinks for exporting ReasoningEvent telemetry to various destinations:
//! - JSON Lines files (for offline analysis)
//! - CSV files (for spreadsheet import)
//! - Prometheus metrics (for real-time monitoring)
//!
//! ## Example
//!
//! ```ignore
//! use symthaea::consciousness::reasoning_engine::telemetry::*;
//!
//! // Create a multi-sink exporter
//! let exporter = TelemetryExporter::builder()
//!     .add_jsonl_sink("/var/log/symthaea/reasoning.jsonl")
//!     .add_prometheus_sink("0.0.0.0:9091")
//!     .build();
//!
//! // Export events
//! exporter.export(&event)?;
//! ```

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use super::types::ReasoningEvent;

// ─────────────────────────────────────────────────────────────────────────────
// TelemetrySink Trait
// ─────────────────────────────────────────────────────────────────────────────

/// Trait for telemetry export destinations.
pub trait TelemetrySink: Send + Sync {
    /// Export a single reasoning event.
    fn export(&self, event: &ReasoningEvent) -> Result<(), TelemetryError>;

    /// Flush any buffered data.
    fn flush(&self) -> Result<(), TelemetryError>;

    /// Human-readable name for this sink.
    fn name(&self) -> &str;
}

/// Telemetry export errors.
#[derive(Debug)]
pub enum TelemetryError {
    /// I/O error during export.
    Io(std::io::Error),
    /// Serialization error.
    Serialization(String),
    /// Sink not ready.
    NotReady(String),
}

impl std::fmt::Display for TelemetryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TelemetryError::Io(e) => write!(f, "I/O error: {}", e),
            TelemetryError::Serialization(s) => write!(f, "Serialization error: {}", s),
            TelemetryError::NotReady(s) => write!(f, "Sink not ready: {}", s),
        }
    }
}

impl std::error::Error for TelemetryError {}

impl From<std::io::Error> for TelemetryError {
    fn from(e: std::io::Error) -> Self {
        TelemetryError::Io(e)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON Lines Sink
// ─────────────────────────────────────────────────────────────────────────────

/// Sink that writes events as JSON Lines to a file.
///
/// Each event is written as a single line of JSON, making it easy to
/// process with standard tools like `jq`, `grep`, and streaming analytics.
pub struct JsonLinesSink {
    #[allow(dead_code)] // Retained for diagnostic/Display use
    path: PathBuf,
    writer: Mutex<BufWriter<File>>,
}

impl JsonLinesSink {
    /// Create a new JSON Lines sink.
    ///
    /// Opens the file for appending (creates if doesn't exist).
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, TelemetryError> {
        let path = path.into();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        Ok(Self {
            path,
            writer: Mutex::new(BufWriter::new(file)),
        })
    }
}

impl TelemetrySink for JsonLinesSink {
    fn export(&self, event: &ReasoningEvent) -> Result<(), TelemetryError> {
        let json = serde_json::to_string(event)
            .map_err(|e| TelemetryError::Serialization(e.to_string()))?;
        let mut writer = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        writeln!(writer, "{}", json)?;
        Ok(())
    }

    fn flush(&self) -> Result<(), TelemetryError> {
        let mut writer = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        writer.flush()?;
        Ok(())
    }

    fn name(&self) -> &str {
        "jsonl"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CSV Sink
// ─────────────────────────────────────────────────────────────────────────────

/// Sink that writes events as CSV rows.
///
/// Useful for spreadsheet analysis and simple data pipelines.
pub struct CsvSink {
    #[allow(dead_code)] // Retained for diagnostic/Display use
    path: PathBuf,
    writer: Mutex<BufWriter<File>>,
    header_written: Mutex<bool>,
}

impl CsvSink {
    /// Create a new CSV sink.
    pub fn new(path: impl Into<PathBuf>) -> Result<Self, TelemetryError> {
        let path = path.into();
        let file = OpenOptions::new().create(true).append(true).open(&path)?;
        let header_written = file.metadata()?.len() > 0;
        Ok(Self {
            path,
            writer: Mutex::new(BufWriter::new(file)),
            header_written: Mutex::new(header_written),
        })
    }

    fn write_header(&self, writer: &mut BufWriter<File>) -> Result<(), TelemetryError> {
        let header = "cycle_id,wall_time_us,budget_tier,phi_raw,reliability,phi_eff,gamma,\
            evs,did_simulate,mcts_iterations,plan_confidence,gate_decision,risk_level";
        writeln!(writer, "{}", header)?;
        Ok(())
    }
}

impl TelemetrySink for CsvSink {
    fn export(&self, event: &ReasoningEvent) -> Result<(), TelemetryError> {
        let mut writer = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let mut header_written = self
            .header_written
            .lock()
            .unwrap_or_else(|e| e.into_inner());

        if !*header_written {
            self.write_header(&mut writer)?;
            *header_written = true;
        }

        let row = format!(
            "{},{},{:?},{:.4},{:.4},{:.4},{:.4},{:.4},{},{},{:.4},{},{}",
            event.cycle_id,
            event.wall_time_us,
            event.budget_tier,
            event.phi_raw,
            event.reliability,
            event.phi_eff,
            event.gamma,
            event.evs,
            event.did_simulate,
            event.mcts_iterations,
            event.plan_confidence,
            event.gate_decision,
            event
                .risk_level
                .map_or("None".to_string(), |r| format!("{:?}", r)),
        );
        writeln!(writer, "{}", row)?;
        Ok(())
    }

    fn flush(&self) -> Result<(), TelemetryError> {
        let mut writer = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        writer.flush()?;
        Ok(())
    }

    fn name(&self) -> &str {
        "csv"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Prometheus Metrics Sink
// ─────────────────────────────────────────────────────────────────────────────

/// Prometheus metrics collector for reasoning engine telemetry.
///
/// Exposes metrics that can be scraped by Prometheus:
/// - `reasoning_cycle_duration_us`: Histogram of cycle durations
/// - `reasoning_phi_eff`: Gauge of current effective Φ
/// - `reasoning_reliability`: Gauge of current reliability
/// - `reasoning_budget_tier`: Counter of cycles per tier
/// - `reasoning_gate_decisions`: Counter of gate decisions
pub struct PrometheusMetrics {
    // Atomic counters and gauges for thread-safe updates
    cycle_count: std::sync::atomic::AtomicU64,
    tier0_count: std::sync::atomic::AtomicU64,
    tier1_count: std::sync::atomic::AtomicU64,
    tier2_count: std::sync::atomic::AtomicU64,
    gate_allowed_count: std::sync::atomic::AtomicU64,
    gate_blocked_count: std::sync::atomic::AtomicU64,
    // Gauges stored as atomic u64 (f64 bits)
    phi_eff_bits: std::sync::atomic::AtomicU64,
    reliability_bits: std::sync::atomic::AtomicU64,
    gamma_bits: std::sync::atomic::AtomicU64,
    // Duration histogram buckets (cumulative counts)
    duration_bucket_100us: std::sync::atomic::AtomicU64,
    duration_bucket_500us: std::sync::atomic::AtomicU64,
    duration_bucket_1ms: std::sync::atomic::AtomicU64,
    duration_bucket_2ms: std::sync::atomic::AtomicU64,
    duration_bucket_5ms: std::sync::atomic::AtomicU64,
    duration_bucket_10ms: std::sync::atomic::AtomicU64,
    duration_bucket_20ms: std::sync::atomic::AtomicU64,
    duration_bucket_inf: std::sync::atomic::AtomicU64,
    duration_sum: std::sync::atomic::AtomicU64,
}

impl PrometheusMetrics {
    /// Create a new Prometheus metrics collector.
    pub fn new() -> Self {
        use std::sync::atomic::AtomicU64;
        Self {
            cycle_count: AtomicU64::new(0),
            tier0_count: AtomicU64::new(0),
            tier1_count: AtomicU64::new(0),
            tier2_count: AtomicU64::new(0),
            gate_allowed_count: AtomicU64::new(0),
            gate_blocked_count: AtomicU64::new(0),
            phi_eff_bits: AtomicU64::new(0),
            reliability_bits: AtomicU64::new(0),
            gamma_bits: AtomicU64::new(0),
            duration_bucket_100us: AtomicU64::new(0),
            duration_bucket_500us: AtomicU64::new(0),
            duration_bucket_1ms: AtomicU64::new(0),
            duration_bucket_2ms: AtomicU64::new(0),
            duration_bucket_5ms: AtomicU64::new(0),
            duration_bucket_10ms: AtomicU64::new(0),
            duration_bucket_20ms: AtomicU64::new(0),
            duration_bucket_inf: AtomicU64::new(0),
            duration_sum: AtomicU64::new(0),
        }
    }

    /// Record a reasoning event.
    pub fn record(&self, event: &ReasoningEvent) {
        use crate::consciousness::temporal_planning::types::BudgetTier;
        use std::sync::atomic::Ordering::Relaxed;

        self.cycle_count.fetch_add(1, Relaxed);

        // Update tier counters
        match event.budget_tier {
            BudgetTier::Tier0 => {
                self.tier0_count.fetch_add(1, Relaxed);
            }
            BudgetTier::Tier1 => {
                self.tier1_count.fetch_add(1, Relaxed);
            }
            BudgetTier::Tier2 => {
                self.tier2_count.fetch_add(1, Relaxed);
            }
        }

        // Update gate counters
        if event.gate_decision.contains("Allowed") {
            self.gate_allowed_count.fetch_add(1, Relaxed);
        } else if !event.gate_decision.is_empty() {
            self.gate_blocked_count.fetch_add(1, Relaxed);
        }

        // Update gauges
        self.phi_eff_bits.store(event.phi_eff.to_bits(), Relaxed);
        self.reliability_bits
            .store(event.reliability.to_bits(), Relaxed);
        self.gamma_bits.store(event.gamma.to_bits(), Relaxed);

        // Update duration histogram
        let us = event.wall_time_us;
        if us <= 100 {
            self.duration_bucket_100us.fetch_add(1, Relaxed);
        }
        if us <= 500 {
            self.duration_bucket_500us.fetch_add(1, Relaxed);
        }
        if us <= 1_000 {
            self.duration_bucket_1ms.fetch_add(1, Relaxed);
        }
        if us <= 2_000 {
            self.duration_bucket_2ms.fetch_add(1, Relaxed);
        }
        if us <= 5_000 {
            self.duration_bucket_5ms.fetch_add(1, Relaxed);
        }
        if us <= 10_000 {
            self.duration_bucket_10ms.fetch_add(1, Relaxed);
        }
        if us <= 20_000 {
            self.duration_bucket_20ms.fetch_add(1, Relaxed);
        }
        self.duration_bucket_inf.fetch_add(1, Relaxed);
        self.duration_sum.fetch_add(us, Relaxed);
    }

    /// Render metrics in Prometheus text format.
    pub fn render(&self) -> String {
        use std::sync::atomic::Ordering::Relaxed;

        let mut out = String::new();

        // HELP and TYPE declarations
        out.push_str("# HELP reasoning_cycles_total Total reasoning cycles\n");
        out.push_str("# TYPE reasoning_cycles_total counter\n");
        out.push_str(&format!(
            "reasoning_cycles_total {}\n",
            self.cycle_count.load(Relaxed)
        ));

        out.push_str("# HELP reasoning_tier_cycles_total Cycles per budget tier\n");
        out.push_str("# TYPE reasoning_tier_cycles_total counter\n");
        out.push_str(&format!(
            "reasoning_tier_cycles_total{{tier=\"0\"}} {}\n",
            self.tier0_count.load(Relaxed)
        ));
        out.push_str(&format!(
            "reasoning_tier_cycles_total{{tier=\"1\"}} {}\n",
            self.tier1_count.load(Relaxed)
        ));
        out.push_str(&format!(
            "reasoning_tier_cycles_total{{tier=\"2\"}} {}\n",
            self.tier2_count.load(Relaxed)
        ));

        out.push_str("# HELP reasoning_gate_decisions_total Gate decisions by result\n");
        out.push_str("# TYPE reasoning_gate_decisions_total counter\n");
        out.push_str(&format!(
            "reasoning_gate_decisions_total{{result=\"allowed\"}} {}\n",
            self.gate_allowed_count.load(Relaxed)
        ));
        out.push_str(&format!(
            "reasoning_gate_decisions_total{{result=\"blocked\"}} {}\n",
            self.gate_blocked_count.load(Relaxed)
        ));

        out.push_str("# HELP reasoning_phi_eff Current effective phi\n");
        out.push_str("# TYPE reasoning_phi_eff gauge\n");
        out.push_str(&format!(
            "reasoning_phi_eff {:.4}\n",
            f64::from_bits(self.phi_eff_bits.load(Relaxed))
        ));

        out.push_str("# HELP reasoning_reliability Current reliability R\n");
        out.push_str("# TYPE reasoning_reliability gauge\n");
        out.push_str(&format!(
            "reasoning_reliability {:.4}\n",
            f64::from_bits(self.reliability_bits.load(Relaxed))
        ));

        out.push_str("# HELP reasoning_gamma Current gamma calibration\n");
        out.push_str("# TYPE reasoning_gamma gauge\n");
        out.push_str(&format!(
            "reasoning_gamma {:.4}\n",
            f64::from_bits(self.gamma_bits.load(Relaxed))
        ));

        out.push_str(
            "# HELP reasoning_cycle_duration_us Cycle duration histogram (microseconds)\n",
        );
        out.push_str("# TYPE reasoning_cycle_duration_us histogram\n");
        out.push_str(&format!(
            "reasoning_cycle_duration_us_bucket{{le=\"100\"}} {}\n",
            self.duration_bucket_100us.load(Relaxed)
        ));
        out.push_str(&format!(
            "reasoning_cycle_duration_us_bucket{{le=\"500\"}} {}\n",
            self.duration_bucket_500us.load(Relaxed)
        ));
        out.push_str(&format!(
            "reasoning_cycle_duration_us_bucket{{le=\"1000\"}} {}\n",
            self.duration_bucket_1ms.load(Relaxed)
        ));
        out.push_str(&format!(
            "reasoning_cycle_duration_us_bucket{{le=\"2000\"}} {}\n",
            self.duration_bucket_2ms.load(Relaxed)
        ));
        out.push_str(&format!(
            "reasoning_cycle_duration_us_bucket{{le=\"5000\"}} {}\n",
            self.duration_bucket_5ms.load(Relaxed)
        ));
        out.push_str(&format!(
            "reasoning_cycle_duration_us_bucket{{le=\"10000\"}} {}\n",
            self.duration_bucket_10ms.load(Relaxed)
        ));
        out.push_str(&format!(
            "reasoning_cycle_duration_us_bucket{{le=\"20000\"}} {}\n",
            self.duration_bucket_20ms.load(Relaxed)
        ));
        out.push_str(&format!(
            "reasoning_cycle_duration_us_bucket{{le=\"+Inf\"}} {}\n",
            self.duration_bucket_inf.load(Relaxed)
        ));
        out.push_str(&format!(
            "reasoning_cycle_duration_us_sum {}\n",
            self.duration_sum.load(Relaxed)
        ));
        out.push_str(&format!(
            "reasoning_cycle_duration_us_count {}\n",
            self.cycle_count.load(Relaxed)
        ));

        out
    }
}

impl Default for PrometheusMetrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Prometheus sink that updates in-memory metrics.
pub struct PrometheusSink {
    metrics: Arc<PrometheusMetrics>,
}

impl PrometheusSink {
    /// Create a new Prometheus sink.
    pub fn new(metrics: Arc<PrometheusMetrics>) -> Self {
        Self { metrics }
    }
}

impl TelemetrySink for PrometheusSink {
    fn export(&self, event: &ReasoningEvent) -> Result<(), TelemetryError> {
        self.metrics.record(event);
        Ok(())
    }

    fn flush(&self) -> Result<(), TelemetryError> {
        Ok(()) // Prometheus metrics are always "flushed"
    }

    fn name(&self) -> &str {
        "prometheus"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-Sink Exporter
// ─────────────────────────────────────────────────────────────────────────────

/// Telemetry exporter that fans out to multiple sinks.
pub struct TelemetryExporter {
    sinks: Vec<Box<dyn TelemetrySink>>,
}

impl TelemetryExporter {
    /// Create a new exporter with the given sinks.
    pub fn new(sinks: Vec<Box<dyn TelemetrySink>>) -> Self {
        Self { sinks }
    }

    /// Create an exporter builder.
    pub fn builder() -> TelemetryExporterBuilder {
        TelemetryExporterBuilder::new()
    }

    /// Export an event to all sinks.
    pub fn export(&self, event: &ReasoningEvent) -> Result<(), TelemetryError> {
        for sink in &self.sinks {
            sink.export(event)?;
        }
        Ok(())
    }

    /// Flush all sinks.
    pub fn flush(&self) -> Result<(), TelemetryError> {
        for sink in &self.sinks {
            sink.flush()?;
        }
        Ok(())
    }

    /// Get the names of all configured sinks.
    pub fn sink_names(&self) -> Vec<&str> {
        self.sinks.iter().map(|s| s.name()).collect()
    }
}

/// Builder for TelemetryExporter.
pub struct TelemetryExporterBuilder {
    sinks: Vec<Box<dyn TelemetrySink>>,
    prometheus_metrics: Option<Arc<PrometheusMetrics>>,
}

impl TelemetryExporterBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            sinks: Vec::new(),
            prometheus_metrics: None,
        }
    }

    /// Add a JSON Lines file sink.
    pub fn add_jsonl_sink(mut self, path: impl Into<PathBuf>) -> Result<Self, TelemetryError> {
        let sink = JsonLinesSink::new(path)?;
        self.sinks.push(Box::new(sink));
        Ok(self)
    }

    /// Add a CSV file sink.
    pub fn add_csv_sink(mut self, path: impl Into<PathBuf>) -> Result<Self, TelemetryError> {
        let sink = CsvSink::new(path)?;
        self.sinks.push(Box::new(sink));
        Ok(self)
    }

    /// Add a Prometheus metrics sink.
    ///
    /// Returns the PrometheusMetrics handle so you can serve it via HTTP.
    pub fn add_prometheus_sink(mut self) -> (Self, Arc<PrometheusMetrics>) {
        let metrics = Arc::new(PrometheusMetrics::new());
        let sink = PrometheusSink::new(Arc::clone(&metrics));
        self.sinks.push(Box::new(sink));
        self.prometheus_metrics = Some(Arc::clone(&metrics));
        (self, metrics)
    }

    /// Add a custom sink.
    pub fn add_sink(mut self, sink: Box<dyn TelemetrySink>) -> Self {
        self.sinks.push(sink);
        self
    }

    /// Build the exporter.
    pub fn build(self) -> TelemetryExporter {
        TelemetryExporter::new(self.sinks)
    }
}

impl Default for TelemetryExporterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Production Configuration Helper
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for production telemetry.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Path for JSON Lines log file (None to disable).
    pub jsonl_path: Option<PathBuf>,
    /// Path for CSV export file (None to disable).
    pub csv_path: Option<PathBuf>,
    /// Whether to enable Prometheus metrics.
    pub prometheus_enabled: bool,
}

impl TelemetryConfig {
    /// Create a production-ready configuration.
    ///
    /// Logs to `/var/log/symthaea/reasoning.jsonl` and exposes Prometheus metrics.
    pub fn production() -> Self {
        Self {
            jsonl_path: Some(PathBuf::from("/var/log/symthaea/reasoning.jsonl")),
            csv_path: None,
            prometheus_enabled: true,
        }
    }

    /// Create a development configuration.
    ///
    /// Logs to current directory, no Prometheus.
    pub fn development() -> Self {
        Self {
            jsonl_path: Some(PathBuf::from("reasoning.jsonl")),
            csv_path: Some(PathBuf::from("reasoning.csv")),
            prometheus_enabled: false,
        }
    }

    /// Create a minimal configuration (Prometheus only).
    pub fn minimal() -> Self {
        Self {
            jsonl_path: None,
            csv_path: None,
            prometheus_enabled: true,
        }
    }

    /// Build an exporter from this configuration.
    pub fn build_exporter(
        &self,
    ) -> Result<(TelemetryExporter, Option<Arc<PrometheusMetrics>>), TelemetryError> {
        let mut builder = TelemetryExporter::builder();
        let mut prometheus_metrics = None;

        if let Some(ref path) = self.jsonl_path {
            builder = builder.add_jsonl_sink(path)?;
        }

        if let Some(ref path) = self.csv_path {
            builder = builder.add_csv_sink(path)?;
        }

        if self.prometheus_enabled {
            let (b, metrics) = builder.add_prometheus_sink();
            builder = b;
            prometheus_metrics = Some(metrics);
        }

        Ok((builder.build(), prometheus_metrics))
    }
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self::development()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::consciousness::temporal_planning::types::BudgetTier;

    fn make_test_event() -> ReasoningEvent {
        let mut event = ReasoningEvent::new(1);
        event.wall_time_us = 1500;
        event.budget_tier = BudgetTier::Tier1;
        event.phi_raw = 0.8;
        event.reliability = 0.7;
        event.phi_eff = 0.56;
        event.gamma = 2.0;
        event.evs = 0.4;
        event.did_simulate = true;
        event.mcts_iterations = 50;
        event.plan_confidence = 0.8;
        event.gate_decision = "Allowed".to_string();
        event
    }

    #[test]
    fn test_prometheus_metrics_record() {
        let metrics = PrometheusMetrics::new();
        let event = make_test_event();

        metrics.record(&event);

        let rendered = metrics.render();
        assert!(rendered.contains("reasoning_cycles_total 1"));
        assert!(rendered.contains("reasoning_tier_cycles_total{tier=\"1\"} 1"));
        assert!(rendered.contains("reasoning_gate_decisions_total{result=\"allowed\"} 1"));
    }

    #[test]
    fn test_prometheus_histogram_buckets() {
        let metrics = PrometheusMetrics::new();

        // Record events at different durations
        for us in [50, 200, 800, 1500, 3000, 7000, 15000, 25000] {
            let mut event = make_test_event();
            event.wall_time_us = us;
            metrics.record(&event);
        }

        let rendered = metrics.render();
        // 50us fits in all buckets up to +Inf
        assert!(rendered.contains("reasoning_cycle_duration_us_bucket{le=\"100\"} 1"));
        // 200us, 800us fit in 500us+ buckets
        assert!(rendered.contains("reasoning_cycle_duration_us_bucket{le=\"500\"} 2"));
        // All 8 events in +Inf
        assert!(rendered.contains("reasoning_cycle_duration_us_bucket{le=\"+Inf\"} 8"));
    }

    #[test]
    fn test_telemetry_exporter_builder() {
        let (builder, metrics) = TelemetryExporter::builder().add_prometheus_sink();

        let exporter = builder.build();
        assert_eq!(exporter.sink_names(), vec!["prometheus"]);
        assert!(metrics.render().contains("reasoning_cycles_total"));
    }

    #[test]
    fn test_telemetry_config_production() {
        let config = TelemetryConfig::production();
        assert!(config.jsonl_path.is_some());
        assert!(config.prometheus_enabled);
    }

    #[test]
    fn test_telemetry_config_development() {
        let config = TelemetryConfig::development();
        assert!(config.jsonl_path.is_some());
        assert!(config.csv_path.is_some());
        assert!(!config.prometheus_enabled);
    }
}
