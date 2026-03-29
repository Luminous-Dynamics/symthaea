// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mycelix OpenTelemetry Tracing Library
//!
//! Provides unified observability infrastructure:
//! - Distributed tracing with OTLP export (Jaeger, Zipkin, etc.)
//! - Prometheus metrics integration
//! - Structured logging with trace correlation
//!
//! # Example
//!
//! ```rust,ignore
//! use mycelix_otel_tracing::{TracingConfig, init_tracing};
//!
//! #[tokio::main]
//! async fn main() {
//!     let config = TracingConfig::builder()
//!         .service_name("mycelix-fl")
//!         .otlp_endpoint("http://localhost:4317")
//!         .build();
//!
//!     let _guard = init_tracing(config).expect("Failed to init tracing");
//!
//!     tracing::info!("Service started");
//! }
//! ```

pub mod config;
pub mod metrics;
pub mod tracer;
pub mod propagator;

pub use config::{TracingConfig, TracingConfigBuilder};
pub use metrics::{MetricsRegistry, Counter, Histogram, Gauge};
pub use tracer::{init_tracing, TracingGuard};

use thiserror::Error;

/// Errors that can occur in tracing operations
#[derive(Error, Debug)]
pub enum TracingError {
    #[error("Failed to initialize tracer: {0}")]
    InitError(String),

    #[error("Failed to export telemetry: {0}")]
    ExportError(String),

    #[error("Invalid configuration: {0}")]
    ConfigError(String),

    #[error("OpenTelemetry error: {0}")]
    OtelError(#[from] opentelemetry::trace::TraceError),

    #[error("Metrics error: {0}")]
    MetricsError(String),
}

/// Result type for tracing operations
pub type TracingResult<T> = Result<T, TracingError>;

/// Version of the library
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Federated Learning specific metrics
pub mod fl_metrics {
    use super::metrics::{Counter, Histogram, Gauge};
    use std::sync::OnceLock;

    static FL_METRICS: OnceLock<FlMetrics> = OnceLock::new();

    /// Federated Learning metrics collection
    pub struct FlMetrics {
        /// Number of training rounds completed
        pub rounds_completed: Counter,
        /// Number of Byzantine clients detected
        pub byzantine_detected: Counter,
        /// Aggregation latency in milliseconds
        pub aggregation_latency_ms: Histogram,
        /// Number of active clients
        pub active_clients: Gauge,
        /// Current model accuracy
        pub model_accuracy: Gauge,
        /// Privacy budget consumed (epsilon)
        pub privacy_epsilon_consumed: Gauge,
    }

    impl FlMetrics {
        /// Initialize FL metrics
        pub fn init(registry: &super::MetricsRegistry) -> &'static Self {
            FL_METRICS.get_or_init(|| {
                Self {
                    rounds_completed: registry.counter(
                        "fl_rounds_completed_total",
                        "Total federated learning rounds completed",
                    ),
                    byzantine_detected: registry.counter(
                        "fl_byzantine_detected_total",
                        "Total Byzantine clients detected",
                    ),
                    aggregation_latency_ms: registry.histogram(
                        "fl_aggregation_latency_ms",
                        "Aggregation latency in milliseconds",
                        vec![0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0],
                    ),
                    active_clients: registry.gauge(
                        "fl_active_clients",
                        "Number of currently active FL clients",
                    ),
                    model_accuracy: registry.gauge(
                        "fl_model_accuracy",
                        "Current model accuracy (0-1)",
                    ),
                    privacy_epsilon_consumed: registry.gauge(
                        "fl_privacy_epsilon_consumed",
                        "Privacy budget consumed (epsilon)",
                    ),
                }
            })
        }

        /// Get the global FL metrics instance
        pub fn get() -> Option<&'static Self> {
            FL_METRICS.get()
        }
    }

    /// Record a completed round
    pub fn record_round_completed() {
        if let Some(m) = FlMetrics::get() {
            m.rounds_completed.inc();
        }
    }

    /// Record Byzantine client detection
    pub fn record_byzantine_detected(count: u64) {
        if let Some(m) = FlMetrics::get() {
            m.byzantine_detected.inc_by(count);
        }
    }

    /// Record aggregation latency
    pub fn record_aggregation_latency(ms: f64) {
        if let Some(m) = FlMetrics::get() {
            m.aggregation_latency_ms.observe(ms);
        }
    }

    /// Set active clients count
    pub fn set_active_clients(count: i64) {
        if let Some(m) = FlMetrics::get() {
            m.active_clients.set(count as f64);
        }
    }

    /// Set model accuracy
    pub fn set_model_accuracy(accuracy: f64) {
        if let Some(m) = FlMetrics::get() {
            m.model_accuracy.set(accuracy);
        }
    }

    /// Set privacy epsilon consumed
    pub fn set_privacy_epsilon(epsilon: f64) {
        if let Some(m) = FlMetrics::get() {
            m.privacy_epsilon_consumed.set(epsilon);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert_eq!(VERSION, "0.1.0");
    }
}
