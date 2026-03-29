// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Metrics collection using OpenTelemetry

use opentelemetry::metrics::{Meter, MeterProvider};
use opentelemetry_sdk::metrics::SdkMeterProvider;
use std::sync::Arc;

/// Counter metric - monotonically increasing value
#[derive(Clone)]
pub struct Counter {
    inner: opentelemetry::metrics::Counter<u64>,
}

impl Counter {
    /// Increment the counter by 1
    pub fn inc(&self) {
        self.inner.add(1, &[]);
    }

    /// Increment the counter by a specific amount
    pub fn inc_by(&self, value: u64) {
        self.inner.add(value, &[]);
    }

    /// Increment with attributes
    pub fn inc_with_attrs(&self, attrs: &[opentelemetry::KeyValue]) {
        self.inner.add(1, attrs);
    }
}

/// Gauge metric - can go up or down
#[derive(Clone)]
pub struct Gauge {
    inner: opentelemetry::metrics::Gauge<f64>,
}

impl Gauge {
    /// Set the gauge value
    pub fn set(&self, value: f64) {
        self.inner.record(value, &[]);
    }

    /// Set with attributes
    pub fn set_with_attrs(&self, value: f64, attrs: &[opentelemetry::KeyValue]) {
        self.inner.record(value, attrs);
    }
}

/// Histogram metric - records distributions
#[derive(Clone)]
pub struct Histogram {
    inner: opentelemetry::metrics::Histogram<f64>,
}

impl Histogram {
    /// Observe a value
    pub fn observe(&self, value: f64) {
        self.inner.record(value, &[]);
    }

    /// Observe with attributes
    pub fn observe_with_attrs(&self, value: f64, attrs: &[opentelemetry::KeyValue]) {
        self.inner.record(value, attrs);
    }
}

/// Timer guard for automatic duration recording
pub struct Timer {
    histogram: Histogram,
    start: std::time::Instant,
    attrs: Vec<opentelemetry::KeyValue>,
}

impl Timer {
    /// Create a new timer
    pub fn new(histogram: &Histogram) -> Self {
        Self {
            histogram: histogram.clone(),
            start: std::time::Instant::now(),
            attrs: Vec::new(),
        }
    }

    /// Create a timer with attributes
    pub fn with_attrs(histogram: &Histogram, attrs: Vec<opentelemetry::KeyValue>) -> Self {
        Self {
            histogram: histogram.clone(),
            start: std::time::Instant::now(),
            attrs,
        }
    }

    /// Stop the timer and record the duration
    pub fn stop(self) -> f64 {
        let duration = self.start.elapsed().as_secs_f64() * 1000.0; // Convert to ms
        if self.attrs.is_empty() {
            self.histogram.observe(duration);
        } else {
            self.histogram.observe_with_attrs(duration, &self.attrs);
        }
        duration
    }
}

/// Metrics registry for creating and managing metrics
pub struct MetricsRegistry {
    meter: Meter,
    #[allow(dead_code)]
    provider: Arc<SdkMeterProvider>,
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new(provider: SdkMeterProvider, name: &'static str) -> Self {
        let meter = provider.meter(name);
        Self {
            meter,
            provider: Arc::new(provider),
        }
    }

    /// Create a counter metric
    pub fn counter(&self, name: &'static str, description: &'static str) -> Counter {
        Counter {
            inner: self.meter
                .u64_counter(name)
                .with_description(description)
                .build(),
        }
    }

    /// Create a gauge metric
    pub fn gauge(&self, name: &'static str, description: &'static str) -> Gauge {
        Gauge {
            inner: self.meter
                .f64_gauge(name)
                .with_description(description)
                .build(),
        }
    }

    /// Create a histogram metric
    pub fn histogram(&self, name: &'static str, description: &'static str, _buckets: Vec<f64>) -> Histogram {
        Histogram {
            inner: self.meter
                .f64_histogram(name)
                .with_description(description)
                .build(),
        }
    }

    /// Create a histogram with default buckets
    pub fn histogram_default(&self, name: &'static str, description: &'static str) -> Histogram {
        self.histogram(
            name,
            description,
            vec![0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
    }

    /// Start a timer for a histogram
    pub fn start_timer(&self, histogram: &Histogram) -> Timer {
        Timer::new(histogram)
    }
}

/// Pre-defined common metrics
pub struct CommonMetrics {
    /// HTTP request duration
    pub http_request_duration_ms: Histogram,
    /// HTTP request count
    pub http_requests_total: Counter,
    /// Active connections
    pub active_connections: Gauge,
    /// Error count
    pub errors_total: Counter,
}

impl CommonMetrics {
    /// Create common metrics
    pub fn new(registry: &MetricsRegistry) -> Self {
        Self {
            http_request_duration_ms: registry.histogram(
                "http_request_duration_ms",
                "HTTP request duration in milliseconds",
                vec![5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0],
            ),
            http_requests_total: registry.counter(
                "http_requests_total",
                "Total HTTP requests",
            ),
            active_connections: registry.gauge(
                "active_connections",
                "Number of active connections",
            ),
            errors_total: registry.counter(
                "errors_total",
                "Total errors",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use opentelemetry_sdk::metrics::SdkMeterProvider;

    fn create_test_registry() -> MetricsRegistry {
        let provider = SdkMeterProvider::default();
        MetricsRegistry::new(provider, "test")
    }

    #[test]
    fn test_counter() {
        let registry = create_test_registry();
        let counter = registry.counter("test_counter", "A test counter");

        counter.inc();
        counter.inc_by(5);
        // Counter operations don't panic
    }

    #[test]
    fn test_gauge() {
        let registry = create_test_registry();
        let gauge = registry.gauge("test_gauge", "A test gauge");

        gauge.set(42.0);
        gauge.set(-10.0);
        // Gauge operations don't panic
    }

    #[test]
    fn test_histogram() {
        let registry = create_test_registry();
        let histogram = registry.histogram_default("test_histogram", "A test histogram");

        histogram.observe(0.5);
        histogram.observe(1.5);
        histogram.observe(10.0);
        // Histogram operations don't panic
    }

    #[test]
    fn test_timer() {
        let registry = create_test_registry();
        let histogram = registry.histogram_default("test_timer", "A test timer");

        let timer = Timer::new(&histogram);
        std::thread::sleep(std::time::Duration::from_millis(10));
        let duration = timer.stop();

        assert!(duration >= 10.0);
    }
}
