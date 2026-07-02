// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tracer initialization and management

use crate::{TracingConfig, TracingError, TracingResult, MetricsRegistry};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_sdk::{
    trace::{Sampler, TracerProvider as SdkTracerProvider},
    metrics::SdkMeterProvider,
    Resource,
};
use opentelemetry::KeyValue;
use tracing_subscriber::{
    layer::SubscriberExt,
    util::SubscriberInitExt,
    EnvFilter,
    Layer,
};

/// Guard that shuts down tracing when dropped
pub struct TracingGuard {
    tracer_provider: Option<SdkTracerProvider>,
    meter_provider: Option<SdkMeterProvider>,
}

impl TracingGuard {
    /// Get the metrics registry if metrics are enabled
    pub fn metrics_registry(&self) -> Option<MetricsRegistry> {
        self.meter_provider.as_ref().map(|p| {
            MetricsRegistry::new(p.clone(), "mycelix")
        })
    }
}

impl Drop for TracingGuard {
    fn drop(&mut self) {
        if let Some(provider) = self.tracer_provider.take() {
            if let Err(e) = provider.shutdown() {
                eprintln!("Error shutting down tracer provider: {:?}", e);
            }
        }
        if let Some(provider) = self.meter_provider.take() {
            if let Err(e) = provider.shutdown() {
                eprintln!("Error shutting down meter provider: {:?}", e);
            }
        }
    }
}

/// Initialize the tracing system
pub fn init_tracing(config: TracingConfig) -> TracingResult<TracingGuard> {
    let mut guard = TracingGuard {
        tracer_provider: None,
        meter_provider: None,
    };

    // Build resource attributes
    let mut attributes = vec![
        KeyValue::new("service.name", config.service_name.clone()),
    ];

    if let Some(ref version) = config.service_version {
        attributes.push(KeyValue::new("service.version", version.clone()));
    }

    if let Some(ref env) = config.environment {
        attributes.push(KeyValue::new("deployment.environment", env.clone()));
    }

    for (key, value) in &config.resource_attributes {
        attributes.push(KeyValue::new(key.clone(), value.clone()));
    }

    let resource = Resource::new(attributes);

    // Build env filter
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&config.log_level));

    // Build layers
    let mut layers: Vec<Box<dyn Layer<_> + Send + Sync>> = Vec::new();

    // Console output layer
    if config.console_output {
        if config.json_logs {
            let json_layer = tracing_subscriber::fmt::layer()
                .json()
                .with_target(true)
                .with_thread_ids(true)
                .with_file(true)
                .with_line_number(true)
                .boxed();
            layers.push(json_layer);
        } else {
            let fmt_layer = tracing_subscriber::fmt::layer()
                .with_target(true)
                .with_thread_ids(false)
                .with_file(false)
                .with_line_number(false)
                .boxed();
            layers.push(fmt_layer);
        }
    }

    // OTLP tracing layer
    if config.enable_tracing {
        if let Some(ref endpoint) = config.otlp_endpoint {
            let sampler = if config.sampling_ratio >= 1.0 {
                Sampler::AlwaysOn
            } else if config.sampling_ratio <= 0.0 {
                Sampler::AlwaysOff
            } else {
                Sampler::TraceIdRatioBased(config.sampling_ratio)
            };

            // Try to create OTLP exporter - this may fail if endpoint is not reachable
            match create_otlp_tracer(&endpoint, sampler, resource.clone()) {
                Ok((provider, tracer)) => {
                    let otel_layer = tracing_opentelemetry::layer().with_tracer(tracer).boxed();
                    layers.push(otel_layer);
                    guard.tracer_provider = Some(provider);
                }
                Err(e) => {
                    tracing::warn!("Failed to create OTLP tracer: {}. Continuing without OTLP export.", e);
                }
            }
        }
    }

    // Metrics - use default no-op provider if OTLP not configured
    if config.enable_metrics {
        let meter_provider = SdkMeterProvider::default();
        guard.meter_provider = Some(meter_provider);
    }

    // Initialize subscriber
    tracing_subscriber::registry()
        .with(env_filter)
        .with(layers)
        .try_init()
        .map_err(|e| TracingError::InitError(e.to_string()))?;

    tracing::info!(
        service = %config.service_name,
        version = config.service_version.as_deref().unwrap_or("unknown"),
        environment = config.environment.as_deref().unwrap_or("unknown"),
        "Tracing initialized"
    );

    Ok(guard)
}

/// Create OTLP tracer (simplified version)
fn create_otlp_tracer(
    _endpoint: &str,
    sampler: Sampler,
    resource: Resource,
) -> TracingResult<(SdkTracerProvider, opentelemetry_sdk::trace::Tracer)> {
    // For now, create a simple tracer without OTLP export
    // Full OTLP integration requires runtime async setup
    let provider = SdkTracerProvider::builder()
        .with_sampler(sampler)
        .with_resource(resource)
        .build();

    let tracer = provider.tracer("mycelix");

    Ok((provider, tracer))
}

/// Initialize tracing with default development configuration
pub fn init_dev_tracing(service_name: &str) -> TracingResult<TracingGuard> {
    init_tracing(TracingConfig::development(service_name))
}

/// Initialize tracing from environment variables
pub fn init_tracing_from_env() -> TracingResult<TracingGuard> {
    init_tracing(TracingConfig::from_env())
}

/// Span extension trait for adding common attributes
pub trait SpanExt {
    /// Set the status to error
    fn set_error(&self, error: &dyn std::error::Error);

    /// Add a string attribute
    fn set_attribute(&self, key: &str, value: impl Into<String>);
}

impl SpanExt for tracing::Span {
    fn set_error(&self, error: &dyn std::error::Error) {
        self.record("otel.status_code", "ERROR");
        self.record("error.message", error.to_string().as_str());
    }

    fn set_attribute(&self, key: &str, value: impl Into<String>) {
        self.record(key, value.into().as_str());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dev_config() {
        // Just verify config creation doesn't panic
        let config = TracingConfig::development("test");
        assert_eq!(config.service_name, "test");
        assert!(config.console_output);
    }

    #[test]
    fn test_env_config() {
        // Just verify env config creation doesn't panic
        let config = TracingConfig::from_env();
        assert!(!config.service_name.is_empty());
    }
}
