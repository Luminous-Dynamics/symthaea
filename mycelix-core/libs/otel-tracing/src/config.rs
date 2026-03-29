// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Configuration for OpenTelemetry tracing

use serde::{Deserialize, Serialize};

/// Configuration for the tracing system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TracingConfig {
    /// Name of the service for trace attribution
    pub service_name: String,

    /// Service version
    pub service_version: Option<String>,

    /// Environment (production, staging, development)
    pub environment: Option<String>,

    /// OTLP endpoint for trace export (e.g., "http://localhost:4317")
    pub otlp_endpoint: Option<String>,

    /// Whether to enable console output
    pub console_output: bool,

    /// Whether to enable JSON format for logs
    pub json_logs: bool,

    /// Log level filter (e.g., "info", "debug", "trace")
    pub log_level: String,

    /// Prometheus metrics endpoint port
    pub prometheus_port: Option<u16>,

    /// Sampling ratio for traces (0.0 to 1.0)
    pub sampling_ratio: f64,

    /// Additional resource attributes
    pub resource_attributes: Vec<(String, String)>,

    /// Whether to enable metrics
    pub enable_metrics: bool,

    /// Whether to enable distributed tracing
    pub enable_tracing: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "mycelix".to_string(),
            service_version: None,
            environment: Some("development".to_string()),
            otlp_endpoint: None,
            console_output: true,
            json_logs: false,
            log_level: "info".to_string(),
            prometheus_port: None,
            sampling_ratio: 1.0,
            resource_attributes: Vec::new(),
            enable_metrics: true,
            enable_tracing: true,
        }
    }
}

impl TracingConfig {
    /// Create a new builder for TracingConfig
    pub fn builder() -> TracingConfigBuilder {
        TracingConfigBuilder::default()
    }

    /// Create a minimal development configuration
    pub fn development(service_name: &str) -> Self {
        Self {
            service_name: service_name.to_string(),
            environment: Some("development".to_string()),
            console_output: true,
            json_logs: false,
            log_level: "debug".to_string(),
            ..Default::default()
        }
    }

    /// Create a production configuration
    pub fn production(service_name: &str, otlp_endpoint: &str) -> Self {
        Self {
            service_name: service_name.to_string(),
            environment: Some("production".to_string()),
            otlp_endpoint: Some(otlp_endpoint.to_string()),
            console_output: false,
            json_logs: true,
            log_level: "info".to_string(),
            sampling_ratio: 0.1, // 10% sampling in production
            prometheus_port: Some(9090),
            ..Default::default()
        }
    }

    /// Load configuration from environment variables
    pub fn from_env() -> Self {
        Self {
            service_name: std::env::var("OTEL_SERVICE_NAME")
                .unwrap_or_else(|_| "mycelix".to_string()),
            service_version: std::env::var("SERVICE_VERSION").ok(),
            environment: std::env::var("ENVIRONMENT").ok(),
            otlp_endpoint: std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok(),
            console_output: std::env::var("CONSOLE_OUTPUT")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(true),
            json_logs: std::env::var("JSON_LOGS")
                .map(|v| v == "true" || v == "1")
                .unwrap_or(false),
            log_level: std::env::var("LOG_LEVEL")
                .or_else(|_| std::env::var("RUST_LOG"))
                .unwrap_or_else(|_| "info".to_string()),
            prometheus_port: std::env::var("PROMETHEUS_PORT")
                .ok()
                .and_then(|p| p.parse().ok()),
            sampling_ratio: std::env::var("OTEL_SAMPLING_RATIO")
                .ok()
                .and_then(|r| r.parse().ok())
                .unwrap_or(1.0),
            resource_attributes: Vec::new(),
            enable_metrics: std::env::var("ENABLE_METRICS")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
            enable_tracing: std::env::var("ENABLE_TRACING")
                .map(|v| v != "false" && v != "0")
                .unwrap_or(true),
        }
    }
}

/// Builder for TracingConfig
#[derive(Debug, Default)]
pub struct TracingConfigBuilder {
    config: TracingConfig,
}

impl TracingConfigBuilder {
    /// Set the service name
    pub fn service_name(mut self, name: impl Into<String>) -> Self {
        self.config.service_name = name.into();
        self
    }

    /// Set the service version
    pub fn service_version(mut self, version: impl Into<String>) -> Self {
        self.config.service_version = Some(version.into());
        self
    }

    /// Set the environment
    pub fn environment(mut self, env: impl Into<String>) -> Self {
        self.config.environment = Some(env.into());
        self
    }

    /// Set the OTLP endpoint
    pub fn otlp_endpoint(mut self, endpoint: impl Into<String>) -> Self {
        self.config.otlp_endpoint = Some(endpoint.into());
        self
    }

    /// Enable or disable console output
    pub fn console_output(mut self, enabled: bool) -> Self {
        self.config.console_output = enabled;
        self
    }

    /// Enable or disable JSON logs
    pub fn json_logs(mut self, enabled: bool) -> Self {
        self.config.json_logs = enabled;
        self
    }

    /// Set the log level
    pub fn log_level(mut self, level: impl Into<String>) -> Self {
        self.config.log_level = level.into();
        self
    }

    /// Set the Prometheus port
    pub fn prometheus_port(mut self, port: u16) -> Self {
        self.config.prometheus_port = Some(port);
        self
    }

    /// Set the sampling ratio
    pub fn sampling_ratio(mut self, ratio: f64) -> Self {
        self.config.sampling_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    /// Add a resource attribute
    pub fn resource_attribute(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.config.resource_attributes.push((key.into(), value.into()));
        self
    }

    /// Enable or disable metrics
    pub fn enable_metrics(mut self, enabled: bool) -> Self {
        self.config.enable_metrics = enabled;
        self
    }

    /// Enable or disable tracing
    pub fn enable_tracing(mut self, enabled: bool) -> Self {
        self.config.enable_tracing = enabled;
        self
    }

    /// Build the configuration
    pub fn build(self) -> TracingConfig {
        self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TracingConfig::default();
        assert_eq!(config.service_name, "mycelix");
        assert!(config.console_output);
        assert!(!config.json_logs);
    }

    #[test]
    fn test_builder() {
        let config = TracingConfig::builder()
            .service_name("test-service")
            .otlp_endpoint("http://localhost:4317")
            .sampling_ratio(0.5)
            .build();

        assert_eq!(config.service_name, "test-service");
        assert_eq!(config.otlp_endpoint, Some("http://localhost:4317".to_string()));
        assert!((config.sampling_ratio - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_development_config() {
        let config = TracingConfig::development("dev-service");
        assert_eq!(config.service_name, "dev-service");
        assert!(config.console_output);
        assert_eq!(config.log_level, "debug");
    }

    #[test]
    fn test_production_config() {
        let config = TracingConfig::production("prod-service", "http://otel:4317");
        assert_eq!(config.service_name, "prod-service");
        assert!(!config.console_output);
        assert!(config.json_logs);
        assert!((config.sampling_ratio - 0.1).abs() < f64::EPSILON);
    }
}
