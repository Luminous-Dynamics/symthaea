// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Observability & Telemetry
//!
//! OpenTelemetry distributed tracing, metrics, and logging

use opentelemetry::{
    global,
    sdk::{
        export::trace::SpanExporter,
        propagation::TraceContextPropagator,
        trace::{self, RandomIdGenerator, Sampler, Tracer},
        Resource,
    },
    trace::{TraceError, TracerProvider},
    KeyValue,
};
use opentelemetry_otlp::{ExportConfig, Protocol, WithExportConfig};
use opentelemetry_semantic_conventions as semconv;
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tracing::Subscriber;
use tracing_opentelemetry::OpenTelemetryLayer;
use tracing_subscriber::{layer::SubscriberExt, registry::LookupSpan, EnvFilter, Layer};

/// Telemetry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    pub service_name: String,
    pub service_version: String,
    pub environment: String,
    pub otlp_endpoint: Option<String>,
    pub sentry_dsn: Option<String>,
    pub sample_rate: f64,
    pub log_level: String,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            service_name: "mycelix-mail-api".to_string(),
            service_version: env!("CARGO_PKG_VERSION").to_string(),
            environment: std::env::var("ENVIRONMENT").unwrap_or_else(|_| "development".to_string()),
            otlp_endpoint: std::env::var("OTLP_ENDPOINT").ok(),
            sentry_dsn: std::env::var("SENTRY_DSN").ok(),
            sample_rate: 1.0,
            log_level: "info".to_string(),
        }
    }
}

/// Initialize telemetry stack
pub fn init_telemetry(config: &TelemetryConfig) -> Result<(), Box<dyn std::error::Error>> {
    // Set global propagator
    global::set_text_map_propagator(TraceContextPropagator::new());

    // Build resource
    let resource = Resource::new(vec![
        KeyValue::new(semconv::resource::SERVICE_NAME, config.service_name.clone()),
        KeyValue::new(semconv::resource::SERVICE_VERSION, config.service_version.clone()),
        KeyValue::new(semconv::resource::DEPLOYMENT_ENVIRONMENT, config.environment.clone()),
    ]);

    // Initialize tracer if OTLP endpoint is configured
    if let Some(endpoint) = &config.otlp_endpoint {
        let tracer = opentelemetry_otlp::new_pipeline()
            .tracing()
            .with_exporter(
                opentelemetry_otlp::new_exporter()
                    .tonic()
                    .with_endpoint(endpoint)
                    .with_timeout(Duration::from_secs(3)),
            )
            .with_trace_config(
                trace::config()
                    .with_sampler(Sampler::TraceIdRatioBased(config.sample_rate))
                    .with_id_generator(RandomIdGenerator::default())
                    .with_resource(resource.clone()),
            )
            .install_batch(opentelemetry::runtime::Tokio)?;

        // Create tracing subscriber with OpenTelemetry layer
        let telemetry_layer = tracing_opentelemetry::layer().with_tracer(tracer);

        let subscriber = tracing_subscriber::registry()
            .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                EnvFilter::new(&config.log_level)
            }))
            .with(tracing_subscriber::fmt::layer().json())
            .with(telemetry_layer);

        tracing::subscriber::set_global_default(subscriber)?;
    } else {
        // Simple logging without OTLP
        let subscriber = tracing_subscriber::registry()
            .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| {
                EnvFilter::new(&config.log_level)
            }))
            .with(tracing_subscriber::fmt::layer().json());

        tracing::subscriber::set_global_default(subscriber)?;
    }

    // Initialize Sentry if DSN is configured
    if let Some(dsn) = &config.sentry_dsn {
        let _guard = sentry::init((
            dsn.as_str(),
            sentry::ClientOptions {
                release: Some(config.service_version.clone().into()),
                environment: Some(config.environment.clone().into()),
                sample_rate: config.sample_rate as f32,
                traces_sample_rate: config.sample_rate as f32,
                attach_stacktrace: true,
                ..Default::default()
            },
        ));
    }

    Ok(())
}

/// Shutdown telemetry providers
pub fn shutdown_telemetry() {
    global::shutdown_tracer_provider();
}

/// Custom span attributes for email operations
pub mod attributes {
    use opentelemetry::KeyValue;

    pub fn email_id(id: &str) -> KeyValue {
        KeyValue::new("email.id", id.to_string())
    }

    pub fn email_action(action: &str) -> KeyValue {
        KeyValue::new("email.action", action.to_string())
    }

    pub fn user_id(id: &str) -> KeyValue {
        KeyValue::new("user.id", id.to_string())
    }

    pub fn tenant_id(id: &str) -> KeyValue {
        KeyValue::new("tenant.id", id.to_string())
    }

    pub fn trust_score(score: f64) -> KeyValue {
        KeyValue::new("trust.score", score)
    }

    pub fn is_encrypted(encrypted: bool) -> KeyValue {
        KeyValue::new("email.encrypted", encrypted)
    }

    pub fn recipient_count(count: i64) -> KeyValue {
        KeyValue::new("email.recipient_count", count)
    }

    pub fn attachment_count(count: i64) -> KeyValue {
        KeyValue::new("email.attachment_count", count)
    }
}

/// Metrics definitions
pub mod metrics {
    use once_cell::sync::Lazy;
    use prometheus::{
        register_counter_vec, register_histogram_vec, register_gauge_vec,
        CounterVec, HistogramVec, GaugeVec, Opts, DEFAULT_BUCKETS,
    };

    // Email metrics
    pub static EMAILS_SENT: Lazy<CounterVec> = Lazy::new(|| {
        register_counter_vec!(
            Opts::new("mycelix_emails_sent_total", "Total emails sent"),
            &["status", "encrypted", "tenant"]
        )
        .unwrap()
    });

    pub static EMAILS_RECEIVED: Lazy<CounterVec> = Lazy::new(|| {
        register_counter_vec!(
            Opts::new("mycelix_emails_received_total", "Total emails received"),
            &["spam_status", "trust_level", "tenant"]
        )
        .unwrap()
    });

    pub static EMAIL_SEND_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
        register_histogram_vec!(
            "mycelix_email_send_duration_seconds",
            "Time to send an email",
            &["encrypted"],
            vec![0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        .unwrap()
    });

    // Trust metrics
    pub static TRUST_SCORE_DISTRIBUTION: Lazy<HistogramVec> = Lazy::new(|| {
        register_histogram_vec!(
            "mycelix_trust_score_distribution",
            "Distribution of trust scores",
            &["context"],
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        )
        .unwrap()
    });

    pub static ATTESTATIONS_CREATED: Lazy<CounterVec> = Lazy::new(|| {
        register_counter_vec!(
            Opts::new("mycelix_attestations_created_total", "Total attestations created"),
            &["context", "tenant"]
        )
        .unwrap()
    });

    // API metrics
    pub static HTTP_REQUESTS: Lazy<CounterVec> = Lazy::new(|| {
        register_counter_vec!(
            Opts::new("mycelix_http_requests_total", "Total HTTP requests"),
            &["method", "path", "status"]
        )
        .unwrap()
    });

    pub static HTTP_REQUEST_DURATION: Lazy<HistogramVec> = Lazy::new(|| {
        register_histogram_vec!(
            "mycelix_http_request_duration_seconds",
            "HTTP request duration",
            &["method", "path"],
            DEFAULT_BUCKETS.to_vec()
        )
        .unwrap()
    });

    // Connection metrics
    pub static ACTIVE_CONNECTIONS: Lazy<GaugeVec> = Lazy::new(|| {
        register_gauge_vec!(
            Opts::new("mycelix_active_connections", "Active connections"),
            &["type"]
        )
        .unwrap()
    });

    pub static WEBSOCKET_CONNECTIONS: Lazy<GaugeVec> = Lazy::new(|| {
        register_gauge_vec!(
            Opts::new("mycelix_websocket_connections", "Active WebSocket connections"),
            &["tenant"]
        )
        .unwrap()
    });

    // Queue metrics
    pub static EMAIL_QUEUE_SIZE: Lazy<GaugeVec> = Lazy::new(|| {
        register_gauge_vec!(
            Opts::new("mycelix_email_queue_size", "Email queue size"),
            &["status"]
        )
        .unwrap()
    });

    pub static EMAIL_QUEUE_LATENCY: Lazy<HistogramVec> = Lazy::new(|| {
        register_histogram_vec!(
            "mycelix_email_queue_latency_seconds",
            "Time from queue to send",
            &["priority"],
            vec![1.0, 5.0, 15.0, 30.0, 60.0, 300.0]
        )
        .unwrap()
    });

    // Encryption metrics
    pub static ENCRYPTION_OPERATIONS: Lazy<CounterVec> = Lazy::new(|| {
        register_counter_vec!(
            Opts::new("mycelix_encryption_operations_total", "Encryption operations"),
            &["operation", "algorithm"]
        )
        .unwrap()
    });

    pub static KEY_ROTATIONS: Lazy<CounterVec> = Lazy::new(|| {
        register_counter_vec!(
            Opts::new("mycelix_key_rotations_total", "Key rotations"),
            &["type"]
        )
        .unwrap()
    });
}

/// Alerting rules configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub name: String,
    pub condition: String,
    pub threshold: f64,
    pub duration: Duration,
    pub severity: AlertSeverity,
    pub channels: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Default alerting rules
pub fn default_alert_rules() -> Vec<AlertRule> {
    vec![
        AlertRule {
            name: "High Error Rate".to_string(),
            condition: "rate(mycelix_http_requests_total{status=~\"5..\"}[5m]) / rate(mycelix_http_requests_total[5m]) > 0.05".to_string(),
            threshold: 0.05,
            duration: Duration::from_secs(300),
            severity: AlertSeverity::Critical,
            channels: vec!["slack".to_string(), "pagerduty".to_string()],
        },
        AlertRule {
            name: "Email Delivery Failures".to_string(),
            condition: "rate(mycelix_emails_sent_total{status=\"failed\"}[5m]) > 10".to_string(),
            threshold: 10.0,
            duration: Duration::from_secs(300),
            severity: AlertSeverity::Warning,
            channels: vec!["slack".to_string()],
        },
        AlertRule {
            name: "High Spam Rate".to_string(),
            condition: "rate(mycelix_emails_received_total{spam_status=\"spam\"}[1h]) / rate(mycelix_emails_received_total[1h]) > 0.3".to_string(),
            threshold: 0.3,
            duration: Duration::from_secs(3600),
            severity: AlertSeverity::Warning,
            channels: vec!["slack".to_string()],
        },
        AlertRule {
            name: "Queue Backlog".to_string(),
            condition: "mycelix_email_queue_size{status=\"pending\"} > 1000".to_string(),
            threshold: 1000.0,
            duration: Duration::from_secs(600),
            severity: AlertSeverity::Warning,
            channels: vec!["slack".to_string()],
        },
        AlertRule {
            name: "Low Trust Score Spike".to_string(),
            condition: "rate(mycelix_emails_received_total{trust_level=\"unknown\"}[1h]) > 100".to_string(),
            threshold: 100.0,
            duration: Duration::from_secs(3600),
            severity: AlertSeverity::Info,
            channels: vec!["slack".to_string()],
        },
    ]
}
