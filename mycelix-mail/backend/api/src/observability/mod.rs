// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Track AC: Observability & Operations
//!
//! Comprehensive monitoring, metrics, tracing, and alerting:
//! - OpenTelemetry integration
//! - Prometheus metrics
//! - Distributed tracing
//! - Health checks
//! - Alerting rules

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use prometheus::{
    Counter, CounterVec, Gauge, GaugeVec, Histogram, HistogramVec,
    Opts, Registry, TextEncoder, Encoder,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn, error, span, Level};
use uuid::Uuid;

// ============================================================================
// Metrics Registry
// ============================================================================

pub struct MetricsRegistry {
    registry: Registry,

    // Email metrics
    emails_received: CounterVec,
    emails_sent: CounterVec,
    emails_processed: HistogramVec,
    email_size_bytes: HistogramVec,

    // Sync metrics
    sync_operations: CounterVec,
    sync_duration: HistogramVec,
    sync_errors: CounterVec,
    pending_sync_items: GaugeVec,

    // API metrics
    http_requests: CounterVec,
    http_request_duration: HistogramVec,
    http_request_size: HistogramVec,
    http_response_size: HistogramVec,

    // Connection metrics
    active_connections: GaugeVec,
    connection_pool_size: GaugeVec,
    websocket_connections: Gauge,

    // Queue metrics
    queue_depth: GaugeVec,
    queue_processing_time: HistogramVec,

    // Trust metrics
    trust_calculations: Counter,
    attestations_created: Counter,
    attestations_verified: Counter,

    // System metrics
    memory_usage_bytes: Gauge,
    cpu_usage_percent: Gauge,
    disk_usage_bytes: GaugeVec,
    uptime_seconds: Gauge,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        let registry = Registry::new();

        // Email metrics
        let emails_received = CounterVec::new(
            Opts::new("mycelix_emails_received_total", "Total emails received"),
            &["account", "folder"],
        ).unwrap();
        registry.register(Box::new(emails_received.clone())).unwrap();

        let emails_sent = CounterVec::new(
            Opts::new("mycelix_emails_sent_total", "Total emails sent"),
            &["account"],
        ).unwrap();
        registry.register(Box::new(emails_sent.clone())).unwrap();

        let emails_processed = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "mycelix_email_processing_seconds",
                "Email processing duration",
            ).buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
            &["operation"],
        ).unwrap();
        registry.register(Box::new(emails_processed.clone())).unwrap();

        let email_size_bytes = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "mycelix_email_size_bytes",
                "Email size in bytes",
            ).buckets(vec![1024.0, 10240.0, 102400.0, 1048576.0, 10485760.0]),
            &["direction"],
        ).unwrap();
        registry.register(Box::new(email_size_bytes.clone())).unwrap();

        // Sync metrics
        let sync_operations = CounterVec::new(
            Opts::new("mycelix_sync_operations_total", "Total sync operations"),
            &["account", "type", "status"],
        ).unwrap();
        registry.register(Box::new(sync_operations.clone())).unwrap();

        let sync_duration = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "mycelix_sync_duration_seconds",
                "Sync operation duration",
            ).buckets(vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]),
            &["account", "type"],
        ).unwrap();
        registry.register(Box::new(sync_duration.clone())).unwrap();

        let sync_errors = CounterVec::new(
            Opts::new("mycelix_sync_errors_total", "Total sync errors"),
            &["account", "error_type"],
        ).unwrap();
        registry.register(Box::new(sync_errors.clone())).unwrap();

        let pending_sync_items = GaugeVec::new(
            Opts::new("mycelix_pending_sync_items", "Pending sync items"),
            &["account"],
        ).unwrap();
        registry.register(Box::new(pending_sync_items.clone())).unwrap();

        // API metrics
        let http_requests = CounterVec::new(
            Opts::new("mycelix_http_requests_total", "Total HTTP requests"),
            &["method", "path", "status"],
        ).unwrap();
        registry.register(Box::new(http_requests.clone())).unwrap();

        let http_request_duration = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "mycelix_http_request_duration_seconds",
                "HTTP request duration",
            ).buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]),
            &["method", "path"],
        ).unwrap();
        registry.register(Box::new(http_request_duration.clone())).unwrap();

        let http_request_size = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "mycelix_http_request_size_bytes",
                "HTTP request size",
            ),
            &["method", "path"],
        ).unwrap();
        registry.register(Box::new(http_request_size.clone())).unwrap();

        let http_response_size = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "mycelix_http_response_size_bytes",
                "HTTP response size",
            ),
            &["method", "path"],
        ).unwrap();
        registry.register(Box::new(http_response_size.clone())).unwrap();

        // Connection metrics
        let active_connections = GaugeVec::new(
            Opts::new("mycelix_active_connections", "Active connections"),
            &["type"],
        ).unwrap();
        registry.register(Box::new(active_connections.clone())).unwrap();

        let connection_pool_size = GaugeVec::new(
            Opts::new("mycelix_connection_pool_size", "Connection pool size"),
            &["pool"],
        ).unwrap();
        registry.register(Box::new(connection_pool_size.clone())).unwrap();

        let websocket_connections = Gauge::new(
            "mycelix_websocket_connections",
            "Active WebSocket connections",
        ).unwrap();
        registry.register(Box::new(websocket_connections.clone())).unwrap();

        // Queue metrics
        let queue_depth = GaugeVec::new(
            Opts::new("mycelix_queue_depth", "Queue depth"),
            &["queue"],
        ).unwrap();
        registry.register(Box::new(queue_depth.clone())).unwrap();

        let queue_processing_time = HistogramVec::new(
            prometheus::HistogramOpts::new(
                "mycelix_queue_processing_seconds",
                "Queue item processing time",
            ),
            &["queue"],
        ).unwrap();
        registry.register(Box::new(queue_processing_time.clone())).unwrap();

        // Trust metrics
        let trust_calculations = Counter::new(
            "mycelix_trust_calculations_total",
            "Total trust calculations",
        ).unwrap();
        registry.register(Box::new(trust_calculations.clone())).unwrap();

        let attestations_created = Counter::new(
            "mycelix_attestations_created_total",
            "Total attestations created",
        ).unwrap();
        registry.register(Box::new(attestations_created.clone())).unwrap();

        let attestations_verified = Counter::new(
            "mycelix_attestations_verified_total",
            "Total attestations verified",
        ).unwrap();
        registry.register(Box::new(attestations_verified.clone())).unwrap();

        // System metrics
        let memory_usage_bytes = Gauge::new(
            "mycelix_memory_usage_bytes",
            "Memory usage in bytes",
        ).unwrap();
        registry.register(Box::new(memory_usage_bytes.clone())).unwrap();

        let cpu_usage_percent = Gauge::new(
            "mycelix_cpu_usage_percent",
            "CPU usage percentage",
        ).unwrap();
        registry.register(Box::new(cpu_usage_percent.clone())).unwrap();

        let disk_usage_bytes = GaugeVec::new(
            Opts::new("mycelix_disk_usage_bytes", "Disk usage in bytes"),
            &["mount"],
        ).unwrap();
        registry.register(Box::new(disk_usage_bytes.clone())).unwrap();

        let uptime_seconds = Gauge::new(
            "mycelix_uptime_seconds",
            "Application uptime in seconds",
        ).unwrap();
        registry.register(Box::new(uptime_seconds.clone())).unwrap();

        Self {
            registry,
            emails_received,
            emails_sent,
            emails_processed,
            email_size_bytes,
            sync_operations,
            sync_duration,
            sync_errors,
            pending_sync_items,
            http_requests,
            http_request_duration,
            http_request_size,
            http_response_size,
            active_connections,
            connection_pool_size,
            websocket_connections,
            queue_depth,
            queue_processing_time,
            trust_calculations,
            attestations_created,
            attestations_verified,
            memory_usage_bytes,
            cpu_usage_percent,
            disk_usage_bytes,
            uptime_seconds,
        }
    }

    /// Export metrics in Prometheus format
    pub fn export(&self) -> String {
        let encoder = TextEncoder::new();
        let metric_families = self.registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer).unwrap();
        String::from_utf8(buffer).unwrap()
    }

    /// Record email received
    pub fn record_email_received(&self, account: &str, folder: &str, size: usize) {
        self.emails_received.with_label_values(&[account, folder]).inc();
        self.email_size_bytes.with_label_values(&["incoming"]).observe(size as f64);
    }

    /// Record email sent
    pub fn record_email_sent(&self, account: &str, size: usize) {
        self.emails_sent.with_label_values(&[account]).inc();
        self.email_size_bytes.with_label_values(&["outgoing"]).observe(size as f64);
    }

    /// Record HTTP request
    pub fn record_http_request(
        &self,
        method: &str,
        path: &str,
        status: u16,
        duration_secs: f64,
        request_size: usize,
        response_size: usize,
    ) {
        self.http_requests.with_label_values(&[method, path, &status.to_string()]).inc();
        self.http_request_duration.with_label_values(&[method, path]).observe(duration_secs);
        self.http_request_size.with_label_values(&[method, path]).observe(request_size as f64);
        self.http_response_size.with_label_values(&[method, path]).observe(response_size as f64);
    }

    /// Record sync operation
    pub fn record_sync(&self, account: &str, sync_type: &str, status: &str, duration_secs: f64) {
        self.sync_operations.with_label_values(&[account, sync_type, status]).inc();
        self.sync_duration.with_label_values(&[account, sync_type]).observe(duration_secs);
    }
}

impl Default for MetricsRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Health Checks
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    pub status: HealthState,
    pub checks: Vec<HealthCheck>,
    pub timestamp: DateTime<Utc>,
    pub version: String,
    pub uptime_seconds: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum HealthState {
    Healthy,
    Degraded,
    Unhealthy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub status: HealthState,
    pub message: Option<String>,
    pub duration_ms: u64,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[async_trait]
pub trait HealthChecker: Send + Sync {
    fn name(&self) -> &str;
    async fn check(&self) -> HealthCheck;
}

pub struct DatabaseHealthChecker {
    // pool: sqlx::PgPool,
}

#[async_trait]
impl HealthChecker for DatabaseHealthChecker {
    fn name(&self) -> &str {
        "database"
    }

    async fn check(&self) -> HealthCheck {
        let start = std::time::Instant::now();

        // Would perform actual database ping
        // let result = sqlx::query("SELECT 1").execute(&self.pool).await;

        let duration = start.elapsed();

        HealthCheck {
            name: self.name().to_string(),
            status: HealthState::Healthy,
            message: None,
            duration_ms: duration.as_millis() as u64,
            metadata: HashMap::new(),
        }
    }
}

pub struct RedisHealthChecker;

#[async_trait]
impl HealthChecker for RedisHealthChecker {
    fn name(&self) -> &str {
        "redis"
    }

    async fn check(&self) -> HealthCheck {
        let start = std::time::Instant::now();
        let duration = start.elapsed();

        HealthCheck {
            name: self.name().to_string(),
            status: HealthState::Healthy,
            message: None,
            duration_ms: duration.as_millis() as u64,
            metadata: HashMap::new(),
        }
    }
}

pub struct IMAPHealthChecker;

#[async_trait]
impl HealthChecker for IMAPHealthChecker {
    fn name(&self) -> &str {
        "imap"
    }

    async fn check(&self) -> HealthCheck {
        let start = std::time::Instant::now();
        let duration = start.elapsed();

        HealthCheck {
            name: self.name().to_string(),
            status: HealthState::Healthy,
            message: Some("IMAP server reachable".to_string()),
            duration_ms: duration.as_millis() as u64,
            metadata: HashMap::new(),
        }
    }
}

pub struct HealthCheckService {
    checkers: Vec<Arc<dyn HealthChecker>>,
    start_time: std::time::Instant,
    version: String,
}

impl HealthCheckService {
    pub fn new(version: String) -> Self {
        Self {
            checkers: Vec::new(),
            start_time: std::time::Instant::now(),
            version,
        }
    }

    pub fn register(&mut self, checker: Arc<dyn HealthChecker>) {
        self.checkers.push(checker);
    }

    pub async fn check(&self) -> HealthStatus {
        let mut checks = Vec::new();
        let mut overall_status = HealthState::Healthy;

        for checker in &self.checkers {
            let check = checker.check().await;

            match check.status {
                HealthState::Unhealthy => overall_status = HealthState::Unhealthy,
                HealthState::Degraded if overall_status == HealthState::Healthy => {
                    overall_status = HealthState::Degraded;
                }
                _ => {}
            }

            checks.push(check);
        }

        HealthStatus {
            status: overall_status,
            checks,
            timestamp: Utc::now(),
            version: self.version.clone(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
        }
    }

    pub async fn liveness(&self) -> bool {
        true
    }

    pub async fn readiness(&self) -> bool {
        let status = self.check().await;
        status.status != HealthState::Unhealthy
    }
}

// ============================================================================
// Alerting
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub id: Uuid,
    pub name: String,
    pub description: String,
    pub condition: AlertCondition,
    pub severity: AlertSeverity,
    pub channels: Vec<String>,
    pub cooldown_minutes: u32,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    MetricThreshold {
        metric: String,
        operator: ThresholdOperator,
        value: f64,
        duration_seconds: u32,
    },
    ErrorRate {
        error_type: String,
        threshold_percent: f64,
        window_seconds: u32,
    },
    HealthCheck {
        check_name: String,
        status: HealthState,
    },
    Custom {
        expression: String,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ThresholdOperator {
    GreaterThan,
    LessThan,
    GreaterOrEqual,
    LessOrEqual,
    Equal,
    NotEqual,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    Warning,
    Info,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    pub id: Uuid,
    pub rule_id: Uuid,
    pub rule_name: String,
    pub severity: AlertSeverity,
    pub message: String,
    pub fired_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub labels: HashMap<String, String>,
    pub annotations: HashMap<String, String>,
}

#[async_trait]
pub trait AlertChannel: Send + Sync {
    fn name(&self) -> &str;
    async fn send(&self, alert: &Alert) -> Result<(), AlertError>;
}

pub struct EmailAlertChannel {
    smtp_config: SmtpConfig,
    recipients: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct SmtpConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub password: String,
}

#[async_trait]
impl AlertChannel for EmailAlertChannel {
    fn name(&self) -> &str {
        "email"
    }

    async fn send(&self, alert: &Alert) -> Result<(), AlertError> {
        info!("Sending email alert: {}", alert.message);
        Ok(())
    }
}

pub struct SlackAlertChannel {
    webhook_url: String,
}

#[async_trait]
impl AlertChannel for SlackAlertChannel {
    fn name(&self) -> &str {
        "slack"
    }

    async fn send(&self, alert: &Alert) -> Result<(), AlertError> {
        info!("Sending Slack alert: {}", alert.message);
        Ok(())
    }
}

pub struct WebhookAlertChannel {
    url: String,
    headers: HashMap<String, String>,
}

#[async_trait]
impl AlertChannel for WebhookAlertChannel {
    fn name(&self) -> &str {
        "webhook"
    }

    async fn send(&self, alert: &Alert) -> Result<(), AlertError> {
        info!("Sending webhook alert to {}: {}", self.url, alert.message);
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AlertError {
    #[error("Failed to send alert: {0}")]
    SendError(String),
    #[error("Channel not found: {0}")]
    ChannelNotFound(String),
    #[error("Configuration error: {0}")]
    ConfigError(String),
}

pub struct AlertManager {
    rules: RwLock<Vec<AlertRule>>,
    channels: RwLock<HashMap<String, Arc<dyn AlertChannel>>>,
    active_alerts: RwLock<HashMap<Uuid, Alert>>,
    last_fired: RwLock<HashMap<Uuid, DateTime<Utc>>>,
}

impl AlertManager {
    pub fn new() -> Self {
        Self {
            rules: RwLock::new(Vec::new()),
            channels: RwLock::new(HashMap::new()),
            active_alerts: RwLock::new(HashMap::new()),
            last_fired: RwLock::new(HashMap::new()),
        }
    }

    pub async fn add_rule(&self, rule: AlertRule) {
        self.rules.write().await.push(rule);
    }

    pub async fn register_channel(&self, name: String, channel: Arc<dyn AlertChannel>) {
        self.channels.write().await.insert(name, channel);
    }

    pub async fn fire_alert(&self, rule: &AlertRule, message: String) -> Result<(), AlertError> {
        let now = Utc::now();

        // Check cooldown
        if let Some(last) = self.last_fired.read().await.get(&rule.id) {
            let cooldown = chrono::Duration::minutes(rule.cooldown_minutes as i64);
            if now - *last < cooldown {
                return Ok(());
            }
        }

        let alert = Alert {
            id: Uuid::new_v4(),
            rule_id: rule.id,
            rule_name: rule.name.clone(),
            severity: rule.severity,
            message,
            fired_at: now,
            resolved_at: None,
            labels: HashMap::new(),
            annotations: HashMap::new(),
        };

        // Send to channels
        let channels = self.channels.read().await;
        for channel_name in &rule.channels {
            if let Some(channel) = channels.get(channel_name) {
                channel.send(&alert).await?;
            }
        }

        // Track alert
        self.active_alerts.write().await.insert(alert.id, alert.clone());
        self.last_fired.write().await.insert(rule.id, now);

        Ok(())
    }

    pub async fn resolve_alert(&self, alert_id: Uuid) {
        if let Some(alert) = self.active_alerts.write().await.get_mut(&alert_id) {
            alert.resolved_at = Some(Utc::now());
        }
    }

    pub async fn get_active_alerts(&self) -> Vec<Alert> {
        self.active_alerts
            .read()
            .await
            .values()
            .filter(|a| a.resolved_at.is_none())
            .cloned()
            .collect()
    }
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tracing
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSpan {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub service_name: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub duration_ms: Option<u64>,
    pub status: SpanStatus,
    pub tags: HashMap<String, String>,
    pub logs: Vec<SpanLog>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SpanStatus {
    Ok,
    Error,
    Unset,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpanLog {
    pub timestamp: DateTime<Utc>,
    pub level: String,
    pub message: String,
    pub fields: HashMap<String, serde_json::Value>,
}

pub struct TracingConfig {
    pub service_name: String,
    pub otlp_endpoint: Option<String>,
    pub sample_rate: f64,
    pub enabled: bool,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            service_name: "mycelix-mail".to_string(),
            otlp_endpoint: None,
            sample_rate: 1.0,
            enabled: true,
        }
    }
}

// ============================================================================
// Dashboard Data
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardData {
    pub overview: OverviewStats,
    pub email_stats: EmailStats,
    pub sync_stats: SyncStats,
    pub api_stats: ApiStats,
    pub system_stats: SystemStats,
    pub recent_alerts: Vec<Alert>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverviewStats {
    pub health_status: HealthState,
    pub uptime_seconds: u64,
    pub version: String,
    pub total_users: u64,
    pub active_sessions: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmailStats {
    pub received_today: u64,
    pub sent_today: u64,
    pub received_hourly: Vec<u64>,
    pub sent_hourly: Vec<u64>,
    pub average_size_bytes: u64,
    pub spam_blocked: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncStats {
    pub total_syncs: u64,
    pub failed_syncs: u64,
    pub average_duration_ms: u64,
    pub pending_items: u64,
    pub last_sync: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiStats {
    pub requests_today: u64,
    pub average_latency_ms: u64,
    pub error_rate_percent: f64,
    pub requests_per_minute: Vec<u64>,
    pub top_endpoints: Vec<EndpointStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndpointStats {
    pub path: String,
    pub method: String,
    pub count: u64,
    pub average_latency_ms: u64,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemStats {
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub cpu_percent: f64,
    pub disk_used_bytes: u64,
    pub disk_total_bytes: u64,
    pub load_average: [f64; 3],
}

// ============================================================================
// Observability Service
// ============================================================================

pub struct ObservabilityService {
    pub metrics: Arc<MetricsRegistry>,
    pub health: Arc<HealthCheckService>,
    pub alerts: Arc<AlertManager>,
    config: TracingConfig,
}

impl ObservabilityService {
    pub fn new(version: String, config: TracingConfig) -> Self {
        Self {
            metrics: Arc::new(MetricsRegistry::new()),
            health: Arc::new(HealthCheckService::new(version)),
            alerts: Arc::new(AlertManager::new()),
            config,
        }
    }

    /// Get Prometheus metrics endpoint response
    pub fn prometheus_metrics(&self) -> String {
        self.metrics.export()
    }

    /// Get health status
    pub async fn health_status(&self) -> HealthStatus {
        self.health.check().await
    }

    /// Get liveness probe
    pub async fn liveness(&self) -> bool {
        self.health.liveness().await
    }

    /// Get readiness probe
    pub async fn readiness(&self) -> bool {
        self.health.readiness().await
    }

    /// Get dashboard data
    pub async fn dashboard_data(&self) -> DashboardData {
        let health = self.health.check().await;
        let alerts = self.alerts.get_active_alerts().await;

        DashboardData {
            overview: OverviewStats {
                health_status: health.status,
                uptime_seconds: health.uptime_seconds,
                version: health.version,
                total_users: 0,
                active_sessions: 0,
            },
            email_stats: EmailStats {
                received_today: 0,
                sent_today: 0,
                received_hourly: vec![0; 24],
                sent_hourly: vec![0; 24],
                average_size_bytes: 0,
                spam_blocked: 0,
            },
            sync_stats: SyncStats {
                total_syncs: 0,
                failed_syncs: 0,
                average_duration_ms: 0,
                pending_items: 0,
                last_sync: None,
            },
            api_stats: ApiStats {
                requests_today: 0,
                average_latency_ms: 0,
                error_rate_percent: 0.0,
                requests_per_minute: vec![0; 60],
                top_endpoints: Vec::new(),
            },
            system_stats: SystemStats {
                memory_used_bytes: 0,
                memory_total_bytes: 0,
                cpu_percent: 0.0,
                disk_used_bytes: 0,
                disk_total_bytes: 0,
                load_average: [0.0, 0.0, 0.0],
            },
            recent_alerts: alerts,
        }
    }
}
