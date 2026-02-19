//! Visual Dashboard Components
//!
//! Provides real-time monitoring dashboard for trust network health,
//! agent activities, and system alerts.
//!
//! Features:
//! - Live metrics aggregation and streaming
//! - Time-series data for visualization
//! - Alert panel with severity-based actions
//! - Widget system for customizable layouts
//! - Event stream for real-time updates

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Maximum history length for time series
    pub max_history: usize,
    /// Metrics aggregation window (seconds)
    pub aggregation_window_secs: u64,
    /// Alert retention period (seconds)
    pub alert_retention_secs: u64,
    /// Event stream buffer size
    pub event_buffer_size: usize,
    /// Update interval (milliseconds)
    pub update_interval_ms: u64,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            max_history: 1000,
            aggregation_window_secs: 60,
            alert_retention_secs: 86400, // 24 hours
            event_buffer_size: 100,
            update_interval_ms: 1000,
        }
    }
}

/// Live metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveMetrics {
    /// Timestamp
    pub timestamp: u64,
    /// Total active agents
    pub active_agents: usize,
    /// Total trust score (aggregate)
    pub total_trust: f64,
    /// Average trust score
    pub average_trust: f64,
    /// Network health (0.0-1.0)
    pub network_health: f64,
    /// Transactions per second
    pub tps: f64,
    /// Active alerts count by severity
    pub alerts: AlertCounts,
    /// Phi coherence (collective) - INFORMATIONAL ONLY
    /// Do not gate behavior based on this value. Use agent-specific k_coherence for gating.
    pub collective_phi: f64,
    /// Byzantine threat level (0.0-1.0)
    pub byzantine_threat: f64,
}

impl Default for LiveMetrics {
    fn default() -> Self {
        Self {
            timestamp: 0,
            active_agents: 0,
            total_trust: 0.0,
            average_trust: 0.0,
            network_health: 1.0,
            tps: 0.0,
            alerts: AlertCounts::default(),
            collective_phi: 0.0,
            byzantine_threat: 0.0,
        }
    }
}

/// Alert counts by severity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlertCounts {
    /// Critical alerts
    pub critical: usize,
    /// High severity alerts
    pub high: usize,
    /// Medium severity alerts
    pub medium: usize,
    /// Low severity alerts
    pub low: usize,
}

impl AlertCounts {
    /// Total alert count
    pub fn total(&self) -> usize {
        self.critical + self.high + self.medium + self.low
    }

    /// Whether there are any critical alerts
    pub fn has_critical(&self) -> bool {
        self.critical > 0
    }
}

/// Input for metrics aggregation
#[derive(Debug, Clone)]
pub struct MetricsInput {
    /// Agent trust scores
    pub trust_scores: Vec<f64>,
    /// Transaction count in window
    pub transaction_count: usize,
    /// Alert events
    pub alerts: Vec<DashboardAlert>,
    /// Phi measurements
    pub phi_values: Vec<f64>,
    /// Detected threats
    pub threats: Vec<f64>,
}

/// Metrics aggregator
pub struct MetricsAggregator {
    /// Configuration
    config: DashboardConfig,
    /// Historical metrics
    history: VecDeque<LiveMetrics>,
    /// Current window start
    window_start: u64,
    /// Transactions in current window
    window_transactions: usize,
}

impl MetricsAggregator {
    /// Create new aggregator
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
            window_start: 0,
            window_transactions: 0,
        }
    }

    /// Update metrics with new input
    pub fn update(&mut self, input: MetricsInput, timestamp: u64) -> LiveMetrics {
        // Calculate TPS
        if timestamp >= self.window_start + self.config.aggregation_window_secs {
            self.window_transactions = 0;
            self.window_start = timestamp;
        }
        self.window_transactions += input.transaction_count;

        let window_elapsed = (timestamp - self.window_start).max(1) as f64;
        let tps = self.window_transactions as f64 / window_elapsed;

        // Calculate averages
        let active_agents = input.trust_scores.len();
        let total_trust: f64 = input.trust_scores.iter().sum();
        let average_trust = if active_agents > 0 {
            total_trust / active_agents as f64
        } else {
            0.0
        };

        // Alert counts
        let mut alerts = AlertCounts::default();
        for alert in &input.alerts {
            match alert.severity {
                AlertSeverity::Critical => alerts.critical += 1,
                AlertSeverity::High => alerts.high += 1,
                AlertSeverity::Medium => alerts.medium += 1,
                AlertSeverity::Low => alerts.low += 1,
            }
        }

        // Collective Phi
        let collective_phi = if !input.phi_values.is_empty() {
            input.phi_values.iter().sum::<f64>() / input.phi_values.len() as f64
        } else {
            0.0
        };

        // Byzantine threat level
        let byzantine_threat = if !input.threats.is_empty() {
            input.threats.iter().copied().fold(0.0_f64, f64::max)
        } else {
            0.0
        };

        // Network health (composite)
        let health_factors = [
            average_trust,
            1.0 - byzantine_threat,
            if alerts.critical > 0 { 0.5 } else { 1.0 },
            collective_phi.min(1.0),
        ];
        let network_health = health_factors.iter().sum::<f64>() / health_factors.len() as f64;

        let metrics = LiveMetrics {
            timestamp,
            active_agents,
            total_trust,
            average_trust,
            network_health,
            tps,
            alerts,
            collective_phi,
            byzantine_threat,
        };

        // Add to history
        self.history.push_back(metrics.clone());
        while self.history.len() > self.config.max_history {
            self.history.pop_front();
        }

        metrics
    }

    /// Get metrics history
    pub fn history(&self) -> &VecDeque<LiveMetrics> {
        &self.history
    }

    /// Get latest metrics
    pub fn latest(&self) -> Option<&LiveMetrics> {
        self.history.back()
    }
}

/// Dashboard event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardEvent {
    /// Event ID
    pub event_id: String,
    /// Timestamp
    pub timestamp: u64,
    /// Event type
    pub event_type: DashboardEventType,
    /// Priority
    pub priority: EventPriority,
    /// Source (agent ID, system, etc.)
    pub source: String,
    /// Description
    pub description: String,
    /// Additional data
    pub data: HashMap<String, String>,
}

/// Dashboard event types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DashboardEventType {
    /// Agent created
    AgentCreated,
    /// Agent suspended
    AgentSuspended,
    /// Trust update
    TrustUpdate,
    /// Attack detected
    AttackDetected,
    /// Consensus reached
    ConsensusReached,
    /// Threshold crossed
    ThresholdCrossed,
    /// System health change
    HealthChange,
    /// Budget exhausted
    BudgetExhausted,
    /// Custom event
    Custom,
}

/// Event priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventPriority {
    /// Low priority (informational)
    Low = 0,
    /// Medium priority
    Medium = 1,
    /// High priority (requires attention)
    High = 2,
    /// Critical priority (requires immediate action)
    Critical = 3,
}

/// Event stream for real-time updates
pub struct EventStream {
    /// Configuration
    config: DashboardConfig,
    /// Event buffer
    buffer: VecDeque<DashboardEvent>,
    /// Event counter
    event_counter: u64,
}

impl EventStream {
    /// Create new event stream
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            buffer: VecDeque::new(),
            event_counter: 0,
        }
    }

    /// Push new event
    pub fn push(&mut self, event: DashboardEvent) {
        self.buffer.push_back(event);
        while self.buffer.len() > self.config.event_buffer_size {
            self.buffer.pop_front();
        }
    }

    /// Create and push event
    pub fn emit(
        &mut self,
        event_type: DashboardEventType,
        priority: EventPriority,
        source: &str,
        description: &str,
    ) -> String {
        self.event_counter += 1;
        let event_id = format!("evt_{}", self.event_counter);

        let event = DashboardEvent {
            event_id: event_id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            event_type,
            priority,
            source: source.to_string(),
            description: description.to_string(),
            data: HashMap::new(),
        };

        self.push(event);
        event_id
    }

    /// Get recent events
    pub fn recent(&self, count: usize) -> Vec<&DashboardEvent> {
        self.buffer.iter().rev().take(count).collect()
    }

    /// Get events by priority
    pub fn by_priority(&self, min_priority: EventPriority) -> Vec<&DashboardEvent> {
        self.buffer
            .iter()
            .filter(|e| e.priority >= min_priority)
            .collect()
    }

    /// Get events by type
    pub fn by_type(&self, event_type: DashboardEventType) -> Vec<&DashboardEvent> {
        self.buffer
            .iter()
            .filter(|e| e.event_type == event_type)
            .collect()
    }

    /// Clear buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    /// Timestamp
    pub timestamp: u64,
    /// Value
    pub value: f64,
    /// Optional label
    pub label: Option<String>,
}

/// Time series for charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    /// Series name
    pub name: String,
    /// Data points
    pub points: Vec<DataPoint>,
    /// Chart type
    pub chart_type: ChartType,
    /// Color (hex)
    pub color: String,
}

impl TimeSeries {
    /// Create new time series
    pub fn new(name: &str, chart_type: ChartType) -> Self {
        Self {
            name: name.to_string(),
            points: vec![],
            chart_type,
            color: "#3b82f6".to_string(), // Default blue
        }
    }

    /// Add data point
    pub fn add_point(&mut self, timestamp: u64, value: f64) {
        self.points.push(DataPoint {
            timestamp,
            value,
            label: None,
        });
    }

    /// Set color
    pub fn with_color(mut self, color: &str) -> Self {
        self.color = color.to_string();
        self
    }

    /// Get latest value
    pub fn latest(&self) -> Option<f64> {
        self.points.last().map(|p| p.value)
    }

    /// Get min value
    pub fn min(&self) -> Option<f64> {
        self.points.iter().map(|p| p.value).reduce(f64::min)
    }

    /// Get max value
    pub fn max(&self) -> Option<f64> {
        self.points.iter().map(|p| p.value).reduce(f64::max)
    }
}

/// Chart type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChartType {
    /// Line chart
    Line,
    /// Area chart
    Area,
    /// Bar chart
    Bar,
    /// Scatter plot
    Scatter,
}

/// Chart data builder
pub struct ChartDataBuilder {
    /// Series collection
    series: Vec<TimeSeries>,
}

impl Default for ChartDataBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ChartDataBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self { series: vec![] }
    }

    /// Add series
    pub fn add_series(&mut self, series: TimeSeries) -> &mut Self {
        self.series.push(series);
        self
    }

    /// Build trust over time chart from metrics history
    pub fn trust_over_time(metrics: &VecDeque<LiveMetrics>) -> TimeSeries {
        let mut series = TimeSeries::new("Average Trust", ChartType::Line).with_color("#10b981"); // Green

        for m in metrics {
            series.add_point(m.timestamp, m.average_trust);
        }

        series
    }

    /// Build health over time chart
    pub fn health_over_time(metrics: &VecDeque<LiveMetrics>) -> TimeSeries {
        let mut series = TimeSeries::new("Network Health", ChartType::Area).with_color("#3b82f6"); // Blue

        for m in metrics {
            series.add_point(m.timestamp, m.network_health);
        }

        series
    }

    /// Build TPS chart
    pub fn tps_over_time(metrics: &VecDeque<LiveMetrics>) -> TimeSeries {
        let mut series = TimeSeries::new("Transactions/sec", ChartType::Line).with_color("#f59e0b"); // Amber

        for m in metrics {
            series.add_point(m.timestamp, m.tps);
        }

        series
    }

    /// Build alert trend chart
    pub fn alert_trend(metrics: &VecDeque<LiveMetrics>) -> TimeSeries {
        let mut series = TimeSeries::new("Active Alerts", ChartType::Bar).with_color("#ef4444"); // Red

        for m in metrics {
            series.add_point(m.timestamp, m.alerts.total() as f64);
        }

        series
    }

    /// Get all series
    pub fn build(self) -> Vec<TimeSeries> {
        self.series
    }
}

/// Dashboard alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardAlert {
    /// Alert ID
    pub alert_id: String,
    /// Timestamp
    pub timestamp: u64,
    /// Severity
    pub severity: AlertSeverity,
    /// Status
    pub status: AlertStatus,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Source (agent ID, system component)
    pub source: String,
    /// Recommended actions
    pub actions: Vec<AlertAction>,
    /// Acknowledgment info
    pub acknowledged_by: Option<String>,
    /// Resolution info
    pub resolved_at: Option<u64>,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Low (informational)
    Low = 0,
    /// Medium
    Medium = 1,
    /// High
    High = 2,
    /// Critical
    Critical = 3,
}

/// Alert status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertStatus {
    /// New alert
    New,
    /// Acknowledged
    Acknowledged,
    /// In progress
    InProgress,
    /// Resolved
    Resolved,
    /// Dismissed
    Dismissed,
}

/// Recommended action for alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertAction {
    /// Action type
    pub action_type: AlertActionType,
    /// Label
    pub label: String,
    /// Description
    pub description: String,
    /// Whether action requires confirmation
    pub requires_confirmation: bool,
}

/// Alert action types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertActionType {
    /// Acknowledge alert
    Acknowledge,
    /// Investigate further
    Investigate,
    /// Quarantine agent
    Quarantine,
    /// Adjust threshold
    AdjustThreshold,
    /// Escalate to human
    Escalate,
    /// Dismiss alert
    Dismiss,
    /// Custom action
    Custom,
}

/// Alert panel manager
pub struct AlertPanel {
    /// Configuration
    config: DashboardConfig,
    /// Active alerts
    alerts: HashMap<String, DashboardAlert>,
    /// Alert counter
    alert_counter: u64,
}

impl AlertPanel {
    /// Create new alert panel
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            alerts: HashMap::new(),
            alert_counter: 0,
        }
    }

    /// Create new alert
    pub fn create_alert(
        &mut self,
        severity: AlertSeverity,
        title: &str,
        description: &str,
        source: &str,
    ) -> String {
        self.alert_counter += 1;
        let alert_id = format!("alert_{}", self.alert_counter);

        let alert = DashboardAlert {
            alert_id: alert_id.clone(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            severity,
            status: AlertStatus::New,
            title: title.to_string(),
            description: description.to_string(),
            source: source.to_string(),
            actions: Self::default_actions(severity),
            acknowledged_by: None,
            resolved_at: None,
        };

        self.alerts.insert(alert_id.clone(), alert);
        alert_id
    }

    /// Get default actions for severity
    fn default_actions(severity: AlertSeverity) -> Vec<AlertAction> {
        let mut actions = vec![AlertAction {
            action_type: AlertActionType::Acknowledge,
            label: "Acknowledge".to_string(),
            description: "Mark alert as seen".to_string(),
            requires_confirmation: false,
        }];

        match severity {
            AlertSeverity::Critical => {
                actions.push(AlertAction {
                    action_type: AlertActionType::Quarantine,
                    label: "Quarantine".to_string(),
                    description: "Isolate affected agent".to_string(),
                    requires_confirmation: true,
                });
                actions.push(AlertAction {
                    action_type: AlertActionType::Escalate,
                    label: "Escalate".to_string(),
                    description: "Escalate to human operator".to_string(),
                    requires_confirmation: false,
                });
            }
            AlertSeverity::High => {
                actions.push(AlertAction {
                    action_type: AlertActionType::Investigate,
                    label: "Investigate".to_string(),
                    description: "Open investigation panel".to_string(),
                    requires_confirmation: false,
                });
            }
            AlertSeverity::Medium | AlertSeverity::Low => {
                actions.push(AlertAction {
                    action_type: AlertActionType::Dismiss,
                    label: "Dismiss".to_string(),
                    description: "Dismiss this alert".to_string(),
                    requires_confirmation: false,
                });
            }
        }

        actions
    }

    /// Acknowledge alert
    pub fn acknowledge(&mut self, alert_id: &str, user: &str) -> bool {
        if let Some(alert) = self.alerts.get_mut(alert_id) {
            alert.status = AlertStatus::Acknowledged;
            alert.acknowledged_by = Some(user.to_string());
            true
        } else {
            false
        }
    }

    /// Resolve alert
    pub fn resolve(&mut self, alert_id: &str) -> bool {
        if let Some(alert) = self.alerts.get_mut(alert_id) {
            alert.status = AlertStatus::Resolved;
            alert.resolved_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs(),
            );
            true
        } else {
            false
        }
    }

    /// Dismiss alert
    pub fn dismiss(&mut self, alert_id: &str) -> bool {
        if let Some(alert) = self.alerts.get_mut(alert_id) {
            alert.status = AlertStatus::Dismissed;
            true
        } else {
            false
        }
    }

    /// Get active alerts (not resolved/dismissed)
    pub fn active_alerts(&self) -> Vec<&DashboardAlert> {
        self.alerts
            .values()
            .filter(|a| !matches!(a.status, AlertStatus::Resolved | AlertStatus::Dismissed))
            .collect()
    }

    /// Get alerts by severity
    pub fn by_severity(&self, severity: AlertSeverity) -> Vec<&DashboardAlert> {
        self.alerts
            .values()
            .filter(|a| a.severity == severity)
            .collect()
    }

    /// Get critical alerts
    pub fn critical_alerts(&self) -> Vec<&DashboardAlert> {
        self.by_severity(AlertSeverity::Critical)
            .into_iter()
            .filter(|a| !matches!(a.status, AlertStatus::Resolved | AlertStatus::Dismissed))
            .collect()
    }

    /// Clean up old resolved alerts
    pub fn cleanup(&mut self, current_time: u64) {
        let retention = self.config.alert_retention_secs;
        self.alerts.retain(|_, alert| match alert.resolved_at {
            Some(resolved) => current_time - resolved < retention,
            None => true,
        });
    }

    /// Get alert count by status
    pub fn count_by_status(&self) -> HashMap<AlertStatus, usize> {
        let mut counts = HashMap::new();
        for alert in self.alerts.values() {
            *counts.entry(alert.status).or_insert(0) += 1;
        }
        counts
    }
}

/// Dashboard widget
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Widget {
    /// Widget ID
    pub widget_id: String,
    /// Widget type
    pub widget_type: WidgetType,
    /// Position
    pub position: WidgetPosition,
    /// Size
    pub size: WidgetSize,
    /// Title
    pub title: String,
    /// Configuration
    pub config: HashMap<String, String>,
}

/// Widget types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WidgetType {
    /// Metrics summary
    MetricsSummary,
    /// Trust chart
    TrustChart,
    /// Health gauge
    HealthGauge,
    /// Alert list
    AlertList,
    /// Event stream
    EventStream,
    /// Agent table
    AgentTable,
    /// TPS graph
    TpsGraph,
    /// Phi meter
    PhiMeter,
}

/// Widget position
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WidgetPosition {
    /// X coordinate (grid units)
    pub x: u32,
    /// Y coordinate (grid units)
    pub y: u32,
}

/// Widget size
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct WidgetSize {
    /// Width (grid units)
    pub width: u32,
    /// Height (grid units)
    pub height: u32,
}

/// Get default dashboard layout
pub fn default_layout() -> Vec<Widget> {
    vec![
        Widget {
            widget_id: "metrics_summary".to_string(),
            widget_type: WidgetType::MetricsSummary,
            position: WidgetPosition { x: 0, y: 0 },
            size: WidgetSize {
                width: 12,
                height: 2,
            },
            title: "Network Overview".to_string(),
            config: HashMap::new(),
        },
        Widget {
            widget_id: "trust_chart".to_string(),
            widget_type: WidgetType::TrustChart,
            position: WidgetPosition { x: 0, y: 2 },
            size: WidgetSize {
                width: 6,
                height: 4,
            },
            title: "Trust Trends".to_string(),
            config: HashMap::new(),
        },
        Widget {
            widget_id: "health_gauge".to_string(),
            widget_type: WidgetType::HealthGauge,
            position: WidgetPosition { x: 6, y: 2 },
            size: WidgetSize {
                width: 3,
                height: 4,
            },
            title: "Network Health".to_string(),
            config: HashMap::new(),
        },
        Widget {
            widget_id: "phi_meter".to_string(),
            widget_type: WidgetType::PhiMeter,
            position: WidgetPosition { x: 9, y: 2 },
            size: WidgetSize {
                width: 3,
                height: 4,
            },
            title: "Collective Phi".to_string(),
            config: HashMap::new(),
        },
        Widget {
            widget_id: "alert_list".to_string(),
            widget_type: WidgetType::AlertList,
            position: WidgetPosition { x: 0, y: 6 },
            size: WidgetSize {
                width: 6,
                height: 4,
            },
            title: "Active Alerts".to_string(),
            config: HashMap::new(),
        },
        Widget {
            widget_id: "event_stream".to_string(),
            widget_type: WidgetType::EventStream,
            position: WidgetPosition { x: 6, y: 6 },
            size: WidgetSize {
                width: 6,
                height: 4,
            },
            title: "Event Stream".to_string(),
            config: HashMap::new(),
        },
    ]
}

/// Main dashboard state
pub struct Dashboard {
    /// Configuration
    pub config: DashboardConfig,
    /// Metrics aggregator
    pub metrics: MetricsAggregator,
    /// Event stream
    pub events: EventStream,
    /// Alert panel
    pub alerts: AlertPanel,
    /// Widget layout
    pub widgets: Vec<Widget>,
}

impl Dashboard {
    /// Create new dashboard
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            metrics: MetricsAggregator::new(config.clone()),
            events: EventStream::new(config.clone()),
            alerts: AlertPanel::new(config.clone()),
            widgets: default_layout(),
            config,
        }
    }

    /// Update dashboard with new data
    pub fn update(&mut self, input: MetricsInput, timestamp: u64) -> LiveMetrics {
        let metrics = self.metrics.update(input.clone(), timestamp);

        // Emit events for significant changes
        if metrics.byzantine_threat > 0.5 {
            self.events.emit(
                DashboardEventType::AttackDetected,
                EventPriority::High,
                "system",
                &format!(
                    "Byzantine threat level elevated: {:.2}",
                    metrics.byzantine_threat
                ),
            );
        }

        if metrics.network_health < 0.7 {
            self.events.emit(
                DashboardEventType::HealthChange,
                EventPriority::Medium,
                "system",
                &format!("Network health degraded: {:.2}", metrics.network_health),
            );
        }

        metrics
    }

    /// Get current dashboard state
    pub fn state(&self) -> DashboardState {
        DashboardState {
            metrics: self.metrics.latest().cloned().unwrap_or_default(),
            active_alerts: self.alerts.active_alerts().len(),
            critical_alerts: self.alerts.critical_alerts().len(),
            recent_events: self.events.recent(10).len(),
        }
    }
}

/// Dashboard state summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardState {
    /// Current metrics
    pub metrics: LiveMetrics,
    /// Active alert count
    pub active_alerts: usize,
    /// Critical alert count
    pub critical_alerts: usize,
    /// Recent event count
    pub recent_events: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_aggregator() {
        let config = DashboardConfig::default();
        let mut agg = MetricsAggregator::new(config);

        let input = MetricsInput {
            trust_scores: vec![0.5, 0.6, 0.7, 0.8],
            transaction_count: 10,
            alerts: vec![],
            phi_values: vec![0.65],
            threats: vec![0.1],
        };

        let metrics = agg.update(input, 1000);

        assert_eq!(metrics.active_agents, 4);
        assert!((metrics.average_trust - 0.65).abs() < 0.01);
        assert!(metrics.network_health > 0.5);
    }

    #[test]
    fn test_event_stream() {
        let config = DashboardConfig::default();
        let mut stream = EventStream::new(config);

        stream.emit(
            DashboardEventType::AgentCreated,
            EventPriority::Low,
            "test",
            "Test agent created",
        );

        stream.emit(
            DashboardEventType::AttackDetected,
            EventPriority::Critical,
            "detector",
            "Attack detected",
        );

        assert_eq!(stream.recent(10).len(), 2);

        let critical = stream.by_priority(EventPriority::Critical);
        assert_eq!(critical.len(), 1);
    }

    #[test]
    fn test_alert_panel() {
        let config = DashboardConfig::default();
        let mut panel = AlertPanel::new(config);

        let alert_id = panel.create_alert(
            AlertSeverity::High,
            "Test Alert",
            "Test description",
            "test",
        );

        assert_eq!(panel.active_alerts().len(), 1);

        panel.acknowledge(&alert_id, "admin");
        let alert = panel.alerts.get(&alert_id).unwrap();
        assert_eq!(alert.status, AlertStatus::Acknowledged);

        panel.resolve(&alert_id);
        assert_eq!(panel.active_alerts().len(), 0);
    }

    #[test]
    fn test_time_series() {
        let mut series = TimeSeries::new("Test", ChartType::Line);

        series.add_point(1000, 0.5);
        series.add_point(2000, 0.7);
        series.add_point(3000, 0.6);

        assert_eq!(series.latest(), Some(0.6));
        assert_eq!(series.min(), Some(0.5));
        assert_eq!(series.max(), Some(0.7));
    }

    #[test]
    fn test_chart_data_builder() {
        let config = DashboardConfig::default();
        let mut agg = MetricsAggregator::new(config);

        // Add some data
        for i in 0..5 {
            let input = MetricsInput {
                trust_scores: vec![0.5 + i as f64 * 0.05],
                transaction_count: 10,
                alerts: vec![],
                phi_values: vec![0.6],
                threats: vec![0.1],
            };
            agg.update(input, 1000 + i * 60);
        }

        let trust_series = ChartDataBuilder::trust_over_time(agg.history());
        assert_eq!(trust_series.points.len(), 5);
    }

    #[test]
    fn test_default_layout() {
        let layout = default_layout();
        assert!(!layout.is_empty());

        // Check for key widgets
        let widget_types: Vec<_> = layout.iter().map(|w| w.widget_type).collect();
        assert!(widget_types.contains(&WidgetType::MetricsSummary));
        assert!(widget_types.contains(&WidgetType::AlertList));
    }

    #[test]
    fn test_dashboard() {
        let config = DashboardConfig::default();
        let mut dashboard = Dashboard::new(config);

        let input = MetricsInput {
            trust_scores: vec![0.5, 0.6, 0.7],
            transaction_count: 5,
            alerts: vec![],
            phi_values: vec![0.7],
            threats: vec![0.2],
        };

        let metrics = dashboard.update(input, 1000);
        assert!(metrics.network_health > 0.0);

        let state = dashboard.state();
        assert_eq!(state.active_alerts, 0);
    }
}
