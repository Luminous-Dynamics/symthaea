//! # Visual Dashboard Components
//!
//! Real-time trust visualization data structures and event streams.
//!
//! ## Features
//!
//! - **Live Metrics**: Real-time trust aggregations
//! - **Event Streams**: WebSocket-ready event format
//! - **Chart Data**: Pre-aggregated data for visualizations
//! - **Alerts Panel**: Prioritized alert display

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

// ============================================================================
// Configuration
// ============================================================================

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardConfig {
    /// Update interval (ms)
    pub update_interval_ms: u64,
    /// Maximum events to retain
    pub max_events: usize,
    /// Maximum data points for charts
    pub max_chart_points: usize,
    /// Alert retention (ms)
    pub alert_retention_ms: u64,
    /// Enable real-time mode
    pub realtime_enabled: bool,
}

impl Default for DashboardConfig {
    fn default() -> Self {
        Self {
            update_interval_ms: 1000,
            max_events: 1000,
            max_chart_points: 100,
            alert_retention_ms: 3600_000,
            realtime_enabled: true,
        }
    }
}

// ============================================================================
// Live Metrics
// ============================================================================

/// Real-time metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiveMetrics {
    /// Timestamp
    pub timestamp: u64,
    /// Total agents
    pub total_agents: u64,
    /// Active agents (non-suspended)
    pub active_agents: u64,
    /// Average trust score
    pub average_trust: f64,
    /// Median trust score
    pub median_trust: f64,
    /// Trust standard deviation
    pub trust_stddev: f64,
    /// Total KREDIT in circulation
    pub total_kredit: u64,
    /// Active proposals
    pub active_proposals: u32,
    /// Consensus success rate (last hour)
    pub consensus_success_rate: f64,
    /// Network health score (0-100)
    pub network_health: u32,
    /// Alerts count by severity
    pub alerts_by_severity: AlertCounts,
}

/// Alert counts by severity
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AlertCounts {
    pub critical: u32,
    pub high: u32,
    pub medium: u32,
    pub low: u32,
}

impl AlertCounts {
    pub fn total(&self) -> u32 {
        self.critical + self.high + self.medium + self.low
    }
}

/// Metrics aggregator
#[derive(Debug)]
pub struct MetricsAggregator {
    config: DashboardConfig,
    history: VecDeque<LiveMetrics>,
    current: LiveMetrics,
}

impl MetricsAggregator {
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            history: VecDeque::new(),
            current: LiveMetrics {
                timestamp: 0,
                total_agents: 0,
                active_agents: 0,
                average_trust: 0.0,
                median_trust: 0.0,
                trust_stddev: 0.0,
                total_kredit: 0,
                active_proposals: 0,
                consensus_success_rate: 0.0,
                network_health: 100,
                alerts_by_severity: AlertCounts::default(),
            },
        }
    }

    /// Update metrics from raw data
    pub fn update(&mut self, data: MetricsInput) {
        self.current.timestamp = data.timestamp;
        self.current.total_agents = data.agents.len() as u64;
        self.current.active_agents = data.agents.iter()
            .filter(|a| a.is_active)
            .count() as u64;

        // Calculate trust statistics
        let trust_scores: Vec<f64> = data.agents.iter()
            .map(|a| a.trust_score)
            .collect();

        if !trust_scores.is_empty() {
            self.current.average_trust = trust_scores.iter().sum::<f64>()
                / trust_scores.len() as f64;

            let mut sorted = trust_scores.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            self.current.median_trust = sorted[sorted.len() / 2];

            let variance: f64 = trust_scores.iter()
                .map(|t| (t - self.current.average_trust).powi(2))
                .sum::<f64>() / trust_scores.len() as f64;
            self.current.trust_stddev = variance.sqrt();
        }

        self.current.total_kredit = data.agents.iter()
            .map(|a| a.kredit_balance as u64)
            .sum();

        self.current.active_proposals = data.active_proposals;
        self.current.consensus_success_rate = data.consensus_success_rate;

        // Update alerts BEFORE calculating health (health depends on alerts)
        self.current.alerts_by_severity = data.alerts;

        // Calculate network health
        self.current.network_health = self.calculate_health();

        // Store in history
        self.history.push_back(self.current.clone());
        while self.history.len() > self.config.max_chart_points {
            self.history.pop_front();
        }
    }

    fn calculate_health(&self) -> u32 {
        let mut health = 100.0;

        // Penalize low average trust
        if self.current.average_trust < 0.5 {
            health -= (0.5 - self.current.average_trust) * 40.0;
        }

        // Penalize high variability
        if self.current.trust_stddev > 0.3 {
            health -= (self.current.trust_stddev - 0.3) * 20.0;
        }

        // Penalize critical alerts
        health -= self.current.alerts_by_severity.critical as f64 * 10.0;
        health -= self.current.alerts_by_severity.high as f64 * 5.0;

        // Penalize low consensus success
        if self.current.consensus_success_rate < 0.8 {
            health -= (0.8 - self.current.consensus_success_rate) * 30.0;
        }

        health.max(0.0).min(100.0) as u32
    }

    /// Get current metrics
    pub fn current(&self) -> &LiveMetrics {
        &self.current
    }

    /// Get metrics history
    pub fn history(&self) -> impl Iterator<Item = &LiveMetrics> {
        self.history.iter()
    }
}

/// Input data for metrics update
#[derive(Debug, Clone)]
pub struct MetricsInput {
    pub timestamp: u64,
    pub agents: Vec<AgentMetricInput>,
    pub active_proposals: u32,
    pub consensus_success_rate: f64,
    pub alerts: AlertCounts,
}

/// Agent data for metrics
#[derive(Debug, Clone)]
pub struct AgentMetricInput {
    pub id: String,
    pub trust_score: f64,
    pub kredit_balance: i64,
    pub is_active: bool,
}

// ============================================================================
// Event Stream
// ============================================================================

/// Dashboard event for real-time updates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardEvent {
    /// Event ID
    pub id: String,
    /// Event type
    pub event_type: DashboardEventType,
    /// Timestamp
    pub timestamp: u64,
    /// Payload (JSON-serializable)
    pub payload: serde_json::Value,
    /// Priority (for ordering)
    pub priority: EventPriority,
}

/// Dashboard event types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DashboardEventType {
    // Agent events
    AgentCreated,
    AgentUpdated,
    AgentSuspended,
    AgentRevoked,
    TrustChanged,
    KreditChanged,

    // Consensus events
    ProposalCreated,
    VoteCast,
    ConsensusReached,
    ConsensusFailed,

    // Alert events
    AlertCreated,
    AlertAcknowledged,
    AlertResolved,

    // System events
    MetricsUpdated,
    HealthChanged,
    NetworkAnomaly,

    // Federation events
    SwarmJoined,
    SwarmLeft,
    CrossSwarmTransfer,
}

/// Event priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

/// Event stream manager
#[derive(Debug)]
pub struct EventStream {
    config: DashboardConfig,
    events: VecDeque<DashboardEvent>,
    subscribers: Vec<String>, // Subscriber IDs (simplified)
    sequence: u64,
}

impl EventStream {
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            events: VecDeque::new(),
            subscribers: Vec::new(),
            sequence: 0,
        }
    }

    /// Emit an event
    pub fn emit(&mut self, event_type: DashboardEventType, payload: serde_json::Value, priority: EventPriority, timestamp: u64) {
        self.sequence += 1;

        let event = DashboardEvent {
            id: format!("evt-{}", self.sequence),
            event_type,
            timestamp,
            payload,
            priority,
        };

        self.events.push_back(event);

        // Trim old events
        while self.events.len() > self.config.max_events {
            self.events.pop_front();
        }
    }

    /// Get recent events
    pub fn recent(&self, limit: usize) -> impl Iterator<Item = &DashboardEvent> {
        self.events.iter().rev().take(limit)
    }

    /// Get events since sequence number
    pub fn since(&self, sequence: u64) -> impl Iterator<Item = &DashboardEvent> {
        let seq_str = format!("evt-{}", sequence);
        self.events.iter()
            .skip_while(move |e| e.id <= seq_str)
    }

    /// Get events by type
    pub fn by_type(&self, event_type: DashboardEventType) -> impl Iterator<Item = &DashboardEvent> {
        let type_discriminant = std::mem::discriminant(&event_type);
        self.events.iter()
            .filter(move |e| std::mem::discriminant(&e.event_type) == type_discriminant)
    }

    /// Subscribe to events (returns subscriber ID)
    pub fn subscribe(&mut self) -> String {
        let id = format!("sub-{}", self.subscribers.len());
        self.subscribers.push(id.clone());
        id
    }

    /// Unsubscribe
    pub fn unsubscribe(&mut self, subscriber_id: &str) {
        self.subscribers.retain(|s| s != subscriber_id);
    }
}

// ============================================================================
// Chart Data
// ============================================================================

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: u64,
    pub value: f64,
    pub label: Option<String>,
}

/// Time series for charts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeries {
    pub name: String,
    pub data: Vec<DataPoint>,
    pub color: Option<String>,
    pub chart_type: ChartType,
}

/// Chart types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ChartType {
    Line,
    Area,
    Bar,
    Scatter,
}

/// Chart data builder
#[derive(Debug)]
pub struct ChartDataBuilder {
    max_points: usize,
}

impl ChartDataBuilder {
    pub fn new(max_points: usize) -> Self {
        Self { max_points }
    }

    /// Build trust over time chart
    pub fn trust_over_time(&self, history: &[LiveMetrics]) -> TimeSeries {
        let data: Vec<DataPoint> = history.iter()
            .rev()
            .take(self.max_points)
            .rev()
            .map(|m| DataPoint {
                timestamp: m.timestamp,
                value: m.average_trust,
                label: None,
            })
            .collect();

        TimeSeries {
            name: "Average Trust".to_string(),
            data,
            color: Some("#3B82F6".to_string()), // Blue
            chart_type: ChartType::Line,
        }
    }

    /// Build network health chart
    pub fn health_over_time(&self, history: &[LiveMetrics]) -> TimeSeries {
        let data: Vec<DataPoint> = history.iter()
            .rev()
            .take(self.max_points)
            .rev()
            .map(|m| DataPoint {
                timestamp: m.timestamp,
                value: m.network_health as f64,
                label: None,
            })
            .collect();

        TimeSeries {
            name: "Network Health".to_string(),
            data,
            color: Some("#10B981".to_string()), // Green
            chart_type: ChartType::Area,
        }
    }

    /// Build agent count chart
    pub fn agents_over_time(&self, history: &[LiveMetrics]) -> Vec<TimeSeries> {
        let total: Vec<DataPoint> = history.iter()
            .rev()
            .take(self.max_points)
            .rev()
            .map(|m| DataPoint {
                timestamp: m.timestamp,
                value: m.total_agents as f64,
                label: None,
            })
            .collect();

        let active: Vec<DataPoint> = history.iter()
            .rev()
            .take(self.max_points)
            .rev()
            .map(|m| DataPoint {
                timestamp: m.timestamp,
                value: m.active_agents as f64,
                label: None,
            })
            .collect();

        vec![
            TimeSeries {
                name: "Total Agents".to_string(),
                data: total,
                color: Some("#6366F1".to_string()), // Indigo
                chart_type: ChartType::Line,
            },
            TimeSeries {
                name: "Active Agents".to_string(),
                data: active,
                color: Some("#8B5CF6".to_string()), // Purple
                chart_type: ChartType::Line,
            },
        ]
    }

    /// Build trust distribution histogram
    pub fn trust_distribution(&self, trust_scores: &[f64]) -> Vec<DataPoint> {
        let bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let labels = ["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"];

        bins.windows(2)
            .zip(labels.iter())
            .map(|(window, label)| {
                let count = trust_scores.iter()
                    .filter(|&&t| t >= window[0] && t < window[1])
                    .count();
                DataPoint {
                    timestamp: 0,
                    value: count as f64,
                    label: Some(label.to_string()),
                }
            })
            .collect()
    }

    /// Build KREDIT flow chart
    pub fn kredit_flow(&self, history: &[LiveMetrics]) -> TimeSeries {
        let data: Vec<DataPoint> = history.iter()
            .rev()
            .take(self.max_points)
            .rev()
            .map(|m| DataPoint {
                timestamp: m.timestamp,
                value: m.total_kredit as f64,
                label: None,
            })
            .collect();

        TimeSeries {
            name: "Total KREDIT".to_string(),
            data,
            color: Some("#F59E0B".to_string()), // Amber
            chart_type: ChartType::Area,
        }
    }
}

// ============================================================================
// Alerts Panel
// ============================================================================

/// Dashboard alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardAlert {
    /// Alert ID
    pub id: String,
    /// Severity
    pub severity: AlertSeverity,
    /// Title
    pub title: String,
    /// Description
    pub description: String,
    /// Affected agents
    pub affected_agents: Vec<String>,
    /// Timestamp
    pub timestamp: u64,
    /// Status
    pub status: AlertStatus,
    /// Actions available
    pub actions: Vec<AlertAction>,
}

/// Alert severity
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Alert status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertStatus {
    New,
    Acknowledged,
    Investigating,
    Resolved,
    Dismissed,
}

/// Available alert action
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertAction {
    pub id: String,
    pub label: String,
    pub action_type: AlertActionType,
}

/// Alert action types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertActionType {
    Acknowledge,
    Investigate,
    Quarantine,
    Dismiss,
    Escalate,
    Custom(String),
}

/// Alert panel manager
#[derive(Debug)]
pub struct AlertPanel {
    config: DashboardConfig,
    alerts: Vec<DashboardAlert>,
    sequence: u64,
}

impl AlertPanel {
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            config,
            alerts: Vec::new(),
            sequence: 0,
        }
    }

    /// Create alert
    pub fn create_alert(
        &mut self,
        severity: AlertSeverity,
        title: String,
        description: String,
        affected_agents: Vec<String>,
        timestamp: u64,
    ) -> String {
        self.sequence += 1;
        let id = format!("alert-{}", self.sequence);

        let actions = match severity {
            AlertSeverity::Critical => vec![
                AlertAction {
                    id: "ack".to_string(),
                    label: "Acknowledge".to_string(),
                    action_type: AlertActionType::Acknowledge,
                },
                AlertAction {
                    id: "quarantine".to_string(),
                    label: "Quarantine".to_string(),
                    action_type: AlertActionType::Quarantine,
                },
                AlertAction {
                    id: "escalate".to_string(),
                    label: "Escalate".to_string(),
                    action_type: AlertActionType::Escalate,
                },
            ],
            AlertSeverity::High => vec![
                AlertAction {
                    id: "ack".to_string(),
                    label: "Acknowledge".to_string(),
                    action_type: AlertActionType::Acknowledge,
                },
                AlertAction {
                    id: "investigate".to_string(),
                    label: "Investigate".to_string(),
                    action_type: AlertActionType::Investigate,
                },
            ],
            _ => vec![
                AlertAction {
                    id: "ack".to_string(),
                    label: "Acknowledge".to_string(),
                    action_type: AlertActionType::Acknowledge,
                },
                AlertAction {
                    id: "dismiss".to_string(),
                    label: "Dismiss".to_string(),
                    action_type: AlertActionType::Dismiss,
                },
            ],
        };

        let alert = DashboardAlert {
            id: id.clone(),
            severity,
            title,
            description,
            affected_agents,
            timestamp,
            status: AlertStatus::New,
            actions,
        };

        self.alerts.push(alert);
        id
    }

    /// Update alert status
    pub fn update_status(&mut self, alert_id: &str, status: AlertStatus) -> bool {
        if let Some(alert) = self.alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.status = status;
            return true;
        }
        false
    }

    /// Get active alerts (not resolved/dismissed)
    pub fn active_alerts(&self) -> impl Iterator<Item = &DashboardAlert> {
        self.alerts.iter().filter(|a| {
            !matches!(a.status, AlertStatus::Resolved | AlertStatus::Dismissed)
        })
    }

    /// Get alerts by severity
    pub fn by_severity(&self, severity: AlertSeverity) -> impl Iterator<Item = &DashboardAlert> {
        self.alerts.iter().filter(move |a| a.severity == severity)
    }

    /// Get alert counts
    pub fn counts(&self) -> AlertCounts {
        let mut counts = AlertCounts::default();
        for alert in self.active_alerts() {
            match alert.severity {
                AlertSeverity::Critical => counts.critical += 1,
                AlertSeverity::High => counts.high += 1,
                AlertSeverity::Medium => counts.medium += 1,
                AlertSeverity::Low => counts.low += 1,
            }
        }
        counts
    }

    /// Clean old resolved alerts
    pub fn cleanup(&mut self, current_time: u64) {
        self.alerts.retain(|a| {
            if matches!(a.status, AlertStatus::Resolved | AlertStatus::Dismissed) {
                current_time - a.timestamp < self.config.alert_retention_ms
            } else {
                true
            }
        });
    }
}

// ============================================================================
// Dashboard Widgets
// ============================================================================

/// Widget data for dashboard layout
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Widget {
    pub id: String,
    pub widget_type: WidgetType,
    pub title: String,
    pub position: WidgetPosition,
    pub size: WidgetSize,
    pub refresh_interval_ms: u64,
}

/// Widget types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetType {
    MetricCard { metric: String, format: String },
    LineChart { series: Vec<String> },
    BarChart { series: String },
    PieChart { series: String },
    AlertList { max_items: usize },
    AgentTable { columns: Vec<String> },
    NetworkGraph,
    Heatmap { x_axis: String, y_axis: String },
}

/// Widget position (grid-based)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetPosition {
    pub x: u32,
    pub y: u32,
}

/// Widget size
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetSize {
    pub width: u32,
    pub height: u32,
}

/// Default dashboard layout
pub fn default_layout() -> Vec<Widget> {
    vec![
        Widget {
            id: "health".to_string(),
            widget_type: WidgetType::MetricCard {
                metric: "network_health".to_string(),
                format: "{value}%".to_string(),
            },
            title: "Network Health".to_string(),
            position: WidgetPosition { x: 0, y: 0 },
            size: WidgetSize { width: 2, height: 1 },
            refresh_interval_ms: 1000,
        },
        Widget {
            id: "agents".to_string(),
            widget_type: WidgetType::MetricCard {
                metric: "active_agents".to_string(),
                format: "{value}".to_string(),
            },
            title: "Active Agents".to_string(),
            position: WidgetPosition { x: 2, y: 0 },
            size: WidgetSize { width: 2, height: 1 },
            refresh_interval_ms: 1000,
        },
        Widget {
            id: "trust_avg".to_string(),
            widget_type: WidgetType::MetricCard {
                metric: "average_trust".to_string(),
                format: "{value:.2}".to_string(),
            },
            title: "Average Trust".to_string(),
            position: WidgetPosition { x: 4, y: 0 },
            size: WidgetSize { width: 2, height: 1 },
            refresh_interval_ms: 1000,
        },
        Widget {
            id: "alerts".to_string(),
            widget_type: WidgetType::MetricCard {
                metric: "alert_count".to_string(),
                format: "{value}".to_string(),
            },
            title: "Active Alerts".to_string(),
            position: WidgetPosition { x: 6, y: 0 },
            size: WidgetSize { width: 2, height: 1 },
            refresh_interval_ms: 1000,
        },
        Widget {
            id: "trust_chart".to_string(),
            widget_type: WidgetType::LineChart {
                series: vec!["average_trust".to_string(), "median_trust".to_string()],
            },
            title: "Trust Over Time".to_string(),
            position: WidgetPosition { x: 0, y: 1 },
            size: WidgetSize { width: 4, height: 2 },
            refresh_interval_ms: 5000,
        },
        Widget {
            id: "health_chart".to_string(),
            widget_type: WidgetType::LineChart {
                series: vec!["network_health".to_string()],
            },
            title: "Health Over Time".to_string(),
            position: WidgetPosition { x: 4, y: 1 },
            size: WidgetSize { width: 4, height: 2 },
            refresh_interval_ms: 5000,
        },
        Widget {
            id: "alert_list".to_string(),
            widget_type: WidgetType::AlertList { max_items: 10 },
            title: "Recent Alerts".to_string(),
            position: WidgetPosition { x: 0, y: 3 },
            size: WidgetSize { width: 4, height: 2 },
            refresh_interval_ms: 2000,
        },
        Widget {
            id: "trust_dist".to_string(),
            widget_type: WidgetType::BarChart {
                series: "trust_distribution".to_string(),
            },
            title: "Trust Distribution".to_string(),
            position: WidgetPosition { x: 4, y: 3 },
            size: WidgetSize { width: 4, height: 2 },
            refresh_interval_ms: 10000,
        },
    ]
}

// ============================================================================
// Dashboard State
// ============================================================================

/// Complete dashboard state
#[derive(Debug)]
pub struct Dashboard {
    pub config: DashboardConfig,
    pub metrics: MetricsAggregator,
    pub events: EventStream,
    pub alerts: AlertPanel,
    pub layout: Vec<Widget>,
}

impl Dashboard {
    pub fn new(config: DashboardConfig) -> Self {
        Self {
            metrics: MetricsAggregator::new(config.clone()),
            events: EventStream::new(config.clone()),
            alerts: AlertPanel::new(config.clone()),
            layout: default_layout(),
            config,
        }
    }

    /// Get full dashboard state as JSON
    pub fn to_json(&self) -> serde_json::Value {
        let chart_builder = ChartDataBuilder::new(self.config.max_chart_points);
        let history: Vec<_> = self.metrics.history().cloned().collect();

        serde_json::json!({
            "metrics": self.metrics.current(),
            "charts": {
                "trust_over_time": chart_builder.trust_over_time(&history),
                "health_over_time": chart_builder.health_over_time(&history),
                "agents_over_time": chart_builder.agents_over_time(&history),
            },
            "alerts": {
                "counts": self.alerts.counts(),
                "active": self.alerts.active_alerts().collect::<Vec<_>>(),
            },
            "layout": &self.layout,
        })
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_aggregator() {
        let mut agg = MetricsAggregator::new(DashboardConfig::default());

        let input = MetricsInput {
            timestamp: 1000,
            agents: vec![
                AgentMetricInput {
                    id: "agent-1".to_string(),
                    trust_score: 0.8,
                    kredit_balance: 1000,
                    is_active: true,
                },
                AgentMetricInput {
                    id: "agent-2".to_string(),
                    trust_score: 0.6,
                    kredit_balance: 500,
                    is_active: true,
                },
            ],
            active_proposals: 5,
            consensus_success_rate: 0.9,
            alerts: AlertCounts::default(),
        };

        agg.update(input);

        assert_eq!(agg.current().total_agents, 2);
        assert!((agg.current().average_trust - 0.7).abs() < 0.01);
    }

    #[test]
    fn test_event_stream() {
        let mut stream = EventStream::new(DashboardConfig::default());

        stream.emit(
            DashboardEventType::AgentCreated,
            serde_json::json!({"agent_id": "agent-1"}),
            EventPriority::Normal,
            1000,
        );

        stream.emit(
            DashboardEventType::TrustChanged,
            serde_json::json!({"agent_id": "agent-1", "old": 0.5, "new": 0.7}),
            EventPriority::High,
            2000,
        );

        assert_eq!(stream.recent(10).count(), 2);
    }

    #[test]
    fn test_alert_panel() {
        let mut panel = AlertPanel::new(DashboardConfig::default());

        let id = panel.create_alert(
            AlertSeverity::High,
            "Trust Manipulation Detected".to_string(),
            "Agent showing suspicious trust patterns".to_string(),
            vec!["agent-1".to_string()],
            1000,
        );

        assert!(panel.active_alerts().count() == 1);

        panel.update_status(&id, AlertStatus::Resolved);
        assert!(panel.active_alerts().count() == 0);
    }

    #[test]
    fn test_chart_data() {
        let builder = ChartDataBuilder::new(100);

        let history = vec![
            LiveMetrics {
                timestamp: 1000,
                total_agents: 10,
                active_agents: 8,
                average_trust: 0.7,
                median_trust: 0.75,
                trust_stddev: 0.1,
                total_kredit: 10000,
                active_proposals: 5,
                consensus_success_rate: 0.9,
                network_health: 85,
                alerts_by_severity: AlertCounts::default(),
            },
            LiveMetrics {
                timestamp: 2000,
                total_agents: 12,
                active_agents: 10,
                average_trust: 0.72,
                median_trust: 0.76,
                trust_stddev: 0.09,
                total_kredit: 12000,
                active_proposals: 3,
                consensus_success_rate: 0.95,
                network_health: 90,
                alerts_by_severity: AlertCounts::default(),
            },
        ];

        let trust_series = builder.trust_over_time(&history);
        assert_eq!(trust_series.data.len(), 2);
    }

    #[test]
    fn test_network_health_calculation() {
        let mut agg = MetricsAggregator::new(DashboardConfig::default());

        // Healthy network
        let healthy_input = MetricsInput {
            timestamp: 1000,
            agents: vec![
                AgentMetricInput { id: "a1".to_string(), trust_score: 0.8, kredit_balance: 1000, is_active: true },
                AgentMetricInput { id: "a2".to_string(), trust_score: 0.9, kredit_balance: 1000, is_active: true },
            ],
            active_proposals: 5,
            consensus_success_rate: 0.95,
            alerts: AlertCounts::default(),
        };
        agg.update(healthy_input);
        assert!(agg.current().network_health >= 80);

        // Unhealthy network
        let mut unhealthy_agg = MetricsAggregator::new(DashboardConfig::default());
        let unhealthy_input = MetricsInput {
            timestamp: 1000,
            agents: vec![
                AgentMetricInput { id: "a1".to_string(), trust_score: 0.2, kredit_balance: 1000, is_active: true },
                AgentMetricInput { id: "a2".to_string(), trust_score: 0.3, kredit_balance: 1000, is_active: true },
            ],
            active_proposals: 5,
            consensus_success_rate: 0.5,
            alerts: AlertCounts { critical: 2, high: 3, ..Default::default() },
        };
        unhealthy_agg.update(unhealthy_input);
        assert!(unhealthy_agg.current().network_health < 50);
    }

    #[test]
    fn test_default_layout() {
        let layout = default_layout();
        assert!(!layout.is_empty());
        assert!(layout.iter().any(|w| matches!(w.widget_type, WidgetType::AlertList { .. })));
    }
}
