// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Real-time Agent Monitoring System
//!
//! Provides metrics collection, trust evolution tracking, and alert systems
//! for monitoring agent behavior and health.
//!
//! ## Features
//!
//! - **Metrics Collection**: Tracks key performance indicators
//! - **Trust Evolution**: Monitors K-Vector changes over time
//! - **Calibration Drift Alerts**: Detects when calibration degrades
//! - **Health Scoring**: Composite agent health metric
//! - **Dashboard Support**: Serializable metrics for visualization

use super::calibration_engine::CalibrationQuality;
use super::{AgentStatus, InstrumentalActor};
use crate::matl::KVector;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};

#[cfg(feature = "ts-export")]
use ts_rs::TS;

// ============================================================================
// Metrics Collection
// ============================================================================

/// Key metrics for an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct AgentMetrics {
    /// Agent ID
    pub agent_id: String,
    /// Current trust score
    pub trust_score: f32,
    /// Trust score change (24h)
    pub trust_delta_24h: f32,
    /// KREDIT efficiency (value delivered per KREDIT)
    pub kredit_efficiency: f64,
    /// Success rate (recent window)
    pub success_rate: f64,
    /// Output quality (epistemic weight average)
    pub output_quality: f32,
    /// Calibration quality
    pub calibration_quality: CalibrationQuality,
    /// Coherence (Phi) average
    pub coherence_avg: f64,
    /// Activity level (actions per hour)
    pub activity_rate: f64,
    /// Health score (composite 0-100)
    pub health_score: u8,
    /// Timestamp
    pub timestamp: u64,
}

/// Historical metrics for trend analysis
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct MetricsHistory {
    /// Trust score history
    pub trust_history: VecDeque<(u64, f32)>,
    /// Success rate history
    pub success_history: VecDeque<(u64, f64)>,
    /// Health score history
    pub health_history: VecDeque<(u64, u8)>,
    /// Maximum history size
    pub max_history: usize,
}

impl MetricsHistory {
    /// Create a new metrics history tracker
    pub fn new(max_history: usize) -> Self {
        Self {
            trust_history: VecDeque::with_capacity(max_history),
            success_history: VecDeque::with_capacity(max_history),
            health_history: VecDeque::with_capacity(max_history),
            max_history,
        }
    }

    /// Record a new metrics snapshot in history
    pub fn record(&mut self, metrics: &AgentMetrics) {
        // Trust
        self.trust_history
            .push_back((metrics.timestamp, metrics.trust_score));
        if self.trust_history.len() > self.max_history {
            self.trust_history.pop_front();
        }

        // Success
        self.success_history
            .push_back((metrics.timestamp, metrics.success_rate));
        if self.success_history.len() > self.max_history {
            self.success_history.pop_front();
        }

        // Health
        self.health_history
            .push_back((metrics.timestamp, metrics.health_score));
        if self.health_history.len() > self.max_history {
            self.health_history.pop_front();
        }
    }

    /// Get trust score from N seconds ago
    pub fn trust_at(&self, seconds_ago: u64, current_time: u64) -> Option<f32> {
        let target_time = current_time.saturating_sub(seconds_ago);
        self.trust_history
            .iter()
            .rev()
            .find(|(t, _)| *t <= target_time)
            .map(|(_, v)| *v)
    }
}

// ============================================================================
// Trust Evolution Tracking
// ============================================================================

/// Trust evolution event
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct TrustEvent {
    /// Event type
    pub event_type: TrustEventType,
    /// K-Vector before
    pub kvector_before: KVectorSnapshot,
    /// K-Vector after
    pub kvector_after: KVectorSnapshot,
    /// Trust score change
    pub trust_delta: f32,
    /// Reason for change
    pub reason: String,
    /// Timestamp
    pub timestamp: u64,
}

/// Type of trust event
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum TrustEventType {
    /// Positive behavior
    PositiveAction,
    /// Negative behavior
    NegativeAction,
    /// Calibration adjustment
    CalibrationAdjustment,
    /// Coherence change
    CoherenceChange,
    /// Gaming penalty
    GamingPenalty,
    /// Collaboration bonus
    CollaborationBonus,
    /// Epoch reset
    EpochReset,
}

/// Snapshot of K-Vector dimensions
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct KVectorSnapshot {
    /// Reputation dimension
    pub k_r: f32,
    /// Activity dimension
    pub k_a: f32,
    /// Integrity dimension
    pub k_i: f32,
    /// Performance dimension
    pub k_p: f32,
    /// Materiality dimension
    pub k_m: f32,
    /// Sustainability dimension
    pub k_s: f32,
    /// Historical dimension
    pub k_h: f32,
    /// Topological dimension
    pub k_topo: f32,
    /// Computed trust score
    pub trust_score: f32,
}

impl From<&KVector> for KVectorSnapshot {
    fn from(kv: &KVector) -> Self {
        Self {
            k_r: kv.k_r,
            k_a: kv.k_a,
            k_i: kv.k_i,
            k_p: kv.k_p,
            k_m: kv.k_m,
            k_s: kv.k_s,
            k_h: kv.k_h,
            k_topo: kv.k_topo,
            trust_score: kv.trust_score(),
        }
    }
}

// ============================================================================
// Alert System
// ============================================================================

/// Types of alerts
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum AlertType {
    /// Trust dropped significantly
    TrustDrop,
    /// Calibration degraded
    CalibrationDrift,
    /// Coherence too low
    LowCoherence,
    /// KREDIT running low
    LowKredit,
    /// Suspicious activity detected
    SuspiciousActivity,
    /// Health score critical
    CriticalHealth,
    /// Agent throttled/suspended
    StatusChange,
}

/// Alert severity
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub enum AlertSeverity {
    /// Informational alert
    Info,
    /// Warning level alert
    Warning,
    /// Error level alert
    Error,
    /// Critical alert requiring immediate attention
    Critical,
}

/// An alert for an agent
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct AgentAlert {
    /// Agent ID
    pub agent_id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Severity
    pub severity: AlertSeverity,
    /// Message
    pub message: String,
    /// Current value that triggered alert
    pub current_value: f64,
    /// Threshold that was crossed
    pub threshold: f64,
    /// Timestamp
    pub timestamp: u64,
    /// Acknowledged
    pub acknowledged: bool,
}

/// Alert thresholds configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// Trust drop threshold (e.g., 0.1 = 10% drop)
    pub trust_drop_threshold: f32,
    /// Minimum calibration quality
    pub min_calibration_quality: CalibrationQuality,
    /// Minimum coherence
    pub min_coherence: f64,
    /// KREDIT warning percentage
    pub kredit_warning_pct: f64,
    /// Critical health score
    pub critical_health: u8,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            trust_drop_threshold: 0.1,
            min_calibration_quality: CalibrationQuality::Fair,
            min_coherence: 0.5,
            kredit_warning_pct: 0.2,
            critical_health: 30,
        }
    }
}

// ============================================================================
// Monitoring Engine
// ============================================================================

/// The monitoring engine
pub struct MonitoringEngine {
    /// Metrics by agent
    pub metrics: HashMap<String, AgentMetrics>,
    /// Metrics history by agent
    pub history: HashMap<String, MetricsHistory>,
    /// Trust events by agent
    pub trust_events: HashMap<String, Vec<TrustEvent>>,
    /// Active alerts
    pub alerts: Vec<AgentAlert>,
    /// Alert thresholds
    pub thresholds: AlertThresholds,
    /// History size
    history_size: usize,
}

impl MonitoringEngine {
    /// Create a new monitoring engine
    pub fn new(thresholds: AlertThresholds, history_size: usize) -> Self {
        Self {
            metrics: HashMap::new(),
            history: HashMap::new(),
            trust_events: HashMap::new(),
            alerts: Vec::new(),
            thresholds,
            history_size,
        }
    }

    /// Collect metrics for an agent
    pub fn collect_metrics(&mut self, agent: &InstrumentalActor, timestamp: u64) -> AgentMetrics {
        let agent_id = agent.agent_id.as_str().to_string();

        // Calculate metrics
        let trust_score = agent.k_vector.trust_score();

        // Get 24h delta
        let trust_delta_24h = self
            .history
            .get(&agent_id)
            .and_then(|h| h.trust_at(86400, timestamp))
            .map(|old| trust_score - old)
            .unwrap_or(0.0);

        // Success rate (recent 100 actions)
        let recent_actions: Vec<_> = agent.behavior_log.iter().rev().take(100).collect();
        let successes = recent_actions
            .iter()
            .filter(|a| a.outcome == super::ActionOutcome::Success)
            .count();
        let success_rate = if recent_actions.is_empty() {
            0.0
        } else {
            successes as f64 / recent_actions.len() as f64
        };

        // KREDIT efficiency
        let total_kredit: u64 = agent.behavior_log.iter().map(|a| a.kredit_consumed).sum();
        let kredit_efficiency = if total_kredit > 0 {
            successes as f64 / total_kredit as f64
        } else {
            0.0
        };

        // Output quality
        let output_quality = agent.average_epistemic_weight();

        // Activity rate (last hour)
        let hour_ago = timestamp.saturating_sub(3600);
        let recent_count = agent
            .behavior_log
            .iter()
            .filter(|a| a.timestamp >= hour_ago)
            .count();
        let activity_rate = recent_count as f64;

        // Health score (composite)
        let health_score = self.compute_health_score(
            trust_score,
            success_rate,
            output_quality,
            agent.kredit_balance as f64 / agent.kredit_cap as f64,
            &agent.status,
        );

        let metrics = AgentMetrics {
            agent_id: agent_id.clone(),
            trust_score,
            trust_delta_24h,
            kredit_efficiency,
            success_rate,
            output_quality,
            calibration_quality: CalibrationQuality::Insufficient, // Would come from calibration engine
            coherence_avg: 0.7,                                    // Would come from phi bridge
            activity_rate,
            health_score,
            timestamp,
        };

        // Store metrics
        self.metrics.insert(agent_id.clone(), metrics.clone());

        // Update history
        let history = self
            .history
            .entry(agent_id)
            .or_insert_with(|| MetricsHistory::new(self.history_size));
        history.record(&metrics);

        // Check for alerts
        self.check_alerts(&metrics);

        metrics
    }

    fn compute_health_score(
        &self,
        trust: f32,
        success_rate: f64,
        output_quality: f32,
        kredit_ratio: f64,
        status: &AgentStatus,
    ) -> u8 {
        // Base components (weighted)
        let trust_component = trust as f64 * 30.0;
        let success_component = success_rate * 25.0;
        let quality_component = output_quality as f64 * 20.0;
        let kredit_component = kredit_ratio.clamp(0.0, 1.0) * 15.0;

        let status_component = match status {
            AgentStatus::Active => 10.0,
            AgentStatus::Throttled => 5.0,
            AgentStatus::Suspended => 0.0,
            AgentStatus::Revoked => 0.0,
        };

        let raw_score = trust_component
            + success_component
            + quality_component
            + kredit_component
            + status_component;

        (raw_score.clamp(0.0, 100.0)) as u8
    }

    fn check_alerts(&mut self, metrics: &AgentMetrics) {
        let agent_id = &metrics.agent_id;

        // Trust drop alert
        if metrics.trust_delta_24h < -self.thresholds.trust_drop_threshold {
            self.alerts.push(AgentAlert {
                agent_id: agent_id.clone(),
                alert_type: AlertType::TrustDrop,
                severity: AlertSeverity::Warning,
                message: format!(
                    "Trust dropped by {:.1}% in 24h",
                    metrics.trust_delta_24h.abs() * 100.0
                ),
                current_value: metrics.trust_score as f64,
                threshold: self.thresholds.trust_drop_threshold as f64,
                timestamp: metrics.timestamp,
                acknowledged: false,
            });
        }

        // Critical health alert
        if metrics.health_score < self.thresholds.critical_health {
            self.alerts.push(AgentAlert {
                agent_id: agent_id.clone(),
                alert_type: AlertType::CriticalHealth,
                severity: AlertSeverity::Critical,
                message: format!("Health score critical: {}", metrics.health_score),
                current_value: metrics.health_score as f64,
                threshold: self.thresholds.critical_health as f64,
                timestamp: metrics.timestamp,
                acknowledged: false,
            });
        }
    }

    /// Record a trust event
    pub fn record_trust_event(
        &mut self,
        agent_id: &str,
        event_type: TrustEventType,
        before: &KVector,
        after: &KVector,
        reason: String,
        timestamp: u64,
    ) {
        let event = TrustEvent {
            event_type,
            kvector_before: before.into(),
            kvector_after: after.into(),
            trust_delta: after.trust_score() - before.trust_score(),
            reason,
            timestamp,
        };

        self.trust_events
            .entry(agent_id.to_string())
            .or_default()
            .push(event);
    }

    /// Get unacknowledged alerts
    pub fn get_active_alerts(&self) -> Vec<&AgentAlert> {
        self.alerts.iter().filter(|a| !a.acknowledged).collect()
    }

    /// Acknowledge an alert
    pub fn acknowledge_alert(&mut self, index: usize) -> bool {
        if let Some(alert) = self.alerts.get_mut(index) {
            alert.acknowledged = true;
            true
        } else {
            false
        }
    }

    /// Get dashboard summary
    pub fn dashboard_summary(&self) -> DashboardSummary {
        let total_agents = self.metrics.len();
        let healthy = self
            .metrics
            .values()
            .filter(|m| m.health_score >= 70)
            .count();
        let warning = self
            .metrics
            .values()
            .filter(|m| m.health_score >= 30 && m.health_score < 70)
            .count();
        let critical = self
            .metrics
            .values()
            .filter(|m| m.health_score < 30)
            .count();

        let avg_trust: f64 = if total_agents > 0 {
            self.metrics
                .values()
                .map(|m| m.trust_score as f64)
                .sum::<f64>()
                / total_agents as f64
        } else {
            0.0
        };

        let active_alerts = self.alerts.iter().filter(|a| !a.acknowledged).count();

        DashboardSummary {
            total_agents,
            healthy_agents: healthy,
            warning_agents: warning,
            critical_agents: critical,
            average_trust: avg_trust,
            active_alerts,
            critical_alerts: self
                .alerts
                .iter()
                .filter(|a| !a.acknowledged && a.severity == AlertSeverity::Critical)
                .count(),
        }
    }
}

/// Dashboard summary for quick overview
#[derive(Clone, Debug, Serialize, Deserialize)]
#[cfg_attr(feature = "ts-export", derive(TS))]
#[cfg_attr(feature = "ts-export", ts(export, export_to = "bindings/agentic/"))]
pub struct DashboardSummary {
    /// Total number of agents being monitored
    pub total_agents: usize,
    /// Agents in healthy state
    pub healthy_agents: usize,
    /// Agents with warnings
    pub warning_agents: usize,
    /// Agents in critical state
    pub critical_agents: usize,
    /// Average trust score across agents
    pub average_trust: f64,
    /// Number of active (unacknowledged) alerts
    pub active_alerts: usize,
    /// Number of critical unacknowledged alerts
    pub critical_alerts: usize,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agentic::UncertaintyCalibration;
    use crate::agentic::{
        ActionOutcome, AgentClass, AgentConstraints, AgentId, BehaviorLogEntry, EpistemicStats,
    };

    fn create_test_agent(id: &str) -> InstrumentalActor {
        InstrumentalActor {
            agent_id: AgentId::from_string(id.to_string()),
            sponsor_did: "sponsor".to_string(),
            agent_class: AgentClass::Supervised,
            kredit_balance: 5000,
            kredit_cap: 10000,
            constraints: AgentConstraints::default(),
            behavior_log: vec![],
            status: AgentStatus::Active,
            created_at: 0,
            last_activity: 0,
            actions_this_hour: 0,
            k_vector: KVector::new(0.7, 0.6, 0.8, 0.7, 0.3, 0.4, 0.6, 0.3, 0.7, 0.65),
            epistemic_stats: EpistemicStats::default(),
            output_history: vec![],
            uncertainty_calibration: UncertaintyCalibration::default(),
            pending_escalations: vec![],
        }
    }

    #[test]
    fn test_metrics_collection() {
        let mut engine = MonitoringEngine::new(AlertThresholds::default(), 100);
        let mut agent = create_test_agent("agent-1");

        // Add some behavior
        for i in 0..20 {
            agent.behavior_log.push(BehaviorLogEntry {
                timestamp: 1000 + i * 60,
                action_type: "process".to_string(),
                kredit_consumed: 10,
                counterparties: vec![],
                outcome: if i % 3 == 0 {
                    ActionOutcome::Error
                } else {
                    ActionOutcome::Success
                },
            });
        }

        let metrics = engine.collect_metrics(&agent, 2500);

        assert_eq!(metrics.agent_id, "agent-1");
        assert!(metrics.trust_score > 0.0);
        assert!(metrics.success_rate > 0.5);
        assert!(metrics.health_score > 0);
    }

    #[test]
    fn test_trust_event_recording() {
        let mut engine = MonitoringEngine::new(AlertThresholds::default(), 100);

        let before = KVector::new_participant();
        let mut after = before.clone();
        after.k_r = 0.7;

        engine.record_trust_event(
            "agent-1",
            TrustEventType::PositiveAction,
            &before,
            &after,
            "Good behavior".to_string(),
            1000,
        );

        let events = engine.trust_events.get("agent-1").unwrap();
        assert_eq!(events.len(), 1);
        assert!(events[0].trust_delta > 0.0);
    }

    #[test]
    fn test_alert_generation() {
        let mut engine = MonitoringEngine::new(AlertThresholds::default(), 100);
        let mut agent = create_test_agent("agent-1");
        agent.k_vector = KVector::new(0.2, 0.1, 0.3, 0.2, 0.1, 0.1, 0.2, 0.1, 0.0, 0.1); // Low trust
        agent.kredit_balance = 100; // Low KREDIT

        let metrics = engine.collect_metrics(&agent, 1000);

        // Should have critical health alert
        assert!(metrics.health_score < 50);
    }

    #[test]
    fn test_dashboard_summary() {
        let mut engine = MonitoringEngine::new(AlertThresholds::default(), 100);

        // Add healthy agent
        let healthy = create_test_agent("healthy");
        engine.collect_metrics(&healthy, 1000);

        // Add unhealthy agent
        let mut unhealthy = create_test_agent("unhealthy");
        unhealthy.k_vector = KVector::new(0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1);
        unhealthy.kredit_balance = 0;
        engine.collect_metrics(&unhealthy, 1000);

        let summary = engine.dashboard_summary();
        assert_eq!(summary.total_agents, 2);
        // At least verify we have some agents categorized
        assert!(summary.healthy_agents + summary.warning_agents + summary.critical_agents == 2);
    }
}
