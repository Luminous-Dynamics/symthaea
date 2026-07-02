// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Anomaly Detection System
//!
//! Component 19: Detects unusual patterns in the system that may indicate
//! problems or opportunities.
//!
//! Monitors for:
//! - Sudden success rate changes (pattern degradation or improvement)
//! - Unusual collective behavior (echo chambers, coordinated activity)
//! - Trust anomalies (rapid changes, suspicious vouching patterns)
//! - Orphaned dependencies (deprecated patterns with active dependents)
//! - Calibration drift (predictions becoming unreliable)
//! - Usage anomalies (sudden spikes or drops)

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use std::collections::HashMap;

use super::{DomainId, PatternId, SymthaeaId};

// ==============================================================================
// COMPONENT 19: ANOMALY DETECTION SYSTEM
// ==============================================================================
//
// This component monitors the WisdomEngine for unusual patterns that may
// indicate problems (degradation, manipulation, drift) or opportunities
// (emerging patterns, novel discoveries).
//
// Philosophy: Early warning system for system health. We detect anomalies,
// not judge them - anomalies may be good (breakthrough) or bad (attack).

/// Types of anomalies that can be detected
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AnomalyType {
    /// Sudden drop in pattern success rate
    SuccessRateDrop,

    /// Sudden increase in pattern success rate
    SuccessRateSpike,

    /// Pattern usage dropped significantly
    UsageDrop,

    /// Pattern usage spiked suddenly
    UsageSpike,

    /// High agreement levels suggesting echo chamber
    EchoChamber,

    /// Coordinated activity from multiple agents
    CoordinatedActivity,

    /// Rapid trust score changes
    TrustVolatility,

    /// Suspicious vouching patterns (circular, mass)
    SuspiciousVouching,

    /// Active pattern depends on deprecated pattern
    OrphanedDependency,

    /// Circular dependency detected
    CircularDependency,

    /// Prediction calibration drifting
    CalibrationDrift,

    /// Overconfident predictions (high confidence, low accuracy)
    Overconfidence,

    /// Pattern conflicts increasing
    ConflictEscalation,

    /// Domain becoming saturated with similar patterns
    DomainSaturation,

    /// Pattern not used despite high relevance
    UnderutilizedPattern,

    /// New pattern rapidly gaining adoption
    RapidAdoption,

    /// Multiple agents independently discovered same pattern
    ConvergentDiscovery,

    /// System health metrics degrading
    SystemHealthDecline,

    /// Custom anomaly type
    Custom(String),
}

impl AnomalyType {
    /// Get a human-readable name for this anomaly type
    pub fn name(&self) -> &str {
        match self {
            AnomalyType::SuccessRateDrop => "Success Rate Drop",
            AnomalyType::SuccessRateSpike => "Success Rate Spike",
            AnomalyType::UsageDrop => "Usage Drop",
            AnomalyType::UsageSpike => "Usage Spike",
            AnomalyType::EchoChamber => "Echo Chamber",
            AnomalyType::CoordinatedActivity => "Coordinated Activity",
            AnomalyType::TrustVolatility => "Trust Volatility",
            AnomalyType::SuspiciousVouching => "Suspicious Vouching",
            AnomalyType::OrphanedDependency => "Orphaned Dependency",
            AnomalyType::CircularDependency => "Circular Dependency",
            AnomalyType::CalibrationDrift => "Calibration Drift",
            AnomalyType::Overconfidence => "Overconfidence",
            AnomalyType::ConflictEscalation => "Conflict Escalation",
            AnomalyType::DomainSaturation => "Domain Saturation",
            AnomalyType::UnderutilizedPattern => "Underutilized Pattern",
            AnomalyType::RapidAdoption => "Rapid Adoption",
            AnomalyType::ConvergentDiscovery => "Convergent Discovery",
            AnomalyType::SystemHealthDecline => "System Health Decline",
            AnomalyType::Custom(name) => name,
        }
    }

    /// Check if this anomaly type is typically negative
    pub fn is_typically_negative(&self) -> bool {
        matches!(
            self,
            AnomalyType::SuccessRateDrop
                | AnomalyType::UsageDrop
                | AnomalyType::EchoChamber
                | AnomalyType::CoordinatedActivity
                | AnomalyType::TrustVolatility
                | AnomalyType::SuspiciousVouching
                | AnomalyType::OrphanedDependency
                | AnomalyType::CircularDependency
                | AnomalyType::CalibrationDrift
                | AnomalyType::Overconfidence
                | AnomalyType::ConflictEscalation
                | AnomalyType::SystemHealthDecline
        )
    }

    /// Check if this anomaly type is typically positive
    pub fn is_typically_positive(&self) -> bool {
        matches!(
            self,
            AnomalyType::SuccessRateSpike
                | AnomalyType::RapidAdoption
                | AnomalyType::ConvergentDiscovery
        )
    }

    /// Check if this anomaly type is neutral (could be good or bad)
    pub fn is_neutral(&self) -> bool {
        !self.is_typically_negative() && !self.is_typically_positive()
    }
}

/// Severity levels for anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AnomalySeverity {
    /// Informational - worth noting but not concerning
    Info,

    /// Low severity - minor deviation from normal
    Low,

    /// Medium severity - notable deviation requiring attention
    Medium,

    /// High severity - significant issue requiring action
    High,

    /// Critical severity - immediate action required
    Critical,
}

impl AnomalySeverity {
    /// Get a numeric value for sorting (higher = more severe)
    pub fn value(&self) -> u8 {
        match self {
            AnomalySeverity::Info => 0,
            AnomalySeverity::Low => 1,
            AnomalySeverity::Medium => 2,
            AnomalySeverity::High => 3,
            AnomalySeverity::Critical => 4,
        }
    }

    /// Create severity from a deviation magnitude (0.0-1.0+)
    pub fn from_magnitude(magnitude: f32) -> Self {
        if magnitude < 0.1 {
            AnomalySeverity::Info
        } else if magnitude < 0.25 {
            AnomalySeverity::Low
        } else if magnitude < 0.5 {
            AnomalySeverity::Medium
        } else if magnitude < 0.75 {
            AnomalySeverity::High
        } else {
            AnomalySeverity::Critical
        }
    }

    /// Get an indicator symbol
    pub fn indicator(&self) -> &str {
        match self {
            AnomalySeverity::Info => "ℹ",
            AnomalySeverity::Low => "⚪",
            AnomalySeverity::Medium => "🟡",
            AnomalySeverity::High => "🟠",
            AnomalySeverity::Critical => "🔴",
        }
    }
}

/// A detected anomaly
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Anomaly {
    /// Unique identifier for this anomaly instance
    pub anomaly_id: u64,

    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,

    /// Severity level
    pub severity: AnomalySeverity,

    /// Patterns affected by this anomaly
    pub affected_patterns: Vec<PatternId>,

    /// Agents involved (if applicable)
    pub involved_agents: Vec<SymthaeaId>,

    /// Domains affected (if applicable)
    pub affected_domains: Vec<DomainId>,

    /// When the anomaly was detected
    pub detected_at: u64,

    /// When the anomaly started (if known)
    pub started_at: Option<u64>,

    /// Human-readable description
    pub description: String,

    /// Detailed evidence supporting this detection
    pub evidence: Vec<String>,

    /// Recommended actions
    pub recommended_actions: Vec<String>,

    /// Numeric magnitude of the anomaly (0.0-1.0+)
    pub magnitude: f32,

    /// Whether this anomaly has been acknowledged
    pub acknowledged: bool,

    /// Whether this anomaly has been resolved
    pub resolved: bool,

    /// Resolution notes (if resolved)
    pub resolution_notes: Option<String>,

    /// Related anomaly IDs (for tracking patterns of anomalies)
    pub related_anomalies: Vec<u64>,
}

impl Anomaly {
    /// Create a new anomaly
    pub fn new(
        anomaly_id: u64,
        anomaly_type: AnomalyType,
        severity: AnomalySeverity,
        description: &str,
        timestamp: u64,
    ) -> Self {
        Self {
            anomaly_id,
            anomaly_type,
            severity,
            affected_patterns: Vec::new(),
            involved_agents: Vec::new(),
            affected_domains: Vec::new(),
            detected_at: timestamp,
            started_at: None,
            description: description.to_string(),
            evidence: Vec::new(),
            recommended_actions: Vec::new(),
            magnitude: 0.0,
            acknowledged: false,
            resolved: false,
            resolution_notes: None,
            related_anomalies: Vec::new(),
        }
    }

    /// Add affected patterns
    pub fn with_patterns(mut self, patterns: Vec<PatternId>) -> Self {
        self.affected_patterns = patterns;
        self
    }

    /// Add involved agents
    pub fn with_agents(mut self, agents: Vec<SymthaeaId>) -> Self {
        self.involved_agents = agents;
        self
    }

    /// Add affected domains
    pub fn with_domains(mut self, domains: Vec<DomainId>) -> Self {
        self.affected_domains = domains;
        self
    }

    /// Add evidence
    pub fn with_evidence(mut self, evidence: Vec<String>) -> Self {
        self.evidence = evidence;
        self
    }

    /// Add recommended actions
    pub fn with_actions(mut self, actions: Vec<String>) -> Self {
        self.recommended_actions = actions;
        self
    }

    /// Set magnitude
    pub fn with_magnitude(mut self, magnitude: f32) -> Self {
        self.magnitude = magnitude;
        self
    }

    /// Set start time
    pub fn with_start_time(mut self, started_at: u64) -> Self {
        self.started_at = Some(started_at);
        self
    }

    /// Acknowledge the anomaly
    pub fn acknowledge(&mut self) {
        self.acknowledged = true;
    }

    /// Resolve the anomaly
    pub fn resolve(&mut self, notes: &str) {
        self.resolved = true;
        self.resolution_notes = Some(notes.to_string());
    }

    /// Check if this anomaly is active (not resolved)
    pub fn is_active(&self) -> bool {
        !self.resolved
    }

    /// Get a short summary
    pub fn summary(&self) -> String {
        format!(
            "{} {} [{}]: {}",
            self.severity.indicator(),
            self.anomaly_type.name(),
            if self.resolved {
                "RESOLVED"
            } else if self.acknowledged {
                "ACK"
            } else {
                "NEW"
            },
            self.description
        )
    }

    /// Format as detailed report
    pub fn detailed_report(&self) -> String {
        let mut report = String::new();

        report.push_str(&format!("=== Anomaly #{} ===\n", self.anomaly_id));
        report.push_str(&format!("Type: {}\n", self.anomaly_type.name()));
        report.push_str(&format!(
            "Severity: {:?} {}\n",
            self.severity,
            self.severity.indicator()
        ));
        report.push_str(&format!("Magnitude: {:.2}\n", self.magnitude));
        report.push_str(&format!("Detected: {}\n", self.detected_at));
        if let Some(started) = self.started_at {
            report.push_str(&format!("Started: {}\n", started));
        }
        report.push_str(&format!("\nDescription: {}\n", self.description));

        if !self.affected_patterns.is_empty() {
            report.push_str(&format!(
                "\nAffected Patterns: {:?}\n",
                self.affected_patterns
            ));
        }
        if !self.involved_agents.is_empty() {
            report.push_str(&format!("Involved Agents: {:?}\n", self.involved_agents));
        }
        if !self.affected_domains.is_empty() {
            report.push_str(&format!("Affected Domains: {:?}\n", self.affected_domains));
        }

        if !self.evidence.is_empty() {
            report.push_str("\nEvidence:\n");
            for e in &self.evidence {
                report.push_str(&format!("  - {}\n", e));
            }
        }

        if !self.recommended_actions.is_empty() {
            report.push_str("\nRecommended Actions:\n");
            for a in &self.recommended_actions {
                report.push_str(&format!("  - {}\n", a));
            }
        }

        if self.resolved {
            report.push_str(&format!("\nStatus: RESOLVED\n"));
            if let Some(notes) = &self.resolution_notes {
                report.push_str(&format!("Resolution: {}\n", notes));
            }
        } else if self.acknowledged {
            report.push_str("\nStatus: ACKNOWLEDGED\n");
        } else {
            report.push_str("\nStatus: NEW - Requires attention\n");
        }

        report
    }
}

/// Configuration for anomaly detection
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnomalyConfig {
    /// Threshold for success rate drop detection (e.g., 0.2 = 20% drop)
    pub success_rate_drop_threshold: f32,

    /// Threshold for success rate spike detection
    pub success_rate_spike_threshold: f32,

    /// Threshold for usage drop detection (percentage)
    pub usage_drop_threshold: f32,

    /// Threshold for usage spike detection (multiplier)
    pub usage_spike_threshold: f32,

    /// Echo chamber agreement threshold (e.g., 0.95 = 95% agreement)
    pub echo_chamber_threshold: f32,

    /// Minimum agents for coordinated activity detection
    pub coordinated_activity_min_agents: usize,

    /// Trust volatility threshold (change per time period)
    pub trust_volatility_threshold: f32,

    /// Calibration drift threshold
    pub calibration_drift_threshold: f32,

    /// Overconfidence threshold (confidence - accuracy gap)
    pub overconfidence_threshold: f32,

    /// Domain saturation threshold (similar patterns count)
    pub domain_saturation_threshold: usize,

    /// Rapid adoption threshold (usage growth rate)
    pub rapid_adoption_threshold: f32,

    /// Minimum time window for trend detection (time units)
    pub min_detection_window: u64,

    /// Enable/disable specific detection types
    pub enabled_detections: HashMap<String, bool>,

    /// Suppress anomalies below this severity
    pub min_severity: AnomalySeverity,
}

impl Default for AnomalyConfig {
    fn default() -> Self {
        Self {
            success_rate_drop_threshold: 0.2,
            success_rate_spike_threshold: 0.3,
            usage_drop_threshold: 0.5,
            usage_spike_threshold: 3.0,
            echo_chamber_threshold: 0.95,
            coordinated_activity_min_agents: 5,
            trust_volatility_threshold: 0.3,
            calibration_drift_threshold: 0.15,
            overconfidence_threshold: 0.2,
            domain_saturation_threshold: 20,
            rapid_adoption_threshold: 2.0,
            min_detection_window: 3600, // 1 hour
            enabled_detections: HashMap::new(),
            min_severity: AnomalySeverity::Low,
        }
    }
}

impl AnomalyConfig {
    /// Create a sensitive configuration (detects more anomalies)
    pub fn sensitive() -> Self {
        Self {
            success_rate_drop_threshold: 0.1,
            success_rate_spike_threshold: 0.2,
            usage_drop_threshold: 0.3,
            echo_chamber_threshold: 0.9,
            trust_volatility_threshold: 0.2,
            calibration_drift_threshold: 0.1,
            overconfidence_threshold: 0.15,
            min_severity: AnomalySeverity::Info,
            ..Default::default()
        }
    }

    /// Create a relaxed configuration (fewer false positives)
    pub fn relaxed() -> Self {
        Self {
            success_rate_drop_threshold: 0.3,
            success_rate_spike_threshold: 0.4,
            usage_drop_threshold: 0.7,
            echo_chamber_threshold: 0.98,
            trust_volatility_threshold: 0.5,
            calibration_drift_threshold: 0.25,
            overconfidence_threshold: 0.3,
            min_severity: AnomalySeverity::Medium,
            ..Default::default()
        }
    }

    /// Check if a detection type is enabled
    pub fn is_enabled(&self, detection_type: &str) -> bool {
        self.enabled_detections
            .get(detection_type)
            .copied()
            .unwrap_or(true)
    }

    /// Enable a detection type
    pub fn enable(&mut self, detection_type: &str) {
        self.enabled_detections
            .insert(detection_type.to_string(), true);
    }

    /// Disable a detection type
    pub fn disable(&mut self, detection_type: &str) {
        self.enabled_detections
            .insert(detection_type.to_string(), false);
    }
}

/// Historical data point for trend detection
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DataPoint {
    /// Timestamp
    pub timestamp: u64,

    /// Value at this time
    pub value: f32,

    /// Optional metadata
    pub metadata: Option<String>,
}

impl DataPoint {
    /// Create a new data point
    pub fn new(timestamp: u64, value: f32) -> Self {
        Self {
            timestamp,
            value,
            metadata: None,
        }
    }
}

/// Time series for tracking metrics over time
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct TimeSeries {
    /// Data points
    pub points: Vec<DataPoint>,

    /// Maximum points to retain
    pub max_points: usize,
}

impl TimeSeries {
    /// Create a new time series
    pub fn new(max_points: usize) -> Self {
        Self {
            points: Vec::new(),
            max_points,
        }
    }

    /// Add a data point
    pub fn add(&mut self, timestamp: u64, value: f32) {
        self.points.push(DataPoint::new(timestamp, value));

        // Trim if needed
        if self.points.len() > self.max_points {
            self.points.remove(0);
        }
    }

    /// Get the latest value
    pub fn latest(&self) -> Option<f32> {
        self.points.last().map(|p| p.value)
    }

    /// Get the average value
    pub fn average(&self) -> Option<f32> {
        if self.points.is_empty() {
            return None;
        }
        Some(self.points.iter().map(|p| p.value).sum::<f32>() / self.points.len() as f32)
    }

    /// Get the trend (positive = increasing, negative = decreasing)
    pub fn trend(&self) -> Option<f32> {
        if self.points.len() < 2 {
            return None;
        }

        // Simple linear regression slope
        let n = self.points.len() as f32;
        let sum_x: f32 = (0..self.points.len()).map(|i| i as f32).sum();
        let sum_y: f32 = self.points.iter().map(|p| p.value).sum();
        let sum_xy: f32 = self
            .points
            .iter()
            .enumerate()
            .map(|(i, p)| i as f32 * p.value)
            .sum();
        let sum_xx: f32 = (0..self.points.len()).map(|i| (i as f32).powi(2)).sum();

        let denominator = n * sum_xx - sum_x.powi(2);
        if denominator.abs() < 0.0001 {
            return Some(0.0);
        }

        Some((n * sum_xy - sum_x * sum_y) / denominator)
    }

    /// Get the volatility (standard deviation)
    pub fn volatility(&self) -> Option<f32> {
        let avg = self.average()?;
        if self.points.len() < 2 {
            return None;
        }

        let variance = self
            .points
            .iter()
            .map(|p| (p.value - avg).powi(2))
            .sum::<f32>()
            / (self.points.len() - 1) as f32;

        Some(variance.sqrt())
    }

    /// Detect sudden change from recent average
    pub fn detect_sudden_change(&self, threshold: f32) -> Option<f32> {
        if self.points.len() < 3 {
            return None;
        }

        let recent = self.points.last()?.value;
        let previous_avg = self.points[..self.points.len() - 1]
            .iter()
            .map(|p| p.value)
            .sum::<f32>()
            / (self.points.len() - 1) as f32;

        let change = (recent - previous_avg).abs();
        if change > threshold {
            Some(change)
        } else {
            None
        }
    }

    /// Get points within a time window
    pub fn in_window(&self, start: u64, end: u64) -> Vec<&DataPoint> {
        self.points
            .iter()
            .filter(|p| p.timestamp >= start && p.timestamp <= end)
            .collect()
    }
}

/// Statistics for the anomaly detector
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnomalyStats {
    /// Total anomalies detected
    pub total_detected: u64,

    /// Currently active anomalies
    pub active_count: usize,

    /// Anomalies by type
    pub by_type: HashMap<String, u64>,

    /// Anomalies by severity
    pub by_severity: HashMap<String, u64>,

    /// Average time to acknowledgement
    pub avg_time_to_ack: Option<f32>,

    /// Average time to resolution
    pub avg_time_to_resolution: Option<f32>,

    /// Most affected patterns
    pub most_affected_patterns: Vec<(PatternId, u32)>,

    /// Most involved agents
    pub most_involved_agents: Vec<(SymthaeaId, u32)>,
}

/// The anomaly detector registry
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AnomalyDetector {
    /// Configuration
    config: AnomalyConfig,

    /// Detected anomalies
    anomalies: Vec<Anomaly>,

    /// Next anomaly ID
    next_anomaly_id: u64,

    /// Time series for pattern success rates
    pattern_success_rates: HashMap<PatternId, TimeSeries>,

    /// Time series for pattern usage
    pattern_usage: HashMap<PatternId, TimeSeries>,

    /// Time series for agent trust scores
    agent_trust: HashMap<SymthaeaId, TimeSeries>,

    /// Time series for calibration accuracy
    calibration_history: TimeSeries,

    /// Time series for system health
    system_health_history: TimeSeries,

    /// Statistics
    stats: AnomalyStats,
}

impl Default for AnomalyDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyDetector {
    /// Create a new anomaly detector
    pub fn new() -> Self {
        Self {
            config: AnomalyConfig::default(),
            anomalies: Vec::new(),
            next_anomaly_id: 1,
            pattern_success_rates: HashMap::new(),
            pattern_usage: HashMap::new(),
            agent_trust: HashMap::new(),
            calibration_history: TimeSeries::new(100),
            system_health_history: TimeSeries::new(100),
            stats: AnomalyStats::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AnomalyConfig) -> Self {
        Self {
            config,
            ..Self::new()
        }
    }

    /// Get current configuration
    pub fn config(&self) -> &AnomalyConfig {
        &self.config
    }

    /// Update configuration
    pub fn set_config(&mut self, config: AnomalyConfig) {
        self.config = config;
    }

    /// Record a pattern success rate observation
    pub fn record_success_rate(&mut self, pattern_id: PatternId, rate: f32, timestamp: u64) {
        self.pattern_success_rates
            .entry(pattern_id)
            .or_insert_with(|| TimeSeries::new(50))
            .add(timestamp, rate);
    }

    /// Record a pattern usage observation
    pub fn record_usage(&mut self, pattern_id: PatternId, usage_count: u64, timestamp: u64) {
        self.pattern_usage
            .entry(pattern_id)
            .or_insert_with(|| TimeSeries::new(50))
            .add(timestamp, usage_count as f32);
    }

    /// Record an agent trust score observation
    pub fn record_trust(&mut self, agent_id: SymthaeaId, trust: f32, timestamp: u64) {
        self.agent_trust
            .entry(agent_id)
            .or_insert_with(|| TimeSeries::new(50))
            .add(timestamp, trust);
    }

    /// Record calibration accuracy
    pub fn record_calibration(&mut self, accuracy: f32, timestamp: u64) {
        self.calibration_history.add(timestamp, accuracy);
    }

    /// Record system health
    pub fn record_system_health(&mut self, health: f32, timestamp: u64) {
        self.system_health_history.add(timestamp, health);
    }

    /// Detect success rate anomalies for a pattern
    pub fn detect_success_rate_anomaly(
        &mut self,
        pattern_id: PatternId,
        timestamp: u64,
    ) -> Option<Anomaly> {
        let series = self.pattern_success_rates.get(&pattern_id)?;

        // Check for sudden drop
        if let Some(change) = series.detect_sudden_change(self.config.success_rate_drop_threshold) {
            let current = series.latest()?;
            let avg = series.average()?;

            if current < avg {
                // It's a drop
                let magnitude = change / avg.max(0.01);
                let severity = AnomalySeverity::from_magnitude(magnitude);

                if severity >= self.config.min_severity {
                    let anomaly = Anomaly::new(
                        self.next_anomaly_id,
                        AnomalyType::SuccessRateDrop,
                        severity,
                        &format!(
                            "Pattern {} success rate dropped from {:.1}% to {:.1}%",
                            pattern_id,
                            avg * 100.0,
                            current * 100.0
                        ),
                        timestamp,
                    )
                    .with_patterns(vec![pattern_id])
                    .with_magnitude(magnitude)
                    .with_evidence(vec![
                        format!("Previous average: {:.2}", avg),
                        format!("Current value: {:.2}", current),
                        format!("Drop magnitude: {:.2}", change),
                    ])
                    .with_actions(vec![
                        "Investigate recent changes to the pattern".to_string(),
                        "Check for environmental changes".to_string(),
                        "Consider deprecating if trend continues".to_string(),
                    ]);

                    self.next_anomaly_id += 1;
                    return Some(anomaly);
                }
            } else {
                // It's a spike (positive anomaly)
                if change > self.config.success_rate_spike_threshold {
                    let magnitude = change / avg.max(0.01);
                    let severity = AnomalySeverity::from_magnitude(magnitude * 0.5); // Less severe for positive

                    if severity >= self.config.min_severity {
                        let anomaly = Anomaly::new(
                            self.next_anomaly_id,
                            AnomalyType::SuccessRateSpike,
                            severity,
                            &format!(
                                "Pattern {} success rate improved from {:.1}% to {:.1}%",
                                pattern_id,
                                avg * 100.0,
                                current * 100.0
                            ),
                            timestamp,
                        )
                        .with_patterns(vec![pattern_id])
                        .with_magnitude(magnitude)
                        .with_evidence(vec![
                            format!("Previous average: {:.2}", avg),
                            format!("Current value: {:.2}", current),
                        ])
                        .with_actions(vec![
                            "Investigate what caused the improvement".to_string(),
                            "Consider promoting this pattern".to_string(),
                        ]);

                        self.next_anomaly_id += 1;
                        return Some(anomaly);
                    }
                }
            }
        }

        None
    }

    /// Detect trust volatility for an agent
    pub fn detect_trust_anomaly(
        &mut self,
        agent_id: SymthaeaId,
        timestamp: u64,
    ) -> Option<Anomaly> {
        let series = self.agent_trust.get(&agent_id)?;

        if let Some(volatility) = series.volatility() {
            if volatility > self.config.trust_volatility_threshold {
                let severity = AnomalySeverity::from_magnitude(volatility);

                if severity >= self.config.min_severity {
                    let anomaly = Anomaly::new(
                        self.next_anomaly_id,
                        AnomalyType::TrustVolatility,
                        severity,
                        &format!(
                            "Agent {} shows high trust score volatility ({:.2})",
                            agent_id, volatility
                        ),
                        timestamp,
                    )
                    .with_agents(vec![agent_id])
                    .with_magnitude(volatility)
                    .with_evidence(vec![
                        format!("Volatility: {:.2}", volatility),
                        format!("Average trust: {:.2}", series.average().unwrap_or(0.0)),
                    ])
                    .with_actions(vec![
                        "Review agent's recent activity".to_string(),
                        "Check for suspicious patterns".to_string(),
                    ]);

                    self.next_anomaly_id += 1;
                    return Some(anomaly);
                }
            }
        }

        None
    }

    /// Detect calibration drift
    pub fn detect_calibration_drift(&mut self, timestamp: u64) -> Option<Anomaly> {
        if let Some(trend) = self.calibration_history.trend() {
            // Negative trend = declining accuracy
            if trend < -self.config.calibration_drift_threshold {
                let magnitude = trend.abs();
                let severity = AnomalySeverity::from_magnitude(magnitude);

                if severity >= self.config.min_severity {
                    let anomaly = Anomaly::new(
                        self.next_anomaly_id,
                        AnomalyType::CalibrationDrift,
                        severity,
                        &format!("System calibration declining (trend: {:.3})", trend),
                        timestamp,
                    )
                    .with_magnitude(magnitude)
                    .with_evidence(vec![
                        format!("Trend: {:.3}", trend),
                        format!(
                            "Current accuracy: {:.2}",
                            self.calibration_history.latest().unwrap_or(0.0)
                        ),
                    ])
                    .with_actions(vec![
                        "Review prediction accuracy".to_string(),
                        "Consider recalibrating confidence scores".to_string(),
                        "Check for environmental changes".to_string(),
                    ]);

                    self.next_anomaly_id += 1;
                    return Some(anomaly);
                }
            }
        }

        None
    }

    /// Detect system health decline
    pub fn detect_system_health_decline(&mut self, timestamp: u64) -> Option<Anomaly> {
        if let Some(trend) = self.system_health_history.trend() {
            if trend < -0.1 {
                let magnitude = trend.abs();
                let severity = if magnitude > 0.3 {
                    AnomalySeverity::High
                } else if magnitude > 0.2 {
                    AnomalySeverity::Medium
                } else {
                    AnomalySeverity::Low
                };

                if severity >= self.config.min_severity {
                    let anomaly = Anomaly::new(
                        self.next_anomaly_id,
                        AnomalyType::SystemHealthDecline,
                        severity,
                        &format!("System health declining (trend: {:.3})", trend),
                        timestamp,
                    )
                    .with_magnitude(magnitude)
                    .with_evidence(vec![
                        format!("Trend: {:.3}", trend),
                        format!(
                            "Current health: {:.2}",
                            self.system_health_history.latest().unwrap_or(0.0)
                        ),
                    ])
                    .with_actions(vec![
                        "Run comprehensive health diagnostics".to_string(),
                        "Review recent changes".to_string(),
                        "Check resource utilization".to_string(),
                    ]);

                    self.next_anomaly_id += 1;
                    return Some(anomaly);
                }
            }
        }

        None
    }

    /// Create an orphaned dependency anomaly
    pub fn create_orphaned_dependency_anomaly(
        &mut self,
        dependent_id: PatternId,
        deprecated_id: PatternId,
        timestamp: u64,
    ) -> Anomaly {
        let anomaly = Anomaly::new(
            self.next_anomaly_id,
            AnomalyType::OrphanedDependency,
            AnomalySeverity::Medium,
            &format!(
                "Pattern {} depends on deprecated pattern {}",
                dependent_id, deprecated_id
            ),
            timestamp,
        )
        .with_patterns(vec![dependent_id, deprecated_id])
        .with_actions(vec![
            format!("Update pattern {} to use alternative", dependent_id),
            format!("Consider migrating from pattern {}", deprecated_id),
        ]);

        self.next_anomaly_id += 1;
        anomaly
    }

    /// Create a circular dependency anomaly
    pub fn create_circular_dependency_anomaly(
        &mut self,
        patterns: Vec<PatternId>,
        timestamp: u64,
    ) -> Anomaly {
        let anomaly = Anomaly::new(
            self.next_anomaly_id,
            AnomalyType::CircularDependency,
            AnomalySeverity::High,
            &format!(
                "Circular dependency detected among patterns: {:?}",
                patterns
            ),
            timestamp,
        )
        .with_patterns(patterns)
        .with_actions(vec![
            "Break the dependency cycle".to_string(),
            "Review pattern relationships".to_string(),
        ]);

        self.next_anomaly_id += 1;
        anomaly
    }

    /// Create an echo chamber anomaly
    pub fn create_echo_chamber_anomaly(
        &mut self,
        patterns: Vec<PatternId>,
        agreement_level: f32,
        timestamp: u64,
    ) -> Anomaly {
        let severity = if agreement_level > 0.98 {
            AnomalySeverity::High
        } else {
            AnomalySeverity::Medium
        };

        let anomaly = Anomaly::new(
            self.next_anomaly_id,
            AnomalyType::EchoChamber,
            severity,
            &format!(
                "High agreement ({:.1}%) on patterns may indicate echo chamber",
                agreement_level * 100.0
            ),
            timestamp,
        )
        .with_patterns(patterns)
        .with_magnitude(agreement_level)
        .with_actions(vec![
            "Seek diverse perspectives".to_string(),
            "Review pattern sources".to_string(),
            "Consider alternative patterns".to_string(),
        ]);

        self.next_anomaly_id += 1;
        anomaly
    }

    /// Create a convergent discovery anomaly (positive)
    pub fn create_convergent_discovery_anomaly(
        &mut self,
        pattern_id: PatternId,
        discoverers: Vec<SymthaeaId>,
        timestamp: u64,
    ) -> Anomaly {
        let anomaly = Anomaly::new(
            self.next_anomaly_id,
            AnomalyType::ConvergentDiscovery,
            AnomalySeverity::Info,
            &format!(
                "Pattern {} independently discovered by {} agents",
                pattern_id,
                discoverers.len()
            ),
            timestamp,
        )
        .with_patterns(vec![pattern_id])
        .with_agents(discoverers)
        .with_actions(vec![
            "Consider promoting this pattern".to_string(),
            "Document the pattern well".to_string(),
        ]);

        self.next_anomaly_id += 1;
        anomaly
    }

    /// Add an anomaly to the registry
    pub fn add_anomaly(&mut self, anomaly: Anomaly) {
        // Update stats
        self.stats.total_detected += 1;
        *self
            .stats
            .by_type
            .entry(anomaly.anomaly_type.name().to_string())
            .or_insert(0) += 1;
        *self
            .stats
            .by_severity
            .entry(format!("{:?}", anomaly.severity))
            .or_insert(0) += 1;

        self.anomalies.push(anomaly);
    }

    /// Get all anomalies
    pub fn all_anomalies(&self) -> &[Anomaly] {
        &self.anomalies
    }

    /// Get active anomalies
    pub fn active_anomalies(&self) -> Vec<&Anomaly> {
        self.anomalies.iter().filter(|a| a.is_active()).collect()
    }

    /// Get anomalies by type
    pub fn anomalies_by_type(&self, anomaly_type: &AnomalyType) -> Vec<&Anomaly> {
        self.anomalies
            .iter()
            .filter(|a| &a.anomaly_type == anomaly_type)
            .collect()
    }

    /// Get anomalies by severity
    pub fn anomalies_by_severity(&self, min_severity: AnomalySeverity) -> Vec<&Anomaly> {
        self.anomalies
            .iter()
            .filter(|a| a.severity >= min_severity)
            .collect()
    }

    /// Get anomalies affecting a pattern
    pub fn anomalies_for_pattern(&self, pattern_id: PatternId) -> Vec<&Anomaly> {
        self.anomalies
            .iter()
            .filter(|a| a.affected_patterns.contains(&pattern_id))
            .collect()
    }

    /// Get anomaly by ID
    pub fn get_anomaly(&self, anomaly_id: u64) -> Option<&Anomaly> {
        self.anomalies.iter().find(|a| a.anomaly_id == anomaly_id)
    }

    /// Get mutable anomaly by ID
    pub fn get_anomaly_mut(&mut self, anomaly_id: u64) -> Option<&mut Anomaly> {
        self.anomalies
            .iter_mut()
            .find(|a| a.anomaly_id == anomaly_id)
    }

    /// Acknowledge an anomaly
    pub fn acknowledge_anomaly(&mut self, anomaly_id: u64) -> bool {
        if let Some(anomaly) = self.get_anomaly_mut(anomaly_id) {
            anomaly.acknowledge();
            true
        } else {
            false
        }
    }

    /// Resolve an anomaly
    pub fn resolve_anomaly(&mut self, anomaly_id: u64, notes: &str) -> bool {
        if let Some(anomaly) = self.get_anomaly_mut(anomaly_id) {
            anomaly.resolve(notes);
            true
        } else {
            false
        }
    }

    /// Get statistics
    pub fn stats(&self) -> AnomalyStats {
        let mut stats = self.stats.clone();
        stats.active_count = self.active_anomalies().len();
        stats
    }

    /// Generate a summary report
    pub fn summary_report(&self) -> String {
        let active = self.active_anomalies();
        let mut report = String::new();

        report.push_str("=== Anomaly Summary ===\n\n");
        report.push_str(&format!("Total detected: {}\n", self.stats.total_detected));
        report.push_str(&format!("Currently active: {}\n\n", active.len()));

        if active.is_empty() {
            report.push_str("No active anomalies.\n");
        } else {
            report.push_str("Active Anomalies:\n");
            for anomaly in active {
                report.push_str(&format!("  {}\n", anomaly.summary()));
            }
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_type_properties() {
        assert!(AnomalyType::SuccessRateDrop.is_typically_negative());
        assert!(AnomalyType::RapidAdoption.is_typically_positive());
        assert!(AnomalyType::UsageSpike.is_neutral());
    }

    #[test]
    fn test_anomaly_severity_ordering() {
        assert!(AnomalySeverity::Critical > AnomalySeverity::High);
        assert!(AnomalySeverity::High > AnomalySeverity::Medium);
        assert!(AnomalySeverity::Medium > AnomalySeverity::Low);
        assert!(AnomalySeverity::Low > AnomalySeverity::Info);
    }

    #[test]
    fn test_anomaly_severity_from_magnitude() {
        assert_eq!(AnomalySeverity::from_magnitude(0.05), AnomalySeverity::Info);
        assert_eq!(AnomalySeverity::from_magnitude(0.15), AnomalySeverity::Low);
        assert_eq!(
            AnomalySeverity::from_magnitude(0.35),
            AnomalySeverity::Medium
        );
        assert_eq!(AnomalySeverity::from_magnitude(0.6), AnomalySeverity::High);
        assert_eq!(
            AnomalySeverity::from_magnitude(0.9),
            AnomalySeverity::Critical
        );
    }

    #[test]
    fn test_anomaly_creation() {
        let anomaly = Anomaly::new(
            1,
            AnomalyType::SuccessRateDrop,
            AnomalySeverity::Medium,
            "Test anomaly",
            1000,
        )
        .with_patterns(vec![42])
        .with_magnitude(0.3);

        assert_eq!(anomaly.anomaly_id, 1);
        assert_eq!(anomaly.affected_patterns, vec![42]);
        assert_eq!(anomaly.magnitude, 0.3);
        assert!(anomaly.is_active());
    }

    #[test]
    fn test_anomaly_lifecycle() {
        let mut anomaly = Anomaly::new(
            1,
            AnomalyType::SuccessRateDrop,
            AnomalySeverity::Medium,
            "Test",
            1000,
        );

        assert!(!anomaly.acknowledged);
        assert!(!anomaly.resolved);
        assert!(anomaly.is_active());

        anomaly.acknowledge();
        assert!(anomaly.acknowledged);
        assert!(anomaly.is_active());

        anomaly.resolve("Fixed the issue");
        assert!(anomaly.resolved);
        assert!(!anomaly.is_active());
        assert_eq!(
            anomaly.resolution_notes,
            Some("Fixed the issue".to_string())
        );
    }

    #[test]
    fn test_time_series_operations() {
        let mut series = TimeSeries::new(10);

        series.add(100, 0.5);
        series.add(200, 0.6);
        series.add(300, 0.7);
        series.add(400, 0.8);

        assert_eq!(series.latest(), Some(0.8));
        assert!((series.average().unwrap() - 0.65).abs() < 0.01);
        assert!(series.trend().unwrap() > 0.0); // Increasing
    }

    #[test]
    fn test_time_series_volatility() {
        let mut series = TimeSeries::new(10);

        // Constant values = low volatility
        series.add(100, 0.5);
        series.add(200, 0.5);
        series.add(300, 0.5);

        let vol = series.volatility().unwrap();
        assert!(vol < 0.01);

        // High variance = high volatility
        let mut series2 = TimeSeries::new(10);
        series2.add(100, 0.1);
        series2.add(200, 0.9);
        series2.add(300, 0.2);
        series2.add(400, 0.8);

        let vol2 = series2.volatility().unwrap();
        assert!(vol2 > 0.3);
    }

    #[test]
    fn test_anomaly_detector_creation() {
        let detector = AnomalyDetector::new();
        assert_eq!(detector.all_anomalies().len(), 0);
    }

    #[test]
    fn test_anomaly_detector_record_and_detect() {
        let mut detector = AnomalyDetector::new();

        // Record stable success rates
        detector.record_success_rate(1, 0.8, 100);
        detector.record_success_rate(1, 0.79, 200);
        detector.record_success_rate(1, 0.81, 300);

        // Should not detect anomaly
        let anomaly = detector.detect_success_rate_anomaly(1, 400);
        assert!(anomaly.is_none());

        // Record sudden drop
        detector.record_success_rate(1, 0.4, 400);

        // Should detect anomaly
        let anomaly = detector.detect_success_rate_anomaly(1, 500);
        assert!(anomaly.is_some());
        let anomaly = anomaly.unwrap();
        assert_eq!(anomaly.anomaly_type, AnomalyType::SuccessRateDrop);
    }

    #[test]
    fn test_anomaly_detector_orphaned_dependency() {
        let mut detector = AnomalyDetector::new();

        let anomaly = detector.create_orphaned_dependency_anomaly(1, 2, 1000);
        detector.add_anomaly(anomaly);

        assert_eq!(detector.all_anomalies().len(), 1);
        assert_eq!(detector.active_anomalies().len(), 1);

        let anomalies = detector.anomalies_for_pattern(1);
        assert_eq!(anomalies.len(), 1);
    }

    #[test]
    fn test_anomaly_detector_echo_chamber() {
        let mut detector = AnomalyDetector::new();

        let anomaly = detector.create_echo_chamber_anomaly(vec![1, 2, 3], 0.97, 1000);
        assert_eq!(anomaly.anomaly_type, AnomalyType::EchoChamber);
        assert!(anomaly.magnitude > 0.9);
    }

    #[test]
    fn test_anomaly_config_presets() {
        let sensitive = AnomalyConfig::sensitive();
        let relaxed = AnomalyConfig::relaxed();

        assert!(sensitive.success_rate_drop_threshold < relaxed.success_rate_drop_threshold);
        assert!(sensitive.min_severity < relaxed.min_severity);
    }

    #[test]
    fn test_anomaly_summary_report() {
        let mut detector = AnomalyDetector::new();

        let anomaly = detector.create_orphaned_dependency_anomaly(1, 2, 1000);
        detector.add_anomaly(anomaly);

        let report = detector.summary_report();
        assert!(report.contains("Anomaly Summary"));
        assert!(report.contains("Currently active: 1"));
    }
}
