// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Defense Common - Shared utilities for Byzantine defense zomes
//!
//! This module provides:
//! - Comprehensive error types for all failure modes
//! - Structured logging for production monitoring
//! - Performance metrics and health tracking
//! - Runtime configuration management
//!
//! Author: Luminous Dynamics Research Team
//! Date: December 2025

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use thiserror::Error;

// =============================================================================
// Error Types
// =============================================================================

/// Comprehensive error types for Byzantine defense operations
#[derive(Error, Debug, Clone, Serialize, Deserialize)]
pub enum DefenseError {
    // Input validation errors
    #[error("Empty gradient list: at least one gradient required")]
    EmptyGradientList,

    #[error("Gradient dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    #[error("Invalid gradient values: contains NaN or Inf")]
    InvalidGradientValues,

    #[error("Invalid round number: {0} (must be positive)")]
    InvalidRoundNumber(u64),

    #[error("Invalid node ID: {reason}")]
    InvalidNodeId { reason: String },

    // Configuration errors
    #[error("Invalid z-score threshold: {value} (must be in range 0.5-10.0)")]
    InvalidZThreshold { value: f64 },

    #[error("Invalid reputation alpha: {value} (must be in range 0.01-0.5)")]
    InvalidReputationAlpha { value: f64 },

    #[error("Invalid Krum parameter k: {k} (must be >= 1 and <= n-f)")]
    InvalidKrumK { k: usize },

    #[error("Invalid trim ratio: {value} (must be in range 0.0-0.45)")]
    InvalidTrimRatio { value: f64 },

    // Byzantine detection errors
    #[error("Byzantine node detected: {node_id} (z-score: {z_score:.2}, threshold: {threshold:.2})")]
    ByzantineNodeDetected {
        node_id: String,
        z_score: f64,
        threshold: f64,
    },

    #[error("Too many Byzantine nodes: {count} of {total} ({percentage:.1}% > {max_percentage:.1}%)")]
    TooManyByzantineNodes {
        count: usize,
        total: usize,
        percentage: f64,
        max_percentage: f64,
    },

    #[error("Reputation blacklisted: node {node_id} has reputation {reputation:.3} < {threshold:.3}")]
    ReputationBlacklisted {
        node_id: String,
        reputation: f64,
        threshold: f64,
    },

    // Aggregation errors
    #[error("Aggregation failed: insufficient honest nodes ({count} < {min_required})")]
    InsufficientHonestNodes { count: usize, min_required: usize },

    #[error("Aggregation quality too low: {quality:.3} < {threshold:.3}")]
    LowAggregationQuality { quality: f64, threshold: f64 },

    // System errors
    #[error("Fixed-point overflow: {operation}")]
    FixedPointOverflow { operation: String },

    #[error("DHT operation failed: {reason}")]
    DhtOperationFailed { reason: String },

    #[error("Cross-zome call failed: {zome} -> {function}: {reason}")]
    CrossZomeCallFailed {
        zome: String,
        function: String,
        reason: String,
    },

    #[error("Timeout waiting for {operation} (elapsed: {elapsed_ms}ms, limit: {limit_ms}ms)")]
    Timeout {
        operation: String,
        elapsed_ms: u64,
        limit_ms: u64,
    },

    // Internal errors
    #[error("Internal error: {message}")]
    InternalError { message: String },
}

impl DefenseError {
    /// Create a new dimension mismatch error
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        DefenseError::DimensionMismatch { expected, actual }
    }

    /// Create a new Byzantine node detected error
    pub fn byzantine_detected(node_id: impl Into<String>, z_score: f64, threshold: f64) -> Self {
        DefenseError::ByzantineNodeDetected {
            node_id: node_id.into(),
            z_score,
            threshold,
        }
    }

    /// Create a new cross-zome call failed error
    pub fn cross_zome_failed(
        zome: impl Into<String>,
        function: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        DefenseError::CrossZomeCallFailed {
            zome: zome.into(),
            function: function.into(),
            reason: reason.into(),
        }
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        match self {
            DefenseError::DhtOperationFailed { .. } => true,
            DefenseError::CrossZomeCallFailed { .. } => true,
            DefenseError::Timeout { .. } => true,
            _ => false,
        }
    }

    /// Get error severity level (1-5, with 5 being most severe)
    pub fn severity(&self) -> u8 {
        match self {
            // Level 1: Informational (expected conditions)
            DefenseError::ByzantineNodeDetected { .. } => 1,
            DefenseError::ReputationBlacklisted { .. } => 1,

            // Level 2: Warning (unusual but handled)
            DefenseError::TooManyByzantineNodes { .. } => 2,
            DefenseError::LowAggregationQuality { .. } => 2,

            // Level 3: Error (requires attention)
            DefenseError::EmptyGradientList => 3,
            DefenseError::DimensionMismatch { .. } => 3,
            DefenseError::InsufficientHonestNodes { .. } => 3,

            // Level 4: Severe (configuration/system issue)
            DefenseError::InvalidZThreshold { .. } => 4,
            DefenseError::InvalidReputationAlpha { .. } => 4,
            DefenseError::DhtOperationFailed { .. } => 4,
            DefenseError::CrossZomeCallFailed { .. } => 4,
            DefenseError::Timeout { .. } => 4,

            // Level 5: Critical (should not happen)
            DefenseError::FixedPointOverflow { .. } => 5,
            DefenseError::InvalidGradientValues => 5,
            DefenseError::InternalError { .. } => 5,

            _ => 3,
        }
    }
}

/// Result type alias for defense operations
pub type DefenseResult<T> = Result<T, DefenseError>;

// =============================================================================
// Logging
// =============================================================================

/// Log level enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum LogLevel {
    Trace = 0,
    Debug = 1,
    #[default]
    Info = 2,
    Warn = 3,
    Error = 4,
    Critical = 5,
}

impl std::fmt::Display for LogLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LogLevel::Trace => write!(f, "TRACE"),
            LogLevel::Debug => write!(f, "DEBUG"),
            LogLevel::Info => write!(f, "INFO"),
            LogLevel::Warn => write!(f, "WARN"),
            LogLevel::Error => write!(f, "ERROR"),
            LogLevel::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Structured log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Timestamp (round number in FL context)
    pub round: u64,
    /// Log level
    pub level: LogLevel,
    /// Source component
    pub component: String,
    /// Operation being performed
    pub operation: String,
    /// Log message
    pub message: String,
    /// Additional structured data
    pub data: HashMap<String, String>,
}

impl LogEntry {
    /// Create a new log entry
    pub fn new(
        round: u64,
        level: LogLevel,
        component: impl Into<String>,
        operation: impl Into<String>,
        message: impl Into<String>,
    ) -> Self {
        LogEntry {
            round,
            level,
            component: component.into(),
            operation: operation.into(),
            message: message.into(),
            data: HashMap::new(),
        }
    }

    /// Add a key-value pair to the log data
    pub fn with_data(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }

    /// Format as a log line
    pub fn format(&self) -> String {
        let data_str = if self.data.is_empty() {
            String::new()
        } else {
            let pairs: Vec<String> = self
                .data
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect();
            format!(" [{}]", pairs.join(", "))
        };

        format!(
            "[R{:06}] {} [{}::{}] {}{}",
            self.round, self.level, self.component, self.operation, self.message, data_str
        )
    }
}

/// Logger trait for defense components
pub trait DefenseLogger {
    fn log(&self, entry: LogEntry);

    fn trace(&self, round: u64, component: &str, op: &str, msg: &str) {
        self.log(LogEntry::new(round, LogLevel::Trace, component, op, msg));
    }

    fn debug(&self, round: u64, component: &str, op: &str, msg: &str) {
        self.log(LogEntry::new(round, LogLevel::Debug, component, op, msg));
    }

    fn info(&self, round: u64, component: &str, op: &str, msg: &str) {
        self.log(LogEntry::new(round, LogLevel::Info, component, op, msg));
    }

    fn warn(&self, round: u64, component: &str, op: &str, msg: &str) {
        self.log(LogEntry::new(round, LogLevel::Warn, component, op, msg));
    }

    fn error(&self, round: u64, component: &str, op: &str, msg: &str) {
        self.log(LogEntry::new(round, LogLevel::Error, component, op, msg));
    }
}

/// Simple in-memory logger (for testing and development)
#[derive(Default)]
pub struct MemoryLogger {
    min_level: LogLevel,
    entries: std::sync::Mutex<Vec<LogEntry>>,
}

impl MemoryLogger {
    pub fn new(min_level: LogLevel) -> Self {
        MemoryLogger {
            min_level,
            entries: std::sync::Mutex::new(Vec::new()),
        }
    }

    pub fn get_entries(&self) -> Vec<LogEntry> {
        self.entries.lock().expect("logger mutex poisoned").clone()
    }

    pub fn clear(&self) {
        self.entries.lock().expect("logger mutex poisoned").clear();
    }
}

impl DefenseLogger for MemoryLogger {
    fn log(&self, entry: LogEntry) {
        if entry.level >= self.min_level {
            self.entries.lock().expect("logger mutex poisoned").push(entry);
        }
    }
}

// =============================================================================
// Metrics
// =============================================================================

/// Performance metrics for defense operations
#[derive(Debug, Default)]
pub struct DefenseMetrics {
    // Detection metrics
    pub rounds_processed: AtomicU64,
    pub nodes_analyzed: AtomicU64,
    pub byzantine_detected: AtomicU64,
    pub false_positives: AtomicU64,

    // Aggregation metrics
    pub aggregations_performed: AtomicU64,
    pub gradients_aggregated: AtomicU64,
    pub gradients_excluded: AtomicU64,

    // Reputation metrics
    pub reputation_updates: AtomicU64,
    pub nodes_blacklisted: AtomicU64,
    pub sleeper_agents_detected: AtomicU64,

    // Performance metrics
    pub total_detection_time_us: AtomicU64,
    pub total_aggregation_time_us: AtomicU64,

    // Error metrics
    pub errors_recoverable: AtomicU64,
    pub errors_fatal: AtomicU64,
}

impl DefenseMetrics {
    /// Create new metrics instance
    pub fn new() -> Self {
        DefenseMetrics::default()
    }

    /// Record a round being processed
    pub fn record_round(&self, nodes: usize, byzantine: usize) {
        self.rounds_processed.fetch_add(1, Ordering::Relaxed);
        self.nodes_analyzed.fetch_add(nodes as u64, Ordering::Relaxed);
        self.byzantine_detected.fetch_add(byzantine as u64, Ordering::Relaxed);
    }

    /// Record an aggregation
    pub fn record_aggregation(&self, used: usize, excluded: usize) {
        self.aggregations_performed.fetch_add(1, Ordering::Relaxed);
        self.gradients_aggregated.fetch_add(used as u64, Ordering::Relaxed);
        self.gradients_excluded.fetch_add(excluded as u64, Ordering::Relaxed);
    }

    /// Record detection time in microseconds
    pub fn record_detection_time(&self, us: u64) {
        self.total_detection_time_us.fetch_add(us, Ordering::Relaxed);
    }

    /// Record an error
    pub fn record_error(&self, recoverable: bool) {
        if recoverable {
            self.errors_recoverable.fetch_add(1, Ordering::Relaxed);
        } else {
            self.errors_fatal.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Get snapshot of all metrics
    pub fn snapshot(&self) -> MetricsSnapshot {
        MetricsSnapshot {
            rounds_processed: self.rounds_processed.load(Ordering::Relaxed),
            nodes_analyzed: self.nodes_analyzed.load(Ordering::Relaxed),
            byzantine_detected: self.byzantine_detected.load(Ordering::Relaxed),
            false_positives: self.false_positives.load(Ordering::Relaxed),
            aggregations_performed: self.aggregations_performed.load(Ordering::Relaxed),
            gradients_aggregated: self.gradients_aggregated.load(Ordering::Relaxed),
            gradients_excluded: self.gradients_excluded.load(Ordering::Relaxed),
            reputation_updates: self.reputation_updates.load(Ordering::Relaxed),
            nodes_blacklisted: self.nodes_blacklisted.load(Ordering::Relaxed),
            sleeper_agents_detected: self.sleeper_agents_detected.load(Ordering::Relaxed),
            total_detection_time_us: self.total_detection_time_us.load(Ordering::Relaxed),
            total_aggregation_time_us: self.total_aggregation_time_us.load(Ordering::Relaxed),
            errors_recoverable: self.errors_recoverable.load(Ordering::Relaxed),
            errors_fatal: self.errors_fatal.load(Ordering::Relaxed),
        }
    }
}

/// Serializable metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSnapshot {
    pub rounds_processed: u64,
    pub nodes_analyzed: u64,
    pub byzantine_detected: u64,
    pub false_positives: u64,
    pub aggregations_performed: u64,
    pub gradients_aggregated: u64,
    pub gradients_excluded: u64,
    pub reputation_updates: u64,
    pub nodes_blacklisted: u64,
    pub sleeper_agents_detected: u64,
    pub total_detection_time_us: u64,
    pub total_aggregation_time_us: u64,
    pub errors_recoverable: u64,
    pub errors_fatal: u64,
}

impl MetricsSnapshot {
    /// Calculate derived metrics
    pub fn detection_rate(&self) -> f64 {
        if self.nodes_analyzed == 0 {
            0.0
        } else {
            self.byzantine_detected as f64 / self.nodes_analyzed as f64
        }
    }

    pub fn exclusion_rate(&self) -> f64 {
        let total = self.gradients_aggregated + self.gradients_excluded;
        if total == 0 {
            0.0
        } else {
            self.gradients_excluded as f64 / total as f64
        }
    }

    pub fn avg_detection_time_us(&self) -> f64 {
        if self.rounds_processed == 0 {
            0.0
        } else {
            self.total_detection_time_us as f64 / self.rounds_processed as f64
        }
    }

    pub fn error_rate(&self) -> f64 {
        let total_ops = self.rounds_processed + self.aggregations_performed;
        if total_ops == 0 {
            0.0
        } else {
            (self.errors_recoverable + self.errors_fatal) as f64 / total_ops as f64
        }
    }
}

// =============================================================================
// Configuration
// =============================================================================

/// Defense configuration with validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefenseConfig {
    // Statistical detection (Layer 1)
    /// Z-score threshold for Byzantine detection (default: 2.5)
    pub z_threshold: f64,
    /// Whether to use MAD (true) or standard deviation (false)
    pub use_mad: bool,

    // Reputation (Layer 2)
    /// EMA alpha for reputation updates (default: 0.2)
    pub reputation_alpha: f64,
    /// Minimum reputation to participate (default: 0.3)
    pub min_reputation: f64,
    /// Sleeper agent detection z-score threshold (default: 2.0)
    pub sleeper_threshold: f64,

    // Aggregation (Layer 3)
    /// Default aggregation method
    pub default_method: AggregationMethodConfig,
    /// Number of gradients to select for Multi-Krum (default: auto)
    pub krum_k: Option<usize>,
    /// Trim ratio for trimmed mean (default: 0.1)
    pub trim_ratio: f64,

    // System
    /// Maximum Byzantine ratio before failing (default: 0.45)
    pub max_byzantine_ratio: f64,
    /// Minimum number of honest nodes required (default: 3)
    pub min_honest_nodes: usize,
    /// Operation timeout in milliseconds (default: 5000)
    pub timeout_ms: u64,

    // Logging
    /// Minimum log level
    pub log_level: LogLevel,
    /// Whether to log individual node analyses
    pub log_node_details: bool,
}

/// Aggregation method for configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum AggregationMethodConfig {
    Mean,
    CoordinateMedian,
    Krum,
    MultiKrum,
    TrimmedMean,
}

impl Default for DefenseConfig {
    fn default() -> Self {
        DefenseConfig {
            // Layer 1: Statistical
            z_threshold: 2.5,
            use_mad: true,

            // Layer 2: Reputation
            reputation_alpha: 0.2,
            min_reputation: 0.3,
            sleeper_threshold: 2.0,

            // Layer 3: Aggregation
            default_method: AggregationMethodConfig::MultiKrum,
            krum_k: None, // Auto-calculate
            trim_ratio: 0.1,

            // System
            max_byzantine_ratio: 0.45,
            min_honest_nodes: 3,
            timeout_ms: 5000,

            // Logging
            log_level: LogLevel::Info,
            log_node_details: false,
        }
    }
}

impl DefenseConfig {
    /// Validate configuration
    pub fn validate(&self) -> DefenseResult<()> {
        // Z-threshold validation
        if self.z_threshold < 0.5 || self.z_threshold > 10.0 {
            return Err(DefenseError::InvalidZThreshold {
                value: self.z_threshold,
            });
        }

        // Reputation alpha validation
        if self.reputation_alpha < 0.01 || self.reputation_alpha > 0.5 {
            return Err(DefenseError::InvalidReputationAlpha {
                value: self.reputation_alpha,
            });
        }

        // Trim ratio validation
        if self.trim_ratio < 0.0 || self.trim_ratio > 0.45 {
            return Err(DefenseError::InvalidTrimRatio {
                value: self.trim_ratio,
            });
        }

        Ok(())
    }

    /// Create a strict configuration (more aggressive detection)
    pub fn strict() -> Self {
        DefenseConfig {
            z_threshold: 2.0,
            reputation_alpha: 0.3,
            min_reputation: 0.4,
            ..Default::default()
        }
    }

    /// Create a permissive configuration (fewer false positives)
    pub fn permissive() -> Self {
        DefenseConfig {
            z_threshold: 3.0,
            reputation_alpha: 0.1,
            min_reputation: 0.2,
            ..Default::default()
        }
    }

    /// Calculate Krum k parameter based on number of nodes
    pub fn compute_krum_k(&self, n: usize) -> usize {
        if let Some(k) = self.krum_k {
            k.min(n)
        } else {
            // Auto-calculate: k = n - f - 1, where f = floor(0.3 * n)
            let f = (n as f64 * 0.3).floor() as usize;
            let k = n.saturating_sub(f).saturating_sub(1);
            k.max(1)
        }
    }
}

// =============================================================================
// Health Check
// =============================================================================

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatus {
    /// Overall health (0.0 - 1.0)
    pub health_score: f64,
    /// Whether system is operational
    pub is_healthy: bool,
    /// Individual component statuses
    pub components: HashMap<String, ComponentHealth>,
    /// Current metrics snapshot
    pub metrics: MetricsSnapshot,
    /// Warnings
    pub warnings: Vec<String>,
}

/// Individual component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub name: String,
    pub status: ComponentStatus,
    pub message: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ComponentStatus {
    Healthy,
    Degraded,
    Unhealthy,
}

impl HealthStatus {
    /// Create health status from metrics
    pub fn from_metrics(metrics: &DefenseMetrics, config: &DefenseConfig) -> Self {
        let snapshot = metrics.snapshot();
        let mut warnings = Vec::new();
        let mut components = HashMap::new();

        // Check detection rate
        let detection_rate = snapshot.detection_rate();
        let detection_status = if detection_rate > config.max_byzantine_ratio {
            warnings.push(format!(
                "High Byzantine rate: {:.1}% detected",
                detection_rate * 100.0
            ));
            ComponentHealth {
                name: "detection".to_string(),
                status: ComponentStatus::Degraded,
                message: format!("{:.1}% Byzantine detected", detection_rate * 100.0),
            }
        } else {
            ComponentHealth {
                name: "detection".to_string(),
                status: ComponentStatus::Healthy,
                message: format!("{:.1}% Byzantine detected", detection_rate * 100.0),
            }
        };
        components.insert("detection".to_string(), detection_status);

        // Check error rate
        let error_rate = snapshot.error_rate();
        let error_status = if error_rate > 0.05 {
            warnings.push(format!("High error rate: {:.1}%", error_rate * 100.0));
            ComponentHealth {
                name: "errors".to_string(),
                status: ComponentStatus::Unhealthy,
                message: format!("{:.1}% error rate", error_rate * 100.0),
            }
        } else if error_rate > 0.01 {
            ComponentHealth {
                name: "errors".to_string(),
                status: ComponentStatus::Degraded,
                message: format!("{:.1}% error rate", error_rate * 100.0),
            }
        } else {
            ComponentHealth {
                name: "errors".to_string(),
                status: ComponentStatus::Healthy,
                message: format!("{:.1}% error rate", error_rate * 100.0),
            }
        };
        components.insert("errors".to_string(), error_status);

        // Calculate overall health score
        let health_score: f64 = {
            let mut score: f64 = 1.0;
            for component in components.values() {
                match component.status {
                    ComponentStatus::Healthy => {}
                    ComponentStatus::Degraded => score -= 0.2,
                    ComponentStatus::Unhealthy => score -= 0.5,
                }
            }
            if score < 0.0 { 0.0 } else { score }
        };

        let is_healthy = components.values().all(|c| c.status != ComponentStatus::Unhealthy);

        HealthStatus {
            health_score,
            is_healthy,
            components,
            metrics: snapshot,
            warnings,
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_severity() {
        assert_eq!(
            DefenseError::ByzantineNodeDetected {
                node_id: "test".to_string(),
                z_score: 3.0,
                threshold: 2.5
            }
            .severity(),
            1
        );

        assert_eq!(DefenseError::EmptyGradientList.severity(), 3);

        assert_eq!(
            DefenseError::FixedPointOverflow {
                operation: "mul".to_string()
            }
            .severity(),
            5
        );
    }

    #[test]
    fn test_config_validation() {
        let valid = DefenseConfig::default();
        assert!(valid.validate().is_ok());

        let invalid_z = DefenseConfig {
            z_threshold: 0.1,
            ..Default::default()
        };
        assert!(invalid_z.validate().is_err());

        let invalid_alpha = DefenseConfig {
            reputation_alpha: 0.001,
            ..Default::default()
        };
        assert!(invalid_alpha.validate().is_err());
    }

    #[test]
    fn test_krum_k_calculation() {
        let config = DefenseConfig::default();

        // n=10: f=3, k=10-3-1=6
        assert_eq!(config.compute_krum_k(10), 6);

        // n=5: f=1, k=5-1-1=3
        assert_eq!(config.compute_krum_k(5), 3);

        // n=3: f=0, k=3-0-1=2
        assert_eq!(config.compute_krum_k(3), 2);
    }

    #[test]
    fn test_metrics() {
        let metrics = DefenseMetrics::new();

        metrics.record_round(10, 3);
        metrics.record_aggregation(7, 3);
        metrics.record_detection_time(1000);

        let snapshot = metrics.snapshot();
        assert_eq!(snapshot.rounds_processed, 1);
        assert_eq!(snapshot.nodes_analyzed, 10);
        assert_eq!(snapshot.byzantine_detected, 3);
        assert_eq!(snapshot.gradients_aggregated, 7);
        assert_eq!(snapshot.gradients_excluded, 3);

        assert!((snapshot.detection_rate() - 0.3).abs() < 0.01);
    }

    #[test]
    fn test_logger() {
        let logger = MemoryLogger::new(LogLevel::Info);

        logger.debug(1, "test", "op", "debug message");
        logger.info(1, "test", "op", "info message");
        logger.warn(1, "test", "op", "warn message");

        let entries = logger.get_entries();
        // Debug should be filtered out
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].level, LogLevel::Info);
        assert_eq!(entries[1].level, LogLevel::Warn);
    }

    #[test]
    fn test_log_entry_format() {
        let entry = LogEntry::new(42, LogLevel::Info, "defense", "detect", "Found Byzantine node")
            .with_data("node_id", "abc123")
            .with_data("z_score", "3.14");

        let formatted = entry.format();
        assert!(formatted.contains("[R000042]"));
        assert!(formatted.contains("INFO"));
        assert!(formatted.contains("defense::detect"));
        assert!(formatted.contains("Found Byzantine node"));
    }

    #[test]
    fn test_health_status() {
        let metrics = DefenseMetrics::new();
        let config = DefenseConfig::default();

        // Healthy state
        metrics.record_round(100, 10);
        metrics.record_aggregation(90, 10);

        let health = HealthStatus::from_metrics(&metrics, &config);
        assert!(health.is_healthy);
        assert!(health.health_score > 0.8);
    }
}
