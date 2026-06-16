// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Core Infrastructure for Recursive Improvement
//!
//! Foundational types for monitoring, identifying bottlenecks, and
//! orchestrating self-improvement.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

/// Unique identifier for a system component
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComponentId(pub String);

impl ComponentId {
    /// Create a new component ID
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the ID as a string reference
    pub fn as_str(&self) -> &str {
        &self.0
    }

    // Well-known component identifiers used across the system

    pub fn primitive_evolution() -> Self {
        Self("PrimitiveEvolution".into())
    }
    pub fn hrm() -> Self {
        Self("HRM".into())
    }
    pub fn meta_cognition() -> Self {
        Self("MetaCognition".into())
    }
    pub fn byzantine_collective() -> Self {
        Self("ByzantineCollective".into())
    }
    pub fn meta_learning() -> Self {
        Self("MetaLearning".into())
    }
    pub fn causal_defense() -> Self {
        Self("CausalDefense".into())
    }
    pub fn unified_intelligence() -> Self {
        Self("UnifiedIntelligence".into())
    }
    pub fn collective_sharing() -> Self {
        Self("CollectiveSharing".into())
    }
    pub fn cache() -> Self {
        Self("Cache".into())
    }
    pub fn integration() -> Self {
        Self("Integration".into())
    }
}

impl From<&str> for ComponentId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for ComponentId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl std::fmt::Display for ComponentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BottleneckType {
    /// Computation taking too long
    Latency,
    /// Memory usage too high
    Memory,
    /// Throughput too low
    Throughput,
    /// Accuracy below threshold
    Accuracy,
    /// Energy consumption too high
    Energy,
    /// Integration failure between components
    Integration,
    /// Resource contention
    Contention,
    /// General computation bottleneck
    Computation,
    /// I/O bound bottleneck
    IO,
    /// Low accuracy detected
    LowAccuracy,
    /// Low Phi (integrated information) detected
    LowPhi,
    /// Oscillation in metrics
    Oscillation,
    /// Phi stagnation (not improving)
    PhiStagnation,
    /// Resource exhaustion
    ResourceExhaustion,
}

/// A detected bottleneck in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Which component is bottlenecked
    pub component_id: ComponentId,

    /// Alias for component_id (backward compatibility)
    pub id: ComponentId,

    /// Alias for component_id (backward compatibility)
    pub component: ComponentId,

    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,

    /// Severity (0.0 = minor, 1.0 = critical)
    pub severity: f32,

    /// Description of the bottleneck
    pub description: String,

    /// Suggested improvements
    pub suggestions: Vec<String>,

    /// When detected
    pub detected_at: u64,

    /// Evidence/metrics supporting this bottleneck
    pub evidence: HashMap<String, f64>,
}

impl Bottleneck {
    /// Create a new bottleneck
    pub fn new(
        component_id: impl Into<ComponentId>,
        bottleneck_type: BottleneckType,
        severity: f32,
        description: impl Into<String>,
    ) -> Self {
        let cid: ComponentId = component_id.into();
        Self {
            id: cid.clone(),
            component: cid.clone(),
            component_id: cid,
            bottleneck_type,
            severity: severity.clamp(0.0, 1.0),
            description: description.into(),
            suggestions: Vec::new(),
            detected_at: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis() as u64,
            evidence: HashMap::new(),
        }
    }

    /// Add a suggestion
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestions.push(suggestion.into());
        self
    }

    /// Add evidence
    pub fn with_evidence(mut self, key: impl Into<String>, value: f64) -> Self {
        self.evidence.insert(key.into(), value);
        self
    }
}

/// Types of improvements that can be applied
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImprovementType {
    /// Optimize algorithm/implementation
    Optimization,
    /// Add caching layer
    Caching,
    /// Parallelize computation
    Parallelization,
    /// Reduce memory footprint
    MemoryReduction,
    /// Improve accuracy/quality
    AccuracyImprovement,
    /// Add batching
    Batching,
    /// Restructure architecture
    Restructure,
    /// Add redundancy
    Redundancy,
    /// Tune hyperparameters
    HyperparameterTuning,
    // Extended variants used by improvement_generator and gradient_optimizer
    /// Increase cache size
    IncreaseCacheSize { from: usize, to: usize },
    /// Parallelize (alias for Parallelization used in some modules)
    Parallelize {
        component: ComponentId,
        threads: usize,
    },
    /// Increase evolution rate for adaptive systems
    IncreaseEvolutionRate,
    /// Add synthetic training data
    AddSyntheticData { count: usize },
    /// Optimize specific algorithm
    OptimizeAlgorithm {
        component: ComponentId,
        optimization: String,
    },
    /// Tune a specific hyperparameter
    TuneHyperparameter {
        name: String,
        old_value: f64,
        new_value: f64,
    },
}

/// Configuration for the performance monitor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// How often to sample metrics (ms)
    pub sample_interval_ms: u64,

    /// How many samples to keep in history
    pub history_size: usize,

    /// Latency threshold for bottleneck detection (ms)
    pub latency_threshold_ms: f64,

    /// Memory threshold for bottleneck detection (bytes)
    pub memory_threshold_bytes: u64,

    /// Accuracy threshold for bottleneck detection (0-1)
    pub accuracy_threshold: f32,

    /// Whether to auto-detect bottlenecks
    pub auto_detect: bool,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            sample_interval_ms: 100,
            history_size: 1000,
            latency_threshold_ms: 100.0,
            memory_threshold_bytes: 1024 * 1024 * 1024, // 1 GB
            accuracy_threshold: 0.9,
            auto_detect: true,
        }
    }
}

/// Performance metrics for a component
#[derive(Debug, Clone, Default)]
pub struct ComponentMetrics {
    /// Latency samples (ms)
    pub latencies: VecDeque<f64>,

    /// Memory usage samples (bytes)
    pub memory_usage: VecDeque<u64>,

    /// Throughput samples (ops/sec)
    pub throughput: VecDeque<f64>,

    /// Accuracy samples (0-1)
    pub accuracy: VecDeque<f32>,

    /// Total invocations
    pub invocations: u64,

    /// Total errors
    pub errors: u64,
}

impl ComponentMetrics {
    /// Create with a maximum history size
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            latencies: VecDeque::with_capacity(capacity),
            memory_usage: VecDeque::with_capacity(capacity),
            throughput: VecDeque::with_capacity(capacity),
            accuracy: VecDeque::with_capacity(capacity),
            invocations: 0,
            errors: 0,
        }
    }

    /// Record a latency sample
    pub fn record_latency(&mut self, latency_ms: f64, max_size: usize) {
        if self.latencies.len() >= max_size {
            self.latencies.pop_front();
        }
        self.latencies.push_back(latency_ms);
        self.invocations += 1;
    }

    /// Record an error
    pub fn record_error(&mut self) {
        self.errors += 1;
    }

    /// Record accuracy
    pub fn record_accuracy(&mut self, accuracy: f32, max_size: usize) {
        if self.accuracy.len() >= max_size {
            self.accuracy.pop_front();
        }
        self.accuracy.push_back(accuracy);
    }

    /// Get average latency
    pub fn avg_latency(&self) -> Option<f64> {
        if self.latencies.is_empty() {
            None
        } else {
            Some(self.latencies.iter().sum::<f64>() / self.latencies.len() as f64)
        }
    }

    /// Get average accuracy
    pub fn avg_accuracy(&self) -> Option<f32> {
        if self.accuracy.is_empty() {
            None
        } else {
            Some(self.accuracy.iter().sum::<f32>() / self.accuracy.len() as f32)
        }
    }

    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        if self.invocations == 0 {
            0.0
        } else {
            self.errors as f64 / self.invocations as f64
        }
    }
}

/// Performance monitor for system components
#[derive(Debug)]
pub struct PerformanceMonitor {
    /// Configuration
    config: MonitorConfig,

    /// Metrics per component
    metrics: HashMap<ComponentId, ComponentMetrics>,

    /// Detected bottlenecks
    bottlenecks: Vec<Bottleneck>,

    /// When the monitor was created
    created_at: Instant,
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            config,
            metrics: HashMap::new(),
            bottlenecks: Vec::new(),
            created_at: Instant::now(),
        }
    }

    /// Start timing an operation
    pub fn start_timer(&self) -> Instant {
        Instant::now()
    }

    /// Record latency for a component
    pub fn record_latency(&mut self, component_id: impl Into<ComponentId>, start: Instant) {
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        let id = component_id.into();
        let max_size = self.config.history_size;

        self.metrics
            .entry(id)
            .or_insert_with(|| ComponentMetrics::with_capacity(max_size))
            .record_latency(latency_ms, max_size);
    }

    /// Record accuracy for a component
    pub fn record_accuracy(&mut self, component_id: impl Into<ComponentId>, accuracy: f32) {
        let id = component_id.into();
        let max_size = self.config.history_size;

        self.metrics
            .entry(id)
            .or_insert_with(|| ComponentMetrics::with_capacity(max_size))
            .record_accuracy(accuracy, max_size);
    }

    /// Record an error
    pub fn record_error(&mut self, component_id: impl Into<ComponentId>) {
        let id = component_id.into();
        let max_size = self.config.history_size;

        self.metrics
            .entry(id)
            .or_insert_with(|| ComponentMetrics::with_capacity(max_size))
            .record_error();
    }

    /// Get metrics for a component
    pub fn get_metrics(&self, component_id: &ComponentId) -> Option<&ComponentMetrics> {
        self.metrics.get(component_id)
    }

    /// Detect bottlenecks across all components
    pub fn detect_bottlenecks(&mut self) -> Vec<Bottleneck> {
        let mut new_bottlenecks = Vec::new();

        for (id, metrics) in &self.metrics {
            // Check latency
            if let Some(avg_latency) = metrics.avg_latency() {
                if avg_latency > self.config.latency_threshold_ms {
                    let severity =
                        ((avg_latency / self.config.latency_threshold_ms) - 1.0).min(1.0) as f32;
                    new_bottlenecks.push(
                        Bottleneck::new(
                            id.clone(),
                            BottleneckType::Latency,
                            severity,
                            format!(
                                "Average latency {:.2}ms exceeds threshold {:.2}ms",
                                avg_latency, self.config.latency_threshold_ms
                            ),
                        )
                        .with_evidence("avg_latency_ms", avg_latency)
                        .with_suggestion("Consider caching frequent operations")
                        .with_suggestion("Review algorithm complexity"),
                    );
                }
            }

            // Check accuracy
            if let Some(avg_accuracy) = metrics.avg_accuracy() {
                if avg_accuracy < self.config.accuracy_threshold {
                    let severity = (self.config.accuracy_threshold - avg_accuracy)
                        / self.config.accuracy_threshold;
                    new_bottlenecks.push(
                        Bottleneck::new(
                            id.clone(),
                            BottleneckType::Accuracy,
                            severity,
                            format!(
                                "Average accuracy {:.2}% below threshold {:.2}%",
                                avg_accuracy * 100.0,
                                self.config.accuracy_threshold * 100.0
                            ),
                        )
                        .with_evidence("avg_accuracy", avg_accuracy as f64)
                        .with_suggestion("Review training data quality")
                        .with_suggestion("Consider model architecture changes"),
                    );
                }
            }

            // Check error rate
            let error_rate = metrics.error_rate();
            if error_rate > 0.05 {
                let severity = (error_rate * 10.0).min(1.0) as f32;
                new_bottlenecks.push(
                    Bottleneck::new(
                        id.clone(),
                        BottleneckType::Integration,
                        severity,
                        format!(
                            "Error rate {:.2}% exceeds acceptable threshold",
                            error_rate * 100.0
                        ),
                    )
                    .with_evidence("error_rate", error_rate)
                    .with_suggestion("Add error handling and retry logic")
                    .with_suggestion("Review component interface contracts"),
                );
            }
        }

        self.bottlenecks.extend(new_bottlenecks.clone());
        new_bottlenecks
    }

    /// Get all detected bottlenecks
    pub fn get_bottlenecks(&self) -> &[Bottleneck] {
        &self.bottlenecks
    }

    /// Clear old bottlenecks
    pub fn clear_bottlenecks(&mut self) {
        self.bottlenecks.clear();
    }

    /// Get uptime
    pub fn uptime(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get a snapshot of current stats for all components
    pub fn get_stats(&self) -> HashMap<ComponentId, ComponentMetrics> {
        self.metrics.clone()
    }

    /// Record a phi (integrated information) measurement for a component
    pub fn record_phi(&mut self, component_id: impl Into<ComponentId>, phi: f64) {
        // Store phi as an accuracy metric (0.0-1.0 range)
        self.record_accuracy(component_id, phi as f32);
    }
}

/// Accuracy metric types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccuracyMetric {
    /// Classification accuracy
    Classification,
    /// Regression mean squared error
    MeanSquaredError,
    /// F1 score
    F1Score,
    /// Precision
    Precision,
    /// Recall
    Recall,
    /// Custom metric
    Custom,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_component_id() {
        let id = ComponentId::new("test-component");
        assert_eq!(id.as_str(), "test-component");
    }

    #[test]
    fn test_bottleneck_creation() {
        let bottleneck = Bottleneck::new("test", BottleneckType::Latency, 0.8, "Test bottleneck");
        assert_eq!(bottleneck.severity, 0.8);
    }

    #[test]
    fn test_performance_monitor() {
        let config = MonitorConfig::default();
        let mut monitor = PerformanceMonitor::new(config);

        let start = monitor.start_timer();
        std::thread::sleep(std::time::Duration::from_millis(10));
        monitor.record_latency("test", start);

        let metrics = monitor.get_metrics(&ComponentId::new("test")).unwrap();
        assert!(metrics.avg_latency().unwrap() >= 10.0);
    }
}
