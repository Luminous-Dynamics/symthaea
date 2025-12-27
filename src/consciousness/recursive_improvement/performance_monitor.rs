//! Performance Monitoring for Recursive Self-Improvement
//!
//! This module tracks system metrics (Phi, latency, accuracy) and identifies bottlenecks.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use super::improvement_types::ImprovementType;

// Helper function for default Instant
fn instant_now() -> Instant {
    Instant::now()
}

/// Performance monitor that tracks system metrics over time
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceMonitor {
    /// Phi (consciousness) measurements over time
    phi_history: VecDeque<PhiMeasurement>,

    /// Reasoning latency measurements
    latency_history: VecDeque<LatencyMeasurement>,

    /// Accuracy measurements
    accuracy_history: VecDeque<AccuracyMeasurement>,

    /// Detected bottlenecks
    bottlenecks: Vec<Bottleneck>,

    /// Performance statistics
    stats: PerformanceStats,

    /// Configuration
    config: MonitorConfig,
}

/// Single Phi measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiMeasurement {
    #[serde(skip, default = "instant_now")]
    pub timestamp: Instant,
    pub phi: f64,
    pub component_count: usize,
    pub context: String,
}

/// Single latency measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyMeasurement {
    #[serde(skip, default = "instant_now")]
    pub timestamp: Instant,
    pub operation: String,
    pub duration: Duration,
    pub component: ComponentId,
}

/// Single accuracy measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccuracyMeasurement {
    #[serde(skip, default = "instant_now")]
    pub timestamp: Instant,
    pub metric_type: AccuracyMetric,
    pub value: f64,
    pub context: String,
}

/// Type of accuracy metric
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AccuracyMetric {
    /// Byzantine attack detection accuracy
    AttackDetection,
    /// Primitive fitness prediction
    FitnessPrediction,
    /// Collective reasoning accuracy
    CollectiveReasoning,
    /// Meta-learning prediction
    MetaLearning,
}

/// Identified performance bottleneck
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bottleneck {
    /// Bottleneck identifier
    pub id: String,
    /// Which component is bottlenecked
    pub component: ComponentId,
    /// Type of bottleneck
    pub bottleneck_type: BottleneckType,
    /// Severity (0.0 = minor, 1.0 = critical)
    pub severity: f64,
    /// Description
    pub description: String,
    /// Suggested fix
    pub suggested_fix: Option<ImprovementType>,
    /// When detected
    #[serde(skip, default = "instant_now")]
    pub detected_at: Instant,
}

/// Type of performance bottleneck
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BottleneckType {
    /// High latency
    Latency,
    /// Low accuracy
    Accuracy,
    /// Low accuracy (alias for rule-based generator)
    LowAccuracy,
    /// Phi not improving
    PhiStagnation,
    /// Low Phi (consciousness measure)
    LowPhi,
    /// Memory pressure
    Memory,
    /// Resource exhaustion (memory, CPU, etc.)
    ResourceExhaustion,
    /// CPU bottleneck
    Computation,
    /// I/O bottleneck
    IO,
    /// System oscillation / instability
    Oscillation,
}

/// System component identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ComponentId {
    /// Primitive evolution
    PrimitiveEvolution,
    /// HRM reasoning
    HRM,
    /// Meta-cognitive monitoring
    MetaCognition,
    /// Byzantine collective
    ByzantineCollective,
    /// Meta-learning defense
    MetaLearning,
    /// Causal defense
    CausalDefense,
    /// Unified intelligence
    UnifiedIntelligence,
    /// Collective primitive sharing
    CollectiveSharing,
    /// Cache system
    Cache,
    /// Integration / consciousness binding
    Integration,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    /// Average Phi over window
    pub avg_phi: f64,
    /// Phi trend (positive = improving)
    pub phi_trend: f64,
    /// Average latency per component
    pub avg_latency: HashMap<ComponentId, Duration>,
    /// Average accuracy per metric
    pub avg_accuracy: HashMap<AccuracyMetric, f64>,
    /// Total bottlenecks detected
    pub bottlenecks_detected: usize,
    /// Measurements window size
    pub window_size: usize,
}

/// Monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfig {
    /// How many measurements to keep
    pub history_size: usize,
    /// Minimum Phi trend for "improving"
    pub phi_improvement_threshold: f64,
    /// Maximum acceptable latency per operation
    pub latency_threshold: Duration,
    /// Minimum acceptable accuracy
    pub accuracy_threshold: f64,
    /// How many measurements for trend analysis
    pub trend_window: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            history_size: 1000,
            phi_improvement_threshold: 0.01,
            latency_threshold: Duration::from_millis(100),
            accuracy_threshold: 0.85,
            trend_window: 50,
        }
    }
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new(config: MonitorConfig) -> Self {
        Self {
            phi_history: VecDeque::with_capacity(config.history_size),
            latency_history: VecDeque::with_capacity(config.history_size),
            accuracy_history: VecDeque::with_capacity(config.history_size),
            bottlenecks: Vec::new(),
            stats: PerformanceStats {
                avg_phi: 0.0,
                phi_trend: 0.0,
                avg_latency: HashMap::new(),
                avg_accuracy: HashMap::new(),
                bottlenecks_detected: 0,
                window_size: config.trend_window,
            },
            config,
        }
    }

    /// Record a Phi measurement
    pub fn record_phi(&mut self, phi: f64, component_count: usize, context: String) {
        let measurement = PhiMeasurement {
            timestamp: Instant::now(),
            phi,
            component_count,
            context,
        };

        self.phi_history.push_back(measurement);

        while self.phi_history.len() > self.config.history_size {
            self.phi_history.pop_front();
        }

        self.update_phi_stats();
    }

    /// Record a latency measurement
    pub fn record_latency(&mut self, operation: String, duration: Duration, component: ComponentId) {
        let measurement = LatencyMeasurement {
            timestamp: Instant::now(),
            operation,
            duration,
            component,
        };

        self.latency_history.push_back(measurement);

        while self.latency_history.len() > self.config.history_size {
            self.latency_history.pop_front();
        }

        self.update_latency_stats();

        if duration > self.config.latency_threshold {
            self.detect_latency_bottleneck(component, duration);
        }
    }

    /// Record an accuracy measurement
    pub fn record_accuracy(&mut self, metric_type: AccuracyMetric, value: f64, context: String) {
        let measurement = AccuracyMeasurement {
            timestamp: Instant::now(),
            metric_type,
            value,
            context,
        };

        self.accuracy_history.push_back(measurement);

        while self.accuracy_history.len() > self.config.history_size {
            self.accuracy_history.pop_front();
        }

        self.update_accuracy_stats();

        if value < self.config.accuracy_threshold {
            self.detect_accuracy_bottleneck(metric_type, value);
        }
    }

    fn update_phi_stats(&mut self) {
        if self.phi_history.is_empty() {
            return;
        }

        let window_size = self.config.trend_window.min(self.phi_history.len());
        let recent: Vec<f64> = self.phi_history.iter()
            .rev()
            .take(window_size)
            .map(|m| m.phi)
            .collect();

        self.stats.avg_phi = recent.iter().sum::<f64>() / recent.len() as f64;

        if recent.len() >= 2 {
            let chronological: Vec<f64> = recent.iter().rev().copied().collect();
            self.stats.phi_trend = calculate_trend(&chronological);

            if self.stats.phi_trend.abs() < self.config.phi_improvement_threshold {
                self.detect_phi_stagnation();
            }
        }
    }

    fn update_latency_stats(&mut self) {
        let mut component_latencies: HashMap<ComponentId, Vec<Duration>> = HashMap::new();

        for measurement in self.latency_history.iter().rev().take(self.config.trend_window) {
            component_latencies.entry(measurement.component)
                .or_insert_with(Vec::new)
                .push(measurement.duration);
        }

        self.stats.avg_latency.clear();
        for (component, latencies) in component_latencies {
            let avg = latencies.iter()
                .map(|d| d.as_micros())
                .sum::<u128>() / latencies.len() as u128;
            self.stats.avg_latency.insert(component, Duration::from_micros(avg as u64));
        }
    }

    fn update_accuracy_stats(&mut self) {
        let mut metric_values: HashMap<AccuracyMetric, Vec<f64>> = HashMap::new();

        for measurement in self.accuracy_history.iter().rev().take(self.config.trend_window) {
            metric_values.entry(measurement.metric_type)
                .or_insert_with(Vec::new)
                .push(measurement.value);
        }

        self.stats.avg_accuracy.clear();
        for (metric, values) in metric_values {
            let avg = values.iter().sum::<f64>() / values.len() as f64;
            self.stats.avg_accuracy.insert(metric, avg);
        }
    }

    fn detect_latency_bottleneck(&mut self, component: ComponentId, duration: Duration) {
        let severity = (duration.as_micros() as f64 / self.config.latency_threshold.as_micros() as f64).min(1.0);

        let bottleneck = Bottleneck {
            id: format!("latency_{:?}_{}", component, Instant::now().elapsed().as_millis()),
            component,
            bottleneck_type: BottleneckType::Latency,
            severity,
            description: format!(
                "{:?} operation took {:?} (threshold: {:?})",
                component, duration, self.config.latency_threshold
            ),
            suggested_fix: Some(suggest_latency_fix(component)),
            detected_at: Instant::now(),
        };

        self.bottlenecks.push(bottleneck);
        self.stats.bottlenecks_detected += 1;
    }

    fn detect_accuracy_bottleneck(&mut self, metric_type: AccuracyMetric, value: f64) {
        let severity = (self.config.accuracy_threshold - value) / self.config.accuracy_threshold;

        let bottleneck = Bottleneck {
            id: format!("accuracy_{:?}_{}", metric_type, Instant::now().elapsed().as_millis()),
            component: metric_to_component(metric_type),
            bottleneck_type: BottleneckType::Accuracy,
            severity,
            description: format!(
                "{:?} accuracy is {:.1}% (threshold: {:.1}%)",
                metric_type, value * 100.0, self.config.accuracy_threshold * 100.0
            ),
            suggested_fix: Some(suggest_accuracy_fix(metric_type)),
            detected_at: Instant::now(),
        };

        self.bottlenecks.push(bottleneck);
        self.stats.bottlenecks_detected += 1;
    }

    fn detect_phi_stagnation(&mut self) {
        if self.phi_history.len() < self.config.trend_window {
            return;
        }

        let bottleneck = Bottleneck {
            id: format!("phi_stagnation_{}", Instant::now().elapsed().as_millis()),
            component: ComponentId::PrimitiveEvolution,
            bottleneck_type: BottleneckType::PhiStagnation,
            severity: 0.7,
            description: format!(
                "Phi not improving (trend: {:.4}, threshold: {:.4})",
                self.stats.phi_trend, self.config.phi_improvement_threshold
            ),
            suggested_fix: Some(ImprovementType::IncreaseEvolutionRate),
            detected_at: Instant::now(),
        };

        self.bottlenecks.push(bottleneck);
        self.stats.bottlenecks_detected += 1;
    }

    /// Get current performance statistics
    pub fn get_stats(&self) -> &PerformanceStats {
        &self.stats
    }

    /// Get recent bottlenecks
    pub fn get_bottlenecks(&self, limit: usize) -> Vec<&Bottleneck> {
        self.bottlenecks.iter()
            .rev()
            .take(limit)
            .collect()
    }

    /// Get critical bottlenecks (severity > 0.7)
    pub fn get_critical_bottlenecks(&self) -> Vec<&Bottleneck> {
        self.bottlenecks.iter()
            .filter(|b| b.severity > 0.7)
            .collect()
    }
}

// Helper functions

/// Calculate trend using simple linear regression
pub fn calculate_trend(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f64;
    let sum_x: f64 = (0..values.len()).map(|i| i as f64).sum();
    let sum_y: f64 = values.iter().sum();
    let sum_xy: f64 = values.iter().enumerate().map(|(i, y)| i as f64 * y).sum();
    let sum_xx: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

    let denominator = n * sum_xx - sum_x.powi(2);
    if denominator.abs() < 1e-10 {
        return 0.0;
    }

    (n * sum_xy - sum_x * sum_y) / denominator
}

/// Map accuracy metric to component
pub fn metric_to_component(metric: AccuracyMetric) -> ComponentId {
    match metric {
        AccuracyMetric::AttackDetection => ComponentId::ByzantineCollective,
        AccuracyMetric::FitnessPrediction => ComponentId::PrimitiveEvolution,
        AccuracyMetric::CollectiveReasoning => ComponentId::CollectiveSharing,
        AccuracyMetric::MetaLearning => ComponentId::MetaLearning,
    }
}

/// Suggest fix for latency bottleneck
pub fn suggest_latency_fix(component: ComponentId) -> ImprovementType {
    match component {
        ComponentId::PrimitiveEvolution => ImprovementType::OptimizePrimitiveSelection,
        ComponentId::ByzantineCollective => ImprovementType::ParallelizeVoting,
        ComponentId::MetaCognition => ImprovementType::CacheMetaDecisions,
        ComponentId::Cache => ImprovementType::ExpandCacheSize,
        _ => ImprovementType::AddCaching,
    }
}

/// Suggest fix for accuracy bottleneck
pub fn suggest_accuracy_fix(metric: AccuracyMetric) -> ImprovementType {
    match metric {
        AccuracyMetric::AttackDetection => ImprovementType::ExpandAttackPatterns,
        AccuracyMetric::FitnessPrediction => ImprovementType::RefineFitnessFunction,
        AccuracyMetric::CollectiveReasoning => ImprovementType::ImproveConsensus,
        AccuracyMetric::MetaLearning => ImprovementType::ExpandMetaLearningWindow,
    }
}

impl BottleneckType {
    /// Get severity weight for this bottleneck type
    pub fn severity_weight(&self) -> f64 {
        match self {
            BottleneckType::PhiStagnation => 1.0,
            BottleneckType::LowPhi => 0.95,
            BottleneckType::Accuracy | BottleneckType::LowAccuracy => 0.8,
            BottleneckType::Latency => 0.7,
            BottleneckType::Memory | BottleneckType::ResourceExhaustion => 0.9,
            BottleneckType::Computation => 0.6,
            BottleneckType::IO => 0.5,
            BottleneckType::Oscillation => 0.85,
        }
    }

    /// Whether this bottleneck type requires immediate attention
    pub fn is_critical(&self) -> bool {
        matches!(self,
            BottleneckType::PhiStagnation |
            BottleneckType::LowPhi |
            BottleneckType::ResourceExhaustion |
            BottleneckType::Oscillation
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_monitor_creation() {
        let config = MonitorConfig::default();
        let monitor = PerformanceMonitor::new(config);
        assert_eq!(monitor.get_stats().bottlenecks_detected, 0);
    }

    #[test]
    fn test_phi_recording() {
        let config = MonitorConfig::default();
        let mut monitor = PerformanceMonitor::new(config);

        for i in 0..10 {
            monitor.record_phi(0.5 + i as f64 * 0.01, 100, "test".to_string());
        }

        assert!(monitor.get_stats().avg_phi > 0.0);
    }

    #[test]
    fn test_bottleneck_detection() {
        let mut config = MonitorConfig::default();
        config.accuracy_threshold = 0.9;
        let mut monitor = PerformanceMonitor::new(config);

        monitor.record_accuracy(AccuracyMetric::AttackDetection, 0.5, "low".to_string());

        assert!(monitor.get_stats().bottlenecks_detected > 0);
    }

    #[test]
    fn test_calculate_trend() {
        let increasing = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(calculate_trend(&increasing) > 0.0);

        let decreasing = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        assert!(calculate_trend(&decreasing) < 0.0);

        let flat = vec![3.0, 3.0, 3.0, 3.0];
        assert!(calculate_trend(&flat).abs() < 0.01);
    }
}
