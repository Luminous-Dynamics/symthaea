//! # Ultimate Breakthrough #3: Recursive Self-Improvement
//!
//! This module implements the REVOLUTIONARY recursive self-improvement system
//! where the AI uses causal reasoning to understand and improve its own architecture!
//!
//! **Key Innovation**: The first AI system that uses causal analysis to optimize
//! its own design, creating a feedback loop of continuous improvement.
//!
//! ## The Paradigm Shift
//!
//! Traditional AI:
//! - Architecture designed by humans
//! - Hyperparameters tuned manually
//! - Improvements require expert intervention
//! - **Result**: Static design, limited optimization
//!
//! Recursive Self-Improvement:
//! - Monitors its own performance continuously
//! - Identifies bottlenecks using causal analysis
//! - Generates architectural improvements automatically
//! - Tests improvements safely and adopts successful ones
//! - **Result**: Autonomous evolution toward optimal design!
//!
//! ## Architecture
//!
//! 1. **PerformanceMonitor**: Tracks metrics (Φ, latency, accuracy)
//! 2. **ArchitecturalCausalGraph**: Models component interactions
//! 3. **BottleneckDetector**: Identifies performance problems
//! 4. **ImprovementGenerator**: Proposes optimizations
//! 5. **SafeExperiment**: Tests improvements in sandbox
//! 6. **RecursiveOptimizer**: Coordinates the improvement loop
//!
//! ## Safety Guarantees
//!
//! - All improvements tested in isolation
//! - Automatic rollback if performance degrades
//! - Conservative adoption (multiple validations required)
//! - Bounded exploration (limits on change magnitude)
//! - Human oversight for major architectural changes

use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};
use std::f64::consts::PI;

// Re-export self_model types for backward compatibility with existing code
pub use crate::consciousness::recursive_improvement::self_model::{
    CapabilityDomain, SelfModel, SelfModelConfig,
    BehaviorPrediction, CalibrationStats, CapabilityEstimate,
    ImprovementTrajectory, ControllerConfig, ControllerState, ControllerOutput,
    RecommendedAction, DesiredSelfState, ImprovementMethod, ImprovementStep,
    KnownLimitation, PredictionRecord, UnifiedImprovementController, ControllerStats,
};

// Re-export world_model types for backward compatibility with existing code
pub use crate::consciousness::recursive_improvement::world_model::{
    LatentConsciousnessState, ConsciousnessAction, ConsciousnessTransition,
    ConsciousnessDynamicsModel, RewardPredictor, Counterfactual,
    ConsciousnessWorldModel, WorldModelConfig, WorldModelStats, WorldModelSummary,
};

// Re-export meta_cognitive types for backward compatibility with existing code
pub use crate::consciousness::recursive_improvement::meta_cognitive::{
    CognitiveResourceType, CognitiveResources, SubsystemId, SubsystemHealth,
    MetaGoal, MetaGoalType, MetaCognitiveConfig, MetaCognitiveStats,
    AttentionBroadcast, BroadcastContentType, MetaCognitiveController,
    MetaCognitiveSummary,
};

// Re-export all router types from the routers module for backward compatibility
// This consolidates the modular router implementations (Phase 5G/5H improvements)
pub use crate::consciousness::recursive_improvement::routers::*;

// Re-export routing_hub types for backward compatibility
// TODO: routing_hub depends on routers that aren't implemented yet
// pub use crate::consciousness::recursive_improvement::routing_hub::{
//     RoutingMode, RouterType, UnifiedRoutingDecision, RouterPerformance,
//     RoutingHubConfig, ConsciousnessRoutingHub,
// };

// Re-export benchmark_suite types for backward compatibility
// TODO: benchmark_suite depends on routers that aren't implemented yet
// pub use crate::consciousness::recursive_improvement::benchmark_suite::{
//     RouterBenchmark, ComparativeBenchmark, BenchmarkConfig, RouterBenchmarkSuite,
// };

// ═══════════════════════════════════════════════════════════════════
// PERFORMANCE MONITORING
// ═══════════════════════════════════════════════════════════════════

/// Performance monitor that tracks system metrics over time
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceMonitor {
    /// Φ (consciousness) measurements over time
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

// Helper function for default Instant
fn instant_now() -> Instant {
    Instant::now()
}

/// Single Φ measurement
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

    /// Φ not improving
    PhiStagnation,

    /// Low Φ (consciousness measure)
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
    /// Average Φ over window
    pub avg_phi: f64,

    /// Φ trend (positive = improving)
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

    /// Minimum Φ trend for "improving"
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
            phi_improvement_threshold: 0.01, // 1% improvement
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

    /// Record a Φ measurement
    pub fn record_phi(&mut self, phi: f64, component_count: usize, context: String) {
        let measurement = PhiMeasurement {
            timestamp: Instant::now(),
            phi,
            component_count,
            context,
        };

        self.phi_history.push_back(measurement);

        // Trim history if needed
        while self.phi_history.len() > self.config.history_size {
            self.phi_history.pop_front();
        }

        // Update statistics
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

        // Trim history
        while self.latency_history.len() > self.config.history_size {
            self.latency_history.pop_front();
        }

        // Update statistics
        self.update_latency_stats();

        // Check for latency bottlenecks
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

        // Trim history
        while self.accuracy_history.len() > self.config.history_size {
            self.accuracy_history.pop_front();
        }

        // Update statistics
        self.update_accuracy_stats();

        // Check for accuracy bottlenecks
        if value < self.config.accuracy_threshold {
            self.detect_accuracy_bottleneck(metric_type, value);
        }
    }

    /// Update Φ statistics
    fn update_phi_stats(&mut self) {
        if self.phi_history.is_empty() {
            return;
        }

        // Calculate average Φ over window
        let window_size = self.config.trend_window.min(self.phi_history.len());
        let recent: Vec<f64> = self.phi_history.iter()
            .rev()
            .take(window_size)
            .map(|m| m.phi)
            .collect();

        self.stats.avg_phi = recent.iter().sum::<f64>() / recent.len() as f64;

        // Calculate trend (linear regression slope)
        // Note: recent is in reverse order (newest first), so we need to reverse for trend calc
        if recent.len() >= 2 {
            let chronological: Vec<f64> = recent.iter().rev().copied().collect();
            self.stats.phi_trend = calculate_trend(&chronological);

            // Check for Φ stagnation
            if self.stats.phi_trend.abs() < self.config.phi_improvement_threshold {
                self.detect_phi_stagnation();
            }
        }
    }

    /// Update latency statistics
    fn update_latency_stats(&mut self) {
        // Calculate average latency per component
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

    /// Update accuracy statistics
    fn update_accuracy_stats(&mut self) {
        // Calculate average accuracy per metric type
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

    /// Detect latency bottleneck
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

    /// Detect accuracy bottleneck
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

    /// Detect Φ stagnation
    fn detect_phi_stagnation(&mut self) {
        if self.phi_history.len() < self.config.trend_window {
            return; // Not enough data
        }

        let bottleneck = Bottleneck {
            id: format!("phi_stagnation_{}", Instant::now().elapsed().as_millis()),
            component: ComponentId::PrimitiveEvolution,
            bottleneck_type: BottleneckType::PhiStagnation,
            severity: 0.7, // Moderate severity
            description: format!(
                "Φ not improving (trend: {:.4}, threshold: {:.4})",
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

// ═══════════════════════════════════════════════════════════════════
// IMPROVEMENT TYPES
// ═══════════════════════════════════════════════════════════════════

/// Type of architectural improvement
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ImprovementType {
    /// Increase cache size
    IncreaseCacheSize { from: usize, to: usize },

    /// Parallelize operation
    Parallelize { component: ComponentId, threads: usize },

    /// Increase evolution rate
    IncreaseEvolutionRate,

    /// Add synthetic training data
    AddSyntheticData { count: usize },

    /// Tune hyperparameter
    TuneHyperparameter { name: String, old_value: f64, new_value: f64 },

    /// Optimize algorithm
    OptimizeAlgorithm { component: ComponentId, optimization: String },
}

// ═══════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════

/// Calculate trend (linear regression slope)
fn calculate_trend(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let n = values.len() as f64;
    let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
    let y_sum: f64 = values.iter().sum();
    let xy_sum: f64 = values.iter()
        .enumerate()
        .map(|(i, &y)| i as f64 * y)
        .sum();
    let x2_sum: f64 = (0..values.len()).map(|i| (i * i) as f64).sum();

    let slope = (n * xy_sum - x_sum * y_sum) / (n * x2_sum - x_sum * x_sum);
    slope
}

/// Suggest fix for latency bottleneck
fn suggest_latency_fix(component: ComponentId) -> ImprovementType {
    match component {
        ComponentId::Cache => ImprovementType::IncreaseCacheSize {
            from: 1000,
            to: 5000,
        },
        ComponentId::PrimitiveEvolution => ImprovementType::Parallelize {
            component,
            threads: 4,
        },
        ComponentId::MetaLearning => ImprovementType::OptimizeAlgorithm {
            component,
            optimization: "Use faster pattern matching".to_string(),
        },
        _ => ImprovementType::Parallelize {
            component,
            threads: 2,
        },
    }
}

/// Suggest fix for accuracy bottleneck
fn suggest_accuracy_fix(metric: AccuracyMetric) -> ImprovementType {
    match metric {
        AccuracyMetric::AttackDetection | AccuracyMetric::MetaLearning => {
            ImprovementType::AddSyntheticData { count: 100 }
        }
        AccuracyMetric::FitnessPrediction => {
            ImprovementType::TuneHyperparameter {
                name: "learning_rate".to_string(),
                old_value: 0.01,
                new_value: 0.001,
            }
        }
        AccuracyMetric::CollectiveReasoning => {
            ImprovementType::IncreaseEvolutionRate
        }
    }
}

/// Map accuracy metric to component
fn metric_to_component(metric: AccuracyMetric) -> ComponentId {
    match metric {
        AccuracyMetric::AttackDetection => ComponentId::ByzantineCollective,
        AccuracyMetric::FitnessPrediction => ComponentId::PrimitiveEvolution,
        AccuracyMetric::CollectiveReasoning => ComponentId::UnifiedIntelligence,
        AccuracyMetric::MetaLearning => ComponentId::MetaLearning,
    }
}

// Re-export architectural_graph types for backward compatibility
pub use crate::consciousness::recursive_improvement::architectural_graph::{
    ArchitecturalCausalGraph, ComponentNode, ArchitecturalEdge,
    CausalRelationship, PerformanceImpact, CausalChain, GraphStats,
};

// Re-export safe_experiment types for backward compatibility
pub use crate::consciousness::recursive_improvement::safe_experiment::{
    SafeExperiment, SystemSnapshot, ArchitecturalImprovement,
    SuccessCriteria, RollbackCondition, ExperimentStatus,
    ValidationRun, ExperimentConfig,
};

// Re-export improvement_generator types for backward compatibility
pub use crate::consciousness::recursive_improvement::improvement_generator::{
    ImprovementGenerator, ImprovementRecord, ImprovementOutcome,
    ImprovementPatterns, CausalPattern, GeneratorConfig, GeneratorStats,
};

// Re-export recursive_optimizer types for backward compatibility
pub use crate::consciousness::recursive_improvement::recursive_optimizer::{
    RecursiveOptimizer, OptimizationCycle, OptimizerConfig, OptimizerStats,
};

// Re-export gradient_optimizer types for backward compatibility
pub use crate::consciousness::recursive_improvement::gradient_optimizer::{
    ConsciousnessGradientOptimizer, ArchitecturalParameter, ConsciousnessGradient,
    OptimizationObjective, AdamState, GradientOptimizerConfig, GradientOptimizerStats,
    GradientStep,
};

// Re-export intrinsic_motivation types for backward compatibility
pub use crate::consciousness::recursive_improvement::intrinsic_motivation::{
    IntrinsicMotivationSystem, DriveType, DriveState, AutonomousGoal,
    CuriosityModule, CompetenceModule, AutonomyModule,
    MotivationConfig, MotivationStats, MotivatedAction,
};

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_performance_monitor_basic() {
        let mut monitor = PerformanceMonitor::new(MonitorConfig::default());

        // Record some measurements
        monitor.record_phi(0.5, 3, "test".to_string());
        monitor.record_phi(0.6, 3, "test".to_string());
        monitor.record_phi(0.7, 3, "test".to_string());

        let stats = monitor.get_stats();
        assert!(stats.avg_phi > 0.0);
        assert!(stats.phi_trend > 0.0); // Should be improving
    }

    #[test]
    fn test_latency_bottleneck_detection() {
        let mut monitor = PerformanceMonitor::new(MonitorConfig::default());

        // Record high latency
        monitor.record_latency(
            "test".to_string(),
            Duration::from_millis(200), // Above threshold
            ComponentId::Cache,
        );

        let bottlenecks = monitor.get_bottlenecks(10);
        assert_eq!(bottlenecks.len(), 1);
        assert_eq!(bottlenecks[0].bottleneck_type, BottleneckType::Latency);
    }

    #[test]
    fn test_trend_calculation() {
        // Increasing trend
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let trend = calculate_trend(&values);
        assert!(trend > 0.9 && trend < 1.1); // Should be ~1.0

        // Decreasing trend
        let values = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let trend = calculate_trend(&values);
        assert!(trend < -0.9 && trend > -1.1); // Should be ~-1.0

        // Flat trend
        let values = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        let trend = calculate_trend(&values);
        assert!(trend.abs() < 0.01); // Should be ~0.0
    }
}
