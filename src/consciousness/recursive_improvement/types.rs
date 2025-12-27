//! Shared types for the recursive improvement system
//!
//! This module contains core types used across all recursive improvement submodules.

use serde::{Deserialize, Serialize};

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
    /// Add caching layer
    AddCaching,
    /// Optimize primitive selection
    OptimizePrimitiveSelection,
    /// Parallelize Byzantine voting
    ParallelizeVoting,
    /// Cache meta decisions
    CacheMetaDecisions,
    /// Expand cache size
    ExpandCacheSize,
    /// Expand attack detection patterns
    ExpandAttackPatterns,
    /// Refine fitness function
    RefineFitnessFunction,
    /// Improve consensus mechanism
    ImproveConsensus,
    /// Expand meta-learning window
    ExpandMetaLearningWindow,
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
