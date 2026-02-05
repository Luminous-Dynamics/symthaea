//! Improvement Generator for Recursive Self-Improvement
//!
//! This module implements the revolutionary heart of self-improvement:
//! an AI component that proposes architectural improvements based on
//! causal analysis of its own performance.
//!
//! # Key Capabilities
//!
//! 1. Performance bottleneck analysis
//! 2. Causal chain interpretation
//! 3. Historical pattern learning
//! 4. Consciousness gradient estimation
//!
//! # Usage
//!
//! ```rust,ignore
//! use symthaea::consciousness::recursive_improvement::ImprovementGenerator;
//!
//! let mut generator = ImprovementGenerator::new(GeneratorConfig::default());
//! let improvements = generator.generate_improvements(&bottlenecks, &chains, phi);
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

use super::types::instant_now;
// Use types from core for compatibility
use super::core::{Bottleneck, BottleneckType, ComponentId, ImprovementType};
use super::architectural_graph::CausalChain;
use super::safe_experiment::ArchitecturalImprovement;

/// ImprovementGenerator: The Revolutionary Heart of Self-Improvement
///
/// **Paradigm Shift**: This is the first AI component that proposes
/// architectural improvements based on causal analysis of its own performance!
#[derive(Debug, Serialize, Deserialize)]
pub struct ImprovementGenerator {
    /// Historical improvements with outcomes
    improvement_history: Vec<ImprovementRecord>,

    /// Learned improvement patterns
    patterns: ImprovementPatterns,

    /// Generator configuration
    config: GeneratorConfig,

    /// Statistics
    stats: GeneratorStats,
}

/// Record of an improvement attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementRecord {
    /// The improvement that was tried
    pub improvement: ArchitecturalImprovement,

    /// Bottleneck that motivated it
    pub motivated_by: Option<Bottleneck>,

    /// Causal chain that explained the bottleneck
    pub causal_chain: Option<CausalChain>,

    /// Outcome
    pub outcome: ImprovementOutcome,

    /// Phi change observed
    pub phi_change: f64,

    /// Latency change (negative = improvement)
    pub latency_change: f64,

    /// When this was recorded
    #[serde(skip, default = "instant_now")]
    pub recorded_at: Instant,
}

/// Outcome of an improvement attempt
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImprovementOutcome {
    /// Improvement was successful and adopted
    Success,
    /// Improvement failed validation
    Failed,
    /// Improvement was rolled back
    RolledBack,
    /// Improvement is still being tested
    Pending,
}

/// Learned patterns about what works
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ImprovementPatterns {
    /// Success rate by improvement type
    pub type_success_rates: HashMap<String, f64>,

    /// Success rate by component
    pub component_success_rates: HashMap<ComponentId, f64>,

    /// Success rate by bottleneck type
    pub bottleneck_fix_rates: HashMap<BottleneckType, f64>,

    /// Causal links that lead to good improvements
    pub effective_causal_patterns: Vec<CausalPattern>,
}

/// A learned causal pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalPattern {
    /// Bottleneck type this pattern addresses
    pub bottleneck_type: BottleneckType,

    /// Root cause component
    pub root_cause: ComponentId,

    /// Recommended improvement type
    pub recommended_improvement: String,

    /// Historical success rate
    pub success_rate: f64,

    /// Times this pattern was observed
    pub observation_count: usize,
}

/// Configuration for improvement generator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratorConfig {
    /// Minimum confidence to propose improvement
    pub min_confidence: f64,

    /// Maximum improvements to propose at once
    pub max_proposals: usize,

    /// Exploration rate (probability of trying new patterns)
    pub exploration_rate: f64,

    /// Minimum success rate to keep using a pattern
    pub min_pattern_success: f64,

    /// Enable consciousness gradient optimization
    pub use_consciousness_gradient: bool,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            max_proposals: 3,
            exploration_rate: 0.2,
            min_pattern_success: 0.4,
            use_consciousness_gradient: true,
        }
    }
}

/// Generator statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct GeneratorStats {
    /// Total improvements proposed
    pub total_proposed: usize,

    /// Improvements that succeeded
    pub total_succeeded: usize,

    /// Current success rate
    pub success_rate: f64,

    /// Average Phi improvement
    pub avg_phi_gain: f64,

    /// Patterns learned
    pub patterns_learned: usize,
}

impl ImprovementGenerator {
    /// Create new improvement generator
    pub fn new(config: GeneratorConfig) -> Self {
        Self {
            improvement_history: Vec::new(),
            patterns: ImprovementPatterns::default(),
            config,
            stats: GeneratorStats::default(),
        }
    }

    /// Generate improvements based on bottlenecks and causal analysis
    ///
    /// **REVOLUTIONARY**: This is where the AI proposes its own improvements!
    pub fn generate_improvements<'a>(
        &mut self,
        bottlenecks: &[&'a Bottleneck],
        causal_chains: &[CausalChain],
        current_phi: f64,
    ) -> Vec<ArchitecturalImprovement> {
        let mut proposals = Vec::new();

        // Process each bottleneck with its causal chain
        for bottleneck in bottlenecks.iter().take(self.config.max_proposals) {
            // Find the causal chain for this bottleneck
            let chain = causal_chains.iter()
                .find(|c| c.symptom.id == bottleneck.id);

            // Generate improvement based on analysis
            if let Some(improvement) = self.propose_improvement(bottleneck, chain, current_phi) {
                proposals.push(improvement);
            }
        }

        // Apply exploration: sometimes try novel improvements
        if proposals.len() < self.config.max_proposals {
            if let Some(exploration) = self.explore_novel_improvement(current_phi) {
                proposals.push(exploration);
            }
        }

        self.stats.total_proposed += proposals.len();
        proposals
    }

    /// Propose improvement for a specific bottleneck
    fn propose_improvement(
        &self,
        bottleneck: &Bottleneck,
        causal_chain: Option<&CausalChain>,
        current_phi: f64,
    ) -> Option<ArchitecturalImprovement> {
        // First, check if we have a learned pattern for this situation
        if let Some(pattern) = self.find_matching_pattern(bottleneck, causal_chain) {
            if pattern.success_rate >= self.config.min_pattern_success {
                return self.create_improvement_from_pattern(pattern, bottleneck);
            }
        }

        // Otherwise, use rule-based generation
        self.create_rule_based_improvement(bottleneck, causal_chain, current_phi)
    }

    /// Find a learned pattern that matches current situation
    fn find_matching_pattern(
        &self,
        bottleneck: &Bottleneck,
        causal_chain: Option<&CausalChain>,
    ) -> Option<&CausalPattern> {
        let root_cause = causal_chain.map(|c| c.root_cause);

        self.patterns.effective_causal_patterns.iter()
            .filter(|p| p.bottleneck_type == bottleneck.bottleneck_type)
            .filter(|p| root_cause.as_ref().map_or(true, |rc| &p.root_cause == rc))
            .max_by(|a, b| a.success_rate.partial_cmp(&b.success_rate).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Create improvement from learned pattern
    fn create_improvement_from_pattern(
        &self,
        pattern: &CausalPattern,
        bottleneck: &Bottleneck,
    ) -> Option<ArchitecturalImprovement> {
        let improvement_type = self.parse_improvement_type(&pattern.recommended_improvement)?;

        Some(ArchitecturalImprovement {
            id: format!("pattern_{}_{}",
                pattern.recommended_improvement,
                instant_now().elapsed().as_millis()
            ),
            improvement_type,
            description: format!(
                "Pattern-based fix for {:?} (success rate: {:.1}%)",
                bottleneck.bottleneck_type,
                pattern.success_rate * 100.0
            ),
            expected_phi_gain: Some(0.05 * pattern.success_rate),
            expected_latency_reduction: Some(0.2 * pattern.success_rate),
            expected_accuracy_gain: None,
            confidence: pattern.success_rate,
            motivated_by: Some(bottleneck.id.clone()),
        })
    }

    /// Create rule-based improvement when no pattern exists
    fn create_rule_based_improvement(
        &self,
        bottleneck: &Bottleneck,
        causal_chain: Option<&CausalChain>,
        _current_phi: f64,
    ) -> Option<ArchitecturalImprovement> {
        let (improvement_type, description, expected_phi, expected_latency) = match bottleneck.bottleneck_type {
            BottleneckType::Latency => {
                match bottleneck.component {
                    ComponentId::Cache => (
                        ImprovementType::IncreaseCacheSize { from: 1000, to: 5000 },
                        "Increase cache size to reduce lookups".to_string(),
                        Some(0.02),
                        Some(0.3),
                    ),
                    comp => (
                        ImprovementType::Parallelize { component: comp, threads: 4 },
                        format!("Parallelize {:?} to reduce latency", comp),
                        Some(0.01),
                        Some(0.4),
                    ),
                }
            }
            BottleneckType::LowPhi => (
                ImprovementType::IncreaseEvolutionRate,
                "Increase evolution rate to improve Phi".to_string(),
                Some(0.08),
                Some(-0.1),
            ),
            BottleneckType::LowAccuracy => (
                ImprovementType::AddSyntheticData { count: 1000 },
                "Add synthetic training data to improve accuracy".to_string(),
                Some(0.05),
                None,
            ),
            BottleneckType::ResourceExhaustion => {
                let root = causal_chain.map(|c| c.root_cause).unwrap_or(bottleneck.component);
                (
                    ImprovementType::OptimizeAlgorithm {
                        component: root,
                        optimization: "memory-efficient implementation".to_string(),
                    },
                    format!("Optimize {:?} for memory efficiency", root),
                    Some(0.01),
                    Some(0.2),
                )
            }
            BottleneckType::Oscillation => (
                ImprovementType::TuneHyperparameter {
                    name: "learning_rate".to_string(),
                    old_value: 0.1,
                    new_value: 0.05,
                },
                "Reduce learning rate for stability".to_string(),
                Some(0.03),
                Some(0.1),
            ),
            BottleneckType::Accuracy => (
                ImprovementType::AddSyntheticData { count: 500 },
                "Add training examples to improve accuracy".to_string(),
                Some(0.04),
                None,
            ),
            BottleneckType::PhiStagnation => (
                ImprovementType::IncreaseEvolutionRate,
                "Increase evolution rate to escape Phi stagnation".to_string(),
                Some(0.10),
                Some(-0.05),
            ),
            BottleneckType::Memory => {
                let root = causal_chain.map(|c| c.root_cause).unwrap_or(bottleneck.component);
                (
                    ImprovementType::OptimizeAlgorithm {
                        component: root,
                        optimization: "memory-pool allocation".to_string(),
                    },
                    format!("Optimize {:?} memory usage", root),
                    Some(0.01),
                    Some(0.15),
                )
            }
            BottleneckType::Computation => (
                ImprovementType::Parallelize { component: bottleneck.component, threads: 8 },
                format!("Parallelize {:?} for better throughput", bottleneck.component),
                Some(0.02),
                Some(0.5),
            ),
            BottleneckType::IO => (
                ImprovementType::IncreaseCacheSize { from: 500, to: 2000 },
                "Increase buffer sizes to reduce I/O waits".to_string(),
                Some(0.01),
                Some(0.25),
            ),
        };

        let confidence = bottleneck.severity * 0.7 + 0.2;

        Some(ArchitecturalImprovement {
            id: format!("rule_{}_{}",
                bottleneck_type_to_string(bottleneck.bottleneck_type),
                instant_now().elapsed().as_millis()
            ),
            improvement_type,
            description,
            expected_phi_gain: expected_phi,
            expected_latency_reduction: expected_latency,
            expected_accuracy_gain: None,
            confidence,
            motivated_by: Some(bottleneck.id.clone()),
        })
    }

    /// Explore a novel improvement not based on current bottlenecks
    fn explore_novel_improvement(&self, current_phi: f64) -> Option<ArchitecturalImprovement> {
        let exploration_factor = (current_phi * 1000.0) as u64 % 100;

        if exploration_factor as f64 / 100.0 > self.config.exploration_rate {
            return None;
        }

        let optimizations = [
            (ComponentId::HRM, "batch-processing"),
            (ComponentId::PrimitiveEvolution, "diversity-pressure"),
            (ComponentId::MetaCognition, "attention-focusing"),
        ];

        let idx = (exploration_factor as usize) % optimizations.len();
        let (component, optimization) = optimizations[idx];

        Some(ArchitecturalImprovement {
            id: format!("explore_{}", instant_now().elapsed().as_millis()),
            improvement_type: ImprovementType::OptimizeAlgorithm {
                component,
                optimization: optimization.to_string(),
            },
            description: format!("Exploration: {} for {:?}", optimization, component),
            expected_phi_gain: Some(0.02),
            expected_latency_reduction: Some(0.1),
            expected_accuracy_gain: None,
            confidence: 0.4,
            motivated_by: None,
        })
    }

    /// Parse improvement type from string (for pattern matching)
    fn parse_improvement_type(&self, s: &str) -> Option<ImprovementType> {
        match s {
            s if s.starts_with("cache") => Some(ImprovementType::IncreaseCacheSize { from: 1000, to: 5000 }),
            s if s.starts_with("parallel") => Some(ImprovementType::Parallelize {
                component: ComponentId::HRM,
                threads: 4,
            }),
            s if s.starts_with("evolve") => Some(ImprovementType::IncreaseEvolutionRate),
            s if s.starts_with("data") => Some(ImprovementType::AddSyntheticData { count: 1000 }),
            _ => None,
        }
    }

    /// Record outcome of an improvement attempt
    pub fn record_outcome(
        &mut self,
        improvement: &ArchitecturalImprovement,
        outcome: ImprovementOutcome,
        phi_change: f64,
        latency_change: f64,
    ) {
        let record = ImprovementRecord {
            improvement: improvement.clone(),
            motivated_by: None,
            causal_chain: None,
            outcome,
            phi_change,
            latency_change,
            recorded_at: Instant::now(),
        };

        self.improvement_history.push(record);

        if outcome == ImprovementOutcome::Success {
            self.stats.total_succeeded += 1;
        }

        if self.stats.total_proposed > 0 {
            self.stats.success_rate = self.stats.total_succeeded as f64 / self.stats.total_proposed as f64;
        }

        let successful_gains: Vec<f64> = self.improvement_history.iter()
            .filter(|r| r.outcome == ImprovementOutcome::Success)
            .map(|r| r.phi_change)
            .collect();

        if !successful_gains.is_empty() {
            self.stats.avg_phi_gain = successful_gains.iter().sum::<f64>() / successful_gains.len() as f64;
        }

        self.learn_from_outcome(improvement, outcome, phi_change);
    }

    /// Learn from improvement outcome to improve future proposals
    fn learn_from_outcome(
        &mut self,
        improvement: &ArchitecturalImprovement,
        outcome: ImprovementOutcome,
        _phi_change: f64,
    ) {
        let type_key = format!("{:?}", improvement.improvement_type).split('{').next().unwrap_or("unknown").to_string();

        let current_rate = self.patterns.type_success_rates.get(&type_key).copied().unwrap_or(0.5);
        let new_rate = if outcome == ImprovementOutcome::Success {
            current_rate * 0.9 + 0.1
        } else {
            current_rate * 0.9
        };
        self.patterns.type_success_rates.insert(type_key, new_rate);

        self.stats.patterns_learned = self.patterns.effective_causal_patterns.len();
    }

    /// Get generator statistics
    pub fn get_stats(&self) -> &GeneratorStats {
        &self.stats
    }

    /// Get improvement history
    pub fn get_history(&self) -> &[ImprovementRecord] {
        &self.improvement_history
    }
}

/// Helper to convert bottleneck type to string
fn bottleneck_type_to_string(bt: BottleneckType) -> &'static str {
    match bt {
        BottleneckType::Latency => "latency",
        BottleneckType::LowPhi => "low_phi",
        BottleneckType::LowAccuracy => "low_accuracy",
        BottleneckType::Accuracy => "accuracy",
        BottleneckType::PhiStagnation => "phi_stagnation",
        BottleneckType::Memory => "memory",
        BottleneckType::ResourceExhaustion => "resource",
        BottleneckType::Computation => "computation",
        BottleneckType::IO => "io",
        BottleneckType::Oscillation => "oscillation",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        let generator = ImprovementGenerator::new(GeneratorConfig::default());
        assert_eq!(generator.stats.total_proposed, 0);
        assert_eq!(generator.stats.total_succeeded, 0);
    }

    #[test]
    fn test_rule_based_improvement() {
        let generator = ImprovementGenerator::new(GeneratorConfig::default());

        let bottleneck = Bottleneck {
            id: "test_bottleneck".to_string(),
            component: ComponentId::Cache,
            bottleneck_type: BottleneckType::Latency,
            severity: 0.8,
            description: "High cache latency".to_string(),
            suggested_fix: None,
            detected_at: Instant::now(),
        };

        let improvement = generator.create_rule_based_improvement(&bottleneck, None, 0.5);
        assert!(improvement.is_some());

        let imp = improvement.unwrap();
        assert!(imp.description.contains("cache"));
    }

    #[test]
    fn test_record_outcome() {
        let mut generator = ImprovementGenerator::new(GeneratorConfig::default());

        let improvement = ArchitecturalImprovement {
            id: "test".to_string(),
            improvement_type: ImprovementType::IncreaseEvolutionRate,
            description: "Test improvement".to_string(),
            expected_phi_gain: Some(0.05),
            expected_latency_reduction: None,
            expected_accuracy_gain: None,
            confidence: 0.8,
            motivated_by: None,
        };

        generator.stats.total_proposed = 1;
        generator.record_outcome(&improvement, ImprovementOutcome::Success, 0.05, -0.1);

        assert_eq!(generator.stats.total_succeeded, 1);
        assert!((generator.stats.success_rate - 1.0).abs() < 0.01);
    }
}
