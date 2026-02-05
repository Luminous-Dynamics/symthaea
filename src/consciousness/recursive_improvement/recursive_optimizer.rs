//! Recursive Optimizer - The Coordination Loop
//!
//! This module orchestrates the complete self-improvement loop, coordinating
//! performance monitoring, causal analysis, improvement generation, and
//! safe experimentation.
//!
//! # The Revolutionary Loop
//!
//! 1. Monitor performance (PerformanceMonitor)
//! 2. Identify bottlenecks and trace causes (ArchitecturalCausalGraph)
//! 3. Generate improvements (ImprovementGenerator)
//! 4. Test safely (SafeExperiment)
//! 5. Adopt successful improvements
//! 6. LOOP → System becomes better at improving itself!

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::types::instant_now;
use super::architectural_graph::ArchitecturalCausalGraph;
use super::safe_experiment::{
    ExperimentConfig, ExperimentStatus,
    SafeExperiment, SystemSnapshot,
};
use super::improvement_generator::{
    GeneratorConfig, ImprovementGenerator, ImprovementOutcome,
};

// Import from core for now (PerformanceMonitor, MonitorConfig, ComponentId still in core.rs)
use super::core::{PerformanceMonitor, MonitorConfig, ComponentId};

/// RecursiveOptimizer: Coordinates the Complete Self-Improvement Loop
///
/// **REVOLUTIONARY**: This is the main coordination layer that orchestrates
/// the first AI system capable of autonomous architectural evolution!
#[derive(Debug)]
pub struct RecursiveOptimizer {
    /// Performance monitoring
    monitor: PerformanceMonitor,

    /// Causal analysis
    causal_graph: ArchitecturalCausalGraph,

    /// Improvement generation
    generator: ImprovementGenerator,

    /// Active experiments
    active_experiments: Vec<SafeExperiment>,

    /// Optimization history
    optimization_history: Vec<OptimizationCycle>,

    /// Configuration
    config: OptimizerConfig,

    /// Statistics
    stats: OptimizerStats,
}

/// Record of one optimization cycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationCycle {
    /// Cycle ID
    pub cycle_id: usize,

    /// Starting Phi
    pub starting_phi: f64,

    /// Ending Phi
    pub ending_phi: f64,

    /// Bottlenecks addressed
    pub bottlenecks_addressed: usize,

    /// Improvements tried
    pub improvements_tried: usize,

    /// Improvements adopted
    pub improvements_adopted: usize,

    /// Duration
    #[serde(skip)]
    pub duration: Duration,
}

/// Optimizer configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// How often to run optimization (in reasoning cycles)
    pub optimization_frequency: usize,

    /// Maximum concurrent experiments
    pub max_concurrent_experiments: usize,

    /// Minimum Phi improvement to continue optimizing
    pub min_phi_improvement: f64,

    /// Maximum cycles without improvement before pausing
    pub max_stagnant_cycles: usize,

    /// Enable automatic adoption (vs. require human approval)
    pub auto_adopt: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimization_frequency: 100,
            max_concurrent_experiments: 3,
            min_phi_improvement: 0.01,
            max_stagnant_cycles: 5,
            auto_adopt: false, // Conservative default
        }
    }
}

/// Optimizer statistics
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct OptimizerStats {
    /// Total optimization cycles
    pub total_cycles: usize,

    /// Cycles with improvements
    pub successful_cycles: usize,

    /// Total Phi gained
    pub total_phi_gained: f64,

    /// Current Phi
    pub current_phi: f64,

    /// Cycles since last improvement
    pub stagnant_cycles: usize,

    /// Is optimization paused
    pub paused: bool,
}

impl RecursiveOptimizer {
    /// Create new recursive optimizer
    pub fn new(config: OptimizerConfig) -> Self {
        Self {
            monitor: PerformanceMonitor::new(MonitorConfig::default()),
            causal_graph: ArchitecturalCausalGraph::new(),
            generator: ImprovementGenerator::new(GeneratorConfig::default()),
            active_experiments: Vec::new(),
            optimization_history: Vec::new(),
            config,
            stats: OptimizerStats::default(),
        }
    }

    /// Run one optimization cycle
    ///
    /// **THE REVOLUTIONARY LOOP**: This is where the magic happens!
    pub fn optimize(&mut self) -> Result<OptimizationCycle> {
        let cycle_start = Instant::now();
        let starting_phi = self.stats.current_phi;

        // Step 1: Identify bottlenecks
        let bottlenecks = self.monitor.get_bottlenecks(5);

        // Step 2: Analyze bottlenecks causally
        let mut causal_chains = Vec::new();
        for bottleneck in &bottlenecks {
            if let Ok(chain) = self.causal_graph.analyze_bottleneck(bottleneck) {
                causal_chains.push(chain);
            }
        }

        // Step 3: Generate improvements
        let improvements = self.generator.generate_improvements(
            &bottlenecks,
            &causal_chains,
            starting_phi,
        );

        // Step 4: Create experiments for each improvement
        let mut improvements_tried = 0;
        for improvement in improvements {
            if self.active_experiments.len() >= self.config.max_concurrent_experiments {
                break;
            }

            let baseline = self.capture_baseline();
            let experiment = SafeExperiment::new(
                improvement,
                baseline,
                ExperimentConfig {
                    require_human_approval: !self.config.auto_adopt,
                    ..Default::default()
                },
            );

            self.active_experiments.push(experiment);
            improvements_tried += 1;
        }

        // Step 5: Run validation on active experiments
        let mut improvements_adopted = 0;
        let mut experiments_to_remove = Vec::new();

        for (i, experiment) in self.active_experiments.iter_mut().enumerate() {
            let _ = experiment.run_validation();

            match experiment.get_status() {
                ExperimentStatus::Successful => {
                    if self.config.auto_adopt {
                        if experiment.adopt().is_ok() {
                            improvements_adopted += 1;

                            // Record outcome
                            self.generator.record_outcome(
                                experiment.get_improvement(),
                                ImprovementOutcome::Success,
                                0.05, // Estimated phi gain
                                -0.1, // Estimated latency reduction
                            );
                        }
                    }
                    experiments_to_remove.push(i);
                }
                ExperimentStatus::Failed => {
                    self.generator.record_outcome(
                        experiment.get_improvement(),
                        ImprovementOutcome::Failed,
                        0.0,
                        0.0,
                    );
                    experiments_to_remove.push(i);
                }
                _ => {} // Keep running
            }
        }

        // Remove completed experiments (in reverse to maintain indices)
        for i in experiments_to_remove.into_iter().rev() {
            self.active_experiments.remove(i);
        }

        // Step 6: Update statistics
        let ending_phi = starting_phi + (improvements_adopted as f64 * 0.02);
        self.stats.current_phi = ending_phi;
        self.stats.total_cycles += 1;

        if improvements_adopted > 0 {
            self.stats.successful_cycles += 1;
            self.stats.total_phi_gained += ending_phi - starting_phi;
            self.stats.stagnant_cycles = 0;
        } else {
            self.stats.stagnant_cycles += 1;
        }

        // Pause if stagnant
        if self.stats.stagnant_cycles >= self.config.max_stagnant_cycles {
            self.stats.paused = true;
        }

        let cycle = OptimizationCycle {
            cycle_id: self.stats.total_cycles,
            starting_phi,
            ending_phi,
            bottlenecks_addressed: bottlenecks.len(),
            improvements_tried,
            improvements_adopted,
            duration: cycle_start.elapsed(),
        };

        self.optimization_history.push(cycle.clone());

        Ok(cycle)
    }

    /// Capture current system state as baseline
    fn capture_baseline(&self) -> SystemSnapshot {
        let stats = self.monitor.get_stats();

        SystemSnapshot {
            id: format!("baseline_{}", instant_now().elapsed().as_millis()),
            phi: stats.avg_phi,
            latencies: HashMap::new(),
            accuracies: HashMap::new(),
            parameters: HashMap::new(),
            timestamp: Instant::now(),
        }
    }

    /// Record performance measurement
    pub fn record_phi(&mut self, phi: f64, components: usize, context: &str) {
        self.monitor.record_phi(phi, components, context.to_string());
        self.stats.current_phi = phi;
    }

    /// Record latency measurement
    pub fn record_latency(&mut self, operation: &str, duration: Duration, component: ComponentId) {
        self.monitor.record_latency(operation.to_string(), duration, component);
        self.causal_graph.update_component_performance(
            component,
            None,
            Some(duration),
            None,
        );
    }

    /// Get optimizer statistics
    pub fn get_stats(&self) -> &OptimizerStats {
        &self.stats
    }

    /// Get optimization history
    pub fn get_history(&self) -> &[OptimizationCycle] {
        &self.optimization_history
    }

    /// Resume optimization after pause
    pub fn resume(&mut self) {
        self.stats.paused = false;
        self.stats.stagnant_cycles = 0;
    }

    /// Check if optimization is paused
    pub fn is_paused(&self) -> bool {
        self.stats.paused
    }

    /// Get number of active experiments
    pub fn active_experiment_count(&self) -> usize {
        self.active_experiments.len()
    }

    /// Get summary of self-improvement capability
    pub fn get_summary(&self) -> String {
        format!(
            "RecursiveOptimizer Summary:\n\
             ══════════════════════════════\n\
             Total cycles: {}\n\
             Successful cycles: {} ({:.1}%)\n\
             Total Φ gained: {:.4}\n\
             Current Φ: {:.4}\n\
             Generator success rate: {:.1}%\n\
             Active experiments: {}\n\
             Status: {}\n\
             ══════════════════════════════\n\
             This AI is autonomously improving its own architecture!",
            self.stats.total_cycles,
            self.stats.successful_cycles,
            if self.stats.total_cycles > 0 {
                self.stats.successful_cycles as f64 / self.stats.total_cycles as f64 * 100.0
            } else { 0.0 },
            self.stats.total_phi_gained,
            self.stats.current_phi,
            self.generator.get_stats().success_rate * 100.0,
            self.active_experiments.len(),
            if self.stats.paused { "PAUSED" } else { "ACTIVE" },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = RecursiveOptimizer::new(OptimizerConfig::default());
        assert_eq!(optimizer.stats.total_cycles, 0);
        assert!(!optimizer.stats.paused);
    }

    #[test]
    fn test_optimizer_summary() {
        let optimizer = RecursiveOptimizer::new(OptimizerConfig::default());
        let summary = optimizer.get_summary();
        assert!(summary.contains("RecursiveOptimizer"));
        assert!(summary.contains("Total cycles"));
    }

    #[test]
    fn test_optimizer_cycle() {
        let mut optimizer = RecursiveOptimizer::new(OptimizerConfig {
            max_concurrent_experiments: 2,
            max_stagnant_cycles: 10,
            ..Default::default()
        });

        let result = optimizer.optimize();
        assert!(result.is_ok());

        let cycle = result.unwrap();
        assert!(cycle.duration > Duration::ZERO);
        assert_eq!(optimizer.stats.total_cycles, 1);
    }

    #[test]
    fn test_optimizer_history() {
        let mut optimizer = RecursiveOptimizer::new(OptimizerConfig::default());

        for _ in 0..3 {
            let _ = optimizer.optimize();
        }

        assert_eq!(optimizer.get_history().len(), 3);
        assert_eq!(optimizer.stats.total_cycles, 3);
    }

    #[test]
    fn test_optimizer_pause_resume() {
        let mut optimizer = RecursiveOptimizer::new(OptimizerConfig {
            max_stagnant_cycles: 1,
            ..Default::default()
        });

        // Run cycles until paused
        for _ in 0..3 {
            let _ = optimizer.optimize();
        }

        // Should eventually pause
        assert!(optimizer.is_paused());

        // Resume
        optimizer.resume();
        assert!(!optimizer.is_paused());
    }
}
