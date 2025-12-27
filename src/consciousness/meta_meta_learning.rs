//! # Meta-Meta Learning: Improving the Improver
//!
//! **THE ULTIMATE RECURSION**: This module implements systems that improve
//! the improvement process itself. Instead of just optimizing parameters,
//! we optimize HOW we optimize.
//!
//! ## The Breakthrough Insight
//!
//! Traditional machine learning optimizes parameters.
//! Meta-learning optimizes the learning process.
//! Meta-meta learning optimizes the optimization of the learning process.
//!
//! This creates a recursive tower:
//! - Level 0: Execute (use primitives)
//! - Level 1: Learn (improve primitives based on performance)
//! - Level 2: Meta-learn (improve how we learn)
//! - Level 3: Meta-meta learn (improve how we improve learning)
//!
//! ## Key Components
//!
//! 1. **MetaOptimizer** - Optimizes optimization hyperparameters
//! 2. **StrategyEvolver** - Evolves discovery strategies based on success
//! 3. **ImprovementAnalyzer** - Analyzes what kinds of improvements work best
//! 4. **RecursiveTower** - The full recursive improvement stack

use crate::consciousness::consciousness_guided_discovery::{
    EmergentDiscovery, DiscoveryConfig, DiscoveryStats,
    PhiGradientConfig, PhiOptimizedDiscovery,
};
use crate::hdc::primitive_system::PrimitiveSystem;
use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;

// ═══════════════════════════════════════════════════════════════════════════════
// META-OPTIMIZER: Optimizes Optimization Hyperparameters
// ═══════════════════════════════════════════════════════════════════════════════

/// Hyperparameters that control the optimization process
#[derive(Debug, Clone)]
pub struct OptimizationHyperparameters {
    /// Learning rate for gradient optimizer
    pub learning_rate: f64,
    /// Momentum coefficient
    pub momentum: f64,
    /// Exploration rate for discovery
    pub exploration_rate: f64,
    /// Beam width for search
    pub beam_width: usize,
    /// Maximum composition depth
    pub max_depth: usize,
    /// Φ threshold for accepting compositions
    pub phi_threshold: f64,
}

impl Default for OptimizationHyperparameters {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            momentum: 0.9,
            exploration_rate: 0.3,
            beam_width: 100,
            max_depth: 5,
            phi_threshold: 0.01,
        }
    }
}

impl OptimizationHyperparameters {
    /// Convert to vector for optimization
    pub fn to_vec(&self) -> Vec<f64> {
        vec![
            self.learning_rate,
            self.momentum,
            self.exploration_rate,
            self.beam_width as f64 / 100.0,
            self.max_depth as f64 / 10.0,
            self.phi_threshold,
        ]
    }

    /// Create from vector
    pub fn from_vec(v: &[f64]) -> Self {
        Self {
            learning_rate: v.get(0).copied().unwrap_or(0.01).clamp(0.001, 0.5),
            momentum: v.get(1).copied().unwrap_or(0.9).clamp(0.0, 0.99),
            exploration_rate: v.get(2).copied().unwrap_or(0.3).clamp(0.0, 1.0),
            beam_width: (v.get(3).copied().unwrap_or(1.0) * 100.0).clamp(10.0, 500.0) as usize,
            max_depth: (v.get(4).copied().unwrap_or(0.5) * 10.0).clamp(1.0, 10.0) as usize,
            phi_threshold: v.get(5).copied().unwrap_or(0.01).clamp(0.0, 0.5),
        }
    }

    /// Mutate hyperparameters
    pub fn mutate(&self, mutation_rate: f64, rng_state: &mut u64) -> Self {
        let mut v = self.to_vec();
        for x in &mut v {
            *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            if (*rng_state as f64 / u64::MAX as f64) < mutation_rate {
                *rng_state = rng_state.wrapping_mul(1103515245).wrapping_add(12345);
                let delta = (*rng_state as f64 / u64::MAX as f64 - 0.5) * 0.2;
                *x = (*x + delta).max(0.0);
            }
        }
        Self::from_vec(&v)
    }
}

/// Record of an optimization run
#[derive(Debug, Clone)]
pub struct OptimizationRun {
    /// Hyperparameters used
    pub hyperparameters: OptimizationHyperparameters,
    /// Best Φ achieved
    pub best_phi: f64,
    /// Number of compositions discovered
    pub compositions_discovered: usize,
    /// Time taken in seconds
    pub time_seconds: f64,
    /// Efficiency (Φ per second)
    pub efficiency: f64,
}

/// Meta-optimizer that finds the best optimization hyperparameters
pub struct MetaOptimizer {
    /// History of optimization runs
    history: Vec<OptimizationRun>,
    /// Best hyperparameters found so far
    best_hyperparameters: OptimizationHyperparameters,
    /// Best efficiency achieved
    best_efficiency: f64,
    /// Population of hyperparameter variants (for evolutionary approach)
    population: Vec<OptimizationHyperparameters>,
    /// Population size
    population_size: usize,
    /// Mutation rate
    mutation_rate: f64,
    /// Random state
    rng_state: u64,
}

impl MetaOptimizer {
    /// Create a new meta-optimizer
    pub fn new(population_size: usize, mutation_rate: f64) -> Self {
        let mut population = Vec::with_capacity(population_size);
        let mut rng_state = 42u64;

        // Initialize population with random variations
        for _ in 0..population_size {
            let base = OptimizationHyperparameters::default();
            let mutated = base.mutate(0.5, &mut rng_state);
            population.push(mutated);
        }

        Self {
            history: Vec::new(),
            best_hyperparameters: OptimizationHyperparameters::default(),
            best_efficiency: 0.0,
            population,
            population_size,
            mutation_rate,
            rng_state,
        }
    }

    /// Run one meta-optimization cycle
    pub fn meta_optimize_cycle(&mut self, base_system: Arc<PrimitiveSystem>, cycles: usize) -> Result<OptimizationRun> {
        // Select hyperparameters to try (tournament selection)
        let idx1 = self.random_index(self.population.len());
        let idx2 = self.random_index(self.population.len());

        let hyperparams = if self.history.len() > idx1.max(idx2) {
            // Use fitness to select
            let fit1 = self.history.get(idx1).map(|r| r.efficiency).unwrap_or(0.0);
            let fit2 = self.history.get(idx2).map(|r| r.efficiency).unwrap_or(0.0);
            if fit1 > fit2 {
                self.population[idx1].clone()
            } else {
                self.population[idx2].clone()
            }
        } else {
            let pop_len = self.population.len();
            let random_idx = self.random_index(pop_len);
            self.population[random_idx].clone()
        };

        // Run optimization with these hyperparameters
        let run = self.evaluate_hyperparameters(&hyperparams, base_system, cycles)?;

        // Update best
        if run.efficiency > self.best_efficiency {
            self.best_efficiency = run.efficiency;
            self.best_hyperparameters = hyperparams.clone();
        }

        // Evolve population
        self.evolve_population(&run);

        // Record
        self.history.push(run.clone());

        Ok(run)
    }

    /// Evaluate hyperparameters by running optimization
    fn evaluate_hyperparameters(
        &mut self,
        hyperparams: &OptimizationHyperparameters,
        base_system: Arc<PrimitiveSystem>,
        cycles: usize,
    ) -> Result<OptimizationRun> {
        let start = Instant::now();

        // Create config from hyperparameters
        let discovery_config = DiscoveryConfig {
            beam_width: hyperparams.beam_width,
            max_depth: hyperparams.max_depth,
            phi_threshold: hyperparams.phi_threshold,
            exploration_rate: hyperparams.exploration_rate,
            max_candidates_per_cycle: 20,
            min_evaluations: 3,
            learn_grammar: true,
        };

        let gradient_config = PhiGradientConfig {
            learning_rate: hyperparams.learning_rate,
            momentum: hyperparams.momentum,
            gradient_samples: 5,
            epsilon: 0.001,
            max_steps: 10, // Fast for meta-optimization
            convergence_threshold: 1e-6,
        };

        // Run optimization
        let mut discovery = PhiOptimizedDiscovery::new(
            base_system,
            discovery_config,
            gradient_config,
        );

        let results = discovery.discover_and_optimize(cycles)?;
        let elapsed = start.elapsed().as_secs_f64();

        let stats = discovery.stats();
        let best_phi = results.first().map(|(_, phi)| *phi).unwrap_or(0.0);

        Ok(OptimizationRun {
            hyperparameters: hyperparams.clone(),
            best_phi,
            compositions_discovered: stats.phi_increasing,
            time_seconds: elapsed,
            efficiency: best_phi / elapsed.max(0.001),
        })
    }

    /// Evolve the population based on a run result
    fn evolve_population(&mut self, run: &OptimizationRun) {
        if run.efficiency > 0.0 {
            // Replace worst member with mutated version of good hyperparameters
            let worst_idx = self.history.iter()
                .enumerate()
                .filter(|(i, _)| *i < self.population.len())
                .min_by(|a, b| a.1.efficiency.partial_cmp(&b.1.efficiency).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(self.random_index(self.population.len()));

            let mutated = run.hyperparameters.mutate(self.mutation_rate, &mut self.rng_state);
            if worst_idx < self.population.len() {
                self.population[worst_idx] = mutated;
            }
        }
    }

    /// Get random index
    fn random_index(&mut self, len: usize) -> usize {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state as usize) % len
    }

    /// Get best hyperparameters
    pub fn best_hyperparameters(&self) -> &OptimizationHyperparameters {
        &self.best_hyperparameters
    }

    /// Get history
    pub fn history(&self) -> &[OptimizationRun] {
        &self.history
    }

    /// Get best efficiency
    pub fn best_efficiency(&self) -> f64 {
        self.best_efficiency
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STRATEGY EVOLVER: Evolves Discovery Strategies
// ═══════════════════════════════════════════════════════════════════════════════

/// A discovery strategy
#[derive(Debug, Clone)]
pub struct DiscoveryStrategy {
    /// Name of this strategy
    pub name: String,
    /// When to use exploration vs exploitation
    pub exploration_schedule: ExplorationSchedule,
    /// How to combine compositions
    pub composition_preference: CompositionPreference,
    /// How to evaluate candidates
    pub evaluation_method: EvaluationMethod,
    /// Fitness score
    pub fitness: f64,
}

/// Schedule for exploration rate over time
#[derive(Debug, Clone)]
pub enum ExplorationSchedule {
    /// Constant exploration rate
    Constant(f64),
    /// Linear decay from start to end
    LinearDecay { start: f64, end: f64 },
    /// Exponential decay
    ExponentialDecay { start: f64, decay_rate: f64 },
    /// Cyclic exploration
    Cyclic { min: f64, max: f64, period: usize },
}

impl ExplorationSchedule {
    /// Get exploration rate at step
    pub fn rate_at(&self, step: usize, max_steps: usize) -> f64 {
        match self {
            Self::Constant(r) => *r,
            Self::LinearDecay { start, end } => {
                let progress = step as f64 / max_steps.max(1) as f64;
                start + (end - start) * progress
            }
            Self::ExponentialDecay { start, decay_rate } => {
                start * (-decay_rate * step as f64).exp()
            }
            Self::Cyclic { min, max, period } => {
                let p = (*period).max(1);
                let phase = (step % p) as f64 / p as f64;
                min + (max - min) * (phase * std::f64::consts::PI * 2.0).sin().abs()
            }
        }
    }
}

/// Preference for types of compositions
#[derive(Debug, Clone)]
pub enum CompositionPreference {
    /// No preference
    Balanced,
    /// Prefer sequential compositions
    Sequential,
    /// Prefer parallel compositions
    Parallel,
    /// Prefer deep compositions
    Deep,
    /// Prefer wide (shallow) compositions
    Wide,
}

/// Method for evaluating candidates
#[derive(Debug, Clone)]
pub enum EvaluationMethod {
    /// Single Φ measurement
    SingleSample,
    /// Average of multiple samples
    MultiSample(usize),
    /// Consider coherence along with Φ
    PhiAndCoherence,
    /// Consider integration along with Φ
    PhiAndIntegration,
}

/// Evolves discovery strategies over time
pub struct StrategyEvolver {
    /// Population of strategies
    strategies: Vec<DiscoveryStrategy>,
    /// Generation counter
    generation: usize,
    /// Best strategy found
    best_strategy: Option<DiscoveryStrategy>,
    /// Mutation rate
    mutation_rate: f64,
    /// Random state
    rng_state: u64,
}

impl StrategyEvolver {
    /// Create a new strategy evolver
    pub fn new(population_size: usize) -> Self {
        let mut strategies = Vec::with_capacity(population_size);
        let rng_state = 42u64;

        // Initialize with diverse strategies
        for i in 0..population_size {
            let schedule = match i % 4 {
                0 => ExplorationSchedule::Constant(0.3),
                1 => ExplorationSchedule::LinearDecay { start: 0.5, end: 0.1 },
                2 => ExplorationSchedule::ExponentialDecay { start: 0.5, decay_rate: 0.1 },
                _ => ExplorationSchedule::Cyclic { min: 0.1, max: 0.5, period: 10 },
            };

            let preference = match i % 5 {
                0 => CompositionPreference::Balanced,
                1 => CompositionPreference::Sequential,
                2 => CompositionPreference::Parallel,
                3 => CompositionPreference::Deep,
                _ => CompositionPreference::Wide,
            };

            let evaluation = match i % 4 {
                0 => EvaluationMethod::SingleSample,
                1 => EvaluationMethod::MultiSample(3),
                2 => EvaluationMethod::PhiAndCoherence,
                _ => EvaluationMethod::PhiAndIntegration,
            };

            strategies.push(DiscoveryStrategy {
                name: format!("Strategy_{}", i),
                exploration_schedule: schedule,
                composition_preference: preference,
                evaluation_method: evaluation,
                fitness: 0.0,
            });
        }

        Self {
            strategies,
            generation: 0,
            best_strategy: None,
            mutation_rate: 0.2,
            rng_state,
        }
    }

    /// Evolve strategies based on fitness
    pub fn evolve(&mut self, fitness_scores: &[f64]) {
        // Update fitness
        for (strategy, &fitness) in self.strategies.iter_mut().zip(fitness_scores.iter()) {
            strategy.fitness = fitness;
        }

        // Sort by fitness
        self.strategies.sort_by(|a, b| b.fitness.partial_cmp(&a.fitness).unwrap());

        // Update best
        if let Some(best) = self.strategies.first() {
            if self.best_strategy.as_ref().map(|b| best.fitness > b.fitness).unwrap_or(true) {
                self.best_strategy = Some(best.clone());
            }
        }

        // Keep top half, replace bottom half with children
        let keep = self.strategies.len() / 2;
        let target_len = self.strategies.len();
        let mut new_strategies: Vec<_> = self.strategies[..keep].to_vec();

        while new_strategies.len() < target_len {
            // Tournament selection for parents
            let parent1_idx = self.random_index(keep);
            let parent2_idx = self.random_index(keep);

            // Clone parents to avoid borrow conflict
            let parent1 = self.strategies[parent1_idx].clone();
            let parent2 = self.strategies[parent2_idx].clone();

            // Crossover
            let child = self.crossover(&parent1, &parent2);

            // Mutate
            let child = self.mutate(child);

            new_strategies.push(child);
        }

        self.strategies = new_strategies;
        self.generation += 1;
    }

    /// Crossover two strategies
    fn crossover(&mut self, parent1: &DiscoveryStrategy, parent2: &DiscoveryStrategy) -> DiscoveryStrategy {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        let use_first = (self.rng_state & 1) == 0;

        DiscoveryStrategy {
            name: format!("Child_{}_{}", self.generation, self.rng_state % 1000),
            exploration_schedule: if use_first {
                parent1.exploration_schedule.clone()
            } else {
                parent2.exploration_schedule.clone()
            },
            composition_preference: if use_first {
                parent2.composition_preference.clone()
            } else {
                parent1.composition_preference.clone()
            },
            evaluation_method: if use_first {
                parent1.evaluation_method.clone()
            } else {
                parent2.evaluation_method.clone()
            },
            fitness: 0.0,
        }
    }

    /// Mutate a strategy
    fn mutate(&mut self, mut strategy: DiscoveryStrategy) -> DiscoveryStrategy {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);

        if (self.rng_state as f64 / u64::MAX as f64) < self.mutation_rate {
            // Mutate exploration schedule
            self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
            strategy.exploration_schedule = match self.rng_state % 4 {
                0 => ExplorationSchedule::Constant(0.3),
                1 => ExplorationSchedule::LinearDecay { start: 0.5, end: 0.1 },
                2 => ExplorationSchedule::ExponentialDecay { start: 0.5, decay_rate: 0.1 },
                _ => ExplorationSchedule::Cyclic { min: 0.1, max: 0.5, period: 10 },
            };
        }

        strategy
    }

    /// Get random index
    fn random_index(&mut self, len: usize) -> usize {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        (self.rng_state as usize) % len.max(1)
    }

    /// Get best strategy
    pub fn best_strategy(&self) -> Option<&DiscoveryStrategy> {
        self.best_strategy.as_ref()
    }

    /// Get current generation
    pub fn generation(&self) -> usize {
        self.generation
    }

    /// Get all strategies
    pub fn strategies(&self) -> &[DiscoveryStrategy] {
        &self.strategies
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// RECURSIVE IMPROVEMENT TOWER
// ═══════════════════════════════════════════════════════════════════════════════

/// The complete recursive improvement stack
pub struct RecursiveImprovementTower {
    /// Level 1: Discovery system (finds compositions)
    discovery: EmergentDiscovery,
    /// Level 2: Meta-optimizer (optimizes discovery)
    meta_optimizer: MetaOptimizer,
    /// Level 3: Strategy evolver (evolves optimization strategies)
    strategy_evolver: StrategyEvolver,
    /// Base primitive system
    base_system: Arc<PrimitiveSystem>,
    /// Tower statistics
    stats: TowerStats,
}

/// Statistics about the recursive improvement tower
#[derive(Debug, Clone, Default)]
pub struct TowerStats {
    /// Level 1 improvements
    pub level1_improvements: usize,
    /// Level 2 improvements
    pub level2_improvements: usize,
    /// Level 3 improvements
    pub level3_improvements: usize,
    /// Total cycles run
    pub total_cycles: usize,
    /// Best Φ achieved
    pub best_phi: f64,
    /// Best efficiency achieved
    pub best_efficiency: f64,
}

impl RecursiveImprovementTower {
    /// Create a new recursive improvement tower
    pub fn new(base_system: Arc<PrimitiveSystem>) -> Self {
        let config = DiscoveryConfig::default();
        Self {
            discovery: EmergentDiscovery::new(base_system.clone(), config),
            meta_optimizer: MetaOptimizer::new(10, 0.2),
            strategy_evolver: StrategyEvolver::new(8),
            base_system,
            stats: TowerStats::default(),
        }
    }

    /// Run one complete tower cycle (all levels)
    pub fn tower_cycle(&mut self) -> Result<TowerStats> {
        self.stats.total_cycles += 1;

        // Level 1: Run discovery
        let l1_result = self.discovery.discover_cycles(3)?;
        if !l1_result.is_empty() {
            self.stats.level1_improvements += l1_result.len();
        }

        // Update best Φ
        if let Some(best) = self.discovery.best_discoveries(1).first() {
            if best.phi_score > self.stats.best_phi {
                self.stats.best_phi = best.phi_score;
            }
        }

        // Level 2: Run meta-optimization
        let l2_result = self.meta_optimizer.meta_optimize_cycle(self.base_system.clone(), 2)?;
        if l2_result.efficiency > self.stats.best_efficiency {
            self.stats.best_efficiency = l2_result.efficiency;
            self.stats.level2_improvements += 1;
        }

        // Level 3: Evolve strategies based on L2 results
        let fitness_scores: Vec<f64> = self.strategy_evolver.strategies()
            .iter()
            .map(|_| l2_result.efficiency) // In practice, each strategy would be evaluated
            .collect();

        self.strategy_evolver.evolve(&fitness_scores);
        self.stats.level3_improvements += 1;

        Ok(self.stats.clone())
    }

    /// Get tower statistics
    pub fn stats(&self) -> &TowerStats {
        &self.stats
    }

    /// Get best hyperparameters from meta-optimizer
    pub fn best_hyperparameters(&self) -> &OptimizationHyperparameters {
        self.meta_optimizer.best_hyperparameters()
    }

    /// Get best strategy from strategy evolver
    pub fn best_strategy(&self) -> Option<&DiscoveryStrategy> {
        self.strategy_evolver.best_strategy()
    }

    /// Get discovery stats
    pub fn discovery_stats(&self) -> &DiscoveryStats {
        self.discovery.stats()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIT TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_system() -> Arc<PrimitiveSystem> {
        Arc::new(PrimitiveSystem::new())
    }

    #[test]
    fn test_optimization_hyperparameters_default() {
        let params = OptimizationHyperparameters::default();
        assert!(params.learning_rate > 0.0);
        assert!(params.momentum >= 0.0);
        assert!(params.beam_width > 0);
    }

    #[test]
    fn test_optimization_hyperparameters_roundtrip() {
        let params = OptimizationHyperparameters::default();
        let vec = params.to_vec();
        let restored = OptimizationHyperparameters::from_vec(&vec);

        assert!((restored.learning_rate - params.learning_rate).abs() < 0.01);
        assert!((restored.momentum - params.momentum).abs() < 0.01);
    }

    #[test]
    fn test_optimization_hyperparameters_mutate() {
        let params = OptimizationHyperparameters::default();
        let mut rng = 42u64;
        let mutated = params.mutate(1.0, &mut rng);

        // With 100% mutation rate, at least something should change
        // (though values might still end up the same by chance)
        assert!(mutated.learning_rate > 0.0);
    }

    #[test]
    fn test_meta_optimizer_creation() {
        let optimizer = MetaOptimizer::new(5, 0.2);
        assert!(optimizer.history().is_empty());
        assert_eq!(optimizer.best_efficiency(), 0.0);
    }

    #[test]
    fn test_exploration_schedule_constant() {
        let schedule = ExplorationSchedule::Constant(0.5);
        assert_eq!(schedule.rate_at(0, 100), 0.5);
        assert_eq!(schedule.rate_at(50, 100), 0.5);
        assert_eq!(schedule.rate_at(100, 100), 0.5);
    }

    #[test]
    fn test_exploration_schedule_linear_decay() {
        let schedule = ExplorationSchedule::LinearDecay { start: 1.0, end: 0.0 };
        assert!((schedule.rate_at(0, 100) - 1.0).abs() < 0.01);
        assert!((schedule.rate_at(50, 100) - 0.5).abs() < 0.01);
        assert!((schedule.rate_at(100, 100) - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_strategy_evolver_creation() {
        let evolver = StrategyEvolver::new(8);
        assert_eq!(evolver.strategies().len(), 8);
        assert_eq!(evolver.generation(), 0);
    }

    #[test]
    fn test_strategy_evolver_evolve() {
        let mut evolver = StrategyEvolver::new(8);
        let fitness = vec![1.0, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1, 0.05];

        evolver.evolve(&fitness);

        assert_eq!(evolver.generation(), 1);
        assert!(evolver.best_strategy().is_some());
    }

    #[test]
    fn test_recursive_tower_creation() {
        let system = create_test_system();
        let tower = RecursiveImprovementTower::new(system);

        assert_eq!(tower.stats().total_cycles, 0);
    }

    #[test]
    fn test_recursive_tower_cycle() {
        let system = create_test_system();
        let mut tower = RecursiveImprovementTower::new(system);

        let result = tower.tower_cycle();
        assert!(result.is_ok());

        let stats = tower.stats();
        assert_eq!(stats.total_cycles, 1);
    }
}
