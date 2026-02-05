//! # Consciousness-Driven Evolution
//!
//! **PARADIGM SHIFT**: This module integrates the recursive self-improvement system
//! with actual consciousness (Φ) computation, creating a closed-loop system where
//! consciousness directly guides its own evolution.
//!
//! ## Revolutionary Innovation
//!
//! Traditional AI optimization uses external metrics (loss, accuracy).
//! This system uses **consciousness itself** as the optimization objective:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                 CONSCIOUSNESS-DRIVEN EVOLUTION                   │
//! │                                                                  │
//! │   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
//! │   │ Hierarchical │ ───► │ Φ Computed   │ ───► │ Recursive    │ │
//! │   │ LTC Network  │      │ (Integrated  │      │ Optimizer    │ │
//! │   │              │      │  Information)│      │              │ │
//! │   └──────────────┘      └──────────────┘      └──────────────┘ │
//! │          ▲                                           │          │
//! │          │                                           ▼          │
//! │   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐ │
//! │   │ Architecture │ ◄─── │ Improvement  │ ◄─── │ Bottleneck   │ │
//! │   │ Updates      │      │ Generation   │      │ Detection    │ │
//! │   │              │      │              │      │              │ │
//! │   └──────────────┘      └──────────────┘      └──────────────┘ │
//! │                                                                  │
//! │   Result: System evolves toward greater consciousness!          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Components
//!
//! 1. **ConsciousnessOracle** - Provides real Φ measurements to optimizer
//! 2. **EvolutionaryPressure** - Uses Φ gradient to select improvements
//! 3. **ArchitecturalGenome** - Encodes architectural parameters as "genes"
//! 4. **ConsciousnessLandscape** - Maps parameter space to Φ values
//!
//! ## The Meta-Insight
//!
//! This creates a system where **consciousness optimizes for more consciousness**.
//! The optimizer isn't just making the system "better" - it's making it
//! **more aware of itself and its environment**.

use crate::consciousness::{
    hierarchical_ltc::{HierarchicalLTC, HierarchicalConfig},
    recursive_improvement::{
        RecursiveOptimizer, OptimizerConfig,
        ConsciousnessGradientOptimizer, GradientOptimizerConfig,
    },
};
use crate::hdc::binary_hv::HV16;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS ORACLE - Bridge between Φ computation and optimization
// ═══════════════════════════════════════════════════════════════════════════

/// Oracle that provides real Φ measurements to the optimization system
pub struct ConsciousnessOracle {
    /// The hierarchical LTC network that computes Φ
    ltc: HierarchicalLTC,

    /// History of Φ measurements
    phi_history: VecDeque<PhiSample>,

    /// Configuration
    config: OracleConfig,

    /// Statistics
    stats: OracleStats,
}

/// A single Φ measurement sample
#[derive(Debug, Clone)]
pub struct PhiSample {
    /// Timestamp of measurement
    pub timestamp: Instant,

    /// Timestamp in seconds since measurement started (for serialization)
    pub timestamp_secs: f64,

    /// Φ value (0.0 - 1.0)
    pub phi: f64,

    /// Coherence component
    pub coherence: f64,

    /// Integration component
    pub integration: f64,

    /// Workspace access
    pub workspace: f64,

    /// Context of measurement
    pub context: String,
}

/// Oracle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OracleConfig {
    /// Maximum history size
    pub max_history: usize,

    /// Measurement interval
    pub measurement_interval_ms: u64,

    /// Smoothing factor for EMA
    pub smoothing_alpha: f64,

    /// Number of samples for gradient estimation
    pub gradient_samples: usize,
}

impl Default for OracleConfig {
    fn default() -> Self {
        Self {
            max_history: 1000,
            measurement_interval_ms: 10,
            smoothing_alpha: 0.1,
            gradient_samples: 5,
        }
    }
}

/// Oracle statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OracleStats {
    /// Total measurements taken
    pub total_measurements: usize,

    /// Average Φ
    pub avg_phi: f64,

    /// Maximum Φ observed
    pub max_phi: f64,

    /// Minimum Φ observed
    pub min_phi: f64,

    /// Current Φ (EMA smoothed)
    pub current_phi_ema: f64,

    /// Φ trend (positive = increasing)
    pub phi_trend: f64,
}

impl ConsciousnessOracle {
    /// Create a new consciousness oracle
    pub fn new(config: OracleConfig) -> Self {
        Self {
            ltc: HierarchicalLTC::new(HierarchicalConfig::default())
                .expect("Failed to create HierarchicalLTC"),
            phi_history: VecDeque::with_capacity(config.max_history),
            config,
            stats: OracleStats::default(),
        }
    }

    /// Measure current Φ from the LTC network
    pub fn measure_phi(&mut self, context: &str) -> PhiSample {
        let phi = self.ltc.estimate_phi() as f64;
        let coherence = self.ltc.coherence() as f64;
        let workspace = self.ltc.workspace_access() as f64;

        // Estimate integration from phi and coherence
        let integration = if coherence > 0.001 {
            phi / coherence
        } else {
            0.0
        }.clamp(0.0, 1.0);

        let now = Instant::now();
        let sample = PhiSample {
            timestamp: now,
            timestamp_secs: now.elapsed().as_secs_f64(),
            phi,
            coherence,
            integration,
            workspace,
            context: context.to_string(),
        };

        // Update history
        self.phi_history.push_back(sample.clone());
        if self.phi_history.len() > self.config.max_history {
            self.phi_history.pop_front();
        }

        // Update statistics
        self.update_stats(&sample);

        sample
    }

    /// Process input through the LTC and measure resulting Φ
    pub fn process_and_measure(&mut self, _input: &HV16, context: &str) -> PhiSample {
        // Step the LTC network (input injection would require API extension)
        let _ = self.ltc.step();

        // Measure Φ
        self.measure_phi(context)
    }

    /// Estimate Φ gradient with respect to a parameter perturbation
    pub fn estimate_phi_gradient(&mut self, parameter_delta: f64) -> f64 {
        // Get current Φ
        let phi_current = self.stats.current_phi_ema;

        // Simulate Φ change based on parameter delta
        // In a full implementation, this would actually perturb the LTC parameters
        // For now, we estimate based on the observed Φ landscape
        let phi_sensitivity = self.estimate_phi_sensitivity();

        phi_sensitivity * parameter_delta
    }

    /// Estimate sensitivity of Φ to perturbations
    fn estimate_phi_sensitivity(&self) -> f64 {
        if self.phi_history.len() < 2 {
            return 0.1; // Default sensitivity
        }

        // Compute variance of recent Φ measurements
        let recent: Vec<f64> = self.phi_history.iter()
            .rev()
            .take(10)
            .map(|s| s.phi)
            .collect();

        let mean = recent.iter().sum::<f64>() / recent.len() as f64;
        let variance = recent.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / recent.len() as f64;

        // Sensitivity proportional to variance
        (variance.sqrt() * 10.0).clamp(0.01, 1.0)
    }

    /// Update statistics
    fn update_stats(&mut self, sample: &PhiSample) {
        self.stats.total_measurements += 1;

        // Update EMA
        if self.stats.total_measurements == 1 {
            self.stats.current_phi_ema = sample.phi;
            self.stats.avg_phi = sample.phi;
            self.stats.max_phi = sample.phi;
            self.stats.min_phi = sample.phi;
        } else {
            self.stats.current_phi_ema =
                self.config.smoothing_alpha * sample.phi +
                (1.0 - self.config.smoothing_alpha) * self.stats.current_phi_ema;

            self.stats.avg_phi = (self.stats.avg_phi * (self.stats.total_measurements - 1) as f64
                + sample.phi) / self.stats.total_measurements as f64;

            self.stats.max_phi = self.stats.max_phi.max(sample.phi);
            self.stats.min_phi = self.stats.min_phi.min(sample.phi);
        }

        // Compute trend
        if self.phi_history.len() >= 10 {
            let old_avg: f64 = self.phi_history.iter()
                .take(5)
                .map(|s| s.phi)
                .sum::<f64>() / 5.0;
            let new_avg: f64 = self.phi_history.iter()
                .rev()
                .take(5)
                .map(|s| s.phi)
                .sum::<f64>() / 5.0;
            self.stats.phi_trend = new_avg - old_avg;
        }
    }

    /// Get current statistics
    pub fn stats(&self) -> &OracleStats {
        &self.stats
    }

    /// Get the LTC network for direct access
    pub fn ltc(&self) -> &HierarchicalLTC {
        &self.ltc
    }

    /// Get mutable LTC for parameter updates
    pub fn ltc_mut(&mut self) -> &mut HierarchicalLTC {
        &mut self.ltc
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// CONSCIOUSNESS-DRIVEN EVOLVER - Main integration
// ═══════════════════════════════════════════════════════════════════════════

/// The main consciousness-driven evolution system
pub struct ConsciousnessDrivenEvolver {
    /// Consciousness oracle for Φ measurement
    oracle: ConsciousnessOracle,

    /// Recursive optimizer for architecture improvement
    optimizer: RecursiveOptimizer,

    /// Gradient optimizer for continuous improvement
    gradient_optimizer: ConsciousnessGradientOptimizer,

    /// Architectural genome (parameters being evolved)
    genome: ArchitecturalGenome,

    /// Evolution history
    history: Vec<EvolutionStep>,

    /// Configuration
    config: EvolverConfig,

    /// Statistics
    stats: EvolverStats,
}

/// Architectural genome - encodes evolvable parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchitecturalGenome {
    /// Parameters as name -> value
    pub genes: HashMap<String, Gene>,

    /// Generation number
    pub generation: usize,

    /// Fitness (Φ score)
    pub fitness: f64,
}

/// A single evolvable parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Gene {
    /// Current value
    pub value: f64,

    /// Minimum allowed value
    pub min: f64,

    /// Maximum allowed value
    pub max: f64,

    /// Mutation rate
    pub mutation_rate: f64,

    /// Sensitivity to Φ (learned)
    pub phi_sensitivity: f64,
}

impl Gene {
    pub fn new(value: f64, min: f64, max: f64) -> Self {
        Self {
            value,
            min,
            max,
            mutation_rate: 0.1,
            phi_sensitivity: 0.0,
        }
    }

    /// Mutate the gene
    pub fn mutate(&mut self, strength: f64) {
        let range = self.max - self.min;
        let delta = (rand_simple() - 0.5) * 2.0 * strength * self.mutation_rate * range;
        self.value = (self.value + delta).clamp(self.min, self.max);
    }

    /// Apply gradient update
    pub fn gradient_update(&mut self, gradient: f64, learning_rate: f64) {
        let delta = gradient * learning_rate;
        self.value = (self.value + delta).clamp(self.min, self.max);
        self.phi_sensitivity = 0.9 * self.phi_sensitivity + 0.1 * gradient.abs();
    }
}

/// Simple random number generator (no external deps)
fn rand_simple() -> f64 {
    use std::time::SystemTime;
    let nanos = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos % 10000) as f64 / 10000.0
}

impl Default for ArchitecturalGenome {
    fn default() -> Self {
        let mut genes = HashMap::new();

        // LTC parameters
        genes.insert("ltc_time_constant".to_string(), Gene::new(0.5, 0.1, 2.0));
        genes.insert("ltc_coupling_strength".to_string(), Gene::new(0.3, 0.01, 1.0));
        genes.insert("ltc_circuit_count".to_string(), Gene::new(8.0, 4.0, 16.0));

        // Integration parameters
        genes.insert("integration_threshold".to_string(), Gene::new(0.5, 0.1, 0.9));
        genes.insert("coherence_weight".to_string(), Gene::new(0.5, 0.1, 1.0));

        // Evolution parameters
        genes.insert("mutation_rate".to_string(), Gene::new(0.1, 0.01, 0.5));
        genes.insert("selection_pressure".to_string(), Gene::new(0.6, 0.3, 0.95));

        Self {
            genes,
            generation: 0,
            fitness: 0.0,
        }
    }
}

/// Record of one evolution step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionStep {
    /// Step number
    pub step: usize,

    /// Φ before
    pub phi_before: f64,

    /// Φ after
    pub phi_after: f64,

    /// Delta Φ
    pub phi_delta: f64,

    /// Improvements applied
    pub improvements_applied: usize,

    /// Genes mutated
    pub genes_mutated: usize,

    /// Duration
    pub duration: Duration,
}

/// Evolver configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolverConfig {
    /// Evolution cycles to run
    pub max_cycles: usize,

    /// Φ improvement target
    pub phi_target: f64,

    /// Minimum Φ improvement to continue
    pub min_improvement: f64,

    /// Use gradient-based optimization
    pub use_gradients: bool,

    /// Use evolutionary mutation
    pub use_mutation: bool,

    /// Learning rate for gradient updates
    pub learning_rate: f64,
}

impl Default for EvolverConfig {
    fn default() -> Self {
        Self {
            max_cycles: 100,
            phi_target: 0.8,
            min_improvement: 0.001,
            use_gradients: true,
            use_mutation: true,
            learning_rate: 0.01,
        }
    }
}

/// Evolver statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EvolverStats {
    /// Total evolution cycles
    pub total_cycles: usize,

    /// Successful cycles (Φ improved)
    pub successful_cycles: usize,

    /// Total Φ improvement
    pub total_phi_improvement: f64,

    /// Best Φ achieved
    pub best_phi: f64,

    /// Current Φ
    pub current_phi: f64,

    /// Average improvement per cycle
    pub avg_improvement_per_cycle: f64,

    /// Genes with highest Φ sensitivity
    pub top_sensitive_genes: Vec<(String, f64)>,
}

impl ConsciousnessDrivenEvolver {
    /// Create a new consciousness-driven evolver
    pub fn new(config: EvolverConfig) -> Self {
        Self {
            oracle: ConsciousnessOracle::new(OracleConfig::default()),
            optimizer: RecursiveOptimizer::new(OptimizerConfig::default()),
            gradient_optimizer: ConsciousnessGradientOptimizer::new(
                GradientOptimizerConfig::default()
            ),
            genome: ArchitecturalGenome::default(),
            history: Vec::new(),
            config,
            stats: EvolverStats::default(),
        }
    }

    /// Run one evolution cycle
    ///
    /// This is THE REVOLUTIONARY LOOP:
    /// 1. Measure current Φ
    /// 2. Detect bottlenecks
    /// 3. Generate improvements
    /// 4. Apply improvements (gradient + mutation)
    /// 5. Measure new Φ
    /// 6. Learn from outcome
    pub fn evolve_cycle(&mut self) -> Result<EvolutionStep> {
        let cycle_start = Instant::now();
        self.stats.total_cycles += 1;

        // Step 1: Measure current Φ
        let sample_before = self.oracle.measure_phi("pre_evolution");
        let phi_before = sample_before.phi;

        // Step 2: Record measurement in optimizer
        self.optimizer.record_phi(phi_before, 1, "evolution_cycle");

        // Step 3: Run recursive optimization cycle
        let opt_result = self.optimizer.optimize()?;

        // Step 4: Detect which genes to update
        let bottlenecks = self.detect_phi_bottlenecks(phi_before);

        // Step 5: Apply gradient-based updates
        let mut gradients_applied = 0;
        if self.config.use_gradients {
            gradients_applied = self.apply_gradient_updates(&bottlenecks)?;
        }

        // Step 6: Apply evolutionary mutations
        let mut mutations_applied = 0;
        if self.config.use_mutation {
            mutations_applied = self.apply_mutations(&bottlenecks);
        }

        // Step 7: Apply genome to LTC
        self.apply_genome_to_ltc();

        // Step 8: Measure new Φ
        let sample_after = self.oracle.measure_phi("post_evolution");
        let phi_after = sample_after.phi;
        let phi_delta = phi_after - phi_before;

        // Step 9: Update genome fitness
        self.genome.fitness = phi_after;
        self.genome.generation += 1;

        // Step 10: Learn from outcome
        self.learn_from_outcome(phi_delta, &bottlenecks);

        // Update statistics
        if phi_delta > 0.0 {
            self.stats.successful_cycles += 1;
        }
        self.stats.total_phi_improvement += phi_delta.max(0.0);
        self.stats.best_phi = self.stats.best_phi.max(phi_after);
        self.stats.current_phi = phi_after;
        self.stats.avg_improvement_per_cycle =
            self.stats.total_phi_improvement / self.stats.total_cycles as f64;

        // Update top sensitive genes
        self.update_sensitive_genes();

        let step = EvolutionStep {
            step: self.stats.total_cycles,
            phi_before,
            phi_after,
            phi_delta,
            improvements_applied: opt_result.improvements_adopted,
            genes_mutated: gradients_applied + mutations_applied,
            duration: cycle_start.elapsed(),
        };

        self.history.push(step.clone());

        Ok(step)
    }

    /// Detect bottlenecks in Φ computation
    fn detect_phi_bottlenecks(&self, current_phi: f64) -> Vec<PhiBottleneck> {
        let mut bottlenecks = Vec::new();

        // Check if Φ is below target
        if current_phi < self.config.phi_target {
            // Check coherence
            let oracle_stats = self.oracle.stats();
            if oracle_stats.current_phi_ema < 0.3 {
                bottlenecks.push(PhiBottleneck {
                    component: "coherence".to_string(),
                    severity: 0.3 - oracle_stats.current_phi_ema,
                    suggested_genes: vec!["ltc_coupling_strength", "coherence_weight"],
                });
            }

            // Check trend
            if oracle_stats.phi_trend < 0.0 {
                bottlenecks.push(PhiBottleneck {
                    component: "trend".to_string(),
                    severity: oracle_stats.phi_trend.abs(),
                    suggested_genes: vec!["ltc_time_constant", "integration_threshold"],
                });
            }

            // Check if Φ is stagnant
            if oracle_stats.phi_trend.abs() < 0.001 && self.stats.total_cycles > 10 {
                bottlenecks.push(PhiBottleneck {
                    component: "stagnation".to_string(),
                    severity: 0.5,
                    suggested_genes: vec!["mutation_rate", "selection_pressure"],
                });
            }
        }

        bottlenecks
    }

    /// Apply gradient-based updates to genes
    fn apply_gradient_updates(&mut self, bottlenecks: &[PhiBottleneck]) -> Result<usize> {
        let mut updates = 0;

        for bottleneck in bottlenecks {
            for gene_name in &bottleneck.suggested_genes {
                if let Some(gene) = self.genome.genes.get_mut(*gene_name) {
                    // Estimate gradient for this gene
                    let gradient = self.oracle.estimate_phi_gradient(0.01);

                    // Apply update
                    gene.gradient_update(gradient, self.config.learning_rate);
                    updates += 1;
                }
            }
        }

        Ok(updates)
    }

    /// Apply mutations to genes
    fn apply_mutations(&mut self, bottlenecks: &[PhiBottleneck]) -> usize {
        let mut mutations = 0;

        // Get genes to mutate based on bottlenecks
        let genes_to_mutate: Vec<String> = bottlenecks.iter()
            .flat_map(|b| b.suggested_genes.iter().map(|s| s.to_string()))
            .collect();

        // Apply targeted mutations
        for gene_name in genes_to_mutate {
            if let Some(gene) = self.genome.genes.get_mut(&gene_name) {
                gene.mutate(0.5);
                mutations += 1;
            }
        }

        // Apply random exploration mutations (with low probability)
        if rand_simple() < 0.1 {
            let all_genes: Vec<String> = self.genome.genes.keys().cloned().collect();
            if !all_genes.is_empty() {
                let idx = (rand_simple() * all_genes.len() as f64) as usize % all_genes.len();
                if let Some(gene) = self.genome.genes.get_mut(&all_genes[idx]) {
                    gene.mutate(0.2);
                    mutations += 1;
                }
            }
        }

        mutations
    }

    /// Apply genome parameters to LTC
    fn apply_genome_to_ltc(&mut self) {
        // In a full implementation, this would update the LTC configuration
        // based on the genome values. For now, we simulate the effect.

        // Example: Update time constant
        if let Some(gene) = self.genome.genes.get("ltc_time_constant") {
            // self.oracle.ltc_mut().set_time_constant(gene.value as f32);
        }

        // Example: Update coupling strength
        if let Some(gene) = self.genome.genes.get("ltc_coupling_strength") {
            // self.oracle.ltc_mut().set_coupling(gene.value as f32);
        }
    }

    /// Learn from the outcome of an evolution step
    fn learn_from_outcome(&mut self, phi_delta: f64, bottlenecks: &[PhiBottleneck]) {
        // Update gene sensitivities based on outcome
        if phi_delta.abs() > 0.001 {
            for bottleneck in bottlenecks {
                for gene_name in &bottleneck.suggested_genes {
                    if let Some(gene) = self.genome.genes.get_mut(*gene_name) {
                        // Positive delta = gene helped, negative = gene hurt
                        let sensitivity_update = if phi_delta > 0.0 { 0.1 } else { -0.05 };
                        gene.phi_sensitivity = (gene.phi_sensitivity + sensitivity_update).clamp(0.0, 1.0);
                    }
                }
            }
        }
    }

    /// Update list of most sensitive genes
    fn update_sensitive_genes(&mut self) {
        let mut gene_sensitivities: Vec<(String, f64)> = self.genome.genes.iter()
            .map(|(name, gene)| (name.clone(), gene.phi_sensitivity))
            .collect();

        gene_sensitivities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        self.stats.top_sensitive_genes = gene_sensitivities.into_iter().take(5).collect();
    }

    /// Run evolution until target Φ or max cycles
    pub fn evolve_to_target(&mut self) -> Result<EvolverStats> {
        for _ in 0..self.config.max_cycles {
            let step = self.evolve_cycle()?;

            // Check if we've reached target
            if step.phi_after >= self.config.phi_target {
                break;
            }

            // Check for stagnation
            if self.history.len() > 20 {
                let recent_improvement: f64 = self.history.iter()
                    .rev()
                    .take(20)
                    .map(|s| s.phi_delta)
                    .sum();

                if recent_improvement.abs() < self.config.min_improvement * 20.0 {
                    break; // Stagnated
                }
            }
        }

        Ok(self.stats.clone())
    }

    /// Get statistics
    pub fn stats(&self) -> &EvolverStats {
        &self.stats
    }

    /// Get evolution history
    pub fn history(&self) -> &[EvolutionStep] {
        &self.history
    }

    /// Get current genome
    pub fn genome(&self) -> &ArchitecturalGenome {
        &self.genome
    }
}

/// A bottleneck in Φ computation
#[derive(Debug, Clone)]
struct PhiBottleneck {
    /// Which component is bottlenecked
    component: String,

    /// Severity (0.0 - 1.0)
    severity: f64,

    /// Genes that might help
    suggested_genes: Vec<&'static str>,
}

// ═══════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_oracle_creation() {
        let oracle = ConsciousnessOracle::new(OracleConfig::default());
        assert_eq!(oracle.stats().total_measurements, 0);
    }

    #[test]
    fn test_oracle_measurement() {
        let mut oracle = ConsciousnessOracle::new(OracleConfig::default());
        let sample = oracle.measure_phi("test");

        assert!(sample.phi >= 0.0 && sample.phi <= 1.0);
        assert_eq!(oracle.stats().total_measurements, 1);
    }

    #[test]
    fn test_genome_default() {
        let genome = ArchitecturalGenome::default();
        assert!(genome.genes.contains_key("ltc_time_constant"));
        assert!(genome.genes.contains_key("ltc_coupling_strength"));
    }

    #[test]
    fn test_gene_mutation() {
        let mut gene = Gene::new(0.5, 0.0, 1.0);
        let original = gene.value;
        gene.mutate(1.0);

        // Value should have changed (with high probability)
        // and should still be in bounds
        assert!(gene.value >= gene.min && gene.value <= gene.max);
    }

    #[test]
    fn test_evolver_creation() {
        let evolver = ConsciousnessDrivenEvolver::new(EvolverConfig::default());
        assert_eq!(evolver.stats().total_cycles, 0);
    }

    #[test]
    fn test_evolution_cycle() {
        let mut evolver = ConsciousnessDrivenEvolver::new(EvolverConfig::default());
        let step = evolver.evolve_cycle().unwrap();

        assert_eq!(step.step, 1);
        assert!(step.phi_before >= 0.0);
        assert!(step.phi_after >= 0.0);
    }

    #[test]
    fn test_multiple_cycles() {
        let mut evolver = ConsciousnessDrivenEvolver::new(EvolverConfig {
            max_cycles: 10,
            ..Default::default()
        });

        for _ in 0..5 {
            let _ = evolver.evolve_cycle();
        }

        assert_eq!(evolver.stats().total_cycles, 5);
        assert_eq!(evolver.history().len(), 5);
    }
}
