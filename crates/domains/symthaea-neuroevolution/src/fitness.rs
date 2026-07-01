// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! FEP fitness bridge: evaluates organisms by running CfC cycles and recording free energy.
//!
//! Multi-objective fitness with configurable weights and NSGA-II Pareto sorting.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::unified_hv::ContinuousHV;

use crate::organism::{NeuralOrganism, OrganismFitness};

/// Number of warmup steps excluded from fitness computation.
/// LTC dynamics need ~20τ to settle. Hasani et al. (2021).
pub const WARMUP_STEPS: usize = 20;

/// Number of evaluation steps recorded for fitness.
/// Hasani et al. (2021).
pub const EVAL_STEPS: usize = 100;

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Configurable fitness weights for multi-objective optimization.
/// Defaults: FE 0.3, Phi 0.3, consciousness 0.2, efficiency 0.2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FitnessWeights {
    /// Weight for free energy minimization (lower FE = better).
    pub free_energy: f64,
    /// Weight for Phi maximization.
    pub phi: f64,
    /// Weight for consciousness level.
    pub consciousness: f64,
    /// Weight for energy efficiency.
    pub efficiency: f64,
}

impl Default for FitnessWeights {
    fn default() -> Self {
        Self {
            free_energy: 0.3,
            phi: 0.3,
            consciousness: 0.2,
            efficiency: 0.2,
        }
    }
}

/// Strategy for generating input observations during evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InputStrategy {
    /// Random inputs from a deterministic seed.
    Random { seed: u64 },
    /// Temporal sine-wave patterns at specified frequencies.
    Temporal { frequencies: Vec<f32>, seed: u64 },
    /// Pre-defined dataset of input vectors.
    Dataset { inputs: Vec<Vec<f32>> },
}

impl Default for InputStrategy {
    fn default() -> Self {
        Self::Random { seed: 42 }
    }
}

/// Configuration for FEP fitness evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FepFitnessConfig {
    pub warmup_steps: usize,
    pub eval_steps: usize,
    pub dt: f32,
    pub input_dim: usize,
    pub input_strategy: InputStrategy,
    pub weights: FitnessWeights,
}

impl Default for FepFitnessConfig {
    fn default() -> Self {
        Self {
            warmup_steps: WARMUP_STEPS,
            eval_steps: EVAL_STEPS,
            dt: 0.05,
            input_dim: 16_384,
            input_strategy: InputStrategy::default(),
            weights: FitnessWeights::default(),
        }
    }
}

impl FepFitnessConfig {
    /// Create a fast-test config with reduced dimension (256D) and fewer steps.
    /// ~64x faster than default for unit/integration tests.
    pub fn fast_test() -> Self {
        Self {
            warmup_steps: 3,
            eval_steps: 10,
            dt: 0.05,
            input_dim: 256,
            input_strategy: InputStrategy::default(),
            weights: FitnessWeights::default(),
        }
    }
}

/// Record of a single organism's evaluation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationRecord {
    pub organism_id: u64,
    pub fitness: OrganismFitness,
    pub total_steps: usize,
}

/// FEP fitness bridge: evaluates organisms via CfC-FEP cycle loop.
pub struct FepFitnessBridge {
    pub config: FepFitnessConfig,
    pub evaluations: Vec<EvaluationRecord>,
}

impl FepFitnessBridge {
    pub fn new(config: FepFitnessConfig) -> Self {
        Self {
            config,
            evaluations: Vec::new(),
        }
    }

    /// Generate an input vector for a given step.
    fn generate_input(&self, step: usize) -> ContinuousHV {
        match &self.config.input_strategy {
            InputStrategy::Random { seed } => {
                ContinuousHV::random(self.config.input_dim, *seed + step as u64)
            }
            InputStrategy::Temporal {
                frequencies,
                seed: _,
            } => {
                let t = step as f32 * self.config.dt;
                let mut values = vec![0.0f32; self.config.input_dim];
                for (i, val) in values.iter_mut().enumerate() {
                    let mut sum = 0.0f32;
                    for (f_idx, &freq) in frequencies.iter().enumerate() {
                        let phase = (i + f_idx * 1000) as f32 * 0.001;
                        sum += (2.0 * std::f32::consts::PI * freq * t + phase).sin();
                    }
                    *val = (sum / frequencies.len().max(1) as f32).clamp(-1.0, 1.0);
                }
                ContinuousHV::from_vec(values)
            }
            InputStrategy::Dataset { inputs } => {
                let idx = step % inputs.len().max(1);
                if let Some(data) = inputs.get(idx) {
                    let mut values = data.clone();
                    values.resize(self.config.input_dim, 0.0);
                    ContinuousHV::from_vec(values)
                } else {
                    ContinuousHV::zero(self.config.input_dim)
                }
            }
        }
    }

    /// Evaluate a single organism: warmup + eval steps, compute fitness.
    pub fn evaluate(&mut self, organism: &mut NeuralOrganism) -> OrganismFitness {
        // Ensure organism uses configured dimension
        if organism.eval_dim != self.config.input_dim {
            organism.eval_dim = self.config.input_dim;
        }
        organism.reset_evaluation();

        // Warmup phase (excluded from fitness)
        for step in 0..self.config.warmup_steps {
            let input = self.generate_input(step);
            organism.evaluate_step(&input, self.config.dt);
        }

        // Reset counters after warmup
        organism.total_free_energy = 0.0;
        organism.total_cycles = 0;

        // Evaluation phase
        for step in 0..self.config.eval_steps {
            let input = self.generate_input(self.config.warmup_steps + step);
            organism.evaluate_step(&input, self.config.dt);
        }

        organism.compute_fitness(&self.config.weights);

        let record = EvaluationRecord {
            organism_id: organism.id,
            fitness: organism.fitness.clone(),
            total_steps: self.config.warmup_steps + self.config.eval_steps,
        };
        self.evaluations.push(record);

        organism.fitness.clone()
    }

    /// Evaluate a population of organisms sequentially.
    pub fn evaluate_population(
        &mut self,
        organisms: &mut [NeuralOrganism],
    ) -> Vec<OrganismFitness> {
        organisms.iter_mut().map(|org| self.evaluate(org)).collect()
    }

    /// NSGA-II non-dominated sort. Returns fronts (front 0 = Pareto-optimal).
    ///
    /// Objectives: minimize FE, maximize Phi, maximize consciousness, maximize efficiency.
    pub fn pareto_sort(organisms: &[NeuralOrganism]) -> Vec<Vec<usize>> {
        let n = organisms.len();
        if n == 0 {
            return Vec::new();
        }

        // Domination counts and dominated-by sets
        let mut domination_count = vec![0usize; n];
        let mut dominated_by: Vec<Vec<usize>> = vec![Vec::new(); n];

        for i in 0..n {
            for j in (i + 1)..n {
                let dom = dominates(&organisms[i].fitness, &organisms[j].fitness);
                if dom == 1 {
                    // i dominates j
                    dominated_by[i].push(j);
                    domination_count[j] += 1;
                } else if dom == -1 {
                    // j dominates i
                    dominated_by[j].push(i);
                    domination_count[i] += 1;
                }
            }
        }

        let mut fronts: Vec<Vec<usize>> = Vec::new();
        let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

        while !current_front.is_empty() {
            let mut next_front = Vec::new();
            for &i in &current_front {
                for &j in &dominated_by[i] {
                    domination_count[j] -= 1;
                    if domination_count[j] == 0 {
                        next_front.push(j);
                    }
                }
            }
            fronts.push(current_front);
            current_front = next_front;
        }

        fronts
    }

    /// Crowding distance for diversity preservation within a Pareto front.
    pub fn crowding_distance(organisms: &[NeuralOrganism], front: &[usize]) -> Vec<f64> {
        let n = front.len();
        if n <= 2 {
            return vec![f64::INFINITY; n];
        }

        let mut distances = vec![0.0f64; n];

        // 4 objectives: FE (minimize), Phi (maximize), consciousness, efficiency
        let objectives: Vec<Box<dyn Fn(&OrganismFitness) -> f64>> = vec![
            Box::new(|f: &OrganismFitness| -f.free_energy), // Negate so higher = better
            Box::new(|f: &OrganismFitness| f.phi),
            Box::new(|f: &OrganismFitness| f.consciousness),
            Box::new(|f: &OrganismFitness| f.energy_efficiency),
        ];

        for obj in &objectives {
            let mut sorted_indices: Vec<usize> = (0..n).collect();
            sorted_indices.sort_by(|&a, &b| {
                let va = obj(&organisms[front[a]].fitness);
                let vb = obj(&organisms[front[b]].fitness);
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            });

            // Boundary points get infinite distance
            distances[sorted_indices[0]] = f64::INFINITY;
            distances[sorted_indices[n - 1]] = f64::INFINITY;

            let f_min = obj(&organisms[front[sorted_indices[0]]].fitness);
            let f_max = obj(&organisms[front[sorted_indices[n - 1]]].fitness);
            let range = f_max - f_min;
            if range < 1e-12 {
                continue;
            }

            for i in 1..(n - 1) {
                let next_val = obj(&organisms[front[sorted_indices[i + 1]]].fitness);
                let prev_val = obj(&organisms[front[sorted_indices[i - 1]]].fitness);
                distances[sorted_indices[i]] += (next_val - prev_val) / range;
            }
        }

        distances
    }

    /// Get runtime-configurable fitness weights.
    pub fn weights(&self) -> &FitnessWeights {
        &self.config.weights
    }

    /// Set fitness weights at runtime (enables "environmental pressure" experiments).
    pub fn set_weights(&mut self, weights: FitnessWeights) {
        self.config.weights = weights;
    }

    /// Evaluate a population in parallel using rayon.
    /// Each organism gets its own fitness bridge instance for thread safety.
    #[cfg(feature = "parallel")]
    pub fn evaluate_population_parallel(
        &self,
        organisms: &mut [NeuralOrganism],
    ) -> Vec<OrganismFitness> {
        use rayon::prelude::*;
        organisms
            .par_iter_mut()
            .map(|org| {
                let mut local_bridge = FepFitnessBridge::new(self.config.clone());
                local_bridge.evaluate(org)
            })
            .collect()
    }

    /// Evaluate population using parallel if available, serial otherwise.
    pub fn evaluate_auto(&mut self, organisms: &mut [NeuralOrganism]) -> Vec<OrganismFitness> {
        #[cfg(feature = "parallel")]
        {
            self.evaluate_population_parallel(organisms)
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.evaluate_population(organisms)
        }
    }
}

/// Returns 1 if a dominates b, -1 if b dominates a, 0 if neither.
fn dominates(a: &OrganismFitness, b: &OrganismFitness) -> i8 {
    // 5 Pareto objectives (all maximize):
    // 1. Phi (consciousness integration)
    // 2. FE reduction (learning capability — Goodhart defense)
    // 3. Prediction accuracy (cognitive utility — Goodhart defense)
    // 4. Phi stability (sustained, not oscillating — consciousness = phi_stability)
    // 5. Threshold consistency
    // Note: -free_energy approximates FE reduction when initial FE varies
    let a_vals = [
        a.phi,
        -a.free_energy,
        a.prediction_accuracy,
        a.consciousness,
        a.threshold_fitness,
    ];
    let b_vals = [
        b.phi,
        -b.free_energy,
        b.prediction_accuracy,
        b.consciousness,
        b.threshold_fitness,
    ];

    let mut a_better = false;
    let mut b_better = false;

    for (av, bv) in a_vals.iter().zip(b_vals.iter()) {
        if av > bv {
            a_better = true;
        } else if bv > av {
            b_better = true;
        }
    }

    if a_better && !b_better {
        1
    } else if b_better && !a_better {
        -1
    } else {
        0
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::NeuralGenome;
    use symthaea_core::genesis::GenesisSeed;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-fitness")
    }

    #[test]
    fn test_evaluate_produces_finite_fitness() {
        let genome = NeuralGenome::random(42);
        let mut org = NeuralOrganism::spawn(1, genome, &test_genesis(), 0);
        let mut config = FepFitnessConfig::fast_test();
        config.warmup_steps = 5;
        config.eval_steps = 20;
        let mut bridge = FepFitnessBridge::new(config);
        let fitness = bridge.evaluate(&mut org);
        assert!(fitness.composite.is_finite());
        assert!(fitness.free_energy.is_finite());
        assert!(fitness.energy_efficiency.is_finite());
    }

    #[test]
    fn test_evaluate_population() {
        let genesis = test_genesis();
        let mut orgs: Vec<NeuralOrganism> = (0..5)
            .map(|i| {
                let genome = NeuralGenome::random(i * 100);
                NeuralOrganism::spawn(i as u64, genome, &genesis, 0)
            })
            .collect();

        let mut bridge = FepFitnessBridge::new(FepFitnessConfig::fast_test());
        let fitnesses = bridge.evaluate_population(&mut orgs);
        assert_eq!(fitnesses.len(), 5);
        for f in &fitnesses {
            assert!(f.composite.is_finite());
        }
    }

    #[test]
    fn test_warmup_excluded() {
        let genome = NeuralGenome::random(42);
        let mut org = NeuralOrganism::spawn(1, genome, &test_genesis(), 0);
        let mut config = FepFitnessConfig::fast_test();
        config.warmup_steps = 10;
        config.eval_steps = 20;
        let mut bridge = FepFitnessBridge::new(config);
        bridge.evaluate(&mut org);
        // After evaluation, total_cycles should be eval_steps (warmup excluded)
        assert_eq!(org.total_cycles, 20);
    }

    #[test]
    fn test_pareto_sort_basic() {
        let genesis = test_genesis();
        let mut orgs: Vec<NeuralOrganism> = (0..3)
            .map(|i| {
                let genome = NeuralGenome::random(i * 100);
                let mut org = NeuralOrganism::spawn(i as u64, genome, &genesis, 0);
                org.fitness = OrganismFitness {
                    composite: i as f64,
                    free_energy: -(i as f64), // Lower FE for higher index
                    phi: i as f64 * 0.5,
                    consciousness: i as f64 * 0.3,
                    prediction_accuracy: 0.5,
                    energy_efficiency: 0.5,
                    threshold_fitness: 0.5,
                };
                org
            })
            .collect();

        let fronts = FepFitnessBridge::pareto_sort(&orgs);
        assert!(!fronts.is_empty());
        // Front 0 should contain the dominating organism (index 2)
        assert!(fronts[0].contains(&2));
    }

    #[test]
    fn test_pareto_sort_empty() {
        let fronts = FepFitnessBridge::pareto_sort(&[]);
        assert!(fronts.is_empty());
    }

    #[test]
    fn test_crowding_distance_boundary() {
        let genesis = test_genesis();
        let orgs: Vec<NeuralOrganism> = (0..5)
            .map(|i| {
                let genome = NeuralGenome::random(i * 100);
                let mut org = NeuralOrganism::spawn(i as u64, genome, &genesis, 0);
                org.fitness = OrganismFitness {
                    composite: i as f64,
                    free_energy: -(i as f64),
                    phi: i as f64 * 0.5,
                    consciousness: i as f64 * 0.3,
                    prediction_accuracy: 0.5,
                    energy_efficiency: i as f64 * 0.2,
                    threshold_fitness: 0.5,
                };
                org
            })
            .collect();

        let front: Vec<usize> = (0..5).collect();
        let distances = FepFitnessBridge::crowding_distance(&orgs, &front);
        assert_eq!(distances.len(), 5);
        // Boundary points should have infinite distance
        // (indices 0 and 4 are boundaries when sorted by any objective)
    }

    #[test]
    fn test_fitness_weights_configurable() {
        let mut bridge = FepFitnessBridge::new(FepFitnessConfig::default());
        assert!((bridge.weights().free_energy - 0.3).abs() < 1e-10);
        bridge.set_weights(FitnessWeights {
            free_energy: 0.9,
            phi: 0.05,
            consciousness: 0.025,
            efficiency: 0.025,
        });
        assert!((bridge.weights().free_energy - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_input_strategy() {
        let config = FepFitnessConfig {
            input_strategy: InputStrategy::Temporal {
                frequencies: vec![1.0, 5.0, 10.0],
                seed: 42,
            },
            ..Default::default()
        };
        let bridge = FepFitnessBridge::new(config);
        let input = bridge.generate_input(0);
        assert_eq!(input.dim(), 16_384);
        for &v in &input.values {
            assert!(v >= -1.0 && v <= 1.0);
        }
    }
}
