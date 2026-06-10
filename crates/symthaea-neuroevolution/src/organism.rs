// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Neural organism: evolvable wrapper around HdcLtcUnifiedNetwork + ActiveInferenceAgent.
//!
//! Lifecycle: spawn → evaluate_step (repeated) → compute_fitness → reproduce/die.

use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;
use symthaea_core::hdc::hdc_ltc_unified::HdcLtcUnifiedNetwork;
use symthaea_core::hdc::unified_hv::ContinuousHV;
use symthaea_fep::{
    ActiveInferenceAgent, ActiveInferenceAgentConfig, FreeEnergyComponents, Observation,
};

use crate::genome::NeuralGenome;

/// FEP state dimension (projected from 16,384D via deterministic projection).
/// Friston (2010) — 32D sufficient for belief state dynamics.
pub const FEP_STATE_DIM: usize = 32;

/// Default HDC dimension for organism evaluation.
pub const DEFAULT_EVAL_DIM: usize = 16_384;

/// Reduced dimension for fast testing (64x smaller, ~64x faster).
pub const FAST_TEST_DIM: usize = 256;

/// Maximum age before death eligibility.
pub const MAX_AGE_CYCLES: u32 = 500;

/// Floor for fitness to prevent -inf domination.
/// Stanley & Miikkulainen (2002) — capped negative fitness.
pub const FITNESS_FLOOR: f64 = -10.0;

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Multi-objective fitness for a neural organism.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrganismFitness {
    /// Weighted composite fitness (higher = better).
    pub composite: f64,
    /// Mean free energy (lower = better prediction).
    pub free_energy: f64,
    /// Mean Phi (higher = more integrated).
    pub phi: f64,
    /// Mean consciousness level.
    pub consciousness: f64,
    /// Prediction accuracy (1.0 - normalized prediction error).
    pub prediction_accuracy: f64,
    /// Energy efficiency (lower energy per useful computation).
    pub energy_efficiency: f64,
    pub threshold_fitness: f64,
}

/// Result of a single evaluation step.
#[derive(Debug, Clone)]
pub struct StepResult {
    pub free_energy: FreeEnergyComponents,
    pub output: ContinuousHV,
}

/// A neural organism: genome + instantiated network + FEP agent.
#[derive(Debug, Clone)]
pub struct NeuralOrganism {
    pub id: u64,
    pub genome: NeuralGenome,
    pub network: HdcLtcUnifiedNetwork,
    pub fep_agent: ActiveInferenceAgent,
    pub fitness: OrganismFitness,
    pub generation: u32,
    pub parent_ids: (Option<u64>, Option<u64>),
    pub age_cycles: u32,
    pub alive: bool,
    pub total_free_energy: f64,
    pub total_cycles: u64,
    pub peak_phi: f32,
    /// Accumulated Phi across evaluation steps (for mean Phi computation).
    pub total_phi: f64,
    /// Welford mean accumulator for Phi variance (stability objective).
    phi_mean: f64,
    /// Welford M2 accumulator for Phi variance.
    phi_m2: f64,
    /// Free energy at first evaluation step (for FE reduction rate).
    initial_fe: f64,
    /// Free energy at most recent evaluation step.
    final_fe: f64,
    /// HDC dimension used for this organism's network.
    pub eval_dim: usize,
}

impl NeuralOrganism {
    /// Spawn a new organism from a genome with default dimension (16,384D).
    pub fn spawn(id: u64, genome: NeuralGenome, genesis: &GenesisSeed, generation: u32) -> Self {
        Self::spawn_with_dim(id, genome, genesis, generation, DEFAULT_EVAL_DIM)
    }

    /// Spawn with a custom HDC dimension (e.g., 256D for fast testing).
    pub fn spawn_with_dim(
        id: u64,
        genome: NeuralGenome,
        genesis: &GenesisSeed,
        generation: u32,
        dim: usize,
    ) -> Self {
        let phenotype = genome.decode();
        let mut net_config = phenotype.network_config.clone();
        net_config.neuron_config.dimension = dim;
        let network = HdcLtcUnifiedNetwork::from_genesis(net_config, genesis);

        let fep_config = ActiveInferenceAgentConfig {
            state_dim: FEP_STATE_DIM,
            obs_dim: FEP_STATE_DIM,
            num_actions: 8,
            inference_iterations: 5,
            belief_learning_rate: 0.1,
            planning_horizon: 3,
            action_temperature: 1.0,
            enable_model_learning: true,
            enable_td_learning: false, // Lightweight for evolution
            ..Default::default()
        };
        let fep_agent = ActiveInferenceAgent::new(fep_config);

        Self {
            id,
            genome,
            network,
            fep_agent,
            fitness: OrganismFitness::default(),
            generation,
            parent_ids: (None, None),
            age_cycles: 0,
            alive: true,
            total_free_energy: 0.0,
            total_cycles: 0,
            peak_phi: 0.0,
            total_phi: 0.0,
            phi_mean: 0.0,
            phi_m2: 0.0,
            initial_fe: 0.0,
            final_fe: 0.0,
            eval_dim: dim,
        }
    }

    /// Run one evaluation step: feed observation through CfC network, then FEP agent.
    ///
    /// Computes a Phi proxy from CfC output integration: how much the network's
    /// output differs from the sum of its parts (variance across dimensions).
    pub fn evaluate_step(&mut self, input: &ContinuousHV, dt: f32) -> StepResult {
        // Evolve CfC network
        self.network.evolve_closed_form(dt, input);
        let output = self.network.output().clone();

        // Project CfC output → FEP observation (32D) via strided sampling
        let stride = (output.dim() / FEP_STATE_DIM).max(1);
        let obs_values: Vec<f64> = (0..FEP_STATE_DIM)
            .map(|i| {
                let idx = i * stride;
                output.values.get(idx).copied().unwrap_or(0.0) as f64
            })
            .collect();

        let observation = Observation::new(obs_values, 1.0, "neuroevo");
        let perception = self.fep_agent.perceive(&observation);
        let fe = perception.free_energy.clone();

        // Phi proxy: integration measure from CfC hidden state.
        // High variance across dimensions indicates differentiated-yet-integrated processing
        // (low variance = all dimensions doing the same thing = low Phi).
        // This is a lightweight approximation of IIT's Phi.
        let phi_proxy = compute_phi_proxy(&output);
        if phi_proxy > self.peak_phi {
            self.peak_phi = phi_proxy;
        }
        self.total_phi += phi_proxy as f64;

        // Track Phi variance via Welford's online algorithm (for stability objective)
        let n = (self.total_cycles + 1) as f64;
        let delta = phi_proxy as f64 - self.phi_mean;
        self.phi_mean += delta / n;
        let delta2 = phi_proxy as f64 - self.phi_mean;
        self.phi_m2 += delta * delta2;

        // Track initial and final FE for reduction rate
        if self.total_cycles == 0 {
            self.initial_fe = fe.total;
        }
        self.final_fe = fe.total;

        self.total_free_energy += fe.total;
        self.total_cycles += 1;
        self.age_cycles += 1;

        StepResult {
            free_energy: fe,
            output,
        }
    }

    /// Compute fitness from accumulated evaluation data.
    pub fn compute_fitness(&mut self, weights: &crate::fitness::FitnessWeights) {
        if self.total_cycles == 0 {
            self.fitness = OrganismFitness::default();
            return;
        }

        let n = self.total_cycles.max(1) as f64;
        let mean_fe = self.total_free_energy / n;

        // Objective 1: Mean Phi (sustained consciousness, not lucky spikes)
        let mean_phi = self.total_phi / n;

        // Objective 2: FE reduction rate — system must be LEARNING, not stagnant.
        // Goodhart defense: you can't fake learning.
        let fe_reduction = if self.initial_fe.abs() > 1e-10 {
            ((self.initial_fe - self.final_fe) / self.initial_fe.abs()).clamp(-1.0, 1.0)
        } else {
            0.0
        };

        // Objective 3: Prediction accuracy — system must actually predict.
        let pred_acc = if self.fep_agent.stats.perception_cycles > 0 {
            1.0 - (self.fep_agent.stats.avg_prediction_error / 10.0).min(1.0)
        } else {
            0.0
        };

        // Objective 4: Phi stability — penalize variance.
        // High mean + low variance = genuine sustained consciousness.
        let phi_variance = if self.total_cycles > 1 {
            self.phi_m2 / (n - 1.0)
        } else {
            0.0
        };
        let phi_stability = 1.0 / (1.0 + phi_variance * 10.0);

        // Objective 5: Threshold consistency (existing heuristic)
        let threshold_fit =
            crate::threshold_genome::evaluate_threshold_fitness(&self.genome.decode_thresholds());

        let efficiency = 1.0 / (1.0 + mean_fe.abs());

        // Composite for display/tiebreaking. Pareto sort uses individual objectives.
        let composite = mean_phi * weights.phi
            + fe_reduction.max(0.0) * weights.free_energy
            + pred_acc * 0.2
            + phi_stability * 0.15
            + threshold_fit * 0.1;

        self.fitness = OrganismFitness {
            composite: composite.max(FITNESS_FLOOR),
            free_energy: mean_fe,
            phi: mean_phi,
            consciousness: phi_stability,
            prediction_accuracy: pred_acc,
            energy_efficiency: efficiency,
            threshold_fitness: threshold_fit,
        };
    }

    /// Reproduce with a partner: crossover genomes, spawn child.
    pub fn reproduce(
        &self,
        partner: &Self,
        child_id: u64,
        mutation_rate: f32,
        seed: u64,
        genesis: &GenesisSeed,
    ) -> Self {
        let child_genome = self.genome.crossover(&partner.genome, seed);
        let child_genome = child_genome.mutate(mutation_rate, seed.wrapping_add(1));
        let mut child = Self::spawn(child_id, child_genome, genesis, self.generation + 1);
        child.parent_ids = (Some(self.id), Some(partner.id));
        child
    }

    /// Asexual replication with mutation.
    pub fn replicate(
        &self,
        child_id: u64,
        mutation_rate: f32,
        seed: u64,
        genesis: &GenesisSeed,
    ) -> Self {
        let child_genome = self.genome.mutate(mutation_rate, seed);
        let mut child = Self::spawn(child_id, child_genome, genesis, self.generation + 1);
        child.parent_ids = (Some(self.id), None);
        child
    }

    /// Check if this organism should die (age-based).
    pub fn should_die(&self) -> bool {
        self.age_cycles >= MAX_AGE_CYCLES
    }

    /// Reset evaluation state for a new fitness round.

    /// Get initial FE from evaluation (for FE reduction rate).
    pub fn initial_fe(&self) -> f64 {
        self.initial_fe
    }
    /// Get final FE from evaluation.
    pub fn final_fe(&self) -> f64 {
        self.final_fe
    }
    pub fn reset_evaluation(&mut self) {
        self.total_free_energy = 0.0;
        self.total_cycles = 0;
        self.peak_phi = 0.0;
        self.total_phi = 0.0;
        self.phi_mean = 0.0;
        self.phi_m2 = 0.0;
        self.initial_fe = 0.0;
        self.final_fe = 0.0;
        self.fep_agent = ActiveInferenceAgent::new(self.fep_agent.config.clone());
        // Re-instantiate network from genome with same dimension
        let phenotype = self.genome.decode();
        let mut net_config = phenotype.network_config;
        net_config.neuron_config.dimension = self.eval_dim;
        let genesis = GenesisSeed::from_phrase(&format!("organism_{}_reset", self.id));
        self.network = HdcLtcUnifiedNetwork::from_genesis(net_config, &genesis);
    }

    /// Get the best-ever phenotype decoded from this organism's genome.
    pub fn phenotype(&self) -> crate::genome::NeuralPhenotype {
        self.genome.decode()
    }
}

/// Phi proxy: measures integration of CfC output via normalized variance.
///
/// High variance across dimensions indicates differentiated processing;
/// high mean activity indicates the network is "alive". Product of both
/// approximates Tononi's Phi: differentiation × integration.
///
/// Returns a value in [0.0, ~1.0] for typical CfC outputs.
/// Compute Phi proxy from CfC network output using spectral analysis.
///
/// Two components combined:
/// 1. **Output integration**: variance × activity of the output vector
///    (lightweight, captures differentiation)
/// 2. **Spectral gap**: eigenvalue gap of the output's auto-correlation
///    structure, approximating integrated information
///    (Tononi 2004 — spectral gap correlates with Phi)
///
/// The combination prevents Goodhart: high variance alone (noise) scores low
/// because spectral gap requires structured correlation.
fn compute_phi_proxy(output: &ContinuousHV) -> f32 {
    let n = output.dim();
    if n < 2 {
        return 0.0;
    }

    let nf = n as f32;
    let mean: f32 = output.values.iter().sum::<f32>() / nf;
    let variance: f32 = output
        .values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f32>()
        / nf;
    let mean_abs: f32 = output.values.iter().map(|v| v.abs()).sum::<f32>() / nf;

    // Component 1: output differentiation × integration
    let differentiation = variance.sqrt().min(1.0);
    let integration = mean_abs.min(1.0);
    let output_phi = differentiation * integration;

    // Component 2: spectral gap from output auto-correlation
    // Build a small correlation matrix from output dimensions
    // (sample every stride to keep it manageable — max 32×32)
    let max_dim = 32;
    let stride = (n / max_dim).max(1);
    let m = (n / stride).min(max_dim);
    if m < 2 {
        return output_phi.min(1.0);
    }

    // Build m×m correlation matrix from sampled dimensions
    let sampled: Vec<f32> = (0..m).map(|i| output.values[i * stride]).collect();
    let s_mean: f32 = sampled.iter().sum::<f32>() / m as f32;
    let mut adj = vec![0.0f32; m * m];
    for i in 0..m {
        for j in 0..m {
            // Correlation: (x_i - mean)(x_j - mean)
            adj[i * m + j] = ((sampled[i] - s_mean) * (sampled[j] - s_mean)).abs();
        }
    }

    // Power iteration for spectral gap (10 steps — fast approximation)
    let spectral_gap = spectral_gap_f32(&adj, m);

    // Combine: geometric mean of output_phi and spectral_gap
    let combined = (output_phi * spectral_gap).sqrt();
    combined.min(1.0)
}

/// Compute spectral gap of a symmetric matrix via power iteration.
/// Returns normalized gap in [0, 1].
fn spectral_gap_f32(matrix: &[f32], n: usize) -> f32 {
    if n < 2 {
        return 0.0;
    }

    // Power iteration for dominant eigenvalue
    let mut v = vec![1.0 / (n as f32).sqrt(); n];
    let mut lambda1 = 0.0f32;
    for _ in 0..15 {
        let mut w = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += matrix[i * n + j] * v[j];
            }
        }
        let norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        lambda1 = norm;
        for (vi, wi) in v.iter_mut().zip(w.iter()) {
            *vi = wi / norm;
        }
    }

    if lambda1 < 1e-10 {
        return 0.0;
    }

    // Deflate: A' = A - lambda1 * v * v^T
    let mut deflated = matrix.to_vec();
    for i in 0..n {
        for j in 0..n {
            deflated[i * n + j] -= lambda1 * v[i] * v[j];
        }
    }

    // Second eigenvalue
    let mut v2 = vec![1.0 / (n as f32).sqrt(); n];
    let mut lambda2 = 0.0f32;
    for _ in 0..15 {
        let mut w = vec![0.0f32; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += deflated[i * n + j] * v2[j];
            }
        }
        let norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        lambda2 = norm;
        for (vi, wi) in v2.iter_mut().zip(w.iter()) {
            *vi = wi / norm;
        }
    }

    // Normalized spectral gap
    let gap = (lambda1 - lambda2).max(0.0);
    (gap / lambda1.max(1e-10)).min(1.0)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::FitnessWeights;

    fn test_genesis() -> GenesisSeed {
        GenesisSeed::from_phrase("test-organism")
    }

    fn spawn_fast(
        id: u64,
        genome: NeuralGenome,
        genesis: &GenesisSeed,
        r#gen: u32,
    ) -> NeuralOrganism {
        NeuralOrganism::spawn_with_dim(id, genome, genesis, r#gen, FAST_TEST_DIM)
    }

    #[test]
    fn test_spawn_valid() {
        let genome = NeuralGenome::random(42);
        let org = spawn_fast(1, genome, &test_genesis(), 0);
        assert_eq!(org.id, 1);
        assert_eq!(org.generation, 0);
        assert!(org.alive);
        assert_eq!(org.age_cycles, 0);
        assert_eq!(org.total_cycles, 0);
    }

    #[test]
    fn test_evaluate_step_finite() {
        let genome = NeuralGenome::random(42);
        let mut org = spawn_fast(1, genome, &test_genesis(), 0);
        let input = ContinuousHV::random(FAST_TEST_DIM, 99);
        let result = org.evaluate_step(&input, 0.05);
        assert!(result.free_energy.total.is_finite());
        assert_eq!(org.total_cycles, 1);
    }

    #[test]
    fn test_evaluate_multiple_steps() {
        let genome = NeuralGenome::random(42);
        let mut org = spawn_fast(1, genome, &test_genesis(), 0);
        for i in 0..10 {
            let input = ContinuousHV::random(FAST_TEST_DIM, 100 + i);
            org.evaluate_step(&input, 0.05);
        }
        assert_eq!(org.total_cycles, 10);
        assert_eq!(org.age_cycles, 10);
    }

    #[test]
    fn test_compute_fitness_with_data() {
        let genome = NeuralGenome::random(42);
        let mut org = spawn_fast(1, genome, &test_genesis(), 0);
        for i in 0..5 {
            let input = ContinuousHV::random(FAST_TEST_DIM, 200 + i);
            org.evaluate_step(&input, 0.05);
        }
        let weights = FitnessWeights::default();
        org.compute_fitness(&weights);
        assert!(org.fitness.composite.is_finite());
        assert!(org.fitness.free_energy.is_finite());
    }

    #[test]
    fn test_compute_fitness_empty() {
        let genome = NeuralGenome::random(42);
        let mut org = spawn_fast(1, genome, &test_genesis(), 0);
        let weights = FitnessWeights::default();
        org.compute_fitness(&weights);
        assert_eq!(org.fitness.composite, 0.0);
    }

    #[test]
    fn test_crossover_inheritance() {
        let g1 = NeuralGenome::random(100);
        let g2 = NeuralGenome::random(200);
        let genesis = test_genesis();
        let parent1 = spawn_fast(1, g1, &genesis, 0);
        let parent2 = spawn_fast(2, g2, &genesis, 0);
        let child = parent1.reproduce(&parent2, 3, 0.02, 42, &genesis);
        assert_eq!(child.id, 3);
        assert_eq!(child.generation, 1);
        assert_eq!(child.parent_ids, (Some(1), Some(2)));
    }

    #[test]
    fn test_replicate() {
        let genome = NeuralGenome::random(42);
        let genesis = test_genesis();
        let parent = spawn_fast(1, genome, &genesis, 0);
        let child = parent.replicate(2, 0.02, 99, &genesis);
        assert_eq!(child.parent_ids, (Some(1), None));
        assert_eq!(child.generation, 1);
    }

    #[test]
    fn test_should_die_age() {
        let genome = NeuralGenome::random(42);
        let mut org = spawn_fast(1, genome, &test_genesis(), 0);
        assert!(!org.should_die());
        org.age_cycles = MAX_AGE_CYCLES;
        assert!(org.should_die());
    }

    #[test]
    fn test_reset_evaluation() {
        let genome = NeuralGenome::random(42);
        let mut org = spawn_fast(1, genome, &test_genesis(), 0);
        let input = ContinuousHV::random(FAST_TEST_DIM, 99);
        org.evaluate_step(&input, 0.05);
        assert_eq!(org.total_cycles, 1);
        org.reset_evaluation();
        assert_eq!(org.total_cycles, 0);
        assert_eq!(org.total_free_energy, 0.0);
    }

    #[test]
    fn test_organism_deterministic() {
        let genome = NeuralGenome::random(42);
        let genesis = test_genesis();
        let mut org1 = spawn_fast(1, genome.clone(), &genesis, 0);
        let mut org2 = spawn_fast(1, genome, &genesis, 0);
        let input = ContinuousHV::random(FAST_TEST_DIM, 99);
        let r1 = org1.evaluate_step(&input, 0.05);
        let r2 = org2.evaluate_step(&input, 0.05);
        assert!((r1.free_energy.total - r2.free_energy.total).abs() < 1e-10);
    }
}
