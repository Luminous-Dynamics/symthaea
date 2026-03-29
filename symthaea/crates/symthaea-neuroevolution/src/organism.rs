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

        let mean_fe = self.total_free_energy / self.total_cycles as f64;
        let mean_phi = self.peak_phi as f64; // Best observed Phi

        // Prediction accuracy from FEP agent stats
        let pred_acc = if self.fep_agent.stats.perception_cycles > 0 {
            1.0 - (self.fep_agent.stats.avg_prediction_error / 10.0).min(1.0)
        } else {
            0.0
        };

        // Energy efficiency: lower mean FE per cycle = better
        let efficiency = 1.0 / (1.0 + mean_fe.abs());

        // Composite: negate FE (lower is better), add Phi and others
        let composite = -mean_fe * weights.free_energy
            + mean_phi * weights.phi
            + mean_phi * weights.consciousness // Use phi as consciousness proxy
            + efficiency * weights.efficiency;

        self.fitness = OrganismFitness {
            composite: composite.max(FITNESS_FLOOR),
            free_energy: mean_fe,
            phi: mean_phi,
            consciousness: mean_phi,
            prediction_accuracy: pred_acc,
            energy_efficiency: efficiency,
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
    pub fn reset_evaluation(&mut self) {
        self.total_free_energy = 0.0;
        self.total_cycles = 0;
        self.peak_phi = 0.0;
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
fn compute_phi_proxy(output: &ContinuousHV) -> f32 {
    let n = output.dim() as f32;
    if n < 2.0 {
        return 0.0;
    }

    let mean: f32 = output.values.iter().sum::<f32>() / n;
    let variance: f32 = output
        .values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f32>()
        / n;
    let mean_abs: f32 = output.values.iter().map(|v| v.abs()).sum::<f32>() / n;

    // Phi proxy = sqrt(variance) * mean_activity, clamped to [0, 1]
    let differentiation = variance.sqrt().min(1.0);
    let integration = mean_abs.min(1.0);
    (differentiation * integration).min(1.0)
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
        gen: u32,
    ) -> NeuralOrganism {
        NeuralOrganism::spawn_with_dim(id, genome, genesis, gen, FAST_TEST_DIM)
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
