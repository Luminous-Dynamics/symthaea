// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Neuroevolution Manager
//!
//! Runs CfC-HDC neuroevolution at interval 71 (co-prime scheduling).
//! Accumulates Phi/FE/consciousness between runs, advances one generation
//! per trigger (~250ms), feeds best genome's phenotype back as CfC
//! hyperparameter suggestions.
//!
//! ## Science
//!
//! - Stanley & Miikkulainen (2002). Evolving Neural Networks through Augmenting Topologies.
//! - Hasani et al. (2021). Liquid Time-constant Networks.
//! - Friston (2010). The free-energy principle: a unified brain theory?

use serde::{Deserialize, Serialize};
use symthaea_neuroevolution::{
    FepFitnessConfig, FitnessWeights, GenerationSnapshot, NeuralGenome, NeuralPhenotype,
    NeuroevolutionConfig, NeuroevolutionEngine,
};

use crate::cognitive_loop::subsystem_trait::{CycleSnapshot, SubsystemOutput};
use crate::cognitive_loop::thresholds;

/// Manager interval (co-prime with other managers).
pub const NEUROEVOLUTION_MANAGER_INTERVAL: usize = 71;

// ═══════════════════════════════════════════════════════════════════════════════
// TELEMETRY
// ═══════════════════════════════════════════════════════════════════════════════

/// Telemetry from the neuroevolution manager.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NeuroevolutionTelemetry {
    /// Current generation number.
    pub generation: u32,
    /// Best fitness in current generation.
    pub best_fitness: f64,
    /// Mean fitness in current generation.
    pub mean_fitness: f64,
    /// Population diversity (mean pairwise genome distance).
    pub diversity: f64,
    /// Number of species.
    pub species_count: usize,
    /// BLAKE3 hash of the active (best) genome (first 8 bytes as hex).
    pub active_genome_hash: String,
    /// Total generations completed.
    pub total_generations: u32,
    /// Best evolved tau_base (0.0 before first generation).
    pub best_tau_base: f32,
    /// Best evolved learning rate (0.0 before first generation).
    pub best_learning_rate: f32,
    /// Best evolved layer count (0 before first generation).
    pub best_layer_count: usize,
}

/// Champion topology suggestion fed back to the cognitive loop.
#[derive(Debug, Clone, Default)]
pub struct ChampionSuggestion {
    /// LR modulation (ratio to default, >1 = boost).
    pub lr_modulation: f64,
    /// Suggested tau_base from evolved genome.
    pub tau_base: f32,
    /// Suggested gating steepness from evolved genome.
    pub gating_steepness: f32,
    /// Suggested layer count from evolved topology.
    pub layer_count: usize,
    /// Whether a champion has been found yet.
    pub active: bool,
}

// ═══════════════════════════════════════════════════════════════════════════════
// MANAGER
// ═══════════════════════════════════════════════════════════════════════════════

/// Neuroevolution manager: evolves CfC-HDC neural organisms.
pub struct NeuroevolutionManager {
    engine: NeuroevolutionEngine,
    interval: usize,
    cycle_count: u64,
    last_snapshot: Option<GenerationSnapshot>,
    telemetry: NeuroevolutionTelemetry,
    initialized: bool,
    /// Accumulated observations between manager runs.
    accumulated_psi: Vec<f64>,
    accumulated_prediction_error: Vec<f32>,
    accumulated_confidence: Vec<f64>,
    /// Latest champion topology suggestion.
    champion: ChampionSuggestion,
    pub pending_mutations: Vec<symthaea_wisdom::meta_cognition::MutationSuggestion>,
}

impl NeuroevolutionManager {
    /// Create a new neuroevolution manager.
    pub fn new() -> Self {
        let config = NeuroevolutionConfig {
            population_size: thresholds::NEUROEVO_POPULATION_SIZE,
            tournament_size: thresholds::NEUROEVO_TOURNAMENT_SIZE,
            elitism_fraction: thresholds::NEUROEVO_ELITISM_FRACTION as f64,
            mutation_rate: thresholds::NEUROEVO_MUTATION_RATE,
            crossover_rate: thresholds::NEUROEVO_CROSSOVER_RATE as f64,
            max_generations: 200,
            convergence_patience: thresholds::NEUROEVO_CONVERGENCE_PATIENCE as u32,
            speciation_threshold: thresholds::NEUROEVO_SPECIATION_THRESHOLD,
            fitness_config: FepFitnessConfig {
                warmup_steps: thresholds::NEUROEVO_WARMUP_STEPS,
                eval_steps: thresholds::NEUROEVO_EVAL_STEPS,
                ..Default::default()
            },
            genesis_phrase: "symthaea-cognitive-neuroevolution".to_string(),
        };

        Self {
            engine: NeuroevolutionEngine::new(config),
            interval: NEUROEVOLUTION_MANAGER_INTERVAL,
            cycle_count: 0,
            last_snapshot: None,
            telemetry: NeuroevolutionTelemetry::default(),
            initialized: false,
            accumulated_psi: Vec::new(),
            accumulated_prediction_error: Vec::new(),
            accumulated_confidence: Vec::new(),
            champion: ChampionSuggestion::default(),
            pending_mutations: Vec::new(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: NeuroevolutionConfig) -> Self {
        Self {
            engine: NeuroevolutionEngine::new(config),
            interval: NEUROEVOLUTION_MANAGER_INTERVAL,
            cycle_count: 0,
            last_snapshot: None,
            telemetry: NeuroevolutionTelemetry::default(),
            initialized: false,
            accumulated_psi: Vec::new(),
            accumulated_prediction_error: Vec::new(),
            accumulated_confidence: Vec::new(),
            pending_mutations: Vec::new(),
            champion: ChampionSuggestion::default(),
        }
    }

    pub fn inject_mutations(
        &mut self,
        mutations: Vec<symthaea_wisdom::meta_cognition::MutationSuggestion>,
    ) {
        self.pending_mutations = mutations;
    }
    /// Process a cognitive cycle: accumulate observations, run evolution on interval.
    pub fn process(&mut self, snapshot: &CycleSnapshot) -> SubsystemOutput {
        self.cycle_count += 1;

        // Accumulate observations every cycle
        self.accumulated_psi.push(snapshot.unified_psi);
        self.accumulated_prediction_error
            .push(snapshot.prediction_error);
        self.accumulated_confidence
            .push(snapshot.prediction_confidence);

        // Only run evolution on interval
        if self.cycle_count % self.interval as u64 != 0 {
            return SubsystemOutput::default();
        }

        // Initialize on first trigger
        if !self.initialized {
            self.engine.initialize();
            self.initialized = true;
        }

        // Run one generation
        let gen_snapshot = self.engine.step_generation();

        // Lamarckian injection: convert MetaCognitive suggestions to bit-range targets
        if !self.pending_mutations.is_empty() {
            use symthaea_wisdom::meta_cognition::MutationTarget;
            let targets: Vec<(usize, usize, f32)> = self
                .pending_mutations
                .iter()
                .map(|s| {
                    let (start, bits) = match s.target {
                        MutationTarget::FepSurpriseScale => (400, 12),
                        MutationTarget::FepLrDecay => (412, 12),
                        MutationTarget::DreamBaseInterval => (436, 10),
                        MutationTarget::NeuromodArousalDecay => (490, 12),
                        MutationTarget::HomeostasisPullCruise => (610, 12),
                        MutationTarget::FlowExplorationIncrement => (562, 12),
                        MutationTarget::SelfModelWeightHigh => (598, 12),
                    };
                    (start, bits, s.confidence)
                })
                .collect();
            self.engine.inject_lamarckian(&targets);
            self.pending_mutations.clear();
        }
        self.last_snapshot = Some(gen_snapshot.clone());

        // Update telemetry
        self.telemetry = NeuroevolutionTelemetry {
            generation: gen_snapshot.generation,
            best_fitness: gen_snapshot.best_fitness,
            mean_fitness: gen_snapshot.mean_fitness,
            diversity: gen_snapshot.diversity,
            species_count: gen_snapshot.species_count,
            active_genome_hash: self.genome_hash(),
            total_generations: gen_snapshot.generation,
            best_tau_base: 0.0,
            best_learning_rate: 0.0,
            best_layer_count: 0,
        };

        // Clear accumulated observations
        self.accumulated_psi.clear();
        self.accumulated_prediction_error.clear();
        self.accumulated_confidence.clear();

        // Feed back champion topology as subsystem output
        let mut output = SubsystemOutput::default();
        if let Some((genome, _)) = self.engine.best_ever() {
            let phenotype = genome.decode();
            let lr_ratio = phenotype.neuron_config.learning_rate / 0.01;
            output.lr_modulation = 1.0 + (lr_ratio as f64 - 1.0) * 0.1;

            // Update champion suggestion with full topology
            self.champion = ChampionSuggestion {
                lr_modulation: output.lr_modulation,
                tau_base: phenotype.neuron_config.tau_base,
                gating_steepness: phenotype.neuron_config.gating_steepness,
                layer_count: phenotype.network_config.layer_sizes.len(),
                active: true,
            };

            // Add topology info to telemetry
            self.telemetry.best_tau_base = phenotype.neuron_config.tau_base;
            self.telemetry.best_learning_rate = phenotype.neuron_config.learning_rate;
            self.telemetry.best_layer_count = phenotype.network_config.layer_sizes.len();
        }

        output
    }

    /// Get current telemetry.
    pub fn telemetry(&self) -> &NeuroevolutionTelemetry {
        &self.telemetry
    }

    /// Get the best-ever genome.
    pub fn best_genome(&self) -> Option<&NeuralGenome> {
        self.engine.best_ever().map(|(g, _)| g)
    }

    /// Get the latest champion topology suggestion.
    pub fn champion_suggestion(&self) -> &ChampionSuggestion {
        &self.champion
    }

    /// Get the best-ever phenotype (decoded genome).
    pub fn best_phenotype(&self) -> Option<NeuralPhenotype> {
        self.engine.best_ever().map(|(g, _)| g.decode())
    }

    /// Set fitness weights at runtime.
    pub fn set_fitness_weights(&mut self, weights: FitnessWeights) {
        self.engine.set_fitness_weights(weights);
    }

    fn genome_hash(&self) -> String {
        if let Some((genome, _)) = self.engine.best_ever() {
            let hash = blake3::hash(&genome.hv.0);
            hex::encode(&hash.as_bytes()[..8])
        } else {
            "none".to_string()
        }
    }
}

impl Default for NeuroevolutionManager {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    fn test_snapshot() -> CycleSnapshot {
        let mut snap = CycleSnapshot::default();
        snap.unified_psi = 0.5;
        snap.prediction_error = 0.3;
        snap.prediction_confidence = 0.6;
        snap
    }

    #[test]
    fn test_manager_creation() {
        let mgr = NeuroevolutionManager::new();
        assert_eq!(mgr.interval, NEUROEVOLUTION_MANAGER_INTERVAL);
        assert!(!mgr.initialized);
    }

    #[test]
    fn test_accumulates_observations() {
        let mut mgr = NeuroevolutionManager::new();
        let snap = test_snapshot();
        mgr.process(&snap);
        assert_eq!(mgr.accumulated_psi.len(), 1);
    }

    #[test]
    fn test_interval_skips() {
        let mut mgr = NeuroevolutionManager::new();
        let snap = test_snapshot();

        // Process 70 cycles (interval is 71)
        for _ in 0..70 {
            mgr.process(&snap);
        }
        assert!(!mgr.initialized, "Should not initialize before interval");
    }

    #[test]
    fn test_triggers_at_interval() {
        let mut mgr = NeuroevolutionManager::with_config(NeuroevolutionConfig {
            population_size: 5,
            fitness_config: FepFitnessConfig {
                warmup_steps: 2,
                eval_steps: 5,
                ..Default::default()
            },
            ..Default::default()
        });
        let snap = test_snapshot();

        for _ in 0..NEUROEVOLUTION_MANAGER_INTERVAL {
            mgr.process(&snap);
        }
        assert!(mgr.initialized);
        assert_eq!(mgr.telemetry.generation, 1);
    }

    #[test]
    fn test_telemetry_populated() {
        let mut mgr = NeuroevolutionManager::with_config(NeuroevolutionConfig {
            population_size: 5,
            fitness_config: FepFitnessConfig {
                warmup_steps: 2,
                eval_steps: 5,
                ..Default::default()
            },
            ..Default::default()
        });
        let snap = test_snapshot();

        for _ in 0..NEUROEVOLUTION_MANAGER_INTERVAL {
            mgr.process(&snap);
        }

        let tel = mgr.telemetry();
        assert_eq!(tel.generation, 1);
        assert!(tel.best_fitness.is_finite());
        assert!(tel.diversity >= 0.0);
        assert!(!tel.active_genome_hash.is_empty());
    }

    #[test]
    fn test_genome_swap_suggestion() {
        let mut mgr = NeuroevolutionManager::with_config(NeuroevolutionConfig {
            population_size: 5,
            fitness_config: FepFitnessConfig {
                warmup_steps: 2,
                eval_steps: 5,
                ..Default::default()
            },
            ..Default::default()
        });
        let snap = test_snapshot();

        for _ in 0..NEUROEVOLUTION_MANAGER_INTERVAL {
            mgr.process(&snap);
        }

        assert!(mgr.best_genome().is_some());
    }

    #[test]
    fn test_set_fitness_weights() {
        let mut mgr = NeuroevolutionManager::new();
        mgr.set_fitness_weights(FitnessWeights {
            free_energy: 0.9,
            phi: 0.05,
            consciousness: 0.025,
            efficiency: 0.025,
        });
        // Success: weights applied without panic (no getter exposed, but manager is still functional)
        assert_eq!(
            mgr.interval, NEUROEVOLUTION_MANAGER_INTERVAL,
            "manager should remain valid after setting weights"
        );
    }

    #[test]
    fn test_default_impl() {
        let mgr = NeuroevolutionManager::default();
        assert_eq!(mgr.interval, NEUROEVOLUTION_MANAGER_INTERVAL);
    }
}
