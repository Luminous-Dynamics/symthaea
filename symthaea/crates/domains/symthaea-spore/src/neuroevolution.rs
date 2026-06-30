// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! MicroEvolver: tick-based neuroevolution for WASM consciousness kernel.
//!
//! Stripped-down evolutionary loop that evaluates at most 1-2 organisms per tick,
//! never blocking the JS event loop beyond ~5ms. Population of 5-10 organisms,
//! BinaryHV genomes only (20KB total memory).
//!
//! When a generation completes and a challenger beats the current champion,
//! `TickResult::ChampionSwapped` signals the SporeEngine to update its CfC config.

use serde::{Deserialize, Serialize};
use symthaea_core::hdc::binary_hv::BinaryHV;
use symthaea_core::hdc::hdc_ltc_unified::{HdcLtcUnifiedNetwork, UnifiedNetworkConfig};
use symthaea_core::hdc::unified_hv::ContinuousHV;

// When the neuroevolution feature is enabled, use the shared genome decoder
// to avoid duplicating the bit-layout logic.
#[cfg(feature = "neuroevolution")]
use symthaea_neuroevolution::NeuralGenome;

// When the feature is not enabled, use the inline fallback decoder.
#[cfg(not(feature = "neuroevolution"))]
use symthaea_core::hdc::hdc_ltc_unified::{UnifiedActivation, UnifiedConfig};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Default micro-population size (kept tiny for WASM memory budget).
const DEFAULT_MICRO_POP: usize = 8;

/// Maximum CfC evolution steps per organism evaluation.
const DEFAULT_EVAL_STEPS: usize = 100;

/// Maximum CfC steps executed per tick call (~5ms at 50us/step).
const DEFAULT_MAX_STEPS_PER_TICK: usize = 100;

/// Warmup steps excluded from fitness.
const WARMUP_STEPS: usize = 20;

/// Mutation rate for micro-population.
const MICRO_MUTATION_RATE: f32 = 0.02;

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the micro-evolver.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroEvolverConfig {
    pub population_size: usize,
    pub eval_steps: usize,
    pub max_steps_per_tick: usize,
    pub warmup_steps: usize,
    pub mutation_rate: f32,
    pub dt: f32,
}

impl Default for MicroEvolverConfig {
    fn default() -> Self {
        Self {
            population_size: DEFAULT_MICRO_POP,
            eval_steps: DEFAULT_EVAL_STEPS,
            max_steps_per_tick: DEFAULT_MAX_STEPS_PER_TICK,
            warmup_steps: WARMUP_STEPS,
            mutation_rate: MICRO_MUTATION_RATE,
            dt: 0.05,
        }
    }
}

/// A micro-organism: just a genome + accumulated fitness.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroOrganism {
    pub genome: BinaryHV,
    pub fitness: f64,
    pub eval_count: usize,
}

/// State machine phase for tick-based progression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EvalPhase {
    /// Evaluating a specific organism at a specific step.
    Evaluating {
        organism_index: usize,
        step: usize,
        accumulated_error: f64,
    },
    /// All organisms evaluated; selecting parents.
    Selecting,
    /// Breeding next generation.
    Breeding,
}

/// Result of a single tick.
#[derive(Debug, Clone)]
pub enum TickResult {
    /// Evaluation still in progress.
    InProgress { evaluated: usize, remaining: usize },
    /// A full generation just completed.
    GenerationComplete { snapshot: MicroSnapshot },
    /// Champion was swapped — SporeEngine should update its CfC config.
    ChampionSwapped { new_config: UnifiedNetworkConfig },
}

/// Snapshot of a micro-generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroSnapshot {
    pub generation: u32,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub champion_index: usize,
}

/// Tick-based micro-evolver for WASM.
pub struct MicroEvolver {
    population: Vec<MicroOrganism>,
    current_champion: usize,
    champion_fitness: f64,
    generation: u32,
    phase: EvalPhase,
    config: MicroEvolverConfig,
    /// Transient network used during evaluation (only 1 at a time).
    eval_network: Option<HdcLtcUnifiedNetwork>,
    /// Input cache for deterministic evaluation.
    eval_seed: u64,
}

impl MicroEvolver {
    /// Create a new micro-evolver with the given config.
    pub fn new(config: MicroEvolverConfig) -> Self {
        let mut population = Vec::with_capacity(config.population_size);
        for i in 0..config.population_size {
            population.push(MicroOrganism {
                genome: BinaryHV::random(42 + i as u64),
                fitness: 0.0,
                eval_count: 0,
            });
        }

        Self {
            population,
            current_champion: 0,
            champion_fitness: f64::NEG_INFINITY,
            generation: 0,
            phase: EvalPhase::Evaluating {
                organism_index: 0,
                step: 0,
                accumulated_error: 0.0,
            },
            config,
            eval_network: None,
            eval_seed: 12345,
        }
    }

    /// Run one tick of evolution. Bounded to `max_steps_per_tick` CfC steps.
    pub fn tick(&mut self) -> TickResult {
        match self.phase.clone() {
            EvalPhase::Evaluating {
                organism_index,
                step,
                accumulated_error,
            } => self.tick_evaluate(organism_index, step, accumulated_error),
            EvalPhase::Selecting => self.tick_select(),
            EvalPhase::Breeding => self.tick_breed(),
        }
    }

    fn tick_evaluate(
        &mut self,
        org_idx: usize,
        start_step: usize,
        mut accumulated_error: f64,
    ) -> TickResult {
        let total_steps = self.config.warmup_steps + self.config.eval_steps;

        // Ensure network is instantiated for this organism
        if self.eval_network.is_none() || start_step == 0 {
            let net_config = decode_network_config(&self.population[org_idx].genome);
            self.eval_network = Some(HdcLtcUnifiedNetwork::new(net_config, org_idx as u64));
            accumulated_error = 0.0;
        }

        let network = self.eval_network.as_mut().unwrap();
        let mut steps_this_tick = 0;
        let mut current_step = start_step;

        while current_step < total_steps && steps_this_tick < self.config.max_steps_per_tick {
            let input = generate_eval_input(self.eval_seed + current_step as u64);
            network.evolve_closed_form(self.config.dt, &input);

            // After warmup, accumulate prediction error as fitness signal
            if current_step >= self.config.warmup_steps {
                let output = network.output();
                // Simple fitness: mean absolute value (higher activity = lower free energy proxy)
                let activity: f32 =
                    output.values.iter().map(|v| v.abs()).sum::<f32>() / output.dim() as f32;
                accumulated_error += activity as f64;
            }

            current_step += 1;
            steps_this_tick += 1;
        }

        if current_step >= total_steps {
            // This organism's evaluation is complete
            let fitness = accumulated_error / self.config.eval_steps as f64;
            self.population[org_idx].fitness = fitness;
            self.population[org_idx].eval_count += 1;

            // Drop network to free memory
            self.eval_network = None;

            let next_org = org_idx + 1;
            if next_org >= self.population.len() {
                // All organisms evaluated
                self.phase = EvalPhase::Selecting;
                TickResult::InProgress {
                    evaluated: self.population.len(),
                    remaining: 0,
                }
            } else {
                self.phase = EvalPhase::Evaluating {
                    organism_index: next_org,
                    step: 0,
                    accumulated_error: 0.0,
                };
                TickResult::InProgress {
                    evaluated: next_org,
                    remaining: self.population.len() - next_org,
                }
            }
        } else {
            // Save progress for next tick
            self.phase = EvalPhase::Evaluating {
                organism_index: org_idx,
                step: current_step,
                accumulated_error,
            };
            TickResult::InProgress {
                evaluated: org_idx,
                remaining: self.population.len() - org_idx,
            }
        }
    }

    fn tick_select(&mut self) -> TickResult {
        // Find best organism
        let best_idx = self
            .population
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.fitness
                    .partial_cmp(&b.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        let best_fitness = self.population[best_idx].fitness;
        let mean_fitness =
            self.population.iter().map(|o| o.fitness).sum::<f64>() / self.population.len() as f64;

        let snapshot = MicroSnapshot {
            generation: self.generation,
            best_fitness,
            mean_fitness,
            champion_index: best_idx,
        };

        // Check if we should swap champion
        let should_swap = best_fitness > self.champion_fitness;
        if should_swap {
            self.current_champion = best_idx;
            self.champion_fitness = best_fitness;
        }

        self.phase = EvalPhase::Breeding;

        if should_swap {
            let new_config = decode_network_config(&self.population[best_idx].genome);
            TickResult::ChampionSwapped { new_config }
        } else {
            TickResult::GenerationComplete { snapshot }
        }
    }

    fn tick_breed(&mut self) -> TickResult {
        self.generation += 1;
        self.eval_seed = self.eval_seed.wrapping_add(self.generation as u64 * 1000);

        // Sort by fitness (keep best as elite)
        let mut indices: Vec<usize> = (0..self.population.len()).collect();
        indices.sort_by(|&a, &b| {
            self.population[b]
                .fitness
                .partial_cmp(&self.population[a].fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Elite: keep top 2
        let elite_count = 2.min(self.population.len());
        let mut new_pop = Vec::with_capacity(self.population.len());

        for &idx in indices.iter().take(elite_count) {
            new_pop.push(self.population[idx].clone());
        }

        // Fill rest with mutated copies of top half
        let top_half = self.population.len() / 2;
        let mut child_seed = self.generation as u64 * 7919;
        while new_pop.len() < self.population.len() {
            let parent_idx = indices[(child_seed as usize) % top_half.max(1)];
            child_seed = child_seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let child_genome = self.population[parent_idx]
                .genome
                .add_noise(self.config.mutation_rate, child_seed);
            new_pop.push(MicroOrganism {
                genome: child_genome,
                fitness: 0.0,
                eval_count: 0,
            });
        }

        self.population = new_pop;
        self.phase = EvalPhase::Evaluating {
            organism_index: 0,
            step: 0,
            accumulated_error: 0.0,
        };

        let snapshot = MicroSnapshot {
            generation: self.generation,
            best_fitness: self.champion_fitness,
            mean_fitness: 0.0, // Not computed yet for new generation
            champion_index: 0,
        };

        TickResult::GenerationComplete { snapshot }
    }

    /// Get current generation.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Get champion fitness.
    pub fn champion_fitness(&self) -> f64 {
        self.champion_fitness
    }

    /// Get population size.
    pub fn population_size(&self) -> usize {
        self.population.len()
    }

    /// Estimated memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        // Genomes: pop_size * 2048
        let genome_mem = self.population.len() * 2048;
        // Eval network (when active): ~400KB for a small network
        let net_mem = if self.eval_network.is_some() {
            400_000
        } else {
            0
        };
        genome_mem + net_mem
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// GENOME DECODING
// ═══════════════════════════════════════════════════════════════════════════════

/// When the `neuroevolution` feature is enabled, delegate to the shared crate.
#[cfg(feature = "neuroevolution")]
fn decode_network_config(hv: &BinaryHV) -> UnifiedNetworkConfig {
    let genome = NeuralGenome { hv: *hv };
    let phenotype = genome.decode();
    phenotype.network_config
}

/// Fallback: inline decoder for WASM builds without the neuroevolution crate.
#[cfg(not(feature = "neuroevolution"))]
fn extract_bits(hv: &BinaryHV, start_bit: usize, num_bits: usize) -> u64 {
    let mut value: u64 = 0;
    for i in 0..num_bits {
        let bit_index = start_bit + i;
        let byte_index = bit_index / 8;
        let bit_offset = bit_index % 8;
        if hv.0[byte_index] & (1 << bit_offset) != 0 {
            value |= 1 << i;
        }
    }
    value
}

#[cfg(not(feature = "neuroevolution"))]
fn bits_to_linear(raw: u64, num_bits: usize, lo: f32, hi: f32) -> f32 {
    let max_val = ((1u64 << num_bits) - 1) as f32;
    if max_val == 0.0 {
        return lo;
    }
    lo + (raw as f32 / max_val) * (hi - lo)
}

#[cfg(not(feature = "neuroevolution"))]
fn bits_to_log(raw: u64, num_bits: usize, lo: f32, hi: f32) -> f32 {
    let t = bits_to_linear(raw, num_bits, 0.0, 1.0);
    let log_lo = lo.ln();
    let log_hi = hi.ln();
    (log_lo + t * (log_hi - log_lo)).exp()
}

#[cfg(not(feature = "neuroevolution"))]
fn decode_network_config(hv: &BinaryHV) -> UnifiedNetworkConfig {
    let tau_base = bits_to_log(extract_bits(hv, 0, 16), 16, 0.01, 1.0);
    let backbone_tau = bits_to_linear(extract_bits(hv, 16, 16), 16, 0.1, 2.0);

    let activation_raw = extract_bits(hv, 32, 3) as usize;
    let activation = match activation_raw % 5 {
        0 => UnifiedActivation::Tanh,
        1 => UnifiedActivation::Sigmoid,
        2 => UnifiedActivation::SiLU,
        3 => UnifiedActivation::Identity,
        _ => UnifiedActivation::BoundedTanh { scale: 1.5 },
    };

    let learning_rate = bits_to_log(extract_bits(hv, 35, 16), 16, 0.0001, 0.1);
    let momentum = bits_to_linear(extract_bits(hv, 51, 16), 16, 0.8, 0.999);
    let weight_decay = bits_to_linear(extract_bits(hv, 67, 16), 16, 0.0, 0.01);
    let gating_steepness = bits_to_linear(extract_bits(hv, 83, 16), 16, 0.1, 5.0);

    let layer_count_raw = extract_bits(hv, 231, 3) as usize;
    let layer_count = (layer_count_raw % 5) + 1;

    let mut layer_sizes = Vec::with_capacity(layer_count);
    for i in 0..layer_count {
        let size_raw = extract_bits(hv, 234 + i * 16, 16) as usize;
        let size = (size_raw % 32) + 1;
        layer_sizes.push(size);
    }

    let skip_connections = extract_bits(hv, 314, 1) != 0;
    let use_layer_binding = extract_bits(hv, 315, 1) != 0;

    let neuron_config = UnifiedConfig {
        tau_base,
        backbone_tau,
        dimension: 16_384,
        activation,
        learning_rate,
        momentum,
        weight_decay,
        gating_steepness,
        interp_bias: 0.0,
        fourier_frequencies: Vec::new(), // Skip Fourier for WASM perf
        fourier_amplitude: 0.1,
    };

    UnifiedNetworkConfig {
        layer_sizes,
        neuron_config,
        use_layer_binding,
        skip_connections,
    }
}

/// Generate a deterministic input vector for evaluation.
fn generate_eval_input(seed: u64) -> ContinuousHV {
    ContinuousHV::random(16_384, seed)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_micro_evolver_creation() {
        let evolver = MicroEvolver::new(MicroEvolverConfig::default());
        assert_eq!(evolver.population_size(), DEFAULT_MICRO_POP);
        assert_eq!(evolver.generation(), 0);
    }

    #[test]
    fn test_memory_budget() {
        let evolver = MicroEvolver::new(MicroEvolverConfig {
            population_size: 10,
            ..Default::default()
        });
        // 10 organisms x 2KB = 20KB genome storage
        let mem = evolver.memory_usage();
        assert!(mem <= 25_000, "Memory usage: {mem} bytes (expected <25KB)");
    }

    #[test]
    fn test_tick_progress() {
        let mut evolver = MicroEvolver::new(MicroEvolverConfig {
            population_size: 3,
            eval_steps: 10,
            warmup_steps: 5,
            max_steps_per_tick: 200, // High enough to complete 1 organism per tick
            ..Default::default()
        });

        let result = evolver.tick();
        match result {
            TickResult::InProgress { .. } => {} // Expected
            _ => panic!("Expected InProgress on first tick"),
        }
    }

    #[test]
    fn test_generation_completes() {
        let mut evolver = MicroEvolver::new(MicroEvolverConfig {
            population_size: 3,
            eval_steps: 5,
            warmup_steps: 2,
            max_steps_per_tick: 1000, // Complete entire organisms quickly
            ..Default::default()
        });

        let mut gen_complete = false;
        for _ in 0..50 {
            match evolver.tick() {
                TickResult::GenerationComplete { .. } => {
                    gen_complete = true;
                    break;
                }
                TickResult::ChampionSwapped { .. } => {
                    gen_complete = true;
                    break;
                }
                TickResult::InProgress { .. } => {}
            }
        }
        assert!(gen_complete, "Generation should complete within 50 ticks");
    }

    #[test]
    fn test_multi_generation() {
        let mut evolver = MicroEvolver::new(MicroEvolverConfig {
            population_size: 3,
            eval_steps: 5,
            warmup_steps: 2,
            max_steps_per_tick: 1000,
            ..Default::default()
        });

        let mut generations_completed = 0;
        for _ in 0..200 {
            match evolver.tick() {
                TickResult::GenerationComplete { .. } | TickResult::ChampionSwapped { .. } => {
                    generations_completed += 1;
                    if generations_completed >= 3 {
                        break;
                    }
                }
                _ => {}
            }
        }
        assert!(
            generations_completed >= 3,
            "Should complete 3+ generations, got {generations_completed}"
        );
    }

    #[test]
    fn test_tick_bounded_steps() {
        let mut evolver = MicroEvolver::new(MicroEvolverConfig {
            population_size: 3,
            eval_steps: 200,
            warmup_steps: 50,
            max_steps_per_tick: 50, // Only 50 steps per tick
            ..Default::default()
        });

        // First tick should NOT complete an organism (needs 250 steps, budget is 50)
        let result = evolver.tick();
        match result {
            TickResult::InProgress { evaluated, .. } => {
                assert_eq!(evaluated, 0, "Should still be on first organism");
            }
            _ => panic!("Expected InProgress with limited tick budget"),
        }
    }

    #[test]
    fn test_champion_swap() {
        let mut evolver = MicroEvolver::new(MicroEvolverConfig {
            population_size: 3,
            eval_steps: 5,
            warmup_steps: 2,
            max_steps_per_tick: 1000,
            ..Default::default()
        });

        // Run until first generation completes — should swap champion
        let mut swapped = false;
        for _ in 0..50 {
            match evolver.tick() {
                TickResult::ChampionSwapped { new_config } => {
                    assert!(!new_config.layer_sizes.is_empty());
                    swapped = true;
                    break;
                }
                TickResult::GenerationComplete { .. } => {
                    // Generation 0 might not swap (initial champion)
                    break;
                }
                _ => {}
            }
        }
        // Champion swap happens when a new best is found
        // (first r#gen always swaps since champion_fitness starts at NEG_INFINITY)
        assert!(swapped, "First generation should swap champion");
    }

    #[test]
    fn test_decode_network_config_valid() {
        let genome = BinaryHV::random(42);
        let config = decode_network_config(&genome);
        assert!(!config.layer_sizes.is_empty());
        assert!(config.layer_sizes.len() <= 5);
        for &size in &config.layer_sizes {
            assert!(size >= 1 && size <= 32);
        }
        assert!(config.neuron_config.tau_base >= 0.01);
        assert!(config.neuron_config.tau_base <= 1.0);
    }

    #[test]
    fn test_serialization_roundtrip() {
        let config = MicroEvolverConfig::default();
        let json = serde_json::to_string(&config).unwrap();
        let decoded: MicroEvolverConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(decoded.population_size, config.population_size);
        assert_eq!(decoded.eval_steps, config.eval_steps);
    }
}
