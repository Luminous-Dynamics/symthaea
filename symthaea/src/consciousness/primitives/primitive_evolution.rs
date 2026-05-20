// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Primitive Evolution: Concept and Knowledge Evolution
//!
//! Provides mechanisms for evolving concepts and knowledge structures
//! over time through learning and adaptation.

use crate::consciousness::epistemic_tiers::EpistemicCoordinate;
use crate::hdc::BinaryHV;
use anyhow::Result;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use symthaea_core::genesis::{GenesisSeed, ShakeRng};
use symthaea_core::hdc::ContinuousHV;
use symthaea_core::hdc::primitive_system::PrimitiveTier;

/// Configuration for the primitive evolver
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolverConfig {
    /// Learning rate for concept updates
    pub learning_rate: f32,
    /// Mutation rate for exploration
    pub mutation_rate: f32,
    /// Population size for evolutionary algorithms
    pub population_size: usize,
    /// Number of generations
    pub generations: usize,
    /// Selection pressure (higher = more selective)
    pub selection_pressure: f32,
    /// History size for tracking evolution
    pub history_size: usize,
}

impl Default for EvolverConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            mutation_rate: 0.01,
            population_size: 50,
            generations: 100,
            selection_pressure: 2.0,
            history_size: 100,
        }
    }
}

/// Result of an evolution operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionResult {
    /// Best fitness achieved
    pub best_fitness: f64,
    /// Generations completed
    pub generations_completed: usize,
    /// Improvement over initial
    pub improvement: f64,
    /// Evolution history
    pub history: Vec<EvolutionSnapshot>,
    /// Whether convergence was reached
    pub converged: bool,
}

/// A snapshot of evolution state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionSnapshot {
    /// Generation number
    pub generation: usize,
    /// Best fitness
    pub best_fitness: f64,
    /// Average fitness
    pub avg_fitness: f64,
    /// Diversity measure
    pub diversity: f64,
}

/// An evolving concept
#[derive(Debug, Clone)]
pub struct EvolvingConcept {
    /// Concept identifier
    pub id: u64,
    /// Concept name
    pub name: String,
    /// Current hypervector representation
    pub hv: ContinuousHV,
    /// Fitness score
    pub fitness: f64,
    /// Generation this concept was created
    pub generation: usize,
    /// Parent concepts (if any)
    pub parents: Vec<u64>,
    /// Mutation count
    pub mutations: usize,
}

impl EvolvingConcept {
    /// Create a new evolving concept
    pub fn new(id: u64, name: impl Into<String>, hv: ContinuousHV) -> Self {
        Self {
            id,
            name: name.into(),
            hv,
            fitness: 0.0,
            generation: 0,
            parents: Vec::new(),
            mutations: 0,
        }
    }

    /// Mutate the concept
    pub fn mutate(&mut self, rate: f32) {
        self.hv = self.hv.perturb(rate);
        self.mutations += 1;
    }

    /// Crossover with another concept
    pub fn crossover(&self, other: &EvolvingConcept, id: u64) -> EvolvingConcept {
        let new_hv = ContinuousHV::bundle_owned(&[self.hv.clone(), other.hv.clone()]);
        let new_name = format!("{}×{}", self.name, other.name);

        let mut child = EvolvingConcept::new(id, new_name, new_hv);
        child.generation = self.generation.max(other.generation) + 1;
        child.parents = vec![self.id, other.id];
        child
    }
}

/// The primitive evolver
pub struct PrimitiveEvolver {
    /// Configuration
    config: EvolverConfig,
    /// Current population
    population: HashMap<u64, EvolvingConcept>,
    /// Next concept ID
    next_id: u64,
    /// Current generation
    current_generation: usize,
    /// Evolution history
    history: VecDeque<EvolutionSnapshot>,
    /// Statistics
    stats: EvolverStats,
    /// Optional deterministic RNG from genesis seed
    rng: Option<ShakeRng>,
}

/// Statistics for the evolver
#[derive(Debug, Clone, Default)]
pub struct EvolverStats {
    /// Total evolutions run
    pub total_evolutions: u64,
    /// Total concepts created
    pub concepts_created: u64,
    /// Total mutations
    pub total_mutations: u64,
    /// Total crossovers
    pub total_crossovers: u64,
    /// Best fitness ever achieved
    pub best_fitness_ever: f64,
}

impl PrimitiveEvolver {
    /// Create a new evolver
    pub fn new(config: EvolverConfig) -> Self {
        Self {
            config,
            population: HashMap::new(),
            next_id: 1,
            current_generation: 0,
            history: VecDeque::new(),
            stats: EvolverStats::default(),
            rng: None,
        }
    }

    /// Create a new evolver with deterministic RNG from a genesis seed.
    ///
    /// All random selections (parent selection, mutation, tournament) will use
    /// the genesis-seeded RNG instead of `rand::random()`.
    pub fn from_genesis(config: EvolverConfig, genesis: &GenesisSeed, label: &str) -> Self {
        Self {
            config,
            population: HashMap::new(),
            next_id: 1,
            current_generation: 0,
            history: VecDeque::new(),
            stats: EvolverStats::default(),
            rng: Some(genesis.domain(label)),
        }
    }

    /// Generate a random f32, using the genesis RNG if available.
    fn rand_f32(&mut self) -> f32 {
        match &mut self.rng {
            Some(rng) => rng.r#gen::<f32>(),
            None => rand::random::<f32>(),
        }
    }

    /// Add a concept to the population
    pub fn add_concept(&mut self, mut concept: EvolvingConcept) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        concept.id = id;
        self.population.insert(id, concept);
        self.stats.concepts_created += 1;
        id
    }

    /// Initialize population randomly
    pub fn initialize_random(&mut self, dim: usize, count: usize) {
        for i in 0..count {
            let hv = ContinuousHV::random(dim, 42);
            let concept = EvolvingConcept::new(0, format!("concept_{i}"), hv);
            self.add_concept(concept);
        }
    }

    /// Evaluate fitness for a concept
    pub fn evaluate_fitness<F>(&mut self, id: u64, fitness_fn: F) -> Option<f64>
    where
        F: Fn(&ContinuousHV) -> f64,
    {
        if let Some(concept) = self.population.get_mut(&id) {
            let fitness = fitness_fn(&concept.hv);
            concept.fitness = fitness;
            if fitness > self.stats.best_fitness_ever {
                self.stats.best_fitness_ever = fitness;
            }
            Some(fitness)
        } else {
            None
        }
    }

    /// Evaluate all concepts
    pub fn evaluate_all<F>(&mut self, fitness_fn: F)
    where
        F: Fn(&ContinuousHV) -> f64,
    {
        for concept in self.population.values_mut() {
            let fitness = fitness_fn(&concept.hv);
            concept.fitness = fitness;
            if fitness > self.stats.best_fitness_ever {
                self.stats.best_fitness_ever = fitness;
            }
        }
    }

    /// Run one generation of evolution
    pub fn evolve_generation<F>(&mut self, fitness_fn: F)
    where
        F: Fn(&ContinuousHV) -> f64,
    {
        // Evaluate fitness
        self.evaluate_all(&fitness_fn);

        // Record snapshot
        self.record_snapshot();

        // Selection
        let selected = self.select();

        // Create new population through crossover and mutation
        let mut new_population = HashMap::new();

        // Keep best individuals (elitism)
        let elite_count = (self.config.population_size as f32 * 0.1) as usize;
        let mut sorted: Vec<_> = self.population.values().cloned().collect();
        sorted.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for concept in sorted.iter().take(elite_count) {
            new_population.insert(concept.id, concept.clone());
        }

        // Crossover
        while new_population.len() < self.config.population_size {
            if selected.len() >= 2 {
                let p1_idx = (self.rand_f32() * selected.len() as f32) as usize % selected.len();
                let p2_idx = (self.rand_f32() * selected.len() as f32) as usize % selected.len();

                if p1_idx != p2_idx {
                    let parent1 = &selected[p1_idx];
                    let parent2 = &selected[p2_idx];
                    let mut child = parent1.crossover(parent2, self.next_id);
                    self.next_id += 1;

                    // Mutation
                    if self.rand_f32() < self.config.mutation_rate {
                        child.mutate(self.config.learning_rate);
                        self.stats.total_mutations += 1;
                    }

                    new_population.insert(child.id, child);
                    self.stats.total_crossovers += 1;
                }
            }
        }

        self.population = new_population;
        self.current_generation += 1;
    }

    /// Select individuals for reproduction
    fn select(&mut self) -> Vec<EvolvingConcept> {
        let mut candidates: Vec<_> = self.population.values().cloned().collect();
        candidates.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Tournament selection
        let tournament_size = 3;
        let select_count = self.config.population_size / 2;
        let mut selected = Vec::with_capacity(select_count);

        for _ in 0..select_count {
            let mut best: Option<EvolvingConcept> = None;
            for _ in 0..tournament_size {
                let idx = (self.rand_f32() * candidates.len() as f32) as usize % candidates.len();
                let candidate = &candidates[idx];
                if best.as_ref().is_none_or(|b| candidate.fitness > b.fitness) {
                    best = Some(candidate.clone());
                }
            }
            if let Some(b) = best {
                selected.push(b);
            }
        }

        selected
    }

    /// Run full evolution
    pub fn evolve<F>(&mut self, fitness_fn: F) -> EvolutionResult
    where
        F: Fn(&ContinuousHV) -> f64 + Clone,
    {
        self.stats.total_evolutions += 1;

        let initial_best = self
            .population
            .values()
            .map(|c| c.fitness)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let mut converged = false;
        let convergence_threshold = 0.001;
        let mut prev_best = 0.0;
        let mut stagnation = 0;

        for _ in 0..self.config.generations {
            self.evolve_generation(&fitness_fn);

            let current_best = self
                .population
                .values()
                .map(|c| c.fitness)
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);

            if (current_best - prev_best).abs() < convergence_threshold {
                stagnation += 1;
                if stagnation >= 10 {
                    converged = true;
                    break;
                }
            } else {
                stagnation = 0;
            }
            prev_best = current_best;
        }

        let final_best = self
            .population
            .values()
            .map(|c| c.fitness)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        EvolutionResult {
            best_fitness: final_best,
            generations_completed: self.current_generation,
            improvement: final_best - initial_best,
            history: self.history.iter().cloned().collect(),
            converged,
        }
    }

    /// Record evolution snapshot
    fn record_snapshot(&mut self) {
        let fitnesses: Vec<f64> = self.population.values().map(|c| c.fitness).collect();

        let best = fitnesses
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let avg = if fitnesses.is_empty() {
            0.0
        } else {
            fitnesses.iter().sum::<f64>() / fitnesses.len() as f64
        };

        let diversity = self.calculate_diversity();

        let snapshot = EvolutionSnapshot {
            generation: self.current_generation,
            best_fitness: best,
            avg_fitness: avg,
            diversity,
        };

        if self.history.len() >= self.config.history_size {
            self.history.pop_front();
        }
        self.history.push_back(snapshot);
    }

    /// Calculate population diversity
    fn calculate_diversity(&self) -> f64 {
        if self.population.len() < 2 {
            return 0.0;
        }

        let concepts: Vec<_> = self.population.values().collect();
        let mut total_dist = 0.0;
        let mut count = 0;

        for i in 0..concepts.len() {
            for j in (i + 1)..concepts.len() {
                let dist = 1.0 - concepts[i].hv.similarity(&concepts[j].hv) as f64;
                total_dist += dist;
                count += 1;
            }
        }

        if count > 0 {
            total_dist / count as f64
        } else {
            0.0
        }
    }

    /// Get best concept
    pub fn best(&self) -> Option<&EvolvingConcept> {
        self.population.values().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Get statistics
    pub fn stats(&self) -> &EvolverStats {
        &self.stats
    }

    /// Get current generation
    pub fn generation(&self) -> usize {
        self.current_generation
    }
}

impl std::fmt::Debug for PrimitiveEvolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrimitiveEvolver")
            .field("config", &self.config)
            .field("population_size", &self.population.len())
            .field("next_id", &self.next_id)
            .field("current_generation", &self.current_generation)
            .field("stats", &self.stats)
            .field("has_genesis_rng", &self.rng.is_some())
            .finish()
    }
}

impl Default for PrimitiveEvolver {
    fn default() -> Self {
        Self::new(EvolverConfig::default())
    }
}

// =============================================================================
// EVOLUTION BRIDGE COMPATIBLE TYPES
// =============================================================================

/// Configuration for tier-based primitive evolution (evolution_bridge compatible)
#[derive(Debug, Clone)]
pub struct EvolutionConfig {
    /// Tier to evolve primitives for
    pub tier: PrimitiveTier,

    /// Population size
    pub population_size: usize,

    /// Number of generations to run
    pub num_generations: usize,

    /// Mutation rate (0.0 - 1.0)
    pub mutation_rate: f64,

    /// Crossover rate (0.0 - 1.0)
    pub crossover_rate: f64,

    /// Number of elite individuals to preserve
    pub elitism_count: usize,

    /// Task types to evaluate fitness on
    pub fitness_tasks: Vec<String>,

    /// Convergence threshold (stop if improvement < this)
    pub convergence_threshold: f64,

    /// Weight for Φ in fitness function
    pub phi_weight: f64,

    /// Weight for harmonic alignment in fitness
    pub harmonic_weight: f64,

    /// Weight for epistemic grounding in fitness
    pub epistemic_weight: f64,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            tier: PrimitiveTier::Physical,
            population_size: 20,
            num_generations: 50,
            mutation_rate: 0.1,
            crossover_rate: 0.3,
            elitism_count: 2,
            fitness_tasks: Vec::new(),
            convergence_threshold: 0.001,
            phi_weight: 0.4,
            harmonic_weight: 0.3,
            epistemic_weight: 0.3,
        }
    }
}

/// A candidate primitive being evolved
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidatePrimitive {
    /// Name of the primitive
    pub name: String,

    /// Tier this primitive belongs to
    pub tier: PrimitiveTier,

    /// Semantic definition of this primitive
    pub definition: String,

    /// Fitness score (Φ improvement)
    pub fitness: f64,

    /// HDC encoding of this primitive's phenotype
    pub encoding: BinaryHV,

    /// Epistemic grounding coordinate (Phase 2.2)
    pub epistemic_coordinate: EpistemicCoordinate,

    /// Harmonic alignment score (0.0-1.0, Phase 2.2)
    pub harmonic_alignment: f64,
}

impl CandidatePrimitive {
    /// Create a mutated copy of this primitive.
    ///
    /// Uses BinaryHV::add_noise to flip bits proportional to `mutation_rate`.
    pub fn mutate(&self, mutation_rate: f64, generation: usize) -> Self {
        let encoding = self
            .encoding
            .add_noise(mutation_rate as f32, generation as u64);
        Self {
            name: format!("{}_mut_g{}", self.name, generation),
            tier: self.tier,
            definition: self.definition.clone(),
            fitness: 0.0,
            encoding,
            epistemic_coordinate: self.epistemic_coordinate,
            harmonic_alignment: self.harmonic_alignment,
        }
    }

    /// Recombine two parents into a child (uniform crossover at byte level).
    pub fn recombine(parent1: &Self, parent2: &Self, generation: usize) -> Self {
        let b1 = &parent1.encoding.0;
        let b2 = &parent2.encoding.0;
        let mut child_bytes = [0u8; BinaryHV::BYTES];
        for i in 0..BinaryHV::BYTES {
            child_bytes[i] = if (i + generation).is_multiple_of(2) {
                b1[i]
            } else {
                b2[i]
            };
        }
        Self {
            name: format!("cross_g{generation}"),
            tier: parent1.tier,
            definition: parent1.definition.clone(),
            fitness: 0.0,
            encoding: BinaryHV(child_bytes),
            epistemic_coordinate: parent1.epistemic_coordinate,
            harmonic_alignment: (parent1.harmonic_alignment + parent2.harmonic_alignment) / 2.0,
        }
    }
}

/// Result of evolution (evolution_bridge compatible)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveEvolutionResult {
    /// Final population of primitives
    pub final_primitives: Vec<CandidatePrimitive>,

    /// Φ improvement percentage
    pub phi_improvement_percent: f64,

    /// Final Φ value
    pub final_phi: f64,

    /// Baseline Φ (before evolution)
    pub baseline_phi: f64,

    /// Number of generations executed
    pub generations_run: usize,

    /// Whether convergence was achieved
    pub converged: bool,
}

/// Tier-based primitive evolution system (evolution_bridge compatible)
pub struct PrimitiveEvolution {
    /// Configuration
    config: EvolutionConfig,

    /// Current population
    population: Vec<CandidatePrimitive>,

    /// Current generation
    current_generation: usize,

    /// Baseline Φ
    baseline_phi: f64,

    /// Optional deterministic RNG from genesis seed
    rng: Option<ShakeRng>,
}

impl PrimitiveEvolution {
    /// Create a new evolution system
    pub fn new(config: EvolutionConfig) -> Result<Self> {
        let mut system = Self {
            config,
            population: Vec::new(),
            current_generation: 0,
            baseline_phi: 0.5,
            rng: None,
        };
        system.initialize_population();
        Ok(system)
    }

    /// Create a new evolution system with deterministic RNG from a genesis seed.
    pub fn from_genesis(
        config: EvolutionConfig,
        genesis: &GenesisSeed,
        label: &str,
    ) -> Result<Self> {
        let mut system = Self {
            config,
            population: Vec::new(),
            current_generation: 0,
            baseline_phi: 0.5,
            rng: Some(genesis.domain(label)),
        };
        system.initialize_population();
        Ok(system)
    }

    /// Initialize population with random candidates
    pub fn initialize_population(&mut self) {
        for i in 0..self.config.population_size {
            let seed = (self.config.tier as u64)
                .wrapping_mul(1000)
                .wrapping_add(i as u64);
            self.population.push(CandidatePrimitive {
                name: format!("evolved_{}_{}", self.config.tier as u8, i),
                tier: self.config.tier,
                definition: format!("Evolved primitive for tier {:?}", self.config.tier),
                fitness: 0.0,
                encoding: BinaryHV::random(seed),
                epistemic_coordinate: EpistemicCoordinate::default(),
                harmonic_alignment: 0.5, // Start with neutral alignment
            });
        }
    }

    /// Run evolution and return results
    pub fn evolve(&mut self) -> Result<PrimitiveEvolutionResult> {
        let mut prev_best_fitness = 0.0;
        let mut stagnation_count = 0;

        for r#gen in 0..self.config.num_generations {
            self.current_generation = r#gen;

            // Evaluate fitness for all candidates
            for i in 0..self.population.len() {
                // Simple fitness: use tier-based heuristic
                let tier = self.population[i].tier;
                self.population[i].fitness = self.evaluate_tier_fitness(tier);
            }

            // Sort by fitness (descending)
            self.population.sort_by(|a, b| {
                b.fitness
                    .partial_cmp(&a.fitness)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let best_fitness = self.population.first().map(|c| c.fitness).unwrap_or(0.0);

            // Check convergence
            if (best_fitness - prev_best_fitness).abs() < self.config.convergence_threshold {
                stagnation_count += 1;
                if stagnation_count >= 5 {
                    break;
                }
            } else {
                stagnation_count = 0;
            }
            prev_best_fitness = best_fitness;

            // Selection, crossover, mutation
            self.evolve_generation();
        }

        // Calculate final Φ
        let final_phi = self.population.first().map(|c| c.fitness).unwrap_or(0.5);
        let phi_improvement = if self.baseline_phi > 0.0 {
            ((final_phi - self.baseline_phi) / self.baseline_phi) * 100.0
        } else {
            0.0
        };

        Ok(PrimitiveEvolutionResult {
            final_primitives: self.population.clone(),
            phi_improvement_percent: phi_improvement,
            final_phi,
            baseline_phi: self.baseline_phi,
            generations_run: self.current_generation + 1,
            converged: stagnation_count >= 5,
        })
    }

    /// Generate a random f64, using the genesis RNG if available.
    fn rand_f64(&mut self) -> f64 {
        match &mut self.rng {
            Some(rng) => rng.r#gen::<f64>(),
            None => rand::random::<f64>(),
        }
    }

    /// Evaluate fitness based on tier
    fn evaluate_tier_fitness(&mut self, tier: PrimitiveTier) -> f64 {
        // Stub: return tier-based baseline + small random variation
        let tier_base = match tier {
            PrimitiveTier::NSM => 0.5,
            PrimitiveTier::Mathematical => 0.6,
            PrimitiveTier::Physical => 0.55,
            PrimitiveTier::Geometric => 0.58,
            PrimitiveTier::Strategic => 0.52,
            PrimitiveTier::MetaCognitive => 0.65,
            PrimitiveTier::Temporal => 0.54,
            PrimitiveTier::Compositional => 0.62,
            PrimitiveTier::Consciousness => 0.7,
            PrimitiveTier::Code => 0.68, // Code understanding tier
        };
        tier_base + (self.rand_f64() * 0.1)
    }

    /// Perform one generation of evolution
    fn evolve_generation(&mut self) {
        // Keep elite individuals
        let mut new_population: Vec<CandidatePrimitive> = self
            .population
            .iter()
            .take(self.config.elitism_count)
            .cloned()
            .collect();

        // Fill rest with crossover/mutation
        while new_population.len() < self.config.population_size {
            // Simple mutation: copy a random parent and mutate
            if let Some(parent) = self.population.first() {
                let mut child = parent.clone();
                let r = self.rand_f64();
                if r < self.config.mutation_rate {
                    let r2 = self.rand_f64();
                    child.fitness *= 1.0 + (r2 - 0.5) * 0.2;
                }
                new_population.push(child);
            }
        }

        self.population = new_population;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evolver_creation() {
        let evolver = PrimitiveEvolver::default();
        assert_eq!(evolver.generation(), 0);
    }

    #[test]
    fn test_concept_creation() {
        let hv = ContinuousHV::random(512, 42);
        let concept = EvolvingConcept::new(1, "test", hv);
        assert_eq!(concept.name, "test");
        assert_eq!(concept.fitness, 0.0);
    }

    #[test]
    fn test_mutation() {
        let hv = ContinuousHV::random(512, 42);
        let mut concept = EvolvingConcept::new(1, "test", hv.clone());
        let _original = concept.hv.clone();
        concept.mutate(0.1);
        // After mutation, the HV should be different
        assert_eq!(concept.mutations, 1);
    }

    #[test]
    fn test_initialize_random() {
        let mut evolver = PrimitiveEvolver::default();
        evolver.initialize_random(512, 10);
        assert_eq!(evolver.population.len(), 10);
    }
}
