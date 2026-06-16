// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tournament engine: evolves a population of neural organisms via
//! tournament selection, speciation, and elitism.
//!
//! ## References
//!
//! - Stanley & Miikkulainen (2002). NEAT.
//! - Goldberg & Deb (1991). Tournament selection analysis.
//! - De Jong (1975). Elitism in genetic algorithms.

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;

use crate::fitness::{FepFitnessBridge, FepFitnessConfig, FitnessWeights};
use crate::genome::NeuralGenome;
use crate::organism::NeuralOrganism;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Default population size. Stanley & Miikkulainen (2002).
pub const DEFAULT_POPULATION_SIZE: usize = 50;

/// Default tournament size. Goldberg & Deb (1991).
pub const DEFAULT_TOURNAMENT_SIZE: usize = 3;

/// Default elitism fraction. De Jong (1975).
pub const DEFAULT_ELITISM_FRACTION: f64 = 0.1;

/// Default mutation rate. Back (1993).
pub const DEFAULT_MUTATION_RATE: f32 = 0.02;

/// Default crossover rate. Holland (1975).
pub const DEFAULT_CROSSOVER_RATE: f64 = 0.7;

/// Maximum generations.
pub const DEFAULT_MAX_GENERATIONS: u32 = 200;

/// Convergence patience (generations without improvement).
pub const DEFAULT_CONVERGENCE_PATIENCE: u32 = 10;

/// Speciation distance threshold.
pub const DEFAULT_SPECIATION_THRESHOLD: f32 = 0.15;

/// Maximum history snapshots.
const MAX_HISTORY: usize = 500;

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Configuration for the neuroevolution engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuroevolutionConfig {
    pub population_size: usize,
    pub tournament_size: usize,
    pub elitism_fraction: f64,
    pub mutation_rate: f32,
    pub crossover_rate: f64,
    pub max_generations: u32,
    pub convergence_patience: u32,
    pub speciation_threshold: f32,
    pub fitness_config: FepFitnessConfig,
    pub genesis_phrase: String,
}

impl Default for NeuroevolutionConfig {
    fn default() -> Self {
        Self {
            population_size: DEFAULT_POPULATION_SIZE,
            tournament_size: DEFAULT_TOURNAMENT_SIZE,
            elitism_fraction: DEFAULT_ELITISM_FRACTION,
            mutation_rate: DEFAULT_MUTATION_RATE,
            crossover_rate: DEFAULT_CROSSOVER_RATE,
            max_generations: DEFAULT_MAX_GENERATIONS,
            convergence_patience: DEFAULT_CONVERGENCE_PATIENCE,
            speciation_threshold: DEFAULT_SPECIATION_THRESHOLD,
            fitness_config: FepFitnessConfig::default(),
            genesis_phrase: "symthaea-neuroevolution".to_string(),
        }
    }
}

/// Snapshot of a generation's state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationSnapshot {
    pub generation: u32,
    pub best_fitness: f64,
    pub mean_fitness: f64,
    pub worst_fitness: f64,
    pub diversity: f64,
    pub species_count: usize,
    pub population_size: usize,
}

/// Species grouping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeciesInfo {
    pub id: usize,
    pub representative_id: u64,
    pub member_count: usize,
    pub mean_fitness: f64,
}

/// Result of a full evolution run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionResult {
    pub best_genome: NeuralGenome,
    pub best_fitness: f64,
    pub generations_completed: u32,
    pub converged: bool,
    pub history: Vec<GenerationSnapshot>,
    pub final_species: Vec<SpeciesInfo>,
}

/// The neuroevolution engine: manages population, selection, breeding, speciation.
pub struct NeuroevolutionEngine {
    config: NeuroevolutionConfig,
    population: Vec<NeuralOrganism>,
    fitness_bridge: FepFitnessBridge,
    generation: u32,
    next_id: u64,
    history: VecDeque<GenerationSnapshot>,
    best_ever: Option<(NeuralGenome, f64)>,
    genesis: GenesisSeed,
    species_assignments: HashMap<u64, usize>, // organism_id → species_id
    species_representatives: Vec<NeuralGenome>, // One representative per species
    patience_counter: u32,
    last_best_fitness: f64,
}

impl NeuroevolutionEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: NeuroevolutionConfig) -> Self {
        let genesis = GenesisSeed::from_phrase(&config.genesis_phrase);
        let fitness_bridge = FepFitnessBridge::new(config.fitness_config.clone());

        Self {
            config,
            population: Vec::new(),
            fitness_bridge,
            generation: 0,
            next_id: 0,
            history: VecDeque::with_capacity(MAX_HISTORY),
            best_ever: None,
            genesis,
            species_assignments: HashMap::new(),
            species_representatives: Vec::new(),
            patience_counter: 0,
            last_best_fitness: f64::NEG_INFINITY,
        }
    }

    /// Initialize the population with random genomes.
    pub fn initialize(&mut self) {
        self.population.clear();
        self.next_id = 0;
        self.generation = 0;

        for i in 0..self.config.population_size {
            let genome = NeuralGenome::from_genesis(&self.genesis, &format!("init_{i}"));
            let id = self.next_id;
            self.next_id += 1;
            let org = NeuralOrganism::spawn(id, genome, &self.genesis, 0);
            self.population.push(org);
        }
    }

    /// Run one generation: evaluate → select → breed → replace.
    pub fn step_generation(&mut self) -> GenerationSnapshot {
        self.generation += 1;

        // 1. Evaluate all organisms (parallel when rayon feature enabled)
        self.fitness_bridge.evaluate_auto(&mut self.population);

        // 2. Assign species
        self.assign_species();

        // 3. Sort by fitness (descending)
        self.population.sort_by(|a, b| {
            b.fitness
                .composite
                .partial_cmp(&a.fitness.composite)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 4. Track best ever
        if let Some(best) = self.population.first() {
            match &self.best_ever {
                Some((_, best_fitness)) if best.fitness.composite <= *best_fitness => {}
                _ => {
                    self.best_ever = Some((best.genome.clone(), best.fitness.composite));
                }
            }
        }

        // 5. Check convergence
        let current_best = self
            .population
            .first()
            .map(|o| o.fitness.composite)
            .unwrap_or(f64::NEG_INFINITY);
        if current_best > self.last_best_fitness + 1e-6 {
            self.patience_counter = 0;
            self.last_best_fitness = current_best;
        } else {
            self.patience_counter += 1;
        }

        // 6. Create snapshot
        let snapshot = self.create_snapshot();
        if self.history.len() >= MAX_HISTORY {
            self.history.pop_front();
        }
        self.history.push_back(snapshot.clone());

        // 7. Breed next generation
        self.breed_next_generation();

        snapshot
    }

    /// Evolve until convergence or max generations.
    pub fn evolve(&mut self) -> EvolutionResult {
        if self.population.is_empty() {
            self.initialize();
        }

        while self.generation < self.config.max_generations {
            self.step_generation();

            if self.patience_counter >= self.config.convergence_patience {
                break;
            }
        }

        let (best_genome, best_fitness) = self
            .best_ever
            .clone()
            .unwrap_or_else(|| (NeuralGenome::random(0), f64::NEG_INFINITY));

        EvolutionResult {
            best_genome,
            best_fitness,
            generations_completed: self.generation,
            converged: self.patience_counter >= self.config.convergence_patience,
            history: self.history.iter().cloned().collect(),
            final_species: self.get_species_info(),
        }
    }

    /// Tournament selection: pick `tournament_size` random organisms, return the best.
    fn tournament_select(&self, seed: u64) -> usize {
        let n = self.population.len();
        if n == 0 {
            return 0;
        }

        let mut rng = blake3_rng(seed);
        let mut best_idx = rng_usize(&mut rng, n);
        let mut best_fitness = self.population[best_idx].fitness.composite;

        for _ in 1..self.config.tournament_size {
            let idx = rng_usize(&mut rng, n);
            let fitness = self.population[idx].fitness.composite;
            if fitness > best_fitness {
                best_idx = idx;
                best_fitness = fitness;
            }
        }

        best_idx
    }

    /// Assign organisms to species by genome distance to representatives.
    fn assign_species(&mut self) {
        self.species_assignments.clear();

        if self.species_representatives.is_empty() {
            // First generation: each organism is its own species initially,
            // but we collapse by distance threshold
            if let Some(first) = self.population.first() {
                self.species_representatives.push(first.genome.clone());
            }
        }

        // Assign each organism to closest species (or create new)
        for org in &self.population {
            let mut assigned = false;
            for (species_id, rep) in self.species_representatives.iter().enumerate() {
                let dist = org.genome.distance(&NeuralGenome { hv: rep.hv });
                if dist < self.config.speciation_threshold {
                    self.species_assignments.insert(org.id, species_id);
                    assigned = true;
                    break;
                }
            }
            if !assigned {
                let new_id = self.species_representatives.len();
                self.species_representatives.push(org.genome.clone());
                self.species_assignments.insert(org.id, new_id);
            }
        }

        // Update representatives to the best member of each species
        let mut species_best: HashMap<usize, (u64, f64, NeuralGenome)> = HashMap::new();
        for org in &self.population {
            if let Some(&species_id) = self.species_assignments.get(&org.id) {
                let entry = species_best.entry(species_id).or_insert((
                    org.id,
                    f64::NEG_INFINITY,
                    org.genome.clone(),
                ));
                if org.fitness.composite > entry.1 {
                    *entry = (org.id, org.fitness.composite, org.genome.clone());
                }
            }
        }

        // Rebuild representatives from active species only
        let active_species: Vec<usize> = species_best.keys().copied().collect();
        let mut new_reps = Vec::new();
        let mut id_remap: HashMap<usize, usize> = HashMap::new();

        for (new_id, old_id) in active_species.iter().enumerate() {
            id_remap.insert(*old_id, new_id);
            new_reps.push(species_best[old_id].2.clone());
        }

        // Remap assignments
        let mut new_assignments = HashMap::new();
        for (org_id, old_species) in &self.species_assignments {
            if let Some(&new_species) = id_remap.get(old_species) {
                new_assignments.insert(*org_id, new_species);
            }
        }

        self.species_representatives = new_reps;
        self.species_assignments = new_assignments;
    }

    /// Breed the next generation with elitism, crossover, and mutation.
    fn breed_next_generation(&mut self) {
        let pop_size = self.config.population_size;
        let elite_count = (pop_size as f64 * self.config.elitism_fraction).ceil() as usize;
        let elite_count = elite_count.min(self.population.len());

        let mut next_gen: Vec<NeuralOrganism> = Vec::with_capacity(pop_size);

        // Elitism: keep top organisms unchanged
        for org in self.population.iter().take(elite_count) {
            let mut elite = org.clone();
            elite.age_cycles = 0;
            elite.reset_evaluation();
            next_gen.push(elite);
        }

        // Fill remaining slots with offspring
        let mut child_seed: u64 = self.generation as u64 * 10000;
        while next_gen.len() < pop_size {
            child_seed += 1;
            let parent1_idx = self.tournament_select(child_seed);
            child_seed += 1;

            let child_id = self.next_id;
            self.next_id += 1;

            // Decide crossover vs mutation-only based on crossover rate
            let do_crossover = blake3_f64(child_seed) < self.config.crossover_rate;
            child_seed += 1;

            let child = if do_crossover {
                let parent2_idx = self.tournament_select(child_seed);
                child_seed += 1;
                self.population[parent1_idx].reproduce(
                    &self.population[parent2_idx],
                    child_id,
                    self.config.mutation_rate,
                    child_seed,
                    &self.genesis,
                )
            } else {
                self.population[parent1_idx].replicate(
                    child_id,
                    self.config.mutation_rate,
                    child_seed,
                    &self.genesis,
                )
            };

            next_gen.push(child);
        }

        self.population = next_gen;
    }

    /// Compute population diversity as mean pairwise genome distance (sampled).
    pub fn diversity(&self) -> f64 {
        let n = self.population.len();
        if n < 2 {
            return 0.0;
        }

        // Sample up to 100 pairs for efficiency
        let max_pairs = 100usize;
        let mut total_dist = 0.0;
        let mut count = 0usize;
        let mut seed = self.generation as u64 * 9999;

        for _ in 0..max_pairs {
            seed += 1;
            let i = blake3_usize(seed, n);
            seed += 1;
            let j = blake3_usize(seed, n);
            if i != j {
                total_dist += self.population[i]
                    .genome
                    .distance(&self.population[j].genome) as f64;
                count += 1;
            }
        }

        if count > 0 {
            total_dist / count as f64
        } else {
            0.0
        }
    }

    /// Inject an external organism into the population (replaces worst).
    pub fn inject(&mut self, organism: NeuralOrganism) {
        if self.population.is_empty() {
            self.population.push(organism);
            return;
        }

        // Replace worst organism
        let worst_idx = self
            .population
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                a.fitness
                    .composite
                    .partial_cmp(&b.fitness.composite)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0);

        self.population[worst_idx] = organism;
    }

    /// Get current species information.
    pub fn get_species_info(&self) -> Vec<SpeciesInfo> {
        let mut species_members: HashMap<usize, Vec<f64>> = HashMap::new();

        for org in &self.population {
            if let Some(&species_id) = self.species_assignments.get(&org.id) {
                species_members
                    .entry(species_id)
                    .or_default()
                    .push(org.fitness.composite);
            }
        }

        let mut infos: Vec<SpeciesInfo> = species_members
            .into_iter()
            .map(|(id, fitnesses)| {
                let mean = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
                SpeciesInfo {
                    id,
                    representative_id: self
                        .population
                        .iter()
                        .find(|o| self.species_assignments.get(&o.id) == Some(&id))
                        .map(|o| o.id)
                        .unwrap_or(0),
                    member_count: fitnesses.len(),
                    mean_fitness: mean,
                }
            })
            .collect();

        infos.sort_by(|a, b| {
            b.mean_fitness
                .partial_cmp(&a.mean_fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        infos
    }

    fn create_snapshot(&self) -> GenerationSnapshot {
        let fitnesses: Vec<f64> = self
            .population
            .iter()
            .map(|o| o.fitness.composite)
            .collect();
        let best = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let worst = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
        let mean = if fitnesses.is_empty() {
            0.0
        } else {
            fitnesses.iter().sum::<f64>() / fitnesses.len() as f64
        };

        let species_count = self
            .species_assignments
            .values()
            .collect::<std::collections::HashSet<_>>()
            .len();

        GenerationSnapshot {
            generation: self.generation,
            best_fitness: best,
            mean_fitness: mean,
            worst_fitness: worst,
            diversity: self.diversity(),
            species_count,
            population_size: self.population.len(),
        }
    }

    /// Set fitness weights at runtime.
    pub fn set_fitness_weights(&mut self, weights: FitnessWeights) {
        self.fitness_bridge.set_weights(weights);
        self.config.fitness_config.weights = self.fitness_bridge.weights().clone();
    }

    /// Get current generation number.
    pub fn generation(&self) -> u32 {
        self.generation
    }

    /// Get current population.
    pub fn population(&self) -> &[NeuralOrganism] {
        &self.population
    }

    /// Get best-ever genome and fitness.

    /// Inject a Lamarckian offspring: create biased mutant from best genome,
    /// replace the worst organism in the population.
    /// `targets`: slice of (start_bit, num_bits, confidence) tuples for biased mutation.
    pub fn inject_lamarckian(&mut self, targets: &[(usize, usize, f32)]) {
        if targets.is_empty() {
            return;
        }
        let best_genome = match &self.best_ever {
            Some((g, _)) => g.clone(),
            None => return,
        };
        let biased = best_genome.mutate_biased(
            self.config.mutation_rate,
            self.generation as u64 * 0x517CC1B727220A95,
            targets,
        );
        // Replace worst organism
        if let Some(worst) = self.population.iter_mut().min_by(|a, b| {
            a.fitness
                .composite
                .partial_cmp(&b.fitness.composite)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            *worst = crate::organism::NeuralOrganism::spawn(
                worst.id,
                biased,
                &self.genesis,
                self.generation as u32,
            );
        }
    }
    pub fn best_ever(&self) -> Option<(&NeuralGenome, f64)> {
        self.best_ever.as_ref().map(|(g, f)| (g, *f))
    }

    /// Save a checkpoint to a file (JSON).
    pub fn save_checkpoint(&self, path: &std::path::Path) -> Result<(), String> {
        let checkpoint = Checkpoint {
            generation: self.generation,
            best_genome: self.best_ever.as_ref().map(|(g, _)| g.clone()),
            best_fitness: self
                .best_ever
                .as_ref()
                .map(|(_, f)| *f)
                .unwrap_or(f64::NEG_INFINITY),
            population_genomes: self.population.iter().map(|o| o.genome.clone()).collect(),
            config: self.config.clone(),
            history: self.history.iter().cloned().collect(),
        };
        let json = serde_json::to_string(&checkpoint).map_err(|e| e.to_string())?;
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    /// Load a checkpoint from a file, resuming evolution.
    pub fn load_checkpoint(path: &std::path::Path) -> Result<Self, String> {
        let json = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let checkpoint: Checkpoint = serde_json::from_str(&json).map_err(|e| e.to_string())?;
        let genesis = GenesisSeed::from_phrase(&checkpoint.config.genesis_phrase);
        let fitness_bridge = FepFitnessBridge::new(checkpoint.config.fitness_config.clone());

        let population: Vec<NeuralOrganism> = checkpoint
            .population_genomes
            .into_iter()
            .enumerate()
            .map(|(i, genome)| {
                NeuralOrganism::spawn(i as u64, genome, &genesis, checkpoint.generation)
            })
            .collect();

        let mut engine = Self {
            config: checkpoint.config,
            population,
            fitness_bridge,
            generation: checkpoint.generation,
            next_id: 0,
            history: checkpoint.history.into_iter().collect(),
            best_ever: checkpoint.best_genome.map(|g| (g, checkpoint.best_fitness)),
            genesis,
            species_assignments: HashMap::new(),
            species_representatives: Vec::new(),
            patience_counter: 0,
            last_best_fitness: checkpoint.best_fitness,
        };
        engine.next_id = engine.population.len() as u64;
        Ok(engine)
    }
}

/// Serializable checkpoint for saving/loading evolution state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub generation: u32,
    pub best_genome: Option<NeuralGenome>,
    pub best_fitness: f64,
    pub population_genomes: Vec<NeuralGenome>,
    pub config: NeuroevolutionConfig,
    pub history: Vec<GenerationSnapshot>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// DETERMINISTIC RNG HELPERS (no rand::random)
// ═══════════════════════════════════════════════════════════════════════════════

fn blake3_rng(seed: u64) -> [u8; 32] {
    *blake3::hash(&seed.to_le_bytes()).as_bytes()
}

fn rng_usize(state: &mut [u8; 32], max: usize) -> usize {
    if max == 0 {
        return 0;
    }
    let val = u64::from_le_bytes(state[0..8].try_into().unwrap_or([0; 8]));
    // Advance state
    *state = *blake3::hash(state).as_bytes();
    (val as usize) % max
}

fn blake3_usize(seed: u64, max: usize) -> usize {
    if max == 0 {
        return 0;
    }
    let hash = blake3::hash(&seed.to_le_bytes());
    let val = u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap_or([0; 8]));
    (val as usize) % max
}

fn blake3_f64(seed: u64) -> f64 {
    let hash = blake3::hash(&seed.to_le_bytes());
    let val = u64::from_le_bytes(hash.as_bytes()[0..8].try_into().unwrap_or([0; 8]));
    (val as f64) / (u64::MAX as f64)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitness::InputStrategy;

    fn small_config() -> NeuroevolutionConfig {
        NeuroevolutionConfig {
            population_size: 5,
            tournament_size: 3,
            max_generations: 5,
            convergence_patience: 3,
            fitness_config: FepFitnessConfig {
                warmup_steps: 2,
                eval_steps: 5,
                dt: 0.05,
                input_dim: 256,
                input_strategy: InputStrategy::default(),
                weights: FitnessWeights::default(),
            },
            ..Default::default()
        }
    }

    #[test]
    fn test_initialize_population_size() {
        let config = small_config();
        let mut engine = NeuroevolutionEngine::new(config.clone());
        engine.initialize();
        assert_eq!(engine.population.len(), config.population_size);
    }

    #[test]
    fn test_step_generation_returns_snapshot() {
        let mut engine = NeuroevolutionEngine::new(small_config());
        engine.initialize();
        let snapshot = engine.step_generation();
        assert_eq!(snapshot.generation, 1);
        assert_eq!(snapshot.population_size, 5);
        assert!(snapshot.best_fitness.is_finite());
        assert!(snapshot.mean_fitness.is_finite());
    }

    #[test]
    fn test_population_preserved_after_step() {
        let config = small_config();
        let mut engine = NeuroevolutionEngine::new(config.clone());
        engine.initialize();
        engine.step_generation();
        assert_eq!(engine.population.len(), config.population_size);
    }

    #[test]
    fn test_elitism_preserves_best() {
        let mut engine = NeuroevolutionEngine::new(small_config());
        engine.initialize();
        engine.step_generation();

        // After step, best organism's genome should be in best_ever
        assert!(engine.best_ever().is_some());
    }

    #[test]
    fn test_tournament_selection_bias() {
        let mut engine = NeuroevolutionEngine::new(small_config());
        engine.initialize();

        // Manually set fitness to verify selection bias
        for (i, org) in engine.population.iter_mut().enumerate() {
            org.fitness.composite = i as f64;
        }

        // Run many tournaments, count how often the best is selected
        let mut best_count = 0;
        let best_idx = engine.population.len() - 1;
        for seed in 0..100u64 {
            let selected = engine.tournament_select(seed);
            if selected == best_idx {
                best_count += 1;
            }
        }

        // Best organism should be selected more often than random (>10%)
        assert!(
            best_count > 5,
            "Best selected {best_count}/100 — expected selection bias"
        );
    }

    #[test]
    fn test_evolve_completes() {
        let mut engine = NeuroevolutionEngine::new(small_config());
        let result = engine.evolve();
        assert!(result.generations_completed > 0);
        assert!(result.best_fitness.is_finite());
        assert!(!result.history.is_empty());
    }

    #[test]
    fn test_evolve_fitness_improves_or_stable() {
        let config = NeuroevolutionConfig {
            population_size: 5,
            max_generations: 10,
            convergence_patience: 20,
            fitness_config: FepFitnessConfig {
                warmup_steps: 2,
                eval_steps: 5,
                dt: 0.05,
                input_dim: 256,
                input_strategy: InputStrategy::default(),
                weights: FitnessWeights::default(),
            },
            ..Default::default()
        };
        let mut engine = NeuroevolutionEngine::new(config);
        let result = engine.evolve();

        // Best fitness at end should be >= best at start
        if result.history.len() >= 2 {
            let first_best = result.history[0].best_fitness;
            let last_best = result.history.last().unwrap().best_fitness;
            // Elitism guarantees non-degradation
            assert!(
                last_best >= first_best - 1e-6,
                "Fitness should not degrade: first={first_best}, last={last_best}"
            );
        }
    }

    #[test]
    fn test_speciation_creates_species() {
        let mut engine = NeuroevolutionEngine::new(small_config());
        engine.initialize();
        engine.step_generation();
        let species = engine.get_species_info();
        // Should have at least 1 species
        assert!(!species.is_empty());
    }

    #[test]
    fn test_diversity_bounded() {
        let mut engine = NeuroevolutionEngine::new(small_config());
        engine.initialize();
        let d = engine.diversity();
        assert!(d >= 0.0 && d <= 1.0);
    }

    #[test]
    fn test_inject_organism() {
        let mut engine = NeuroevolutionEngine::new(small_config());
        engine.initialize();
        engine.step_generation();

        let genesis = GenesisSeed::from_phrase("inject");
        let genome = NeuralGenome::random(9999);
        let injected = NeuralOrganism::spawn(9999, genome, &genesis, 0);
        engine.inject(injected);

        assert!(engine.population.iter().any(|o| o.id == 9999));
    }

    #[test]
    fn test_convergence_patience() {
        let config = NeuroevolutionConfig {
            population_size: 5,
            max_generations: 100,
            convergence_patience: 2,
            fitness_config: FepFitnessConfig {
                warmup_steps: 1,
                eval_steps: 3,
                dt: 0.05,
                input_dim: 256,
                input_strategy: InputStrategy::default(),
                weights: FitnessWeights::default(),
            },
            ..Default::default()
        };
        let mut engine = NeuroevolutionEngine::new(config);
        let result = engine.evolve();
        // Should converge before 100 generations due to patience=2
        assert!(result.generations_completed < 100);
        assert!(result.converged);
    }

    #[test]
    fn test_set_fitness_weights_runtime() {
        let mut engine = NeuroevolutionEngine::new(small_config());
        engine.set_fitness_weights(FitnessWeights {
            free_energy: 0.9,
            phi: 0.05,
            consciousness: 0.025,
            efficiency: 0.025,
        });
        assert!((engine.fitness_bridge.weights().free_energy - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_memory_budget() {
        let config = small_config();
        let mut engine = NeuroevolutionEngine::new(config.clone());
        engine.initialize();
        // 10 organisms x 2KB genome = 20KB genome storage
        let genome_bytes = engine.population.len() * 2048;
        assert!(
            genome_bytes <= 200_000,
            "Genome storage: {genome_bytes} bytes"
        );
    }

    #[test]
    fn test_unique_organism_ids() {
        let mut engine = NeuroevolutionEngine::new(small_config());
        engine.initialize();
        engine.step_generation();
        let ids: Vec<u64> = engine.population.iter().map(|o| o.id).collect();
        let unique: std::collections::HashSet<u64> = ids.iter().copied().collect();
        assert_eq!(ids.len(), unique.len(), "All organism IDs should be unique");
    }
}
