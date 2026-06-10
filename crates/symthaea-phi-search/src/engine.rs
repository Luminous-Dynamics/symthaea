// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Phi architecture search engine with multiple optimization strategies.

use serde::{Deserialize, Serialize};

use symthaea_core::hdc::HDC_DIMENSION;

use super::genome::ArchitectureGenome;
use super::phenotype::DecodedArchitecture;
use super::phi_gradient::{GradientVelocity, PhiGradient};

// ═══════════════════════════════════════════════════════════════════════════════
// SEARCH CONFIGURATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Search strategy for architecture optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SearchStrategy {
    /// Random sampling from architecture space
    Random,
    /// Evolutionary search with mutation and selection
    Evolutionary,
    /// Gradient-guided search using Phi gradients
    GradientGuided,
    /// Hybrid: evolutionary exploration + gradient exploitation
    Hybrid,
    /// Gradient-guided evolution: mutations follow Phi gradient direction
    GradientEvolutionary,
    /// Momentum-based optimization with adaptive learning rate
    MomentumOptimization,
    /// Multi-population island model with gradient migration
    IslandGradient,
}

/// Configuration for architecture search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchConfig {
    /// Population size for evolutionary search
    pub population_size: usize,

    /// Number of elite individuals to preserve
    pub elite_count: usize,

    /// Mutation rate (0.0-1.0)
    pub mutation_rate: f32,

    /// Crossover rate (0.0-1.0)
    pub crossover_rate: f32,

    /// Learning rate for gradient-guided search
    pub learning_rate: f32,

    /// Epsilon for gradient computation
    pub gradient_epsilon: f32,

    /// Number of gradient steps between evolutionary generations
    pub gradient_steps_per_generation: usize,

    /// Minimum nodes to search
    pub min_nodes: usize,

    /// Maximum nodes to search
    pub max_nodes: usize,

    /// HDC dimension
    pub hdc_dim: usize,

    /// Random seed
    pub seed: u64,

    /// Whether to use parallelism
    pub parallel: bool,

    /// Number of random architectures to sample for random search
    pub random_samples: usize,

    /// Momentum coefficient for momentum-based optimization (0.0-1.0)
    pub momentum: f32,

    /// Noise scale for gradient-guided mutations (exploration)
    pub gradient_noise_scale: f32,

    /// Number of islands for island model
    pub num_islands: usize,

    /// Migration interval (generations between migrations)
    pub migration_interval: usize,

    /// Migration rate (fraction of population to migrate)
    pub migration_rate: f32,

    /// Convergence threshold (stop when gradient magnitude < this)
    pub convergence_threshold: f64,

    /// Maximum generations without improvement before early stopping
    pub patience: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            elite_count: 2,
            mutation_rate: 0.3,
            crossover_rate: 0.6,
            learning_rate: 0.1,
            gradient_epsilon: 0.01,
            gradient_steps_per_generation: 5,
            min_nodes: 8,
            max_nodes: 64,
            hdc_dim: HDC_DIMENSION,
            seed: 42,
            parallel: false,
            random_samples: 100,
            momentum: 0.9,
            gradient_noise_scale: 0.05,
            num_islands: 4,
            migration_interval: 5,
            migration_rate: 0.1,
            convergence_threshold: 1e-6,
            patience: 20,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SEARCH RESULT
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of architecture search
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Best Phi achieved
    pub best_phi: f64,

    /// Best architecture genome found
    pub best_architecture: ArchitectureGenome,

    /// History of best Phi values per generation
    pub phi_history: Vec<f64>,

    /// Total number of architectures evaluated
    pub evaluations: usize,

    /// Search strategy used
    pub strategy: SearchStrategy,

    /// Elapsed time in milliseconds (if tracked)
    pub elapsed_ms: Option<u64>,
}

/// Individual in the population
#[derive(Debug, Clone)]
pub struct Individual {
    /// Architecture genome
    pub genome: ArchitectureGenome,

    /// Phi fitness value
    pub fitness: f64,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PHI ARCHITECTURE SEARCH
// ═══════════════════════════════════════════════════════════════════════════════

/// Main architecture search engine
///
/// Uses consciousness (Phi) as the fitness function to evolve network architectures
/// toward higher levels of integrated information.
pub struct PhiArchitectureSearch {
    /// Search configuration
    config: SearchConfig,

    /// Current population (for evolutionary search)
    population: Vec<Individual>,

    /// Best individual found so far
    pub(crate) best: Option<Individual>,

    /// History of best Phi values
    phi_history: Vec<f64>,

    /// Generation counter
    pub(crate) generation: usize,

    /// Total evaluations
    pub(crate) evaluations: usize,

    /// PRNG state
    rng_state: u64,
}

impl PhiArchitectureSearch {
    /// Create a new architecture search engine
    pub fn new(config: SearchConfig) -> Self {
        Self {
            rng_state: config.seed,
            config,
            population: Vec::new(),
            best: None,
            phi_history: Vec::new(),
            generation: 0,
            evaluations: 0,
        }
    }

    /// Initialize population
    fn initialize_population(&mut self) {
        self.population.clear();

        for i in 0..self.config.population_size {
            let genome = ArchitectureGenome::random(self.config.seed + i as u64 * 1000);
            let arch = DecodedArchitecture::from_genome(&genome);
            let fitness = arch.compute_phi();
            self.evaluations += 1;

            let individual = Individual { genome, fitness };

            if self.best.is_none()
                || fitness
                    > self
                        .best
                        .as_ref()
                        .expect("best individual must be set after initialization")
                        .fitness
            {
                self.best = Some(individual.clone());
            }

            self.population.push(individual);
        }

        // Sort by fitness (descending)
        self.population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Run random search
    pub fn random_search(&mut self) -> SearchResult {
        let mut best_genome = ArchitectureGenome::default();
        let mut best_phi = 0.0;
        let mut phi_history = Vec::new();

        for i in 0..self.config.random_samples {
            let genome = ArchitectureGenome::random(self.config.seed + i as u64);
            let arch = DecodedArchitecture::from_genome(&genome);
            let phi = arch.compute_phi();
            self.evaluations += 1;

            if phi > best_phi {
                best_phi = phi;
                best_genome = genome;
            }

            phi_history.push(best_phi);
        }

        SearchResult {
            best_phi,
            best_architecture: best_genome,
            phi_history,
            evaluations: self.evaluations,
            strategy: SearchStrategy::Random,
            elapsed_ms: None,
        }
    }

    /// Run evolutionary search
    pub fn evolutionary_search(&mut self, generations: usize) -> SearchResult {
        self.initialize_population();
        self.phi_history.push(
            self.best
                .as_ref()
                .expect("best individual must be set after initialization")
                .fitness,
        );

        for r#gen in 0..generations {
            self.generation = r#gen + 1;
            self.evolve_generation();
            self.phi_history.push(
                self.best
                    .as_ref()
                    .expect("best individual must be set after initialization")
                    .fitness,
            );
        }

        SearchResult {
            best_phi: self
                .best
                .as_ref()
                .expect("best individual must be set after initialization")
                .fitness,
            best_architecture: self
                .best
                .as_ref()
                .expect("best individual must be set after initialization")
                .genome
                .clone(),
            phi_history: self.phi_history.clone(),
            evaluations: self.evaluations,
            strategy: SearchStrategy::Evolutionary,
            elapsed_ms: None,
        }
    }

    /// Run gradient-guided search
    pub fn gradient_search(&mut self, steps: usize) -> SearchResult {
        // Start with random architecture
        let mut genome = ArchitectureGenome::random(self.config.seed);
        let arch = DecodedArchitecture::from_genome(&genome);
        let mut best_phi = arch.compute_phi();
        self.evaluations += 1;

        let mut phi_history = vec![best_phi];

        for _step in 0..steps {
            // Compute gradient
            let gradient = PhiGradient::compute(&genome, self.config.gradient_epsilon);
            self.evaluations += 12; // ~6 parameters * 2 (forward/backward)

            // Apply gradient
            gradient.apply(&mut genome, self.config.learning_rate);

            // Evaluate new architecture
            let arch = DecodedArchitecture::from_genome(&genome);
            let phi = arch.compute_phi();
            self.evaluations += 1;

            if phi > best_phi {
                best_phi = phi;
            }

            phi_history.push(best_phi);
        }

        SearchResult {
            best_phi,
            best_architecture: genome,
            phi_history,
            evaluations: self.evaluations,
            strategy: SearchStrategy::GradientGuided,
            elapsed_ms: None,
        }
    }

    /// Run hybrid search (evolutionary + gradient)
    pub fn hybrid_search(&mut self, generations: usize) -> SearchResult {
        self.initialize_population();
        self.phi_history.push(
            self.best
                .as_ref()
                .expect("best individual must be set after initialization")
                .fitness,
        );

        for r#gen in 0..generations {
            self.generation = r#gen + 1;

            // Evolutionary step
            self.evolve_generation();

            // Gradient refinement on top individuals
            self.gradient_refine_elites();

            self.phi_history.push(
                self.best
                    .as_ref()
                    .expect("best individual must be set after initialization")
                    .fitness,
            );
        }

        SearchResult {
            best_phi: self
                .best
                .as_ref()
                .expect("best individual must be set after initialization")
                .fitness,
            best_architecture: self
                .best
                .as_ref()
                .expect("best individual must be set after initialization")
                .genome
                .clone(),
            phi_history: self.phi_history.clone(),
            evaluations: self.evaluations,
            strategy: SearchStrategy::Hybrid,
            elapsed_ms: None,
        }
    }

    /// Main search entry point
    pub fn search(&mut self, strategy: SearchStrategy, iterations: usize) -> SearchResult {
        match strategy {
            SearchStrategy::Random => self.random_search(),
            SearchStrategy::Evolutionary => self.evolutionary_search(iterations),
            SearchStrategy::GradientGuided => self.gradient_search(iterations),
            SearchStrategy::Hybrid => self.hybrid_search(iterations),
            SearchStrategy::GradientEvolutionary => self.gradient_evolutionary_search(iterations),
            SearchStrategy::MomentumOptimization => self.momentum_search(iterations),
            SearchStrategy::IslandGradient => self.island_gradient_search(iterations),
        }
    }

    /// Gradient-guided evolutionary search
    ///
    /// Combines evolutionary selection with gradient-informed mutations.
    /// Mutations follow the Phi gradient direction rather than being random,
    /// enabling more efficient exploration of the consciousness landscape.
    pub fn gradient_evolutionary_search(&mut self, generations: usize) -> SearchResult {
        self.initialize_population();
        self.phi_history.push(
            self.best
                .as_ref()
                .expect("best individual must be set after initialization")
                .fitness,
        );

        // Compute initial gradients for the population
        let mut population_gradients: Vec<PhiGradient> = self
            .population
            .iter()
            .map(|ind| PhiGradient::compute(&ind.genome, self.config.gradient_epsilon))
            .collect();
        self.evaluations += self.population.len() * 12;

        let mut no_improvement_count = 0;
        let mut prev_best = self
            .best
            .as_ref()
            .expect("best individual must be set after initialization")
            .fitness;

        for r#gen in 0..generations {
            self.generation = r#gen + 1;

            // Gradient-guided evolution step
            self.evolve_generation_with_gradients(&population_gradients);

            // Update gradients for new population
            population_gradients = self
                .population
                .iter()
                .map(|ind| PhiGradient::compute(&ind.genome, self.config.gradient_epsilon))
                .collect();
            self.evaluations += self.population.len() * 12;

            let current_best = self
                .best
                .as_ref()
                .expect("best individual must be set after initialization")
                .fitness;
            self.phi_history.push(current_best);

            // Early stopping check
            if (current_best - prev_best).abs() < self.config.convergence_threshold {
                no_improvement_count += 1;
                if no_improvement_count >= self.config.patience {
                    break;
                }
            } else {
                no_improvement_count = 0;
            }
            prev_best = current_best;
        }

        SearchResult {
            best_phi: self
                .best
                .as_ref()
                .expect("best individual must be set after initialization")
                .fitness,
            best_architecture: self
                .best
                .as_ref()
                .expect("best individual must be set after initialization")
                .genome
                .clone(),
            phi_history: self.phi_history.clone(),
            evaluations: self.evaluations,
            strategy: SearchStrategy::GradientEvolutionary,
            elapsed_ms: None,
        }
    }

    /// Evolve generation using gradient-guided mutations
    fn evolve_generation_with_gradients(&mut self, gradients: &[PhiGradient]) {
        let mut new_population = Vec::with_capacity(self.config.population_size);

        // Preserve elites (unchanged)
        for i in 0..self.config.elite_count.min(self.population.len()) {
            new_population.push(self.population[i].clone());
        }

        // Generate offspring with gradient-guided mutations
        while new_population.len() < self.config.population_size {
            // Tournament selection
            let parent_idx = self.tournament_select_idx();
            let parent_genome = self.population[parent_idx].genome.clone();
            let parent_gradient = &gradients[parent_idx.min(gradients.len() - 1)];

            let mut child_genome = parent_genome;

            // Apply gradient-guided mutation
            child_genome.mutate_with_gradient(
                parent_gradient,
                self.config.learning_rate,
                self.config.gradient_noise_scale,
                self.next_u64(),
            );

            // Evaluate
            let arch = DecodedArchitecture::from_genome(&child_genome);
            let fitness = arch.compute_phi();
            self.evaluations += 1;

            let child = Individual {
                genome: child_genome,
                fitness,
            };

            // Update best
            if fitness
                > self
                    .best
                    .as_ref()
                    .expect("best individual must be set after initialization")
                    .fitness
            {
                self.best = Some(child.clone());
            }

            new_population.push(child);
        }

        // Replace population
        self.population = new_population;
        self.population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Tournament selection returning index
    fn tournament_select_idx(&mut self) -> usize {
        let tournament_size = 3;
        let mut best_idx = (self.next_u64() as usize) % self.population.len();

        for _ in 1..tournament_size {
            let idx = (self.next_u64() as usize) % self.population.len();
            if self.population[idx].fitness > self.population[best_idx].fitness {
                best_idx = idx;
            }
        }

        best_idx
    }

    /// Momentum-based optimization search
    ///
    /// Uses momentum to accelerate convergence and escape local optima.
    /// Maintains velocity vectors for smooth optimization trajectories.
    pub fn momentum_search(&mut self, steps: usize) -> SearchResult {
        // Initialize multiple starting points
        let num_starts = self.config.population_size.min(5);
        let mut best_genome = ArchitectureGenome::random(self.config.seed);
        let mut best_phi = 0.0;
        let mut phi_history = Vec::new();

        for start in 0..num_starts {
            let mut genome = ArchitectureGenome::random(self.config.seed + start as u64 * 1000);
            let mut velocity = GradientVelocity::new();

            let arch = DecodedArchitecture::from_genome(&genome);
            let mut current_phi = arch.compute_phi();
            let _ = current_phi; // Suppress unused_assignments warning - will be overwritten
            self.evaluations += 1;

            let mut prev_gradient: Option<PhiGradient> = None;

            for step in 0..steps / num_starts {
                // Compute gradient
                let gradient = PhiGradient::compute(&genome, self.config.gradient_epsilon);
                self.evaluations += 12;

                // Check for convergence (gradient alignment)
                if let Some(ref prev) = prev_gradient {
                    let similarity = gradient.cosine_similarity(prev);
                    if gradient.magnitude < self.config.convergence_threshold || similarity > 0.99 {
                        // Converged, try random restart
                        break;
                    }
                }

                // Apply momentum update
                genome.mutate_with_momentum(
                    &gradient,
                    &mut velocity,
                    self.config.momentum,
                    self.config.learning_rate,
                );

                // Evaluate
                let arch = DecodedArchitecture::from_genome(&genome);
                current_phi = arch.compute_phi();
                self.evaluations += 1;

                if current_phi > best_phi {
                    best_phi = current_phi;
                    best_genome = genome.clone();
                }

                phi_history.push(best_phi);
                prev_gradient = Some(gradient);

                // Velocity decay for annealing effect
                if step % 10 == 0 {
                    velocity.decay(0.95);
                }
            }
        }

        SearchResult {
            best_phi,
            best_architecture: best_genome,
            phi_history,
            evaluations: self.evaluations,
            strategy: SearchStrategy::MomentumOptimization,
            elapsed_ms: None,
        }
    }

    /// Island model with gradient-guided migration
    ///
    /// Maintains multiple populations (islands) that evolve independently,
    /// with periodic migration of best individuals. Migrants are refined
    /// using gradient descent before being inserted into target islands.
    pub fn island_gradient_search(&mut self, generations: usize) -> SearchResult {
        let num_islands = self.config.num_islands.max(2);
        let island_size = self.config.population_size / num_islands;

        // Initialize islands
        let mut islands: Vec<Vec<Individual>> = (0..num_islands)
            .map(|i| {
                let mut island = Vec::with_capacity(island_size);
                for j in 0..island_size {
                    let genome = ArchitectureGenome::random(
                        self.config.seed + (i * island_size + j) as u64 * 100,
                    );
                    let arch = DecodedArchitecture::from_genome(&genome);
                    let fitness = arch.compute_phi();
                    self.evaluations += 1;
                    island.push(Individual { genome, fitness });
                }
                island.sort_by(|a, b| {
                    b.fitness
                        .partial_cmp(&a.fitness)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                island
            })
            .collect();

        // Track global best
        let mut global_best: Option<Individual> = None;
        for island in &islands {
            if let Some(best) = island.first() {
                if global_best.is_none()
                    || best.fitness
                        > global_best
                            .as_ref()
                            .expect("global_best must be set after first island")
                            .fitness
                {
                    global_best = Some(best.clone());
                }
            }
        }

        let mut phi_history = vec![
            global_best
                .as_ref()
                .expect("global_best must be set after first island")
                .fitness,
        ];

        for r#gen in 0..generations {
            // Evolve each island independently
            for island in &mut islands {
                self.evolve_island(island);
            }

            // Migration phase
            if r#gen > 0 && r#gen % self.config.migration_interval == 0 {
                self.migrate_with_gradient_refinement(&mut islands);
            }

            // Update global best
            for island in &islands {
                if let Some(best) = island.first() {
                    if best.fitness
                        > global_best
                            .as_ref()
                            .expect("global_best must be set after first island")
                            .fitness
                    {
                        global_best = Some(best.clone());
                    }
                }
            }

            phi_history.push(
                global_best
                    .as_ref()
                    .expect("global_best must be set after first island")
                    .fitness,
            );
        }

        let best = global_best.expect("global_best must be set after island evolution");
        SearchResult {
            best_phi: best.fitness,
            best_architecture: best.genome,
            phi_history,
            evaluations: self.evaluations,
            strategy: SearchStrategy::IslandGradient,
            elapsed_ms: None,
        }
    }

    /// Evolve a single island population
    fn evolve_island(&mut self, island: &mut Vec<Individual>) {
        if island.is_empty() {
            return;
        }

        let elite_count = 1.max(island.len() / 5);
        let mut new_island = Vec::with_capacity(island.len());

        // Preserve elites
        for i in 0..elite_count.min(island.len()) {
            new_island.push(island[i].clone());
        }

        // Generate offspring
        while new_island.len() < island.len() {
            // Simple tournament selection within island
            let idx1 = (self.next_u64() as usize) % island.len();
            let idx2 = (self.next_u64() as usize) % island.len();
            let parent = if island[idx1].fitness > island[idx2].fitness {
                &island[idx1]
            } else {
                &island[idx2]
            };

            let mut child_genome = parent.genome.clone();
            child_genome.mutate(self.config.mutation_rate, self.next_u64());

            let arch = DecodedArchitecture::from_genome(&child_genome);
            let fitness = arch.compute_phi();
            self.evaluations += 1;

            new_island.push(Individual {
                genome: child_genome,
                fitness,
            });
        }

        *island = new_island;
        island.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Migrate best individuals between islands with gradient refinement
    fn migrate_with_gradient_refinement(&mut self, islands: &mut [Vec<Individual>]) {
        let num_islands = islands.len();
        if num_islands < 2 {
            return;
        }

        let migrants_per_island =
            (islands[0].len() as f32 * self.config.migration_rate).ceil() as usize;
        let migrants_per_island = migrants_per_island.max(1);

        // Collect migrants from each island (best individuals)
        let migrants: Vec<Vec<Individual>> = islands
            .iter()
            .map(|island| island.iter().take(migrants_per_island).cloned().collect())
            .collect();

        // Ring migration: island i receives migrants from island (i-1) mod n
        for i in 0..num_islands {
            let source = (i + num_islands - 1) % num_islands;

            for migrant in &migrants[source] {
                // Gradient-refine migrant before insertion
                let mut refined_genome = migrant.genome.clone();
                let gradient = PhiGradient::compute(&refined_genome, self.config.gradient_epsilon);
                self.evaluations += 12;

                gradient.apply(&mut refined_genome, self.config.learning_rate);

                let arch = DecodedArchitecture::from_genome(&refined_genome);
                let fitness = arch.compute_phi();
                self.evaluations += 1;

                // Replace worst individual in target island
                if !islands[i].is_empty()
                    && fitness > islands[i].last().expect("island must not be empty").fitness
                {
                    islands[i].pop();
                    islands[i].push(Individual {
                        genome: refined_genome,
                        fitness,
                    });
                    islands[i].sort_by(|a, b| {
                        b.fitness
                            .partial_cmp(&a.fitness)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                }
            }
        }
    }

    /// Evolve one generation
    fn evolve_generation(&mut self) {
        let mut new_population = Vec::with_capacity(self.config.population_size);

        // Preserve elites
        for i in 0..self.config.elite_count.min(self.population.len()) {
            new_population.push(self.population[i].clone());
        }

        // Generate offspring
        while new_population.len() < self.config.population_size {
            // Clone genomes to avoid borrow conflicts with RNG methods
            let parent1_genome = self.tournament_select().genome.clone();
            let parent2_genome = self.tournament_select().genome.clone();
            let crossover_rate = self.config.crossover_rate;

            let mut child_genome = if self.next_f32() < crossover_rate {
                parent1_genome.crossover(&parent2_genome, self.next_u64())
            } else {
                parent1_genome
            };

            // Mutate
            child_genome.mutate(self.config.mutation_rate, self.next_u64());

            // Evaluate
            let arch = DecodedArchitecture::from_genome(&child_genome);
            let fitness = arch.compute_phi();
            self.evaluations += 1;

            let child = Individual {
                genome: child_genome,
                fitness,
            };

            // Update best
            if self.best.is_none()
                || fitness
                    > self
                        .best
                        .as_ref()
                        .expect("best individual must be set after initialization")
                        .fitness
            {
                self.best = Some(child.clone());
            }

            new_population.push(child);
        }

        // Replace population
        self.population = new_population;
        self.population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Tournament selection
    fn tournament_select(&mut self) -> &Individual {
        let tournament_size = 3;
        let mut best_idx = (self.next_u64() as usize) % self.population.len();

        for _ in 1..tournament_size {
            let idx = (self.next_u64() as usize) % self.population.len();
            if self.population[idx].fitness > self.population[best_idx].fitness {
                best_idx = idx;
            }
        }

        &self.population[best_idx]
    }

    /// Gradient refinement on elite individuals
    fn gradient_refine_elites(&mut self) {
        for i in 0..self.config.elite_count.min(self.population.len()) {
            let mut genome = self.population[i].genome.clone();

            for _ in 0..self.config.gradient_steps_per_generation {
                let gradient = PhiGradient::compute(&genome, self.config.gradient_epsilon);
                self.evaluations += 12;
                gradient.apply(&mut genome, self.config.learning_rate);
            }

            // Evaluate refined genome
            let arch = DecodedArchitecture::from_genome(&genome);
            let fitness = arch.compute_phi();
            self.evaluations += 1;

            if fitness > self.population[i].fitness {
                self.population[i].genome = genome;
                self.population[i].fitness = fitness;

                if fitness
                    > self
                        .best
                        .as_ref()
                        .expect("best individual must be set after initialization")
                        .fitness
                {
                    self.best = Some(self.population[i].clone());
                }
            }
        }

        // Re-sort
        self.population.sort_by(|a, b| {
            b.fitness
                .partial_cmp(&a.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    // PRNG helpers

    fn next_u64(&mut self) -> u64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        self.rng_state
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u64() as f32 / u64::MAX as f32
    }

    /// Get current best architecture
    pub fn best(&self) -> Option<&Individual> {
        self.best.as_ref()
    }

    /// Get current population (sorted by fitness)
    pub fn population(&self) -> &[Individual] {
        &self.population
    }

    /// Get search statistics
    pub fn stats(&self) -> SearchStats {
        SearchStats {
            generation: self.generation,
            evaluations: self.evaluations,
            best_phi: self.best.as_ref().map(|b| b.fitness).unwrap_or(0.0),
            population_size: self.population.len(),
            avg_phi: if self.population.is_empty() {
                0.0
            } else {
                self.population.iter().map(|i| i.fitness).sum::<f64>()
                    / self.population.len() as f64
            },
            phi_std: if self.population.len() < 2 {
                0.0
            } else {
                let avg = self.population.iter().map(|i| i.fitness).sum::<f64>()
                    / self.population.len() as f64;
                let variance = self
                    .population
                    .iter()
                    .map(|i| (i.fitness - avg).powi(2))
                    .sum::<f64>()
                    / (self.population.len() - 1) as f64;
                variance.sqrt()
            },
        }
    }

    /// Reset search state
    pub fn reset(&mut self) {
        self.population.clear();
        self.best = None;
        self.phi_history.clear();
        self.generation = 0;
        self.evaluations = 0;
        self.rng_state = self.config.seed;
    }
}

/// Search statistics
#[derive(Debug, Clone)]
pub struct SearchStats {
    pub generation: usize,
    pub evaluations: usize,
    pub best_phi: f64,
    pub population_size: usize,
    pub avg_phi: f64,
    pub phi_std: f64,
}

impl std::fmt::Display for SearchStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Gen {}: Best Phi = {:.4}, Avg = {:.4} +/- {:.4} ({} evals)",
            self.generation, self.best_phi, self.avg_phi, self.phi_std, self.evaluations
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a small, fast config for unit tests.
    fn small_config() -> SearchConfig {
        SearchConfig {
            population_size: 6,
            elite_count: 1,
            hdc_dim: 256,
            min_nodes: 4,
            max_nodes: 8,
            random_samples: 8,
            seed: 42,
            parallel: false,
            gradient_steps_per_generation: 2,
            num_islands: 2,
            migration_interval: 2,
            migration_rate: 0.2,
            patience: 5,
            ..SearchConfig::default()
        }
    }

    #[test]
    fn test_search_config_default_produces_valid_config() {
        let cfg = SearchConfig::default();

        assert!(cfg.population_size > 0);
        assert!(cfg.elite_count <= cfg.population_size);
        assert!(cfg.mutation_rate >= 0.0 && cfg.mutation_rate <= 1.0);
        assert!(cfg.crossover_rate >= 0.0 && cfg.crossover_rate <= 1.0);
        assert!(cfg.learning_rate > 0.0);
        assert!(cfg.gradient_epsilon > 0.0);
        assert!(cfg.min_nodes <= cfg.max_nodes);
        assert!(cfg.hdc_dim > 0);
        assert!(cfg.random_samples > 0);
        assert!(cfg.momentum >= 0.0 && cfg.momentum <= 1.0);
        assert!(cfg.num_islands >= 1);
        assert!(cfg.migration_interval >= 1);
        assert!(cfg.convergence_threshold > 0.0);
        assert!(cfg.patience > 0);
    }

    #[test]
    fn test_new_creates_engine_successfully() {
        let cfg = small_config();
        let searcher = PhiArchitectureSearch::new(cfg.clone());

        assert_eq!(searcher.generation, 0);
        assert_eq!(searcher.evaluations, 0);
        assert!(searcher.best.is_none());
        assert!(searcher.population().is_empty());
    }

    #[test]
    fn test_stats_zeroed_initially() {
        let searcher = PhiArchitectureSearch::new(small_config());
        let stats = searcher.stats();

        assert_eq!(stats.generation, 0);
        assert_eq!(stats.evaluations, 0);
        assert_eq!(stats.population_size, 0);
        assert!((stats.best_phi - 0.0).abs() < f64::EPSILON);
        assert!((stats.avg_phi - 0.0).abs() < f64::EPSILON);
        assert!((stats.phi_std - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_best_returns_none_before_search() {
        let searcher = PhiArchitectureSearch::new(small_config());
        assert!(searcher.best().is_none());
    }

    #[test]
    fn test_population_populated_after_evolutionary_search() {
        let cfg = small_config();
        let pop_size = cfg.population_size;
        let mut searcher = PhiArchitectureSearch::new(cfg);

        searcher.evolutionary_search(1);

        assert_eq!(searcher.population().len(), pop_size);
        // Population should be sorted descending by fitness
        let pop = searcher.population();
        for w in pop.windows(2) {
            assert!(
                w[0].fitness >= w[1].fitness || (w[0].fitness - w[1].fitness).abs() < 1e-12,
                "Population not sorted: {} < {}",
                w[0].fitness,
                w[1].fitness
            );
        }
    }

    #[test]
    fn test_random_search_returns_finite_best_phi() {
        let mut searcher = PhiArchitectureSearch::new(small_config());
        let result = searcher.random_search();

        assert!(result.best_phi.is_finite(), "best_phi must be finite");
        assert!(result.best_phi >= 0.0, "best_phi must be non-negative");
        assert_eq!(result.strategy, SearchStrategy::Random);
        assert!(
            result.evaluations >= 8,
            "should have evaluated at least random_samples"
        );
        assert!(!result.phi_history.is_empty());
    }

    #[test]
    fn test_evolutionary_search_returns_positive_evaluations() {
        let mut searcher = PhiArchitectureSearch::new(small_config());
        let result = searcher.evolutionary_search(2);

        assert!(result.evaluations > 0);
        assert!(result.best_phi.is_finite());
        assert!(result.best_phi >= 0.0);
        assert_eq!(result.strategy, SearchStrategy::Evolutionary);
        // phi_history: initial + 2 generations = at least 3 entries
        assert!(result.phi_history.len() >= 3);
    }

    #[test]
    fn test_gradient_search_returns_finite_phi() {
        let mut searcher = PhiArchitectureSearch::new(small_config());
        let result = searcher.gradient_search(3);

        assert!(result.best_phi.is_finite(), "best_phi must be finite");
        assert!(result.best_phi >= 0.0);
        assert_eq!(result.strategy, SearchStrategy::GradientGuided);
        // phi_history: initial + 3 steps = 4 entries
        assert_eq!(result.phi_history.len(), 4);
        assert!(result.evaluations > 0);
    }

    #[test]
    fn test_search_dispatcher_routes_all_strategies() {
        let strategies = [
            (SearchStrategy::Random, "Random"),
            (SearchStrategy::Evolutionary, "Evolutionary"),
            (SearchStrategy::GradientGuided, "GradientGuided"),
            (SearchStrategy::Hybrid, "Hybrid"),
            (SearchStrategy::GradientEvolutionary, "GradientEvolutionary"),
            (SearchStrategy::MomentumOptimization, "MomentumOptimization"),
            (SearchStrategy::IslandGradient, "IslandGradient"),
        ];

        for (strategy, label) in strategies {
            let mut searcher = PhiArchitectureSearch::new(small_config());
            let result = searcher.search(strategy, 2);

            assert!(
                result.best_phi.is_finite(),
                "{label}: best_phi must be finite"
            );
            assert!(
                result.best_phi >= 0.0,
                "{label}: best_phi must be non-negative"
            );
            assert_eq!(
                result.strategy, strategy,
                "{label}: returned strategy must match requested"
            );
            assert!(result.evaluations > 0, "{label}: evaluations must be > 0");
        }
    }

    #[test]
    fn test_reset_clears_state() {
        let cfg = small_config();
        let mut searcher = PhiArchitectureSearch::new(cfg);

        // Run a search to populate state
        searcher.evolutionary_search(2);
        assert!(searcher.best().is_some());
        assert!(searcher.evaluations > 0);
        assert!(!searcher.population().is_empty());

        // Reset
        searcher.reset();

        assert!(searcher.best().is_none());
        assert_eq!(searcher.evaluations, 0);
        assert_eq!(searcher.generation, 0);
        assert!(searcher.population().is_empty());

        let stats = searcher.stats();
        assert_eq!(stats.evaluations, 0);
        assert_eq!(stats.generation, 0);
        assert_eq!(stats.population_size, 0);
    }

    #[test]
    fn test_stats_after_search_reflect_state() {
        let cfg = small_config();
        let pop_size = cfg.population_size;
        let mut searcher = PhiArchitectureSearch::new(cfg);

        searcher.evolutionary_search(3);
        let stats = searcher.stats();

        assert_eq!(stats.generation, 3);
        assert!(stats.evaluations > 0);
        assert_eq!(stats.population_size, pop_size);
        assert!(stats.best_phi.is_finite());
        assert!(stats.avg_phi.is_finite());
        assert!(stats.phi_std.is_finite());
        assert!(stats.avg_phi >= 0.0);
        assert!(stats.phi_std >= 0.0);

        // Display impl should not panic
        let display = format!("{}", stats);
        assert!(display.contains("Gen 3"));
    }
}
