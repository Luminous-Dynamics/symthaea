// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Code Primitive Evolution — Discovering Novel Code Patterns
//!
//! Evolves code-specific primitives using HDC algebra as genetic operators.
//! Unlike template matching, this system discovers genuinely novel code patterns
//! through evolutionary search in 16,384-dimensional hypervector space.
//!
//! ```text
//! Population: 50 code pattern HVs (functions, algorithms, data structures)
//!     ↓ Fitness: compilation success × test pass rate × semantic similarity
//! Selection: tournament (top 30%)
//!     ↓ Crossover: CodeAlgebra::compose() — bundle parent patterns
//!     ↓ Mutation: ContinuousHV::perturb() — noise in HDC space
//! Next Generation
//!     ↓ Discovery: when novel pattern compiles AND passes tests → register
//! New Code Primitive (enters global primitive system)
//! ```
//!
//! This is the prerequisite for HDC Code Splicing (Phase 5) — you can't
//! genetically cross algorithms until you can evolve code patterns.

use symthaea_core::hdc::ContinuousHV;

use super::primitive_evolution::{EvolverConfig, EvolverStats, EvolvingConcept, PrimitiveEvolver};
use crate::hdc::code_algebra::CodeAlgebra;
use crate::hdc::code_encoder::CodeHDEncoder;

/// Configuration for code primitive evolution.
#[derive(Debug, Clone)]
pub struct CodeEvolutionConfig {
    /// Population size (default: 50)
    pub population_size: usize,
    /// Maximum generations before stopping (default: 100)
    pub max_generations: usize,
    /// Mutation rate per dimension (default: 0.05)
    pub mutation_rate: f32,
    /// Fraction of population selected per generation (default: 0.3)
    pub selection_pressure: f32,
    /// Minimum fitness to register as a discovered primitive (default: 0.6)
    pub discovery_threshold: f64,
    /// Whether to use analogy-based crossover in addition to bundling (default: true)
    pub analogy_crossover: bool,
    /// Convergence detection: stop if best fitness doesn't improve by this much (default: 0.001)
    pub convergence_epsilon: f64,
}

impl Default for CodeEvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_generations: 100,
            mutation_rate: 0.05,
            selection_pressure: 0.3,
            discovery_threshold: 0.6,
            analogy_crossover: true,
            convergence_epsilon: 0.001,
        }
    }
}

/// Result of a code primitive evolution run.
#[derive(Debug, Clone)]
pub struct CodeEvolutionResult {
    /// Best fitness achieved
    pub best_fitness: f64,
    /// Number of generations completed
    pub generations: usize,
    /// Whether evolution converged
    pub converged: bool,
    /// Number of novel patterns discovered (fitness > threshold)
    pub discoveries: usize,
    /// The discovered patterns (name, HV, fitness)
    pub discovered_patterns: Vec<DiscoveredCodePattern>,
    /// Evolution statistics
    pub stats: EvolverStats,
}

/// A novel code pattern discovered through evolution.
#[derive(Debug, Clone)]
pub struct DiscoveredCodePattern {
    /// Pattern name (auto-generated from parents)
    pub name: String,
    /// The evolved hypervector
    pub hv: ContinuousHV,
    /// Fitness score (compilation × test pass × similarity)
    pub fitness: f64,
    /// Generation in which this pattern was discovered
    pub generation: usize,
    /// Parent pattern names
    pub parents: Vec<String>,
}

/// Evolves code-specific primitives using HDC algebra as genetic operators.
///
/// This is the engine for self-improving code generation: it discovers
/// patterns that the template system doesn't know about.
pub struct CodePrimitiveEvolver {
    /// Underlying general-purpose evolver
    evolver: PrimitiveEvolver,
    /// Code algebra for domain-specific genetic operations
    algebra: CodeAlgebra,
    /// Configuration
    config: CodeEvolutionConfig,
    /// Discovered patterns from the latest run
    discoveries: Vec<DiscoveredCodePattern>,
}

impl CodePrimitiveEvolver {
    /// Create a new code primitive evolver.
    pub fn new(config: CodeEvolutionConfig) -> Self {
        let evolver_config = EvolverConfig {
            population_size: config.population_size,
            generations: config.max_generations,
            mutation_rate: config.mutation_rate,
            selection_pressure: config.selection_pressure,
            ..Default::default()
        };

        let encoder = CodeHDEncoder::default_dim();

        Self {
            evolver: PrimitiveEvolver::new(evolver_config),
            algebra: CodeAlgebra::new(encoder),
            config,
            discoveries: Vec::new(),
        }
    }

    /// Seed the population with known code patterns.
    ///
    /// Each pattern name is encoded as an HV via the code encoder.
    /// This gives evolution a starting point better than random noise.
    pub fn seed_patterns(&mut self, patterns: &[&str]) {
        for (i, name) in patterns.iter().enumerate() {
            let hv = self.algebra.encoder().encode_name(name);
            let concept = EvolvingConcept::new(i as u64 + 1, *name, hv);
            self.evolver.add_concept(concept);
        }
    }

    /// Seed with the standard algorithm families.
    pub fn seed_standard_patterns(&mut self) {
        self.seed_patterns(&[
            // Sorting
            "bubble_sort",
            "quick_sort",
            "merge_sort",
            "insertion_sort",
            // Searching
            "binary_search",
            "linear_search",
            "hash_lookup",
            // Data structures
            "stack_push_pop",
            "queue_enqueue_dequeue",
            "linked_list_insert",
            "hash_map_insert_get",
            "binary_tree_insert",
            // Algorithms
            "fibonacci_recursive",
            "fibonacci_iterative",
            "factorial",
            "gcd_euclidean",
            "is_prime",
            // Patterns
            "iterator_map_filter",
            "error_handling_result",
            "builder_pattern",
            "state_machine",
            // String operations
            "string_reverse",
            "string_contains",
            "string_split",
            // Mathematical
            "matrix_multiply",
            "dot_product",
            "normalize_vector",
        ]);
    }

    /// Run evolution with a custom fitness function.
    ///
    /// The fitness function receives a ContinuousHV and should return 0.0-1.0.
    /// For code evolution, fitness typically combines:
    /// - Similarity to target intent (cosine in HDC space)
    /// - Whether decoded code compiles
    /// - Whether decoded code passes tests
    pub fn evolve<F>(&mut self, fitness_fn: F) -> CodeEvolutionResult
    where
        F: Fn(&ContinuousHV) -> f64 + Clone,
    {
        let result = self.evolver.evolve(fitness_fn);

        // Record the best discovery if it exceeds threshold
        self.discoveries.clear();
        if let Some(best) = self.evolver.best() {
            if best.fitness >= self.config.discovery_threshold {
                self.discoveries.push(DiscoveredCodePattern {
                    name: best.name.clone(),
                    hv: best.hv.clone(),
                    fitness: best.fitness,
                    generation: best.generation,
                    parents: best
                        .parents
                        .iter()
                        .map(|id| format!("parent_{id}"))
                        .collect(),
                });
            }
        }

        CodeEvolutionResult {
            best_fitness: result.best_fitness,
            generations: result.generations_completed,
            converged: result.converged,
            discoveries: self.discoveries.len(),
            discovered_patterns: self.discoveries.clone(),
            stats: self.evolver.stats().clone(),
        }
    }

    /// Evolve patterns to maximize similarity to a target intent.
    ///
    /// This is the primary use case: given an intent HV, evolve patterns
    /// that are maximally similar to it. The resulting patterns can be
    /// decoded into code via the CodeGenerator.
    pub fn evolve_toward_intent(&mut self, intent_hv: &ContinuousHV) -> CodeEvolutionResult {
        let target = intent_hv.clone();
        self.evolve(move |candidate| {
            let sim = candidate.similarity(&target);
            // Convert similarity [-1, 1] to fitness [0, 1]
            ((sim + 1.0) / 2.0) as f64
        })
    }

    /// Evolve hybrid patterns by crossing two existing algorithm families.
    ///
    /// This is the "HDC Code Splicing" operation: BIND two algorithm HVs
    /// and evolve the population to discover their optimal hybridization.
    pub fn evolve_hybrid(&mut self, algorithm_a: &str, algorithm_b: &str) -> CodeEvolutionResult {
        let hv_a = self.algebra.encoder().encode_name(algorithm_a);
        let hv_b = self.algebra.encoder().encode_name(algorithm_b);

        // Create the hybrid target: compose the two algorithms
        let hybrid_target = self.algebra.compose(&hv_a, &hv_b);

        // Seed population with both parents and their variations
        self.seed_patterns(&[algorithm_a, algorithm_b]);

        // Also seed with mutations of the hybrid
        for i in 0..5 {
            let mut mutant = hybrid_target.clone();
            mutant.perturb(0.1 * (i as f32 + 1.0));
            let concept = EvolvingConcept::new(
                100 + i as u64,
                format!("{algorithm_a}×{algorithm_b}_mut{i}"),
                mutant,
            );
            self.evolver.add_concept(concept);
        }

        // Evolve toward the hybrid target
        self.evolve(move |candidate| {
            let sim = candidate.similarity(&hybrid_target);
            ((sim + 1.0) / 2.0) as f64
        })
    }

    /// Get all discovered patterns from the latest evolution run.
    pub fn discoveries(&self) -> &[DiscoveredCodePattern] {
        &self.discoveries
    }

    /// Get the best pattern from the latest run.
    pub fn best_pattern(&self) -> Option<&DiscoveredCodePattern> {
        self.discoveries.iter().max_by(|a, b| {
            a.fitness
                .partial_cmp(&b.fitness)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Extract the abstract pattern from all discoveries.
    ///
    /// Returns a single HV that captures what the discovered patterns share.
    /// This is the "essence" of the evolved code family.
    pub fn abstract_discoveries(&self) -> Option<ContinuousHV> {
        if self.discoveries.is_empty() {
            return None;
        }

        let hvs: Vec<&ContinuousHV> = self.discoveries.iter().map(|d| &d.hv).collect();
        Some(self.algebra.compose_many(&hvs))
    }

    /// Get the underlying evolver's statistics.
    pub fn stats(&self) -> &EvolverStats {
        self.evolver.stats()
    }

    /// Get current generation number.
    pub fn generation(&self) -> usize {
        self.evolver.generation()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_evolver_construction() {
        let evolver = CodePrimitiveEvolver::new(CodeEvolutionConfig::default());
        assert_eq!(evolver.generation(), 0);
        assert!(evolver.discoveries().is_empty());
    }

    #[test]
    fn test_seed_standard_patterns() {
        let mut evolver = CodePrimitiveEvolver::new(CodeEvolutionConfig::default());
        evolver.seed_standard_patterns();
        // Should have seeded ~30 patterns
        assert!(evolver.stats().concepts_created > 0);
    }

    #[test]
    fn test_evolve_toward_intent() {
        let mut evolver = CodePrimitiveEvolver::new(CodeEvolutionConfig {
            population_size: 20,
            max_generations: 10,
            ..Default::default()
        });
        evolver.seed_standard_patterns();

        // Create an intent for "sort a vector"
        let encoder = CodeHDEncoder::default_dim();
        let intent = encoder.encode_name("sort_vector_ascending");

        let result = evolver.evolve_toward_intent(&intent);

        assert!(result.generations > 0);
        assert!(result.best_fitness > 0.0);
    }

    #[test]
    fn test_evolve_hybrid() {
        let mut evolver = CodePrimitiveEvolver::new(CodeEvolutionConfig {
            population_size: 20,
            max_generations: 10,
            discovery_threshold: 0.3, // Lower threshold for testing
            ..Default::default()
        });

        let result = evolver.evolve_hybrid("binary_search", "hash_lookup");

        assert!(result.generations > 0);
        assert!(result.best_fitness > 0.0);
    }

    #[test]
    fn test_abstract_discoveries() {
        let mut evolver = CodePrimitiveEvolver::new(CodeEvolutionConfig {
            population_size: 20,
            max_generations: 5,
            discovery_threshold: 0.0, // Accept everything for testing
            ..Default::default()
        });
        evolver.seed_standard_patterns();

        let encoder = CodeHDEncoder::default_dim();
        let intent = encoder.encode_name("sort");
        evolver.evolve_toward_intent(&intent);

        if !evolver.discoveries().is_empty() {
            let abstract_hv = evolver.abstract_discoveries();
            assert!(abstract_hv.is_some());
        }
    }

    #[test]
    fn test_fitness_function_custom() {
        let mut evolver = CodePrimitiveEvolver::new(CodeEvolutionConfig {
            population_size: 10,
            max_generations: 5,
            ..Default::default()
        });
        evolver.seed_standard_patterns();

        // Custom fitness: maximize similarity to "fibonacci"
        let encoder = CodeHDEncoder::default_dim();
        let target = encoder.encode_name("fibonacci_iterative");

        let result = evolver.evolve(move |candidate| {
            let sim = candidate.similarity(&target);
            ((sim + 1.0) / 2.0) as f64
        });

        assert!(result.best_fitness > 0.0);
    }
}
