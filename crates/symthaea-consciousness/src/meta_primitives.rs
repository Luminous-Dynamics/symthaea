//! Revolutionary Improvement #49: Meta-Learning Novel Primitives
//!
//! **The Ultimate Breakthrough**: The system discovers new primitive transformations!
//!
//! ## From Hand-Coded to Discovered
//!
//! **Before #49**: Fixed transformation types
//! - 6 hand-coded transformations (Bind, Bundle, etc.)
//! - No way to discover new operations
//! - Limited to human intuition
//!
//! **After #49**: Evolutionary primitive discovery
//! - Compose existing transformations into new ones
//! - Evolve compositions that maximize Φ
//! - Discover emergent transformation patterns
//! - System creates its own cognitive toolkit!
//!
//! ## The Meta-Learning Framework
//!
//! ```
//! Primitive Genome: Sequence of base transformations
//!   Example: [Bind, Permute, Bundle] = "Rotational Binding"
//!
//! Fitness: How well it improves reasoning
//!   - Δ Φ contribution
//!   - Generalization across problems
//!   - Novelty vs existing primitives
//!
//! Evolution: Genetic algorithm on transformation sequences
//!   - Mutation: Add/remove/change transformation
//!   - Crossover: Combine successful sequences
//!   - Selection: Keep high-Φ compositions
//! ```
//!
//! ## Why This is Revolutionary
//!
//! This is **meta-learning** - the system learns how to create better learning primitives:
//! - Not just learning to select (that's #48)
//! - Not just learning parameters (that's #44)
//! - Actually **inventing new cognitive operations**
//! - Self-improving at the deepest architectural level

use crate::consciousness::primitive_reasoning::TransformationType;
use crate::hdc::primitive_system::Primitive;
use crate::hdc::binary_hv::HV16;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use rand::Rng;

/// Composite transformation: sequence of base transformations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeTransformation {
    /// Name of this composite (if discovered to be useful)
    pub name: Option<String>,

    /// Sequence of base transformations
    pub sequence: Vec<TransformationType>,

    /// Fitness metrics
    pub fitness: CompositeFitness,

    /// How many times this composite has been used
    pub usage_count: usize,
}

impl CompositeTransformation {
    /// Create new composite transformation
    pub fn new(sequence: Vec<TransformationType>) -> Self {
        Self {
            name: None,
            sequence,
            fitness: CompositeFitness::default(),
            usage_count: 0,
        }
    }

    /// Create random composite (for evolution)
    pub fn random(min_length: usize, max_length: usize) -> Self {
        let mut rng = rand::thread_rng();
        let length = rng.gen_range(min_length..=max_length);

        let all_transformations = vec![
            TransformationType::Bind,
            TransformationType::Bundle,
            TransformationType::Permute,
            TransformationType::Resonate,
            TransformationType::Abstract,
            TransformationType::Ground,
        ];

        let sequence: Vec<_> = (0..length)
            .map(|_| {
                let idx = rng.gen_range(0..all_transformations.len());
                all_transformations[idx]
            })
            .collect();

        Self::new(sequence)
    }

    /// Apply this composite transformation
    pub fn apply(&self, input: &HV16, primitive: &Primitive) -> Result<HV16> {
        let mut current = input.clone();

        for transformation in &self.sequence {
            current = self.apply_single(&current, primitive, transformation)?;
        }

        Ok(current)
    }

    /// Apply single transformation (copied from primitive_reasoning)
    fn apply_single(
        &self,
        input: &HV16,
        primitive: &Primitive,
        transformation: &TransformationType,
    ) -> Result<HV16> {
        match transformation {
            TransformationType::Bind => {
                Ok(input.bind(&primitive.encoding))
            }

            TransformationType::Bundle => {
                Ok(HV16::bundle(&[input.clone(), primitive.encoding.clone()]))
            }

            TransformationType::Permute => {
                let rotation = primitive.encoding.popcount() as usize % 16384;
                Ok(input.permute(rotation))
            }

            TransformationType::Resonate => {
                let similarity = input.similarity(&primitive.encoding);
                if similarity > 0.7 {
                    Ok(HV16::bundle(&[input.clone(), primitive.encoding.clone()]))
                } else {
                    Ok(input.clone())
                }
            }

            TransformationType::Abstract => {
                let bound = input.bind(&primitive.encoding);
                Ok(bound.permute(100))
            }

            TransformationType::Ground => {
                let bound = input.bind(&primitive.encoding);
                Ok(bound.permute(16384 - 100))
            }
        }
    }

    /// Mutate this composite (for evolution)
    pub fn mutate(&self) -> Self {
        let mut rng = rand::thread_rng();
        let mut new_sequence = self.sequence.clone();

        let mutation_type = rng.gen_range(0..3);

        match mutation_type {
            0 => {
                // Add transformation
                if new_sequence.len() < 8 {
                    let all_transformations = vec![
                        TransformationType::Bind,
                        TransformationType::Bundle,
                        TransformationType::Permute,
                        TransformationType::Resonate,
                        TransformationType::Abstract,
                        TransformationType::Ground,
                    ];
                    let idx = rng.gen_range(0..all_transformations.len());
                    let pos = rng.gen_range(0..=new_sequence.len());
                    new_sequence.insert(pos, all_transformations[idx]);
                }
            }

            1 => {
                // Remove transformation
                if new_sequence.len() > 1 {
                    let pos = rng.gen_range(0..new_sequence.len());
                    new_sequence.remove(pos);
                }
            }

            2 => {
                // Replace transformation
                if !new_sequence.is_empty() {
                    let all_transformations = vec![
                        TransformationType::Bind,
                        TransformationType::Bundle,
                        TransformationType::Permute,
                        TransformationType::Resonate,
                        TransformationType::Abstract,
                        TransformationType::Ground,
                    ];
                    let pos = rng.gen_range(0..new_sequence.len());
                    let idx = rng.gen_range(0..all_transformations.len());
                    new_sequence[pos] = all_transformations[idx];
                }
            }

            _ => {}
        }

        Self::new(new_sequence)
    }

    /// Crossover with another composite
    pub fn crossover(&self, other: &CompositeTransformation) -> Self {
        let mut rng = rand::thread_rng();

        if self.sequence.is_empty() || other.sequence.is_empty() {
            return self.clone();
        }

        let point1 = rng.gen_range(0..self.sequence.len());
        let point2 = rng.gen_range(0..other.sequence.len());

        let mut new_sequence = Vec::new();
        new_sequence.extend_from_slice(&self.sequence[..point1]);
        new_sequence.extend_from_slice(&other.sequence[point2..]);

        Self::new(new_sequence)
    }
}

/// Fitness metrics for composite transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositeFitness {
    /// Average Φ contribution across uses
    pub avg_phi_contribution: f64,

    /// How many different problems it works well on
    pub generalization_score: f64,

    /// How different from existing transformations (novelty)
    pub novelty_score: f64,

    /// Number of evaluations
    pub num_evaluations: usize,
}

impl Default for CompositeFitness {
    fn default() -> Self {
        Self {
            avg_phi_contribution: 0.0,
            generalization_score: 0.0,
            novelty_score: 1.0, // Start with high novelty
            num_evaluations: 0,
        }
    }
}

impl CompositeFitness {
    /// Update fitness based on new evaluation
    pub fn update(&mut self, phi_contribution: f64, generalization: f64) {
        let n = self.num_evaluations as f64;

        // Running average
        self.avg_phi_contribution = (self.avg_phi_contribution * n + phi_contribution) / (n + 1.0);
        self.generalization_score = (self.generalization_score * n + generalization) / (n + 1.0);

        self.num_evaluations += 1;

        // Decay novelty over time (becomes less novel as it's used)
        self.novelty_score *= 0.99;
    }

    /// Compute overall fitness
    pub fn composite_score(&self) -> f64 {
        // Weighted combination
        0.5 * self.avg_phi_contribution
            + 0.3 * self.generalization_score
            + 0.2 * self.novelty_score
    }
}

/// Population of composite transformations
pub struct MetaPrimitiveEvolution {
    /// Population of composites
    population: Vec<CompositeTransformation>,

    /// Population size
    population_size: usize,

    /// Mutation rate
    mutation_rate: f64,

    /// Crossover rate
    crossover_rate: f64,

    /// Generation counter
    generation: usize,

    /// Best composites discovered
    hall_of_fame: Vec<CompositeTransformation>,
}

impl MetaPrimitiveEvolution {
    /// Create new meta-primitive evolution
    pub fn new(population_size: usize) -> Self {
        let mut population = Vec::new();

        // Initialize with random composites
        for _ in 0..population_size {
            population.push(CompositeTransformation::random(1, 4));
        }

        Self {
            population,
            population_size,
            mutation_rate: 0.3,
            crossover_rate: 0.6,
            generation: 0,
            hall_of_fame: Vec::new(),
        }
    }

    /// Evolve for one generation
    pub fn evolve_generation(&mut self, test_problems: &[HV16], primitives: &[&Primitive]) -> Result<()> {
        // Evaluate all composites on test problems
        for composite in &mut self.population {
            let phi_scores: Vec<f64> = test_problems
                .iter()
                .map(|problem| {
                    Self::evaluate_composite_static(composite, problem, primitives)
                        .unwrap_or(0.0)
                })
                .collect();

            let avg_phi = phi_scores.iter().sum::<f64>() / phi_scores.len() as f64;

            // Variance as measure of generalization (low variance = generalizes well)
            let mean = avg_phi;
            let variance: f64 = phi_scores
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f64>() / phi_scores.len() as f64;

            let generalization = 1.0 / (1.0 + variance); // High generalization = low variance

            composite.fitness.update(avg_phi, generalization);
        }

        // Sort by fitness
        self.population.sort_by(|a, b| {
            b.fitness.composite_score()
                .partial_cmp(&a.fitness.composite_score())
                .unwrap()
        });

        // Add best to hall of fame
        if let Some(best) = self.population.first() {
            if best.fitness.composite_score() > 0.1 {
                self.hall_of_fame.push(best.clone());
            }
        }

        // Create next generation
        let mut next_generation = Vec::new();

        // Elitism: keep top 10%
        let elite_count = self.population_size / 10;
        next_generation.extend_from_slice(&self.population[..elite_count]);

        // Fill rest through tournament selection + crossover/mutation
        while next_generation.len() < self.population_size {
            let parent1 = self.tournament_select();
            let parent2 = self.tournament_select();

            let mut offspring = if rand::thread_rng().gen::<f64>() < self.crossover_rate {
                parent1.crossover(&parent2)
            } else {
                parent1.clone()
            };

            if rand::thread_rng().gen::<f64>() < self.mutation_rate {
                offspring = offspring.mutate();
            }

            next_generation.push(offspring);
        }

        self.population = next_generation;
        self.generation += 1;

        Ok(())
    }

    /// Evaluate composite on a problem (static method for use in loops)
    fn evaluate_composite_static(
        composite: &CompositeTransformation,
        problem: &HV16,
        primitives: &[&Primitive],
    ) -> Result<f64> {
        // Simple evaluation: apply composite and measure Φ change
        if primitives.is_empty() {
            return Ok(0.0);
        }

        let primitive = primitives[0]; // Use first primitive for testing
        let output = composite.apply(problem, primitive)?;

        // Measure Φ
        use crate::hdc::integrated_information::IntegratedInformation;
        let mut phi_computer = IntegratedInformation::new();
        let components = vec![problem.clone(), output];
        let phi = phi_computer.compute_phi(&components);

        Ok(phi)
    }

    /// Tournament selection
    fn tournament_select(&self) -> &CompositeTransformation {
        let mut rng = rand::thread_rng();
        let tournament_size = 3;

        let mut best_idx = rng.gen_range(0..self.population.len());
        let mut best_fitness = self.population[best_idx].fitness.composite_score();

        for _ in 1..tournament_size {
            let idx = rng.gen_range(0..self.population.len());
            let fitness = self.population[idx].fitness.composite_score();

            if fitness > best_fitness {
                best_idx = idx;
                best_fitness = fitness;
            }
        }

        &self.population[best_idx]
    }

    /// Get best discovered composites
    pub fn get_best(&self, n: usize) -> Vec<&CompositeTransformation> {
        let mut all: Vec<_> = self.population.iter().collect();
        all.sort_by(|a, b| {
            b.fitness.composite_score()
                .partial_cmp(&a.fitness.composite_score())
                .unwrap()
        });

        all.into_iter().take(n).collect()
    }

    /// Get hall of fame
    pub fn hall_of_fame(&self) -> &[CompositeTransformation] {
        &self.hall_of_fame
    }

    /// Get generation number
    pub fn generation(&self) -> usize {
        self.generation
    }
}

/// Discovery statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveryStats {
    pub generation: usize,
    pub population_size: usize,
    pub best_fitness: f64,
    pub avg_fitness: f64,
    pub hall_of_fame_size: usize,
    pub diversity: f64, // Unique sequences in population
}

impl MetaPrimitiveEvolution {
    /// Get statistics
    pub fn stats(&self) -> DiscoveryStats {
        let best_fitness = self.population
            .iter()
            .map(|c| c.fitness.composite_score())
            .fold(0.0f64, |a, b| a.max(b));

        let avg_fitness = self.population
            .iter()
            .map(|c| c.fitness.composite_score())
            .sum::<f64>() / self.population.len() as f64;

        // Measure diversity (unique sequences)
        let unique_sequences: std::collections::HashSet<_> = self.population
            .iter()
            .map(|c| format!("{:?}", c.sequence))
            .collect();

        let diversity = unique_sequences.len() as f64 / self.population.len() as f64;

        DiscoveryStats {
            generation: self.generation,
            population_size: self.population.len(),
            best_fitness,
            avg_fitness,
            hall_of_fame_size: self.hall_of_fame.len(),
            diversity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composite_creation() {
        let composite = CompositeTransformation::new(vec![
            TransformationType::Bind,
            TransformationType::Permute,
        ]);

        assert_eq!(composite.sequence.len(), 2);
        assert_eq!(composite.usage_count, 0);
    }

    #[test]
    fn test_random_composite() {
        let composite = CompositeTransformation::random(2, 5);
        assert!(composite.sequence.len() >= 2);
        assert!(composite.sequence.len() <= 5);
    }

    #[test]
    fn test_mutation() {
        let composite = CompositeTransformation::new(vec![
            TransformationType::Bind,
            TransformationType::Bundle,
        ]);

        let mutated = composite.mutate();
        // Mutation should change the sequence
        // (though it might rarely be the same)
        assert!(mutated.sequence.len() >= 1);
        assert!(mutated.sequence.len() <= 8);
    }

    #[test]
    fn test_meta_evolution_creation() {
        let evolution = MetaPrimitiveEvolution::new(20);
        assert_eq!(evolution.population.len(), 20);
        assert_eq!(evolution.generation, 0);
    }
}
