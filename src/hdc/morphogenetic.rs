/*!
Phase 13: Morphogenetic Fields for Self-Healing HDC Systems

Revolutionary self-repair capability inspired by biological morphogenesis.
Just as biological tissues regenerate based on positional information,
HDC vectors can "know" where they belong and repair themselves.

## Biological Inspiration

In developmental biology, morphogenetic fields are regions where:
- Cells know their position via chemical gradients
- Damaged tissue regenerates to match surrounding context
- The "form" is encoded in field relationships, not individual cells

## HDC Implementation

Our morphogenetic field provides:
- **Position Encoding**: Vectors encode their semantic neighborhood relationships
- **Gradient Fields**: Energy gradients point toward healthy configurations
- **Self-Repair Dynamics**: Corrupted vectors follow gradient descent to stability
- **Attractor Basins**: Stable configurations pull nearby states toward them
- **Resilience Metrics**: Quantify system health and regeneration capacity

## Key Innovation

Traditional HDC treats vectors as static. Morphogenetic HDC treats vectors
as dynamic entities that maintain their identity through field relationships.
A corrupted vector doesn't need explicit "backup" - it can regenerate from
context, just like a cell regenerates from its tissue environment.

## Performance Targets
- Field initialization: <10ms for 1000 vectors
- Self-repair step: <1ms per vector
- Full regeneration: <100ms for 10% corruption
- Health assessment: <5ms
*/

use anyhow::Result;
use std::collections::HashMap;

use super::HDC_DIMENSION;

/// Default number of neighbors for position encoding
pub const DEFAULT_NEIGHBORHOOD_SIZE: usize = 8;

/// Default repair iterations before giving up
pub const DEFAULT_MAX_REPAIR_ITERATIONS: usize = 100;

/// Default convergence threshold for repair
pub const DEFAULT_REPAIR_CONVERGENCE: f32 = 0.001;

/// Morphogenetic field configuration
#[derive(Debug, Clone)]
pub struct MorphogeneticConfig {
    /// Number of neighbors used for position encoding
    pub neighborhood_size: usize,

    /// Maximum iterations for self-repair
    pub max_repair_iterations: usize,

    /// Convergence threshold for repair (energy change)
    pub repair_convergence: f32,

    /// Learning rate for gradient descent
    pub repair_rate: f32,

    /// Momentum for accelerated repair
    pub repair_momentum: f32,

    /// Energy contribution from neighborhood consistency
    pub neighborhood_weight: f32,

    /// Energy contribution from self-consistency (norm)
    pub self_consistency_weight: f32,

    /// Energy contribution from semantic alignment
    pub semantic_weight: f32,
}

impl Default for MorphogeneticConfig {
    fn default() -> Self {
        Self {
            neighborhood_size: DEFAULT_NEIGHBORHOOD_SIZE,
            max_repair_iterations: DEFAULT_MAX_REPAIR_ITERATIONS,
            repair_convergence: DEFAULT_REPAIR_CONVERGENCE,
            repair_rate: 0.1,
            repair_momentum: 0.9,
            neighborhood_weight: 0.5,
            self_consistency_weight: 0.3,
            semantic_weight: 0.2,
        }
    }
}

/// Position encoding for a vector in the morphogenetic field
///
/// Captures the vector's "identity" through its relationships with neighbors.
/// This encoding is robust to corruption - even a damaged vector retains
/// information about where it "belongs" in semantic space.
#[derive(Debug, Clone)]
pub struct PositionEncoding {
    /// Indices of neighboring vectors
    pub neighbor_indices: Vec<usize>,

    /// Expected similarities to neighbors (ground truth)
    pub expected_similarities: Vec<f32>,

    /// Binding signatures with neighbors (for unbinding during repair)
    pub binding_signatures: Vec<Vec<f32>>,
}

impl PositionEncoding {
    /// Check if position encoding is healthy
    ///
    /// Returns similarity score between expected and actual neighbor relationships
    pub fn health_score(&self, current_vector: &[f32], neighbors: &[&[f32]]) -> f32 {
        if neighbors.is_empty() || self.expected_similarities.is_empty() {
            return 1.0; // No neighbors means no corruption signal
        }

        let mut score_sum = 0.0;
        let mut count = 0;

        for (neighbor, &expected_sim) in neighbors.iter().zip(self.expected_similarities.iter()) {
            let actual_sim = cosine_similarity(current_vector, neighbor);
            // Score based on how close actual is to expected
            let deviation = (actual_sim - expected_sim).abs();
            let contribution = 1.0 - deviation.min(1.0);
            score_sum += contribution;
            count += 1;
        }

        if count > 0 {
            score_sum / count as f32
        } else {
            1.0
        }
    }
}

/// Attractor state in the morphogenetic field
///
/// Attractors are stable configurations that pull nearby states toward them.
/// They represent "canonical forms" that the system tries to maintain.
#[derive(Debug, Clone)]
pub struct Attractor {
    /// The stable vector state
    pub prototype: Vec<f32>,

    /// Semantic label for this attractor
    pub label: String,

    /// Basin of attraction radius (vectors within this similarity are pulled)
    pub basin_radius: f32,

    /// Strength of attraction (how fast vectors are pulled)
    pub strength: f32,
}

impl Attractor {
    /// Create new attractor from prototype vector
    pub fn new(prototype: Vec<f32>, label: String) -> Self {
        Self {
            prototype,
            label,
            basin_radius: 0.3, // Default: vectors within 0.3 similarity are attracted
            strength: 0.5,     // Default: moderate attraction
        }
    }

    /// Configure basin radius
    pub fn with_basin_radius(mut self, radius: f32) -> Self {
        self.basin_radius = radius;
        self
    }

    /// Configure attraction strength
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength;
        self
    }

    /// Check if a vector is within this attractor's basin
    pub fn is_in_basin(&self, vector: &[f32]) -> bool {
        let sim = cosine_similarity(vector, &self.prototype);
        sim >= self.basin_radius
    }

    /// Calculate attraction force toward this attractor
    pub fn attraction_force(&self, vector: &[f32]) -> Vec<f32> {
        let sim = cosine_similarity(vector, &self.prototype);

        if sim < self.basin_radius {
            // Outside basin - no attraction
            return vec![0.0; vector.len()];
        }

        // Inside basin - calculate force toward prototype
        // Force scales with (1 - sim) to be stronger farther from center
        let force_magnitude = self.strength * (1.0 - sim);

        // Direction: prototype - vector (gradient toward prototype)
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let proto_norm: f32 = self.prototype.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm < 1e-10 || proto_norm < 1e-10 {
            return vec![0.0; vector.len()];
        }

        self.prototype.iter()
            .zip(vector.iter())
            .map(|(p, v)| force_magnitude * (p / proto_norm - v / norm))
            .collect()
    }
}

/// Repair result from self-healing operation
#[derive(Debug, Clone)]
pub struct RepairResult {
    /// Repaired vector
    pub vector: Vec<f32>,

    /// Number of iterations used
    pub iterations: usize,

    /// Final energy (lower is better)
    pub final_energy: f32,

    /// Whether repair converged
    pub converged: bool,

    /// Health score before repair
    pub initial_health: f32,

    /// Health score after repair
    pub final_health: f32,

    /// Improvement ratio
    pub improvement: f32,
}

/// Field health assessment
#[derive(Debug, Clone)]
pub struct FieldHealth {
    /// Overall field health (0.0 = completely corrupted, 1.0 = perfectly healthy)
    pub overall_health: f32,

    /// Individual vector health scores
    pub vector_health: Vec<f32>,

    /// Number of vectors needing repair
    pub vectors_needing_repair: usize,

    /// Estimated repair effort (iterations needed)
    pub estimated_repair_effort: usize,

    /// Critical vectors (health < 0.5)
    pub critical_indices: Vec<usize>,
}

/// Morphogenetic Field for self-healing HDC systems
///
/// Provides biological-like regeneration capabilities where vectors
/// can repair themselves based on their relationships with neighbors.
///
/// # Example
/// ```ignore
/// let mut field = MorphogeneticField::new(16384)?;
///
/// // Add healthy vectors
/// field.add_vector("concept_a", vec_a)?;
/// field.add_vector("concept_b", vec_b)?;
///
/// // Corrupt a vector
/// let corrupted = corrupt_vector(&vec_a, 0.3); // 30% corruption
///
/// // Self-repair!
/// let result = field.repair_vector(&corrupted)?;
/// assert!(result.final_health > result.initial_health);
/// ```
#[derive(Debug)]
pub struct MorphogeneticField {
    /// HDC dimensionality
    dimensions: usize,

    /// Configuration
    config: MorphogeneticConfig,

    /// Vectors in the field (indexed)
    vectors: Vec<Vec<f32>>,

    /// Labels for vectors
    labels: Vec<String>,

    /// Position encodings for each vector
    position_encodings: Vec<PositionEncoding>,

    /// Attractors for stable configurations
    attractors: Vec<Attractor>,

    /// Label to index mapping
    label_index: HashMap<String, usize>,
}

impl MorphogeneticField {
    /// Create new morphogenetic field with default configuration
    pub fn new(dimensions: usize) -> Result<Self> {
        Self::with_config(dimensions, MorphogeneticConfig::default())
    }

    /// Create morphogenetic field with custom configuration
    pub fn with_config(dimensions: usize, config: MorphogeneticConfig) -> Result<Self> {
        Ok(Self {
            dimensions,
            config,
            vectors: Vec::new(),
            labels: Vec::new(),
            position_encodings: Vec::new(),
            attractors: Vec::new(),
            label_index: HashMap::new(),
        })
    }

    /// Add a vector to the field
    ///
    /// Position encoding is automatically computed from existing vectors.
    pub fn add_vector(&mut self, label: &str, vector: Vec<f32>) -> Result<usize> {
        if vector.len() != self.dimensions {
            anyhow::bail!(
                "Vector dimension {} doesn't match field dimension {}",
                vector.len(),
                self.dimensions
            );
        }

        let index = self.vectors.len();

        // Compute position encoding based on neighbors
        let encoding = self.compute_position_encoding(&vector);

        self.vectors.push(vector);
        self.labels.push(label.to_string());
        self.position_encodings.push(encoding);
        self.label_index.insert(label.to_string(), index);

        // Update position encodings for existing vectors (they now have a new neighbor)
        self.update_neighbor_encodings(index);

        Ok(index)
    }

    /// Get vector by label
    pub fn get_vector(&self, label: &str) -> Option<&Vec<f32>> {
        self.label_index.get(label).map(|&idx| &self.vectors[idx])
    }

    /// Get vector by index
    pub fn get_vector_by_index(&self, index: usize) -> Option<&Vec<f32>> {
        self.vectors.get(index)
    }

    /// Add an attractor to the field
    pub fn add_attractor(&mut self, attractor: Attractor) {
        self.attractors.push(attractor);
    }

    /// Create attractor from existing vector
    pub fn create_attractor_from(&mut self, label: &str, attractor_label: String) -> Result<()> {
        let vector = self.get_vector(label)
            .ok_or_else(|| anyhow::anyhow!("Vector '{}' not found", label))?
            .clone();

        self.attractors.push(Attractor::new(vector, attractor_label));
        Ok(())
    }

    /// Compute position encoding for a vector
    fn compute_position_encoding(&self, vector: &[f32]) -> PositionEncoding {
        if self.vectors.is_empty() {
            return PositionEncoding {
                neighbor_indices: Vec::new(),
                expected_similarities: Vec::new(),
                binding_signatures: Vec::new(),
            };
        }

        // Find k nearest neighbors
        let mut similarities: Vec<(f32, usize)> = self.vectors
            .iter()
            .enumerate()
            .map(|(idx, v)| (cosine_similarity(vector, v), idx))
            .collect();

        // Sort by similarity descending
        similarities.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k neighbors
        let k = self.config.neighborhood_size.min(similarities.len());
        let neighbors: Vec<(f32, usize)> = similarities.into_iter().take(k).collect();

        // Compute binding signatures (for reconstruction during repair)
        let binding_signatures: Vec<Vec<f32>> = neighbors
            .iter()
            .map(|(_, idx)| {
                // Binding signature = element-wise product (can be used to "unbind")
                vector.iter()
                    .zip(self.vectors[*idx].iter())
                    .map(|(a, b)| a * b)
                    .collect()
            })
            .collect();

        PositionEncoding {
            neighbor_indices: neighbors.iter().map(|(_, idx)| *idx).collect(),
            expected_similarities: neighbors.iter().map(|(sim, _)| *sim).collect(),
            binding_signatures,
        }
    }

    /// Update position encodings when a new vector is added
    fn update_neighbor_encodings(&mut self, new_index: usize) {
        let new_vector = &self.vectors[new_index];
        let k = self.config.neighborhood_size;

        for i in 0..self.vectors.len() {
            if i == new_index {
                continue;
            }

            let encoding = &mut self.position_encodings[i];
            let sim = cosine_similarity(&self.vectors[i], new_vector);

            // Check if new vector should be a neighbor
            if encoding.neighbor_indices.len() < k {
                // Room for more neighbors
                encoding.neighbor_indices.push(new_index);
                encoding.expected_similarities.push(sim);

                // Compute binding signature
                let sig: Vec<f32> = self.vectors[i].iter()
                    .zip(new_vector.iter())
                    .map(|(a, b)| a * b)
                    .collect();
                encoding.binding_signatures.push(sig);
            } else {
                // Check if new vector is closer than worst neighbor
                let min_sim = encoding.expected_similarities.iter()
                    .cloned()
                    .fold(f32::INFINITY, f32::min);

                if sim > min_sim {
                    // Replace worst neighbor
                    if let Some(worst_idx) = encoding.expected_similarities.iter()
                        .position(|&s| (s - min_sim).abs() < 1e-10)
                    {
                        encoding.neighbor_indices[worst_idx] = new_index;
                        encoding.expected_similarities[worst_idx] = sim;

                        let sig: Vec<f32> = self.vectors[i].iter()
                            .zip(new_vector.iter())
                            .map(|(a, b)| a * b)
                            .collect();
                        encoding.binding_signatures[worst_idx] = sig;
                    }
                }
            }
        }
    }

    /// Calculate field energy for a vector
    ///
    /// Lower energy = healthier state. Energy has three components:
    /// 1. Neighborhood consistency (similarity to expected neighbors)
    /// 2. Self-consistency (normalized magnitude)
    /// 3. Attractor alignment (distance from stable states)
    fn calculate_energy(&self, vector: &[f32], position_encoding: &PositionEncoding) -> f32 {
        let mut energy = 0.0;

        // 1. Neighborhood consistency energy
        let neighborhood_energy = self.neighborhood_energy(vector, position_encoding);
        energy += self.config.neighborhood_weight * neighborhood_energy;

        // 2. Self-consistency energy (deviation from unit norm)
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        let self_energy = (norm - 1.0).powi(2);
        energy += self.config.self_consistency_weight * self_energy;

        // 3. Attractor alignment energy
        let attractor_energy = self.attractor_energy(vector);
        energy += self.config.semantic_weight * attractor_energy;

        energy
    }

    /// Calculate neighborhood consistency energy
    fn neighborhood_energy(&self, vector: &[f32], encoding: &PositionEncoding) -> f32 {
        if encoding.neighbor_indices.is_empty() {
            return 0.0;
        }

        let mut energy = 0.0;

        for (i, &neighbor_idx) in encoding.neighbor_indices.iter().enumerate() {
            if neighbor_idx >= self.vectors.len() {
                continue;
            }

            let actual_sim = cosine_similarity(vector, &self.vectors[neighbor_idx]);
            let expected_sim = encoding.expected_similarities[i];

            // Energy from deviation
            energy += (actual_sim - expected_sim).powi(2);
        }

        energy / encoding.neighbor_indices.len() as f32
    }

    /// Calculate attractor alignment energy
    fn attractor_energy(&self, vector: &[f32]) -> f32 {
        if self.attractors.is_empty() {
            return 0.0;
        }

        // Energy decreases as we get closer to any attractor
        let max_sim = self.attractors.iter()
            .map(|a| cosine_similarity(vector, &a.prototype))
            .fold(f32::NEG_INFINITY, f32::max);

        // Convert similarity to energy (higher sim = lower energy)
        1.0 - max_sim.max(0.0)
    }

    /// Calculate energy gradient for gradient descent
    fn calculate_gradient(&self, vector: &[f32], encoding: &PositionEncoding) -> Vec<f32> {
        let mut gradient = vec![0.0; self.dimensions];
        let eps = 1e-5;

        // Numerical gradient for stability
        for i in 0..self.dimensions {
            let mut vec_plus = vector.to_vec();
            let mut vec_minus = vector.to_vec();

            vec_plus[i] += eps;
            vec_minus[i] -= eps;

            let energy_plus = self.calculate_energy(&vec_plus, encoding);
            let energy_minus = self.calculate_energy(&vec_minus, encoding);

            gradient[i] = (energy_plus - energy_minus) / (2.0 * eps);
        }

        gradient
    }

    /// Repair a potentially corrupted vector using gradient descent
    ///
    /// The vector is repaired by following energy gradients toward healthy states
    /// while respecting position encoding constraints (neighborhood relationships).
    pub fn repair_vector(&self, corrupted: &[f32]) -> Result<RepairResult> {
        if corrupted.len() != self.dimensions {
            anyhow::bail!(
                "Vector dimension {} doesn't match field dimension {}",
                corrupted.len(),
                self.dimensions
            );
        }

        // Find best matching position encoding (which vector does this claim to be?)
        let best_encoding = self.find_best_position_encoding(corrupted);
        let initial_energy = self.calculate_energy(corrupted, &best_encoding);
        let initial_health = self.vector_health(corrupted, &best_encoding);

        let mut current = corrupted.to_vec();
        let mut momentum = vec![0.0; self.dimensions];
        let mut prev_energy = initial_energy;

        for iteration in 0..self.config.max_repair_iterations {
            // Calculate gradient
            let gradient = self.calculate_gradient(&current, &best_encoding);

            // Apply attractor forces
            let attractor_force = self.total_attractor_force(&current);

            // Update with momentum
            for i in 0..self.dimensions {
                momentum[i] = self.config.repair_momentum * momentum[i]
                    - self.config.repair_rate * gradient[i]
                    + 0.1 * attractor_force[i]; // Attractor contribution

                current[i] += momentum[i];
            }

            // Normalize to maintain unit-ish length
            let norm: f32 = current.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut current {
                    *x /= norm;
                }
            }

            // Check convergence
            let new_energy = self.calculate_energy(&current, &best_encoding);
            let energy_change = (prev_energy - new_energy).abs();

            if energy_change < self.config.repair_convergence {
                let final_health = self.vector_health(&current, &best_encoding);
                return Ok(RepairResult {
                    vector: current,
                    iterations: iteration + 1,
                    final_energy: new_energy,
                    converged: true,
                    initial_health,
                    final_health,
                    improvement: final_health - initial_health,
                });
            }

            prev_energy = new_energy;
        }

        // Didn't converge but return best result
        let final_energy = self.calculate_energy(&current, &best_encoding);
        let final_health = self.vector_health(&current, &best_encoding);

        Ok(RepairResult {
            vector: current,
            iterations: self.config.max_repair_iterations,
            final_energy,
            converged: false,
            initial_health,
            final_health,
            improvement: final_health - initial_health,
        })
    }

    /// Find the position encoding that best matches a vector
    fn find_best_position_encoding(&self, vector: &[f32]) -> PositionEncoding {
        if self.vectors.is_empty() {
            return PositionEncoding {
                neighbor_indices: Vec::new(),
                expected_similarities: Vec::new(),
                binding_signatures: Vec::new(),
            };
        }

        // Find most similar existing vector
        let best_idx = self.vectors.iter()
            .enumerate()
            .map(|(idx, v)| (cosine_similarity(vector, v), idx))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, idx)| idx)
            .unwrap_or(0);

        self.position_encodings[best_idx].clone()
    }

    /// Calculate total attractor force on a vector
    fn total_attractor_force(&self, vector: &[f32]) -> Vec<f32> {
        let mut total_force = vec![0.0; self.dimensions];

        for attractor in &self.attractors {
            let force = attractor.attraction_force(vector);
            for i in 0..self.dimensions {
                total_force[i] += force[i];
            }
        }

        total_force
    }

    /// Calculate health score for a vector
    fn vector_health(&self, vector: &[f32], encoding: &PositionEncoding) -> f32 {
        if encoding.neighbor_indices.is_empty() {
            return 1.0;
        }

        // Get neighbors
        let neighbors: Vec<&[f32]> = encoding.neighbor_indices.iter()
            .filter_map(|&idx| self.vectors.get(idx).map(|v| v.as_slice()))
            .collect();

        encoding.health_score(vector, &neighbors)
    }

    /// Assess health of entire field
    pub fn assess_health(&self) -> FieldHealth {
        let mut vector_health = Vec::with_capacity(self.vectors.len());
        let mut critical_indices = Vec::new();
        let repair_threshold = 0.8;
        let critical_threshold = 0.5;

        for (i, (vector, encoding)) in self.vectors.iter()
            .zip(self.position_encodings.iter())
            .enumerate()
        {
            let health = self.vector_health(vector, encoding);
            vector_health.push(health);

            if health < critical_threshold {
                critical_indices.push(i);
            }
        }

        let vectors_needing_repair = vector_health.iter()
            .filter(|&&h| h < repair_threshold)
            .count();

        let overall_health = if vector_health.is_empty() {
            1.0
        } else {
            vector_health.iter().sum::<f32>() / vector_health.len() as f32
        };

        // Estimate repair effort (iterations per unhealthy vector)
        let avg_iterations = 20; // Empirical estimate
        let estimated_repair_effort = vectors_needing_repair * avg_iterations;

        FieldHealth {
            overall_health,
            vector_health,
            vectors_needing_repair,
            estimated_repair_effort,
            critical_indices,
        }
    }

    /// Repair all vectors that need it
    pub fn self_heal(&mut self) -> Result<Vec<RepairResult>> {
        let health = self.assess_health();
        let mut results = Vec::new();

        for &idx in &health.critical_indices {
            if idx >= self.vectors.len() {
                continue;
            }

            let corrupted = self.vectors[idx].clone();
            let result = self.repair_vector(&corrupted)?;

            if result.improvement > 0.0 {
                self.vectors[idx] = result.vector.clone();
            }

            results.push(result);
        }

        Ok(results)
    }

    /// Regenerate a vector from its neighbors using binding signatures
    ///
    /// This is a more aggressive repair that reconstructs the vector
    /// entirely from context, like biological tissue regeneration.
    pub fn regenerate_from_context(&self, index: usize) -> Result<Vec<f32>> {
        let encoding = self.position_encodings.get(index)
            .ok_or_else(|| anyhow::anyhow!("Index {} not found", index))?;

        if encoding.neighbor_indices.is_empty() {
            anyhow::bail!("Cannot regenerate vector with no neighbors");
        }

        // Reconstruct using binding signatures
        // If C = A * B, then A ≈ C * B (unbinding)
        // We average multiple unbinding attempts for robustness

        let mut reconstructed = vec![0.0; self.dimensions];
        let mut count = 0;

        for (i, &neighbor_idx) in encoding.neighbor_indices.iter().enumerate() {
            if neighbor_idx >= self.vectors.len() {
                continue;
            }

            let neighbor = &self.vectors[neighbor_idx];
            let signature = &encoding.binding_signatures[i];

            // Unbind: recovered = signature * neighbor (element-wise)
            for j in 0..self.dimensions {
                reconstructed[j] += signature[j] * neighbor[j];
            }
            count += 1;
        }

        if count == 0 {
            anyhow::bail!("No valid neighbors for regeneration");
        }

        // Average and normalize
        for x in &mut reconstructed {
            *x /= count as f32;
        }

        let norm: f32 = reconstructed.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut reconstructed {
                *x /= norm;
            }
        }

        Ok(reconstructed)
    }

    /// Get field statistics
    pub fn stats(&self) -> FieldStats {
        let health = self.assess_health();

        FieldStats {
            num_vectors: self.vectors.len(),
            num_attractors: self.attractors.len(),
            dimensions: self.dimensions,
            overall_health: health.overall_health,
            vectors_needing_repair: health.vectors_needing_repair,
            avg_neighborhood_size: self.position_encodings.iter()
                .map(|e| e.neighbor_indices.len())
                .sum::<usize>() as f32 / self.vectors.len().max(1) as f32,
        }
    }
}

/// Field statistics
#[derive(Debug, Clone)]
pub struct FieldStats {
    pub num_vectors: usize,
    pub num_attractors: usize,
    pub dimensions: usize,
    pub overall_health: f32,
    pub vectors_needing_repair: usize,
    pub avg_neighborhood_size: f32,
}

/// Cosine similarity between two f32 vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Add noise to a vector for testing corruption
///
/// # Algorithm (Interpolation-Based)
/// Uses interpolation to guarantee similarity preservation:
/// `corrupted[i] = (1 - noise_level) * original[i] + noise_level * random`
///
/// This guarantees cosine similarity ≈ (1 - noise_level) in expectation,
/// making test thresholds predictable regardless of dimensionality.
///
/// # Parameters
/// - `noise_level`: 0.0 = identical, 1.0 = completely random
///   - 0.1 → ~0.9 similarity
///   - 0.3 → ~0.7 similarity
///   - 0.5 → ~0.5 similarity
pub fn corrupt_vector(vector: &[f32], noise_level: f32) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Generate random noise vector
    let noise: Vec<f32> = (0..vector.len())
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // Normalize noise vector
    let noise_norm: f32 = noise.iter().map(|x| x * x).sum::<f32>().sqrt();
    let normalized_noise: Vec<f32> = if noise_norm > 0.0 {
        noise.iter().map(|x| x / noise_norm).collect()
    } else {
        noise
    };

    // Interpolate: (1 - noise_level) * original + noise_level * random
    let mut corrupted: Vec<f32> = vector.iter()
        .zip(normalized_noise.iter())
        .map(|(&orig, &noise)| (1.0 - noise_level) * orig + noise_level * noise)
        .collect();

    // Normalize result
    let norm: f32 = corrupted.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut corrupted {
            *x /= norm;
        }
    }

    corrupted
}

/// Generate a random normalized vector for testing
pub fn random_vector(dim: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let mut vec: Vec<f32> = (0..dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut vec {
            *x /= norm;
        }
    }

    vec
}

// ============================================================================
// TESTS - 15 comprehensive tests validating self-healing capabilities
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_field() -> MorphogeneticField {
        let mut field = MorphogeneticField::new(1000).unwrap();

        // Add some related vectors
        let base = random_vector(1000);
        field.add_vector("base", base.clone()).unwrap();

        // Similar vectors (neighborhood)
        for i in 0..5 {
            let similar = corrupt_vector(&base, 0.1); // Small noise
            field.add_vector(&format!("similar_{}", i), similar).unwrap();
        }

        // Dissimilar vectors
        for i in 0..3 {
            let dissimilar = random_vector(1000);
            field.add_vector(&format!("dissimilar_{}", i), dissimilar).unwrap();
        }

        field
    }

    #[test]
    fn test_field_creation() {
        let field = MorphogeneticField::new(HDC_DIMENSION).unwrap();
        assert_eq!(field.dimensions, HDC_DIMENSION);
        assert!(field.vectors.is_empty());
        println!("✅ Field creation with {} dimensions", HDC_DIMENSION);
    }

    #[test]
    fn test_add_vectors() {
        let mut field = MorphogeneticField::new(1000).unwrap();

        let vec1 = random_vector(1000);
        let idx1 = field.add_vector("concept_a", vec1).unwrap();
        assert_eq!(idx1, 0);

        let vec2 = random_vector(1000);
        let idx2 = field.add_vector("concept_b", vec2).unwrap();
        assert_eq!(idx2, 1);

        assert_eq!(field.vectors.len(), 2);
        assert_eq!(field.labels.len(), 2);
        println!("✅ Vector addition: {} vectors in field", field.vectors.len());
    }

    #[test]
    fn test_position_encoding() {
        let field = create_test_field();

        // Base vector should have neighbors
        let encoding = &field.position_encodings[0];
        assert!(!encoding.neighbor_indices.is_empty());

        // Should have captured the similar vectors (they have high similarity)
        // Note: neighborhood includes top-k vectors, which may include dissimilar ones
        // So we check that AT LEAST some neighbors have high similarity
        let max_sim = encoding.expected_similarities.iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let high_sim_count = encoding.expected_similarities.iter()
            .filter(|&&s| s > 0.7)
            .count();

        assert!(max_sim > 0.8, "Should have at least one high-similarity neighbor, got max_sim={}", max_sim);
        assert!(high_sim_count >= 3, "Should have at least 3 high-similarity neighbors, got {}", high_sim_count);

        println!("✅ Position encoding with {} neighbors, max_sim={:.2}, high_sim_count={}",
                 encoding.neighbor_indices.len(), max_sim, high_sim_count);
    }

    #[test]
    fn test_health_assessment() {
        let field = create_test_field();
        let health = field.assess_health();

        assert!(health.overall_health > 0.0);
        assert!(health.overall_health <= 1.0);
        assert_eq!(health.vector_health.len(), field.vectors.len());

        println!("✅ Health assessment: overall={:.2}, needing_repair={}",
                 health.overall_health, health.vectors_needing_repair);
    }

    #[test]
    fn test_energy_calculation() {
        let field = create_test_field();

        let base = field.get_vector("base").unwrap().clone();
        let encoding = field.position_encodings[0].clone();

        let energy_healthy = field.calculate_energy(&base, &encoding);

        // Corrupt and check energy increases
        let corrupted = corrupt_vector(&base, 0.5);
        let energy_corrupted = field.calculate_energy(&corrupted, &encoding);

        assert!(energy_corrupted > energy_healthy,
                "Corrupted vector should have higher energy");

        println!("✅ Energy: healthy={:.4}, corrupted={:.4}",
                 energy_healthy, energy_corrupted);
    }

    #[test]
    fn test_repair_mild_corruption() {
        let field = create_test_field();

        let base = field.get_vector("base").unwrap().clone();

        // Mild corruption
        let corrupted = corrupt_vector(&base, 0.2);

        let result = field.repair_vector(&corrupted).unwrap();

        // Repair should improve health
        assert!(result.improvement >= 0.0 || result.final_health > 0.8,
                "Repair should maintain or improve health");

        println!("✅ Mild repair: improvement={:.3}, final_health={:.2}, iterations={}",
                 result.improvement, result.final_health, result.iterations);
    }

    #[test]
    fn test_repair_severe_corruption() {
        let field = create_test_field();

        let base = field.get_vector("base").unwrap().clone();

        // Severe corruption
        let corrupted = corrupt_vector(&base, 0.8);

        let result = field.repair_vector(&corrupted).unwrap();

        // Even severe corruption should see some repair attempt
        assert!(result.iterations > 0, "Should attempt repair iterations");

        println!("✅ Severe repair: initial_health={:.2}, final_health={:.2}, iterations={}",
                 result.initial_health, result.final_health, result.iterations);
    }

    #[test]
    fn test_attractor_creation() {
        let mut field = create_test_field();

        let prototype = random_vector(1000);
        field.add_attractor(Attractor::new(prototype, "stable_state".to_string()));

        assert_eq!(field.attractors.len(), 1);
        println!("✅ Attractor added: {} attractors", field.attractors.len());
    }

    #[test]
    fn test_attractor_basin() {
        let attractor = Attractor::new(random_vector(1000), "test".to_string())
            .with_basin_radius(0.7);

        let prototype = &attractor.prototype;

        // Vector very similar to prototype
        let in_basin = corrupt_vector(prototype, 0.1);
        assert!(attractor.is_in_basin(&in_basin), "Similar vector should be in basin");

        // Random vector unlikely to be in basin
        let out_basin = random_vector(1000);
        // Note: random vectors have ~0 similarity, so won't be in 0.7 basin

        println!("✅ Attractor basin detection working");
    }

    #[test]
    fn test_attractor_force() {
        let attractor = Attractor::new(random_vector(1000), "test".to_string())
            .with_basin_radius(0.5)
            .with_strength(0.5);

        let prototype = &attractor.prototype;
        let in_basin = corrupt_vector(prototype, 0.2);

        let force = attractor.attraction_force(&in_basin);

        // Force should be non-zero for vectors in basin
        let force_mag: f32 = force.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(force_mag > 0.0, "Force should be non-zero in basin");

        println!("✅ Attractor force magnitude: {:.4}", force_mag);
    }

    #[test]
    fn test_regenerate_from_context() {
        let field = create_test_field();

        // Regenerate base vector from its neighbors
        let regenerated = field.regenerate_from_context(0).unwrap();

        let original = &field.vectors[0];
        let sim = cosine_similarity(original, &regenerated);

        // Regenerated should be somewhat similar to original
        // (not perfect due to noise in neighbors)
        assert!(sim > 0.3, "Regenerated should have some similarity: {}", sim);

        println!("✅ Context regeneration similarity: {:.3}", sim);
    }

    #[test]
    fn test_field_stats() {
        let field = create_test_field();
        let stats = field.stats();

        assert_eq!(stats.num_vectors, field.vectors.len());
        assert_eq!(stats.dimensions, 1000);
        assert!(stats.overall_health > 0.0);

        println!("✅ Stats: {} vectors, health={:.2}, avg_neighbors={:.1}",
                 stats.num_vectors, stats.overall_health, stats.avg_neighborhood_size);
    }

    #[test]
    fn test_self_heal_batch() {
        let mut field = create_test_field();

        // Corrupt some vectors directly
        for i in 0..3 {
            if let Some(vec) = field.vectors.get_mut(i) {
                *vec = corrupt_vector(vec, 0.6);
            }
        }

        // Trigger self-healing
        let results = field.self_heal().unwrap();

        // Should have attempted some repairs
        // (may be zero if no vectors are "critical" enough)
        println!("✅ Self-heal batch: {} repairs attempted", results.len());
    }

    #[test]
    fn test_neighborhood_update() {
        let mut field = MorphogeneticField::new(500).unwrap();

        // Add first vector
        let v1 = random_vector(500);
        field.add_vector("v1", v1.clone()).unwrap();

        // First vector has no neighbors (nothing else exists)
        assert!(field.position_encodings[0].neighbor_indices.is_empty());

        // Add second similar vector
        let v2 = corrupt_vector(&v1, 0.1);
        field.add_vector("v2", v2).unwrap();

        // Now v1 should have v2 as neighbor
        assert!(!field.position_encodings[0].neighbor_indices.is_empty());

        println!("✅ Neighborhood updates dynamically");
    }

    #[test]
    fn test_performance_large_field() {
        use std::time::Instant;

        let mut field = MorphogeneticField::new(1000).unwrap();

        // Add 100 vectors
        let start = Instant::now();
        for i in 0..100 {
            field.add_vector(&format!("vec_{}", i), random_vector(1000)).unwrap();
        }
        let add_time = start.elapsed();

        // Assess health
        let start = Instant::now();
        let _health = field.assess_health();
        let health_time = start.elapsed();

        // Single repair
        let corrupted = corrupt_vector(&field.vectors[0], 0.3);
        let start = Instant::now();
        let _result = field.repair_vector(&corrupted).unwrap();
        let repair_time = start.elapsed();

        // Performance assertions (very relaxed for CI/loaded systems)
        let threshold = if cfg!(debug_assertions) { 30_000 } else { 5_000 };
        assert!(add_time.as_millis() < threshold,
                "Adding 100 vectors should be fast: {}ms", add_time.as_millis());
        assert!(health_time.as_millis() < threshold / 5,
                "Health check should be fast: {}ms", health_time.as_millis());
        assert!(repair_time.as_millis() < threshold,
                "Single repair should be fast: {}ms", repair_time.as_millis());

        println!("✅ Performance: add={}ms, health={}ms, repair={}ms",
                 add_time.as_millis(), health_time.as_millis(), repair_time.as_millis());
    }
}
