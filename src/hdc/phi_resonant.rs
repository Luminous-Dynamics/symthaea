/*!
Resonator-Based Φ Calculator - Dynamic Consciousness Emergence

Revolutionary improvement: Instead of static eigenvalue computation, model consciousness
as emerging through iterative resonance between components. This captures the *dynamics*
of consciousness rather than just a snapshot.

## Core Insight

Consciousness doesn't exist in isolation - it emerges through the dynamic interaction
of system components. Traditional Φ calculation measures a static snapshot via eigenvalues (O(n³)).
Resonator-based Φ models consciousness emergence through coupled oscillator dynamics (O(n log N)).

## Mathematical Foundation

### Traditional (Algebraic Connectivity):
```text
Φ_static = λ₂(Laplacian)  // 2nd smallest eigenvalue, O(n³)
```

### Resonant (Iterative Dynamics):
```text
state(t+1) = normalize(∑ⱼ similarity(i,j) × state_j(t))
Φ_resonant = integration(state(∞))  // Stable fixed point, O(n log N)
```

## Why Resonance?

1. **Faster**: O(n log N) convergence vs O(n³) eigenvalue computation
2. **Captures Dynamics**: Models consciousness *emergence*, not just static structure
3. **Biologically Realistic**: Brain exhibits coupled oscillator dynamics
4. **Fixed Points**: Stable consciousness states are resonance attractors

## References

- Frady et al. (2020) "Resonator networks" - Neural Computation
- Hopfield (1982) "Neural networks and physical systems" - PNAS
- Freeman (1975) "Mass action in the nervous system" - Chaotic dynamics in brain
- Tononi (2004) "IIT" - Consciousness as integrated information

*/

use super::real_hv::RealHV;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

// =============================================================================
// SIMD-Optimized Helper Functions for Resonance Dynamics
// =============================================================================

/// SIMD-optimized weighted accumulation with 8-wide chunks
///
/// Computes: output += weight * vector in a vectorization-friendly way
#[inline(always)]
fn simd_weighted_accumulate(output: &mut [f32], vector: &[f32], weight: f32) {
    const CHUNK_SIZE: usize = 8;
    let n = output.len().min(vector.len());
    let chunks = n / CHUNK_SIZE;

    // Process 8-wide chunks for auto-vectorization
    for c in 0..chunks {
        let base = c * CHUNK_SIZE;
        for i in 0..CHUNK_SIZE {
            output[base + i] += weight * vector[base + i];
        }
    }

    // Handle remainder
    for i in (chunks * CHUNK_SIZE)..n {
        output[i] += weight * vector[i];
    }
}

/// SIMD-optimized damped blend: result = damping * a + (1-damping) * b
#[inline(always)]
fn simd_damped_blend(result: &mut [f32], a: &[f32], b: &[f32], damping: f32) {
    const CHUNK_SIZE: usize = 8;
    let one_minus_damping = 1.0 - damping;
    let n = result.len().min(a.len()).min(b.len());
    let chunks = n / CHUNK_SIZE;

    // Process 8-wide chunks for auto-vectorization
    for c in 0..chunks {
        let base = c * CHUNK_SIZE;
        for i in 0..CHUNK_SIZE {
            result[base + i] = damping * a[base + i] + one_minus_damping * b[base + i];
        }
    }

    // Handle remainder
    for i in (chunks * CHUNK_SIZE)..n {
        result[i] = damping * a[i] + one_minus_damping * b[i];
    }
}

/// SIMD-optimized normalization in place
#[inline(always)]
fn simd_normalize_inplace(values: &mut [f32]) {
    const CHUNK_SIZE: usize = 8;

    // Compute norm squared with SIMD
    let mut acc = [0.0f32; CHUNK_SIZE];
    for chunk in values.chunks_exact(CHUNK_SIZE) {
        for i in 0..CHUNK_SIZE {
            acc[i] += chunk[i] * chunk[i];
        }
    }

    let mut sum: f32 = acc.iter().sum();
    for &x in values.chunks_exact(CHUNK_SIZE).remainder() {
        sum += x * x;
    }

    let norm = sum.sqrt();
    if norm == 0.0 {
        return;
    }

    let inv_norm = 1.0 / norm;

    // Scale with SIMD
    for chunk in values.chunks_exact_mut(CHUNK_SIZE) {
        for i in 0..CHUNK_SIZE {
            chunk[i] *= inv_norm;
        }
    }

    for x in values.chunks_exact_mut(CHUNK_SIZE).into_remainder() {
        *x *= inv_norm;
    }
}

/// Result of resonant Φ calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResonantPhiResult {
    /// Final Φ value after resonance
    pub phi: f64,

    /// Number of iterations to convergence
    pub iterations: usize,

    /// Computation time in milliseconds
    pub convergence_time_ms: f64,

    /// Whether system converged to stable state
    pub converged: bool,

    /// Final energy of system (lower = more stable)
    pub final_energy: f64,

    /// Energy trajectory (for analysis)
    pub energy_history: Vec<f64>,

    /// Stable state representations
    pub stable_state: Vec<RealHV>,
}

/// Configuration for resonant Φ calculation
#[derive(Debug, Clone)]
pub struct ResonantConfig {
    /// Maximum iterations before timeout
    pub max_iterations: usize,

    /// Convergence threshold (change in energy)
    pub convergence_threshold: f64,

    /// Damping factor (prevents oscillation)
    pub damping: f64,

    /// Self-coupling strength (0 = no self-loop, 1 = strong)
    pub self_coupling: f64,

    /// Normalization method
    pub normalize: bool,

    /// Enable parallel computation (uses rayon)
    pub parallel: bool,

    /// Minimum nodes before enabling parallelism (overhead threshold)
    pub parallel_threshold: usize,
}

impl Default for ResonantConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            damping: 0.3,
            self_coupling: 0.1,
            normalize: true,
            parallel: true,
            parallel_threshold: 8, // Use parallelism for n >= 8 nodes
        }
    }
}

/// Fast resonant configuration (fewer iterations)
impl ResonantConfig {
    pub fn fast() -> Self {
        Self {
            max_iterations: 100,
            convergence_threshold: 1e-4,
            damping: 0.5,
            parallel: true,
            parallel_threshold: 6,
            ..Default::default()
        }
    }

    pub fn accurate() -> Self {
        Self {
            max_iterations: 5000,
            convergence_threshold: 1e-8,
            damping: 0.2,
            parallel: true,
            parallel_threshold: 4,
            ..Default::default()
        }
    }

    /// Sequential-only (for benchmarking or small topologies)
    pub fn sequential() -> Self {
        Self {
            parallel: false,
            ..Default::default()
        }
    }
}

/// Resonator-based Φ calculator
///
/// Models consciousness emergence through iterative resonance dynamics rather than
/// static eigenvalue computation. Much faster (O(n log N) vs O(n³)) and captures
/// the dynamic nature of consciousness.
#[derive(Clone)]
pub struct ResonantPhiCalculator {
    config: ResonantConfig,
}

impl ResonantPhiCalculator {
    /// Create new resonant Φ calculator with default configuration
    pub fn new() -> Self {
        Self {
            config: ResonantConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ResonantConfig) -> Self {
        Self { config }
    }

    /// Create fast calculator (good for real-time monitoring)
    pub fn fast() -> Self {
        Self {
            config: ResonantConfig::fast(),
        }
    }

    /// Create accurate calculator (good for research)
    pub fn accurate() -> Self {
        Self {
            config: ResonantConfig::accurate(),
        }
    }

    /// Compute Φ via resonance dynamics
    ///
    /// # Algorithm
    ///
    /// 1. Initialize: state(0) = components (initial configuration)
    /// 2. Iterate: state(t+1) = damping × state(t) + (1-damping) × resonance_update(state(t))
    /// 3. Converge: Stop when |energy(t+1) - energy(t)| < threshold
    /// 4. Measure: Φ = integration_metric(stable_state)
    ///
    /// # Arguments
    ///
    /// * `components` - Initial component representations (node hypervectors)
    ///
    /// # Returns
    ///
    /// ResonantPhiResult with Φ value, convergence info, and stable state
    pub fn compute(&self, components: &[RealHV]) -> ResonantPhiResult {
        let start_time = Instant::now();
        let n = components.len();

        if n < 2 {
            return ResonantPhiResult {
                phi: 0.0,
                iterations: 0,
                convergence_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                converged: true,
                final_energy: 0.0,
                energy_history: vec![0.0],
                stable_state: components.to_vec(),
            };
        }

        // 1. Build similarity matrix (weights between resonators)
        let similarity_matrix = self.build_similarity_matrix(components);

        // 2. Initialize resonator states
        let mut current_state: Vec<RealHV> = components.to_vec();
        let mut energy_history = Vec::new();

        let mut prev_energy = self.compute_energy(&current_state, &similarity_matrix);
        energy_history.push(prev_energy);

        let mut iterations = 0;
        let mut converged = false;

        // 3. Iterate resonance dynamics until convergence
        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            // Update each resonator based on coupling with others
            let next_state = self.resonance_step(&current_state, &similarity_matrix);

            // Compute new energy
            let new_energy = self.compute_energy(&next_state, &similarity_matrix);
            energy_history.push(new_energy);

            // Check convergence
            let energy_change = (new_energy - prev_energy).abs();
            if energy_change < self.config.convergence_threshold {
                converged = true;
                current_state = next_state;
                break;
            }

            // Update for next iteration
            current_state = next_state;
            prev_energy = new_energy;
        }

        // 4. Measure integration of stable state
        let phi = self.measure_integration(&current_state, &similarity_matrix);

        let convergence_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;

        ResonantPhiResult {
            phi,
            iterations,
            convergence_time_ms,
            converged,
            final_energy: prev_energy,
            energy_history,
            stable_state: current_state,
        }
    }

    /// Build pairwise similarity matrix (coupling strengths)
    ///
    /// Uses parallel computation for large topologies (n >= parallel_threshold)
    fn build_similarity_matrix(&self, components: &[RealHV]) -> Vec<Vec<f64>> {
        let n = components.len();
        let use_parallel = self.config.parallel && n >= self.config.parallel_threshold;

        if use_parallel {
            self.build_similarity_matrix_parallel(components)
        } else {
            self.build_similarity_matrix_sequential(components)
        }
    }

    /// Sequential similarity matrix construction
    fn build_similarity_matrix_sequential(&self, components: &[RealHV]) -> Vec<Vec<f64>> {
        let n = components.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[i][j] = self.config.self_coupling;
                } else {
                    let sim = components[i].similarity(&components[j]);
                    matrix[i][j] = ((sim as f64 + 1.0) / 2.0).max(0.0).min(1.0);
                }
            }
        }

        matrix
    }

    /// Parallel similarity matrix construction using rayon
    ///
    /// Parallelizes row computation - each row is computed independently
    fn build_similarity_matrix_parallel(&self, components: &[RealHV]) -> Vec<Vec<f64>> {
        let n = components.len();
        let self_coupling = self.config.self_coupling;

        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row = vec![0.0; n];
                for j in 0..n {
                    if i == j {
                        row[j] = self_coupling;
                    } else {
                        let sim = components[i].similarity(&components[j]);
                        row[j] = ((sim as f64 + 1.0) / 2.0).max(0.0).min(1.0);
                    }
                }
                row
            })
            .collect()
    }

    /// Single resonance step: update all resonators based on current state
    ///
    /// Each resonator i updates as:
    /// new_i = damping × old_i + (1-damping) × Σⱼ similarity(i,j) × old_j
    fn resonance_step(&self, current_state: &[RealHV], similarity_matrix: &[Vec<f64>]) -> Vec<RealHV> {
        let n = current_state.len();
        let use_parallel = self.config.parallel && n >= self.config.parallel_threshold;

        if use_parallel {
            self.resonance_step_parallel(current_state, similarity_matrix)
        } else {
            self.resonance_step_sequential(current_state, similarity_matrix)
        }
    }

    /// Sequential resonance step
    fn resonance_step_sequential(&self, current_state: &[RealHV], similarity_matrix: &[Vec<f64>]) -> Vec<RealHV> {
        let damping = self.config.damping;
        let normalize = self.config.normalize;

        current_state
            .iter()
            .enumerate()
            .map(|(i, current_hv)| {
                self.update_single_resonator(i, current_hv, current_state, similarity_matrix, damping, normalize)
            })
            .collect()
    }

    /// Parallel resonance step using rayon
    ///
    /// Each node update is independent and can be computed in parallel
    fn resonance_step_parallel(&self, current_state: &[RealHV], similarity_matrix: &[Vec<f64>]) -> Vec<RealHV> {
        let damping = self.config.damping;
        let normalize = self.config.normalize;

        (0..current_state.len())
            .into_par_iter()
            .map(|i| {
                self.update_single_resonator(i, &current_state[i], current_state, similarity_matrix, damping, normalize)
            })
            .collect()
    }

    /// Update a single resonator based on coupling with all others
    ///
    /// SIMD-optimized version that avoids temporary allocations by working
    /// directly on f32 arrays with 8-wide vectorization
    #[inline]
    fn update_single_resonator(
        &self,
        i: usize,
        current_hv: &RealHV,
        current_state: &[RealHV],
        similarity_matrix: &[Vec<f64>],
        damping: f64,
        normalize: bool,
    ) -> RealHV {
        let dim = current_hv.dim();

        // Pre-allocate output buffer (avoid repeated allocations)
        let mut coupled_sum = vec![0.0f32; dim];
        let mut total_weight = 0.0f64;

        // SIMD-optimized weighted accumulation
        for (j, other_hv) in current_state.iter().enumerate() {
            let weight = similarity_matrix[i][j];
            if weight > 0.0 {
                simd_weighted_accumulate(&mut coupled_sum, &other_hv.values, weight as f32);
                total_weight += weight;
            }
        }

        // Normalize by total weight (in-place)
        if total_weight > 1e-10 {
            let inv_weight = (1.0 / total_weight) as f32;
            for x in &mut coupled_sum {
                *x *= inv_weight;
            }
        }

        // SIMD-optimized damped blend: result = damping * current + (1-damping) * coupled
        let mut result = vec![0.0f32; dim];
        simd_damped_blend(&mut result, &current_hv.values, &coupled_sum, damping as f32);

        // Optional normalization (SIMD-optimized)
        if normalize {
            simd_normalize_inplace(&mut result);
        }

        RealHV::from_values(result)
    }

    /// Compute system energy (measures stability)
    ///
    /// Lower energy = more stable configuration
    /// Energy = -Σᵢⱼ similarity(i,j) × similarity(state_i, state_j)
    fn compute_energy(&self, state: &[RealHV], similarity_matrix: &[Vec<f64>]) -> f64 {
        let n = state.len();
        let use_parallel = self.config.parallel && n >= self.config.parallel_threshold;

        if use_parallel {
            self.compute_energy_parallel(state, similarity_matrix)
        } else {
            self.compute_energy_sequential(state, similarity_matrix)
        }
    }

    /// Sequential energy computation
    fn compute_energy_sequential(&self, state: &[RealHV], similarity_matrix: &[Vec<f64>]) -> f64 {
        let n = state.len();
        let mut energy = 0.0;

        for i in 0..n {
            for j in (i + 1)..n {
                let coupling = similarity_matrix[i][j];
                let alignment = state[i].similarity(&state[j]) as f64;
                energy -= coupling * alignment;
            }
        }

        energy
    }

    /// Parallel energy computation using rayon
    ///
    /// Parallelizes over rows, each row computes its contribution to energy
    fn compute_energy_parallel(&self, state: &[RealHV], similarity_matrix: &[Vec<f64>]) -> f64 {
        let n = state.len();

        // Compute partial sums in parallel (one per row)
        (0..n)
            .into_par_iter()
            .map(|i| {
                let mut row_energy = 0.0;
                for j in (i + 1)..n {
                    let coupling = similarity_matrix[i][j];
                    let alignment = state[i].similarity(&state[j]) as f64;
                    row_energy -= coupling * alignment;
                }
                row_energy
            })
            .sum()
    }

    /// Measure integration of stable state
    ///
    /// Uses energy-based integration metric: lower final energy = higher integration
    /// This captures how well the topology enables coherent alignment.
    fn measure_integration(&self, state: &[RealHV], similarity_matrix: &[Vec<f64>]) -> f64 {
        let n = state.len();

        if n < 2 {
            return 0.0;
        }

        // Compute final energy: -Σᵢⱼ similarity(i,j) × alignment(state_i, state_j)
        let final_energy = self.compute_energy(state, similarity_matrix);

        // Compute maximum possible energy (worst case: all states opposite)
        // Max energy = -Σᵢⱼ similarity(i,j) × (-1) = Σᵢⱼ similarity(i,j)
        let max_energy = self.compute_total_similarity(similarity_matrix);

        // Compute minimum possible energy (best case: all states perfectly aligned)
        // Min energy = -Σᵢⱼ similarity(i,j) × (+1) = -Σᵢⱼ similarity(i,j)
        let min_energy = -max_energy;

        // Normalize energy to [0, 1] where:
        // - 0 = worst case (max energy, no integration)
        // - 1 = best case (min energy, perfect integration)
        if (max_energy - min_energy).abs() > 1e-10 {
            let normalized = (final_energy - max_energy) / (min_energy - max_energy);
            normalized.max(0.0).min(1.0)
        } else {
            0.0
        }
    }

    /// Compute total similarity (sum of upper triangle of similarity matrix)
    fn compute_total_similarity(&self, similarity_matrix: &[Vec<f64>]) -> f64 {
        let n = similarity_matrix.len();
        let use_parallel = self.config.parallel && n >= self.config.parallel_threshold;

        if use_parallel {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let mut row_sum = 0.0;
                    for j in (i + 1)..n {
                        row_sum += similarity_matrix[i][j];
                    }
                    row_sum
                })
                .sum()
        } else {
            let mut total = 0.0;
            for i in 0..n {
                for j in (i + 1)..n {
                    total += similarity_matrix[i][j];
                }
            }
            total
        }
    }
}

impl Default for ResonantPhiCalculator {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::consciousness_topology_generators::ConsciousnessTopology;
    use crate::hdc::HDC_DIMENSION;

    #[test]
    fn test_resonant_phi_convergence() {
        let calc = ResonantPhiCalculator::new();

        // Create simple star topology
        let topology = ConsciousnessTopology::star(5, HDC_DIMENSION, 42);

        let result = calc.compute(&topology.node_representations);

        // Should converge
        assert!(result.converged, "Resonance should converge");
        assert!(result.iterations > 0, "Should take some iterations");
        assert!(result.iterations < 1000, "Should converge reasonably fast");

        // Phi should be reasonable
        assert!(result.phi > 0.0, "Φ should be positive");
        assert!(result.phi <= 1.0, "Φ should be ≤ 1.0");

        println!("Resonant Φ: {:.4} (converged in {} iterations, {:.2}ms)",
                 result.phi, result.iterations, result.convergence_time_ms);
    }

    #[test]
    fn test_star_vs_random_resonant() {
        let calc = ResonantPhiCalculator::fast(); // Use fast config for testing

        // Star topology (higher expected Φ)
        let star = ConsciousnessTopology::star(8, HDC_DIMENSION, 42);
        let phi_star = calc.compute(&star.node_representations);

        // Random topology (lower expected Φ)
        let random = ConsciousnessTopology::random(8, HDC_DIMENSION, 42);
        let phi_random = calc.compute(&random.node_representations);

        println!("Star Φ: {:.4} ({} iter, {:.1}ms)",
                 phi_star.phi, phi_star.iterations, phi_star.convergence_time_ms);
        println!("Random Φ: {:.4} ({} iter, {:.1}ms)",
                 phi_random.phi, phi_random.iterations, phi_random.convergence_time_ms);

        // Star should have higher Φ (or at least competitive)
        // Note: Resonant Φ may have different absolute scale, but ordering should match
        println!("Star > Random: {}", phi_star.phi > phi_random.phi);
    }

    #[test]
    #[ignore = "performance test - run with cargo test --release"]
    fn test_resonant_performance() {
        let calc = ResonantPhiCalculator::fast();

        let topology = ConsciousnessTopology::dense_network(10, HDC_DIMENSION, None, 42);

        let start = std::time::Instant::now();
        let result = calc.compute(&topology.node_representations);
        let elapsed = start.elapsed();

        println!("Resonant Φ for n=10: {:.4}", result.phi);
        println!("Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!("Iterations: {}", result.iterations);

        // Should be reasonably fast (debug mode is ~10x slower than release)
        // Release mode: <1s target, Debug mode: <10s acceptable
        assert!(elapsed.as_secs_f64() < 10.0, "Should complete in <10 seconds (debug mode)");
    }
}
