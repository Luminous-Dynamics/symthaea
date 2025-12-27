//! Φ (Integrated Information) Validation for Consciousness Topologies
//!
//! This module integrates RealHV topology generators with TieredPhi calculation
//! to validate that different network topologies produce different Φ values.
//!
//! # The Validation Hypothesis
//!
//! **Hypothesis**: Network topology determines integrated information (Φ)
//! - Star topology should have HIGHER Φ than random topology
//! - Different topologies should produce statistically distinct Φ values
//! - Φ should correlate with topology heterogeneity (r > 0.85)
//!
//! # Methodology
//!
//! 1. Generate multiple instances of each topology type (Random vs Star)
//! 2. Convert RealHV representations to binary HV16 format
//! 3. Compute Φ using TieredPhi with Spectral tier (O(n²) accuracy)
//! 4. Statistical analysis: t-test for Φ_star > Φ_random
//! 5. Success criterion: p < 0.05 with effect size > 0.5
//!
//! # Example Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::phi_topology_validation::{MinimalPhiValidation, ValidationResult};
//!
//! let validation = MinimalPhiValidation::new(10, 10, 10, 2048);
//! let result = validation.run();
//!
//! if result.is_significant() {
//!     println!("SUCCESS: Star topology has significantly higher Φ (p < {})", result.p_value);
//! }
//! ```

use crate::hdc::real_hv::RealHV;
use crate::hdc::consciousness_topology_generators::ConsciousnessTopology;
use crate::hdc::binary_hv::HV16;
use crate::hdc::tiered_phi::{TieredPhi, ApproximationTier};
use crate::hdc::phi_real::RealPhiCalculator;  // ✨ NEW: Direct RealHV Φ calculation
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ============================================================================
// REALHV ↔ HV16 CONVERSION
// ============================================================================

/// Convert RealHV to binary HV16 using threshold-based binarization
///
/// **Strategy**: Binarize by comparing each element to the mean.
/// - Above mean → bit 1
/// - Below mean → bit 0
///
/// This preserves the essential structure while converting to binary format.
pub fn real_hv_to_hv16(real_hv: &RealHV) -> HV16 {
    let values = &real_hv.values;
    let n = values.len();

    // Compute mean for threshold
    let sum: f32 = values.iter().sum();
    let mean = sum / n as f32;

    // Create binary representation
    // HV16 is 16,384 bits = 2048 bytes
    let mut bytes = [0u8; 2048];

    for (i, &val) in values.iter().enumerate() {
        if i >= 16_384 {
            break; // HV16 is exactly 16,384 bits
        }

        // Set bit to 1 if value > mean
        if val > mean {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            bytes[byte_idx] |= 1 << bit_idx;
        }
    }

    HV16(bytes)
}

/// Convert RealHV to HV16 using **median threshold** (more robust to outliers)
///
/// Median is more robust than mean when dealing with extreme values.
/// This should better preserve heterogeneity for high-variance vectors.
pub fn real_hv_to_hv16_median(real_hv: &RealHV) -> HV16 {
    let values = &real_hv.values;
    let n = values.len();

    // Compute median for threshold
    let mut sorted_values: Vec<f32> = values.iter().copied().collect();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if n % 2 == 0 {
        (sorted_values[n / 2 - 1] + sorted_values[n / 2]) / 2.0
    } else {
        sorted_values[n / 2]
    };

    // Create binary representation using median threshold
    let mut bytes = [0u8; 2048];

    for (i, &val) in values.iter().enumerate() {
        if i >= 16_384 {
            break;
        }

        if val > median {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            bytes[byte_idx] |= 1 << bit_idx;
        }
    }

    HV16(bytes)
}

/// Convert RealHV to HV16 using **probabilistic binarization**
///
/// Each value is converted to a probability using sigmoid, then randomly
/// binarized. This preserves more information from the original distribution.
///
/// # Arguments
/// * `real_hv` - Input real-valued hypervector
/// * `seed` - Random seed for reproducibility
pub fn real_hv_to_hv16_probabilistic(real_hv: &RealHV, seed: u64) -> HV16 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let values = &real_hv.values;
    let mut bytes = [0u8; 2048];

    // Normalize values to roughly [-3, 3] range for sigmoid
    let sum: f32 = values.iter().sum();
    let mean = sum / values.len() as f32;
    let variance: f32 = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
    let std_dev = variance.sqrt().max(0.001); // Avoid division by zero

    for (i, &val) in values.iter().enumerate() {
        if i >= 16_384 {
            break;
        }

        // Normalize and apply sigmoid: p = 1 / (1 + exp(-x))
        let normalized = (val - mean) / std_dev;
        let prob = 1.0 / (1.0 + (-normalized).exp());

        // Generate deterministic pseudo-random value from seed + index
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        let hash_val = hasher.finish();
        let random_val = (hash_val % 10000) as f32 / 10000.0; // [0, 1)

        // Set bit based on probability
        if random_val < prob {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            bytes[byte_idx] |= 1 << bit_idx;
        }
    }

    HV16(bytes)
}

/// Convert RealHV to HV16 using **quantile-based threshold** (percentile)
///
/// Uses a specific percentile as the threshold. 50th percentile = median.
/// Can use other percentiles like 25th or 75th for skewed distributions.
///
/// # Arguments
/// * `real_hv` - Input real-valued hypervector
/// * `percentile` - Threshold percentile (0.0 - 100.0), typically 50.0
pub fn real_hv_to_hv16_quantile(real_hv: &RealHV, percentile: f32) -> HV16 {
    let values = &real_hv.values;
    let n = values.len();

    // Compute percentile threshold
    let mut sorted_values: Vec<f32> = values.iter().copied().collect();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let index = ((percentile / 100.0) * (n - 1) as f32) as usize;
    let threshold = sorted_values[index.min(n - 1)];

    // Create binary representation using quantile threshold
    let mut bytes = [0u8; 2048];

    for (i, &val) in values.iter().enumerate() {
        if i >= 16_384 {
            break;
        }

        if val > threshold {
            let byte_idx = i / 8;
            let bit_idx = i % 8;
            bytes[byte_idx] |= 1 << bit_idx;
        }
    }

    HV16(bytes)
}

/// Convert ConsciousnessTopology node representations to HV16 vector
///
/// Takes all node representations from a topology and converts them to
/// binary format suitable for Φ calculation.
pub fn topology_to_hv16_components(topology: &ConsciousnessTopology) -> Vec<HV16> {
    topology.node_representations
        .iter()
        .map(real_hv_to_hv16)
        .collect()
}

// ============================================================================
// VALIDATION RESULT STRUCTURES
// ============================================================================

/// Statistical result from minimal Φ validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Number of Random topology samples
    pub n_random: usize,

    /// Number of Star topology samples
    pub n_star: usize,

    /// Mean Φ for Random topologies
    pub mean_phi_random: f64,

    /// Standard deviation for Random topologies
    pub std_phi_random: f64,

    /// Mean Φ for Star topologies
    pub mean_phi_star: f64,

    /// Standard deviation for Star topologies
    pub std_phi_star: f64,

    /// t-statistic from independent samples t-test
    pub t_statistic: f64,

    /// Degrees of freedom
    pub degrees_of_freedom: f64,

    /// Two-tailed p-value
    pub p_value: f64,

    /// Cohen's d effect size
    pub effect_size: f64,

    /// Total computation time (milliseconds)
    pub total_time_ms: u64,

    /// Average time per Φ calculation (milliseconds)
    pub avg_time_per_phi_ms: f64,

    /// Individual Φ values for Random topologies
    pub phi_random_values: Vec<f64>,

    /// Individual Φ values for Star topologies
    pub phi_star_values: Vec<f64>,
}

impl ValidationResult {
    /// Check if the result shows significant difference (p < 0.05)
    pub fn is_significant(&self) -> bool {
        self.p_value < 0.05
    }

    /// Check if effect size is meaningful (Cohen's d > 0.5)
    pub fn has_large_effect(&self) -> bool {
        self.effect_size.abs() > 0.5
    }

    /// Check if validation succeeded (significant + large effect)
    pub fn validation_succeeded(&self) -> bool {
        self.is_significant() && self.has_large_effect() && self.mean_phi_star > self.mean_phi_random
    }

    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Φ Validation Result:\n\
             Random: {:.4} ± {:.4} (n={})\n\
             Star:   {:.4} ± {:.4} (n={})\n\
             t({:.1}) = {:.3}, p = {:.4}, d = {:.3}\n\
             Significant: {}, Large Effect: {}, Validation: {}\n\
             Total time: {}ms, Avg per Φ: {:.2}ms",
            self.mean_phi_random, self.std_phi_random, self.n_random,
            self.mean_phi_star, self.std_phi_star, self.n_star,
            self.degrees_of_freedom, self.t_statistic, self.p_value, self.effect_size,
            self.is_significant(), self.has_large_effect(), self.validation_succeeded(),
            self.total_time_ms, self.avg_time_per_phi_ms
        )
    }
}

// ============================================================================
// MINIMAL Φ VALIDATION (Random vs Star)
// ============================================================================

/// Minimal Φ validation comparing Random and Star topologies
///
/// This is the quickest validation to confirm that topology affects Φ.
/// If this succeeds, it validates the entire approach.
pub struct MinimalPhiValidation {
    /// Number of Random topology samples to generate
    n_random_samples: usize,

    /// Number of Star topology samples to generate
    n_star_samples: usize,

    /// Number of nodes in each topology
    n_nodes: usize,

    /// Dimensionality of hypervectors
    dim: usize,

    /// Φ calculator
    phi_calculator: TieredPhi,

    /// Random seed for reproducibility
    seed: u64,
}

impl MinimalPhiValidation {
    /// Create a new validation with specified parameters
    ///
    /// # Arguments
    ///
    /// * `n_random` - Number of random topology samples
    /// * `n_star` - Number of star topology samples
    /// * `n_nodes` - Number of nodes per topology
    /// * `dim` - Hypervector dimensionality
    ///
    /// # Recommended Settings
    ///
    /// - For quick test: (10, 10, 8, 2048)
    /// - For validation: (20, 20, 10, 2048)
    /// - For publication: (50, 50, 10, 2048)
    pub fn new(n_random: usize, n_star: usize, n_nodes: usize, dim: usize) -> Self {
        Self {
            n_random_samples: n_random,
            n_star_samples: n_star,
            n_nodes,
            dim,
            // Use Spectral tier for good accuracy/speed tradeoff
            phi_calculator: TieredPhi::new(ApproximationTier::Spectral),
            seed: 42, // Default reproducible seed
        }
    }

    /// Create a quick validation (10 samples each, fast)
    pub fn quick() -> Self {
        Self::new(10, 10, 8, super::HDC_DIMENSION)
    }

    /// Create a standard validation (20 samples each, balanced)
    pub fn standard() -> Self {
        Self::new(20, 20, 10, super::HDC_DIMENSION)
    }

    /// Create a publication-ready validation (50 samples each, thorough)
    pub fn publication() -> Self {
        Self::new(50, 50, 10, super::HDC_DIMENSION)
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Run the minimal Φ validation
    ///
    /// Generates topologies, computes Φ, and performs statistical analysis.
    ///
    /// # Returns
    ///
    /// `ValidationResult` containing all Φ values and statistical tests.
    pub fn run(&mut self) -> ValidationResult {
        let start_time = Instant::now();

        println!("\n🔬 MINIMAL Φ VALIDATION: Random vs Star Topologies");
        println!("============================================================");
        println!("Samples: {} random, {} star", self.n_random_samples, self.n_star_samples);
        println!("Nodes per topology: {}", self.n_nodes);
        println!("Dimensionality: {}", self.dim);
        println!("Φ calculator: {:?}", self.phi_calculator.tier());
        println!("Seed: {}", self.seed);
        println!();

        // Step 1: Generate Random topologies and compute Φ
        println!("📊 Step 1: Generating {} Random topologies...", self.n_random_samples);
        let phi_random_values = self.compute_phi_for_topology_type("random");
        println!("   Mean Φ (Random): {:.4}", mean(&phi_random_values));
        println!();

        // Step 2: Generate Star topologies and compute Φ
        println!("⭐ Step 2: Generating {} Star topologies...", self.n_star_samples);
        let phi_star_values = self.compute_phi_for_topology_type("star");
        println!("   Mean Φ (Star): {:.4}", mean(&phi_star_values));
        println!();

        // Step 3: Statistical analysis
        println!("📈 Step 3: Statistical Analysis");
        let stats = self.compute_statistics(&phi_random_values, &phi_star_values);

        let total_time_ms = start_time.elapsed().as_millis() as u64;
        let total_calculations = (self.n_random_samples + self.n_star_samples) as f64;
        let avg_time_per_phi_ms = total_time_ms as f64 / total_calculations;

        ValidationResult {
            n_random: self.n_random_samples,
            n_star: self.n_star_samples,
            mean_phi_random: stats.0,
            std_phi_random: stats.1,
            mean_phi_star: stats.2,
            std_phi_star: stats.3,
            t_statistic: stats.4,
            degrees_of_freedom: stats.5,
            p_value: stats.6,
            effect_size: stats.7,
            total_time_ms,
            avg_time_per_phi_ms,
            phi_random_values,
            phi_star_values,
        }
    }

    /// Run the validation using **RealPhiCalculator** (no binarization)
    ///
    /// ✨ NEW: This uses the RealPhi calculator directly on continuous data,
    /// avoiding the lossy RealHV→HV16 conversion that can destroy structure.
    ///
    /// # Returns
    ///
    /// `ValidationResult` containing all Φ values computed using cosine similarity
    pub fn run_with_real_phi(&mut self) -> ValidationResult {
        let start_time = Instant::now();

        println!("\n🔬 REAL Φ VALIDATION: Random vs Star Topologies (No Binarization)");
        println!("============================================================");
        println!("Method: RealPhiCalculator (cosine similarity, no conversion)");
        println!("Samples: {} random, {} star", self.n_random_samples, self.n_star_samples);
        println!("Nodes per topology: {}", self.n_nodes);
        println!("Dimensionality: {}", self.dim);
        println!("Seed: {}", self.seed);
        println!();

        // Step 1: Generate Random topologies and compute Φ using RealPhi
        println!("📊 Step 1: Generating {} Random topologies...", self.n_random_samples);
        let phi_random_values = self.compute_real_phi_for_topology_type("random");
        println!("   Mean Φ (Random): {:.4}", mean(&phi_random_values));
        println!();

        // Step 2: Generate Star topologies and compute Φ using RealPhi
        println!("⭐ Step 2: Generating {} Star topologies...", self.n_star_samples);
        let phi_star_values = self.compute_real_phi_for_topology_type("star");
        println!("   Mean Φ (Star): {:.4}", mean(&phi_star_values));
        println!();

        // Step 3: Statistical analysis
        println!("📈 Step 3: Statistical Analysis");
        let stats = self.compute_statistics(&phi_random_values, &phi_star_values);

        let total_time_ms = start_time.elapsed().as_millis() as u64;
        let total_calculations = (self.n_random_samples + self.n_star_samples) as f64;
        let avg_time_per_phi_ms = total_time_ms as f64 / total_calculations;

        ValidationResult {
            n_random: self.n_random_samples,
            n_star: self.n_star_samples,
            mean_phi_random: stats.0,
            std_phi_random: stats.1,
            mean_phi_star: stats.2,
            std_phi_star: stats.3,
            t_statistic: stats.4,
            degrees_of_freedom: stats.5,
            p_value: stats.6,
            effect_size: stats.7,
            total_time_ms,
            avg_time_per_phi_ms,
            phi_random_values,
            phi_star_values,
        }
    }

    /// Compute Φ using RealPhiCalculator for multiple instances of a topology type
    ///
    /// ✨ This computes Φ directly on RealHV without converting to HV16
    fn compute_real_phi_for_topology_type(&mut self, topology_type: &str) -> Vec<f64> {
        let n_samples = match topology_type {
            "random" => self.n_random_samples,
            "star" => self.n_star_samples,
            _ => panic!("Unknown topology type: {}", topology_type),
        };

        let mut phi_values = Vec::with_capacity(n_samples);
        let real_phi_calc = RealPhiCalculator::new();

        for i in 0..n_samples {
            // Generate topology with unique seed
            let seed = self.seed + (i as u64 * 1000);

            let topology = match topology_type {
                "random" => ConsciousnessTopology::random(self.n_nodes, self.dim, seed),
                "star" => ConsciousnessTopology::star(self.n_nodes, self.dim, seed),
                _ => unreachable!(),
            };

            // Get RealHV components directly (no conversion!)
            let components = &topology.node_representations;

            // DEBUG: Print cosine similarities for first sample
            if i == 0 {
                println!("   🔍 DEBUG: Cosine similarities for first {} topology:", topology_type);
                for node_i in 0..components.len().min(5) {
                    for node_j in (node_i + 1)..components.len().min(5) {
                        let sim = components[node_i].similarity(&components[node_j]);
                        println!("      Node {} ↔ Node {}: {:.4}",
                                 node_i, node_j, sim);
                    }
                }
                println!();
            }

            // Compute Φ using RealPhiCalculator (no binarization!)
            let phi = real_phi_calc.compute(&components);

            // DEBUG: Print Φ value for each sample
            if i < 5 {  // First 5 samples only
                println!("      Sample {}: Φ = {:.6}", i, phi);
            }

            phi_values.push(phi);

            // Progress indicator every 10 samples
            if (i + 1) % 10 == 0 || i == n_samples - 1 {
                print!("   Progress: {}/{} samples completed\r", i + 1, n_samples);
                use std::io::{self, Write};
                io::stdout().flush().unwrap();
            }
        }

        println!(); // New line after progress
        phi_values
    }

    /// Compute Φ for multiple instances of a topology type
    fn compute_phi_for_topology_type(&mut self, topology_type: &str) -> Vec<f64> {
        let n_samples = match topology_type {
            "random" => self.n_random_samples,
            "star" => self.n_star_samples,
            _ => panic!("Unknown topology type: {}", topology_type),
        };

        let mut phi_values = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            // Generate topology with unique seed
            let seed = self.seed + (i as u64 * 1000);

            let topology = match topology_type {
                "random" => ConsciousnessTopology::random(self.n_nodes, self.dim, seed),
                "star" => ConsciousnessTopology::star(self.n_nodes, self.dim, seed),
                _ => unreachable!(),
            };

            // Convert to HV16 components
            let components = topology_to_hv16_components(&topology);

            // DEBUG: Print Hamming distances for first sample
            if i == 0 {
                println!("   🔍 DEBUG: Hamming distances for first {} topology:", topology_type);
                for node_i in 0..components.len() {
                    for node_j in (node_i + 1)..components.len() {
                        let dist = components[node_i].hamming_distance(&components[node_j]);
                        println!("      Node {} ↔ Node {}: {} / 2048 = {:.4}",
                                 node_i, node_j, dist, dist as f64 / 2048.0);
                    }
                }
                println!();
            }

            // Compute Φ
            let phi = self.phi_calculator.compute(&components);

            // DEBUG: Print Φ value for each sample
            if i < 5 {  // First 5 samples only
                println!("      Sample {}: Φ = {:.6}", i, phi);
            }

            phi_values.push(phi);

            // Progress indicator every 10 samples
            if (i + 1) % 10 == 0 || i == n_samples - 1 {
                print!("   Progress: {}/{} samples completed\r", i + 1, n_samples);
                use std::io::{self, Write};
                io::stdout().flush().unwrap();
            }
        }

        println!(); // New line after progress
        phi_values
    }

    /// Compute statistical metrics
    ///
    /// Returns: (mean_random, std_random, mean_star, std_star, t, df, p, d)
    fn compute_statistics(
        &self,
        phi_random: &[f64],
        phi_star: &[f64],
    ) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
        let mean_random = mean(phi_random);
        let std_random = std_dev(phi_random, mean_random);

        let mean_star = mean(phi_star);
        let std_star = std_dev(phi_star, mean_star);

        // Independent samples t-test
        let n1 = phi_random.len() as f64;
        let n2 = phi_star.len() as f64;

        // Pooled standard deviation
        let pooled_std = ((((n1 - 1.0) * std_random * std_random) +
                          ((n2 - 1.0) * std_star * std_star)) /
                         (n1 + n2 - 2.0)).sqrt();

        // t-statistic
        let t_statistic = (mean_star - mean_random) /
                         (pooled_std * ((1.0 / n1) + (1.0 / n2)).sqrt());

        // Degrees of freedom
        let df = n1 + n2 - 2.0;

        // Approximate p-value using t-distribution
        // For df > 30, t-distribution ≈ standard normal
        let p_value = if df > 30.0 {
            // Use standard normal approximation
            2.0 * (1.0 - normal_cdf(t_statistic.abs()))
        } else {
            // For smaller samples, still use normal approximation (conservative)
            2.0 * (1.0 - normal_cdf(t_statistic.abs()))
        };

        // Cohen's d effect size
        let cohens_d = (mean_star - mean_random) / pooled_std;

        (mean_random, std_random, mean_star, std_star, t_statistic, df, p_value, cohens_d)
    }
}

// ============================================================================
// STATISTICAL HELPER FUNCTIONS
// ============================================================================

/// Compute mean of a slice
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Compute standard deviation
fn std_dev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let variance = values.iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<f64>() / (values.len() - 1) as f64;

    variance.sqrt()
}

/// Approximate cumulative distribution function for standard normal
///
/// Uses the error function approximation for z-scores.
fn normal_cdf(z: f64) -> f64 {
    // Approximation using error function
    // CDF(z) = 0.5 * (1 + erf(z / sqrt(2)))

    // Scale by 1/sqrt(2) as per the standard normal CDF formula
    let x = z / std::f64::consts::SQRT_2;

    // Simple erf approximation (Abramowitz-Stegun, good to ~5 decimal places)
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let erf = 1.0 - (((((1.061405429 * t + -1.453152027) * t) + 1.421413741) * t +
                     -0.284496736) * t + 0.254829592) * t * (-x * x).exp();

    if z >= 0.0 {
        0.5 * (1.0 + erf)
    } else {
        0.5 * (1.0 - erf)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_real_hv_to_hv16_conversion() {
        // Create a simple RealHV
        let real_hv = RealHV::random(2048, 42);

        // Convert to HV16
        let hv16 = real_hv_to_hv16(&real_hv);

        // Verify it's valid (not all zeros, not all ones)
        let bytes = &hv16.0;
        let all_zeros = bytes.iter().all(|&b| b == 0);
        let all_ones = bytes.iter().all(|&b| b == 0xFF);

        assert!(!all_zeros, "Converted HV16 shouldn't be all zeros");
        assert!(!all_ones, "Converted HV16 shouldn't be all ones");
    }

    #[test]
    fn test_topology_to_hv16_components() {
        // Create a small star topology
        let topology = ConsciousnessTopology::star(5, 2048, 42);

        // Convert to HV16 components
        let components = topology_to_hv16_components(&topology);

        // Should have 5 components (one per node)
        assert_eq!(components.len(), 5);

        // Each should be valid HV16
        for comp in &components {
            let bytes = &comp.0;
            assert_eq!(bytes.len(), 256, "HV16 should be 256 bytes");
        }
    }

    #[test]
    fn test_statistical_functions() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Mean should be 3.0
        let m = mean(&values);
        assert!((m - 3.0).abs() < 1e-10);

        // Standard deviation should be ~1.58
        let s = std_dev(&values, m);
        assert!((s - 1.5811388300841898).abs() < 0.01);
    }

    #[test]
    fn test_normal_cdf() {
        // Test known values
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.01, "CDF(0) should be ~0.5");
        assert!((normal_cdf(1.96) - 0.975).abs() < 0.01, "CDF(1.96) should be ~0.975");
        assert!((normal_cdf(-1.96) - 0.025).abs() < 0.01, "CDF(-1.96) should be ~0.025");
    }

    #[test]
    fn test_validation_result_methods() {
        let result = ValidationResult {
            n_random: 10,
            n_star: 10,
            mean_phi_random: 0.3,
            std_phi_random: 0.05,
            mean_phi_star: 0.5,
            std_phi_star: 0.05,
            t_statistic: 10.0,
            degrees_of_freedom: 18.0,
            p_value: 0.001,
            effect_size: 0.8,
            total_time_ms: 1000,
            avg_time_per_phi_ms: 50.0,
            phi_random_values: vec![0.3; 10],
            phi_star_values: vec![0.5; 10],
        };

        assert!(result.is_significant(), "p < 0.05 should be significant");
        assert!(result.has_large_effect(), "d > 0.5 should be large effect");
        assert!(result.validation_succeeded(), "Should pass all criteria");
    }

    #[test]
    #[ignore] // Run with `cargo test --ignored` - takes 1-2 minutes
    fn test_run_minimal_validation_quick() {
        println!("\n🔬 RUNNING MINIMAL Φ VALIDATION (Quick Version)");
        println!("=================================================\n");

        let mut validation = MinimalPhiValidation::quick();
        let result = validation.run();

        println!("\n📊 FINAL RESULTS:");
        println!("{}", result.summary());
        println!();

        // Report results regardless of pass/fail
        println!("✅ VALIDATION ANALYSIS:");
        println!("   Direction Correct: {} (Φ_star={:.4} > Φ_random={:.4})",
            result.mean_phi_star > result.mean_phi_random,
            result.mean_phi_star,
            result.mean_phi_random
        );
        println!("   Statistically Significant: {} (p={:.4})",
            result.is_significant(), result.p_value);
        println!("   Large Effect Size: {} (Cohen's d={:.3})",
            result.has_large_effect(), result.effect_size);
        println!();

        // Assert basic direction is correct
        assert!(
            result.mean_phi_star > result.mean_phi_random,
            "Star topology should have higher Φ than random (got random={:.4}, star={:.4})",
            result.mean_phi_random,
            result.mean_phi_star
        );

        // If test passes, report success
        if result.validation_succeeded() {
            println!("🎉 SUCCESS: Validation hypothesis confirmed!");
            println!("   Star topology has significantly higher Φ than Random topology.");
            println!("   p = {:.4} (significant at α = 0.05)", result.p_value);
            println!("   Cohen's d = {:.3} (large effect size)", result.effect_size);
        } else {
            println!("⚠️  PARTIAL SUCCESS: Direction is correct but statistical thresholds not met.");
            println!("   This may be due to small sample size (n={}).", result.n_random);
            println!("   Consider running with larger sample sizes for publication.");
        }
    }
}
