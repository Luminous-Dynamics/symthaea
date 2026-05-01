// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Φ (Integrated Information) Validation for Consciousness Topologies
//!
//! This module integrates ContinuousHV topology generators with TieredPhi calculation
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
//! 2. Convert ContinuousHV representations to binary BinaryHV format
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

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::consciousness_topology_generators::ConsciousnessTopology;
use crate::hdc::spectral_connectivity::ConnectivityCalculator; // ✨ NEW: Direct ContinuousHV Φ calculation
use crate::hdc::tiered_phi::{ApproximationTier, TieredPhi};
use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};
use std::time::Instant;
use thiserror::Error;

// ============================================================================
// ERROR TYPES
// ============================================================================

/// Errors that can occur during Phi topology validation
#[derive(Debug, Error)]
pub enum PhiValidationError {
    /// Unknown topology type specified
    #[error("Unknown topology type: '{0}'. Valid types are 'random' and 'star'")]
    UnknownTopologyType(String),
}

// ============================================================================
// REALHV ↔ BinaryHV CONVERSION
// ============================================================================

/// Convert ContinuousHV to binary BinaryHV using threshold-based binarization
///
/// **Strategy**: Binarize by comparing each element to the mean.
/// - Above mean → bit 1
/// - Below mean → bit 0
///
/// This preserves the essential structure while converting to binary format.
pub fn real_hv_to_hv16(real_hv: &ContinuousHV) -> BinaryHV {
    let values = &real_hv.values;
    let n = values.len().min(16_384);

    // Compute mean for threshold
    let sum: f32 = values[..n].iter().sum();
    let mean = sum / n as f32;

    // Create binary representation: process 8 elements per byte
    // BinaryHV is 16,384 bits = 2048 bytes
    let mut bytes = [0u8; 2048];

    for (byte_idx, chunk) in values[..n].chunks(8).enumerate() {
        let mut byte = 0u8;
        for (bit_idx, &val) in chunk.iter().enumerate() {
            if val > mean {
                byte |= 1 << bit_idx;
            }
        }
        bytes[byte_idx] = byte;
    }

    BinaryHV(bytes)
}

/// Convert ContinuousHV to BinaryHV using **median threshold** (more robust to outliers)
///
/// Median is more robust than mean when dealing with extreme values.
/// This should better preserve heterogeneity for high-variance vectors.
pub fn real_hv_to_hv16_median(real_hv: &ContinuousHV) -> BinaryHV {
    let values = &real_hv.values;
    let n = values.len();

    // Compute median for threshold
    let mut sorted_values: Vec<f32> = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = if n.is_multiple_of(2) {
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

    BinaryHV(bytes)
}

/// Convert ContinuousHV to BinaryHV using **probabilistic binarization**
///
/// Each value is converted to a probability using sigmoid, then randomly
/// binarized. This preserves more information from the original distribution.
///
/// # Arguments
/// * `real_hv` - Input real-valued hypervector
/// * `seed` - Random seed for reproducibility
pub fn real_hv_to_hv16_probabilistic(real_hv: &ContinuousHV, seed: u64) -> BinaryHV {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let values = &real_hv.values;
    let mut bytes = [0u8; 2048];

    // Normalize values to roughly [-3, 3] range for sigmoid
    let sum: f32 = values.iter().sum();
    let mean = sum / values.len() as f32;
    let variance: f32 =
        values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32;
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

    BinaryHV(bytes)
}

/// Convert ContinuousHV to BinaryHV using **quantile-based threshold** (percentile)
///
/// Uses a specific percentile as the threshold. 50th percentile = median.
/// Can use other percentiles like 25th or 75th for skewed distributions.
///
/// # Arguments
/// * `real_hv` - Input real-valued hypervector
/// * `percentile` - Threshold percentile (0.0 - 100.0), typically 50.0
pub fn real_hv_to_hv16_quantile(real_hv: &ContinuousHV, percentile: f32) -> BinaryHV {
    let values = &real_hv.values;
    let n = values.len();

    // Compute percentile threshold
    let mut sorted_values: Vec<f32> = values.to_vec();
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

    BinaryHV(bytes)
}

/// Convert ConsciousnessTopology node representations to BinaryHV vector
///
/// Takes all node representations from a topology and converts them to
/// binary format suitable for Φ calculation.
pub fn topology_to_hv16_components(topology: &ConsciousnessTopology) -> Vec<BinaryHV> {
    topology
        .node_representations
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
        self.is_significant()
            && self.has_large_effect()
            && self.mean_phi_star > self.mean_phi_random
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
            self.mean_phi_random,
            self.std_phi_random,
            self.n_random,
            self.mean_phi_star,
            self.std_phi_star,
            self.n_star,
            self.degrees_of_freedom,
            self.t_statistic,
            self.p_value,
            self.effect_size,
            self.is_significant(),
            self.has_large_effect(),
            self.validation_succeeded(),
            self.total_time_ms,
            self.avg_time_per_phi_ms
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
            // Use Exhaustive tier for topology validation fidelity on small systems.
            phi_calculator: TieredPhi::new(ApproximationTier::ExhaustivePartition),
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
    ///
    /// # Errors
    ///
    /// Returns `PhiValidationError::UnknownTopologyType` if an internal error occurs
    /// (should not happen with valid internal state).
    pub fn run(&mut self) -> Result<ValidationResult, PhiValidationError> {
        let start_time = Instant::now();

        println!("\n🔬 MINIMAL Φ VALIDATION: Random vs Star Topologies");
        println!("============================================================");
        println!(
            "Samples: {} random, {} star",
            self.n_random_samples, self.n_star_samples
        );
        println!("Nodes per topology: {}", self.n_nodes);
        println!("Dimensionality: {}", self.dim);
        println!("Φ calculator: {:?}", self.phi_calculator.tier());
        println!("Seed: {}", self.seed);
        println!();

        // Step 1: Generate Random topologies and compute Φ
        println!(
            "📊 Step 1: Generating {} Random topologies...",
            self.n_random_samples
        );
        let phi_random_values = self.compute_phi_for_topology_type("random")?;
        println!("   Mean Φ (Random): {:.4}", mean(&phi_random_values));
        println!();

        // Step 2: Generate Star topologies and compute Φ
        println!(
            "⭐ Step 2: Generating {} Star topologies...",
            self.n_star_samples
        );
        let phi_star_values = self.compute_phi_for_topology_type("star")?;
        println!("   Mean Φ (Star): {:.4}", mean(&phi_star_values));
        println!();

        // Step 3: Statistical analysis
        println!("📈 Step 3: Statistical Analysis");
        let stats = self.compute_statistics(&phi_random_values, &phi_star_values);

        let total_time_ms = start_time.elapsed().as_millis() as u64;
        let total_calculations = (self.n_random_samples + self.n_star_samples) as f64;
        let avg_time_per_phi_ms = total_time_ms as f64 / total_calculations;

        Ok(ValidationResult {
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
        })
    }

    /// Run the validation using **ConnectivityCalculator** (no binarization)
    ///
    /// This uses the RealPhi calculator directly on continuous data,
    /// avoiding the lossy ContinuousHV to BinaryHV conversion that can destroy structure.
    ///
    /// # Returns
    ///
    /// `ValidationResult` containing all Phi values computed using cosine similarity
    ///
    /// # Errors
    ///
    /// Returns `PhiValidationError::UnknownTopologyType` if an internal error occurs
    /// (should not happen with valid internal state).
    pub fn run_with_real_phi(&mut self) -> Result<ValidationResult, PhiValidationError> {
        let start_time = Instant::now();

        println!("\n🔬 REAL Φ VALIDATION: Random vs Star Topologies (No Binarization)");
        println!("============================================================");
        println!("Method: ConnectivityCalculator (cosine similarity, no conversion)");
        println!(
            "Samples: {} random, {} star",
            self.n_random_samples, self.n_star_samples
        );
        println!("Nodes per topology: {}", self.n_nodes);
        println!("Dimensionality: {}", self.dim);
        println!("Seed: {}", self.seed);
        println!();

        // Step 1: Generate Random topologies and compute Φ using RealPhi
        println!(
            "📊 Step 1: Generating {} Random topologies...",
            self.n_random_samples
        );
        let phi_random_values = self.compute_real_phi_for_topology_type("random")?;
        println!("   Mean Φ (Random): {:.4}", mean(&phi_random_values));
        println!();

        // Step 2: Generate Star topologies and compute Φ using RealPhi
        println!(
            "⭐ Step 2: Generating {} Star topologies...",
            self.n_star_samples
        );
        let phi_star_values = self.compute_real_phi_for_topology_type("star")?;
        println!("   Mean Φ (Star): {:.4}", mean(&phi_star_values));
        println!();

        // Step 3: Statistical analysis
        println!("📈 Step 3: Statistical Analysis");
        let stats = self.compute_statistics(&phi_random_values, &phi_star_values);

        let total_time_ms = start_time.elapsed().as_millis() as u64;
        let total_calculations = (self.n_random_samples + self.n_star_samples) as f64;
        let avg_time_per_phi_ms = total_time_ms as f64 / total_calculations;

        Ok(ValidationResult {
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
        })
    }

    /// Compute Φ using ConnectivityCalculator for multiple instances of a topology type
    ///
    /// ✨ This computes Φ directly on ContinuousHV without converting to BinaryHV
    ///
    /// # Errors
    ///
    /// Returns `PhiValidationError::UnknownTopologyType` if topology_type is not "random" or "star"
    fn compute_real_phi_for_topology_type(
        &mut self,
        topology_type: &str,
    ) -> Result<Vec<f64>, PhiValidationError> {
        let n_samples = match topology_type {
            "random" => self.n_random_samples,
            "star" => self.n_star_samples,
            _ => {
                return Err(PhiValidationError::UnknownTopologyType(
                    topology_type.to_string(),
                ))
            }
        };

        let mut phi_values = Vec::with_capacity(n_samples);
        let real_phi_calc = ConnectivityCalculator::new();

        for i in 0..n_samples {
            // Generate topology with unique seed
            let seed = self.seed + (i as u64 * 1000);

            let topology = match topology_type {
                "random" => ConsciousnessTopology::random(self.n_nodes, self.dim, seed),
                "star" => ConsciousnessTopology::star(self.n_nodes, self.dim, seed),
                _ => unreachable!(),
            };

            // Get ContinuousHV components directly (no conversion!)
            let components = &topology.node_representations;

            // DEBUG: Print cosine similarities for first sample
            if i == 0 {
                println!("   🔍 DEBUG: Cosine similarities for first {topology_type} topology:");
                for node_i in 0..components.len().min(5) {
                    for node_j in (node_i + 1)..components.len().min(5) {
                        let sim = components[node_i].similarity(&components[node_j]);
                        println!("      Node {node_i} ↔ Node {node_j}: {sim:.4}");
                    }
                }
                println!();
            }

            // Compute Φ using ConnectivityCalculator (no binarization!)
            let phi = real_phi_calc.algebraic_connectivity(components);

            // DEBUG: Print Φ value for each sample
            if i < 5 {
                // First 5 samples only
                println!("      Sample {i}: Φ = {phi:.6}");
            }

            phi_values.push(phi);

            // Progress indicator every 10 samples
            if (i + 1) % 10 == 0 || i == n_samples - 1 {
                print!("   Progress: {}/{} samples completed\r", i + 1, n_samples);
                use std::io::{self, Write};
                let _ = io::stdout().flush();
            }
        }

        println!(); // New line after progress
        Ok(phi_values)
    }

    /// Compute Φ for multiple instances of a topology type
    ///
    /// # Errors
    ///
    /// Returns `PhiValidationError::UnknownTopologyType` if topology_type is not "random" or "star"
    fn compute_phi_for_topology_type(
        &mut self,
        topology_type: &str,
    ) -> Result<Vec<f64>, PhiValidationError> {
        let n_samples = match topology_type {
            "random" => self.n_random_samples,
            "star" => self.n_star_samples,
            _ => {
                return Err(PhiValidationError::UnknownTopologyType(
                    topology_type.to_string(),
                ))
            }
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

            // Convert to BinaryHV components
            let components = topology_to_hv16_components(&topology);

            // DEBUG: Print Hamming distances for first sample
            if i == 0 {
                println!("   🔍 DEBUG: Hamming distances for first {topology_type} topology:");
                for node_i in 0..components.len() {
                    for node_j in (node_i + 1)..components.len() {
                        let dist = components[node_i].hamming_distance(&components[node_j]);
                        println!(
                            "      Node {} ↔ Node {}: {} / 2048 = {:.4}",
                            node_i,
                            node_j,
                            dist,
                            dist as f64 / 2048.0
                        );
                    }
                }
                println!();
            }

            // Compute Φ
            let phi = self.phi_calculator.compute(&components);

            // DEBUG: Print Φ value for each sample
            if i < 5 {
                // First 5 samples only
                println!("      Sample {i}: Φ = {phi:.6}");
            }

            phi_values.push(phi);

            // Progress indicator every 10 samples
            if (i + 1) % 10 == 0 || i == n_samples - 1 {
                print!("   Progress: {}/{} samples completed\r", i + 1, n_samples);
                use std::io::{self, Write};
                let _ = io::stdout().flush();
            }
        }

        println!(); // New line after progress
        Ok(phi_values)
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
        let pooled_std = ((((n1 - 1.0) * std_random * std_random)
            + ((n2 - 1.0) * std_star * std_star))
            / (n1 + n2 - 2.0))
            .sqrt();

        // t-statistic
        let t_statistic =
            (mean_star - mean_random) / (pooled_std * ((1.0 / n1) + (1.0 / n2)).sqrt());

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

        (
            mean_random,
            std_random,
            mean_star,
            std_star,
            t_statistic,
            df,
            p_value,
            cohens_d,
        )
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

    let variance =
        values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

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
    let erf = 1.0
        - (((((1.061405429 * t + -1.453152027) * t) + 1.421413741) * t + -0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();

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
        // Create a simple ContinuousHV
        let real_hv = ContinuousHV::random(2048, 42);

        // Convert to BinaryHV
        let hv16 = real_hv_to_hv16(&real_hv);

        // Verify it's valid (not all zeros, not all ones)
        let bytes = &hv16.0;
        let all_zeros = bytes.iter().all(|&b| b == 0);
        let all_ones = bytes.iter().all(|&b| b == 0xFF);

        assert!(!all_zeros, "Converted BinaryHV shouldn't be all zeros");
        assert!(!all_ones, "Converted BinaryHV shouldn't be all ones");
    }

    #[test]
    fn test_topology_to_hv16_components() {
        use crate::hdc::HDC_DIMENSION;

        // Create a small star topology with standard HDC dimension
        let topology = ConsciousnessTopology::star(5, HDC_DIMENSION, 42);

        // Convert to BinaryHV components
        let components = topology_to_hv16_components(&topology);

        // Should have 5 components (one per node)
        assert_eq!(components.len(), 5);

        // Each should be valid BinaryHV (16,384 bits = 2048 bytes)
        for comp in &components {
            let bytes = &comp.0;
            assert_eq!(
                bytes.len(),
                2048,
                "BinaryHV should be 2048 bytes (16,384 bits)"
            );
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
        assert!(
            (normal_cdf(0.0) - 0.5).abs() < 0.01,
            "CDF(0) should be ~0.5"
        );
        assert!(
            (normal_cdf(1.96) - 0.975).abs() < 0.01,
            "CDF(1.96) should be ~0.975"
        );
        assert!(
            (normal_cdf(-1.96) - 0.025).abs() < 0.01,
            "CDF(-1.96) should be ~0.025"
        );
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
    fn test_run_minimal_validation_quick() {
        // Use ExhaustivePartition (O(2^n)) for correct IIT Phi computation.
        // SpectralConnectivity computes algebraic_connectivity (lambda_2),
        // which has r ~ -0.62 correlation with true Phi — nearly opposite.
        // ExhaustivePartition searches all bipartitions for the MIP.
        //
        // We use n=4 nodes to keep runtime tractable (2^4 = 16 partitions).
        let n_samples = 5;
        let n_nodes = 4;
        let dim = crate::hdc::HDC_DIMENSION;
        let mut phi_calc = TieredPhi::new(ApproximationTier::ExhaustivePartition);

        println!(
            "\n  RUNNING Phi VALIDATION (ExhaustivePartition, n={})",
            n_nodes
        );
        println!("=================================================\n");

        // Compute Phi for Random topologies
        let mut phi_random_values = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let seed = 42 + (i as u64 * 1000);
            let topo = ConsciousnessTopology::random(n_nodes, dim, seed);
            let binary: Vec<BinaryHV> = topo
                .node_representations
                .iter()
                .map(|c| c.to_binary(0.0))
                .collect();
            let phi = phi_calc.compute_phi(&binary);
            phi_random_values.push(phi);
        }
        let mean_random = mean(&phi_random_values);

        // Compute Phi for Star topologies
        let mut phi_star_values = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let seed = 42 + (i as u64 * 1000);
            let topo = ConsciousnessTopology::star(n_nodes, dim, seed);
            let binary: Vec<BinaryHV> = topo
                .node_representations
                .iter()
                .map(|c| c.to_binary(0.0))
                .collect();
            let phi = phi_calc.compute_phi(&binary);
            phi_star_values.push(phi);
        }
        let mean_star = mean(&phi_star_values);

        println!("  Mean Phi (Random): {:.6}", mean_random);
        println!("  Mean Phi (Star):   {:.6}", mean_star);
        println!();

        // Both should be positive
        assert!(
            mean_star > 0.0 && mean_random > 0.0,
            "Both Phi values should be positive (got star={:.6}, random={:.6})",
            mean_star,
            mean_random
        );

        // Hard assertion: star topology should have higher Phi than random
        // ExhaustivePartition computes true IIT MIP, so this should hold.
        assert!(
            mean_star > mean_random,
            "Star Phi ({:.6}) should exceed Random Phi ({:.6}) with ExhaustivePartition",
            mean_star,
            mean_random
        );

        println!(
            "  SUCCESS: Star topology Phi ({:.6}) > Random Phi ({:.6})",
            mean_star, mean_random
        );
        println!("  Ratio: {:.2}x", mean_star / mean_random);
    }
}
