//! Hypervector-Based Integrated Information (Φ) Measurement
//!
//! This module implements REAL Φ measurement using hypervector approximation,
//! providing Byzantine detection through information-theoretic analysis.
//!
//! # Mathematical Foundation
//!
//! Traditional IIT Φ: O(2^n) complexity - intractable for neural networks
//! Hypervector Φ: O(L²) where L = num_layers - tractable!
//!
//! ```text
//! Φ_hv ≈ total_integration - sum_of_parts_integration
//! ```
//!
//! Where integration = average pairwise cosine similarity between layer hypervectors.
//!
//! # Key Insight
//!
//! - Honest gradients increase system Φ (coherent learning)
//! - Byzantine gradients decrease system Φ (information destruction)
//!
//! # Example
//!
//! ```rust
//! use fl_aggregator::phi::{PhiMeasurer, PhiConfig};
//! use std::collections::HashMap;
//!
//! // Create measurer
//! let mut measurer = PhiMeasurer::new(PhiConfig::default());
//!
//! // Detect Byzantine nodes via Φ degradation
//! let mut gradients = HashMap::new();
//! gradients.insert("node1".to_string(), vec![1.0, 2.0, 3.0]);
//! gradients.insert("node2".to_string(), vec![1.1, 2.1, 3.1]);
//! gradients.insert("byzantine".to_string(), vec![100.0, -100.0, 50.0]);
//!
//! let (byzantine_nodes, system_phi) = measurer.detect_byzantine_by_phi(&gradients, 1.1);
//! println!("Detected Byzantine: {:?}", byzantine_nodes);
//! ```

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;

/// Simple xorshift64* PRNG for deterministic random projections.
/// This avoids external rand dependency while providing good statistical properties.
#[derive(Debug, Clone)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545f4914f6cdd1d)
    }

    /// Generate f32 in range [0, 1)
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Generate f32 in specified range
    fn gen_range(&mut self, min: f32, max: f32) -> f32 {
        min + self.next_f32() * (max - min)
    }
}

/// Default hypervector dimension for Phi measurement
pub const PHI_DIMENSION: usize = 2048;

/// Integrated Information metrics for a gradient/model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiMetrics {
    /// Φ of model before gradient application
    pub phi_before: f32,
    /// Φ of model after gradient application
    pub phi_after: f32,
    /// Change in Φ (positive = learning, negative = degradation)
    pub phi_gain: f32,
    /// Confidence in the gradient's quality (0-1)
    pub epistemic_confidence: f32,
    /// Per-layer contribution to total Φ
    #[serde(skip_serializing_if = "Option::is_none")]
    pub layer_contributions: Option<HashMap<String, f32>>,
}

impl PhiMetrics {
    /// Create new PhiMetrics
    pub fn new(phi_before: f32, phi_after: f32, epistemic_confidence: f32) -> Self {
        Self {
            phi_before,
            phi_after,
            phi_gain: phi_after - phi_before,
            epistemic_confidence,
            layer_contributions: None,
        }
    }

    /// Add layer contributions
    pub fn with_layer_contributions(mut self, contributions: HashMap<String, f32>) -> Self {
        self.layer_contributions = Some(contributions);
        self
    }
}

/// Result from a single Φ measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiMeasurementResult {
    /// Total integrated information
    pub phi_total: f32,
    /// Per-layer Φ contributions
    pub phi_layers: Vec<f32>,
    /// How much integration exceeds sum of parts
    pub integration_gain: f32,
    /// Time taken to compute (milliseconds)
    pub computation_time_ms: f64,
    /// Number of layers analyzed
    pub layer_count: usize,
}

/// Configuration for Phi measurement.
#[derive(Debug, Clone)]
pub struct PhiConfig {
    /// Hypervector dimension (2048 recommended for accuracy)
    pub dimension: usize,
    /// Scaling factor learned from ground truth
    pub calibration_factor: f32,
    /// Weight layers by causal importance
    pub use_causal_weighting: bool,
    /// Random seed for reproducible projections
    pub seed: u64,
}

impl Default for PhiConfig {
    fn default() -> Self {
        Self {
            dimension: PHI_DIMENSION,
            calibration_factor: 1.0,
            use_causal_weighting: true,
            seed: 42,
        }
    }
}

impl PhiConfig {
    /// Create config with custom dimension
    pub fn with_dimension(mut self, dimension: usize) -> Self {
        self.dimension = dimension;
        self
    }

    /// Create config with custom seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Disable causal weighting
    pub fn without_causal_weighting(mut self) -> Self {
        self.use_causal_weighting = false;
        self
    }
}

/// Measures Integrated Information (Φ) using hypervector approximation.
///
/// This provides Byzantine detection via Φ degradation - Byzantine gradients
/// decrease system coherence, so removing them increases Φ.
#[derive(Debug)]
pub struct PhiMeasurer {
    config: PhiConfig,
    rng: Xorshift64,
    /// Cache for projection matrices (keyed by input dimension)
    projection_cache: HashMap<usize, Array2<f32>>,
}

impl PhiMeasurer {
    /// Create a new Phi measurer with the given configuration.
    pub fn new(config: PhiConfig) -> Self {
        let rng = Xorshift64::new(config.seed);
        Self {
            config,
            rng,
            projection_cache: HashMap::new(),
        }
    }

    /// Get or create random projection matrix for given input dimension.
    ///
    /// Uses Gaussian random projection which preserves distances approximately
    /// (Johnson-Lindenstrauss lemma).
    fn get_projection_matrix(&mut self, input_dim: usize) -> &Array2<f32> {
        if !self.projection_cache.contains_key(&input_dim) {
            let mut projection = Array2::zeros((input_dim, self.config.dimension));
            let scale = 1.0f32 / (self.config.dimension as f32).sqrt();

            for i in 0..input_dim {
                for j in 0..self.config.dimension {
                    // Generate Gaussian random value using Box-Muller transform
                    let u1: f32 = self.rng.gen_range(0.0001, 1.0);
                    let u2: f32 = self.rng.next_f32();
                    let z: f32 = (-2.0f32 * u1.ln()).sqrt() * (2.0f32 * std::f32::consts::PI * u2).cos();
                    projection[[i, j]] = z * scale;
                }
            }

            self.projection_cache.insert(input_dim, projection);
        }
        self.projection_cache.get(&input_dim).unwrap()
    }

    /// Encode arbitrary data to a hypervector.
    ///
    /// Uses random projection to map from input space to hypervector space.
    pub fn encode_to_hypervector(&mut self, data: &[f32]) -> Array1<f32> {
        if data.is_empty() {
            return Array1::zeros(self.config.dimension);
        }

        let input_dim = data.len();
        let projection = self.get_projection_matrix(input_dim).clone();

        // Compute hypervector via matrix multiplication
        let mut hypervector = Array1::zeros(self.config.dimension);
        for j in 0..self.config.dimension {
            let mut sum = 0.0f32;
            for (i, &val) in data.iter().enumerate() {
                sum += val * projection[[i, j]];
            }
            hypervector[j] = sum;
        }

        // Normalize to unit length
        let norm = hypervector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            hypervector.mapv_inplace(|x| x / norm);
        }

        hypervector
    }

    /// Compute cosine similarity between two hypervectors.
    pub fn cosine_similarity(&self, hv1: &Array1<f32>, hv2: &Array1<f32>) -> f32 {
        let dot_product: f32 = hv1.iter().zip(hv2.iter()).map(|(a, b)| a * b).sum();
        let norm1 = hv1.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2 = hv2.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 < 1e-8 || norm2 < 1e-8 {
            return 0.0;
        }

        dot_product / (norm1 * norm2)
    }

    /// Compute integration metric for a set of hypervectors.
    ///
    /// Integration = average pairwise cosine similarity.
    /// Higher values indicate more coherent, integrated information.
    pub fn compute_integration(&self, hypervectors: &[Array1<f32>]) -> f32 {
        let n = hypervectors.len();
        if n <= 1 {
            return 0.0;
        }

        let mut total_similarity = 0.0f32;
        let mut num_pairs = 0usize;

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = self.cosine_similarity(&hypervectors[i], &hypervectors[j]);
                total_similarity += sim.max(0.0); // Only positive similarities
                num_pairs += 1;
            }
        }

        if num_pairs == 0 {
            return 0.0;
        }

        total_similarity / num_pairs as f32
    }

    /// Compute Φ (integrated information) from layer hypervectors.
    ///
    /// Φ ≈ total_integration - sum_of_parts_integration
    ///
    /// This is the core IIT-inspired computation using hypervector approximation.
    pub fn measure_phi_from_hypervectors(
        &self,
        layer_hypervectors: &[Array1<f32>],
        layer_weights: Option<&[f32]>,
    ) -> (f32, Array2<f32>) {
        let n = layer_hypervectors.len();

        if n == 0 {
            return (0.0, Array2::zeros((0, 0)));
        }

        if n == 1 {
            return (0.0, Array2::from_elem((1, 1), 1.0));
        }

        // Apply layer weights if provided
        let weighted_hvs: Vec<Array1<f32>> = if let Some(weights) = layer_weights {
            if self.config.use_causal_weighting {
                layer_hypervectors
                    .iter()
                    .zip(weights.iter())
                    .map(|(hv, &w)| hv.mapv(|x| x * w))
                    .collect()
            } else {
                layer_hypervectors.to_vec()
            }
        } else {
            layer_hypervectors.to_vec()
        };

        // Compute integration matrix (pairwise similarities)
        let mut integration_matrix = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    integration_matrix[[i, j]] = 1.0;
                } else {
                    integration_matrix[[i, j]] =
                        self.cosine_similarity(&weighted_hvs[i], &weighted_hvs[j]);
                }
            }
        }

        // Total integration (whole system)
        let total_integration = self.compute_integration(&weighted_hvs);

        // Parts integration (sum of individual components)
        // For hypervectors, individual component "integration" is 0
        let parts_integration = 0.0f32;

        // Φ = how much more integrated the whole is than the parts
        let phi = (total_integration - parts_integration) * self.config.calibration_factor;

        (phi, integration_matrix)
    }

    /// Measure Φ from layer activations, returning a structured result.
    pub fn measure_phi_from_activations(
        &mut self,
        layer_activations: &[Vec<f32>],
    ) -> PhiMeasurementResult {
        let start = Instant::now();

        // Convert activations to hypervectors
        let layer_hvs: Vec<Array1<f32>> = layer_activations
            .iter()
            .map(|act| self.encode_to_hypervector(act))
            .collect();

        // Measure Φ
        let (phi_total, integration_matrix) = self.measure_phi_from_hypervectors(&layer_hvs, None);

        // Compute per-layer contributions
        let n = layer_hvs.len();
        let phi_layers: Vec<f32> = (0..n)
            .map(|i| {
                if n > 1 {
                    // Layer's contribution = average of its row in integration matrix (excluding diagonal)
                    let row_sum: f32 = (0..n).map(|j| integration_matrix[[i, j]]).sum();
                    (row_sum - 1.0) / (n - 1) as f32
                } else {
                    0.0
                }
            })
            .collect();

        let computation_time = start.elapsed().as_secs_f64() * 1000.0;

        PhiMeasurementResult {
            phi_total,
            phi_layers,
            integration_gain: phi_total, // In our formulation, phi IS the integration gain
            computation_time_ms: computation_time,
            layer_count: n,
        }
    }

    /// Compute causal importance weights for each layer.
    ///
    /// Later layers have higher causal impact on output (generally).
    /// Also considers parameter count as proxy for capacity.
    pub fn compute_causal_weights(&self, layer_sizes: &[usize]) -> Vec<f32> {
        let n = layer_sizes.len();
        if n == 0 {
            return vec![];
        }

        let weights: Vec<f32> = layer_sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                // Positional weight (later layers = higher impact)
                let position_weight = (i + 1) as f32 / n as f32;
                // Capacity weight (more parameters = more expressive)
                let capacity_weight = (1.0 + size as f32).ln();
                position_weight * capacity_weight
            })
            .collect();

        // Normalize
        let total: f32 = weights.iter().sum();
        if total > 0.0 {
            weights.iter().map(|w| w / total).collect()
        } else {
            vec![1.0 / n as f32; n]
        }
    }

    /// Measure Φ of a gradient (represented as list of layer gradients).
    pub fn measure_gradient_phi(&mut self, layer_gradients: &[Vec<f32>]) -> f32 {
        if layer_gradients.is_empty() {
            return 0.0;
        }

        let layer_hvs: Vec<Array1<f32>> = layer_gradients
            .iter()
            .map(|g| self.encode_to_hypervector(g))
            .collect();

        let (phi, _) = self.measure_phi_from_hypervectors(&layer_hvs, None);
        phi
    }

    /// Compute epistemic confidence in a gradient.
    ///
    /// Based on:
    /// 1. Gradient-model alignment (should be somewhat aligned)
    /// 2. Gradient coherence across layers
    /// 3. Gradient magnitude reasonableness
    pub fn compute_epistemic_confidence(
        &self,
        model_hvs: &[Array1<f32>],
        gradient_hvs: &[Array1<f32>],
    ) -> f32 {
        if model_hvs.len() != gradient_hvs.len() {
            return 0.5; // Default medium confidence
        }

        let mut confidence_factors = Vec::new();

        // Factor 1: Gradient-model alignment
        let alignments: Vec<f32> = model_hvs
            .iter()
            .zip(gradient_hvs.iter())
            .map(|(m, g)| self.cosine_similarity(m, g).abs())
            .collect();

        let avg_alignment = if alignments.is_empty() {
            0.5
        } else {
            alignments.iter().sum::<f32>() / alignments.len() as f32
        };
        // Optimal alignment is moderate (not orthogonal, not identical)
        let alignment_score = 1.0 - (avg_alignment - 0.5).abs() * 2.0;
        confidence_factors.push(alignment_score);

        // Factor 2: Gradient coherence
        let gradient_coherence = self.compute_integration(gradient_hvs);
        // Moderate coherence is best (not random, not identical)
        let coherence_score = 1.0 - (gradient_coherence - 0.5).abs() * 2.0;
        confidence_factors.push(coherence_score);

        // Factor 3: Gradient magnitude distribution
        let norms: Vec<f32> = gradient_hvs
            .iter()
            .map(|hv| hv.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();

        let norm_std = if norms.len() > 1 {
            let mean = norms.iter().sum::<f32>() / norms.len() as f32;
            let variance =
                norms.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / norms.len() as f32;
            variance.sqrt()
        } else {
            0.0
        };
        // Low variance in norms is good
        let magnitude_score = (-norm_std).exp();
        confidence_factors.push(magnitude_score);

        // Combine factors
        let confidence =
            confidence_factors.iter().sum::<f32>() / confidence_factors.len() as f32;

        confidence.clamp(0.0, 1.0)
    }

    /// Detect Byzantine nodes via Φ degradation.
    ///
    /// Key insight: Byzantine gradients decrease system Φ when removed.
    /// If removing a gradient INCREASES system Φ, that gradient is
    /// information-destroying (likely Byzantine).
    ///
    /// # Arguments
    ///
    /// * `gradients` - HashMap of node_id -> gradient values
    /// * `threshold_multiplier` - Threshold for detection (1.1 = 10% improvement required)
    ///
    /// # Returns
    ///
    /// Tuple of (list of Byzantine node IDs, system Φ)
    pub fn detect_byzantine_by_phi(
        &mut self,
        gradients: &HashMap<String, Vec<f32>>,
        threshold_multiplier: f32,
    ) -> (Vec<String>, f32) {
        if gradients.is_empty() {
            return (vec![], 0.0);
        }

        // Encode all gradients to hypervectors
        let gradient_hvs: HashMap<String, Array1<f32>> = gradients
            .iter()
            .map(|(node_id, grad)| (node_id.clone(), self.encode_to_hypervector(grad)))
            .collect();

        // Compute system Φ with all gradients
        let all_hvs: Vec<Array1<f32>> = gradient_hvs.values().cloned().collect();
        let system_phi = self.compute_integration(&all_hvs);

        // Check each gradient's contribution
        let mut byzantine_candidates = Vec::new();

        for (node_id, _node_hv) in &gradient_hvs {
            // Compute Φ without this node
            let other_hvs: Vec<Array1<f32>> = gradient_hvs
                .iter()
                .filter(|(nid, _)| *nid != node_id)
                .map(|(_, hv)| hv.clone())
                .collect();

            if other_hvs.is_empty() {
                continue;
            }

            let phi_without = self.compute_integration(&other_hvs);

            // If removing this gradient INCREASES Φ, it's destroying information
            if phi_without > system_phi * threshold_multiplier {
                byzantine_candidates.push(node_id.clone());
                tracing::info!(
                    node_id = %node_id,
                    phi_without = phi_without,
                    system_phi = system_phi,
                    "Byzantine candidate detected via Phi degradation"
                );
            }
        }

        (byzantine_candidates, system_phi)
    }

    /// Detect Byzantine nodes using Array1 gradients (for integration with existing code).
    pub fn detect_byzantine_by_phi_array(
        &mut self,
        gradients: &HashMap<String, Array1<f32>>,
        threshold_multiplier: f32,
    ) -> (Vec<String>, f32) {
        let vec_gradients: HashMap<String, Vec<f32>> = gradients
            .iter()
            .map(|(k, v)| (k.clone(), v.to_vec()))
            .collect();
        self.detect_byzantine_by_phi(&vec_gradients, threshold_multiplier)
    }

    /// Compute full Φ metrics including before/after gradient application.
    pub fn measure_full_metrics(
        &mut self,
        model_layers: &[Vec<f32>],
        gradient_layers: &[Vec<f32>],
        learning_rate: f32,
    ) -> PhiMetrics {
        // Measure Φ before
        let model_hvs: Vec<Array1<f32>> = model_layers
            .iter()
            .map(|p| self.encode_to_hypervector(p))
            .collect();

        let layer_sizes: Vec<usize> = model_layers.iter().map(|l| l.len()).collect();
        let layer_weights = self.compute_causal_weights(&layer_sizes);

        let (phi_before, _) =
            self.measure_phi_from_hypervectors(&model_hvs, Some(&layer_weights));

        // Encode gradients
        let gradient_hvs: Vec<Array1<f32>> = gradient_layers
            .iter()
            .map(|g| self.encode_to_hypervector(g))
            .collect();

        // Compute "after" Φ by combining model and gradient hypervectors
        // (simulates gradient descent direction)
        let after_hvs: Vec<Array1<f32>> = model_hvs
            .iter()
            .zip(gradient_hvs.iter())
            .map(|(model_hv, grad_hv)| {
                // New state ≈ model - lr * gradient (in HV space)
                let mut after_hv = model_hv - &(grad_hv * learning_rate);
                // Normalize
                let norm = after_hv.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-8 {
                    after_hv.mapv_inplace(|x| x / norm);
                }
                after_hv
            })
            .collect();

        let (phi_after, _) =
            self.measure_phi_from_hypervectors(&after_hvs, Some(&layer_weights));

        // Compute epistemic confidence
        let epistemic_confidence = self.compute_epistemic_confidence(&model_hvs, &gradient_hvs);

        PhiMetrics::new(phi_before, phi_after, epistemic_confidence)
    }
}

impl Clone for PhiMeasurer {
    fn clone(&self) -> Self {
        Self {
            config: self.config.clone(),
            rng: Xorshift64::new(self.config.seed),
            projection_cache: self.projection_cache.clone(),
        }
    }
}

/// Convenience function to measure Φ of arbitrary data.
pub fn measure_phi(data: &[Vec<f32>], dimension: usize) -> f32 {
    if data.is_empty() || data.len() == 1 {
        return 0.0; // Single array has no integration
    }

    let mut measurer = PhiMeasurer::new(PhiConfig::default().with_dimension(dimension));
    let hvs: Vec<Array1<f32>> = data.iter().map(|v| measurer.encode_to_hypervector(v)).collect();
    let (phi, _) = measurer.measure_phi_from_hypervectors(&hvs, None);
    phi
}

/// Result of Byzantine detection via Phi analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhiByzantineResult {
    /// Detected Byzantine node IDs
    pub byzantine_nodes: Vec<String>,
    /// System-wide Phi value
    pub system_phi: f32,
    /// Per-node Phi contribution (Phi when node is removed)
    pub node_contributions: HashMap<String, f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_to_hypervector() {
        let mut measurer = PhiMeasurer::new(PhiConfig::default());
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let hv = measurer.encode_to_hypervector(&data);

        assert_eq!(hv.len(), PHI_DIMENSION);
        // Check normalization
        let norm: f32 = hv.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Hypervector should be normalized");
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let measurer = PhiMeasurer::new(PhiConfig::default());
        let hv1 = Array1::from(vec![1.0, 0.0, 0.0]);
        let hv2 = Array1::from(vec![1.0, 0.0, 0.0]);

        let sim = measurer.cosine_similarity(&hv1, &hv2);
        assert!((sim - 1.0).abs() < 0.001, "Identical vectors should have similarity 1.0");
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let measurer = PhiMeasurer::new(PhiConfig::default());
        let hv1 = Array1::from(vec![1.0, 0.0, 0.0]);
        let hv2 = Array1::from(vec![0.0, 1.0, 0.0]);

        let sim = measurer.cosine_similarity(&hv1, &hv2);
        assert!(sim.abs() < 0.001, "Orthogonal vectors should have similarity 0.0");
    }

    #[test]
    fn test_identical_layers_high_phi() {
        let mut measurer = PhiMeasurer::new(PhiConfig::default().with_seed(42));

        // Identical layers should have high integration
        let identical_layers: Vec<Vec<f32>> = (0..5).map(|_| vec![1.0; 100]).collect();
        let phi = measurer.measure_gradient_phi(&identical_layers);

        assert!(phi > 0.7, "Identical layers should have high Φ, got {}", phi);
    }

    #[test]
    fn test_random_layers_low_phi() {
        let mut measurer = PhiMeasurer::new(PhiConfig::default().with_seed(42));
        let mut rng = Xorshift64::new(123);

        // Random layers should have low integration
        let random_layers: Vec<Vec<f32>> = (0..5)
            .map(|_| (0..100).map(|_| rng.gen_range(-1.0, 1.0)).collect())
            .collect();
        let phi = measurer.measure_gradient_phi(&random_layers);

        assert!(phi < 0.3, "Random layers should have low Φ, got {}", phi);
    }

    #[test]
    fn test_correlated_layers_medium_phi() {
        let mut measurer = PhiMeasurer::new(PhiConfig::default().with_seed(42));
        let mut rng = Xorshift64::new(456);

        // Correlated layers should have medium integration
        // Use larger perturbations to create meaningful variation while keeping correlation
        let base: Vec<f32> = (0..100).map(|_| rng.gen_range(-1.0, 1.0)).collect();
        let correlated_layers: Vec<Vec<f32>> = (0..5)
            .map(|_| {
                base.iter()
                    .map(|&x| x + rng.gen_range(-0.5, 0.5))
                    .collect()
            })
            .collect();
        let phi = measurer.measure_gradient_phi(&correlated_layers);

        // Correlated (not identical) layers should have high but not perfect Φ
        assert!(
            phi > 0.3,
            "Correlated layers should have Φ > 0.3, got {}",
            phi
        );
    }

    #[test]
    fn test_byzantine_detection() {
        let mut measurer = PhiMeasurer::new(PhiConfig::default().with_seed(42));
        let mut rng = Xorshift64::new(789);

        // Create honest gradients (similar to each other)
        let base: Vec<f32> = (0..100).map(|_| rng.gen_range(-1.0, 1.0)).collect();

        let mut gradients = HashMap::new();
        gradients.insert(
            "honest_1".to_string(),
            base.iter()
                .map(|&x| x + rng.gen_range(-0.05, 0.05))
                .collect(),
        );
        gradients.insert(
            "honest_2".to_string(),
            base.iter()
                .map(|&x| x + rng.gen_range(-0.05, 0.05))
                .collect(),
        );
        gradients.insert(
            "honest_3".to_string(),
            base.iter()
                .map(|&x| x + rng.gen_range(-0.05, 0.05))
                .collect(),
        );

        // Byzantine gradient (very different)
        gradients.insert(
            "byzantine".to_string(),
            (0..100).map(|_| rng.gen_range(-10.0, 10.0)).collect(),
        );

        let (byzantine_nodes, system_phi) = measurer.detect_byzantine_by_phi(&gradients, 1.05);

        assert!(
            byzantine_nodes.contains(&"byzantine".to_string()),
            "Should detect Byzantine node, got {:?}",
            byzantine_nodes
        );
        assert!(system_phi > 0.0, "System Φ should be positive");
    }

    #[test]
    fn test_no_false_positives_honest_nodes() {
        let mut measurer = PhiMeasurer::new(PhiConfig::default().with_seed(42));
        let mut rng = Xorshift64::new(999);

        // All honest gradients
        let base: Vec<f32> = (0..100).map(|_| rng.gen_range(-1.0, 1.0)).collect();

        let mut gradients = HashMap::new();
        for i in 0..5 {
            gradients.insert(
                format!("honest_{}", i),
                base.iter()
                    .map(|&x| x + rng.gen_range(-0.05, 0.05))
                    .collect(),
            );
        }

        let (byzantine_nodes, _) = measurer.detect_byzantine_by_phi(&gradients, 1.1);

        assert!(
            byzantine_nodes.is_empty(),
            "Should not detect Byzantine nodes among honest ones, got {:?}",
            byzantine_nodes
        );
    }

    #[test]
    fn test_phi_metrics() {
        let mut measurer = PhiMeasurer::new(PhiConfig::default().with_seed(42));
        let mut rng = Xorshift64::new(111);

        let model_layers: Vec<Vec<f32>> = (0..3)
            .map(|_| (0..50).map(|_| rng.gen_range(-1.0, 1.0)).collect())
            .collect();

        let gradient_layers: Vec<Vec<f32>> = (0..3)
            .map(|_| (0..50).map(|_| rng.gen_range(-0.1, 0.1)).collect())
            .collect();

        let metrics = measurer.measure_full_metrics(&model_layers, &gradient_layers, 0.01);

        assert!(
            metrics.epistemic_confidence >= 0.0 && metrics.epistemic_confidence <= 1.0,
            "Epistemic confidence should be in [0, 1]"
        );
    }

    #[test]
    fn test_causal_weights() {
        let measurer = PhiMeasurer::new(PhiConfig::default());
        let layer_sizes = vec![100, 200, 50];

        let weights = measurer.compute_causal_weights(&layer_sizes);

        assert_eq!(weights.len(), 3);
        let sum: f32 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 0.001,
            "Weights should sum to 1.0, got {}",
            sum
        );
        // Later layers should generally have higher weights
        assert!(
            weights[2] > weights[0],
            "Last layer should have higher weight than first"
        );
    }

    #[test]
    fn test_measure_phi_from_activations() {
        let mut measurer = PhiMeasurer::new(PhiConfig::default().with_seed(42));

        let activations: Vec<Vec<f32>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![1.2, 2.2, 3.2],
        ];

        let result = measurer.measure_phi_from_activations(&activations);

        assert_eq!(result.layer_count, 3);
        assert_eq!(result.phi_layers.len(), 3);
        assert!(result.computation_time_ms >= 0.0);
        assert!(result.phi_total >= 0.0);
    }

    #[test]
    fn test_empty_input_handling() {
        let mut measurer = PhiMeasurer::new(PhiConfig::default());

        // Empty gradient
        let empty_hv = measurer.encode_to_hypervector(&[]);
        assert_eq!(empty_hv.len(), PHI_DIMENSION);

        // Empty gradients map
        let empty_map: HashMap<String, Vec<f32>> = HashMap::new();
        let (byzantine, phi) = measurer.detect_byzantine_by_phi(&empty_map, 1.1);
        assert!(byzantine.is_empty());
        assert_eq!(phi, 0.0);
    }

    #[test]
    fn test_convenience_measure_phi() {
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
            vec![1.0, 2.0, 3.0],
        ];

        let phi = measure_phi(&data, 512);
        assert!(phi > 0.7, "Identical data should have high Φ");
    }

    #[test]
    fn test_deterministic_with_seed() {
        let config = PhiConfig::default().with_seed(12345);
        let mut measurer1 = PhiMeasurer::new(config.clone());
        let mut measurer2 = PhiMeasurer::new(config);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let hv1 = measurer1.encode_to_hypervector(&data);
        let hv2 = measurer2.encode_to_hypervector(&data);

        for (a, b) in hv1.iter().zip(hv2.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "Same seed should produce identical results"
            );
        }
    }
}
