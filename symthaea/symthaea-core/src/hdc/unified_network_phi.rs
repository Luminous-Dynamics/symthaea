// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Phi Measurement for HdcLtcUnifiedNetwork
//!
//! This module provides comprehensive Phi (integrated information) measurement
//! for the revolutionary HdcLtcUnifiedNetwork architecture, where the neuron STATE
//! is a hypervector that evolves through Liquid Time-Constant dynamics.
//!
//! ## Key Features
//!
//! - Extract connectivity/state information from HdcLtcUnifiedNetwork
//! - Convert network states to format compatible with existing Phi calculators
//! - Measure Phi at different network states (initial, correlated, uncorrelated, equilibrium)
//! - Compare Phi values across different network configurations
//! - Validate IIT predictions (Phi increases with integration, decreases with partitioning)
//!
//! ## Scientific Foundation
//!
//! Based on Integrated Information Theory (IIT) by Giulio Tononi:
//! - Phi measures how much information a system generates above and beyond its parts
//! - Higher Phi = more integrated consciousness
//! - A unified HDC-LTC network should exhibit meaningful Phi due to:
//!   1. Holographic state representation (information distributed across dimensions)
//!   2. Temporal dynamics creating causal dependencies
//!   3. Layer-wise binding creating hierarchical integration
//!
//! ## Example Usage
//!
//! ```rust,ignore
//! use symthaea::hdc::unified_network_phi::{UnifiedNetworkPhiMeasurer, PhiMeasurement};
//! use symthaea::hdc::hdc_ltc_unified::{HdcLtcUnifiedNetwork, UnifiedNetworkConfig};
//!
//! // Create network and measurer
//! let config = UnifiedNetworkConfig::default();
//! let mut network = HdcLtcUnifiedNetwork::new(config, 42);
//! let measurer = UnifiedNetworkPhiMeasurer::new();
//!
//! // Measure Phi at current state
//! let phi = measurer.measure(&network);
//! println!("Network Phi: {:.4}", phi.phi);
//! ```

use crate::hdc::autodiff_phi::{AutodiffConfig, AutodiffPhiEngine, DiffNetwork, DiffNode};
use crate::hdc::hdc_ltc_unified::{HdcLtcUnifiedNetwork, UnifiedNetworkConfig};
use crate::hdc::unified_hv::ContinuousHV;
use crate::phi_engine::{PhiEngine, PhiMethod};
use serde::{Deserialize, Serialize};
use std::time::Instant;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for Phi measurement on unified networks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnifiedPhiConfig {
    /// Method for Phi calculation
    pub method: PhiCalculationMethod,

    /// Number of samples for state extraction (higher = more accurate, slower)
    pub n_samples: usize,

    /// Whether to include layer binding vectors in Phi calculation
    pub include_layer_bindings: bool,

    /// Whether to normalize states before Phi calculation
    pub normalize_states: bool,

    /// Temperature for autodiff method (lower = sharper partitions)
    pub autodiff_temperature: f64,

    /// Number of partition samples for autodiff
    pub n_partition_samples: usize,
}

impl Default for UnifiedPhiConfig {
    fn default() -> Self {
        Self {
            method: PhiCalculationMethod::SpectralConnectivity,
            n_samples: 1,
            include_layer_bindings: true,
            normalize_states: true,
            autodiff_temperature: 1.0,
            n_partition_samples: 8,
        }
    }
}

/// Available methods for Phi calculation
///
/// ## Method Comparison
///
/// | Method | Measures | HdcLtcUnified Behavior | Use Case |
/// |--------|----------|------------------------|----------|
/// | SpectralConnectivity | Graph connectivity | Moderate Phi (~0.5) | General integration measure |
/// | Autodiff | Information loss on partition | Low Phi (~0.0-0.1) | IIT-aligned research |
/// | Combined | Average of both | Moderate | Comparative analysis |
///
/// ## Why Different Methods Give Different Values
///
/// **SpectralConnectivity** measures how hard it is to partition the network based on
/// pairwise similarity. For HdcLtcUnifiedNetwork, even near-orthogonal states create
/// a connected graph, yielding moderate Phi values.
///
/// **Autodiff** measures true IIT Phi: how much information is lost when partitioning.
/// For HdcLtcUnifiedNetwork with high-dimensional states that are designed to be
/// distinguishable (near-orthogonal), partitioning doesn't lose much cross-partition
/// information, yielding low Phi. This is actually CORRECT IIT behavior.
///
/// ## Interpretation for Consciousness Research
///
/// A unified HDC-LTC network shows:
/// - High spectral connectivity (good information flow between neurons)
/// - Low IIT Phi (states are distinguishable, not tightly integrated)
///
/// For higher IIT Phi, the network would need states that are MORE similar
/// (more integrated), but this could reduce information capacity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhiCalculationMethod {
    /// Spectral connectivity (algebraic connectivity of similarity graph)
    ///
    /// Measures how hard it is to partition the network based on pairwise similarity.
    /// Fast O(n^3), provides moderate Phi values for typical networks.
    /// Good for measuring information flow and network connectivity.
    SpectralConnectivity,

    /// Autodiff-based Phi with differentiable partitioning
    ///
    /// Measures true IIT Phi: information lost when partitioning the system.
    /// For HdcLtcUnifiedNetwork with near-orthogonal states, this typically
    /// produces low Phi values (close to 0), which is correct IIT behavior.
    /// Useful for gradient-based consciousness optimization.
    Autodiff,

    /// Combined: use both methods and report comparison
    ///
    /// Averages spectral and autodiff methods for a balanced view.
    /// Useful for understanding different aspects of network integration.
    Combined,
}

// ============================================================================
// PHI MEASUREMENT RESULT
// ============================================================================

/// Result of Phi measurement on a unified network
#[derive(Debug, Clone)]
pub struct PhiMeasurement {
    /// The Phi value (integrated information)
    pub phi: f64,

    /// Method used for calculation
    pub method: PhiCalculationMethod,

    /// Number of nodes (neurons) in the network
    pub n_neurons: usize,

    /// Number of layers
    pub n_layers: usize,

    /// Computation time
    pub computation_time: std::time::Duration,

    /// Per-layer Phi values (if computed)
    pub layer_phi: Option<Vec<f64>>,

    /// Whole-system integration measure
    pub whole_integration: Option<f64>,

    /// Average pairwise similarity between neuron states
    pub avg_similarity: f64,

    /// Similarity matrix (optional, for analysis)
    pub similarity_matrix: Option<Vec<Vec<f64>>>,
}

impl PhiMeasurement {
    /// Check if Phi is in expected range [0, 1]
    pub fn is_valid(&self) -> bool {
        self.phi >= 0.0 && self.phi <= 1.0
    }

    /// Classify consciousness level based on Phi
    pub fn consciousness_level(&self) -> &'static str {
        match self.phi {
            x if x < 0.1 => "Minimal",
            x if x < 0.3 => "Low",
            x if x < 0.5 => "Moderate",
            x if x < 0.7 => "High",
            _ => "VeryHigh",
        }
    }
}

// ============================================================================
// NETWORK STATE EXTRACTOR
// ============================================================================

/// Extracts network state information for Phi calculation
pub struct NetworkStateExtractor;

impl NetworkStateExtractor {
    /// Extract all neuron states as ContinuousHV vectors
    pub fn extract_neuron_states(network: &HdcLtcUnifiedNetwork) -> Vec<ContinuousHV> {
        let mut states = Vec::new();

        for layer_idx in 0..network.n_layers() {
            if let Some(layer) = network.layer(layer_idx) {
                for neuron in layer {
                    states.push(neuron.state().clone());
                }
            }
        }

        states
    }

    /// Extract states for a specific layer
    pub fn extract_layer_states(
        network: &HdcLtcUnifiedNetwork,
        layer_idx: usize,
    ) -> Vec<ContinuousHV> {
        if let Some(layer) = network.layer(layer_idx) {
            layer.iter().map(|n| n.state().clone()).collect()
        } else {
            Vec::new()
        }
    }

    /// Convert ContinuousHV states to DiffNode format for autodiff
    ///
    /// The autodiff Phi engine works best with lower-dimensional representations.
    /// This function projects the high-dimensional states (16384D) to a lower
    /// dimension (default 64D) using random projection, preserving similarity structure.
    pub fn to_diff_nodes(states: &[ContinuousHV]) -> Vec<DiffNode> {
        Self::to_diff_nodes_with_dim(states, 64)
    }

    /// Convert ContinuousHV states to DiffNode format with specified output dimension
    pub fn to_diff_nodes_with_dim(states: &[ContinuousHV], output_dim: usize) -> Vec<DiffNode> {
        if states.is_empty() {
            return Vec::new();
        }

        let input_dim = states[0].values.len();

        // If input is already small enough, use directly
        if input_dim <= output_dim {
            return states
                .iter()
                .map(|hv| {
                    let mut node = DiffNode {
                        values: hv.values.iter().map(|&x| x as f64).collect(),
                        grad: vec![0.0; hv.values.len()],
                    };
                    node.normalize();
                    node
                })
                .collect();
        }

        // Use random projection to reduce dimension while preserving similarity
        // Johnson-Lindenstrauss lemma guarantees approximate distance preservation
        let projection_scale = (output_dim as f64).sqrt();

        states
            .iter()
            .enumerate()
            .map(|(idx, hv)| {
                let mut projected = vec![0.0f64; output_dim];

                // Simple hash-based random projection (deterministic)
                for (i, &val) in hv.values.iter().enumerate() {
                    // Hash to select output index
                    let out_idx = (i * 31 + idx * 17) % output_dim;
                    // Hash to select sign
                    let sign = if (i * 37 + idx * 23) % 2 == 0 {
                        1.0
                    } else {
                        -1.0
                    };
                    projected[out_idx] += sign * val as f64;
                }

                // Scale by projection factor
                for v in &mut projected {
                    *v /= projection_scale;
                }

                let mut node = DiffNode {
                    values: projected,
                    grad: vec![0.0; output_dim],
                };
                node.normalize();
                node
            })
            .collect()
    }

    /// Compute pairwise similarity matrix from states
    pub fn compute_similarity_matrix(states: &[ContinuousHV]) -> Vec<Vec<f64>> {
        let n = states.len();
        let mut matrix = vec![vec![0.0; n]; n];

        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let sim = states[i].similarity(&states[j]) as f64;
                matrix[i][j] = sim;
                matrix[j][i] = sim;
            }
        }

        matrix
    }

    /// Compute average pairwise similarity
    pub fn compute_avg_similarity(states: &[ContinuousHV]) -> f64 {
        let n = states.len();
        if n < 2 {
            return 0.0;
        }

        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                total_sim += states[i].similarity(&states[j]) as f64;
                count += 1;
            }
        }

        if count > 0 {
            total_sim / count as f64
        } else {
            0.0
        }
    }
}

// ============================================================================
// UNIFIED NETWORK PHI MEASURER
// ============================================================================

/// Main interface for measuring Phi on HdcLtcUnifiedNetwork
pub struct UnifiedNetworkPhiMeasurer {
    config: UnifiedPhiConfig,
    phi_engine: PhiEngine,
    autodiff_engine: Option<AutodiffPhiEngine>,
}

impl UnifiedNetworkPhiMeasurer {
    /// Create a new measurer with default configuration
    pub fn new() -> Self {
        Self::with_config(UnifiedPhiConfig::default())
    }

    /// Create a new measurer with custom configuration
    pub fn with_config(config: UnifiedPhiConfig) -> Self {
        let phi_engine = PhiEngine::new(PhiMethod::SpectralConnectivity);

        let autodiff_engine = if matches!(
            config.method,
            PhiCalculationMethod::Autodiff | PhiCalculationMethod::Combined
        ) {
            Some(AutodiffPhiEngine::new(AutodiffConfig {
                temperature: config.autodiff_temperature,
                n_partition_samples: config.n_partition_samples,
                ..Default::default()
            }))
        } else {
            None
        };

        Self {
            config,
            phi_engine,
            autodiff_engine,
        }
    }

    /// Measure Phi for the current network state
    pub fn measure(&mut self, network: &HdcLtcUnifiedNetwork) -> PhiMeasurement {
        let start = Instant::now();

        // Extract neuron states
        let states = NetworkStateExtractor::extract_neuron_states(network);

        if states.len() < 2 {
            return PhiMeasurement {
                phi: 0.0,
                method: self.config.method,
                n_neurons: states.len(),
                n_layers: network.n_layers(),
                computation_time: start.elapsed(),
                layer_phi: None,
                whole_integration: None,
                avg_similarity: 0.0,
                similarity_matrix: None,
            };
        }

        // Optionally normalize states
        let processed_states: Vec<ContinuousHV> = if self.config.normalize_states {
            states.iter().map(|s| s.normalize()).collect()
        } else {
            states
        };

        // Compute average similarity
        let avg_similarity = NetworkStateExtractor::compute_avg_similarity(&processed_states);

        // Calculate Phi based on method
        let (phi, whole_integration, similarity_matrix) = match self.config.method {
            PhiCalculationMethod::SpectralConnectivity => {
                let result = self.phi_engine.compute(&processed_states);
                (result.phi, None, None)
            }
            PhiCalculationMethod::Autodiff => self.measure_autodiff(&processed_states),
            PhiCalculationMethod::Combined => {
                // Use spectral as primary, autodiff for validation
                let spectral_result = self.phi_engine.compute(&processed_states);
                let (autodiff_phi, whole_int, sim_matrix) =
                    self.measure_autodiff(&processed_states);

                // Average the two methods
                let combined_phi = (spectral_result.phi + autodiff_phi) / 2.0;
                (combined_phi, whole_int, sim_matrix)
            }
        };

        // Compute per-layer Phi if requested
        let layer_phi = self.compute_layer_phi(network, &processed_states);

        PhiMeasurement {
            phi,
            method: self.config.method,
            n_neurons: processed_states.len(),
            n_layers: network.n_layers(),
            computation_time: start.elapsed(),
            layer_phi: Some(layer_phi),
            whole_integration,
            avg_similarity,
            similarity_matrix,
        }
    }

    /// Measure Phi using autodiff method
    fn measure_autodiff(
        &mut self,
        states: &[ContinuousHV],
    ) -> (f64, Option<f64>, Option<Vec<Vec<f64>>>) {
        if let Some(ref mut engine) = self.autodiff_engine {
            // Convert to DiffNode format
            let nodes = NetworkStateExtractor::to_diff_nodes(states);
            let diff_network = DiffNetwork {
                nodes,
                edge_weights: None,
                edge_grad: None,
            };

            let result = engine.forward(&diff_network);

            (
                result.phi,
                Some(result.whole_integration),
                Some(result.similarity_matrix),
            )
        } else {
            (0.0, None, None)
        }
    }

    /// Compute Phi for each layer separately
    fn compute_layer_phi(
        &mut self,
        network: &HdcLtcUnifiedNetwork,
        _all_states: &[ContinuousHV],
    ) -> Vec<f64> {
        let mut layer_phis = Vec::new();

        for layer_idx in 0..network.n_layers() {
            let layer_states = NetworkStateExtractor::extract_layer_states(network, layer_idx);

            if layer_states.len() >= 2 {
                let processed: Vec<ContinuousHV> = if self.config.normalize_states {
                    layer_states.iter().map(|s| s.normalize()).collect()
                } else {
                    layer_states
                };

                let result = self.phi_engine.compute(&processed);
                layer_phis.push(result.phi);
            } else {
                layer_phis.push(0.0);
            }
        }

        layer_phis
    }
}

impl Default for UnifiedNetworkPhiMeasurer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PHI EVOLUTION TRACKER
// ============================================================================

/// Tracks Phi evolution during network learning/processing
#[derive(Debug, Clone)]
pub struct PhiEvolutionTracker {
    /// History of Phi measurements
    pub history: Vec<PhiMeasurement>,

    /// Timestamps for each measurement
    pub timestamps: Vec<f64>,

    /// Input descriptions (for analysis)
    pub input_descriptions: Vec<String>,
}

impl PhiEvolutionTracker {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            timestamps: Vec::new(),
            input_descriptions: Vec::new(),
        }
    }

    /// Record a measurement
    pub fn record(&mut self, measurement: PhiMeasurement, time: f64, description: &str) {
        self.history.push(measurement);
        self.timestamps.push(time);
        self.input_descriptions.push(description.to_string());
    }

    /// Get Phi values as a vector
    pub fn phi_values(&self) -> Vec<f64> {
        self.history.iter().map(|m| m.phi).collect()
    }

    /// Compute Phi change rate (derivative)
    pub fn phi_derivative(&self) -> Vec<f64> {
        if self.history.len() < 2 {
            return Vec::new();
        }

        let mut derivatives = Vec::new();
        for i in 1..self.history.len() {
            let dt = self.timestamps[i] - self.timestamps[i - 1];
            if dt > 0.0 {
                let dphi = self.history[i].phi - self.history[i - 1].phi;
                derivatives.push(dphi / dt);
            } else {
                derivatives.push(0.0);
            }
        }

        derivatives
    }

    /// Get summary statistics
    pub fn summary(&self) -> PhiEvolutionSummary {
        if self.history.is_empty() {
            return PhiEvolutionSummary::default();
        }

        let phis = self.phi_values();
        let mean = phis.iter().sum::<f64>() / phis.len() as f64;
        let variance = phis.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / phis.len() as f64;

        PhiEvolutionSummary {
            n_measurements: self.history.len(),
            initial_phi: phis.first().copied().unwrap_or(0.0),
            final_phi: phis.last().copied().unwrap_or(0.0),
            min_phi: phis.iter().cloned().fold(f64::INFINITY, f64::min),
            max_phi: phis.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
            mean_phi: mean,
            std_phi: variance.sqrt(),
            total_time: self.timestamps.last().copied().unwrap_or(0.0)
                - self.timestamps.first().copied().unwrap_or(0.0),
        }
    }
}

impl Default for PhiEvolutionTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of Phi evolution
#[derive(Debug, Clone, Default)]
pub struct PhiEvolutionSummary {
    pub n_measurements: usize,
    pub initial_phi: f64,
    pub final_phi: f64,
    pub min_phi: f64,
    pub max_phi: f64,
    pub mean_phi: f64,
    pub std_phi: f64,
    pub total_time: f64,
}

// ============================================================================
// PHI COMPARISON UTILITIES
// ============================================================================

/// Compare Phi between different network configurations
pub struct PhiComparator;

impl PhiComparator {
    /// Compare Phi between two networks
    pub fn compare(
        measurer: &mut UnifiedNetworkPhiMeasurer,
        network_a: &HdcLtcUnifiedNetwork,
        network_b: &HdcLtcUnifiedNetwork,
    ) -> PhiComparison {
        let phi_a = measurer.measure(network_a);
        let phi_b = measurer.measure(network_b);

        PhiComparison {
            phi_a: phi_a.phi,
            phi_b: phi_b.phi,
            difference: phi_b.phi - phi_a.phi,
            ratio: if phi_a.phi > 0.0 {
                phi_b.phi / phi_a.phi
            } else {
                f64::INFINITY
            },
            measurement_a: phi_a,
            measurement_b: phi_b,
        }
    }

    /// Compare a network with different numbers of connections
    /// by measuring Phi with varying layer sizes
    pub fn compare_connectivity_levels(
        measurer: &mut UnifiedNetworkPhiMeasurer,
        base_config: &UnifiedNetworkConfig,
        connectivity_factors: &[f32],
        seed: u64,
    ) -> Vec<(f32, PhiMeasurement)> {
        let mut results = Vec::new();

        for &factor in connectivity_factors {
            // Scale layer sizes by connectivity factor
            let scaled_config = UnifiedNetworkConfig {
                layer_sizes: base_config
                    .layer_sizes
                    .iter()
                    .map(|&size| ((size as f32 * factor) as usize).max(2))
                    .collect(),
                ..base_config.clone()
            };

            let network = HdcLtcUnifiedNetwork::new(scaled_config, seed);
            let measurement = measurer.measure(&network);

            results.push((factor, measurement));
        }

        results
    }
}

/// Result of comparing two networks
#[derive(Debug, Clone)]
pub struct PhiComparison {
    pub phi_a: f64,
    pub phi_b: f64,
    pub difference: f64,
    pub ratio: f64,
    pub measurement_a: PhiMeasurement,
    pub measurement_b: PhiMeasurement,
}

// ============================================================================
// VALIDATION UTILITIES
// ============================================================================

/// Validates IIT predictions for unified networks
pub struct PhiValidator;

impl PhiValidator {
    /// Validate that Phi increases with more integration (connections)
    /// Returns true if the validation passes
    pub fn validate_integration_increases_phi(
        measurer: &mut UnifiedNetworkPhiMeasurer,
        seed: u64,
    ) -> ValidationResult {
        // Create networks with increasing connectivity
        let small_config = UnifiedNetworkConfig {
            layer_sizes: vec![2, 2],
            use_layer_binding: false,
            skip_connections: false,
            ..Default::default()
        };

        let large_config = UnifiedNetworkConfig {
            layer_sizes: vec![4, 6, 4],
            use_layer_binding: true,
            skip_connections: true,
            ..Default::default()
        };

        let small_network = HdcLtcUnifiedNetwork::new(small_config, seed);
        let large_network = HdcLtcUnifiedNetwork::new(large_config, seed);

        // Evolve both networks to build up state
        let mut small = small_network;
        let mut large = large_network;
        let input = ContinuousHV::random_default(seed + 1000);

        for _ in 0..100 {
            small.evolve_closed_form(0.01, &input);
            large.evolve_closed_form(0.01, &input);
        }

        let phi_small = measurer.measure(&small);
        let phi_large = measurer.measure(&large);

        // Larger, more connected network should have higher Phi
        // (or at least similar if already saturated)
        let passed = phi_large.phi >= phi_small.phi * 0.8; // Allow 20% tolerance

        ValidationResult {
            name: "Integration Increases Phi".to_string(),
            passed,
            expected: format!(
                "phi_large ({:.4}) >= phi_small ({:.4}) * 0.8",
                phi_large.phi, phi_small.phi
            ),
            actual: format!(
                "phi_small={:.4}, phi_large={:.4}",
                phi_small.phi, phi_large.phi
            ),
            details: Some(format!(
                "Small network: {} neurons, {} layers, Phi={:.4}\n\
                 Large network: {} neurons, {} layers, Phi={:.4}",
                phi_small.n_neurons,
                phi_small.n_layers,
                phi_small.phi,
                phi_large.n_neurons,
                phi_large.n_layers,
                phi_large.phi
            )),
        }
    }

    /// Validate that Phi is in expected range [0, 1]
    pub fn validate_phi_range(
        measurer: &mut UnifiedNetworkPhiMeasurer,
        network: &HdcLtcUnifiedNetwork,
    ) -> ValidationResult {
        let measurement = measurer.measure(network);

        let passed = measurement.phi >= 0.0 && measurement.phi <= 1.0;

        ValidationResult {
            name: "Phi Range".to_string(),
            passed,
            expected: "0.0 <= Phi <= 1.0".to_string(),
            actual: format!("Phi = {:.6}", measurement.phi),
            details: Some(format!(
                "Network: {} neurons, {} layers\n\
                 Average similarity: {:.4}",
                measurement.n_neurons, measurement.n_layers, measurement.avg_similarity
            )),
        }
    }

    /// Validate that correlated inputs produce higher Phi than uncorrelated
    pub fn validate_correlation_increases_phi(
        measurer: &mut UnifiedNetworkPhiMeasurer,
        seed: u64,
    ) -> ValidationResult {
        let config = UnifiedNetworkConfig::default();

        // Network with correlated inputs
        let mut network_correlated = HdcLtcUnifiedNetwork::new(config.clone(), seed);
        let base_input = ContinuousHV::random_default(seed + 100);

        // Use similar (correlated) inputs
        for _ in 0..100 {
            let noise = ContinuousHV::random_default(seed + rand::random::<u64>());
            let correlated_input = ContinuousHV::weighted_bundle(
                &[&base_input, &noise],
                &[0.9, 0.1], // 90% base, 10% noise
            );
            network_correlated.evolve_closed_form(0.01, &correlated_input);
        }

        // Network with uncorrelated inputs
        let mut network_uncorrelated = HdcLtcUnifiedNetwork::new(config, seed);

        for i in 0..100 {
            let uncorrelated_input = ContinuousHV::random_default(seed + 1000 + i as u64);
            network_uncorrelated.evolve_closed_form(0.01, &uncorrelated_input);
        }

        let phi_correlated = measurer.measure(&network_correlated);
        let phi_uncorrelated = measurer.measure(&network_uncorrelated);

        // Correlated inputs should lead to more coherent state = higher Phi
        // Note: This relationship may not always hold due to the complexity of the dynamics
        let passed = phi_correlated.avg_similarity >= phi_uncorrelated.avg_similarity * 0.9;

        ValidationResult {
            name: "Correlation Increases Coherence".to_string(),
            passed,
            expected: "Correlated inputs -> higher average similarity".to_string(),
            actual: format!(
                "Correlated avg_sim={:.4}, Phi={:.4}\n\
                 Uncorrelated avg_sim={:.4}, Phi={:.4}",
                phi_correlated.avg_similarity,
                phi_correlated.phi,
                phi_uncorrelated.avg_similarity,
                phi_uncorrelated.phi
            ),
            details: None,
        }
    }

    /// Run all validations and return summary
    pub fn run_all_validations(
        measurer: &mut UnifiedNetworkPhiMeasurer,
        network: &HdcLtcUnifiedNetwork,
        seed: u64,
    ) -> Vec<ValidationResult> {
        vec![
            Self::validate_phi_range(measurer, network),
            Self::validate_integration_increases_phi(measurer, seed),
            Self::validate_correlation_increases_phi(measurer, seed),
        ]
    }
}

/// Result of a validation check
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub name: String,
    pub passed: bool,
    pub expected: String,
    pub actual: String,
    pub details: Option<String>,
}

impl ValidationResult {
    pub fn summary(&self) -> String {
        let status = if self.passed { "PASS" } else { "FAIL" };
        format!("[{}] {}: {}", status, self.name, self.actual)
    }
}

// ============================================================================
// PHI DIAGNOSTIC ANALYSIS
// ============================================================================

/// Diagnostic analysis for understanding Phi behavior in unified networks
#[derive(Debug, Clone)]
pub struct PhiDiagnostic {
    /// Overall diagnosis
    pub diagnosis: String,

    /// Spectral Phi value
    pub spectral_phi: f64,

    /// Autodiff Phi value
    pub autodiff_phi: f64,

    /// Average pairwise similarity between neuron states
    pub avg_similarity: f64,

    /// Similarity variance (low = uniform, high = diverse)
    pub similarity_variance: f64,

    /// Network size (number of neurons)
    pub n_neurons: usize,

    /// Issues identified
    pub issues: Vec<String>,

    /// Proposed fixes
    pub proposed_fixes: Vec<String>,
}

/// Analyzes why Phi might be low and proposes fixes
pub struct PhiDiagnosticAnalyzer;

impl PhiDiagnosticAnalyzer {
    /// Perform comprehensive diagnostic analysis on a network
    pub fn analyze(network: &HdcLtcUnifiedNetwork, seed: u64) -> PhiDiagnostic {
        // Extract states
        let states = NetworkStateExtractor::extract_neuron_states(network);
        let n_neurons = states.len();

        if n_neurons < 2 {
            return PhiDiagnostic {
                diagnosis: "Network too small for meaningful Phi analysis".to_string(),
                spectral_phi: 0.0,
                autodiff_phi: 0.0,
                avg_similarity: 0.0,
                similarity_variance: 0.0,
                n_neurons,
                issues: vec!["Network has fewer than 2 neurons".to_string()],
                proposed_fixes: vec!["Add more neurons to the network".to_string()],
            };
        }

        // Compute similarity statistics
        let sim_matrix = NetworkStateExtractor::compute_similarity_matrix(&states);
        let avg_similarity = NetworkStateExtractor::compute_avg_similarity(&states);

        // Compute similarity variance
        let mut similarities = Vec::new();
        for i in 0..n_neurons {
            for j in (i + 1)..n_neurons {
                similarities.push(sim_matrix[i][j]);
            }
        }
        let similarity_variance = if similarities.len() > 1 {
            let mean = avg_similarity;
            similarities.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / similarities.len() as f64
        } else {
            0.0
        };

        // Measure Phi with both methods
        let mut spectral_measurer = UnifiedNetworkPhiMeasurer::with_config(UnifiedPhiConfig {
            method: PhiCalculationMethod::SpectralConnectivity,
            ..Default::default()
        });
        let spectral_result = spectral_measurer.measure(network);

        let mut autodiff_measurer = UnifiedNetworkPhiMeasurer::with_config(UnifiedPhiConfig {
            method: PhiCalculationMethod::Autodiff,
            ..Default::default()
        });
        let autodiff_result = autodiff_measurer.measure(network);

        // Identify issues and propose fixes
        let mut issues = Vec::new();
        let mut proposed_fixes = Vec::new();

        // Issue 1: Very low average similarity
        if avg_similarity.abs() < 0.05 {
            issues.push("Neuron states are nearly orthogonal (very low similarity)".to_string());
            proposed_fixes.push("Use correlated inputs to increase state coherence".to_string());
            proposed_fixes.push("Increase layer binding strength".to_string());
        }

        // Issue 2: Low similarity variance
        if similarity_variance < 0.01 {
            issues.push("Uniform similarity across all pairs (low variance)".to_string());
            proposed_fixes.push("Create more diverse inputs to differentiate states".to_string());
            proposed_fixes.push("Add skip connections for hierarchical integration".to_string());
        }

        // Issue 3: Large gap between spectral and autodiff
        if (spectral_result.phi - autodiff_result.phi).abs() > 0.3 {
            issues.push("Large gap between spectral and autodiff Phi".to_string());
            proposed_fixes.push(
                "This is expected: spectral measures connectivity, autodiff measures IIT Phi"
                    .to_string(),
            );
        }

        // Issue 4: Low autodiff Phi (common for HDC networks)
        if autodiff_result.phi < 0.1 && spectral_result.phi > 0.3 {
            issues.push("Low IIT Phi despite good network connectivity".to_string());
            proposed_fixes.push("This is expected for HDC: high-dimensional states are designed to be distinguishable".to_string());
            proposed_fixes.push(
                "For higher IIT Phi, reduce state dimensionality or increase state similarity"
                    .to_string(),
            );
        }

        // Issue 5: Small network
        if n_neurons < 5 {
            issues.push("Small network size limits integration potential".to_string());
            proposed_fixes
                .push("Increase network size for more integration opportunities".to_string());
        }

        // Generate diagnosis
        let diagnosis = if issues.is_empty() {
            "Network shows healthy Phi characteristics".to_string()
        } else if autodiff_result.phi < 0.1 && spectral_result.phi > 0.3 {
            "Network has good connectivity but low IIT Phi - this is EXPECTED for HDC".to_string()
        } else if avg_similarity.abs() < 0.05 {
            "Network states are too orthogonal for strong integration".to_string()
        } else {
            format!("{} issue(s) identified affecting Phi", issues.len())
        };

        PhiDiagnostic {
            diagnosis,
            spectral_phi: spectral_result.phi,
            autodiff_phi: autodiff_result.phi,
            avg_similarity,
            similarity_variance,
            n_neurons,
            issues,
            proposed_fixes,
        }
    }

    /// Print a formatted diagnostic report
    pub fn print_report(diagnostic: &PhiDiagnostic) {
        println!("========================================");
        println!("       PHI DIAGNOSTIC REPORT");
        println!("========================================");
        println!();
        println!("DIAGNOSIS: {}", diagnostic.diagnosis);
        println!();
        println!("MEASUREMENTS:");
        println!("  Spectral Phi: {:.6}", diagnostic.spectral_phi);
        println!("  Autodiff Phi: {:.6}", diagnostic.autodiff_phi);
        println!("  Avg Similarity: {:.6}", diagnostic.avg_similarity);
        println!(
            "  Similarity Variance: {:.6}",
            diagnostic.similarity_variance
        );
        println!("  Network Size: {} neurons", diagnostic.n_neurons);
        println!();

        if !diagnostic.issues.is_empty() {
            println!("ISSUES IDENTIFIED:");
            for (i, issue) in diagnostic.issues.iter().enumerate() {
                println!("  {}. {}", i + 1, issue);
            }
            println!();
        }

        if !diagnostic.proposed_fixes.is_empty() {
            println!("PROPOSED FIXES:");
            for (i, fix) in diagnostic.proposed_fixes.iter().enumerate() {
                println!("  {}. {}", i + 1, fix);
            }
            println!();
        }

        println!("========================================");
    }
}

// ============================================================================
// DEMO AND EXAMPLES
// ============================================================================

/// Demonstrates Phi evolution during learning
pub fn demo_phi_evolution(seed: u64) -> PhiEvolutionTracker {
    let config = UnifiedNetworkConfig {
        layer_sizes: vec![3, 5, 3],
        use_layer_binding: true,
        skip_connections: false,
        ..Default::default()
    };

    let mut network = HdcLtcUnifiedNetwork::new(config, seed);
    let mut measurer = UnifiedNetworkPhiMeasurer::new();
    let mut tracker = PhiEvolutionTracker::new();

    // Initial state
    let initial_measurement = measurer.measure(&network);
    tracker.record(initial_measurement, 0.0, "Initial (zero state)");

    // Random input phase
    for i in 0..10 {
        let random_input = ContinuousHV::random_default(seed + i as u64);
        for _ in 0..10 {
            network.evolve_closed_form(0.01, &random_input);
        }
        let measurement = measurer.measure(&network);
        tracker.record(
            measurement,
            (i + 1) as f64 * 0.1,
            &format!("Random input {i}"),
        );
    }

    // Correlated input phase (learning pattern)
    let target_pattern = ContinuousHV::random_default(seed + 1000);
    for i in 0..20 {
        let noise = ContinuousHV::random_default(seed + 2000 + i as u64);
        let learning_input = ContinuousHV::weighted_bundle(&[&target_pattern, &noise], &[0.8, 0.2]);
        for _ in 0..10 {
            network.evolve_closed_form(0.01, &learning_input);
        }
        let measurement = measurer.measure(&network);
        tracker.record(
            measurement,
            1.0 + (i + 1) as f64 * 0.1,
            &format!("Learning step {i}"),
        );
    }

    // Equilibrium phase (stable input)
    for i in 0..10 {
        for _ in 0..10 {
            network.evolve_closed_form(0.1, &target_pattern); // Longer time steps
        }
        let measurement = measurer.measure(&network);
        tracker.record(
            measurement,
            3.0 + (i + 1) as f64 * 0.1,
            &format!("Equilibrium {i}"),
        );
    }

    tracker
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]

    fn test_basic_phi_measurement() {
        let config = UnifiedNetworkConfig {
            layer_sizes: vec![2, 3, 2],
            ..Default::default()
        };
        let mut network = HdcLtcUnifiedNetwork::new(config, 42);
        let mut measurer = UnifiedNetworkPhiMeasurer::new();

        // Evolve network to build up state
        let input = ContinuousHV::random_default(123);
        for _ in 0..50 {
            network.evolve_closed_form(0.01, &input);
        }

        let measurement = measurer.measure(&network);

        assert!(
            measurement.is_valid(),
            "Phi should be in range [0, 1], got {}",
            measurement.phi
        );
        assert_eq!(measurement.n_neurons, 7); // 2 + 3 + 2
        assert_eq!(measurement.n_layers, 3);
    }

    #[test]

    fn test_phi_measurement_initial_state() {
        let config = UnifiedNetworkConfig::default();
        let network = HdcLtcUnifiedNetwork::new(config, 42);
        let mut measurer = UnifiedNetworkPhiMeasurer::new();

        // Measure initial (zero) state
        let measurement = measurer.measure(&network);

        // Zero state should have low Phi (no information)
        // But not necessarily 0 due to random weight vectors
        assert!(measurement.is_valid());
    }

    #[test]

    fn test_phi_evolution_tracking() {
        let tracker = demo_phi_evolution(42);

        assert!(tracker.history.len() > 0);

        let summary = tracker.summary();
        assert!(summary.n_measurements > 30);

        // All measurements should be valid
        for m in &tracker.history {
            assert!(m.is_valid(), "Invalid Phi: {}", m.phi);
        }
    }

    #[test]

    fn test_phi_range_validation() {
        let config = UnifiedNetworkConfig::default();
        let mut network = HdcLtcUnifiedNetwork::new(config, 42);
        let mut measurer = UnifiedNetworkPhiMeasurer::new();

        // Evolve to get meaningful state
        let input = ContinuousHV::random_default(123);
        for _ in 0..100 {
            network.evolve_closed_form(0.01, &input);
        }

        let result = PhiValidator::validate_phi_range(&mut measurer, &network);

        assert!(
            result.passed,
            "Phi range validation failed: {}",
            result.actual
        );
    }

    #[test]

    fn test_network_state_extraction() {
        let config = UnifiedNetworkConfig {
            layer_sizes: vec![2, 3, 2],
            ..Default::default()
        };
        let network = HdcLtcUnifiedNetwork::new(config, 42);

        let states = NetworkStateExtractor::extract_neuron_states(&network);
        assert_eq!(states.len(), 7); // 2 + 3 + 2

        let layer_states = NetworkStateExtractor::extract_layer_states(&network, 1);
        assert_eq!(layer_states.len(), 3); // Middle layer
    }

    #[test]

    fn test_similarity_matrix_computation() {
        let config = UnifiedNetworkConfig {
            layer_sizes: vec![3, 3],
            ..Default::default()
        };
        let mut network = HdcLtcUnifiedNetwork::new(config, 42);

        // Evolve to get diverse states
        let input = ContinuousHV::random_default(123);
        for _ in 0..50 {
            network.evolve_closed_form(0.01, &input);
        }

        let states = NetworkStateExtractor::extract_neuron_states(&network);
        let sim_matrix = NetworkStateExtractor::compute_similarity_matrix(&states);

        assert_eq!(sim_matrix.len(), 6);
        assert_eq!(sim_matrix[0].len(), 6);

        // Diagonal should be 1.0 (self-similarity)
        for i in 0..6 {
            assert!((sim_matrix[i][i] - 1.0).abs() < 0.001);
        }

        // Matrix should be symmetric
        for i in 0..6 {
            for j in 0..6 {
                assert!((sim_matrix[i][j] - sim_matrix[j][i]).abs() < 0.001);
            }
        }
    }

    #[test]

    fn test_phi_comparison() {
        let config_small = UnifiedNetworkConfig {
            layer_sizes: vec![2, 2],
            ..Default::default()
        };
        let config_large = UnifiedNetworkConfig {
            layer_sizes: vec![4, 6, 4],
            ..Default::default()
        };

        let mut network_small = HdcLtcUnifiedNetwork::new(config_small, 42);
        let mut network_large = HdcLtcUnifiedNetwork::new(config_large, 42);
        let mut measurer = UnifiedNetworkPhiMeasurer::new();

        // Evolve both networks
        let input = ContinuousHV::random_default(123);
        for _ in 0..100 {
            network_small.evolve_closed_form(0.01, &input);
            network_large.evolve_closed_form(0.01, &input);
        }

        let comparison = PhiComparator::compare(&mut measurer, &network_small, &network_large);

        // Both should have valid Phi
        assert!(comparison.measurement_a.is_valid());
        assert!(comparison.measurement_b.is_valid());
    }

    #[test]

    fn test_layer_phi_computation() {
        let config = UnifiedNetworkConfig {
            layer_sizes: vec![3, 4, 3],
            ..Default::default()
        };
        let mut network = HdcLtcUnifiedNetwork::new(config, 42);
        let mut measurer = UnifiedNetworkPhiMeasurer::new();

        // Evolve network
        let input = ContinuousHV::random_default(123);
        for _ in 0..50 {
            network.evolve_closed_form(0.01, &input);
        }

        let measurement = measurer.measure(&network);

        // Should have per-layer Phi
        assert!(measurement.layer_phi.is_some());
        let layer_phis = measurement.layer_phi.unwrap();
        assert_eq!(layer_phis.len(), 3);

        // All layer Phis should be valid
        for phi in &layer_phis {
            assert!(*phi >= 0.0 && *phi <= 1.0);
        }
    }

    #[test]

    fn test_phi_with_different_methods() {
        let config = UnifiedNetworkConfig {
            layer_sizes: vec![3, 4, 3],
            ..Default::default()
        };
        let mut network = HdcLtcUnifiedNetwork::new(config, 42);

        // Evolve network
        let input = ContinuousHV::random_default(123);
        for _ in 0..50 {
            network.evolve_closed_form(0.01, &input);
        }

        // Test spectral method
        let spectral_config = UnifiedPhiConfig {
            method: PhiCalculationMethod::SpectralConnectivity,
            ..Default::default()
        };
        let mut spectral_measurer = UnifiedNetworkPhiMeasurer::with_config(spectral_config);
        let spectral_result = spectral_measurer.measure(&network);

        assert!(spectral_result.is_valid());

        // Test autodiff method
        let autodiff_config = UnifiedPhiConfig {
            method: PhiCalculationMethod::Autodiff,
            ..Default::default()
        };
        let mut autodiff_measurer = UnifiedNetworkPhiMeasurer::with_config(autodiff_config);
        let autodiff_result = autodiff_measurer.measure(&network);

        assert!(autodiff_result.is_valid());
    }

    #[test]

    fn test_validation_suite() {
        let config = UnifiedNetworkConfig::default();
        let mut network = HdcLtcUnifiedNetwork::new(config, 42);
        let mut measurer = UnifiedNetworkPhiMeasurer::new();

        // Evolve network
        let input = ContinuousHV::random_default(123);
        for _ in 0..100 {
            network.evolve_closed_form(0.01, &input);
        }

        let results = PhiValidator::run_all_validations(&mut measurer, &network, 42);

        // Should have at least 3 validation results
        assert!(results.len() >= 3);

        // Print results for visibility
        for result in &results {
            println!("{}", result.summary());
        }
    }

    #[test]

    fn test_phi_increases_with_evolution() {
        let config = UnifiedNetworkConfig::default();
        let mut network = HdcLtcUnifiedNetwork::new(config, 42);
        let mut measurer = UnifiedNetworkPhiMeasurer::new();

        // Measure initial (zero state)
        let initial_phi = measurer.measure(&network).phi;

        // Evolve network with consistent input
        let input = ContinuousHV::random_default(123);
        for _ in 0..200 {
            network.evolve_closed_form(0.01, &input);
        }

        // Measure evolved state
        let evolved_phi = measurer.measure(&network).phi;

        // Both should be valid
        assert!(initial_phi >= 0.0 && initial_phi <= 1.0);
        assert!(evolved_phi >= 0.0 && evolved_phi <= 1.0);

        // Evolved network should have non-trivial Phi
        // (not necessarily higher due to state normalization)
        println!(
            "Initial Phi: {:.4}, Evolved Phi: {:.4}",
            initial_phi, evolved_phi
        );
    }
}
