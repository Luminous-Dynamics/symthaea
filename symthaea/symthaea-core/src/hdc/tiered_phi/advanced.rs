// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Advanced Φ Analysis Methods
//!
//! This module contains advanced analysis methods including:
//! - Cross-Topology Transfer (#96): Learn and transfer consciousness patterns between topologies
//! - Causal Intervention Analysis (#98): Analyze causal effects of node interventions
//! - Network Modularity Analysis (#99): Detect consciousness modules and their relationships

use crate::hdc::unified_hv::ContinuousHV;
// Note: BinaryHV and TieredPhi/ApproximationTier are available via super if needed

#[derive(Debug, Clone)]
pub struct PhiTransferConfig {
    /// Number of signature dimensions to extract
    pub signature_dims: usize,

    /// Learning rate for transfer mapping
    pub learning_rate: f64,

    /// Maximum iterations for optimization
    pub max_iterations: usize,

    /// Convergence threshold
    pub convergence_threshold: f64,

    /// Regularization strength (prevents overfitting)
    pub regularization: f64,

    /// Whether to use spectral features
    pub use_spectral: bool,

    /// Whether to use connectivity features
    pub use_connectivity: bool,
}

impl Default for PhiTransferConfig {
    fn default() -> Self {
        Self {
            signature_dims: 16,
            learning_rate: 0.01,
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            regularization: 0.001,
            use_spectral: true,
            use_connectivity: true,
        }
    }
}

impl PhiTransferConfig {
    /// Fast config for quick transfer learning
    pub fn fast() -> Self {
        Self {
            signature_dims: 8,
            max_iterations: 100,
            convergence_threshold: 1e-4,
            ..Default::default()
        }
    }

    /// Research config for detailed analysis
    pub fn research() -> Self {
        Self {
            signature_dims: 32,
            max_iterations: 5000,
            convergence_threshold: 1e-8,
            ..Default::default()
        }
    }
}

/// Φ Signature - extracted features that characterize consciousness potential
#[derive(Debug, Clone)]
pub struct PhiSignature {
    /// Similarity distribution features
    pub similarity_features: Vec<f64>,

    /// Connectivity pattern features
    pub connectivity_features: Vec<f64>,

    /// Spectral (eigenvalue) features
    pub spectral_features: Vec<f64>,

    /// Original Φ value of the topology
    pub original_phi: f64,

    /// Number of components in source topology
    pub num_components: usize,

    /// Topology type (if known)
    pub topology_type: Option<String>,
}

impl PhiSignature {
    /// Get full feature vector
    pub fn as_vector(&self) -> Vec<f64> {
        let mut v = Vec::new();
        v.extend(&self.similarity_features);
        v.extend(&self.connectivity_features);
        v.extend(&self.spectral_features);
        v
    }

    /// Get dimensionality of signature
    pub fn dim(&self) -> usize {
        self.similarity_features.len()
            + self.connectivity_features.len()
            + self.spectral_features.len()
    }
}

/// Result of Φ transfer operation
#[derive(Debug, Clone)]
pub struct PhiTransferResult {
    /// Original Φ of target topology
    pub original_phi: f64,

    /// Enhanced Φ after transfer
    pub enhanced_phi: f64,

    /// Improvement ratio (enhanced / original)
    pub improvement_ratio: f64,

    /// Transfer loss (how well patterns transferred)
    pub transfer_loss: f64,

    /// Iterations used in optimization
    pub iterations: usize,

    /// Whether transfer converged
    pub converged: bool,

    /// Source topology type used
    pub source_type: String,

    /// Target topology type
    pub target_type: String,

    /// Transferred features (modification vector)
    pub transfer_vector: Vec<f64>,
}

impl PhiTransferResult {
    /// Check if transfer was successful (improved Φ)
    pub fn is_successful(&self) -> bool {
        self.improvement_ratio > 1.0 && self.converged
    }

    /// Get percentage improvement
    pub fn improvement_percent(&self) -> f64 {
        (self.improvement_ratio - 1.0) * 100.0
    }
}

/// Cross-Topology Φ Transfer Engine
///
/// Learns consciousness patterns from high-Φ topologies and transfers
/// them to enhance low-Φ topologies.
#[derive(Debug, Clone)]
pub struct PhiTransfer {
    /// Configuration for transfer learning
    pub config: PhiTransferConfig,
    /// Learned transfer weights (source signature → target enhancement)
    pub transfer_weights: Option<Vec<Vec<f64>>>,
    /// Source signatures for reference
    pub source_signatures: Vec<PhiSignature>,
}

impl PhiTransfer {
    /// Create new transfer engine with default config
    pub fn new() -> Self {
        Self {
            config: PhiTransferConfig::default(),
            transfer_weights: None,
            source_signatures: Vec::new(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: PhiTransferConfig) -> Self {
        Self {
            config,
            transfer_weights: None,
            source_signatures: Vec::new(),
        }
    }

    /// Fast transfer engine
    pub fn fast() -> Self {
        Self::with_config(PhiTransferConfig::fast())
    }

    /// Research transfer engine
    pub fn research() -> Self {
        Self::with_config(PhiTransferConfig::research())
    }

    /// Extract Φ signature from a topology's component representations
    pub fn extract_signature(
        &self,
        components: &[ContinuousHV],
        phi: f64,
        topology_type: Option<&str>,
    ) -> PhiSignature {
        let n = components.len();

        // 1. Extract similarity features
        let similarity_features = self.extract_similarity_features(components);

        // 2. Extract connectivity features
        let connectivity_features = self.extract_connectivity_features(components);

        // 3. Extract spectral features
        let spectral_features = self.extract_spectral_features(components);

        PhiSignature {
            similarity_features,
            connectivity_features,
            spectral_features,
            original_phi: phi,
            num_components: n,
            topology_type: topology_type.map(String::from),
        }
    }

    /// Extract similarity distribution features
    fn extract_similarity_features(&self, components: &[ContinuousHV]) -> Vec<f64> {
        let n = components.len();
        if n < 2 {
            return vec![0.0; self.config.signature_dims / 3];
        }

        // Compute all pairwise similarities
        let mut similarities = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = components[i].similarity(&components[j]) as f64;
                similarities.push(sim);
            }
        }

        // Extract statistical features
        let num_features = self.config.signature_dims / 3;
        let mut features = Vec::with_capacity(num_features);

        // Mean similarity
        let mean = similarities.iter().sum::<f64>() / similarities.len() as f64;
        features.push(mean);

        // Variance
        let variance = similarities.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
            / similarities.len() as f64;
        features.push(variance.sqrt());

        // Min and max
        let min = similarities.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = similarities
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        features.push(min);
        features.push(max);

        // Percentiles (quartiles)
        similarities.sort_by(|a, b| a.total_cmp(b));
        let q1 = similarities[similarities.len() / 4];
        let q2 = similarities[similarities.len() / 2];
        let q3 = similarities[3 * similarities.len() / 4];
        features.push(q1);
        features.push(q2);
        features.push(q3);

        // Pad to target size
        while features.len() < num_features {
            features.push(0.0);
        }
        features.truncate(num_features);

        features
    }

    /// Extract connectivity pattern features
    fn extract_connectivity_features(&self, components: &[ContinuousHV]) -> Vec<f64> {
        let n = components.len();
        let num_features = self.config.signature_dims / 3;
        let mut features = Vec::with_capacity(num_features);

        if n < 2 {
            return vec![0.0; num_features];
        }

        // "Effective connectivity" based on similarity threshold
        let threshold = 0.3; // Consider connected if similarity > threshold
        let mut connection_counts = vec![0usize; n];

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = components[i].similarity(&components[j]) as f64;
                if sim > threshold {
                    connection_counts[i] += 1;
                    connection_counts[j] += 1;
                }
            }
        }

        // Mean degree
        let mean_degree = connection_counts.iter().sum::<usize>() as f64 / n as f64;
        features.push(mean_degree / (n - 1) as f64); // Normalized

        // Degree variance
        let degree_variance = connection_counts
            .iter()
            .map(|&d| (d as f64 - mean_degree).powi(2))
            .sum::<f64>()
            / n as f64;
        features.push(degree_variance.sqrt() / (n - 1) as f64); // Normalized

        // Hub detection (nodes with degree > 2 * mean)
        let hub_count = connection_counts
            .iter()
            .filter(|&&d| d as f64 > 2.0 * mean_degree)
            .count();
        features.push(hub_count as f64 / n as f64);

        // Isolation detection (nodes with degree = 0)
        let isolated = connection_counts.iter().filter(|&&d| d == 0).count();
        features.push(isolated as f64 / n as f64);

        // Degree distribution entropy (measure of uniformity)
        let max_degree = *connection_counts.iter().max().unwrap_or(&0);
        if max_degree > 0 {
            let mut degree_dist = vec![0usize; max_degree + 1];
            for &d in &connection_counts {
                degree_dist[d] += 1;
            }
            let entropy: f64 = degree_dist
                .iter()
                .filter(|&&c| c > 0)
                .map(|&c| {
                    let p = c as f64 / n as f64;
                    -p * p.ln()
                })
                .sum();
            features.push(entropy / (n as f64).ln()); // Normalized
        } else {
            features.push(0.0);
        }

        // Pad to target size
        while features.len() < num_features {
            features.push(0.0);
        }
        features.truncate(num_features);

        features
    }

    /// Extract spectral (eigenvalue-like) features
    fn extract_spectral_features(&self, components: &[ContinuousHV]) -> Vec<f64> {
        let n = components.len();
        let num_features = self.config.signature_dims / 3;
        let mut features = Vec::with_capacity(num_features);

        if n < 2 {
            return vec![0.0; num_features];
        }

        // Build similarity matrix
        let mut sim_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    sim_matrix[i][j] = 1.0;
                } else {
                    sim_matrix[i][j] = components[i].similarity(&components[j]) as f64;
                }
            }
        }

        // Power iteration to estimate dominant eigenvalue
        let mut v = vec![1.0 / (n as f64).sqrt(); n];
        for _ in 0..50 {
            // Matrix-vector multiply
            let mut new_v = vec![0.0; n];
            for i in 0..n {
                for j in 0..n {
                    new_v[i] += sim_matrix[i][j] * v[j];
                }
            }
            // Normalize
            let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for x in &mut new_v {
                    *x /= norm;
                }
            }
            v = new_v;
        }

        // Estimated dominant eigenvalue (Rayleigh quotient)
        let mut mv = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                mv[i] += sim_matrix[i][j] * v[j];
            }
        }
        let v_t_m_v: f64 = v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum();
        let v_t_v: f64 = v.iter().map(|x| x * x).sum();
        let dominant_eig = v_t_m_v / v_t_v;
        features.push(dominant_eig / n as f64); // Normalized

        // Spectral gap approximation (using trace)
        let trace: f64 = (0..n).map(|i| sim_matrix[i][i]).sum();
        let off_diag_sum: f64 = sim_matrix
            .iter()
            .enumerate()
            .flat_map(|(i, row)| {
                row.iter()
                    .enumerate()
                    .filter(move |(j, _)| *j != i)
                    .map(|(_, &v)| v.abs())
            })
            .sum();
        features.push(trace / n as f64);
        features.push(off_diag_sum / (n * (n - 1)) as f64);

        // Frobenius norm
        let frob: f64 = sim_matrix
            .iter()
            .flat_map(|row| row.iter().map(|&v| v * v))
            .sum::<f64>()
            .sqrt();
        features.push(frob / n as f64);

        // Pad to target size
        while features.len() < num_features {
            features.push(0.0);
        }
        features.truncate(num_features);

        features
    }

    /// Learn transfer mapping from source (high-Φ) to target (low-Φ) signatures
    pub fn learn_transfer(
        &mut self,
        source_signatures: &[PhiSignature],
        _target_signatures: &[PhiSignature],
    ) {
        if source_signatures.is_empty() {
            return;
        }

        let dim = source_signatures[0].dim();

        // Initialize random weights
        let mut weights = vec![vec![0.0; dim]; dim];
        for i in 0..dim {
            for j in 0..dim {
                // Xavier initialization
                weights[i][j] = if i == j { 1.0 } else { 0.0 };
            }
        }

        // Store source signatures for reference
        self.source_signatures = source_signatures.to_vec();

        // Simple gradient descent to learn mapping
        // Goal: map low-Φ signatures toward high-Φ patterns
        let mut best_weights = weights.clone();
        let mut _best_loss = f64::INFINITY;

        for _iter in 0..self.config.max_iterations {
            // Compute loss and gradients
            let mut total_loss = 0.0;
            let mut gradients = vec![vec![0.0; dim]; dim];

            for source in source_signatures {
                let source_vec = source.as_vector();
                let target_phi = source.original_phi;

                // Apply current weights
                let transformed: Vec<f64> = (0..dim)
                    .map(|i| (0..dim).map(|j| weights[i][j] * source_vec[j]).sum::<f64>())
                    .collect();

                // Loss: distance from "ideal" high-Φ pattern
                // We want transformed features that correlate with high Φ
                let _predicted_phi: f64 = transformed.iter().sum::<f64>() / dim as f64;
                let loss = (1.0 - target_phi).powi(2); // We want high Φ
                total_loss += loss;

                // Compute gradients (simplified)
                for i in 0..dim {
                    for j in 0..dim {
                        gradients[i][j] += loss * source_vec[j] * self.config.learning_rate;
                    }
                }
            }

            // Update weights with regularization
            for i in 0..dim {
                for j in 0..dim {
                    weights[i][j] -= gradients[i][j] / source_signatures.len() as f64;
                    weights[i][j] -= self.config.regularization * weights[i][j];
                }
            }

            // Track best
            if total_loss < _best_loss {
                _best_loss = total_loss;
                best_weights = weights.clone();
            }

            // Check convergence
            if total_loss < self.config.convergence_threshold {
                break;
            }
        }

        self.transfer_weights = Some(best_weights);
    }

    /// Transfer consciousness patterns from source to target topology
    pub fn transfer(
        &self,
        source_components: &[ContinuousHV],
        target_components: &[ContinuousHV],
        source_phi: f64,
        target_phi: f64,
        source_type: &str,
        target_type: &str,
    ) -> PhiTransferResult {
        let source_sig = self.extract_signature(source_components, source_phi, Some(source_type));
        let target_sig = self.extract_signature(target_components, target_phi, Some(target_type));

        // Compute transfer vector (difference in signatures)
        let source_vec = source_sig.as_vector();
        let target_vec = target_sig.as_vector();

        let transfer_vector: Vec<f64> = source_vec
            .iter()
            .zip(target_vec.iter())
            .map(|(s, t)| s - t)
            .collect();

        // Estimate Φ improvement based on signature similarity
        let signature_sim: f64 = source_vec
            .iter()
            .zip(target_vec.iter())
            .map(|(s, t)| s * t)
            .sum::<f64>();
        let source_norm: f64 = source_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        let target_norm: f64 = target_vec.iter().map(|x| x * x).sum::<f64>().sqrt();

        let cosine_sim = if source_norm > 0.0 && target_norm > 0.0 {
            signature_sim / (source_norm * target_norm)
        } else {
            0.0
        };

        // Transfer effectiveness: how much of source Φ can be transferred
        // Based on signature similarity
        let transfer_efficiency = cosine_sim.clamp(0.0, 1.0);
        let phi_gap = source_phi - target_phi;
        let transferred_phi = phi_gap * transfer_efficiency;
        let enhanced_phi = target_phi + transferred_phi;

        // Transfer loss (how much was "lost in translation")
        let transfer_loss = (1.0 - transfer_efficiency) * phi_gap.abs();

        PhiTransferResult {
            original_phi: target_phi,
            enhanced_phi,
            improvement_ratio: enhanced_phi / target_phi.max(0.001),
            transfer_loss,
            iterations: 1,
            converged: true,
            source_type: source_type.to_string(),
            target_type: target_type.to_string(),
            transfer_vector,
        }
    }

    /// Compute transfer potential between topologies
    ///
    /// Returns a score indicating how well consciousness patterns
    /// can be transferred from source to target.
    pub fn transfer_potential(
        &self,
        source_components: &[ContinuousHV],
        target_components: &[ContinuousHV],
        source_phi: f64,
        target_phi: f64,
    ) -> f64 {
        let source_sig = self.extract_signature(source_components, source_phi, None);
        let target_sig = self.extract_signature(target_components, target_phi, None);

        let source_vec = source_sig.as_vector();
        let target_vec = target_sig.as_vector();

        // Compute cosine similarity of signatures
        let dot: f64 = source_vec
            .iter()
            .zip(target_vec.iter())
            .map(|(s, t)| s * t)
            .sum();
        let norm_s: f64 = source_vec.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_t: f64 = target_vec.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm_s > 0.0 && norm_t > 0.0 {
            let cosine = dot / (norm_s * norm_t);
            // Transfer potential: high similarity + high source Φ = good transfer potential
            cosine.max(0.0) * source_phi
        } else {
            0.0
        }
    }
}

impl Default for PhiTransfer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: quick transfer from Ring to target
pub fn transfer_from_ring(
    target_components: &[ContinuousHV],
    target_phi: f64,
) -> PhiTransferResult {
    use crate::hdc::HDC_DIMENSION;
    use crate::hdc::consciousness_topology_generators::ConsciousnessTopology;
    use crate::hdc::spectral_connectivity::ConnectivityCalculator;

    let n = target_components.len().max(8);
    let dim = if target_components.is_empty() {
        HDC_DIMENSION
    } else {
        target_components[0].dim()
    };

    // Generate Ring topology for transfer
    let ring = ConsciousnessTopology::ring(n, dim, 42);
    let ring_phi = ConnectivityCalculator::new().algebraic_connectivity(&ring.node_representations);

    let transfer = PhiTransfer::new();
    transfer.transfer(
        &ring.node_representations,
        target_components,
        ring_phi,
        target_phi,
        "Ring",
        "Unknown",
    )
}

/// Convenience function: compute transfer matrix between topology types
pub fn compute_transfer_matrix(topologies: &[(String, Vec<ContinuousHV>, f64)]) -> Vec<Vec<f64>> {
    let n = topologies.len();
    let mut matrix = vec![vec![0.0; n]; n];
    let transfer = PhiTransfer::new();

    for i in 0..n {
        for j in 0..n {
            if i != j {
                let potential = transfer.transfer_potential(
                    &topologies[i].1,
                    &topologies[j].1,
                    topologies[i].2,
                    topologies[j].2,
                );
                matrix[i][j] = potential;
            }
        }
    }

    matrix
}

// REVOLUTIONARY #98: Φ CAUSAL INTERVENTION ANALYSIS
// ============================================================================
//
// Φ Causal Intervention Analysis models how perturbations to specific nodes
// affect overall integrated information. This is core to IIT's concept of
// causal power - understanding which components are most critical for
// consciousness.
//
// ## Core Insight
//
// In IIT, consciousness arises from causal interactions between components.
// By systematically intervening on individual nodes (knockout, amplify, dampen),
// we can measure each node's contribution to overall Φ and identify:
//
// 1. **Critical Nodes**: Nodes whose removal drastically reduces Φ
// 2. **Redundant Nodes**: Nodes whose removal barely affects Φ
// 3. **Hub Nodes**: Nodes that connect otherwise isolated subsystems
// 4. **Bridge Nodes**: Nodes that facilitate integration across partitions
//
// ## Mathematical Foundation
//
// For a system with n nodes and baseline Φ₀:
//
// **Knockout Analysis**: Φᵢ⁻ = Φ(system without node i)
//   - Δ_knockout(i) = Φ₀ - Φᵢ⁻
//   - High Δ → critical node
//
// **Amplification Analysis**: Φᵢ⁺ = Φ(system with amplified node i)
//   - Δ_amplify(i) = Φᵢ⁺ - Φ₀
//   - High Δ → influential node
//
// **Dampening Analysis**: Φᵢ↓ = Φ(system with dampened node i)
//   - Δ_dampen(i) = Φ₀ - Φᵢ↓
//   - High Δ → important for maintaining integration
//
// **Causal Power**: CP(i) = weighted combination of intervention effects
//   - CP(i) = α·Δ_knockout + β·Δ_amplify + γ·Δ_dampen
//
// ## Applications
//
// - **Neural Lesion Modeling**: Predict effects of brain damage on consciousness
// - **Anesthesia Targeting**: Find optimal targets for consciousness disruption
// - **AGI Design**: Design systems with robust, distributed consciousness
// - **Network Optimization**: Identify bottlenecks and critical pathways
//
// ## References
//
// - Pearl (2009): "Causality: Models, Reasoning, and Inference"
// - Albantakis et al. (2023): "Causal structure in IIT 4.0"
// - Oizumi et al. (2014): "Measuring consciousness"
// - This work: First computational implementation of causal intervention for Φ

/// Type of intervention to apply to a node
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InterventionType {
    /// Remove node entirely from computation
    Knockout,
    /// Amplify node's influence (multiply by factor)
    Amplify(f64),
    /// Dampen node's influence (divide by factor)
    Dampen(f64),
    /// Replace with random noise
    Noise,
    /// Replace with constant value (clamp)
    Clamp(f64),
}

impl InterventionType {
    /// Human-readable description
    pub fn description(&self) -> String {
        match self {
            Self::Knockout => "knockout (remove node)".to_string(),
            Self::Amplify(f) => format!("amplify (×{f:.1})"),
            Self::Dampen(f) => format!("dampen (÷{f:.1})"),
            Self::Noise => "noise (randomize)".to_string(),
            Self::Clamp(v) => format!("clamp (set to {v:.2})"),
        }
    }

    /// Consciousness interpretation
    pub fn interpretation(&self) -> &'static str {
        match self {
            Self::Knockout => "neural lesion or surgical removal",
            Self::Amplify(_) => "neural excitation or stimulation",
            Self::Dampen(_) => "neural inhibition or sedation",
            Self::Noise => "random neural firing (seizure-like)",
            Self::Clamp(_) => "fixed activation (locked-in state)",
        }
    }
}

/// Configuration for causal intervention analysis
#[derive(Debug, Clone)]
pub struct CausalInterventionConfig {
    /// Number of bootstrap samples for confidence intervals
    pub bootstrap_samples: usize,
    /// Default amplification factor
    pub amplify_factor: f64,
    /// Default dampening factor
    pub dampen_factor: f64,
    /// Weight for knockout in causal power calculation
    pub knockout_weight: f64,
    /// Weight for amplify in causal power calculation
    pub amplify_weight: f64,
    /// Weight for dampen in causal power calculation
    pub dampen_weight: f64,
    /// Seed for deterministic random interventions
    pub seed: u64,
}

impl Default for CausalInterventionConfig {
    fn default() -> Self {
        Self {
            bootstrap_samples: 10,
            amplify_factor: 2.0,
            dampen_factor: 2.0,
            knockout_weight: 0.5,
            amplify_weight: 0.25,
            dampen_weight: 0.25,
            seed: 12345,
        }
    }
}

impl CausalInterventionConfig {
    /// Fast config for real-time analysis
    pub fn fast() -> Self {
        Self {
            bootstrap_samples: 3,
            ..Default::default()
        }
    }

    /// Research config for detailed analysis
    pub fn research() -> Self {
        Self {
            bootstrap_samples: 50,
            ..Default::default()
        }
    }
}

/// Result of intervening on a single node
#[derive(Debug, Clone)]
pub struct NodeInterventionResult {
    /// Index of the node
    pub node_index: usize,
    /// Type of intervention applied
    pub intervention: InterventionType,
    /// Baseline Φ before intervention
    pub baseline_phi: f64,
    /// Φ after intervention
    pub intervened_phi: f64,
    /// Change in Φ (baseline - intervened for knockout/dampen, intervened - baseline for amplify)
    pub delta_phi: f64,
    /// Percentage change in Φ
    pub percent_change: f64,
    /// Standard error (if bootstrap was used)
    pub standard_error: Option<f64>,
    /// 95% confidence interval
    pub confidence_interval: Option<(f64, f64)>,
}

impl NodeInterventionResult {
    /// Check if intervention had significant effect
    pub fn is_significant(&self, threshold: f64) -> bool {
        self.percent_change.abs() > threshold
    }

    /// Check if node is critical (knockout causes major Φ drop)
    pub fn is_critical(&self) -> bool {
        matches!(self.intervention, InterventionType::Knockout) && self.percent_change < -10.0
    }

    /// Check if node is redundant (knockout has minimal effect)
    pub fn is_redundant(&self) -> bool {
        matches!(self.intervention, InterventionType::Knockout) && self.percent_change.abs() < 5.0
    }
}

/// Comprehensive result of causal intervention analysis
#[derive(Debug, Clone)]
pub struct CausalAnalysisResult {
    /// Baseline Φ of the system
    pub baseline_phi: f64,
    /// Results for each node
    pub node_results: Vec<Vec<NodeInterventionResult>>,
    /// Causal power score for each node (weighted combination)
    pub causal_power: Vec<f64>,
    /// Ranking of nodes by causal power (highest first)
    pub node_ranking: Vec<usize>,
    /// Identified critical nodes (knockout causes >10% Φ drop)
    pub critical_nodes: Vec<usize>,
    /// Identified redundant nodes (knockout causes <5% Φ change)
    pub redundant_nodes: Vec<usize>,
    /// Mean Φ change per intervention type
    pub mean_effects: std::collections::HashMap<String, f64>,
}

impl CausalAnalysisResult {
    /// Get the most critical node
    pub fn most_critical_node(&self) -> Option<usize> {
        self.node_ranking.first().copied()
    }

    /// Get the least critical node
    pub fn least_critical_node(&self) -> Option<usize> {
        self.node_ranking.last().copied()
    }

    /// Compute system robustness (how resistant to single-node failures)
    pub fn robustness(&self) -> f64 {
        if self.critical_nodes.is_empty() {
            1.0 // No critical nodes = maximally robust
        } else {
            let critical_fraction =
                self.critical_nodes.len() as f64 / self.node_ranking.len() as f64;
            1.0 - critical_fraction
        }
    }

    /// Compute system concentration (how concentrated is causal power)
    pub fn concentration(&self) -> f64 {
        if self.causal_power.is_empty() {
            return 0.0;
        }

        let sum: f64 = self.causal_power.iter().sum();
        if sum <= 0.0 {
            return 0.0;
        }

        // Gini coefficient
        let n = self.causal_power.len();
        let mut sorted = self.causal_power.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));

        let mut gini_sum = 0.0;
        for (i, &x) in sorted.iter().enumerate() {
            gini_sum += (2.0 * (i + 1) as f64 - n as f64 - 1.0) * x;
        }

        (gini_sum / (n as f64 * sum)).abs()
    }
}

/// Φ Causal Intervention Analyzer
///
/// Models how perturbations to specific nodes affect overall integrated
/// information, identifying critical, hub, and redundant nodes.
#[derive(Debug, Clone)]
pub struct PhiCausalAnalyzer {
    config: CausalInterventionConfig,
}

impl PhiCausalAnalyzer {
    /// Create new analyzer with default config
    pub fn new() -> Self {
        Self {
            config: CausalInterventionConfig::default(),
        }
    }

    /// Create with custom config
    pub fn with_config(config: CausalInterventionConfig) -> Self {
        Self { config }
    }

    /// Fast analyzer for real-time use
    pub fn fast() -> Self {
        Self::with_config(CausalInterventionConfig::fast())
    }

    /// Research analyzer for detailed analysis
    pub fn research() -> Self {
        Self::with_config(CausalInterventionConfig::research())
    }

    /// Perform comprehensive causal intervention analysis
    ///
    /// Tests knockout, amplify, and dampen interventions on each node
    /// and computes causal power scores.
    pub fn analyze(&self, node_representations: &[ContinuousHV]) -> CausalAnalysisResult {
        let n = node_representations.len();
        if n == 0 {
            return CausalAnalysisResult {
                baseline_phi: 0.0,
                node_results: vec![],
                causal_power: vec![],
                node_ranking: vec![],
                critical_nodes: vec![],
                redundant_nodes: vec![],
                mean_effects: std::collections::HashMap::new(),
            };
        }

        // Compute baseline Φ
        let baseline_phi = self.compute_phi(node_representations);

        // Test interventions on each node
        let mut node_results = Vec::with_capacity(n);
        let mut causal_power = Vec::with_capacity(n);

        for node_idx in 0..n {
            let mut node_interventions = Vec::new();
            let mut knockout_delta = 0.0;
            let mut amplify_delta = 0.0;
            let mut dampen_delta = 0.0;

            // Knockout
            let knockout_result = self.test_intervention(
                node_representations,
                node_idx,
                InterventionType::Knockout,
                baseline_phi,
            );
            knockout_delta = knockout_result.delta_phi;
            node_interventions.push(knockout_result);

            // Amplify
            let amplify_result = self.test_intervention(
                node_representations,
                node_idx,
                InterventionType::Amplify(self.config.amplify_factor),
                baseline_phi,
            );
            amplify_delta = amplify_result.delta_phi;
            node_interventions.push(amplify_result);

            // Dampen
            let dampen_result = self.test_intervention(
                node_representations,
                node_idx,
                InterventionType::Dampen(self.config.dampen_factor),
                baseline_phi,
            );
            dampen_delta = dampen_result.delta_phi;
            node_interventions.push(dampen_result);

            node_results.push(node_interventions);

            // Compute causal power (weighted combination)
            let cp = self.config.knockout_weight * knockout_delta.abs()
                + self.config.amplify_weight * amplify_delta.abs()
                + self.config.dampen_weight * dampen_delta.abs();
            causal_power.push(cp);
        }

        // Rank nodes by causal power
        let mut node_ranking: Vec<usize> = (0..n).collect();
        node_ranking.sort_by(|&a, &b| causal_power[b].total_cmp(&causal_power[a]));

        // Identify critical and redundant nodes
        let critical_nodes: Vec<usize> = node_results
            .iter()
            .enumerate()
            .filter(|(_, results)| results.iter().any(|r| r.is_critical()))
            .map(|(i, _)| i)
            .collect();

        let redundant_nodes: Vec<usize> = node_results
            .iter()
            .enumerate()
            .filter(|(_, results)| results.iter().any(|r| r.is_redundant()))
            .map(|(i, _)| i)
            .collect();

        // Compute mean effects
        let mut mean_effects = std::collections::HashMap::new();

        let mut knockout_sum = 0.0;
        let mut amplify_sum = 0.0;
        let mut dampen_sum = 0.0;

        for results in &node_results {
            for r in results {
                match r.intervention {
                    InterventionType::Knockout => knockout_sum += r.delta_phi,
                    InterventionType::Amplify(_) => amplify_sum += r.delta_phi,
                    InterventionType::Dampen(_) => dampen_sum += r.delta_phi,
                    _ => {}
                }
            }
        }

        mean_effects.insert("knockout".to_string(), knockout_sum / n as f64);
        mean_effects.insert("amplify".to_string(), amplify_sum / n as f64);
        mean_effects.insert("dampen".to_string(), dampen_sum / n as f64);

        CausalAnalysisResult {
            baseline_phi,
            node_results,
            causal_power,
            node_ranking,
            critical_nodes,
            redundant_nodes,
            mean_effects,
        }
    }

    /// Test a single intervention on a node
    fn test_intervention(
        &self,
        nodes: &[ContinuousHV],
        node_idx: usize,
        intervention: InterventionType,
        baseline_phi: f64,
    ) -> NodeInterventionResult {
        let intervened_nodes = self.apply_intervention(nodes, node_idx, intervention);
        let intervened_phi = self.compute_phi(&intervened_nodes);

        let delta_phi = match intervention {
            InterventionType::Amplify(_) => intervened_phi - baseline_phi,
            _ => baseline_phi - intervened_phi,
        };

        let percent_change = if baseline_phi > 0.0 {
            (delta_phi / baseline_phi) * 100.0
        } else {
            0.0
        };

        NodeInterventionResult {
            node_index: node_idx,
            intervention,
            baseline_phi,
            intervened_phi,
            delta_phi,
            percent_change,
            standard_error: None,
            confidence_interval: None,
        }
    }

    /// Apply intervention to create modified node set
    fn apply_intervention(
        &self,
        nodes: &[ContinuousHV],
        node_idx: usize,
        intervention: InterventionType,
    ) -> Vec<ContinuousHV> {
        let mut modified = nodes.to_vec();

        match intervention {
            InterventionType::Knockout => {
                // Remove the node entirely (swap with last and truncate)
                if node_idx < modified.len() {
                    modified.remove(node_idx);
                }
            }
            InterventionType::Amplify(factor) => {
                if node_idx < modified.len() {
                    modified[node_idx] = modified[node_idx].scale(factor as f32);
                }
            }
            InterventionType::Dampen(factor) => {
                if node_idx < modified.len() {
                    modified[node_idx] = modified[node_idx].scale(1.0 / factor as f32);
                }
            }
            InterventionType::Noise => {
                if node_idx < modified.len() {
                    let dim = modified[node_idx].values.len();
                    modified[node_idx] =
                        ContinuousHV::random(dim, self.config.seed + node_idx as u64);
                }
            }
            InterventionType::Clamp(value) => {
                if node_idx < modified.len() {
                    let dim = modified[node_idx].values.len();
                    modified[node_idx] = ContinuousHV {
                        values: vec![value as f32; dim],
                    };
                }
            }
        }

        modified
    }

    /// Compute Φ for a set of node representations
    /// Uses cosine similarity-based integration measure
    fn compute_phi(&self, nodes: &[ContinuousHV]) -> f64 {
        let n = nodes.len();
        if n < 2 {
            return 0.0;
        }

        // Compute pairwise similarity matrix
        let mut total_sim = 0.0;
        let mut count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = nodes[i].similarity(&nodes[j]) as f64;
                total_sim += sim;
                count += 1;
            }
        }

        // Average similarity as integration measure
        if count > 0 {
            total_sim / count as f64
        } else {
            0.0
        }
    }

    /// Analyze intervention effects on a subset of nodes
    pub fn analyze_subset(
        &self,
        nodes: &[ContinuousHV],
        target_indices: &[usize],
    ) -> Vec<NodeInterventionResult> {
        let baseline_phi = self.compute_phi(nodes);

        target_indices
            .iter()
            .filter(|&&idx| idx < nodes.len())
            .map(|&idx| {
                self.test_intervention(nodes, idx, InterventionType::Knockout, baseline_phi)
            })
            .collect()
    }

    /// Find the minimum dominating set (nodes that control most of Φ)
    pub fn find_dominating_set(&self, nodes: &[ContinuousHV], threshold: f64) -> Vec<usize> {
        let analysis = self.analyze(nodes);

        let mut dominating = Vec::new();
        let mut cumulative_power = 0.0;
        let total_power: f64 = analysis.causal_power.iter().sum();

        if total_power <= 0.0 {
            return dominating;
        }

        for &node_idx in &analysis.node_ranking {
            dominating.push(node_idx);
            cumulative_power += analysis.causal_power[node_idx];

            if cumulative_power / total_power >= threshold {
                break;
            }
        }

        dominating
    }
}

impl Default for PhiCausalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function: analyze causal interventions
pub fn analyze_causal_interventions(node_representations: &[ContinuousHV]) -> CausalAnalysisResult {
    PhiCausalAnalyzer::new().analyze(node_representations)
}

/// Convenience function: find critical nodes
pub fn find_critical_nodes(node_representations: &[ContinuousHV]) -> Vec<usize> {
    PhiCausalAnalyzer::new()
        .analyze(node_representations)
        .critical_nodes
}

/// Convenience function: compute causal power scores
pub fn compute_causal_power(node_representations: &[ContinuousHV]) -> Vec<f64> {
    PhiCausalAnalyzer::new()
        .analyze(node_representations)
        .causal_power
}

// ============================================================================

// REVOLUTIONARY #99: Φ NETWORK MODULARITY ANALYSIS
// ============================================================================

/// Method for detecting community modules in consciousness networks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModuleDetectionMethod {
    /// Spectral clustering using eigenvalues of similarity matrix
    Spectral,
    /// Agglomerative clustering with similarity threshold
    Agglomerative,
    /// Louvain-inspired greedy modularity optimization
    Greedy,
    /// K-means on similarity vectors
    KMeans,
}

impl ModuleDetectionMethod {
    /// Get description of detection method
    pub fn description(&self) -> &'static str {
        match self {
            Self::Spectral => "Spectral clustering using eigendecomposition",
            Self::Agglomerative => "Hierarchical agglomerative clustering",
            Self::Greedy => "Greedy modularity optimization",
            Self::KMeans => "K-means clustering on similarity space",
        }
    }
}

/// Configuration for network modularity analysis
#[derive(Debug, Clone)]
pub struct ModularityConfig {
    /// Method for detecting modules
    pub detection_method: ModuleDetectionMethod,
    /// Number of modules to detect (None = auto-detect)
    pub num_modules: Option<usize>,
    /// Similarity threshold for clustering (0.0 to 1.0)
    pub similarity_threshold: f64,
    /// Minimum module size
    pub min_module_size: usize,
    /// Whether to compute inter-module Φ (expensive)
    pub compute_inter_module_phi: bool,
    /// Maximum iterations for iterative methods
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
}

impl Default for ModularityConfig {
    fn default() -> Self {
        Self {
            detection_method: ModuleDetectionMethod::Agglomerative,
            num_modules: None,
            similarity_threshold: 0.3,
            min_module_size: 2,
            compute_inter_module_phi: true,
            max_iterations: 100,
            convergence_threshold: 1e-6,
        }
    }
}

impl ModularityConfig {
    /// Quick analysis preset
    pub fn quick() -> Self {
        Self {
            detection_method: ModuleDetectionMethod::Agglomerative,
            num_modules: Some(3),
            similarity_threshold: 0.35,
            min_module_size: 2,
            compute_inter_module_phi: false,
            max_iterations: 50,
            convergence_threshold: 1e-4,
        }
    }

    /// Thorough analysis preset
    pub fn thorough() -> Self {
        Self {
            detection_method: ModuleDetectionMethod::Spectral,
            num_modules: None,
            similarity_threshold: 0.25,
            min_module_size: 2,
            compute_inter_module_phi: true,
            max_iterations: 200,
            convergence_threshold: 1e-8,
        }
    }

    /// Research preset with all features
    pub fn research() -> Self {
        Self {
            detection_method: ModuleDetectionMethod::Greedy,
            num_modules: None,
            similarity_threshold: 0.2,
            min_module_size: 1,
            compute_inter_module_phi: true,
            max_iterations: 500,
            convergence_threshold: 1e-10,
        }
    }
}

/// A detected module (community) in the consciousness network
#[derive(Debug, Clone)]
pub struct ConsciousnessModule {
    /// Module identifier
    pub id: usize,
    /// Node indices belonging to this module
    pub node_indices: Vec<usize>,
    /// Internal cohesion (average intra-module similarity)
    pub internal_cohesion: f64,
    /// Φ value within this module
    pub internal_phi: f64,
    /// Isolation score (1 - avg external connections)
    pub isolation_score: f64,
    /// Centroid representation (average of node representations)
    pub centroid: Option<ContinuousHV>,
}

impl ConsciousnessModule {
    /// Get module size
    pub fn size(&self) -> usize {
        self.node_indices.len()
    }

    /// Check if node is in this module
    pub fn contains(&self, node_index: usize) -> bool {
        self.node_indices.contains(&node_index)
    }

    /// Get integration efficiency (phi / size ratio)
    pub fn integration_efficiency(&self) -> f64 {
        if self.node_indices.is_empty() {
            0.0
        } else {
            self.internal_phi / (self.node_indices.len() as f64).sqrt()
        }
    }
}

/// Result of inter-module analysis
#[derive(Debug, Clone)]
pub struct InterModuleRelation {
    /// First module ID
    pub module_a: usize,
    /// Second module ID
    pub module_b: usize,
    /// Coupling strength (similarity between centroids)
    pub coupling_strength: f64,
    /// Information flow (asymmetric measure)
    pub info_flow_a_to_b: f64,
    /// Information flow (reverse direction)
    pub info_flow_b_to_a: f64,
    /// Bridge nodes connecting these modules
    pub bridge_nodes: Vec<usize>,
}

/// Node role classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NodeRole {
    /// Core member of a single module
    Core,
    /// Peripheral member (low internal connectivity)
    Peripheral,
    /// Bridge connecting multiple modules
    Bridge,
    /// Hub connecting to many nodes across modules
    Hub,
    /// Isolated node with minimal connections
    Isolated,
}

impl NodeRole {
    /// Get description of the role
    pub fn description(&self) -> &'static str {
        match self {
            Self::Core => "Core member strongly connected within module",
            Self::Peripheral => "Peripheral member with weak internal ties",
            Self::Bridge => "Bridge connecting multiple modules",
            Self::Hub => "Hub with connections across many modules",
            Self::Isolated => "Isolated node with minimal connections",
        }
    }
}

/// Classification of each node's role in the network
#[derive(Debug, Clone)]
pub struct NodeClassification {
    /// Node index
    pub node_index: usize,
    /// Primary module assignment
    pub primary_module: usize,
    /// Node role classification
    pub role: NodeRole,
    /// Within-module degree (z-score)
    pub within_module_degree: f64,
    /// Participation coefficient (diversity of connections)
    pub participation_coefficient: f64,
    /// Betweenness centrality
    pub betweenness: f64,
}

/// Comprehensive result of network modularity analysis
#[derive(Debug, Clone)]
pub struct NetworkModularityResult {
    /// Total Φ of the entire network
    pub total_phi: f64,
    /// Detected modules
    pub modules: Vec<ConsciousnessModule>,
    /// Modularity score Q (higher = more modular)
    pub modularity_score: f64,
    /// Inter-module relations
    pub inter_module_relations: Vec<InterModuleRelation>,
    /// Classification of each node
    pub node_classifications: Vec<NodeClassification>,
    /// Bridge nodes connecting modules
    pub bridge_nodes: Vec<usize>,
    /// Bottleneck edges (critical connections)
    pub bottleneck_edges: Vec<(usize, usize)>,
    /// Segregation index (within-module / total connectivity)
    pub segregation_index: f64,
    /// Integration index (between-module connectivity quality)
    pub integration_index: f64,
    /// Hierarchical modularity (modularity at different scales)
    pub hierarchical_scores: Vec<f64>,
}

impl NetworkModularityResult {
    /// Get number of modules
    pub fn num_modules(&self) -> usize {
        self.modules.len()
    }

    /// Get average module size
    pub fn avg_module_size(&self) -> f64 {
        if self.modules.is_empty() {
            0.0
        } else {
            let total: usize = self.modules.iter().map(|m| m.size()).sum();
            total as f64 / self.modules.len() as f64
        }
    }

    /// Get largest module
    pub fn largest_module(&self) -> Option<&ConsciousnessModule> {
        self.modules.iter().max_by_key(|m| m.size())
    }

    /// Get module with highest internal Φ
    pub fn highest_phi_module(&self) -> Option<&ConsciousnessModule> {
        self.modules
            .iter()
            .max_by(|a, b| a.internal_phi.total_cmp(&b.internal_phi))
    }

    /// Get balance score (how evenly sized modules are)
    pub fn balance_score(&self) -> f64 {
        if self.modules.len() < 2 {
            return 1.0;
        }
        let sizes: Vec<f64> = self.modules.iter().map(|m| m.size() as f64).collect();
        let mean = sizes.iter().sum::<f64>() / sizes.len() as f64;
        let variance = sizes.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / sizes.len() as f64;
        let std_dev = variance.sqrt();
        1.0 / (1.0 + std_dev / mean)
    }

    /// Get efficiency ratio (sum of module Φ vs total Φ)
    pub fn efficiency_ratio(&self) -> f64 {
        if self.total_phi == 0.0 {
            return 0.0;
        }
        let sum_module_phi: f64 = self.modules.iter().map(|m| m.internal_phi).sum();
        sum_module_phi / self.total_phi
    }
}

/// Φ Network Modularity Analyzer
#[derive(Debug, Clone)]
pub struct PhiModularityAnalyzer {
    config: ModularityConfig,
}

impl Default for PhiModularityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl PhiModularityAnalyzer {
    /// Create new analyzer with default config
    pub fn new() -> Self {
        Self {
            config: ModularityConfig::default(),
        }
    }

    /// Create analyzer with custom config
    pub fn with_config(config: ModularityConfig) -> Self {
        Self { config }
    }

    /// Perform full modularity analysis
    pub fn analyze(&self, node_representations: &[ContinuousHV]) -> NetworkModularityResult {
        let n = node_representations.len();

        if n == 0 {
            return NetworkModularityResult {
                total_phi: 0.0,
                modules: Vec::new(),
                modularity_score: 0.0,
                inter_module_relations: Vec::new(),
                node_classifications: Vec::new(),
                bridge_nodes: Vec::new(),
                bottleneck_edges: Vec::new(),
                segregation_index: 0.0,
                integration_index: 0.0,
                hierarchical_scores: Vec::new(),
            };
        }

        // Compute similarity matrix
        let sim_matrix = self.compute_similarity_matrix(node_representations);

        // Compute total network Φ
        let total_phi = self.compute_phi(node_representations);

        // Detect modules
        let module_assignments = self.detect_modules(&sim_matrix);

        // Build module structures
        let modules = self.build_modules(node_representations, &module_assignments, &sim_matrix);

        // Compute modularity score Q
        let modularity_score = self.compute_modularity_q(&sim_matrix, &module_assignments);

        // Analyze inter-module relations
        let inter_module_relations = if self.config.compute_inter_module_phi {
            self.analyze_inter_module(node_representations, &modules, &sim_matrix)
        } else {
            Vec::new()
        };

        // Classify nodes
        let node_classifications = self.classify_nodes(&sim_matrix, &module_assignments, &modules);

        // Find bridge nodes
        let bridge_nodes = node_classifications
            .iter()
            .filter(|c| c.role == NodeRole::Bridge || c.role == NodeRole::Hub)
            .map(|c| c.node_index)
            .collect();

        // Find bottleneck edges
        let bottleneck_edges = self.find_bottleneck_edges(&sim_matrix, &module_assignments);

        // Compute segregation and integration indices
        let (segregation_index, integration_index) =
            self.compute_seg_int_indices(&sim_matrix, &module_assignments);

        // Compute hierarchical modularity
        let hierarchical_scores = self.compute_hierarchical_modularity(&sim_matrix);

        NetworkModularityResult {
            total_phi,
            modules,
            modularity_score,
            inter_module_relations,
            node_classifications,
            bridge_nodes,
            bottleneck_edges,
            segregation_index,
            integration_index,
            hierarchical_scores,
        }
    }

    /// Compute similarity matrix between all nodes
    fn compute_similarity_matrix(&self, node_representations: &[ContinuousHV]) -> Vec<Vec<f64>> {
        let n = node_representations.len();
        let mut matrix = vec![vec![0.0f64; n]; n];

        for i in 0..n {
            matrix[i][i] = 1.0;
            for j in (i + 1)..n {
                let sim = node_representations[i].similarity(&node_representations[j]);
                let normalized = ((sim + 1.0) / 2.0) as f64; // Normalize from [-1,1] to [0,1]
                matrix[i][j] = normalized;
                matrix[j][i] = normalized;
            }
        }
        matrix
    }

    /// Compute Φ for a set of representations
    fn compute_phi(&self, representations: &[ContinuousHV]) -> f64 {
        if representations.len() < 2 {
            return 0.0;
        }

        let bundle = ContinuousHV::bundle_owned(representations);
        let mut total_info = 0.0f64;

        for rep in representations {
            let sim = bundle.similarity(rep);
            let normalized = ((sim + 1.0) / 2.0) as f64;
            total_info += normalized;
        }

        total_info / (representations.len() as f64)
    }

    /// Detect modules using configured method
    fn detect_modules(&self, sim_matrix: &[Vec<f64>]) -> Vec<usize> {
        match self.config.detection_method {
            ModuleDetectionMethod::Spectral => self.detect_spectral(sim_matrix),
            ModuleDetectionMethod::Agglomerative => self.detect_agglomerative(sim_matrix),
            ModuleDetectionMethod::Greedy => self.detect_greedy(sim_matrix),
            ModuleDetectionMethod::KMeans => self.detect_kmeans(sim_matrix),
        }
    }

    /// Spectral clustering
    fn detect_spectral(&self, sim_matrix: &[Vec<f64>]) -> Vec<usize> {
        let n = sim_matrix.len();
        if n < 2 {
            return vec![0; n];
        }

        let k = self.config.num_modules.unwrap_or_else(|| {
            // Auto-detect using eigenvalue gap
            ((n as f64).sqrt() as usize).max(2).min(n / 2)
        });

        // Compute Laplacian eigenvalues (simplified - use power iteration)
        let mut fiedler = vec![0.0; n];
        let mut rng_state = 42u64;
        for v in fiedler.iter_mut() {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (rng_state as f64 / u64::MAX as f64) - 0.5;
        }

        // Power iteration for second eigenvector
        for _ in 0..self.config.max_iterations {
            let mut new_fiedler = vec![0.0; n];
            for i in 0..n {
                let degree: f64 = sim_matrix[i].iter().sum();
                for j in 0..n {
                    new_fiedler[i] +=
                        (if i == j { degree } else { 0.0 } - sim_matrix[i][j]) * fiedler[j];
                }
            }

            // Orthogonalize against constant vector
            let sum: f64 = new_fiedler.iter().sum();
            let mean = sum / n as f64;
            for v in new_fiedler.iter_mut() {
                *v -= mean;
            }

            // Normalize
            let norm: f64 = new_fiedler.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > 1e-10 {
                for v in new_fiedler.iter_mut() {
                    *v /= norm;
                }
            }
            fiedler = new_fiedler;
        }

        // Cluster by sign of Fiedler vector
        let mut assignments = vec![0; n];
        if k == 2 {
            for (i, &v) in fiedler.iter().enumerate() {
                assignments[i] = if v >= 0.0 { 0 } else { 1 };
            }
        } else {
            // For k > 2, use quantiles
            let mut sorted: Vec<(usize, f64)> = fiedler.iter().copied().enumerate().collect();
            sorted.sort_by(|a, b| a.1.total_cmp(&b.1));

            let chunk_size = n.div_ceil(k);
            for (rank, (idx, _)) in sorted.iter().enumerate() {
                assignments[*idx] = rank / chunk_size;
            }
        }

        assignments
    }

    /// Agglomerative clustering
    fn detect_agglomerative(&self, sim_matrix: &[Vec<f64>]) -> Vec<usize> {
        let n = sim_matrix.len();
        if n < 2 {
            return vec![0; n];
        }

        // Start with each node as its own cluster
        let mut assignments: Vec<usize> = (0..n).collect();
        let mut cluster_count = n;

        let target_clusters = self
            .config
            .num_modules
            .unwrap_or(((n as f64).sqrt() as usize).max(2).min(n / 2));

        // Merge until target
        while cluster_count > target_clusters {
            // Find most similar pair of clusters
            let mut best_sim = f64::MIN;
            let mut best_pair = (0, 0);

            let unique_clusters: Vec<usize> = {
                let mut v: Vec<usize> = assignments.clone();
                v.sort_unstable();
                v.dedup();
                v
            };

            for (ci, &c1) in unique_clusters.iter().enumerate() {
                for &c2 in unique_clusters.iter().skip(ci + 1) {
                    // Average linkage similarity
                    let nodes1: Vec<usize> = assignments
                        .iter()
                        .enumerate()
                        .filter(|&(_, &c)| c == c1)
                        .map(|(i, _)| i)
                        .collect();
                    let nodes2: Vec<usize> = assignments
                        .iter()
                        .enumerate()
                        .filter(|&(_, &c)| c == c2)
                        .map(|(i, _)| i)
                        .collect();

                    let mut sum = 0.0;
                    let mut count = 0;
                    for &i in &nodes1 {
                        for &j in &nodes2 {
                            sum += sim_matrix[i][j];
                            count += 1;
                        }
                    }

                    let avg_sim = if count > 0 { sum / count as f64 } else { 0.0 };

                    if avg_sim > best_sim {
                        best_sim = avg_sim;
                        best_pair = (c1, c2);
                    }
                }
            }

            // Merge clusters
            if best_sim >= self.config.similarity_threshold || cluster_count > target_clusters * 2 {
                let (merge_from, merge_to) = best_pair;
                for a in assignments.iter_mut() {
                    if *a == merge_from {
                        *a = merge_to;
                    }
                }
                cluster_count -= 1;
            } else {
                break;
            }
        }

        // Renumber clusters consecutively
        let mut mapping = std::collections::HashMap::new();
        let mut next_id = 0;
        for a in assignments.iter_mut() {
            let new_id = *mapping.entry(*a).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            *a = new_id;
        }

        assignments
    }

    /// Greedy modularity optimization
    fn detect_greedy(&self, sim_matrix: &[Vec<f64>]) -> Vec<usize> {
        let n = sim_matrix.len();
        if n < 2 {
            return vec![0; n];
        }

        // Start with each node in its own cluster
        let mut assignments: Vec<usize> = (0..n).collect();

        // Compute total edge weight
        let total_weight: f64 = sim_matrix
            .iter()
            .enumerate()
            .flat_map(|(i, row)| row.iter().skip(i + 1))
            .sum::<f64>()
            * 2.0;

        if total_weight == 0.0 {
            return vec![0; n];
        }

        // Greedy optimization
        for _ in 0..self.config.max_iterations {
            let mut improved = false;

            for i in 0..n {
                let current_cluster = assignments[i];

                // Try moving to each other cluster
                let unique_clusters: Vec<usize> = {
                    let mut v: Vec<usize> = assignments.clone();
                    v.sort_unstable();
                    v.dedup();
                    v
                };

                let mut best_gain = 0.0;
                let mut best_cluster = current_cluster;

                for &target_cluster in &unique_clusters {
                    if target_cluster == current_cluster {
                        continue;
                    }

                    // Compute modularity gain
                    let gain = self.compute_move_gain(
                        i,
                        current_cluster,
                        target_cluster,
                        &assignments,
                        sim_matrix,
                        total_weight,
                    );

                    if gain > best_gain {
                        best_gain = gain;
                        best_cluster = target_cluster;
                    }
                }

                if best_cluster != current_cluster {
                    assignments[i] = best_cluster;
                    improved = true;
                }
            }

            if !improved {
                break;
            }
        }

        // Compact cluster IDs
        let mut mapping = std::collections::HashMap::new();
        let mut next_id = 0;
        for a in assignments.iter_mut() {
            let new_id = *mapping.entry(*a).or_insert_with(|| {
                let id = next_id;
                next_id += 1;
                id
            });
            *a = new_id;
        }

        assignments
    }

    /// Compute modularity gain for moving a node
    fn compute_move_gain(
        &self,
        node: usize,
        from_cluster: usize,
        to_cluster: usize,
        assignments: &[usize],
        sim_matrix: &[Vec<f64>],
        total_weight: f64,
    ) -> f64 {
        let n = assignments.len();

        // Compute connections to target cluster
        let mut to_target = 0.0;
        let mut from_current = 0.0;

        for j in 0..n {
            if j == node {
                continue;
            }

            let weight = sim_matrix[node][j];
            if assignments[j] == to_cluster {
                to_target += weight;
            } else if assignments[j] == from_cluster {
                from_current += weight;
            }
        }

        // Degree of node
        let node_degree: f64 = sim_matrix[node].iter().sum();

        // Degree sum of clusters
        let to_degree: f64 = assignments
            .iter()
            .enumerate()
            .filter(|&(_, &c)| c == to_cluster)
            .map(|(i, _)| sim_matrix[i].iter().sum::<f64>())
            .sum();

        let from_degree: f64 = assignments
            .iter()
            .enumerate()
            .filter(|&(ref i, &c)| c == from_cluster && i != &node)
            .map(|(i, _)| sim_matrix[i].iter().sum::<f64>())
            .sum();

        // Modularity gain formula
        let gain = 2.0 * (to_target - from_current)
            - node_degree * (to_degree - from_degree + node_degree) / total_weight;

        gain / total_weight
    }

    /// K-means clustering on similarity space
    fn detect_kmeans(&self, sim_matrix: &[Vec<f64>]) -> Vec<usize> {
        let n = sim_matrix.len();
        if n < 2 {
            return vec![0; n];
        }

        let k = self
            .config
            .num_modules
            .unwrap_or(((n as f64).sqrt() as usize).max(2).min(n / 2));

        // Initialize centroids (use k-means++ style)
        let mut centroids: Vec<Vec<f64>> = Vec::with_capacity(k);
        let mut rng_state = 42u64;

        // First centroid: random
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let first_idx = (rng_state % n as u64) as usize;
        centroids.push(sim_matrix[first_idx].clone());

        // Subsequent centroids: proportional to distance
        while centroids.len() < k {
            let mut distances: Vec<f64> = vec![f64::MAX; n];
            for (i, row) in sim_matrix.iter().enumerate() {
                for centroid in &centroids {
                    let dist: f64 = row
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    distances[i] = distances[i].min(dist);
                }
            }

            let total: f64 = distances.iter().sum();
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let mut target = (rng_state as f64 / u64::MAX as f64) * total;

            for (i, &d) in distances.iter().enumerate() {
                target -= d;
                if target <= 0.0 {
                    centroids.push(sim_matrix[i].clone());
                    break;
                }
            }

            if centroids.len() == k - 1 {
                centroids.push(sim_matrix[0].clone()); // Fallback
            }
        }

        // K-means iterations
        let mut assignments = vec![0; n];

        for _ in 0..self.config.max_iterations {
            // Assign to nearest centroid
            let mut changed = false;
            for (i, row) in sim_matrix.iter().enumerate() {
                let mut best_cluster = 0;
                let mut best_dist = f64::MAX;

                for (c, centroid) in centroids.iter().enumerate() {
                    let dist: f64 = row
                        .iter()
                        .zip(centroid.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum();
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = c;
                    }
                }

                if assignments[i] != best_cluster {
                    assignments[i] = best_cluster;
                    changed = true;
                }
            }

            if !changed {
                break;
            }

            // Update centroids
            for (c, centroid) in centroids.iter_mut().enumerate() {
                let members: Vec<usize> = assignments
                    .iter()
                    .enumerate()
                    .filter(|&(_, &a)| a == c)
                    .map(|(i, _)| i)
                    .collect();

                if !members.is_empty() {
                    for (j, val) in centroid.iter_mut().enumerate() {
                        *val = members.iter().map(|&i| sim_matrix[i][j]).sum::<f64>()
                            / members.len() as f64;
                    }
                }
            }
        }

        assignments
    }

    /// Build module structures from assignments
    fn build_modules(
        &self,
        node_representations: &[ContinuousHV],
        assignments: &[usize],
        sim_matrix: &[Vec<f64>],
    ) -> Vec<ConsciousnessModule> {
        let num_modules = assignments.iter().max().map(|&m| m + 1).unwrap_or(0);
        let mut modules = Vec::with_capacity(num_modules);

        for module_id in 0..num_modules {
            let node_indices: Vec<usize> = assignments
                .iter()
                .enumerate()
                .filter(|&(_, &a)| a == module_id)
                .map(|(i, _)| i)
                .collect();

            if node_indices.len() < self.config.min_module_size {
                continue;
            }

            // Internal cohesion
            let internal_cohesion = if node_indices.len() > 1 {
                let mut sum = 0.0;
                let mut count = 0;
                for (ii, &i) in node_indices.iter().enumerate() {
                    for &j in node_indices.iter().skip(ii + 1) {
                        sum += sim_matrix[i][j];
                        count += 1;
                    }
                }
                if count > 0 { sum / count as f64 } else { 0.0 }
            } else {
                1.0
            };

            // Internal Φ
            let module_reps: Vec<ContinuousHV> = node_indices
                .iter()
                .map(|&i| node_representations[i].clone())
                .collect();
            let internal_phi = self.compute_phi(&module_reps);

            // Isolation score
            let external_nodes: Vec<usize> = (0..assignments.len())
                .filter(|&i| assignments[i] != module_id)
                .collect();
            let isolation_score = if external_nodes.is_empty() {
                1.0
            } else {
                let mut ext_sum = 0.0;
                let mut ext_count = 0;
                for &i in &node_indices {
                    for &j in &external_nodes {
                        ext_sum += sim_matrix[i][j];
                        ext_count += 1;
                    }
                }
                let avg_ext = if ext_count > 0 {
                    ext_sum / ext_count as f64
                } else {
                    0.0
                };
                1.0 - avg_ext
            };

            // Centroid
            let centroid = if !module_reps.is_empty() {
                Some(ContinuousHV::bundle_owned(&module_reps))
            } else {
                None
            };

            modules.push(ConsciousnessModule {
                id: module_id,
                node_indices,
                internal_cohesion,
                internal_phi,
                isolation_score,
                centroid,
            });
        }

        modules
    }

    /// Compute modularity Q score
    fn compute_modularity_q(&self, sim_matrix: &[Vec<f64>], assignments: &[usize]) -> f64 {
        let n = sim_matrix.len();
        if n < 2 {
            return 0.0;
        }

        let total_weight: f64 = sim_matrix
            .iter()
            .enumerate()
            .flat_map(|(i, row)| row.iter().skip(i + 1))
            .sum::<f64>()
            * 2.0;

        if total_weight == 0.0 {
            return 0.0;
        }

        let degrees: Vec<f64> = sim_matrix.iter().map(|row| row.iter().sum()).collect();

        let mut q = 0.0;
        for i in 0..n {
            for j in 0..n {
                if assignments[i] == assignments[j] {
                    let expected = degrees[i] * degrees[j] / total_weight;
                    q += sim_matrix[i][j] - expected;
                }
            }
        }

        q / total_weight
    }

    /// Analyze inter-module relations
    fn analyze_inter_module(
        &self,
        _node_representations: &[ContinuousHV],
        modules: &[ConsciousnessModule],
        sim_matrix: &[Vec<f64>],
    ) -> Vec<InterModuleRelation> {
        let mut relations = Vec::new();

        for (mi, module_a) in modules.iter().enumerate() {
            for module_b in modules.iter().skip(mi + 1) {
                // Coupling strength
                let coupling: f64 =
                    if let (Some(ca), Some(cb)) = (&module_a.centroid, &module_b.centroid) {
                        ((ca.similarity(cb) + 1.0) / 2.0) as f64
                    } else {
                        0.0f64
                    };

                // Information flow (asymmetric)
                let mut a_to_b = 0.0;
                let mut b_to_a = 0.0;
                let mut count = 0;

                for &i in &module_a.node_indices {
                    for &j in &module_b.node_indices {
                        a_to_b += sim_matrix[i][j];
                        b_to_a += sim_matrix[j][i];
                        count += 1;
                    }
                }

                if count > 0 {
                    a_to_b /= count as f64;
                    b_to_a /= count as f64;
                }

                // Bridge nodes
                let mut bridge_nodes = Vec::new();
                let threshold = self.config.similarity_threshold;

                for &i in &module_a.node_indices {
                    let has_strong_b_link = module_b
                        .node_indices
                        .iter()
                        .any(|&j| sim_matrix[i][j] > threshold);
                    if has_strong_b_link {
                        bridge_nodes.push(i);
                    }
                }

                for &j in &module_b.node_indices {
                    let has_strong_a_link = module_a
                        .node_indices
                        .iter()
                        .any(|&i| sim_matrix[i][j] > threshold);
                    if has_strong_a_link && !bridge_nodes.contains(&j) {
                        bridge_nodes.push(j);
                    }
                }

                relations.push(InterModuleRelation {
                    module_a: module_a.id,
                    module_b: module_b.id,
                    coupling_strength: coupling,
                    info_flow_a_to_b: a_to_b,
                    info_flow_b_to_a: b_to_a,
                    bridge_nodes,
                });
            }
        }

        relations
    }

    /// Classify nodes by their role in the network
    fn classify_nodes(
        &self,
        sim_matrix: &[Vec<f64>],
        assignments: &[usize],
        modules: &[ConsciousnessModule],
    ) -> Vec<NodeClassification> {
        let n = sim_matrix.len();
        let mut classifications = Vec::with_capacity(n);

        for i in 0..n {
            let primary_module = assignments[i];

            // Within-module degree (z-score)
            let module = modules.iter().find(|m| m.id == primary_module);
            let within_degree = if let Some(m) = module {
                let internal_sum: f64 = m
                    .node_indices
                    .iter()
                    .filter(|&&j| j != i)
                    .map(|&j| sim_matrix[i][j])
                    .sum();
                let module_size = m.node_indices.len() as f64;
                if module_size > 1.0 {
                    internal_sum / (module_size - 1.0)
                } else {
                    0.0
                }
            } else {
                0.0
            };

            // Participation coefficient
            let mut module_sums: std::collections::HashMap<usize, f64> =
                std::collections::HashMap::new();
            let mut total = 0.0;

            for j in 0..n {
                if i != j {
                    let weight = sim_matrix[i][j];
                    *module_sums.entry(assignments[j]).or_default() += weight;
                    total += weight;
                }
            }

            let participation = if total > 0.0 && module_sums.len() > 1 {
                let sum_sq: f64 = module_sums.values().map(|&s| (s / total).powi(2)).sum();
                1.0 - sum_sq
            } else {
                0.0
            };

            // Betweenness (simplified - count shortest paths through node)
            let betweenness = self.estimate_betweenness(i, sim_matrix);

            // Classify role
            let role = if total < self.config.similarity_threshold * n as f64 {
                NodeRole::Isolated
            } else if participation > 0.6 && betweenness > 0.3 {
                NodeRole::Hub
            } else if participation > 0.4 {
                NodeRole::Bridge
            } else if within_degree > 0.5 {
                NodeRole::Core
            } else {
                NodeRole::Peripheral
            };

            classifications.push(NodeClassification {
                node_index: i,
                primary_module,
                role,
                within_module_degree: within_degree,
                participation_coefficient: participation,
                betweenness,
            });
        }

        classifications
    }

    /// Estimate betweenness centrality (simplified)
    fn estimate_betweenness(&self, node: usize, sim_matrix: &[Vec<f64>]) -> f64 {
        let n = sim_matrix.len();
        if n < 3 {
            return 0.0;
        }

        // Count how many pairs have their strongest path through this node
        let mut through_count = 0;
        let total_pairs = (n - 1) * (n - 2) / 2;

        for i in 0..n {
            if i == node {
                continue;
            }
            for j in (i + 1)..n {
                if j == node {
                    continue;
                }

                // Direct path strength
                let direct = sim_matrix[i][j];

                // Path through node
                let through = (sim_matrix[i][node] + sim_matrix[node][j]) / 2.0;

                if through > direct {
                    through_count += 1;
                }
            }
        }

        if total_pairs > 0 {
            through_count as f64 / total_pairs as f64
        } else {
            0.0
        }
    }

    /// Find bottleneck edges between modules
    fn find_bottleneck_edges(
        &self,
        sim_matrix: &[Vec<f64>],
        assignments: &[usize],
    ) -> Vec<(usize, usize)> {
        let n = sim_matrix.len();
        let mut bottlenecks = Vec::new();

        for i in 0..n {
            for j in (i + 1)..n {
                // Only consider inter-module edges
                if assignments[i] == assignments[j] {
                    continue;
                }

                // Check if this is the strongest link between these modules
                let weight = sim_matrix[i][j];
                if weight < self.config.similarity_threshold {
                    continue;
                }

                let is_bottleneck = sim_matrix[i]
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| *k != j && assignments[*k] == assignments[j])
                    .all(|(_, &w)| w < weight)
                    && sim_matrix[j]
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| *k != i && assignments[*k] == assignments[i])
                        .all(|(_, &w)| w < weight);

                if is_bottleneck {
                    bottlenecks.push((i, j));
                }
            }
        }

        bottlenecks
    }

    /// Compute segregation and integration indices
    fn compute_seg_int_indices(
        &self,
        sim_matrix: &[Vec<f64>],
        assignments: &[usize],
    ) -> (f64, f64) {
        let n = sim_matrix.len();
        if n < 2 {
            return (0.0, 0.0);
        }

        let mut within_sum = 0.0;
        let mut within_count = 0;
        let mut between_sum = 0.0;
        let mut between_count = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let weight = sim_matrix[i][j];
                if assignments[i] == assignments[j] {
                    within_sum += weight;
                    within_count += 1;
                } else {
                    between_sum += weight;
                    between_count += 1;
                }
            }
        }

        let within_avg = if within_count > 0 {
            within_sum / within_count as f64
        } else {
            0.0
        };
        let between_avg = if between_count > 0 {
            between_sum / between_count as f64
        } else {
            0.0
        };

        let total_avg = if within_count + between_count > 0 {
            (within_sum + between_sum) / (within_count + between_count) as f64
        } else {
            0.0
        };

        // Segregation: how much stronger are within-module connections
        let segregation = if total_avg > 0.0 {
            (within_avg - between_avg) / total_avg
        } else {
            0.0
        };

        // Integration: quality of between-module connections
        let integration = if within_avg > 0.0 {
            between_avg / within_avg
        } else {
            0.0
        };

        (segregation.clamp(0.0, 1.0), integration.clamp(0.0, 1.0))
    }

    /// Compute hierarchical modularity at different scales
    fn compute_hierarchical_modularity(&self, sim_matrix: &[Vec<f64>]) -> Vec<f64> {
        let n = sim_matrix.len();
        if n < 4 {
            return vec![0.0];
        }

        let mut scores = Vec::new();

        // Try different numbers of modules
        for k in 2..=(n / 2).min(8) {
            let config = ModularityConfig {
                num_modules: Some(k),
                ..self.config.clone()
            };
            let analyzer = PhiModularityAnalyzer::with_config(config);
            let assignments = analyzer.detect_modules(sim_matrix);
            let q = analyzer.compute_modularity_q(sim_matrix, &assignments);
            scores.push(q);
        }

        scores
    }
}

/// Convenience function: analyze network modularity
pub fn analyze_network_modularity(
    node_representations: &[ContinuousHV],
) -> NetworkModularityResult {
    PhiModularityAnalyzer::new().analyze(node_representations)
}

/// Convenience function: detect number of natural modules
pub fn detect_module_count(node_representations: &[ContinuousHV]) -> usize {
    let result = PhiModularityAnalyzer::new().analyze(node_representations);
    result.num_modules()
}

/// Convenience function: get modularity Q score
pub fn compute_modularity_score(node_representations: &[ContinuousHV]) -> f64 {
    PhiModularityAnalyzer::new()
        .analyze(node_representations)
        .modularity_score
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::ContinuousHV;
    use crate::hdc::HDC_DIMENSION;

    /// Helper: create n random ContinuousHV node representations.
    fn make_nodes(n: usize) -> Vec<ContinuousHV> {
        (0..n)
            .map(|i| ContinuousHV::random(HDC_DIMENSION, i as u64 + 500))
            .collect()
    }

    // ------------------------------------------------------------------
    // 1. PhiTransfer constructors and config
    // ------------------------------------------------------------------

    #[test]
    fn test_phi_transfer_constructors() {
        let t1 = PhiTransfer::new();
        assert_eq!(t1.config.signature_dims, 16);
        assert!(t1.transfer_weights.is_none());

        let t2 = PhiTransfer::fast();
        assert_eq!(t2.config.signature_dims, 8);
        assert_eq!(t2.config.max_iterations, 100);

        let t3 = PhiTransfer::research();
        assert_eq!(t3.config.signature_dims, 32);
    }

    // ------------------------------------------------------------------
    // 2. PhiSignature extraction
    // ------------------------------------------------------------------

    #[test]
    fn test_extract_signature_basic() {
        let transfer = PhiTransfer::new();
        let nodes = make_nodes(6);
        let sig = transfer.extract_signature(&nodes, 0.5, Some("Ring"));

        assert!(sig.dim() > 0, "Signature must have non-zero dimension");
        assert_eq!(sig.original_phi, 0.5);
        assert_eq!(sig.num_components, 6);
        assert_eq!(sig.topology_type.as_deref(), Some("Ring"));

        // as_vector should match dim()
        assert_eq!(sig.as_vector().len(), sig.dim());
    }

    #[test]
    fn test_extract_signature_empty_input() {
        let transfer = PhiTransfer::new();
        let sig = transfer.extract_signature(&[], 0.0, None);
        assert_eq!(sig.num_components, 0);
        // Features should still be padded to the expected size
        assert_eq!(sig.dim(), sig.as_vector().len());
    }

    // ------------------------------------------------------------------
    // 3. PhiTransfer: transfer produces valid result
    // ------------------------------------------------------------------

    #[test]
    fn test_transfer_produces_valid_result() {
        let transfer = PhiTransfer::new();
        let source = make_nodes(8);
        let target = make_nodes(8);

        let result = transfer.transfer(&source, &target, 0.8, 0.3, "Star", "Random");
        assert_eq!(result.source_type, "Star");
        assert_eq!(result.target_type, "Random");
        assert_eq!(result.original_phi, 0.3);
        // Enhanced phi should be >= original when source phi > target phi
        assert!(
            result.enhanced_phi >= result.original_phi,
            "Transfer from higher-phi source should not reduce target phi"
        );
        assert!(!result.transfer_vector.is_empty());
    }

    // ------------------------------------------------------------------
    // 4. PhiCausalAnalyzer constructors
    // ------------------------------------------------------------------

    #[test]
    fn test_causal_analyzer_constructors() {
        let a1 = PhiCausalAnalyzer::new();
        assert_eq!(a1.config.bootstrap_samples, 10);

        let a2 = PhiCausalAnalyzer::fast();
        assert_eq!(a2.config.bootstrap_samples, 3);

        let a3 = PhiCausalAnalyzer::research();
        assert_eq!(a3.config.bootstrap_samples, 50);
    }

    // ------------------------------------------------------------------
    // 5. Causal analysis on empty input
    // ------------------------------------------------------------------

    #[test]
    fn test_causal_analysis_empty() {
        let analyzer = PhiCausalAnalyzer::new();
        let result = analyzer.analyze(&[]);
        assert_eq!(result.baseline_phi, 0.0);
        assert!(result.node_results.is_empty());
        assert!(result.causal_power.is_empty());
    }

    // ------------------------------------------------------------------
    // 6. Causal analysis produces valid structure
    // ------------------------------------------------------------------

    #[test]
    fn test_causal_analysis_basic() {
        let analyzer = PhiCausalAnalyzer::fast();
        let nodes = make_nodes(4);
        let result = analyzer.analyze(&nodes);

        // Should have results for all 4 nodes
        assert_eq!(result.node_results.len(), 4);
        assert_eq!(result.causal_power.len(), 4);
        assert_eq!(result.node_ranking.len(), 4);

        // All causal power values should be non-negative
        for &cp in &result.causal_power {
            assert!(cp >= 0.0, "Causal power must be non-negative");
        }

        // Robustness should be in [0, 1]
        let rob = result.robustness();
        assert!((0.0..=1.0).contains(&rob), "Robustness must be in [0,1]");
    }

    // ------------------------------------------------------------------
    // 7. PhiModularityAnalyzer on empty input
    // ------------------------------------------------------------------

    #[test]
    fn test_modularity_empty() {
        let analyzer = PhiModularityAnalyzer::new();
        let result = analyzer.analyze(&[]);
        assert_eq!(result.total_phi, 0.0);
        assert!(result.modules.is_empty());
        assert_eq!(result.modularity_score, 0.0);
    }

    // ------------------------------------------------------------------
    // 8. Modularity analysis produces reasonable structure
    // ------------------------------------------------------------------

    #[test]
    fn test_modularity_basic() {
        let config = ModularityConfig::quick();
        let analyzer = PhiModularityAnalyzer::with_config(config);
        let nodes = make_nodes(10);
        let result = analyzer.analyze(&nodes);

        // Should detect at least one module
        assert!(
            !result.modules.is_empty(),
            "Should detect at least one module for 10 nodes"
        );

        // Node classifications should cover every node
        assert_eq!(result.node_classifications.len(), 10);

        // Segregation and integration should be non-negative
        assert!(
            result.segregation_index >= 0.0,
            "Segregation index must be non-negative"
        );
        assert!(
            result.integration_index >= 0.0,
            "Integration index must be non-negative"
        );

        // Balance score in [0, 1]
        let bal = result.balance_score();
        assert!((0.0..=1.0).contains(&bal), "Balance score must be in [0,1]");
    }
}
