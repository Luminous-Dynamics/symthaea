//! Cross-Modal Binding Mechanisms for Hyperdimensional Computing
//!
//! This module implements sophisticated cross-modal binding operations that allow
//! information from different modalities (visual, auditory, semantic, temporal, etc.)
//! to be bound together into coherent hyperdimensional representations.
//!
//! # Key Concepts
//!
//! ## Multi-Modal Fusion
//! Different sensory modalities have fundamentally different statistical properties.
//! Our approach uses modality-specific projection matrices and learned alignment
//! to create a unified representation space.
//!
//! ## Binding Operators
//! - **Symmetric Binding**: Order-invariant binding for associative memories
//! - **Asymmetric Binding**: Order-preserving binding for sequential/relational data
//! - **Hierarchical Binding**: Nested binding for compositional structures
//! - **Attentional Binding**: Weighted binding based on relevance/salience
//!
//! # Scientific Foundation
//!
//! Based on research in:
//! - Vector Symbolic Architectures (VSA) for multi-modal fusion
//! - Binding problem in neuroscience (how brain combines features)
//! - Cross-modal attention mechanisms in transformers
//! - Holographic reduced representations (HRR)
//!
//! # Example
//!
//! ```ignore
//! use symthaea::hdc::cross_modal_binding::{CrossModalBinder, Modality};
//!
//! let binder = CrossModalBinder::new(16384);
//!
//! // Bind visual and auditory features
//! let visual = some_visual_representation;
//! let auditory = some_auditory_representation;
//! let bound = binder.bind_symmetric(&visual, &auditory, Modality::Visual, Modality::Auditory);
//! ```

use crate::hdc::real_hv::RealHV as ContinuousHV;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Represents different sensory/cognitive modalities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Modality {
    /// Visual/spatial information
    Visual,
    /// Auditory/temporal information
    Auditory,
    /// Semantic/conceptual information
    Semantic,
    /// Temporal/sequential information
    Temporal,
    /// Proprioceptive/embodied information
    Proprioceptive,
    /// Emotional/valence information
    Emotional,
    /// Abstract/symbolic information
    Symbolic,
    /// Motor/action information
    Motor,
}

impl Modality {
    /// Get a unique seed offset for this modality (for projection matrix generation)
    pub fn seed_offset(&self) -> u64 {
        match self {
            Modality::Visual => 1000,
            Modality::Auditory => 2000,
            Modality::Semantic => 3000,
            Modality::Temporal => 4000,
            Modality::Proprioceptive => 5000,
            Modality::Emotional => 6000,
            Modality::Symbolic => 7000,
            Modality::Motor => 8000,
        }
    }

    /// Get the characteristic dimensionality weighting for this modality
    /// (some modalities naturally have higher/lower intrinsic dimensions)
    pub fn dimension_weight(&self) -> f32 {
        match self {
            Modality::Visual => 1.0,        // Full spatial complexity
            Modality::Auditory => 0.8,      // Temporal but lower spatial
            Modality::Semantic => 1.0,      // Full conceptual space
            Modality::Temporal => 0.6,      // Lower dimensional (sequential)
            Modality::Proprioceptive => 0.7, // Body space
            Modality::Emotional => 0.5,     // Valence/arousal dimensions
            Modality::Symbolic => 0.9,      // Discrete but high variety
            Modality::Motor => 0.7,         // Action space
        }
    }
}

/// Configuration for cross-modal binding operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalConfig {
    /// Hypervector dimension
    pub dimension: usize,
    /// Base seed for reproducibility
    pub base_seed: u64,
    /// Whether to learn modality alignments
    pub learn_alignment: bool,
    /// Alignment learning rate
    pub alignment_lr: f32,
    /// Temperature for soft binding operations
    pub temperature: f32,
    /// Whether to normalize intermediate results
    pub normalize_intermediate: bool,
}

impl Default for CrossModalConfig {
    fn default() -> Self {
        Self {
            dimension: super::HDC_DIMENSION,
            base_seed: 42,
            learn_alignment: true,
            alignment_lr: 0.01,
            temperature: 1.0,
            normalize_intermediate: true,
        }
    }
}

/// Learned alignment parameters between modality pairs
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModalityAlignment {
    /// Source modality
    pub source: Modality,
    /// Target modality
    pub target: Modality,
    /// Alignment strength (learned)
    pub strength: f32,
    /// Phase offset for binding (learned)
    pub phase_offset: f32,
    /// Number of alignment updates
    pub update_count: usize,
}

impl ModalityAlignment {
    pub fn new(source: Modality, target: Modality) -> Self {
        Self {
            source,
            target,
            strength: 1.0,
            phase_offset: 0.0,
            update_count: 0,
        }
    }

    /// Update alignment based on binding success
    pub fn update(&mut self, success_signal: f32, lr: f32) {
        // Simple gradient update
        self.strength += lr * success_signal * (1.0 - self.strength);
        self.strength = self.strength.clamp(0.1, 2.0);
        self.update_count += 1;
    }
}

/// Result of a cross-modal binding operation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingResult {
    /// The bound hypervector
    pub bound_hv: ContinuousHV,
    /// Coherence measure (how well modalities aligned)
    pub coherence: f32,
    /// Information preserved from each modality
    pub preservation: HashMap<Modality, f32>,
    /// Binding type used
    pub binding_type: BindingType,
}

/// Types of binding operations
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum BindingType {
    /// Order-invariant symmetric binding
    Symmetric,
    /// Order-preserving asymmetric binding
    Asymmetric,
    /// Nested hierarchical binding
    Hierarchical,
    /// Attention-weighted binding
    Attentional,
    /// Circular convolution (HRR-style)
    Convolution,
}

/// Cross-modal binding system for hyperdimensional computing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossModalBinder {
    /// Configuration
    config: CrossModalConfig,
    /// Modality-specific projection bases
    modality_bases: HashMap<Modality, ContinuousHV>,
    /// Learned alignments between modality pairs
    alignments: HashMap<(Modality, Modality), ModalityAlignment>,
    /// Role vectors for asymmetric binding
    role_vectors: Vec<ContinuousHV>,
}

impl CrossModalBinder {
    /// Create a new cross-modal binder with default configuration
    pub fn new(dimension: usize) -> Self {
        let config = CrossModalConfig {
            dimension,
            ..Default::default()
        };
        Self::with_config(config)
    }

    /// Create a new cross-modal binder with custom configuration
    pub fn with_config(config: CrossModalConfig) -> Self {
        // Generate modality-specific basis vectors
        let mut modality_bases = HashMap::new();
        for modality in [
            Modality::Visual,
            Modality::Auditory,
            Modality::Semantic,
            Modality::Temporal,
            Modality::Proprioceptive,
            Modality::Emotional,
            Modality::Symbolic,
            Modality::Motor,
        ] {
            let seed = config.base_seed + modality.seed_offset();
            let basis = ContinuousHV::random(config.dimension, seed);
            modality_bases.insert(modality, basis);
        }

        // Generate role vectors for asymmetric binding (up to 16 roles)
        let role_vectors: Vec<ContinuousHV> = (0..16)
            .map(|i| ContinuousHV::random(config.dimension, config.base_seed + 10000 + i))
            .collect();

        Self {
            config,
            modality_bases,
            alignments: HashMap::new(),
            role_vectors,
        }
    }

    /// Get modality-specific projection of a hypervector
    fn project_to_modality(&self, hv: &ContinuousHV, modality: Modality) -> ContinuousHV {
        let basis = self.modality_bases.get(&modality)
            .expect("All modalities should have bases");

        // Project by binding with modality basis (creates modality-specific subspace)
        let projected = hv.bind(basis);

        // Apply modality weight
        let weight = modality.dimension_weight();
        projected.scale(weight)
    }

    /// Symmetric binding: order-invariant combination of two modalities
    ///
    /// Uses element-wise multiplication followed by normalization.
    /// The result is identical regardless of argument order.
    pub fn bind_symmetric(
        &self,
        hv1: &ContinuousHV,
        hv2: &ContinuousHV,
        mod1: Modality,
        mod2: Modality,
    ) -> BindingResult {
        // Project to modality-specific subspaces
        let proj1 = self.project_to_modality(hv1, mod1);
        let proj2 = self.project_to_modality(hv2, mod2);

        // Symmetric binding via element-wise multiplication
        let bound = proj1.bind(&proj2);

        // Normalize if configured
        let bound_hv = if self.config.normalize_intermediate {
            bound.normalize()
        } else {
            bound
        };

        // Measure coherence (how well the binding preserved information)
        let coherence = self.measure_coherence(&bound_hv, &[(&proj1, mod1), (&proj2, mod2)]);

        // Measure preservation for each modality
        let mut preservation = HashMap::new();
        preservation.insert(mod1, bound_hv.similarity(&proj1).abs());
        preservation.insert(mod2, bound_hv.similarity(&proj2).abs());

        BindingResult {
            bound_hv,
            coherence,
            preservation,
            binding_type: BindingType::Symmetric,
        }
    }

    /// Asymmetric binding: order-preserving combination with role markers
    ///
    /// Uses role vectors to mark the position/role of each element,
    /// preserving sequential or relational structure.
    pub fn bind_asymmetric(
        &self,
        hv1: &ContinuousHV,
        hv2: &ContinuousHV,
        mod1: Modality,
        mod2: Modality,
    ) -> BindingResult {
        // Project to modality-specific subspaces
        let proj1 = self.project_to_modality(hv1, mod1);
        let proj2 = self.project_to_modality(hv2, mod2);

        // Bind with role vectors to mark positions
        let role1 = &self.role_vectors[0];
        let role2 = &self.role_vectors[1];

        let marked1 = proj1.bind(role1);
        let marked2 = proj2.bind(role2);

        // Bundle the marked vectors
        let bound = ContinuousHV::bundle(&[marked1.clone(), marked2.clone()]);

        let bound_hv = if self.config.normalize_intermediate {
            bound.normalize()
        } else {
            bound
        };

        // Measure coherence
        let coherence = self.measure_coherence(&bound_hv, &[(&proj1, mod1), (&proj2, mod2)]);

        // Measure preservation
        let mut preservation = HashMap::new();
        preservation.insert(mod1, bound_hv.similarity(&marked1).abs());
        preservation.insert(mod2, bound_hv.similarity(&marked2).abs());

        BindingResult {
            bound_hv,
            coherence,
            preservation,
            binding_type: BindingType::Asymmetric,
        }
    }

    /// Hierarchical binding: nested structure for compositional semantics
    ///
    /// Binds multiple levels of structure, preserving constituency relations.
    pub fn bind_hierarchical(
        &self,
        constituents: &[(ContinuousHV, Modality)],
        structure: &HierarchicalStructure,
    ) -> BindingResult {
        if constituents.is_empty() {
            return BindingResult {
                bound_hv: ContinuousHV::zero(self.config.dimension),
                coherence: 0.0,
                preservation: HashMap::new(),
                binding_type: BindingType::Hierarchical,
            };
        }

        // Project all constituents
        let projected: Vec<(ContinuousHV, Modality)> = constituents
            .iter()
            .map(|(hv, mod_)| (self.project_to_modality(hv, *mod_), *mod_))
            .collect();

        // Build hierarchical structure recursively
        let bound_hv = self.build_hierarchy(&projected, structure, 0);

        let bound_hv = if self.config.normalize_intermediate {
            bound_hv.normalize()
        } else {
            bound_hv
        };

        // Measure coherence and preservation
        let refs: Vec<(&ContinuousHV, Modality)> = projected.iter()
            .map(|(hv, m)| (hv, *m))
            .collect();
        let coherence = self.measure_coherence(&bound_hv, &refs);

        let mut preservation = HashMap::new();
        for (hv, mod_) in &projected {
            let sim = bound_hv.similarity(hv).abs();
            preservation.entry(*mod_)
                .and_modify(|v| *v = (*v + sim) / 2.0)
                .or_insert(sim);
        }

        BindingResult {
            bound_hv,
            coherence,
            preservation,
            binding_type: BindingType::Hierarchical,
        }
    }

    /// Attentional binding: weighted combination based on relevance scores
    ///
    /// Uses attention weights to emphasize important modalities.
    pub fn bind_attentional(
        &self,
        inputs: &[(ContinuousHV, Modality, f32)], // (hv, modality, attention_weight)
    ) -> BindingResult {
        if inputs.is_empty() {
            return BindingResult {
                bound_hv: ContinuousHV::zero(self.config.dimension),
                coherence: 0.0,
                preservation: HashMap::new(),
                binding_type: BindingType::Attentional,
            };
        }

        // Apply softmax to attention weights with temperature
        let weights: Vec<f32> = inputs.iter().map(|(_, _, w)| *w).collect();
        let softmax_weights = self.softmax(&weights);

        // Project and weight each input
        let weighted_hvs: Vec<ContinuousHV> = inputs
            .iter()
            .zip(softmax_weights.iter())
            .map(|((hv, mod_, _), w)| {
                let projected = self.project_to_modality(hv, *mod_);
                projected.scale(*w)
            })
            .collect();

        // Bundle weighted vectors
        let bound = ContinuousHV::bundle(&weighted_hvs);

        let bound_hv = if self.config.normalize_intermediate {
            bound.normalize()
        } else {
            bound
        };

        // Calculate coherence and preservation
        let refs: Vec<(&ContinuousHV, Modality)> = inputs.iter()
            .map(|(hv, m, _)| (hv, *m))
            .collect();
        let coherence = self.measure_coherence(&bound_hv, &refs);

        let mut preservation = HashMap::new();
        for ((hv, mod_, _), w) in inputs.iter().zip(softmax_weights.iter()) {
            let projected = self.project_to_modality(hv, *mod_);
            let sim = bound_hv.similarity(&projected).abs() * w;
            preservation.entry(*mod_)
                .and_modify(|v| *v += sim)
                .or_insert(sim);
        }

        BindingResult {
            bound_hv,
            coherence,
            preservation,
            binding_type: BindingType::Attentional,
        }
    }

    /// Circular convolution binding (HRR-style)
    ///
    /// Uses FFT-based circular convolution for binding.
    /// More computationally expensive but has nice mathematical properties.
    pub fn bind_convolution(
        &self,
        hv1: &ContinuousHV,
        hv2: &ContinuousHV,
        mod1: Modality,
        mod2: Modality,
    ) -> BindingResult {
        let proj1 = self.project_to_modality(hv1, mod1);
        let proj2 = self.project_to_modality(hv2, mod2);

        // Circular convolution in the spatial domain
        // (For efficiency, this should use FFT, but we implement direct convolution for clarity)
        let n = proj1.values.len();
        let mut result = vec![0.0f32; n];

        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..n {
                let idx = (i + n - j) % n;
                sum += proj1.values[j] * proj2.values[idx];
            }
            result[i] = sum / (n as f32).sqrt();
        }

        let bound_hv = ContinuousHV { values: result };
        let bound_hv = if self.config.normalize_intermediate {
            bound_hv.normalize()
        } else {
            bound_hv
        };

        let coherence = self.measure_coherence(&bound_hv, &[(&proj1, mod1), (&proj2, mod2)]);

        let mut preservation = HashMap::new();
        preservation.insert(mod1, bound_hv.similarity(&proj1).abs());
        preservation.insert(mod2, bound_hv.similarity(&proj2).abs());

        BindingResult {
            bound_hv,
            coherence,
            preservation,
            binding_type: BindingType::Convolution,
        }
    }

    /// Unbind: attempt to recover one component from a bound representation
    pub fn unbind(
        &self,
        bound: &ContinuousHV,
        known: &ContinuousHV,
        known_modality: Modality,
        binding_type: BindingType,
    ) -> ContinuousHV {
        let projected_known = self.project_to_modality(known, known_modality);

        match binding_type {
            BindingType::Symmetric | BindingType::Hierarchical => {
                // For symmetric binding, unbind by multiplying with inverse
                let inverse = projected_known.inverse();
                bound.bind(&inverse)
            }
            BindingType::Asymmetric => {
                // For asymmetric, need to unbind role marker too
                let role = &self.role_vectors[0]; // Assume first role
                let role_inv = role.inverse();
                let unrolled = bound.bind(&role_inv);
                let known_inv = projected_known.inverse();
                unrolled.bind(&known_inv)
            }
            BindingType::Attentional => {
                // Attentional uses bundling, so unbinding is approximate
                bound.bind(&projected_known.inverse())
            }
            BindingType::Convolution => {
                // Circular correlation (inverse of convolution)
                self.circular_correlation(bound, &projected_known)
            }
        }
    }

    /// Update alignment based on binding success/failure feedback
    pub fn update_alignment(&mut self, mod1: Modality, mod2: Modality, success_signal: f32) {
        let key = (mod1, mod2);
        let alignment = self.alignments
            .entry(key)
            .or_insert_with(|| ModalityAlignment::new(mod1, mod2));

        alignment.update(success_signal, self.config.alignment_lr);
    }

    /// Measure coherence of a binding (how well information was integrated)
    fn measure_coherence(&self, bound: &ContinuousHV, sources: &[(&ContinuousHV, Modality)]) -> f32 {
        if sources.is_empty() {
            return 0.0;
        }

        // Average similarity to source projections, weighted by modality importance
        let mut total_sim = 0.0f32;
        let mut total_weight = 0.0f32;

        for (hv, mod_) in sources {
            let weight = mod_.dimension_weight();
            let sim = bound.similarity(hv).abs();
            total_sim += sim * weight;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            total_sim / total_weight
        } else {
            0.0
        }
    }

    /// Softmax with temperature
    fn softmax(&self, values: &[f32]) -> Vec<f32> {
        let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_vals: Vec<f32> = values
            .iter()
            .map(|v| ((v - max_val) / self.config.temperature).exp())
            .collect();
        let sum: f32 = exp_vals.iter().sum();
        exp_vals.iter().map(|v| v / sum).collect()
    }

    /// Build hierarchical structure recursively
    fn build_hierarchy(
        &self,
        constituents: &[(ContinuousHV, Modality)],
        structure: &HierarchicalStructure,
        depth: usize,
    ) -> ContinuousHV {
        match structure {
            HierarchicalStructure::Leaf(idx) => {
                if *idx < constituents.len() {
                    constituents[*idx].0.clone()
                } else {
                    ContinuousHV::zero(self.config.dimension)
                }
            }
            HierarchicalStructure::Branch(children) => {
                // Recursively build children
                let child_hvs: Vec<ContinuousHV> = children
                    .iter()
                    .map(|child| self.build_hierarchy(constituents, child, depth + 1))
                    .collect();

                // Bind children with depth-specific role
                if child_hvs.is_empty() {
                    return ContinuousHV::zero(self.config.dimension);
                }

                // Use depth as role index
                let role_idx = depth.min(self.role_vectors.len() - 1);
                let role = &self.role_vectors[role_idx];

                // Bundle children and bind with role
                let bundled = ContinuousHV::bundle(&child_hvs);
                bundled.bind(role)
            }
        }
    }

    /// Circular correlation (inverse of circular convolution)
    fn circular_correlation(&self, a: &ContinuousHV, b: &ContinuousHV) -> ContinuousHV {
        let n = a.values.len();
        let mut result = vec![0.0f32; n];

        for i in 0..n {
            let mut sum = 0.0f32;
            for j in 0..n {
                let idx = (j + i) % n;
                sum += a.values[j] * b.values[idx];
            }
            result[i] = sum / (n as f32).sqrt();
        }

        ContinuousHV { values: result }
    }
}

/// Hierarchical structure for compositional binding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HierarchicalStructure {
    /// Leaf node referencing constituent index
    Leaf(usize),
    /// Branch node with children
    Branch(Vec<HierarchicalStructure>),
}

impl HierarchicalStructure {
    /// Create a flat structure (all leaves under one branch)
    pub fn flat(n: usize) -> Self {
        HierarchicalStructure::Branch(
            (0..n).map(HierarchicalStructure::Leaf).collect()
        )
    }

    /// Create a binary tree structure
    pub fn binary_tree(n: usize) -> Self {
        fn build(start: usize, end: usize) -> HierarchicalStructure {
            if start >= end {
                return HierarchicalStructure::Leaf(start);
            }
            if end - start == 1 {
                return HierarchicalStructure::Leaf(start);
            }
            let mid = (start + end) / 2;
            HierarchicalStructure::Branch(vec![
                build(start, mid),
                build(mid, end),
            ])
        }
        build(0, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_hv(dim: usize, seed: u64) -> ContinuousHV {
        ContinuousHV::random(dim, seed)
    }

    #[test]
    fn test_symmetric_binding_is_commutative() {
        let binder = CrossModalBinder::new(2048);
        let hv1 = create_test_hv(2048, 1);
        let hv2 = create_test_hv(2048, 2);

        let result1 = binder.bind_symmetric(&hv1, &hv2, Modality::Visual, Modality::Auditory);
        let result2 = binder.bind_symmetric(&hv2, &hv1, Modality::Auditory, Modality::Visual);

        // Symmetric binding should be approximately commutative
        // (not exact due to normalization differences)
        let sim = result1.bound_hv.similarity(&result2.bound_hv);
        assert!(sim > 0.8, "Symmetric binding should be approximately commutative, got {}", sim);
    }

    #[test]
    fn test_asymmetric_binding_preserves_order() {
        let binder = CrossModalBinder::new(2048);
        let hv1 = create_test_hv(2048, 1);
        let hv2 = create_test_hv(2048, 2);

        let result1 = binder.bind_asymmetric(&hv1, &hv2, Modality::Visual, Modality::Auditory);
        let result2 = binder.bind_asymmetric(&hv2, &hv1, Modality::Auditory, Modality::Visual);

        // Asymmetric binding should NOT be commutative
        let sim = result1.bound_hv.similarity(&result2.bound_hv);
        assert!(sim < 0.7, "Asymmetric binding should not be commutative, got {}", sim);
    }

    #[test]
    fn test_attentional_binding_respects_weights() {
        let binder = CrossModalBinder::new(2048);
        let hv1 = create_test_hv(2048, 1);
        let hv2 = create_test_hv(2048, 2);

        // Heavily weight hv1
        let result = binder.bind_attentional(&[
            (hv1.clone(), Modality::Visual, 10.0),
            (hv2.clone(), Modality::Auditory, 0.1),
        ]);

        // Result should be more similar to hv1's projection
        let proj1 = binder.project_to_modality(&hv1, Modality::Visual);
        let sim1 = result.bound_hv.similarity(&proj1).abs();

        assert!(sim1 > 0.5, "Attentional binding should favor heavily weighted input, got {}", sim1);
    }

    #[test]
    fn test_hierarchical_binding() {
        let binder = CrossModalBinder::new(2048);
        let constituents: Vec<(ContinuousHV, Modality)> = (0..4)
            .map(|i| (create_test_hv(2048, i), Modality::Semantic))
            .collect();

        let structure = HierarchicalStructure::binary_tree(4);
        let result = binder.bind_hierarchical(&constituents, &structure);

        assert!(result.coherence > 0.0, "Hierarchical binding should have positive coherence");
        assert!(result.bound_hv.values.iter().any(|v| *v != 0.0), "Result should not be zero");
    }

    #[test]
    fn test_unbinding_symmetric() {
        let binder = CrossModalBinder::new(2048);
        let hv1 = create_test_hv(2048, 1);
        let hv2 = create_test_hv(2048, 2);

        let bound = binder.bind_symmetric(&hv1, &hv2, Modality::Visual, Modality::Auditory);
        let recovered = binder.unbind(&bound.bound_hv, &hv1, Modality::Visual, BindingType::Symmetric);

        // The recovered vector should have some similarity to hv2's projection
        let proj2 = binder.project_to_modality(&hv2, Modality::Auditory);
        let sim = recovered.similarity(&proj2).abs();

        // Unbinding is approximate, but should show some recovery
        assert!(sim > 0.1, "Unbinding should partially recover the other component, got {}", sim);
    }

    #[test]
    fn test_modality_projection_consistency() {
        let binder = CrossModalBinder::new(2048);
        let hv = create_test_hv(2048, 42);

        // Same input with same modality should give same projection
        let proj1 = binder.project_to_modality(&hv, Modality::Visual);
        let proj2 = binder.project_to_modality(&hv, Modality::Visual);

        let sim = proj1.similarity(&proj2);
        assert!((sim - 1.0).abs() < 0.0001, "Same projection should be identical");

        // Different modalities should give different projections
        let proj3 = binder.project_to_modality(&hv, Modality::Auditory);
        let sim_diff = proj1.similarity(&proj3).abs();
        assert!(sim_diff < 0.5, "Different modality projections should differ, got {}", sim_diff);
    }

    #[test]
    fn test_convolution_binding() {
        let binder = CrossModalBinder::new(512); // Smaller for speed
        let hv1 = create_test_hv(512, 1);
        let hv2 = create_test_hv(512, 2);

        let result = binder.bind_convolution(&hv1, &hv2, Modality::Visual, Modality::Auditory);

        assert!(result.coherence > 0.0, "Convolution binding should have positive coherence");
        assert!(result.bound_hv.values.iter().any(|v| *v != 0.0), "Result should not be zero");
    }
}
