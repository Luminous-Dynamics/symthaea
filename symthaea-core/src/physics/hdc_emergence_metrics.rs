// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # HDC-Native Emergence Metrics
//!
//! This module provides structural measures intrinsic to HDC algebra, replacing
//! the need to measure similarity to labeled concept vectors ("qualia", "awareness").
//!
//! ## Core Insight
//!
//! Instead of asking "how similar is this to consciousness?", we measure:
//! - **Binding Depth**: How many bind operations deep is a composition?
//! - **Recoverability**: Can we unbind to recover original components?
//! - **Coherence**: Does bind create fundamentally different structure than bundle?
//! - **Compositional Information**: Entropy of the composition tree structure
//!
//! ## Theoretical Basis
//!
//! These metrics are grounded in HDC theory:
//! - Binding creates orthogonal structures (new information)
//! - Bundling creates superpositions (preserves information)
//! - Permutation encodes sequence/position
//!
//! The metrics measure structural properties without requiring semantic labels.

use crate::hdc::unified_hv::ContinuousHV;
use serde::{Deserialize, Serialize};

/// A node in the composition tree representing HDC operations
#[derive(Debug, Clone)]
pub enum CompositionNode {
    /// An atomic (leaf) vector with a label
    Atomic { label: String, vector: ContinuousHV },
    /// Binding of two nodes: left ⊗ right
    Bind {
        left: Box<CompositionNode>,
        right: Box<CompositionNode>,
        result: ContinuousHV,
    },
    /// Bundling of multiple nodes with weights
    Bundle {
        children: Vec<CompositionNode>,
        weights: Vec<f32>,
        result: ContinuousHV,
    },
    /// Permutation of a node by shift amount
    Permute {
        child: Box<CompositionNode>,
        shift: usize,
        result: ContinuousHV,
    },
}

impl CompositionNode {
    /// Create an atomic node
    pub fn atomic(label: impl Into<String>, vector: ContinuousHV) -> Self {
        CompositionNode::Atomic {
            label: label.into(),
            vector,
        }
    }

    /// Create a bind node
    pub fn bind(left: CompositionNode, right: CompositionNode) -> Self {
        let result = left.vector().bind(right.vector());
        CompositionNode::Bind {
            left: Box::new(left),
            right: Box::new(right),
            result,
        }
    }

    /// Create a bundle node
    pub fn bundle(children: Vec<CompositionNode>, weights: Vec<f32>) -> Self {
        let refs: Vec<&ContinuousHV> = children.iter().map(|c| c.vector()).collect();
        let result = ContinuousHV::weighted_bundle(&refs, &weights);
        CompositionNode::Bundle {
            children,
            weights,
            result,
        }
    }

    /// Create a bundle node with uniform weights
    pub fn bundle_uniform(children: Vec<CompositionNode>) -> Self {
        let n = children.len();
        let weights = vec![1.0; n];
        Self::bundle(children, weights)
    }

    /// Create a permute node
    pub fn permute(child: CompositionNode, shift: usize) -> Self {
        let result = child.vector().permute(shift);
        CompositionNode::Permute {
            child: Box::new(child),
            shift,
            result,
        }
    }

    /// Get the result vector for this node
    pub fn vector(&self) -> &ContinuousHV {
        match self {
            CompositionNode::Atomic { vector, .. } => vector,
            CompositionNode::Bind { result, .. } => result,
            CompositionNode::Bundle { result, .. } => result,
            CompositionNode::Permute { result, .. } => result,
        }
    }

    /// Check if this is an atomic node
    pub fn is_atomic(&self) -> bool {
        matches!(self, CompositionNode::Atomic { .. })
    }

    /// Get the label if this is an atomic node
    pub fn label(&self) -> Option<&str> {
        match self {
            CompositionNode::Atomic { label, .. } => Some(label),
            _ => None,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// METRIC 1: BINDING DEPTH
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of binding depth analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BindingDepthResult {
    /// Maximum path length through Bind nodes
    pub max_depth: usize,
    /// Average depth across all leaf paths
    pub avg_depth: f64,
    /// Total number of Bind operations in the tree
    pub bind_count: usize,
    /// Total number of nodes in the tree
    pub total_nodes: usize,
}

/// Analyzer for binding depth metrics
#[derive(Debug, Clone, Default)]
pub struct BindingDepthAnalyzer;

impl BindingDepthAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self
    }

    /// Compute the maximum binding depth (max path through Bind nodes)
    pub fn max_depth(&self, node: &CompositionNode) -> usize {
        match node {
            CompositionNode::Atomic { .. } => 0,
            CompositionNode::Bind { left, right, .. } => {
                1 + self.max_depth(left).max(self.max_depth(right))
            }
            CompositionNode::Bundle { children, .. } => children
                .iter()
                .map(|c| self.max_depth(c))
                .max()
                .unwrap_or(0),
            CompositionNode::Permute { child, .. } => self.max_depth(child),
        }
    }

    /// Compute the average depth across all leaf paths
    pub fn avg_depth(&self, node: &CompositionNode) -> f64 {
        let (total_depth, leaf_count) = self.depth_sum_and_count(node, 0);
        if leaf_count == 0 {
            0.0
        } else {
            total_depth as f64 / leaf_count as f64
        }
    }

    fn depth_sum_and_count(&self, node: &CompositionNode, current_depth: usize) -> (usize, usize) {
        match node {
            CompositionNode::Atomic { .. } => (current_depth, 1),
            CompositionNode::Bind { left, right, .. } => {
                let (sum_l, count_l) = self.depth_sum_and_count(left, current_depth + 1);
                let (sum_r, count_r) = self.depth_sum_and_count(right, current_depth + 1);
                (sum_l + sum_r, count_l + count_r)
            }
            CompositionNode::Bundle { children, .. } => {
                let mut total_sum = 0;
                let mut total_count = 0;
                for child in children {
                    let (sum, count) = self.depth_sum_and_count(child, current_depth);
                    total_sum += sum;
                    total_count += count;
                }
                (total_sum, total_count)
            }
            CompositionNode::Permute { child, .. } => {
                self.depth_sum_and_count(child, current_depth)
            }
        }
    }

    /// Count total Bind operations in the tree
    pub fn bind_count(&self, node: &CompositionNode) -> usize {
        match node {
            CompositionNode::Atomic { .. } => 0,
            CompositionNode::Bind { left, right, .. } => {
                1 + self.bind_count(left) + self.bind_count(right)
            }
            CompositionNode::Bundle { children, .. } => {
                children.iter().map(|c| self.bind_count(c)).sum()
            }
            CompositionNode::Permute { child, .. } => self.bind_count(child),
        }
    }

    /// Count total nodes in the tree
    pub fn total_nodes(&self, node: &CompositionNode) -> usize {
        match node {
            CompositionNode::Atomic { .. } => 1,
            CompositionNode::Bind { left, right, .. } => {
                1 + self.total_nodes(left) + self.total_nodes(right)
            }
            CompositionNode::Bundle { children, .. } => {
                1 + children.iter().map(|c| self.total_nodes(c)).sum::<usize>()
            }
            CompositionNode::Permute { child, .. } => 1 + self.total_nodes(child),
        }
    }

    /// Compute full binding depth analysis
    pub fn analyze(&self, node: &CompositionNode) -> BindingDepthResult {
        BindingDepthResult {
            max_depth: self.max_depth(node),
            avg_depth: self.avg_depth(node),
            bind_count: self.bind_count(node),
            total_nodes: self.total_nodes(node),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// METRIC 2: RECOVERABILITY INDEX
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of recoverability analysis for a single binding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoverabilityResult {
    /// Similarity of recovered A to original A
    pub recovered_a_similarity: f32,
    /// Similarity of recovered B to original B
    pub recovered_b_similarity: f32,
    /// Average recoverability
    pub avg_recoverability: f32,
    /// Whether recovery is considered successful
    pub is_recoverable: bool,
}

/// Result of tree-wide recoverability analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeRecoverabilityResult {
    /// Average recoverability across all Bind nodes
    pub avg_recoverability: f32,
    /// Minimum recoverability found
    pub min_recoverability: f32,
    /// Maximum recoverability found
    pub max_recoverability: f32,
    /// Number of Bind nodes analyzed
    pub bind_count: usize,
    /// Number of recoverable bindings
    pub recoverable_count: usize,
}

/// Analyzer for binding recoverability
#[derive(Debug, Clone)]
pub struct RecoverabilityAnalyzer {
    /// Threshold for considering a binding recoverable
    pub success_threshold: f32,
}

impl Default for RecoverabilityAnalyzer {
    fn default() -> Self {
        Self {
            success_threshold: 0.7,
        }
    }
}

impl RecoverabilityAnalyzer {
    /// Create a new analyzer with default threshold
    pub fn new() -> Self {
        Self::default()
    }

    /// Create with custom success threshold
    pub fn with_threshold(threshold: f32) -> Self {
        Self {
            success_threshold: threshold,
        }
    }

    /// Compute recoverability for a single binding
    ///
    /// Given A ⊗ B = C, compute:
    /// - similarity(C ⊗ B, A) - can we recover A?
    /// - similarity(C ⊗ A, B) - can we recover B?
    pub fn binding_recoverability(
        &self,
        a: &ContinuousHV,
        b: &ContinuousHV,
    ) -> RecoverabilityResult {
        // C = A ⊗ B
        let c = a.bind(b);

        // Try to recover A: C ⊗ B should ≈ A
        // (A ⊗ B) ⊗ B = A ⊗ (B ⊗ B)
        // For continuous HVs, B ⊗ B ≠ identity, but should still preserve some structure
        let recovered_a = c.bind(b);
        let recovered_a_similarity = recovered_a.similarity(a);

        // Try to recover B: C ⊗ A should ≈ B
        let recovered_b = c.bind(a);
        let recovered_b_similarity = recovered_b.similarity(b);

        let avg = (recovered_a_similarity + recovered_b_similarity) / 2.0;
        let is_recoverable = avg >= self.success_threshold;

        RecoverabilityResult {
            recovered_a_similarity,
            recovered_b_similarity,
            avg_recoverability: avg,
            is_recoverable,
        }
    }

    /// Analyze recoverability across all Bind nodes in a tree
    pub fn tree_recoverability(&self, node: &CompositionNode) -> TreeRecoverabilityResult {
        let mut recoverabilities = Vec::new();
        self.collect_recoverabilities(node, &mut recoverabilities);

        if recoverabilities.is_empty() {
            return TreeRecoverabilityResult {
                avg_recoverability: 1.0, // No bindings = fully recoverable
                min_recoverability: 1.0,
                max_recoverability: 1.0,
                bind_count: 0,
                recoverable_count: 0,
            };
        }

        let avg = recoverabilities.iter().sum::<f32>() / recoverabilities.len() as f32;
        let min = recoverabilities
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let max = recoverabilities
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let recoverable_count = recoverabilities
            .iter()
            .filter(|&&r| r >= self.success_threshold)
            .count();

        TreeRecoverabilityResult {
            avg_recoverability: avg,
            min_recoverability: min,
            max_recoverability: max,
            bind_count: recoverabilities.len(),
            recoverable_count,
        }
    }

    fn collect_recoverabilities(&self, node: &CompositionNode, results: &mut Vec<f32>) {
        match node {
            CompositionNode::Atomic { .. } => {}
            CompositionNode::Bind { left, right, .. } => {
                let result = self.binding_recoverability(left.vector(), right.vector());
                results.push(result.avg_recoverability);
                self.collect_recoverabilities(left, results);
                self.collect_recoverabilities(right, results);
            }
            CompositionNode::Bundle { children, .. } => {
                for child in children {
                    self.collect_recoverabilities(child, results);
                }
            }
            CompositionNode::Permute { child, .. } => {
                self.collect_recoverabilities(child, results);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// METRIC 3: BINDING COHERENCE
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of coherence analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoherenceResult {
    /// Coherence score: 1 - |similarity(A⊗B, bundle(A,B))|
    pub coherence: f32,
    /// Similarity between bound and bundled
    pub bound_bundle_similarity: f32,
    /// Similarity of bound to input A
    pub bound_a_similarity: f32,
    /// Similarity of bound to input B
    pub bound_b_similarity: f32,
}

/// Tree-wide coherence result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeCoherenceResult {
    /// Average coherence across all Bind nodes
    pub avg_coherence: f32,
    /// Minimum coherence found
    pub min_coherence: f32,
    /// Maximum coherence found
    pub max_coherence: f32,
    /// Number of Bind nodes analyzed
    pub bind_count: usize,
}

/// Analyzer for binding coherence
///
/// Coherence measures whether bind creates fundamentally different structure than bundle.
/// High coherence (→1) means bind creates orthogonal/new structure.
/// Low coherence (→0) means bind ≈ bundle (no new structure).
#[derive(Debug, Clone, Default)]
pub struct BindingCoherenceAnalyzer;

impl BindingCoherenceAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self
    }

    /// Compute coherence for a single binding
    ///
    /// Coherence = 1 - |similarity(A⊗B, bundle(A,B))|
    pub fn coherence(&self, a: &ContinuousHV, b: &ContinuousHV) -> f32 {
        let bound = a.bind(b);
        let bundled = ContinuousHV::bundle(&[a, b]);

        let sim = bound.similarity(&bundled).abs();
        1.0 - sim
    }

    /// Compute detailed coherence analysis
    pub fn detailed_coherence(&self, a: &ContinuousHV, b: &ContinuousHV) -> CoherenceResult {
        let bound = a.bind(b);
        let bundled = ContinuousHV::bundle(&[a, b]);

        let bound_bundle_similarity = bound.similarity(&bundled);
        let bound_a_similarity = bound.similarity(a);
        let bound_b_similarity = bound.similarity(b);

        CoherenceResult {
            coherence: 1.0 - bound_bundle_similarity.abs(),
            bound_bundle_similarity,
            bound_a_similarity,
            bound_b_similarity,
        }
    }

    /// Analyze coherence across all Bind nodes in a tree
    pub fn tree_coherence(&self, node: &CompositionNode) -> TreeCoherenceResult {
        let mut coherences = Vec::new();
        self.collect_coherences(node, &mut coherences);

        if coherences.is_empty() {
            return TreeCoherenceResult {
                avg_coherence: 0.0,
                min_coherence: 0.0,
                max_coherence: 0.0,
                bind_count: 0,
            };
        }

        let avg = coherences.iter().sum::<f32>() / coherences.len() as f32;
        let min = coherences.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = coherences.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        TreeCoherenceResult {
            avg_coherence: avg,
            min_coherence: min,
            max_coherence: max,
            bind_count: coherences.len(),
        }
    }

    fn collect_coherences(&self, node: &CompositionNode, results: &mut Vec<f32>) {
        match node {
            CompositionNode::Atomic { .. } => {}
            CompositionNode::Bind { left, right, .. } => {
                let coherence = self.coherence(left.vector(), right.vector());
                results.push(coherence);
                self.collect_coherences(left, results);
                self.collect_coherences(right, results);
            }
            CompositionNode::Bundle { children, .. } => {
                for child in children {
                    self.collect_coherences(child, results);
                }
            }
            CompositionNode::Permute { child, .. } => {
                self.collect_coherences(child, results);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// METRIC 4: COMPOSITIONAL INFORMATION
// ═══════════════════════════════════════════════════════════════════════════════

/// Node type counts
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NodeCounts {
    pub atomic: usize,
    pub bind: usize,
    pub bundle: usize,
    pub permute: usize,
}

impl NodeCounts {
    pub fn total(&self) -> usize {
        self.atomic + self.bind + self.bundle + self.permute
    }
}

/// Branching statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchingStats {
    /// Average number of children at Bundle nodes
    pub avg_bundle_width: f64,
    /// Maximum Bundle width
    pub max_bundle_width: usize,
    /// Total number of Bundle nodes
    pub bundle_count: usize,
}

/// Analyzer for compositional information
#[derive(Debug, Clone, Default)]
pub struct CompositionalInformationAnalyzer;

impl CompositionalInformationAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self
    }

    /// Count nodes by type
    pub fn node_counts(&self, node: &CompositionNode) -> NodeCounts {
        let mut counts = NodeCounts::default();
        self.count_nodes(node, &mut counts);
        counts
    }

    fn count_nodes(&self, node: &CompositionNode, counts: &mut NodeCounts) {
        match node {
            CompositionNode::Atomic { .. } => {
                counts.atomic += 1;
            }
            CompositionNode::Bind { left, right, .. } => {
                counts.bind += 1;
                self.count_nodes(left, counts);
                self.count_nodes(right, counts);
            }
            CompositionNode::Bundle { children, .. } => {
                counts.bundle += 1;
                for child in children {
                    self.count_nodes(child, counts);
                }
            }
            CompositionNode::Permute { child, .. } => {
                counts.permute += 1;
                self.count_nodes(child, counts);
            }
        }
    }

    /// Compute structural entropy of the composition tree
    ///
    /// H_struct = -Σ (n_i/N) log(n_i/N) where n_i is count of each node type
    pub fn structural_entropy(&self, node: &CompositionNode) -> f64 {
        let counts = self.node_counts(node);
        let total = counts.total() as f64;

        if total == 0.0 {
            return 0.0;
        }

        let type_counts = [
            counts.atomic as f64,
            counts.bind as f64,
            counts.bundle as f64,
            counts.permute as f64,
        ];

        let mut entropy = 0.0;
        for &count in &type_counts {
            if count > 0.0 {
                let p = count / total;
                entropy -= p * p.log2();
            }
        }

        entropy
    }

    /// Compute branching statistics for Bundle nodes
    pub fn branching_stats(&self, node: &CompositionNode) -> BranchingStats {
        let mut widths = Vec::new();
        self.collect_bundle_widths(node, &mut widths);

        if widths.is_empty() {
            return BranchingStats {
                avg_bundle_width: 0.0,
                max_bundle_width: 0,
                bundle_count: 0,
            };
        }

        let avg = widths.iter().sum::<usize>() as f64 / widths.len() as f64;
        let max = *widths.iter().max().unwrap_or(&0);

        BranchingStats {
            avg_bundle_width: avg,
            max_bundle_width: max,
            bundle_count: widths.len(),
        }
    }

    fn collect_bundle_widths(&self, node: &CompositionNode, widths: &mut Vec<usize>) {
        match node {
            CompositionNode::Atomic { .. } => {}
            CompositionNode::Bind { left, right, .. } => {
                self.collect_bundle_widths(left, widths);
                self.collect_bundle_widths(right, widths);
            }
            CompositionNode::Bundle { children, .. } => {
                widths.push(children.len());
                for child in children {
                    self.collect_bundle_widths(child, widths);
                }
            }
            CompositionNode::Permute { child, .. } => {
                self.collect_bundle_widths(child, widths);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// UNIFIED EMERGENCE METRICS
// ═══════════════════════════════════════════════════════════════════════════════

/// Complete emergence analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceAnalysis {
    /// Binding depth analysis
    pub depth: BindingDepthResult,
    /// Recoverability analysis
    pub recoverability: TreeRecoverabilityResult,
    /// Coherence analysis
    pub coherence: TreeCoherenceResult,
    /// Node type counts
    pub node_counts: NodeCounts,
    /// Structural entropy
    pub structural_entropy: f64,
    /// Branching statistics
    pub branching: BranchingStats,
    /// Unified emergence score
    pub emergence_score: f64,
}

/// Unified metrics calculator combining all four analyzers
#[derive(Debug, Clone)]
pub struct EmergenceMetrics {
    depth_analyzer: BindingDepthAnalyzer,
    recoverability_analyzer: RecoverabilityAnalyzer,
    coherence_analyzer: BindingCoherenceAnalyzer,
    compositional_analyzer: CompositionalInformationAnalyzer,
}

impl Default for EmergenceMetrics {
    fn default() -> Self {
        Self::new()
    }
}

impl EmergenceMetrics {
    /// Create a new metrics calculator with default analyzers
    pub fn new() -> Self {
        Self {
            depth_analyzer: BindingDepthAnalyzer::new(),
            recoverability_analyzer: RecoverabilityAnalyzer::new(),
            coherence_analyzer: BindingCoherenceAnalyzer::new(),
            compositional_analyzer: CompositionalInformationAnalyzer::new(),
        }
    }

    /// Create with custom recoverability threshold
    pub fn with_recoverability_threshold(mut self, threshold: f32) -> Self {
        self.recoverability_analyzer = RecoverabilityAnalyzer::with_threshold(threshold);
        self
    }

    /// Access the depth analyzer
    pub fn depth_analyzer(&self) -> &BindingDepthAnalyzer {
        &self.depth_analyzer
    }

    /// Access the recoverability analyzer
    pub fn recoverability_analyzer(&self) -> &RecoverabilityAnalyzer {
        &self.recoverability_analyzer
    }

    /// Access the coherence analyzer
    pub fn coherence_analyzer(&self) -> &BindingCoherenceAnalyzer {
        &self.coherence_analyzer
    }

    /// Access the compositional analyzer
    pub fn compositional_analyzer(&self) -> &CompositionalInformationAnalyzer {
        &self.compositional_analyzer
    }

    /// Compute unified emergence score
    ///
    /// Combines all metrics into a single score:
    /// - 0.3 * normalized depth (deeper = more emergent)
    /// - 0.3 * coherence (bind creates new structure)
    /// - 0.2 * (1 - recoverability) (can't unbind = integrated)
    /// - 0.2 * normalized structural entropy (diverse composition)
    pub fn emergence_score(&self, node: &CompositionNode) -> f64 {
        let depth = self.depth_analyzer.analyze(node);
        let recoverability = self.recoverability_analyzer.tree_recoverability(node);
        let coherence = self.coherence_analyzer.tree_coherence(node);
        let structural_entropy = self.compositional_analyzer.structural_entropy(node);

        // Normalize depth (assume max reasonable depth is 10)
        let normalized_depth = (depth.max_depth as f64 / 10.0).min(1.0);

        // Coherence is already in [0, 1]
        let coherence_score = coherence.avg_coherence as f64;

        // Invert recoverability (low recoverability = high integration)
        let integration_score = 1.0 - recoverability.avg_recoverability as f64;

        // Normalize structural entropy (max is log2(4) = 2 for 4 node types)
        let normalized_entropy = (structural_entropy / 2.0).min(1.0);

        // Weighted combination
        0.3 * normalized_depth
            + 0.3 * coherence_score
            + 0.2 * integration_score
            + 0.2 * normalized_entropy
    }

    /// Perform complete emergence analysis
    pub fn analyze(&self, node: &CompositionNode) -> EmergenceAnalysis {
        let depth = self.depth_analyzer.analyze(node);
        let recoverability = self.recoverability_analyzer.tree_recoverability(node);
        let coherence = self.coherence_analyzer.tree_coherence(node);
        let node_counts = self.compositional_analyzer.node_counts(node);
        let structural_entropy = self.compositional_analyzer.structural_entropy(node);
        let branching = self.compositional_analyzer.branching_stats(node);
        let emergence_score = self.emergence_score(node);

        EmergenceAnalysis {
            depth,
            recoverability,
            coherence,
            node_counts,
            structural_entropy,
            branching,
            emergence_score,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hdc::unified_hv::HDC_DIMENSION;

    fn create_test_vector(seed: u64) -> ContinuousHV {
        ContinuousHV::random(HDC_DIMENSION, seed)
    }

    fn create_atomic(label: &str, seed: u64) -> CompositionNode {
        CompositionNode::atomic(label, create_test_vector(seed))
    }

    #[test]
    fn test_composition_node_creation() {
        let a = create_atomic("A", 1);
        let b = create_atomic("B", 2);

        assert!(a.is_atomic());
        assert_eq!(a.label(), Some("A"));

        let bound = CompositionNode::bind(a, b);
        assert!(!bound.is_atomic());
        assert!(bound.vector().norm() > 0.0);
    }

    #[test]
    fn test_binding_depth_atomic() {
        let analyzer = BindingDepthAnalyzer::new();
        let node = create_atomic("A", 1);

        assert_eq!(analyzer.max_depth(&node), 0);
        assert_eq!(analyzer.bind_count(&node), 0);
        assert_eq!(analyzer.total_nodes(&node), 1);
    }

    #[test]
    fn test_binding_depth_single_bind() {
        let analyzer = BindingDepthAnalyzer::new();
        let a = create_atomic("A", 1);
        let b = create_atomic("B", 2);
        let bound = CompositionNode::bind(a, b);

        assert_eq!(analyzer.max_depth(&bound), 1);
        assert_eq!(analyzer.bind_count(&bound), 1);
        assert_eq!(analyzer.total_nodes(&bound), 3);
    }

    #[test]
    fn test_binding_depth_nested() {
        let analyzer = BindingDepthAnalyzer::new();
        let a = create_atomic("A", 1);
        let b = create_atomic("B", 2);
        let c = create_atomic("C", 3);

        // ((A ⊗ B) ⊗ C)
        let ab = CompositionNode::bind(a, b);
        let abc = CompositionNode::bind(ab, c);

        assert_eq!(analyzer.max_depth(&abc), 2);
        assert_eq!(analyzer.bind_count(&abc), 2);
        assert_eq!(analyzer.total_nodes(&abc), 5);
    }

    #[test]
    fn test_binding_depth_with_bundle() {
        let analyzer = BindingDepthAnalyzer::new();
        let a = create_atomic("A", 1);
        let b = create_atomic("B", 2);
        let c = create_atomic("C", 3);

        // Bundle(A, B) ⊗ C
        let bundle = CompositionNode::bundle_uniform(vec![a, b]);
        let bound = CompositionNode::bind(bundle, c);

        assert_eq!(analyzer.max_depth(&bound), 1);
        assert_eq!(analyzer.bind_count(&bound), 1);
    }

    #[test]
    fn test_recoverability_analyzer() {
        let analyzer = RecoverabilityAnalyzer::new();
        let a = create_test_vector(1);
        let b = create_test_vector(2);

        let result = analyzer.binding_recoverability(&a, &b);

        // For random vectors, recoverability depends on HDC algebra properties
        assert!(result.avg_recoverability >= -1.0 && result.avg_recoverability <= 1.0);
    }

    #[test]
    fn test_tree_recoverability() {
        let analyzer = RecoverabilityAnalyzer::new();
        let a = create_atomic("A", 1);
        let b = create_atomic("B", 2);
        let bound = CompositionNode::bind(a, b);

        let result = analyzer.tree_recoverability(&bound);

        assert_eq!(result.bind_count, 1);
        assert!(result.avg_recoverability >= -1.0 && result.avg_recoverability <= 1.0);
    }

    #[test]
    fn test_coherence_analyzer() {
        let analyzer = BindingCoherenceAnalyzer::new();
        let a = create_test_vector(1);
        let b = create_test_vector(2);

        let coherence = analyzer.coherence(&a, &b);

        // Coherence should be in [0, 1]
        assert!(coherence >= 0.0 && coherence <= 1.0);
    }

    #[test]
    fn test_coherence_high_for_random_vectors() {
        let analyzer = BindingCoherenceAnalyzer::new();
        let a = create_test_vector(1);
        let b = create_test_vector(2);

        let result = analyzer.detailed_coherence(&a, &b);

        // For random vectors, bind should create structure very different from bundle
        // So coherence should be high
        assert!(
            result.coherence > 0.5,
            "Random vectors should have high coherence: {}",
            result.coherence
        );
    }

    #[test]
    fn test_node_counts() {
        let analyzer = CompositionalInformationAnalyzer::new();
        let a = create_atomic("A", 1);
        let b = create_atomic("B", 2);
        let c = create_atomic("C", 3);

        // Bundle(A, B) ⊗ C
        let bundle = CompositionNode::bundle_uniform(vec![a, b]);
        let bound = CompositionNode::bind(bundle, c);

        let counts = analyzer.node_counts(&bound);

        assert_eq!(counts.atomic, 3);
        assert_eq!(counts.bind, 1);
        assert_eq!(counts.bundle, 1);
        assert_eq!(counts.permute, 0);
        assert_eq!(counts.total(), 5);
    }

    #[test]
    fn test_structural_entropy() {
        let analyzer = CompositionalInformationAnalyzer::new();

        // Single atomic node has zero entropy (all same type)
        let atomic = create_atomic("A", 1);
        let entropy_atomic = analyzer.structural_entropy(&atomic);
        assert_eq!(entropy_atomic, 0.0);

        // Mixed structure has positive entropy
        let a = create_atomic("A", 1);
        let b = create_atomic("B", 2);
        let bound = CompositionNode::bind(a, b);
        let entropy_mixed = analyzer.structural_entropy(&bound);
        assert!(
            entropy_mixed > 0.0,
            "Mixed structure should have positive entropy"
        );
    }

    #[test]
    fn test_branching_stats() {
        let analyzer = CompositionalInformationAnalyzer::new();
        let a = create_atomic("A", 1);
        let b = create_atomic("B", 2);
        let c = create_atomic("C", 3);

        // Bundle of 3 elements
        let bundle = CompositionNode::bundle_uniform(vec![a, b, c]);
        let stats = analyzer.branching_stats(&bundle);

        assert_eq!(stats.bundle_count, 1);
        assert_eq!(stats.max_bundle_width, 3);
        assert!((stats.avg_bundle_width - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_emergence_score_increases_with_depth() {
        let metrics = EmergenceMetrics::new();

        // Shallow: A ⊗ B
        let a1 = create_atomic("A", 1);
        let b1 = create_atomic("B", 2);
        let shallow = CompositionNode::bind(a1, b1);

        // Deep: ((A ⊗ B) ⊗ C) ⊗ D
        let a2 = create_atomic("A", 1);
        let b2 = create_atomic("B", 2);
        let c2 = create_atomic("C", 3);
        let d2 = create_atomic("D", 4);
        let ab = CompositionNode::bind(a2, b2);
        let abc = CompositionNode::bind(ab, c2);
        let deep = CompositionNode::bind(abc, d2);

        let score_shallow = metrics.emergence_score(&shallow);
        let score_deep = metrics.emergence_score(&deep);

        assert!(
            score_deep > score_shallow,
            "Deeper structure should have higher emergence: shallow={}, deep={}",
            score_shallow,
            score_deep
        );
    }

    #[test]
    fn test_full_analysis() {
        let metrics = EmergenceMetrics::new();
        let a = create_atomic("A", 1);
        let b = create_atomic("B", 2);
        let c = create_atomic("C", 3);

        // Bundle(A, B) ⊗ C
        let bundle = CompositionNode::bundle_uniform(vec![a, b]);
        let bound = CompositionNode::bind(bundle, c);

        let analysis = metrics.analyze(&bound);

        assert_eq!(analysis.depth.max_depth, 1);
        assert_eq!(analysis.depth.bind_count, 1);
        assert_eq!(analysis.node_counts.atomic, 3);
        assert_eq!(analysis.node_counts.bundle, 1);
        assert!(analysis.structural_entropy > 0.0);
        assert!(analysis.emergence_score >= 0.0 && analysis.emergence_score <= 1.0);
    }

    #[test]
    fn test_permute_node() {
        let analyzer = BindingDepthAnalyzer::new();
        let a = create_atomic("A", 1);
        let permuted = CompositionNode::permute(a, 100);

        assert_eq!(analyzer.max_depth(&permuted), 0);
        assert_eq!(analyzer.total_nodes(&permuted), 2);
    }

    #[test]
    fn test_atomic_emergence_score_is_low() {
        let metrics = EmergenceMetrics::new();
        let atomic = create_atomic("A", 1);

        let score = metrics.emergence_score(&atomic);

        // Single atomic node should have very low emergence
        assert!(
            score < 0.3,
            "Atomic node should have low emergence score: {}",
            score
        );
    }

    #[test]
    fn test_complex_tree_analysis() {
        let metrics = EmergenceMetrics::new();

        // Build a more complex tree:
        // Bundle(A⊗B, C⊗D, Permute(E))
        let a = create_atomic("A", 1);
        let b = create_atomic("B", 2);
        let c = create_atomic("C", 3);
        let d = create_atomic("D", 4);
        let e = create_atomic("E", 5);

        let ab = CompositionNode::bind(a, b);
        let cd = CompositionNode::bind(c, d);
        let pe = CompositionNode::permute(e, 500);

        let complex = CompositionNode::bundle_uniform(vec![ab, cd, pe]);

        let analysis = metrics.analyze(&complex);

        // Should have 2 binds, 1 bundle, 1 permute, 5 atomic
        assert_eq!(analysis.node_counts.bind, 2);
        assert_eq!(analysis.node_counts.bundle, 1);
        assert_eq!(analysis.node_counts.permute, 1);
        assert_eq!(analysis.node_counts.atomic, 5);
        assert_eq!(analysis.depth.max_depth, 1);
        assert_eq!(analysis.coherence.bind_count, 2);
    }
}
