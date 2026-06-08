// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Resonant Hypergraph Network (RHN)
//!
//! This module implements the initial Hierarchical Cantor Hypervector (HCH) topology
//! inside the broader Resonant Hypergraph Network research line. RHN treats topology
//! as an engineering variable: Cantor hierarchy for multi-scale isolation, hypercube
//! routing for proof-friendly coordinates, and small-world/prototype routing for
//! semantic retrieval.
//!
//! ## Structure
//!
//! - **L0: Global Apex** — global workspace and active-inference state.
//! - **L1: Meso-Hull** — broad context domains (visual, linguistic, etc.).
//! - **L2: Micro-Bundle** — object-level feature binding.
//! - **L3: Atomic Leaves** — low-level sensory primitives.
//!
//! This partitioning reduces representational crosstalk under heavy bundling by
//! physically separating information at different scales.

use super::unified_hv::ContinuousHV;
use std::ops::Range;

/// Aggregation strategy for bundling hypervectors.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum BundleMode {
    /// Pure additive accumulation (no normalization).
    Sum,
    /// Element-wise mean (Sum / Count).
    Mean,
    /// Project to unit L2 norm after each update.
    UnitNormalize,
    /// Cap the norm at a maximum value.
    Clipped { max_norm: f32 },
    /// Binarize via majority sign {-1, 0, 1}.
    MajoritySign,
}

impl Default for BundleMode {
    fn default() -> Self {
        Self::Sum
    }
}

/// Trait for routing information to specific hierarchical nodes.
pub trait CantorRouter {
    /// Determine which child index to route to (read-only).
    fn route(&self, role: &ContinuousHV, context: &ContinuousHV, branching: usize) -> usize;

    /// Route and record the placement (may mutate internal load counters).
    fn route_and_record(
        &self,
        role: &ContinuousHV,
        context: &ContinuousHV,
        branching: usize,
    ) -> usize {
        self.route(role, context, branching)
    }
}

/// Default hash-based deterministic router.
pub struct HashRouter;

impl CantorRouter for HashRouter {
    fn route(&self, role: &ContinuousHV, _context: &ContinuousHV, branching: usize) -> usize {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        for &val in role.as_slice().iter().take(16) {
            val.to_bits().hash(&mut hasher);
        }
        (hasher.finish() as usize) % branching
    }
}

/// Random router for baseline comparison.
pub struct RandomRouter {
    pub seed: u64,
}

impl CantorRouter for RandomRouter {
    fn route(&self, _role: &ContinuousHV, _context: &ContinuousHV, branching: usize) -> usize {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.seed.hash(&mut hasher);
        (hasher.finish() as usize) % branching
    }
}

/// Routes based on the maximum value in the first N dimensions.
pub struct PrefixMaxRouter;

impl CantorRouter for PrefixMaxRouter {
    fn route(&self, role: &ContinuousHV, _context: &ContinuousHV, branching: usize) -> usize {
        let mut best_val = -1.0;
        let mut best_idx = 0;
        let slice = role.as_slice();
        for i in 0..branching.min(slice.len()) {
            if slice[i] > best_val {
                best_val = slice[i];
                best_idx = i;
            }
        }
        best_idx
    }
}

/// Proof-friendly router that maps a role/context pair to a Boolean hypercube coordinate.
///
/// For power-of-two branch counts, the returned leaf index is the coordinate encoded by
/// `dimensions` sign bits. This keeps routing inspectable for binary-field proof systems
/// while still remaining deterministic for HDC experiments.
pub struct HypercubeRouter {
    pub dimensions: usize,
    pub seed: u64,
}

impl HypercubeRouter {
    pub fn node_count(&self) -> usize {
        1usize << self.dimensions
    }

    pub fn coordinate_bits(&self, role: &ContinuousHV, context: &ContinuousHV) -> Vec<bool> {
        let role_slice = role.as_slice();
        let context_slice = context.as_slice();
        let role_len = role_slice.len().max(1);
        let context_len = context_slice.len().max(1);
        let offset = self.seed as usize % role_len;

        (0..self.dimensions)
            .map(|bit| {
                let role_val = role_slice[(offset + bit) % role_len];
                let context_val = if context_slice.is_empty() {
                    0.0
                } else {
                    context_slice[(offset + bit) % context_len]
                };
                role_val + 0.5 * context_val >= 0.0
            })
            .collect()
    }

    pub fn hamming_neighbors(index: usize, dimensions: usize) -> Vec<usize> {
        (0..dimensions).map(|bit| index ^ (1usize << bit)).collect()
    }
}

impl CantorRouter for HypercubeRouter {
    fn route(&self, role: &ContinuousHV, context: &ContinuousHV, branching: usize) -> usize {
        let mut coord = 0usize;
        for (bit, high) in self.coordinate_bits(role, context).into_iter().enumerate() {
            if high {
                coord |= 1usize << bit;
            }
        }

        if branching == 0 { 0 } else { coord % branching }
    }
}

fn semantic_routing_query(role: &ContinuousHV, context: &ContinuousHV) -> ContinuousHV {
    if context.as_slice().iter().any(|v| v.abs() > 1e-8) {
        role.bind(context)
    } else {
        role.clone()
    }
}

/// Per-leaf semantic prototypes derived from assigned roles or intents.
///
/// This is intentionally simple and deterministic: callers decide the assignment policy
/// (hash, hypercube, oracle, etc.), then this index bundles assigned semantic vectors into
/// stable leaf keys for prototype and small-world routing.
#[derive(Debug, Clone)]
pub struct PrototypeIndex {
    pub leaf_keys: Vec<ContinuousHV>,
    pub counts: Vec<usize>,
}

impl PrototypeIndex {
    pub fn from_assignments(
        assignments: &[(usize, ContinuousHV)],
        branching: usize,
        leaf_dim: usize,
        mode: BundleMode,
    ) -> Self {
        let mut leaf_keys = vec![ContinuousHV::zero(leaf_dim); branching];
        let mut counts = vec![0usize; branching];

        for (leaf_idx, vector) in assignments {
            if *leaf_idx >= branching {
                continue;
            }

            let target = leaf_keys[*leaf_idx].as_mut_slice();
            let source = vector.as_slice();
            for i in 0..leaf_dim.min(source.len()) {
                target[i] += source[i];
            }
            counts[*leaf_idx] += 1;
        }

        for (idx, key) in leaf_keys.iter_mut().enumerate() {
            match mode {
                BundleMode::Sum => {}
                BundleMode::Mean => {
                    if counts[idx] > 0 {
                        for v in key.as_mut_slice() {
                            *v /= counts[idx] as f32;
                        }
                    }
                }
                BundleMode::UnitNormalize | BundleMode::Clipped { .. } => {
                    normalize_hv(key);
                }
                BundleMode::MajoritySign => {
                    for v in key.as_mut_slice() {
                        *v = if *v > 0.0 {
                            1.0
                        } else if *v < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                    }
                }
            }
        }

        Self { leaf_keys, counts }
    }
}

fn normalize_hv(hv: &mut ContinuousHV) {
    let norm_sq: f32 = hv.as_slice().iter().map(|v| v * v).sum();
    let norm = norm_sq.sqrt();
    if norm > 1e-8 {
        for v in hv.as_mut_slice() {
            *v /= norm;
        }
    }
}

/// Lightweight semantic router for older ablations.
///
/// This routes by deterministic signed projections from the role/context state. It is
/// intentionally simple: v0.5+ uses explicit prototype and hypercube routers for stronger
/// topology comparisons.
pub struct SemanticRouter;

impl CantorRouter for SemanticRouter {
    fn route(&self, role: &ContinuousHV, context: &ContinuousHV, branching: usize) -> usize {
        if branching == 0 {
            return 0;
        }

        let role_slice = role.as_slice();
        let context_slice = context.as_slice();
        let len = role_slice.len().max(context_slice.len()).max(1);
        let mut score = 0usize;

        for bit in 0..branching.next_power_of_two().trailing_zeros() as usize {
            let idx = (bit * 17 + 3) % len;
            let role_val = role_slice
                .get(idx % role_slice.len().max(1))
                .copied()
                .unwrap_or(0.0);
            let context_val = context_slice
                .get(idx % context_slice.len().max(1))
                .copied()
                .unwrap_or(0.0);
            if role_val + context_val >= 0.0 {
                score |= 1usize << bit;
            }
        }

        score % branching
    }
}

/// Semantic prototype router: routes to the leaf with the most similar key.
pub struct PrototypeRouter {
    pub leaf_keys: Vec<ContinuousHV>,
}

impl CantorRouter for PrototypeRouter {
    fn route(&self, role: &ContinuousHV, context: &ContinuousHV, branching: usize) -> usize {
        let query = semantic_routing_query(role, context);
        self.leaf_keys
            .iter()
            .take(branching)
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                query
                    .similarity(a)
                    .partial_cmp(&query.similarity(b))
                    .unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap_or(0)
    }
}

/// Small-world semantic router.
///
/// Starts from a proof-friendly hypercube coordinate, then searches the local Hamming
/// neighborhood plus a small deterministic shortcut set against semantic leaf prototypes.
/// This is the first HCH router aimed at Broca-style associative retrieval rather than
/// purely uniform load distribution.
pub struct SmallWorldRouter {
    pub dimensions: usize,
    pub seed: u64,
    pub leaf_keys: Vec<ContinuousHV>,
    pub shortcuts: usize,
}

impl SmallWorldRouter {
    fn push_candidate(candidates: &mut Vec<usize>, candidate: usize, branching: usize) {
        if candidate < branching && !candidates.contains(&candidate) {
            candidates.push(candidate);
        }
    }

    fn shortcut(&self, base: usize, jump: usize, branching: usize) -> usize {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        self.seed.hash(&mut hasher);
        base.hash(&mut hasher);
        jump.hash(&mut hasher);
        (hasher.finish() as usize) % branching
    }
}

impl CantorRouter for SmallWorldRouter {
    fn route(&self, role: &ContinuousHV, context: &ContinuousHV, branching: usize) -> usize {
        if branching == 0 {
            return 0;
        }

        let hypercube = HypercubeRouter {
            dimensions: self.dimensions,
            seed: self.seed,
        };
        let base = hypercube.route(role, context, branching);
        if self.leaf_keys.is_empty() {
            return base;
        }

        let mut candidates = Vec::new();
        Self::push_candidate(&mut candidates, base, branching);
        for neighbor in HypercubeRouter::hamming_neighbors(base, self.dimensions) {
            Self::push_candidate(&mut candidates, neighbor, branching);
        }
        for jump in 0..self.shortcuts {
            let shortcut = self.shortcut(base, jump, branching);
            Self::push_candidate(&mut candidates, shortcut, branching);
        }

        let query = semantic_routing_query(role, context);
        candidates
            .into_iter()
            .filter(|idx| *idx < self.leaf_keys.len())
            .max_by(|a, b| {
                query
                    .similarity(&self.leaf_keys[*a])
                    .partial_cmp(&query.similarity(&self.leaf_keys[*b]))
                    .unwrap()
            })
            .unwrap_or(base)
    }
}

/// Oracle router for upper-bound testing.
pub struct OracleRouter {
    pub target_leaf: usize,
}

impl CantorRouter for OracleRouter {
    fn route(&self, _role: &ContinuousHV, _context: &ContinuousHV, _branching: usize) -> usize {
        self.target_leaf
    }
}

/// Load-balanced hash router using power-of-k-choices.
pub struct LoadBalancedHashRouter {
    pub counts: std::sync::Arc<std::sync::Mutex<Vec<usize>>>,
    pub probes: usize,
}

impl LoadBalancedHashRouter {
    pub fn new(branching: usize, probes: usize) -> Self {
        Self {
            counts: std::sync::Arc::new(std::sync::Mutex::new(vec![0; branching])),
            probes,
        }
    }
}

impl CantorRouter for LoadBalancedHashRouter {
    fn route(&self, role: &ContinuousHV, _context: &ContinuousHV, branching: usize) -> usize {
        use std::hash::{Hash, Hasher};
        let mut best_idx = 0;
        let mut min_count = usize::MAX;

        let counts = self.counts.lock().unwrap();

        for i in 0..self.probes {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            i.hash(&mut hasher);
            for &val in role.as_slice().iter().take(16) {
                val.to_bits().hash(&mut hasher);
            }
            let idx = (hasher.finish() as usize) % branching;
            if counts[idx] < min_count {
                min_count = counts[idx];
                best_idx = idx;
            }
        }
        best_idx
    }

    fn route_and_record(
        &self,
        role: &ContinuousHV,
        context: &ContinuousHV,
        branching: usize,
    ) -> usize {
        let best_idx = self.route(role, context, branching);
        let mut counts = self.counts.lock().unwrap();
        counts[best_idx] += 1;
        best_idx
    }
}

/// Topology configuration for Hierarchical Cantor Hypervectors.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CantorHdcConfig {
    /// Total dimensions in the hypervector.
    pub total_dim: usize,
    /// Number of levels in the hierarchy (e.g., 4).
    pub levels: usize,
    /// Branching factor per level (e.g., 4).
    pub branching: usize,
    /// Minimum dimension for a leaf node (e.g., 256).
    pub leaf_dim: usize,
    /// Aggregation mode for bundling.
    pub bundle_mode: BundleMode,
}

/// Broader RHN alias for the current HCH configuration.
pub type ResonantHypergraphConfig = CantorHdcConfig;

impl Default for CantorHdcConfig {
    fn default() -> Self {
        Self {
            total_dim: 16_384,
            levels: 4,
            branching: 4,
            leaf_dim: 256,
            bundle_mode: BundleMode::Sum,
        }
    }
}

/// A node in the Cantor hierarchy.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CantorNode {
    /// Level in the hierarchy (0 = global apex).
    pub level: usize,
    /// Index within the level.
    pub index: usize,
    /// Range of dimensions in the parent hypervector.
    pub range: Range<usize>,
    /// Index of parent node in the flat nodes vector.
    pub parent: Option<usize>,
    /// Indices of child nodes in the flat nodes vector.
    pub children: Vec<usize>,
}

impl CantorNode {
    /// Get the coordinate of this node (path of indices from root).
    pub fn coordinate(&self, nodes: &[CantorNode]) -> Vec<usize> {
        let mut coord = Vec::new();
        let mut current = self;
        while let Some(parent_idx) = current.parent {
            coord.push(current.index);
            current = &nodes[parent_idx];
        }
        coord.reverse();
        coord
    }
}

/// Pyramid Cantor Vector: a hypervector with hierarchical topology.
#[derive(Debug, Clone)]
pub struct PyramidCantorVector {
    /// Configuration used for this vector.
    pub config: CantorHdcConfig,
    /// The underlying hypervector data.
    pub data: ContinuousHV,
    /// Metadata for all nodes in the hierarchy.
    pub nodes: Vec<CantorNode>,
}

/// Broader RHN alias for the current Cantor-backed topology vector.
pub type ResonantHypergraphVector = PyramidCantorVector;

impl PyramidCantorVector {
    /// Create a new Pyramid Cantor Vector with the given configuration.
    ///
    /// If `vector` is provided, it must match `config.total_dim`.
    /// Otherwise, a zero vector is initialized.
    pub fn new(config: CantorHdcConfig, vector: Option<ContinuousHV>) -> Self {
        let dim = config.total_dim;
        let data = vector.unwrap_or_else(|| ContinuousHV::zero(dim));
        assert_eq!(
            data.dim(),
            dim,
            "Vector dimension must match config total_dim"
        );

        let mut nodes = Vec::new();
        let mut level_counters = vec![0; config.levels];
        Self::build_hierarchy(&config, 0, &mut level_counters, 0..dim, None, &mut nodes);

        Self {
            config,
            data,
            nodes,
        }
    }

    /// Recursive helper to build the node metadata.
    fn build_hierarchy(
        config: &CantorHdcConfig,
        level: usize,
        level_counters: &mut [usize],
        range: Range<usize>,
        parent: Option<usize>,
        nodes: &mut Vec<CantorNode>,
    ) -> usize {
        let node_idx = nodes.len();
        let index = level_counters[level];
        level_counters[level] += 1;

        // Placeholder for children
        nodes.push(CantorNode {
            level,
            index,
            range: range.clone(),
            parent,
            children: Vec::new(),
        });

        if level + 1 < config.levels && range.len() >= config.branching * config.leaf_dim {
            let child_dim = range.len() / config.branching;
            let mut children = Vec::new();

            for i in 0..config.branching {
                let child_start = range.start + i * child_dim;
                let child_end = if i == config.branching - 1 {
                    range.end
                } else {
                    child_start + child_dim
                };

                let child_node_idx = Self::build_hierarchy(
                    config,
                    level + 1,
                    level_counters,
                    child_start..child_end,
                    Some(node_idx),
                    nodes,
                );
                children.push(child_node_idx);
            }

            nodes[node_idx].children = children;
        }

        node_idx
    }

    /// Get a node by level and index.
    pub fn find_node(&self, level: usize, index: usize) -> Option<&CantorNode> {
        self.nodes
            .iter()
            .find(|n| n.level == level && n.index == index)
    }

    /// Split a node into children, increasing local resolution.
    pub fn split_at_node(&mut self, node_idx: usize) -> bool {
        let node = &self.nodes[node_idx];
        if !node.children.is_empty() {
            return false; // Already split
        }

        let range = node.range.clone();
        let branching = self.config.branching;
        if range.len() < branching * 8 {
            return false; // Too small to split further
        }

        let child_dim = range.len() / branching;
        let mut children = Vec::new();
        let current_level = node.level;

        let child_level = current_level + 1;
        let mut next_level_index = self
            .nodes
            .iter()
            .filter(|existing| existing.level == child_level)
            .map(|existing| existing.index)
            .max()
            .map(|idx| idx + 1)
            .unwrap_or(0);

        for i in 0..branching {
            let child_start = range.start + i * child_dim;
            let child_end = if i == branching - 1 {
                range.end
            } else {
                child_start + child_dim
            };

            let child_idx = self.nodes.len();
            self.nodes.push(CantorNode {
                level: child_level,
                index: next_level_index,
                range: child_start..child_end,
                parent: Some(node_idx),
                children: Vec::new(),
            });
            next_level_index += 1;
            children.push(child_idx);
        }

        self.nodes[node_idx].children = children;
        true
    }

    /// Find the best match for a role in the routed leaf and its neighbors.
    pub fn retrieve_multi_leaf(
        &self,
        role: &ContinuousHV,
        context: &ContinuousHV,
        router: &dyn CantorRouter,
        codebook: &[ContinuousHV],
        top_k_leaves: usize,
    ) -> Vec<(usize, f32)> {
        let candidates = self.routed_leaf_candidates(role, context, router, top_k_leaves);

        let mut best_by_codebook = vec![f32::NEG_INFINITY; codebook.len()];
        for leaf_idx in candidates {
            if let Some(node) = self.find_node(1, leaf_idx) {
                let data = self.node_data(node);
                let recovered = ContinuousHV::from_slice(data).bind(&role.inverse());
                for (idx, cand) in codebook.iter().enumerate() {
                    best_by_codebook[idx] = best_by_codebook[idx].max(recovered.similarity(cand));
                }
            }
        }

        let mut results: Vec<(usize, f32)> = best_by_codebook
            .into_iter()
            .enumerate()
            .filter(|(_, score)| score.is_finite())
            .collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    fn routed_leaf_candidates(
        &self,
        role: &ContinuousHV,
        context: &ContinuousHV,
        router: &dyn CantorRouter,
        top_k_leaves: usize,
    ) -> Vec<usize> {
        let branching = self.config.branching;
        if branching == 0 || top_k_leaves == 0 {
            return Vec::new();
        }

        let primary_idx = router.route(role, context, branching);
        let mut candidates = vec![primary_idx];

        let dimensions = branching.next_power_of_two().trailing_zeros() as usize;
        for neighbor in HypercubeRouter::hamming_neighbors(primary_idx, dimensions) {
            if neighbor < branching && !candidates.contains(&neighbor) {
                candidates.push(neighbor);
                if candidates.len() >= top_k_leaves {
                    return candidates;
                }
            }
        }

        for offset in 1..branching {
            let neighbor = (primary_idx + offset) % branching;
            if !candidates.contains(&neighbor) {
                candidates.push(neighbor);
                if candidates.len() >= top_k_leaves {
                    break;
                }
            }
        }

        candidates
    }

    /// Access the data for a specific node.
    pub fn node_data(&self, node: &CantorNode) -> &[f32] {
        &self.data.as_slice()[node.range.clone()]
    }

    /// Access the mutable data for a specific node.
    pub fn node_data_mut(&mut self, node: &CantorNode) -> &mut [f32] {
        &mut self.data.as_mut_slice()[node.range.clone()]
    }

    /// Bundle a concept within a node, splitting if saturation is detected.
    pub fn bundle_at_node_adaptive(
        &mut self,
        node_idx: usize,
        other: &ContinuousHV,
        load_threshold: usize,
        current_load: usize,
    ) -> usize {
        if current_load >= load_threshold {
            if self.split_at_node(node_idx) {
                // Node was split, route to a child
                // For simplicity, we'll use a local hash router here
                use std::hash::{Hash, Hasher};
                let mut hasher = std::collections::hash_map::DefaultHasher::new();
                for &val in other.as_slice().iter().take(16) {
                    val.to_bits().hash(&mut hasher);
                }
                let branching = self.config.branching;
                let child_local_idx = (hasher.finish() as usize) % branching;
                let child_node_idx = self.nodes[node_idx].children[child_local_idx];

                // Recurse into child
                return self.bundle_at_node_adaptive(child_node_idx, other, load_threshold, 0);
            }
        }

        let node = &self.nodes[node_idx].clone();
        self.bundle_at_node(node, other);
        node_idx
    }

    /// Bundle a concept redundantly across multiple nodes (e.g. for small-world robustness).
    pub fn bundle_multi_leaf(
        &mut self,
        role: &ContinuousHV,
        context: &ContinuousHV,
        router: &dyn CantorRouter,
        binding: &ContinuousHV,
        k: usize,
    ) {
        for leaf_idx in self.routed_leaf_candidates(role, context, router, k) {
            if let Some(node) = self.find_node(1, leaf_idx).cloned() {
                self.bundle_at_node(&node, binding);
            }
        }
    }

    /// Bind a concept ONLY within a specific node.
    pub fn bind_at_node(&mut self, node: &CantorNode, other: &ContinuousHV) {
        let other_data = other.as_slice();
        let target = self.node_data_mut(node);
        let len = target.len().min(other_data.len());

        for i in 0..len {
            target[i] *= other_data[i];
        }
    }

    /// Bundle a concept ONLY within a specific node.
    pub fn bundle_at_node(&mut self, node: &CantorNode, other: &ContinuousHV) {
        let other_data = other.as_slice();
        let mode = self.config.bundle_mode;
        let target = self.node_data_mut(node);
        let len = target.len().min(other_data.len());

        for i in 0..len {
            target[i] += other_data[i];
        }

        match mode {
            BundleMode::Sum => {}
            BundleMode::Mean => {
                // Approximate mean requires tracking counts; here we use simple decay or scaling
                // For this v0.3, we'll implement simple UnitNormalize as a proxy for stable aggregation
                let mut norm_sq: f32 = 0.0;
                for &v in target.iter() {
                    norm_sq += v * v;
                }
                let norm = norm_sq.sqrt();
                if norm > 1e-8 {
                    for v in target.iter_mut() {
                        *v /= norm;
                    }
                }
            }
            BundleMode::UnitNormalize => {
                let mut norm_sq: f32 = 0.0;
                for &v in target.iter() {
                    norm_sq += v * v;
                }
                let norm = norm_sq.sqrt();
                if norm > 1e-8 {
                    for v in target.iter_mut() {
                        *v /= norm;
                    }
                }
            }
            BundleMode::Clipped { max_norm } => {
                let mut norm_sq: f32 = 0.0;
                for &v in target.iter() {
                    norm_sq += v * v;
                }
                let norm = norm_sq.sqrt();
                if norm > max_norm {
                    let factor = max_norm / norm;
                    for v in target.iter_mut() {
                        *v *= factor;
                    }
                }
            }
            BundleMode::MajoritySign => {
                for v in target.iter_mut() {
                    *v = if *v > 0.0 {
                        1.0
                    } else if *v < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    /// Upward Broadcast: Promote information from children to parent.
    ///
    /// Bundles all children of the given node into the node itself.
    pub fn broadcast_up(&mut self, node_idx: usize) {
        let children_indices = self.nodes[node_idx].children.clone();
        if children_indices.is_empty() {
            return;
        }

        let parent_range = self.nodes[node_idx].range.clone();
        let parent_len = parent_range.len();

        let mut aggregate = vec![0.0; parent_len];

        for &child_idx in &children_indices {
            let child_range = self.nodes[child_idx].range.clone();
            let child_data = &self.data.as_slice()[child_range];

            // Map child data to parent space (simple tiling or alignment)
            for i in 0..parent_len {
                aggregate[i] += child_data[i % child_data.len()];
            }
        }

        // Normalize and update parent
        let target = &mut self.data.as_mut_slice()[parent_range];
        for i in 0..parent_len {
            target[i] = aggregate[i] / children_indices.len() as f32;
        }
    }

    /// Downward Modulation: Modulate child nodes with parent state.
    ///
    /// Binds parent state into each child.
    pub fn modulate_down(&mut self, node_idx: usize) {
        let node = &self.nodes[node_idx];
        let parent_range = node.range.clone();
        let parent_data = self.data.as_slice()[parent_range].to_vec();

        let children_indices = node.children.clone();
        for &child_idx in &children_indices {
            let child_range = self.nodes[child_idx].range.clone();
            let child_len = child_range.len();
            let target = &mut self.data.as_mut_slice()[child_range];

            for i in 0..child_len {
                target[i] *= parent_data[i % parent_data.len()];
            }
        }
    }

    /// Compute similarity between two pyramids at a specific level.
    pub fn sim_level(&self, other: &PyramidCantorVector, level: usize) -> f32 {
        let mut total_sim = 0.0;
        let mut count = 0;

        for (n_self, n_other) in self.nodes.iter().zip(other.nodes.iter()) {
            if n_self.level == level {
                let d_self = &self.data.as_slice()[n_self.range.clone()];
                let d_other = &other.data.as_slice()[n_other.range.clone()];

                total_sim += cosine_similarity(d_self, d_other);
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            total_sim / count as f32
        }
    }

    /// Fractal Similarity: weighted sum of similarity across all levels.
    pub fn sim_fractal(&self, other: &PyramidCantorVector) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for level in 0..self.config.levels {
            let weight = 1.0 / (level + 1) as f32;
            weighted_sum += self.sim_level(other, level) * weight;
            total_weight += weight;
        }

        weighted_sum / total_weight
    }
}

/// Helper for cosine similarity.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a.sqrt() * norm_b.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cantor_topology() {
        let config = CantorHdcConfig {
            total_dim: 1024,
            levels: 3,
            branching: 2,
            leaf_dim: 128,
            ..CantorHdcConfig::default()
        };

        let pyramid = PyramidCantorVector::new(config, None);

        // L0: 1 node (0-1023)
        // L1: 2 nodes (0-511, 512-1023)
        // L2: 4 nodes (0-255, 256-511, 512-767, 768-1023)

        assert_eq!(pyramid.nodes.len(), 1 + 2 + 4);

        let l0 = pyramid.find_node(0, 0).unwrap();
        assert_eq!(l0.range, 0..1024);
        assert_eq!(l0.children.len(), 2);

        let l1_0 = pyramid.find_node(1, 0).unwrap();
        assert_eq!(l1_0.range, 0..512);

        let l1_1 = pyramid.find_node(1, 1).unwrap();
        assert_eq!(l1_1.range, 512..1024);

        let l2_0 = pyramid.find_node(2, 0).unwrap();
        assert_eq!(l2_0.range, 0..256);
        assert_eq!(l2_0.coordinate(&pyramid.nodes), vec![0, 0]);

        let l2_1 = pyramid.find_node(2, 1).unwrap();
        assert_eq!(l2_1.coordinate(&pyramid.nodes), vec![0, 1]);
    }

    #[test]
    fn test_broadcast_and_modulate() {
        let config = CantorHdcConfig {
            total_dim: 1024,
            levels: 2,
            branching: 2,
            leaf_dim: 512,
            ..CantorHdcConfig::default()
        };

        let mut pyramid = PyramidCantorVector::new(config, None);

        // Set children to some values
        {
            let c0 = pyramid.find_node(1, 0).unwrap();
            let c0_range = c0.range.clone();
            pyramid.data.values[c0_range].fill(1.0);

            let c1 = pyramid.find_node(1, 1).unwrap();
            let c1_range = c1.range.clone();
            pyramid.data.values[c1_range].fill(-1.0);
        }

        // Broadcast up to L0
        pyramid.broadcast_up(0);

        let l0 = pyramid.find_node(0, 0).unwrap();
        let l0_data = pyramid.node_data(l0);

        // Average of [1.0, 1.0, ...] and [-1.0, -1.0, ...] should be 0.0
        for &val in l0_data {
            assert_eq!(val, 0.0);
        }
    }

    #[test]
    fn test_adaptive_splitting() {
        let config = CantorHdcConfig {
            total_dim: 1024,
            levels: 1,
            branching: 4,
            leaf_dim: 128,
            ..CantorHdcConfig::default()
        };

        let mut pyramid = PyramidCantorVector::new(config, None);
        assert_eq!(pyramid.nodes.len(), 1); // Only root

        let role = ContinuousHV::random(1024, 42);

        // Bundle with threshold 1. First bundle should be at root.
        let node_idx = pyramid.bundle_at_node_adaptive(0, &role, 1, 0);
        assert_eq!(node_idx, 0);
        assert_eq!(pyramid.nodes.len(), 1);

        // Second bundle at root should trigger split.
        let node_idx2 = pyramid.bundle_at_node_adaptive(0, &role, 1, 1);
        assert!(node_idx2 > 0);
        assert_eq!(pyramid.nodes.len(), 1 + 4); // Root + 4 children
        assert_eq!(pyramid.nodes[0].children.len(), 4);

        let child_indices: Vec<usize> = pyramid.nodes[0]
            .children
            .iter()
            .map(|idx| pyramid.nodes[*idx].index)
            .collect();
        assert_eq!(child_indices, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_multi_leaf_retrieval_deduplicates_codebook_scores() {
        let config = CantorHdcConfig {
            total_dim: 1024,
            levels: 2,
            branching: 4,
            leaf_dim: 256,
            bundle_mode: BundleMode::UnitNormalize,
        };
        let mut pyramid = PyramidCantorVector::new(config, None);
        let role = ContinuousHV::random(256, 91);
        let target = ContinuousHV::random(256, 92);
        let decoy = ContinuousHV::random(256, 93);
        let binding = role.bind(&target);
        let router = OracleRouter { target_leaf: 0 };

        pyramid.bundle_multi_leaf(&role, &ContinuousHV::zero(256), &router, &binding, 2);

        let results = pyramid.retrieve_multi_leaf(
            &role,
            &ContinuousHV::zero(256),
            &router,
            &[target, decoy],
            2,
        );

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0);
    }
}
