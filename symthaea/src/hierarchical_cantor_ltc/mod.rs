//! # Hierarchical Cantor-LTC/HDC Network
//!
//! A 7-level self-similar architecture combining:
//! - **Cantor Set Topology**: Recursive ternary structure (τ ratio = 1/3)
//! - **Liquid Time-Constant Networks**: Continuous-time neural dynamics
//! - **Hyperdimensional Computing**: 16,384D holographic representations
//!
//! ## Key Innovation
//!
//! HDC binding (⊗) replaces matrix multiplication in LTC dynamics:
//! ```text
//! dx/dt = (-x + σ(W⊗x + parent⊗bundle(children) + bias)) / τ
//! ```
//!
//! This achieves:
//! - **True Unification**: HDC + LTC operate on the SAME hypervector space
//! - **Self-Similarity**: Each node is a microcosm of the whole
//! - **Scale-Depth Theory**: 7 levels enable meta-cognitive capability
//!
//! ## Architecture (Cantor Ternary Structure)
//!
//! ```text
//! Level 0 (Root):     [████████████████████████████]    τ = 1000ms
//!                            /                 \
//! Level 1:           [████████]               [████████]    τ = 333ms
//!                     /      \                /      \
//! Level 2:        [██]      [██]          [██]      [██]    τ = 111ms
//!                  /\        /\            /\        /\
//! Level 3:       [█][█]    [█][█]        [█][█]    [█][█]   τ = 37ms
//!                ...        ...          ...        ...
//! Level 7:       (leaf nodes, τ ≈ 0.46ms)
//! ```

use symthaea_core::hdc::{RealHV, HDC_DIMENSION};
use std::f32::consts::E;

// =============================================================================
// CONFIGURATION
// =============================================================================

/// Configuration for the Hierarchical Cantor-LTC Network
#[derive(Clone, Debug)]
pub struct CantorLtcConfig {
    /// HDC dimension (default: 16,384)
    pub dimension: usize,
    /// Number of levels (default: 7 for meta-cognition)
    pub max_depth: usize,
    /// Time constant ratio between levels (Cantor = 1/3)
    pub tau_ratio: f32,
    /// Base time constant at root (ms)
    pub base_tau: f32,
    /// Learning rate for Hebbian updates
    pub learning_rate: f32,
    /// Enable hierarchical Φ measurement
    pub measure_phi: bool,
}

impl Default for CantorLtcConfig {
    fn default() -> Self {
        Self {
            dimension: HDC_DIMENSION,
            max_depth: 7,
            tau_ratio: 1.0 / 3.0,  // Cantor ternary ratio
            base_tau: 1000.0,      // 1 second at root
            learning_rate: 0.01,
            measure_phi: true,
        }
    }
}

impl CantorLtcConfig {
    /// Calculate τ for a given level
    pub fn tau_at_level(&self, level: usize) -> f32 {
        self.base_tau * self.tau_ratio.powi(level as i32)
    }

    /// Total number of nodes (2^(depth+1) - 1)
    pub fn total_nodes(&self) -> usize {
        (1 << (self.max_depth + 1)) - 1  // 255 for depth 7
    }
}

// =============================================================================
// CANTOR-LTC NODE
// =============================================================================

/// A single node in the Hierarchical Cantor-LTC Network
///
/// Each node contains:
/// - A 16,384D hypervector state
/// - A time constant τ (decreasing with depth)
/// - Optional children (binary tree structure)
/// - History for BPTT training
/// - Local Φ measurement
#[derive(Clone)]
pub struct CantorLtcNode {
    /// 16,384D HDC state vector
    pub state: RealHV,
    /// Time constant (ms) - decreases with depth
    pub tau: f32,
    /// Level in the hierarchy (0 = root, 7 = leaves)
    pub level: usize,
    /// Node index for identification
    pub index: usize,
    /// Children (left, right) - None for leaf nodes
    pub children: Option<(Box<CantorLtcNode>, Box<CantorLtcNode>)>,
    /// State history for BPTT (backpropagation through time)
    pub history: Vec<RealHV>,
    /// Local integrated information (Φ)
    pub local_phi: f64,
    /// Weight vector for HDC binding (learned)
    pub weight: RealHV,
    /// Bias vector
    pub bias: RealHV,
}

impl CantorLtcNode {
    /// Create a new CantorLtcNode at the specified level
    pub fn new(level: usize, index: usize, config: &CantorLtcConfig) -> Self {
        let dim = config.dimension;
        let tau = config.tau_at_level(level);
        let seed = (level * 1000 + index) as u64;

        Self {
            state: RealHV::random(dim, seed),
            tau,
            level,
            index,
            children: None,
            history: Vec::with_capacity(100),
            local_phi: 0.0,
            weight: RealHV::random(dim, seed + 10000),
            bias: RealHV::random(dim, seed + 20000).scale(0.1),
        }
    }

    /// Create a full tree recursively
    pub fn create_tree(level: usize, index: usize, config: &CantorLtcConfig) -> Self {
        let mut node = Self::new(level, index, config);

        if level < config.max_depth {
            let left_index = 2 * index + 1;
            let right_index = 2 * index + 2;
            let left = Box::new(Self::create_tree(level + 1, left_index, config));
            let right = Box::new(Self::create_tree(level + 1, right_index, config));
            node.children = Some((left, right));
        }

        node
    }

    /// LTC dynamics step for a single node (non-recursive)
    ///
    /// The core innovation: uses HDC binding (⊗) instead of matrix multiplication
    ///
    /// ```text
    /// dx/dt = (-x + σ(W⊗x + parent⊗bundle(children) + bias)) / τ
    /// ```
    fn step_single(&mut self, dt: f32, parent_state: Option<&RealHV>) {
        // Save history for BPTT
        self.history.push(self.state.clone());
        if self.history.len() > 100 {
            self.history.remove(0);
        }

        // Compute pre-activation: W ⊗ x
        let weighted_state = self.weight.bind(&self.state);

        // Add parent influence (if not root)
        let parent_contribution = match parent_state {
            Some(parent) => {
                // Bundle children if they exist
                let child_bundle = self.get_child_bundle();
                parent.bind(&child_bundle)
            }
            None => RealHV::zero(self.state.values.len()),
        };

        // Add bias and compute full pre-activation
        let pre_activation = RealHV::bundle(&[
            weighted_state,
            parent_contribution,
            self.bias.clone(),
        ]);

        // Apply sigmoid activation
        let activated: Vec<f32> = pre_activation.values.iter()
            .map(|&x| 1.0 / (1.0 + E.powf(-x)))
            .collect();
        let target = RealHV { values: activated };

        // LTC dynamics: dx/dt = (-x + target) / τ
        let dx: Vec<f32> = self.state.values.iter()
            .zip(target.values.iter())
            .map(|(&x, &t)| (-x + t) / self.tau)
            .collect();

        // Euler stability: dt should be < 2*tau for stable integration.
        // For leaf nodes (tau ≈ 0.46ms), this is critical.
        // We clamp effective_dt to the stability limit to ensure numerical stability.
        // Note: This is not a debug_assert because the clamping handles it gracefully,
        // and some legitimate use cases (e.g., rhythm detection at 30fps) may have
        // dt larger than 2*tau for deep leaf nodes.
        let effective_dt = dt.min(1.9 * self.tau);

        // Euler integration
        for (s, d) in self.state.values.iter_mut().zip(dx.iter()) {
            *s += effective_dt * d;
        }
    }

    /// LTC dynamics step with HDC binding (iterative, avoids stack overflow)
    ///
    /// Uses breadth-first traversal with an explicit work queue to avoid
    /// deep recursion that could overflow the call stack.
    pub fn step(&mut self, dt: f32, parent_state: Option<&RealHV>) {
        use std::collections::HashMap;

        // Store parent states by node index for efficient lookup
        let mut parent_states: HashMap<usize, RealHV> = HashMap::new();

        // First, process this node (the root or starting node)
        self.step_single(dt, parent_state);

        // Store this node's state for its children to use
        parent_states.insert(self.index, self.state.clone());

        // Now use BFS to process children iteratively
        // We use raw pointers to avoid borrow checker issues with the tree structure
        let mut current_level: Vec<*mut CantorLtcNode> = Vec::new();

        // Add children of this node to current level
        if let Some((ref mut left, ref mut right)) = self.children {
            current_level.push(left.as_mut() as *mut CantorLtcNode);
            current_level.push(right.as_mut() as *mut CantorLtcNode);
        }

        // Process level by level
        while !current_level.is_empty() {
            let mut next_level: Vec<*mut CantorLtcNode> = Vec::new();

            for node_ptr in current_level {
                // SAFETY: We have exclusive access to the tree through &mut self,
                // and we're processing each node exactly once per level.
                let node = unsafe { &mut *node_ptr };

                // Get parent state from our cache
                // For a node at index i, parent is at (i-1)/2
                let parent_index = (node.index - 1) / 2;
                let parent_state_for_child = parent_states.get(&parent_index);

                node.step_single(dt, parent_state_for_child);

                // Store this node's updated state for its children
                parent_states.insert(node.index, node.state.clone());

                // Add children to next level
                if let Some((ref mut left, ref mut right)) = node.children {
                    next_level.push(left.as_mut() as *mut CantorLtcNode);
                    next_level.push(right.as_mut() as *mut CantorLtcNode);
                }
            }

            current_level = next_level;
        }
    }

    /// Get bundled state of children (or zero if leaf)
    fn get_child_bundle(&self) -> RealHV {
        match &self.children {
            Some((left, right)) => {
                RealHV::bundle(&[left.state.clone(), right.state.clone()])
            }
            None => RealHV::zero(self.state.values.len()),
        }
    }

    /// Compute local Φ (integrated information) for this subtree
    pub fn compute_local_phi(&mut self) -> f64 {
        let n = self.count_nodes() as f64;
        if n < 2.0 {
            self.local_phi = 0.0;
            return 0.0;
        }

        // Collect all states in subtree
        let states = self.collect_states();

        // Compute pairwise similarities
        let mut total_sim = 0.0;
        let mut count = 0;
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                total_sim += states[i].similarity(&states[j]) as f64;
                count += 1;
            }
        }

        let avg_sim = if count > 0 {
            total_sim / count as f64
        } else {
            0.0
        };

        // Φ approximation: integration beyond random
        // Optimal similarity around 0.4 (related but distinct)
        let optimal_sim = 0.4;
        let deviation = (avg_sim - optimal_sim).abs();
        self.local_phi = (1.0 - deviation * 2.0).max(0.0) * n.ln() / 10.0;

        self.local_phi
    }

    /// Count total nodes in this subtree
    fn count_nodes(&self) -> usize {
        1 + match &self.children {
            Some((left, right)) => left.count_nodes() + right.count_nodes(),
            None => 0,
        }
    }

    /// Collect all states in this subtree
    fn collect_states(&self) -> Vec<RealHV> {
        let mut states = vec![self.state.clone()];
        if let Some((ref left, ref right)) = self.children {
            states.extend(left.collect_states());
            states.extend(right.collect_states());
        }
        states
    }

    /// Get state at specific level (for cross-level queries)
    pub fn get_level_states(&self, target_level: usize) -> Vec<&RealHV> {
        if self.level == target_level {
            vec![&self.state]
        } else if self.level < target_level {
            match &self.children {
                Some((left, right)) => {
                    let mut states = left.get_level_states(target_level);
                    states.extend(right.get_level_states(target_level));
                    states
                }
                None => vec![],
            }
        } else {
            vec![]
        }
    }
}

// =============================================================================
// HIERARCHICAL CANTOR-LTC NETWORK
// =============================================================================

/// The complete Hierarchical Cantor-LTC Network
///
/// A 7-level self-similar structure with 255 nodes, each operating
/// at different time scales (1000ms root → 0.46ms leaves).
///
/// ## Usage
///
/// ```rust,ignore
/// use symthaea::hierarchical_cantor_ltc::{HierarchicalCantorLtcNetwork, CantorLtcConfig};
///
/// let config = CantorLtcConfig::default();
/// let mut network = HierarchicalCantorLtcNetwork::new(config);
///
/// // Simulate for 1 second
/// for _ in 0..1000 {
///     network.step(1.0); // 1ms steps
/// }
///
/// // Get hierarchical Φ
/// let phi = network.compute_hierarchical_phi();
/// println!("Hierarchical Φ = {:.4}", phi);
/// ```
pub struct HierarchicalCantorLtcNetwork {
    /// Root node (contains full tree)
    pub root: CantorLtcNode,
    /// Configuration
    pub config: CantorLtcConfig,
    /// Total simulation time (ms)
    pub time: f32,
    /// Global hierarchical Φ
    pub global_phi: f64,
    /// Per-level Φ measurements
    pub level_phi: Vec<f64>,
}

impl HierarchicalCantorLtcNetwork {
    /// Create a new Hierarchical Cantor-LTC Network
    pub fn new(config: CantorLtcConfig) -> Self {
        let root = CantorLtcNode::create_tree(0, 0, &config);
        let max_depth = config.max_depth;

        Self {
            root,
            config,
            time: 0.0,
            global_phi: 0.0,
            level_phi: vec![0.0; max_depth + 1],
        }
    }

    /// Step the entire network forward by dt milliseconds
    pub fn step(&mut self, dt: f32) {
        self.root.step(dt, None);
        self.time += dt;
    }

    /// Step the network with external input (inject at root)
    pub fn step_with_input(&mut self, dt: f32, input: &RealHV) {
        // Blend input with root state before stepping
        self.root.state = RealHV::bundle(&[
            self.root.state.clone(),
            input.clone(),
        ]);
        self.step(dt);
    }

    /// Compute hierarchical Φ (weighted sum across levels)
    ///
    /// Higher levels contribute more to global consciousness
    /// (they integrate more information across time scales)
    pub fn compute_hierarchical_phi(&mut self) -> f64 {
        // Compute Φ at each level
        for level in 0..=self.config.max_depth {
            let level_states = self.root.get_level_states(level);
            self.level_phi[level] = self.compute_level_phi(&level_states);
        }

        // Weight by level (higher levels = more integration = more weight)
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;
        for (level, &phi) in self.level_phi.iter().enumerate() {
            let weight = (self.config.max_depth - level + 1) as f64;
            weighted_sum += weight * phi;
            weight_sum += weight;
        }

        self.global_phi = if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        };

        self.global_phi
    }

    /// Compute Φ for a set of states at a single level
    fn compute_level_phi(&self, states: &[&RealHV]) -> f64 {
        let n = states.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mut total_sim = 0.0;
        let mut count = 0;
        for i in 0..states.len() {
            for j in (i + 1)..states.len() {
                total_sim += states[i].similarity(states[j]) as f64;
                count += 1;
            }
        }

        let avg_sim = if count > 0 {
            total_sim / count as f64
        } else {
            0.0
        };

        // Φ approximation
        let optimal_sim = 0.4;
        let deviation = (avg_sim - optimal_sim).abs();
        (1.0 - deviation * 2.0).max(0.0) * n.ln() / 10.0
    }

    /// Get the root state (global representation)
    pub fn get_root_state(&self) -> &RealHV {
        &self.root.state
    }

    /// Get states at a specific level
    pub fn get_level_states(&self, level: usize) -> Vec<&RealHV> {
        self.root.get_level_states(level)
    }

    /// Query by similarity (find matching nodes)
    pub fn query(&self, query: &RealHV, top_k: usize) -> Vec<(usize, usize, f32)> {
        let mut results = Vec::new();
        self.query_recursive(&self.root, query, &mut results);
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(top_k);
        results
    }

    fn query_recursive(&self, node: &CantorLtcNode, query: &RealHV, results: &mut Vec<(usize, usize, f32)>) {
        let similarity = node.state.similarity(query);
        results.push((node.level, node.index, similarity));

        if let Some((ref left, ref right)) = node.children {
            self.query_recursive(left, query, results);
            self.query_recursive(right, query, results);
        }
    }

    /// Get summary statistics
    pub fn summary(&self) -> NetworkSummary {
        NetworkSummary {
            total_nodes: self.config.total_nodes(),
            max_depth: self.config.max_depth,
            base_tau: self.config.base_tau,
            leaf_tau: self.config.tau_at_level(self.config.max_depth),
            simulation_time: self.time,
            global_phi: self.global_phi,
            level_phi: self.level_phi.clone(),
        }
    }
}

/// Summary statistics for the network
#[derive(Clone, Debug)]
pub struct NetworkSummary {
    pub total_nodes: usize,
    pub max_depth: usize,
    pub base_tau: f32,
    pub leaf_tau: f32,
    pub simulation_time: f32,
    pub global_phi: f64,
    pub level_phi: Vec<f64>,
}

impl std::fmt::Display for NetworkSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "╔══════════════════════════════════════════════════════════════╗")?;
        writeln!(f, "║       Hierarchical Cantor-LTC Network Summary                ║")?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  Total Nodes:      {:>6}                                    ║", self.total_nodes)?;
        writeln!(f, "║  Max Depth:        {:>6}                                    ║", self.max_depth)?;
        writeln!(f, "║  Root τ:           {:>6.2} ms                                ║", self.base_tau)?;
        writeln!(f, "║  Leaf τ:           {:>6.2} ms                                ║", self.leaf_tau)?;
        writeln!(f, "║  Simulation Time:  {:>6.2} ms                                ║", self.simulation_time)?;
        writeln!(f, "║  Global Φ:         {:>6.4}                                  ║", self.global_phi)?;
        writeln!(f, "╠══════════════════════════════════════════════════════════════╣")?;
        writeln!(f, "║  Per-Level Φ:                                                ║")?;
        for (level, phi) in self.level_phi.iter().enumerate() {
            let tau = self.base_tau * (1.0_f32 / 3.0).powi(level as i32);
            writeln!(f, "║    Level {}: Φ = {:.4} (τ = {:>7.2} ms)                    ║", level, phi, tau)?;
        }
        writeln!(f, "╚══════════════════════════════════════════════════════════════╝")
    }
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_tau_calculation() {
        let config = CantorLtcConfig::default();

        assert!((config.tau_at_level(0) - 1000.0).abs() < 0.01);
        assert!((config.tau_at_level(1) - 333.33).abs() < 0.5);
        assert!((config.tau_at_level(2) - 111.11).abs() < 0.5);
        assert!((config.tau_at_level(7) - 0.46).abs() < 0.1);
    }

    #[test]
    fn test_node_count() {
        let config = CantorLtcConfig::default();
        assert_eq!(config.total_nodes(), 255);  // 2^8 - 1
    }

    #[test]
    fn test_tree_creation() {
        let config = CantorLtcConfig {
            max_depth: 3,  // Small tree for testing
            ..Default::default()
        };
        let tree = CantorLtcNode::create_tree(0, 0, &config);

        assert_eq!(tree.level, 0);
        assert!(tree.children.is_some());
        assert_eq!(tree.count_nodes(), 15);  // 2^4 - 1
    }

    #[test]
    fn test_network_creation() {
        let config = CantorLtcConfig {
            max_depth: 3,
            ..Default::default()
        };
        let network = HierarchicalCantorLtcNetwork::new(config);

        assert_eq!(network.root.level, 0);
        assert!((network.root.tau - 1000.0).abs() < 0.01);
    }

    #[test]
    fn test_network_stepping() {
        let config = CantorLtcConfig {
            max_depth: 3,
            ..Default::default()
        };
        let mut network = HierarchicalCantorLtcNetwork::new(config);

        // Step forward
        for _ in 0..100 {
            network.step(1.0);
        }

        assert!((network.time - 100.0).abs() < 0.01);
    }

    #[test]
    fn test_phi_computation() {
        let config = CantorLtcConfig {
            max_depth: 3,
            ..Default::default()
        };
        let mut network = HierarchicalCantorLtcNetwork::new(config);

        // Run simulation
        for _ in 0..100 {
            network.step(1.0);
        }

        let phi = network.compute_hierarchical_phi();
        assert!(phi >= 0.0);
        assert!(phi <= 1.0);
    }

    #[test]
    fn test_query() {
        let config = CantorLtcConfig {
            max_depth: 3,
            ..Default::default()
        };
        let network = HierarchicalCantorLtcNetwork::new(config);

        let query = RealHV::random(HDC_DIMENSION, 12345);
        let results = network.query(&query, 5);

        assert!(!results.is_empty());
        assert!(results.len() <= 5);
        // Results should be sorted by similarity (descending)
        for i in 0..results.len().saturating_sub(1) {
            assert!(results[i].2 >= results[i + 1].2);
        }
    }

    #[test]
    fn test_level_states() {
        let config = CantorLtcConfig {
            max_depth: 3,
            ..Default::default()
        };
        let network = HierarchicalCantorLtcNetwork::new(config);

        // Level 0 should have 1 node (root)
        assert_eq!(network.get_level_states(0).len(), 1);

        // Level 1 should have 2 nodes
        assert_eq!(network.get_level_states(1).len(), 2);

        // Level 2 should have 4 nodes
        assert_eq!(network.get_level_states(2).len(), 4);

        // Level 3 should have 8 nodes
        assert_eq!(network.get_level_states(3).len(), 8);
    }
}
