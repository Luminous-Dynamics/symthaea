//! Multi-Scale Fractal Integration
//!
//! # Research Direction: Fractal Consciousness Architecture
//!
//! This module implements a self-similar, multi-scale topology where the same
//! consciousness-optimizing patterns appear at different scales - fractals within fractals.
//!
//! # Theoretical Foundation
//!
//! Consciousness may exhibit fractal properties:
//! - **Self-similarity**: Same patterns at micro (neurons), meso (areas), macro (networks)
//! - **Scale invariance**: Φ optimization principles apply at every level
//! - **Nested integration**: Each node contains its own integrated micro-network
//! - **Cross-scale bridges**: Information flows between scales, not just within
//!
//! # Architecture
//!
//! ```text
//! Scale 3 (Macro): Global Workspace
//!     │
//!     ├── Module Hub ─────────── Module Hub
//!     │       │                       │
//!     │   Scale 2 (Meso):         Scale 2 (Meso):
//!     │   ┌───────────┐          ┌───────────┐
//!     │   │ ●─●─●     │          │ ●─●─●     │
//!     │   │ │ │ │     │          │ │ │ │     │
//!     │   │ ●─●─●     │          │ ●─●─●     │
//!     │   └───────────┘          └───────────┘
//!     │       │                       │
//!     │   Scale 1 (Micro):        Scale 1 (Micro):
//!     │   Each node contains      Each node contains
//!     │   its own mini-network    its own mini-network
//! ```
//!
//! # Key Insight
//!
//! If bridge ratio ~40-45% optimizes Φ at one scale, the same principle
//! should apply at every scale. This creates a fractal of optimal integration.

use super::real_hv::RealHV;
use super::phi_real::RealPhiCalculator;
use std::collections::HashMap;

/// Configuration for fractal topology
#[derive(Clone, Debug)]
pub struct FractalConfig {
    /// Number of scales (depth of fractal)
    pub n_scales: usize,
    /// Nodes per scale (branching factor)
    pub nodes_per_scale: usize,
    /// Target bridge ratio at each scale
    pub bridge_ratio: f64,
    /// Target density at each scale
    pub density: f64,
    /// Cross-scale coupling strength
    pub cross_scale_coupling: f64,
    /// HDC dimension
    pub dim: usize,
}

impl Default for FractalConfig {
    fn default() -> Self {
        Self {
            n_scales: 3,
            nodes_per_scale: 8,
            bridge_ratio: 0.425,  // Optimal from bridge hypothesis
            density: 0.10,
            cross_scale_coupling: 0.3,
            dim: 2048,
        }
    }
}

/// A node that can contain a sub-topology (fractal recursion)
#[derive(Clone)]
pub struct FractalNode {
    /// Unique ID
    pub id: usize,
    /// Current scale level (0 = finest, higher = coarser)
    pub scale: usize,
    /// Module assignment at this scale
    pub module: usize,
    /// HDC state representation
    pub state: RealHV,
    /// Activity level
    pub activity: f64,
    /// Sub-topology (fractal children) - None at finest scale
    pub children: Option<Box<FractalSubTopology>>,
    /// Connections at this scale
    pub connections: Vec<usize>,
}

impl FractalNode {
    /// Create a leaf node (finest scale, no children)
    pub fn leaf(id: usize, module: usize, dim: usize, seed: u64) -> Self {
        Self {
            id,
            scale: 0,
            module,
            state: RealHV::random(dim, seed),
            activity: 0.0,
            children: None,
            connections: Vec::new(),
        }
    }

    /// Create a fractal node with sub-topology
    pub fn fractal(id: usize, scale: usize, module: usize, dim: usize, seed: u64, sub: FractalSubTopology) -> Self {
        Self {
            id,
            scale,
            module,
            state: RealHV::random(dim, seed),
            activity: 0.0,
            children: Some(Box::new(sub)),
            connections: Vec::new(),
        }
    }

    /// Integrate state from children (bottom-up)
    pub fn integrate_from_children(&mut self) {
        if let Some(ref children) = self.children {
            // Bundle all child states to create this node's state
            let child_states: Vec<RealHV> = children.nodes.values()
                .map(|n| n.state.clone())
                .collect();

            if !child_states.is_empty() {
                self.state = RealHV::bundle(&child_states);
            }

            // Activity is average of child activities
            let total_activity: f64 = children.nodes.values()
                .map(|n| n.activity)
                .sum();
            self.activity = total_activity / children.nodes.len().max(1) as f64;
        }
    }

    /// Propagate state to children (top-down)
    pub fn propagate_to_children(&mut self, coupling: f64) {
        if let Some(ref mut children) = self.children {
            let parent_influence = self.state.scale(coupling as f32);
            for child in children.nodes.values_mut() {
                child.state = child.state.add(&parent_influence).normalize();
            }
        }
    }

    /// Compute local Φ (for this node's sub-topology)
    pub fn local_phi(&self, phi_calc: &RealPhiCalculator) -> f64 {
        if let Some(ref children) = self.children {
            let child_states: Vec<RealHV> = children.nodes.values()
                .map(|n| n.state.clone())
                .collect();
            phi_calc.compute(&child_states)
        } else {
            // Leaf node - no local Φ
            0.0
        }
    }

    /// Get depth of fractal below this node
    pub fn depth(&self) -> usize {
        match &self.children {
            Some(sub) => 1 + sub.nodes.values()
                .map(|n| n.depth())
                .max()
                .unwrap_or(0),
            None => 0,
        }
    }

    /// Count total nodes in fractal subtree
    pub fn total_nodes(&self) -> usize {
        match &self.children {
            Some(sub) => 1 + sub.nodes.values()
                .map(|n| n.total_nodes())
                .sum::<usize>(),
            None => 1,
        }
    }
}

impl std::fmt::Debug for FractalNode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FractalNode")
            .field("id", &self.id)
            .field("scale", &self.scale)
            .field("module", &self.module)
            .field("activity", &self.activity)
            .field("has_children", &self.children.is_some())
            .field("connections", &self.connections)
            .finish()
    }
}

/// A sub-topology (one scale of the fractal)
#[derive(Clone)]
pub struct FractalSubTopology {
    /// Nodes at this scale
    pub nodes: HashMap<usize, FractalNode>,
    /// Edges at this scale
    pub edges: Vec<(usize, usize)>,
    /// Scale level
    pub scale: usize,
    /// Number of modules at this scale
    pub n_modules: usize,
}

impl FractalSubTopology {
    /// Create empty sub-topology
    pub fn new(scale: usize, n_modules: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            scale,
            n_modules,
        }
    }

    /// Add a node
    pub fn add_node(&mut self, node: FractalNode) {
        self.nodes.insert(node.id, node);
    }

    /// Add an edge
    pub fn add_edge(&mut self, from: usize, to: usize) {
        if !self.edges.contains(&(from, to)) && !self.edges.contains(&(to, from)) {
            self.edges.push((from, to));

            // Update node connections
            if let Some(node) = self.nodes.get_mut(&from) {
                if !node.connections.contains(&to) {
                    node.connections.push(to);
                }
            }
            if let Some(node) = self.nodes.get_mut(&to) {
                if !node.connections.contains(&from) {
                    node.connections.push(from);
                }
            }
        }
    }

    /// Compute Φ for this topology level
    pub fn compute_phi(&self, phi_calc: &RealPhiCalculator) -> f64 {
        let states: Vec<RealHV> = self.nodes.values()
            .map(|n| n.state.clone())
            .collect();
        phi_calc.compute(&states)
    }

    /// Get bridge ratio at this scale
    pub fn bridge_ratio(&self) -> f64 {
        if self.edges.is_empty() {
            return 0.0;
        }

        let bridge_count = self.edges.iter()
            .filter(|&&(a, b)| {
                let ma = self.nodes.get(&a).map(|n| n.module);
                let mb = self.nodes.get(&b).map(|n| n.module);
                ma != mb
            })
            .count();

        bridge_count as f64 / self.edges.len() as f64
    }
}

impl std::fmt::Debug for FractalSubTopology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FractalSubTopology")
            .field("scale", &self.scale)
            .field("n_nodes", &self.nodes.len())
            .field("n_edges", &self.edges.len())
            .field("n_modules", &self.n_modules)
            .finish()
    }
}

/// Multi-scale fractal consciousness topology
#[derive(Clone)]
pub struct FractalConsciousness {
    /// Root topology (coarsest scale)
    root: FractalSubTopology,
    /// Configuration
    config: FractalConfig,
    /// Φ calculator
    phi_calc: RealPhiCalculator,
    /// ID counter for unique node IDs
    next_id: usize,
}

impl FractalConsciousness {
    /// Create a new fractal consciousness topology
    pub fn new(config: FractalConfig) -> Self {
        let mut fc = Self {
            root: FractalSubTopology::new(config.n_scales - 1, 4),
            config: config.clone(),
            phi_calc: RealPhiCalculator::new(),
            next_id: 0,
        };

        // Build fractal structure recursively
        fc.build_fractal();
        fc
    }

    /// Build the fractal structure
    fn build_fractal(&mut self) {
        let top_scale = self.config.n_scales - 1;
        let n_modules = 4;  // Like ConsciousnessOptimized

        // Create top-level nodes
        for module in 0..n_modules {
            let node = self.create_fractal_node(top_scale, module);
            self.root.add_node(node);
        }

        // Create consciousness-optimized edges at top level
        self.create_optimal_edges(&mut self.root.clone());

        // Replace root with the connected version
        // (This is a bit awkward due to borrow checker)
        let connected_root = {
            let mut r = self.root.clone();
            self.create_optimal_edges(&mut r);
            r
        };
        self.root = connected_root;
    }

    /// Create a fractal node with children (recursive)
    fn create_fractal_node(&mut self, scale: usize, module: usize) -> FractalNode {
        let id = self.next_id;
        self.next_id += 1;

        if scale == 0 {
            // Base case: leaf node
            FractalNode::leaf(id, module, self.config.dim, id as u64 * 12345)
        } else {
            // Recursive case: create sub-topology
            let mut sub = FractalSubTopology::new(scale - 1, 4);

            // Create child nodes
            for child_module in 0..self.config.nodes_per_scale {
                let child = self.create_fractal_node(scale - 1, child_module % 4);
                sub.add_node(child);
            }

            // Create edges in sub-topology
            self.create_optimal_edges(&mut sub);

            FractalNode::fractal(id, scale, module, self.config.dim, id as u64 * 12345, sub)
        }
    }

    /// Create consciousness-optimized edges for a topology
    fn create_optimal_edges(&self, topo: &mut FractalSubTopology) {
        let node_ids: Vec<usize> = topo.nodes.keys().copied().collect();
        let n = node_ids.len();
        if n < 2 {
            return;
        }

        // Target number of edges based on density
        let max_edges = n * (n - 1) / 2;
        let target_edges = ((max_edges as f64) * self.config.density).ceil() as usize;
        let target_bridges = ((target_edges as f64) * self.config.bridge_ratio).ceil() as usize;

        // First add bridges (cross-module)
        let mut bridge_count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if bridge_count >= target_bridges {
                    break;
                }

                let id_i = node_ids[i];
                let id_j = node_ids[j];

                let module_i = topo.nodes.get(&id_i).map(|n| n.module);
                let module_j = topo.nodes.get(&id_j).map(|n| n.module);

                if module_i != module_j {
                    topo.add_edge(id_i, id_j);
                    bridge_count += 1;
                }
            }
            if bridge_count >= target_bridges {
                break;
            }
        }

        // Then add intra-module edges
        let intra_target = target_edges.saturating_sub(bridge_count);
        let mut intra_count = 0;
        for i in 0..n {
            for j in (i + 1)..n {
                if intra_count >= intra_target {
                    break;
                }

                let id_i = node_ids[i];
                let id_j = node_ids[j];

                if topo.edges.contains(&(id_i, id_j)) || topo.edges.contains(&(id_j, id_i)) {
                    continue;
                }

                let module_i = topo.nodes.get(&id_i).map(|n| n.module);
                let module_j = topo.nodes.get(&id_j).map(|n| n.module);

                if module_i == module_j {
                    topo.add_edge(id_i, id_j);
                    intra_count += 1;
                }
            }
            if intra_count >= intra_target {
                break;
            }
        }
    }

    /// Compute multi-scale Φ (Φ at each scale combined)
    pub fn multi_scale_phi(&self) -> MultiScalePhi {
        let mut scale_phis = Vec::new();

        // Top-level Φ
        let top_phi = self.root.compute_phi(&self.phi_calc);
        scale_phis.push((self.config.n_scales - 1, top_phi));

        // Φ at each lower scale (averaged across all sub-topologies at that scale)
        for scale in (0..self.config.n_scales - 1).rev() {
            let phis_at_scale = self.collect_phis_at_scale(scale);
            if !phis_at_scale.is_empty() {
                let avg_phi = phis_at_scale.iter().sum::<f64>() / phis_at_scale.len() as f64;
                scale_phis.push((scale, avg_phi));
            }
        }

        // Combined Φ (weighted sum, higher scales weighted more)
        let total_weight: f64 = scale_phis.iter()
            .map(|(s, _)| *s as f64 + 1.0)
            .sum();

        let weighted_phi: f64 = scale_phis.iter()
            .map(|(s, phi)| (*s as f64 + 1.0) * phi)
            .sum::<f64>() / total_weight;

        MultiScalePhi {
            scale_phis,
            combined_phi: weighted_phi,
            n_scales: self.config.n_scales,
        }
    }

    /// Collect all Φ values at a specific scale
    fn collect_phis_at_scale(&self, target_scale: usize) -> Vec<f64> {
        let mut phis = Vec::new();
        self.collect_phis_recursive(&self.root, target_scale, &mut phis);
        phis
    }

    fn collect_phis_recursive(&self, topo: &FractalSubTopology, target_scale: usize, phis: &mut Vec<f64>) {
        if topo.scale == target_scale {
            phis.push(topo.compute_phi(&self.phi_calc));
        }

        // Recurse into children
        for node in topo.nodes.values() {
            if let Some(ref children) = node.children {
                self.collect_phis_recursive(children, target_scale, phis);
            }
        }
    }

    /// Perform integration step (bottom-up then top-down)
    pub fn integrate_step(&mut self) {
        // Bottom-up: aggregate child states
        self.integrate_bottom_up(&mut self.root.clone());

        // Top-down: propagate parent influences
        self.integrate_top_down(&mut self.root.clone());
    }

    fn integrate_bottom_up(&self, topo: &mut FractalSubTopology) {
        for node in topo.nodes.values_mut() {
            // First recurse to children
            if let Some(ref mut children) = node.children {
                self.integrate_bottom_up(children);
            }
            // Then integrate from children
            node.integrate_from_children();
        }
    }

    fn integrate_top_down(&self, topo: &mut FractalSubTopology) {
        for node in topo.nodes.values_mut() {
            // First propagate to children
            node.propagate_to_children(self.config.cross_scale_coupling);
            // Then recurse
            if let Some(ref mut children) = node.children {
                self.integrate_top_down(children);
            }
        }
    }

    /// Activate a top-level module
    pub fn activate_module(&mut self, module: usize, input: &RealHV) {
        for node in self.root.nodes.values_mut() {
            if node.module == module {
                node.state = node.state.bind(input);
                node.activity = 1.0;
            }
        }
    }

    /// Get metrics
    pub fn metrics(&self) -> FractalMetrics {
        let ms_phi = self.multi_scale_phi();

        // Count total nodes across all scales
        let total_nodes: usize = self.root.nodes.values()
            .map(|n| n.total_nodes())
            .sum();

        // Count total edges across all scales
        let mut total_edges = self.root.edges.len();
        for node in self.root.nodes.values() {
            total_edges += self.count_edges_recursive(node);
        }

        FractalMetrics {
            n_scales: self.config.n_scales,
            total_nodes,
            total_edges,
            top_level_phi: ms_phi.scale_phis.first().map(|(_, p)| *p).unwrap_or(0.0),
            combined_phi: ms_phi.combined_phi,
            top_level_bridge_ratio: self.root.bridge_ratio(),
            scale_phis: ms_phi.scale_phis,
        }
    }

    fn count_edges_recursive(&self, node: &FractalNode) -> usize {
        match &node.children {
            Some(sub) => {
                sub.edges.len() + sub.nodes.values()
                    .map(|n| self.count_edges_recursive(n))
                    .sum::<usize>()
            }
            None => 0,
        }
    }

    /// Get the root topology
    pub fn root(&self) -> &FractalSubTopology {
        &self.root
    }
}

impl std::fmt::Debug for FractalConsciousness {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FractalConsciousness")
            .field("n_scales", &self.config.n_scales)
            .field("root", &self.root)
            .finish()
    }
}

/// Multi-scale Φ computation result
#[derive(Clone, Debug)]
pub struct MultiScalePhi {
    /// Φ at each scale (scale, phi)
    pub scale_phis: Vec<(usize, f64)>,
    /// Combined (weighted) Φ
    pub combined_phi: f64,
    /// Number of scales
    pub n_scales: usize,
}

impl std::fmt::Display for MultiScalePhi {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MultiScaleΦ: combined={:.4} [", self.combined_phi)?;
        for (i, (scale, phi)) in self.scale_phis.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "S{}={:.4}", scale, phi)?;
        }
        write!(f, "]")
    }
}

/// Fractal consciousness metrics
#[derive(Clone, Debug)]
pub struct FractalMetrics {
    /// Number of scales
    pub n_scales: usize,
    /// Total nodes across all scales
    pub total_nodes: usize,
    /// Total edges across all scales
    pub total_edges: usize,
    /// Φ at top level
    pub top_level_phi: f64,
    /// Combined multi-scale Φ
    pub combined_phi: f64,
    /// Bridge ratio at top level
    pub top_level_bridge_ratio: f64,
    /// Φ at each scale
    pub scale_phis: Vec<(usize, f64)>,
}

impl std::fmt::Display for FractalMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "FractalConsciousness: {} scales, {} nodes, {} edges, Φ_top={:.4}, Φ_combined={:.4}, bridge_ratio={:.1}%",
            self.n_scales,
            self.total_nodes,
            self.total_edges,
            self.top_level_phi,
            self.combined_phi,
            self.top_level_bridge_ratio * 100.0
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fractal_creation() {
        let config = FractalConfig {
            n_scales: 2,
            nodes_per_scale: 4,
            dim: 1024,
            ..Default::default()
        };

        let fc = FractalConsciousness::new(config);
        let metrics = fc.metrics();

        println!("{}", metrics);
        assert!(metrics.total_nodes > 4);  // More than just top level
        assert!(metrics.combined_phi > 0.0);
    }

    #[test]
    fn test_multi_scale_phi() {
        let config = FractalConfig {
            n_scales: 3,
            nodes_per_scale: 4,
            dim: 1024,
            bridge_ratio: 0.425,
            ..Default::default()
        };

        let fc = FractalConsciousness::new(config);
        let ms_phi = fc.multi_scale_phi();

        println!("{}", ms_phi);

        // Should have Φ at each scale
        assert!(!ms_phi.scale_phis.is_empty());
        assert!(ms_phi.combined_phi > 0.0);
    }

    #[test]
    fn test_bridge_ratio_fractal() {
        let config = FractalConfig {
            n_scales: 2,
            nodes_per_scale: 8,
            dim: 1024,
            bridge_ratio: 0.45,
            density: 0.15,
            ..Default::default()
        };

        let target_bridge = config.bridge_ratio;
        let fc = FractalConsciousness::new(config);

        println!("Top-level bridge ratio: {:.1}%", fc.root.bridge_ratio() * 100.0);
        println!("Target: {:.1}%", target_bridge * 100.0);

        // Bridge ratio should be close to target
        let diff = (fc.root.bridge_ratio() - target_bridge).abs();
        assert!(diff < 0.2, "Bridge ratio {:.2} too far from target {:.2}",
                fc.root.bridge_ratio(), target_bridge);
    }

    #[test]
    fn test_fractal_depth() {
        let config = FractalConfig {
            n_scales: 3,
            nodes_per_scale: 4,
            dim: 512,
            ..Default::default()
        };

        let fc = FractalConsciousness::new(config);

        // Check depth of fractal
        let max_depth = fc.root.nodes.values()
            .map(|n| n.depth())
            .max()
            .unwrap_or(0);

        println!("Fractal depth: {}", max_depth);
        // Should have depth = n_scales - 1
        assert!(max_depth >= 1);
    }

    #[test]
    fn test_scale_comparison() {
        println!("\nScale Comparison Test:");
        println!("{}", "─".repeat(50));

        for n_scales in 1..=3 {
            let config = FractalConfig {
                n_scales,
                nodes_per_scale: 4,
                dim: 1024,
                ..Default::default()
            };

            let fc = FractalConsciousness::new(config);
            let metrics = fc.metrics();

            println!("Scales={}: nodes={}, edges={}, Φ_combined={:.4}",
                     n_scales, metrics.total_nodes, metrics.total_edges, metrics.combined_phi);
        }
    }
}
