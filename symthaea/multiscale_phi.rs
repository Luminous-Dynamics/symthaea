// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use nalgebra::DMatrix;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use std::collections::{HashMap, HashSet, VecDeque};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum CoarseGrainError {
    #[error("missing node index {0} in module map")]
    MissingNode(usize),
    #[error("empty graph")]
    EmptyGraph,
    #[error("coarse graining produced no modules")]
    NoModules,
}

/// Trait for coarse-graining strategies.
pub trait CoarseGrainer {
    fn name(&self) -> &'static str;
    fn coarse_grain(&self, graph: &UnGraph<(), ()>) -> Result<UnGraph<(), ()>, CoarseGrainError>;
}

/// Manual module-based coarse graining.
pub struct ManualCoarseGrainer {
    pub modules: HashMap<usize, usize>,
}

impl CoarseGrainer for ManualCoarseGrainer {
    fn name(&self) -> &'static str {
        "manual-module"
    }

    fn coarse_grain(&self, graph: &UnGraph<(), ()>) -> Result<UnGraph<(), ()>, CoarseGrainError> {
        coarse_grain_from_modules(graph, &self.modules)
    }
}

/// True spectral bipartition coarse grainer using the Fiedler vector of the graph Laplacian.
pub struct SpectralCoarseGrainer;

impl CoarseGrainer for SpectralCoarseGrainer {
    fn name(&self) -> &'static str {
        "spectral-bipartition"
    }

    fn coarse_grain(&self, graph: &UnGraph<(), ()>) -> Result<UnGraph<(), ()>, CoarseGrainError> {
        let n = graph.node_count();

        if n == 0 {
            return Err(CoarseGrainError::EmptyGraph);
        }

        if n <= 2 {
            return Ok(graph.clone());
        }

        let laplacian = graph_laplacian(graph);
        let eig = laplacian.symmetric_eigen();

        let mut eigen_order: Vec<usize> = (0..n).collect();
        eigen_order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));

        let fiedler_col = eigen_order.get(1).copied().unwrap_or(0);
        let values: Vec<f64> = (0..n).map(|i| eig.eigenvectors[(i, fiedler_col)]).collect();

        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        let median = sorted[n / 2];

        let mut modules = HashMap::new();
        for (idx, value) in values.iter().enumerate() {
            modules.insert(idx, if *value <= median { 0 } else { 1 });
        }

        coarse_grain_from_modules(graph, &modules)
    }
}

/// Local degree-bin baseline. Useful as an ablation against spectral coarse-graining.
pub struct DegreeBinCoarseGrainer {
    pub bins: usize,
}

impl Default for DegreeBinCoarseGrainer {
    fn default() -> Self {
        Self { bins: 2 }
    }
}

impl CoarseGrainer for DegreeBinCoarseGrainer {
    fn name(&self) -> &'static str {
        "degree-bin-baseline"
    }

    fn coarse_grain(&self, graph: &UnGraph<(), ()>) -> Result<UnGraph<(), ()>, CoarseGrainError> {
        let n = graph.node_count();

        if n == 0 {
            return Err(CoarseGrainError::EmptyGraph);
        }

        let bins = self.bins.max(1);
        let max_degree = graph
            .node_indices()
            .map(|node| graph.neighbors(node).count())
            .max()
            .unwrap_or(0)
            .max(1);

        let mut modules = HashMap::new();

        for node in graph.node_indices() {
            let degree = graph.neighbors(node).count();
            let module = ((degree * bins) / (max_degree + 1)).min(bins - 1);
            modules.insert(node.index(), module);
        }

        coarse_grain_from_modules(graph, &modules)
    }
}

/// Greedy box-covering coarse grainer.
/// Each module is a graph-radius ball around a high-degree uncovered center.
/// This is an exploratory network-renormalization diagnostic, not an optimized
/// minimum-box-cover solver.
pub struct BoxCoveringCoarseGrainer {
    pub radius: usize,
}

impl Default for BoxCoveringCoarseGrainer {
    fn default() -> Self {
        Self { radius: 1 }
    }
}

impl CoarseGrainer for BoxCoveringCoarseGrainer {
    fn name(&self) -> &'static str {
        "greedy-box-covering"
    }

    fn coarse_grain(&self, graph: &UnGraph<(), ()>) -> Result<UnGraph<(), ()>, CoarseGrainError> {
        if graph.node_count() == 0 {
            return Err(CoarseGrainError::EmptyGraph);
        }

        let mut uncovered: HashSet<usize> = graph.node_indices().map(|node| node.index()).collect();
        let mut modules = HashMap::new();
        let mut module_id = 0usize;

        while !uncovered.is_empty() {
            let center = graph
                .node_indices()
                .filter(|node| uncovered.contains(&node.index()))
                .max_by_key(|node| graph.neighbors(*node).count())
                .ok_or(CoarseGrainError::NoModules)?;

            let covered = radius_ball(graph, center, self.radius);

            let mut assigned_any = false;
            for node_idx in covered {
                if uncovered.remove(&node_idx) {
                    modules.insert(node_idx, module_id);
                    assigned_any = true;
                }
            }

            if !assigned_any {
                let idx = center.index();
                uncovered.remove(&idx);
                modules.insert(idx, module_id);
            }

            module_id += 1;
        }

        coarse_grain_from_modules(graph, &modules)
    }
}

/// Backward-compatible odd/even placeholder baseline.
/// This is intentionally not called spectral.
pub struct OddEvenCoarseGrainer;

impl CoarseGrainer for OddEvenCoarseGrainer {
    fn name(&self) -> &'static str {
        "odd-even-placeholder-baseline"
    }

    fn coarse_grain(&self, graph: &UnGraph<(), ()>) -> Result<UnGraph<(), ()>, CoarseGrainError> {
        if graph.node_count() == 0 {
            return Err(CoarseGrainError::EmptyGraph);
        }

        let mut modules = HashMap::new();
        for node in graph.node_indices() {
            modules.insert(node.index(), node.index() % 2);
        }

        coarse_grain_from_modules(graph, &modules)
    }
}

/// Proxy for Integrated Information (Phi) on graphs.
pub struct MultiScalePhi;

impl MultiScalePhi {
    /// Effective Information proxy:
    /// EI = H(<W_i>) - <H(W_i)>,
    /// where W_i is the transition distribution from node i.
    pub fn effective_information(&self, graph: &UnGraph<(), ()>) -> f64 {
        let n = graph.node_count();

        if n == 0 {
            return 0.0;
        }

        let mut avg_w = vec![0.0; n];
        let mut sum_h = 0.0;

        for i in 0..n {
            let idx = NodeIndex::new(i);
            let neighbors: Vec<_> = graph.neighbors(idx).collect();
            let degree = neighbors.len();

            if degree > 0 {
                let p = 1.0 / degree as f64;
                let h_i = -(degree as f64 * p * p.log2());
                sum_h += h_i;

                for neighbor in neighbors {
                    avg_w[neighbor.index()] += p / n as f64;
                }
            } else {
                avg_w[i] += 1.0 / n as f64;
            }
        }

        let h_avg_w: f64 = avg_w
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum();

        let avg_h = sum_h / n as f64;
        (h_avg_w - avg_h).max(0.0)
    }

    pub fn phi_proxy(&self, graph: &UnGraph<(), ()>) -> f64 {
        self.effective_information(graph)
    }

    pub fn integration_survival(&self, original: &UnGraph<(), ()>, coarse: &UnGraph<(), ()>) -> f64 {
        let before = self.phi_proxy(original);
        let after = self.phi_proxy(coarse);

        if before > f64::EPSILON {
            (after / before).clamp(0.0, 2.0)
        } else {
            0.0
        }
    }

    pub fn scale_collapse_error(&self, original: &UnGraph<(), ()>, coarse: &UnGraph<(), ()>) -> f64 {
        let before = self.phi_proxy(original);
        let after = self.phi_proxy(coarse);

        (before - after).abs()
    }
}

fn coarse_grain_from_modules(
    graph: &UnGraph<(), ()>,
    modules: &HashMap<usize, usize>,
) -> Result<UnGraph<(), ()>, CoarseGrainError> {
    if graph.node_count() == 0 {
        return Err(CoarseGrainError::EmptyGraph);
    }

    for node in graph.node_indices() {
        if !modules.contains_key(&node.index()) {
            return Err(CoarseGrainError::MissingNode(node.index()));
        }
    }

    let mut new_graph = UnGraph::<(), ()>::default();
    let mut module_to_node = HashMap::new();

    for &module in modules.values() {
        module_to_node
            .entry(module)
            .or_insert_with(|| new_graph.add_node(()));
    }

    if module_to_node.is_empty() {
        return Err(CoarseGrainError::NoModules);
    }

    for edge in graph.edge_references() {
        let source_module = modules[&edge.source().index()];
        let target_module = modules[&edge.target().index()];

        if source_module != target_module {
            let source = module_to_node[&source_module];
            let target = module_to_node[&target_module];
            new_graph.update_edge(source, target, ());
        }
    }

    Ok(new_graph)
}

fn graph_laplacian(graph: &UnGraph<(), ()>) -> DMatrix<f64> {
    let n = graph.node_count();
    let mut matrix = DMatrix::<f64>::zeros(n, n);

    for node in graph.node_indices() {
        let i = node.index();
        let degree = graph.neighbors(node).count() as f64;
        matrix[(i, i)] = degree;

        for neighbor in graph.neighbors(node) {
            matrix[(i, neighbor.index())] -= 1.0;
        }
    }

    matrix
}

fn radius_ball(graph: &UnGraph<(), ()>, center: NodeIndex, radius: usize) -> Vec<usize> {
    let mut seen = HashSet::new();
    let mut queue = VecDeque::new();

    seen.insert(center.index());
    queue.push_back((center, 0usize));

    while let Some((node, depth)) = queue.pop_front() {
        if depth >= radius {
            continue;
        }

        for neighbor in graph.neighbors(node) {
            if seen.insert(neighbor.index()) {
                queue.push_back((neighbor, depth + 1));
            }
        }
    }

    seen.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_manual_coarse_grainer_reduces_nodes() {
        let mut graph = UnGraph::<(), ()>::default();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        graph.add_edge(n0, n1, ());

        let mut modules = HashMap::new();
        modules.insert(0, 0);
        modules.insert(1, 0);

        let grainer = ManualCoarseGrainer { modules };
        let coarse = grainer.coarse_grain(&graph).unwrap();

        assert_eq!(coarse.node_count(), 1);
        assert_eq!(coarse.edge_count(), 0);
    }

    #[test]
    fn test_coarse_grain_error_for_missing_node() {
        let mut graph = UnGraph::<(), ()>::default();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        graph.add_edge(n0, n1, ());

        let mut modules = HashMap::new();
        modules.insert(0, 0);

        let grainer = ManualCoarseGrainer { modules };
        assert!(grainer.coarse_grain(&graph).is_err());
    }

    #[test]
    fn test_spectral_coarse_grainer_is_finite() {
        let mut graph = UnGraph::<(), ()>::default();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());
        let d = graph.add_node(());

        graph.add_edge(a, b, ());
        graph.add_edge(b, c, ());
        graph.add_edge(c, d, ());

        let grainer = SpectralCoarseGrainer;
        let coarse = grainer.coarse_grain(&graph).unwrap();
        let analyzer = MultiScalePhi;

        assert!(coarse.node_count() <= graph.node_count());
        assert!(analyzer.phi_proxy(&coarse).is_finite());
    }

    #[test]
    fn test_box_covering_coarse_grainer_reduces_path() {
        let mut graph = UnGraph::<(), ()>::default();
        let nodes: Vec<_> = (0..6).map(|_| graph.add_node(())).collect();
        for i in 0..5 {
            graph.add_edge(nodes[i], nodes[i + 1], ());
        }

        let grainer = BoxCoveringCoarseGrainer { radius: 1 };
        let coarse = grainer.coarse_grain(&graph).unwrap();

        assert!(coarse.node_count() < graph.node_count());
    }

    #[test]
    fn test_degree_bin_baseline_is_finite() {
        let mut graph = UnGraph::<(), ()>::default();
        let a = graph.add_node(());
        let b = graph.add_node(());
        let c = graph.add_node(());

        graph.add_edge(a, b, ());
        graph.add_edge(a, c, ());

        let grainer = DegreeBinCoarseGrainer::default();
        let coarse = grainer.coarse_grain(&graph).unwrap();

        assert!(coarse.node_count() <= graph.node_count());
    }

    #[test]
    fn test_integration_proxy_nonnegative() {
        let mut graph = UnGraph::<(), ()>::default();
        let a = graph.add_node(());
        let b = graph.add_node(());
        graph.add_edge(a, b, ());

        let analyzer = MultiScalePhi;

        assert!(analyzer.phi_proxy(&graph) >= 0.0);
    }
}
