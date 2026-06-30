// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
use nalgebra::DMatrix;
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
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

pub trait CoarseGrainer {
    fn name(&self) -> &'static str;
    fn coarse_grain(&self, graph: &UnGraph<(), ()>) -> Result<UnGraph<(), ()>, CoarseGrainError>;
}

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
        if graph.node_count() == 0 {
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

#[derive(Clone, Copy, Debug)]
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
        let modules = self.cover_modules(graph)?;
        coarse_grain_from_modules(graph, &modules)
    }
}

impl BoxCoveringCoarseGrainer {
    pub fn cover_modules(
        &self,
        graph: &UnGraph<(), ()>,
    ) -> Result<HashMap<usize, usize>, CoarseGrainError> {
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

        Ok(modules)
    }

    pub fn box_count(&self, graph: &UnGraph<(), ()>) -> Result<usize, CoarseGrainError> {
        let modules = self.cover_modules(graph)?;
        Ok(modules.values().copied().collect::<HashSet<_>>().len())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BoxDimensionEstimate {
    pub radii: Vec<usize>,
    pub counts: Vec<usize>,
    pub dimension: f64,
    pub r_squared: f64,
}

pub struct BoxDimensionEstimator;

impl BoxDimensionEstimator {
    pub fn estimate(
        graph: &UnGraph<(), ()>,
        max_radius: usize,
    ) -> Result<BoxDimensionEstimate, CoarseGrainError> {
        if graph.node_count() == 0 {
            return Err(CoarseGrainError::EmptyGraph);
        }

        let max_radius = max_radius.max(1);
        let mut radii = Vec::new();
        let mut counts = Vec::new();

        for radius in 1..=max_radius {
            let grainer = BoxCoveringCoarseGrainer { radius };
            let count = grainer.box_count(graph)?;
            radii.push(radius);
            counts.push(count);
        }

        let xs: Vec<f64> = radii.iter().map(|&r| (1.0 / r as f64).ln()).collect();
        let ys: Vec<f64> = counts.iter().map(|&c| (c.max(1) as f64).ln()).collect();
        let (slope, r_squared) = linear_slope_and_r2(&xs, &ys);

        Ok(BoxDimensionEstimate {
            radii,
            counts,
            dimension: slope.max(0.0),
            r_squared,
        })
    }
}

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

pub struct MultiScalePhi;

impl MultiScalePhi {
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

    pub fn integration_survival(
        &self,
        original: &UnGraph<(), ()>,
        coarse: &UnGraph<(), ()>,
    ) -> f64 {
        let before = self.phi_proxy(original);
        let after = self.phi_proxy(coarse);

        if before > f64::EPSILON {
            (after / before).clamp(0.0, 2.0)
        } else {
            0.0
        }
    }

    pub fn scale_collapse_error(
        &self,
        original: &UnGraph<(), ()>,
        coarse: &UnGraph<(), ()>,
    ) -> f64 {
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

fn linear_slope_and_r2(xs: &[f64], ys: &[f64]) -> (f64, f64) {
    if xs.len() != ys.len() || xs.len() < 2 {
        return (0.0, 0.0);
    }

    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;

    let ss_xy = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();

    let ss_xx = xs.iter().map(|x| (x - mean_x).powi(2)).sum::<f64>();

    if ss_xx <= f64::EPSILON {
        return (0.0, 0.0);
    }

    let slope = ss_xy / ss_xx;
    let intercept = mean_y - slope * mean_x;

    let ss_tot = ys.iter().map(|y| (y - mean_y).powi(2)).sum::<f64>();
    let ss_res = xs
        .iter()
        .zip(ys)
        .map(|(x, y)| {
            let y_hat = slope * x + intercept;
            (y - y_hat).powi(2)
        })
        .sum::<f64>();

    let r_squared = if ss_tot > f64::EPSILON {
        (1.0 - ss_res / ss_tot).clamp(0.0, 1.0)
    } else {
        0.0
    };

    (slope, r_squared)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_box_dimension_estimator_is_finite() {
        let mut graph = UnGraph::<(), ()>::default();
        let nodes: Vec<_> = (0..10).map(|_| graph.add_node(())).collect();
        for i in 0..9 {
            graph.add_edge(nodes[i], nodes[i + 1], ());
        }

        let estimate = BoxDimensionEstimator::estimate(&graph, 3).unwrap();
        assert_eq!(estimate.radii.len(), 3);
        assert!(estimate.dimension.is_finite());
        assert!(estimate.r_squared.is_finite());
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
    fn test_integration_proxy_nonnegative() {
        let mut graph = UnGraph::<(), ()>::default();
        let a = graph.add_node(());
        let b = graph.add_node(());
        graph.add_edge(a, b, ());

        let analyzer = MultiScalePhi;
        assert!(analyzer.phi_proxy(&graph) >= 0.0);
    }
}
