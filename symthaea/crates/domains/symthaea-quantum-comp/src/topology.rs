//! Lightweight topology-oriented metrics.
//!
//! These are research proxies, not a replacement for persistent homology tools.

use crate::errors::{QuantumCompError, Result};

/// A compact topology summary for a point/similarity graph.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopologySummary {
    /// Number of graph nodes.
    pub nodes: usize,
    /// Number of undirected graph edges.
    pub edges: usize,
    /// Number of connected components.
    pub components: usize,
    /// Cycle-rank proxy: `edges - nodes + components`.
    pub beta1_proxy: isize,
    /// Density of the threshold graph in `[0, 1]`.
    pub edge_density: f32,
    /// Mean degree of the threshold graph.
    pub mean_degree: f32,
}

/// Builds a threshold graph from a symmetric similarity matrix and returns a
/// beta-1-like cycle proxy plus simple graph statistics.
pub fn threshold_graph_summary(
    similarities: &[Vec<f32>],
    threshold: f32,
) -> Result<TopologySummary> {
    if !(0.0..=1.0).contains(&threshold) {
        return Err(QuantumCompError::InvalidProbability);
    }
    let n = similarities.len();
    if n == 0 {
        return Err(QuantumCompError::InvalidDimension);
    }
    for row in similarities {
        if row.len() != n {
            return Err(QuantumCompError::DimensionMismatch {
                expected: n,
                actual: row.len(),
            });
        }
    }

    let mut parent: Vec<usize> = (0..n).collect();
    let mut edges = 0usize;

    for (i, row) in similarities.iter().enumerate() {
        for (j, similarity) in row.iter().enumerate().skip(i + 1) {
            if *similarity >= threshold {
                edges += 1;
                union(&mut parent, i, j);
            }
        }
    }

    let mut roots = Vec::new();
    for i in 0..n {
        let r = find(&mut parent, i);
        if !roots.contains(&r) {
            roots.push(r);
        }
    }
    let components = roots.len();
    let beta1_proxy = edges as isize - n as isize + components as isize;
    let possible_edges = n.saturating_mul(n.saturating_sub(1)) / 2;
    let edge_density = if possible_edges == 0 {
        0.0
    } else {
        edges as f32 / possible_edges as f32
    };
    let mean_degree = if n == 0 {
        0.0
    } else {
        2.0 * edges as f32 / n as f32
    };

    Ok(TopologySummary {
        nodes: n,
        edges,
        components,
        beta1_proxy,
        edge_density,
        mean_degree,
    })
}

fn find(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
        parent[x] = find(parent, parent[x]);
    }
    parent[x]
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let ra = find(parent, a);
    let rb = find(parent, b);
    if ra != rb {
        parent[rb] = ra;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangle_has_one_cycle_proxy() {
        let sim = vec![
            vec![1.0, 0.9, 0.9],
            vec![0.9, 1.0, 0.9],
            vec![0.9, 0.9, 1.0],
        ];
        let s = threshold_graph_summary(&sim, 0.8).unwrap();
        assert_eq!(s.beta1_proxy, 1);
        assert_eq!(s.edges, 3);
    }
}
