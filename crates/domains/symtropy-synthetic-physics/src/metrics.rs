// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Graph metrics — the observable state of a [`SyntheticGraph`] at one tick.
//!
//! These are the values fed into:
//! 1. [`AttractorClassifier`] — to classify attractor type
//! 2. [`StructuralFreeEnergy`] — to compute the update objective
//! 3. `symthaea-projection` — to emit [`ProjectionFrame`]s for the Time-Waterfall

use serde::{Deserialize, Serialize};

use crate::{attractor::AttractorClass, graph::SyntheticGraph};

/// A complete metric snapshot for one graph at one tick.
///
/// All fields are `f64` or `usize` so they can be serialized into
/// `ProjectionFrame::scalar_metrics` without loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetrics {
    /// Tick index at which this snapshot was taken.
    pub tick: u64,
    /// Total node count.
    pub node_count: usize,
    /// Total edge count.
    pub edge_count: usize,
    /// Maximum node degree.
    pub max_degree: u32,
    /// Average node degree.
    pub avg_degree: f64,
    /// Estimated intrinsic dimension (log N / log D proxy).
    pub estimated_dimension: f64,
    /// Graph diameter (estimated via BFS from one source).
    pub graph_diameter: usize,
    /// Global clustering coefficient (Watts-Strogatz).
    pub clustering_coefficient: f64,
    /// Spectral gap proxy (sqrt of max_degree * avg_degree).
    pub spectral_gap_proxy: f64,
    /// Edge churn this tick (adds + removes).
    pub edge_churn: usize,
    /// Shannon entropy of degree distribution (bits).
    pub degree_entropy: f64,
    /// Structural free energy (lower = more coherent).
    pub structural_free_energy: f64,
    /// Holonomy drift (curvature proxy — zero until Phase 4).
    pub holonomy_drift: f64,
    /// Betti-0: number of connected components.
    pub betti_0: usize,
    /// Betti-1 proxy: estimated cycles = E - V + C (cycle rank).
    pub betti_1_proxy: isize,
    /// Current attractor classification (updated by classifier).
    pub attractor_class: AttractorClass,
    /// Confidence score for the attractor classification [0, 1].
    pub classifier_confidence: f64,
}

impl GraphMetrics {
    /// Compute the full metric set from a [`SyntheticGraph`] at a given tick.
    pub fn compute(graph: &SyntheticGraph, tick: u64) -> Self {
        let node_count = graph.node_count();
        let edge_count = graph.edge_count();
        let max_degree = graph.max_degree();

        let avg_degree = if node_count > 0 {
            2.0 * edge_count as f64 / node_count as f64
        } else {
            0.0
        };

        let estimated_dimension = graph.estimate_dimension();
        let graph_diameter = graph.estimated_diameter();
        let clustering_coefficient = graph.clustering_coefficient();
        let spectral_gap_proxy = (max_degree as f64 * avg_degree).sqrt();
        let degree_entropy = graph.degree_entropy();
        let betti_0 = graph.connected_components();

        // Cycle rank (Euler characteristic proxy): |E| - |V| + |components|
        let betti_1_proxy = edge_count as isize - node_count as isize + betti_0 as isize;

        Self {
            tick,
            node_count,
            edge_count,
            max_degree,
            avg_degree,
            estimated_dimension,
            graph_diameter,
            clustering_coefficient,
            spectral_gap_proxy,
            edge_churn: 0, // filled by update loop
            degree_entropy,
            structural_free_energy: 0.0, // filled by free_energy module
            holonomy_drift: 0.0,         // Phase 4
            betti_0,
            betti_1_proxy,
            attractor_class: AttractorClass::Unknown,
            classifier_confidence: 0.0,
        }
    }

    /// Produce a human-readable one-line summary for logging.
    pub fn summary_line(&self) -> String {
        format!(
            "tick={} N={} E={} dim={:.2} D={} CC={:.3} β₀={} β₁≈{} F={:.3} class={:?}",
            self.tick,
            self.node_count,
            self.edge_count,
            self.estimated_dimension,
            self.graph_diameter,
            self.clustering_coefficient,
            self.betti_0,
            self.betti_1_proxy,
            self.structural_free_energy,
            self.attractor_class,
        )
    }

    /// Convert scalar metrics to a flat map for projection frame export.
    pub fn to_scalar_map(&self) -> std::collections::HashMap<String, f64> {
        let mut map = std::collections::HashMap::new();
        map.insert("node_count".to_string(), self.node_count as f64);
        map.insert("edge_count".to_string(), self.edge_count as f64);
        map.insert("max_degree".to_string(), self.max_degree as f64);
        map.insert("avg_degree".to_string(), self.avg_degree);
        map.insert("estimated_dimension".to_string(), self.estimated_dimension);
        map.insert("graph_diameter".to_string(), self.graph_diameter as f64);
        map.insert(
            "clustering_coefficient".to_string(),
            self.clustering_coefficient,
        );
        map.insert("spectral_gap_proxy".to_string(), self.spectral_gap_proxy);
        map.insert("edge_churn".to_string(), self.edge_churn as f64);
        map.insert("degree_entropy".to_string(), self.degree_entropy);
        map.insert(
            "structural_free_energy".to_string(),
            self.structural_free_energy,
        );
        map.insert("holonomy_drift".to_string(), self.holonomy_drift);
        map.insert("betti_0".to_string(), self.betti_0 as f64);
        map.insert("betti_1_proxy".to_string(), self.betti_1_proxy as f64);
        map.insert(
            "classifier_confidence".to_string(),
            self.classifier_confidence,
        );
        map
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::SyntheticGraph;

    #[test]
    fn metrics_from_initial_graph() {
        let graph = SyntheticGraph::new_with_seed(42, 16);
        let m = GraphMetrics::compute(&graph, 0);
        assert_eq!(m.node_count, 16);
        assert!(m.estimated_dimension >= 1.0);
        assert!(m.betti_0 >= 1);
        assert!(m.clustering_coefficient >= 0.0);
        assert!(m.clustering_coefficient <= 1.0);
        println!("{}", m.summary_line());
    }

    #[test]
    fn scalar_map_has_all_keys() {
        let graph = SyntheticGraph::new_with_seed(7, 8);
        let m = GraphMetrics::compute(&graph, 0);
        let map = m.to_scalar_map();
        assert!(map.contains_key("estimated_dimension"));
        assert!(map.contains_key("betti_0"));
        assert!(map.contains_key("structural_free_energy"));
    }
}
