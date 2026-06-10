// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tier 4 topology generators: Neural and extended topologies.
//!
//! CorticalColumn, Feedforward, Recurrent, Bipartite, CorePeriphery,
//! BowTie, Attention, Residual, PetersenGraph, CompleteBipartite.

use super::super::unified_hv::ContinuousHV;
use super::types::{ConsciousnessTopology, TopologyType};

impl ConsciousnessTopology {
    // ==========================================================================
    // TIER 4: Extended Topologies (Revolutionary #102)
    // ==========================================================================

    /// Cortical Column - 6-layer hierarchical structure like mammalian cortex
    ///
    /// Layers: L1 (sparse) → L2/3 (feedback) → L4 (input) → L5 (output) → L6 (thalamic)
    /// This models the canonical microcircuit of the neocortex.
    ///
    /// # Arguments
    /// * `neurons_per_layer` - Neurons in each of 6 layers
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed
    pub fn cortical_column(neurons_per_layer: usize, dim: usize, _seed: u64) -> Self {
        let n_layers = 6;
        let n_nodes = neurons_per_layer * n_layers;

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut edges = Vec::new();

        // Within-layer connections (dense local connectivity)
        for layer in 0..n_layers {
            let start = layer * neurons_per_layer;
            for i in start..(start + neurons_per_layer) {
                for j in (i + 1)..(start + neurons_per_layer) {
                    edges.push((i, j));
                }
            }
        }

        // Between-layer connections (feedforward + feedback)
        // L4 (input layer, index 3) receives from L6 and projects to L2/3
        // L2/3 (index 1-2) projects to L5 and receives feedback from L5
        // L5 (index 4) projects to L6 and thalamus
        for i in 0..neurons_per_layer {
            // L4 → L2/3 (feedforward)
            let l4_neuron = 3 * neurons_per_layer + i;
            let l23_neuron = neurons_per_layer + (i % neurons_per_layer);
            edges.push((l4_neuron.min(l23_neuron), l4_neuron.max(l23_neuron)));

            // L2/3 → L5 (feedforward)
            let l5_neuron = 4 * neurons_per_layer + i;
            edges.push((l23_neuron.min(l5_neuron), l23_neuron.max(l5_neuron)));

            // L5 → L6 (feedforward)
            let l6_neuron = 5 * neurons_per_layer + i;
            edges.push((l5_neuron.min(l6_neuron), l5_neuron.max(l6_neuron)));

            // L6 → L4 (feedback loop via thalamus)
            edges.push((l6_neuron.min(l4_neuron), l6_neuron.max(l4_neuron)));

            // L5 → L2/3 (feedback)
            edges.push((l5_neuron.min(l23_neuron), l5_neuron.max(l23_neuron)));
        }

        edges.sort_unstable();
        edges.dedup();

        // Build adjacency and representations
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (i, j) in &edges {
            adjacency[*i].push(*j);
            adjacency[*j].push(*i);
        }

        let node_representations: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let connections: Vec<ContinuousHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                if connections.is_empty() {
                    node_identities[i].clone()
                } else {
                    ContinuousHV::bundle_owned(&connections)
                }
            })
            .collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::CorticalColumn,
            edges,
        }
    }

    /// Feedforward network - layered structure like neural networks
    ///
    /// Each layer only connects to the next layer (no recurrence).
    /// Models perception pipelines and deep learning architectures.
    ///
    /// # Arguments
    /// * `layers` - Vec of layer sizes, e.g., [3, 4, 2] for 3→4→2
    /// * `dim` - Hypervector dimension
    /// * `seed` - Random seed
    pub fn feedforward(layers: &[usize], dim: usize, _seed: u64) -> Self {
        let n_nodes: usize = layers.iter().sum();

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut edges = Vec::new();
        let mut offset = 0;

        // Connect each layer to the next (all-to-all between adjacent layers)
        for window in layers.windows(2) {
            let layer_size = window[0];
            let next_layer_size = window[1];

            for i in 0..layer_size {
                for j in 0..next_layer_size {
                    let from = offset + i;
                    let to = offset + layer_size + j;
                    edges.push((from.min(to), from.max(to)));
                }
            }
            offset += layer_size;
        }

        // Build adjacency and representations
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (i, j) in &edges {
            adjacency[*i].push(*j);
            adjacency[*j].push(*i);
        }

        let node_representations: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let connections: Vec<ContinuousHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                if connections.is_empty() {
                    node_identities[i].clone()
                } else {
                    ContinuousHV::bundle_owned(&connections)
                }
            })
            .collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Feedforward,
            edges,
        }
    }

    /// Recurrent network - includes feedback loops
    ///
    /// Like feedforward but with additional recurrent connections within layers.
    /// Models memory and temporal processing (like RNNs/LSTMs).
    pub fn recurrent(layers: &[usize], dim: usize, _seed: u64) -> Self {
        let n_nodes: usize = layers.iter().sum();

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut edges = Vec::new();
        let mut offset = 0;

        for (layer_idx, &layer_size) in layers.iter().enumerate() {
            // Within-layer recurrent connections (each node connects to next in layer)
            for i in 0..layer_size {
                let next_i = (i + 1) % layer_size;
                if layer_size > 1 {
                    let from = offset + i;
                    let to = offset + next_i;
                    edges.push((from.min(to), from.max(to)));
                }
            }

            // Feedforward to next layer (if not last layer)
            if layer_idx < layers.len() - 1 {
                let next_layer_size = layers[layer_idx + 1];
                for i in 0..layer_size {
                    for j in 0..next_layer_size {
                        let from = offset + i;
                        let to = offset + layer_size + j;
                        edges.push((from.min(to), from.max(to)));
                    }
                }
            }

            offset += layer_size;
        }

        edges.sort_unstable();
        edges.dedup();

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (i, j) in &edges {
            adjacency[*i].push(*j);
            adjacency[*j].push(*i);
        }

        let node_representations: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let connections: Vec<ContinuousHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                if connections.is_empty() {
                    node_identities[i].clone()
                } else {
                    ContinuousHV::bundle_owned(&connections)
                }
            })
            .collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Recurrent,
            edges,
        }
    }

    /// Bipartite graph - two groups with connections only between groups
    ///
    /// Models sensory processing (inputs → outputs with no lateral connections).
    /// Like retina → V1 or encoder → decoder.
    pub fn bipartite(
        n_left: usize,
        n_right: usize,
        connection_prob: f64,
        dim: usize,
        seed: u64,
    ) -> Self {
        let n_nodes = n_left + n_right;

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut edges = Vec::new();

        // Connect left to right with given probability
        for i in 0..n_left {
            for j in 0..n_right {
                let edge_seed = seed.wrapping_add((i * n_right + j) as u64);
                if (edge_seed % 100) as f64 / 100.0 < connection_prob {
                    edges.push((i, n_left + j));
                }
            }
        }

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (i, j) in &edges {
            adjacency[*i].push(*j);
            adjacency[*j].push(*i);
        }

        let node_representations: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let connections: Vec<ContinuousHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                if connections.is_empty() {
                    node_identities[i].clone()
                } else {
                    ContinuousHV::bundle_owned(&connections)
                }
            })
            .collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Bipartite,
            edges,
        }
    }

    /// Core-Periphery structure - dense core with sparse peripheral connections
    ///
    /// Models real-world networks where a small core is highly interconnected
    /// while peripheral nodes connect mainly to the core.
    pub fn core_periphery(core_size: usize, periphery_size: usize, dim: usize, _seed: u64) -> Self {
        let n_nodes = core_size + periphery_size;

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut edges = Vec::new();

        // Dense core connections (complete graph)
        for i in 0..core_size {
            for j in (i + 1)..core_size {
                edges.push((i, j));
            }
        }

        // Peripheral nodes connect to 1-2 core nodes each
        for p in 0..periphery_size {
            let peripheral_node = core_size + p;
            // Connect to core node (p mod core_size)
            let core_node = p % core_size;
            edges.push((core_node, peripheral_node));
            // Connect to one more core node for connectivity
            if core_size > 1 {
                let second_core = (p + 1) % core_size;
                edges.push((second_core, peripheral_node));
            }
        }

        edges.sort_unstable();
        edges.dedup();

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (i, j) in &edges {
            adjacency[*i].push(*j);
            adjacency[*j].push(*i);
        }

        let node_representations: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let connections: Vec<ContinuousHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                if connections.is_empty() {
                    node_identities[i].clone()
                } else {
                    ContinuousHV::bundle_owned(&connections)
                }
            })
            .collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::CorePeriphery,
            edges,
        }
    }

    /// Bow-Tie structure - IN → CORE → OUT
    ///
    /// Classic web/biological network structure: input nodes feed into a strongly
    /// connected core, which feeds into output nodes. Models metabolic networks,
    /// gene regulatory networks, and the web.
    pub fn bow_tie(n_in: usize, n_core: usize, n_out: usize, dim: usize, _seed: u64) -> Self {
        let n_nodes = n_in + n_core + n_out;

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut edges = Vec::new();

        // IN nodes connect to CORE
        for i in 0..n_in {
            for c in 0..n_core {
                edges.push((i, n_in + c));
            }
        }

        // CORE is strongly connected (complete graph)
        for i in 0..n_core {
            for j in (i + 1)..n_core {
                edges.push((n_in + i, n_in + j));
            }
        }

        // CORE connects to OUT
        for c in 0..n_core {
            for o in 0..n_out {
                edges.push((n_in + c, n_in + n_core + o));
            }
        }

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (i, j) in &edges {
            adjacency[*i].push(*j);
            adjacency[*j].push(*i);
        }

        let node_representations: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let connections: Vec<ContinuousHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                if connections.is_empty() {
                    node_identities[i].clone()
                } else {
                    ContinuousHV::bundle_owned(&connections)
                }
            })
            .collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::BowTie,
            edges,
        }
    }

    /// Attention network - Query-Key-Value structure
    ///
    /// Models transformer attention: queries attend to keys, which gate values.
    /// Three-layer structure with all-to-all attention weights.
    pub fn attention(
        n_queries: usize,
        n_keys: usize,
        n_values: usize,
        dim: usize,
        _seed: u64,
    ) -> Self {
        let n_nodes = n_queries + n_keys + n_values;

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut edges = Vec::new();

        // Queries attend to all Keys
        for q in 0..n_queries {
            for k in 0..n_keys {
                edges.push((q, n_queries + k));
            }
        }

        // Keys gate Values (1-to-1 for simplicity, assuming n_keys == n_values)
        for k in 0..n_keys.min(n_values) {
            edges.push((n_queries + k, n_queries + n_keys + k));
        }

        // Queries also directly connect to Values (residual-like)
        for q in 0..n_queries {
            for v in 0..n_values {
                edges.push((q.min(n_queries + n_keys + v), q.max(n_queries + n_keys + v)));
            }
        }

        edges.sort_unstable();
        edges.dedup();

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (i, j) in &edges {
            adjacency[*i].push(*j);
            adjacency[*j].push(*i);
        }

        let node_representations: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let connections: Vec<ContinuousHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                if connections.is_empty() {
                    node_identities[i].clone()
                } else {
                    ContinuousHV::bundle_owned(&connections)
                }
            })
            .collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Attention,
            edges,
        }
    }

    /// Residual network - skip connections like ResNets
    ///
    /// Layered structure where each layer has skip connections to layers 2 ahead.
    /// Models deep learning architectures with gradient highways.
    pub fn residual(layers: &[usize], dim: usize, _seed: u64) -> Self {
        let n_nodes: usize = layers.iter().sum();

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut edges = Vec::new();
        let mut offsets: Vec<usize> = vec![0];
        let mut sum = 0;
        for &size in layers {
            sum += size;
            offsets.push(sum);
        }

        // Regular feedforward connections
        for (layer_idx, window) in layers.windows(2).enumerate() {
            let layer_size = window[0];
            let next_layer_size = window[1];

            for i in 0..layer_size {
                for j in 0..next_layer_size {
                    let from = offsets[layer_idx] + i;
                    let to = offsets[layer_idx + 1] + j;
                    edges.push((from.min(to), from.max(to)));
                }
            }
        }

        // Skip connections (layer i to layer i+2)
        for layer_idx in 0..(layers.len().saturating_sub(2)) {
            let layer_size = layers[layer_idx];
            let skip_layer_size = layers[layer_idx + 2];

            for i in 0..layer_size.min(skip_layer_size) {
                let from = offsets[layer_idx] + i;
                let to = offsets[layer_idx + 2] + i;
                edges.push((from.min(to), from.max(to)));
            }
        }

        edges.sort_unstable();
        edges.dedup();

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (i, j) in &edges {
            adjacency[*i].push(*j);
            adjacency[*j].push(*i);
        }

        let node_representations: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let connections: Vec<ContinuousHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                if connections.is_empty() {
                    node_identities[i].clone()
                } else {
                    ContinuousHV::bundle_owned(&connections)
                }
            })
            .collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::Residual,
            edges,
        }
    }

    /// Petersen Graph - famous 10-node highly symmetric graph
    ///
    /// One of the most famous graphs in graph theory. Has remarkable properties:
    /// - 10 vertices, 15 edges
    /// - 3-regular (each vertex has exactly 3 neighbors)
    /// - Highly symmetric (120 automorphisms)
    /// - Non-planar, vertex-transitive
    pub fn petersen_graph(dim: usize, _seed: u64) -> Self {
        let n_nodes = 10;

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        // Petersen graph edges (fixed structure)
        // Outer pentagon: 0-1-2-3-4-0
        // Inner pentagram: 5-7-9-6-8-5
        // Spokes: 0-5, 1-6, 2-7, 3-8, 4-9
        let edges = vec![
            // Outer pentagon
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 0),
            // Inner pentagram
            (5, 7),
            (7, 9),
            (9, 6),
            (6, 8),
            (8, 5),
            // Spokes
            (0, 5),
            (1, 6),
            (2, 7),
            (3, 8),
            (4, 9),
        ];

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (i, j) in &edges {
            adjacency[*i].push(*j);
            adjacency[*j].push(*i);
        }

        let node_representations: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let connections: Vec<ContinuousHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                ContinuousHV::bundle_owned(&connections)
            })
            .collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::PetersenGraph,
            edges,
        }
    }

    /// Complete Bipartite Graph K_{n,m} - all-to-all between two groups
    ///
    /// Every node in group A connects to every node in group B.
    /// Models perfect encoder-decoder relationships.
    pub fn complete_bipartite(n: usize, m: usize, dim: usize, _seed: u64) -> Self {
        let n_nodes = n + m;

        let node_identities: Vec<ContinuousHV> =
            (0..n_nodes).map(|i| ContinuousHV::basis(i, dim)).collect();

        let mut edges = Vec::new();

        // All nodes in first group connect to all nodes in second group
        for i in 0..n {
            for j in 0..m {
                edges.push((i, n + j));
            }
        }

        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n_nodes];
        for (i, j) in &edges {
            adjacency[*i].push(*j);
            adjacency[*j].push(*i);
        }

        let node_representations: Vec<ContinuousHV> = (0..n_nodes)
            .map(|i| {
                let connections: Vec<ContinuousHV> = adjacency[i]
                    .iter()
                    .map(|&neighbor| node_identities[i].bind(&node_identities[neighbor]))
                    .collect();
                if connections.is_empty() {
                    node_identities[i].clone()
                } else {
                    ContinuousHV::bundle_owned(&connections)
                }
            })
            .collect();

        Self {
            n_nodes,
            dim,
            node_representations,
            node_identities,
            topology_type: TopologyType::CompleteBipartite,
            edges,
        }
    }
}
