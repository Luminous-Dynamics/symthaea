// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! # symthaea-graph-theory
//!
//! Foundational graph algorithms. The workspace had only Dijkstra (in
//! `symthaea-operations-research`), yet graphs are everywhere here — causal
//! reasoning, dependency registries, social coherence, chronicle causal chains,
//! the DHT. This crate supplies the missing structural core.
//!
//! Pure `std`, zero dependencies, no `symthaea-core` link. Every algorithm is
//! checked against a worked example.
//!
//! ## Contents
//! - [`graph::Graph`] — adjacency-list graph (directed or undirected) with BFS,
//!   DFS, BFS distances, connected components, topological sort (Kahn), and
//!   directed-cycle detection
//! - [`algorithms`] — degree centrality, PageRank (power iteration), minimum
//!   spanning tree (Kruskal + union-find), and maximum flow (Edmonds-Karp)
//!
//! ## Example
//!
//! ```
//! use symthaea_graph_theory::Graph;
//! // A directed cycle has no topological order.
//! let mut g = Graph::new(3, true);
//! g.add_edge(0, 1).add_edge(1, 2).add_edge(2, 0);
//! assert!(g.topological_sort().is_none());
//! assert!(g.has_directed_cycle());
//! ```

pub mod algorithms;
pub mod graph;

pub use algorithms::{degree_centrality, kruskal, max_flow, pagerank};
pub use graph::Graph;
