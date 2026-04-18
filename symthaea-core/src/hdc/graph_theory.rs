// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Graph Theory & Combinatorics Engine
//!
//! Graph data structure with HDC-encoded nodes/edges, standard graph algorithms,
//! and combinatorial counting functions.
//!
//! ## Algorithms
//! - BFS, DFS, Dijkstra shortest path, Kruskal MST
//! - Topological sort, cycle detection
//! - Graph coloring (greedy)
//! - Combinatorial counting: permutations, combinations, Catalan, Stirling

use crate::hdc::binary_hv::BinaryHV;
use crate::hdc::primitive_system::seed_from_name;
use std::collections::{BinaryHeap, HashSet, VecDeque};

// ─── Graph Type ──────────────────────────────────────────────────────────────

/// A graph represented by adjacency lists with optional edge weights.
#[derive(Debug, Clone)]
pub struct Graph {
    /// Number of vertices
    pub n: usize,
    /// Adjacency list: vertex → [(neighbor, weight)]
    pub adj: Vec<Vec<(usize, f64)>>,
    /// Whether the graph is directed
    pub directed: bool,
}

impl Graph {
    /// Create an empty graph
    pub fn new(n: usize, directed: bool) -> Self {
        Self {
            n,
            adj: vec![Vec::new(); n],
            directed,
        }
    }

    /// Add an edge (with weight 1.0)
    pub fn add_edge(&mut self, u: usize, v: usize) {
        self.add_weighted_edge(u, v, 1.0);
    }

    /// Add a weighted edge
    pub fn add_weighted_edge(&mut self, u: usize, v: usize, weight: f64) {
        self.adj[u].push((v, weight));
        if !self.directed {
            self.adj[v].push((u, weight));
        }
    }

    /// Number of edges
    pub fn num_edges(&self) -> usize {
        let total: usize = self.adj.iter().map(|a| a.len()).sum();
        if self.directed {
            total
        } else {
            total / 2
        }
    }

    /// HDC encoding of the graph
    pub fn encode(&self) -> BinaryHV {
        let graph_prim = BinaryHV::random(seed_from_name("GRAPH"));
        let n_hv = BinaryHV::random(seed_from_name(&format!("GRAPH_N_{}", self.n)));
        let e_hv = BinaryHV::random(seed_from_name(&format!("GRAPH_E_{}", self.num_edges())));
        graph_prim.bind(&n_hv).bind(&e_hv)
    }

    // ─── BFS ─────────────────────────────────────────────────────────────

    /// Breadth-first search from a source vertex.
    /// Returns distances from source (usize::MAX if unreachable).
    pub fn bfs(&self, source: usize) -> Vec<usize> {
        let mut dist = vec![usize::MAX; self.n];
        let mut queue = VecDeque::new();
        dist[source] = 0;
        queue.push_back(source);

        while let Some(u) = queue.pop_front() {
            for &(v, _) in &self.adj[u] {
                if dist[v] == usize::MAX {
                    dist[v] = dist[u] + 1;
                    queue.push_back(v);
                }
            }
        }
        dist
    }

    // ─── DFS ─────────────────────────────────────────────────────────────

    /// Depth-first search from a source. Returns vertices in DFS visit order.
    pub fn dfs(&self, source: usize) -> Vec<usize> {
        let mut visited = vec![false; self.n];
        let mut order = Vec::new();
        self.dfs_recursive(source, &mut visited, &mut order);
        order
    }

    fn dfs_recursive(&self, u: usize, visited: &mut Vec<bool>, order: &mut Vec<usize>) {
        visited[u] = true;
        order.push(u);
        for &(v, _) in &self.adj[u] {
            if !visited[v] {
                self.dfs_recursive(v, visited, order);
            }
        }
    }

    // ─── Dijkstra ────────────────────────────────────────────────────────

    /// Dijkstra's shortest path from source.
    /// Returns (distances, predecessors). Distance is f64::INFINITY if unreachable.
    pub fn dijkstra(&self, source: usize) -> (Vec<f64>, Vec<Option<usize>>) {
        let mut dist = vec![f64::INFINITY; self.n];
        let mut pred = vec![None; self.n];
        let mut heap = BinaryHeap::new();

        dist[source] = 0.0;
        heap.push(std::cmp::Reverse(DijkstraEntry {
            dist: 0.0,
            vertex: source,
        }));

        while let Some(std::cmp::Reverse(entry)) = heap.pop() {
            if entry.dist > dist[entry.vertex] {
                continue;
            }

            for &(v, w) in &self.adj[entry.vertex] {
                let new_dist = dist[entry.vertex] + w;
                if new_dist < dist[v] {
                    dist[v] = new_dist;
                    pred[v] = Some(entry.vertex);
                    heap.push(std::cmp::Reverse(DijkstraEntry {
                        dist: new_dist,
                        vertex: v,
                    }));
                }
            }
        }

        (dist, pred)
    }

    /// Reconstruct the shortest path from source to target
    pub fn shortest_path(&self, source: usize, target: usize) -> Option<Vec<usize>> {
        let (dist, pred) = self.dijkstra(source);
        if dist[target].is_infinite() {
            return None;
        }

        let mut path = Vec::new();
        let mut current = target;
        while let Some(prev) = pred[current] {
            path.push(current);
            current = prev;
        }
        path.push(source);
        path.reverse();
        Some(path)
    }

    // ─── Kruskal MST ────────────────────────────────────────────────────

    /// Kruskal's minimum spanning tree.
    /// Returns (MST edges, total weight). Only for undirected graphs.
    pub fn kruskal_mst(&self) -> (Vec<(usize, usize, f64)>, f64) {
        assert!(!self.directed, "MST requires undirected graph");

        // Collect all edges
        let mut edges: Vec<(f64, usize, usize)> = Vec::new();
        for u in 0..self.n {
            for &(v, w) in &self.adj[u] {
                if u < v {
                    edges.push((w, u, v));
                }
            }
        }
        edges.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut uf = UnionFind::new(self.n);
        let mut mst = Vec::new();
        let mut total_weight = 0.0;

        for (w, u, v) in edges {
            if uf.find(u) != uf.find(v) {
                uf.union(u, v);
                mst.push((u, v, w));
                total_weight += w;
                if mst.len() == self.n - 1 {
                    break;
                }
            }
        }

        (mst, total_weight)
    }

    // ─── Topological Sort ────────────────────────────────────────────────

    /// Topological sort (Kahn's algorithm). Returns None if graph has a cycle.
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        assert!(self.directed, "Topological sort requires directed graph");

        let mut in_degree = vec![0usize; self.n];
        for u in 0..self.n {
            for &(v, _) in &self.adj[u] {
                in_degree[v] += 1;
            }
        }

        let mut queue: VecDeque<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|(_, &d)| d == 0)
            .map(|(i, _)| i)
            .collect();

        let mut order = Vec::new();

        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &(v, _) in &self.adj[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 {
                    queue.push_back(v);
                }
            }
        }

        if order.len() == self.n {
            Some(order)
        } else {
            None // Cycle detected
        }
    }

    // ─── Cycle Detection ─────────────────────────────────────────────────

    /// Detect if the graph contains a cycle
    pub fn has_cycle(&self) -> bool {
        if self.directed {
            self.topological_sort().is_none()
        } else {
            let mut visited = vec![false; self.n];
            for start in 0..self.n {
                if !visited[start] && self.dfs_cycle_undirected(start, usize::MAX, &mut visited) {
                    return true;
                }
            }
            false
        }
    }

    fn dfs_cycle_undirected(&self, u: usize, parent: usize, visited: &mut Vec<bool>) -> bool {
        visited[u] = true;
        for &(v, _) in &self.adj[u] {
            if !visited[v] {
                if self.dfs_cycle_undirected(v, u, visited) {
                    return true;
                }
            } else if v != parent {
                return true;
            }
        }
        false
    }

    // ─── Graph Coloring ──────────────────────────────────────────────────

    /// Greedy graph coloring. Returns (coloring, number of colors used).
    pub fn greedy_coloring(&self) -> (Vec<usize>, usize) {
        let mut color = vec![usize::MAX; self.n];
        let mut num_colors = 0;

        for u in 0..self.n {
            let neighbor_colors: HashSet<usize> = self.adj[u]
                .iter()
                .filter_map(|&(v, _)| {
                    if color[v] != usize::MAX {
                        Some(color[v])
                    } else {
                        None
                    }
                })
                .collect();

            let mut c = 0;
            while neighbor_colors.contains(&c) {
                c += 1;
            }
            color[u] = c;
            num_colors = num_colors.max(c + 1);
        }

        (color, num_colors)
    }

    /// Check if the graph is connected
    pub fn is_connected(&self) -> bool {
        if self.n == 0 {
            return true;
        }
        let dist = self.bfs(0);
        dist.iter().all(|&d| d != usize::MAX)
    }

    /// Check if the graph is bipartite
    pub fn is_bipartite(&self) -> bool {
        let mut color: Vec<i32> = vec![-1; self.n];
        let mut queue = VecDeque::new();

        for start in 0..self.n {
            if color[start] != -1 {
                continue;
            }
            color[start] = 0;
            queue.push_back(start);

            while let Some(u) = queue.pop_front() {
                for &(v, _) in &self.adj[u] {
                    if color[v] == -1 {
                        color[v] = 1 - color[u];
                        queue.push_back(v);
                    } else if color[v] == color[u] {
                        return false;
                    }
                }
            }
        }
        true
    }

    // ── Spectral Graph Theory ───────────────────────────────────────────

    /// Compute the graph Laplacian matrix L = D - A.
    ///
    /// D = degree matrix (diagonal), A = adjacency matrix.
    /// Returns L as a dense Vec<Vec<f64>> (use sparse for large graphs).
    pub fn laplacian_matrix(&self) -> Vec<Vec<f64>> {
        let mut l = vec![vec![0.0; self.n]; self.n];
        for u in 0..self.n {
            let mut degree = 0.0;
            for &(v, w) in &self.adj[u] {
                l[u][v] -= w; // off-diagonal: -A_{uv}
                degree += w;
            }
            l[u][u] = degree; // diagonal: D_{uu}
        }
        l
    }

    /// Compute eigenvalues of the graph Laplacian (sorted ascending).
    ///
    /// Uses the linear algebra module's eigendecomposition.
    /// Returns all n eigenvalues: λ₁ = 0 ≤ λ₂ ≤ ... ≤ λₙ.
    ///
    /// Key spectral quantities:
    /// - λ₁ = 0 always (connected component count = multiplicity of 0)
    /// - λ₂ = algebraic connectivity (Fiedler value) — measures connectivity
    /// - λₙ = spectral radius — upper bound on diameter
    pub fn laplacian_eigenvalues(&self) -> Vec<f64> {
        let l = self.laplacian_matrix();
        let flat: Vec<f64> = l.iter().flat_map(|row| row.iter().copied()).collect();
        let mat = super::linear_algebra::HdcMatrix::new(flat, self.n, self.n);
        let (mut eigenvalues, _, _) = mat.eigen_symmetric();
        eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        eigenvalues
    }

    /// Algebraic connectivity (Fiedler value): second-smallest eigenvalue λ₂.
    ///
    /// - λ₂ = 0 means the graph is disconnected
    /// - λ₂ > 0 means the graph is connected
    /// - Larger λ₂ = more robust connectivity
    /// - Related to Cheeger constant: h(G) ≥ λ₂ / 2 (Cheeger inequality)
    pub fn algebraic_connectivity(&self) -> f64 {
        let evals = self.laplacian_eigenvalues();
        if evals.len() >= 2 { evals[1] } else { 0.0 }
    }

    /// Spectral gap: λ₂ (difference between first two distinct eigenvalues).
    /// Large spectral gap → rapid mixing, strong expansion.
    pub fn spectral_gap(&self) -> f64 {
        self.algebraic_connectivity()
    }

    /// Cheeger constant lower bound: h(G) ≥ λ₂ / 2.
    /// Measures the bottleneck of the graph (min cut / max balance).
    pub fn cheeger_lower_bound(&self) -> f64 {
        self.algebraic_connectivity() / 2.0
    }

    /// Number of connected components (= multiplicity of eigenvalue 0).
    pub fn num_components_spectral(&self) -> usize {
        let evals = self.laplacian_eigenvalues();
        evals.iter().filter(|&&e| e.abs() < 1e-6).count()
    }
}

// ─── Dijkstra Helper ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DijkstraEntry {
    dist: f64,
    vertex: usize,
}

impl PartialEq for DijkstraEntry {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for DijkstraEntry {}
impl PartialOrd for DijkstraEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DijkstraEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ─── Union-Find ──────────────────────────────────────────────────────────────

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
    }
}

// ─── Combinatorics ───────────────────────────────────────────────────────────

/// Combinatorial counting functions
pub struct Combinatorics;

impl Combinatorics {
    /// Number of permutations P(n, k) = n! / (n-k)!
    pub fn permutations(n: u64, k: u64) -> u64 {
        if k > n {
            return 0;
        }
        (n - k + 1..=n).product()
    }

    /// Binomial coefficient C(n, k) = n! / (k! (n-k)!)
    pub fn combinations(n: u64, k: u64) -> u64 {
        if k > n {
            return 0;
        }
        let k = k.min(n - k);
        let mut result: u64 = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// Factorial n!
    pub fn factorial(n: u64) -> u64 {
        (1..=n).product::<u64>().max(1)
    }

    /// Catalan number C_n = C(2n, n) / (n+1)
    pub fn catalan(n: u64) -> u64 {
        Self::combinations(2 * n, n) / (n + 1)
    }

    /// Stirling numbers of the second kind S(n, k)
    /// (number of ways to partition n elements into k non-empty subsets)
    pub fn stirling_second(n: u64, k: u64) -> u64 {
        if n == 0 && k == 0 {
            return 1;
        }
        if n == 0 || k == 0 || k > n {
            return 0;
        }

        // Use recurrence: S(n,k) = k * S(n-1,k) + S(n-1,k-1)
        let mut prev = vec![0u64; (k + 1) as usize];
        prev[0] = 1; // S(0,0) = 1

        for i in 1..=n {
            let mut curr = vec![0u64; (k + 1) as usize];
            for j in 1..=k.min(i) {
                curr[j as usize] = j * prev[j as usize] + prev[(j - 1) as usize];
            }
            prev = curr;
        }

        prev[k as usize]
    }

    /// Bell number B_n (total number of partitions of n elements)
    pub fn bell(n: u64) -> u64 {
        (0..=n).map(|k| Self::stirling_second(n, k)).sum()
    }

    /// Derangements D(n) = n! * Σ (-1)^k / k!
    pub fn derangements(n: u64) -> u64 {
        if n == 0 {
            return 1;
        }
        if n == 1 {
            return 0;
        }
        let mut d = vec![0u64; (n + 1) as usize];
        d[0] = 1;
        d[1] = 0;
        for i in 2..=n as usize {
            d[i] = (i as u64 - 1) * (d[i - 1] + d[i - 2]);
        }
        d[n as usize]
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── BFS ──────────────────────────────────────────────────────────────

    #[test]
    fn test_bfs() {
        let mut g = Graph::new(5, false);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        g.add_edge(3, 4);
        let dist = g.bfs(0);
        assert_eq!(dist, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_bfs_disconnected() {
        let mut g = Graph::new(4, false);
        g.add_edge(0, 1);
        // 2 and 3 are isolated
        let dist = g.bfs(0);
        assert_eq!(dist[0], 0);
        assert_eq!(dist[1], 1);
        assert_eq!(dist[2], usize::MAX);
    }

    // ── DFS ──────────────────────────────────────────────────────────────

    #[test]
    fn test_dfs() {
        let mut g = Graph::new(4, false);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 3);
        let order = g.dfs(0);
        assert_eq!(order[0], 0);
        assert_eq!(order.len(), 4);
    }

    // ── Dijkstra ─────────────────────────────────────────────────────────

    #[test]
    fn test_dijkstra() {
        let mut g = Graph::new(4, false);
        g.add_weighted_edge(0, 1, 1.0);
        g.add_weighted_edge(1, 2, 2.0);
        g.add_weighted_edge(0, 2, 10.0);
        g.add_weighted_edge(2, 3, 1.0);
        let (dist, _) = g.dijkstra(0);
        assert!((dist[0] - 0.0).abs() < 1e-10);
        assert!((dist[1] - 1.0).abs() < 1e-10);
        assert!((dist[2] - 3.0).abs() < 1e-10);
        assert!((dist[3] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_shortest_path() {
        let mut g = Graph::new(4, false);
        g.add_weighted_edge(0, 1, 1.0);
        g.add_weighted_edge(1, 2, 1.0);
        g.add_weighted_edge(2, 3, 1.0);
        g.add_weighted_edge(0, 3, 10.0);
        let path = g.shortest_path(0, 3).unwrap();
        assert_eq!(path, vec![0, 1, 2, 3]);
    }

    // ── Kruskal MST ──────────────────────────────────────────────────────

    #[test]
    fn test_kruskal_mst() {
        let mut g = Graph::new(4, false);
        g.add_weighted_edge(0, 1, 1.0);
        g.add_weighted_edge(1, 2, 2.0);
        g.add_weighted_edge(2, 3, 3.0);
        g.add_weighted_edge(0, 3, 4.0);
        g.add_weighted_edge(0, 2, 5.0);
        let (mst, weight) = g.kruskal_mst();
        assert_eq!(mst.len(), 3);
        assert!((weight - 6.0).abs() < 1e-10); // 1 + 2 + 3
    }

    // ── Topological Sort ─────────────────────────────────────────────────

    #[test]
    fn test_topological_sort() {
        let mut g = Graph::new(4, true);
        g.add_edge(0, 1);
        g.add_edge(0, 2);
        g.add_edge(1, 3);
        g.add_edge(2, 3);
        let order = g.topological_sort().unwrap();
        assert_eq!(order[0], 0);
        assert_eq!(*order.last().unwrap(), 3);
    }

    #[test]
    fn test_topological_sort_cycle() {
        let mut g = Graph::new(3, true);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 0);
        assert!(g.topological_sort().is_none());
    }

    // ── Cycle Detection ──────────────────────────────────────────────────

    #[test]
    fn test_cycle_detection() {
        let mut g = Graph::new(3, false);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        assert!(!g.has_cycle());

        g.add_edge(2, 0);
        assert!(g.has_cycle());
    }

    // ── Graph Coloring ───────────────────────────────────────────────────

    #[test]
    fn test_greedy_coloring_bipartite() {
        let mut g = Graph::new(4, false);
        g.add_edge(0, 1);
        g.add_edge(2, 3);
        g.add_edge(0, 3);
        let (coloring, num_colors) = g.greedy_coloring();
        assert!(num_colors <= 3);
        // Verify no adjacent vertices share color
        for u in 0..4 {
            for &(v, _) in &g.adj[u] {
                assert_ne!(coloring[u], coloring[v]);
            }
        }
    }

    // ── Properties ───────────────────────────────────────────────────────

    #[test]
    fn test_connected() {
        let mut g = Graph::new(3, false);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        assert!(g.is_connected());

        let g2 = Graph::new(3, false);
        assert!(!g2.is_connected());
    }

    #[test]
    fn test_bipartite() {
        let mut g = Graph::new(4, false);
        g.add_edge(0, 1);
        g.add_edge(2, 3);
        g.add_edge(0, 3);
        assert!(g.is_bipartite());

        // Triangle is not bipartite
        let mut g2 = Graph::new(3, false);
        g2.add_edge(0, 1);
        g2.add_edge(1, 2);
        g2.add_edge(2, 0);
        assert!(!g2.is_bipartite());
    }

    // ── Combinatorics ────────────────────────────────────────────────────

    #[test]
    fn test_factorial() {
        assert_eq!(Combinatorics::factorial(0), 1);
        assert_eq!(Combinatorics::factorial(5), 120);
        assert_eq!(Combinatorics::factorial(10), 3628800);
    }

    #[test]
    fn test_permutations() {
        assert_eq!(Combinatorics::permutations(5, 3), 60);
        assert_eq!(Combinatorics::permutations(5, 5), 120);
        assert_eq!(Combinatorics::permutations(5, 0), 1);
    }

    #[test]
    fn test_combinations() {
        assert_eq!(Combinatorics::combinations(5, 2), 10);
        assert_eq!(Combinatorics::combinations(10, 3), 120);
        assert_eq!(Combinatorics::combinations(5, 0), 1);
        assert_eq!(Combinatorics::combinations(5, 5), 1);
    }

    #[test]
    fn test_catalan() {
        assert_eq!(Combinatorics::catalan(0), 1);
        assert_eq!(Combinatorics::catalan(1), 1);
        assert_eq!(Combinatorics::catalan(2), 2);
        assert_eq!(Combinatorics::catalan(3), 5);
        assert_eq!(Combinatorics::catalan(4), 14);
        assert_eq!(Combinatorics::catalan(5), 42);
    }

    #[test]
    fn test_stirling_second() {
        assert_eq!(Combinatorics::stirling_second(0, 0), 1);
        assert_eq!(Combinatorics::stirling_second(3, 2), 3);
        assert_eq!(Combinatorics::stirling_second(4, 2), 7);
        assert_eq!(Combinatorics::stirling_second(5, 3), 25);
    }

    #[test]
    fn test_bell() {
        assert_eq!(Combinatorics::bell(0), 1);
        assert_eq!(Combinatorics::bell(1), 1);
        assert_eq!(Combinatorics::bell(2), 2);
        assert_eq!(Combinatorics::bell(3), 5);
        assert_eq!(Combinatorics::bell(4), 15);
        assert_eq!(Combinatorics::bell(5), 52);
    }

    #[test]
    fn test_derangements() {
        assert_eq!(Combinatorics::derangements(0), 1);
        assert_eq!(Combinatorics::derangements(1), 0);
        assert_eq!(Combinatorics::derangements(2), 1);
        assert_eq!(Combinatorics::derangements(3), 2);
        assert_eq!(Combinatorics::derangements(4), 9);
        assert_eq!(Combinatorics::derangements(5), 44);
    }

    // ── Spectral graph theory tests ─────────────────────────────────────

    #[test]
    fn test_laplacian_complete_graph() {
        // K₃ (complete graph on 3 vertices): L = [[2,-1,-1],[-1,2,-1],[-1,-1,2]]
        let mut g = Graph::new(3, false);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(0, 2);
        let l = g.laplacian_matrix();
        assert!((l[0][0] - 2.0).abs() < 1e-10);
        assert!((l[0][1] - (-1.0)).abs() < 1e-10);
        assert!((l[1][2] - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_laplacian_eigenvalues_path() {
        // Path graph P₄: 0-1-2-3
        // Eigenvalues: 0, 2-√2, 2, 2+√2
        let mut g = Graph::new(4, false);
        g.add_edge(0, 1);
        g.add_edge(1, 2);
        g.add_edge(2, 3);
        let evals = g.laplacian_eigenvalues();
        assert_eq!(evals.len(), 4);
        assert!(evals[0].abs() < 1e-6, "smallest eigenvalue should be 0, got {}", evals[0]);
        assert!(evals[1] > 0.2, "λ₂ should be > 0 (connected): {}", evals[1]);
    }

    #[test]
    fn test_algebraic_connectivity_connected() {
        // Complete graph K₅: λ₂ = 5 (all non-zero eigenvalues = n)
        let mut g = Graph::new(5, false);
        for i in 0..5 {
            for j in (i + 1)..5 {
                g.add_edge(i, j);
            }
        }
        let ac = g.algebraic_connectivity();
        assert!((ac - 5.0).abs() < 0.1, "K₅ algebraic connectivity should be 5, got {}", ac);
    }

    #[test]
    fn test_algebraic_connectivity_disconnected() {
        // Two isolated components: {0,1}, {2,3}
        let mut g = Graph::new(4, false);
        g.add_edge(0, 1);
        g.add_edge(2, 3);
        let ac = g.algebraic_connectivity();
        assert!(ac < 1e-6, "disconnected graph should have λ₂ ≈ 0, got {}", ac);
    }

    #[test]
    fn test_num_components_spectral() {
        // Three components: {0,1}, {2}, {3,4}
        let mut g = Graph::new(5, false);
        g.add_edge(0, 1);
        g.add_edge(3, 4);
        let nc = g.num_components_spectral();
        assert_eq!(nc, 3, "should have 3 components, got {}", nc);
    }

    #[test]
    fn test_cheeger_bound() {
        // Complete graph has high Cheeger bound
        let mut g = Graph::new(4, false);
        for i in 0..4 { for j in (i+1)..4 { g.add_edge(i, j); } }
        let ch = g.cheeger_lower_bound();
        assert!(ch > 1.0, "K₄ should have high Cheeger bound: {}", ch);
    }
}
