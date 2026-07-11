// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! An adjacency-list graph over integer nodes `0..n`, with traversal,
//! connectivity, topological sort, and cycle detection.

use std::collections::VecDeque;

/// A graph on `n` nodes (ids `0..n`). `directed` selects whether `add_edge`
/// also inserts the reverse edge.
#[derive(Debug, Clone)]
pub struct Graph {
    n: usize,
    directed: bool,
    adj: Vec<Vec<usize>>,
}

impl Graph {
    /// A graph with `n` isolated nodes.
    pub fn new(n: usize, directed: bool) -> Graph {
        Graph {
            n,
            directed,
            adj: vec![Vec::new(); n],
        }
    }

    /// Add an edge `u → v` (and `v → u` when undirected). Out-of-range endpoints
    /// are ignored.
    pub fn add_edge(&mut self, u: usize, v: usize) -> &mut Graph {
        if u < self.n && v < self.n {
            self.adj[u].push(v);
            if !self.directed {
                self.adj[v].push(u);
            }
        }
        self
    }

    /// Number of nodes.
    pub fn node_count(&self) -> usize {
        self.n
    }

    /// Whether the graph is directed.
    pub fn is_directed(&self) -> bool {
        self.directed
    }

    /// Out-neighbours of `u`.
    pub fn neighbors(&self, u: usize) -> &[usize] {
        &self.adj[u]
    }

    /// Breadth-first traversal order from `start`.
    pub fn bfs(&self, start: usize) -> Vec<usize> {
        let mut order = Vec::new();
        let mut seen = vec![false; self.n];
        let mut q = VecDeque::from([start]);
        seen[start] = true;
        while let Some(u) = q.pop_front() {
            order.push(u);
            for &v in &self.adj[u] {
                if !seen[v] {
                    seen[v] = true;
                    q.push_back(v);
                }
            }
        }
        order
    }

    /// Shortest-path distance (in edges) from `start` to every node; `None` for
    /// unreachable nodes.
    pub fn bfs_distances(&self, start: usize) -> Vec<Option<usize>> {
        let mut dist = vec![None; self.n];
        let mut q = VecDeque::from([start]);
        dist[start] = Some(0);
        while let Some(u) = q.pop_front() {
            let du = dist[u].unwrap();
            for &v in &self.adj[u] {
                if dist[v].is_none() {
                    dist[v] = Some(du + 1);
                    q.push_back(v);
                }
            }
        }
        dist
    }

    /// Depth-first traversal order from `start`.
    pub fn dfs(&self, start: usize) -> Vec<usize> {
        let mut order = Vec::new();
        let mut seen = vec![false; self.n];
        let mut stack = vec![start];
        while let Some(u) = stack.pop() {
            if seen[u] {
                continue;
            }
            seen[u] = true;
            order.push(u);
            // Push neighbours in reverse so the smallest is visited first.
            for &v in self.adj[u].iter().rev() {
                if !seen[v] {
                    stack.push(v);
                }
            }
        }
        order
    }

    /// Connected components (treating edges as undirected). For a directed
    /// graph this yields components of the underlying undirected graph only if
    /// it was built undirected; otherwise it reflects forward reachability.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let mut comp = vec![usize::MAX; self.n];
        let mut components = Vec::new();
        for s in 0..self.n {
            if comp[s] != usize::MAX {
                continue;
            }
            let id = components.len();
            let mut members = Vec::new();
            let mut q = VecDeque::from([s]);
            comp[s] = id;
            while let Some(u) = q.pop_front() {
                members.push(u);
                for &v in &self.adj[u] {
                    if comp[v] == usize::MAX {
                        comp[v] = id;
                        q.push_back(v);
                    }
                }
            }
            members.sort_unstable();
            components.push(members);
        }
        components
    }

    /// In-degree of every node (counts incoming edges; for undirected graphs
    /// this equals the degree).
    fn in_degrees(&self) -> Vec<usize> {
        let mut indeg = vec![0usize; self.n];
        for u in 0..self.n {
            for &v in &self.adj[u] {
                indeg[v] += 1;
            }
        }
        indeg
    }

    /// A topological ordering (Kahn's algorithm) for a directed acyclic graph.
    /// `None` if the graph has a directed cycle. Ties broken by smallest id.
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        let mut indeg = self.in_degrees();
        // Min-ordered frontier via a sorted vec used as a stack of ready nodes.
        let mut ready: Vec<usize> = (0..self.n).filter(|&u| indeg[u] == 0).collect();
        ready.sort_unstable_by(|a, b| b.cmp(a)); // pop smallest
        let mut order = Vec::new();
        while let Some(u) = ready.pop() {
            order.push(u);
            for &v in &self.adj[u] {
                indeg[v] -= 1;
                if indeg[v] == 0 {
                    // Insert keeping `ready` descending so pop yields the min.
                    let pos = ready.partition_point(|&x| x > v);
                    ready.insert(pos, v);
                }
            }
        }
        (order.len() == self.n).then_some(order)
    }

    /// Whether a **directed** cycle exists (equivalently, no topological order).
    pub fn has_directed_cycle(&self) -> bool {
        self.topological_sort().is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn undirected_two_triangles() -> Graph {
        // {0,1,2} triangle and {3,4,5} triangle — two components.
        let mut g = Graph::new(6, false);
        g.add_edge(0, 1).add_edge(1, 2).add_edge(2, 0);
        g.add_edge(3, 4).add_edge(4, 5).add_edge(5, 3);
        g
    }

    #[test]
    fn bfs_and_distances() {
        let mut g = Graph::new(4, true);
        g.add_edge(0, 1).add_edge(0, 2).add_edge(2, 3);
        assert_eq!(g.bfs(0), vec![0, 1, 2, 3]);
        let d = g.bfs_distances(0);
        assert_eq!(d[3], Some(2));
        assert_eq!(d[1], Some(1));
    }

    #[test]
    fn components() {
        let comps = undirected_two_triangles().connected_components();
        assert_eq!(comps.len(), 2);
        assert_eq!(comps[0], vec![0, 1, 2]);
        assert_eq!(comps[1], vec![3, 4, 5]);
    }

    #[test]
    fn topological_sort_of_dag() {
        // 0→1→3, 0→2→3 : a valid order must have 0 first and 3 last.
        let mut g = Graph::new(4, true);
        g.add_edge(0, 1)
            .add_edge(0, 2)
            .add_edge(1, 3)
            .add_edge(2, 3);
        let order = g.topological_sort().unwrap();
        assert_eq!(order.first(), Some(&0));
        assert_eq!(order.last(), Some(&3));
        assert!(!g.has_directed_cycle());
    }

    #[test]
    fn cycle_has_no_topological_order() {
        let mut g = Graph::new(3, true);
        g.add_edge(0, 1).add_edge(1, 2).add_edge(2, 0);
        assert!(g.topological_sort().is_none());
        assert!(g.has_directed_cycle());
    }
}
