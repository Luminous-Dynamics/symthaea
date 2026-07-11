// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Centrality (degree, PageRank) plus weighted-edge algorithms — minimum
//! spanning tree (Kruskal) and maximum flow (Edmonds-Karp).

use crate::graph::Graph;
use std::collections::VecDeque;

/// Degree centrality: each node's degree normalised by `n − 1`.
pub fn degree_centrality(g: &Graph) -> Vec<f64> {
    let n = g.node_count();
    if n <= 1 {
        return vec![0.0; n];
    }
    (0..n)
        .map(|u| g.neighbors(u).len() as f64 / (n - 1) as f64)
        .collect()
}

/// PageRank via power iteration. `damping` is typically 0.85. Dangling nodes
/// (no out-edges) redistribute their rank uniformly, so the vector always sums
/// to 1.
pub fn pagerank(g: &Graph, damping: f64, iterations: usize) -> Vec<f64> {
    let n = g.node_count();
    if n == 0 {
        return Vec::new();
    }
    let nf = n as f64;
    let mut rank = vec![1.0 / nf; n];
    let out_deg: Vec<usize> = (0..n).map(|u| g.neighbors(u).len()).collect();
    for _ in 0..iterations {
        let mut next = vec![(1.0 - damping) / nf; n];
        let mut dangling = 0.0;
        for u in 0..n {
            if out_deg[u] == 0 {
                dangling += rank[u];
            } else {
                let share = damping * rank[u] / out_deg[u] as f64;
                for &v in g.neighbors(u) {
                    next[v] += share;
                }
            }
        }
        // Distribute dangling mass uniformly.
        let dangling_share = damping * dangling / nf;
        for r in next.iter_mut() {
            *r += dangling_share;
        }
        rank = next;
    }
    rank
}

/// Union-Find (disjoint-set) with path compression and union by rank.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> UnionFind {
        UnionFind {
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
    fn union(&mut self, a: usize, b: usize) -> bool {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra == rb {
            return false;
        }
        match self.rank[ra].cmp(&self.rank[rb]) {
            std::cmp::Ordering::Less => self.parent[ra] = rb,
            std::cmp::Ordering::Greater => self.parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                self.parent[rb] = ra;
                self.rank[ra] += 1;
            }
        }
        true
    }
}

/// The minimum spanning tree (Kruskal) of an undirected weighted graph on `n`
/// nodes given as `(u, v, weight)` edges. Returns the chosen edges and their
/// total weight. For a disconnected graph this is a minimum spanning *forest*.
pub fn kruskal(n: usize, edges: &[(usize, usize, f64)]) -> (Vec<(usize, usize, f64)>, f64) {
    let mut sorted: Vec<&(usize, usize, f64)> = edges.iter().collect();
    sorted.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());
    let mut uf = UnionFind::new(n);
    let mut chosen = Vec::new();
    let mut total = 0.0;
    for &&(u, v, w) in &sorted {
        if u < n && v < n && uf.union(u, v) {
            chosen.push((u, v, w));
            total += w;
        }
    }
    (chosen, total)
}

/// Maximum flow from `source` to `sink` (Edmonds-Karp: BFS-augmenting
/// Ford-Fulkerson) on a directed weighted graph on `n` nodes. Edges are
/// `(from, to, capacity)`; parallel edges accumulate.
pub fn max_flow(n: usize, edges: &[(usize, usize, f64)], source: usize, sink: usize) -> f64 {
    if source >= n || sink >= n || source == sink {
        return 0.0;
    }
    // Residual capacity matrix.
    let mut cap = vec![vec![0.0f64; n]; n];
    for &(u, v, c) in edges {
        if u < n && v < n {
            cap[u][v] += c;
        }
    }
    let mut flow = 0.0;
    loop {
        // BFS for an augmenting path, recording parents.
        let mut parent = vec![usize::MAX; n];
        parent[source] = source;
        let mut q = VecDeque::from([source]);
        while let Some(u) = q.pop_front() {
            for v in 0..n {
                if parent[v] == usize::MAX && cap[u][v] > 1e-12 {
                    parent[v] = u;
                    q.push_back(v);
                }
            }
        }
        if parent[sink] == usize::MAX {
            break; // no augmenting path
        }
        // Bottleneck along the path.
        let mut bottleneck = f64::INFINITY;
        let mut v = sink;
        while v != source {
            let u = parent[v];
            bottleneck = bottleneck.min(cap[u][v]);
            v = u;
        }
        // Apply the augmenting flow.
        let mut v = sink;
        while v != source {
            let u = parent[v];
            cap[u][v] -= bottleneck;
            cap[v][u] += bottleneck;
            v = u;
        }
        flow += bottleneck;
    }
    flow
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pagerank_sums_to_one_and_ranks_hub() {
        // Star pointing inward: 1,2,3 all → 0. Node 0 should dominate.
        let mut g = Graph::new(4, true);
        g.add_edge(1, 0).add_edge(2, 0).add_edge(3, 0);
        let pr = pagerank(&g, 0.85, 100);
        assert!((pr.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        assert!(pr[0] > pr[1] && pr[0] > pr[2] && pr[0] > pr[3]);
    }

    #[test]
    fn degree_centrality_basic() {
        let mut g = Graph::new(3, false);
        g.add_edge(0, 1).add_edge(0, 2); // node 0 connects to both others
        let dc = degree_centrality(&g);
        assert!((dc[0] - 1.0).abs() < 1e-12); // 2/(3-1)
        assert!((dc[1] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn kruskal_minimum_spanning_tree() {
        // Classic 4-node graph; MST weight is 1+2+... let's use known edges.
        // Edges: (0-1,1) (1-2,2) (0-2,2) (2-3,3) (0-3,10)
        // MST picks 0-1(1), 1-2(2), 2-3(3) → total 6.
        let edges = [
            (0, 1, 1.0),
            (1, 2, 2.0),
            (0, 2, 2.0),
            (2, 3, 3.0),
            (0, 3, 10.0),
        ];
        let (chosen, total) = kruskal(4, &edges);
        assert_eq!(chosen.len(), 3); // n-1 edges
        assert!((total - 6.0).abs() < 1e-12);
    }

    #[test]
    fn max_flow_classic() {
        // Network 0→1(3), 0→2(2), 1→2(1), 1→3(2), 2→3(3).
        // Min cut {0}|{1,2,3} = 3+2 = 5, so max flow from 0 to 3 is 5
        // (augmenting paths 0→1→3:2, 0→2→3:2, 0→1→2→3:1).
        let edges = [
            (0, 1, 3.0),
            (0, 2, 2.0),
            (1, 2, 1.0),
            (1, 3, 2.0),
            (2, 3, 3.0),
        ];
        let f = max_flow(4, &edges, 0, 3);
        assert!((f - 5.0).abs() < 1e-9, "flow={f}");
    }
}
