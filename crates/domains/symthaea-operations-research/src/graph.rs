// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Dijkstra's single-source shortest paths on a non-negative weighted graph.

/// Shortest-path distances from `source` to every node, given an adjacency list
/// `adjacency[u] = [(v, weight), …]` with non-negative weights. Unreachable
/// nodes are `f64::INFINITY`.
pub fn dijkstra(adjacency: &[Vec<(usize, f64)>], source: usize) -> Vec<f64> {
    let n = adjacency.len();
    let mut dist = vec![f64::INFINITY; n];
    let mut visited = vec![false; n];
    if source >= n {
        return dist;
    }
    dist[source] = 0.0;

    for _ in 0..n {
        // Pick the unvisited node with the smallest tentative distance.
        let mut u = None;
        let mut best = f64::INFINITY;
        for (i, &d) in dist.iter().enumerate() {
            if !visited[i] && d < best {
                best = d;
                u = Some(i);
            }
        }
        let Some(u) = u else { break };
        visited[u] = true;
        for &(v, w) in &adjacency[u] {
            if dist[u] + w < dist[v] {
                dist[v] = dist[u] + w;
            }
        }
    }
    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> Vec<Vec<(usize, f64)>> {
        // 0→1 (4), 0→2 (1), 2→1 (2), 1→3 (1), 2→3 (5).
        vec![
            vec![(1, 4.0), (2, 1.0)],
            vec![(3, 1.0)],
            vec![(1, 2.0), (3, 5.0)],
            vec![],
        ]
    }

    #[test]
    fn shortest_paths_prefer_the_cheap_route() {
        let d = dijkstra(&sample_graph(), 0);
        // 0→2→1→3 = 1+2+1 = 4 beats 0→1 = 4 direct but 0→2→1 = 3 for node 1.
        assert_eq!(d, vec![0.0, 3.0, 1.0, 4.0]);
    }

    #[test]
    fn unreachable_node_is_infinite() {
        // Node 1 has no outgoing edges; from source 3 nothing is reachable.
        let d = dijkstra(&sample_graph(), 3);
        assert_eq!(d[3], 0.0);
        assert!(d[0].is_infinite());
    }
}
