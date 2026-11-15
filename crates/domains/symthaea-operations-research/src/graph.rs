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

/// Exact shortest-path tree for a graph with non-negative integer edge weights.
///
/// Distances use u128 so sums of u64 edge weights do not lose precision.
/// predecessor_edges[v] stores the caller-defined edge index used to reach v,
/// allowing callers to preserve path provenance without reimplementing Dijkstra.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShortestPathTree {
    /// Exact distance from the source; None means unreachable.
    pub distances: Vec<Option<u128>>,
    /// Previous node on the selected deterministic shortest path.
    pub predecessor_nodes: Vec<Option<usize>>,
    /// Caller-defined edge index used to reach each node.
    pub predecessor_edges: Vec<Option<usize>>,
}

/// Errors returned by dijkstra_u64.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DijkstraError {
    /// The requested source node is outside the graph.
    InvalidSource { source: usize, node_count: usize },
    /// An edge references a node outside the graph.
    InvalidNeighbor {
        from: usize,
        to: usize,
        node_count: usize,
    },
    /// A path-distance sum exceeded the supported integer range.
    DistanceOverflow { from: usize, to: usize },
}

impl std::fmt::Display for DijkstraError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for DijkstraError {}

/// Deterministic Dijkstra with exact u64 edge weights and path provenance.
///
/// adjacency[u] contains (neighbor, weight, edge_index) tuples. The edge_index
/// is opaque to this function and is copied into the resulting predecessor
/// tree. Nodes are selected by (distance, node_index) and each adjacency list
/// is consumed in caller-supplied order; callers that require order-independent
/// path provenance should sort nodes and edges first.
///
/// Zero-weight edges are supported. Negative weights are unrepresentable.
/// Invalid node references and arithmetic overflow are returned as errors.
pub fn dijkstra_u64(
    adjacency: &[Vec<(usize, u64, usize)>],
    source: usize,
) -> Result<ShortestPathTree, DijkstraError> {
    let node_count = adjacency.len();
    if source >= node_count {
        return Err(DijkstraError::InvalidSource {
            source,
            node_count,
        });
    }

    for (from, edges) in adjacency.iter().enumerate() {
        for &(to, _, _) in edges {
            if to >= node_count {
                return Err(DijkstraError::InvalidNeighbor {
                    from,
                    to,
                    node_count,
                });
            }
        }
    }

    let mut distances = vec![None; node_count];
    let mut predecessor_nodes = vec![None; node_count];
    let mut predecessor_edges = vec![None; node_count];
    let mut visited = vec![false; node_count];
    distances[source] = Some(0_u128);

    for _ in 0..node_count {
        let mut next: Option<(usize, u128)> = None;

        for (node, distance) in distances.iter().enumerate() {
            if visited[node] {
                continue;
            }
            let Some(distance) = *distance else {
                continue;
            };

            let replace = match next {
                None => true,
                Some((best_node, best_distance)) => {
                    distance < best_distance
                        || (distance == best_distance && node < best_node)
                }
            };
            if replace {
                next = Some((node, distance));
            }
        }

        let Some((node, distance_to_node)) = next else {
            break;
        };
        visited[node] = true;

        for &(neighbor, weight, edge_index) in &adjacency[node] {
            if visited[neighbor] {
                continue;
            }

            let candidate = distance_to_node
                .checked_add(u128::from(weight))
                .ok_or(DijkstraError::DistanceOverflow {
                    from: node,
                    to: neighbor,
                })?;

            let improves = distances[neighbor]
                .map(|current| candidate < current)
                .unwrap_or(true);

            if improves {
                distances[neighbor] = Some(candidate);
                predecessor_nodes[neighbor] = Some(node);
                predecessor_edges[neighbor] = Some(edge_index);
            }
        }
    }

    Ok(ShortestPathTree {
        distances,
        predecessor_nodes,
        predecessor_edges,
    })
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
