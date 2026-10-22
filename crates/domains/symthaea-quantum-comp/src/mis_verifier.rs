//! Independent verification and tiny-instance exact reference for maximum independent set.
//!
//! This module is a benchmark-validation primitive, not a competitive MIS solver.
//! It checks candidate solutions independently of the solver that produced them
//! and supplies a bounded exact reference for small correctness fixtures. It does
//! not claim state-of-the-art optimization performance or quantum advantage.

use core::fmt;

/// Maximum graph size accepted by the deliberately small exhaustive-reference solver.
///
/// Keep this bound conservative: this reference is for correctness anchors, not
/// for competing with production exact or heuristic solvers.
pub const EXACT_ORACLE_MAX_VERTICES: usize = 24;

/// A simple undirected graph used for maximum-independent-set benchmark cases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisGraph {
    vertex_count: usize,
    /// Canonical edges (u, v) with u < v, sorted and deduplicated.
    edges: Vec<(usize, usize)>,
    /// Sorted adjacency lists used for deterministic validation and search.
    adjacency: Vec<Vec<usize>>,
}

/// Errors raised while constructing a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisGraphError {
    /// An edge references a vertex outside the permitted vertex range.
    VertexOutOfRange {
        /// First endpoint of the invalid edge.
        u: usize,
        /// Second endpoint of the invalid edge.
        v: usize,
        /// Number of vertices in the graph.
        vertex_count: usize,
    },
    /// A simple graph cannot contain a self-loop.
    SelfLoop {
        /// Vertex with the self-loop.
        vertex: usize,
    },
}

impl fmt::Display for MisGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexOutOfRange { u, v, vertex_count } => write!(
                f,
                "edge ({u}, {v}) references a vertex outside 0..{vertex_count}"
            ),
            Self::SelfLoop { vertex } => write!(f, "self-loop at vertex {vertex} is not allowed"),
        }
    }
}

impl std::error::Error for MisGraphError {}

/// A violation found when checking a candidate independent set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisViolation {
    /// The candidate listed a vertex more than once.
    DuplicateVertex {
        /// Repeated vertex.
        vertex: usize,
    },
    /// The candidate listed a vertex outside the graph.
    VertexOutOfRange {
        /// Invalid vertex.
        vertex: usize,
        /// Number of vertices in the graph.
        vertex_count: usize,
    },
    /// Both endpoints of an edge were selected.
    AdjacentSelectedVertices {
        /// First selected endpoint.
        u: usize,
        /// Second selected endpoint.
        v: usize,
    },
    /// The reported objective differs from the unique, in-range selected count.
    ObjectiveMismatch {
        /// Objective supplied by the candidate solver.
        reported: usize,
        /// Cardinality computed independently from valid, unique selected vertices.
        computed: usize,
    },
}

/// Result of independent candidate-solution verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisVerificationReport {
    /// True only when no violation was found.
    pub feasible: bool,
    /// Number of unique, in-range vertices in the submitted list.
    ///
    /// This is the independently computed cardinality. A candidate with duplicate
    /// or out-of-range vertices is still marked infeasible even though this count
    /// remains available for diagnostics.
    pub computed_objective: usize,
    /// All detected violations, in deterministic order.
    pub violations: Vec<MisViolation>,
}

/// Error raised by the deliberately bounded exact reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExactMisError {
    /// The graph is too large for this exhaustive correctness oracle.
    VertexLimitExceeded {
        /// Actual number of vertices.
        vertex_count: usize,
        /// Maximum supported size.
        maximum: usize,
    },
}

impl fmt::Display for ExactMisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::VertexLimitExceeded {
                vertex_count,
                maximum,
            } => write!(
                f,
                "exact MIS reference supports at most {maximum} vertices, got {vertex_count}"
            ),
        }
    }
}

impl std::error::Error for ExactMisError {}

impl MisGraph {
    /// Construct a simple undirected graph.
    ///
    /// Edge endpoints must be in range and self-loops are rejected. Reversed and
    /// repeated copies of the same undirected edge are canonicalized, so equivalent
    /// input orderings result in the same graph value.
    pub fn new(
        vertex_count: usize,
        edges: impl IntoIterator<Item = (usize, usize)>,
    ) -> Result<Self, MisGraphError> {
        let mut canonical_edges = Vec::new();

        for (u, v) in edges {
            if u >= vertex_count || v >= vertex_count {
                return Err(MisGraphError::VertexOutOfRange {
                    u,
                    v,
                    vertex_count,
                });
            }
            if u == v {
                return Err(MisGraphError::SelfLoop { vertex: u });
            }
            canonical_edges.push(if u < v { (u, v) } else { (v, u) });
        }

        canonical_edges.sort_unstable();
        canonical_edges.dedup();

        let mut adjacency = vec![Vec::new(); vertex_count];
        for &(u, v) in &canonical_edges {
            adjacency[u].push(v);
            adjacency[v].push(u);
        }
        for neighbors in &mut adjacency {
            neighbors.sort_unstable();
        }

        Ok(Self {
            vertex_count,
            edges: canonical_edges,
            adjacency,
        })
    }

    /// Number of vertices in the graph.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Canonical, sorted, unique undirected edges.
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Independently verify a candidate independent set.
    ///
    /// The checker does not trust the candidate solver's reported objective. It
    /// derives cardinality from unique in-range submitted vertices and checks every
    /// selected edge. All detected violations are returned, not just the first one.
    pub fn verify_solution(
        &self,
        selected_vertices: &[usize],
        reported_objective: Option<usize>,
    ) -> MisVerificationReport {
        let mut seen = vec![false; self.vertex_count];
        let mut violations = Vec::new();

        for &vertex in selected_vertices {
            if vertex >= self.vertex_count {
                violations.push(MisViolation::VertexOutOfRange {
                    vertex,
                    vertex_count: self.vertex_count,
                });
            } else if seen[vertex] {
                violations.push(MisViolation::DuplicateVertex { vertex });
            } else {
                seen[vertex] = true;
            }
        }

        // Iterate canonical edges so each selected conflict is emitted exactly once.
        for &(u, v) in &self.edges {
            if seen[u] && seen[v] {
                violations.push(MisViolation::AdjacentSelectedVertices { u, v });
            }
        }

        let computed_objective = seen.iter().filter(|&&selected| selected).count();
        if let Some(reported) = reported_objective {
            if reported != computed_objective {
                violations.push(MisViolation::ObjectiveMismatch {
                    reported,
                    computed: computed_objective,
                });
            }
        }

        MisVerificationReport {
            feasible: violations.is_empty(),
            computed_objective,
            violations,
        }
    }

    /// Find an exact maximum independent set for a small graph.
    ///
    /// This bounded branch-and-bound implementation is intended only as an
    /// independent correctness oracle for small fixtures. It is not intended to
    /// compete with dedicated MIS solvers. The result is deterministic and the
    /// returned vertex indices are sorted.
    pub fn exact_maximum_independent_set(&self) -> Result<Vec<usize>, ExactMisError> {
        if self.vertex_count > EXACT_ORACLE_MAX_VERTICES {
            return Err(ExactMisError::VertexLimitExceeded {
                vertex_count: self.vertex_count,
                maximum: EXACT_ORACLE_MAX_VERTICES,
            });
        }

        let candidates: Vec<usize> = (0..self.vertex_count).collect();
        let mut selected = Vec::new();
        let mut best = Vec::new();
        self.search_exact(&candidates, &mut selected, &mut best);
        Ok(best)
    }

    fn search_exact(
        &self,
        candidates: &[usize],
        selected: &mut Vec<usize>,
        best: &mut Vec<usize>,
    ) {
        if selected.len() > best.len() {
            *best = selected.clone();
        }
        if selected.len() + candidates.len() <= best.len() || candidates.is_empty() {
            return;
        }

        // Include the first candidate, then exclude it. Filtering neighbors keeps
        // every recursive include-branch feasible by construction.
        let vertex = candidates[0];
        let remaining = &candidates[1..];
        let non_neighbors: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|&other| self.adjacency[vertex].binary_search(&other).is_err())
            .collect();

        selected.push(vertex);
        self.search_exact(&non_neighbors, selected, best);
        selected.pop();

        self.search_exact(remaining, selected, best);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_graph_has_empty_maximum_independent_set() {
        let graph = MisGraph::new(0, []).expect("empty graph is valid");
        let solution = graph
            .exact_maximum_independent_set()
            .expect("small exact oracle should run");
        assert!(solution.is_empty());
        assert!(graph.verify_solution(&solution, Some(0)).feasible);
    }

    #[test]
    fn edgeless_graph_selects_every_vertex() {
        let graph = MisGraph::new(4, []).expect("edgeless graph is valid");
        let solution = graph
            .exact_maximum_independent_set()
            .expect("small exact oracle should run");
        assert_eq!(solution, vec![0, 1, 2, 3]);
        assert!(graph.verify_solution(&solution, Some(4)).feasible);
    }

    #[test]
    fn complete_graph_has_one_vertex_optimum() {
        let graph = MisGraph::new(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
            .expect("complete graph is valid");
        let solution = graph
            .exact_maximum_independent_set()
            .expect("small exact oracle should run");
        assert_eq!(solution.len(), 1);
        assert!(graph.verify_solution(&solution, Some(1)).feasible);
    }

    #[test]
    fn path_graph_has_expected_optimum_cardinality() {
        let graph = MisGraph::new(4, [(0, 1), (1, 2), (2, 3)])
            .expect("path graph is valid");
        let solution = graph
            .exact_maximum_independent_set()
            .expect("small exact oracle should run");
        assert_eq!(solution.len(), 2);
        assert!(graph.verify_solution(&solution, Some(2)).feasible);
    }

    #[test]
    fn duplicate_and_reversed_edges_are_canonicalized() {
        let graph = MisGraph::new(3, [(1, 0), (0, 1), (2, 1)])
            .expect("valid graph");
        assert_eq!(graph.edges(), &[(0, 1), (1, 2)]);
    }

    #[test]
    fn rejects_out_of_range_edge_endpoint() {
        assert_eq!(
            MisGraph::new(2, [(0, 2)]).expect_err("endpoint is invalid"),
            MisGraphError::VertexOutOfRange {
                u: 0,
                v: 2,
                vertex_count: 2,
            }
        );
    }

    #[test]
    fn rejects_self_loop() {
        assert_eq!(
            MisGraph::new(1, [(0, 0)]).expect_err("self loop is invalid"),
            MisGraphError::SelfLoop { vertex: 0 }
        );
    }

    #[test]
    fn checker_rejects_adjacent_selected_vertices_and_wrong_objective() {
        let graph = MisGraph::new(3, [(0, 1)]).expect("valid graph");
        let report = graph.verify_solution(&[0, 1, 2], Some(99));
        assert!(!report.feasible);
        assert_eq!(report.computed_objective, 3);
        assert!(report.violations.contains(&MisViolation::AdjacentSelectedVertices {
            u: 0,
            v: 1,
        }));
        assert!(report.violations.contains(&MisViolation::ObjectiveMismatch {
            reported: 99,
            computed: 3,
        }));
    }

    #[test]
    fn checker_rejects_duplicates_and_unknown_vertices() {
        let graph = MisGraph::new(2, [(0, 1)]).expect("valid graph");
        let report = graph.verify_solution(&[0, 0, 7], Some(1));
        assert!(!report.feasible);
        assert_eq!(report.computed_objective, 1);
        assert!(report.violations.contains(&MisViolation::DuplicateVertex { vertex: 0 }));
        assert!(report.violations.contains(&MisViolation::VertexOutOfRange {
            vertex: 7,
            vertex_count: 2,
        }));
    }

    #[test]
    fn exact_oracle_matches_independent_brute_force_on_every_graph_up_to_five_vertices() {
        // Exhaustively cover every labelled simple graph with 0..=5 vertices.
        // The oracle under test uses branch-and-bound; this test independently
        // enumerates vertex subsets and checks feasibility directly from the edges.
        for vertex_count in 0..=5 {
            let possible_edges: Vec<(usize, usize)> = (0..vertex_count)
                .flat_map(|u| ((u + 1)..vertex_count).map(move |v| (u, v)))
                .collect();
            let graph_variants = 1usize << possible_edges.len();

            for graph_mask in 0..graph_variants {
                let mut edges = Vec::new();
                for (edge_index, &(u, v)) in possible_edges.iter().enumerate() {
                    if graph_mask & (1usize << edge_index) != 0 {
                        edges.push((u, v));
                    }
                }

                let graph = MisGraph::new(vertex_count, edges.clone())
                    .expect("generated simple graph should be valid");
                let exact = graph
                    .exact_maximum_independent_set()
                    .expect("graphs up to five vertices are within the exact limit");

                let mut independently_computed_optimum = 0usize;
                for subset_mask in 0..(1usize << vertex_count) {
                    let feasible = edges.iter().all(|&(u, v)| {
                        subset_mask & (1usize << u) == 0 || subset_mask & (1usize << v) == 0
                    });
                    if feasible {
                        independently_computed_optimum = independently_computed_optimum
                            .max(subset_mask.count_ones() as usize);
                    }
                }

                assert_eq!(
                    exact.len(),
                    independently_computed_optimum,
                    "mismatch for {vertex_count} vertices and graph edge mask {graph_mask}"
                );
                assert!(
                    graph.verify_solution(&exact, Some(exact.len())).feasible,
                    "oracle returned an invalid independent set for edge mask {graph_mask}"
                );
            }
        }
    }

    #[test]
    fn exact_oracle_refuses_graphs_above_its_declared_limit() {
        let graph = MisGraph::new(EXACT_ORACLE_MAX_VERTICES + 1, [])
            .expect("large edgeless graph itself is valid");
        assert_eq!(
            graph.exact_maximum_independent_set().expect_err("must refuse"),
            ExactMisError::VertexLimitExceeded {
                vertex_count: EXACT_ORACLE_MAX_VERTICES + 1,
                maximum: EXACT_ORACLE_MAX_VERTICES,
            }
        );
    }
}
