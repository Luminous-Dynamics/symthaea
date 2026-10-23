//! Seeded generator for simple d-regular MIS graph instances.
//!
//! This uses the configuration model: create d stubs per vertex, shuffle and pair them,
//! rejecting any pairing containing loops or repeated undirected edges. With a uniform
//! stub permutation, rejection conditional on simplicity yields the uniform simple
//! regular-graph distribution. The implementation's PRNG is deterministic and not
//! cryptographic; its primary guarantee is exact replay from a seed.

use core::fmt;
use std::collections::HashSet;

use crate::mis_verifier::{MisGraph, MisGraphError};
use crate::rng::XorShift64;

/// Seeded random-regular graph request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MisRandomRegularConfig {
    /// Positive number of vertices.
    pub vertex_count: usize,
    /// Degree required at every vertex; must be less than vertex count.
    pub degree: usize,
    /// Seed for deterministic pseudo-random shuffling.
    pub seed: u64,
    /// Maximum complete stub pairings to attempt.
    pub max_attempts: usize,
}

impl MisRandomRegularConfig {
    /// Standard configuration with a bounded rejection budget.
    pub const fn new(vertex_count: usize, degree: usize, seed: u64) -> Self {
        Self { vertex_count, degree, seed, max_attempts: 10_000 }
    }
}

/// Provenance returned with the generated graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MisRandomRegularReport {
    /// Exact generator configuration.
    pub config: MisRandomRegularConfig,
    /// Number of full permutations tried, including the accepted one.
    pub attempts: usize,
    /// Edge count of the accepted graph.
    pub edge_count: usize,
}

/// Rejected graph request or exhausted attempt budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisRandomRegularError {
    /// At least one vertex is required.
    ZeroVertices,
    /// A simple graph requires degree less than its vertex count.
    DegreeNotLessThanVertices { vertex_count: usize, degree: usize },
    /// A regular graph requires an even vertex-count times degree.
    OddStubCount { vertex_count: usize, degree: usize },
    /// Vertex count times degree overflowed usize arithmetic.
    StubCountOverflow { vertex_count: usize, degree: usize },
    /// No attempts were allowed.
    ZeroRetryBudget,
    /// No simple pairing was found within the configured budget.
    RetryBudgetExhausted { max_attempts: usize },
    /// The independently validating graph constructor rejected the generated edges.
    GraphConstruction(MisGraphError),
}

impl fmt::Display for MisRandomRegularError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroVertices => write!(f, "vertex_count must be positive"),
            Self::DegreeNotLessThanVertices { vertex_count, degree } =>
                write!(f, "degree {degree} must be less than vertex_count {vertex_count}"),
            Self::OddStubCount { vertex_count, degree } =>
                write!(f, "vertex_count * degree must be even, got {vertex_count} * {degree}"),
            Self::StubCountOverflow { vertex_count, degree } =>
                write!(f, "vertex_count * degree overflows usize: {vertex_count} * {degree}"),
            Self::ZeroRetryBudget => write!(f, "max_attempts must be positive"),
            Self::RetryBudgetExhausted { max_attempts } =>
                write!(f, "no simple pairing found in {max_attempts} attempts"),
            Self::GraphConstruction(error) => write!(f, "graph construction failed: {error}"),
        }
    }
}

impl std::error::Error for MisRandomRegularError {}

/// Generate a simple d-regular graph with deterministic replay.
///
/// Degree zero is supported (the graph is edgeless). For nonzero degree, the routine
/// shuffles d copies of each vertex, pairs stubs, and rejects the entire pairing if it
/// contains a self-loop or duplicate edge. Dense regular graphs may exhaust the finite
/// retry budget; exhaustion is reported as an error rather than returning a malformed
/// graph. The target research workload is sparse random 3-regular graphs, not dense graphs.
pub fn generate_mis_random_regular(
    config: MisRandomRegularConfig,
) -> Result<(MisGraph, MisRandomRegularReport), MisRandomRegularError> {
    if config.vertex_count == 0 { return Err(MisRandomRegularError::ZeroVertices); }
    if config.max_attempts == 0 { return Err(MisRandomRegularError::ZeroRetryBudget); }
    if config.degree >= config.vertex_count {
        return Err(MisRandomRegularError::DegreeNotLessThanVertices {
            vertex_count: config.vertex_count, degree: config.degree,
        });
    }
    let stub_count = config.vertex_count.checked_mul(config.degree).ok_or(
        MisRandomRegularError::StubCountOverflow {
            vertex_count: config.vertex_count, degree: config.degree,
        },
    )?;
    if stub_count % 2 != 0 {
        return Err(MisRandomRegularError::OddStubCount {
            vertex_count: config.vertex_count, degree: config.degree,
        });
    }
    if config.degree == 0 {
        let graph = MisGraph::new(config.vertex_count, [])
            .map_err(MisRandomRegularError::GraphConstruction)?;
        return Ok((graph, MisRandomRegularReport { config, attempts: 1, edge_count: 0 }));
    }

    let mut stubs = Vec::with_capacity(stub_count);
    for vertex in 0..config.vertex_count {
        for _ in 0..config.degree { stubs.push(vertex); }
    }
    let mut rng = XorShift64::new(config.seed);
    for attempt in 1..=config.max_attempts {
        let mut shuffled = stubs.clone();
        for index in (1..shuffled.len()).rev() {
            let swap_with = sample_below(&mut rng, index + 1);
            shuffled.swap(index, swap_with);
        }
        let mut edges = Vec::with_capacity(stub_count / 2);
        let mut seen = HashSet::with_capacity(stub_count / 2);
        let mut simple = true;
        for pair in shuffled.chunks_exact(2) {
            let (u, v) = (pair[0], pair[1]);
            if u == v { simple = false; break; }
            let edge = if u < v { (u, v) } else { (v, u) };
            if !seen.insert(edge) { simple = false; break; }
            edges.push(edge);
        }
        if !simple { continue; }

        let graph = MisGraph::new(config.vertex_count, edges)
            .map_err(MisRandomRegularError::GraphConstruction)?;
        return Ok((graph, MisRandomRegularReport {
            config, attempts: attempt, edge_count: stub_count / 2,
        }));
    }
    Err(MisRandomRegularError::RetryBudgetExhausted {
        max_attempts: config.max_attempts,
    })
}

/// Unbiased bounded conversion from one u64 draw, using rejection of the final
/// incomplete residue bucket.
fn sample_below(rng: &mut XorShift64, upper: usize) -> usize {
    debug_assert!(upper > 0);
    let bound = upper as u64;
    let accept_below = u64::MAX - (u64::MAX % bound);
    loop {
        let value = rng.next_u64();
        if value < accept_below { return (value % bound) as usize; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn degrees(graph: &MisGraph) -> Vec<usize> {
        let mut degree = vec![0; graph.vertex_count()];
        for &(u, v) in graph.edges() { degree[u] += 1; degree[v] += 1; }
        degree
    }

    #[test]
    fn random_three_regular_graph_has_exact_degree_and_edge_count() {
        let config = MisRandomRegularConfig::new(50, 3, 0x5152_4547_3030_3031);
        let (graph, report) = generate_mis_random_regular(config).unwrap();
        assert_eq!(graph.vertex_count(), 50);
        assert_eq!(graph.edges().len(), 75);
        assert_eq!(report.edge_count, 75);
        assert!((1..=config.max_attempts).contains(&report.attempts));
        assert!(degrees(&graph).iter().all(|&d| d == 3));
        assert!(graph.edges().iter().all(|&(u,v)| u < v));
        assert!(graph.edges().windows(2).all(|w| w[0] < w[1]));
    }

    #[test]
    fn same_seed_replays_identical_graph_and_report() {
        let config = MisRandomRegularConfig::new(100, 3, 123_456);
        assert_eq!(generate_mis_random_regular(config).unwrap(),
                   generate_mis_random_regular(config).unwrap());
    }

    #[test]
    fn zero_degree_returns_an_edgeless_graph() {
        let (graph, report) = generate_mis_random_regular(
            MisRandomRegularConfig::new(17, 0, 7)).unwrap();
        assert_eq!(graph.vertex_count(), 17);
        assert!(graph.edges().is_empty());
        assert_eq!(report.edge_count, 0);
    }

    #[test]
    fn invalid_degree_parity_and_retry_budget_fail_closed() {
        assert_eq!(generate_mis_random_regular(MisRandomRegularConfig::new(5, 5, 1)),
            Err(MisRandomRegularError::DegreeNotLessThanVertices { vertex_count: 5, degree: 5 }));
        assert_eq!(generate_mis_random_regular(MisRandomRegularConfig::new(5, 3, 1)),
            Err(MisRandomRegularError::OddStubCount { vertex_count: 5, degree: 3 }));
        assert_eq!(generate_mis_random_regular(MisRandomRegularConfig::new(0, 0, 1)),
            Err(MisRandomRegularError::ZeroVertices));
        assert_eq!(generate_mis_random_regular(MisRandomRegularConfig {
            max_attempts: 0, ..MisRandomRegularConfig::new(20, 3, 2)
        }), Err(MisRandomRegularError::ZeroRetryBudget));
    }

    #[test]
    fn degree_one_graph_is_a_perfect_matching() {
        let (graph, _) = generate_mis_random_regular(MisRandomRegularConfig::new(40, 1, 88)).unwrap();
        assert_eq!(graph.edges().len(), 20);
        assert!(degrees(&graph).iter().all(|&d| d == 1));
    }
}
