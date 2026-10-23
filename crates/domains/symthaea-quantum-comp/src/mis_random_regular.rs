//! Seeded generation of simple regular graphs for QCOMP random-regular MIS studies.
//!
//! The generator uses the configuration model: create d stubs per vertex, shuffle the
//! stubs with a deterministic pseudo-random generator, and pair them. A candidate with
//! a self-loop or repeated undirected edge is rejected in full. Conditional on a
//! uniform stub permutation, rejection sampling yields the uniform simple d-regular
//! graph distribution; this implementation uses the crate's seeded non-cryptographic
//! PRNG, so its operational guarantee is deterministic replay, not cryptographic or
//! formal randomness.
//!
//! The main target is the 3-regular MIS workload studied by fixed-angle QAOA-guided
//! greedy algorithms. Generated graphs are separate from QOBLIB's pinned named cohort.
//! The output should be content-digested and shared across every solver arm.

use core::fmt;

use crate::mis_verifier::{MisGraph, MisGraphError};
use crate::rng::XorShift64;

/// Configuration for one seeded simple d-regular graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MisRandomRegularConfig {
    /// Number of graph vertices; must be greater than zero.
    pub vertex_count: usize,
    /// Required degree of every vertex; must be less than vertex_count.
    pub degree: usize,
    /// Seed for deterministic pseudo-random stub permutation.
    pub seed: u64,
    /// Maximum whole-permutation attempts before returning an error.
    pub max_attempts: usize,
}

impl MisRandomRegularConfig {
    /// Creates a configuration using a conservative finite retry limit.
    pub const fn new(vertex_count: usize, degree: usize, seed: u64) -> Self {
        Self {
            vertex_count,
            degree,
            seed,
            max_attempts: 10_000,
        }
    }
}

/// Number of permutation attempts used to obtain one valid simple regular graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MisRandomRegularReport {
    /// Configuration used for deterministic replay.
    pub config: MisRandomRegularConfig,
    /// Number of full shuffled-stub pairings attempted, including the accepted one.
    pub attempts: usize,
    /// Number of unique undirected edges in the accepted graph.
    pub edge_count: usize,
}

/// Invalid graph-generation request or exhausted rejection budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisRandomRegularError {
    /// No regular graph was requested with a positive vertex count.
    ZeroVertices,
    /// Simple d-regular graphs require degree < vertex count.
    DegreeNotLessThanVertices { vertex_count: usize, degree: usize },
    /// n*d must be even for a d-regular graph to exist.
    OddStubCount { vertex_count: usize, degree: usize },
    /// The number of stubs overflowed usize arithmetic.
    StubCountOverflow { vertex_count: usize, degree: usize },
    /// No attempts were permitted.
    ZeroRetryBudget,
    /// No simple pairing was found within the bounded number of attempts.
    RetryBudgetExhausted {
        /// Configured maximum number of attempts.
        max_attempts: usize,
    },
    /// Constructing the accepted graph unexpectedly failed.
    GraphConstruction(MisGraphError),
}

impl fmt::Display for MisRandomRegularError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroVertices => write!(f, "random regular graph requires at least one vertex"),
            Self::DegreeNotLessThanVertices { vertex_count, degree } => write!(
                f,
                "degree {degree} must be smaller than vertex count {vertex_count} for a simple graph"
            ),
            Self::OddStubCount { vertex_count, degree } => write!(
                f,
                "vertex_count * degree must be even, got {vertex_count} * {degree}"
            ),
            Self::StubCountOverflow { vertex_count, degree } => write!(
                f,
                "vertex_count * degree overflows usize: {vertex_count} * {degree}"
            ),
            Self::ZeroRetryBudget => write!(f, "max_attempts must be positive"),
            Self::RetryBudgetExhausted { max_attempts } => write!(
                f,
                "no simple regular graph found after {max_attempts} full pairing attempts"
            ),
            Self::GraphConstruction(error) => write!(f, "failed to construct graph: {error}"),
        }
    }
}

impl std::error::Error for MisRandomRegularError {}

/// Generates a seeded simple undirected d-regular graph with deterministic replay.
///
/// For a candidate permutation, each consecutive stub pair becomes an undirected edge.
/// The entire pairing is discarded if it creates a self-loop or duplicate edge. If a
/// simple graph is accepted, all vertices have exactly the configured degree and
/// the result has vertex_count * degree / 2 canonical edges.
///
/// The routine has a finite retry limit and fails explicitly rather than silently
/// returning a non-regular graph. Degree zero is supported for positive vertex counts
/// and returns an edgeless graph. Degree one and typical sparse degree-three cases are
/// supported; dense regular graphs can require many rejections and are not the target
/// workload. Use the same returned graph value/digest for every paired solver arm.
pub fn generate_mis_random_regular(
    config: MisRandomRegularConfig,
) -> Result<(MisGraph, MisRandomRegularReport), MisRandomRegularError> {
    if config.vertex_count == 0 {
        return Err(MisRandomRegularError::ZeroVertices);
    }
    if config.max_attempts == 0 {
        return Err(MisRandomRegularError::ZeroRetryBudget);
    }
    if config.degree >= config.vertex_count {
        return Err(MisRandomRegularError::DegreeNotLessThanVertices {
            vertex_count: config.vertex_count,
            degree: config.degree,
        });
    }
    let stub_count = config.vertex_count.checked_mul(config.degree).ok_or(
        MisRandomRegularError::StubCountOverflow {
            vertex_count: config.vertex_count,
            degree: config.degree,
        },
    )?;
    if stub_count % 2 != 0 {
        return Err(MisRandomRegularError::OddStubCount {
            vertex_count: config.vertex_count,
            degree: config.degree,
        });
    }
    if config.degree == 0 {
        let graph = MisGraph::new(config.vertex_count, [])
            .map_err(MisRandomRegularError::GraphConstruction)?;
        return Ok((
            graph,
            MisRandomRegularReport {
                config,
                attempts: 1,
                edge_count: 0,
            },
        ));
    }

    let mut stubs = Vec::with_capacity(stub_count);
    for vertex in 0..config.vertex_count {
        for _ in 0..config.degree {
            stubs.push(vertex);
        }
    }
    let mut rng = XorShift64::new(config.seed);

    for attempt in 1..=config.max_attempts {
        // Every attempt begins from the same canonical stub multiset.
        let mut shuffled = stubs.clone();
        for index in (1..shuffled.len()).rev() {
            let swap_with = sample_below(&mut rng, index + 1);
            shuffled.swap(index, swap_with);
        }

        let mut edges = Vec::with_capacity(stub_count / 2);
        let mut seen = std::collections::HashSet::with_capacity(stub_count / 2);
        let mut simple = true;
        for pair in shuffled.chunks_exact(2) {
            let u = pair[0];
            let v = pair[1];
            if u == v {
                simple = false;
                break;
            }
            let canonical = if u < v { (u, v) } else { (v, u) };
            if !seen.insert(canonical) {
                simple = false;
                break;
            }
            edges.push(canonical);
        }
        if !simple {
            continue;
        }

        // The construction invariant makes these validations redundant in normal
        // operation, but retaining MisGraph::new keeps the verifier boundary explicit.
        let graph = MisGraph::new(config.vertex_count, edges)
            .map_err(MisRandomRegularError::GraphConstruction)?;
        return Ok((
            graph,
            MisRandomRegularReport {
                config,
                attempts: attempt,
                edge_count: stub_count / 2,
            },
        ));
    }

    Err(MisRandomRegularError::RetryBudgetExhausted {
        max_attempts: config.max_attempts,
    })
}

/// Samples a value in 0..upper without modulo bias for a u64 source.
///
/// The accepted prefix has a size divisible by upper. In practice upper is
/// bounded by the in-memory stub count, far below u64::MAX.
fn sample_below(rng: &mut XorShift64, upper: usize) -> usize {
    debug_assert!(upper > 0);
    let bound = upper as u64;
    let accept_below = u64::MAX - (u64::MAX % bound);
    loop {
        let value = rng.next_u64();
        if value < accept_below {
            return (value % bound) as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn degrees(graph: &MisGraph) -> Vec<usize> {
        let mut result = vec![0; graph.vertex_count()];
        for &(u, v) in graph.edges() {
            result[u] += 1;
            result[v] += 1;
        }
        result
    }

    #[test]
    fn seeded_three_regular_graph_has_exact_degree_and_edge_count() {
        let config = MisRandomRegularConfig::new(50, 3, 0x5152_4547_3030_3031);
        let (graph, report) = generate_mis_random_regular(config).unwrap();
        assert_eq!(graph.vertex_count(), 50);
        assert_eq!(graph.edges().len(), 75);
        assert_eq!(report.edge_count, 75);
        assert!(report.attempts >= 1 && report.attempts <= config.max_attempts);
        assert!(degrees(&graph).iter().all(|&degree| degree == 3));
        assert!(graph.edges().windows(2).all(|pair| pair[0] < pair[1]));
        assert!(graph.edges().iter().all(|(u, v)| u < v));
    }

    #[test]
    fn same_seed_replays_identical_canonical_graph_and_report() {
        let config = MisRandomRegularConfig::new(100, 3, 123_456);
        let first = generate_mis_random_regular(config).unwrap();
        let second = generate_mis_random_regular(config).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn zero_degree_returns_an_edgeless_regular_graph() {
        let (graph, report) =
            generate_mis_random_regular(MisRandomRegularConfig::new(17, 0, 7)).unwrap();
        assert_eq!(graph.vertex_count(), 17);
        assert!(graph.edges().is_empty());
        assert_eq!(report.edge_count, 0);
    }

    #[test]
    fn invalid_degree_and_parity_fail_closed() {
        assert_eq!(
            generate_mis_random_regular(MisRandomRegularConfig::new(5, 5, 1)),
            Err(MisRandomRegularError::DegreeNotLessThanVertices {
                vertex_count: 5,
                degree: 5,
            })
        );
        assert_eq!(
            generate_mis_random_regular(MisRandomRegularConfig::new(5, 3, 1)),
            Err(MisRandomRegularError::OddStubCount {
                vertex_count: 5,
                degree: 3,
            })
        );
    }

    #[test]
    fn invalid_vertex_and_retry_budget_fail_closed() {
        assert_eq!(
            generate_mis_random_regular(MisRandomRegularConfig::new(0, 0, 1)),
            Err(MisRandomRegularError::ZeroVertices)
        );
        assert_eq!(
            generate_mis_random_regular(MisRandomRegularConfig {
                max_attempts: 0,
                ..MisRandomRegularConfig::new(20, 3, 2)
            }),
            Err(MisRandomRegularError::ZeroRetryBudget)
        );
    }

    #[test]
    fn degree_one_graph_has_exactly_one_edge_per_vertex_pair() {
        let (graph, _) = generate_mis_random_regular(MisRandomRegularConfig::new(40, 1, 88))
            .unwrap();
        assert_eq!(graph.edges().len(), 20);
        assert!(degrees(&graph).iter().all(|&degree| degree == 1));
    }
}
