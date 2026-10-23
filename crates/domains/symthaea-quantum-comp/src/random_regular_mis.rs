//! Reproducible configuration-model sampling for regular-graph MIS experiments.
//!
//! The generator creates a deterministic pseudo-random pairing of vertex_count *
//! degree stubs, rejects pairings containing loops or repeated undirected edges,
//! and returns a sorted simple regular graph. It is intended to produce pinned
//! research workloads, not cryptographic randomness or a claim about random-number
//! generator quality. Record the seed, attempt limit, algorithm version and returned
//! graph digest in the experiment manifest. Validate distributional properties before
//! using a generated cohort for confirmatory scientific claims.

use core::fmt;
use crate::rng::XorShift64;

/// Version tag for this deterministic generator's externally visible semantics.
pub const RANDOM_REGULAR_MIS_GENERATOR_VERSION: &str = "configuration-pair-reject-v1";

/// A simple undirected d-regular graph generated with explicit provenance metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RandomRegularMisGraph {
    vertex_count: usize,
    degree: usize,
    seed: u64,
    attempts_used: usize,
    /// Sorted canonical zero-based edges, each represented by u < v.
    edges: Vec<(usize, usize)>,
}

impl RandomRegularMisGraph {
    /// Number of vertices.
    pub fn vertex_count(&self) -> usize {
        self.vertex_count
    }

    /// Required degree of each vertex.
    pub fn degree(&self) -> usize {
        self.degree
    }

    /// Seed passed to the generator.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Number of pairing attempts performed, including the successful attempt.
    pub fn attempts_used(&self) -> usize {
        self.attempts_used
    }

    /// Version tag for the generator algorithm.
    pub fn generator_version(&self) -> &'static str {
        RANDOM_REGULAR_MIS_GENERATOR_VERSION
    }

    /// Sorted, canonical zero-based undirected edges.
    pub fn edges(&self) -> &[(usize, usize)] {
        &self.edges
    }

    /// Degree of every vertex, recomputed from the returned edge list.
    pub fn degree_sequence(&self) -> Vec<usize> {
        let mut degrees = vec![0usize; self.vertex_count];
        for &(u, v) in &self.edges {
            degrees[u] += 1;
            degrees[v] += 1;
        }
        degrees
    }

    /// Recheck structural invariants without trusting stored construction state.
    pub fn validate(&self) -> Result<(), RandomRegularMisError> {
        if self.edges.len() != self.vertex_count * self.degree / 2 {
            return Err(RandomRegularMisError::InvariantViolation {
                reason: "edge count differs from n*d/2",
            });
        }
        let mut previous: Option<(usize, usize)> = None;
        let mut degrees = vec![0usize; self.vertex_count];
        for &(u, v) in &self.edges {
            if u >= self.vertex_count || v >= self.vertex_count {
                return Err(RandomRegularMisError::InvariantViolation {
                    reason: "edge endpoint is out of range",
                });
            }
            if u >= v {
                return Err(RandomRegularMisError::InvariantViolation {
                    reason: "edge is not canonical with u < v",
                });
            }
            if previous.is_some_and(|edge| edge >= (u, v)) {
                return Err(RandomRegularMisError::InvariantViolation {
                    reason: "edges are unsorted or duplicated",
                });
            }
            degrees[u] += 1;
            degrees[v] += 1;
            previous = Some((u, v));
        }
        if degrees.iter().any(|&degree| degree != self.degree) {
            return Err(RandomRegularMisError::InvariantViolation {
                reason: "vertex degree differs from requested regular degree",
            });
        }
        Ok(())
    }

    /// Serialize to deterministic DIMACS edge format using one-based source indices.
    ///
    /// The first comment records the generator version and configuration. The caller
    /// should separately bind the exact output bytes to the benchmark run receipt.
    pub fn to_dimacs(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!(
            "c generated_by={} seed={} attempts={} degree={}\n",
            RANDOM_REGULAR_MIS_GENERATOR_VERSION, self.seed, self.attempts_used, self.degree
        ));
        output.push_str(&format!(
            "p edge {} {}\n",
            self.vertex_count,
            self.edges.len()
        ));
        for &(u, v) in &self.edges {
            output.push_str(&format!("e {} {}\n", u + 1, v + 1));
        }
        output
    }
}

/// Failure while generating or validating a regular graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RandomRegularMisError {
    /// An empty graph is not a benchmark instance for this generator.
    EmptyGraph,
    /// A simple graph cannot have degree greater than or equal to its vertex count.
    DegreeNotSmallerThanVertexCount {
        /// Requested number of vertices.
        vertex_count: usize,
        /// Requested degree.
        degree: usize,
    },
    /// The product n*d overflowed usize.
    StubCountOverflow {
        /// Number of vertices.
        vertex_count: usize,
        /// Degree.
        degree: usize,
    },
    /// The requested degree sequence has an odd total number of stubs.
    OddStubCount {
        /// Number of vertices.
        vertex_count: usize,
        /// Degree.
        degree: usize,
    },
    /// The bounded rejection loop must allow at least one attempt.
    ZeroAttemptLimit,
    /// No simple graph was produced in the configured number of attempts.
    AttemptsExhausted {
        /// Number of vertices.
        vertex_count: usize,
        /// Requested degree.
        degree: usize,
        /// Seed that defines the random stream.
        seed: u64,
        /// Number of unsuccessful pairings.
        attempts: usize,
    },
    /// A constructed graph failed an independent invariant check.
    InvariantViolation {
        /// Brief invariant that failed.
        reason: &'static str,
    },
}

impl fmt::Display for RandomRegularMisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGraph => write!(f, "vertex_count must be positive"),
            Self::DegreeNotSmallerThanVertexCount { vertex_count, degree } => {
                write!(f, "degree {degree} must be less than vertex_count {vertex_count}")
            }
            Self::StubCountOverflow { vertex_count, degree } => {
                write!(f, "stub count overflows usize for n={vertex_count}, d={degree}")
            }
            Self::OddStubCount { vertex_count, degree } => {
                write!(f, "n*d must be even for a d-regular graph (n={vertex_count}, d={degree})")
            }
            Self::ZeroAttemptLimit => write!(f, "max_attempts must be positive"),
            Self::AttemptsExhausted { vertex_count, degree, seed, attempts } => write!(
                f,
                "no simple d-regular graph found in {attempts} attempts (n={vertex_count}, d={degree}, seed={seed})"
            ),
            Self::InvariantViolation { reason } => write!(f, "regular graph invariant failed: {reason}"),
        }
    }
}

impl std::error::Error for RandomRegularMisError {}

/// Generate a seeded simple d-regular graph by rejection sampling of stub pairings.
///
/// Preconditions fail closed: vertex_count must be positive, degree < vertex_count,
/// n*d must not overflow and must be even, and max_attempts must be positive. Every
/// vertex contributes degree stubs; a shuffled pairing is accepted only if it has no
/// self-loops and no repeated undirected edges. Edge records are sorted and canonicalized
/// before return.
///
/// The PRNG is deterministic and non-cryptographic. This gives stable replay, but
/// generator distribution and source-family generalization must be validated before
/// using the cohort for confirmatory claims. For higher degrees, rejection rates can
/// grow sharply; an exhausted attempt budget returns an error rather than returning
/// a malformed or degree-relaxed graph.
pub fn generate_random_regular_mis_graph(
    vertex_count: usize,
    degree: usize,
    seed: u64,
    max_attempts: usize,
) -> Result<RandomRegularMisGraph, RandomRegularMisError> {
    if vertex_count == 0 {
        return Err(RandomRegularMisError::EmptyGraph);
    }
    if degree >= vertex_count {
        return Err(RandomRegularMisError::DegreeNotSmallerThanVertexCount {
            vertex_count,
            degree,
        });
    }
    let stub_count = vertex_count
        .checked_mul(degree)
        .ok_or(RandomRegularMisError::StubCountOverflow {
            vertex_count,
            degree,
        })?;
    if stub_count % 2 != 0 {
        return Err(RandomRegularMisError::OddStubCount {
            vertex_count,
            degree,
        });
    }
    if max_attempts == 0 {
        return Err(RandomRegularMisError::ZeroAttemptLimit);
    }

    let mut rng = XorShift64::new(seed);
    let mut stubs = Vec::with_capacity(stub_count);
    for vertex in 0..vertex_count {
        for _ in 0..degree {
            stubs.push(vertex);
        }
    }

    for attempt in 1..=max_attempts {
        // Fisher-Yates: rejection-sampled bounded integers avoid modulo bias.
        for i in (1..stubs.len()).rev() {
            let j = uniform_below(&mut rng, i + 1);
            stubs.swap(i, j);
        }

        let mut edges = Vec::with_capacity(stub_count / 2);
        let mut has_self_loop = false;
        for pair in stubs.chunks_exact(2) {
            let (a, b) = (pair[0], pair[1]);
            if a == b {
                has_self_loop = true;
                break;
            }
            edges.push(if a < b { (a, b) } else { (b, a) });
        }
        if has_self_loop {
            continue;
        }

        edges.sort_unstable();
        if edges.windows(2).any(|pair| pair[0] == pair[1]) {
            continue;
        }

        let graph = RandomRegularMisGraph {
            vertex_count,
            degree,
            seed,
            attempts_used: attempt,
            edges,
        };
        graph.validate()?;
        return Ok(graph);
    }

    Err(RandomRegularMisError::AttemptsExhausted {
        vertex_count,
        degree,
        seed,
        attempts: max_attempts,
    })
}

/// Return a bounded integer under the source PRNG's word stream with modulo bias removed.
fn uniform_below(rng: &mut XorShift64, upper: usize) -> usize {
    debug_assert!(upper > 0);
    let upper = upper as u64;
    let threshold = upper.wrapping_neg() % upper;
    loop {
        let value = rng.next_u64();
        if value >= threshold {
            return (value % upper) as usize;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_graph_is_simple_canonical_and_regular() {
        let graph = generate_random_regular_mis_graph(64, 3, 0xC0FFEE, 20_000).unwrap();
        graph.validate().unwrap();
        assert_eq!(graph.edges().len(), 64 * 3 / 2);
        assert!(graph.degree_sequence().iter().all(|&degree| degree == 3));
        assert!(graph.edges().windows(2).all(|pair| pair[0] < pair[1]));
    }

    #[test]
    fn deterministic_seed_replays_identical_graph_and_metadata() {
        let a = generate_random_regular_mis_graph(128, 4, 0x5EED, 100_000).unwrap();
        let b = generate_random_regular_mis_graph(128, 4, 0x5EED, 100_000).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.to_dimacs(), b.to_dimacs());
    }

    #[test]
    fn dimacs_output_is_one_based_sorted_and_records_provenance() {
        let graph = generate_random_regular_mis_graph(8, 2, 7, 10_000).unwrap();
        let text = graph.to_dimacs();
        assert!(text.starts_with("c generated_by=configuration-pair-reject-v1 seed=7 attempts="));
        assert!(text.contains("\np edge 8 8\n"));
        assert!(text.lines().skip(2).all(|line| line.starts_with("e ")));
        assert!(text.lines().skip(2).all(|line| {
            let tokens: Vec<_> = line.split_whitespace().collect();
            tokens[1].parse::<usize>().unwrap() < tokens[2].parse::<usize>().unwrap()
        }));
    }

    #[test]
    fn degree_zero_produces_an_edgeless_regular_graph() {
        let graph = generate_random_regular_mis_graph(12, 0, 99, 1).unwrap();
        assert!(graph.edges().is_empty());
        assert_eq!(graph.degree_sequence(), vec![0; 12]);
        assert_eq!(graph.attempts_used(), 1);
    }

    #[test]
    fn rejects_invalid_graph_and_attempt_configurations() {
        assert_eq!(
            generate_random_regular_mis_graph(0, 0, 1, 10),
            Err(RandomRegularMisError::EmptyGraph)
        );
        assert_eq!(
            generate_random_regular_mis_graph(4, 4, 1, 10),
            Err(RandomRegularMisError::DegreeNotSmallerThanVertexCount { vertex_count: 4, degree: 4 })
        );
        assert_eq!(
            generate_random_regular_mis_graph(5, 3, 1, 10),
            Err(RandomRegularMisError::OddStubCount { vertex_count: 5, degree: 3 })
        );
        assert_eq!(
            generate_random_regular_mis_graph(8, 3, 1, 0),
            Err(RandomRegularMisError::ZeroAttemptLimit)
        );
    }

    #[test]
    fn distinct_seeds_can_produce_distinct_graphs_and_every_graph_is_regular() {
        let mut seen = std::collections::BTreeSet::new();
        for seed in 1..=16u64 {
            let graph = generate_random_regular_mis_graph(30, 3, seed, 20_000).unwrap();
            graph.validate().unwrap();
            assert!(graph.degree_sequence().iter().all(|&degree| degree == 3));
            seen.insert(graph.edges().to_vec());
        }
        assert!(seen.len() > 1, "different seeds should not collapse to one graph");
    }

    #[test]
    fn small_degree_three_graph_sample_has_expected_triangle_class_frequency() {
        // There are exactly 70 labelled simple 3-regular graphs on six vertices:
        // 10 labelled K3,3 graphs (no triangles) and 60 labelled triangular prisms
        // (with triangles). Uniform configuration-model pairing conditioned on
        // simplicity therefore gives P(K3,3)=1/7. This is a coarse deterministic
        // regression check for gross sampling bias, not a distribution proof.
        fn has_triangle(graph: &RandomRegularMisGraph) -> bool {
            let mut adjacent = vec![vec![false; graph.vertex_count()]; graph.vertex_count()];
            for &(u, v) in graph.edges() {
                adjacent[u][v] = true;
                adjacent[v][u] = true;
            }
            for u in 0..graph.vertex_count() {
                for v in (u + 1)..graph.vertex_count() {
                    if !adjacent[u][v] {
                        continue;
                    }
                    for w in (v + 1)..graph.vertex_count() {
                        if adjacent[u][w] && adjacent[v][w] {
                            return true;
                        }
                    }
                }
            }
            false
        }

        const SAMPLES: usize = 5_000;
        let mut k33_count = 0usize;
        for seed in 1..=(SAMPLES as u64) {
            let graph = generate_random_regular_mis_graph(6, 3, seed, 20_000)
                .expect("small simple regular graph should be generated");
            graph.validate().expect("sampled graph remains simple and regular");
            if !has_triangle(&graph) {
                k33_count += 1;
            }
        }

        let observed = k33_count as f64 / SAMPLES as f64;
        let expected = 1.0 / 7.0;
        assert!(
            (observed - expected).abs() < 0.04,
            "K3,3 sample fraction {observed:.4} too far from reference {expected:.4}"
        );
    }

    #[test]
    fn exhausted_attempt_budget_fails_closed_without_changing_seed() {
        // For this generator version, the first pairing for n=6,d=3,seed=1
        // contains a self-loop. With one allowed attempt the function must fail
        // rather than silently switch seeds or return a non-regular graph.
        assert_eq!(
            generate_random_regular_mis_graph(6, 3, 1, 1),
            Err(RandomRegularMisError::AttemptsExhausted {
                vertex_count: 6,
                degree: 3,
                seed: 1,
                attempts: 1,
            })
        );
    }

}
