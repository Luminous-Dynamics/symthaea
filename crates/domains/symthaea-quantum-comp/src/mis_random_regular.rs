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

/// Maximum vertex count accepted by the seeded regular-MIS generator.
pub const MAX_MIS_RANDOM_REGULAR_VERTICES: usize = 100_000;
/// Maximum number of stubs accepted before allocating the generator buffers.
/// The current paper-shaped cohort only needs 15,000 stubs at N=5,000, degree 3.
pub const MAX_MIS_RANDOM_REGULAR_STUBS: usize = 1_000_000;

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
        Self { vertex_count, degree, seed, max_attempts: 100_000 }
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
    /// Vertex count exceeds the explicit per-call allocation ceiling.
    VertexCountLimitExceeded { actual: usize, maximum: usize },
    /// Stub storage exceeds the explicit per-call allocation ceiling.
    StubCountLimitExceeded { actual: usize, maximum: usize },
    /// A regular graph requires an even vertex-count times degree.
    OddStubCount { vertex_count: usize, degree: usize },
    /// Vertex count times degree overflowed usize arithmetic.
    StubCountOverflow { vertex_count: usize, degree: usize },
    /// No attempts were allowed.
    ZeroRetryBudget,
    /// No simple pairing was found within the configured budget.
    RetryBudgetExhausted {
        /// Number of vertices in the requested graph.
        vertex_count: usize,
        /// Requested regular degree.
        degree: usize,
        /// Seed defining the deterministic random stream.
        seed: u64,
        /// Number of complete pairings attempted.
        max_attempts: usize,
    },
    /// The independently validating graph constructor rejected the generated edges.
    GraphConstruction(MisGraphError),
}

impl fmt::Display for MisRandomRegularError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroVertices => write!(f, "vertex_count must be positive"),
            Self::VertexCountLimitExceeded { actual, maximum } => write!(f, "vertex_count {actual} exceeds allocation ceiling {maximum}"),
            Self::StubCountLimitExceeded { actual, maximum } => write!(f, "stub count {actual} exceeds allocation ceiling {maximum}"),
            Self::DegreeNotLessThanVertices { vertex_count, degree } =>
                write!(f, "degree {degree} must be less than vertex_count {vertex_count}"),
            Self::OddStubCount { vertex_count, degree } =>
                write!(f, "vertex_count * degree must be even, got {vertex_count} * {degree}"),
            Self::StubCountOverflow { vertex_count, degree } =>
                write!(f, "vertex_count * degree overflows usize: {vertex_count} * {degree}"),
            Self::ZeroRetryBudget => write!(f, "max_attempts must be positive"),
            Self::RetryBudgetExhausted { vertex_count, degree, seed, max_attempts } => write!(
                f,
                "no simple pairing found in {max_attempts} attempts (n={vertex_count}, degree={degree}, seed={seed})"
            ),
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
    if config.vertex_count > MAX_MIS_RANDOM_REGULAR_VERTICES {
        return Err(MisRandomRegularError::VertexCountLimitExceeded {
            actual: config.vertex_count,
            maximum: MAX_MIS_RANDOM_REGULAR_VERTICES,
        });
    }
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
    if stub_count > MAX_MIS_RANDOM_REGULAR_STUBS {
        return Err(MisRandomRegularError::StubCountLimitExceeded {
            actual: stub_count,
            maximum: MAX_MIS_RANDOM_REGULAR_STUBS,
        });
    }
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

    let mut canonical_stubs = Vec::with_capacity(stub_count);
    for vertex in 0..config.vertex_count {
        for _ in 0..config.degree { canonical_stubs.push(vertex); }
    }
    let mut rng = XorShift64::new(config.seed);
    for attempt in 1..=config.max_attempts {
        // Every retry starts from the same canonical stub multiset, then consumes the
        // next segment of the seeded PRNG stream for a fresh Fisher-Yates permutation.
        // This is the seed-stream contract used by QCOMP-RRG-CLASSICAL-PREFLIGHT-001.
        // Re-shuffling a failed permutation is distributionally reasonable but does
        // not produce identical graph bytes for the same cross-language seed fixture.
        let mut stubs = canonical_stubs.clone();
        for index in (1..stubs.len()).rev() {
            let swap_with = sample_below(&mut rng, index + 1);
            stubs.swap(index, swap_with);
        }
        let mut edges = Vec::with_capacity(stub_count / 2);
        let mut seen = HashSet::with_capacity(stub_count / 2);
        let mut simple = true;
        for pair in stubs.chunks_exact(2) {
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
        vertex_count: config.vertex_count,
        degree: config.degree,
        seed: config.seed,
        max_attempts: config.max_attempts,
    })
}

/// Unbiased bounded conversion from one u64 draw, using rejection of the final
/// incomplete residue bucket.
fn sample_below(rng: &mut XorShift64, upper: usize) -> usize {
    debug_assert!(upper > 0);
    let bound = upper as u64;
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let value = rng.next_u64();
        if value >= threshold { return (value % bound) as usize; }
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

    fn canonical_graph_fnv1a64(graph: &MisGraph) -> String {
        let mut canonical =
            format!("p edge {} {}\\n", graph.vertex_count(), graph.edges().len()).into_bytes();
        for &(u, v) in graph.edges() {
            canonical.extend_from_slice(format!("e {} {}\\n", u + 1, v + 1).as_bytes());
        }
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for byte in canonical {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{hash:016x}")
    }

    #[test]
    fn seeded_graphs_match_independent_javascript_reference_for_all_70_frozen_instances() {
        // Generator-seed/acceptance/hash vectors extracted from
        // QCOMP-RRG-CLASSICAL-PREFLIGHT-001.json at blob
        // 3a4dc72106e8de2f73afe3af33a66cee2beee70e.
        // FNV-1a is only a compact regression fingerprint, not cryptographic provenance.
        const FIXTURES: &[(&str, usize, usize, u64, usize, &str)] = &[
            ("rrg3-n50-i001", 50, 3, 202660230150u64, 5, "5ef6d3ec4c3dea16"),
            ("rrg3-n50-i002", 50, 3, 202660238069u64, 1, "7e5760f553fcb36c"),
            ("rrg3-n50-i003", 50, 3, 202660245988u64, 10, "dff5e7f1bd974004"),
            ("rrg3-n50-i004", 50, 3, 202660253907u64, 5, "26497251480cc382"),
            ("rrg3-n50-i005", 50, 3, 202660261826u64, 1, "092cec71f6bb0578"),
            ("rrg3-n50-i006", 50, 3, 202660269745u64, 3, "fc1ab7cb2cf32812"),
            ("rrg3-n50-i007", 50, 3, 202660277664u64, 20, "061184f83bba4a6a"),
            ("rrg3-n50-i008", 50, 3, 202660285583u64, 4, "39899551ca8f101c"),
            ("rrg3-n50-i009", 50, 3, 202660293502u64, 1, "5792e165a52cdc84"),
            ("rrg3-n50-i010", 50, 3, 202660301421u64, 4, "c5f9835001d1e706"),
            ("rrg3-n50-i011", 50, 3, 202660309340u64, 19, "7dc9b7ed1601d52e"),
            ("rrg3-n50-i012", 50, 3, 202660317259u64, 1, "824d48d7f9a7b090"),
            ("rrg3-n50-i013", 50, 3, 202660325178u64, 3, "e13c30a7bb4bdee4"),
            ("rrg3-n50-i014", 50, 3, 202660333097u64, 18, "e5e29d385b3bd2e4"),
            ("rrg3-n50-i015", 50, 3, 202660341016u64, 6, "4326f2889ee970ac"),
            ("rrg3-n50-i016", 50, 3, 202660348935u64, 6, "36ef37adc663370e"),
            ("rrg3-n50-i017", 50, 3, 202660356854u64, 4, "9da24068efc329de"),
            ("rrg3-n50-i018", 50, 3, 202660364773u64, 2, "eabae4c1607240ec"),
            ("rrg3-n50-i019", 50, 3, 202660372692u64, 2, "66bd22140b37abb4"),
            ("rrg3-n50-i020", 50, 3, 202660380611u64, 2, "54e11882a387aabc"),
            ("rrg3-n100-i001", 100, 3, 202710230300u64, 3, "e2c3805240071e8c"),
            ("rrg3-n100-i002", 100, 3, 202710238219u64, 6, "76a75279fb6704d6"),
            ("rrg3-n100-i003", 100, 3, 202710246138u64, 5, "3ac293112617e4e4"),
            ("rrg3-n100-i004", 100, 3, 202710254057u64, 13, "830d8caed70ab77c"),
            ("rrg3-n100-i005", 100, 3, 202710261976u64, 15, "642d4f7f97dc2564"),
            ("rrg3-n100-i006", 100, 3, 202710269895u64, 1, "850602fa5799f7aa"),
            ("rrg3-n100-i007", 100, 3, 202710277814u64, 1, "bb8d05c98215fcf8"),
            ("rrg3-n100-i008", 100, 3, 202710285733u64, 2, "68808e7506fb19b2"),
            ("rrg3-n100-i009", 100, 3, 202710293652u64, 1, "c955863ba7591e50"),
            ("rrg3-n100-i010", 100, 3, 202710301571u64, 13, "0529a7db5be34280"),
            ("rrg3-n100-i011", 100, 3, 202710309490u64, 4, "7c3e323805fb82ae"),
            ("rrg3-n100-i012", 100, 3, 202710317409u64, 1, "e718aa1a737e462e"),
            ("rrg3-n100-i013", 100, 3, 202710325328u64, 8, "34eb5c3fe02e5e38"),
            ("rrg3-n100-i014", 100, 3, 202710333247u64, 9, "c9e56ac1eb4f462a"),
            ("rrg3-n100-i015", 100, 3, 202710341166u64, 15, "8d577a255d15f8c4"),
            ("rrg3-n100-i016", 100, 3, 202710349085u64, 2, "707812d99d31fd94"),
            ("rrg3-n100-i017", 100, 3, 202710357004u64, 7, "30504114f5aa4040"),
            ("rrg3-n100-i018", 100, 3, 202710364923u64, 2, "b5f18de96d6eca10"),
            ("rrg3-n100-i019", 100, 3, 202710372842u64, 8, "71135f1529438b64"),
            ("rrg3-n100-i020", 100, 3, 202710380761u64, 4, "346451aafc7c8bce"),
            ("rrg3-n200-i001", 200, 3, 202810230600u64, 2, "1fa8cd44ba27c3cf"),
            ("rrg3-n200-i002", 200, 3, 202810238519u64, 4, "bfe63e7001742b11"),
            ("rrg3-n200-i003", 200, 3, 202810246438u64, 4, "6372160663e48a1d"),
            ("rrg3-n200-i004", 200, 3, 202810254357u64, 6, "da113d590973d987"),
            ("rrg3-n200-i005", 200, 3, 202810262276u64, 5, "b7810970cf5116d9"),
            ("rrg3-n200-i006", 200, 3, 202810270195u64, 1, "a52a6e53b0dcc5df"),
            ("rrg3-n200-i007", 200, 3, 202810278114u64, 6, "8342f597572e2dbd"),
            ("rrg3-n200-i008", 200, 3, 202810286033u64, 9, "05231ea71610c2ff"),
            ("rrg3-n200-i009", 200, 3, 202810293952u64, 11, "79e08dc3bb8f3cd1"),
            ("rrg3-n200-i010", 200, 3, 202810301871u64, 17, "cc2f0448a8af5af7"),
            ("rrg3-n200-i011", 200, 3, 202810309790u64, 3, "c0d49ded2cc29b71"),
            ("rrg3-n200-i012", 200, 3, 202810317709u64, 1, "146cde0fc183aff7"),
            ("rrg3-n200-i013", 200, 3, 202810325628u64, 9, "e926d144a80aabfd"),
            ("rrg3-n200-i014", 200, 3, 202810333547u64, 14, "03284b8c55a18947"),
            ("rrg3-n200-i015", 200, 3, 202810341466u64, 12, "6d13dd5125b00897"),
            ("rrg3-n200-i016", 200, 3, 202810349385u64, 8, "36645235df5cea2f"),
            ("rrg3-n200-i017", 200, 3, 202810357304u64, 2, "c31f5a3d731611d7"),
            ("rrg3-n200-i018", 200, 3, 202810365223u64, 6, "f62c1a97edf3e02b"),
            ("rrg3-n200-i019", 200, 3, 202810373142u64, 28, "de34d0a9daa31b7d"),
            ("rrg3-n200-i020", 200, 3, 202810381061u64, 2, "17e0c9acc05597b5"),
            ("rrg3-n500-i001", 500, 3, 203110231500u64, 12, "8264a4b397287086"),
            ("rrg3-n500-i002", 500, 3, 203110239419u64, 4, "af4636788344d2ce"),
            ("rrg3-n500-i003", 500, 3, 203110247338u64, 8, "d1fce8e357c550e8"),
            ("rrg3-n500-i004", 500, 3, 203110255257u64, 33, "b09307afb9c20802"),
            ("rrg3-n500-i005", 500, 3, 203110263176u64, 2, "619f29382a46acd8"),
            ("rrg3-n500-i006", 500, 3, 203110271095u64, 4, "773a4d0cf29cf42e"),
            ("rrg3-n500-i007", 500, 3, 203110279014u64, 4, "9df71ff63770bf18"),
            ("rrg3-n500-i008", 500, 3, 203110286933u64, 12, "95d6188704a85eee"),
            ("rrg3-n500-i009", 500, 3, 203110294852u64, 7, "879dfb4ca13cab32"),
            ("rrg3-n500-i010", 500, 3, 203110302771u64, 13, "75fee262d2e1b8f4"),
        ];

        for &(id, vertex_count, degree, seed, expected_attempts, expected_hash) in FIXTURES {
            let config = MisRandomRegularConfig::new(vertex_count, degree, seed);
            let (graph, report) = generate_mis_random_regular(config)
                .unwrap_or_else(|error| panic!("{id}: generation failed: {error}"));
            assert_eq!(report.attempts, expected_attempts, "{id}: accepted attempt count drift");
            assert_eq!(report.edge_count, vertex_count * degree / 2, "{id}: edge-count drift");
            assert_eq!(graph.edges().len(), vertex_count * degree / 2, "{id}: wrong edge count");
            assert!(degrees(&graph).iter().all(|&actual| actual == degree), "{id}: degree drift");
            assert_eq!(
                canonical_graph_fnv1a64(&graph),
                expected_hash,
                "{id}: canonical DIMACS graph fingerprint drift for seed {seed}"
            );
        }
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
    fn rejects_vertex_and_stub_limits_before_allocating() {
        assert_eq!(
            generate_mis_random_regular(MisRandomRegularConfig::new(
                MAX_MIS_RANDOM_REGULAR_VERTICES + 1, 0, 7
            )),
            Err(MisRandomRegularError::VertexCountLimitExceeded {
                actual: MAX_MIS_RANDOM_REGULAR_VERTICES + 1,
                maximum: MAX_MIS_RANDOM_REGULAR_VERTICES,
            })
        );
        assert_eq!(
            generate_mis_random_regular(MisRandomRegularConfig::new(100_000, 12, 17)),
            Err(MisRandomRegularError::StubCountLimitExceeded {
                actual: 1_200_000,
                maximum: MAX_MIS_RANDOM_REGULAR_STUBS,
            })
        );
    }

    #[test]
    fn small_degree_three_graph_sample_has_expected_triangle_class_frequency() {
        // There are exactly 70 labelled simple 3-regular graphs on six vertices:
        // 10 labelled K3,3 graphs (no triangles) and 60 labelled triangular prisms
        // (with triangles). Uniform pairing conditioned on simplicity predicts P(K3,3)=1/7.
        fn has_triangle(graph: &MisGraph) -> bool {
            let mut adjacent = vec![vec![false; graph.vertex_count()]; graph.vertex_count()];
            for &(u, v) in graph.edges() {
                adjacent[u][v] = true;
                adjacent[v][u] = true;
            }
            for u in 0..graph.vertex_count() {
                for v in (u + 1)..graph.vertex_count() {
                    if !adjacent[u][v] { continue; }
                    for w in (v + 1)..graph.vertex_count() {
                        if adjacent[u][w] && adjacent[v][w] { return true; }
                    }
                }
            }
            false
        }

        const SAMPLES: usize = 5_000;
        let mut k33_count = 0usize;
        for seed in 1..=(SAMPLES as u64) {
            let (graph, _) = generate_mis_random_regular(
                MisRandomRegularConfig::new(6, 3, seed)
            ).expect("small simple cubic graph should be generated");
            if !has_triangle(&graph) { k33_count += 1; }
        }
        let observed = k33_count as f64 / SAMPLES as f64;
        let expected = 1.0 / 7.0;
        assert!(
            (observed - expected).abs() < 0.04,
            "K3,3 sample fraction {observed:.4} too far from reference {expected:.4}"
        );
    }

    #[test]
    fn retry_exhaustion_preserves_workload_identity() {
        // The one-attempt seed is selected to produce a non-simple first pairing.
        let config = MisRandomRegularConfig {
            max_attempts: 1,
            ..MisRandomRegularConfig::new(6, 3, 1)
        };
        assert_eq!(
            generate_mis_random_regular(config),
            Err(MisRandomRegularError::RetryBudgetExhausted {
                vertex_count: 6,
                degree: 3,
                seed: 1,
                max_attempts: 1,
            })
        );
    }

    #[test]
    fn degree_one_graph_is_a_perfect_matching() {
        let (graph, _) = generate_mis_random_regular(MisRandomRegularConfig::new(40, 1, 88)).unwrap();
        assert_eq!(graph.edges().len(), 20);
        assert!(degrees(&graph).iter().all(|&d| d == 1));
    }
}
