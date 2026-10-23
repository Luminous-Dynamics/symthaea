//! Deterministic and seeded-randomized MIS heuristic portfolio.
//!
//! This module is a classical control track for testing whether a more expensive
//! informer (including a quantum informer) adds value beyond strong cheap baselines.
//! It is not a state-of-the-art MIS solver. Every candidate is independently verified,
//! selection depends on verified cardinality rather than timing, and randomized seeds,
//! tie-breaking rules and local-search budgets are part of the explicit configuration.
//!
//! The optional bounded 1-for-2 neighborhood move replaces one selected vertex with
//! two unselected nonadjacent vertices whose only selected neighbor is the removed
//! vertex. Each accepted move is followed by a deterministic maximality augmentation.
//! A hard pair-check budget prevents this improvement pass from growing quadratically
//! without limit on dense graphs.

use core::fmt;
use std::time::Instant;

use crate::mis_verifier::{MisGraph, MisViolation};
use crate::rng::XorShift64;

/// Maximum supported randomized restarts, keeping the portfolio budget explicit.
pub const MAX_MIS_RANDOMIZED_RESTARTS: usize = 256;

/// Maximum accepted one-for-two neighborhood moves.
pub const MAX_MIS_LOCAL_SEARCH_ROUNDS: usize = 1_024;

/// Greedy construction policy evaluated by the portfolio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MisGreedyPolicy {
    /// Select a minimum residual-degree active vertex, breaking ties by lower index.
    MinimumResidualDegree,
    /// Select a maximum residual-degree active vertex, breaking ties by lower index.
    MaximumResidualDegree,
    /// Select a minimum static-degree active vertex, breaking ties by lower index.
    MinimumStaticDegree,
    /// Select a minimum residual-degree vertex, uniformly tie-broken by a seeded PRNG.
    RandomizedMinimumResidualDegree,
}

impl MisGreedyPolicy {
    /// Stable policy label for machine-readable run receipts.
    pub const fn label(self) -> &'static str {
        match self {
            Self::MinimumResidualDegree => "greedy_min_residual_degree_low_index",
            Self::MaximumResidualDegree => "greedy_max_residual_degree_low_index",
            Self::MinimumStaticDegree => "greedy_min_static_degree_low_index",
            Self::RandomizedMinimumResidualDegree => "greedy_min_residual_degree_random_ties",
        }
    }
}

/// Explicit run budget for a classical MIS portfolio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MisPortfolioConfig {
    /// Number of seeded randomized minimum-residual-degree restarts, 0..=256.
    pub randomized_restarts: usize,
    /// Base seed; each restart derives its seed deterministically from this value.
    pub seed: u64,
    /// Apply the bounded 1-for-2 local-search pass to each constructed candidate.
    pub enable_one_for_two: bool,
    /// Maximum accepted 1-for-2 moves per candidate, 0..=1024.
    pub max_local_search_rounds: usize,
    /// Maximum nonadjacent-pair checks per local-search round.
    pub max_pair_checks_per_round: usize,
}

impl Default for MisPortfolioConfig {
    fn default() -> Self {
        Self {
            randomized_restarts: 16,
            seed: 0x4d49_5350_4f52_5431,
            enable_one_for_two: true,
            max_local_search_rounds: 32,
            max_pair_checks_per_round: 100_000,
        }
    }
}

/// Configuration or invariant error raised by a portfolio run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MisPortfolioError {
    /// Too many randomized restarts were requested.
    TooManyRandomizedRestarts { requested: usize, maximum: usize },
    /// Too many local-search rounds were requested.
    TooManyLocalSearchRounds { requested: usize, maximum: usize },
    /// Pair-check limit must be positive when local search is enabled.
    ZeroPairCheckLimit,
    /// A generated solution failed independent feasibility verification.
    GeneratedSolutionInvalid {
        /// Policy that generated the rejected candidate.
        policy_id: String,
        /// Violations reported by the independent graph verifier.
        violations: Vec<MisViolation>,
    },
}

impl fmt::Display for MisPortfolioError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooManyRandomizedRestarts { requested, maximum } => {
                write!(f, "requested {requested} randomized restarts, maximum is {maximum}")
            }
            Self::TooManyLocalSearchRounds { requested, maximum } => {
                write!(f, "requested {requested} local-search rounds, maximum is {maximum}")
            }
            Self::ZeroPairCheckLimit => {
                write!(f, "pair-check limit must be positive when one-for-two search is enabled")
            }
            Self::GeneratedSolutionInvalid { policy_id, violations } => {
                write!(f, "policy {policy_id:?} generated invalid MIS candidate: {violations:?}")
            }
        }
    }
}

impl std::error::Error for MisPortfolioError {}

/// One evaluated heuristic arm with its own resource and outcome record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisPortfolioArm {
    /// Stable policy label, including restart number when applicable.
    pub policy_id: String,
    /// Seed used for a randomized arm, otherwise None.
    pub seed: Option<u64>,
    /// Independently verified cardinality of the selected set.
    pub objective: usize,
    /// Result of the independent graph verifier.
    pub feasible: bool,
    /// Number of accepted one-for-two neighborhood moves.
    pub local_improvements: usize,
    /// Number of candidate vertex pairs examined by local search.
    pub pair_checks: usize,
    /// Returned vertex indices, zero-based and sorted.
    pub vertices: Vec<usize>,
    /// Local diagnostic wall time in nanoseconds. Do not compare across machines.
    pub elapsed_nanos: u128,
}

/// Summary of the selected candidate and all evaluated arms.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MisPortfolioReport {
    /// Index into arms of the selected result; ties preserve the earlier registered arm.
    pub selected_arm_index: usize,
    /// Selected policy identifier.
    pub selected_policy_id: String,
    /// Selected verified solution cardinality.
    pub selected_objective: usize,
    /// Selected solution passes independent feasibility verification.
    pub feasible: bool,
    /// Selected zero-based vertices in sorted order.
    pub selected_vertices: Vec<usize>,
    /// Every evaluated policy/restart, including its outcome and local cost metadata.
    pub arms: Vec<MisPortfolioArm>,
    /// Total elapsed wall time for portfolio orchestration and all arm execution.
    pub total_elapsed_nanos: u128,
}

fn adjacency_lists(graph: &MisGraph) -> Vec<Vec<usize>> {
    let mut adjacency = vec![Vec::new(); graph.vertex_count()];
    for &(u, v) in graph.edges() {
        adjacency[u].push(v);
        adjacency[v].push(u);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
    }
    adjacency
}

fn greedy_candidate(
    graph: &MisGraph,
    policy: MisGreedyPolicy,
    seed: Option<u64>,
) -> Vec<usize> {
    let n = graph.vertex_count();
    let adjacency = adjacency_lists(graph);
    let static_degrees: Vec<usize> = adjacency.iter().map(Vec::len).collect();
    let mut active = vec![true; n];
    let mut selected = Vec::new();
    let mut rng = seed.map(XorShift64::new);

    loop {
        let mut best_score: Option<usize> = None;
        let mut tied = Vec::new();

        for vertex in 0..n {
            if !active[vertex] {
                continue;
            }
            let score = match policy {
                MisGreedyPolicy::MinimumStaticDegree => static_degrees[vertex],
                MisGreedyPolicy::MinimumResidualDegree
                | MisGreedyPolicy::RandomizedMinimumResidualDegree
                | MisGreedyPolicy::MaximumResidualDegree => adjacency[vertex]
                    .iter()
                    .filter(|&&neighbor| active[neighbor])
                    .count(),
            };
            let normalized_score = if policy == MisGreedyPolicy::MaximumResidualDegree {
                usize::MAX - score
            } else {
                score
            };

            match best_score {
                None => {
                    best_score = Some(normalized_score);
                    tied.push(vertex);
                }
                Some(best) if normalized_score < best => {
                    best_score = Some(normalized_score);
                    tied.clear();
                    tied.push(vertex);
                }
                Some(best) if normalized_score == best => tied.push(vertex),
                Some(_) => {}
            }
        }

        let Some(_) = best_score else { break };
        let vertex = if policy == MisGreedyPolicy::RandomizedMinimumResidualDegree {
            let index = rng
                .as_mut()
                .and_then(|source| source.next_usize(tied.len()))
                .unwrap_or(0);
            tied[index]
        } else {
            // Vertices were visited in ascending order, so the first tied value is the
            // stable low-index tie-break for all deterministic arms.
            tied[0]
        };

        active[vertex] = false;
        selected.push(vertex);
        for &neighbor in &adjacency[vertex] {
            active[neighbor] = false;
        }
    }
    selected.sort_unstable();
    selected
}

/// Adds any currently feasible vertices in ascending-index order until maximal.
fn augment_to_maximal(graph: &MisGraph, selected: &mut Vec<usize>, adjacency: &[Vec<usize>]) {
    let mut chosen = vec![false; graph.vertex_count()];
    for &vertex in selected.iter() {
        chosen[vertex] = true;
    }
    for vertex in 0..graph.vertex_count() {
        if !chosen[vertex] && adjacency[vertex].iter().all(|&other| !chosen[other]) {
            chosen[vertex] = true;
        }
    }
    *selected = chosen
        .iter()
        .enumerate()
        .filter_map(|(vertex, &is_chosen)| is_chosen.then_some(vertex))
        .collect();
}

/// Apply bounded 1-for-2 improving moves while preserving MIS feasibility.
fn improve_one_for_two(
    graph: &MisGraph,
    selected: &mut Vec<usize>,
    max_rounds: usize,
    max_pair_checks_per_round: usize,
) -> (usize, usize) {
    if max_rounds == 0 || graph.vertex_count() < 3 {
        return (0, 0);
    }
    let adjacency = adjacency_lists(graph);
    let mut accepted_moves = 0;
    let mut total_pair_checks = 0;

    for _ in 0..max_rounds {
        let mut is_selected = vec![false; graph.vertex_count()];
        for &vertex in selected.iter() {
            is_selected[vertex] = true;
        }

        let mut selected_neighbor_count = vec![0usize; graph.vertex_count()];
        let mut sole_selected_neighbor = vec![None; graph.vertex_count()];
        for &(u, v) in graph.edges() {
            match (is_selected[u], is_selected[v]) {
                (true, false) => {
                    selected_neighbor_count[v] += 1;
                    sole_selected_neighbor[v] = Some(u);
                }
                (false, true) => {
                    selected_neighbor_count[u] += 1;
                    sole_selected_neighbor[u] = Some(v);
                }
                _ => {}
            }
        }

        let mut replacement: Option<(usize, usize, usize)> = None;
        let mut pair_checks_this_round = 0usize;
        'selected_vertex: for &remove in selected.iter() {
            let candidates: Vec<usize> = (0..graph.vertex_count())
                .filter(|&vertex| {
                    !is_selected[vertex]
                        && selected_neighbor_count[vertex] == 1
                        && sole_selected_neighbor[vertex] == Some(remove)
                })
                .collect();

            for left_index in 0..candidates.len() {
                let left = candidates[left_index];
                for &right in candidates.iter().skip(left_index + 1) {
                    if pair_checks_this_round >= max_pair_checks_per_round {
                        break 'selected_vertex;
                    }
                    pair_checks_this_round += 1;
                    if adjacency[left].binary_search(&right).is_err() {
                        replacement = Some((remove, left, right));
                        break 'selected_vertex;
                    }
                }
            }
        }
        total_pair_checks = total_pair_checks.saturating_add(pair_checks_this_round);

        let Some((remove, left, right)) = replacement else {
            break;
        };
        selected.retain(|&vertex| vertex != remove);
        selected.push(left);
        selected.push(right);
        selected.sort_unstable();
        augment_to_maximal(graph, selected, &adjacency);
        accepted_moves += 1;
    }
    (accepted_moves, total_pair_checks)
}

/// Evaluate cheap deterministic controls plus bounded randomized restarts and return
/// the best independently verified solution under the declared portfolio budget.
///
/// Candidate ranking is by verified objective only. Equal objectives preserve arm
/// registration order, so noisy wall-clock timing never changes the selected result.
pub fn run_mis_portfolio(
    graph: &MisGraph,
    config: MisPortfolioConfig,
) -> Result<MisPortfolioReport, MisPortfolioError> {
    if config.randomized_restarts > MAX_MIS_RANDOMIZED_RESTARTS {
        return Err(MisPortfolioError::TooManyRandomizedRestarts {
            requested: config.randomized_restarts,
            maximum: MAX_MIS_RANDOMIZED_RESTARTS,
        });
    }
    if config.max_local_search_rounds > MAX_MIS_LOCAL_SEARCH_ROUNDS {
        return Err(MisPortfolioError::TooManyLocalSearchRounds {
            requested: config.max_local_search_rounds,
            maximum: MAX_MIS_LOCAL_SEARCH_ROUNDS,
        });
    }
    if config.enable_one_for_two && config.max_pair_checks_per_round == 0 {
        return Err(MisPortfolioError::ZeroPairCheckLimit);
    }

    let portfolio_start = Instant::now();
    let mut registered: Vec<(String, MisGreedyPolicy, Option<u64>)> = vec![
        (
            MisGreedyPolicy::MinimumResidualDegree.label().to_owned(),
            MisGreedyPolicy::MinimumResidualDegree,
            None,
        ),
        (
            MisGreedyPolicy::MaximumResidualDegree.label().to_owned(),
            MisGreedyPolicy::MaximumResidualDegree,
            None,
        ),
        (
            MisGreedyPolicy::MinimumStaticDegree.label().to_owned(),
            MisGreedyPolicy::MinimumStaticDegree,
            None,
        ),
    ];
    for restart in 0..config.randomized_restarts {
        let seed = config.seed.wrapping_add(
            (restart as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        );
        registered.push((
            format!("{}_restart_{restart:03}", MisGreedyPolicy::RandomizedMinimumResidualDegree.label()),
            MisGreedyPolicy::RandomizedMinimumResidualDegree,
            Some(seed),
        ));
    }

    let mut arms = Vec::with_capacity(registered.len());
    for (policy_id, policy, seed) in registered {
        let arm_start = Instant::now();
        let mut vertices = greedy_candidate(graph, policy, seed);
        let (local_improvements, pair_checks) = if config.enable_one_for_two {
            improve_one_for_two(
                graph,
                &mut vertices,
                config.max_local_search_rounds,
                config.max_pair_checks_per_round,
            )
        } else {
            (0, 0)
        };
        vertices.sort_unstable();
        let verification = graph.verify_solution(&vertices, Some(vertices.len()));
        if !verification.feasible {
            return Err(MisPortfolioError::GeneratedSolutionInvalid {
                policy_id,
                violations: verification.violations,
            });
        }
        arms.push(MisPortfolioArm {
            policy_id,
            seed,
            objective: verification.computed_objective,
            feasible: verification.feasible,
            local_improvements,
            pair_checks,
            vertices,
            elapsed_nanos: arm_start.elapsed().as_nanos(),
        });
    }

    // Strict > preserves registration order among equal objective values.
    let mut selected_arm_index = 0usize;
    for index in 1..arms.len() {
        if arms[index].objective > arms[selected_arm_index].objective {
            selected_arm_index = index;
        }
    }
    let selected = &arms[selected_arm_index];
    Ok(MisPortfolioReport {
        selected_arm_index,
        selected_policy_id: selected.policy_id.clone(),
        selected_objective: selected.objective,
        feasible: selected.feasible,
        selected_vertices: selected.vertices.clone(),
        arms,
        total_elapsed_nanos: portfolio_start.elapsed().as_nanos(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn portfolio_is_feasible_and_matches_tiny_exact_oracle_on_path() {
        let graph = MisGraph::new(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]).unwrap();
        let report = run_mis_portfolio(&graph, MisPortfolioConfig::default()).unwrap();
        let exact = graph.exact_maximum_independent_set().unwrap();
        assert_eq!(report.selected_objective, exact.len());
        assert!(report.feasible);
        assert!(report.arms.iter().all(|arm| arm.feasible));
        assert_eq!(report.arms.len(), 19);
    }

    #[test]
    fn bounded_local_swap_recovers_all_leaves_of_a_star() {
        let graph = MisGraph::new(4, [(0, 1), (0, 2), (0, 3)]).unwrap();
        let report = run_mis_portfolio(
            &graph,
            MisPortfolioConfig {
                randomized_restarts: 0,
                seed: 7,
                enable_one_for_two: true,
                max_local_search_rounds: 4,
                max_pair_checks_per_round: 100,
            },
        )
        .unwrap();
        let max_degree = report.arms.iter()
            .find(|arm| arm.policy_id == MisGreedyPolicy::MaximumResidualDegree.label())
            .unwrap();
        assert_eq!(max_degree.objective, 3);
        assert_eq!(max_degree.local_improvements, 1);
        assert!(max_degree.feasible);
        assert_eq!(report.selected_objective, 3);
    }

    #[test]
    fn selection_is_reproducible_and_independent_of_wall_clock_timing() {
        let graph = MisGraph::new(7, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)])
            .unwrap();
        let config = MisPortfolioConfig { randomized_restarts: 5, ..MisPortfolioConfig::default() };
        let left = run_mis_portfolio(&graph, config).unwrap();
        let right = run_mis_portfolio(&graph, config).unwrap();
        assert_eq!(left.selected_policy_id, right.selected_policy_id);
        assert_eq!(left.selected_objective, right.selected_objective);
        assert_eq!(left.selected_vertices, right.selected_vertices);
        assert_eq!(left.arms.iter().map(|a| (&a.policy_id, a.seed, a.objective, &a.vertices))
            .collect::<Vec<_>>(),
            right.arms.iter().map(|a| (&a.policy_id, a.seed, a.objective, &a.vertices))
            .collect::<Vec<_>>());
    }

    #[test]
    fn rejects_unbounded_portfolio_configuration() {
        let graph = MisGraph::new(0, []).unwrap();
        assert_eq!(
            run_mis_portfolio(&graph, MisPortfolioConfig {
                randomized_restarts: MAX_MIS_RANDOMIZED_RESTARTS + 1,
                ..MisPortfolioConfig::default()
            }).unwrap_err(),
            MisPortfolioError::TooManyRandomizedRestarts {
                requested: MAX_MIS_RANDOMIZED_RESTARTS + 1,
                maximum: MAX_MIS_RANDOMIZED_RESTARTS
            }
        );
        assert_eq!(
            run_mis_portfolio(&graph, MisPortfolioConfig {
                enable_one_for_two: true,
                max_pair_checks_per_round: 0,
                ..MisPortfolioConfig::default()
            }).unwrap_err(),
            MisPortfolioError::ZeroPairCheckLimit
        );
    }
}
