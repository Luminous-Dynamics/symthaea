// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Deterministic seeded bounded RRT over tri-state continuous validity oracles.
//!
//! This solver can produce only `Feasible` or `Unknown`. There is intentionally
//! no sampling-based `CertifiedInfeasible` variant: exhausting a finite sample
//! budget does not prove that a continuous path does not exist.

use blake3::Hasher;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use thiserror::Error;

use crate::continuous_reachability::{
    AnalyticBoxValidityOracle, ContinuousPathReplay, ContinuousPathValidationReceipt,
    ContinuousReachabilityError, ContinuousValidityOracle, EuclideanPlanningProblem,
    OracleVerdict, validate_euclidean_path,
};
use crate::reachability::UnknownReason;

/// Hard ceiling on V1 RRT iterations.
pub const MAX_RRT_ITERATIONS: u32 = 1_000_000;
/// Hard ceiling on V1 Euclidean sampling dimension.
pub const MAX_RRT_DIMENSION: usize = 64;

/// Exact deterministic RRT search profile.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RrtConfig {
    seed: u64,
    max_iterations: u32,
    step_size: f64,
    goal_bias: f64,
    goal_connection_radius: f64,
}

impl RrtConfig {
    /// Construct a bounded deterministic RRT configuration.
    pub fn new(
        seed: u64,
        max_iterations: u32,
        step_size: f64,
        goal_bias: f64,
        goal_connection_radius: f64,
    ) -> Result<Self, SamplingReachabilityError> {
        if max_iterations == 0 || max_iterations > MAX_RRT_ITERATIONS {
            return Err(SamplingReachabilityError::InvalidConfig {
                reason: format!(
                    "max_iterations must be in 1..={MAX_RRT_ITERATIONS}, got {max_iterations}"
                ),
            });
        }
        if !step_size.is_finite() || step_size <= 0.0 {
            return Err(SamplingReachabilityError::InvalidConfig {
                reason: format!("step_size must be finite and > 0, got {step_size}"),
            });
        }
        if !goal_bias.is_finite() || !(0.0..=1.0).contains(&goal_bias) {
            return Err(SamplingReachabilityError::InvalidConfig {
                reason: format!("goal_bias must be finite and in [0,1], got {goal_bias}"),
            });
        }
        if !goal_connection_radius.is_finite() || goal_connection_radius <= 0.0 {
            return Err(SamplingReachabilityError::InvalidConfig {
                reason: format!(
                    "goal_connection_radius must be finite and > 0, got {goal_connection_radius}"
                ),
            });
        }
        Ok(Self {
            seed,
            max_iterations,
            step_size,
            goal_bias,
            goal_connection_radius,
        })
    }

    /// Named deterministic reference profile.
    pub fn reference_v1(seed: u64) -> Self {
        Self::new(seed, 10_000, 0.5, 0.05, 0.75).expect("reference RRT config is valid")
    }

    /// Deterministic RNG seed.
    pub fn seed(&self) -> u64 {
        self.seed
    }
    /// Maximum tree-expansion attempts.
    pub fn max_iterations(&self) -> u32 {
        self.max_iterations
    }
    /// Maximum extension length.
    pub fn step_size(&self) -> f64 {
        self.step_size
    }
    /// Probability of sampling the exact goal.
    pub fn goal_bias(&self) -> f64 {
        self.goal_bias
    }
    /// Maximum distance at which a direct goal connection is attempted.
    pub fn goal_connection_radius(&self) -> f64 {
        self.goal_connection_radius
    }

    /// Exact configuration identity.
    pub fn identity(&self) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(b"symthaea-seeded-rrt-config-v1\0");
        hasher.update(&self.seed.to_le_bytes());
        hasher.update(&self.max_iterations.to_le_bytes());
        hasher.update(&self.step_size.to_bits().to_le_bytes());
        hasher.update(&self.goal_bias.to_bits().to_le_bytes());
        hasher.update(&self.goal_connection_radius.to_bits().to_le_bytes());
        *hasher.finalize().as_bytes()
    }
}

/// Oracle extension that exposes exact finite Euclidean sampling bounds.
pub trait BoundedEuclideanSamplingOracle: ContinuousValidityOracle {
    /// Inclusive coordinate lower bounds used for deterministic sampling.
    fn sampling_min(&self) -> &[f64];
    /// Inclusive coordinate upper bounds used for deterministic sampling.
    fn sampling_max(&self) -> &[f64];
}

impl BoundedEuclideanSamplingOracle for AnalyticBoxValidityOracle {
    fn sampling_min(&self) -> &[f64] {
        self.domain().min()
    }

    fn sampling_max(&self) -> &[f64] {
        self.domain().max()
    }
}

/// Exact execution receipt for one bounded RRT attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SamplingSolverReceipt {
    solver_identity: [u8; 32],
    config_identity: [u8; 32],
    problem_identity: [u8; 32],
    oracle_identity: [u8; 32],
    iterations: u32,
    accepted_nodes: u64,
    state_queries: u64,
    segment_queries: u64,
    encountered_unknown: bool,
}

impl SamplingSolverReceipt {
    /// Exact algorithm + config identity.
    pub fn solver_identity(&self) -> [u8; 32] {
        self.solver_identity
    }
    /// Exact RRT configuration identity.
    pub fn config_identity(&self) -> [u8; 32] {
        self.config_identity
    }
    /// Exact planning-problem identity.
    pub fn problem_identity(&self) -> [u8; 32] {
        self.problem_identity
    }
    /// Exact validity-oracle identity.
    pub fn oracle_identity(&self) -> [u8; 32] {
        self.oracle_identity
    }
    /// Number of expansion iterations actually executed.
    pub fn iterations(&self) -> u32 {
        self.iterations
    }
    /// Number of accepted tree nodes including the root.
    pub fn accepted_nodes(&self) -> u64 {
        self.accepted_nodes
    }
    /// State-oracle query count used by the solver itself.
    pub fn state_queries(&self) -> u64 {
        self.state_queries
    }
    /// Segment-oracle query count used by the solver itself.
    pub fn segment_queries(&self) -> u64 {
        self.segment_queries
    }
    /// Whether any solver query returned `Unknown`.
    pub fn encountered_unknown(&self) -> bool {
        self.encountered_unknown
    }
}

/// Proof-strength result for a bounded sampling planner.
///
/// There is deliberately no infeasibility variant.
#[derive(Clone, Debug, PartialEq)]
pub enum SamplingReachabilityResult {
    /// A candidate path survived independent replay against the exact oracle.
    Feasible {
        /// Canonical waypoint path.
        path: Vec<Vec<f64>>,
        /// Independent replay receipt, separate from the RRT search state.
        validation: ContinuousPathValidationReceipt,
        /// Search execution receipt.
        solver: SamplingSolverReceipt,
    },
    /// The bounded sampling attempt cannot justify feasibility or infeasibility.
    Unknown {
        /// Shared proof-strength unknown category.
        reason: UnknownReason,
        /// Bounded diagnostic detail.
        detail: String,
        /// Best known tree path toward the goal, not promoted to feasibility.
        best_partial_path: Vec<Vec<f64>>,
        /// Search execution receipt.
        solver: SamplingSolverReceipt,
    },
}

/// Fail-closed RRT construction/execution errors.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum SamplingReachabilityError {
    /// Solver profile is malformed.
    #[error("invalid RRT configuration: {reason}")]
    InvalidConfig { reason: String },
    /// Sampling domain does not match the exact problem dimension or is numerically unusable.
    #[error("invalid RRT sampling domain: {reason}")]
    InvalidSamplingDomain { reason: String },
    /// Start or goal is known invalid, making the planning query malformed for this solver.
    #[error("{endpoint} endpoint is invalid: {reason}")]
    InvalidEndpoint {
        endpoint: &'static str,
        reason: String,
    },
    /// Continuous substrate returned a hard contract/evaluation error.
    #[error(transparent)]
    Continuous(#[from] ContinuousReachabilityError),
    /// Solver-generated candidate contradicted independent replay.
    #[error("RRT candidate failed independent replay: {reason}")]
    CandidateReplayContradiction { reason: String },
    /// Internal tree reconstruction failed.
    #[error("RRT tree reconstruction failed: {reason}")]
    TreeReconstruction { reason: String },
    /// Numeric distance/steering arithmetic became non-finite.
    #[error("RRT numerical failure: {reason}")]
    Numerical { reason: String },
}

#[derive(Clone, Debug)]
struct TreeNode {
    state: Vec<f64>,
    parent: Option<usize>,
}

/// Run deterministic seeded RRT over an exact bounded Euclidean oracle.
///
/// Sampling exhaustion can only return [`SamplingReachabilityResult::Unknown`].
pub fn seeded_rrt<O: BoundedEuclideanSamplingOracle>(
    problem: &EuclideanPlanningProblem,
    oracle: &O,
    config: RrtConfig,
) -> Result<SamplingReachabilityResult, SamplingReachabilityError> {
    validate_sampling_domain(problem, oracle)?;
    let config_identity = config.identity();
    let solver_identity = sampling_solver_identity(config_identity);
    let mut state_queries = 0_u64;
    let mut segment_queries = 0_u64;
    let mut encountered_unknown = false;

    for (endpoint, state) in [("start", problem.start()), ("goal", problem.goal())] {
        state_queries += 1;
        match oracle.state_verdict(state)? {
            OracleVerdict::Valid => {}
            OracleVerdict::Invalid { reason } => {
                return Err(SamplingReachabilityError::InvalidEndpoint { endpoint, reason });
            }
            OracleVerdict::Unknown { reason } => {
                encountered_unknown = true;
                let solver = receipt(
                    problem,
                    oracle,
                    config_identity,
                    solver_identity,
                    0,
                    1,
                    state_queries,
                    segment_queries,
                    encountered_unknown,
                );
                return Ok(SamplingReachabilityResult::Unknown {
                    reason: UnknownReason::ValidityOracleIncomplete,
                    detail: format!("{endpoint} endpoint validity is unknown: {reason}"),
                    best_partial_path: vec![problem.start().to_vec()],
                    solver,
                });
            }
        }
    }

    // Always try the direct segment first. Failure is only local information.
    segment_queries += 1;
    match oracle.segment_verdict(problem.start(), problem.goal())? {
        OracleVerdict::Valid => {
            return promote_candidate(
                problem,
                oracle,
                vec![problem.start().to_vec(), problem.goal().to_vec()],
                config_identity,
                solver_identity,
                0,
                1,
                state_queries,
                segment_queries,
                encountered_unknown,
            );
        }
        OracleVerdict::Invalid { .. } => {}
        OracleVerdict::Unknown { .. } => encountered_unknown = true,
    }

    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut tree = vec![TreeNode {
        state: problem.start().to_vec(),
        parent: None,
    }];
    let mut iterations_executed = 0_u32;

    for iteration in 1..=config.max_iterations {
        iterations_executed = iteration;
        let target = if rng.gen_range(0.0..1.0) < config.goal_bias {
            problem.goal().to_vec()
        } else {
            sample_uniform(oracle, &mut rng)?
        };
        let nearest = nearest_node(&tree, &target)?;
        let candidate = steer(&tree[nearest].state, &target, config.step_size)?;
        if candidate == tree[nearest].state {
            continue;
        }

        state_queries += 1;
        match oracle.state_verdict(&candidate)? {
            OracleVerdict::Valid => {}
            OracleVerdict::Invalid { .. } => continue,
            OracleVerdict::Unknown { .. } => {
                encountered_unknown = true;
                continue;
            }
        }
        segment_queries += 1;
        match oracle.segment_verdict(&tree[nearest].state, &candidate)? {
            OracleVerdict::Valid => {}
            OracleVerdict::Invalid { .. } => continue,
            OracleVerdict::Unknown { .. } => {
                encountered_unknown = true;
                continue;
            }
        }

        let new_index = tree.len();
        tree.push(TreeNode {
            state: candidate,
            parent: Some(nearest),
        });
        let goal_distance = euclidean_distance(&tree[new_index].state, problem.goal())?;
        if goal_distance <= config.goal_connection_radius {
            segment_queries += 1;
            match oracle.segment_verdict(&tree[new_index].state, problem.goal())? {
                OracleVerdict::Valid => {
                    let mut path = reconstruct_tree_path(&tree, new_index)?;
                    if path.last().map(Vec::as_slice) != Some(problem.goal()) {
                        path.push(problem.goal().to_vec());
                    }
                    return promote_candidate(
                        problem,
                        oracle,
                        path,
                        config_identity,
                        solver_identity,
                        iteration,
                        tree.len() as u64,
                        state_queries,
                        segment_queries,
                        encountered_unknown,
                    );
                }
                OracleVerdict::Invalid { .. } => {}
                OracleVerdict::Unknown { .. } => encountered_unknown = true,
            }
        }
    }

    let best = best_partial_index(&tree, problem.goal())?;
    let best_partial_path = reconstruct_tree_path(&tree, best)?;
    let reason = if encountered_unknown {
        UnknownReason::ValidityOracleIncomplete
    } else {
        UnknownReason::BudgetExceeded
    };
    let detail = if encountered_unknown {
        "RRT budget exhausted after at least one undecidable validity query".to_string()
    } else {
        format!(
            "RRT exhausted {} expansion attempts without a validated goal path",
            config.max_iterations
        )
    };
    let solver = receipt(
        problem,
        oracle,
        config_identity,
        solver_identity,
        iterations_executed,
        tree.len() as u64,
        state_queries,
        segment_queries,
        encountered_unknown,
    );
    Ok(SamplingReachabilityResult::Unknown {
        reason,
        detail,
        best_partial_path,
        solver,
    })
}

fn promote_candidate<O: BoundedEuclideanSamplingOracle>(
    problem: &EuclideanPlanningProblem,
    oracle: &O,
    path: Vec<Vec<f64>>,
    config_identity: [u8; 32],
    solver_identity: [u8; 32],
    iterations: u32,
    accepted_nodes: u64,
    state_queries: u64,
    segment_queries: u64,
    encountered_unknown: bool,
) -> Result<SamplingReachabilityResult, SamplingReachabilityError> {
    let solver = receipt(
        problem,
        oracle,
        config_identity,
        solver_identity,
        iterations,
        accepted_nodes,
        state_queries,
        segment_queries,
        encountered_unknown,
    );
    match validate_euclidean_path(problem, oracle, &path)? {
        ContinuousPathReplay::Valid(validation) => Ok(SamplingReachabilityResult::Feasible {
            path,
            validation,
            solver,
        }),
        ContinuousPathReplay::Unknown { reason, .. } => Ok(SamplingReachabilityResult::Unknown {
            reason: UnknownReason::ValidityOracleIncomplete,
            detail: format!("candidate replay became unknown: {reason}"),
            best_partial_path: path,
            solver,
        }),
        ContinuousPathReplay::Invalid { reason, .. } => {
            Err(SamplingReachabilityError::CandidateReplayContradiction { reason })
        }
    }
}

fn receipt<O: BoundedEuclideanSamplingOracle>(
    problem: &EuclideanPlanningProblem,
    oracle: &O,
    config_identity: [u8; 32],
    solver_identity: [u8; 32],
    iterations: u32,
    accepted_nodes: u64,
    state_queries: u64,
    segment_queries: u64,
    encountered_unknown: bool,
) -> SamplingSolverReceipt {
    SamplingSolverReceipt {
        solver_identity,
        config_identity,
        problem_identity: problem.identity(),
        oracle_identity: oracle.profile().identity(),
        iterations,
        accepted_nodes,
        state_queries,
        segment_queries,
        encountered_unknown,
    }
}

fn validate_sampling_domain<O: BoundedEuclideanSamplingOracle>(
    problem: &EuclideanPlanningProblem,
    oracle: &O,
) -> Result<(), SamplingReachabilityError> {
    let dimension = problem.space().dimension();
    if dimension == 0 || dimension > MAX_RRT_DIMENSION {
        return Err(SamplingReachabilityError::InvalidSamplingDomain {
            reason: format!("RRT dimension must be in 1..={MAX_RRT_DIMENSION}, got {dimension}"),
        });
    }
    let min = oracle.sampling_min();
    let max = oracle.sampling_max();
    if min.len() != dimension || max.len() != dimension {
        return Err(SamplingReachabilityError::InvalidSamplingDomain {
            reason: "sampling bound dimension does not match problem".to_string(),
        });
    }
    for index in 0..dimension {
        let span = max[index] - min[index];
        if !min[index].is_finite()
            || !max[index].is_finite()
            || !span.is_finite()
            || span <= 0.0
        {
            return Err(SamplingReachabilityError::InvalidSamplingDomain {
                reason: format!(
                    "axis {index} requires finite strict bounds with finite span, got [{}, {}]",
                    min[index], max[index]
                ),
            });
        }
    }
    if oracle.profile().identity() != problem.oracle_identity() {
        return Err(SamplingReachabilityError::InvalidSamplingDomain {
            reason: "oracle identity does not match exact planning problem".to_string(),
        });
    }
    Ok(())
}

fn sample_uniform<O: BoundedEuclideanSamplingOracle>(
    oracle: &O,
    rng: &mut StdRng,
) -> Result<Vec<f64>, SamplingReachabilityError> {
    let min = oracle.sampling_min();
    let max = oracle.sampling_max();
    let mut sample = Vec::with_capacity(min.len());
    for index in 0..min.len() {
        let value = rng.gen_range(min[index]..max[index]);
        if !value.is_finite() {
            return Err(SamplingReachabilityError::Numerical {
                reason: format!("sampled non-finite coordinate on axis {index}"),
            });
        }
        sample.push(value + 0.0);
    }
    Ok(sample)
}

fn nearest_node(tree: &[TreeNode], target: &[f64]) -> Result<usize, SamplingReachabilityError> {
    let mut best_index = 0usize;
    let mut best_distance = f64::INFINITY;
    for (index, node) in tree.iter().enumerate() {
        let distance = euclidean_distance(&node.state, target)?;
        if distance < best_distance {
            best_distance = distance;
            best_index = index;
        }
    }
    Ok(best_index)
}

fn best_partial_index(
    tree: &[TreeNode],
    goal: &[f64],
) -> Result<usize, SamplingReachabilityError> {
    nearest_node(tree, goal)
}

fn steer(
    from: &[f64],
    target: &[f64],
    step_size: f64,
) -> Result<Vec<f64>, SamplingReachabilityError> {
    let distance = euclidean_distance(from, target)?;
    if distance == 0.0 {
        return Ok(from.to_vec());
    }
    if distance <= step_size {
        return Ok(target.iter().map(|value| *value + 0.0).collect());
    }
    let scale = step_size / distance;
    if !scale.is_finite() {
        return Err(SamplingReachabilityError::Numerical {
            reason: "steering scale became non-finite".to_string(),
        });
    }
    let mut result = Vec::with_capacity(from.len());
    for index in 0..from.len() {
        let value = from[index] + scale * (target[index] - from[index]);
        if !value.is_finite() {
            return Err(SamplingReachabilityError::Numerical {
                reason: format!("steering produced non-finite coordinate {index}"),
            });
        }
        result.push(value + 0.0);
    }
    Ok(result)
}

fn euclidean_distance(a: &[f64], b: &[f64]) -> Result<f64, SamplingReachabilityError> {
    if a.len() != b.len() {
        return Err(SamplingReachabilityError::Numerical {
            reason: "distance dimension mismatch".to_string(),
        });
    }
    let mut distance = 0.0_f64;
    for index in 0..a.len() {
        let delta = a[index] - b[index];
        if !delta.is_finite() {
            return Err(SamplingReachabilityError::Numerical {
                reason: format!("distance delta became non-finite on axis {index}"),
            });
        }
        distance = distance.hypot(delta);
    }
    if distance.is_finite() {
        Ok(distance)
    } else {
        Err(SamplingReachabilityError::Numerical {
            reason: "distance became non-finite".to_string(),
        })
    }
}

fn reconstruct_tree_path(
    tree: &[TreeNode],
    index: usize,
) -> Result<Vec<Vec<f64>>, SamplingReachabilityError> {
    if index >= tree.len() {
        return Err(SamplingReachabilityError::TreeReconstruction {
            reason: "node index outside tree".to_string(),
        });
    }
    let mut path = Vec::new();
    let mut cursor = Some(index);
    let mut steps = 0usize;
    while let Some(current) = cursor {
        path.push(tree[current].state.clone());
        cursor = tree[current].parent;
        steps += 1;
        if steps > tree.len() {
            return Err(SamplingReachabilityError::TreeReconstruction {
                reason: "parent chain contains a cycle".to_string(),
            });
        }
    }
    path.reverse();
    Ok(path)
}

fn sampling_solver_identity(config_identity: [u8; 32]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-deterministic-seeded-rrt-v1\0");
    hasher.update(&config_identity);
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::continuous_reachability::AxisAlignedBox;

    fn box2(min: [f64; 2], max: [f64; 2]) -> AxisAlignedBox {
        AxisAlignedBox::new(min.to_vec(), max.to_vec()).unwrap()
    }

    fn central_obstacle_oracle() -> AnalyticBoxValidityOracle {
        AnalyticBoxValidityOracle::new(
            box2([0.0, 0.0], [10.0, 10.0]),
            vec![box2([4.0, 4.0], [6.0, 6.0])],
            vec![],
        )
        .unwrap()
    }

    #[test]
    fn open_world_promotes_only_after_independent_replay() {
        let oracle = AnalyticBoxValidityOracle::new(
            box2([0.0, 0.0], [10.0, 10.0]),
            vec![],
            vec![],
        )
        .unwrap();
        let problem = EuclideanPlanningProblem::new(
            2,
            vec![1.0, 1.0],
            vec![9.0, 9.0],
            &oracle,
        )
        .unwrap();
        let result = seeded_rrt(&problem, &oracle, RrtConfig::reference_v1(7)).unwrap();
        let SamplingReachabilityResult::Feasible {
            path,
            validation,
            solver,
        } = result
        else {
            panic!("open world should use valid direct segment");
        };
        assert_eq!(path.len(), 2);
        let replay = validate_euclidean_path(&problem, &oracle, &path).unwrap();
        let ContinuousPathReplay::Valid(replay) = replay else {
            panic!("returned path must replay valid");
        };
        assert_eq!(replay.path_identity(), validation.path_identity());
        assert_eq!(solver.iterations(), 0);
    }

    #[test]
    fn tiny_budget_failure_is_unknown_even_when_detour_exists() {
        let oracle = central_obstacle_oracle();
        let problem = EuclideanPlanningProblem::new(
            2,
            vec![1.0, 5.0],
            vec![9.0, 5.0],
            &oracle,
        )
        .unwrap();
        let config = RrtConfig::new(11, 1, 0.25, 1.0, 0.25).unwrap();
        let result = seeded_rrt(&problem, &oracle, config).unwrap();
        let SamplingReachabilityResult::Unknown { reason, solver, .. } = result else {
            panic!("one tiny step must not certify infeasibility");
        };
        assert_eq!(reason, UnknownReason::BudgetExceeded);
        assert_eq!(solver.iterations(), 1);
    }

    #[test]
    fn unknown_direct_region_propagates_if_budget_cannot_find_detour() {
        let oracle = AnalyticBoxValidityOracle::new(
            box2([0.0, 0.0], [10.0, 10.0]),
            vec![],
            vec![box2([4.0, 4.0], [6.0, 6.0])],
        )
        .unwrap();
        let problem = EuclideanPlanningProblem::new(
            2,
            vec![1.0, 5.0],
            vec![9.0, 5.0],
            &oracle,
        )
        .unwrap();
        let config = RrtConfig::new(5, 1, 0.25, 1.0, 0.25).unwrap();
        let result = seeded_rrt(&problem, &oracle, config).unwrap();
        let SamplingReachabilityResult::Unknown { reason, solver, .. } = result else {
            panic!("undecidable direct corridor must remain unknown");
        };
        assert_eq!(reason, UnknownReason::ValidityOracleIncomplete);
        assert!(solver.encountered_unknown());
    }

    #[test]
    fn deterministic_seeded_rrt_can_find_synthetic_detour() {
        let oracle = central_obstacle_oracle();
        let problem = EuclideanPlanningProblem::new(
            2,
            vec![1.0, 5.0],
            vec![9.0, 5.0],
            &oracle,
        )
        .unwrap();
        let config = RrtConfig::new(42, 20_000, 0.5, 0.10, 0.75).unwrap();
        let result = seeded_rrt(&problem, &oracle, config).unwrap();
        let SamplingReachabilityResult::Feasible {
            path,
            validation,
            solver,
        } = result
        else {
            panic!("reference seeded RRT should find central-box detour");
        };
        assert!(path.len() >= 3);
        assert!(solver.iterations() > 0);
        assert!(validation.total_cost() > 8.0);
        assert!(matches!(
            validate_euclidean_path(&problem, &oracle, &path).unwrap(),
            ContinuousPathReplay::Valid(_)
        ));
    }

    #[test]
    fn config_identity_binds_seed_and_budget() {
        let a = RrtConfig::new(1, 100, 0.5, 0.1, 0.75).unwrap();
        let b = RrtConfig::new(2, 100, 0.5, 0.1, 0.75).unwrap();
        let c = RrtConfig::new(1, 101, 0.5, 0.1, 0.75).unwrap();
        assert_ne!(a.identity(), b.identity());
        assert_ne!(a.identity(), c.identity());
    }

    #[test]
    fn invalid_endpoint_is_a_bad_query_not_infeasibility() {
        let oracle = central_obstacle_oracle();
        let problem = EuclideanPlanningProblem::new(
            2,
            vec![5.0, 5.0],
            vec![9.0, 5.0],
            &oracle,
        )
        .unwrap();
        let error = seeded_rrt(&problem, &oracle, RrtConfig::reference_v1(1)).unwrap_err();
        assert!(matches!(
            error,
            SamplingReachabilityError::InvalidEndpoint { endpoint: "start", .. }
        ));
    }
}