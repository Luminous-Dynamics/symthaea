// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Planner-neutral reachability results with proof-strength distinctions.
//!
//! The governing rules are intentionally strict:
//! - a solver finding a path is not enough: the path is independently replayed;
//! - a solver failing to find a path is not proof of infeasibility;
//! - `CertifiedInfeasible` requires a separately verifiable certificate.
//!
//! V1 begins with a complete finite directed-graph reference solver. Continuous
//! and sampling planners can reuse the result vocabulary without inheriting the
//! finite solver's proof authority.

use std::collections::BTreeSet;

use blake3::Hasher;
use thiserror::Error;

/// Maximum node count admitted by the V1 finite reference solver.
pub const MAX_FINITE_GRAPH_NODES: usize = 100_000;
/// Maximum edge count admitted by the V1 finite reference solver.
pub const MAX_FINITE_GRAPH_EDGES: usize = 1_000_000;

/// Reasons a bounded planner may return `Unknown` without claiming impossibility.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum UnknownReason {
    /// Explicit compute/search budget was exhausted.
    BudgetExceeded,
    /// Search completed its bounded attempt without discovering a path.
    NoPathFound,
    /// A required validity oracle could not decide one or more queries.
    ValidityOracleIncomplete,
    /// Numerical arithmetic failed or became non-finite.
    NumericalFailure,
    /// Query left the exact modeled domain.
    ModelOutOfDomain,
    /// Constraint projection/localization failed without proving infeasibility.
    ConstraintProjectionFailure,
    /// Solver does not support the supplied state-space/problem profile.
    UnsupportedSpace,
}

/// Solver execution receipt shared by reachability result variants.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SolverReceipt {
    solver_identity: [u8; 32],
    problem_identity: [u8; 32],
    expanded_states: u64,
    validity_queries: u64,
    complete_search: bool,
}

impl SolverReceipt {
    /// Exact solver/profile identity.
    pub fn solver_identity(&self) -> [u8; 32] {
        self.solver_identity
    }
    /// Exact problem identity.
    pub fn problem_identity(&self) -> [u8; 32] {
        self.problem_identity
    }
    /// Number of states expanded by the solver.
    pub fn expanded_states(&self) -> u64 {
        self.expanded_states
    }
    /// Number of edge/validity checks performed.
    pub fn validity_queries(&self) -> u64 {
        self.validity_queries
    }
    /// Whether this execution exhausted the solver's complete problem domain.
    pub fn complete_search(&self) -> bool {
        self.complete_search
    }
}

/// Independent replay receipt for a feasible path witness.
#[derive(Clone, Debug, PartialEq)]
pub struct PathValidationReceipt {
    problem_identity: [u8; 32],
    validator_identity: [u8; 32],
    path_identity: [u8; 32],
    segment_count: usize,
    total_cost: f64,
}

impl PathValidationReceipt {
    /// Exact problem identity against which the path was replayed.
    pub fn problem_identity(&self) -> [u8; 32] {
        self.problem_identity
    }
    /// Independent path-validator identity.
    pub fn validator_identity(&self) -> [u8; 32] {
        self.validator_identity
    }
    /// Identity binding exact problem, ordered states and validated cost.
    pub fn path_identity(&self) -> [u8; 32] {
        self.path_identity
    }
    /// Number of validated transitions/segments.
    pub fn segment_count(&self) -> usize {
        self.segment_count
    }
    /// Recomputed path cost under exact finite-graph edge costs.
    pub fn total_cost(&self) -> f64 {
        self.total_cost
    }
}

/// Validated feasible path plus solver and independent replay receipts.
#[derive(Clone, Debug, PartialEq)]
pub struct FeasiblePath<S> {
    states: Vec<S>,
    cost: f64,
    validation: PathValidationReceipt,
    solver: SolverReceipt,
}

impl<S> FeasiblePath<S> {
    /// Ordered path states.
    pub fn states(&self) -> &[S] {
        &self.states
    }
    /// Independently recomputed path cost.
    pub fn cost(&self) -> f64 {
        self.cost
    }
    /// Independent path validation receipt.
    pub fn validation(&self) -> &PathValidationReceipt {
        &self.validation
    }
    /// Solver execution receipt that proposed the path.
    pub fn solver(&self) -> &SolverReceipt {
        &self.solver
    }
}

/// Certified infeasibility result carrying a separately verified certificate.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedInfeasible<C> {
    certificate: C,
    verification: CertificateVerificationReceipt,
    solver: SolverReceipt,
}

impl<C> CertifiedInfeasible<C> {
    /// Independently checkable infeasibility certificate.
    pub fn certificate(&self) -> &C {
        &self.certificate
    }
    /// Certificate-verification receipt.
    pub fn verification(&self) -> &CertificateVerificationReceipt {
        &self.verification
    }
    /// Solver execution receipt that produced the candidate certificate.
    pub fn solver(&self) -> &SolverReceipt {
        &self.solver
    }
}

/// Bounded/indeterminate reachability result.
#[derive(Clone, Debug, PartialEq)]
pub struct UnknownReachability<S> {
    reason: UnknownReason,
    detail: String,
    best_partial_path: Option<Vec<S>>,
    solver: SolverReceipt,
}

impl<S> UnknownReachability<S> {
    /// Typed reason no strong reachability conclusion is justified.
    pub fn reason(&self) -> &UnknownReason {
        &self.reason
    }
    /// Human-readable bounded failure detail.
    pub fn detail(&self) -> &str {
        &self.detail
    }
    /// Optional best partial path retained for diagnostics, not promoted to feasibility.
    pub fn best_partial_path(&self) -> Option<&[S]> {
        self.best_partial_path.as_deref()
    }
    /// Solver execution receipt.
    pub fn solver(&self) -> &SolverReceipt {
        &self.solver
    }
}

/// Planner-neutral proof-strength reachability result.
#[derive(Clone, Debug, PartialEq)]
pub enum ReachabilityResult<S, C> {
    /// An independently replayed valid path exists.
    Feasible(FeasiblePath<S>),
    /// A separately verified certificate establishes infeasibility under exact assumptions.
    CertifiedInfeasible(CertifiedInfeasible<C>),
    /// The bounded attempt does not justify either feasibility or infeasibility.
    Unknown(UnknownReachability<S>),
}

/// Directed finite graph edge with non-negative scalar cost.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FiniteGraphEdge {
    from: u32,
    to: u32,
    cost: f64,
}

impl FiniteGraphEdge {
    /// Construct an edge candidate. Full validation occurs in [`FiniteGraph::new`].
    pub fn new(from: u32, to: u32, cost: f64) -> Self {
        Self { from, to, cost }
    }
    /// Source node.
    pub fn from(&self) -> u32 {
        self.from
    }
    /// Target node.
    pub fn to(&self) -> u32 {
        self.to
    }
    /// Non-negative transition cost.
    pub fn cost(&self) -> f64 {
        self.cost
    }
}

/// Exact finite directed graph used by the complete reference solver.
#[derive(Clone, Debug)]
pub struct FiniteGraph {
    node_count: usize,
    edges: Vec<FiniteGraphEdge>,
    outgoing: Vec<Vec<FiniteGraphEdge>>,
    identity: [u8; 32],
}

impl FiniteGraph {
    /// Construct and canonicalize an exact directed graph.
    pub fn new(
        node_count: usize,
        mut edges: Vec<FiniteGraphEdge>,
    ) -> Result<Self, ReachabilityError> {
        if node_count == 0 || node_count > MAX_FINITE_GRAPH_NODES {
            return Err(ReachabilityError::InvalidGraph {
                reason: format!(
                    "node count must be in 1..={MAX_FINITE_GRAPH_NODES}, got {node_count}"
                ),
            });
        }
        if edges.len() > MAX_FINITE_GRAPH_EDGES {
            return Err(ReachabilityError::InvalidGraph {
                reason: format!(
                    "edge count {} exceeds maximum {MAX_FINITE_GRAPH_EDGES}",
                    edges.len()
                ),
            });
        }
        for edge in &mut edges {
            if edge.from as usize >= node_count || edge.to as usize >= node_count {
                return Err(ReachabilityError::InvalidGraph {
                    reason: format!(
                        "edge {} -> {} references node outside 0..{}",
                        edge.from,
                        edge.to,
                        node_count - 1
                    ),
                });
            }
            if !edge.cost.is_finite() || edge.cost < 0.0 {
                return Err(ReachabilityError::InvalidGraph {
                    reason: format!(
                        "edge {} -> {} has invalid non-negative finite cost {}",
                        edge.from, edge.to, edge.cost
                    ),
                });
            }
            edge.cost += 0.0;
        }
        edges.sort_by_key(|edge| (edge.from, edge.to));
        for pair in edges.windows(2) {
            if pair[0].from == pair[1].from && pair[0].to == pair[1].to {
                return Err(ReachabilityError::InvalidGraph {
                    reason: format!(
                        "duplicate directed edge {} -> {}",
                        pair[0].from, pair[0].to
                    ),
                });
            }
        }
        let mut outgoing = vec![Vec::new(); node_count];
        for edge in &edges {
            outgoing[edge.from as usize].push(*edge);
        }
        let identity = hash_graph(node_count, &edges);
        Ok(Self {
            node_count,
            edges,
            outgoing,
            identity,
        })
    }

    /// Number of nodes, identified by `0..node_count`.
    pub fn node_count(&self) -> usize {
        self.node_count
    }
    /// Canonically ordered directed edges.
    pub fn edges(&self) -> &[FiniteGraphEdge] {
        &self.edges
    }
    /// Deterministic graph identity independent of input edge order.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }

    fn outgoing(&self, node: u32) -> &[FiniteGraphEdge] {
        &self.outgoing[node as usize]
    }

    fn edge_cost(&self, from: u32, to: u32) -> Option<f64> {
        self.outgoing(from)
            .iter()
            .find(|edge| edge.to == to)
            .map(|edge| edge.cost)
    }
}

/// Start/goal query over one exact finite graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FiniteGraphQuery {
    start: u32,
    goal: u32,
}

impl FiniteGraphQuery {
    /// Construct a finite graph query. Node-range validation uses the target graph.
    pub fn new(start: u32, goal: u32) -> Self {
        Self { start, goal }
    }
    /// Start node.
    pub fn start(&self) -> u32 {
        self.start
    }
    /// Goal node.
    pub fn goal(&self) -> u32 {
        self.goal
    }
}

/// Closed-set cut certificate proving a finite directed goal is unreachable.
///
/// If a set contains the start, excludes the goal, and every outgoing edge from
/// the set remains in the set, then no directed path from start to goal exists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteGraphClosureCertificate {
    graph_identity: [u8; 32],
    problem_identity: [u8; 32],
    start: u32,
    goal: u32,
    closed_nodes: Vec<u32>,
    identity: [u8; 32],
}

impl FiniteGraphClosureCertificate {
    /// Exact graph identity.
    pub fn graph_identity(&self) -> [u8; 32] {
        self.graph_identity
    }
    /// Exact graph+query problem identity.
    pub fn problem_identity(&self) -> [u8; 32] {
        self.problem_identity
    }
    /// Start node required to lie inside the closed set.
    pub fn start(&self) -> u32 {
        self.start
    }
    /// Goal node required to lie outside the closed set.
    pub fn goal(&self) -> u32 {
        self.goal
    }
    /// Canonically ordered closed node set.
    pub fn closed_nodes(&self) -> &[u32] {
        &self.closed_nodes
    }
    /// Deterministic certificate identity.
    pub fn identity(&self) -> [u8; 32] {
        self.identity
    }
}

/// Independent verification receipt for an infeasibility certificate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertificateVerificationReceipt {
    problem_identity: [u8; 32],
    certificate_identity: [u8; 32],
    verifier_identity: [u8; 32],
}

impl CertificateVerificationReceipt {
    /// Exact problem identity.
    pub fn problem_identity(&self) -> [u8; 32] {
        self.problem_identity
    }
    /// Verified certificate identity.
    pub fn certificate_identity(&self) -> [u8; 32] {
        self.certificate_identity
    }
    /// Independent verifier/profile identity.
    pub fn verifier_identity(&self) -> [u8; 32] {
        self.verifier_identity
    }
}

/// Fail-closed finite reachability errors.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum ReachabilityError {
    /// Finite graph/profile is malformed.
    #[error("invalid finite graph: {reason}")]
    InvalidGraph { reason: String },
    /// Query references a node outside the exact graph.
    #[error("invalid graph query: {reason}")]
    InvalidQuery { reason: String },
    /// Candidate path fails independent replay.
    #[error("invalid path witness: {reason}")]
    InvalidPath { reason: String },
    /// Infeasibility certificate fails independent verification.
    #[error("invalid infeasibility certificate: {reason}")]
    InvalidCertificate { reason: String },
}

/// Independently replay and validate one finite graph path.
pub fn validate_finite_graph_path(
    graph: &FiniteGraph,
    query: FiniteGraphQuery,
    path: &[u32],
) -> Result<PathValidationReceipt, ReachabilityError> {
    validate_query(graph, query)?;
    if path.is_empty() {
        return Err(ReachabilityError::InvalidPath {
            reason: "path is empty".to_string(),
        });
    }
    if path[0] != query.start {
        return Err(ReachabilityError::InvalidPath {
            reason: format!("path starts at {}, expected {}", path[0], query.start),
        });
    }
    if *path.last().expect("non-empty path has last state") != query.goal {
        return Err(ReachabilityError::InvalidPath {
            reason: format!(
                "path ends at {}, expected {}",
                path[path.len() - 1],
                query.goal
            ),
        });
    }
    for &node in path {
        if node as usize >= graph.node_count {
            return Err(ReachabilityError::InvalidPath {
                reason: format!("path references out-of-range node {node}"),
            });
        }
    }

    let mut total_cost = 0.0_f64;
    for pair in path.windows(2) {
        let Some(cost) = graph.edge_cost(pair[0], pair[1]) else {
            return Err(ReachabilityError::InvalidPath {
                reason: format!("missing directed edge {} -> {}", pair[0], pair[1]),
            });
        };
        total_cost += cost;
        if !total_cost.is_finite() {
            return Err(ReachabilityError::InvalidPath {
                reason: "path cost overflowed/non-finite".to_string(),
            });
        }
    }

    let problem_identity = finite_problem_identity(graph, query);
    let validator_identity = finite_path_validator_identity(graph.identity);
    let path_identity = hash_graph_path(problem_identity, path, total_cost);
    Ok(PathValidationReceipt {
        problem_identity,
        validator_identity,
        path_identity,
        segment_count: path.len().saturating_sub(1),
        total_cost,
    })
}

/// Independently verify a finite closed-set infeasibility certificate.
pub fn verify_finite_graph_certificate(
    graph: &FiniteGraph,
    query: FiniteGraphQuery,
    certificate: &FiniteGraphClosureCertificate,
) -> Result<CertificateVerificationReceipt, ReachabilityError> {
    validate_query(graph, query)?;
    let problem_identity = finite_problem_identity(graph, query);
    if certificate.graph_identity != graph.identity {
        return Err(ReachabilityError::InvalidCertificate {
            reason: "graph identity mismatch".to_string(),
        });
    }
    if certificate.problem_identity != problem_identity
        || certificate.start != query.start
        || certificate.goal != query.goal
    {
        return Err(ReachabilityError::InvalidCertificate {
            reason: "query/problem identity mismatch".to_string(),
        });
    }
    if certificate.closed_nodes.is_empty() {
        return Err(ReachabilityError::InvalidCertificate {
            reason: "closed set is empty".to_string(),
        });
    }
    if certificate
        .closed_nodes
        .windows(2)
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(ReachabilityError::InvalidCertificate {
            reason: "closed nodes must be strictly increasing and unique".to_string(),
        });
    }
    if certificate
        .closed_nodes
        .iter()
        .any(|node| *node as usize >= graph.node_count)
    {
        return Err(ReachabilityError::InvalidCertificate {
            reason: "closed set references out-of-range node".to_string(),
        });
    }

    let closed: BTreeSet<u32> = certificate.closed_nodes.iter().copied().collect();
    if !closed.contains(&query.start) {
        return Err(ReachabilityError::InvalidCertificate {
            reason: "closed set does not contain start".to_string(),
        });
    }
    if closed.contains(&query.goal) {
        return Err(ReachabilityError::InvalidCertificate {
            reason: "closed set contains goal".to_string(),
        });
    }
    for &node in &certificate.closed_nodes {
        for edge in graph.outgoing(node) {
            if !closed.contains(&edge.to) {
                return Err(ReachabilityError::InvalidCertificate {
                    reason: format!(
                        "closed-set violation: edge {} -> {} exits certificate set",
                        edge.from, edge.to
                    ),
                });
            }
        }
    }

    let expected_identity = hash_graph_certificate(
        graph.identity,
        problem_identity,
        query,
        &certificate.closed_nodes,
    );
    if expected_identity != certificate.identity {
        return Err(ReachabilityError::InvalidCertificate {
            reason: "certificate identity mismatch".to_string(),
        });
    }

    Ok(CertificateVerificationReceipt {
        problem_identity,
        certificate_identity: certificate.identity,
        verifier_identity: finite_certificate_verifier_identity(),
    })
}

/// Run deterministic complete Dijkstra search over one finite non-negative graph.
///
/// `CertifiedInfeasible` is emitted only after constructing and independently
/// verifying a closed-set cut certificate. Arithmetic overflow becomes `Unknown`.
pub fn complete_finite_dijkstra(
    graph: &FiniteGraph,
    query: FiniteGraphQuery,
) -> Result<ReachabilityResult<u32, FiniteGraphClosureCertificate>, ReachabilityError> {
    validate_query(graph, query)?;
    let problem_identity = finite_problem_identity(graph, query);
    let solver_identity = finite_dijkstra_solver_identity();

    if query.start == query.goal {
        let path = vec![query.start];
        let validation = validate_finite_graph_path(graph, query, &path)?;
        let solver = SolverReceipt {
            solver_identity,
            problem_identity,
            expanded_states: 0,
            validity_queries: 0,
            complete_search: false,
        };
        return Ok(ReachabilityResult::Feasible(FeasiblePath {
            states: path,
            cost: validation.total_cost,
            validation,
            solver,
        }));
    }

    let n = graph.node_count;
    let mut distance = vec![f64::INFINITY; n];
    let mut predecessor: Vec<Option<u32>> = vec![None; n];
    let mut settled = vec![false; n];
    distance[query.start as usize] = 0.0;
    let mut expanded_states = 0_u64;
    let mut validity_queries = 0_u64;

    loop {
        let mut current: Option<usize> = None;
        let mut best_distance = f64::INFINITY;
        for node in 0..n {
            if settled[node] || !distance[node].is_finite() {
                continue;
            }
            if distance[node] < best_distance {
                best_distance = distance[node];
                current = Some(node);
            }
        }
        let Some(current) = current else {
            break;
        };
        settled[current] = true;
        expanded_states += 1;
        let current_node = current as u32;

        if current_node == query.goal {
            let path = reconstruct_path(query.start, query.goal, &predecessor)?;
            let validation = validate_finite_graph_path(graph, query, &path)?;
            let solver = SolverReceipt {
                solver_identity,
                problem_identity,
                expanded_states,
                validity_queries,
                complete_search: false,
            };
            return Ok(ReachabilityResult::Feasible(FeasiblePath {
                states: path,
                cost: validation.total_cost,
                validation,
                solver,
            }));
        }

        for edge in graph.outgoing(current_node) {
            validity_queries += 1;
            let candidate = distance[current] + edge.cost;
            if !candidate.is_finite() {
                let partial = reconstruct_path(query.start, current_node, &predecessor).ok();
                let solver = SolverReceipt {
                    solver_identity,
                    problem_identity,
                    expanded_states,
                    validity_queries,
                    complete_search: false,
                };
                return Ok(ReachabilityResult::Unknown(UnknownReachability {
                    reason: UnknownReason::NumericalFailure,
                    detail: format!(
                        "cost overflow while relaxing edge {} -> {}",
                        edge.from, edge.to
                    ),
                    best_partial_path: partial,
                    solver,
                }));
            }
            let target = edge.to as usize;
            if candidate < distance[target] {
                distance[target] = candidate;
                predecessor[target] = Some(current_node);
            }
        }
    }

    let closed_nodes: Vec<u32> = distance
        .iter()
        .enumerate()
        .filter_map(|(node, value)| value.is_finite().then_some(node as u32))
        .collect();
    let certificate = build_graph_certificate(graph, query, closed_nodes);
    let verification = verify_finite_graph_certificate(graph, query, &certificate)?;
    let solver = SolverReceipt {
        solver_identity,
        problem_identity,
        expanded_states,
        validity_queries,
        complete_search: true,
    };
    Ok(ReachabilityResult::CertifiedInfeasible(CertifiedInfeasible {
        certificate,
        verification,
        solver,
    }))
}

fn validate_query(graph: &FiniteGraph, query: FiniteGraphQuery) -> Result<(), ReachabilityError> {
    if query.start as usize >= graph.node_count || query.goal as usize >= graph.node_count {
        return Err(ReachabilityError::InvalidQuery {
            reason: format!(
                "query {} -> {} exceeds graph node range 0..{}",
                query.start,
                query.goal,
                graph.node_count - 1
            ),
        });
    }
    Ok(())
}

fn reconstruct_path(
    start: u32,
    goal: u32,
    predecessor: &[Option<u32>],
) -> Result<Vec<u32>, ReachabilityError> {
    let mut reversed = vec![goal];
    let mut cursor = goal;
    let mut steps = 0usize;
    while cursor != start {
        let Some(previous) = predecessor.get(cursor as usize).copied().flatten() else {
            return Err(ReachabilityError::InvalidPath {
                reason: format!("missing predecessor while reconstructing node {cursor}"),
            });
        };
        reversed.push(previous);
        cursor = previous;
        steps += 1;
        if steps > predecessor.len() {
            return Err(ReachabilityError::InvalidPath {
                reason: "predecessor chain contains a cycle".to_string(),
            });
        }
    }
    reversed.reverse();
    Ok(reversed)
}

fn build_graph_certificate(
    graph: &FiniteGraph,
    query: FiniteGraphQuery,
    mut closed_nodes: Vec<u32>,
) -> FiniteGraphClosureCertificate {
    closed_nodes.sort_unstable();
    closed_nodes.dedup();
    let problem_identity = finite_problem_identity(graph, query);
    let identity = hash_graph_certificate(graph.identity, problem_identity, query, &closed_nodes);
    FiniteGraphClosureCertificate {
        graph_identity: graph.identity,
        problem_identity,
        start: query.start,
        goal: query.goal,
        closed_nodes,
        identity,
    }
}

fn finite_problem_identity(graph: &FiniteGraph, query: FiniteGraphQuery) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-finite-reachability-problem-v1\0");
    hasher.update(&graph.identity);
    hasher.update(&query.start.to_le_bytes());
    hasher.update(&query.goal.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn finite_dijkstra_solver_identity() -> [u8; 32] {
    *blake3::hash(b"symthaea-complete-finite-dijkstra-v1").as_bytes()
}

fn finite_path_validator_identity(graph_identity: [u8; 32]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-finite-path-validator-v1\0");
    hasher.update(&graph_identity);
    *hasher.finalize().as_bytes()
}

fn finite_certificate_verifier_identity() -> [u8; 32] {
    *blake3::hash(b"symthaea-finite-closed-set-verifier-v1").as_bytes()
}

fn hash_graph(node_count: usize, edges: &[FiniteGraphEdge]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-finite-directed-graph-v1\0");
    hasher.update(&(node_count as u64).to_le_bytes());
    hasher.update(&(edges.len() as u64).to_le_bytes());
    for edge in edges {
        hasher.update(&edge.from.to_le_bytes());
        hasher.update(&edge.to.to_le_bytes());
        hasher.update(&edge.cost.to_bits().to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

fn hash_graph_path(problem_identity: [u8; 32], path: &[u32], cost: f64) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-finite-path-witness-v1\0");
    hasher.update(&problem_identity);
    hasher.update(&(path.len() as u64).to_le_bytes());
    for node in path {
        hasher.update(&node.to_le_bytes());
    }
    hasher.update(&cost.to_bits().to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn hash_graph_certificate(
    graph_identity: [u8; 32],
    problem_identity: [u8; 32],
    query: FiniteGraphQuery,
    nodes: &[u32],
) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(b"symthaea-finite-closed-set-certificate-v1\0");
    hasher.update(&graph_identity);
    hasher.update(&problem_identity);
    hasher.update(&query.start.to_le_bytes());
    hasher.update(&query.goal.to_le_bytes());
    hasher.update(&(nodes.len() as u64).to_le_bytes());
    for node in nodes {
        hasher.update(&node.to_le_bytes());
    }
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_graph() -> FiniteGraph {
        FiniteGraph::new(
            4,
            vec![
                FiniteGraphEdge::new(0, 2, 2.0),
                FiniteGraphEdge::new(2, 3, 2.0),
                FiniteGraphEdge::new(0, 1, 1.0),
                FiniteGraphEdge::new(1, 3, 2.0),
            ],
        )
        .unwrap()
    }

    #[test]
    fn complete_dijkstra_returns_independently_validated_path() {
        let graph = sample_graph();
        let query = FiniteGraphQuery::new(0, 3);
        let result = complete_finite_dijkstra(&graph, query).unwrap();
        let ReachabilityResult::Feasible(path) = result else {
            panic!("expected feasible result");
        };
        assert_eq!(path.states(), &[0, 1, 3]);
        assert_eq!(path.cost(), 3.0);
        assert_eq!(path.validation().segment_count(), 2);
        assert!(!path.solver().complete_search());
        let replay = validate_finite_graph_path(&graph, query, path.states()).unwrap();
        assert_eq!(replay.path_identity(), path.validation().path_identity());
    }

    #[test]
    fn disconnected_graph_emits_verified_closed_set_certificate() {
        let graph = FiniteGraph::new(
            4,
            vec![
                FiniteGraphEdge::new(0, 1, 1.0),
                FiniteGraphEdge::new(2, 3, 1.0),
            ],
        )
        .unwrap();
        let query = FiniteGraphQuery::new(0, 3);
        let result = complete_finite_dijkstra(&graph, query).unwrap();
        let ReachabilityResult::CertifiedInfeasible(result) = result else {
            panic!("expected certified infeasible result");
        };
        assert_eq!(result.certificate().closed_nodes(), &[0, 1]);
        assert!(result.solver().complete_search());
        let replay = verify_finite_graph_certificate(&graph, query, result.certificate()).unwrap();
        assert_eq!(
            replay.certificate_identity(),
            result.verification().certificate_identity()
        );
    }

    #[test]
    fn invalid_waypoint_path_is_rejected_by_independent_validator() {
        let graph = sample_graph();
        let query = FiniteGraphQuery::new(0, 3);
        let error = validate_finite_graph_path(&graph, query, &[0, 3]).unwrap_err();
        assert!(matches!(error, ReachabilityError::InvalidPath { .. }));
    }

    #[test]
    fn tampered_certificate_cannot_mint_infeasibility() {
        let graph = FiniteGraph::new(3, vec![FiniteGraphEdge::new(0, 1, 1.0)]).unwrap();
        let query = FiniteGraphQuery::new(0, 2);
        let result = complete_finite_dijkstra(&graph, query).unwrap();
        let ReachabilityResult::CertifiedInfeasible(result) = result else {
            panic!("expected certified infeasible result");
        };
        let mut certificate = result.certificate().clone();
        certificate.closed_nodes.push(2);
        assert!(verify_finite_graph_certificate(&graph, query, &certificate).is_err());
    }

    #[test]
    fn graph_identity_is_independent_of_input_edge_order() {
        let a = FiniteGraph::new(
            3,
            vec![
                FiniteGraphEdge::new(0, 1, 2.0),
                FiniteGraphEdge::new(1, 2, 3.0),
            ],
        )
        .unwrap();
        let b = FiniteGraph::new(
            3,
            vec![
                FiniteGraphEdge::new(1, 2, 3.0),
                FiniteGraphEdge::new(0, 1, 2.0),
            ],
        )
        .unwrap();
        assert_eq!(a.identity(), b.identity());
    }

    #[test]
    fn changing_edge_cost_changes_graph_identity() {
        let a = FiniteGraph::new(2, vec![FiniteGraphEdge::new(0, 1, 1.0)]).unwrap();
        let b = FiniteGraph::new(2, vec![FiniteGraphEdge::new(0, 1, 2.0)]).unwrap();
        assert_ne!(a.identity(), b.identity());
    }

    #[test]
    fn start_equal_goal_is_valid_zero_segment_witness() {
        let graph = sample_graph();
        let query = FiniteGraphQuery::new(2, 2);
        let result = complete_finite_dijkstra(&graph, query).unwrap();
        let ReachabilityResult::Feasible(path) = result else {
            panic!("expected feasible result");
        };
        assert_eq!(path.states(), &[2]);
        assert_eq!(path.cost(), 0.0);
        assert_eq!(path.validation().segment_count(), 0);
    }

    #[test]
    fn arithmetic_overflow_is_unknown_not_infeasible() {
        let huge = f64::MAX * 0.75;
        let graph = FiniteGraph::new(
            3,
            vec![
                FiniteGraphEdge::new(0, 1, huge),
                FiniteGraphEdge::new(1, 2, huge),
            ],
        )
        .unwrap();
        let result = complete_finite_dijkstra(&graph, FiniteGraphQuery::new(0, 2)).unwrap();
        let ReachabilityResult::Unknown(unknown) = result else {
            panic!("overflow must remain unknown");
        };
        assert_eq!(unknown.reason(), &UnknownReason::NumericalFailure);
        assert_eq!(unknown.best_partial_path(), Some(&[0, 1][..]));
        assert!(!unknown.solver().complete_search());
    }

    #[test]
    fn invalid_negative_or_duplicate_edges_fail_closed() {
        assert!(FiniteGraph::new(2, vec![FiniteGraphEdge::new(0, 1, -1.0)]).is_err());
        assert!(
            FiniteGraph::new(
                2,
                vec![
                    FiniteGraphEdge::new(0, 1, 1.0),
                    FiniteGraphEdge::new(0, 1, 2.0),
                ],
            )
            .is_err()
        );
    }
}