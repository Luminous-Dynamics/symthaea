// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Qualified causal identification for fully observed Markovian DAGs.
//!
//! This module deliberately has a narrower claim than [`super::IDAlgorithm`].  It implements
//! only the fully observed / no-latent-confounder case, where the interventional distribution
//! is identified by the truncated DAG factorization (g-formula):
//!
//! `P_x(y) = Σ_{v \ (x ∪ y)} ∏_{V_i ∉ x} P(V_i | pa_i)`.
//!
//! Graphs containing bidirected edges fail closed.  They must go through a separately qualified
//! latent-variable identification path rather than inheriting an unproven completeness claim.

use std::collections::{HashSet, VecDeque};
use std::fmt;

use super::{CausalExpression, CausalGraphWithLatents};

/// Exact semantic contract of this implementation.
pub const QUALIFIED_MARKOVIAN_ID_VERSION: &str = "rq-005b-markovian-g-formula-v1";

/// Fail-closed validation errors for the qualified Markovian identification path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualifiedMarkovianIdError {
    EmptyGraph,
    EmptyOutcome,
    NodeIndexOutOfRange {
        role: &'static str,
        index: usize,
        node_count: usize,
    },
    DuplicateTreatment(usize),
    DuplicateOutcome(usize),
    TreatmentOutcomeOverlap(usize),
    DirectedEdgeOutOfRange {
        parent: usize,
        child: usize,
        node_count: usize,
    },
    BidirectedEdgeOutOfRange {
        left: usize,
        right: usize,
        node_count: usize,
    },
    DirectedSelfLoop(usize),
    DuplicateDirectedEdge { parent: usize, child: usize },
    DirectedCycle,
    UnsupportedLatentGraph { bidirected_edges: usize },
}

impl fmt::Display for QualifiedMarkovianIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyGraph => write!(f, "causal graph contains no nodes"),
            Self::EmptyOutcome => write!(f, "causal query contains no outcome variables"),
            Self::NodeIndexOutOfRange {
                role,
                index,
                node_count,
            } => write!(
                f,
                "{role} node index {index} is outside graph of {node_count} nodes"
            ),
            Self::DuplicateTreatment(index) => {
                write!(f, "treatment node {index} appears more than once")
            }
            Self::DuplicateOutcome(index) => {
                write!(f, "outcome node {index} appears more than once")
            }
            Self::TreatmentOutcomeOverlap(index) => write!(
                f,
                "node {index} cannot be both treatment and outcome in one qualified query"
            ),
            Self::DirectedEdgeOutOfRange {
                parent,
                child,
                node_count,
            } => write!(
                f,
                "directed edge {parent}->{child} is outside graph of {node_count} nodes"
            ),
            Self::BidirectedEdgeOutOfRange {
                left,
                right,
                node_count,
            } => write!(
                f,
                "bidirected edge {left}<->{right} is outside graph of {node_count} nodes"
            ),
            Self::DirectedSelfLoop(node) => {
                write!(f, "directed self-loop at node {node} violates DAG semantics")
            }
            Self::DuplicateDirectedEdge { parent, child } => {
                write!(f, "duplicate directed edge {parent}->{child}")
            }
            Self::DirectedCycle => write!(f, "directed graph contains a cycle"),
            Self::UnsupportedLatentGraph { bidirected_edges } => write!(
                f,
                "qualified Markovian ID does not accept latent graphs ({bidirected_edges} bidirected edges)"
            ),
        }
    }
}

impl std::error::Error for QualifiedMarkovianIdError {}

/// Narrow, evidence-oriented identification path for fully observed causal DAGs.
#[derive(Debug, Clone, Copy, Default)]
pub struct QualifiedMarkovianId;

impl QualifiedMarkovianId {
    pub fn new() -> Self {
        Self
    }

    /// Identify `P(outcome | do(treatment))` by exact truncated DAG factorization.
    ///
    /// This method validates the complete graph/query shape before returning an estimand.  It
    /// intentionally rejects every graph with a bidirected edge; latent-variable identification
    /// remains outside this qualification boundary until RQ-005 establishes it independently.
    pub fn identify(
        &self,
        graph: &CausalGraphWithLatents,
        treatment: &[usize],
        outcome: &[usize],
    ) -> Result<CausalExpression, QualifiedMarkovianIdError> {
        let topo = validate_and_toposort(graph, treatment, outcome)?;
        let treatment_set: HashSet<usize> = treatment.iter().copied().collect();
        let outcome_set: HashSet<usize> = outcome.iter().copied().collect();

        let mut factors = Vec::new();
        for node in topo {
            if treatment_set.contains(&node) {
                // Intervention truncates the structural factor for X.
                continue;
            }
            let mut parents = graph.parents(node);
            parents.sort_unstable();
            parents.dedup();
            factors.push(CausalExpression::Probability {
                outcome: vec![node],
                conditioning: parents,
            });
        }

        let inner = match factors.len() {
            0 => CausalExpression::Product(Vec::new()),
            1 => factors.remove(0),
            _ => CausalExpression::Product(factors),
        };

        let sum_over: Vec<usize> = (0..graph.nodes.len())
            .filter(|node| !treatment_set.contains(node) && !outcome_set.contains(node))
            .collect();

        if sum_over.is_empty() {
            Ok(inner)
        } else {
            Ok(CausalExpression::Sum {
                sum_over,
                inner: Box::new(inner),
            })
        }
    }
}

fn validate_and_toposort(
    graph: &CausalGraphWithLatents,
    treatment: &[usize],
    outcome: &[usize],
) -> Result<Vec<usize>, QualifiedMarkovianIdError> {
    let n = graph.nodes.len();
    if n == 0 {
        return Err(QualifiedMarkovianIdError::EmptyGraph);
    }
    if outcome.is_empty() {
        return Err(QualifiedMarkovianIdError::EmptyOutcome);
    }

    let mut treatments = HashSet::new();
    for &node in treatment {
        if node >= n {
            return Err(QualifiedMarkovianIdError::NodeIndexOutOfRange {
                role: "treatment",
                index: node,
                node_count: n,
            });
        }
        if !treatments.insert(node) {
            return Err(QualifiedMarkovianIdError::DuplicateTreatment(node));
        }
    }

    let mut outcomes = HashSet::new();
    for &node in outcome {
        if node >= n {
            return Err(QualifiedMarkovianIdError::NodeIndexOutOfRange {
                role: "outcome",
                index: node,
                node_count: n,
            });
        }
        if !outcomes.insert(node) {
            return Err(QualifiedMarkovianIdError::DuplicateOutcome(node));
        }
        if treatments.contains(&node) {
            return Err(QualifiedMarkovianIdError::TreatmentOutcomeOverlap(node));
        }
    }

    for &(left, right) in &graph.bidirected {
        if left >= n || right >= n {
            return Err(QualifiedMarkovianIdError::BidirectedEdgeOutOfRange {
                left,
                right,
                node_count: n,
            });
        }
    }
    if !graph.bidirected.is_empty() {
        return Err(QualifiedMarkovianIdError::UnsupportedLatentGraph {
            bidirected_edges: graph.bidirected.len(),
        });
    }

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut indegree = vec![0usize; n];
    let mut seen_edges = HashSet::new();
    for &(parent, child) in &graph.directed {
        if parent >= n || child >= n {
            return Err(QualifiedMarkovianIdError::DirectedEdgeOutOfRange {
                parent,
                child,
                node_count: n,
            });
        }
        if parent == child {
            return Err(QualifiedMarkovianIdError::DirectedSelfLoop(parent));
        }
        if !seen_edges.insert((parent, child)) {
            return Err(QualifiedMarkovianIdError::DuplicateDirectedEdge { parent, child });
        }
        adjacency[parent].push(child);
        indegree[child] += 1;
    }

    // Deterministic Kahn ordering: lowest node index wins every tie.
    let mut ready: VecDeque<usize> = (0..n).filter(|&node| indegree[node] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(node) = ready.pop_front() {
        order.push(node);
        let mut children = adjacency[node].clone();
        children.sort_unstable();
        for child in children {
            indegree[child] -= 1;
            if indegree[child] == 0 {
                // Keep the ready frontier ordered without relying on HashSet iteration order.
                let insert_at = ready
                    .iter()
                    .position(|&queued| child < queued)
                    .unwrap_or(ready.len());
                ready.insert(insert_at, child);
            }
        }
    }

    if order.len() != n {
        return Err(QualifiedMarkovianIdError::DirectedCycle);
    }
    Ok(order)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observed_confounder_requires_g_formula_adjustment() {
        // Z → X, Z → Y, X → Y.
        // P(Y|do(X)) = Σ_Z P(Z) P(Y|X,Z), not P(Y|X).
        let graph = CausalGraphWithLatents::new(
            vec!["X".into(), "Y".into(), "Z".into()],
            vec![(2, 0), (2, 1), (0, 1)],
            vec![],
        );
        let expression = QualifiedMarkovianId::new()
            .identify(&graph, &[0], &[1])
            .unwrap_or_else(|err| panic!("Markovian identification must succeed: {err}"));
        assert_eq!(
            expression.to_string(&graph.nodes),
            "Σ_{Z} [P(Z) × P(Y|X,Z)]"
        );
    }

    #[test]
    fn mediator_is_marginalized_under_intervention() {
        // X → M → Y.
        let graph = CausalGraphWithLatents::new(
            vec!["X".into(), "M".into(), "Y".into()],
            vec![(0, 1), (1, 2)],
            vec![],
        );
        let expression = QualifiedMarkovianId::new()
            .identify(&graph, &[0], &[2])
            .unwrap_or_else(|err| panic!("Markovian identification must succeed: {err}"));
        assert_eq!(
            expression.to_string(&graph.nodes),
            "Σ_{M} [P(M|X) × P(Y|M)]"
        );
    }

    #[test]
    fn direct_effect_has_no_unnecessary_sum() {
        let graph = CausalGraphWithLatents::new(
            vec!["X".into(), "Y".into()],
            vec![(0, 1)],
            vec![],
        );
        let expression = QualifiedMarkovianId::new()
            .identify(&graph, &[0], &[1])
            .unwrap_or_else(|err| panic!("Markovian identification must succeed: {err}"));
        assert_eq!(expression.to_string(&graph.nodes), "P(Y|X)");
    }

    #[test]
    fn latent_graph_fails_closed() {
        let graph = CausalGraphWithLatents::new(
            vec!["X".into(), "Y".into()],
            vec![(0, 1)],
            vec![(0, 1)],
        );
        assert!(matches!(
            QualifiedMarkovianId::new().identify(&graph, &[0], &[1]),
            Err(QualifiedMarkovianIdError::UnsupportedLatentGraph { .. })
        ));
    }

    #[test]
    fn cycle_fails_closed() {
        let graph = CausalGraphWithLatents::new(
            vec!["A".into(), "B".into()],
            vec![(0, 1), (1, 0)],
            vec![],
        );
        assert!(matches!(
            QualifiedMarkovianId::new().identify(&graph, &[0], &[1]),
            Err(QualifiedMarkovianIdError::DirectedCycle)
        ));
    }

    #[test]
    fn invalid_query_indices_fail_closed() {
        let graph = CausalGraphWithLatents::new(vec!["X".into()], vec![], vec![]);
        assert!(matches!(
            QualifiedMarkovianId::new().identify(&graph, &[1], &[0]),
            Err(QualifiedMarkovianIdError::NodeIndexOutOfRange {
                role: "treatment",
                ..
            })
        ));
    }

    #[test]
    fn duplicate_edges_are_not_silently_normalized() {
        let graph = CausalGraphWithLatents::new(
            vec!["X".into(), "Y".into()],
            vec![(0, 1), (0, 1)],
            vec![],
        );
        assert!(matches!(
            QualifiedMarkovianId::new().identify(&graph, &[0], &[1]),
            Err(QualifiedMarkovianIdError::DuplicateDirectedEdge {
                parent: 0,
                child: 1
            })
        ));
    }
}
