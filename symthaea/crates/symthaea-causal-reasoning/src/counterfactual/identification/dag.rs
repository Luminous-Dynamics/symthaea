// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Causal DAG and query/outcome types.
//!
//! Defines the core graph structures and type definitions used throughout
//! the causal identification subsystem.

use serde::{Deserialize, Serialize};
use std::collections::{HashSet, VecDeque};

// ─────────────────────────────────────────────────────────────────────────────
// Causal DAG
// ─────────────────────────────────────────────────────────────────────────────

/// A directed acyclic graph representing causal structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalDAG {
    /// Node names.
    pub nodes: Vec<String>,
    /// Directed edges: (parent_idx, child_idx).
    pub edges: Vec<(usize, usize)>,
}

impl CausalDAG {
    pub fn new(nodes: Vec<String>, edges: Vec<(usize, usize)>) -> Self {
        Self { nodes, edges }
    }

    /// Get parents of a node.
    pub fn parents(&self, node: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|(_, c)| *c == node)
            .map(|(p, _)| *p)
            .collect()
    }

    /// Get children of a node.
    pub fn children(&self, node: usize) -> Vec<usize> {
        self.edges
            .iter()
            .filter(|(p, _)| *p == node)
            .map(|(_, c)| *c)
            .collect()
    }

    /// Get ancestors of a node (transitive parents).
    pub fn ancestors(&self, node: usize) -> HashSet<usize> {
        let mut result = HashSet::new();
        let mut stack = self.parents(node);
        while let Some(n) = stack.pop() {
            if result.insert(n) {
                stack.extend(self.parents(n));
            }
        }
        result
    }

    /// Get descendants of a node (transitive children).
    pub fn descendants(&self, node: usize) -> HashSet<usize> {
        let mut result = HashSet::new();
        let mut stack = self.children(node);
        while let Some(n) = stack.pop() {
            if result.insert(n) {
                stack.extend(self.children(n));
            }
        }
        result
    }

    /// Check if there is a directed path from `from` to `to`.
    pub fn has_path(&self, from: usize, to: usize) -> bool {
        self.descendants(from).contains(&to)
    }

    /// Find node index by name.
    pub fn node_index(&self, name: &str) -> Option<usize> {
        self.nodes.iter().position(|n| n == name)
    }

    /// Number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Get all edges as an iterator.
    pub fn edges(&self) -> impl Iterator<Item = &(usize, usize)> {
        self.edges.iter()
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // d-Separation (Full Implementation)
    // ─────────────────────────────────────────────────────────────────────────────

    /// Full d-separation test: X ⊥ Y | Z in graph G.
    ///
    /// Returns true if all paths between X and Y are blocked by Z.
    /// Uses the Bayes-Ball algorithm for efficient d-separation testing.
    ///
    /// A path is blocked by Z if:
    /// - For chains (A→B→C) or forks (A←B→C): B ∈ Z
    /// - For colliders (A→B←C): B ∉ Z AND no descendant of B is in Z
    pub fn is_d_separated(&self, x: usize, y: usize, z: &HashSet<usize>) -> bool {
        // Use Bayes-Ball algorithm: find all nodes reachable from X
        // given conditioning on Z. If Y is not reachable, they are d-separated.
        let reachable = self.bayes_ball_reachable(x, z);
        !reachable.contains(&y)
    }

    /// Bayes-Ball algorithm: find all nodes d-connected to `source` given conditioning set Z.
    ///
    /// The algorithm tracks whether we're visiting a node from a child or parent,
    /// which determines whether we can traverse through the node.
    fn bayes_ball_reachable(&self, source: usize, z: &HashSet<usize>) -> HashSet<usize> {
        // Pre-compute which nodes would activate a collider.
        // A collider at M is activated if M ∈ Z or any descendant of M is in Z.
        // Equivalently: M ∈ Z or M is an ancestor of some node in Z.
        let mut collider_activated: HashSet<usize> = z.clone();
        for &node in z {
            collider_activated.extend(self.ancestors(node));
        }

        // Track visited (node, from_child) pairs to avoid cycles
        let mut visited: HashSet<(usize, bool)> = HashSet::new();
        // Queue: (node, came_from_child)
        let mut queue: VecDeque<(usize, bool)> = VecDeque::new();
        let mut reachable: HashSet<usize> = HashSet::new();

        // Start from source, considering both directions
        queue.push_back((source, false)); // as if from parent
        queue.push_back((source, true)); // as if from child

        while let Some((node, from_child)) = queue.pop_front() {
            if !visited.insert((node, from_child)) {
                continue;
            }

            reachable.insert(node);
            let is_conditioned = z.contains(&node);

            if from_child {
                // Came from a child (upstream traversal, we went up an edge)
                if !is_conditioned {
                    // Not conditioned: can continue upstream to parents
                    // Parent sees us as coming FROM a child (we're going up)
                    for &parent in &self.parents(node) {
                        queue.push_back((parent, true));
                    }
                    // Can also go downstream to other children (fork pattern: A←B→C)
                    // Child sees us as coming FROM a parent (we're going down)
                    for &child in &self.children(node) {
                        queue.push_back((child, false));
                    }
                }
                // If conditioned, we're blocked for chains/forks
                // (no collider case when arriving from child)
            } else {
                // Came from a parent (downstream traversal, we went down an edge)
                if !is_conditioned {
                    // Not conditioned: can continue downstream to children
                    // Child sees us as coming FROM a parent
                    for &child in &self.children(node) {
                        queue.push_back((child, false));
                    }
                }
                // Check collider activation: if this node is in Z or has a descendant in Z,
                // the collider path is opened, allowing us to go UP to parents
                if collider_activated.contains(&node) {
                    // Collider is activated: can traverse upstream to parents
                    // Parent sees us as coming FROM a child
                    for &parent in &self.parents(node) {
                        queue.push_back((parent, true));
                    }
                }
            }
        }

        reachable
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Graph Surgery Operations (for do-calculus)
    // ─────────────────────────────────────────────────────────────────────────────

    /// Create a mutilated graph with incoming edges to specified nodes removed.
    ///
    /// This produces G̅_X (G with arrows into X removed), used in do-calculus
    /// to represent the effect of intervention do(X).
    pub fn remove_incoming(&self, nodes: &[usize]) -> CausalDAG {
        let node_set: HashSet<usize> = nodes.iter().copied().collect();
        let new_edges: Vec<(usize, usize)> = self
            .edges
            .iter()
            .filter(|(_, child)| !node_set.contains(child))
            .copied()
            .collect();
        CausalDAG {
            nodes: self.nodes.clone(),
            edges: new_edges,
        }
    }

    /// Create a mutilated graph with outgoing edges from specified nodes removed.
    ///
    /// This produces G_Z_ (G with arrows from Z removed), used in Rule 2.
    pub fn remove_outgoing(&self, nodes: &[usize]) -> CausalDAG {
        let node_set: HashSet<usize> = nodes.iter().copied().collect();
        let new_edges: Vec<(usize, usize)> = self
            .edges
            .iter()
            .filter(|(parent, _)| !node_set.contains(parent))
            .copied()
            .collect();
        CausalDAG {
            nodes: self.nodes.clone(),
            edges: new_edges,
        }
    }

    /// Create a mutilated graph for Rule 3: G̅_X,Z(W).
    ///
    /// This removes:
    /// - All incoming edges to X
    /// - Outgoing edges from Z, except those leading to ancestors of W
    ///
    /// Used in Rule 3 to determine if do(Z) can be removed.
    pub fn remove_for_rule3(
        &self,
        x_nodes: &[usize],
        z_nodes: &[usize],
        w_nodes: &[usize],
    ) -> CausalDAG {
        let x_set: HashSet<usize> = x_nodes.iter().copied().collect();
        let z_set: HashSet<usize> = z_nodes.iter().copied().collect();

        // Compute ancestors of W (nodes that lead to W)
        let mut w_ancestors: HashSet<usize> = HashSet::new();
        for &w in w_nodes {
            w_ancestors.extend(self.ancestors(w));
            w_ancestors.insert(w);
        }

        let new_edges: Vec<(usize, usize)> = self
            .edges
            .iter()
            .filter(|(parent, child)| {
                // Remove incoming edges to X
                if x_set.contains(child) {
                    return false;
                }
                // Remove outgoing edges from Z, except those to W ancestors
                if z_set.contains(parent) && !w_ancestors.contains(child) {
                    return false;
                }
                true
            })
            .copied()
            .collect();

        CausalDAG {
            nodes: self.nodes.clone(),
            edges: new_edges,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Causal Query
// ─────────────────────────────────────────────────────────────────────────────

/// A causal query: "What is the effect of X on Y?"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalQuery {
    /// Treatment variable (intervention target).
    pub treatment: usize,
    /// Outcome variable.
    pub outcome: usize,
    /// Conditioning set (optional).
    pub conditioning: Vec<usize>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Outcome Types
// ─────────────────────────────────────────────────────────────────────────────

/// Result of attempting to identify a causal effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CausalQueryOutcome {
    /// Causal effect is identified: we have a valid estimand.
    Identified {
        estimand: CausalEstimand,
        method: IdentificationMethod,
        confidence: f64,
    },
    /// Causal effect cannot be identified from observational data.
    Unidentified {
        reason: UnidentifiedReason,
        missing: Vec<String>,
        suggestions: Vec<String>,
    },
    /// Causal effect can be identified IF an assumption holds.
    AssumptionRequired {
        assumption: CausalAssumption,
        estimand_if_assumed: CausalEstimand,
        plausibility: f64,
    },
}

/// A causal estimand: a formula for computing the causal effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalEstimand {
    /// The estimated causal effect magnitude.
    pub effect: f64,
    /// Variables being adjusted for.
    pub adjustment_set: Vec<usize>,
    /// Human-readable description of the estimand.
    pub description: String,
}

/// Method used for identification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IdentificationMethod {
    /// Pearl Rule 1: d-separation check / insertion-deletion of observations.
    DSeparation,
    /// Backdoor adjustment (special case of Rule 1).
    BackdoorAdjustment,
    /// Frontdoor criterion (special case of Rule 1).
    FrontdoorCriterion,
    /// Pearl Rule 2: Action/observation exchange.
    /// P(y|do(x),do(z),w) = P(y|do(x),z,w) if Y ⊥ Z | X,W in G̅_X,Z_
    Rule2ActionObservation,
    /// Pearl Rule 3: Insertion/deletion of actions.
    /// P(y|do(x),do(z),w) = P(y|do(x),w) if Y ⊥ Z | X,W in G̅_X,Z(W)
    Rule3ActionDeletion,
    /// Shpitser-Pearl ID Algorithm (complete identification).
    IDAlgorithm,
}

/// Reason why a causal effect cannot be identified.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum UnidentifiedReason {
    /// No valid adjustment set exists (all backdoor paths are confounded).
    NoValidAdjustmentSet,
    /// Treatment and outcome are not connected.
    NotConnected,
    /// DAG is too large for exhaustive identification.
    DagTooLarge { nodes: usize, max: usize },
    /// Cyclic graph detected (not a DAG).
    CyclicGraph,
    /// Hedge found: there exists a hedge for P(y|do(x)) proving non-identifiability.
    HedgeFound {
        /// The hedge structure that blocks identification.
        hedge_nodes: Vec<usize>,
    },
}

/// An assumption required for identification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalAssumption {
    /// The assumed condition.
    pub condition: String,
    /// How testable this assumption is.
    pub testability: f64,
}
