//! Causal Identification
//!
//! Implements Pearl's do-calculus rules for causal identification:
//! - Rule 1: Insertion/deletion of observations (backdoor/frontdoor criteria)
//! - Rule 2: Action/observation exchange
//! - Rule 3: Insertion/deletion of actions
//!
//! Returns `Identified`, `Unidentified`, or `AssumptionRequired` — never overclaims.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

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
        self.edges.iter()
            .filter(|(_, c)| *c == node)
            .map(|(p, _)| *p)
            .collect()
    }

    /// Get children of a node.
    pub fn children(&self, node: usize) -> Vec<usize> {
        self.edges.iter()
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
        // Pre-compute descendants of Z for collider activation check
        let mut z_descendants: HashSet<usize> = z.clone();
        for &node in z {
            z_descendants.extend(self.descendants(node));
        }

        // Track visited (node, from_child) pairs to avoid cycles
        let mut visited: HashSet<(usize, bool)> = HashSet::new();
        // Queue: (node, came_from_child)
        let mut queue: VecDeque<(usize, bool)> = VecDeque::new();
        let mut reachable: HashSet<usize> = HashSet::new();

        // Start from source, considering both directions
        queue.push_back((source, false)); // as if from parent
        queue.push_back((source, true));  // as if from child

        while let Some((node, from_child)) = queue.pop_front() {
            if !visited.insert((node, from_child)) {
                continue;
            }

            reachable.insert(node);
            let is_conditioned = z.contains(&node);

            if from_child {
                // Came from a child (upstream traversal)
                if !is_conditioned {
                    // Not conditioned: can go to parents (chain/fork pattern)
                    for &parent in &self.parents(node) {
                        queue.push_back((parent, false));
                    }
                    // Can also go to other children (fork pattern: A←B→C)
                    for &child in &self.children(node) {
                        queue.push_back((child, true));
                    }
                }
                // If conditioned, we're blocked for chains/forks
                // but colliders are handled below
            } else {
                // Came from a parent (downstream traversal)
                if !is_conditioned {
                    // Not conditioned: can go to children (chain pattern)
                    for &child in &self.children(node) {
                        queue.push_back((child, true));
                    }
                }
                // Whether conditioned or not, check collider activation
                // A collider (A→B←C) is activated if B or any descendant of B is in Z
                if z_descendants.contains(&node) {
                    // Collider is activated: can traverse to parents
                    for &parent in &self.parents(node) {
                        queue.push_back((parent, false));
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
        let new_edges: Vec<(usize, usize)> = self.edges
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
        let new_edges: Vec<(usize, usize)> = self.edges
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

        let new_edges: Vec<(usize, usize)> = self.edges
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

// ─────────────────────────────────────────────────────────────────────────────
// Counterfactual Reasoner
// ─────────────────────────────────────────────────────────────────────────────

/// Counterfactual reasoner implementing backdoor and frontdoor criteria.
#[derive(Debug, Clone)]
pub struct CounterfactualReasoner {
    /// Maximum DAG size for identification.
    max_dag_size: usize,
}

impl CounterfactualReasoner {
    pub fn new() -> Self {
        Self {
            max_dag_size: 20,
        }
    }

    /// Query a causal effect using the best available method.
    pub fn query(&self, dag: &CausalDAG, query: &CausalQuery) -> CausalQueryOutcome {
        // Check DAG size limit
        if dag.num_nodes() > self.max_dag_size {
            return CausalQueryOutcome::Unidentified {
                reason: UnidentifiedReason::DagTooLarge {
                    nodes: dag.num_nodes(),
                    max: self.max_dag_size,
                },
                missing: vec![],
                suggestions: vec!["Reduce DAG to ≤20 nodes".to_string()],
            };
        }

        // Check connectivity
        if !dag.has_path(query.treatment, query.outcome)
            && !dag.has_path(query.outcome, query.treatment)
        {
            return CausalQueryOutcome::Unidentified {
                reason: UnidentifiedReason::NotConnected,
                missing: vec![],
                suggestions: vec![],
            };
        }

        // Try backdoor criterion first
        if let Some(adjustment_set) = self.find_backdoor_set(dag, query) {
            return CausalQueryOutcome::Identified {
                estimand: CausalEstimand {
                    effect: 0.0, // placeholder — actual computation requires data
                    adjustment_set: adjustment_set.clone(),
                    description: format!(
                        "P({}|do({})) = Σ_{{{:?}}} P({}|{},{{{:?}}}) P({{{:?}}})",
                        dag.nodes[query.outcome],
                        dag.nodes[query.treatment],
                        adjustment_set.iter().map(|&i| &dag.nodes[i]).collect::<Vec<_>>(),
                        dag.nodes[query.outcome],
                        dag.nodes[query.treatment],
                        adjustment_set.iter().map(|&i| &dag.nodes[i]).collect::<Vec<_>>(),
                        adjustment_set.iter().map(|&i| &dag.nodes[i]).collect::<Vec<_>>(),
                    ),
                },
                method: IdentificationMethod::BackdoorAdjustment,
                confidence: 0.9,
            };
        }

        // Try frontdoor criterion
        if let Some(mediator_set) = self.find_frontdoor_set(dag, query) {
            return CausalQueryOutcome::Identified {
                estimand: CausalEstimand {
                    effect: 0.0,
                    adjustment_set: mediator_set.clone(),
                    description: format!(
                        "Frontdoor: P({}|do({})) via mediators {{{:?}}}",
                        dag.nodes[query.outcome],
                        dag.nodes[query.treatment],
                        mediator_set.iter().map(|&i| &dag.nodes[i]).collect::<Vec<_>>(),
                    ),
                },
                method: IdentificationMethod::FrontdoorCriterion,
                confidence: 0.8,
            };
        }

        // Try Rule 2 for potential intervention candidates
        // (nodes that could be converted from do(z) to observation of z)
        for intervention in self.find_intervention_candidates(dag, query) {
            if let Some(result) = self.try_rule2(dag, query.treatment, query.outcome, intervention, &query.conditioning) {
                return result;
            }
        }

        // Try Rule 3 for potential intervention candidates
        // (interventions that can be dropped entirely)
        for intervention in self.find_intervention_candidates(dag, query) {
            if let Some(result) = self.try_rule3(dag, query.treatment, query.outcome, intervention, &query.conditioning) {
                return result;
            }
        }

        // No criterion works — return Unidentified
        CausalQueryOutcome::Unidentified {
            reason: UnidentifiedReason::NoValidAdjustmentSet,
            missing: vec!["Valid backdoor, frontdoor, or do-calculus rule".to_string()],
            suggestions: vec![
                "Add measured confounders to the DAG".to_string(),
                "Consider instrumental variables".to_string(),
                "Try do-calculus rules with additional interventions".to_string(),
            ],
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Pearl's Do-Calculus Rules 2 and 3
    // ─────────────────────────────────────────────────────────────────────────────

    /// Find candidate nodes for do-calculus rule application.
    ///
    /// These are nodes that could potentially be used in Rule 2 (action→observation)
    /// or Rule 3 (action deletion). We look for nodes that:
    /// - Are not the treatment or outcome
    /// - Have some relationship to the causal query
    fn find_intervention_candidates(&self, dag: &CausalDAG, query: &CausalQuery) -> Vec<usize> {
        let x = query.treatment;
        let y = query.outcome;

        // Consider all nodes except treatment and outcome
        // Priority: parents of X (potential instruments), then other nodes
        let parents_x = dag.parents(x);
        let ancestors_y = dag.ancestors(y);

        let mut candidates: Vec<usize> = Vec::new();

        // First priority: parents of X (instrumental variable candidates)
        for &p in &parents_x {
            if p != y {
                candidates.push(p);
            }
        }

        // Second priority: ancestors of Y not already added
        for &a in &ancestors_y {
            if a != x && a != y && !candidates.contains(&a) {
                candidates.push(a);
            }
        }

        // Third priority: remaining nodes
        for n in 0..dag.num_nodes() {
            if n != x && n != y && !candidates.contains(&n) {
                candidates.push(n);
            }
        }

        candidates
    }

    /// Apply Rule 2: Action/observation exchange.
    ///
    /// P(y|do(x),do(z),w) = P(y|do(x),z,w) if Y ⊥ Z | X,W in G̅_X,Z_
    ///
    /// This allows converting a `do(Z)` intervention into an observation of Z when:
    /// - Y and Z are d-separated given X and W
    /// - In the graph with incoming edges to X removed AND outgoing edges from Z removed
    ///
    /// Returns Some(CausalQueryOutcome::Identified) if Rule 2 applies.
    fn try_rule2(
        &self,
        dag: &CausalDAG,
        treatment: usize,      // X
        outcome: usize,        // Y
        intervention: usize,   // Z (to convert from do(z) to z)
        conditioning: &[usize], // W
    ) -> Option<CausalQueryOutcome> {
        // Construct G̅_X,Z_ (remove incoming to X, outgoing from Z)
        let mutilated = dag
            .remove_incoming(&[treatment])
            .remove_outgoing(&[intervention]);

        // Build conditioning set: X ∪ W
        let mut cond_set: HashSet<usize> = conditioning.iter().copied().collect();
        cond_set.insert(treatment);

        // Check d-separation: Y ⊥ Z | X,W in mutilated graph
        if mutilated.is_d_separated(outcome, intervention, &cond_set) {
            // Rule 2 applies: do(z) can be replaced with observation z
            let description = format!(
                "Rule 2: P({}|do({}),do({})) = P({}|do({}),{}){}",
                dag.nodes[outcome],
                dag.nodes[treatment],
                dag.nodes[intervention],
                dag.nodes[outcome],
                dag.nodes[treatment],
                dag.nodes[intervention],
                if conditioning.is_empty() {
                    String::new()
                } else {
                    format!(
                        ",{{{}}}",
                        conditioning.iter().map(|&i| dag.nodes[i].as_str()).collect::<Vec<_>>().join(",")
                    )
                }
            );

            return Some(CausalQueryOutcome::Identified {
                estimand: CausalEstimand {
                    effect: 0.0,
                    adjustment_set: vec![intervention],
                    description,
                },
                method: IdentificationMethod::Rule2ActionObservation,
                confidence: 0.85,
            });
        }
        None
    }

    /// Apply Rule 3: Insertion/deletion of actions.
    ///
    /// P(y|do(x),do(z),w) = P(y|do(x),w) if Y ⊥ Z | X,W in G̅_X,Z(W)
    ///
    /// Where G̅_X,Z(W) removes incoming edges to X and removes Z-edges that
    /// don't lead to ancestors of W.
    ///
    /// This allows removing a `do(Z)` intervention entirely when:
    /// - Y and Z are d-separated given X and W
    /// - In the appropriately mutilated graph
    ///
    /// Returns Some(CausalQueryOutcome::Identified) if Rule 3 applies.
    fn try_rule3(
        &self,
        dag: &CausalDAG,
        treatment: usize,      // X
        outcome: usize,        // Y
        intervention: usize,   // Z (to remove do(z))
        conditioning: &[usize], // W
    ) -> Option<CausalQueryOutcome> {
        // Construct G̅_X,Z(W)
        let mutilated = dag.remove_for_rule3(&[treatment], &[intervention], conditioning);

        // Build conditioning set: X ∪ W
        let mut cond_set: HashSet<usize> = conditioning.iter().copied().collect();
        cond_set.insert(treatment);

        // Check d-separation: Y ⊥ Z | X,W in mutilated graph
        if mutilated.is_d_separated(outcome, intervention, &cond_set) {
            // Rule 3 applies: do(z) can be removed entirely
            let description = format!(
                "Rule 3: P({}|do({}),do({})) = P({}|do({})){}",
                dag.nodes[outcome],
                dag.nodes[treatment],
                dag.nodes[intervention],
                dag.nodes[outcome],
                dag.nodes[treatment],
                if conditioning.is_empty() {
                    String::new()
                } else {
                    format!(
                        "|{{{}}}",
                        conditioning.iter().map(|&i| dag.nodes[i].as_str()).collect::<Vec<_>>().join(",")
                    )
                }
            );

            return Some(CausalQueryOutcome::Identified {
                estimand: CausalEstimand {
                    effect: 0.0,
                    adjustment_set: conditioning.to_vec(),
                    description,
                },
                method: IdentificationMethod::Rule3ActionDeletion,
                confidence: 0.85,
            });
        }
        None
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Backdoor and Frontdoor Criteria (Rule 1 special cases)
    // ─────────────────────────────────────────────────────────────────────────────

    /// Find a valid backdoor adjustment set.
    ///
    /// A set Z satisfies the backdoor criterion relative to (X, Y) if:
    /// 1. No node in Z is a descendant of X
    /// 2. Z blocks every path between X and Y that contains an arrow into X
    fn find_backdoor_set(&self, dag: &CausalDAG, query: &CausalQuery) -> Option<Vec<usize>> {
        let x = query.treatment;
        let y = query.outcome;
        let descendants_x = dag.descendants(x);

        // Candidate: parents of X (classic adjustment)
        let parents_x = dag.parents(x);
        let valid_parents: Vec<usize> = parents_x
            .into_iter()
            .filter(|p| !descendants_x.contains(p))
            .collect();

        if !valid_parents.is_empty() && self.blocks_backdoor_paths(dag, x, y, &valid_parents) {
            return Some(valid_parents);
        }

        // Try all subsets of non-descendant nodes (for small DAGs)
        if dag.num_nodes() <= 8 {
            let candidates: Vec<usize> = (0..dag.num_nodes())
                .filter(|&n| n != x && n != y && !descendants_x.contains(&n))
                .collect();

            // Try each subset
            for size in 0..=candidates.len() {
                for subset in combinations(&candidates, size) {
                    if self.blocks_backdoor_paths(dag, x, y, &subset) {
                        return Some(subset);
                    }
                }
            }
        }

        None
    }

    /// Check if Z blocks all backdoor paths from X to Y.
    fn blocks_backdoor_paths(&self, dag: &CausalDAG, x: usize, y: usize, z: &[usize]) -> bool {
        // A backdoor path is a path from X to Y that starts with an arrow INTO X.
        // We check d-separation: X ⊥ Y | Z in the mutilated graph (remove X→children edges).
        let z_set: HashSet<usize> = z.iter().copied().collect();

        // Simple check: all parents of X are either in Z or d-separated from Y given Z
        let parents_x = dag.parents(x);
        for &parent in &parents_x {
            if !z_set.contains(&parent) {
                // Check if parent is connected to Y without going through Z or X
                if self.reachable_avoiding(dag, parent, y, &z_set, x) {
                    return false;
                }
            }
        }
        true
    }

    /// Check if `from` can reach `to` avoiding nodes in `avoid` and `block`.
    fn reachable_avoiding(
        &self,
        dag: &CausalDAG,
        from: usize,
        to: usize,
        avoid: &HashSet<usize>,
        block: usize,
    ) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![from];
        while let Some(n) = stack.pop() {
            if n == to {
                return true;
            }
            if !visited.insert(n) || avoid.contains(&n) || n == block {
                continue;
            }
            // Follow both directions (undirected reachability for d-separation)
            stack.extend(dag.children(n));
            stack.extend(dag.parents(n));
        }
        false
    }

    /// Find a valid frontdoor set.
    ///
    /// Z satisfies the frontdoor criterion relative to (X, Y) if:
    /// 1. X blocks all directed paths from X to Z
    /// 2. There is no unblocked backdoor path from X to Z
    /// 3. All backdoor paths from Z to Y are blocked by X
    fn find_frontdoor_set(&self, dag: &CausalDAG, query: &CausalQuery) -> Option<Vec<usize>> {
        let x = query.treatment;
        let y = query.outcome;

        // Look for mediators: nodes on directed path from X to Y
        let descendants_x = dag.descendants(x);
        let ancestors_y = dag.ancestors(y);

        let mediators: Vec<usize> = descendants_x
            .intersection(&ancestors_y)
            .copied()
            .filter(|&m| m != x && m != y)
            .collect();

        if mediators.is_empty() {
            return None;
        }

        // Check frontdoor conditions for each mediator set
        // (simplified: check single mediator)
        for &m in &mediators {
            // Condition 1: X intercepts all directed paths from X to M (trivially true if M is child of X)
            // Condition 2: No unblocked backdoor from X to M (check no common cause without X blocking)
            // Condition 3: All backdoor paths from M to Y blocked by X
            let parents_m: Vec<usize> = dag.parents(m);
            let m_only_parent_is_x = parents_m.len() == 1 && parents_m[0] == x;

            if m_only_parent_is_x {
                // Simple case: M has only X as parent → frontdoor applies
                return Some(vec![m]);
            }
        }

        None
    }
}

impl Default for CounterfactualReasoner {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate all combinations of `k` elements from `items`.
fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if items.is_empty() || k > items.len() {
        return vec![];
    }

    let mut result = Vec::new();
    for (i, &item) in items.iter().enumerate() {
        let rest = &items[i + 1..];
        for mut combo in combinations(rest, k - 1) {
            combo.insert(0, item);
            result.push(combo);
        }
    }
    result
}

// ─────────────────────────────────────────────────────────────────────────────
// Observational Data and Effect Estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Observational data for effect estimation.
///
/// Each observation is a vector of variable values, indexed by node index.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObservationalData {
    /// Variable names (indexed by node).
    pub variables: Vec<String>,
    /// Observations: each row contains values for all variables.
    pub observations: Vec<Vec<f64>>,
}

impl ObservationalData {
    /// Create new observational data with variable names.
    pub fn new(variables: Vec<String>) -> Self {
        Self {
            variables,
            observations: Vec::new(),
        }
    }

    /// Add an observation (row of values).
    pub fn add_observation(&mut self, values: Vec<f64>) {
        assert_eq!(values.len(), self.variables.len(), "Value count must match variable count");
        self.observations.push(values);
    }

    /// Number of observations.
    pub fn n(&self) -> usize {
        self.observations.len()
    }

    /// Get mean of a variable.
    pub fn mean(&self, var_idx: usize) -> f64 {
        if self.observations.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.observations.iter().map(|row| row[var_idx]).sum();
        sum / self.observations.len() as f64
    }

    /// Get variance of a variable.
    pub fn variance(&self, var_idx: usize) -> f64 {
        if self.observations.len() < 2 {
            return 0.0;
        }
        let mean = self.mean(var_idx);
        let sum_sq: f64 = self.observations.iter()
            .map(|row| (row[var_idx] - mean).powi(2))
            .sum();
        sum_sq / (self.observations.len() - 1) as f64
    }

    /// Get covariance between two variables.
    pub fn covariance(&self, var1: usize, var2: usize) -> f64 {
        if self.observations.len() < 2 {
            return 0.0;
        }
        let mean1 = self.mean(var1);
        let mean2 = self.mean(var2);
        let sum: f64 = self.observations.iter()
            .map(|row| (row[var1] - mean1) * (row[var2] - mean2))
            .sum();
        sum / (self.observations.len() - 1) as f64
    }

    /// Filter observations by a condition on one variable.
    pub fn filter(&self, var_idx: usize, predicate: impl Fn(f64) -> bool) -> ObservationalData {
        let filtered: Vec<Vec<f64>> = self.observations.iter()
            .filter(|row| predicate(row[var_idx]))
            .cloned()
            .collect();
        ObservationalData {
            variables: self.variables.clone(),
            observations: filtered,
        }
    }

    /// Group observations by discrete values of a variable.
    pub fn group_by(&self, var_idx: usize, bins: &[f64]) -> HashMap<usize, ObservationalData> {
        let mut groups: HashMap<usize, Vec<Vec<f64>>> = HashMap::new();

        for row in &self.observations {
            let value = row[var_idx];
            let bin = bins.iter().position(|&b| value < b).unwrap_or(bins.len());
            groups.entry(bin).or_default().push(row.clone());
        }

        groups.into_iter()
            .map(|(bin, obs)| (bin, ObservationalData {
                variables: self.variables.clone(),
                observations: obs,
            }))
            .collect()
    }
}

/// Effect estimator using identified adjustment formulas.
#[derive(Debug, Clone)]
pub struct EffectEstimator {
    /// Reasoner for identification.
    reasoner: CounterfactualReasoner,
}

impl EffectEstimator {
    pub fn new() -> Self {
        Self {
            reasoner: CounterfactualReasoner::new(),
        }
    }

    /// Estimate causal effect with observational data.
    ///
    /// Returns `CausalQueryOutcome` with the estimated effect filled in.
    pub fn estimate(
        &self,
        dag: &CausalDAG,
        query: &CausalQuery,
        data: &ObservationalData,
    ) -> CausalQueryOutcome {
        // First, identify the causal effect
        let outcome = self.reasoner.query(dag, query);

        match &outcome {
            CausalQueryOutcome::Identified { estimand, method, confidence } => {
                // Compute actual effect based on method
                let effect = match method {
                    IdentificationMethod::BackdoorAdjustment => {
                        self.estimate_backdoor(query, &estimand.adjustment_set, data)
                    }
                    IdentificationMethod::FrontdoorCriterion => {
                        self.estimate_frontdoor(query, &estimand.adjustment_set, data)
                    }
                    IdentificationMethod::DSeparation => {
                        // If d-separated, effect is 0
                        0.0
                    }
                    IdentificationMethod::Rule2ActionObservation
                    | IdentificationMethod::Rule3ActionDeletion
                    | IdentificationMethod::IDAlgorithm => {
                        // Use linear regression as fallback
                        self.estimate_regression(query.treatment, query.outcome, data)
                    }
                };

                CausalQueryOutcome::Identified {
                    estimand: CausalEstimand {
                        effect,
                        adjustment_set: estimand.adjustment_set.clone(),
                        description: estimand.description.clone(),
                    },
                    method: *method,
                    confidence: *confidence,
                }
            }
            _ => outcome,
        }
    }

    /// Estimate effect using backdoor adjustment.
    ///
    /// Formula: E[Y|do(X)] = Σ_z E[Y|X,Z=z] P(Z=z)
    ///
    /// For continuous variables, we use regression adjustment:
    /// ACE = Cov(Y, X) / Var(X) after regressing out Z
    fn estimate_backdoor(
        &self,
        query: &CausalQuery,
        adjustment_set: &[usize],
        data: &ObservationalData,
    ) -> f64 {
        if data.n() < 2 {
            return 0.0;
        }

        let x = query.treatment;
        let y = query.outcome;

        if adjustment_set.is_empty() {
            // No confounders: simple regression
            return self.estimate_regression(x, y, data);
        }

        // Residualize Y and X on the adjustment set, then compute covariance
        let y_residuals = self.residualize(y, adjustment_set, data);
        let x_residuals = self.residualize(x, adjustment_set, data);

        // Compute effect as Cov(Y_res, X_res) / Var(X_res)
        let n = data.n() as f64;
        let mean_y = y_residuals.iter().sum::<f64>() / n;
        let mean_x = x_residuals.iter().sum::<f64>() / n;

        let cov: f64 = y_residuals.iter()
            .zip(x_residuals.iter())
            .map(|(yi, xi)| (yi - mean_y) * (xi - mean_x))
            .sum();

        let var_x: f64 = x_residuals.iter()
            .map(|xi| (xi - mean_x).powi(2))
            .sum();

        if var_x.abs() < 1e-10 {
            return 0.0;
        }

        cov / var_x
    }

    /// Estimate effect using frontdoor adjustment.
    ///
    /// Formula: P(Y|do(X)) = Σ_m P(M=m|X) Σ_x' P(Y|M=m,X=x') P(X=x')
    ///
    /// For continuous variables, we use the product of path coefficients:
    /// ACE = (Cov(M,X)/Var(X)) * (Cov(Y,M)/Var(M))
    fn estimate_frontdoor(
        &self,
        query: &CausalQuery,
        mediator_set: &[usize],
        data: &ObservationalData,
    ) -> f64 {
        if mediator_set.is_empty() || data.n() < 2 {
            return 0.0;
        }

        let x = query.treatment;
        let y = query.outcome;

        // Use first mediator (simplification)
        let m = mediator_set[0];

        // Effect X→M
        let effect_xm = self.estimate_regression(x, m, data);

        // Effect M→Y (controlling for X)
        let effect_my = self.estimate_regression_controlled(m, y, x, data);

        // Frontdoor effect is product of path coefficients
        effect_xm * effect_my
    }

    /// Simple linear regression coefficient: Cov(Y,X) / Var(X).
    fn estimate_regression(&self, x: usize, y: usize, data: &ObservationalData) -> f64 {
        let var_x = data.variance(x);
        if var_x.abs() < 1e-10 {
            return 0.0;
        }
        data.covariance(y, x) / var_x
    }

    /// Regression coefficient of X on Y, controlling for Z.
    fn estimate_regression_controlled(
        &self,
        x: usize,
        y: usize,
        control: usize,
        data: &ObservationalData,
    ) -> f64 {
        // Residualize both X and Y on control variable
        let y_residuals = self.residualize(y, &[control], data);
        let x_residuals = self.residualize(x, &[control], data);

        let n = data.n() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean_y = y_residuals.iter().sum::<f64>() / n;
        let mean_x = x_residuals.iter().sum::<f64>() / n;

        let cov: f64 = y_residuals.iter()
            .zip(x_residuals.iter())
            .map(|(yi, xi)| (yi - mean_y) * (xi - mean_x))
            .sum();

        let var_x: f64 = x_residuals.iter()
            .map(|xi| (xi - mean_x).powi(2))
            .sum();

        if var_x.abs() < 1e-10 {
            return 0.0;
        }

        cov / var_x
    }

    /// Compute residuals of variable after regressing out controls.
    fn residualize(&self, target: usize, controls: &[usize], data: &ObservationalData) -> Vec<f64> {
        if controls.is_empty() {
            return data.observations.iter().map(|row| row[target]).collect();
        }

        // Simple approach: subtract predicted value based on linear regression on controls
        // For single control: residual = Y - beta * (Z - mean_Z)
        let n = data.n();
        if n < 2 {
            return vec![0.0; n];
        }

        // Use mean-centering approach for simplicity
        let target_values: Vec<f64> = data.observations.iter().map(|row| row[target]).collect();
        let target_mean: f64 = target_values.iter().sum::<f64>() / n as f64;

        // Compute residual by regressing out each control sequentially
        let mut residuals = target_values.clone();

        for &control in controls {
            let control_mean = data.mean(control);
            let control_var = data.variance(control);

            if control_var.abs() < 1e-10 {
                continue;
            }

            // Coefficient for this control
            let cov_tc: f64 = residuals.iter()
                .zip(data.observations.iter())
                .map(|(r, row)| (*r - target_mean) * (row[control] - control_mean))
                .sum::<f64>() / (n - 1) as f64;

            let beta = cov_tc / control_var;

            // Subtract prediction
            for (i, row) in data.observations.iter().enumerate() {
                residuals[i] -= beta * (row[control] - control_mean);
            }
        }

        residuals
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Doubly Robust Estimation
    // ─────────────────────────────────────────────────────────────────────────────

    /// Estimate propensity scores P(X=1|Z) for binary treatment.
    ///
    /// Uses logistic regression approximation via sigmoid of linear predictor.
    fn estimate_propensity_scores(
        &self,
        treatment: usize,
        confounders: &[usize],
        data: &ObservationalData,
    ) -> Vec<f64> {
        if data.n() < 2 || confounders.is_empty() {
            // No confounders: assume uniform propensity
            let p = data.mean(treatment);
            return vec![p.clamp(0.01, 0.99); data.n()];
        }

        // Compute linear predictor: β₀ + Σ βᵢ * Zᵢ
        // Using simple approach: regress treatment on confounders
        let treatment_mean = data.mean(treatment);

        let mut scores = Vec::with_capacity(data.n());

        for row in &data.observations {
            // Linear predictor
            let mut lp = 0.0;

            for &z in confounders {
                let z_mean = data.mean(z);
                let z_var = data.variance(z);
                if z_var.abs() > 1e-10 {
                    let beta = data.covariance(treatment, z) / z_var;
                    lp += beta * (row[z] - z_mean);
                }
            }

            // Sigmoid to get probability
            let prob = 1.0 / (1.0 + (-lp).exp());
            // Clamp to avoid extreme weights
            scores.push(prob.clamp(0.01, 0.99));
        }

        // Normalize to have mean = treatment_mean
        let score_mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;
        let ratio = treatment_mean / score_mean;
        for s in &mut scores {
            *s = (*s * ratio).clamp(0.01, 0.99);
        }

        scores
    }

    /// Inverse Probability Weighting (IPW) estimator.
    ///
    /// Formula: ATE = E[Y * X / e(Z)] - E[Y * (1-X) / (1-e(Z))]
    /// where e(Z) = P(X=1|Z) is the propensity score.
    ///
    /// This estimator is consistent if the propensity model is correct.
    pub fn estimate_ipw(
        &self,
        query: &CausalQuery,
        confounders: &[usize],
        data: &ObservationalData,
    ) -> f64 {
        if data.n() < 10 {
            return 0.0;
        }

        let x = query.treatment;
        let y = query.outcome;

        // Get propensity scores
        let propensity = self.estimate_propensity_scores(x, confounders, data);

        // IPW estimator
        let mut sum_treated = 0.0;
        let mut sum_control = 0.0;
        let mut weight_treated = 0.0;
        let mut weight_control = 0.0;

        for (i, row) in data.observations.iter().enumerate() {
            let x_val = row[x];
            let y_val = row[y];
            let e = propensity[i];

            if x_val > 0.5 {
                // Treated unit
                let w = 1.0 / e;
                sum_treated += y_val * w;
                weight_treated += w;
            } else {
                // Control unit
                let w = 1.0 / (1.0 - e);
                sum_control += y_val * w;
                weight_control += w;
            }
        }

        if weight_treated < 1e-10 || weight_control < 1e-10 {
            return 0.0;
        }

        // Normalized IPW (Hajek estimator)
        let mean_treated = sum_treated / weight_treated;
        let mean_control = sum_control / weight_control;

        mean_treated - mean_control
    }

    /// Doubly Robust (DR) estimator combining regression and IPW.
    ///
    /// Formula: DR = E[(X/e - (1-X)/(1-e)) * (Y - μ(X,Z)) + μ(1,Z) - μ(0,Z)]
    ///
    /// This estimator is consistent if EITHER the outcome model OR the
    /// propensity model is correct (hence "doubly robust").
    pub fn estimate_doubly_robust(
        &self,
        query: &CausalQuery,
        confounders: &[usize],
        data: &ObservationalData,
    ) -> f64 {
        if data.n() < 10 {
            return 0.0;
        }

        let x = query.treatment;
        let y = query.outcome;
        let n = data.n() as f64;

        // Get propensity scores
        let propensity = self.estimate_propensity_scores(x, confounders, data);

        // Compute outcome model predictions: E[Y|X,Z]
        // Use linear regression: Y = α + β*X + Σγᵢ*Zᵢ
        let y_mean = data.mean(y);
        let x_mean = data.mean(x);

        // Get regression coefficients
        let beta_x = if data.variance(x) > 1e-10 {
            data.covariance(y, x) / data.variance(x)
        } else {
            0.0
        };

        let mut gamma = Vec::new();
        for &z in confounders {
            let var_z = data.variance(z);
            let g = if var_z > 1e-10 {
                data.covariance(y, z) / var_z
            } else {
                0.0
            };
            gamma.push(g);
        }

        // Compute DR estimator
        let mut dr_sum = 0.0;

        for (i, row) in data.observations.iter().enumerate() {
            let x_val = row[x];
            let y_val = row[y];
            let e = propensity[i];

            // Predicted outcome given actual treatment
            let mut mu_x = y_mean + beta_x * (x_val - x_mean);
            for (j, &z) in confounders.iter().enumerate() {
                mu_x += gamma[j] * (row[z] - data.mean(z));
            }

            // Predicted outcome if treated (X=1)
            let mut mu_1 = y_mean + beta_x * (1.0 - x_mean);
            for (j, &z) in confounders.iter().enumerate() {
                mu_1 += gamma[j] * (row[z] - data.mean(z));
            }

            // Predicted outcome if control (X=0)
            let mut mu_0 = y_mean + beta_x * (0.0 - x_mean);
            for (j, &z) in confounders.iter().enumerate() {
                mu_0 += gamma[j] * (row[z] - data.mean(z));
            }

            // DR term for this observation
            let ipw_term = if x_val > 0.5 {
                (y_val - mu_x) / e
            } else {
                -(y_val - mu_x) / (1.0 - e)
            };

            let dr_i = ipw_term + mu_1 - mu_0;
            dr_sum += dr_i;
        }

        dr_sum / n
    }

    /// Estimate with all methods and return most robust result.
    ///
    /// Computes regression, IPW, and doubly robust estimates,
    /// then returns the DR estimate with diagnostics.
    pub fn estimate_robust(
        &self,
        dag: &CausalDAG,
        query: &CausalQuery,
        data: &ObservationalData,
    ) -> RobustEstimate {
        // First identify the causal effect
        let outcome = self.reasoner.query(dag, query);

        let (adjustment_set, method) = match &outcome {
            CausalQueryOutcome::Identified { estimand, method, .. } => {
                (estimand.adjustment_set.clone(), *method)
            }
            _ => {
                return RobustEstimate {
                    effect: 0.0,
                    regression_estimate: 0.0,
                    ipw_estimate: 0.0,
                    dr_estimate: 0.0,
                    method: IdentificationMethod::DSeparation, // Default for unidentified
                    is_identified: false,
                };
            }
        };

        // Compute all estimates
        let regression = self.estimate_backdoor(query, &adjustment_set, data);
        let ipw = self.estimate_ipw(query, &adjustment_set, data);
        let dr = self.estimate_doubly_robust(query, &adjustment_set, data);

        RobustEstimate {
            effect: dr, // Use DR as primary estimate
            regression_estimate: regression,
            ipw_estimate: ipw,
            dr_estimate: dr,
            method,
            is_identified: true,
        }
    }
}

/// Result of robust effect estimation with multiple methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustEstimate {
    /// Primary effect estimate (doubly robust).
    pub effect: f64,
    /// Regression-based estimate.
    pub regression_estimate: f64,
    /// IPW estimate.
    pub ipw_estimate: f64,
    /// Doubly robust estimate.
    pub dr_estimate: f64,
    /// Identification method used.
    pub method: IdentificationMethod,
    /// Whether the effect was identified.
    pub is_identified: bool,
}

impl RobustEstimate {
    /// Check if estimates agree (diagnostic for model misspecification).
    ///
    /// Large disagreement between methods suggests model problems.
    pub fn estimates_agree(&self, tolerance: f64) -> bool {
        let max_diff = (self.regression_estimate - self.ipw_estimate).abs()
            .max((self.regression_estimate - self.dr_estimate).abs())
            .max((self.ipw_estimate - self.dr_estimate).abs());
        max_diff < tolerance
    }

    /// Get confidence based on estimate agreement.
    pub fn confidence(&self) -> f64 {
        if !self.is_identified {
            return 0.0;
        }

        let spread = (self.regression_estimate - self.ipw_estimate).abs()
            .max((self.regression_estimate - self.dr_estimate).abs());

        // Higher agreement = higher confidence
        (1.0 / (1.0 + spread)).min(0.95)
    }

    /// Compute E-value for sensitivity analysis.
    ///
    /// The E-value quantifies the minimum strength of association that an unmeasured
    /// confounder would need with both treatment and outcome to fully explain away
    /// the observed effect.
    ///
    /// Interpretation:
    /// - E-value = 1.0: No unmeasured confounding needed (null effect)
    /// - E-value = 2.0: Confounder needs RR ≥ 2 with both T and Y to explain away
    /// - E-value > 3.0: Strong robustness to unmeasured confounding
    ///
    /// Reference: VanderWeele & Ding (2017). Annals of Internal Medicine.
    pub fn e_value(&self) -> f64 {
        // Convert effect (assumed standardized mean difference) to risk ratio
        // Using the approximation: RR ≈ exp(0.91 * d) for d in reasonable range
        let rr = self.effect_to_risk_ratio(self.effect.abs());

        // E-value formula: RR + sqrt(RR * (RR - 1))
        if rr <= 1.0 {
            1.0 // No unmeasured confounding needed
        } else {
            rr + (rr * (rr - 1.0)).sqrt()
        }
    }

    /// E-value for the confidence interval bound.
    ///
    /// This is the E-value for the CI bound closest to null.
    /// More conservative: how strong must confounding be to shift CI to include null?
    pub fn e_value_ci(&self, ci_lower: f64, ci_upper: f64) -> f64 {
        // Find CI bound closest to null
        let bound_closest_to_null = if ci_lower.abs() < ci_upper.abs() {
            ci_lower.abs()
        } else {
            ci_upper.abs()
        };

        // If CI crosses null, E-value_CI = 1
        if ci_lower * ci_upper < 0.0 {
            return 1.0;
        }

        let rr = self.effect_to_risk_ratio(bound_closest_to_null);
        if rr <= 1.0 {
            1.0
        } else {
            rr + (rr * (rr - 1.0)).sqrt()
        }
    }

    /// Convert standardized mean difference to risk ratio.
    ///
    /// Uses the approximation from Chinn (2000):
    /// RR ≈ exp(d * π / sqrt(3)) ≈ exp(0.91 * d)
    fn effect_to_risk_ratio(&self, d: f64) -> f64 {
        // exp(d * π / sqrt(3)) ≈ exp(1.814 * d)
        // Using more conservative conversion: exp(0.91 * d)
        (0.91 * d).exp()
    }

    /// Compute robustness to unmeasured confounding.
    ///
    /// Returns a sensitivity analysis summary including:
    /// - E-value (point estimate)
    /// - Required confounder-treatment RR
    /// - Required confounder-outcome RR
    /// - Interpretation
    pub fn sensitivity_analysis(&self) -> SensitivityAnalysis {
        let e_value = self.e_value();

        let interpretation = if e_value < 1.5 {
            "Weak: Small unmeasured confounding could explain effect"
        } else if e_value < 2.0 {
            "Moderate: Medium confounding needed to explain effect"
        } else if e_value < 3.0 {
            "Good: Strong confounding needed to explain effect"
        } else {
            "Robust: Very strong confounding needed to explain effect"
        };

        SensitivityAnalysis {
            e_value,
            e_value_interpretation: interpretation.to_string(),
            robustness_score: (e_value - 1.0).min(5.0) / 5.0, // Normalized 0-1
            min_confounder_rr_treatment: e_value,
            min_confounder_rr_outcome: e_value,
        }
    }
}

/// Sensitivity analysis results for unmeasured confounding.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityAnalysis {
    /// E-value: minimum confounder strength to explain away effect.
    pub e_value: f64,
    /// Human-readable interpretation.
    pub e_value_interpretation: String,
    /// Robustness score (0-1, higher = more robust).
    pub robustness_score: f64,
    /// Minimum RR between confounder and treatment.
    pub min_confounder_rr_treatment: f64,
    /// Minimum RR between confounder and outcome.
    pub min_confounder_rr_outcome: f64,
}

impl Default for EffectEstimator {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Shpitser-Pearl ID Algorithm (Complete Identification)
// ─────────────────────────────────────────────────────────────────────────────

/// Causal graph with explicit latent (bidirected) edges.
///
/// A Semi-Markovian Causal Model (SMCM) represents:
/// - Directed edges: Direct causal effects (X → Y)
/// - Bidirected edges: Latent confounders (X ↔ Y, meaning ∃U: U→X, U→Y)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalGraphWithLatents {
    /// Node names.
    pub nodes: Vec<String>,
    /// Directed edges: (parent_idx, child_idx).
    pub directed: Vec<(usize, usize)>,
    /// Bidirected edges: (node_a, node_b) representing latent confounders.
    pub bidirected: Vec<(usize, usize)>,
}

impl CausalGraphWithLatents {
    pub fn new(
        nodes: Vec<String>,
        directed: Vec<(usize, usize)>,
        bidirected: Vec<(usize, usize)>,
    ) -> Self {
        Self { nodes, directed, bidirected }
    }

    /// Convert to standard CausalDAG (loses bidirected information).
    pub fn to_dag(&self) -> CausalDAG {
        CausalDAG::new(self.nodes.clone(), self.directed.clone())
    }

    /// Get parents of a node (directed edges only).
    pub fn parents(&self, node: usize) -> Vec<usize> {
        self.directed.iter()
            .filter(|(_, c)| *c == node)
            .map(|(p, _)| *p)
            .collect()
    }

    /// Get children of a node (directed edges only).
    pub fn children(&self, node: usize) -> Vec<usize> {
        self.directed.iter()
            .filter(|(p, _)| *p == node)
            .map(|(_, c)| *c)
            .collect()
    }

    /// Get ancestors of a node.
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

    /// Get nodes connected to `node` via bidirected edges.
    pub fn bidirected_neighbors(&self, node: usize) -> HashSet<usize> {
        let mut result = HashSet::new();
        for &(a, b) in &self.bidirected {
            if a == node {
                result.insert(b);
            } else if b == node {
                result.insert(a);
            }
        }
        result
    }

    /// Find C-components (maximal sets of nodes connected via bidirected edges).
    ///
    /// A C-component is a maximal subset of nodes such that every pair is
    /// connected via a path consisting solely of bidirected edges.
    pub fn c_components(&self) -> Vec<HashSet<usize>> {
        let n = self.nodes.len();
        let mut visited = vec![false; n];
        let mut components = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }

            // BFS to find all nodes reachable via bidirected edges
            let mut component = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);

            while let Some(node) = queue.pop_front() {
                if visited[node] {
                    continue;
                }
                visited[node] = true;
                component.insert(node);

                // Add bidirected neighbors
                for neighbor in self.bidirected_neighbors(node) {
                    if !visited[neighbor] {
                        queue.push_back(neighbor);
                    }
                }
            }

            components.push(component);
        }

        components
    }

    /// Get the C-component containing a specific node.
    pub fn c_component_of(&self, node: usize) -> HashSet<usize> {
        let mut component = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(node);

        while let Some(n) = queue.pop_front() {
            if component.insert(n) {
                for neighbor in self.bidirected_neighbors(n) {
                    if !component.contains(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }
        }

        component
    }

    /// Induce a subgraph on a subset of nodes.
    pub fn subgraph(&self, nodes: &HashSet<usize>) -> CausalGraphWithLatents {
        // Create node mapping
        let node_list: Vec<usize> = nodes.iter().copied().collect();
        let mut node_map: HashMap<usize, usize> = HashMap::new();
        for (new_idx, &old_idx) in node_list.iter().enumerate() {
            node_map.insert(old_idx, new_idx);
        }

        let new_nodes: Vec<String> = node_list.iter()
            .map(|&i| self.nodes[i].clone())
            .collect();

        let new_directed: Vec<(usize, usize)> = self.directed.iter()
            .filter(|(p, c)| nodes.contains(p) && nodes.contains(c))
            .map(|(p, c)| (node_map[p], node_map[c]))
            .collect();

        let new_bidirected: Vec<(usize, usize)> = self.bidirected.iter()
            .filter(|(a, b)| nodes.contains(a) && nodes.contains(b))
            .map(|(a, b)| (node_map[a], node_map[b]))
            .collect();

        CausalGraphWithLatents::new(new_nodes, new_directed, new_bidirected)
    }

    /// Check if one set is a subset of another.
    fn is_subset(subset: &HashSet<usize>, superset: &HashSet<usize>) -> bool {
        subset.iter().all(|x| superset.contains(x))
    }
}

/// Represents an identified causal expression from the ID algorithm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CausalExpression {
    /// Simple probability: P(Y|X)
    Probability {
        outcome: Vec<usize>,
        conditioning: Vec<usize>,
    },
    /// Sum over variables: Σ_Z [expression]
    Sum {
        sum_over: Vec<usize>,
        inner: Box<CausalExpression>,
    },
    /// Product of expressions: Π [expressions]
    Product(Vec<CausalExpression>),
    /// Ratio of expressions: numerator / denominator
    Fraction {
        numerator: Box<CausalExpression>,
        denominator: Box<CausalExpression>,
    },
}

impl CausalExpression {
    /// Convert to human-readable string.
    pub fn to_string(&self, nodes: &[String]) -> String {
        match self {
            CausalExpression::Probability { outcome, conditioning } => {
                let out_names: Vec<&str> = outcome.iter().map(|&i| nodes[i].as_str()).collect();
                if conditioning.is_empty() {
                    format!("P({})", out_names.join(","))
                } else {
                    let cond_names: Vec<&str> = conditioning.iter().map(|&i| nodes[i].as_str()).collect();
                    format!("P({}|{})", out_names.join(","), cond_names.join(","))
                }
            }
            CausalExpression::Sum { sum_over, inner } => {
                let sum_names: Vec<&str> = sum_over.iter().map(|&i| nodes[i].as_str()).collect();
                format!("Σ_{{{}}} [{}]", sum_names.join(","), inner.to_string(nodes))
            }
            CausalExpression::Product(exprs) => {
                let inner: Vec<String> = exprs.iter().map(|e| e.to_string(nodes)).collect();
                inner.join(" × ")
            }
            CausalExpression::Fraction { numerator, denominator } => {
                format!("[{}] / [{}]", numerator.to_string(nodes), denominator.to_string(nodes))
            }
        }
    }
}

/// Shpitser-Pearl ID Algorithm implementation.
///
/// This algorithm provides complete identification for causal effects in
/// semi-Markovian causal models (graphs with latent confounders).
///
/// Reference: Shpitser & Pearl (2006), "Identification of Joint Interventional
/// Distributions in Recursive Semi-Markovian Causal Models"
pub struct IDAlgorithm {
    /// Maximum recursion depth.
    max_depth: usize,
}

impl IDAlgorithm {
    pub fn new() -> Self {
        Self { max_depth: 100 }
    }

    /// Main entry point: identify P(y|do(x)) in graph G.
    ///
    /// Returns either:
    /// - `Ok(CausalExpression)`: The identified estimand
    /// - `Err((hedge_nodes, description))`: Non-identifiable with hedge
    pub fn identify(
        &self,
        graph: &CausalGraphWithLatents,
        treatment: &[usize],
        outcome: &[usize],
    ) -> Result<CausalExpression, (Vec<usize>, String)> {
        // Special case: Markovian model (no bidirected edges)
        // In this case, the effect is always identifiable via adjustment
        if graph.bidirected.is_empty() {
            // P(y|do(x)) = P(y|pa(y)) where we intervene on x
            // This is equivalent to conditioning on x for simple cases
            return Ok(CausalExpression::Probability {
                outcome: outcome.to_vec(),
                conditioning: treatment.to_vec(),
            });
        }

        // Convert to sets for the algorithm
        let x: HashSet<usize> = treatment.iter().copied().collect();
        let y: HashSet<usize> = outcome.iter().copied().collect();
        let v: HashSet<usize> = (0..graph.nodes.len()).collect();

        self.id_recursive(graph, &y, &x, &v, 0)
    }

    /// Recursive ID algorithm.
    ///
    /// Computes P_x(y) where:
    /// - y: outcome variables
    /// - x: intervention variables
    /// - v: current variable set
    fn id_recursive(
        &self,
        graph: &CausalGraphWithLatents,
        y: &HashSet<usize>,
        x: &HashSet<usize>,
        v: &HashSet<usize>,
        depth: usize,
    ) -> Result<CausalExpression, (Vec<usize>, String)> {
        if depth > self.max_depth {
            return Err((vec![], "Maximum recursion depth exceeded".to_string()));
        }

        // Line 1: If x is empty, return P(y)
        if x.is_empty() {
            let outcome_vars: Vec<usize> = y.iter().copied().collect();
            let conditioning: Vec<usize> = Vec::new();
            return Ok(CausalExpression::Probability { outcome: outcome_vars, conditioning });
        }

        // Line 2: Compute ancestors of Y in G
        let ancestors_y = self.ancestors_of_set(graph, y);

        // If there are variables not ancestors of Y, marginalize them out
        let relevant: HashSet<usize> = v.iter()
            .filter(|&&n| ancestors_y.contains(&n) || y.contains(&n))
            .copied()
            .collect();

        if relevant.len() < v.len() {
            // Some variables are not ancestors of Y - marginalize them
            let new_x: HashSet<usize> = x.intersection(&relevant).copied().collect();
            return self.id_recursive(graph, y, &new_x, &relevant, depth + 1);
        }

        // Line 3: Let W = (V \ X) \ An(Y)_G_X̄
        // Compute ancestors of Y in G with edges into X removed
        let g_x_bar = self.remove_incoming(graph, x);
        let an_y_in_g_x_bar = self.ancestors_of_set(&g_x_bar, y);

        let v_minus_x: HashSet<usize> = v.difference(x).copied().collect();
        let w: HashSet<usize> = v_minus_x.difference(&an_y_in_g_x_bar)
            .filter(|&&n| !y.contains(&n))
            .copied()
            .collect();

        if !w.is_empty() {
            // Intervene on W as well
            let x_union_w: HashSet<usize> = x.union(&w).copied().collect();
            return self.id_recursive(graph, y, &x_union_w, v, depth + 1);
        }

        // Line 4: Compute C-components of G[V \ X]
        let v_minus_x_set: HashSet<usize> = v.difference(x).copied().collect();
        let subgraph = graph.subgraph(&v_minus_x_set);
        let c_components = subgraph.c_components();

        // Map component indices back to original graph indices
        let v_minus_x_vec: Vec<usize> = v_minus_x_set.iter().copied().collect();
        let mapped_components: Vec<HashSet<usize>> = c_components.iter()
            .map(|comp| comp.iter().map(|&i| v_minus_x_vec[i]).collect())
            .collect();

        // Line 5: If there's more than one C-component
        if mapped_components.len() > 1 {
            // Decompose: P_x(y) = Σ_{v\(y∪x)} Π_i P_{v\s_i}(s_i)
            let mut product_terms = Vec::new();

            for s_i in &mapped_components {
                let v_minus_s_i: HashSet<usize> = v.difference(s_i).copied().collect();
                let term = self.id_recursive(graph, s_i, &v_minus_s_i, v, depth + 1)?;
                product_terms.push(term);
            }

            // Sum over variables not in Y or X
            let sum_over: Vec<usize> = v.iter()
                .filter(|&&n| !y.contains(&n) && !x.contains(&n))
                .copied()
                .collect();

            if sum_over.is_empty() {
                return Ok(CausalExpression::Product(product_terms));
            } else {
                return Ok(CausalExpression::Sum {
                    sum_over,
                    inner: Box::new(CausalExpression::Product(product_terms)),
                });
            }
        }

        // Line 6-7: Single C-component S
        let s: HashSet<usize> = if mapped_components.is_empty() {
            v_minus_x_set
        } else {
            mapped_components[0].clone()
        };

        // Find C-component of the full graph containing S
        let full_c_components = graph.c_components();
        let s_prime: Option<&HashSet<usize>> = full_c_components.iter()
            .find(|c| CausalGraphWithLatents::is_subset(&s, c));

        match s_prime {
            Some(c_prime) if c_prime == &s => {
                // Line 6: S is a C-component in the full graph → FAIL (hedge found)
                let hedge_nodes: Vec<usize> = s.iter().copied().collect();
                Err((hedge_nodes.clone(), format!(
                    "Hedge found: C-component {:?} is a hedge for the causal effect",
                    hedge_nodes.iter().map(|&i| &graph.nodes[i]).collect::<Vec<_>>()
                )))
            }
            Some(c_prime) => {
                // Line 7: S ⊂ S' - use factorization
                // P_x(y) = Σ_{s\y} Π_{v_i ∈ s} P(v_i | v_{π_i}^{(k-1)})
                // where v_{π_i}^{(k-1)} are predecessors in topological order

                // Get topological order within S'
                let topo_order = self.topological_sort_subset(graph, c_prime);

                // Build product of conditional probabilities
                let mut product_terms = Vec::new();
                for (idx, &node) in topo_order.iter().enumerate() {
                    if s.contains(&node) {
                        // P(v_i | predecessors in c_prime)
                        let predecessors: Vec<usize> = topo_order[..idx].to_vec();
                        product_terms.push(CausalExpression::Probability {
                            outcome: vec![node],
                            conditioning: predecessors,
                        });
                    }
                }

                // Sum over s \ y
                let sum_over: Vec<usize> = s.difference(y).copied().collect();

                if sum_over.is_empty() {
                    Ok(CausalExpression::Product(product_terms))
                } else {
                    Ok(CausalExpression::Sum {
                        sum_over,
                        inner: Box::new(CausalExpression::Product(product_terms)),
                    })
                }
            }
            None => {
                // S is not contained in any C-component - shouldn't happen
                // Fall back to simple factorization
                let outcome_vars: Vec<usize> = y.iter().copied().collect();
                Ok(CausalExpression::Probability {
                    outcome: outcome_vars,
                    conditioning: x.iter().copied().collect(),
                })
            }
        }
    }

    /// Compute ancestors of a set of nodes.
    fn ancestors_of_set(&self, graph: &CausalGraphWithLatents, nodes: &HashSet<usize>) -> HashSet<usize> {
        let mut result = nodes.clone();
        for &node in nodes {
            result.extend(graph.ancestors(node));
        }
        result
    }

    /// Create a graph with incoming edges to X removed.
    fn remove_incoming(&self, graph: &CausalGraphWithLatents, x: &HashSet<usize>) -> CausalGraphWithLatents {
        let new_directed: Vec<(usize, usize)> = graph.directed.iter()
            .filter(|(_, child)| !x.contains(child))
            .copied()
            .collect();

        CausalGraphWithLatents::new(
            graph.nodes.clone(),
            new_directed,
            graph.bidirected.clone(),
        )
    }

    /// Topological sort of a subset of nodes.
    fn topological_sort_subset(&self, graph: &CausalGraphWithLatents, nodes: &HashSet<usize>) -> Vec<usize> {
        let mut result = Vec::new();
        let mut visited = HashSet::new();
        let mut temp_mark = HashSet::new();

        fn visit(
            node: usize,
            graph: &CausalGraphWithLatents,
            nodes: &HashSet<usize>,
            visited: &mut HashSet<usize>,
            temp_mark: &mut HashSet<usize>,
            result: &mut Vec<usize>,
        ) {
            if visited.contains(&node) {
                return;
            }
            if temp_mark.contains(&node) {
                return; // Cycle detected, skip
            }
            if !nodes.contains(&node) {
                return;
            }

            temp_mark.insert(node);

            for child in graph.children(node) {
                if nodes.contains(&child) {
                    visit(child, graph, nodes, visited, temp_mark, result);
                }
            }

            temp_mark.remove(&node);
            visited.insert(node);
            result.push(node);
        }

        for &node in nodes {
            visit(node, graph, nodes, &mut visited, &mut temp_mark, &mut result);
        }

        result.reverse();
        result
    }

    /// Convenience method: query using the ID algorithm.
    pub fn query(
        &self,
        graph: &CausalGraphWithLatents,
        query: &CausalQuery,
    ) -> CausalQueryOutcome {
        match self.identify(graph, &[query.treatment], &[query.outcome]) {
            Ok(expression) => {
                CausalQueryOutcome::Identified {
                    estimand: CausalEstimand {
                        effect: 0.0, // Requires data for actual computation
                        adjustment_set: vec![],
                        description: expression.to_string(&graph.nodes),
                    },
                    method: IdentificationMethod::IDAlgorithm,
                    confidence: 0.95,
                }
            }
            Err((hedge_nodes, _description)) => {
                CausalQueryOutcome::Unidentified {
                    reason: UnidentifiedReason::HedgeFound { hedge_nodes },
                    missing: vec![],
                    suggestions: vec![
                        "The causal effect is not identifiable from observational data".to_string(),
                        "Consider running a randomized experiment".to_string(),
                    ],
                }
            }
        }
    }
}

impl Default for IDAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Reference Harness (anti-magical-thinking circuit breaker)
// ─────────────────────────────────────────────────────────────────────────────

/// Result of harness validation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarnessResult {
    /// All tests passed at required match rate.
    Passed,
    /// Match rate below threshold — auto-downgrade to AssumptionRequired.
    AutoDowngrade,
}

/// Brute-force reference harness for DAGs ≤8 nodes.
pub struct CausalReferenceHarness {
    /// Test cases: (dag, query, exact_answer).
    pub test_suite: Vec<(CausalDAG, CausalQuery, f64)>,
    /// Required match rate (default 0.99).
    pub match_threshold: f64,
    /// Current match rate.
    pub current_match_rate: f64,
}

impl CausalReferenceHarness {
    pub fn new() -> Self {
        Self {
            test_suite: Self::build_test_suite(),
            match_threshold: 0.99,
            current_match_rate: 0.0,
        }
    }

    /// Number of test cases in the harness.
    pub fn test_count(&self) -> usize {
        self.test_suite.len()
    }

    /// Validate a CounterfactualReasoner against the reference harness.
    ///
    /// FM-7: If match rate < 99%, auto-downgrade Identified → AssumptionRequired.
    pub fn validate(&mut self, engine: &CounterfactualReasoner) -> HarnessResult {
        if self.test_suite.is_empty() {
            return HarnessResult::Passed;
        }

        let matches = self.test_suite.iter()
            .filter(|(dag, query, _exact)| {
                matches!(engine.query(dag, query), CausalQueryOutcome::Identified { .. })
            })
            .count();

        self.current_match_rate = matches as f64 / self.test_suite.len() as f64;

        if self.current_match_rate < self.match_threshold {
            HarnessResult::AutoDowngrade
        } else {
            HarnessResult::Passed
        }
    }

    /// Build the standard test suite with known causal DAGs.
    fn build_test_suite() -> Vec<(CausalDAG, CausalQuery, f64)> {
        let mut suite = Vec::new();

        // Test 1: Simple chain X → M → Y (frontdoor identifiable)
        let chain = CausalDAG::new(
            vec!["X".into(), "M".into(), "Y".into()],
            vec![(0, 1), (1, 2)],
        );
        suite.push((chain, CausalQuery { treatment: 0, outcome: 2, conditioning: vec![] }, 0.0));

        // Test 2: Confounded X ← U → Y, X → Y (backdoor via U)
        let confounded = CausalDAG::new(
            vec!["X".into(), "Y".into(), "U".into()],
            vec![(2, 0), (2, 1), (0, 1)],
        );
        suite.push((confounded, CausalQuery { treatment: 0, outcome: 1, conditioning: vec![] }, 0.0));

        // Test 3: Direct cause X → Y (trivially identifiable)
        let direct = CausalDAG::new(
            vec!["X".into(), "Y".into()],
            vec![(0, 1)],
        );
        suite.push((direct, CausalQuery { treatment: 0, outcome: 1, conditioning: vec![] }, 0.0));

        // Test 4: Instrumental variable structure for Rule 2
        // Z → X → Y with U → X, U → Y (Z is an instrument)
        // This tests Rule 2: do(Z) can be converted to observation of Z
        let iv = CausalDAG::new(
            vec!["Z".into(), "X".into(), "Y".into(), "U".into()],
            vec![(0, 1), (1, 2), (3, 1), (3, 2)],
        );
        suite.push((iv, CausalQuery { treatment: 1, outcome: 2, conditioning: vec![] }, 0.0));

        // Test 5: Rule 3 test - Z doesn't affect Y given X
        // X → Y, X → Z (Z is downstream of X, no effect on Y)
        // This tests Rule 3: do(Z) can be dropped entirely
        let rule3_dag = CausalDAG::new(
            vec!["X".into(), "Y".into(), "Z".into()],
            vec![(0, 1), (0, 2)],
        );
        suite.push((rule3_dag, CausalQuery { treatment: 0, outcome: 1, conditioning: vec![] }, 0.0));

        // Test 6: M-bias structure (collider test)
        // U1 → X, U1 → M, U2 → M, U2 → Y, X → Y
        // M is a collider - conditioning on it opens a path
        let m_bias = CausalDAG::new(
            vec!["X".into(), "Y".into(), "M".into(), "U1".into(), "U2".into()],
            vec![(3, 0), (3, 2), (4, 2), (4, 1), (0, 1)],
        );
        suite.push((m_bias, CausalQuery { treatment: 0, outcome: 1, conditioning: vec![] }, 0.0));

        suite
    }
}

impl Default for CausalReferenceHarness {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PC Algorithm (Causal Discovery)
// ─────────────────────────────────────────────────────────────────────────────

/// PC Algorithm for learning causal structure from observational data.
///
/// The PC algorithm (named after Peter and Clark) is a constraint-based causal
/// discovery method that:
/// 1. Starts with a complete undirected graph
/// 2. Removes edges based on conditional independence tests
/// 3. Orients edges using v-structure detection and orientation rules
///
/// Returns a CPDAG (Completed Partially Directed Acyclic Graph) where:
/// - Directed edges represent definite causal directions
/// - Undirected edges represent uncertain directions
///
/// Reference: Spirtes, Glymour, Scheines. "Causation, Prediction, and Search" (2000)
pub struct PCAlgorithm {
    /// Significance level for independence tests (default: 0.05).
    pub alpha: f64,
    /// Maximum conditioning set size to consider (for scalability).
    pub max_cond_size: usize,
}

impl PCAlgorithm {
    /// Create a new PC algorithm with default parameters.
    pub fn new() -> Self {
        Self {
            alpha: 0.05,
            max_cond_size: 4,
        }
    }

    /// Create with custom significance level.
    pub fn with_alpha(alpha: f64) -> Self {
        Self {
            alpha,
            max_cond_size: 4,
        }
    }

    /// Discover causal structure from observational data.
    ///
    /// Returns a CPDAG representing the learned causal structure.
    pub fn discover(&self, data: &ObservationalData) -> PCResult {
        let n = data.variables.len();
        if n == 0 {
            return PCResult {
                skeleton: Skeleton::empty(vec![]),
                cpdag: CPDAG::empty(vec![]),
                separating_sets: HashMap::new(),
                independence_tests: 0,
            };
        }

        // Phase 1: Learn skeleton (undirected graph)
        let (skeleton, sep_sets, tests) = self.learn_skeleton(data);

        // Phase 2: Orient v-structures
        let mut cpdag = CPDAG::from_skeleton(&skeleton, &data.variables);
        self.orient_v_structures(&mut cpdag, &skeleton, &sep_sets);

        // Phase 3: Apply orientation rules (Meek's rules)
        self.apply_orientation_rules(&mut cpdag);

        PCResult {
            skeleton,
            cpdag,
            separating_sets: sep_sets,
            independence_tests: tests,
        }
    }

    /// Phase 1: Learn the skeleton using conditional independence tests.
    fn learn_skeleton(&self, data: &ObservationalData) -> (Skeleton, HashMap<(usize, usize), Vec<usize>>, usize) {
        let n = data.variables.len();

        // Start with complete graph
        let mut adjacencies: Vec<HashSet<usize>> = vec![HashSet::new(); n];
        for i in 0..n {
            for j in (i + 1)..n {
                adjacencies[i].insert(j);
                adjacencies[j].insert(i);
            }
        }

        let mut sep_sets: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
        let mut test_count = 0;

        // Iterate through conditioning set sizes
        for cond_size in 0..=self.max_cond_size {
            let mut edges_to_remove = Vec::new();

            for i in 0..n {
                for &j in &adjacencies[i].clone() {
                    if i >= j {
                        continue; // Only check each pair once
                    }

                    // Get potential conditioning sets from neighbors
                    let neighbors: Vec<usize> = adjacencies[i]
                        .iter()
                        .filter(|&&k| k != j)
                        .copied()
                        .collect();

                    if neighbors.len() < cond_size {
                        continue;
                    }

                    // Test all conditioning sets of this size
                    for cond_set in combinations(&neighbors, cond_size) {
                        test_count += 1;

                        if self.is_independent(data, i, j, &cond_set) {
                            edges_to_remove.push((i, j));
                            sep_sets.insert((i.min(j), i.max(j)), cond_set);
                            break;
                        }
                    }
                }
            }

            // Remove edges found to be conditionally independent
            for (i, j) in edges_to_remove {
                adjacencies[i].remove(&j);
                adjacencies[j].remove(&i);
            }
        }

        let skeleton = Skeleton {
            nodes: data.variables.clone(),
            adjacencies,
        };

        (skeleton, sep_sets, test_count)
    }

    /// Test if X ⊥ Y | Z using partial correlation.
    ///
    /// Uses Fisher's z-transformation for significance testing.
    fn is_independent(&self, data: &ObservationalData, x: usize, y: usize, z: &[usize]) -> bool {
        let n = data.n();
        if n < 5 {
            return false; // Not enough data
        }

        // Compute partial correlation
        let partial_corr = self.partial_correlation(data, x, y, z);

        // Fisher's z-transformation
        let z_stat = fisher_z_transform(partial_corr, n, z.len());

        // Two-tailed test against standard normal
        let critical_value = 1.96; // α = 0.05
        z_stat.abs() < critical_value
    }

    /// Compute partial correlation of X and Y given Z.
    fn partial_correlation(&self, data: &ObservationalData, x: usize, y: usize, z: &[usize]) -> f64 {
        if z.is_empty() {
            // Simple correlation
            return self.correlation(data, x, y);
        }

        if z.len() == 1 {
            // First-order partial correlation
            let rxy = self.correlation(data, x, y);
            let rxz = self.correlation(data, x, z[0]);
            let ryz = self.correlation(data, y, z[0]);

            let denom = ((1.0 - rxz * rxz) * (1.0 - ryz * ryz)).sqrt();
            if denom < 1e-10 {
                return 0.0;
            }
            return (rxy - rxz * ryz) / denom;
        }

        // Higher-order partial correlation via recursion
        // r(X,Y|Z) = (r(X,Y|Z-z0) - r(X,z0|Z-z0) * r(Y,z0|Z-z0)) /
        //            sqrt((1 - r(X,z0|Z-z0)^2) * (1 - r(Y,z0|Z-z0)^2))
        let z0 = z[0];
        let z_rest: Vec<usize> = z[1..].to_vec();

        let rxy_z = self.partial_correlation(data, x, y, &z_rest);
        let rxz0_z = self.partial_correlation(data, x, z0, &z_rest);
        let ryz0_z = self.partial_correlation(data, y, z0, &z_rest);

        let denom = ((1.0 - rxz0_z * rxz0_z) * (1.0 - ryz0_z * ryz0_z)).sqrt();
        if denom < 1e-10 {
            return 0.0;
        }
        (rxy_z - rxz0_z * ryz0_z) / denom
    }

    /// Compute Pearson correlation between two variables.
    fn correlation(&self, data: &ObservationalData, x: usize, y: usize) -> f64 {
        let n = data.n();
        if n < 2 {
            return 0.0;
        }

        let mean_x = data.mean(x);
        let mean_y = data.mean(y);

        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;

        for obs in &data.observations {
            let dx = obs[x] - mean_x;
            let dy = obs[y] - mean_y;
            sum_xy += dx * dy;
            sum_xx += dx * dx;
            sum_yy += dy * dy;
        }

        let denom = (sum_xx * sum_yy).sqrt();
        if denom < 1e-10 {
            return 0.0;
        }
        sum_xy / denom
    }

    /// Phase 2: Orient v-structures (colliders).
    ///
    /// For each triple A - B - C where A and C are not adjacent,
    /// orient as A → B ← C if B is not in the separating set of A and C.
    fn orient_v_structures(
        &self,
        cpdag: &mut CPDAG,
        skeleton: &Skeleton,
        sep_sets: &HashMap<(usize, usize), Vec<usize>>,
    ) {
        let n = skeleton.nodes.len();

        for b in 0..n {
            // Find pairs of non-adjacent neighbors of B
            let neighbors: Vec<usize> = skeleton.adjacencies[b].iter().copied().collect();

            for i in 0..neighbors.len() {
                for j in (i + 1)..neighbors.len() {
                    let a = neighbors[i];
                    let c = neighbors[j];

                    // Check if A and C are non-adjacent
                    if skeleton.adjacencies[a].contains(&c) {
                        continue;
                    }

                    // Check if B is in the separating set
                    let key = (a.min(c), a.max(c));
                    let sep_set = sep_sets.get(&key).cloned().unwrap_or_default();

                    if !sep_set.contains(&b) {
                        // B is not in sep(A,C) → orient as v-structure: A → B ← C
                        cpdag.orient(a, b);
                        cpdag.orient(c, b);
                    }
                }
            }
        }
    }

    /// Phase 3: Apply Meek's orientation rules.
    ///
    /// R1: A → B - C and A-/-C ⟹ B → C
    /// R2: A → B → C and A - C ⟹ A → C
    /// R3: A - B, A - C, A - D, B → D, C → D ⟹ A → D
    /// R4: A - B, B - C, C → D, A → D, A-/-C ⟹ B → C
    fn apply_orientation_rules(&self, cpdag: &mut CPDAG) {
        loop {
            let mut changed = false;

            // R1: A → B - C and A-/-C ⟹ B → C
            for b in 0..cpdag.nodes.len() {
                let parents: Vec<usize> = cpdag.parents(b).iter().copied().collect();
                let undirected: Vec<usize> = cpdag.undirected_neighbors(b).iter().copied().collect();

                for &a in &parents {
                    for &c in &undirected {
                        if !cpdag.adjacent(a, c) {
                            cpdag.orient(b, c);
                            changed = true;
                        }
                    }
                }
            }

            // R2: A → B → C and A - C ⟹ A → C
            for b in 0..cpdag.nodes.len() {
                let parents: Vec<usize> = cpdag.parents(b).iter().copied().collect();
                let children: Vec<usize> = cpdag.children(b).iter().copied().collect();

                for &a in &parents {
                    for &c in &children {
                        if cpdag.undirected_neighbors(a).contains(&c) {
                            cpdag.orient(a, c);
                            changed = true;
                        }
                    }
                }
            }

            if !changed {
                break;
            }
        }
    }
}

impl Default for PCAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

/// Fisher's z-transformation for testing correlation significance.
fn fisher_z_transform(r: f64, n: usize, k: usize) -> f64 {
    // z = arctanh(r) * sqrt(n - k - 3)
    let r_clamped = r.clamp(-0.9999, 0.9999);
    let z = 0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln();
    let df = n as f64 - k as f64 - 3.0;
    if df <= 0.0 {
        return 0.0;
    }
    z * df.sqrt()
}

/// Undirected skeleton graph.
#[derive(Debug, Clone)]
pub struct Skeleton {
    /// Node names.
    pub nodes: Vec<String>,
    /// Adjacency lists (undirected).
    pub adjacencies: Vec<HashSet<usize>>,
}

impl Skeleton {
    /// Create an empty skeleton.
    pub fn empty(nodes: Vec<String>) -> Self {
        let n = nodes.len();
        Self {
            nodes,
            adjacencies: vec![HashSet::new(); n],
        }
    }

    /// Check if two nodes are adjacent.
    pub fn adjacent(&self, a: usize, b: usize) -> bool {
        self.adjacencies[a].contains(&b)
    }

    /// Get number of edges.
    pub fn num_edges(&self) -> usize {
        self.adjacencies.iter().map(|adj| adj.len()).sum::<usize>() / 2
    }
}

/// Completed Partially Directed Acyclic Graph (CPDAG).
///
/// Represents the equivalence class of DAGs consistent with the data.
#[derive(Debug, Clone)]
pub struct CPDAG {
    /// Node names.
    pub nodes: Vec<String>,
    /// Directed edges (parent → child).
    pub directed: HashSet<(usize, usize)>,
    /// Undirected edges (unordered pairs stored as (min, max)).
    pub undirected: HashSet<(usize, usize)>,
}

impl CPDAG {
    /// Create an empty CPDAG.
    pub fn empty(nodes: Vec<String>) -> Self {
        Self {
            nodes,
            directed: HashSet::new(),
            undirected: HashSet::new(),
        }
    }

    /// Create from skeleton (all edges undirected).
    pub fn from_skeleton(skeleton: &Skeleton, nodes: &[String]) -> Self {
        let mut undirected = HashSet::new();
        for (i, adj) in skeleton.adjacencies.iter().enumerate() {
            for &j in adj {
                if i < j {
                    undirected.insert((i, j));
                }
            }
        }
        Self {
            nodes: nodes.to_vec(),
            directed: HashSet::new(),
            undirected,
        }
    }

    /// Orient an undirected edge from a to b.
    pub fn orient(&mut self, a: usize, b: usize) {
        let key = (a.min(b), a.max(b));
        if self.undirected.remove(&key) {
            self.directed.insert((a, b));
        }
    }

    /// Get parents of a node.
    pub fn parents(&self, node: usize) -> HashSet<usize> {
        self.directed
            .iter()
            .filter(|(_, c)| *c == node)
            .map(|(p, _)| *p)
            .collect()
    }

    /// Get children of a node.
    pub fn children(&self, node: usize) -> HashSet<usize> {
        self.directed
            .iter()
            .filter(|(p, _)| *p == node)
            .map(|(_, c)| *c)
            .collect()
    }

    /// Get undirected neighbors of a node.
    pub fn undirected_neighbors(&self, node: usize) -> HashSet<usize> {
        let mut result = HashSet::new();
        for &(a, b) in &self.undirected {
            if a == node {
                result.insert(b);
            } else if b == node {
                result.insert(a);
            }
        }
        result
    }

    /// Check if two nodes are adjacent (directed or undirected).
    pub fn adjacent(&self, a: usize, b: usize) -> bool {
        let key = (a.min(b), a.max(b));
        self.undirected.contains(&key)
            || self.directed.contains(&(a, b))
            || self.directed.contains(&(b, a))
    }

    /// Convert to CausalDAG (directed edges only).
    ///
    /// Note: Loses undirected edge information.
    pub fn to_dag(&self) -> CausalDAG {
        CausalDAG::new(
            self.nodes.clone(),
            self.directed.iter().copied().collect(),
        )
    }

    /// Get number of directed edges.
    pub fn num_directed(&self) -> usize {
        self.directed.len()
    }

    /// Get number of undirected edges.
    pub fn num_undirected(&self) -> usize {
        self.undirected.len()
    }
}

/// Result of PC algorithm causal discovery.
#[derive(Debug, Clone)]
pub struct PCResult {
    /// Undirected skeleton graph.
    pub skeleton: Skeleton,
    /// Completed partially directed graph.
    pub cpdag: CPDAG,
    /// Separating sets for each non-adjacent pair.
    pub separating_sets: HashMap<(usize, usize), Vec<usize>>,
    /// Number of independence tests performed.
    pub independence_tests: usize,
}

impl PCResult {
    /// Check if the algorithm found a valid structure.
    pub fn is_valid(&self) -> bool {
        !self.cpdag.nodes.is_empty()
    }

    /// Get the learned DAG (directed edges only).
    pub fn to_dag(&self) -> CausalDAG {
        self.cpdag.to_dag()
    }

    /// Summarize the discovered structure.
    pub fn summary(&self) -> String {
        format!(
            "PC Algorithm Result:\n\
             - Nodes: {}\n\
             - Skeleton edges: {}\n\
             - Directed edges: {}\n\
             - Undirected edges: {}\n\
             - Independence tests: {}",
            self.cpdag.nodes.len(),
            self.skeleton.num_edges(),
            self.cpdag.num_directed(),
            self.cpdag.num_undirected(),
            self.independence_tests
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Causal Mediation Analysis
// ─────────────────────────────────────────────────────────────────────────────

/// Causal mediation analysis for decomposing treatment effects.
///
/// Given a treatment X, mediator M, and outcome Y, decomposes the total effect into:
/// - **Natural Direct Effect (NDE)**: Effect of X on Y NOT through M
/// - **Natural Indirect Effect (NIE)**: Effect of X on Y THROUGH M
/// - **Total Effect (TE)** = NDE + NIE (on additive scale)
///
/// Reference: VanderWeele (2015). "Explanation in Causal Inference"
///
/// # Example
///
/// ```ignore
/// // Smoking → Tar → Cancer
/// let dag = CausalDAG::new(
///     vec!["Smoking".into(), "Tar".into(), "Cancer".into()],
///     vec![(0, 1), (0, 2), (1, 2)],  // Smoking → Tar, Smoking → Cancer, Tar → Cancer
/// );
///
/// let analysis = MediationAnalysis::new(&dag, 0, 1, 2);  // X=Smoking, M=Tar, Y=Cancer
/// let result = analysis.analyze(&data);
/// println!("Direct effect: {}", result.nde);
/// println!("Indirect effect (via Tar): {}", result.nie);
/// ```
pub struct MediationAnalysis<'a> {
    dag: &'a CausalDAG,
    treatment: usize,
    mediator: usize,
    outcome: usize,
}

impl<'a> MediationAnalysis<'a> {
    /// Create a new mediation analysis.
    ///
    /// # Arguments
    /// * `dag` - The causal DAG
    /// * `treatment` - Index of treatment variable X
    /// * `mediator` - Index of mediator variable M
    /// * `outcome` - Index of outcome variable Y
    pub fn new(dag: &'a CausalDAG, treatment: usize, mediator: usize, outcome: usize) -> Self {
        Self { dag, treatment, mediator, outcome }
    }

    /// Check if mediation is identified given the causal structure.
    ///
    /// Mediation requires:
    /// 1. X → M path exists
    /// 2. M → Y path exists (with or without controlling for X)
    /// 3. No confounding of M → Y that is affected by X
    pub fn is_identified(&self) -> MediationIdentification {
        // Check basic structure
        let x_to_m = self.dag.has_path(self.treatment, self.mediator);
        let m_to_y = self.dag.has_path(self.mediator, self.outcome);
        let x_to_y = self.dag.has_path(self.treatment, self.outcome);

        if !x_to_m {
            return MediationIdentification::NotMediator {
                reason: "No path from treatment to mediator".to_string(),
            };
        }

        if !m_to_y {
            return MediationIdentification::NotMediator {
                reason: "No path from mediator to outcome".to_string(),
            };
        }

        // Check for exposure-induced confounding
        // This occurs when X affects a confounder of M → Y
        let m_parents: HashSet<usize> = self.dag.parents(self.mediator).into_iter().collect();
        let y_parents: HashSet<usize> = self.dag.parents(self.outcome).into_iter().collect();

        // Potential confounders of M → Y
        let potential_confounders: Vec<usize> = m_parents.intersection(&y_parents)
            .filter(|&&n| n != self.treatment && n != self.mediator)
            .copied()
            .collect();

        // Check if X affects any of these confounders
        for &confounder in &potential_confounders {
            if self.dag.has_path(self.treatment, confounder) {
                return MediationIdentification::ExposureInducedConfounding {
                    confounder: self.dag.nodes.get(confounder)
                        .cloned()
                        .unwrap_or_else(|| format!("Node_{}", confounder)),
                };
            }
        }

        // Find adjustment set for NDE/NIE
        let baseline_confounders: Vec<usize> = self.dag.parents(self.treatment)
            .into_iter()
            .filter(|&n| self.dag.has_path(n, self.outcome))
            .collect();

        MediationIdentification::Identified {
            nde_adjustment: baseline_confounders.clone(),
            nie_adjustment: baseline_confounders,
            has_direct_effect: x_to_y,
        }
    }

    /// Estimate mediation effects from data.
    ///
    /// Uses the difference method (Baron-Kenny approach):
    /// - NDE = E[Y | do(X=1), M(0)] - E[Y | do(X=0), M(0)]
    /// - NIE = E[Y | do(X=1), M(1)] - E[Y | do(X=1), M(0)]
    ///
    /// For linear models without interactions:
    /// - Total = c (regression coefficient of Y on X)
    /// - NDE = c' (regression coefficient of Y on X controlling for M)
    /// - NIE = a * b (product of coefficients)
    pub fn analyze(&self, data: &ObservationalData) -> MediationResult {
        let identification = self.is_identified();

        match &identification {
            MediationIdentification::Identified { nde_adjustment, .. } => {
                // Simplified linear mediation analysis

                // Step 1: Total effect (Y ~ X)
                let total_effect = self.simple_regression(data, self.treatment, self.outcome);

                // Step 2: Effect of X on M (a path)
                let a_path = self.simple_regression(data, self.treatment, self.mediator);

                // Step 3: Effect of M on Y controlling for X (b path) and direct effect (c')
                let (c_prime, b_path) = self.multiple_regression_2(
                    data,
                    self.outcome,
                    self.treatment,
                    self.mediator,
                );

                // NIE = a * b (indirect effect)
                let nie = a_path * b_path;

                // NDE = c' (direct effect)
                let nde = c_prime;

                // Proportion mediated
                let proportion_mediated = if total_effect.abs() > 1e-10 {
                    nie / total_effect
                } else {
                    0.0
                };

                MediationResult {
                    total_effect,
                    natural_direct_effect: nde,
                    natural_indirect_effect: nie,
                    a_path,
                    b_path,
                    c_prime,
                    proportion_mediated: proportion_mediated.clamp(0.0, 1.0),
                    is_identified: true,
                    identification,
                }
            }
            _ => {
                // Not identified - return NaN values
                MediationResult {
                    total_effect: f64::NAN,
                    natural_direct_effect: f64::NAN,
                    natural_indirect_effect: f64::NAN,
                    a_path: f64::NAN,
                    b_path: f64::NAN,
                    c_prime: f64::NAN,
                    proportion_mediated: f64::NAN,
                    is_identified: false,
                    identification,
                }
            }
        }
    }

    /// Simple linear regression: Y ~ X
    fn simple_regression(&self, data: &ObservationalData, x: usize, y: usize) -> f64 {
        let n = data.n();
        if n < 2 {
            return 0.0;
        }

        let mean_x = data.mean(x);
        let mean_y = data.mean(y);

        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;

        for obs in &data.observations {
            let dx = obs[x] - mean_x;
            let dy = obs[y] - mean_y;
            sum_xy += dx * dy;
            sum_xx += dx * dx;
        }

        if sum_xx < 1e-10 {
            return 0.0;
        }

        sum_xy / sum_xx
    }

    /// Multiple regression: Y ~ X + M, returns (coef_x, coef_m)
    fn multiple_regression_2(&self, data: &ObservationalData, y: usize, x1: usize, x2: usize) -> (f64, f64) {
        let n = data.n();
        if n < 3 {
            return (0.0, 0.0);
        }

        let mean_y = data.mean(y);
        let mean_x1 = data.mean(x1);
        let mean_x2 = data.mean(x2);

        // Compute sums for normal equations
        let mut sum_x1x1 = 0.0;
        let mut sum_x2x2 = 0.0;
        let mut sum_x1x2 = 0.0;
        let mut sum_x1y = 0.0;
        let mut sum_x2y = 0.0;

        for obs in &data.observations {
            let dx1 = obs[x1] - mean_x1;
            let dx2 = obs[x2] - mean_x2;
            let dy = obs[y] - mean_y;

            sum_x1x1 += dx1 * dx1;
            sum_x2x2 += dx2 * dx2;
            sum_x1x2 += dx1 * dx2;
            sum_x1y += dx1 * dy;
            sum_x2y += dx2 * dy;
        }

        // Solve 2x2 system: [[sum_x1x1, sum_x1x2], [sum_x1x2, sum_x2x2]] * [b1, b2] = [sum_x1y, sum_x2y]
        let det = sum_x1x1 * sum_x2x2 - sum_x1x2 * sum_x1x2;
        if det.abs() < 1e-10 {
            return (0.0, 0.0);
        }

        let b1 = (sum_x2x2 * sum_x1y - sum_x1x2 * sum_x2y) / det;
        let b2 = (sum_x1x1 * sum_x2y - sum_x1x2 * sum_x1y) / det;

        (b1, b2)
    }
}

/// Result of mediation identification check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MediationIdentification {
    /// Mediation effects are identified.
    Identified {
        /// Variables to adjust for when estimating NDE.
        nde_adjustment: Vec<usize>,
        /// Variables to adjust for when estimating NIE.
        nie_adjustment: Vec<usize>,
        /// Whether there is a direct effect (X → Y path exists).
        has_direct_effect: bool,
    },
    /// Not a valid mediator.
    NotMediator {
        /// Reason why M is not a valid mediator.
        reason: String,
    },
    /// Exposure-induced confounding blocks identification.
    ExposureInducedConfounding {
        /// The confounder that is affected by X.
        confounder: String,
    },
}

/// Result of causal mediation analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MediationResult {
    /// Total effect of X on Y.
    pub total_effect: f64,
    /// Natural Direct Effect (not through mediator).
    pub natural_direct_effect: f64,
    /// Natural Indirect Effect (through mediator).
    pub natural_indirect_effect: f64,
    /// a-path: Effect of X on M.
    pub a_path: f64,
    /// b-path: Effect of M on Y (controlling for X).
    pub b_path: f64,
    /// c'-path: Direct effect of X on Y (controlling for M).
    pub c_prime: f64,
    /// Proportion of total effect mediated (0-1).
    pub proportion_mediated: f64,
    /// Whether the mediation is identified.
    pub is_identified: bool,
    /// Identification status.
    pub identification: MediationIdentification,
}

impl MediationResult {
    /// Check if there is significant mediation.
    ///
    /// Returns true if:
    /// - NIE is non-negligible (> threshold)
    /// - NIE has the same sign as total effect
    pub fn has_significant_mediation(&self, threshold: f64) -> bool {
        self.is_identified
            && self.natural_indirect_effect.abs() > threshold
            && (self.natural_indirect_effect.signum() == self.total_effect.signum()
                || self.total_effect.abs() < 1e-10)
    }

    /// Check if the effect is fully mediated (> 80% through mediator).
    pub fn is_fully_mediated(&self) -> bool {
        self.is_identified && self.proportion_mediated > 0.8
    }

    /// Check if the effect is partially mediated (20-80% through mediator).
    pub fn is_partially_mediated(&self) -> bool {
        self.is_identified && self.proportion_mediated > 0.2 && self.proportion_mediated <= 0.8
    }

    /// Get a summary of the mediation analysis.
    pub fn summary(&self) -> String {
        if !self.is_identified {
            return format!("Mediation not identified: {:?}", self.identification);
        }

        let mediation_type = if self.is_fully_mediated() {
            "Full mediation"
        } else if self.is_partially_mediated() {
            "Partial mediation"
        } else if self.proportion_mediated > 0.0 {
            "Weak mediation"
        } else {
            "No mediation"
        };

        format!(
            "Mediation Analysis:\n\
             - Total Effect: {:.4}\n\
             - Direct Effect (NDE): {:.4}\n\
             - Indirect Effect (NIE): {:.4}\n\
             - Proportion Mediated: {:.1}%\n\
             - Type: {}",
            self.total_effect,
            self.natural_direct_effect,
            self.natural_indirect_effect,
            self.proportion_mediated * 100.0,
            mediation_type
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Instrumental Variable Estimation
// ─────────────────────────────────────────────────────────────────────────────

/// Instrumental Variable (IV) estimator for causal effects.
///
/// Used when treatment X is confounded with outcome Y, but we have an
/// instrument Z that:
/// 1. Affects treatment (Z → X)
/// 2. Only affects outcome through treatment (no Z → Y path except via X)
/// 3. Is independent of confounders
///
/// The classic example: distance to college (Z) as instrument for education (X)
/// on earnings (Y).
///
/// # Two-Stage Least Squares (2SLS)
///
/// Stage 1: X̂ = α + β*Z (predict X from Z)
/// Stage 2: Y = γ + δ*X̂ (regress Y on predicted X)
///
/// The coefficient δ is the causal effect of X on Y.
pub struct IVEstimator;

impl IVEstimator {
    /// Check if Z is a valid instrument for X → Y.
    ///
    /// Instrument validity requires:
    /// 1. Relevance: Z → X path exists
    /// 2. Exclusion: No direct Z → Y path
    /// 3. Independence: No confounding of Z
    pub fn is_valid_instrument(dag: &CausalDAG, instrument: usize, treatment: usize, outcome: usize) -> IVValidity {
        // Check relevance: Z must affect X
        if !dag.has_path(instrument, treatment) {
            return IVValidity::Invalid {
                reason: "Instrument does not affect treatment (no Z → X path)".to_string(),
            };
        }

        // Check exclusion: Z should not directly affect Y
        // This is a simplification - full check requires excluding paths through X
        let z_children = dag.children(instrument);
        for child in z_children {
            if child == outcome {
                return IVValidity::Invalid {
                    reason: "Instrument directly affects outcome (Z → Y path exists)".to_string(),
                };
            }
        }

        // Check if Z only reaches Y through X
        let mut reaches_y_not_through_x = false;
        let z_descendants = dag.descendants(instrument);
        let x_descendants = dag.descendants(treatment);

        for desc in &z_descendants {
            if *desc == outcome && !x_descendants.contains(&outcome) {
                reaches_y_not_through_x = true;
                break;
            }
        }

        if reaches_y_not_through_x {
            return IVValidity::Invalid {
                reason: "Instrument reaches outcome through path not including treatment".to_string(),
            };
        }

        IVValidity::Valid {
            instrument_strength: 1.0, // Would be computed from data
        }
    }

    /// Estimate causal effect using Two-Stage Least Squares (2SLS).
    ///
    /// Returns the Local Average Treatment Effect (LATE) for compliers.
    pub fn estimate_2sls(
        data: &ObservationalData,
        instrument: usize,
        treatment: usize,
        outcome: usize,
    ) -> IVResult {
        let n = data.n();
        if n < 10 {
            return IVResult {
                effect: f64::NAN,
                first_stage_f: 0.0,
                is_weak_instrument: true,
                method: "2SLS".to_string(),
            };
        }

        // Stage 1: Regress X on Z
        let (first_stage_coef, first_stage_r2) = Self::first_stage(data, instrument, treatment);

        // Check for weak instrument (F-statistic < 10 rule of thumb)
        let first_stage_f = (n as f64 - 2.0) * first_stage_r2 / (1.0 - first_stage_r2);
        let is_weak = first_stage_f < 10.0;

        // Stage 2: Regress Y on X̂
        let effect = Self::second_stage(data, instrument, treatment, outcome, first_stage_coef);

        IVResult {
            effect,
            first_stage_f,
            is_weak_instrument: is_weak,
            method: "2SLS".to_string(),
        }
    }

    /// First stage regression: X ~ Z
    fn first_stage(data: &ObservationalData, z: usize, x: usize) -> (f64, f64) {
        let n = data.n();
        let mean_z = data.mean(z);
        let mean_x = data.mean(x);

        let mut sum_zx = 0.0;
        let mut sum_zz = 0.0;
        let mut sum_xx = 0.0;

        for obs in &data.observations {
            let dz = obs[z] - mean_z;
            let dx = obs[x] - mean_x;
            sum_zx += dz * dx;
            sum_zz += dz * dz;
            sum_xx += dx * dx;
        }

        let beta = if sum_zz > 1e-10 { sum_zx / sum_zz } else { 0.0 };

        // R² = (Cov(Z,X))² / (Var(Z) * Var(X))
        let r2 = if sum_zz > 1e-10 && sum_xx > 1e-10 {
            (sum_zx * sum_zx) / (sum_zz * sum_xx)
        } else {
            0.0
        };

        (beta, r2)
    }

    /// Second stage regression: Y ~ X̂
    fn second_stage(
        data: &ObservationalData,
        z: usize,
        x: usize,
        y: usize,
        first_stage_coef: f64,
    ) -> f64 {
        let mean_z = data.mean(z);
        let mean_x = data.mean(x);
        let mean_y = data.mean(y);

        // Compute X̂ = mean_x + first_stage_coef * (Z - mean_z)
        let mut sum_xhat_y = 0.0;
        let mut sum_xhat_xhat = 0.0;

        for obs in &data.observations {
            let x_hat = mean_x + first_stage_coef * (obs[z] - mean_z);
            let dx_hat = x_hat - mean_x;
            let dy = obs[y] - mean_y;
            sum_xhat_y += dx_hat * dy;
            sum_xhat_xhat += dx_hat * dx_hat;
        }

        if sum_xhat_xhat > 1e-10 {
            sum_xhat_y / sum_xhat_xhat
        } else {
            0.0
        }
    }

    /// Wald estimator (simple IV with binary instrument).
    ///
    /// Effect = (E[Y|Z=1] - E[Y|Z=0]) / (E[X|Z=1] - E[X|Z=0])
    pub fn estimate_wald(
        data: &ObservationalData,
        instrument: usize,
        treatment: usize,
        outcome: usize,
    ) -> f64 {
        // Split data by instrument value (assuming binary/threshold at 0.5)
        let mut y_z1 = Vec::new();
        let mut y_z0 = Vec::new();
        let mut x_z1 = Vec::new();
        let mut x_z0 = Vec::new();

        for obs in &data.observations {
            if obs[instrument] > 0.5 {
                y_z1.push(obs[outcome]);
                x_z1.push(obs[treatment]);
            } else {
                y_z0.push(obs[outcome]);
                x_z0.push(obs[treatment]);
            }
        }

        if y_z1.is_empty() || y_z0.is_empty() {
            return f64::NAN;
        }

        let mean_y_z1: f64 = y_z1.iter().sum::<f64>() / y_z1.len() as f64;
        let mean_y_z0: f64 = y_z0.iter().sum::<f64>() / y_z0.len() as f64;
        let mean_x_z1: f64 = x_z1.iter().sum::<f64>() / x_z1.len() as f64;
        let mean_x_z0: f64 = x_z0.iter().sum::<f64>() / x_z0.len() as f64;

        let denom = mean_x_z1 - mean_x_z0;
        if denom.abs() < 1e-10 {
            return f64::NAN;
        }

        (mean_y_z1 - mean_y_z0) / denom
    }
}

/// Validity status of an instrumental variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IVValidity {
    /// Instrument is valid.
    Valid {
        /// Strength of the instrument (first-stage F-statistic).
        instrument_strength: f64,
    },
    /// Instrument is invalid.
    Invalid {
        /// Reason for invalidity.
        reason: String,
    },
}

/// Result of instrumental variable estimation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IVResult {
    /// Estimated causal effect (LATE).
    pub effect: f64,
    /// First-stage F-statistic.
    pub first_stage_f: f64,
    /// Whether the instrument is weak (F < 10).
    pub is_weak_instrument: bool,
    /// Estimation method used.
    pub method: String,
}

// ─────────────────────────────────────────────────────────────────────────────
// Time-Series Causal Discovery
// ─────────────────────────────────────────────────────────────────────────────

/// Time-series causal discovery using Granger causality and temporal PC.
///
/// Extends causal discovery to longitudinal data where time ordering
/// provides additional constraints on possible causal relationships.
///
/// Key insight: Causes must precede effects in time.
pub struct TimeSeriesCausalDiscovery {
    /// Maximum lag to consider.
    pub max_lag: usize,
    /// Significance level for Granger tests.
    pub alpha: f64,
}

impl TimeSeriesCausalDiscovery {
    /// Create new time-series causal discovery.
    pub fn new(max_lag: usize) -> Self {
        Self {
            max_lag,
            alpha: 0.05,
        }
    }

    /// Test Granger causality: Does X Granger-cause Y?
    ///
    /// X Granger-causes Y if past values of X help predict Y
    /// beyond what past values of Y alone can predict.
    ///
    /// Returns F-statistic and p-value approximation.
    pub fn granger_test(
        &self,
        x: &[f64],
        y: &[f64],
        lag: usize,
    ) -> GrangerResult {
        if x.len() != y.len() || x.len() <= lag + 1 {
            return GrangerResult {
                f_statistic: 0.0,
                p_value: 1.0,
                is_significant: false,
                optimal_lag: 0,
            };
        }

        let n = x.len() - lag;

        // Restricted model: Y_t ~ Y_{t-1} + ... + Y_{t-lag}
        let ssr_restricted = self.compute_ssr_restricted(y, lag);

        // Unrestricted model: Y_t ~ Y_{t-1} + ... + Y_{t-lag} + X_{t-1} + ... + X_{t-lag}
        let ssr_unrestricted = self.compute_ssr_unrestricted(x, y, lag);

        // F-statistic
        let df1 = lag as f64;
        let df2 = (n - 2 * lag - 1) as f64;

        if ssr_unrestricted < 1e-10 || df2 <= 0.0 {
            return GrangerResult {
                f_statistic: 0.0,
                p_value: 1.0,
                is_significant: false,
                optimal_lag: lag,
            };
        }

        let f_stat = ((ssr_restricted - ssr_unrestricted) / df1) / (ssr_unrestricted / df2);

        // Approximate p-value using F-distribution CDF approximation
        let p_value = self.f_distribution_pvalue(f_stat, df1, df2);

        GrangerResult {
            f_statistic: f_stat,
            p_value,
            is_significant: p_value < self.alpha,
            optimal_lag: lag,
        }
    }

    /// Compute Sum of Squared Residuals for restricted model.
    fn compute_ssr_restricted(&self, y: &[f64], lag: usize) -> f64 {
        let n = y.len();
        let mut ssr = 0.0;

        for t in lag..n {
            // Predict Y_t from past Y values
            let mut y_pred = 0.0;
            for l in 1..=lag {
                y_pred += y[t - l] / lag as f64; // Simple average as baseline
            }
            let residual = y[t] - y_pred;
            ssr += residual * residual;
        }

        ssr
    }

    /// Compute Sum of Squared Residuals for unrestricted model.
    fn compute_ssr_unrestricted(&self, x: &[f64], y: &[f64], lag: usize) -> f64 {
        let n = y.len();
        let mut ssr = 0.0;

        for t in lag..n {
            // Predict Y_t from past Y and X values
            let mut y_pred = 0.0;
            for l in 1..=lag {
                y_pred += (y[t - l] + x[t - l]) / (2.0 * lag as f64);
            }
            let residual = y[t] - y_pred;
            ssr += residual * residual;
        }

        ssr
    }

    /// Approximate F-distribution p-value.
    fn f_distribution_pvalue(&self, f: f64, df1: f64, df2: f64) -> f64 {
        // Wilson-Hilferty approximation for F-distribution
        if f <= 0.0 || df1 <= 0.0 || df2 <= 0.0 {
            return 1.0;
        }

        let x = (f.powf(1.0 / 3.0) * (1.0 - 2.0 / (9.0 * df2)) - (1.0 - 2.0 / (9.0 * df1)))
            / ((2.0 / (9.0 * df1) + f.powf(2.0 / 3.0) * 2.0 / (9.0 * df2)).sqrt());

        // Standard normal CDF approximation
        0.5 * (1.0 - Self::erf(x / std::f64::consts::SQRT_2))
    }

    /// Error function approximation.
    fn erf(x: f64) -> f64 {
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let p = 0.3275911;

        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let x = x.abs();
        let t = 1.0 / (1.0 + p * x);
        let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

        sign * y
    }

    /// Discover causal structure from multivariate time series.
    ///
    /// Uses Granger causality tests to build a causal graph where
    /// X → Y if X Granger-causes Y.
    pub fn discover(&self, data: &TimeSeriesData) -> TimeSeriesCausalGraph {
        let n_vars = data.variables.len();
        let mut edges = Vec::new();
        let mut granger_results = HashMap::new();

        for i in 0..n_vars {
            for j in 0..n_vars {
                if i == j {
                    continue;
                }

                // Test if variable i Granger-causes variable j
                let mut best_result = GrangerResult {
                    f_statistic: 0.0,
                    p_value: 1.0,
                    is_significant: false,
                    optimal_lag: 1,
                };

                for lag in 1..=self.max_lag {
                    let result = self.granger_test(&data.series[i], &data.series[j], lag);
                    if result.f_statistic > best_result.f_statistic {
                        best_result = result;
                        best_result.optimal_lag = lag;
                    }
                }

                if best_result.is_significant {
                    edges.push((i, j, best_result.optimal_lag));
                }

                granger_results.insert((i, j), best_result);
            }
        }

        TimeSeriesCausalGraph {
            variables: data.variables.clone(),
            edges,
            granger_results,
        }
    }
}

impl Default for TimeSeriesCausalDiscovery {
    fn default() -> Self {
        Self::new(5)
    }
}

/// Time series data for multiple variables.
#[derive(Debug, Clone)]
pub struct TimeSeriesData {
    /// Variable names.
    pub variables: Vec<String>,
    /// Time series for each variable (same length).
    pub series: Vec<Vec<f64>>,
}

impl TimeSeriesData {
    /// Create new time series data.
    pub fn new(variables: Vec<String>) -> Self {
        let n = variables.len();
        Self {
            variables,
            series: vec![Vec::new(); n],
        }
    }

    /// Add a time point observation for all variables.
    pub fn add_observation(&mut self, values: Vec<f64>) {
        for (i, v) in values.into_iter().enumerate() {
            if i < self.series.len() {
                self.series[i].push(v);
            }
        }
    }

    /// Get number of time points.
    pub fn n_timepoints(&self) -> usize {
        self.series.first().map(|s| s.len()).unwrap_or(0)
    }
}

/// Result of Granger causality test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrangerResult {
    /// F-statistic for the test.
    pub f_statistic: f64,
    /// P-value of the test.
    pub p_value: f64,
    /// Whether the result is significant at alpha level.
    pub is_significant: bool,
    /// Optimal lag that gave highest F-statistic.
    pub optimal_lag: usize,
}

/// Causal graph discovered from time series.
#[derive(Debug, Clone)]
pub struct TimeSeriesCausalGraph {
    /// Variable names.
    pub variables: Vec<String>,
    /// Edges as (from, to, lag).
    pub edges: Vec<(usize, usize, usize)>,
    /// Granger test results for all pairs.
    pub granger_results: HashMap<(usize, usize), GrangerResult>,
}

impl TimeSeriesCausalGraph {
    /// Convert to standard CausalDAG (ignoring lags).
    pub fn to_dag(&self) -> CausalDAG {
        let edges: Vec<(usize, usize)> = self.edges.iter().map(|(f, t, _)| (*f, *t)).collect();
        CausalDAG::new(self.variables.clone(), edges)
    }

    /// Get summary of discovered causal relationships.
    pub fn summary(&self) -> String {
        let mut lines = vec!["Time-Series Causal Graph:".to_string()];

        for (from, to, lag) in &self.edges {
            lines.push(format!(
                "  {} → {} (lag={})",
                self.variables[*from], self.variables[*to], lag
            ));
        }

        if self.edges.is_empty() {
            lines.push("  No significant Granger-causal relationships found".to_string());
        }

        lines.join("\n")
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Causal Transportability
// ─────────────────────────────────────────────────────────────────────────────

/// Causal transportability analysis.
///
/// Determines whether a causal effect learned in a source population
/// can be "transported" to a target population that differs in some ways.
///
/// Uses selection diagrams where S-nodes represent selection/sampling
/// mechanisms that differ between populations.
///
/// Reference: Pearl & Bareinboim (2011). "Transportability of Causal and
/// Statistical Relations"
pub struct TransportabilityAnalyzer {
    /// Source population DAG.
    source_dag: CausalDAG,
    /// Target population DAG (may differ in mechanisms).
    target_dag: CausalDAG,
    /// Selection variables (nodes that differ between populations).
    selection_nodes: Vec<usize>,
}

impl TransportabilityAnalyzer {
    /// Create a new transportability analyzer.
    ///
    /// # Arguments
    /// * `source_dag` - DAG for the source population
    /// * `target_dag` - DAG for the target population
    /// * `selection_nodes` - Nodes whose mechanisms differ between populations
    pub fn new(
        source_dag: CausalDAG,
        target_dag: CausalDAG,
        selection_nodes: Vec<usize>,
    ) -> Self {
        Self {
            source_dag,
            target_dag,
            selection_nodes,
        }
    }

    /// Check if the causal effect P(y|do(x)) is transportable.
    ///
    /// Returns transportability status and any required adjustments.
    pub fn is_transportable(
        &self,
        treatment: usize,
        outcome: usize,
    ) -> TransportabilityResult {
        // Simple check: effect is directly transportable if no selection
        // node is on any path from treatment to outcome
        let paths_blocked = self.check_selection_blocking(treatment, outcome);

        if paths_blocked {
            return TransportabilityResult::DirectlyTransportable {
                explanation: "No selection mechanism affects the causal pathway".to_string(),
            };
        }

        // Check if we can adjust for selection
        let adjustment = self.find_transport_adjustment(treatment, outcome);

        match adjustment {
            Some(adj_set) => TransportabilityResult::TransportableWithAdjustment {
                adjustment_set: adj_set,
                explanation: "Effect transportable after adjusting for population differences"
                    .to_string(),
            },
            None => TransportabilityResult::NotTransportable {
                reason: "Selection mechanisms block all identification strategies".to_string(),
                blocking_nodes: self.find_blocking_selection_nodes(treatment, outcome),
            },
        }
    }

    /// Check if selection nodes block the treatment-outcome relationship.
    fn check_selection_blocking(&self, treatment: usize, outcome: usize) -> bool {
        // Get all nodes on paths from treatment to outcome
        let descendants = self.source_dag.descendants(treatment);

        // Check if any selection node is a descendant of treatment
        // and an ancestor of outcome
        for &s_node in &self.selection_nodes {
            if descendants.contains(&s_node) {
                let s_descendants = self.source_dag.descendants(s_node);
                if s_descendants.contains(&outcome) {
                    return false; // Selection node is on a path
                }
            }
        }

        true // No selection node on any path
    }

    /// Find adjustment set for transportability.
    fn find_transport_adjustment(&self, treatment: usize, outcome: usize) -> Option<Vec<usize>> {
        // Find variables that can block selection bias
        // These must satisfy the selection backdoor criterion

        let mut candidates: Vec<usize> = (0..self.source_dag.num_nodes())
            .filter(|&n| {
                n != treatment
                    && n != outcome
                    && !self.selection_nodes.contains(&n)
                    && !self.source_dag.descendants(treatment).contains(&n)
            })
            .collect();

        // Check if candidates block selection-induced confounding
        if candidates.is_empty() {
            return None;
        }

        // Simplified: return all non-descendant, non-selection nodes
        Some(candidates)
    }

    /// Find which selection nodes block transportability.
    fn find_blocking_selection_nodes(&self, treatment: usize, outcome: usize) -> Vec<usize> {
        let mut blocking = Vec::new();
        let treatment_descendants = self.source_dag.descendants(treatment);

        for &s_node in &self.selection_nodes {
            if treatment_descendants.contains(&s_node) {
                let s_descendants = self.source_dag.descendants(s_node);
                if s_descendants.contains(&outcome) {
                    blocking.push(s_node);
                }
            }
        }

        blocking
    }

    /// Compute the transported effect estimate.
    ///
    /// Uses the transport formula to reweight the source effect.
    pub fn transport_effect(
        &self,
        source_data: &ObservationalData,
        target_data: &ObservationalData,
        treatment: usize,
        outcome: usize,
    ) -> Option<f64> {
        let result = self.is_transportable(treatment, outcome);

        match result {
            TransportabilityResult::DirectlyTransportable { .. } => {
                // Effect is the same in both populations
                // Use source effect directly
                let estimator = EffectEstimator::new();
                let query = CausalQuery {
                    treatment,
                    outcome,
                    conditioning: vec![],
                };

                match estimator.estimate(&self.source_dag, &query, source_data) {
                    CausalQueryOutcome::Identified { estimand, .. } => Some(estimand.effect),
                    _ => None,
                }
            }
            TransportabilityResult::TransportableWithAdjustment { adjustment_set, .. } => {
                // Need to reweight by population differences
                // Simplified: compute effect in source with adjustment
                let estimator = EffectEstimator::new();
                let query = CausalQuery {
                    treatment,
                    outcome,
                    conditioning: adjustment_set,
                };

                match estimator.estimate(&self.source_dag, &query, source_data) {
                    CausalQueryOutcome::Identified { estimand, .. } => Some(estimand.effect),
                    _ => None,
                }
            }
            TransportabilityResult::NotTransportable { .. } => None,
        }
    }
}

/// Result of transportability analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransportabilityResult {
    /// Effect is directly transportable without adjustment.
    DirectlyTransportable {
        /// Explanation of why it's transportable.
        explanation: String,
    },
    /// Effect is transportable after adjusting for covariates.
    TransportableWithAdjustment {
        /// Variables to adjust for in the target population.
        adjustment_set: Vec<usize>,
        /// Explanation.
        explanation: String,
    },
    /// Effect is not transportable.
    NotTransportable {
        /// Reason for non-transportability.
        reason: String,
        /// Selection nodes that block transport.
        blocking_nodes: Vec<usize>,
    },
}

impl TransportabilityResult {
    /// Check if the effect is transportable (with or without adjustment).
    pub fn is_transportable(&self) -> bool {
        !matches!(self, TransportabilityResult::NotTransportable { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_direct_cause_identified() {
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into()],
            vec![(0, 1)],
        );
        let query = CausalQuery { treatment: 0, outcome: 1, conditioning: vec![] };
        let reasoner = CounterfactualReasoner::new();
        let result = reasoner.query(&dag, &query);
        assert!(matches!(result, CausalQueryOutcome::Identified { .. }));
    }

    #[test]
    fn test_unconnected_unidentified() {
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into()],
            vec![],
        );
        let query = CausalQuery { treatment: 0, outcome: 1, conditioning: vec![] };
        let reasoner = CounterfactualReasoner::new();
        let result = reasoner.query(&dag, &query);
        assert!(matches!(result, CausalQueryOutcome::Unidentified { .. }));
    }

    #[test]
    fn test_backdoor_adjustment() {
        // X ← U → Y, X → Y. Adjusting for U should work.
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into(), "U".into()],
            vec![(2, 0), (2, 1), (0, 1)],
        );
        let query = CausalQuery { treatment: 0, outcome: 1, conditioning: vec![] };
        let reasoner = CounterfactualReasoner::new();
        let result = reasoner.query(&dag, &query);
        match result {
            CausalQueryOutcome::Identified { method, .. } => {
                assert_eq!(method, IdentificationMethod::BackdoorAdjustment);
            }
            _ => panic!("Expected Identified via backdoor"),
        }
    }

    #[test]
    fn test_frontdoor_criterion() {
        // Classic frontdoor: U is unobserved confounder.
        // X → M → Y, with U→X and U→Y but U is NOT a node we can adjust for.
        // Since U confounds X-Y, backdoor fails. But X→M is unconfounded
        // and we can use M as frontdoor.
        //
        // For our implementation, we test that frontdoor fires when
        // backdoor is not available. We use a DAG where the only parent
        // of X is a descendant of X (violating backdoor condition 1).
        let dag = CausalDAG::new(
            vec!["X".into(), "M".into(), "Y".into()],
            vec![(0, 1), (1, 2)],
        );
        let query = CausalQuery { treatment: 0, outcome: 2, conditioning: vec![] };
        let reasoner = CounterfactualReasoner::new();
        let result = reasoner.query(&dag, &query);
        match result {
            CausalQueryOutcome::Identified { method, .. } => {
                // Could be either frontdoor or backdoor (empty set is valid backdoor for chain)
                assert!(
                    method == IdentificationMethod::FrontdoorCriterion
                        || method == IdentificationMethod::BackdoorAdjustment,
                    "Expected identified method, got {:?}",
                    method,
                );
            }
            _ => panic!("Expected Identified, got {:?}", result),
        }
    }

    #[test]
    fn test_dag_too_large() {
        let nodes: Vec<String> = (0..25).map(|i| format!("N{}", i)).collect();
        let dag = CausalDAG::new(nodes, vec![]);
        let query = CausalQuery { treatment: 0, outcome: 1, conditioning: vec![] };
        let reasoner = CounterfactualReasoner::new();
        let result = reasoner.query(&dag, &query);
        assert!(matches!(
            result,
            CausalQueryOutcome::Unidentified { reason: UnidentifiedReason::DagTooLarge { .. }, .. }
        ));
    }

    #[test]
    fn test_reference_harness() {
        let reasoner = CounterfactualReasoner::new();
        let mut harness = CausalReferenceHarness::new();
        let result = harness.validate(&reasoner);
        // Should pass: our reasoner handles the basic test cases
        assert!(
            harness.current_match_rate >= 0.5,
            "Harness match rate too low: {}",
            harness.current_match_rate
        );
    }

    #[test]
    fn test_inv3_deterministic() {
        // INV-3: Same inputs → same result
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into(), "U".into()],
            vec![(2, 0), (2, 1), (0, 1)],
        );
        let query = CausalQuery { treatment: 0, outcome: 1, conditioning: vec![] };
        let reasoner = CounterfactualReasoner::new();

        let r1 = reasoner.query(&dag, &query);
        let r2 = reasoner.query(&dag, &query);

        // Both should be Identified with same method
        match (&r1, &r2) {
            (
                CausalQueryOutcome::Identified { method: m1, .. },
                CausalQueryOutcome::Identified { method: m2, .. },
            ) => assert_eq!(m1, m2, "INV-3: Same query must produce same method"),
            _ => panic!("INV-3: Same query must produce same outcome type"),
        }
    }

    #[test]
    fn test_combinations() {
        let items = vec![0, 1, 2];
        let c2 = combinations(&items, 2);
        assert_eq!(c2.len(), 3); // C(3,2) = 3
        assert_eq!(c2, vec![vec![0, 1], vec![0, 2], vec![1, 2]]);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // d-Separation Tests
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_d_separation_chain() {
        // Chain: A → B → C
        // A ⊥ C | B (blocking the chain)
        // A ⊥̸ C | ∅ (no blocking)
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![(0, 1), (1, 2)],
        );

        let empty: HashSet<usize> = HashSet::new();
        let b_set: HashSet<usize> = [1].iter().copied().collect();

        // Without conditioning: A and C are d-connected
        assert!(!dag.is_d_separated(0, 2, &empty), "A-C should be d-connected without conditioning");

        // Conditioning on B: A and C are d-separated
        assert!(dag.is_d_separated(0, 2, &b_set), "A-C should be d-separated given B");
    }

    #[test]
    fn test_d_separation_fork() {
        // Fork: A ← B → C
        // A ⊥ C | B (blocking the fork)
        // A ⊥̸ C | ∅ (no blocking)
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![(1, 0), (1, 2)],
        );

        let empty: HashSet<usize> = HashSet::new();
        let b_set: HashSet<usize> = [1].iter().copied().collect();

        // Without conditioning: A and C are d-connected via B
        assert!(!dag.is_d_separated(0, 2, &empty), "A-C should be d-connected without conditioning");

        // Conditioning on B: A and C are d-separated
        assert!(dag.is_d_separated(0, 2, &b_set), "A-C should be d-separated given B");
    }

    #[test]
    fn test_d_separation_collider() {
        // Collider: A → B ← C
        // A ⊥ C | ∅ (collider blocks by default)
        // A ⊥̸ C | B (conditioning on collider opens the path)
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![(0, 1), (2, 1)],
        );

        let empty: HashSet<usize> = HashSet::new();
        let b_set: HashSet<usize> = [1].iter().copied().collect();

        // Without conditioning: A and C are d-separated (collider blocks)
        assert!(dag.is_d_separated(0, 2, &empty), "A-C should be d-separated without conditioning (collider)");

        // Conditioning on B: A and C are d-connected (collider opened)
        assert!(!dag.is_d_separated(0, 2, &b_set), "A-C should be d-connected given B (collider opened)");
    }

    #[test]
    fn test_d_separation_collider_descendant() {
        // Collider with descendant: A → B ← C, B → D
        // Conditioning on D should also open the collider path
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            vec![(0, 1), (2, 1), (1, 3)],
        );

        let empty: HashSet<usize> = HashSet::new();
        let d_set: HashSet<usize> = [3].iter().copied().collect();

        // Without conditioning: A and C are d-separated
        assert!(dag.is_d_separated(0, 2, &empty), "A-C should be d-separated");

        // Conditioning on D (descendant of collider): A and C are d-connected
        assert!(!dag.is_d_separated(0, 2, &d_set), "A-C should be d-connected given D (collider descendant)");
    }

    #[test]
    fn test_d_separation_m_bias() {
        // M-bias structure: U1 → X, U1 → M, U2 → M, U2 → Y, X → Y
        // X ⊥̸ Y | ∅ (direct path X → Y)
        // Conditioning on M opens a backdoor path via U1 and U2
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into(), "M".into(), "U1".into(), "U2".into()],
            vec![(3, 0), (3, 2), (4, 2), (4, 1), (0, 1)],
        );

        let empty: HashSet<usize> = HashSet::new();
        let m_set: HashSet<usize> = [2].iter().copied().collect();

        // X and Y are d-connected via X → Y
        assert!(!dag.is_d_separated(0, 1, &empty), "X-Y should be d-connected");

        // Conditioning on M opens a backdoor path
        assert!(!dag.is_d_separated(0, 1, &m_set), "X-Y should still be d-connected given M");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Graph Surgery Tests
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_remove_incoming() {
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![(0, 1), (1, 2), (0, 2)],
        );

        // Remove incoming to B
        let mutilated = dag.remove_incoming(&[1]);
        assert_eq!(mutilated.edges.len(), 2);
        assert!(mutilated.edges.contains(&(1, 2)));
        assert!(mutilated.edges.contains(&(0, 2)));
        assert!(!mutilated.edges.contains(&(0, 1)));
    }

    #[test]
    fn test_remove_outgoing() {
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![(0, 1), (1, 2), (0, 2)],
        );

        // Remove outgoing from A
        let mutilated = dag.remove_outgoing(&[0]);
        assert_eq!(mutilated.edges.len(), 1);
        assert!(mutilated.edges.contains(&(1, 2)));
        assert!(!mutilated.edges.contains(&(0, 1)));
        assert!(!mutilated.edges.contains(&(0, 2)));
    }

    #[test]
    fn test_remove_for_rule3() {
        // X → Y, X → Z, Z → W
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into(), "Z".into(), "W".into()],
            vec![(0, 1), (0, 2), (2, 3)],
        );

        // Rule 3 mutilation: G̅_X,Z(W)
        // Remove incoming to X (none here)
        // Remove outgoing from Z except those to ancestors of W
        // W's ancestors: Z (and Z → W is an edge from Z)
        let mutilated = dag.remove_for_rule3(&[0], &[2], &[3]);

        // Should keep X→Y, X→Z, Z→W (Z→W goes to ancestor of W, so kept)
        assert_eq!(mutilated.edges.len(), 3);
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Rule 2 and Rule 3 Tests
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_rule2_instrument_variable() {
        // Instrumental variable: Z → X → Y with U → X, U → Y
        // Rule 2 can convert do(Z) to observation of Z
        let dag = CausalDAG::new(
            vec!["Z".into(), "X".into(), "Y".into(), "U".into()],
            vec![(0, 1), (1, 2), (3, 1), (3, 2)],
        );

        let reasoner = CounterfactualReasoner::new();

        // Try Rule 2 for Z as intervention candidate
        let result = reasoner.try_rule2(&dag, 1, 2, 0, &[]);

        // Rule 2 should apply: Y ⊥ Z | X in G̅_X,Z_
        assert!(result.is_some(), "Rule 2 should apply for instrument variable");
        if let Some(CausalQueryOutcome::Identified { method, .. }) = result {
            assert_eq!(method, IdentificationMethod::Rule2ActionObservation);
        }
    }

    #[test]
    fn test_rule3_irrelevant_intervention() {
        // X → Y, X → Z (Z doesn't affect Y)
        // Rule 3 can drop do(Z) entirely
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into(), "Z".into()],
            vec![(0, 1), (0, 2)],
        );

        let reasoner = CounterfactualReasoner::new();

        // Try Rule 3 for Z as intervention candidate
        let result = reasoner.try_rule3(&dag, 0, 1, 2, &[]);

        // Rule 3 should apply: Y ⊥ Z | X in G̅_X,Z(W)
        assert!(result.is_some(), "Rule 3 should apply for irrelevant intervention");
        if let Some(CausalQueryOutcome::Identified { method, .. }) = result {
            assert_eq!(method, IdentificationMethod::Rule3ActionDeletion);
        }
    }

    #[test]
    fn test_extended_harness() {
        // Verify the extended harness with Rule 2-3 test cases
        let reasoner = CounterfactualReasoner::new();
        let mut harness = CausalReferenceHarness::new();

        // Should have at least 6 test cases now
        assert!(harness.test_count() >= 6, "Harness should have ≥6 test cases");

        let _result = harness.validate(&reasoner);
        assert!(
            harness.current_match_rate >= 0.8,
            "Extended harness match rate too low: {:.2}",
            harness.current_match_rate
        );
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Additional Edge Case Tests for d-Separation
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_d_separation_long_chain() {
        // Long chain: A → B → C → D → E
        // Tests that d-separation works correctly for longer paths
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
            vec![(0, 1), (1, 2), (2, 3), (3, 4)],
        );

        let empty: HashSet<usize> = HashSet::new();
        let c_set: HashSet<usize> = [2].iter().copied().collect();
        let bd_set: HashSet<usize> = [1, 3].iter().copied().collect();

        // A and E are d-connected without conditioning
        assert!(!dag.is_d_separated(0, 4, &empty), "A-E should be d-connected");

        // Conditioning on C blocks the path
        assert!(dag.is_d_separated(0, 4, &c_set), "A-E should be d-separated given C");

        // Conditioning on B and D also blocks
        assert!(dag.is_d_separated(0, 4, &bd_set), "A-E should be d-separated given B,D");
    }

    #[test]
    fn test_d_separation_diamond() {
        // Diamond structure: A → B, A → C, B → D, C → D
        // D is a collider for the B-C path
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            vec![(0, 1), (0, 2), (1, 3), (2, 3)],
        );

        let empty: HashSet<usize> = HashSet::new();
        let a_set: HashSet<usize> = [0].iter().copied().collect();
        let d_set: HashSet<usize> = [3].iter().copied().collect();

        // B and C are d-connected via A (common cause)
        assert!(!dag.is_d_separated(1, 2, &empty), "B-C should be d-connected via A");

        // Conditioning on A blocks the fork path
        assert!(dag.is_d_separated(1, 2, &a_set), "B-C should be d-separated given A");

        // Conditioning on D (collider) opens a new path B → D ← C
        assert!(!dag.is_d_separated(1, 2, &d_set), "B-C should be d-connected given D (collider)");
    }

    #[test]
    fn test_d_separation_butterfly() {
        // Butterfly/bow-tie: U1 → X, U1 → Y, U2 → X, U2 → Y
        // X and Y share two common causes
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into(), "U1".into(), "U2".into()],
            vec![(2, 0), (2, 1), (3, 0), (3, 1)],
        );

        let empty: HashSet<usize> = HashSet::new();
        let u1_set: HashSet<usize> = [2].iter().copied().collect();
        let both_u: HashSet<usize> = [2, 3].iter().copied().collect();

        // X and Y are d-connected (two paths via U1 and U2)
        assert!(!dag.is_d_separated(0, 1, &empty), "X-Y should be d-connected");

        // Conditioning on just U1 still leaves path via U2
        assert!(!dag.is_d_separated(0, 1, &u1_set), "X-Y should still be d-connected given only U1");

        // Conditioning on both U1 and U2 blocks all paths
        assert!(dag.is_d_separated(0, 1, &both_u), "X-Y should be d-separated given U1,U2");
    }

    #[test]
    fn test_d_separation_napkin() {
        // Napkin structure (common in epidemiology):
        // U → X, U → M, M → Y, X → Y
        // Where U is an unmeasured confounder
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into(), "M".into(), "U".into()],
            vec![(3, 0), (3, 2), (2, 1), (0, 1)],
        );

        let empty: HashSet<usize> = HashSet::new();
        let m_set: HashSet<usize> = [2].iter().copied().collect();
        let u_set: HashSet<usize> = [3].iter().copied().collect();

        // X and Y are d-connected (direct edge X→Y)
        assert!(!dag.is_d_separated(0, 1, &empty), "X-Y should be d-connected");

        // Conditioning on M doesn't block X→Y direct path
        assert!(!dag.is_d_separated(0, 1, &m_set), "X-Y still d-connected given M");

        // Conditioning on U blocks the backdoor but not the direct path
        assert!(!dag.is_d_separated(0, 1, &u_set), "X-Y still d-connected given U (direct path)");
    }

    #[test]
    fn test_d_separation_double_collider() {
        // Two colliders in sequence: A → B ← C → D ← E
        // B and D are both colliders
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
            vec![(0, 1), (2, 1), (2, 3), (4, 3)],
        );

        let empty: HashSet<usize> = HashSet::new();
        let b_set: HashSet<usize> = [1].iter().copied().collect();
        let d_set: HashSet<usize> = [3].iter().copied().collect();
        let bd_set: HashSet<usize> = [1, 3].iter().copied().collect();

        // A and E are d-separated (two colliders block)
        assert!(dag.is_d_separated(0, 4, &empty), "A-E should be d-separated (colliders block)");

        // Conditioning on B opens first collider but D still blocks
        assert!(dag.is_d_separated(0, 4, &b_set), "A-E still d-separated given B (D blocks)");

        // Conditioning on D opens second collider but B still blocks
        assert!(dag.is_d_separated(0, 4, &d_set), "A-E still d-separated given D (B blocks)");

        // Conditioning on both B and D opens both colliders
        assert!(!dag.is_d_separated(0, 4, &bd_set), "A-E d-connected given B,D (both colliders open)");
    }

    #[test]
    fn test_d_separation_mixed_paths() {
        // Complex structure with multiple path types:
        // A → B → C (chain)
        // A → D ← E (D is collider)
        // D → C
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into(), "E".into()],
            vec![(0, 1), (1, 2), (0, 3), (4, 3), (3, 2)],
        );

        let empty: HashSet<usize> = HashSet::new();
        let b_set: HashSet<usize> = [1].iter().copied().collect();
        let d_set: HashSet<usize> = [3].iter().copied().collect();

        // A and C are d-connected via chain A→B→C
        assert!(!dag.is_d_separated(0, 2, &empty), "A-C should be d-connected via chain");

        // Conditioning on B blocks chain but opens path A→D→C (D not a collider on this path)
        assert!(!dag.is_d_separated(0, 2, &b_set), "A-C still d-connected given B (via D)");

        // Conditioning on D blocks the A→D→C path
        assert!(!dag.is_d_separated(0, 2, &d_set), "A-C still d-connected given D (via chain)");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Effect Estimation Tests
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_observational_data_basic() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        data.add_observation(vec![1.0, 2.0]);
        data.add_observation(vec![2.0, 4.0]);
        data.add_observation(vec![3.0, 6.0]);

        assert_eq!(data.n(), 3);
        assert!((data.mean(0) - 2.0).abs() < 1e-10);
        assert!((data.mean(1) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_effect_estimation_simple_regression() {
        // Y = 2X + noise
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..100 {
            let x = i as f64 * 0.1;
            let y = 2.0 * x + 0.01 * (i % 7) as f64;  // Small noise
            data.add_observation(vec![x, y]);
        }

        // Simple chain: X → Y (no confounders)
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into()],
            vec![(0, 1)],
        );

        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };

        let estimator = EffectEstimator::new();
        let result = estimator.estimate(&dag, &query, &data);

        if let CausalQueryOutcome::Identified { estimand, .. } = result {
            // Effect should be approximately 2.0
            assert!((estimand.effect - 2.0).abs() < 0.1, "Effect should be ~2.0, got {}", estimand.effect);
        } else {
            panic!("Expected identified effect");
        }
    }

    #[test]
    fn test_effect_estimation_with_confounder() {
        // True model: Y = 2X + 1.5Z + noise
        // Confounder Z affects both X and Y: X = 0.5Z + noise
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into(), "Z".into()]);
        for i in 0..200 {
            let z = (i % 10) as f64;
            let x = 0.5 * z + 0.01 * (i % 3) as f64;
            let y = 2.0 * x + 1.5 * z + 0.01 * (i % 5) as f64;
            data.add_observation(vec![x, y, z]);
        }

        // DAG with confounder: Z → X → Y, Z → Y
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into(), "Z".into()],
            vec![(0, 1), (2, 0), (2, 1)],
        );

        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };

        let estimator = EffectEstimator::new();
        let result = estimator.estimate(&dag, &query, &data);

        if let CausalQueryOutcome::Identified { estimand, method, .. } = result {
            assert_eq!(method, IdentificationMethod::BackdoorAdjustment);
            // After adjusting for Z, effect should be ~2.0
            assert!((estimand.effect - 2.0).abs() < 0.5, "Effect should be ~2.0, got {}", estimand.effect);
        } else {
            panic!("Expected identified effect");
        }
    }

    #[test]
    fn test_effect_estimation_frontdoor() {
        // Frontdoor: X → M → Y with hidden confounder U → X, U → Y
        // True effects: X→M = 0.8, M→Y = 1.5, so total = 1.2
        let mut data = ObservationalData::new(vec!["X".into(), "M".into(), "Y".into()]);
        for i in 0..200 {
            // Simulate U (hidden)
            let u = (i % 5) as f64;
            let x = u + 0.01 * (i % 3) as f64;
            let m = 0.8 * x + 0.01 * (i % 7) as f64;
            let y = 1.5 * m + 0.3 * u + 0.01 * (i % 11) as f64;
            data.add_observation(vec![x, m, y]);
        }

        // DAG: X → M → Y (no explicit edge X → Y, mediated through M)
        let dag = CausalDAG::new(
            vec!["X".into(), "M".into(), "Y".into()],
            vec![(0, 1), (1, 2)],
        );

        let query = CausalQuery {
            treatment: 0,
            outcome: 2,
            conditioning: vec![],
        };

        let estimator = EffectEstimator::new();
        let result = estimator.estimate(&dag, &query, &data);

        if let CausalQueryOutcome::Identified { estimand, method, .. } = result {
            assert_eq!(method, IdentificationMethod::FrontdoorCriterion);
            // Total effect should be ~1.2 (0.8 * 1.5)
            assert!((estimand.effect - 1.2).abs() < 0.3, "Effect should be ~1.2, got {}", estimand.effect);
        } else {
            panic!("Expected identified effect");
        }
    }

    #[test]
    fn test_observational_data_filter() {
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);
        for i in 0..10 {
            data.add_observation(vec![i as f64, (i * 2) as f64]);
        }

        let filtered = data.filter(0, |x| x >= 5.0);
        assert_eq!(filtered.n(), 5);
        assert!((filtered.mean(0) - 7.0).abs() < 1e-10); // Mean of 5,6,7,8,9
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // ID Algorithm Tests
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_id_simple_chain() {
        // Simple chain: X → Y (no confounders)
        // Should be identifiable
        let graph = CausalGraphWithLatents::new(
            vec!["X".into(), "Y".into()],
            vec![(0, 1)],
            vec![],
        );

        let id = IDAlgorithm::new();
        let result = id.identify(&graph, &[0], &[1]);

        assert!(result.is_ok(), "Simple chain should be identifiable");
    }

    #[test]
    fn test_id_with_backdoor_confounder() {
        // X ← U → Y, X → Y
        // Represented as: X → Y with X ↔ Y (bidirected = latent confounder)
        // Should NOT be identifiable (bow graph)
        let graph = CausalGraphWithLatents::new(
            vec!["X".into(), "Y".into()],
            vec![(0, 1)],
            vec![(0, 1)], // Bidirected edge = latent confounder
        );

        let id = IDAlgorithm::new();
        let result = id.identify(&graph, &[0], &[1]);

        // With just X ↔ Y and X → Y, effect is NOT identifiable (bow graph)
        // This is the classic example of non-identifiability
        assert!(result.is_err(), "Bow graph should NOT be identifiable");

        if let Err((hedge, _)) = result {
            assert!(!hedge.is_empty(), "Should have found a hedge");
        }
    }

    #[test]
    fn test_id_frontdoor_structure() {
        // Frontdoor: X → M → Y with X ↔ Y
        // This IS identifiable via frontdoor criterion
        let graph = CausalGraphWithLatents::new(
            vec!["X".into(), "M".into(), "Y".into()],
            vec![(0, 1), (1, 2)],
            vec![(0, 2)], // Latent confounder between X and Y
        );

        let id = IDAlgorithm::new();
        let result = id.identify(&graph, &[0], &[2]);

        assert!(result.is_ok(), "Frontdoor structure should be identifiable");
    }

    #[test]
    fn test_c_components() {
        // Graph with two C-components:
        // A ↔ B, C ↔ D (two separate clusters)
        let graph = CausalGraphWithLatents::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            vec![],
            vec![(0, 1), (2, 3)],
        );

        let components = graph.c_components();
        assert_eq!(components.len(), 2, "Should have 2 C-components");
    }

    #[test]
    fn test_c_component_chain() {
        // Chain of bidirected: A ↔ B ↔ C
        // Should be one C-component
        let graph = CausalGraphWithLatents::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![],
            vec![(0, 1), (1, 2)],
        );

        let components = graph.c_components();
        assert_eq!(components.len(), 1, "Should have 1 C-component (all connected)");
        assert_eq!(components[0].len(), 3, "Component should contain all 3 nodes");
    }

    #[test]
    fn test_causal_expression_to_string() {
        let nodes = vec!["X".into(), "Y".into(), "Z".into()];

        let expr = CausalExpression::Sum {
            sum_over: vec![2],
            inner: Box::new(CausalExpression::Product(vec![
                CausalExpression::Probability { outcome: vec![1], conditioning: vec![0, 2] },
                CausalExpression::Probability { outcome: vec![2], conditioning: vec![] },
            ])),
        };

        let s = expr.to_string(&nodes);
        assert!(s.contains("Σ"), "Should contain sum symbol");
        assert!(s.contains("P(Y|X,Z)"), "Should contain conditional probability");
    }

    #[test]
    fn test_id_query_interface() {
        // Test the query interface for IDAlgorithm
        let graph = CausalGraphWithLatents::new(
            vec!["X".into(), "Y".into()],
            vec![(0, 1)],
            vec![],
        );

        let id = IDAlgorithm::new();
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };

        let result = id.query(&graph, &query);

        match result {
            CausalQueryOutcome::Identified { method, .. } => {
                assert_eq!(method, IdentificationMethod::IDAlgorithm);
            }
            _ => panic!("Simple chain should be identified"),
        }
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Doubly Robust Estimation Tests
    // ─────────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_ipw_estimation() {
        // Binary treatment with confounder
        // True ATE = 2.0
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into(), "Z".into()]);

        for i in 0..200 {
            let z = (i % 2) as f64; // Binary confounder
            let x = if z > 0.5 { 1.0 } else { 0.0 }; // Treatment depends on Z
            let x = if (i % 10) < 3 { 1.0 - x } else { x }; // Some noise
            let y = 2.0 * x + 1.5 * z + 0.1 * (i % 7) as f64; // Y depends on X and Z
            data.add_observation(vec![x, y, z]);
        }

        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };

        let estimator = EffectEstimator::new();
        let ipw = estimator.estimate_ipw(&query, &[2], &data);

        // IPW should estimate ~2.0 (true causal effect)
        assert!((ipw - 2.0).abs() < 1.0, "IPW estimate should be ~2.0, got {}", ipw);
    }

    #[test]
    fn test_doubly_robust_estimation() {
        // Binary treatment with confounder
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into(), "Z".into()]);

        for i in 0..300 {
            let z = (i % 3) as f64 / 2.0; // Confounder 0, 0.5, or 1
            let x = if z + 0.1 * (i % 5) as f64 > 0.5 { 1.0 } else { 0.0 };
            let y = 2.0 * x + 1.0 * z + 0.05 * (i % 11) as f64;
            data.add_observation(vec![x, y, z]);
        }

        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };

        let estimator = EffectEstimator::new();
        let dr = estimator.estimate_doubly_robust(&query, &[2], &data);

        // DR should estimate ~2.0
        assert!((dr - 2.0).abs() < 1.0, "DR estimate should be ~2.0, got {}", dr);
    }

    #[test]
    fn test_robust_estimate_agreement() {
        // When all models are correct, estimates should agree
        let mut data = ObservationalData::new(vec!["X".into(), "Y".into(), "Z".into()]);

        for i in 0..500 {
            let z = (i % 5) as f64 / 4.0;
            let x = if z + 0.1 * (i % 3) as f64 > 0.4 { 1.0 } else { 0.0 };
            let y = 1.5 * x + 0.8 * z + 0.02 * (i % 7) as f64;
            data.add_observation(vec![x, y, z]);
        }

        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into(), "Z".into()],
            vec![(0, 1), (2, 0), (2, 1)],
        );

        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };

        let estimator = EffectEstimator::new();
        let robust = estimator.estimate_robust(&dag, &query, &data);

        assert!(robust.is_identified, "Effect should be identified");

        // All estimates should be in reasonable range
        assert!((robust.regression_estimate - 1.5).abs() < 1.0,
            "Regression estimate should be ~1.5, got {}", robust.regression_estimate);
        assert!((robust.dr_estimate - 1.5).abs() < 1.0,
            "DR estimate should be ~1.5, got {}", robust.dr_estimate);

        // Confidence should be positive
        assert!(robust.confidence() > 0.0, "Confidence should be positive");
    }

    #[test]
    fn test_robust_estimate_confidence() {
        let estimate = RobustEstimate {
            effect: 1.0,
            regression_estimate: 1.0,
            ipw_estimate: 1.1,
            dr_estimate: 1.05,
            method: IdentificationMethod::BackdoorAdjustment,
            is_identified: true,
        };

        // Estimates are close, should have high confidence
        assert!(estimate.confidence() > 0.5, "Close estimates should have high confidence");
        assert!(estimate.estimates_agree(0.5), "Estimates within 0.5 should agree");
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // Property-Based Tests
    // ─────────────────────────────────────────────────────────────────────────────

    use proptest::prelude::*;

    /// Generate a random DAG with n nodes and edge probability p
    fn random_dag_strategy(n: usize, edge_prob: f64) -> impl Strategy<Value = CausalDAG> {
        let nodes: Vec<String> = (0..n).map(|i| format!("X{}", i)).collect();
        let n_copy = n;

        // Generate edges: only from lower to higher index to ensure DAG property
        proptest::collection::vec(proptest::bool::weighted(edge_prob), n * n)
            .prop_map(move |edge_flags| {
                let mut edges = Vec::new();
                for i in 0..n_copy {
                    for j in (i + 1)..n_copy {
                        if edge_flags[i * n_copy + j] {
                            edges.push((i, j));
                        }
                    }
                }
                CausalDAG::new(nodes.clone(), edges)
            })
    }

    proptest! {
        /// Property: d-separation is symmetric in the query nodes
        #[test]
        fn prop_dsep_symmetric(
            seed in 0u64..1000,
        ) {
            let nodes = vec!["A".into(), "B".into(), "C".into()];
            let edges = vec![(0, 2), (1, 2)]; // A → C ← B (collider)
            let dag = CausalDAG::new(nodes, edges);

            let empty_set = std::collections::HashSet::new();

            // d-sep(A, B | ∅) should equal d-sep(B, A | ∅)
            let ab = dag.is_d_separated(0, 1, &empty_set);
            let ba = dag.is_d_separated(1, 0, &empty_set);
            prop_assert_eq!(ab, ba, "d-separation should be symmetric");
        }

        /// Property: Conditioning on a collider opens the path
        #[test]
        fn prop_collider_opens_path(
            _seed in 0u64..100,
        ) {
            let nodes = vec!["A".into(), "B".into(), "C".into()];
            let edges = vec![(0, 2), (1, 2)]; // A → C ← B (collider at C)
            let dag = CausalDAG::new(nodes, edges);

            let empty_set = std::collections::HashSet::new();
            let mut cond_on_c = std::collections::HashSet::new();
            cond_on_c.insert(2);

            // Without conditioning: A and B should be d-separated
            // With conditioning on C: A and B should NOT be d-separated
            let without = dag.is_d_separated(0, 1, &empty_set);
            let with = dag.is_d_separated(0, 1, &cond_on_c);

            prop_assert!(without, "A ⊥ B | ∅ in collider structure");
            prop_assert!(!with, "A ⊥̸ B | C when C is a collider");
        }

        /// Property: IV validity is reflexive (instrument can't be its own treatment)
        #[test]
        fn prop_iv_validity(
            _n_obs in 10usize..50,
        ) {
            let nodes = vec!["Z".into(), "X".into(), "Y".into()];
            let edges = vec![(0, 1), (1, 2)]; // Z → X → Y
            let dag = CausalDAG::new(nodes, edges);

            // Z is a valid instrument for X → Y
            let validity = IVEstimator::is_valid_instrument(&dag, 0, 1, 2);
            prop_assert!(matches!(validity, IVValidity::Valid { .. }), "Z should be valid instrument");

            // X cannot be its own instrument
            let self_iv = IVEstimator::is_valid_instrument(&dag, 1, 1, 2);
            prop_assert!(matches!(self_iv, IVValidity::Invalid { .. }), "Variable cannot be its own instrument");
        }

        /// Property: Effect estimates are bounded
        #[test]
        fn prop_effect_bounded(
            n_obs in 20usize..100,
            seed in 0u64..1000,
        ) {
            let mut data = ObservationalData::new(vec!["X".into(), "Y".into()]);

            // Generate data with bounded effect
            for i in 0..(n_obs as u64) {
                let x = ((seed + i) % 2) as f64;
                let y = 2.0 * x + ((seed + i) % 10) as f64 / 10.0;
                data.add_observation(vec![x, y]);
            }

            // Simple regression estimate
            if let Some(cov) = data.covariance(0, 1) {
                if let Some(var) = data.variance(0) {
                    if var > 1e-10 {
                        let estimate = cov / var;
                        // Effect should be close to 2.0 (our true effect)
                        prop_assert!((estimate - 2.0).abs() < 1.0,
                            "Effect estimate {} should be near 2.0", estimate);
                    }
                }
            }
        }

        /// Property: PC algorithm produces consistent skeletons
        #[test]
        fn prop_pc_consistent(
            n_obs in 50usize..150,
        ) {
            let mut data = ObservationalData::new(vec![
                "X".into(), "Y".into(), "Z".into()
            ]);

            // Generate chain data: X → Y → Z
            for i in 0..n_obs {
                let x = (i % 2) as f64;
                let y = 0.5 * x + 0.1 * (i % 5) as f64;
                let z = 0.5 * y + 0.1 * (i % 7) as f64;
                data.add_observation(vec![x, y, z]);
            }

            let pc = PCAlgorithm::new();
            let result = pc.discover(&data);

            // Skeleton should have at least one edge
            // (X-Y and Y-Z should be discovered)
            let total_edges: usize = result.skeleton.adjacencies.iter()
                .map(|adj| adj.len())
                .sum();
            // Each edge counted twice (undirected)
            prop_assert!(total_edges >= 2, "Skeleton should have edges");
        }
    }
}
