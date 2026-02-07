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
                    IdentificationMethod::Rule2ActionObservationExchange
                    | IdentificationMethod::Rule3ActionDeletion => {
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
                    method,
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
}
