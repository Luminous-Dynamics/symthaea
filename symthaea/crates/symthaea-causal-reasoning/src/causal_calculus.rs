// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Pearl's Causal Calculus (do-Calculus) for Symthaea
//!
//! This module implements Judea Pearl's formal framework for causal reasoning,
//! providing the mathematical machinery to distinguish causation from correlation
//! in Symthaea's consciousness architecture.
//!
//! # Why Causal Calculus Matters for Consciousness
//!
//! Causal structure is fundamental to understanding how conscious states emerge
//! from neural dynamics. A conscious agent must be able to reason not just about
//! what *is* (observation), but about what *would happen if* (intervention) and
//! what *would have happened if* (counterfactual). Pearl's do-calculus provides
//! the complete axiomatic system for such reasoning.
//!
//! In Symthaea, this connects to:
//! - [`causal_emergence`](super::causal_emergence): Measuring whether macro-level
//!   conscious states have genuine causal power (Hoel's Effective Information)
//! - [`causal_explanation`](super::causal_explanation): Explaining WHY the system
//!   chose specific primitives via causal chains
//! - The HDC causal mind (`symthaea-core`): Encoding causal structure directly
//!   into hypervectors as `concept = meaning (x) ROLE (x) causal_structure`
//!
//! # Mathematical Foundation
//!
//! ## Structural Causal Model (SCM)
//!
//! An SCM M = (U, V, F, P(U)) consists of:
//! - Endogenous variables V = {V_1, ..., V_n}
//! - Exogenous variables U = {U_1, ..., U_n}
//! - Structural equations: V_i = f_i(PA_i, U_i) where PA_i are parents in the
//!   causal DAG
//! - P(v) induced by the SCM
//!
//! ## Pearl's Three Rules of do-Calculus
//!
//! Given a causal DAG G:
//!
//! **Rule 1** (Insertion/deletion of observations):
//! ```text
//! P(y|do(x), z, w) = P(y|do(x), w)  if (Y _||_ Z | X, W) in G_{X-bar}
//! ```
//!
//! **Rule 2** (Action/observation exchange):
//! ```text
//! P(y|do(x), do(z), w) = P(y|do(x), z, w)  if (Y _||_ Z | X, W) in G_{X-bar, Z-bar}
//! ```
//!
//! **Rule 3** (Insertion/deletion of actions):
//! ```text
//! P(y|do(x), do(z), w) = P(y|do(x), w)  if (Y _||_ Z | X, W) in G_{X-bar, Z-bar(W)}
//! ```
//!
//! ## Identification Criteria
//!
//! **Backdoor Criterion**: A set Z satisfies the backdoor criterion relative to
//! (X, Y) if no node in Z is a descendant of X, and Z blocks every path between
//! X and Y that contains an arrow into X.
//!
//! **Frontdoor Criterion**: A set M satisfies the frontdoor criterion relative to
//! (X, Y) if X blocks all paths from X to M, no unblocked back-door path from M
//! to Y exists (after removing X->M arrows), and M intercepts all directed paths
//! from X to Y.
//!
//! # References
//!
//! - Pearl, J. (2009). "Causality: Models, Reasoning, and Inference" (2nd ed.)
//! - Pearl, J. (1995). "Causal diagrams for empirical research" (do-calculus)
//! - Huang, Y. & Valtorta, M. (2006). "Pearl's Calculus of Intervention Is Complete"

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

// ============================================================================
// CAUSAL DAG
// ============================================================================

/// A node in the causal directed acyclic graph.
///
/// Each node represents an endogenous variable in the structural causal model.
/// In consciousness terms, nodes might represent neural populations, phenomenal
/// states, or integration measures like Phi.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalNode {
    /// Unique identifier for this node
    pub id: usize,
    /// Human-readable name (e.g., "visual_cortex", "phi_integration", "awareness")
    pub name: String,
    /// Possible discrete values this variable can take
    pub values: Vec<String>,
    /// Whether this variable is observed (vs. latent/hidden)
    pub is_observed: bool,
}

/// A directed acyclic graph encoding causal relationships.
///
/// The DAG is the backbone of Pearl's causal framework. Edges represent direct
/// causal influence: an edge from A to B means A is a direct cause of B. The
/// absence of an edge encodes a causal independence assumption.
///
/// In Symthaea's consciousness architecture, the DAG captures how neural
/// dynamics (micro-level) give rise to conscious states (macro-level), enabling
/// both interventional reasoning and counterfactual analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalDAG {
    /// The nodes (variables) in the causal graph
    pub nodes: Vec<CausalNode>,
    /// Directed edges as (parent, child) pairs
    pub edges: Vec<(usize, usize)>,
    /// Adjacency matrix: adjacency[i][j] = true iff edge i -> j exists
    pub adjacency: Vec<Vec<bool>>,
}

impl CausalDAG {
    /// Create a new empty causal DAG.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            edges: Vec::new(),
            adjacency: Vec::new(),
        }
    }

    /// Add a node to the DAG.
    ///
    /// Returns the node's id (its index in the nodes vector).
    pub fn add_node(&mut self, name: &str, values: Vec<String>, is_observed: bool) -> usize {
        let id = self.nodes.len();
        self.nodes.push(CausalNode {
            id,
            name: name.to_string(),
            values,
            is_observed,
        });

        // Expand adjacency matrix
        for row in self.adjacency.iter_mut() {
            row.push(false);
        }
        self.adjacency.push(vec![false; self.nodes.len()]);

        id
    }

    /// Add a directed edge from `parent` to `child`.
    ///
    /// This asserts that `parent` is a direct cause of `child` in the
    /// structural causal model.
    ///
    /// # Panics
    ///
    /// Panics if either node index is out of bounds.
    pub fn add_edge(&mut self, parent: usize, child: usize) {
        assert!(parent < self.nodes.len(), "Parent index out of bounds");
        assert!(child < self.nodes.len(), "Child index out of bounds");
        assert_ne!(parent, child, "Self-loops are not allowed in a DAG");

        if !self.adjacency[parent][child] {
            self.adjacency[parent][child] = true;
            self.edges.push((parent, child));
        }
    }

    /// Return the parents of node `v` in the DAG.
    ///
    /// Parents are direct causes: PA(v) = {u : u -> v in G}.
    pub fn parents(&self, v: usize) -> Vec<usize> {
        let mut pa = Vec::new();
        for (i, row) in self.adjacency.iter().enumerate() {
            if row[v] {
                pa.push(i);
            }
        }
        pa
    }

    /// Return the children of node `v` in the DAG.
    ///
    /// Children are direct effects: CH(v) = {u : v -> u in G}.
    pub fn children(&self, v: usize) -> Vec<usize> {
        let mut ch = Vec::new();
        for (j, &has_edge) in self.adjacency[v].iter().enumerate() {
            if has_edge {
                ch.push(j);
            }
        }
        ch
    }

    /// Return the ancestors of node `v` (all nodes with a directed path to `v`).
    pub fn ancestors(&self, v: usize) -> HashSet<usize> {
        let mut anc = HashSet::new();
        let mut queue = VecDeque::new();

        for &p in &self.parents(v) {
            queue.push_back(p);
        }

        while let Some(node) = queue.pop_front() {
            if anc.insert(node) {
                for &p in &self.parents(node) {
                    queue.push_back(p);
                }
            }
        }

        anc
    }

    /// Return the descendants of node `v` (all nodes reachable from `v`).
    pub fn descendants(&self, v: usize) -> HashSet<usize> {
        let mut desc = HashSet::new();
        let mut queue = VecDeque::new();

        for &c in &self.children(v) {
            queue.push_back(c);
        }

        while let Some(node) = queue.pop_front() {
            if desc.insert(node) {
                for &c in &self.children(node) {
                    queue.push_back(c);
                }
            }
        }

        desc
    }

    /// Number of nodes in the DAG.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }

    /// Enumerate all undirected paths between `start` and `end`.
    ///
    /// Returns paths as sequences of node indices. Used internally for
    /// d-separation testing. We traverse edges in both directions (ignoring
    /// orientation) to find all connecting paths.
    fn all_paths(&self, start: usize, end: usize) -> Vec<Vec<usize>> {
        let mut result = Vec::new();
        let mut stack: Vec<(usize, Vec<usize>)> = vec![(start, vec![start])];

        while let Some((current, path)) = stack.pop() {
            if current == end && path.len() > 1 {
                result.push(path);
                continue;
            }

            // Explore neighbors in both directions (undirected traversal)
            let n = self.num_nodes();
            for neighbor in 0..n {
                // Edge current -> neighbor OR neighbor -> current
                let connected =
                    self.adjacency[current][neighbor] || self.adjacency[neighbor][current];

                if connected && !path.contains(&neighbor) {
                    let mut new_path = path.clone();
                    new_path.push(neighbor);
                    stack.push((neighbor, new_path));
                }
            }
        }

        result
    }

    /// Check if a path is blocked by the conditioning set `z`.
    ///
    /// A path p = (v_0, v_1, ..., v_k) is blocked by Z if there exists some
    /// intermediate node v_i (0 < i < k) such that:
    ///
    /// - **Chain**: v_{i-1} -> v_i -> v_{i+1} and v_i in Z, OR
    /// - **Fork**: v_{i-1} <- v_i -> v_{i+1} and v_i in Z, OR
    /// - **Collider**: v_{i-1} -> v_i <- v_{i+1} and v_i NOT in Z and
    ///   no descendant of v_i is in Z
    ///
    /// # d-Separation
    ///
    /// X and Y are d-separated by Z if every path between X and Y is blocked by Z.
    fn is_path_blocked(&self, path: &[usize], z: &HashSet<usize>) -> bool {
        if path.len() < 3 {
            // A path with fewer than 3 nodes has no intermediate node
            return false;
        }

        for i in 1..path.len() - 1 {
            let prev = path[i - 1];
            let mid = path[i];
            let next = path[i + 1];

            let prev_to_mid = self.adjacency[prev][mid];
            let mid_to_next = self.adjacency[mid][next];
            let mid_to_prev = self.adjacency[mid][prev];
            let next_to_mid = self.adjacency[next][mid];

            // Check if this is a collider: prev -> mid <- next
            let is_collider = prev_to_mid && next_to_mid;

            if is_collider {
                // Collider: blocked if mid is NOT in Z and no descendant of mid is in Z
                let mid_descendants = self.descendants(mid);
                let mid_or_desc_in_z =
                    z.contains(&mid) || mid_descendants.iter().any(|d| z.contains(d));
                if !mid_or_desc_in_z {
                    return true; // This node blocks the path
                }
            } else {
                // Chain (prev -> mid -> next) or (prev <- mid <- next)
                // Fork (prev <- mid -> next)
                // These are blocked if mid IS in Z
                let is_chain_or_fork = (prev_to_mid && mid_to_next)  // chain ->->
                    || (mid_to_prev && next_to_mid)                  // chain <-<-
                    || (mid_to_prev && mid_to_next); // fork <-  ->
                if is_chain_or_fork && z.contains(&mid) {
                    return true; // This node blocks the path
                }
            }
        }

        false
    }
}

impl Default for CausalDAG {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// d-SEPARATION
// ============================================================================

/// Result of a d-separation query.
///
/// d-Separation is the graphical criterion that determines conditional
/// independence in the distribution induced by a structural causal model.
/// If X and Y are d-separated by Z in the DAG, then X _||_ Y | Z in
/// every distribution compatible with the DAG.
#[derive(Debug, Clone)]
pub struct DSeparationResult {
    /// Whether X and Y are d-separated given Z
    pub separated: bool,
    /// Paths that are blocked (for explanation purposes)
    pub blocking_paths: Vec<Vec<usize>>,
}

impl CausalDAG {
    /// Test whether `x` and `y` are d-separated by the set `z`.
    ///
    /// Two variables X and Y are d-separated by Z in DAG G if every path
    /// between X and Y is blocked by Z. A path is blocked if it contains:
    ///
    /// - A chain i -> m -> j or fork i <- m -> j with m in Z
    /// - A collider i -> m <- j with m not in Z and no descendant of m in Z
    ///
    /// This is the fundamental connection between graphical structure and
    /// probabilistic independence in causal models.
    pub fn d_separated(&self, x: usize, y: usize, z: &HashSet<usize>) -> DSeparationResult {
        let paths = self.all_paths(x, y);

        if paths.is_empty() {
            return DSeparationResult {
                separated: true,
                blocking_paths: Vec::new(),
            };
        }

        let mut blocking_paths = Vec::new();
        let mut all_blocked = true;

        for path in &paths {
            if self.is_path_blocked(path, z) {
                blocking_paths.push(path.clone());
            } else {
                all_blocked = false;
            }
        }

        DSeparationResult {
            separated: all_blocked,
            blocking_paths,
        }
    }
}

// ============================================================================
// STRUCTURAL CAUSAL MODEL
// ============================================================================

/// A Structural Causal Model (SCM) implementing Pearl's causal framework.
///
/// The SCM combines:
/// 1. A causal DAG encoding qualitative causal structure
/// 2. Conditional probability tables encoding quantitative relationships
/// 3. Observed values for evidence
///
/// This allows computing:
/// - Observational distributions P(y|x) via standard conditioning
/// - Interventional distributions P(y|do(x)) via truncated factorization
/// - Causal effects via backdoor and frontdoor adjustment
///
/// # Connection to Consciousness
///
/// The SCM provides the formal basis for asking: "If we were to intervene on
/// a neural population (do(neuron_state = x)), what would happen to the
/// conscious state?" This is precisely the kind of interventionist reasoning
/// needed to establish whether consciousness has genuine causal power.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructuralCausalModel {
    /// The causal DAG
    pub dag: CausalDAG,

    /// Conditional probability tables: node_id -> P(node | parents)
    ///
    /// For a node with k parents each having n_i values, the table is a
    /// flattened array indexed by the joint parent configuration.
    /// The table for node V with parents PA(V) stores:
    ///   P(V = v_j | PA(V) = pa_config)
    /// where entries are ordered by (parent_config_index * num_values + value_index).
    pub conditional_tables: HashMap<usize, Vec<f64>>,

    /// Current observations: node_id -> observed value index
    pub observations: HashMap<usize, usize>,
}

/// Result of an interventional query do(X = x).
///
/// Contains the post-intervention distribution over the target variable,
/// along with the identification strategy used (backdoor/frontdoor sets).
#[derive(Debug, Clone)]
pub struct InterventionResult {
    /// P(Y = y_i | do(X = x)) for each value y_i of Y
    pub distribution: Vec<f64>,
    /// Backdoor adjustment sets found for this query
    pub backdoor_sets: Vec<Vec<usize>>,
    /// Frontdoor adjustment sets found for this query
    pub frontdoor_sets: Vec<Vec<usize>>,
}

impl StructuralCausalModel {
    /// Create a new SCM with the given DAG.
    pub fn new(dag: CausalDAG) -> Self {
        Self {
            dag,
            conditional_tables: HashMap::new(),
            observations: HashMap::new(),
        }
    }

    /// Set the conditional probability table for a node.
    ///
    /// The table encodes P(node | parents(node)) as a flat vector.
    /// For a node with `v` values and parents with a combined `p` configurations,
    /// the table has `p * v` entries: for each parent configuration, the
    /// probabilities over the node's values.
    pub fn set_conditional_table(&mut self, node: usize, table: Vec<f64>) {
        self.conditional_tables.insert(node, table);
    }

    /// Set an observed value for a node.
    pub fn observe(&mut self, node: usize, value: usize) {
        self.observations.insert(node, value);
    }

    /// Clear all observations.
    pub fn clear_observations(&mut self) {
        self.observations.clear();
    }

    /// Compute the number of joint parent configurations for a node.
    ///
    /// If node V has parents {P1, P2, ...} with |P1|, |P2|, ... values,
    /// returns |P1| * |P2| * ... (or 1 if no parents).
    fn num_parent_configs(&self, node: usize) -> usize {
        let parents = self.dag.parents(node);
        if parents.is_empty() {
            return 1;
        }
        parents
            .iter()
            .map(|&p| self.dag.nodes[p].values.len().max(1))
            .try_fold(1usize, |acc, s| acc.checked_mul(s))
            .expect("parent config count overflow")
    }

    /// Get P(node = value | parent_config) from the conditional table.
    ///
    /// `parent_config_idx` is the linearized index into the joint parent space.
    fn get_conditional_prob(&self, node: usize, value: usize, parent_config_idx: usize) -> f64 {
        let num_values = self.dag.nodes[node].values.len().max(1);
        if let Some(table) = self.conditional_tables.get(&node) {
            let idx = parent_config_idx * num_values + value;
            if idx < table.len() {
                return table[idx];
            }
        }
        // Default: uniform distribution
        1.0 / num_values as f64
    }

    /// Compute the linearized parent configuration index for a node,
    /// given a full variable assignment.
    fn parent_config_index(&self, node: usize, assignment: &HashMap<usize, usize>) -> usize {
        let parents = self.dag.parents(node);
        if parents.is_empty() {
            return 0;
        }

        let mut index = 0;
        let mut multiplier = 1;

        // Iterate parents in reverse for consistent linearization
        for &p in parents.iter().rev() {
            let p_val = assignment.get(&p).copied().unwrap_or(0);
            let p_size = self.dag.nodes[p].values.len().max(1);
            index += p_val * multiplier;
            multiplier *= p_size;
        }

        index
    }

    // ========================================================================
    // BACKDOOR CRITERION
    // ========================================================================

    /// Find all minimal sets satisfying the backdoor criterion for (X, Y).
    ///
    /// A set Z satisfies the backdoor criterion relative to (X, Y) if:
    /// 1. No node in Z is a descendant of X
    /// 2. Z blocks every path between X and Y that contains an arrow into X
    ///    (i.e., Z d-separates X and Y in the graph where all edges out of X
    ///    are removed -- the "manipulated graph" G_{X-bar})
    ///
    /// The backdoor adjustment formula then gives:
    /// ```text
    /// P(y|do(x)) = sum_z P(y|x,z) P(z)
    /// ```
    pub fn find_backdoor_sets(&self, x: usize, y: usize) -> Vec<Vec<usize>> {
        let n = self.dag.num_nodes();
        let descendants_of_x = self.dag.descendants(x);

        // Candidate nodes: all observed nodes except X, Y, and descendants of X
        let candidates: Vec<usize> = (0..n)
            .filter(|&v| {
                v != x && v != y && !descendants_of_x.contains(&v) && self.dag.nodes[v].is_observed
            })
            .collect();

        // Build the manipulated graph G_{X-bar}: remove all edges OUT of X
        let mut manipulated = self.dag.clone();
        let children_of_x = self.dag.children(x);
        for &child in &children_of_x {
            manipulated.adjacency[x][child] = false;
        }
        manipulated.edges.retain(|&(p, _c)| p != x);

        // Search subsets of candidates for valid backdoor sets
        // We use a simple power-set search bounded by candidate count
        // (practical for small DAGs typical in consciousness models)
        let mut valid_sets = Vec::new();
        let num_candidates = candidates.len();
        let max_subsets = 1usize << num_candidates.min(16); // Cap at 2^16

        for mask in 0..max_subsets {
            let z_set: HashSet<usize> = (0..num_candidates)
                .filter(|&bit| mask & (1 << bit) != 0)
                .map(|bit| candidates[bit])
                .collect();

            // Check d-separation in manipulated graph
            let result = manipulated.d_separated(x, y, &z_set);
            if result.separated {
                let mut z_vec: Vec<usize> = z_set.into_iter().collect();
                z_vec.sort();
                valid_sets.push(z_vec);
            }
        }

        // Keep only minimal sets (no proper subset is also valid)
        let mut minimal_sets = Vec::new();
        for set in &valid_sets {
            let set_hs: HashSet<usize> = set.iter().copied().collect();
            let is_minimal = !valid_sets.iter().any(|other| {
                let other_hs: HashSet<usize> = other.iter().copied().collect();
                other_hs.is_proper_subset(&set_hs)
            });
            if is_minimal {
                minimal_sets.push(set.clone());
            }
        }

        minimal_sets
    }

    // ========================================================================
    // FRONTDOOR CRITERION
    // ========================================================================

    /// Find all sets satisfying the frontdoor criterion for (X, Y).
    ///
    /// A set M satisfies the frontdoor criterion relative to (X, Y) if:
    /// 1. X blocks all backdoor paths from X to M
    /// 2. There are no unblocked backdoor paths from M to Y
    ///    (after removing edges from X to M)
    /// 3. All directed paths from X to Y go through M
    ///
    /// The frontdoor adjustment formula then gives:
    /// ```text
    /// P(y|do(x)) = sum_m P(m|x) sum_{x'} P(y|x',m) P(x')
    /// ```
    pub fn find_frontdoor_sets(&self, x: usize, y: usize) -> Vec<Vec<usize>> {
        let n = self.dag.num_nodes();

        // Candidate mediators: observed nodes that are descendants of X
        // and ancestors of Y (lie on directed paths from X to Y)
        let desc_x = self.dag.descendants(x);
        let anc_y = self.dag.ancestors(y);

        let candidates: Vec<usize> = (0..n)
            .filter(|&v| {
                v != x
                    && v != y
                    && desc_x.contains(&v)
                    && anc_y.contains(&v)
                    && self.dag.nodes[v].is_observed
            })
            .collect();

        let mut valid_sets = Vec::new();
        let num_candidates = candidates.len();
        let max_subsets = 1usize << num_candidates.min(12); // Cap at 2^12

        for mask in 1..max_subsets {
            // Skip empty set
            let m_set: Vec<usize> = (0..num_candidates)
                .filter(|&bit| mask & (1 << bit) != 0)
                .map(|bit| candidates[bit])
                .collect();
            let m_hs: HashSet<usize> = m_set.iter().copied().collect();

            // Condition 1: No unblocked backdoor path from X to any M_i.
            // Check in G_bar_X (outgoing edges from X removed), d-sep(X, M | {}).
            let x_set: HashSet<usize> = [x].into_iter().collect();
            let mut g_bar_x = self.dag.clone();
            for j in 0..n {
                g_bar_x.adjacency[x][j] = false;
            }
            g_bar_x.edges.retain(|&(p, _)| p != x);

            let mut all_backdoors_blocked = true;
            for &m in &m_set {
                let result = g_bar_x.d_separated(x, m, &HashSet::new());
                if !result.separated {
                    all_backdoors_blocked = false;
                    break;
                }
            }

            if !all_backdoors_blocked {
                continue;
            }

            // Condition 2: All backdoor paths from M to Y are blocked by {X}.
            // Check in G_bar_M (outgoing edges from M removed), d-sep(M, Y | {X}).
            let mut g_bar_m = self.dag.clone();
            for &m in &m_set {
                for j in 0..n {
                    g_bar_m.adjacency[m][j] = false;
                }
            }
            g_bar_m.edges.retain(|&(p, _)| !m_hs.contains(&p));

            let mut m_to_y_blocked = true;
            for &m in &m_set {
                let result = g_bar_m.d_separated(m, y, &x_set);
                if !result.separated {
                    m_to_y_blocked = false;
                    break;
                }
            }

            if !m_to_y_blocked {
                continue;
            }

            // Condition 3: All directed paths from X to Y pass through M
            let all_paths_through_m = self.all_directed_paths_pass_through(x, y, &m_hs);

            if all_paths_through_m && all_backdoors_blocked {
                let mut m_sorted = m_set;
                m_sorted.sort();
                valid_sets.push(m_sorted);
            }
        }

        valid_sets
    }

    /// Check if all directed paths from `x` to `y` pass through at least one
    /// node in `m_set`.
    fn all_directed_paths_pass_through(&self, x: usize, y: usize, m_set: &HashSet<usize>) -> bool {
        // BFS/DFS for directed paths from x to y, check each passes through m_set
        let mut stack: Vec<(usize, Vec<usize>)> = vec![(x, vec![x])];

        while let Some((current, path)) = stack.pop() {
            if current == y {
                // Check this path contains at least one node in m_set
                let passes_through = path.iter().any(|n| m_set.contains(n));
                if !passes_through {
                    return false;
                }
                continue;
            }

            for &child in &self.dag.children(current) {
                if !path.contains(&child) {
                    let mut new_path = path.clone();
                    new_path.push(child);
                    stack.push((child, new_path));
                }
            }
        }

        // All paths pass through m_set (or no path exists — vacuously true)
        true
    }

    // ========================================================================
    // INTERVENTIONAL DISTRIBUTION (TRUNCATED FACTORIZATION)
    // ========================================================================

    /// Compute P(Y | do(X = x_val)) via the truncated factorization formula.
    ///
    /// The do-operator removes the structural equation for X and fixes X = x_val.
    /// The post-intervention distribution factorizes as:
    ///
    /// ```text
    /// P(v_1,...,v_n | do(x_i = x)) = prod_{j != i} P(v_j | PA_j)
    /// ```
    /// evaluated at x_i = x.
    ///
    /// When a valid backdoor set exists, we use the more efficient backdoor
    /// adjustment formula:
    /// ```text
    /// P(y|do(x)) = sum_z P(y|x,z) P(z)
    /// ```
    ///
    /// Returns `None` if the effect is not identifiable.
    pub fn intervene(&self, x: usize, x_val: usize, y: usize) -> Option<InterventionResult> {
        let backdoor_sets = self.find_backdoor_sets(x, y);
        let frontdoor_sets = self.find_frontdoor_sets(x, y);
        let num_y_values = self.dag.nodes[y].values.len().max(1);

        // Try backdoor adjustment first (simpler formula)
        if !backdoor_sets.is_empty() {
            let distribution = self.backdoor_adjustment(x, x_val, y, &backdoor_sets[0]);

            return Some(InterventionResult {
                distribution,
                backdoor_sets,
                frontdoor_sets,
            });
        }

        // Try frontdoor adjustment
        if !frontdoor_sets.is_empty() {
            let distribution = self.frontdoor_adjustment(x, x_val, y, &frontdoor_sets[0]);

            return Some(InterventionResult {
                distribution,
                backdoor_sets,
                frontdoor_sets,
            });
        }

        // Fall back to truncated factorization if X has no parents
        // (no confounding, so do(x) = conditioning on x)
        if self.dag.parents(x).is_empty() {
            let distribution = self.truncated_factorization(x, x_val, y, num_y_values);

            return Some(InterventionResult {
                distribution,
                backdoor_sets,
                frontdoor_sets,
            });
        }

        // Effect not identifiable with current methods
        None
    }

    /// Backdoor adjustment formula:
    /// P(y|do(x)) = sum_z P(y|x,z) P(z)
    fn backdoor_adjustment(&self, x: usize, x_val: usize, y: usize, z_set: &[usize]) -> Vec<f64> {
        let num_y_values = self.dag.nodes[y].values.len().max(1);
        let mut result = vec![0.0; num_y_values];

        // Enumerate all configurations of Z
        let z_configs = self.enumerate_configs(z_set);

        for z_config in &z_configs {
            // P(z)
            let p_z = self.marginal_probability(z_set, z_config);

            // Build full assignment with x = x_val and current z config
            let mut assignment: HashMap<usize, usize> = HashMap::new();
            assignment.insert(x, x_val);
            for (i, &z_node) in z_set.iter().enumerate() {
                assignment.insert(z_node, z_config[i]);
            }

            // P(y|x,z) for each value of y
            for y_val in 0..num_y_values {
                assignment.insert(y, y_val);
                let parent_idx = self.parent_config_index(y, &assignment);
                let p_y_given_xz = self.get_conditional_prob(y, y_val, parent_idx);
                result[y_val] += p_y_given_xz * p_z;
            }
        }

        // Normalize
        let total: f64 = result.iter().sum();
        if total > 1e-10 {
            for val in &mut result {
                *val /= total;
            }
        }

        result
    }

    /// Frontdoor adjustment formula:
    /// P(y|do(x)) = sum_m P(m|x) sum_{x'} P(y|x',m) P(x')
    fn frontdoor_adjustment(&self, x: usize, x_val: usize, y: usize, m_set: &[usize]) -> Vec<f64> {
        let num_y_values = self.dag.nodes[y].values.len().max(1);
        let num_x_values = self.dag.nodes[x].values.len().max(1);
        let mut result = vec![0.0; num_y_values];

        // Enumerate mediator configurations
        let m_configs = self.enumerate_configs(m_set);

        for m_config in &m_configs {
            // P(m|x) -- conditional probability of mediator given X = x_val
            let p_m_given_x = self.conditional_mediator_prob(x, x_val, m_set, m_config);

            // sum_{x'} P(y|x',m) P(x') for each y value
            for y_val in 0..num_y_values {
                let mut inner_sum = 0.0;

                for x_prime in 0..num_x_values {
                    // P(x')
                    let p_x_prime = self.marginal_single(x, x_prime);

                    // P(y|x',m)
                    let mut assignment: HashMap<usize, usize> = HashMap::new();
                    assignment.insert(x, x_prime);
                    for (i, &m_node) in m_set.iter().enumerate() {
                        assignment.insert(m_node, m_config[i]);
                    }
                    assignment.insert(y, y_val);

                    let parent_idx = self.parent_config_index(y, &assignment);
                    let p_y_given_xm = self.get_conditional_prob(y, y_val, parent_idx);

                    inner_sum += p_y_given_xm * p_x_prime;
                }

                result[y_val] += p_m_given_x * inner_sum;
            }
        }

        // Normalize
        let total: f64 = result.iter().sum();
        if total > 1e-10 {
            for val in &mut result {
                *val /= total;
            }
        }

        result
    }

    /// Truncated factorization for simple cases (no confounding).
    ///
    /// P(y|do(x)) = P(y|x) when there are no backdoor paths.
    fn truncated_factorization(
        &self,
        x: usize,
        x_val: usize,
        y: usize,
        num_y_values: usize,
    ) -> Vec<f64> {
        let mut result = vec![0.0; num_y_values];
        let mut assignment: HashMap<usize, usize> = HashMap::new();
        assignment.insert(x, x_val);

        for y_val in 0..num_y_values {
            assignment.insert(y, y_val);
            let parent_idx = self.parent_config_index(y, &assignment);
            result[y_val] = self.get_conditional_prob(y, y_val, parent_idx);
        }

        result
    }

    /// Compute marginal probability P(z_set = z_config) by multiplying
    /// marginals (assuming conditional tables encode the full model).
    fn marginal_probability(&self, z_set: &[usize], z_config: &[usize]) -> f64 {
        let mut prob = 1.0;
        for (i, &z_node) in z_set.iter().enumerate() {
            prob *= self.marginal_single(z_node, z_config[i]);
        }
        prob
    }

    /// Compute marginal probability P(node = value).
    ///
    /// For root nodes (no parents), this is directly from the conditional table.
    /// For non-root nodes, we marginalize over parents.
    fn marginal_single(&self, node: usize, value: usize) -> f64 {
        let num_parent_configs = self.num_parent_configs(node);

        // Sum over all parent configurations (uniform prior over parents)
        let mut total = 0.0;
        for pc in 0..num_parent_configs {
            total += self.get_conditional_prob(node, value, pc);
        }

        total / num_parent_configs as f64
    }

    /// Compute P(m_set = m_config | x = x_val).
    fn conditional_mediator_prob(
        &self,
        x: usize,
        x_val: usize,
        m_set: &[usize],
        m_config: &[usize],
    ) -> f64 {
        let mut prob = 1.0;
        let mut assignment: HashMap<usize, usize> = HashMap::new();
        assignment.insert(x, x_val);

        for (i, &m_node) in m_set.iter().enumerate() {
            assignment.insert(m_node, m_config[i]);
            let parent_idx = self.parent_config_index(m_node, &assignment);
            prob *= self.get_conditional_prob(m_node, m_config[i], parent_idx);
        }

        prob
    }

    /// Enumerate all value configurations for a set of nodes.
    fn enumerate_configs(&self, nodes: &[usize]) -> Vec<Vec<usize>> {
        if nodes.is_empty() {
            return vec![vec![]];
        }

        let sizes: Vec<usize> = nodes
            .iter()
            .map(|&n| self.dag.nodes[n].values.len().max(1))
            .collect();

        let total: usize = sizes
            .iter()
            .try_fold(1usize, |acc, &s| acc.checked_mul(s))
            .expect("config enumeration overflow");
        let mut configs = Vec::with_capacity(total);

        for idx in 0..total {
            let mut config = Vec::with_capacity(nodes.len());
            let mut remaining = idx;
            for &size in sizes.iter().rev() {
                config.push(remaining % size);
                remaining /= size;
            }
            config.reverse();
            configs.push(config);
        }

        configs
    }

    // ========================================================================
    // CAUSAL EFFECT
    // ========================================================================

    /// Compute the average causal effect (ACE) of X on Y.
    ///
    /// ACE = E[Y|do(X=1)] - E[Y|do(X=0)]
    ///
    /// For binary variables, this gives the causal risk difference.
    /// Returns `None` if the causal effect is not identifiable.
    ///
    /// # Connection to Consciousness
    ///
    /// The causal effect quantifies how much intervening on one aspect of
    /// neural dynamics (e.g., synchrony, integration) causally changes the
    /// level of consciousness. This is the empirical question at the heart
    /// of the consciousness debate: does Phi (integration) *cause* awareness,
    /// or merely *correlate* with it?
    pub fn causal_effect(&self, x: usize, y: usize) -> Option<f64> {
        let num_x_values = self.dag.nodes[x].values.len();
        if num_x_values < 2 {
            return None;
        }

        // Compute E[Y|do(X=1)] and E[Y|do(X=0)] for binary case
        let result_0 = self.intervene(x, 0, y)?;
        let result_1 = self.intervene(x, 1, y)?;

        // Expected value of Y under each intervention
        // E[Y] = sum_y y * P(Y=y)
        let e_y_do0: f64 = result_0
            .distribution
            .iter()
            .enumerate()
            .map(|(i, &p)| i as f64 * p)
            .sum();

        let e_y_do1: f64 = result_1
            .distribution
            .iter()
            .enumerate()
            .map(|(i, &p)| i as f64 * p)
            .sum();

        Some(e_y_do1 - e_y_do0)
    }
}

// ============================================================================
// TRAIT IMPLEMENTATIONS FOR HASHSET SUBSET CHECK
// ============================================================================

/// Extension trait for proper subset checking on HashSet.
trait ProperSubset {
    fn is_proper_subset(&self, other: &Self) -> bool;
}

impl ProperSubset for HashSet<usize> {
    fn is_proper_subset(&self, other: &Self) -> bool {
        self.len() < other.len() && self.is_subset(other)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a binary variable with values ["0", "1"].
    fn binary_values() -> Vec<String> {
        vec!["0".to_string(), "1".to_string()]
    }

    // ────────────────────────────────────────────────────────────────────────
    // d-Separation tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_d_separation_chain() {
        // Chain: X -> M -> Y
        // X and Y are d-separated by {M}
        // X and Y are NOT d-separated by {}
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary_values(), true);
        let m = dag.add_node("M", binary_values(), true);
        let y = dag.add_node("Y", binary_values(), true);
        dag.add_edge(x, m);
        dag.add_edge(m, y);

        // Conditioning on M blocks the chain
        let z_with_m: HashSet<usize> = [m].into_iter().collect();
        let result = dag.d_separated(x, y, &z_with_m);
        assert!(
            result.separated,
            "Chain X->M->Y should be blocked by conditioning on M"
        );

        // Empty conditioning set does NOT block the chain
        let z_empty: HashSet<usize> = HashSet::new();
        let result = dag.d_separated(x, y, &z_empty);
        assert!(
            !result.separated,
            "Chain X->M->Y should NOT be blocked by empty set"
        );
    }

    #[test]
    fn test_d_separation_fork() {
        // Fork: X <- C -> Y  (C is a common cause)
        // X and Y are d-separated by {C}
        // X and Y are NOT d-separated by {}
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary_values(), true);
        let c = dag.add_node("C", binary_values(), true);
        let y = dag.add_node("Y", binary_values(), true);
        dag.add_edge(c, x);
        dag.add_edge(c, y);

        // Conditioning on C blocks the fork
        let z_with_c: HashSet<usize> = [c].into_iter().collect();
        let result = dag.d_separated(x, y, &z_with_c);
        assert!(
            result.separated,
            "Fork X<-C->Y should be blocked by conditioning on C"
        );

        // Empty set does NOT block the fork
        let z_empty: HashSet<usize> = HashSet::new();
        let result = dag.d_separated(x, y, &z_empty);
        assert!(
            !result.separated,
            "Fork X<-C->Y should NOT be blocked by empty set"
        );
    }

    #[test]
    fn test_d_separation_collider() {
        // Collider: X -> M <- Y
        // X and Y are d-separated by {} (collider blocks by default)
        // X and Y are NOT d-separated by {M} (conditioning on collider opens the path)
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary_values(), true);
        let m = dag.add_node("M", binary_values(), true);
        let y = dag.add_node("Y", binary_values(), true);
        dag.add_edge(x, m);
        dag.add_edge(y, m);

        // Empty conditioning: collider blocks by default
        let z_empty: HashSet<usize> = HashSet::new();
        let result = dag.d_separated(x, y, &z_empty);
        assert!(
            result.separated,
            "Collider X->M<-Y should be blocked by empty set"
        );

        // Conditioning on M: opens the collider path
        let z_with_m: HashSet<usize> = [m].into_iter().collect();
        let result = dag.d_separated(x, y, &z_with_m);
        assert!(
            !result.separated,
            "Collider X->M<-Y should NOT be blocked when conditioning on M"
        );
    }

    #[test]
    fn test_d_separation_collider_descendant() {
        // Collider with descendant: X -> M <- Y, M -> D
        // Conditioning on descendant D also opens the collider
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary_values(), true);
        let m = dag.add_node("M", binary_values(), true);
        let y = dag.add_node("Y", binary_values(), true);
        let d = dag.add_node("D", binary_values(), true);
        dag.add_edge(x, m);
        dag.add_edge(y, m);
        dag.add_edge(m, d);

        // Conditioning on descendant D opens the collider
        let z_with_d: HashSet<usize> = [d].into_iter().collect();
        let result = dag.d_separated(x, y, &z_with_d);
        assert!(
            !result.separated,
            "Conditioning on collider's descendant D should open the path"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Backdoor criterion tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_backdoor_confounded_graph() {
        // Classic confounding: X <- C -> Y, X -> Y
        // C is a confounder. Z = {C} satisfies the backdoor criterion.
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary_values(), true);
        let c = dag.add_node("C", binary_values(), true);
        let y = dag.add_node("Y", binary_values(), true);
        dag.add_edge(c, x);
        dag.add_edge(c, y);
        dag.add_edge(x, y);

        let scm = StructuralCausalModel::new(dag);
        let backdoor_sets = scm.find_backdoor_sets(x, y);

        // {C} should be a valid backdoor set
        assert!(
            !backdoor_sets.is_empty(),
            "Should find at least one backdoor set"
        );
        let has_c = backdoor_sets.iter().any(|set| set.contains(&c));
        assert!(has_c, "Backdoor sets should include the confounder C");
    }

    #[test]
    fn test_backdoor_no_confounding() {
        // No confounding: X -> Y (direct cause, no backdoor path)
        // The empty set {} satisfies the backdoor criterion.
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary_values(), true);
        let y = dag.add_node("Y", binary_values(), true);
        dag.add_edge(x, y);

        let scm = StructuralCausalModel::new(dag);
        let backdoor_sets = scm.find_backdoor_sets(x, y);

        // Empty set should be valid (no confounding to block)
        let has_empty = backdoor_sets.iter().any(|set| set.is_empty());
        assert!(
            has_empty,
            "Empty set should be a valid backdoor set when no confounding exists"
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Frontdoor criterion tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_frontdoor_mediator_graph() {
        // Classic frontdoor: X -> M -> Y, U -> X, U -> Y (U unobserved)
        // M satisfies the frontdoor criterion.
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary_values(), true);
        let m = dag.add_node("M", binary_values(), true);
        let y = dag.add_node("Y", binary_values(), true);
        let u = dag.add_node("U", binary_values(), false); // Unobserved confounder
        dag.add_edge(x, m);
        dag.add_edge(m, y);
        dag.add_edge(u, x);
        dag.add_edge(u, y);

        let scm = StructuralCausalModel::new(dag);
        let frontdoor_sets = scm.find_frontdoor_sets(x, y);

        // {M} should satisfy frontdoor criterion
        let has_m = frontdoor_sets.iter().any(|set| set == &vec![m]);
        assert!(
            has_m,
            "M should satisfy frontdoor criterion in X->M->Y with unobserved confounder U.\n\
             Found sets: {:?}",
            frontdoor_sets
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Intervention tests (truncated factorization)
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_intervention_removes_incoming_edges() {
        // X -> Y, where Y = f(X)
        // P(Y=1|X=0) = 0.2, P(Y=1|X=1) = 0.9
        //
        // do(X=1) should give P(Y=1) = 0.9, NOT the observational P(Y=1)
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary_values(), true);
        let y = dag.add_node("Y", binary_values(), true);
        dag.add_edge(x, y);

        let mut scm = StructuralCausalModel::new(dag);

        // P(X=0) = 0.5, P(X=1) = 0.5
        scm.set_conditional_table(x, vec![0.5, 0.5]);

        // P(Y|X): P(Y=0|X=0)=0.8, P(Y=1|X=0)=0.2, P(Y=0|X=1)=0.1, P(Y=1|X=1)=0.9
        scm.set_conditional_table(y, vec![0.8, 0.2, 0.1, 0.9]);

        // do(X=1)
        let result = scm.intervene(x, 1, y).expect("Should be identifiable");

        // P(Y=0|do(X=1)) should be 0.1, P(Y=1|do(X=1)) should be 0.9
        assert!(
            (result.distribution[0] - 0.1).abs() < 0.05,
            "P(Y=0|do(X=1)) should be ~0.1, got {}",
            result.distribution[0]
        );
        assert!(
            (result.distribution[1] - 0.9).abs() < 0.05,
            "P(Y=1|do(X=1)) should be ~0.9, got {}",
            result.distribution[1]
        );
    }

    #[test]
    fn test_intervention_with_confounder() {
        // Confounded graph: C -> X, C -> Y, X -> Y
        // Observational P(Y|X) != P(Y|do(X)) due to confounding by C
        //
        // With backdoor adjustment on C:
        // P(y|do(x)) = sum_c P(y|x,c) P(c)
        let mut dag = CausalDAG::new();
        let c = dag.add_node("C", binary_values(), true);
        let x = dag.add_node("X", binary_values(), true);
        let y = dag.add_node("Y", binary_values(), true);
        dag.add_edge(c, x);
        dag.add_edge(c, y);
        dag.add_edge(x, y);

        let mut scm = StructuralCausalModel::new(dag);

        // P(C=0)=0.5, P(C=1)=0.5
        scm.set_conditional_table(c, vec![0.5, 0.5]);

        // P(X|C): P(X=0|C=0)=0.9, P(X=1|C=0)=0.1, P(X=0|C=1)=0.1, P(X=1|C=1)=0.9
        scm.set_conditional_table(x, vec![0.9, 0.1, 0.1, 0.9]);

        // P(Y|X,C): parent order is [C, X] (by parent index order)
        // Configs: (C=0,X=0), (C=0,X=1), (C=1,X=0), (C=1,X=1)
        // P(Y=0|C=0,X=0)=0.9, P(Y=1|C=0,X=0)=0.1
        // P(Y=0|C=0,X=1)=0.7, P(Y=1|C=0,X=1)=0.3
        // P(Y=0|C=1,X=0)=0.3, P(Y=1|C=1,X=0)=0.7
        // P(Y=0|C=1,X=1)=0.1, P(Y=1|C=1,X=1)=0.9
        scm.set_conditional_table(y, vec![0.9, 0.1, 0.7, 0.3, 0.3, 0.7, 0.1, 0.9]);

        let result = scm.intervene(x, 1, y).expect("Should be identifiable");

        // P(Y|do(X=1)) = sum_c P(Y|X=1,C=c) P(C=c)
        //              = P(Y|X=1,C=0)*0.5 + P(Y|X=1,C=1)*0.5
        // P(Y=1|do(X=1)) = 0.3*0.5 + 0.9*0.5 = 0.6
        assert!(
            (result.distribution[1] - 0.6).abs() < 0.1,
            "P(Y=1|do(X=1)) should be ~0.6, got {}",
            result.distribution[1]
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // Causal effect tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_causal_effect_simple() {
        // Simple X -> Y with known CPT
        // P(Y=1|X=0) = 0.2, P(Y=1|X=1) = 0.8
        // ACE = E[Y|do(X=1)] - E[Y|do(X=0)] = 0.8 - 0.2 = 0.6
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary_values(), true);
        let y = dag.add_node("Y", binary_values(), true);
        dag.add_edge(x, y);

        let mut scm = StructuralCausalModel::new(dag);
        scm.set_conditional_table(x, vec![0.5, 0.5]);
        scm.set_conditional_table(y, vec![0.8, 0.2, 0.2, 0.8]);

        let ace = scm.causal_effect(x, y).expect("Should be identifiable");
        assert!((ace - 0.6).abs() < 0.05, "ACE should be ~0.6, got {}", ace);
    }

    #[test]
    fn test_causal_effect_no_effect() {
        // X -> M -> Y, but X has no direct or indirect effect when CPTs
        // make M independent of X.
        // P(M=0|X=0) = 0.5, P(M=0|X=1) = 0.5 (X does not affect M)
        // ACE should be ~0.
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary_values(), true);
        let m = dag.add_node("M", binary_values(), true);
        let y = dag.add_node("Y", binary_values(), true);
        dag.add_edge(x, m);
        dag.add_edge(m, y);

        let mut scm = StructuralCausalModel::new(dag);
        scm.set_conditional_table(x, vec![0.5, 0.5]);
        // M is independent of X
        scm.set_conditional_table(m, vec![0.5, 0.5, 0.5, 0.5]);
        scm.set_conditional_table(y, vec![0.9, 0.1, 0.1, 0.9]);

        let ace = scm.causal_effect(x, y).expect("Should be identifiable");
        assert!(
            ace.abs() < 0.1,
            "ACE should be ~0 when X does not affect M, got {}",
            ace
        );
    }

    // ────────────────────────────────────────────────────────────────────────
    // DAG helper tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_dag_parents_children() {
        let mut dag = CausalDAG::new();
        let a = dag.add_node("A", binary_values(), true);
        let b = dag.add_node("B", binary_values(), true);
        let c = dag.add_node("C", binary_values(), true);
        dag.add_edge(a, b);
        dag.add_edge(a, c);
        dag.add_edge(b, c);

        assert_eq!(dag.parents(a), Vec::<usize>::new());
        assert_eq!(dag.parents(b), vec![a]);
        let mut parents_c = dag.parents(c);
        parents_c.sort();
        assert_eq!(parents_c, vec![a, b]);

        let mut children_a = dag.children(a);
        children_a.sort();
        assert_eq!(children_a, vec![b, c]);
    }

    #[test]
    fn test_dag_ancestors_descendants() {
        // A -> B -> C -> D
        let mut dag = CausalDAG::new();
        let a = dag.add_node("A", binary_values(), true);
        let b = dag.add_node("B", binary_values(), true);
        let c = dag.add_node("C", binary_values(), true);
        let d = dag.add_node("D", binary_values(), true);
        dag.add_edge(a, b);
        dag.add_edge(b, c);
        dag.add_edge(c, d);

        let anc_d = dag.ancestors(d);
        assert!(anc_d.contains(&a));
        assert!(anc_d.contains(&b));
        assert!(anc_d.contains(&c));
        assert!(!anc_d.contains(&d));

        let desc_a = dag.descendants(a);
        assert!(desc_a.contains(&b));
        assert!(desc_a.contains(&c));
        assert!(desc_a.contains(&d));
        assert!(!desc_a.contains(&a));
    }

    // ────────────────────────────────────────────────────────────────────────
    // Additional DAG construction & property tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_dag_new_is_empty() {
        let dag = CausalDAG::new();
        assert_eq!(dag.num_nodes(), 0);
        assert!(dag.edges.is_empty());
        assert!(dag.adjacency.is_empty());
    }

    #[test]
    fn test_dag_default_is_empty() {
        let dag = CausalDAG::default();
        assert_eq!(dag.num_nodes(), 0);
        assert!(dag.edges.is_empty());
    }

    #[test]
    fn test_dag_add_node_returns_sequential_ids() {
        let mut dag = CausalDAG::new();
        let a = dag.add_node("A", binary_values(), true);
        let b = dag.add_node("B", binary_values(), false);
        let c = dag.add_node("C", vec!["lo".into(), "mid".into(), "hi".into()], true);
        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(c, 2);
        assert_eq!(dag.num_nodes(), 3);
    }

    #[test]
    fn test_dag_node_properties_stored() {
        let mut dag = CausalDAG::new();
        let vals = vec!["off".into(), "on".into(), "high".into()];
        let id = dag.add_node("sensor", vals.clone(), false);
        assert_eq!(dag.nodes[id].name, "sensor");
        assert_eq!(dag.nodes[id].values, vals);
        assert!(!dag.nodes[id].is_observed);
    }

    #[test]
    fn test_dag_duplicate_edge_not_added() {
        let mut dag = CausalDAG::new();
        let a = dag.add_node("A", binary_values(), true);
        let b = dag.add_node("B", binary_values(), true);
        dag.add_edge(a, b);
        dag.add_edge(a, b); // duplicate
        assert_eq!(dag.edges.len(), 1, "Duplicate edges should not be added");
    }

    #[test]
    fn test_dag_parents_of_root_empty() {
        let mut dag = CausalDAG::new();
        let root = dag.add_node("root", binary_values(), true);
        let child = dag.add_node("child", binary_values(), true);
        dag.add_edge(root, child);
        assert!(dag.parents(root).is_empty());
    }

    #[test]
    fn test_dag_children_of_leaf_empty() {
        let mut dag = CausalDAG::new();
        let parent = dag.add_node("parent", binary_values(), true);
        let leaf = dag.add_node("leaf", binary_values(), true);
        dag.add_edge(parent, leaf);
        assert!(dag.children(leaf).is_empty());
    }

    #[test]
    fn test_dag_ancestors_of_root_empty() {
        let mut dag = CausalDAG::new();
        let root = dag.add_node("root", binary_values(), true);
        let _child = dag.add_node("child", binary_values(), true);
        dag.add_edge(root, _child);
        assert!(dag.ancestors(root).is_empty());
    }

    #[test]
    fn test_dag_descendants_of_leaf_empty() {
        let mut dag = CausalDAG::new();
        let parent = dag.add_node("parent", binary_values(), true);
        let leaf = dag.add_node("leaf", binary_values(), true);
        dag.add_edge(parent, leaf);
        assert!(dag.descendants(leaf).is_empty());
    }

    // ────────────────────────────────────────────────────────────────────────
    // d-Separation additional tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_d_separation_disconnected_nodes() {
        // Two isolated nodes — always d-separated regardless of conditioning
        let mut dag = CausalDAG::new();
        let a = dag.add_node("A", binary_values(), true);
        let b = dag.add_node("B", binary_values(), true);
        let result = dag.d_separated(a, b, &HashSet::new());
        assert!(result.separated, "Disconnected nodes should be d-separated");
    }

    // ────────────────────────────────────────────────────────────────────────
    // SCM construction & observation tests
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_scm_new_has_empty_tables() {
        let dag = CausalDAG::new();
        let scm = StructuralCausalModel::new(dag);
        assert!(scm.conditional_tables.is_empty());
        assert!(scm.observations.is_empty());
    }

    #[test]
    fn test_scm_observe_and_clear() {
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary_values(), true);
        let mut scm = StructuralCausalModel::new(dag);
        scm.observe(x, 1);
        assert_eq!(scm.observations.get(&x), Some(&1));
        scm.clear_observations();
        assert!(scm.observations.is_empty());
    }

    #[test]
    fn test_scm_default_conditional_prob_uniform() {
        // Without an explicit table, get_conditional_prob should return uniform 1/n
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", vec!["a".into(), "b".into(), "c".into()], true);
        let scm = StructuralCausalModel::new(dag);
        let p = scm.get_conditional_prob(x, 0, 0);
        assert!(
            (p - 1.0 / 3.0).abs() < 1e-9,
            "Default prob should be uniform 1/3, got {}",
            p
        );
    }

    #[test]
    fn test_scm_set_conditional_table() {
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary_values(), true);
        let mut scm = StructuralCausalModel::new(dag);
        scm.set_conditional_table(x, vec![0.3, 0.7]);
        let p0 = scm.get_conditional_prob(x, 0, 0);
        let p1 = scm.get_conditional_prob(x, 1, 0);
        assert!((p0 - 0.3).abs() < 1e-9);
        assert!((p1 - 0.7).abs() < 1e-9);
    }

    // ────────────────────────────────────────────────────────────────────────
    // Intervention result invariants
    // ────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_intervention_result_distribution_sums_to_one() {
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary_values(), true);
        let y = dag.add_node("Y", binary_values(), true);
        dag.add_edge(x, y);
        let mut scm = StructuralCausalModel::new(dag);
        scm.set_conditional_table(x, vec![0.5, 0.5]);
        scm.set_conditional_table(y, vec![0.8, 0.2, 0.3, 0.7]);
        let result = scm.intervene(x, 0, y).expect("identifiable");
        let sum: f64 = result.distribution.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "Distribution should sum to 1.0, got {}",
            sum
        );
    }

    #[test]
    fn test_causal_effect_returns_none_for_single_value_node() {
        let mut dag = CausalDAG::new();
        // X has only one value — causal_effect should return None
        let x = dag.add_node("X", vec!["only".into()], true);
        let y = dag.add_node("Y", binary_values(), true);
        dag.add_edge(x, y);
        let scm = StructuralCausalModel::new(dag);
        assert!(scm.causal_effect(x, y).is_none());
    }

    #[test]
    fn test_backdoor_excludes_descendants_of_x() {
        // X -> M -> Y, C -> X, C -> Y
        // M is a descendant of X, so M should NOT appear in backdoor sets
        let mut dag = CausalDAG::new();
        let c = dag.add_node("C", binary_values(), true);
        let x = dag.add_node("X", binary_values(), true);
        let m = dag.add_node("M", binary_values(), true);
        let y = dag.add_node("Y", binary_values(), true);
        dag.add_edge(c, x);
        dag.add_edge(c, y);
        dag.add_edge(x, m);
        dag.add_edge(m, y);
        let scm = StructuralCausalModel::new(dag);
        let sets = scm.find_backdoor_sets(x, y);
        for set in &sets {
            assert!(
                !set.contains(&m),
                "Backdoor sets must not include descendants of X; found M in {:?}",
                set
            );
        }
    }

    #[test]
    fn test_enumerate_configs_empty() {
        let dag = CausalDAG::new();
        let scm = StructuralCausalModel::new(dag);
        let configs = scm.enumerate_configs(&[]);
        let expected: Vec<Vec<usize>> = vec![vec![]];
        assert_eq!(configs, expected);
    }
}
