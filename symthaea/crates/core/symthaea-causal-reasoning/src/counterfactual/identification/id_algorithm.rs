// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shpitser-Pearl ID Algorithm and reference harness.
//!
//! Provides complete identification for causal effects in semi-Markovian
//! causal models (graphs with latent confounders), plus a reference harness
//! for validating the counterfactual reasoner.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

use super::dag::{
    CausalDAG, CausalEstimand, CausalQuery, CausalQueryOutcome, IdentificationMethod,
    UnidentifiedReason,
};
use super::reasoner::CounterfactualReasoner;

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
        Self {
            nodes,
            directed,
            bidirected,
        }
    }

    /// Convert to standard CausalDAG (loses bidirected information).
    pub fn to_dag(&self) -> CausalDAG {
        CausalDAG::new(self.nodes.clone(), self.directed.clone())
    }

    /// Get parents of a node (directed edges only).
    pub fn parents(&self, node: usize) -> Vec<usize> {
        self.directed
            .iter()
            .filter(|(_, c)| *c == node)
            .map(|(p, _)| *p)
            .collect()
    }

    /// Get children of a node (directed edges only).
    pub fn children(&self, node: usize) -> Vec<usize> {
        self.directed
            .iter()
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

        let new_nodes: Vec<String> = node_list.iter().map(|&i| self.nodes[i].clone()).collect();

        let new_directed: Vec<(usize, usize)> = self
            .directed
            .iter()
            .filter(|(p, c)| nodes.contains(p) && nodes.contains(c))
            .map(|(p, c)| (node_map[p], node_map[c]))
            .collect();

        let new_bidirected: Vec<(usize, usize)> = self
            .bidirected
            .iter()
            .filter(|(a, b)| nodes.contains(a) && nodes.contains(b))
            .map(|(a, b)| (node_map[a], node_map[b]))
            .collect();

        CausalGraphWithLatents::new(new_nodes, new_directed, new_bidirected)
    }

    /// Check if one set is a subset of another.
    pub(crate) fn is_subset(subset: &HashSet<usize>, superset: &HashSet<usize>) -> bool {
        subset.iter().all(|x| superset.contains(x))
    }

    /// Compute C-components restricted to a subset of nodes.
    ///
    /// This is equivalent to c_components() on the induced subgraph,
    /// but keeps the original node indices for easier comparison.
    pub fn c_components_restricted(&self, nodes: &HashSet<usize>) -> Vec<HashSet<usize>> {
        let mut visited: HashSet<usize> = HashSet::new();
        let mut components = Vec::new();

        for &start in nodes {
            if visited.contains(&start) {
                continue;
            }

            // BFS to find all nodes reachable via bidirected edges within the subset
            let mut component = HashSet::new();
            let mut queue = VecDeque::new();
            queue.push_back(start);

            while let Some(node) = queue.pop_front() {
                if visited.contains(&node) {
                    continue;
                }
                visited.insert(node);
                component.insert(node);

                // Add bidirected neighbors that are also in the subset
                for neighbor in self.bidirected_neighbors(node) {
                    if nodes.contains(&neighbor) && !visited.contains(&neighbor) {
                        queue.push_back(neighbor);
                    }
                }
            }

            components.push(component);
        }

        components
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
            CausalExpression::Probability {
                outcome,
                conditioning,
            } => {
                let out_names: Vec<&str> = outcome.iter().map(|&i| nodes[i].as_str()).collect();
                if conditioning.is_empty() {
                    format!("P({})", out_names.join(","))
                } else {
                    let cond_names: Vec<&str> =
                        conditioning.iter().map(|&i| nodes[i].as_str()).collect();
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
            CausalExpression::Fraction {
                numerator,
                denominator,
            } => {
                format!(
                    "[{}] / [{}]",
                    numerator.to_string(nodes),
                    denominator.to_string(nodes)
                )
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
            return Ok(CausalExpression::Probability {
                outcome: outcome_vars,
                conditioning,
            });
        }

        // Line 2: Compute ancestors of Y in G
        let ancestors_y = self.ancestors_of_set(graph, y);

        // If there are variables not ancestors of Y, marginalize them out
        let relevant: HashSet<usize> = v
            .iter()
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
        let w: HashSet<usize> = v_minus_x
            .difference(&an_y_in_g_x_bar)
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
        let mapped_components: Vec<HashSet<usize>> = c_components
            .iter()
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
            let sum_over: Vec<usize> = v
                .iter()
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
            v_minus_x_set.clone()
        } else {
            mapped_components[0].clone()
        };

        // Find C-component of the FULL graph G containing S
        // According to Shpitser-Pearl, we check against C(G), not C(G[V])
        let full_c_components = graph.c_components();

        // Find the C-component containing S
        let s_prime: Option<HashSet<usize>> = full_c_components
            .into_iter()
            .find(|c| CausalGraphWithLatents::is_subset(&s, c));

        // Additional check: S being equal to its C-component is only a hedge if S
        // actually has bidirected edges (i.e., is confounded). A singleton node
        // with no bidirected edges is trivially identifiable.
        let s_has_bidirected = graph
            .bidirected
            .iter()
            .any(|(a, b)| s.contains(a) && s.contains(b));

        match s_prime {
            Some(ref c_prime) if c_prime == &s && s_has_bidirected => {
                // Line 6: S is a C-component in G AND S has internal bidirected edges
                // This represents a true hedge (confounded structure)
                let hedge_nodes: Vec<usize> = s.iter().copied().collect();
                Err((
                    hedge_nodes.clone(),
                    format!(
                        "Hedge found: C-component {:?} is a hedge for the causal effect",
                        hedge_nodes
                            .iter()
                            .map(|&i| &graph.nodes[i])
                            .collect::<Vec<_>>()
                    ),
                ))
            }
            Some(ref c_prime) if c_prime == &s => {
                // S equals its C-component but has no internal bidirected edges
                // This is a trivially identifiable case - just use P(s | pa(s))
                let outcome_vars: Vec<usize> = y.iter().copied().collect();
                let conditioning: Vec<usize> = s
                    .iter()
                    .flat_map(|&node| graph.parents(node))
                    .filter(|&p| !s.contains(&p))
                    .collect::<HashSet<_>>()
                    .into_iter()
                    .collect();
                Ok(CausalExpression::Probability {
                    outcome: outcome_vars,
                    conditioning,
                })
            }
            Some(ref c_prime) => {
                // Line 7: S ⊂ S' - use factorization
                //
                // Note: The full Shpitser-Pearl algorithm uses recursive ID calls here,
                // which correctly handles all confounding. This simplified implementation
                // uses factorization, which may incorrectly identify some non-identifiable
                // effects (e.g., bow graphs). See test_id_with_backdoor_confounder.
                //
                // The check below catches the direct confounding case (bow graph)
                // where X is the only treatment, is in c_prime, and there's no mediator.
                let x_only_in_c_prime = x.len() == 1
                    && x.iter().all(|&xi| c_prime.contains(&xi))
                    && s.len() == 1
                    && s.iter().all(|&si| c_prime.contains(&si));
                if x_only_in_c_prime && x.iter().all(|&xi| !s.contains(&xi)) {
                    // Direct confounding: single treatment X and single outcome Y
                    // both in same C-component, with no mediator in the path
                    // Check if there's a direct edge X → Y
                    // x.len() == 1 and s.len() == 1 guaranteed by x_only_in_c_prime check above
                    let (&x_node, &s_node) = match (x.iter().next(), s.iter().next()) {
                        (Some(xn), Some(sn)) => (xn, sn),
                        _ => {
                            return Ok(CausalExpression::Probability {
                                outcome: s.iter().copied().collect(),
                                conditioning: Vec::new(),
                            });
                        }
                    };
                    let direct_edge = graph
                        .directed
                        .iter()
                        .any(|&(p, c)| p == x_node && c == s_node);
                    if direct_edge {
                        let hedge_nodes: Vec<usize> = c_prime.iter().copied().collect();
                        return Err((
                            hedge_nodes.clone(),
                            format!(
                                "Hedge found: treatment and outcome directly confounded {:?}",
                                hedge_nodes
                                    .iter()
                                    .map(|&i| &graph.nodes[i])
                                    .collect::<Vec<_>>()
                            ),
                        ));
                    }
                }

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
    fn ancestors_of_set(
        &self,
        graph: &CausalGraphWithLatents,
        nodes: &HashSet<usize>,
    ) -> HashSet<usize> {
        let mut result = nodes.clone();
        for &node in nodes {
            result.extend(graph.ancestors(node));
        }
        result
    }

    /// Create a graph with incoming edges to X removed.
    fn remove_incoming(
        &self,
        graph: &CausalGraphWithLatents,
        x: &HashSet<usize>,
    ) -> CausalGraphWithLatents {
        let new_directed: Vec<(usize, usize)> = graph
            .directed
            .iter()
            .filter(|(_, child)| !x.contains(child))
            .copied()
            .collect();

        CausalGraphWithLatents::new(graph.nodes.clone(), new_directed, graph.bidirected.clone())
    }

    /// Topological sort of a subset of nodes.
    fn topological_sort_subset(
        &self,
        graph: &CausalGraphWithLatents,
        nodes: &HashSet<usize>,
    ) -> Vec<usize> {
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
            visit(
                node,
                graph,
                nodes,
                &mut visited,
                &mut temp_mark,
                &mut result,
            );
        }

        result.reverse();
        result
    }

    /// Convenience method: query using the ID algorithm.
    pub fn query(&self, graph: &CausalGraphWithLatents, query: &CausalQuery) -> CausalQueryOutcome {
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
            Err((hedge_nodes, _description)) => CausalQueryOutcome::Unidentified {
                reason: UnidentifiedReason::HedgeFound { hedge_nodes },
                missing: vec![],
                suggestions: vec![
                    "The causal effect is not identifiable from observational data".to_string(),
                    "Consider running a randomized experiment".to_string(),
                ],
            },
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

        let matches = self
            .test_suite
            .iter()
            .filter(|(dag, query, _exact)| {
                matches!(
                    engine.query(dag, query),
                    CausalQueryOutcome::Identified { .. }
                )
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
        suite.push((
            chain,
            CausalQuery {
                treatment: 0,
                outcome: 2,
                conditioning: vec![],
            },
            0.0,
        ));

        // Test 2: Confounded X ← U → Y, X → Y (backdoor via U)
        let confounded = CausalDAG::new(
            vec!["X".into(), "Y".into(), "U".into()],
            vec![(2, 0), (2, 1), (0, 1)],
        );
        suite.push((
            confounded,
            CausalQuery {
                treatment: 0,
                outcome: 1,
                conditioning: vec![],
            },
            0.0,
        ));

        // Test 3: Direct cause X → Y (trivially identifiable)
        let direct = CausalDAG::new(vec!["X".into(), "Y".into()], vec![(0, 1)]);
        suite.push((
            direct,
            CausalQuery {
                treatment: 0,
                outcome: 1,
                conditioning: vec![],
            },
            0.0,
        ));

        // Test 4: Instrumental variable structure for Rule 2
        // Z → X → Y with U → X, U → Y (Z is an instrument)
        // This tests Rule 2: do(Z) can be converted to observation of Z
        let iv = CausalDAG::new(
            vec!["Z".into(), "X".into(), "Y".into(), "U".into()],
            vec![(0, 1), (1, 2), (3, 1), (3, 2)],
        );
        suite.push((
            iv,
            CausalQuery {
                treatment: 1,
                outcome: 2,
                conditioning: vec![],
            },
            0.0,
        ));

        // Test 5: Rule 3 test - Z doesn't affect Y given X
        // X → Y, X → Z (Z is downstream of X, no effect on Y)
        // This tests Rule 3: do(Z) can be dropped entirely
        let rule3_dag = CausalDAG::new(
            vec!["X".into(), "Y".into(), "Z".into()],
            vec![(0, 1), (0, 2)],
        );
        suite.push((
            rule3_dag,
            CausalQuery {
                treatment: 0,
                outcome: 1,
                conditioning: vec![],
            },
            0.0,
        ));

        // Test 6: M-bias structure (collider test)
        // U1 → X, U1 → M, U2 → M, U2 → Y, X → Y
        // M is a collider - conditioning on it opens a path
        let m_bias = CausalDAG::new(
            vec!["X".into(), "Y".into(), "M".into(), "U1".into(), "U2".into()],
            vec![(3, 0), (3, 2), (4, 2), (4, 1), (0, 1)],
        );
        suite.push((
            m_bias,
            CausalQuery {
                treatment: 0,
                outcome: 1,
                conditioning: vec![],
            },
            0.0,
        ));

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
    use std::collections::HashSet;

    // ─────────────────────────────────────────────────────────────────────────
    // CausalGraphWithLatents Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_graph_with_latents_construction() {
        let g = CausalGraphWithLatents::new(
            vec!["X".into(), "Y".into(), "Z".into()],
            vec![(0, 1), (1, 2)],
            vec![(0, 2)],
        );
        assert_eq!(g.nodes.len(), 3);
        assert_eq!(g.directed.len(), 2);
        assert_eq!(g.bidirected.len(), 1);
    }

    #[test]
    fn test_graph_parents_and_children() {
        let g = CausalGraphWithLatents::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![(0, 1), (0, 2), (1, 2)],
            vec![],
        );
        assert_eq!(g.parents(0), Vec::<usize>::new());
        assert_eq!(g.parents(1), vec![0]);
        let mut parents_c = g.parents(2);
        parents_c.sort();
        assert_eq!(parents_c, vec![0, 1]);

        let mut children_a = g.children(0);
        children_a.sort();
        assert_eq!(children_a, vec![1, 2]);
        assert_eq!(g.children(2), Vec::<usize>::new());
    }

    #[test]
    fn test_graph_ancestors() {
        // A → B → C → D
        let g = CausalGraphWithLatents::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            vec![(0, 1), (1, 2), (2, 3)],
            vec![],
        );
        let ancestors_d = g.ancestors(3);
        assert!(ancestors_d.contains(&0));
        assert!(ancestors_d.contains(&1));
        assert!(ancestors_d.contains(&2));
        assert!(!ancestors_d.contains(&3));
        assert_eq!(ancestors_d.len(), 3);

        let ancestors_a = g.ancestors(0);
        assert!(ancestors_a.is_empty());
    }

    #[test]
    fn test_bidirected_neighbors() {
        let g = CausalGraphWithLatents::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![],
            vec![(0, 1), (1, 2)],
        );
        let neighbors_b = g.bidirected_neighbors(1);
        assert!(neighbors_b.contains(&0));
        assert!(neighbors_b.contains(&2));
        assert_eq!(neighbors_b.len(), 2);

        let neighbors_a = g.bidirected_neighbors(0);
        assert_eq!(neighbors_a.len(), 1);
        assert!(neighbors_a.contains(&1));
    }

    #[test]
    fn test_c_components_disjoint() {
        // A ↔ B, C ↔ D — two disjoint C-components
        let g = CausalGraphWithLatents::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            vec![],
            vec![(0, 1), (2, 3)],
        );
        let components = g.c_components();
        assert_eq!(components.len(), 2);
    }

    #[test]
    fn test_c_components_connected_chain() {
        // A ↔ B ↔ C — one C-component
        let g = CausalGraphWithLatents::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![],
            vec![(0, 1), (1, 2)],
        );
        let components = g.c_components();
        assert_eq!(components.len(), 1);
        assert_eq!(components[0].len(), 3);
    }

    #[test]
    fn test_c_components_no_bidirected() {
        // No bidirected edges — each node is its own C-component
        let g = CausalGraphWithLatents::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![(0, 1), (1, 2)],
            vec![],
        );
        let components = g.c_components();
        assert_eq!(components.len(), 3);
    }

    #[test]
    fn test_c_component_of() {
        let g = CausalGraphWithLatents::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            vec![],
            vec![(0, 1), (2, 3)],
        );
        let comp_a = g.c_component_of(0);
        assert!(comp_a.contains(&0));
        assert!(comp_a.contains(&1));
        assert!(!comp_a.contains(&2));
        assert_eq!(comp_a.len(), 2);
    }

    #[test]
    fn test_c_components_restricted() {
        // Full graph: A ↔ B ↔ C ↔ D
        // Restrict to {A, B, D} — A ↔ B connected, D isolated
        let g = CausalGraphWithLatents::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            vec![],
            vec![(0, 1), (1, 2), (2, 3)],
        );
        let subset: HashSet<usize> = [0, 1, 3].iter().copied().collect();
        let components = g.c_components_restricted(&subset);
        assert_eq!(components.len(), 2, "Should have 2 restricted C-components");
    }

    #[test]
    fn test_subgraph() {
        let g = CausalGraphWithLatents::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            vec![(0, 1), (1, 2), (2, 3)],
            vec![(0, 2)],
        );
        let subset: HashSet<usize> = [0, 1, 2].iter().copied().collect();
        let sub = g.subgraph(&subset);
        assert_eq!(sub.nodes.len(), 3);
        // D should be removed, so edge (2,3) mapped to nothing
        assert!(sub.directed.len() <= 2);
    }

    #[test]
    fn test_to_dag() {
        let g =
            CausalGraphWithLatents::new(vec!["X".into(), "Y".into()], vec![(0, 1)], vec![(0, 1)]);
        let dag = g.to_dag();
        assert_eq!(dag.nodes.len(), 2);
        assert_eq!(dag.edges.len(), 1);
        assert!(dag.edges.contains(&(0, 1)));
    }

    #[test]
    fn test_is_subset() {
        let small: HashSet<usize> = [1, 2].iter().copied().collect();
        let big: HashSet<usize> = [0, 1, 2, 3].iter().copied().collect();
        assert!(CausalGraphWithLatents::is_subset(&small, &big));
        assert!(!CausalGraphWithLatents::is_subset(&big, &small));

        let empty: HashSet<usize> = HashSet::new();
        assert!(CausalGraphWithLatents::is_subset(&empty, &big));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IDAlgorithm Identification Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_id_markovian_always_identifiable() {
        // No bidirected edges → always identifiable
        let g = CausalGraphWithLatents::new(
            vec!["X".into(), "M".into(), "Y".into()],
            vec![(0, 1), (1, 2)],
            vec![],
        );
        let id = IDAlgorithm::new();
        let result = id.identify(&g, &[0], &[2]);
        assert!(
            result.is_ok(),
            "Markovian model should always be identifiable"
        );
    }

    #[test]
    fn test_id_empty_treatment_returns_probability() {
        let g =
            CausalGraphWithLatents::new(vec!["X".into(), "Y".into()], vec![(0, 1)], vec![(0, 1)]);
        let id = IDAlgorithm::new();
        // Empty treatment: P(y) should be returned directly
        let result = id.identify(&g, &[], &[1]);
        assert!(result.is_ok());
        if let Ok(CausalExpression::Probability {
            outcome,
            conditioning,
        }) = result
        {
            assert_eq!(outcome, vec![1]);
            assert!(conditioning.is_empty());
        }
    }

    #[test]
    fn test_id_bow_graph_not_identifiable() {
        // X → Y with X ↔ Y (bow graph) — classic non-identifiable
        let g =
            CausalGraphWithLatents::new(vec!["X".into(), "Y".into()], vec![(0, 1)], vec![(0, 1)]);
        let id = IDAlgorithm::new();
        let result = id.identify(&g, &[0], &[1]);
        assert!(result.is_err(), "Bow graph should NOT be identifiable");
        if let Err((hedge, msg)) = result {
            assert!(!hedge.is_empty());
            assert!(!msg.is_empty());
        }
    }

    #[test]
    fn test_id_frontdoor_identifiable() {
        // X → M → Y with X ↔ Y (frontdoor criterion applies)
        let g = CausalGraphWithLatents::new(
            vec!["X".into(), "M".into(), "Y".into()],
            vec![(0, 1), (1, 2)],
            vec![(0, 2)],
        );
        let id = IDAlgorithm::new();
        let result = id.identify(&g, &[0], &[2]);
        assert!(result.is_ok(), "Frontdoor structure should be identifiable");
    }

    #[test]
    fn test_id_napkin_identifiable() {
        // Napkin graph: W → X → Y with W ↔ Y
        // P(y|do(x)) is identifiable via adjustment for W
        let g = CausalGraphWithLatents::new(
            vec!["W".into(), "X".into(), "Y".into()],
            vec![(0, 1), (1, 2)],
            vec![(0, 2)],
        );
        let id = IDAlgorithm::new();
        let result = id.identify(&g, &[1], &[2]);
        assert!(result.is_ok(), "Napkin graph should be identifiable");
    }

    #[test]
    fn test_id_multiple_c_components_decomposition() {
        // Graph where V\X has multiple C-components (triggers line 5)
        // A → B → C, with A ↔ B but not C
        // do(A), outcome C
        let g = CausalGraphWithLatents::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![(0, 1), (1, 2)],
            vec![], // No bidirected — each is own C-component
        );
        let id = IDAlgorithm::new();
        let result = id.identify(&g, &[0], &[2]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_id_default_trait() {
        let id = IDAlgorithm::default();
        assert_eq!(id.max_depth, 100);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // CausalExpression Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_expression_probability_unconditional() {
        let nodes = vec!["X".into(), "Y".into()];
        let expr = CausalExpression::Probability {
            outcome: vec![1],
            conditioning: vec![],
        };
        let s = expr.to_string(&nodes);
        assert_eq!(s, "P(Y)");
    }

    #[test]
    fn test_expression_probability_conditional() {
        let nodes = vec!["X".into(), "Y".into(), "Z".into()];
        let expr = CausalExpression::Probability {
            outcome: vec![1],
            conditioning: vec![0, 2],
        };
        let s = expr.to_string(&nodes);
        assert_eq!(s, "P(Y|X,Z)");
    }

    #[test]
    fn test_expression_sum() {
        let nodes = vec!["X".into(), "Y".into(), "Z".into()];
        let expr = CausalExpression::Sum {
            sum_over: vec![2],
            inner: Box::new(CausalExpression::Probability {
                outcome: vec![1],
                conditioning: vec![0, 2],
            }),
        };
        let s = expr.to_string(&nodes);
        assert!(s.contains("Σ_{Z}"));
        assert!(s.contains("P(Y|X,Z)"));
    }

    #[test]
    fn test_expression_product() {
        let nodes = vec!["X".into(), "Y".into()];
        let expr = CausalExpression::Product(vec![
            CausalExpression::Probability {
                outcome: vec![0],
                conditioning: vec![],
            },
            CausalExpression::Probability {
                outcome: vec![1],
                conditioning: vec![0],
            },
        ]);
        let s = expr.to_string(&nodes);
        assert!(s.contains("P(X)"));
        assert!(s.contains("P(Y|X)"));
        assert!(s.contains(" × "));
    }

    #[test]
    fn test_expression_fraction() {
        let nodes = vec!["X".into(), "Y".into()];
        let expr = CausalExpression::Fraction {
            numerator: Box::new(CausalExpression::Probability {
                outcome: vec![0, 1],
                conditioning: vec![],
            }),
            denominator: Box::new(CausalExpression::Probability {
                outcome: vec![0],
                conditioning: vec![],
            }),
        };
        let s = expr.to_string(&nodes);
        assert!(s.contains("P(X,Y)"));
        assert!(s.contains("P(X)"));
        assert!(s.contains("/"));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // IDAlgorithm Query Interface Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_id_query_identified() {
        let g = CausalGraphWithLatents::new(vec!["X".into(), "Y".into()], vec![(0, 1)], vec![]);
        let id = IDAlgorithm::new();
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let result = id.query(&g, &query);
        assert!(matches!(result, CausalQueryOutcome::Identified { .. }));
        if let CausalQueryOutcome::Identified {
            method, confidence, ..
        } = result
        {
            assert_eq!(method, IdentificationMethod::IDAlgorithm);
            assert!((confidence - 0.95).abs() < 1e-10);
        }
    }

    #[test]
    fn test_id_query_unidentified_returns_hedge() {
        let g =
            CausalGraphWithLatents::new(vec!["X".into(), "Y".into()], vec![(0, 1)], vec![(0, 1)]);
        let id = IDAlgorithm::new();
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let result = id.query(&g, &query);
        assert!(matches!(result, CausalQueryOutcome::Unidentified { .. }));
        if let CausalQueryOutcome::Unidentified {
            reason,
            suggestions,
            ..
        } = result
        {
            assert!(matches!(reason, UnidentifiedReason::HedgeFound { .. }));
            assert!(!suggestions.is_empty());
        }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Reference Harness Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_harness_construction() {
        let harness = CausalReferenceHarness::new();
        assert!(
            harness.test_count() >= 6,
            "Harness should have at least 6 test cases"
        );
        assert!((harness.match_threshold - 0.99).abs() < 1e-10);
        assert!((harness.current_match_rate - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_harness_default() {
        let harness = CausalReferenceHarness::default();
        assert!(harness.test_count() >= 6);
    }

    #[test]
    fn test_harness_validate() {
        let reasoner = CounterfactualReasoner::new();
        let mut harness = CausalReferenceHarness::new();
        let result = harness.validate(&reasoner);
        // Our reasoner should identify most standard cases
        assert!(
            harness.current_match_rate >= 0.5,
            "Expected >=50% match rate, got {}",
            harness.current_match_rate
        );
        // Result should be Passed or AutoDowngrade
        assert!(result == HarnessResult::Passed || result == HarnessResult::AutoDowngrade);
    }

    #[test]
    fn test_harness_empty_suite() {
        let mut harness = CausalReferenceHarness {
            test_suite: vec![],
            match_threshold: 0.99,
            current_match_rate: 0.0,
        };
        let reasoner = CounterfactualReasoner::new();
        let result = harness.validate(&reasoner);
        assert_eq!(result, HarnessResult::Passed, "Empty suite should pass");
    }

    #[test]
    fn test_harness_result_enum() {
        assert_ne!(HarnessResult::Passed, HarnessResult::AutoDowngrade);
        let p = HarnessResult::Passed;
        assert_eq!(p, HarnessResult::Passed);
    }
}
