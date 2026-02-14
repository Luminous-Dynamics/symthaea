//! Counterfactual reasoner implementing backdoor and frontdoor criteria,
//! plus Pearl's do-calculus Rules 2 and 3.

use std::collections::HashSet;

use super::dag::{
    CausalDAG, CausalEstimand, CausalQuery, CausalQueryOutcome,
    IdentificationMethod, UnidentifiedReason,
};

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
    pub(crate) fn try_rule2(
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
    pub(crate) fn try_rule3(
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
pub(crate) fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
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
