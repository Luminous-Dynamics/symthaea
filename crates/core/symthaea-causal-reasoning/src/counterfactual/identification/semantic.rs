// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict structural semantics layered over the legacy causal query APIs.
//!
//! These methods are additive while RQ-005 qualifies the stronger contract. They avoid silently
//! changing legacy call sites before the new semantics have executable evidence.

use std::collections::HashSet;

use super::dag::{
    CausalDAG, CausalEstimand, CausalQuery, CausalQueryOutcome, IdentificationMethod,
    UnidentifiedReason,
};
use super::discovery::{IVEstimator, IVValidity};
use super::reasoner::CounterfactualReasoner;

impl CounterfactualReasoner {
    /// Query a causal effect under strict supplied-DAG semantics.
    ///
    /// A valid `CausalDAG` is a fully specified directed causal graph. If intervention target X
    /// has no directed path to outcome Y, intervening on X cannot change Y in that graph. The
    /// effect is therefore structurally identified as zero; it is not epistemically unknown.
    /// Invalid cyclic inputs fail closed instead of being interpreted as DAG evidence.
    ///
    /// For descendant outcomes, this delegates to the existing identification engine so
    /// backdoor/frontdoor/do-calculus behavior remains unchanged.
    pub fn query_semantic(&self, dag: &CausalDAG, query: &CausalQuery) -> CausalQueryOutcome {
        if dag.num_nodes() > 20 {
            // Preserve the legacy resource/qualification boundary before doing more graph work.
            return self.query(dag, query);
        }

        if has_directed_cycle(dag) {
            return CausalQueryOutcome::Unidentified {
                reason: UnidentifiedReason::CyclicGraph,
                missing: vec!["Valid directed acyclic causal graph".into()],
                suggestions: vec!["Resolve or explicitly model feedback before DAG identification".into()],
            };
        }

        if !dag.has_path(query.treatment, query.outcome) {
            return CausalQueryOutcome::Identified {
                estimand: CausalEstimand {
                    effect: 0.0,
                    adjustment_set: vec![],
                    description: format!(
                        "Structural zero: {} is not a descendant of {}; P({}|do({})) = P({})",
                        dag.nodes[query.outcome],
                        dag.nodes[query.treatment],
                        dag.nodes[query.outcome],
                        dag.nodes[query.treatment],
                        dag.nodes[query.outcome],
                    ),
                },
                method: IdentificationMethod::DSeparation,
                // Logical consequence of the supplied DAG. Uncertainty about whether the DAG is
                // correct belongs in a separate graph/model-uncertainty layer.
                confidence: 1.0,
            };
        }

        self.query(dag, query)
    }
}

impl IVEstimator {
    /// Strict structural IV validation.
    ///
    /// This preserves all legacy rejection rules and additionally enforces the exclusion
    /// restriction correctly: after blocking traversal through treatment X, there must be no
    /// directed path from instrument Z to outcome Y.
    pub fn is_valid_instrument_strict(
        dag: &CausalDAG,
        instrument: usize,
        treatment: usize,
        outcome: usize,
    ) -> IVValidity {
        let legacy = Self::is_valid_instrument(dag, instrument, treatment, outcome);
        if matches!(legacy, IVValidity::Invalid { .. }) {
            return legacy;
        }

        if directed_path_avoiding(dag, instrument, outcome, treatment) {
            return IVValidity::Invalid {
                reason: format!(
                    "Instrument has a directed path to outcome that avoids treatment: {} -> ... -> {} without {}",
                    dag.nodes[instrument], dag.nodes[outcome], dag.nodes[treatment]
                ),
            };
        }

        legacy
    }
}

/// Directed reachability while treating one node as blocked.
fn directed_path_avoiding(dag: &CausalDAG, from: usize, to: usize, blocked: usize) -> bool {
    if from == blocked || to == blocked {
        return false;
    }

    let mut visited = HashSet::new();
    let mut stack = vec![from];
    while let Some(node) = stack.pop() {
        if node == to {
            return true;
        }
        if node == blocked || !visited.insert(node) {
            continue;
        }
        for child in dag.children(node) {
            if child != blocked {
                stack.push(child);
            }
        }
    }
    false
}

fn has_directed_cycle(dag: &CausalDAG) -> bool {
    // 0 = unseen, 1 = active DFS stack, 2 = complete.
    let mut state = vec![0_u8; dag.num_nodes()];
    for node in 0..dag.num_nodes() {
        if state[node] == 0 && cycle_from(dag, node, &mut state) {
            return true;
        }
    }
    false
}

fn cycle_from(dag: &CausalDAG, node: usize, state: &mut [u8]) -> bool {
    state[node] = 1;
    for child in dag.children(node) {
        match state[child] {
            1 => return true,
            0 if cycle_from(dag, child, state) => return true,
            _ => {}
        }
    }
    state[node] = 2;
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_descendant_path_is_identified_structural_zero() {
        let dag = CausalDAG::new(vec!["X".into(), "Y".into()], vec![]);
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let outcome = CounterfactualReasoner::new().query_semantic(&dag, &query);
        match outcome {
            CausalQueryOutcome::Identified {
                estimand,
                method,
                confidence,
            } => {
                assert_eq!(method, IdentificationMethod::DSeparation);
                assert_eq!(estimand.effect, 0.0);
                assert!(estimand.adjustment_set.is_empty());
                assert_eq!(confidence, 1.0);
            }
            other => panic!("expected structural zero, got {other:?}"),
        }
    }

    #[test]
    fn reverse_direction_query_is_identified_structural_zero() {
        let dag = CausalDAG::new(vec!["X".into(), "Y".into()], vec![(0, 1)]);
        let query = CausalQuery {
            treatment: 1,
            outcome: 0,
            conditioning: vec![],
        };
        assert!(matches!(
            CounterfactualReasoner::new().query_semantic(&dag, &query),
            CausalQueryOutcome::Identified {
                method: IdentificationMethod::DSeparation,
                ..
            }
        ));
    }

    #[test]
    fn cyclic_input_fails_closed_before_zero_inference() {
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "Y".into()],
            vec![(0, 1), (1, 0)],
        );
        let query = CausalQuery {
            treatment: 0,
            outcome: 2,
            conditioning: vec![],
        };
        assert!(matches!(
            CounterfactualReasoner::new().query_semantic(&dag, &query),
            CausalQueryOutcome::Unidentified {
                reason: UnidentifiedReason::CyclicGraph,
                ..
            }
        ));
    }

    #[test]
    fn descendant_query_preserves_backdoor_identification() {
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into(), "U".into()],
            vec![(2, 0), (2, 1), (0, 1)],
        );
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let outcome = CounterfactualReasoner::new().query_semantic(&dag, &query);
        match outcome {
            CausalQueryOutcome::Identified {
                method, estimand, ..
            } => {
                assert_eq!(method, IdentificationMethod::BackdoorAdjustment);
                assert!(estimand.adjustment_set.contains(&2));
            }
            other => panic!("expected backdoor identification, got {other:?}"),
        }
    }

    #[test]
    fn oversized_dag_preserves_honest_abstention() {
        let nodes: Vec<String> = (0..21).map(|i| format!("N{i}")).collect();
        let dag = CausalDAG::new(nodes, vec![(0, 1)]);
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        assert!(matches!(
            CounterfactualReasoner::new().query_semantic(&dag, &query),
            CausalQueryOutcome::Unidentified {
                reason: UnidentifiedReason::DagTooLarge { .. },
                ..
            }
        ));
    }

    #[test]
    fn valid_instrument_remains_valid() {
        let dag = CausalDAG::new(
            vec!["Z".into(), "X".into(), "Y".into(), "U".into()],
            vec![(0, 1), (1, 2), (3, 1), (3, 2)],
        );
        assert!(matches!(
            IVEstimator::is_valid_instrument_strict(&dag, 0, 1, 2),
            IVValidity::Valid { .. }
        ));
    }

    #[test]
    fn direct_exclusion_violation_remains_invalid() {
        let dag = CausalDAG::new(
            vec!["Z".into(), "X".into(), "Y".into()],
            vec![(0, 1), (1, 2), (0, 2)],
        );
        assert!(matches!(
            IVEstimator::is_valid_instrument_strict(&dag, 0, 1, 2),
            IVValidity::Invalid { .. }
        ));
    }

    #[test]
    fn alternate_path_exclusion_violation_is_rejected() {
        let dag = CausalDAG::new(
            vec!["Z".into(), "X".into(), "Y".into(), "W".into()],
            vec![(0, 1), (1, 2), (0, 3), (3, 2)],
        );
        let legacy = IVEstimator::is_valid_instrument(&dag, 0, 1, 2);
        assert!(
            matches!(legacy, IVValidity::Valid { .. }),
            "fixture must reproduce the legacy false acceptance"
        );
        assert!(matches!(
            IVEstimator::is_valid_instrument_strict(&dag, 0, 1, 2),
            IVValidity::Invalid { .. }
        ));
    }
}
