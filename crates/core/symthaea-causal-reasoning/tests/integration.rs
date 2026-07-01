// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Integration tests for symthaea-causal-reasoning public API.
//!
//! These tests exercise the crate as an external consumer, covering
//! causal_calculus (SCM, d-separation, backdoor/frontdoor, intervention),
//! causal_emergence (EI, determinism, degeneracy, analyzer), and
//! counterfactual identification (lightweight DAG, graph surgery, queries).

use std::collections::HashSet;

// ============================================================================
// CAUSAL CALCULUS — StructuralCausalModel & CausalDAG
// ============================================================================

mod causal_calculus_tests {
    use super::*;
    use symthaea_causal_reasoning::causal_calculus::{CausalDAG, StructuralCausalModel};

    fn binary() -> Vec<String> {
        vec!["0".into(), "1".into()]
    }

    #[test]
    fn empty_dag_has_no_nodes() {
        let dag = CausalDAG::new();
        assert_eq!(dag.num_nodes(), 0);
    }

    #[test]
    fn single_node_dag_parents_children() {
        let mut dag = CausalDAG::new();
        let a = dag.add_node("A", binary(), true);
        assert_eq!(dag.num_nodes(), 1);
        assert!(dag.parents(a).is_empty());
        assert!(dag.children(a).is_empty());
    }

    #[test]
    fn dag_edge_creates_parent_child() {
        let mut dag = CausalDAG::new();
        let a = dag.add_node("A", binary(), true);
        let b = dag.add_node("B", binary(), true);
        dag.add_edge(a, b);

        assert_eq!(dag.parents(b), vec![a]);
        assert_eq!(dag.children(a), vec![b]);
        assert!(dag.parents(a).is_empty());
        assert!(dag.children(b).is_empty());
    }

    #[test]
    fn ancestors_descendants_diamond() {
        // A → B, A → C, B → D, C → D
        let mut dag = CausalDAG::new();
        let a = dag.add_node("A", binary(), true);
        let b = dag.add_node("B", binary(), true);
        let c = dag.add_node("C", binary(), true);
        let d = dag.add_node("D", binary(), true);
        dag.add_edge(a, b);
        dag.add_edge(a, c);
        dag.add_edge(b, d);
        dag.add_edge(c, d);

        assert_eq!(
            dag.ancestors(d),
            [a, b, c].into_iter().collect::<HashSet<_>>()
        );
        assert_eq!(
            dag.descendants(a),
            [b, c, d].into_iter().collect::<HashSet<_>>()
        );
        assert!(dag.ancestors(a).is_empty());
        assert!(dag.descendants(d).is_empty());
    }

    #[test]
    fn scm_intervene_direct_cause() {
        // X → Y, P(X=0)=0.5, P(Y=1|X=0)=0.2, P(Y=1|X=1)=0.9
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary(), true);
        let y = dag.add_node("Y", binary(), true);
        dag.add_edge(x, y);

        let mut scm = StructuralCausalModel::new(dag);
        scm.set_conditional_table(x, vec![0.5, 0.5]);
        scm.set_conditional_table(y, vec![0.8, 0.2, 0.1, 0.9]); // P(Y|X)

        // do(X=1) should give P(Y=1) ≈ 0.9
        let result = scm.intervene(x, 1, y).expect("Should be identifiable");
        assert!(
            (result.distribution[1] - 0.9).abs() < 0.1,
            "P(Y=1|do(X=1)) should be ~0.9, got {}",
            result.distribution[1]
        );
    }

    #[test]
    fn scm_causal_effect_no_confounding() {
        // Simple X → Y with no confounders
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary(), true);
        let y = dag.add_node("Y", binary(), true);
        dag.add_edge(x, y);

        let mut scm = StructuralCausalModel::new(dag);
        scm.set_conditional_table(x, vec![0.5, 0.5]);
        scm.set_conditional_table(y, vec![0.9, 0.1, 0.1, 0.9]);

        let ace = scm.causal_effect(x, y);
        assert!(ace.is_some(), "Causal effect should be identifiable");
        let ace = ace.unwrap();
        assert!(ace.is_finite(), "ACE must be finite, got {}", ace);
    }

    #[test]
    fn scm_backdoor_with_confounder() {
        // U → X, U → Y, X → Y
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary(), true);
        let y = dag.add_node("Y", binary(), true);
        let u = dag.add_node("U", binary(), true);
        dag.add_edge(u, x);
        dag.add_edge(u, y);
        dag.add_edge(x, y);

        let scm = StructuralCausalModel::new(dag);
        let backdoor_sets = scm.find_backdoor_sets(x, y);
        assert!(
            !backdoor_sets.is_empty(),
            "Should find at least one backdoor set (U)"
        );
        // U should be in at least one backdoor set
        let has_u = backdoor_sets.iter().any(|s| s.contains(&u));
        assert!(has_u, "U should appear in a backdoor adjustment set");
    }

    #[test]
    fn scm_frontdoor_with_mediator() {
        // X → M → Y, U → X, U → Y (U unobserved)
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary(), true);
        let m = dag.add_node("M", binary(), true);
        let y = dag.add_node("Y", binary(), true);
        let u = dag.add_node("U", binary(), false); // unobserved
        dag.add_edge(x, m);
        dag.add_edge(m, y);
        dag.add_edge(u, x);
        dag.add_edge(u, y);

        let scm = StructuralCausalModel::new(dag);
        let frontdoor_sets = scm.find_frontdoor_sets(x, y);
        assert!(
            !frontdoor_sets.is_empty(),
            "Should find frontdoor set through M"
        );
    }

    #[test]
    fn d_separation_m_bias() {
        // M-bias: Z1 → X, Z1 → Z3, Z2 → Y, Z2 → Z3
        // X and Y are d-separated by {} (no direct path)
        // Conditioning on Z3 opens a collider path
        let mut dag = CausalDAG::new();
        let x = dag.add_node("X", binary(), true);
        let y = dag.add_node("Y", binary(), true);
        let z1 = dag.add_node("Z1", binary(), true);
        let z2 = dag.add_node("Z2", binary(), true);
        let z3 = dag.add_node("Z3", binary(), true);
        dag.add_edge(z1, x);
        dag.add_edge(z1, z3);
        dag.add_edge(z2, y);
        dag.add_edge(z2, z3);

        let empty: HashSet<usize> = HashSet::new();
        let res = dag.d_separated(x, y, &empty);
        assert!(
            res.separated,
            "X⊥Y with no conditioning in M-bias structure"
        );

        let z3_set: HashSet<usize> = [z3].into_iter().collect();
        let res = dag.d_separated(x, y, &z3_set);
        assert!(!res.separated, "Conditioning on collider Z3 opens the path");
    }
}

// ============================================================================
// CAUSAL EMERGENCE — EI, determinism, degeneracy, analyzer
// ============================================================================

mod causal_emergence_tests {
    use symthaea_causal_reasoning::causal_emergence::{
        CausalEmergenceAnalyzer, StateDiscretizer, degeneracy, determinism, effective_information,
    };

    #[test]
    fn state_discretizer_boundary_values() {
        let disc = StateDiscretizer::new(4, 2);

        // Value 0.0 → first bin
        assert_eq!(disc.to_micro(0.0), 0);
        assert_eq!(disc.to_macro(0.0), 0);

        // Value 1.0 → last bin (clamped)
        assert_eq!(disc.to_micro(1.0), 3);
        assert_eq!(disc.to_macro(1.0), 1);

        // Mid value
        assert_eq!(disc.to_micro(0.5), 2);
        assert_eq!(disc.to_macro(0.5), 1);
    }

    #[test]
    fn determinism_perfect_tpm() {
        // Each state deterministically maps to exactly one next state
        let tpm = vec![
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 0.0, 0.0],
        ];
        let det = determinism(&tpm);
        assert!(
            (det - 1.0).abs() < 0.01,
            "Perfect determinism should be ~1.0, got {}",
            det
        );
    }

    #[test]
    fn determinism_uniform_tpm() {
        // Each state randomly transitions to any state
        let tpm = vec![
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
            vec![0.25, 0.25, 0.25, 0.25],
        ];
        let det = determinism(&tpm);
        assert!(
            det.abs() < 0.01,
            "Uniform transitions should have ~0.0 determinism, got {}",
            det
        );
    }

    #[test]
    fn degeneracy_identity_tpm() {
        // Each state maps to itself — unique outputs, low degeneracy
        let tpm = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let deg = degeneracy(&tpm);
        assert!(
            deg < 0.5,
            "Identity TPM should have low degeneracy, got {}",
            deg
        );
    }

    #[test]
    fn degeneracy_all_same_output() {
        // All inputs lead to the same output distribution
        let tpm = vec![
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        let deg = degeneracy(&tpm);
        assert!(
            (deg - 1.0).abs() < 0.01,
            "All-same-output should have degeneracy ~1.0, got {}",
            deg
        );
    }

    #[test]
    fn effective_information_empty_tpm() {
        let ei = effective_information(&[]);
        assert!(ei.abs() < 1e-10, "Empty TPM should have 0 EI, got {}", ei);
    }

    #[test]
    fn ei_deterministic_greater_than_random() {
        let det_tpm = vec![vec![0.0, 1.0], vec![1.0, 0.0]];
        let rand_tpm = vec![vec![0.5, 0.5], vec![0.5, 0.5]];
        let ei_det = effective_information(&det_tpm);
        let ei_rand = effective_information(&rand_tpm);
        assert!(
            ei_det > ei_rand,
            "Deterministic EI ({}) should exceed random EI ({})",
            ei_det,
            ei_rand
        );
    }

    #[test]
    fn analyzer_reset_clears_observations() {
        let mut analyzer = CausalEmergenceAnalyzer::new();
        analyzer.observe(&[0.5, 0.3], 0.4);
        assert_eq!(analyzer.num_observations(), 1);

        analyzer.reset();
        assert_eq!(analyzer.num_observations(), 0);
    }

    #[test]
    fn analyzer_custom_granularity() {
        let mut analyzer = CausalEmergenceAnalyzer::with_granularity(8, 4);

        for i in 0..500 {
            let micro = vec![
                (i as f64 * 0.1).sin() * 0.5 + 0.5,
                (i as f64 * 0.2).cos() * 0.5 + 0.5,
            ];
            let macro_state = (i as f64 * 0.05).sin() * 0.3 + 0.5;
            analyzer.observe(&micro, macro_state);
        }

        let result = analyzer.analyze();
        assert!(result.micro_ei.is_finite() && result.micro_ei >= 0.0);
        assert!(result.macro_ei.is_finite() && result.macro_ei >= 0.0);
        assert!(!result.interpretation.is_empty());
    }
}

// ============================================================================
// COUNTERFACTUAL IDENTIFICATION — lightweight DAG, graph surgery, queries
// ============================================================================

#[cfg(feature = "counterfactual")]
mod counterfactual_tests {
    use super::*;
    use symthaea_causal_reasoning::counterfactual::identification::{
        CausalDAG, CausalQuery, CausalQueryOutcome, CounterfactualReasoner,
    };

    #[test]
    fn dag_node_index_lookup() {
        let dag = CausalDAG::new(
            vec!["treatment".into(), "mediator".into(), "outcome".into()],
            vec![(0, 1), (1, 2)],
        );
        assert_eq!(dag.node_index("treatment"), Some(0));
        assert_eq!(dag.node_index("mediator"), Some(1));
        assert_eq!(dag.node_index("outcome"), Some(2));
        assert_eq!(dag.node_index("nonexistent"), None);
    }

    #[test]
    fn dag_has_path_transitive() {
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into(), "D".into()],
            vec![(0, 1), (1, 2), (2, 3)],
        );
        assert!(dag.has_path(0, 3), "A → B → C → D should have path");
        assert!(!dag.has_path(3, 0), "No reverse path in DAG");
        assert!(!dag.has_path(0, 0), "No self-loop path");
    }

    #[test]
    fn dag_has_path_no_connection() {
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![(0, 1)], // A → B only, C isolated
        );
        assert!(!dag.has_path(0, 2), "No path from A to isolated C");
        assert!(!dag.has_path(2, 0), "No path from isolated C to A");
    }

    #[test]
    fn graph_surgery_remove_incoming() {
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![(0, 1), (1, 2), (0, 2)],
        );

        // Remove incoming edges to B (i.e., A → B)
        let mutilated = dag.remove_incoming(&[1]);
        assert!(
            !mutilated.edges.contains(&(0, 1)),
            "A → B should be removed"
        );
        assert!(mutilated.edges.contains(&(1, 2)), "B → C should remain");
        assert!(mutilated.edges.contains(&(0, 2)), "A → C should remain");
    }

    #[test]
    fn graph_surgery_remove_outgoing() {
        let dag = CausalDAG::new(
            vec!["A".into(), "B".into(), "C".into()],
            vec![(0, 1), (1, 2), (0, 2)],
        );

        // Remove outgoing edges from B (i.e., B → C)
        let mutilated = dag.remove_outgoing(&[1]);
        assert!(mutilated.edges.contains(&(0, 1)), "A → B should remain");
        assert!(
            !mutilated.edges.contains(&(1, 2)),
            "B → C should be removed"
        );
    }

    #[test]
    fn reasoner_identifies_simple_chain() {
        let dag = CausalDAG::new(
            vec!["X".into(), "M".into(), "Y".into()],
            vec![(0, 1), (1, 2)],
        );
        let query = CausalQuery {
            treatment: 0,
            outcome: 2,
            conditioning: vec![],
        };
        let reasoner = CounterfactualReasoner::new();
        let result = reasoner.query(&dag, &query);
        assert!(
            matches!(result, CausalQueryOutcome::Identified { .. }),
            "Simple chain X → M → Y should be identified"
        );
    }

    #[test]
    fn reasoner_unidentified_disconnected() {
        let dag = CausalDAG::new(
            vec!["X".into(), "Y".into()],
            vec![], // No edges
        );
        let query = CausalQuery {
            treatment: 0,
            outcome: 1,
            conditioning: vec![],
        };
        let reasoner = CounterfactualReasoner::new();
        let result = reasoner.query(&dag, &query);
        assert!(
            matches!(result, CausalQueryOutcome::Unidentified { .. }),
            "Disconnected graph should be unidentified"
        );
    }
}
