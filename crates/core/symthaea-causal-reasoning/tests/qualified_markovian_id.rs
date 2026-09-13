// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use symthaea_causal_reasoning::counterfactual::{
    CausalExpression, CausalGraphWithLatents, QualifiedMarkovianId,
};

fn canonical_by_name(expression: &CausalExpression, nodes: &[String]) -> String {
    match expression {
        CausalExpression::Probability {
            outcome,
            conditioning,
        } => {
            let mut outcome_names: Vec<&str> = outcome.iter().map(|&i| nodes[i].as_str()).collect();
            let mut conditioning_names: Vec<&str> =
                conditioning.iter().map(|&i| nodes[i].as_str()).collect();
            outcome_names.sort_unstable();
            conditioning_names.sort_unstable();
            format!(
                "P({}|{})",
                outcome_names.join(","),
                conditioning_names.join(",")
            )
        }
        CausalExpression::Sum { sum_over, inner } => {
            let mut names: Vec<&str> = sum_over.iter().map(|&i| nodes[i].as_str()).collect();
            names.sort_unstable();
            format!(
                "SUM[{}]({})",
                names.join(","),
                canonical_by_name(inner, nodes)
            )
        }
        CausalExpression::Product(parts) => {
            let mut canonical_parts: Vec<String> = parts
                .iter()
                .map(|part| canonical_by_name(part, nodes))
                .collect();
            canonical_parts.sort_unstable();
            format!("PRODUCT[{}]", canonical_parts.join(";"))
        }
        CausalExpression::Fraction {
            numerator,
            denominator,
        } => format!(
            "FRACTION({})/({})",
            canonical_by_name(numerator, nodes),
            canonical_by_name(denominator, nodes)
        ),
    }
}

#[test]
fn qualified_markovian_id_is_invariant_to_node_index_permutation() {
    // Same named graph with different numeric node assignments:
    // Z → X, Z → Y, X → Y.
    let original = CausalGraphWithLatents::new(
        vec!["X".into(), "Y".into(), "Z".into()],
        vec![(2, 0), (2, 1), (0, 1)],
        vec![],
    );
    let permuted = CausalGraphWithLatents::new(
        vec!["Z".into(), "Y".into(), "X".into()],
        vec![(0, 2), (0, 1), (2, 1)],
        vec![],
    );

    let left = QualifiedMarkovianId::new()
        .identify(&original, &[0], &[1])
        .unwrap_or_else(|err| panic!("original graph must identify: {err}"));
    let right = QualifiedMarkovianId::new()
        .identify(&permuted, &[2], &[1])
        .unwrap_or_else(|err| panic!("permuted graph must identify: {err}"));

    assert_eq!(
        canonical_by_name(&left, &original.nodes),
        canonical_by_name(&right, &permuted.nodes)
    );
}

#[test]
fn qualified_markovian_id_has_exact_repeat_identity() {
    let graph = CausalGraphWithLatents::new(
        vec!["X".into(), "Y".into(), "Z1".into(), "Z2".into()],
        vec![(2, 0), (2, 1), (3, 0), (3, 1), (0, 1)],
        vec![],
    );
    let engine = QualifiedMarkovianId::new();
    let first = engine
        .identify(&graph, &[0], &[1])
        .unwrap_or_else(|err| panic!("first identification must succeed: {err}"));
    let second = engine
        .identify(&graph, &[0], &[1])
        .unwrap_or_else(|err| panic!("second identification must succeed: {err}"));

    let first_json = serde_json::to_string(&first)
        .unwrap_or_else(|err| panic!("first expression must serialize: {err}"));
    let second_json = serde_json::to_string(&second)
        .unwrap_or_else(|err| panic!("second expression must serialize: {err}"));
    assert_eq!(first_json, second_json);
}
