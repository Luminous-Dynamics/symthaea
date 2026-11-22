// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded SynQBF constructor from complete game-value profiles.
//!
//! This implements the finite circuit-composition construction of Lemma 4.7
//! in Akshay et al., SAT 2026: given circuits C_0, ..., C_n for all game-value
//! layers, build a paired-variable decision-list circuit that selects C_i
//! when x_i differs from xhat_i, falling back to C_0 otherwise.
//!
//! The game-value profile and every source layer are checked first. The
//! resulting circuit is then passed to the independent SynQBF verifier.
//! This is a table-backed finite constructor, not the paper's CEGC/AllSATRep
//! proof-construction algorithm for arbitrary QBF instances.

use super::decomposition::CnfProblem;
use super::quantified_game_dag::QbfGameDagBudget;
use super::quantified_game_values::{
    GameValueProfileBudget, GameValueProfileError, GameValueProfileVerificationReport,
    QuantifiedGameValueProfile, verify_game_value_profile,
};
use super::quantified_scope::{Quantifier, ValidatedQuantifierPrefix};
use super::quantified_strategy::BooleanStrategyFunction;
use super::strategy_aig::{
    AigLiteral, AigNode, StrategyAig, StrategyAigBudget, StrategyAigError, StrategyAigReport,
    compile_strategy_function_to_aig, validate_strategy_aig, verify_aig_matches_strategy_function,
};
use super::synqbf::{
    SynQbfBudget, SynQbfError, SynQbfProof, SynQbfVerificationReport, verify_synqbf_proof,
};
use std::collections::BTreeMap;

/// Explicit limits for source layer compilation, AIG composition and final
/// independent SynQBF verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynQbfConstructionBudget {
    pub game_values: GameValueProfileBudget,
    pub layer_aig: StrategyAigBudget,
    pub checker: SynQbfBudget,
    pub max_total_layer_aig_and_nodes: usize,
    pub max_composed_aig_and_nodes: usize,
    pub max_compose_operations: u128,
}

impl Default for SynQbfConstructionBudget {
    fn default() -> Self {
        Self {
            game_values: GameValueProfileBudget::default(),
            layer_aig: StrategyAigBudget::default(),
            checker: SynQbfBudget::default(),
            max_total_layer_aig_and_nodes: 1 << 22,
            max_composed_aig_and_nodes: 1 << 22,
            max_compose_operations: 1 << 24,
        }
    }
}

/// Fail-closed construction errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynQbfConstructionError {
    GameValueProfile(GameValueProfileError),
    LayerAigCompilation(StrategyAigError),
    LayerAigVerification(StrategyAigError),
    LayerAndNodeBudgetExceeded {
        required: usize,
        limit: usize,
    },
    ComposedAndNodeBudgetExceeded {
        required: usize,
        limit: usize,
    },
    ComposeOperationBudgetExceeded {
        required: u128,
        limit: u128,
    },
    InputCountOverflow,
    LayerDependencyScopeMismatch {
        layer: usize,
    },
    LayerInputOutOfRange {
        layer: usize,
        dependency_index: usize,
    },
    CircuitCompositionOutOfRange {
        node: usize,
        node_count: usize,
    },
    FinalAigValidation(StrategyAigError),
    SynQbfVerification(SynQbfError),
    ResourceAccountingOverflow,
}

/// Per-layer source-circuit measurements.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynQbfLayerCompilationReport {
    pub prefix_position: usize,
    pub semantic_dependencies: Vec<usize>,
    pub truth_table_entries: usize,
    pub aig_report: StrategyAigReport,
}

/// Evidence for the finite game-value-to-SynQBF construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynQbfConstructionReport {
    pub game_value_verification: GameValueProfileVerificationReport,
    pub layers: Vec<SynQbfLayerCompilationReport>,
    pub total_layer_aig_and_nodes: usize,
    pub composed_aig_and_nodes: usize,
    pub compose_operations: u128,
    pub final_synqbf_verification: SynQbfVerificationReport,
}

/// A completed finite SynQBF proof candidate with independent checking reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConstructedSynQbfProof {
    pub proof: SynQbfProof,
    pub report: SynQbfConstructionReport,
}

struct AigComposer {
    nodes: Vec<AigNode>,
    unique: BTreeMap<(AigLiteral, AigLiteral), usize>,
    and_nodes: usize,
    operations: u128,
}

impl AigComposer {
    fn with_inputs(input_count: usize) -> Self {
        let nodes = (0..input_count)
            .map(|dependency_index| AigNode::Input { dependency_index })
            .collect();
        Self {
            nodes,
            unique: BTreeMap::new(),
            and_nodes: 0,
            operations: 0,
        }
    }

    fn and(
        &mut self,
        left: AigLiteral,
        right: AigLiteral,
        budget: SynQbfConstructionBudget,
    ) -> Result<AigLiteral, SynQbfConstructionError> {
        self.operations = self
            .operations
            .checked_add(1)
            .ok_or(SynQbfConstructionError::ResourceAccountingOverflow)?;
        if self.operations > budget.max_compose_operations {
            return Err(SynQbfConstructionError::ComposeOperationBudgetExceeded {
                required: self.operations,
                limit: budget.max_compose_operations,
            });
        }

        match (left, right) {
            (AigLiteral::Constant(false), _) | (_, AigLiteral::Constant(false)) => {
                return Ok(AigLiteral::FALSE);
            }
            (AigLiteral::Constant(true), value) | (value, AigLiteral::Constant(true)) => {
                return Ok(value);
            }
            _ => {}
        }
        if left == right {
            return Ok(left);
        }
        if left.is_complement_of(right) {
            return Ok(AigLiteral::FALSE);
        }

        let key = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        if let Some(&existing) = self.unique.get(&key) {
            return Ok(AigLiteral::node(existing));
        }
        if self.and_nodes >= budget.max_composed_aig_and_nodes {
            return Err(SynQbfConstructionError::ComposedAndNodeBudgetExceeded {
                required: self.and_nodes.saturating_add(1),
                limit: budget.max_composed_aig_and_nodes,
            });
        }

        let index = self.nodes.len();
        self.nodes.push(AigNode::And {
            left: key.0,
            right: key.1,
        });
        self.unique.insert(key, index);
        self.and_nodes = self
            .and_nodes
            .checked_add(1)
            .ok_or(SynQbfConstructionError::ResourceAccountingOverflow)?;
        Ok(AigLiteral::node(index))
    }

    fn or(
        &mut self,
        left: AigLiteral,
        right: AigLiteral,
        budget: SynQbfConstructionBudget,
    ) -> Result<AigLiteral, SynQbfConstructionError> {
        Ok(self.and(left.not(), right.not(), budget)?.not())
    }

    fn xor(
        &mut self,
        left: AigLiteral,
        right: AigLiteral,
        budget: SynQbfConstructionBudget,
    ) -> Result<AigLiteral, SynQbfConstructionError> {
        let left_high = self.and(left, right.not(), budget)?;
        let right_high = self.and(left.not(), right, budget)?;
        self.or(left_high, right_high, budget)
    }

    fn mux(
        &mut self,
        selector: AigLiteral,
        when_true: AigLiteral,
        when_false: AigLiteral,
        budget: SynQbfConstructionBudget,
    ) -> Result<AigLiteral, SynQbfConstructionError> {
        let true_branch = self.and(selector, when_true, budget)?;
        let false_branch = self.and(selector.not(), when_false, budget)?;
        self.or(true_branch, false_branch, budget)
    }

    fn clone_local_node(
        &mut self,
        layer_position: usize,
        local_index: usize,
        local_aig: &StrategyAig,
        dependency_to_paired_input: &[usize],
        memo: &mut [Option<AigLiteral>],
        budget: SynQbfConstructionBudget,
    ) -> Result<AigLiteral, SynQbfConstructionError> {
        if let Some(value) = memo.get(local_index).copied().flatten() {
            return Ok(value);
        }
        let node = local_aig.nodes.get(local_index).copied().ok_or(
            SynQbfConstructionError::CircuitCompositionOutOfRange {
                node: local_index,
                node_count: local_aig.nodes.len(),
            },
        )?;
        let value = match node {
            AigNode::Input { dependency_index } => {
                let paired_input = *dependency_to_paired_input.get(dependency_index).ok_or(
                    SynQbfConstructionError::LayerInputOutOfRange {
                        layer: layer_position,
                        dependency_index,
                    },
                )?;
                AigLiteral::node(paired_input)
            }
            AigNode::And { left, right } => {
                let left = self.clone_local_literal(
                    layer_position,
                    left,
                    local_aig,
                    dependency_to_paired_input,
                    memo,
                    budget,
                )?;
                let right = self.clone_local_literal(
                    layer_position,
                    right,
                    local_aig,
                    dependency_to_paired_input,
                    memo,
                    budget,
                )?;
                self.and(left, right, budget)?
            }
        };
        let memo_len = memo.len();
        let slot = memo.get_mut(local_index).ok_or(
            SynQbfConstructionError::CircuitCompositionOutOfRange {
                node: local_index,
                node_count: memo_len,
            },
        )?;
        *slot = Some(value);
        Ok(value)
    }

    fn clone_local_literal(
        &mut self,
        layer_position: usize,
        literal: AigLiteral,
        local_aig: &StrategyAig,
        dependency_to_paired_input: &[usize],
        memo: &mut [Option<AigLiteral>],
        budget: SynQbfConstructionBudget,
    ) -> Result<AigLiteral, SynQbfConstructionError> {
        match literal {
            AigLiteral::Constant(value) => Ok(AigLiteral::Constant(value)),
            AigLiteral::Node { index, inverted } => {
                let base = self.clone_local_node(
                    layer_position,
                    index,
                    local_aig,
                    dependency_to_paired_input,
                    memo,
                    budget,
                )?;
                Ok(if inverted { base.not() } else { base })
            }
        }
    }

    fn clone_local_output(
        &mut self,
        layer_position: usize,
        local_aig: &StrategyAig,
        dependency_to_paired_input: &[usize],
        budget: SynQbfConstructionBudget,
    ) -> Result<AigLiteral, SynQbfConstructionError> {
        let mut memo = vec![None; local_aig.nodes.len()];
        self.clone_local_literal(
            layer_position,
            local_aig.output,
            local_aig,
            dependency_to_paired_input,
            &mut memo,
            budget,
        )
    }
}

/// Compile a fully verified game-value profile into the paired-variable
/// decision-list SynQBF circuit from Lemma 4.7, then verify that proof object
/// independently using Definition 4.3.
pub fn construct_synqbf_from_game_value_profile(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    profile: &QuantifiedGameValueProfile,
    budget: SynQbfConstructionBudget,
) -> Result<ConstructedSynQbfProof, SynQbfConstructionError> {
    let profile_verification =
        verify_game_value_profile(problem, prefix, profile, budget.game_values)
            .map_err(SynQbfConstructionError::GameValueProfile)?;

    let n = profile.prefix_order.len();
    if n > budget.checker.max_variables {
        return Err(SynQbfConstructionError::CheckerVariableBudgetExceeded {
            required: n,
            limit: budget.checker.max_variables,
        });
    }
    let paired_input_count = n
        .checked_mul(2)
        .ok_or(SynQbfConstructionError::InputCountOverflow)?;
    if paired_input_count > budget.checker.aig.max_dependency_variables {
        return Err(SynQbfConstructionError::LayerAigCompilation(
            StrategyAigError::DependencyBudgetExceeded {
                required: paired_input_count,
                limit: budget.checker.aig.max_dependency_variables,
            },
        ));
    }

    let mut layer_circuits = Vec::with_capacity(profile.layers.len());
    let mut layer_reports = Vec::with_capacity(profile.layers.len());
    let mut total_layer_aig_and_nodes = 0usize;

    for layer in &profile.layers {
        let function = BooleanStrategyFunction {
            variable: layer.prefix_position,
            dependencies: layer.prefix_variables.clone(),
            table: layer.values.clone(),
        };
        let aig = compile_strategy_function_to_aig(&function, budget.layer_aig)
            .map_err(SynQbfConstructionError::LayerAigCompilation)?;
        let report = verify_aig_matches_strategy_function(&aig, &function, budget.layer_aig)
            .map_err(SynQbfConstructionError::LayerAigVerification)?;
        total_layer_aig_and_nodes = total_layer_aig_and_nodes
            .checked_add(report.and_nodes)
            .ok_or(SynQbfConstructionError::ResourceAccountingOverflow)?;
        if total_layer_aig_and_nodes > budget.max_total_layer_aig_and_nodes {
            return Err(SynQbfConstructionError::LayerAndNodeBudgetExceeded {
                required: total_layer_aig_and_nodes,
                limit: budget.max_total_layer_aig_and_nodes,
            });
        }
        layer_reports.push(SynQbfLayerCompilationReport {
            prefix_position: layer.prefix_position,
            semantic_dependencies: function.dependencies,
            truth_table_entries: function.table.len(),
            aig_report: report,
        });
        layer_circuits.push(aig);
    }

    // Import C_0 first. For i = 1..n, wrap the current result in
    // if x_i != xhat_i then C_i else result. This leaves C_n outermost.
    let mut composer = AigComposer::with_inputs(paired_input_count);
    let variable_position_by_id = {
        let mut positions = vec![usize::MAX; problem.variables.len()];
        for (position, &(_, variable)) in profile.prefix_order.iter().enumerate() {
            positions[variable] = position;
        }
        positions
    };

    let mut initial_mapping = Vec::with_capacity(layer_circuits[0].dependencies.len());
    for &variable in &layer_circuits[0].dependencies {
        let position = *variable_position_by_id
            .get(variable)
            .ok_or(SynQbfConstructionError::LayerDependencyScopeMismatch { layer: 0 })?;
        if position == usize::MAX {
            return Err(SynQbfConstructionError::LayerDependencyScopeMismatch { layer: 0 });
        }
        initial_mapping.push(position);
    }
    let mut result =
        composer.clone_local_output(0, &layer_circuits[0], &initial_mapping, budget)?;

    for position in 1..=n {
        let layer = &layer_circuits[position];
        let mut mapping = Vec::with_capacity(layer.dependencies.len());
        for &variable in &layer.dependencies {
            let position_of_variable = *variable_position_by_id
                .get(variable)
                .ok_or(SynQbfConstructionError::LayerDependencyScopeMismatch { layer: position })?;
            if position_of_variable == usize::MAX || position_of_variable >= position {
                return Err(SynQbfConstructionError::LayerDependencyScopeMismatch {
                    layer: position,
                });
            }
            mapping.push(position_of_variable);
        }

        let layer_value = composer.clone_local_output(position, layer, &mapping, budget)?;
        let current = AigLiteral::node(position - 1);
        let dual = AigLiteral::node(n + position - 1);
        let differs = composer.xor(current, dual, budget)?;
        result = composer.mux(differs, layer_value, result, budget)?;
    }

    let circuit = StrategyAig {
        dependencies: (0..paired_input_count).collect(),
        decomposition_order: (0..paired_input_count).collect(),
        nodes: composer.nodes,
        output: result,
    };
    validate_strategy_aig(&circuit, budget.checker.aig)
        .map_err(SynQbfConstructionError::FinalAigValidation)?;

    let proof = SynQbfProof {
        variable_count: problem.variables.len(),
        prefix_order: profile.prefix_order.clone(),
        claimed_truth_value: profile.truth_value,
        circuit,
    };
    let synqbf_verification = verify_synqbf_proof(problem, prefix, &proof, budget.checker)
        .map_err(SynQbfConstructionError::SynQbfVerification)?;

    Ok(ConstructedSynQbfProof {
        proof,
        report: SynQbfConstructionReport {
            game_value_verification: profile_verification,
            layers: layer_reports,
            total_layer_aig_and_nodes,
            composed_aig_and_nodes: composer.and_nodes,
            compose_operations: composer.operations,
            final_synqbf_verification: synqbf_verification,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::super::decomposition::Literal;
    use super::super::qdimacs::{QdimacsParseBudget, parse_qdimacs};
    use super::super::quantified_game_dag::{QbfGameDagBudget, build_qbf_game_proof_dag};
    use super::super::quantified_game_values::{
        GameValueProfileBudget, extract_game_value_profile_from_dag,
    };
    use super::super::quantified_scope::QuantifierBlock;
    use super::*;

    fn block(quantifier: Quantifier, variables: &[usize]) -> QuantifierBlock {
        QuantifierBlock {
            quantifier,
            variables: variables.to_vec(),
        }
    }

    fn prefix(variable_count: usize, blocks: Vec<QuantifierBlock>) -> ValidatedQuantifierPrefix {
        ValidatedQuantifierPrefix::new(variable_count, blocks, Default::default()).unwrap()
    }

    fn dag_budget() -> QbfGameDagBudget {
        QbfGameDagBudget {
            max_variables: 8,
            max_total_variable_name_bytes: 256,
            max_clauses: 128,
            max_literals: 512,
            max_proof_nodes: 1 << 12,
            max_builder_state_visits: 1 << 12,
            max_verifier_node_visits: 1 << 12,
            max_residual_literal_visits: 1 << 18,
            max_stored_residual_literals: 1 << 18,
            max_stored_residual_clauses: 1 << 16,
        }
    }

    fn construction_budget() -> SynQbfConstructionBudget {
        SynQbfConstructionBudget {
            game_values: GameValueProfileBudget {
                max_variables: 8,
                max_total_variable_name_bytes: 256,
                max_clauses: 128,
                max_literals: 512,
                max_total_truth_table_entries: 1 << 12,
                max_dag_path_steps: 1 << 14,
                max_matrix_literal_checks: 1 << 14,
            },
            layer_aig: StrategyAigBudget {
                max_dependency_variables: 8,
                max_truth_table_entries: 1 << 8,
                max_and_nodes: 1 << 12,
                max_assignments_checked: 1 << 8,
            },
            checker: SynQbfBudget {
                max_variables: 8,
                max_total_variable_name_bytes: 256,
                max_clauses: 128,
                max_literals: 512,
                max_circuit_evaluations: 1 << 14,
                max_estimated_aig_node_visits: 1 << 24,
                aig: StrategyAigBudget {
                    max_dependency_variables: 16,
                    max_truth_table_entries: 1 << 16,
                    max_and_nodes: 1 << 14,
                    max_assignments_checked: 1 << 16,
                },
            },
            max_total_layer_aig_and_nodes: 1 << 14,
            max_composed_aig_and_nodes: 1 << 14,
            max_compose_operations: 1 << 18,
        }
    }

    fn two_variable_formulas() -> Vec<CnfProblem> {
        let literals = [
            Literal::positive(0),
            Literal::negative(0),
            Literal::positive(1),
            Literal::negative(1),
        ];
        let mut formulas = vec![CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![],
        }];

        for left_mask in 0..16usize {
            let left: Vec<_> = (0..4)
                .filter(|index| ((left_mask >> index) & 1) != 0)
                .map(|index| literals[index])
                .collect();
            formulas.push(CnfProblem {
                variables: vec!["x".into(), "y".into()],
                clauses: vec![left.clone()],
            });

            for right_mask in 0..16usize {
                let right: Vec<_> = (0..4)
                    .filter(|index| ((right_mask >> index) & 1) != 0)
                    .map(|index| literals[index])
                    .collect();
                formulas.push(CnfProblem {
                    variables: vec!["x".into(), "y".into()],
                    clauses: vec![left.clone(), right],
                });
            }
        }
        formulas
    }

    fn paper_example() -> (CnfProblem, ValidatedQuantifierPrefix) {
        let problem = CnfProblem {
            variables: vec!["x1".into(), "x2".into(), "x3".into()],
            clauses: vec![
                vec![
                    Literal::positive(0),
                    Literal::positive(1),
                    Literal::positive(2),
                ],
                vec![
                    Literal::negative(0),
                    Literal::negative(1),
                    Literal::negative(2),
                ],
            ],
        };
        let prefix = prefix(
            3,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
                block(Quantifier::ForAll, &[2]),
            ],
        );
        (problem, prefix)
    }

    #[test]
    fn constructs_and_checks_synqbf_from_paper_game_value_example() {
        let (problem, prefix) = paper_example();
        let dag_budget = dag_budget();
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget).unwrap();
        let profile = extract_game_value_profile_from_dag(
            &problem,
            &prefix,
            &dag,
            dag_budget,
            construction_budget().game_values,
        )
        .unwrap();
        let result = construct_synqbf_from_game_value_profile(
            &problem,
            &prefix,
            &profile,
            construction_budget(),
        )
        .unwrap();
        assert!(result.proof.claimed_truth_value);
        assert_eq!(result.report.layers.len(), 4);
        assert_eq!(
            result
                .report
                .final_synqbf_verification
                .source_equivalence_assignments_checked,
            8
        );
        assert_eq!(
            result
                .report
                .final_synqbf_verification
                .representability_instances_checked,
            7
        );
    }

    #[test]
    fn strict_qdimacs_input_reaches_independently_verified_synqbf_proof() {
        let source = concat!(
            "c forall x exists y: x iff y\n",
            "p cnf 2 2\n",
            "a 1 0\n",
            "e 2 0\n",
            "-1 2 0\n",
            "1 -2 0\n"
        );
        let parsed = parse_qdimacs(source, QdimacsParseBudget::default()).unwrap();
        assert_eq!(parsed.stats.declared_variables, 2);
        assert_eq!(parsed.stats.parsed_clauses, 2);
        assert_eq!(parsed.prefix.variable_count(), 2);

        let dag_budget = dag_budget();
        let (dag, _) =
            build_qbf_game_proof_dag(&parsed.problem, &parsed.prefix, dag_budget).unwrap();
        let profile = extract_game_value_profile_from_dag(
            &parsed.problem,
            &parsed.prefix,
            &dag,
            dag_budget,
            construction_budget().game_values,
        )
        .unwrap();
        let constructed = construct_synqbf_from_game_value_profile(
            &parsed.problem,
            &parsed.prefix,
            &profile,
            construction_budget(),
        )
        .unwrap();
        assert!(constructed.proof.claimed_truth_value);
        assert_eq!(
            constructed
                .report
                .final_synqbf_verification
                .source_equivalence_assignments_checked,
            4
        );
        assert_eq!(
            constructed
                .report
                .final_synqbf_verification
                .representability_conditions_checked,
            3
        );
    }

    #[test]
    fn every_two_variable_cnf_and_quantifier_prefix_constructs_verified_synqbf() {
        let formulas = two_variable_formulas();
        assert_eq!(formulas.len(), 273);

        let orders = [[0usize, 1usize], [1usize, 0usize]];
        for (formula_index, problem) in formulas.iter().enumerate() {
            for order in orders {
                for quantifier_mask in 0..4usize {
                    let blocks: Vec<_> = order
                        .iter()
                        .enumerate()
                        .map(|(position, &variable)| {
                            block(
                                if ((quantifier_mask >> position) & 1) == 0 {
                                    Quantifier::ForAll
                                } else {
                                    Quantifier::Exists
                                },
                                &[variable],
                            )
                        })
                        .collect();
                    let prefix = prefix(2, blocks);
                    let dag_budget = dag_budget();
                    let (dag, _) = build_qbf_game_proof_dag(problem, &prefix, dag_budget).unwrap();
                    let profile = extract_game_value_profile_from_dag(
                        problem,
                        &prefix,
                        &dag,
                        dag_budget,
                        construction_budget().game_values,
                    )
                    .unwrap();
                    let constructed = construct_synqbf_from_game_value_profile(
                        problem,
                        &prefix,
                        &profile,
                        construction_budget(),
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "formula {formula_index}, order {order:?}, quantifier mask {quantifier_mask}: {error:?}"
                        )
                    });

                    assert_eq!(
                        constructed.proof.claimed_truth_value, profile.truth_value,
                        "formula {formula_index}, order {order:?}, quantifier mask {quantifier_mask}"
                    );
                    assert_eq!(constructed.report.layers.len(), 3);
                    assert_eq!(
                        constructed
                            .report
                            .final_synqbf_verification
                            .source_equivalence_assignments_checked,
                        4
                    );
                    assert_eq!(
                        constructed
                            .report
                            .final_synqbf_verification
                            .representability_instances_checked,
                        3
                    );
                    assert_eq!(
                        constructed
                            .report
                            .final_synqbf_verification
                            .representability_conditions_checked,
                        3
                    );
                }
            }
        }
    }

    #[test]
    fn empty_prefix_constructs_true_and_false_constant_synqbf_proofs() {
        for (clauses, expected) in [(vec![], true), (vec![vec![]], false)] {
            let problem = CnfProblem {
                variables: vec![],
                clauses,
            };
            let prefix = prefix(0, vec![]);
            let dag_budget = dag_budget();
            let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget).unwrap();
            let profile = extract_game_value_profile_from_dag(
                &problem,
                &prefix,
                &dag,
                dag_budget,
                construction_budget().game_values,
            )
            .unwrap();
            let constructed = construct_synqbf_from_game_value_profile(
                &problem,
                &prefix,
                &profile,
                construction_budget(),
            )
            .unwrap();
            assert_eq!(constructed.proof.claimed_truth_value, expected);
            assert!(constructed.proof.circuit.dependencies.is_empty());
            assert_eq!(constructed.report.layers.len(), 1);
            assert_eq!(
                constructed
                    .report
                    .final_synqbf_verification
                    .source_equivalence_assignments_checked,
                1
            );
            assert_eq!(
                constructed
                    .report
                    .final_synqbf_verification
                    .root_value_checks,
                1
            );
        }
    }

    #[test]
    fn construction_rejects_tampered_game_value_profile() {
        let (problem, prefix) = paper_example();
        let dag_budget = dag_budget();
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget).unwrap();
        let mut profile = extract_game_value_profile_from_dag(
            &problem,
            &prefix,
            &dag,
            dag_budget,
            construction_budget().game_values,
        )
        .unwrap();
        let old_value = profile.layers[1].values[&vec![false]];
        profile.layers[1].values.insert(vec![false], !old_value);
        assert!(matches!(
            construct_synqbf_from_game_value_profile(
                &problem,
                &prefix,
                &profile,
                construction_budget()
            ),
            Err(SynQbfConstructionError::GameValueProfile(_))
        ));
    }

    #[test]
    fn composition_operation_budget_is_fail_closed() {
        let (problem, prefix) = paper_example();
        let dag_budget = dag_budget();
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget).unwrap();
        let profile = extract_game_value_profile_from_dag(
            &problem,
            &prefix,
            &dag,
            dag_budget,
            construction_budget().game_values,
        )
        .unwrap();
        let tight = SynQbfConstructionBudget {
            max_compose_operations: 0,
            ..construction_budget()
        };
        assert!(matches!(
            construct_synqbf_from_game_value_profile(&problem, &prefix, &profile, tight),
            Err(SynQbfConstructionError::ComposeOperationBudgetExceeded { .. })
        ));
    }
}
