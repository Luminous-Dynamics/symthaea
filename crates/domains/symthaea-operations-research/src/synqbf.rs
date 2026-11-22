// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Bounded independent checker for SynQBF paired-variable circuit proofs.
//!
//! Inputs are ordered [x_1, ..., x_n, xhat_1, ..., xhat_n]. The checker
//! implements Definition 4.3 of Akshay et al., SAT 2026:
//! (1) source matrix equivalence under xhat_i := !x_i;
//! (2) local OR/AND representability at every prefix position;
//! (3) root value agreement under x_i = xhat_i = b_i.
//!
//! The proof circuit reuses StrategyAig. This is an exhaustive checker for
//! bounded CNF source matrices, not a SynQBF proof constructor or QBF solver.

use super::decomposition::CnfProblem;
use super::quantified_scope::{Quantifier, ValidatedQuantifierPrefix};
use super::strategy_aig::{
    StrategyAig, StrategyAigBudget, StrategyAigError, evaluate_strategy_aig, validate_strategy_aig,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynQbfProof {
    pub variable_count: usize,
    pub prefix_order: Vec<(Quantifier, usize)>,
    pub claimed_truth_value: bool,
    /// Circuit inputs are [x_1, ..., x_n, xhat_1, ..., xhat_n].
    pub circuit: StrategyAig,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynQbfBudget {
    pub max_variables: usize,
    pub max_total_variable_name_bytes: usize,
    pub max_clauses: usize,
    pub max_literals: usize,
    pub max_circuit_evaluations: u128,
    pub max_estimated_aig_node_visits: u128,
    pub aig: StrategyAigBudget,
}

impl Default for SynQbfBudget {
    fn default() -> Self {
        Self {
            max_variables: 10,
            max_total_variable_name_bytes: 4 * 1024,
            max_clauses: 1 << 16,
            max_literals: 1 << 20,
            max_circuit_evaluations: 1 << 18,
            max_estimated_aig_node_visits: 1 << 28,
            aig: StrategyAigBudget {
                max_dependency_variables: 20,
                max_truth_table_entries: 1 << 20,
                max_and_nodes: 1 << 20,
                max_assignments_checked: 1 << 20,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynQbfError {
    MatrixPrefixVariableCountMismatch {
        matrix: usize,
        prefix: usize,
    },
    VariableBudgetExceeded {
        required: usize,
        limit: usize,
    },
    VariableNameBudgetExceeded {
        required: usize,
        limit: usize,
    },
    EmptyVariableName {
        index: usize,
    },
    DuplicateVariableName(String),
    ClauseBudgetExceeded {
        required: usize,
        limit: usize,
    },
    LiteralBudgetExceeded {
        required: usize,
        limit: usize,
    },
    LiteralVariableOutOfRange {
        clause: usize,
        literal: usize,
        variable: usize,
        variable_count: usize,
    },
    ProofVariableCountMismatch {
        proof: usize,
        source: usize,
    },
    PrefixBindingMismatch,
    CircuitDependencyCountMismatch {
        expected: usize,
        actual: usize,
    },
    CircuitDependencyOrderMismatch {
        position: usize,
        expected: usize,
        actual: usize,
    },
    CircuitValidation(StrategyAigError),
    CircuitEvaluation(StrategyAigError),
    CircuitEvaluationBudgetExceeded {
        required: u128,
        limit: u128,
    },
    EstimatedAigNodeVisitBudgetExceeded {
        required: u128,
        limit: u128,
    },
    SourceEquivalenceCounterexample {
        prefix_assignment: Vec<bool>,
        source_value: bool,
        circuit_value: bool,
    },
    RepresentabilityViolation {
        prefix_position: usize,
        earlier_assignment: Vec<bool>,
        diagonal_value: bool,
        branch_01_value: bool,
        branch_10_value: bool,
        required_value: bool,
    },
    ClaimedTruthValueMismatch {
        claimed: bool,
        circuit_value: bool,
    },
    ResourceAccountingOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SynQbfVerificationReport {
    pub source_equivalence_assignments_checked: u128,
    pub representability_instances_checked: u128,
    pub representability_conditions_checked: u128,
    pub root_value_checks: u128,
    pub circuit_evaluations: u128,
    /// Conservative structural/evaluation work estimate, not wall-clock time.
    pub estimated_aig_node_visits: u128,
    pub source_matrix_literal_checks: u128,
    pub claimed_truth_value: bool,
}

fn checked_power_of_two(exponent: usize) -> Result<u128, SynQbfError> {
    let shift = u32::try_from(exponent).map_err(|_| SynQbfError::ResourceAccountingOverflow)?;
    1u128
        .checked_shl(shift)
        .ok_or(SynQbfError::ResourceAccountingOverflow)
}

fn ordered_prefix(prefix: &ValidatedQuantifierPrefix) -> Vec<(Quantifier, usize)> {
    prefix
        .blocks()
        .iter()
        .flat_map(|block| {
            block
                .variables
                .iter()
                .copied()
                .map(move |variable| (block.quantifier, variable))
        })
        .collect()
}

fn validate_source(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    budget: SynQbfBudget,
) -> Result<Vec<(Quantifier, usize)>, SynQbfError> {
    if problem.variables.len() != prefix.variable_count() {
        return Err(SynQbfError::MatrixPrefixVariableCountMismatch {
            matrix: problem.variables.len(),
            prefix: prefix.variable_count(),
        });
    }
    if problem.variables.len() > budget.max_variables {
        return Err(SynQbfError::VariableBudgetExceeded {
            required: problem.variables.len(),
            limit: budget.max_variables,
        });
    }
    let name_bytes = problem.variables.iter().try_fold(0usize, |total, name| {
        total
            .checked_add(name.len())
            .ok_or(SynQbfError::ResourceAccountingOverflow)
    })?;
    if name_bytes > budget.max_total_variable_name_bytes {
        return Err(SynQbfError::VariableNameBudgetExceeded {
            required: name_bytes,
            limit: budget.max_total_variable_name_bytes,
        });
    }
    let mut names = std::collections::BTreeSet::new();
    for (index, name) in problem.variables.iter().enumerate() {
        if name.is_empty() {
            return Err(SynQbfError::EmptyVariableName { index });
        }
        if !names.insert(name.as_str()) {
            return Err(SynQbfError::DuplicateVariableName(name.clone()));
        }
    }
    if problem.clauses.len() > budget.max_clauses {
        return Err(SynQbfError::ClauseBudgetExceeded {
            required: problem.clauses.len(),
            limit: budget.max_clauses,
        });
    }
    let literal_count = problem.clauses.iter().try_fold(0usize, |total, clause| {
        total
            .checked_add(clause.len())
            .ok_or(SynQbfError::ResourceAccountingOverflow)
    })?;
    if literal_count > budget.max_literals {
        return Err(SynQbfError::LiteralBudgetExceeded {
            required: literal_count,
            limit: budget.max_literals,
        });
    }
    for (clause_index, clause) in problem.clauses.iter().enumerate() {
        for (literal_index, literal) in clause.iter().enumerate() {
            if literal.variable >= problem.variables.len() {
                return Err(SynQbfError::LiteralVariableOutOfRange {
                    clause: clause_index,
                    literal: literal_index,
                    variable: literal.variable,
                    variable_count: problem.variables.len(),
                });
            }
        }
    }
    Ok(ordered_prefix(prefix))
}

fn source_matrix_value(
    problem: &CnfProblem,
    assignment: &[bool],
    literal_checks: &mut u128,
) -> Result<bool, SynQbfError> {
    for clause in &problem.clauses {
        let mut satisfied = false;
        for literal in clause {
            *literal_checks = literal_checks
                .checked_add(1)
                .ok_or(SynQbfError::ResourceAccountingOverflow)?;
            if assignment[literal.variable] == literal.positive {
                satisfied = true;
                break;
            }
        }
        if !satisfied {
            return Ok(false);
        }
    }
    Ok(true)
}

fn paired_assignment(prefix_values: &[bool]) -> Vec<bool> {
    let mut paired = Vec::with_capacity(prefix_values.len().saturating_mul(2));
    paired.extend_from_slice(prefix_values);
    paired.extend(prefix_values.iter().map(|value| !value));
    paired
}

fn evaluate_counted(
    proof: &SynQbfProof,
    assignment: &[bool],
    budget: SynQbfBudget,
    report: &mut SynQbfVerificationReport,
) -> Result<bool, SynQbfError> {
    report.circuit_evaluations = report
        .circuit_evaluations
        .checked_add(1)
        .ok_or(SynQbfError::ResourceAccountingOverflow)?;
    if report.circuit_evaluations > budget.max_circuit_evaluations {
        return Err(SynQbfError::CircuitEvaluationBudgetExceeded {
            required: report.circuit_evaluations,
            limit: budget.max_circuit_evaluations,
        });
    }
    evaluate_strategy_aig(&proof.circuit, assignment, budget.aig)
        .map_err(SynQbfError::CircuitEvaluation)
}

/// Verify all three SynQBF conditions against an original CNF and ordered
/// prefix. This checker does not trust the proof's claimed value.
pub fn verify_synqbf_proof(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    proof: &SynQbfProof,
    budget: SynQbfBudget,
) -> Result<SynQbfVerificationReport, SynQbfError> {
    let order = validate_source(problem, prefix, budget)?;
    if proof.variable_count != problem.variables.len() {
        return Err(SynQbfError::ProofVariableCountMismatch {
            proof: proof.variable_count,
            source: problem.variables.len(),
        });
    }
    if proof.prefix_order != order {
        return Err(SynQbfError::PrefixBindingMismatch);
    }

    let n = order.len();
    let expected_inputs = n
        .checked_mul(2)
        .ok_or(SynQbfError::ResourceAccountingOverflow)?;
    if proof.circuit.dependencies.len() != expected_inputs {
        return Err(SynQbfError::CircuitDependencyCountMismatch {
            expected: expected_inputs,
            actual: proof.circuit.dependencies.len(),
        });
    }
    for (position, &actual) in proof.circuit.dependencies.iter().enumerate() {
        if actual != position {
            return Err(SynQbfError::CircuitDependencyOrderMismatch {
                position,
                expected: position,
                actual,
            });
        }
    }
    validate_strategy_aig(&proof.circuit, budget.aig).map_err(SynQbfError::CircuitValidation)?;

    let source_assignments = checked_power_of_two(n)?;
    let representability_instances = source_assignments
        .checked_sub(1)
        .ok_or(SynQbfError::ResourceAccountingOverflow)?;
    let expected_evaluations = source_assignments
        .checked_add(
            representability_instances
                .checked_mul(3)
                .ok_or(SynQbfError::ResourceAccountingOverflow)?,
        )
        .and_then(|value| value.checked_add(1))
        .ok_or(SynQbfError::ResourceAccountingOverflow)?;
    if expected_evaluations > budget.max_circuit_evaluations {
        return Err(SynQbfError::CircuitEvaluationBudgetExceeded {
            required: expected_evaluations,
            limit: budget.max_circuit_evaluations,
        });
    }

    // Each evaluation includes structural validation, edge/reachability
    // checks, and straight-line AIG evaluation. This conservative bound is
    // reported as structural work, not as empirical runtime.
    let node_count = u128::try_from(proof.circuit.nodes.len())
        .map_err(|_| SynQbfError::ResourceAccountingOverflow)?;
    let estimated_node_visits = expected_evaluations
        .checked_mul(
            node_count
                .checked_mul(8)
                .and_then(|value| value.checked_add(1))
                .ok_or(SynQbfError::ResourceAccountingOverflow)?,
        )
        .ok_or(SynQbfError::ResourceAccountingOverflow)?;
    if estimated_node_visits > budget.max_estimated_aig_node_visits {
        return Err(SynQbfError::EstimatedAigNodeVisitBudgetExceeded {
            required: estimated_node_visits,
            limit: budget.max_estimated_aig_node_visits,
        });
    }

    let mut report = SynQbfVerificationReport::default();

    // Condition 1: phi(x) iff phihat(x, xhat := not x).
    for mask in 0..source_assignments {
        let prefix_values: Vec<bool> = (0..n)
            .map(|position| ((mask >> position) & 1) != 0)
            .collect();
        let mut source_assignment = vec![false; problem.variables.len()];
        for (position, &(_, variable)) in order.iter().enumerate() {
            source_assignment[variable] = prefix_values[position];
        }
        let source_value = source_matrix_value(
            problem,
            &source_assignment,
            &mut report.source_matrix_literal_checks,
        )?;
        let circuit_value = evaluate_counted(
            proof,
            &paired_assignment(&prefix_values),
            budget,
            &mut report,
        )?;
        if source_value != circuit_value {
            return Err(SynQbfError::SourceEquivalenceCounterexample {
                prefix_assignment: prefix_values,
                source_value,
                circuit_value,
            });
        }
        report.source_equivalence_assignments_checked = report
            .source_equivalence_assignments_checked
            .checked_add(1)
            .ok_or(SynQbfError::ResourceAccountingOverflow)?;
    }

    // Condition 2: quantifier-specific representability at every prefix
    // position and every assignment to the earlier original variables.
    for position in 0..n {
        let history_rows = checked_power_of_two(position)?;
        let quantifier = order[position].0;
        let bit = matches!(quantifier, Quantifier::Exists);
        for mask in 0..history_rows {
            let mut base = vec![false; expected_inputs];
            for earlier in 0..position {
                let value = ((mask >> earlier) & 1) != 0;
                base[earlier] = value;
                base[n + earlier] = !value;
            }
            for later in (position + 1)..n {
                let later_bit = matches!(order[later].0, Quantifier::Exists);
                base[later] = later_bit;
                base[n + later] = later_bit;
            }

            let mut diagonal = base.clone();
            diagonal[position] = bit;
            diagonal[n + position] = bit;

            let mut branch_01 = base.clone();
            branch_01[position] = false;
            branch_01[n + position] = true;

            let mut branch_10 = base;
            branch_10[position] = true;
            branch_10[n + position] = false;

            let diagonal_value = evaluate_counted(proof, &diagonal, budget, &mut report)?;
            let branch_01_value = evaluate_counted(proof, &branch_01, budget, &mut report)?;
            let branch_10_value = evaluate_counted(proof, &branch_10, budget, &mut report)?;
            let required_value = match quantifier {
                Quantifier::Exists => branch_01_value || branch_10_value,
                Quantifier::ForAll => branch_01_value && branch_10_value,
            };
            if diagonal_value != required_value {
                let earlier_assignment: Vec<bool> = (0..position)
                    .map(|index| ((mask >> index) & 1) != 0)
                    .collect();
                return Err(SynQbfError::RepresentabilityViolation {
                    prefix_position: position,
                    earlier_assignment,
                    diagonal_value,
                    branch_01_value,
                    branch_10_value,
                    required_value,
                });
            }
            report.representability_conditions_checked = report
                .representability_conditions_checked
                .checked_add(1)
                .ok_or(SynQbfError::ResourceAccountingOverflow)?;
        }
    }

    // Condition 3: phihat(x_i = xhat_i = b_i) equals the claimed truth value.
    let mut root_assignment = vec![false; expected_inputs];
    for (position, (quantifier, _)) in order.iter().enumerate() {
        let bit = matches!(quantifier, Quantifier::Exists);
        root_assignment[position] = bit;
        root_assignment[n + position] = bit;
    }
    let root_value = evaluate_counted(proof, &root_assignment, budget, &mut report)?;
    if root_value != proof.claimed_truth_value {
        return Err(SynQbfError::ClaimedTruthValueMismatch {
            claimed: proof.claimed_truth_value,
            circuit_value: root_value,
        });
    }

    report.representability_instances_checked = representability_instances;
    report.root_value_checks = 1;
    report.estimated_aig_node_visits = estimated_node_visits;
    report.claimed_truth_value = root_value;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::super::decomposition::Literal;
    use super::super::quantified_scope::QuantifierBlock;
    use super::super::quantified_strategy::BooleanStrategyFunction;
    use super::super::strategy_aig::compile_strategy_function_to_aig;
    use super::*;
    use std::collections::BTreeMap;

    fn block(quantifier: Quantifier, variables: &[usize]) -> QuantifierBlock {
        QuantifierBlock {
            quantifier,
            variables: variables.to_vec(),
        }
    }

    fn prefix(variable_count: usize, blocks: Vec<QuantifierBlock>) -> ValidatedQuantifierPrefix {
        ValidatedQuantifierPrefix::new(variable_count, blocks, Default::default()).unwrap()
    }

    fn aig_budget() -> StrategyAigBudget {
        StrategyAigBudget {
            max_dependency_variables: 12,
            max_truth_table_entries: 1 << 12,
            max_and_nodes: 1 << 12,
            max_assignments_checked: 1 << 12,
        }
    }

    fn budget() -> SynQbfBudget {
        SynQbfBudget {
            max_variables: 6,
            max_total_variable_name_bytes: 256,
            max_clauses: 64,
            max_literals: 256,
            max_circuit_evaluations: 1 << 14,
            max_estimated_aig_node_visits: 1 << 22,
            aig: aig_budget(),
        }
    }

    fn proof_from_values(
        variable_count: usize,
        prefix_order: Vec<(Quantifier, usize)>,
        values: Vec<bool>,
        claimed_truth_value: bool,
    ) -> SynQbfProof {
        let dependencies: Vec<usize> = (0..(2 * variable_count)).collect();
        let table = (0..values.len())
            .map(|mask| {
                let key = (0..dependencies.len())
                    .map(|index| ((mask >> index) & 1) != 0)
                    .collect::<Vec<_>>();
                (key, values[mask])
            })
            .collect::<BTreeMap<_, _>>();
        let function = BooleanStrategyFunction {
            variable: usize::MAX,
            dependencies: dependencies.clone(),
            table,
        };
        let circuit = compile_strategy_function_to_aig(&function, aig_budget()).unwrap();
        SynQbfProof {
            variable_count,
            prefix_order,
            claimed_truth_value,
            circuit,
        }
    }

    #[test]
    fn verifies_simple_existential_and_universal_synqbf_proofs() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![Literal::positive(0)]],
        };

        for (quantifier, expected) in [(Quantifier::Exists, true), (Quantifier::ForAll, false)] {
            let prefix = prefix(1, vec![block(quantifier, &[0])]);
            // phihat(x, xhat) = x.
            let proof = proof_from_values(
                1,
                vec![(quantifier, 0)],
                vec![false, false, true, true],
                expected,
            );
            let report = verify_synqbf_proof(&problem, &prefix, &proof, budget()).unwrap();
            assert_eq!(report.source_equivalence_assignments_checked, 2);
            assert_eq!(report.representability_instances_checked, 1);
            assert_eq!(report.representability_conditions_checked, 1);
            assert_eq!(report.root_value_checks, 1);
        }
    }

    #[test]
    fn verifies_sat_2026_worked_synqbf_proof() {
        // SAT 2026 Example 4.4 uses:
        // forall x1 exists x2 forall x3:
        // ((x1 AND !x3) OR ((x2 OR x3) AND (!x2 OR !x1))).
        // Its equivalent CNF is (x1 OR x2 OR x3) AND
        // (!x1 OR !x2 OR !x3).
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
        let quantifier_order = vec![
            (Quantifier::ForAll, 0),
            (Quantifier::Exists, 1),
            (Quantifier::ForAll, 2),
        ];
        let prefix = prefix(
            3,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
                block(Quantifier::ForAll, &[2]),
            ],
        );

        // phihat = (x1 AND xhat3)
        //          OR ((x2 OR x3) AND (xhat2 OR xhat1))
        //          OR (x1 AND xhat2).
        let mut values = vec![false; 1 << 6];
        for mask in 0..(1 << 6) {
            let bit = |index: usize| ((mask >> index) & 1) != 0;
            let x1 = bit(0);
            let x2 = bit(1);
            let x3 = bit(2);
            let xhat1 = bit(3);
            let xhat2 = bit(4);
            let xhat3 = bit(5);
            values[mask] = (x1 && xhat3) || ((x2 || x3) && (xhat2 || xhat1)) || (x1 && xhat2);
        }
        let proof = proof_from_values(3, quantifier_order, values, true);
        let report = verify_synqbf_proof(&problem, &prefix, &proof, budget()).unwrap();
        assert_eq!(report.source_equivalence_assignments_checked, 8);
        assert_eq!(report.representability_instances_checked, 7);
        assert_eq!(report.representability_conditions_checked, 7);
        assert_eq!(report.root_value_checks, 1);
        assert!(report.circuit_evaluations > 8);
    }

    #[test]
    fn rejects_circuit_that_does_not_represent_source_matrix() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![Literal::positive(0)]],
        };
        let prefix = prefix(1, vec![block(Quantifier::Exists, &[0])]);
        // Constant false disagrees with phi(x)=x when x=1.
        let proof = proof_from_values(1, vec![(Quantifier::Exists, 0)], vec![false; 4], true);
        assert!(matches!(
            verify_synqbf_proof(&problem, &prefix, &proof, budget()),
            Err(SynQbfError::SourceEquivalenceCounterexample { .. })
        ));
    }

    #[test]
    fn rejects_representability_violation_even_when_source_equivalence_holds() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![Literal::positive(0)]],
        };
        let prefix = prefix(1, vec![block(Quantifier::Exists, &[0])]);
        // Under xhat=!x the table agrees with phi=x, but diagonal != OR.
        let proof = proof_from_values(
            1,
            vec![(Quantifier::Exists, 0)],
            vec![false, false, true, false],
            false,
        );
        assert!(matches!(
            verify_synqbf_proof(&problem, &prefix, &proof, budget()),
            Err(SynQbfError::RepresentabilityViolation { .. })
        ));
    }

    #[test]
    fn rejects_root_value_mismatch_after_other_conditions_pass() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![Literal::positive(0)]],
        };
        let prefix = prefix(1, vec![block(Quantifier::Exists, &[0])]);
        let proof = proof_from_values(
            1,
            vec![(Quantifier::Exists, 0)],
            vec![false, false, true, true],
            false,
        );
        assert!(matches!(
            verify_synqbf_proof(&problem, &prefix, &proof, budget()),
            Err(SynQbfError::ClaimedTruthValueMismatch { .. })
        ));
    }

    #[test]
    fn rejects_wrong_paired_input_order() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![Literal::positive(0)]],
        };
        let prefix = prefix(1, vec![block(Quantifier::Exists, &[0])]);
        let mut proof = proof_from_values(
            1,
            vec![(Quantifier::Exists, 0)],
            vec![false, false, true, true],
            true,
        );
        proof.circuit.dependencies = vec![1, 0];
        assert!(matches!(
            verify_synqbf_proof(&problem, &prefix, &proof, budget()),
            Err(SynQbfError::CircuitDependencyOrderMismatch { .. })
        ));
    }

    #[test]
    fn circuit_evaluation_budget_is_fail_closed() {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![Literal::positive(0)]],
        };
        let prefix = prefix(1, vec![block(Quantifier::Exists, &[0])]);
        let proof = proof_from_values(
            1,
            vec![(Quantifier::Exists, 0)],
            vec![false, false, true, true],
            true,
        );
        let tight = SynQbfBudget {
            max_circuit_evaluations: 3,
            ..budget()
        };
        assert!(matches!(
            verify_synqbf_proof(&problem, &prefix, &proof, tight),
            Err(SynQbfError::CircuitEvaluationBudgetExceeded { .. })
        ));
    }
}
