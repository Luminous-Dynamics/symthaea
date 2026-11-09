// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Verified dual-representation portfolio for full-history QBF functions.
//!
//! This compiles every full-history Skolem and Herbrand function from a
//! verified game-value profile into both a ROBDD and an AIG. Semantic support
//! is minimized only after checking equivalence to the original full-history
//! table. Both circuit formats are then exhaustively checked against the
//! reduced table. The source game-value profile is checked against the
//! original CNF/prefix before any circuit is emitted.

use super::quantified_game_values::{
    FullHistoryStrategyFunction, GameValueProfileBudget, GameValueProfileError,
    GameValueProfileVerificationReport, QuantifiedGameValueProfile,
    verify_game_value_profile,
};
use super::quantified_scope::{Quantifier, ValidatedQuantifierPrefix};
use super::quantified_strategy::BooleanStrategyFunction;
use super::strategy_aig::{
    StrategyAig, StrategyAigBudget, StrategyAigError, StrategyAigReport,
    compile_strategy_function_to_aig_with_order, verify_aig_matches_strategy_function,
};
use super::strategy_bdd::{
    ReducedOrderedBdd, StrategyBddBudget, StrategyBddError,
    StrategyBddExactOrderSearchReport, StrategyBddMultiStartReport,
    StrategyFunctionSupportReduction, StrategySupportReductionReport,
    find_exact_minimum_robdd_order, minimize_strategy_function_support,
    multi_start_sift_strategy_function_robdd,
    verify_robdd_matches_strategy_function,
};
use super::decomposition::CnfProblem;

/// How the portfolio selected a ROBDD order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FullHistoryBddSearchMethod {
    ExactPermutationSearch,
    MultiStartFrontierSifting,
}

/// Circuit representations for one full-history quantified function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullHistoryFunctionPortfolioEntry {
    pub prefix_position: usize,
    pub variable: usize,
    pub quantifier: Quantifier,
    pub original_dependencies: Vec<usize>,
    pub reduced_function: BooleanStrategyFunction,
    pub support_reduction: StrategySupportReductionReport,
    pub bdd: ReducedOrderedBdd,
    pub aig: StrategyAig,
    pub bdd_search_method: FullHistoryBddSearchMethod,
    pub bdd_order_evaluations: usize,
    pub bdd_search_complete: bool,
    pub bdd_nodes: usize,
    pub aig_and_nodes: usize,
    pub aig_complemented_edges: usize,
    pub truth_table_rows_checked: u128,
}

/// Limits for one complete full-history circuit portfolio.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FullHistoryPortfolioBudget {
    pub game_values: GameValueProfileBudget,
    pub bdd: StrategyBddBudget,
    pub aig: StrategyAigBudget,
    pub max_total_bdd_nodes: usize,
    pub max_total_aig_and_nodes: usize,
    pub max_total_support_pair_checks: u128,
}

/// Typed failures. Budget exhaustion is never treated as a successful
/// partial portfolio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FullHistoryPortfolioError {
    GameValueProfile(GameValueProfileError),
    StrategyFunctionMissing { variable: usize, quantifier: Quantifier },
    StrategyFunctionScopeMismatch { variable: usize },
    BddOptimization(StrategyBddError),
    BddVerification(StrategyBddError),
    AigCompilation(StrategyAigError),
    AigVerification(StrategyAigError),
    TotalBddNodeBudgetExceeded { required: usize, limit: usize },
    TotalAigNodeBudgetExceeded { required: usize, limit: usize },
    SupportPairCheckBudgetExceeded { required: u128, limit: u128 },
    ResourceAccountingOverflow,
}

/// Aggregate cost and correctness evidence for all full-history functions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullHistoryPortfolioReport {
    pub game_value_verification: GameValueProfileVerificationReport,
    pub entries: Vec<FullHistoryFunctionPortfolioEntry>,
    pub function_count: usize,
    pub original_dependency_count: usize,
    pub reduced_dependency_count: usize,
    pub original_truth_table_entries: u128,
    pub reduced_truth_table_entries: u128,
    pub total_support_pair_checks: u128,
    pub total_bdd_nodes: usize,
    pub total_aig_and_nodes: usize,
    pub total_bdd_order_evaluations: usize,
    pub total_aig_assignments_checked: u128,
}

/// The verified profile's complete function portfolio.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FullHistoryStrategyPortfolio {
    pub variable_count: usize,
    pub prefix_order: Vec<(Quantifier, usize)>,
    pub truth_value: bool,
    pub report: FullHistoryPortfolioReport,
}

fn exact_order_budget_fits(dependency_count: usize, limit: usize) -> bool {
    let mut factorial = 1usize;
    for factor in 2..=dependency_count {
        let Some(next) = factorial.checked_mul(factor) else {
            return false;
        };
        if next > limit {
            return false;
        }
        factorial = next;
    }
    factorial <= limit
}

fn full_history_function<'a>(
    profile: &'a QuantifiedGameValueProfile,
    variable: usize,
    quantifier: Quantifier,
) -> Option<&'a FullHistoryStrategyFunction> {
    let functions = match quantifier {
        Quantifier::Exists => &profile.skolem_functions,
        Quantifier::ForAll => &profile.herbrand_functions,
    };
    functions
        .iter()
        .find(|function| function.variable == variable && function.quantifier == quantifier)
}

/// Compile all full-history functions into support-minimized ROBDD and AIG
/// representations, then independently validate each representation.
///
/// The ROBDD compiler uses exact permutation search when n! fits the declared
/// order budget and otherwise uses the interaction/frontier multi-start
/// heuristic. The chosen ROBDD decision order is reused as the AIG Shannon
/// decomposition order, allowing a controlled comparison on the same order.
pub fn compile_full_history_strategy_portfolio(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    profile: &QuantifiedGameValueProfile,
    budget: FullHistoryPortfolioBudget,
) -> Result<FullHistoryStrategyPortfolio, FullHistoryPortfolioError> {
    let game_value_verification =
        verify_game_value_profile(problem, prefix, profile, budget.game_values)
            .map_err(FullHistoryPortfolioError::GameValueProfile)?;

    let mut entries = Vec::with_capacity(profile.prefix_order.len());
    let mut original_dependency_count = 0usize;
    let mut reduced_dependency_count = 0usize;
    let mut original_truth_table_entries = 0u128;
    let mut reduced_truth_table_entries = 0u128;
    let mut total_support_pair_checks = 0u128;
    let mut total_bdd_nodes = 0usize;
    let mut total_aig_and_nodes = 0usize;
    let mut total_bdd_order_evaluations = 0usize;
    let mut total_aig_assignments_checked = 0u128;

    for (position, &(quantifier, variable)) in profile.prefix_order.iter().enumerate() {
        let source_function = full_history_function(profile, variable, quantifier).ok_or(
            FullHistoryPortfolioError::StrategyFunctionMissing {
                variable,
                quantifier,
            },
        )?;
        let expected_dependencies: Vec<usize> = profile.prefix_order[..position]
            .iter()
            .map(|(_, previous_variable)| *previous_variable)
            .collect();
        if source_function.dependencies != expected_dependencies {
            return Err(FullHistoryPortfolioError::StrategyFunctionScopeMismatch {
                variable,
            });
        }

        let source = BooleanStrategyFunction {
            variable,
            dependencies: source_function.dependencies.clone(),
            table: source_function.table.clone(),
        };
        let support = minimize_strategy_function_support(&source, budget.bdd)
            .map_err(FullHistoryPortfolioError::BddOptimization)?;
        // Recheck support equivalence at the portfolio boundary rather than
        // trusting the minimizer's internal report alone.
        super::strategy_bdd::verify_support_reduction_equivalence(
            &source,
            &support.function,
            budget.bdd,
        )
        .map_err(FullHistoryPortfolioError::BddVerification)?;

        let (bdd, bdd_search_method, bdd_order_evaluations, bdd_search_complete) =
            if exact_order_budget_fits(
                support.function.dependencies.len(),
                budget.bdd.max_variable_order_evaluations,
            ) {
                let exact = find_exact_minimum_robdd_order(&support.function, budget.bdd)
                    .map_err(FullHistoryPortfolioError::BddOptimization)?;
                (
                    exact.bdd,
                    FullHistoryBddSearchMethod::ExactPermutationSearch,
                    exact.orders_evaluated,
                    true,
                )
            } else {
                let heuristic =
                    multi_start_sift_strategy_function_robdd(&support.function, budget.bdd)
                        .map_err(FullHistoryPortfolioError::BddOptimization)?;
                (
                    heuristic.bdd,
                    FullHistoryBddSearchMethod::MultiStartFrontierSifting,
                    heuristic.order_evaluations,
                    heuristic.search_complete,
                )
            };

        let bdd_report = verify_robdd_matches_strategy_function(
            &bdd,
            &support.function,
            budget.bdd,
        )
        .map_err(FullHistoryPortfolioError::BddVerification)?;

        let aig = compile_strategy_function_to_aig_with_order(
            &support.function,
            &bdd.variable_order,
            budget.aig,
        )
        .map_err(FullHistoryPortfolioError::AigCompilation)?;
        let aig_report = verify_aig_matches_strategy_function(
            &aig,
            &support.function,
            budget.aig,
        )
        .map_err(FullHistoryPortfolioError::AigVerification)?;

        original_dependency_count = original_dependency_count
            .checked_add(source.dependencies.len())
            .ok_or(FullHistoryPortfolioError::ResourceAccountingOverflow)?;
        reduced_dependency_count = reduced_dependency_count
            .checked_add(support.function.dependencies.len())
            .ok_or(FullHistoryPortfolioError::ResourceAccountingOverflow)?;
        original_truth_table_entries = original_truth_table_entries
            .checked_add(
                u128::try_from(source.table.len())
                    .map_err(|_| FullHistoryPortfolioError::ResourceAccountingOverflow)?,
            )
            .ok_or(FullHistoryPortfolioError::ResourceAccountingOverflow)?;
        reduced_truth_table_entries = reduced_truth_table_entries
            .checked_add(
                u128::try_from(support.function.table.len())
                    .map_err(|_| FullHistoryPortfolioError::ResourceAccountingOverflow)?,
            )
            .ok_or(FullHistoryPortfolioError::ResourceAccountingOverflow)?;
        total_support_pair_checks = total_support_pair_checks
            .checked_add(support.report.support_pair_checks)
            .ok_or(FullHistoryPortfolioError::ResourceAccountingOverflow)?;
        total_bdd_nodes = total_bdd_nodes
            .checked_add(bdd_report.bdd_nodes)
            .ok_or(FullHistoryPortfolioError::ResourceAccountingOverflow)?;
        total_aig_and_nodes = total_aig_and_nodes
            .checked_add(aig_report.and_nodes)
            .ok_or(FullHistoryPortfolioError::ResourceAccountingOverflow)?;
        total_bdd_order_evaluations = total_bdd_order_evaluations
            .checked_add(bdd_order_evaluations)
            .ok_or(FullHistoryPortfolioError::ResourceAccountingOverflow)?;
        total_aig_assignments_checked = total_aig_assignments_checked
            .checked_add(aig_report.assignments_checked)
            .ok_or(FullHistoryPortfolioError::ResourceAccountingOverflow)?;

        if total_support_pair_checks > budget.max_total_support_pair_checks {
            return Err(FullHistoryPortfolioError::SupportPairCheckBudgetExceeded {
                required: total_support_pair_checks,
                limit: budget.max_total_support_pair_checks,
            });
        }
        if total_bdd_nodes > budget.max_total_bdd_nodes {
            return Err(FullHistoryPortfolioError::TotalBddNodeBudgetExceeded {
                required: total_bdd_nodes,
                limit: budget.max_total_bdd_nodes,
            });
        }
        if total_aig_and_nodes > budget.max_total_aig_and_nodes {
            return Err(FullHistoryPortfolioError::TotalAigNodeBudgetExceeded {
                required: total_aig_and_nodes,
                limit: budget.max_total_aig_and_nodes,
            });
        }

        entries.push(FullHistoryFunctionPortfolioEntry {
            prefix_position: position,
            variable,
            quantifier,
            original_dependencies: source.dependencies,
            reduced_function: support.function,
            support_reduction: support.report,
            bdd_nodes: bdd_report.bdd_nodes,
            aig_and_nodes: aig_report.and_nodes,
            aig_complemented_edges: aig_report.complemented_edges,
            truth_table_rows_checked: bdd_report.assignments_checked,
            bdd,
            aig,
            bdd_search_method,
            bdd_order_evaluations,
            bdd_search_complete,
        });
    }

    Ok(FullHistoryStrategyPortfolio {
        variable_count: profile.variable_count,
        prefix_order: profile.prefix_order.clone(),
        truth_value: profile.truth_value,
        report: FullHistoryPortfolioReport {
            game_value_verification,
            function_count: entries.len(),
            entries,
            original_dependency_count,
            reduced_dependency_count,
            original_truth_table_entries,
            reduced_truth_table_entries,
            total_support_pair_checks,
            total_bdd_nodes,
            total_aig_and_nodes,
            total_bdd_order_evaluations,
            total_aig_assignments_checked,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::decomposition::Literal;
    use super::super::quantified_game_dag::{QbfGameDagBudget, build_qbf_game_proof_dag};
    use super::super::quantified_game_values::extract_game_value_profile_from_dag;
    use super::super::quantified_scope::QuantifierBlock;

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

    fn portfolio_budget() -> FullHistoryPortfolioBudget {
        FullHistoryPortfolioBudget {
            game_values: GameValueProfileBudget {
                max_variables: 8,
                max_total_variable_name_bytes: 256,
                max_clauses: 128,
                max_literals: 512,
                max_total_truth_table_entries: 1 << 12,
                max_dag_path_steps: 1 << 14,
                max_matrix_literal_checks: 1 << 14,
            },
            bdd: StrategyBddBudget {
                max_dependency_variables: 8,
                max_table_entries: 1 << 8,
                max_bdd_nodes: 1 << 12,
                max_support_checks: 1 << 16,
                max_interaction_checks: 1 << 16,
                max_weighted_interaction_checks: 1 << 16,
                max_anf_transform_checks: 1 << 16,
                max_frontier_order_evaluations: 1 << 12,
                max_frontier_refinement_passes: 2,
                max_variable_order_evaluations: 1 << 12,
                max_sifting_passes: 2,
            },
            aig: StrategyAigBudget {
                max_dependency_variables: 8,
                max_truth_table_entries: 1 << 8,
                max_and_nodes: 1 << 12,
                max_assignments_checked: 1 << 8,
            },
            max_total_bdd_nodes: 1 << 14,
            max_total_aig_and_nodes: 1 << 14,
            max_total_support_pair_checks: 1 << 20,
        }
    }

    fn equality_problem() -> (CnfProblem, ValidatedQuantifierPrefix) {
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![
                vec![Literal::negative(0), Literal::positive(1)],
                vec![Literal::positive(0), Literal::negative(1)],
            ],
        };
        let prefix = prefix(
            2,
            vec![
                block(Quantifier::ForAll, &[0]),
                block(Quantifier::Exists, &[1]),
            ],
        );
        (problem, prefix)
    }

    #[test]
    fn compiles_both_full_history_function_families_to_bdd_and_aig() {
        let (problem, prefix) = equality_problem();
        let dag_budget = dag_budget();
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget).unwrap();
        let profile = extract_game_value_profile_from_dag(
            &problem,
            &prefix,
            &dag,
            dag_budget,
            portfolio_budget().game_values,
        )
        .unwrap();
        let portfolio =
            compile_full_history_strategy_portfolio(&problem, &prefix, &profile, portfolio_budget())
                .unwrap();

        assert_eq!(portfolio.report.function_count, 2);
        assert_eq!(portfolio.report.entries[0].quantifier, Quantifier::ForAll);
        assert_eq!(portfolio.report.entries[1].quantifier, Quantifier::Exists);
        assert_eq!(portfolio.report.entries[0].original_dependencies, vec![]);
        assert_eq!(portfolio.report.entries[1].original_dependencies, vec![0]);
        assert_eq!(portfolio.report.entries[0].bdd_nodes, 0);
        assert_eq!(portfolio.report.entries[0].aig_and_nodes, 0);
        assert_eq!(portfolio.report.entries[1].bdd_nodes, 1);
        assert_eq!(portfolio.report.entries[1].aig_and_nodes, 0);
        assert_eq!(portfolio.report.game_value_verification.truth_value, true);
    }

    #[test]
    fn portfolio_rejects_tampered_game_value_profile_before_compilation() {
        let (problem, prefix) = equality_problem();
        let dag_budget = dag_budget();
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget).unwrap();
        let mut profile = extract_game_value_profile_from_dag(
            &problem,
            &prefix,
            &dag,
            dag_budget,
            portfolio_budget().game_values,
        )
        .unwrap();
        profile.layers[1].values.insert(vec![false], !profile.layers[1].values[&vec![false]]);
        assert!(matches!(
            compile_full_history_strategy_portfolio(&problem, &prefix, &profile, portfolio_budget()),
            Err(FullHistoryPortfolioError::GameValueProfile(_))
        ));
    }

    #[test]
    fn total_bdd_node_budget_is_fail_closed() {
        let (problem, prefix) = equality_problem();
        let dag_budget = dag_budget();
        let (dag, _) = build_qbf_game_proof_dag(&problem, &prefix, dag_budget).unwrap();
        let profile = extract_game_value_profile_from_dag(
            &problem,
            &prefix,
            &dag,
            dag_budget,
            portfolio_budget().game_values,
        )
        .unwrap();
        let tight = FullHistoryPortfolioBudget {
            max_total_bdd_nodes: 0,
            ..portfolio_budget()
        };
        assert!(matches!(
            compile_full_history_strategy_portfolio(&problem, &prefix, &profile, tight),
            Err(FullHistoryPortfolioError::TotalBddNodeBudgetExceeded { .. })
        ));
    }
}
