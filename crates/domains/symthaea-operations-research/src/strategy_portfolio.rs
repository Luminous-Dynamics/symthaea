// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Verified portfolio compilation for Boolean strategy functions.
//!
//! The original QBF strategy is validated first against the source matrix and
//! ordered prefix. Each support-minimized function is then compiled to both a
//! ROBDD and an AIG. Each representation is independently checked against the
//! same reduced truth table and encoded using a versioned canonical byte
//! format. Selection compares byte lengths, not incomparable node counts.
//!
//! A compiled circuit is not itself a QBF proof. The source strategy checker
//! remains the authority for whether the Skolem/Herbrand functions win.

use super::decomposition::CnfProblem;
use super::quantified_scope::ValidatedQuantifierPrefix;
use super::quantified_strategy::{
    QbfEvaluationResult, QuantifiedEvaluationBudget, QuantifiedStrategyError,
    StrategyVerificationReport,
};
use super::strategy_aig::{
    AigLiteral, AigNode, StrategyAig, StrategyAigBudget, StrategyAigError,
    StrategyAigReport, compile_strategy_function_to_aig_with_order,
    verify_aig_matches_strategy_function,
};
use super::strategy_bdd::{
    BddRef, CompiledStrategyFunction, StrategyBddBudget, StrategyBddError,
    StrategyBddReport, VerifiedStrategyCompilationError, verify_and_compile_qbf_strategy,
    verify_robdd_matches_strategy_function,
};
use std::convert::TryFrom;

/// Representation selected for one strategy function by canonical byte size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedStrategyRepresentation {
    Robdd,
    Aig,
}

/// Explicit limits on canonical artifact encoding and aggregate output size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrategyPortfolioBudget {
    pub max_single_representation_bytes: usize,
    pub max_total_representation_bytes: usize,
}

impl Default for StrategyPortfolioBudget {
    fn default() -> Self {
        Self {
            max_single_representation_bytes: 1 << 26,
            max_total_representation_bytes: 1 << 28,
        }
    }
}

/// Failures at the source-verification, representation, or encoding boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StrategyPortfolioError {
    SourceOrBddCompilation(VerifiedStrategyCompilationError),
    BddVerification(StrategyBddError),
    AigCompilation(StrategyAigError),
    AigVerification(StrategyAigError),
    RepresentationTooLarge { required: usize, limit: usize },
    TotalPortfolioTooLarge { required: usize, limit: usize },
    EncodingIndexOverflow { value: usize },
    ResourceAccountingOverflow,
}

/// Both checked circuit representations for one support-minimized function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StrategyFunctionPortfolio {
    pub variable: usize,
    pub original_dependencies: Vec<usize>,
    pub reduced_dependencies: Vec<usize>,
    pub bdd: super::strategy_bdd::ReducedOrderedBdd,
    pub aig: StrategyAig,
    pub bdd_verification: StrategyBddReport,
    pub aig_verification: StrategyAigReport,
    pub bdd_bytes: Vec<u8>,
    pub aig_bytes: Vec<u8>,
    pub selected: SelectedStrategyRepresentation,
    pub selected_bytes: usize,
}

/// A source-verified winning strategy with two equivalent checked encodings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedStrategyPortfolio {
    pub value: bool,
    pub source_strategy_verification: StrategyVerificationReport,
    pub functions: Vec<StrategyFunctionPortfolio>,
    pub total_bdd_bytes: usize,
    pub total_aig_bytes: usize,
    pub total_selected_bytes: usize,
    pub total_bdd_nodes: usize,
    pub total_aig_and_nodes: usize,
    pub bdd_search_complete: bool,
}

fn ensure_append(
    bytes: &[u8],
    additional: usize,
    budget: StrategyPortfolioBudget,
) -> Result<(), StrategyPortfolioError> {
    let required = bytes
        .len()
        .checked_add(additional)
        .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
    if required > budget.max_single_representation_bytes {
        return Err(StrategyPortfolioError::RepresentationTooLarge {
            required,
            limit: budget.max_single_representation_bytes,
        });
    }
    Ok(())
}

fn append_byte(
    bytes: &mut Vec<u8>,
    value: u8,
    budget: StrategyPortfolioBudget,
) -> Result<(), StrategyPortfolioError> {
    ensure_append(bytes, 1, budget)?;
    bytes.push(value);
    Ok(())
}

fn append_u64(
    bytes: &mut Vec<u8>,
    value: usize,
    budget: StrategyPortfolioBudget,
) -> Result<(), StrategyPortfolioError> {
    let value = u64::try_from(value)
        .map_err(|_| StrategyPortfolioError::EncodingIndexOverflow { value })?;
    ensure_append(bytes, 8, budget)?;
    bytes.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn append_dependencies(
    bytes: &mut Vec<u8>,
    dependencies: &[usize],
    budget: StrategyPortfolioBudget,
) -> Result<(), StrategyPortfolioError> {
    append_u64(bytes, dependencies.len(), budget)?;
    for &dependency in dependencies {
        append_u64(bytes, dependency, budget)?;
    }
    Ok(())
}

fn append_aig_literal(
    bytes: &mut Vec<u8>,
    literal: AigLiteral,
    budget: StrategyPortfolioBudget,
) -> Result<(), StrategyPortfolioError> {
    match literal {
        AigLiteral::Constant(value) => {
            append_byte(bytes, 0, budget)?;
            append_byte(bytes, u8::from(value), budget)?;
        }
        AigLiteral::Node { index, inverted } => {
            append_byte(bytes, 1, budget)?;
            append_u64(bytes, index, budget)?;
            append_byte(bytes, u8::from(inverted), budget)?;
        }
    }
    Ok(())
}

/// Encode an AIG in the canonical LD-AIG-1 little-endian format.
pub fn encode_strategy_aig(
    aig: &StrategyAig,
    aig_budget: StrategyAigBudget,
    encoding_budget: StrategyPortfolioBudget,
) -> Result<Vec<u8>, StrategyPortfolioError> {
    super::strategy_aig::validate_strategy_aig(aig, aig_budget)
        .map_err(StrategyPortfolioError::AigVerification)?;
    let mut bytes = Vec::new();
    for byte in b"LD-AIG-1" {
        append_byte(&mut bytes, *byte, encoding_budget)?;
    }
    append_dependencies(&mut bytes, &aig.dependencies, encoding_budget)?;
    append_dependencies(&mut bytes, &aig.decomposition_order, encoding_budget)?;
    append_u64(&mut bytes, aig.nodes.len(), encoding_budget)?;

    for node in &aig.nodes {
        match *node {
            AigNode::Input { dependency_index } => {
                append_byte(&mut bytes, 0, encoding_budget)?;
                append_u64(&mut bytes, dependency_index, encoding_budget)?;
            }
            AigNode::And { left, right } => {
                append_byte(&mut bytes, 1, encoding_budget)?;
                append_aig_literal(&mut bytes, left, encoding_budget)?;
                append_aig_literal(&mut bytes, right, encoding_budget)?;
            }
        }
    }
    append_aig_literal(&mut bytes, aig.output, encoding_budget)?;
    Ok(bytes)
}

fn append_bdd_ref(
    bytes: &mut Vec<u8>,
    reference: BddRef,
    budget: StrategyPortfolioBudget,
) -> Result<(), StrategyPortfolioError> {
    match reference {
        BddRef::False => append_byte(bytes, 0, budget)?,
        BddRef::True => append_byte(bytes, 1, budget)?,
        BddRef::Node(index) => {
            append_byte(bytes, 2, budget)?;
            append_u64(bytes, index, budget)?;
        }
    }
    Ok(())
}

/// Encode an ROBDD in the canonical LD-ROBDD-1 little-endian format.
pub fn encode_strategy_robdd(
    bdd: &super::strategy_bdd::ReducedOrderedBdd,
    bdd_budget: StrategyBddBudget,
    encoding_budget: StrategyPortfolioBudget,
) -> Result<Vec<u8>, StrategyPortfolioError> {
    super::strategy_bdd::validate_robdd_with_budget(bdd, bdd_budget)
        .map_err(StrategyPortfolioError::BddVerification)?;
    let mut bytes = Vec::new();
    for byte in b"LD-ROBDD1" {
        append_byte(&mut bytes, *byte, encoding_budget)?;
    }
    append_dependencies(&mut bytes, &bdd.dependencies, encoding_budget)?;
    append_dependencies(&mut bytes, &bdd.variable_order, encoding_budget)?;
    append_u64(&mut bytes, bdd.nodes.len(), encoding_budget)?;

    for node in &bdd.nodes {
        append_u64(&mut bytes, node.dependency_index, encoding_budget)?;
        append_bdd_ref(&mut bytes, node.low, encoding_budget)?;
        append_bdd_ref(&mut bytes, node.high, encoding_budget)?;
    }
    append_bdd_ref(&mut bytes, bdd.root, encoding_budget)?;
    Ok(bytes)
}

/// Verify the original QBF strategy, compile both representations for every
/// support-minimized function, independently verify each, and choose the
/// smaller canonical encoding. The original QBF verification always precedes
/// representation compilation.
pub fn verify_and_compile_qbf_strategy_portfolio(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    candidate: &QbfEvaluationResult,
    qbf_budget: QuantifiedEvaluationBudget,
    bdd_budget: StrategyBddBudget,
    aig_budget: StrategyAigBudget,
    encoding_budget: StrategyPortfolioBudget,
) -> Result<VerifiedStrategyPortfolio, StrategyPortfolioError> {
    let verified = verify_and_compile_qbf_strategy(
        problem,
        prefix,
        candidate,
        qbf_budget,
        bdd_budget,
    )
    .map_err(StrategyPortfolioError::SourceOrBddCompilation)?;

    let mut functions = Vec::with_capacity(verified.representation.functions.len());
    let mut total_bdd_bytes = 0usize;
    let mut total_aig_bytes = 0usize;
    let mut total_selected_bytes = 0usize;
    let mut total_bdd_nodes = 0usize;
    let mut total_aig_and_nodes = 0usize;

    for compiled in &verified.representation.functions {
        let (bdd_bytes, bdd_verification) =
            (encode_strategy_robdd(&compiled.bdd, bdd_budget, encoding_budget)?,
             verify_robdd_matches_strategy_function(
                 &compiled.bdd,
                 &compiled.reduced_function,
                 bdd_budget,
             )
             .map_err(StrategyPortfolioError::BddVerification)?);

        let aig = compile_strategy_function_to_aig_with_order(
            &compiled.reduced_function,
            &compiled.bdd.variable_order,
            aig_budget,
        )
        .map_err(StrategyPortfolioError::AigCompilation)?;
        let aig_verification = verify_aig_matches_strategy_function(
            &aig,
            &compiled.reduced_function,
            aig_budget,
        )
        .map_err(StrategyPortfolioError::AigVerification)?;
        let aig_bytes = encode_strategy_aig(&aig, aig_budget, encoding_budget)?;

        let (selected, selected_bytes) = if bdd_bytes.len() <= aig_bytes.len() {
            (SelectedStrategyRepresentation::Robdd, bdd_bytes.len())
        } else {
            (SelectedStrategyRepresentation::Aig, aig_bytes.len())
        };

        total_bdd_bytes = total_bdd_bytes
            .checked_add(bdd_bytes.len())
            .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
        total_aig_bytes = total_aig_bytes
            .checked_add(aig_bytes.len())
            .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
        total_selected_bytes = total_selected_bytes
            .checked_add(selected_bytes)
            .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
        total_bdd_nodes = total_bdd_nodes
            .checked_add(compiled.bdd.nodes.len())
            .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
        total_aig_and_nodes = total_aig_and_nodes
            .checked_add(aig_verification.and_nodes)
            .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;

        let total_bytes = total_bdd_bytes
            .checked_add(total_aig_bytes)
            .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
        if total_bytes > encoding_budget.max_total_representation_bytes {
            return Err(StrategyPortfolioError::TotalPortfolioTooLarge {
                required: total_bytes,
                limit: encoding_budget.max_total_representation_bytes,
            });
        }

        functions.push(StrategyFunctionPortfolio {
            variable: compiled.variable,
            original_dependencies: compiled.original_dependencies.clone(),
            reduced_dependencies: compiled.reduced_function.dependencies.clone(),
            bdd: compiled.bdd.clone(),
            aig,
            bdd_verification,
            aig_verification,
            bdd_bytes,
            aig_bytes,
            selected,
            selected_bytes,
        });
    }

    Ok(VerifiedStrategyPortfolio {
        value: verified.value,
        source_strategy_verification: verified.source_strategy_verification,
        functions,
        total_bdd_bytes,
        total_aig_bytes,
        total_selected_bytes,
        total_bdd_nodes,
        total_aig_and_nodes,
        bdd_search_complete: verified.representation.optimization_search_complete,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::decomposition::Literal;
    use super::super::quantified_scope::{Quantifier, QuantifierBlock, ValidatedQuantifierPrefix};
    use super::super::quantified_strategy::{evaluate_qbf_with_strategy, StrategyOwner};

    fn qbf_budget() -> QuantifiedEvaluationBudget {
        QuantifiedEvaluationBudget {
            max_variables: 4,
            max_total_variable_name_bytes: 64,
            max_clauses: 16,
            max_literals: 64,
            max_quantifier_node_visits: 1 << 8,
            max_opponent_assignments: 1 << 4,
            max_strategy_table_entries: 1 << 6,
        }
    }

    fn bdd_budget() -> StrategyBddBudget {
        StrategyBddBudget {
            max_dependency_variables: 4,
            max_table_entries: 1 << 4,
            max_bdd_nodes: 1 << 6,
            max_support_checks: 1 << 8,
            max_interaction_checks: 1 << 8,
            max_weighted_interaction_checks: 1 << 8,
            max_frontier_order_evaluations: 1 << 8,
            max_frontier_refinement_passes: 1,
            max_variable_order_evaluations: 1 << 8,
            max_sifting_passes: 1,
        }
    }

    fn aig_budget() -> StrategyAigBudget {
        StrategyAigBudget {
            max_dependency_variables: 4,
            max_truth_table_entries: 1 << 4,
            max_and_nodes: 1 << 6,
            max_assignments_checked: 1 << 4,
        }
    }

    fn encoding_budget() -> StrategyPortfolioBudget {
        StrategyPortfolioBudget {
            max_single_representation_bytes: 1 << 16,
            max_total_representation_bytes: 1 << 18,
        }
    }

    fn equality_qbf() -> (CnfProblem, ValidatedQuantifierPrefix) {
        let problem = CnfProblem {
            variables: vec!["x".into(), "y".into()],
            clauses: vec![
                vec![Literal::negative(0), Literal::positive(1)],
                vec![Literal::positive(0), Literal::negative(1)],
            ],
        };
        let prefix = ValidatedQuantifierPrefix::new(
            2,
            vec![
                QuantifierBlock {
                    quantifier: Quantifier::ForAll,
                    variables: vec![0],
                },
                QuantifierBlock {
                    quantifier: Quantifier::Exists,
                    variables: vec![1],
                },
            ],
            Default::default(),
        )
        .unwrap();
        (problem, prefix)
    }

    #[test]
    fn canonical_encodings_are_deterministic_and_versioned() {
        let (problem, prefix) = equality_qbf();
        let result = evaluate_qbf_with_strategy(&problem, &prefix, qbf_budget()).unwrap();
        let portfolio = verify_and_compile_qbf_strategy_portfolio(
            &problem,
            &prefix,
            &result,
            qbf_budget(),
            bdd_budget(),
            aig_budget(),
            encoding_budget(),
        )
        .unwrap();
        let function = &portfolio.functions[0];

        let bdd_again =
            encode_strategy_robdd(&function.bdd, bdd_budget(), encoding_budget()).unwrap();
        let aig_again =
            encode_strategy_aig(&function.aig, aig_budget(), encoding_budget()).unwrap();
        assert_eq!(function.bdd_bytes, bdd_again);
        assert_eq!(function.aig_bytes, aig_again);
        assert!(function.bdd_bytes.starts_with(b"LD-ROBDD1"));
        assert!(function.aig_bytes.starts_with(b"LD-AIG-1"));
    }

    #[test]
    fn source_strategy_is_checked_before_dual_representation_compilation() {
        let (problem, prefix) = equality_qbf();
        let result = evaluate_qbf_with_strategy(&problem, &prefix, qbf_budget()).unwrap();
        let portfolio = verify_and_compile_qbf_strategy_portfolio(
            &problem,
            &prefix,
            &result,
            qbf_budget(),
            bdd_budget(),
            aig_budget(),
            encoding_budget(),
        )
        .unwrap();

        assert!(portfolio.value);
        assert_eq!(
            portfolio.source_strategy_verification.opponent_assignments_checked,
            2
        );
        assert_eq!(portfolio.functions.len(), 1);
        assert_eq!(portfolio.functions[0].variable, 1);
        assert_eq!(portfolio.functions[0].reduced_dependencies, vec![0]);
        assert_eq!(portfolio.functions[0].bdd_verification.assignments_checked, 2);
        assert_eq!(portfolio.functions[0].aig_verification.assignments_checked, 2);
        assert_eq!(
            portfolio.functions[0].selected_bytes,
            portfolio.functions[0]
                .bdd_bytes
                .len()
                .min(portfolio.functions[0].aig_bytes.len())
        );
        assert_eq!(
            portfolio.total_selected_bytes,
            portfolio.functions[0].selected_bytes
        );
        assert!(portfolio.bdd_search_complete);
        assert_eq!(
            portfolio.functions[0].aig_verification.and_nodes,
            portfolio.total_aig_and_nodes
        );
    }

    #[test]
    fn tampered_strategy_fails_before_any_circuit_is_admitted() {
        let (problem, prefix) = equality_qbf();
        let mut result = evaluate_qbf_with_strategy(&problem, &prefix, qbf_budget()).unwrap();
        result.strategy.functions[0]
            .table
            .insert(vec![false], true);
        assert!(matches!(
            verify_and_compile_qbf_strategy_portfolio(
                &problem,
                &prefix,
                &result,
                qbf_budget(),
                bdd_budget(),
                aig_budget(),
                encoding_budget(),
            ),
            Err(StrategyPortfolioError::SourceOrBddCompilation(
                VerifiedStrategyCompilationError::SourceStrategyInvalid(_)
            ))
        ));
    }

    #[test]
    fn representation_and_aggregate_byte_budgets_fail_closed() {
        let (problem, prefix) = equality_qbf();
        let result = evaluate_qbf_with_strategy(&problem, &prefix, qbf_budget()).unwrap();
        let too_small = StrategyPortfolioBudget {
            max_single_representation_bytes: 8,
            max_total_representation_bytes: 1 << 18,
        };
        assert!(matches!(
            verify_and_compile_qbf_strategy_portfolio(
                &problem,
                &prefix,
                &result,
                qbf_budget(),
                bdd_budget(),
                aig_budget(),
                too_small,
            ),
            Err(StrategyPortfolioError::RepresentationTooLarge { .. })
        ));

        let total_too_small = StrategyPortfolioBudget {
            max_single_representation_bytes: 1 << 16,
            max_total_representation_bytes: 8,
        };
        assert!(matches!(
            verify_and_compile_qbf_strategy_portfolio(
                &problem,
                &prefix,
                &result,
                qbf_budget(),
                bdd_budget(),
                aig_budget(),
                total_too_small,
            ),
            Err(StrategyPortfolioError::TotalPortfolioTooLarge { .. })
        ));
    }

    #[test]
    fn result_owner_mismatch_is_rejected_before_representation_compilation() {
        let (problem, prefix) = equality_qbf();
        let mut result = evaluate_qbf_with_strategy(&problem, &prefix, qbf_budget()).unwrap();
        result.value = false;
        assert!(matches!(
            verify_and_compile_qbf_strategy_portfolio(
                &problem,
                &prefix,
                &result,
                qbf_budget(),
                bdd_budget(),
                aig_budget(),
                encoding_budget(),
            ),
            Err(StrategyPortfolioError::SourceOrBddCompilation(
                VerifiedStrategyCompilationError::ResultOwnerMismatch { .. }
            ))
        ));
    }

    #[test]
    fn empty_strategy_serializes_and_selects_an_empty_portfolio() {
        let problem = CnfProblem {
            variables: vec![],
            clauses: vec![],
        };
        let prefix =
            ValidatedQuantifierPrefix::new(0, vec![], Default::default()).unwrap();
        let result = evaluate_qbf_with_strategy(&problem, &prefix, qbf_budget()).unwrap();
        let portfolio = verify_and_compile_qbf_strategy_portfolio(
            &problem,
            &prefix,
            &result,
            qbf_budget(),
            bdd_budget(),
            aig_budget(),
            encoding_budget(),
        )
        .unwrap();
        assert!(portfolio.value);
        assert!(portfolio.functions.is_empty());
        assert_eq!(portfolio.total_bdd_bytes, 0);
        assert_eq!(portfolio.total_aig_bytes, 0);
        assert_eq!(portfolio.total_selected_bytes, 0);
    }
}
