// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Verified portfolio compilation for Boolean strategy functions.
//!
//! The original QBF strategy is validated first against the source matrix and
//! ordered prefix. Each support-minimized function is then compiled to both a
//! ROBDD and an AIG. Each representation is independently checked against the
//! same reduced truth table and encoded using a versioned canonical byte
//! format. The portfolio exposes a Pareto frontier over encoded bytes, circuit
//! nodes, worst-case evaluation work and independent verification work. Its
//! default selected representation is the smallest encoded member of that
//! frontier; callers retain both alternatives for workload-specific choice.
//!
//! A compiled circuit is not itself a QBF proof. The source strategy checker
//! remains the authority for whether the Skolem/Herbrand functions win.

use super::decomposition::CnfProblem;
use super::quantified_game_dag::{
    QbfGameDagBudget, QbfGameDagError, QbfGameDagNodeKind, QbfGameDagVerificationReport,
    QbfGameProofDag, ResidualLiteral, verify_qbf_game_proof_dag,
};
use super::quantified_proof::{
    QbfGameProof, QbfProofNodeKind, QbfProofVerificationReport, QuantifiedProofBudget,
    QuantifiedProofError, verify_qbf_game_proof,
};
use super::quantified_scope::{Quantifier, ValidatedQuantifierPrefix};
use super::quantified_strategy::{
    QbfEvaluationResult, QuantifiedEvaluationBudget, StrategyVerificationReport,
};
use super::strategy_aig::{
    AigLiteral, AigNode, StrategyAig, StrategyAigBudget, StrategyAigError, StrategyAigReport,
    compile_strategy_function_to_aig_with_order, verify_aig_matches_strategy_function,
};
use super::strategy_bdd::{
    BddRef, StrategyBddBudget, StrategyBddError, StrategyBddReport,
    VerifiedStrategyCompilationError, verify_and_compile_qbf_strategy,
    verify_robdd_matches_strategy_function,
};
use std::convert::TryFrom;

/// Representation selected for one strategy function by canonical byte size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectedStrategyRepresentation {
    Robdd,
    Aig,
}

/// Deterministic structural cost vector for one checked representation.
///
/// `worst_case_evaluation_steps` counts decision-node visits for an ROBDD
/// and full stored-node evaluations for the current straight-line AIG
/// evaluator. `estimated_verification_work_units` is explicitly a structural
/// estimate: graph-node visits plus exhaustive assignments multiplied by
/// worst-case evaluation steps. It is not wall-clock timing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StrategyRepresentationCost {
    pub encoded_bytes: usize,
    pub internal_nodes: usize,
    pub logic_depth: usize,
    pub worst_case_evaluation_steps: usize,
    pub estimated_verification_work_units: u128,
}

/// Return the nondominated encodings. Exact cost ties retain ROBDD as the
/// deterministic representative; incomparable costs preserve both.
pub fn strategy_representation_pareto_frontier(
    robdd: StrategyRepresentationCost,
    aig: StrategyRepresentationCost,
) -> Vec<SelectedStrategyRepresentation> {
    fn dominates(left: StrategyRepresentationCost, right: StrategyRepresentationCost) -> bool {
        let no_worse = left.encoded_bytes <= right.encoded_bytes
            && left.internal_nodes <= right.internal_nodes
            && left.logic_depth <= right.logic_depth
            && left.worst_case_evaluation_steps <= right.worst_case_evaluation_steps
            && left.estimated_verification_work_units <= right.estimated_verification_work_units;
        let strictly_better = left.encoded_bytes < right.encoded_bytes
            || left.internal_nodes < right.internal_nodes
            || left.logic_depth < right.logic_depth
            || left.worst_case_evaluation_steps < right.worst_case_evaluation_steps
            || left.estimated_verification_work_units < right.estimated_verification_work_units;
        no_worse && strictly_better
    }

    if dominates(robdd, aig) || robdd == aig {
        vec![SelectedStrategyRepresentation::Robdd]
    } else if dominates(aig, robdd) {
        vec![SelectedStrategyRepresentation::Aig]
    } else {
        vec![
            SelectedStrategyRepresentation::Robdd,
            SelectedStrategyRepresentation::Aig,
        ]
    }
}

fn select_smallest_pareto_encoding(
    frontier: &[SelectedStrategyRepresentation],
    robdd: StrategyRepresentationCost,
    aig: StrategyRepresentationCost,
) -> (SelectedStrategyRepresentation, StrategyRepresentationCost) {
    let mut selected = frontier[0];
    let mut selected_cost = match selected {
        SelectedStrategyRepresentation::Robdd => robdd,
        SelectedStrategyRepresentation::Aig => aig,
    };
    for &candidate in frontier.iter().skip(1) {
        let cost = match candidate {
            SelectedStrategyRepresentation::Robdd => robdd,
            SelectedStrategyRepresentation::Aig => aig,
        };
        let candidate_key = (
            cost.encoded_bytes,
            cost.estimated_verification_work_units,
            cost.worst_case_evaluation_steps,
            cost.internal_nodes,
            matches!(candidate, SelectedStrategyRepresentation::Aig),
        );
        let selected_key = (
            selected_cost.encoded_bytes,
            selected_cost.estimated_verification_work_units,
            selected_cost.worst_case_evaluation_steps,
            selected_cost.internal_nodes,
            matches!(selected, SelectedStrategyRepresentation::Aig),
        );
        if candidate_key < selected_key {
            selected = candidate;
            selected_cost = cost;
        }
    }
    (selected, selected_cost)
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
    GameProof(QuantifiedProofError),
    GameProofResultMismatch {
        proof_value: bool,
        candidate_value: bool,
    },
    GameDag(QbfGameDagError),
    GameDagResultMismatch {
        proof_value: bool,
        candidate_value: bool,
    },
    ProofEncodingInvalid {
        offset: usize,
    },
    ProofEncodingUnknownNodeTag {
        offset: usize,
        tag: u8,
    },
    ProofEncodingTrailingBytes {
        remaining: usize,
    },
    ProofEncodingRoundTripMismatch,
    BddVerification(StrategyBddError),
    AigCompilation(StrategyAigError),
    AigVerification(StrategyAigError),
    RepresentationTooLarge {
        required: usize,
        limit: usize,
    },
    TotalPortfolioTooLarge {
        required: usize,
        limit: usize,
    },
    EncodingIndexOverflow {
        value: usize,
    },
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
    pub bdd_cost: StrategyRepresentationCost,
    pub aig_cost: StrategyRepresentationCost,
    pub pareto_frontier: Vec<SelectedStrategyRepresentation>,
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
    pub total_bdd_estimated_verification_work: u128,
    pub total_aig_estimated_verification_work: u128,
    pub total_selected_estimated_verification_work: u128,
    pub bdd_search_complete: bool,
}

/// A complete verified bundle: quantified game proof, extracted strategy,
/// and independently checked BDD/AIG representations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedQbfEvidencePortfolio {
    pub value: bool,
    pub game_proof: QbfGameProof,
    pub game_proof_verification: QbfProofVerificationReport,
    pub game_proof_bytes: Vec<u8>,
    pub strategy_portfolio: VerifiedStrategyPortfolio,
    pub total_evidence_bytes: usize,
}

/// A complete verified bundle using the residual-CNF proof DAG, extracted
/// strategy, and independently checked BDD/AIG representations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedQbfDagEvidencePortfolio {
    pub value: bool,
    pub game_proof_dag: QbfGameProofDag,
    pub game_proof_dag_verification: QbfGameDagVerificationReport,
    pub game_proof_dag_bytes: Vec<u8>,
    pub strategy_portfolio: VerifiedStrategyPortfolio,
    pub total_evidence_bytes: usize,
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

fn checked_cost_add(left: u128, right: u128) -> Result<u128, StrategyPortfolioError> {
    left.checked_add(right)
        .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)
}

fn checked_cost_mul(left: u128, right: u128) -> Result<u128, StrategyPortfolioError> {
    left.checked_mul(right)
        .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)
}

fn bdd_representation_cost(
    bdd: &super::strategy_bdd::ReducedOrderedBdd,
    encoded_bytes: usize,
    assignments_checked: u128,
) -> Result<StrategyRepresentationCost, StrategyPortfolioError> {
    let mut depth = vec![0usize; bdd.nodes.len()];
    for (index, node) in bdd.nodes.iter().enumerate() {
        let child_depth = |child: BddRef| -> usize {
            match child {
                BddRef::False | BddRef::True => 0,
                BddRef::Node(child_index) => depth[child_index],
            }
        };
        depth[index] = child_depth(node.low)
            .max(child_depth(node.high))
            .checked_add(1)
            .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
    }

    let logic_depth = match bdd.root {
        BddRef::False | BddRef::True => 0,
        BddRef::Node(root) => depth[root],
    };
    let structural_work = u128::try_from(bdd.nodes.len())
        .map_err(|_| StrategyPortfolioError::ResourceAccountingOverflow)?;
    let evaluation_work = checked_cost_mul(
        assignments_checked,
        u128::try_from(logic_depth)
            .map_err(|_| StrategyPortfolioError::ResourceAccountingOverflow)?,
    )?;
    Ok(StrategyRepresentationCost {
        encoded_bytes,
        internal_nodes: bdd.nodes.len(),
        logic_depth,
        worst_case_evaluation_steps: logic_depth,
        estimated_verification_work_units: checked_cost_add(structural_work, evaluation_work)?,
    })
}

fn aig_representation_cost(
    aig: &StrategyAig,
    encoded_bytes: usize,
    assignments_checked: u128,
    and_nodes: usize,
) -> Result<StrategyRepresentationCost, StrategyPortfolioError> {
    let mut depth = vec![0usize; aig.nodes.len()];
    for (index, node) in aig.nodes.iter().enumerate() {
        if let AigNode::And { left, right } = node {
            let child_depth = |literal: AigLiteral| -> usize {
                match literal {
                    AigLiteral::Constant(_) => 0,
                    AigLiteral::Node { index, .. } => depth[index],
                }
            };
            depth[index] = child_depth(*left)
                .max(child_depth(*right))
                .checked_add(1)
                .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
        }
    }
    let logic_depth = match aig.output {
        AigLiteral::Constant(_) => 0,
        AigLiteral::Node { index, .. } => depth[index],
    };
    // The present AIG evaluator executes its canonical straight-line node list
    // for every assignment; unlike an ROBDD it does not short-circuit by path.
    let worst_case_evaluation_steps = aig.nodes.len();
    let structural_work = u128::try_from(aig.nodes.len())
        .map_err(|_| StrategyPortfolioError::ResourceAccountingOverflow)?;
    let evaluation_work = checked_cost_mul(
        assignments_checked,
        u128::try_from(worst_case_evaluation_steps)
            .map_err(|_| StrategyPortfolioError::ResourceAccountingOverflow)?,
    )?;
    Ok(StrategyRepresentationCost {
        encoded_bytes,
        internal_nodes: and_nodes,
        logic_depth,
        worst_case_evaluation_steps,
        estimated_verification_work_units: checked_cost_add(structural_work, evaluation_work)?,
    })
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
/// Encode a verified QBF game proof in canonical LD-QBFPROOF1 little-endian
/// format. The caller must run the source-bound independent proof checker
/// before accepting this byte sequence as evidence.
pub fn encode_qbf_game_proof(
    proof: &QbfGameProof,
    encoding_budget: StrategyPortfolioBudget,
) -> Result<Vec<u8>, StrategyPortfolioError> {
    let mut bytes = Vec::new();
    for byte in b"LD-QBFPROOF1" {
        append_byte(&mut bytes, *byte, encoding_budget)?;
    }
    append_u64(&mut bytes, proof.variable_count, encoding_budget)?;
    append_u64(&mut bytes, proof.root, encoding_budget)?;
    append_u64(&mut bytes, proof.nodes.len(), encoding_budget)?;

    for node in &proof.nodes {
        append_u64(&mut bytes, node.prefix_position, encoding_budget)?;
        append_byte(&mut bytes, u8::from(node.value), encoding_budget)?;
        match &node.kind {
            QbfProofNodeKind::MatrixLeaf => {
                append_byte(&mut bytes, 0, encoding_budget)?;
            }
            QbfProofNodeKind::EarlyTrue => {
                append_byte(&mut bytes, 1, encoding_budget)?;
            }
            QbfProofNodeKind::EarlyFalseClause { clause_index } => {
                append_byte(&mut bytes, 2, encoding_budget)?;
                append_u64(&mut bytes, *clause_index, encoding_budget)?;
            }
            QbfProofNodeKind::ExistsDecision {
                variable,
                low_child,
                high_child,
            } => {
                append_byte(&mut bytes, 3, encoding_budget)?;
                append_u64(&mut bytes, *variable, encoding_budget)?;
                append_u64(&mut bytes, *low_child, encoding_budget)?;
                append_u64(&mut bytes, *high_child, encoding_budget)?;
            }
            QbfProofNodeKind::ForAllDecision {
                variable,
                low_child,
                high_child,
            } => {
                append_byte(&mut bytes, 4, encoding_budget)?;
                append_u64(&mut bytes, *variable, encoding_budget)?;
                append_u64(&mut bytes, *low_child, encoding_budget)?;
                append_u64(&mut bytes, *high_child, encoding_budget)?;
            }
        }
    }
    Ok(bytes)
}

/// Encode a residual QBF proof DAG in canonical LD-QBFDAG1 little-endian
/// format. The DAG must still be bound to its source matrix and prefix by the
/// independent DAG verifier before acceptance.
pub fn encode_qbf_game_proof_dag(
    proof: &QbfGameProofDag,
    encoding_budget: StrategyPortfolioBudget,
) -> Result<Vec<u8>, StrategyPortfolioError> {
    let mut bytes = Vec::new();
    for byte in b"LD-QBFDAG1" {
        append_byte(&mut bytes, *byte, encoding_budget)?;
    }
    append_u64(&mut bytes, proof.variable_count, encoding_budget)?;
    append_u64(&mut bytes, proof.prefix_order.len(), encoding_budget)?;
    for (quantifier, variable) in &proof.prefix_order {
        let tag = match quantifier {
            Quantifier::Exists => 0,
            Quantifier::ForAll => 1,
        };
        append_byte(&mut bytes, tag, encoding_budget)?;
        append_u64(&mut bytes, *variable, encoding_budget)?;
    }
    append_u64(&mut bytes, proof.root, encoding_budget)?;
    append_u64(&mut bytes, proof.nodes.len(), encoding_budget)?;

    for node in &proof.nodes {
        append_u64(&mut bytes, node.prefix_position, encoding_budget)?;
        append_byte(&mut bytes, u8::from(node.value), encoding_budget)?;
        append_u64(&mut bytes, node.residual.len(), encoding_budget)?;
        for clause in &node.residual {
            append_u64(&mut bytes, clause.len(), encoding_budget)?;
            for literal in clause {
                append_u64(&mut bytes, literal.variable, encoding_budget)?;
                append_byte(&mut bytes, u8::from(literal.positive), encoding_budget)?;
            }
        }
        match &node.kind {
            QbfGameDagNodeKind::TrueTerminal => {
                append_byte(&mut bytes, 0, encoding_budget)?;
            }
            QbfGameDagNodeKind::FalseTerminal => {
                append_byte(&mut bytes, 1, encoding_budget)?;
            }
            QbfGameDagNodeKind::ExistsDecision {
                variable,
                low_child,
                high_child,
            } => {
                append_byte(&mut bytes, 2, encoding_budget)?;
                append_u64(&mut bytes, *variable, encoding_budget)?;
                append_u64(&mut bytes, *low_child, encoding_budget)?;
                append_u64(&mut bytes, *high_child, encoding_budget)?;
            }
            QbfGameDagNodeKind::ForAllDecision {
                variable,
                low_child,
                high_child,
            } => {
                append_byte(&mut bytes, 3, encoding_budget)?;
                append_u64(&mut bytes, *variable, encoding_budget)?;
                append_u64(&mut bytes, *low_child, encoding_budget)?;
                append_u64(&mut bytes, *high_child, encoding_budget)?;
            }
        }
    }
    Ok(bytes)
}

fn read_proof_byte(bytes: &[u8], offset: &mut usize) -> Result<u8, StrategyPortfolioError> {
    let Some(&value) = bytes.get(*offset) else {
        return Err(StrategyPortfolioError::ProofEncodingInvalid { offset: *offset });
    };
    *offset = (*offset)
        .checked_add(1)
        .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
    Ok(value)
}

fn read_proof_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, StrategyPortfolioError> {
    let end = (*offset)
        .checked_add(8)
        .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
    let Some(slice) = bytes.get(*offset..end) else {
        return Err(StrategyPortfolioError::ProofEncodingInvalid { offset: *offset });
    };
    let mut raw = [0u8; 8];
    raw.copy_from_slice(slice);
    *offset = end;
    Ok(u64::from_le_bytes(raw))
}

fn read_proof_usize(bytes: &[u8], offset: &mut usize) -> Result<usize, StrategyPortfolioError> {
    let position = *offset;
    usize::try_from(read_proof_u64(bytes, offset)?)
        .map_err(|_| StrategyPortfolioError::ProofEncodingInvalid { offset: position })
}

fn validate_decoded_proof_tree(proof: &QbfGameProof) -> Result<(), StrategyPortfolioError> {
    if proof.nodes.is_empty()
        || proof.root >= proof.nodes.len()
        || proof.root != proof.nodes.len() - 1
    {
        return Err(StrategyPortfolioError::ProofEncodingInvalid { offset: 0 });
    }
    let mut parent_counts = vec![0usize; proof.nodes.len()];
    for (parent, node) in proof.nodes.iter().enumerate() {
        let children = match &node.kind {
            QbfProofNodeKind::ExistsDecision {
                low_child,
                high_child,
                ..
            }
            | QbfProofNodeKind::ForAllDecision {
                low_child,
                high_child,
                ..
            } => {
                if low_child == high_child {
                    return Err(StrategyPortfolioError::ProofEncodingInvalid { offset: parent });
                }
                [Some(*low_child), Some(*high_child)]
            }
            _ => [None, None],
        };
        for child in children.into_iter().flatten() {
            if child >= parent {
                return Err(StrategyPortfolioError::ProofEncodingInvalid { offset: parent });
            }
            parent_counts[child] = parent_counts[child]
                .checked_add(1)
                .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
            if parent_counts[child] > 1 {
                return Err(StrategyPortfolioError::ProofEncodingInvalid { offset: child });
            }
        }
    }
    for (node, parents) in parent_counts.iter().enumerate() {
        if *parents != usize::from(node != proof.root) {
            return Err(StrategyPortfolioError::ProofEncodingInvalid { offset: node });
        }
    }
    Ok(())
}

/// Decode the canonical LD-QBFPROOF1 format under explicit byte, variable and
/// node limits. The result is only parsed evidence; bind it to a source matrix
/// and ordered prefix with `verify_qbf_game_proof` before trusting its value.
pub fn decode_qbf_game_proof(
    bytes: &[u8],
    proof_budget: QuantifiedProofBudget,
    encoding_budget: StrategyPortfolioBudget,
) -> Result<QbfGameProof, StrategyPortfolioError> {
    if bytes.len() > encoding_budget.max_single_representation_bytes {
        return Err(StrategyPortfolioError::RepresentationTooLarge {
            required: bytes.len(),
            limit: encoding_budget.max_single_representation_bytes,
        });
    }
    const MAGIC: &[u8] = b"LD-QBFPROOF1";
    if bytes.get(..MAGIC.len()) != Some(MAGIC) {
        return Err(StrategyPortfolioError::ProofEncodingInvalid { offset: 0 });
    }
    let mut offset = MAGIC.len();
    let variable_count = read_proof_usize(bytes, &mut offset)?;
    let root = read_proof_usize(bytes, &mut offset)?;
    let node_count = read_proof_usize(bytes, &mut offset)?;

    if variable_count > proof_budget.max_variables {
        return Err(StrategyPortfolioError::ProofEncodingInvalid {
            offset: MAGIC.len(),
        });
    }
    if node_count > proof_budget.max_proof_nodes {
        return Err(StrategyPortfolioError::ProofEncodingInvalid {
            offset: MAGIC.len() + 16,
        });
    }
    // Every node needs at least prefix_position:u64 + value:u8 + tag:u8.
    let minimum_node_bytes = node_count
        .checked_mul(10)
        .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
    let remaining = bytes.len().saturating_sub(offset);
    if minimum_node_bytes > remaining {
        return Err(StrategyPortfolioError::ProofEncodingInvalid { offset });
    }
    if node_count == 0 || root != node_count - 1 {
        return Err(StrategyPortfolioError::ProofEncodingInvalid {
            offset: MAGIC.len() + 8,
        });
    }

    let mut nodes = Vec::new();
    nodes
        .try_reserve_exact(node_count)
        .map_err(|_| StrategyPortfolioError::ProofEncodingInvalid { offset })?;
    for node_index in 0..node_count {
        let prefix_position = read_proof_usize(bytes, &mut offset)?;
        if prefix_position > variable_count {
            return Err(StrategyPortfolioError::ProofEncodingInvalid { offset });
        }
        let value_offset = offset;
        let value = match read_proof_byte(bytes, &mut offset)? {
            0 => false,
            1 => true,
            _ => {
                return Err(StrategyPortfolioError::ProofEncodingInvalid {
                    offset: value_offset,
                });
            }
        };
        let tag_offset = offset;
        let tag = read_proof_byte(bytes, &mut offset)?;
        let kind = match tag {
            0 => QbfProofNodeKind::MatrixLeaf,
            1 => QbfProofNodeKind::EarlyTrue,
            2 => QbfProofNodeKind::EarlyFalseClause {
                clause_index: read_proof_usize(bytes, &mut offset)?,
            },
            3 => QbfProofNodeKind::ExistsDecision {
                variable: read_proof_usize(bytes, &mut offset)?,
                low_child: read_proof_usize(bytes, &mut offset)?,
                high_child: read_proof_usize(bytes, &mut offset)?,
            },
            4 => QbfProofNodeKind::ForAllDecision {
                variable: read_proof_usize(bytes, &mut offset)?,
                low_child: read_proof_usize(bytes, &mut offset)?,
                high_child: read_proof_usize(bytes, &mut offset)?,
            },
            _ => {
                return Err(StrategyPortfolioError::ProofEncodingUnknownNodeTag {
                    offset: tag_offset,
                    tag,
                });
            }
        };
        // Bound source-position fields and proof edges before exposing the
        // parsed object to the source-bound verifier.
        if let QbfProofNodeKind::ExistsDecision {
            variable,
            low_child,
            high_child,
        }
        | QbfProofNodeKind::ForAllDecision {
            variable,
            low_child,
            high_child,
        } = &kind
        {
            if *variable >= variable_count || *low_child >= node_index || *high_child >= node_index
            {
                return Err(StrategyPortfolioError::ProofEncodingInvalid { offset: tag_offset });
            }
        }
        nodes.push(super::quantified_proof::QbfProofNode {
            prefix_position,
            value,
            kind,
        });
    }
    if offset != bytes.len() {
        return Err(StrategyPortfolioError::ProofEncodingTrailingBytes {
            remaining: bytes.len() - offset,
        });
    }

    let proof = QbfGameProof {
        variable_count,
        root,
        nodes,
    };
    validate_decoded_proof_tree(&proof)?;
    Ok(proof)
}

/// Decode canonical LD-QBFDAG1 bytes under explicit source-independent
/// structural limits. Source binding and mathematical correctness still
/// require `verify_qbf_game_proof_dag` with the original problem and prefix.
pub fn decode_qbf_game_proof_dag(
    bytes: &[u8],
    dag_budget: QbfGameDagBudget,
    encoding_budget: StrategyPortfolioBudget,
) -> Result<QbfGameProofDag, StrategyPortfolioError> {
    if bytes.len() > encoding_budget.max_single_representation_bytes {
        return Err(StrategyPortfolioError::RepresentationTooLarge {
            required: bytes.len(),
            limit: encoding_budget.max_single_representation_bytes,
        });
    }
    const MAGIC: &[u8] = b"LD-QBFDAG1";
    if bytes.get(..MAGIC.len()) != Some(MAGIC) {
        return Err(StrategyPortfolioError::ProofEncodingInvalid { offset: 0 });
    }
    let mut offset = MAGIC.len();
    let variable_count = read_proof_usize(bytes, &mut offset)?;
    let prefix_len = read_proof_usize(bytes, &mut offset)?;
    if variable_count > dag_budget.max_variables || prefix_len != variable_count {
        return Err(StrategyPortfolioError::ProofEncodingInvalid { offset });
    }

    let mut prefix_order = Vec::with_capacity(prefix_len);
    let mut prefix_variables = std::collections::BTreeSet::new();
    for _ in 0..prefix_len {
        let quantifier_tag = read_proof_byte(bytes, &mut offset)?;
        let quantifier = match quantifier_tag {
            0 => Quantifier::Exists,
            1 => Quantifier::ForAll,
            tag => {
                return Err(StrategyPortfolioError::ProofEncodingUnknownNodeTag {
                    offset: offset.saturating_sub(1),
                    tag,
                });
            }
        };
        let variable = read_proof_usize(bytes, &mut offset)?;
        if variable >= variable_count || !prefix_variables.insert(variable) {
            return Err(StrategyPortfolioError::ProofEncodingInvalid {
                offset: offset.saturating_sub(8),
            });
        }
        prefix_order.push((quantifier, variable));
    }
    if prefix_variables.len() != variable_count {
        return Err(StrategyPortfolioError::ProofEncodingInvalid { offset });
    }

    let root = read_proof_usize(bytes, &mut offset)?;
    let node_count = read_proof_usize(bytes, &mut offset)?;
    if node_count == 0 || node_count > dag_budget.max_proof_nodes || root != node_count - 1 {
        return Err(StrategyPortfolioError::ProofEncodingInvalid { offset });
    }
    // Each node minimally needs prefix_position:u64, value:u8 and tag:u8.
    let minimum_node_bytes = node_count
        .checked_mul(10)
        .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
    if minimum_node_bytes > bytes.len().saturating_sub(offset) {
        return Err(StrategyPortfolioError::ProofEncodingInvalid { offset });
    }

    let mut nodes = Vec::with_capacity(node_count);
    let mut total_residual_literals = 0u128;
    let mut total_residual_clauses = 0u128;
    for _ in 0..node_count {
        let prefix_position = read_proof_usize(bytes, &mut offset)?;
        let value = match read_proof_byte(bytes, &mut offset)? {
            0 => false,
            1 => true,
            _ => {
                return Err(StrategyPortfolioError::ProofEncodingInvalid {
                    offset: offset.saturating_sub(1),
                });
            }
        };
        let clause_count = read_proof_usize(bytes, &mut offset)?;
        if clause_count > dag_budget.max_clauses {
            return Err(StrategyPortfolioError::ProofEncodingInvalid { offset });
        }
        let clause_count_u128 = u128::try_from(clause_count)
            .map_err(|_| StrategyPortfolioError::ResourceAccountingOverflow)?;
        total_residual_clauses = total_residual_clauses
            .checked_add(clause_count_u128)
            .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
        if total_residual_clauses > dag_budget.max_stored_residual_clauses {
            return Err(StrategyPortfolioError::ProofEncodingInvalid { offset });
        }

        let mut residual = Vec::with_capacity(clause_count);
        let mut node_residual_literals = 0u128;
        for _ in 0..clause_count {
            let literal_count = read_proof_usize(bytes, &mut offset)?;
            if literal_count > dag_budget.max_literals {
                return Err(StrategyPortfolioError::ProofEncodingInvalid { offset });
            }
            let literal_count_u128 = u128::try_from(literal_count)
                .map_err(|_| StrategyPortfolioError::ResourceAccountingOverflow)?;
            node_residual_literals = node_residual_literals
                .checked_add(literal_count_u128)
                .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
            if node_residual_literals > dag_budget.max_literals {
                return Err(StrategyPortfolioError::ProofEncodingInvalid { offset });
            }
            total_residual_literals = total_residual_literals
                .checked_add(literal_count_u128)
                .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
            if total_residual_literals > dag_budget.max_stored_residual_literals {
                return Err(StrategyPortfolioError::ProofEncodingInvalid { offset });
            }

            let mut clause = Vec::with_capacity(literal_count);
            for _ in 0..literal_count {
                let variable = read_proof_usize(bytes, &mut offset)?;
                let positive = match read_proof_byte(bytes, &mut offset)? {
                    0 => false,
                    1 => true,
                    _ => {
                        return Err(StrategyPortfolioError::ProofEncodingInvalid {
                            offset: offset.saturating_sub(1),
                        });
                    }
                };
                if variable >= variable_count {
                    return Err(StrategyPortfolioError::ProofEncodingInvalid {
                        offset: offset.saturating_sub(9),
                    });
                }
                clause.push(ResidualLiteral { variable, positive });
            }
            residual.push(clause);
        }

        let kind_tag_offset = offset;
        let tag = read_proof_byte(bytes, &mut offset)?;
        let kind = match tag {
            0 => QbfGameDagNodeKind::TrueTerminal,
            1 => QbfGameDagNodeKind::FalseTerminal,
            2 => QbfGameDagNodeKind::ExistsDecision {
                variable: read_proof_usize(bytes, &mut offset)?,
                low_child: read_proof_usize(bytes, &mut offset)?,
                high_child: read_proof_usize(bytes, &mut offset)?,
            },
            3 => QbfGameDagNodeKind::ForAllDecision {
                variable: read_proof_usize(bytes, &mut offset)?,
                low_child: read_proof_usize(bytes, &mut offset)?,
                high_child: read_proof_usize(bytes, &mut offset)?,
            },
            unknown => {
                return Err(StrategyPortfolioError::ProofEncodingUnknownNodeTag {
                    offset: kind_tag_offset,
                    tag: unknown,
                });
            }
        };
        nodes.push(super::quantified_game_dag::QbfGameDagNode {
            prefix_position,
            residual,
            value,
            kind,
        });
    }

    if offset != bytes.len() {
        return Err(StrategyPortfolioError::ProofEncodingTrailingBytes {
            remaining: bytes.len().saturating_sub(offset),
        });
    }

    Ok(QbfGameProofDag {
        variable_count,
        prefix_order,
        root,
        nodes,
    })
}

/// Verify and compile the game proof, winning strategy and both circuit
/// representations as one evidence bundle. The independent game proof and
/// strategy verifiers are both mandatory; equal truth values alone are not
/// treated as a substitute for either check.
pub fn verify_and_compile_qbf_evidence_portfolio(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    proof: &QbfGameProof,
    candidate: &QbfEvaluationResult,
    qbf_budget: QuantifiedEvaluationBudget,
    proof_budget: QuantifiedProofBudget,
    bdd_budget: StrategyBddBudget,
    aig_budget: StrategyAigBudget,
    encoding_budget: StrategyPortfolioBudget,
) -> Result<VerifiedQbfEvidencePortfolio, StrategyPortfolioError> {
    let game_proof_verification = verify_qbf_game_proof(problem, prefix, proof, proof_budget)
        .map_err(StrategyPortfolioError::GameProof)?;
    if game_proof_verification.root_value != candidate.value {
        return Err(StrategyPortfolioError::GameProofResultMismatch {
            proof_value: game_proof_verification.root_value,
            candidate_value: candidate.value,
        });
    }
    let game_proof_bytes = encode_qbf_game_proof(proof, encoding_budget)?;
    let decoded_proof = decode_qbf_game_proof(&game_proof_bytes, proof_budget, encoding_budget)?;
    if decoded_proof != *proof {
        return Err(StrategyPortfolioError::ProofEncodingRoundTripMismatch);
    }
    let decoded_verification = verify_qbf_game_proof(problem, prefix, &decoded_proof, proof_budget)
        .map_err(StrategyPortfolioError::GameProof)?;
    if decoded_verification != game_proof_verification {
        return Err(StrategyPortfolioError::ProofEncodingRoundTripMismatch);
    }
    let strategy_portfolio = verify_and_compile_qbf_strategy_portfolio(
        problem,
        prefix,
        candidate,
        qbf_budget,
        bdd_budget,
        aig_budget,
        encoding_budget,
    )?;

    let circuit_bytes = strategy_portfolio
        .total_bdd_bytes
        .checked_add(strategy_portfolio.total_aig_bytes)
        .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
    let total_evidence_bytes = game_proof_bytes
        .len()
        .checked_add(circuit_bytes)
        .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
    if total_evidence_bytes > encoding_budget.max_total_representation_bytes {
        return Err(StrategyPortfolioError::TotalPortfolioTooLarge {
            required: total_evidence_bytes,
            limit: encoding_budget.max_total_representation_bytes,
        });
    }

    Ok(VerifiedQbfEvidencePortfolio {
        value: game_proof_verification.root_value,
        game_proof: decoded_proof,
        game_proof_verification: decoded_verification,
        game_proof_bytes,
        strategy_portfolio,
        total_evidence_bytes,
    })
}

/// Verify and compile a residual-DAG proof, its canonical byte encoding,
/// and the candidate's BDD/AIG strategy portfolio as one source-bound bundle.
pub fn verify_and_compile_qbf_dag_evidence_portfolio(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    proof: &QbfGameProofDag,
    candidate: &QbfEvaluationResult,
    qbf_budget: QuantifiedEvaluationBudget,
    dag_budget: QbfGameDagBudget,
    bdd_budget: StrategyBddBudget,
    aig_budget: StrategyAigBudget,
    encoding_budget: StrategyPortfolioBudget,
) -> Result<VerifiedQbfDagEvidencePortfolio, StrategyPortfolioError> {
    let dag_verification = verify_qbf_game_proof_dag(problem, prefix, proof, dag_budget)
        .map_err(StrategyPortfolioError::GameDag)?;
    if dag_verification.root_value != candidate.value {
        return Err(StrategyPortfolioError::GameDagResultMismatch {
            proof_value: dag_verification.root_value,
            candidate_value: candidate.value,
        });
    }

    let dag_bytes = encode_qbf_game_proof_dag(proof, encoding_budget)?;
    let decoded_dag = decode_qbf_game_proof_dag(&dag_bytes, dag_budget, encoding_budget)?;
    if decoded_dag != *proof {
        return Err(StrategyPortfolioError::ProofEncodingRoundTripMismatch);
    }
    let decoded_verification = verify_qbf_game_proof_dag(problem, prefix, &decoded_dag, dag_budget)
        .map_err(StrategyPortfolioError::GameDag)?;
    if decoded_verification != dag_verification {
        return Err(StrategyPortfolioError::ProofEncodingRoundTripMismatch);
    }

    let strategy_portfolio = verify_and_compile_qbf_strategy_portfolio(
        problem,
        prefix,
        candidate,
        qbf_budget,
        bdd_budget,
        aig_budget,
        encoding_budget,
    )?;
    let circuit_bytes = strategy_portfolio
        .total_bdd_bytes
        .checked_add(strategy_portfolio.total_aig_bytes)
        .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
    let total_evidence_bytes = dag_bytes
        .len()
        .checked_add(circuit_bytes)
        .ok_or(StrategyPortfolioError::ResourceAccountingOverflow)?;
    if total_evidence_bytes > encoding_budget.max_total_representation_bytes {
        return Err(StrategyPortfolioError::TotalPortfolioTooLarge {
            required: total_evidence_bytes,
            limit: encoding_budget.max_total_representation_bytes,
        });
    }

    Ok(VerifiedQbfDagEvidencePortfolio {
        value: decoded_verification.root_value,
        game_proof_dag: decoded_dag,
        game_proof_dag_verification: decoded_verification,
        game_proof_dag_bytes: dag_bytes,
        strategy_portfolio,
        total_evidence_bytes,
    })
}

pub fn verify_and_compile_qbf_strategy_portfolio(
    problem: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    candidate: &QbfEvaluationResult,
    qbf_budget: QuantifiedEvaluationBudget,
    bdd_budget: StrategyBddBudget,
    aig_budget: StrategyAigBudget,
    encoding_budget: StrategyPortfolioBudget,
) -> Result<VerifiedStrategyPortfolio, StrategyPortfolioError> {
    let verified =
        verify_and_compile_qbf_strategy(problem, prefix, candidate, qbf_budget, bdd_budget)
            .map_err(StrategyPortfolioError::SourceOrBddCompilation)?;

    let mut functions = Vec::with_capacity(verified.representation.functions.len());
    let mut total_bdd_bytes = 0usize;
    let mut total_aig_bytes = 0usize;
    let mut total_selected_bytes = 0usize;
    let mut total_bdd_nodes = 0usize;
    let mut total_aig_and_nodes = 0usize;
    let mut total_bdd_estimated_verification_work = 0u128;
    let mut total_aig_estimated_verification_work = 0u128;
    let mut total_selected_estimated_verification_work = 0u128;

    for compiled in &verified.representation.functions {
        let (bdd_bytes, bdd_verification) = (
            encode_strategy_robdd(&compiled.bdd, bdd_budget, encoding_budget)?,
            verify_robdd_matches_strategy_function(
                &compiled.bdd,
                &compiled.reduced_function,
                bdd_budget,
            )
            .map_err(StrategyPortfolioError::BddVerification)?,
        );

        let aig = compile_strategy_function_to_aig_with_order(
            &compiled.reduced_function,
            &compiled.bdd.variable_order,
            aig_budget,
        )
        .map_err(StrategyPortfolioError::AigCompilation)?;
        let aig_verification =
            verify_aig_matches_strategy_function(&aig, &compiled.reduced_function, aig_budget)
                .map_err(StrategyPortfolioError::AigVerification)?;
        let aig_bytes = encode_strategy_aig(&aig, aig_budget, encoding_budget)?;

        let bdd_cost = bdd_representation_cost(
            &compiled.bdd,
            bdd_bytes.len(),
            bdd_verification.assignments_checked,
        )?;
        let aig_cost = aig_representation_cost(
            &aig,
            aig_bytes.len(),
            aig_verification.assignments_checked,
            aig_verification.and_nodes,
        )?;
        let pareto_frontier = strategy_representation_pareto_frontier(bdd_cost, aig_cost);
        let (selected, selected_cost) =
            select_smallest_pareto_encoding(&pareto_frontier, bdd_cost, aig_cost);
        let selected_bytes = selected_cost.encoded_bytes;

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
        total_bdd_estimated_verification_work = checked_cost_add(
            total_bdd_estimated_verification_work,
            bdd_cost.estimated_verification_work_units,
        )?;
        total_aig_estimated_verification_work = checked_cost_add(
            total_aig_estimated_verification_work,
            aig_cost.estimated_verification_work_units,
        )?;
        total_selected_estimated_verification_work = checked_cost_add(
            total_selected_estimated_verification_work,
            selected_cost.estimated_verification_work_units,
        )?;

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
            bdd_cost,
            aig_cost,
            pareto_frontier,
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
        total_bdd_estimated_verification_work,
        total_aig_estimated_verification_work,
        total_selected_estimated_verification_work,
        bdd_search_complete: verified.representation.optimization_search_complete,
    })
}

#[cfg(test)]
mod tests {
    use super::super::decomposition::Literal;
    use super::super::quantified_proof::{QuantifiedProofBudget, build_qbf_game_proof};
    use super::super::quantified_scope::{Quantifier, QuantifierBlock, ValidatedQuantifierPrefix};
    use super::super::quantified_strategy::evaluate_qbf_with_strategy;
    use super::*;

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
            max_anf_transform_checks: 1 << 8,
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

    fn dag_budget() -> QbfGameDagBudget {
        QbfGameDagBudget {
            max_variables: 4,
            max_total_variable_name_bytes: 64,
            max_clauses: 16,
            max_literals: 64,
            max_proof_nodes: 1 << 8,
            max_builder_state_visits: 1 << 8,
            max_verifier_node_visits: 1 << 8,
            max_residual_literal_visits: 1 << 12,
            max_stored_residual_literals: 1 << 12,
            max_stored_residual_clauses: 1 << 10,
        }
    }

    #[test]
    fn dag_proof_and_strategy_portfolio_roundtrip_as_source_bound_evidence() {
        let (problem, prefix) = equality_qbf();
        let candidate = evaluate_qbf_with_strategy(&problem, &prefix, qbf_budget()).unwrap();
        let (proof, _) = super::super::quantified_game_dag::build_qbf_game_proof_dag(
            &problem,
            &prefix,
            dag_budget(),
        )
        .unwrap();
        let portfolio = verify_and_compile_qbf_dag_evidence_portfolio(
            &problem,
            &prefix,
            &proof,
            &candidate,
            qbf_budget(),
            dag_budget(),
            bdd_budget(),
            aig_budget(),
            encoding_budget(),
        )
        .unwrap();
        assert!(portfolio.value);
        assert_eq!(portfolio.game_proof_dag, proof);
        assert_eq!(
            portfolio.game_proof_dag_verification.root_value,
            candidate.value
        );
        assert!(!portfolio.game_proof_dag_bytes.is_empty());
        assert!(portfolio.total_evidence_bytes >= portfolio.game_proof_dag_bytes.len());
    }

    #[test]
    fn dag_proof_encoder_rejects_unknown_tags_and_trailing_bytes() {
        let (problem, prefix) = equality_qbf();
        let (proof, _) = super::super::quantified_game_dag::build_qbf_game_proof_dag(
            &problem,
            &prefix,
            dag_budget(),
        )
        .unwrap();
        let mut bytes = encode_qbf_game_proof_dag(&proof, encoding_budget()).unwrap();
        bytes.push(0);
        assert!(matches!(
            decode_qbf_game_proof_dag(&bytes, dag_budget(), encoding_budget()),
            Err(StrategyPortfolioError::ProofEncodingTrailingBytes { .. })
        ));
    }

    #[test]
    fn dag_evidence_portfolio_rejects_wrong_candidate_value() {
        let (problem, prefix) = equality_qbf();
        let mut candidate = evaluate_qbf_with_strategy(&problem, &prefix, qbf_budget()).unwrap();
        candidate.value = false;
        let (proof, _) = super::super::quantified_game_dag::build_qbf_game_proof_dag(
            &problem,
            &prefix,
            dag_budget(),
        )
        .unwrap();
        assert!(matches!(
            verify_and_compile_qbf_dag_evidence_portfolio(
                &problem,
                &prefix,
                &proof,
                &candidate,
                qbf_budget(),
                dag_budget(),
                bdd_budget(),
                aig_budget(),
                encoding_budget(),
            ),
            Err(StrategyPortfolioError::GameDagResultMismatch { .. })
        ));
    }

    #[test]
    fn representation_pareto_frontier_preserves_incomparable_tradeoffs() {
        let smaller_and_fast_verifier = StrategyRepresentationCost {
            encoded_bytes: 40,
            internal_nodes: 4,
            logic_depth: 4,
            worst_case_evaluation_steps: 4,
            estimated_verification_work_units: 100,
        };
        let larger_but_shallower = StrategyRepresentationCost {
            encoded_bytes: 60,
            internal_nodes: 7,
            logic_depth: 2,
            worst_case_evaluation_steps: 9,
            estimated_verification_work_units: 150,
        };
        assert_eq!(
            strategy_representation_pareto_frontier(
                smaller_and_fast_verifier,
                larger_but_shallower
            ),
            vec![
                SelectedStrategyRepresentation::Robdd,
                SelectedStrategyRepresentation::Aig
            ]
        );
    }

    #[test]
    fn representation_pareto_frontier_removes_a_dominated_encoding() {
        let better = StrategyRepresentationCost {
            encoded_bytes: 20,
            internal_nodes: 2,
            logic_depth: 2,
            worst_case_evaluation_steps: 2,
            estimated_verification_work_units: 20,
        };
        let dominated = StrategyRepresentationCost {
            encoded_bytes: 30,
            internal_nodes: 4,
            logic_depth: 4,
            worst_case_evaluation_steps: 8,
            estimated_verification_work_units: 100,
        };
        assert_eq!(
            strategy_representation_pareto_frontier(better, dominated),
            vec![SelectedStrategyRepresentation::Robdd]
        );
        assert_eq!(
            strategy_representation_pareto_frontier(dominated, better),
            vec![SelectedStrategyRepresentation::Aig]
        );
    }

    #[test]
    fn exact_representation_cost_tie_uses_canonical_robdd_preference() {
        let equal = StrategyRepresentationCost {
            encoded_bytes: 32,
            internal_nodes: 3,
            logic_depth: 2,
            worst_case_evaluation_steps: 2,
            estimated_verification_work_units: 12,
        };
        assert_eq!(
            strategy_representation_pareto_frontier(equal, equal),
            vec![SelectedStrategyRepresentation::Robdd]
        );
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
            portfolio
                .source_strategy_verification
                .opponent_assignments_checked,
            2
        );
        assert_eq!(portfolio.functions.len(), 1);
        assert_eq!(portfolio.functions[0].variable, 1);
        assert_eq!(portfolio.functions[0].reduced_dependencies, vec![0]);
        assert_eq!(
            portfolio.functions[0].bdd_verification.assignments_checked,
            2
        );
        assert_eq!(
            portfolio.functions[0].aig_verification.assignments_checked,
            2
        );
        assert_eq!(
            portfolio.functions[0].selected_bytes,
            portfolio.functions[0]
                .bdd_bytes
                .len()
                .min(portfolio.functions[0].aig_bytes.len())
        );
        assert!(
            portfolio.functions[0]
                .pareto_frontier
                .contains(&portfolio.functions[0].selected)
        );
        assert!(portfolio.functions[0].bdd_cost.encoded_bytes > 0);
        assert!(portfolio.functions[0].aig_cost.encoded_bytes > 0);
        assert_eq!(
            portfolio.total_selected_estimated_verification_work,
            match portfolio.functions[0].selected {
                SelectedStrategyRepresentation::Robdd => {
                    portfolio.functions[0]
                        .bdd_cost
                        .estimated_verification_work_units
                }
                SelectedStrategyRepresentation::Aig => {
                    portfolio.functions[0]
                        .aig_cost
                        .estimated_verification_work_units
                }
            }
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
        result.strategy.functions[0].table.insert(vec![false], true);
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
    fn game_proof_and_strategy_are_bound_in_one_evidence_portfolio() {
        let (problem, prefix) = equality_qbf();
        let candidate = evaluate_qbf_with_strategy(&problem, &prefix, qbf_budget()).unwrap();
        let (proof, _) =
            build_qbf_game_proof(&problem, &prefix, QuantifiedProofBudget::default()).unwrap();
        let portfolio = verify_and_compile_qbf_evidence_portfolio(
            &problem,
            &prefix,
            &proof,
            &candidate,
            qbf_budget(),
            QuantifiedProofBudget::default(),
            bdd_budget(),
            aig_budget(),
            encoding_budget(),
        )
        .unwrap();

        assert!(portfolio.value);
        assert!(portfolio.game_proof_verification.root_value);
        assert_eq!(
            portfolio.game_proof_verification.proof_nodes_checked,
            proof.nodes.len()
        );
        assert!(portfolio.game_proof_bytes.starts_with(b"LD-QBFPROOF1"));
        assert_eq!(portfolio.strategy_portfolio.value, portfolio.value);
        assert_eq!(
            portfolio.total_evidence_bytes,
            portfolio.game_proof_bytes.len()
                + portfolio.strategy_portfolio.total_bdd_bytes
                + portfolio.strategy_portfolio.total_aig_bytes
        );
    }

    #[test]
    fn proof_encoding_round_trips_and_independently_verifies_decoded_artifact() {
        let (problem, prefix) = equality_qbf();
        let candidate = evaluate_qbf_with_strategy(&problem, &prefix, qbf_budget()).unwrap();
        let (proof, _) =
            build_qbf_game_proof(&problem, &prefix, QuantifiedProofBudget::default()).unwrap();
        let bytes = encode_qbf_game_proof(&proof, encoding_budget()).unwrap();
        let decoded =
            decode_qbf_game_proof(&bytes, QuantifiedProofBudget::default(), encoding_budget())
                .unwrap();
        assert_eq!(decoded, proof);
        let checked = super::super::quantified_proof::verify_qbf_game_proof(
            &problem,
            &prefix,
            &decoded,
            QuantifiedProofBudget::default(),
        )
        .unwrap();
        assert_eq!(checked.root_value, candidate.value);
    }

    #[test]
    fn proof_decoder_rejects_truncation_unknown_tag_bad_boolean_and_trailing_bytes() {
        let (problem, prefix) = equality_qbf();
        let (proof, _) =
            build_qbf_game_proof(&problem, &prefix, QuantifiedProofBudget::default()).unwrap();
        let bytes = encode_qbf_game_proof(&proof, encoding_budget()).unwrap();

        assert!(matches!(
            decode_qbf_game_proof(
                &bytes[..bytes.len() - 1],
                QuantifiedProofBudget::default(),
                encoding_budget()
            ),
            Err(StrategyPortfolioError::ProofEncodingInvalid { .. })
        ));

        let mut unknown_tag = bytes.clone();
        // Header (12 bytes) + variable_count/root/node_count (24 bytes), then
        // one node's prefix position (8 bytes) and value (1 byte).
        unknown_tag[45] = 255;
        assert!(matches!(
            decode_qbf_game_proof(
                &unknown_tag,
                QuantifiedProofBudget::default(),
                encoding_budget()
            ),
            Err(StrategyPortfolioError::ProofEncodingUnknownNodeTag { .. })
        ));

        let mut bad_boolean = bytes.clone();
        bad_boolean[44] = 2;
        assert!(matches!(
            decode_qbf_game_proof(
                &bad_boolean,
                QuantifiedProofBudget::default(),
                encoding_budget()
            ),
            Err(StrategyPortfolioError::ProofEncodingInvalid { .. })
        ));

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(matches!(
            decode_qbf_game_proof(
                &trailing,
                QuantifiedProofBudget::default(),
                encoding_budget()
            ),
            Err(StrategyPortfolioError::ProofEncodingTrailingBytes { remaining: 1 })
        ));
    }

    #[test]
    fn proof_decoder_checks_variable_node_and_byte_budgets_before_allocation() {
        let (problem, prefix) = equality_qbf();
        let (proof, _) =
            build_qbf_game_proof(&problem, &prefix, QuantifiedProofBudget::default()).unwrap();
        let bytes = encode_qbf_game_proof(&proof, encoding_budget()).unwrap();
        let small_nodes = QuantifiedProofBudget {
            max_proof_nodes: 1,
            ..QuantifiedProofBudget::default()
        };
        assert!(matches!(
            decode_qbf_game_proof(&bytes, small_nodes, encoding_budget()),
            Err(StrategyPortfolioError::ProofEncodingInvalid { .. })
        ));
        let small_bytes = StrategyPortfolioBudget {
            max_single_representation_bytes: 8,
            max_total_representation_bytes: 1 << 18,
        };
        assert!(matches!(
            decode_qbf_game_proof(&bytes, QuantifiedProofBudget::default(), small_bytes),
            Err(StrategyPortfolioError::RepresentationTooLarge { .. })
        ));
    }

    #[test]
    fn game_proof_mutation_is_rejected_before_strategy_compilation() {
        let (problem, prefix) = equality_qbf();
        let candidate = evaluate_qbf_with_strategy(&problem, &prefix, qbf_budget()).unwrap();
        let (proof, _) =
            build_qbf_game_proof(&problem, &prefix, QuantifiedProofBudget::default()).unwrap();
        let mut tampered = proof.clone();
        tampered.nodes[tampered.root].value = false;
        assert!(matches!(
            verify_and_compile_qbf_evidence_portfolio(
                &problem,
                &prefix,
                &tampered,
                &candidate,
                qbf_budget(),
                QuantifiedProofBudget::default(),
                bdd_budget(),
                aig_budget(),
                encoding_budget(),
            ),
            Err(StrategyPortfolioError::GameProof(
                super::super::quantified_proof::QuantifiedProofError::IncorrectNodeValue { .. }
            ))
        ));
    }

    #[test]
    fn game_proof_and_candidate_truth_must_agree() {
        let (problem, prefix) = equality_qbf();
        let mut candidate = evaluate_qbf_with_strategy(&problem, &prefix, qbf_budget()).unwrap();
        let (proof, _) =
            build_qbf_game_proof(&problem, &prefix, QuantifiedProofBudget::default()).unwrap();
        candidate.value = false;
        assert_eq!(
            verify_and_compile_qbf_evidence_portfolio(
                &problem,
                &prefix,
                &proof,
                &candidate,
                qbf_budget(),
                QuantifiedProofBudget::default(),
                bdd_budget(),
                aig_budget(),
                encoding_budget(),
            )
            .unwrap_err(),
            StrategyPortfolioError::GameProofResultMismatch {
                proof_value: true,
                candidate_value: false,
            }
        );
    }

    #[test]
    fn proof_encoding_is_deterministic_and_single_representation_bounded() {
        let (problem, prefix) = equality_qbf();
        let candidate = evaluate_qbf_with_strategy(&problem, &prefix, qbf_budget()).unwrap();
        let (proof, _) =
            build_qbf_game_proof(&problem, &prefix, QuantifiedProofBudget::default()).unwrap();
        let first = encode_qbf_game_proof(&proof, encoding_budget()).unwrap();
        let second = encode_qbf_game_proof(&proof, encoding_budget()).unwrap();
        assert_eq!(first, second);

        let too_small = StrategyPortfolioBudget {
            max_single_representation_bytes: 8,
            max_total_representation_bytes: 1 << 18,
        };
        assert!(matches!(
            verify_and_compile_qbf_evidence_portfolio(
                &problem,
                &prefix,
                &proof,
                &candidate,
                qbf_budget(),
                QuantifiedProofBudget::default(),
                bdd_budget(),
                aig_budget(),
                too_small,
            ),
            Err(StrategyPortfolioError::RepresentationTooLarge { .. })
        ));
    }

    #[test]
    fn empty_strategy_serializes_and_selects_an_empty_portfolio() {
        let problem = CnfProblem {
            variables: vec![],
            clauses: vec![],
        };
        let prefix = ValidatedQuantifierPrefix::new(0, vec![], Default::default()).unwrap();
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
