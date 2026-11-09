// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Versioned, bounded serialization for SynQBF paired-variable AIG proofs.
//!
//! The artifact carries the ordered quantifier prefix and the paired-input
//! AIG. It does not carry or replace the source matrix: acceptance requires
//! decoding the artifact and independently verifying it against the original
//! CNF and validated prefix with verify_synqbf_proof.
//!
//! Format v1 is deterministic little-endian data with explicit tags, lengths,
//! strict trailing-byte rejection and pre-allocation count checks.

use super::decomposition::CnfProblem;
use super::quantified_scope::{Quantifier, ValidatedQuantifierPrefix};
use super::strategy_aig::{AigLiteral, AigNode, StrategyAig, validate_strategy_aig};
use super::synqbf::{
    SynQbfBudget, SynQbfError, SynQbfProof, SynQbfVerificationReport, verify_synqbf_proof,
};

const MAGIC: &[u8; 8] = b"LDSYNQBF";
const VERSION: u16 = 1;

/// Additional resource limits for portable SynQBF proof artifacts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynQbfArtifactBudget {
    pub max_artifact_bytes: usize,
    pub max_aig_nodes: usize,
    pub proof: SynQbfBudget,
}

impl Default for SynQbfArtifactBudget {
    fn default() -> Self {
        Self {
            max_artifact_bytes: 64 * 1024 * 1024,
            max_aig_nodes: 1 << 21,
            proof: SynQbfBudget::default(),
        }
    }
}

/// Fail-closed format, structure and semantic-verification errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynQbfArtifactError {
    ArtifactByteBudgetExceeded {
        required: usize,
        limit: usize,
    },
    EncodingByteBudgetExceeded {
        required: usize,
        limit: usize,
    },
    BadMagic,
    UnsupportedVersion {
        version: u16,
    },
    TruncatedArtifact {
        offset: usize,
        required: usize,
    },
    TrailingBytes {
        remaining: usize,
    },
    ResourceAccountingOverflow,
    IntegerOutOfRange {
        field: &'static str,
        value: u64,
    },
    VariableBudgetExceeded {
        required: usize,
        limit: usize,
    },
    PrefixCountMismatch {
        expected: usize,
        actual: usize,
    },
    InvalidBooleanTag {
        offset: usize,
        tag: u8,
    },
    InvalidQuantifierTag {
        offset: usize,
        tag: u8,
    },
    PrefixVariableOutOfRange {
        position: usize,
        variable: usize,
        variable_count: usize,
    },
    DuplicatePrefixVariable {
        variable: usize,
    },
    DependencyCountMismatch {
        expected: usize,
        actual: usize,
    },
    AigDependencyBudgetExceeded {
        required: usize,
        limit: usize,
    },
    DependencyOrderMismatch {
        position: usize,
        expected: usize,
        actual: usize,
    },
    DecompositionOrderCountMismatch {
        expected: usize,
        actual: usize,
    },
    AigNodeBudgetExceeded {
        required: usize,
        limit: usize,
    },
    AigAndNodeBudgetExceeded {
        required: usize,
        limit: usize,
    },
    UnknownAigNodeTag {
        offset: usize,
        tag: u8,
    },
    UnknownAigLiteralTag {
        offset: usize,
        tag: u8,
    },
    CircuitValidation(super::strategy_aig::StrategyAigError),
    SemanticVerification(SynQbfError),
}

/// A decoded proof together with independent source-bound verification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedSynQbfArtifact {
    pub proof: SynQbfProof,
    pub verification: SynQbfVerificationReport,
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 0 }
    }

    fn take(&mut self, count: usize) -> Result<&'a [u8], SynQbfArtifactError> {
        let end = self
            .offset
            .checked_add(count)
            .ok_or(SynQbfArtifactError::ResourceAccountingOverflow)?;
        if end > self.bytes.len() {
            return Err(SynQbfArtifactError::TruncatedArtifact {
                offset: self.offset,
                required: count,
            });
        }
        let value = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(value)
    }

    fn u8(&mut self) -> Result<u8, SynQbfArtifactError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, SynQbfArtifactError> {
        let bytes = self.take(2)?;
        Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
    }

    fn u64(&mut self) -> Result<u64, SynQbfArtifactError> {
        let bytes = self.take(8)?;
        Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn usize(&mut self, field: &'static str) -> Result<usize, SynQbfArtifactError> {
        let value = self.u64()?;
        usize::try_from(value).map_err(|_| SynQbfArtifactError::IntegerOutOfRange { field, value })
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.offset)
    }
}

fn append(
    output: &mut Vec<u8>,
    bytes: &[u8],
    budget: SynQbfArtifactBudget,
) -> Result<(), SynQbfArtifactError> {
    let required = output
        .len()
        .checked_add(bytes.len())
        .ok_or(SynQbfArtifactError::ResourceAccountingOverflow)?;
    if required > budget.max_artifact_bytes {
        return Err(SynQbfArtifactError::EncodingByteBudgetExceeded {
            required,
            limit: budget.max_artifact_bytes,
        });
    }
    output.extend_from_slice(bytes);
    Ok(())
}

fn push_u8(
    output: &mut Vec<u8>,
    value: u8,
    budget: SynQbfArtifactBudget,
) -> Result<(), SynQbfArtifactError> {
    append(output, &[value], budget)
}

fn push_u16(
    output: &mut Vec<u8>,
    value: u16,
    budget: SynQbfArtifactBudget,
) -> Result<(), SynQbfArtifactError> {
    append(output, &value.to_le_bytes(), budget)
}

fn push_u64(
    output: &mut Vec<u8>,
    value: usize,
    budget: SynQbfArtifactBudget,
) -> Result<(), SynQbfArtifactError> {
    let value =
        u64::try_from(value).map_err(|_| SynQbfArtifactError::ResourceAccountingOverflow)?;
    append(output, &value.to_le_bytes(), budget)
}

fn encode_literal(
    output: &mut Vec<u8>,
    literal: AigLiteral,
    budget: SynQbfArtifactBudget,
) -> Result<(), SynQbfArtifactError> {
    match literal {
        AigLiteral::Constant(false) => push_u8(output, 0, budget),
        AigLiteral::Constant(true) => push_u8(output, 1, budget),
        AigLiteral::Node {
            index,
            inverted: false,
        } => {
            push_u8(output, 2, budget)?;
            push_u64(output, index, budget)
        }
        AigLiteral::Node {
            index,
            inverted: true,
        } => {
            push_u8(output, 3, budget)?;
            push_u64(output, index, budget)
        }
    }
}

fn decode_literal(reader: &mut Reader<'_>) -> Result<AigLiteral, SynQbfArtifactError> {
    let offset = reader.offset;
    match reader.u8()? {
        0 => Ok(AigLiteral::FALSE),
        1 => Ok(AigLiteral::TRUE),
        2 => Ok(AigLiteral::node(reader.usize("AIG literal node index")?)),
        3 => Ok(AigLiteral::node(reader.usize("AIG literal node index")?).not()),
        tag => Err(SynQbfArtifactError::UnknownAigLiteralTag { offset, tag }),
    }
}

fn validate_prefix_order(
    variable_count: usize,
    prefix_order: &[(Quantifier, usize)],
) -> Result<(), SynQbfArtifactError> {
    if prefix_order.len() != variable_count {
        return Err(SynQbfArtifactError::PrefixCountMismatch {
            expected: variable_count,
            actual: prefix_order.len(),
        });
    }
    let mut seen = vec![false; variable_count];
    for (position, &(_, variable)) in prefix_order.iter().enumerate() {
        if variable >= variable_count {
            return Err(SynQbfArtifactError::PrefixVariableOutOfRange {
                position,
                variable,
                variable_count,
            });
        }
        if seen[variable] {
            return Err(SynQbfArtifactError::DuplicatePrefixVariable { variable });
        }
        seen[variable] = true;
    }
    Ok(())
}

fn validate_proof_structure(
    proof: &SynQbfProof,
    budget: SynQbfArtifactBudget,
) -> Result<(), SynQbfArtifactError> {
    if proof.variable_count > budget.proof.max_variables {
        return Err(SynQbfArtifactError::VariableBudgetExceeded {
            required: proof.variable_count,
            limit: budget.proof.max_variables,
        });
    }
    validate_prefix_order(proof.variable_count, &proof.prefix_order)?;

    let expected_inputs = proof
        .variable_count
        .checked_mul(2)
        .ok_or(SynQbfArtifactError::ResourceAccountingOverflow)?;
    if proof.circuit.dependencies.len() != expected_inputs {
        return Err(SynQbfArtifactError::DependencyCountMismatch {
            expected: expected_inputs,
            actual: proof.circuit.dependencies.len(),
        });
    }
    for (position, &actual) in proof.circuit.dependencies.iter().enumerate() {
        if actual != position {
            return Err(SynQbfArtifactError::DependencyOrderMismatch {
                position,
                expected: position,
                actual,
            });
        }
    }
    if proof.circuit.nodes.len() > budget.max_aig_nodes {
        return Err(SynQbfArtifactError::AigNodeBudgetExceeded {
            required: proof.circuit.nodes.len(),
            limit: budget.max_aig_nodes,
        });
    }
    validate_strategy_aig(&proof.circuit, budget.proof.aig)
        .map_err(SynQbfArtifactError::CircuitValidation)
}

fn decode_quantifier(reader: &mut Reader<'_>) -> Result<Quantifier, SynQbfArtifactError> {
    let offset = reader.offset;
    match reader.u8()? {
        0 => Ok(Quantifier::ForAll),
        1 => Ok(Quantifier::Exists),
        tag => Err(SynQbfArtifactError::InvalidQuantifierTag { offset, tag }),
    }
}

/// Encode a structurally valid SynQBF proof in deterministic v1 format.
///
/// This operation validates the prefix and AIG structure, but does not assert
/// that the proof is semantically valid for any particular source matrix.
pub fn encode_synqbf_proof(
    proof: &SynQbfProof,
    budget: SynQbfArtifactBudget,
) -> Result<Vec<u8>, SynQbfArtifactError> {
    validate_proof_structure(proof, budget)?;
    let mut output = Vec::new();
    append(&mut output, MAGIC, budget)?;
    push_u16(&mut output, VERSION, budget)?;
    push_u64(&mut output, proof.variable_count, budget)?;
    push_u8(&mut output, u8::from(proof.claimed_truth_value), budget)?;

    push_u64(&mut output, proof.prefix_order.len(), budget)?;
    for &(quantifier, variable) in &proof.prefix_order {
        push_u8(
            &mut output,
            match quantifier {
                Quantifier::ForAll => 0,
                Quantifier::Exists => 1,
            },
            budget,
        )?;
        push_u64(&mut output, variable, budget)?;
    }

    push_u64(&mut output, proof.circuit.dependencies.len(), budget)?;
    for &dependency in &proof.circuit.dependencies {
        push_u64(&mut output, dependency, budget)?;
    }
    push_u64(&mut output, proof.circuit.decomposition_order.len(), budget)?;
    for &dependency in &proof.circuit.decomposition_order {
        push_u64(&mut output, dependency, budget)?;
    }

    push_u64(&mut output, proof.circuit.nodes.len(), budget)?;
    let mut and_nodes = 0usize;
    for node in &proof.circuit.nodes {
        match *node {
            AigNode::Input { dependency_index } => {
                push_u8(&mut output, 0, budget)?;
                push_u64(&mut output, dependency_index, budget)?;
            }
            AigNode::And { left, right } => {
                and_nodes = and_nodes
                    .checked_add(1)
                    .ok_or(SynQbfArtifactError::ResourceAccountingOverflow)?;
                if and_nodes > budget.proof.aig.max_and_nodes {
                    return Err(SynQbfArtifactError::AigAndNodeBudgetExceeded {
                        required: and_nodes,
                        limit: budget.proof.aig.max_and_nodes,
                    });
                }
                push_u8(&mut output, 1, budget)?;
                encode_literal(&mut output, left, budget)?;
                encode_literal(&mut output, right, budget)?;
            }
        }
    }
    encode_literal(&mut output, proof.circuit.output, budget)?;
    Ok(output)
}

/// Decode a v1 proof after checking byte, count and graph-structure budgets.
///
/// Source-matrix equivalence, representability and root truth still require
/// verify_synqbf_artifact with the original CNF/prefix.
pub fn decode_synqbf_proof(
    bytes: &[u8],
    budget: SynQbfArtifactBudget,
) -> Result<SynQbfProof, SynQbfArtifactError> {
    if bytes.len() > budget.max_artifact_bytes {
        return Err(SynQbfArtifactError::ArtifactByteBudgetExceeded {
            required: bytes.len(),
            limit: budget.max_artifact_bytes,
        });
    }

    let mut reader = Reader::new(bytes);
    if reader.take(MAGIC.len())? != MAGIC {
        return Err(SynQbfArtifactError::BadMagic);
    }
    let version = reader.u16()?;
    if version != VERSION {
        return Err(SynQbfArtifactError::UnsupportedVersion { version });
    }

    let variable_count = reader.usize("variable count")?;
    if variable_count > budget.proof.max_variables {
        return Err(SynQbfArtifactError::VariableBudgetExceeded {
            required: variable_count,
            limit: budget.proof.max_variables,
        });
    }
    let truth_tag_offset = reader.offset;
    let claimed_truth_value = match reader.u8()? {
        0 => false,
        1 => true,
        tag => {
            return Err(SynQbfArtifactError::InvalidBooleanTag {
                offset: truth_tag_offset,
                tag,
            });
        }
    };

    let prefix_count = reader.usize("prefix count")?;
    if prefix_count != variable_count {
        return Err(SynQbfArtifactError::PrefixCountMismatch {
            expected: variable_count,
            actual: prefix_count,
        });
    }
    let mut prefix_order = Vec::with_capacity(prefix_count);
    let mut prefix_seen = vec![false; variable_count];
    for position in 0..prefix_count {
        let quantifier = decode_quantifier(&mut reader)?;
        let variable = reader.usize("prefix variable")?;
        if variable >= variable_count {
            return Err(SynQbfArtifactError::PrefixVariableOutOfRange {
                position,
                variable,
                variable_count,
            });
        }
        if prefix_seen[variable] {
            return Err(SynQbfArtifactError::DuplicatePrefixVariable { variable });
        }
        prefix_seen[variable] = true;
        prefix_order.push((quantifier, variable));
    }

    let expected_inputs = variable_count
        .checked_mul(2)
        .ok_or(SynQbfArtifactError::ResourceAccountingOverflow)?;
    if expected_inputs > budget.proof.aig.max_dependency_variables {
        return Err(SynQbfArtifactError::AigDependencyBudgetExceeded {
            required: expected_inputs,
            limit: budget.proof.aig.max_dependency_variables,
        });
    }

    let dependency_count = reader.usize("AIG dependency count")?;
    if dependency_count != expected_inputs {
        return Err(SynQbfArtifactError::DependencyCountMismatch {
            expected: expected_inputs,
            actual: dependency_count,
        });
    }
    let mut dependencies = Vec::with_capacity(dependency_count);
    for position in 0..dependency_count {
        let actual = reader.usize("AIG dependency")?;
        if actual != position {
            return Err(SynQbfArtifactError::DependencyOrderMismatch {
                position,
                expected: position,
                actual,
            });
        }
        dependencies.push(actual);
    }

    let decomposition_count = reader.usize("AIG decomposition order count")?;
    if decomposition_count != dependency_count {
        return Err(SynQbfArtifactError::DecompositionOrderCountMismatch {
            expected: dependency_count,
            actual: decomposition_count,
        });
    }
    let mut decomposition_order = Vec::with_capacity(decomposition_count);
    for _ in 0..decomposition_count {
        decomposition_order.push(reader.usize("AIG decomposition dependency")?);
    }

    let node_count = reader.usize("AIG node count")?;
    if node_count > budget.max_aig_nodes {
        return Err(SynQbfArtifactError::AigNodeBudgetExceeded {
            required: node_count,
            limit: budget.max_aig_nodes,
        });
    }
    // An AND node is the shortest possible record: one tag plus two
    // one-byte constant literals. Reserve one byte for the output literal.
    // Reject impossible node counts before Vec::with_capacity can allocate.
    let minimum_node_bytes = node_count
        .checked_mul(3)
        .and_then(|bytes| bytes.checked_add(1))
        .ok_or(SynQbfArtifactError::ResourceAccountingOverflow)?;
    if minimum_node_bytes > reader.remaining() {
        return Err(SynQbfArtifactError::TruncatedArtifact {
            offset: reader.offset,
            required: minimum_node_bytes,
        });
    }
    let mut nodes = Vec::with_capacity(node_count);
    let mut and_nodes = 0usize;
    for _ in 0..node_count {
        let offset = reader.offset;
        match reader.u8()? {
            0 => nodes.push(AigNode::Input {
                dependency_index: reader.usize("AIG input dependency")?,
            }),
            1 => {
                and_nodes = and_nodes
                    .checked_add(1)
                    .ok_or(SynQbfArtifactError::ResourceAccountingOverflow)?;
                if and_nodes > budget.proof.aig.max_and_nodes {
                    return Err(SynQbfArtifactError::AigAndNodeBudgetExceeded {
                        required: and_nodes,
                        limit: budget.proof.aig.max_and_nodes,
                    });
                }
                nodes.push(AigNode::And {
                    left: decode_literal(&mut reader)?,
                    right: decode_literal(&mut reader)?,
                });
            }
            tag => return Err(SynQbfArtifactError::UnknownAigNodeTag { offset, tag }),
        }
    }
    let output = decode_literal(&mut reader)?;
    if reader.remaining() != 0 {
        return Err(SynQbfArtifactError::TrailingBytes {
            remaining: reader.remaining(),
        });
    }

    let proof = SynQbfProof {
        variable_count,
        prefix_order,
        claimed_truth_value,
        circuit: StrategyAig {
            dependencies,
            decomposition_order,
            nodes,
            output,
        },
    };
    validate_proof_structure(&proof, budget)?;
    Ok(proof)
}

/// Decode and independently verify a SynQBF artifact against the original
/// matrix and ordered prefix. This is the acceptance entry point for artifacts.
pub fn verify_synqbf_artifact(
    source: &CnfProblem,
    prefix: &ValidatedQuantifierPrefix,
    bytes: &[u8],
    artifact_budget: SynQbfArtifactBudget,
    verification_budget: SynQbfBudget,
) -> Result<VerifiedSynQbfArtifact, SynQbfArtifactError> {
    let proof = decode_synqbf_proof(bytes, artifact_budget)?;
    let verification = verify_synqbf_proof(source, prefix, &proof, verification_budget)
        .map_err(SynQbfArtifactError::SemanticVerification)?;
    Ok(VerifiedSynQbfArtifact {
        proof,
        verification,
    })
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::super::decomposition::Literal;
    use super::super::qdimacs::{QdimacsParseBudget, parse_qdimacs};
    use super::super::quantified_game_dag::{QbfGameDagBudget, build_qbf_game_proof_dag};
    use super::super::quantified_game_values::extract_game_value_profile_from_dag;
    use super::super::quantified_scope::QuantifierBlock;
    use super::super::quantified_strategy::BooleanStrategyFunction;
    use super::super::strategy_aig::{StrategyAigBudget, compile_strategy_function_to_aig};
    use super::super::synqbf_construction::{
        SynQbfConstructionBudget, construct_synqbf_from_game_value_profile,
    };
    use super::*;

    fn proof_and_source(
        quantifier: Quantifier,
    ) -> (CnfProblem, ValidatedQuantifierPrefix, SynQbfProof) {
        let problem = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![Literal::positive(0)]],
        };
        let prefix = ValidatedQuantifierPrefix::new(
            1,
            vec![QuantifierBlock {
                quantifier,
                variables: vec![0],
            }],
            Default::default(),
        )
        .unwrap();
        // The paired-variable circuit is x. It satisfies all three SynQBF
        // obligations for either quantifier, with a different root truth bit.
        let function = BooleanStrategyFunction {
            variable: 0,
            dependencies: vec![0, 1],
            table: BTreeMap::from([
                (vec![false, false], false),
                (vec![false, true], false),
                (vec![true, false], true),
                (vec![true, true], true),
            ]),
        };
        let circuit =
            compile_strategy_function_to_aig(&function, StrategyAigBudget::default()).unwrap();
        let proof = SynQbfProof {
            variable_count: 1,
            prefix_order: vec![(quantifier, 0)],
            claimed_truth_value: matches!(quantifier, Quantifier::Exists),
            circuit,
        };
        (problem, prefix, proof)
    }

    fn budget() -> SynQbfArtifactBudget {
        SynQbfArtifactBudget {
            max_artifact_bytes: 1 << 16,
            max_aig_nodes: 1 << 12,
            proof: SynQbfBudget {
                max_variables: 8,
                max_total_variable_name_bytes: 256,
                max_clauses: 64,
                max_literals: 256,
                max_circuit_evaluations: 1 << 10,
                max_estimated_aig_node_visits: 1 << 20,
                aig: StrategyAigBudget {
                    max_dependency_variables: 16,
                    max_truth_table_entries: 1 << 12,
                    max_and_nodes: 1 << 12,
                    max_assignments_checked: 1 << 12,
                },
            },
        }
    }

    #[test]
    fn artifact_round_trip_and_source_verification_work_for_both_quantifiers() {
        for quantifier in [Quantifier::Exists, Quantifier::ForAll] {
            let (problem, prefix, proof) = proof_and_source(quantifier);
            let bytes = encode_synqbf_proof(&proof, budget()).unwrap();
            let verified =
                verify_synqbf_artifact(&problem, &prefix, &bytes, budget(), budget().proof)
                    .unwrap();
            assert_eq!(verified.proof, proof);
            assert_eq!(
                verified.verification.claimed_truth_value,
                proof.claimed_truth_value
            );
            assert_eq!(
                verified.verification.source_equivalence_assignments_checked,
                2
            );
            assert_eq!(verified.verification.representability_conditions_checked, 1);
        }
    }

    #[test]
    fn qdimacs_game_dag_synqbf_and_portable_artifact_form_one_vertical_slice() {
        let source = concat!("c exists x: x\n", "p cnf 1 1\n", "e 1 0\n", "1 0\n");
        let parsed = parse_qdimacs(source, QdimacsParseBudget::default()).unwrap();
        let dag_budget = QbfGameDagBudget::default();
        let (dag, _) =
            build_qbf_game_proof_dag(&parsed.problem, &parsed.prefix, dag_budget).unwrap();
        let profile = extract_game_value_profile_from_dag(
            &parsed.problem,
            &parsed.prefix,
            &dag,
            dag_budget,
            SynQbfConstructionBudget::default().game_values,
        )
        .unwrap();
        let constructed = construct_synqbf_from_game_value_profile(
            &parsed.problem,
            &parsed.prefix,
            &profile,
            SynQbfConstructionBudget::default(),
        )
        .unwrap();
        let bytes = encode_synqbf_proof(&constructed.proof, budget()).unwrap();
        let verified = verify_synqbf_artifact(
            &parsed.problem,
            &parsed.prefix,
            &bytes,
            budget(),
            budget().proof,
        )
        .unwrap();

        assert!(verified.proof.claimed_truth_value);
        assert_eq!(verified.proof, constructed.proof);
        assert_eq!(
            verified.verification.source_equivalence_assignments_checked,
            2
        );
        assert_eq!(verified.verification.representability_conditions_checked, 1);
        assert_eq!(parsed.stats.declared_variables, 1);
    }

    #[test]
    fn encoding_is_deterministic() {
        let (_, _, proof) = proof_and_source(Quantifier::Exists);
        assert_eq!(
            encode_synqbf_proof(&proof, budget()).unwrap(),
            encode_synqbf_proof(&proof, budget()).unwrap()
        );
    }

    #[test]
    fn decoder_rejects_truncation_and_trailing_bytes() {
        let (_, _, proof) = proof_and_source(Quantifier::Exists);
        let bytes = encode_synqbf_proof(&proof, budget()).unwrap();
        assert!(matches!(
            decode_synqbf_proof(&bytes[..bytes.len() - 1], budget()),
            Err(SynQbfArtifactError::TruncatedArtifact { .. })
        ));
        let mut trailing = bytes;
        trailing.push(0);
        assert_eq!(
            decode_synqbf_proof(&trailing, budget()).unwrap_err(),
            SynQbfArtifactError::TrailingBytes { remaining: 1 }
        );
    }

    #[test]
    fn decoder_rejects_bad_magic_version_boolean_and_quantifier() {
        let (_, _, proof) = proof_and_source(Quantifier::Exists);
        let bytes = encode_synqbf_proof(&proof, budget()).unwrap();

        let mut bad_magic = bytes.clone();
        bad_magic[0] = b'X';
        assert_eq!(
            decode_synqbf_proof(&bad_magic, budget()).unwrap_err(),
            SynQbfArtifactError::BadMagic
        );

        let mut bad_version = bytes.clone();
        bad_version[8] = 2;
        assert_eq!(
            decode_synqbf_proof(&bad_version, budget()).unwrap_err(),
            SynQbfArtifactError::UnsupportedVersion { version: 2 }
        );

        let mut bad_boolean = bytes.clone();
        bad_boolean[18] = 2;
        assert!(matches!(
            decode_synqbf_proof(&bad_boolean, budget()),
            Err(SynQbfArtifactError::InvalidBooleanTag { .. })
        ));

        let mut bad_quantifier = bytes;
        bad_quantifier[27] = 9;
        assert!(matches!(
            decode_synqbf_proof(&bad_quantifier, budget()),
            Err(SynQbfArtifactError::InvalidQuantifierTag { .. })
        ));
    }

    #[test]
    fn decoder_rejects_noncanonical_dependencies_and_unknown_aig_tag() {
        let (_, _, proof) = proof_and_source(Quantifier::Exists);
        let bytes = encode_synqbf_proof(&proof, budget()).unwrap();

        let mut bad_dependency = bytes.clone();
        bad_dependency[44] = 1;
        assert!(matches!(
            decode_synqbf_proof(&bad_dependency, budget()),
            Err(SynQbfArtifactError::DependencyOrderMismatch { .. })
        ));

        let mut bad_node = bytes;
        // For one variable: prefix 27..36, dependencies 36..60,
        // decomposition order 60..84, node count 84..92, then node tags.
        bad_node[92] = 255;
        assert!(matches!(
            decode_synqbf_proof(&bad_node, budget()),
            Err(SynQbfArtifactError::UnknownAigNodeTag { .. })
        ));
    }

    #[test]
    fn decoder_rejects_impossible_node_count_before_reserving_nodes() {
        let (_, _, proof) = proof_and_source(Quantifier::Exists);
        let mut bytes = encode_synqbf_proof(&proof, budget()).unwrap();
        // The encoded node count starts at byte 84 in this one-variable
        // fixture. Claim more minimum-size records than the artifact holds.
        bytes[84..92].copy_from_slice(&1000u64.to_le_bytes());
        assert!(matches!(
            decode_synqbf_proof(&bytes, budget()),
            Err(SynQbfArtifactError::TruncatedArtifact { .. })
        ));
    }

    #[test]
    fn artifact_byte_and_aig_node_budgets_are_checked() {
        let (_, _, proof) = proof_and_source(Quantifier::Exists);
        let bytes = encode_synqbf_proof(&proof, budget()).unwrap();
        let byte_limited = SynQbfArtifactBudget {
            max_artifact_bytes: bytes.len() - 1,
            ..budget()
        };
        assert!(matches!(
            decode_synqbf_proof(&bytes, byte_limited),
            Err(SynQbfArtifactError::ArtifactByteBudgetExceeded { .. })
        ));

        let node_limited = SynQbfArtifactBudget {
            max_aig_nodes: 1,
            ..budget()
        };
        assert!(matches!(
            decode_synqbf_proof(&bytes, node_limited),
            Err(SynQbfArtifactError::AigNodeBudgetExceeded { .. })
        ));
    }

    #[test]
    fn semantically_wrong_proof_is_rejected_after_successful_decode() {
        let (problem, prefix, proof) = proof_and_source(Quantifier::Exists);
        let bytes = encode_synqbf_proof(&proof, budget()).unwrap();
        let wrong_source = CnfProblem {
            variables: vec!["x".into()],
            clauses: vec![vec![Literal::negative(0)]],
        };
        assert!(matches!(
            verify_synqbf_artifact(&wrong_source, &prefix, &bytes, budget(), budget().proof),
            Err(SynQbfArtifactError::SemanticVerification(
                SynQbfError::SourceEquivalenceCounterexample { .. }
            ))
        ));
        assert!(
            verify_synqbf_artifact(&problem, &prefix, &bytes, budget(), budget().proof).is_ok()
        );
    }
}
