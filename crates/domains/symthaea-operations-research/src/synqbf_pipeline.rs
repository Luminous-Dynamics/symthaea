// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! End-to-end source-bound QDIMACS -> SynQBF artifact pipeline.
//!
//! This module composes existing parser, residual-game DAG, full-history
//! game-value, SynQBF constructor, artifact codec and independent verifier.
//! It does not introduce a new proof calculus or make the artifact
//! self-authenticating: replay always requires the original QDIMACS source.

use super::qdimacs::{QdimacsError, QdimacsParseBudget, QdimacsSourceStats, parse_qdimacs};
use super::quantified_game_dag::{
    QbfGameDagBudget, QbfGameDagBuildReport, QbfGameDagError, build_qbf_game_proof_dag,
};
use super::quantified_game_values::extract_game_value_profile_from_dag;
use super::synqbf::{SynQbfError, SynQbfVerificationReport};
use super::synqbf_artifact::{
    SynQbfArtifactBudget, SynQbfArtifactError, VerifiedSynQbfArtifact, encode_synqbf_proof,
    verify_synqbf_artifact,
};
use super::synqbf_construction::{
    SynQbfConstructionBudget, SynQbfConstructionError, SynQbfConstructionReport,
    construct_synqbf_from_game_value_profile,
};

/// Independent resource budgets for each stage of the portable pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SynQbfPipelineBudget {
    pub qdimacs: QdimacsParseBudget,
    pub game_dag: QbfGameDagBudget,
    pub construction: SynQbfConstructionBudget,
    pub artifact: SynQbfArtifactBudget,
}

impl Default for SynQbfPipelineBudget {
    fn default() -> Self {
        Self {
            qdimacs: QdimacsParseBudget::default(),
            game_dag: QbfGameDagBudget::default(),
            construction: SynQbfConstructionBudget::default(),
            artifact: SynQbfArtifactBudget::default(),
        }
    }
}

/// Errors preserve the stage that failed; no stage failure becomes a truth
/// value, proof acceptance, or successful artifact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SynQbfPipelineError {
    Qdimacs(QdimacsError),
    GameDag(QbfGameDagError),
    Construction(SynQbfConstructionError),
    Artifact(SynQbfArtifactError),
    ConstructionReplayMismatch,
    ResourceAccountingOverflow,
}

/// Measurements from the producer and the independent portable-artifact
/// replay. Byte length is not a cryptographic digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynQbfPipelineReport {
    pub source: QdimacsSourceStats,
    pub game_dag: QbfGameDagBuildReport,
    pub construction: SynQbfConstructionReport,
    pub artifact_bytes: usize,
    pub artifact_verification: SynQbfVerificationReport,
}

/// Portable bytes and the reports produced by the full verified construction
/// chain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BuiltSynQbfArtifact {
    pub bytes: Vec<u8>,
    pub report: SynQbfPipelineReport,
}

/// Result of replaying a portable artifact against a supplied QDIMACS source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayedSynQbfArtifact {
    pub source: QdimacsSourceStats,
    pub verified: VerifiedSynQbfArtifact,
}

/// Parse, construct, serialize, decode and independently verify a SynQBF
/// artifact against the exact source and ordered prefix from the QDIMACS text.
///
/// Construction and artifact verification have separate budgets. The
/// artifact's own proof budget governs the final replay. The returned bytes
/// are deterministic for a fixed source, implementation version and budgets.
pub fn build_portable_synqbf_artifact(
    qdimacs_source: &str,
    budget: SynQbfPipelineBudget,
) -> Result<BuiltSynQbfArtifact, SynQbfPipelineError> {
    let parsed =
        parse_qdimacs(qdimacs_source, budget.qdimacs).map_err(SynQbfPipelineError::Qdimacs)?;
    let (dag, dag_report) =
        build_qbf_game_proof_dag(&parsed.problem, &parsed.prefix, budget.game_dag)
            .map_err(SynQbfPipelineError::GameDag)?;

    let profile = extract_game_value_profile_from_dag(
        &parsed.problem,
        &parsed.prefix,
        &dag,
        budget.game_dag,
        budget.construction.game_values,
    )
    .map_err(|error| {
        SynQbfPipelineError::Construction(SynQbfConstructionError::GameValueProfile(error))
    })?;

    let constructed = construct_synqbf_from_game_value_profile(
        &parsed.problem,
        &parsed.prefix,
        &profile,
        budget.construction,
    )
    .map_err(SynQbfPipelineError::Construction)?;

    let bytes = encode_synqbf_proof(&constructed.proof, budget.artifact)
        .map_err(SynQbfPipelineError::Artifact)?;
    let replayed = verify_synqbf_artifact(
        &parsed.problem,
        &parsed.prefix,
        &bytes,
        budget.artifact,
        budget.artifact.proof,
    )
    .map_err(SynQbfPipelineError::Artifact)?;

    if replayed.proof != constructed.proof
        || replayed.verification != constructed.report.final_synqbf_verification
    {
        return Err(SynQbfPipelineError::ConstructionReplayMismatch);
    }

    let artifact_bytes = bytes.len();
    if artifact_bytes > budget.artifact.max_artifact_bytes {
        return Err(SynQbfPipelineError::Artifact(
            SynQbfArtifactError::ArtifactByteBudgetExceeded {
                required: artifact_bytes,
                limit: budget.artifact.max_artifact_bytes,
            },
        ));
    }

    Ok(BuiltSynQbfArtifact {
        bytes,
        report: SynQbfPipelineReport {
            source: parsed.stats,
            game_dag: dag_report,
            construction: constructed.report,
            artifact_bytes,
            artifact_verification: replayed.verification,
        },
    })
}

/// Independently replay portable SynQBF bytes against QDIMACS source text.
///
/// This is the consumer-side path: it does not trust producer reports,
/// counters, or an embedded acceptance bit. It reparses the source, decodes
/// the artifact and verifies every SynQBF obligation against that source.
pub fn replay_portable_synqbf_artifact(
    qdimacs_source: &str,
    artifact_bytes: &[u8],
    budget: SynQbfPipelineBudget,
) -> Result<ReplayedSynQbfArtifact, SynQbfPipelineError> {
    let parsed = parse_qdimacs(qdimacs_source, budget.qdimacs)
        .map_err(SynQbfPipelineError::Qdimacs)?;
    let verified = verify_synqbf_artifact(
        &parsed.problem,
        &parsed.prefix,
        artifact_bytes,
        budget.artifact,
        budget.artifact.proof,
    )
    .map_err(SynQbfPipelineError::Artifact)?;

    Ok(ReplayedSynQbfArtifact {
        source: parsed.stats,
        verified,
    })
}

#[cfg(test)]
mod tests {
    use super::super::qdimacs::QdimacsParseBudget;
    use super::*;

    fn budget() -> SynQbfPipelineBudget {
        SynQbfPipelineBudget {
            qdimacs: QdimacsParseBudget {
                max_input_bytes: 1 << 16,
                max_lines: 1024,
                max_variables: 8,
                max_clauses: 64,
                max_literals: 256,
                ..QdimacsParseBudget::default()
            },
            game_dag: QbfGameDagBudget {
                max_variables: 8,
                max_total_variable_name_bytes: 256,
                max_clauses: 64,
                max_literals: 256,
                max_proof_nodes: 1 << 12,
                max_builder_state_visits: 1 << 14,
                max_verifier_node_visits: 1 << 14,
                max_residual_literal_visits: 1 << 16,
                max_stored_residual_literals: 1 << 16,
                max_stored_residual_clauses: 1 << 14,
            },
            construction: SynQbfConstructionBudget::default(),
            artifact: SynQbfArtifactBudget::default(),
        }
    }

    fn true_qbf_source() -> &'static str {
        "c exists x: x\np cnf 1 1\ne 1 0\n1 0\n"
    }

    #[test]
    fn producer_and_consumer_replay_the_same_source_bound_artifact() {
        let source = true_qbf_source();
        let first = build_portable_synqbf_artifact(source, budget()).unwrap();
        let second = build_portable_synqbf_artifact(source, budget()).unwrap();
        assert_eq!(first.bytes, second.bytes);
        assert_eq!(first.report, second.report);

        let replayed = replay_portable_synqbf_artifact(source, &first.bytes, budget()).unwrap();
        assert_eq!(replayed.source, first.report.source);
        assert_eq!(
            replayed.verified.verification,
            first.report.artifact_verification
        );
    }

    #[test]
    fn replay_rejects_different_source_matrix_with_same_prefix_shape() {
        let source = true_qbf_source();
        let built = build_portable_synqbf_artifact(source, budget()).unwrap();
        let different_matrix = "c exists x: not x\np cnf 1 1\ne 1 0\n-1 0\n";
        assert!(matches!(
            replay_portable_synqbf_artifact(different_matrix, &built.bytes, budget()),
            Err(SynQbfPipelineError::Artifact(
                SynQbfArtifactError::SemanticVerification(
                    SynQbfError::SourceEquivalenceCounterexample { .. }
                        | SynQbfError::ClaimedTruthValueMismatch { .. }
                )
            ))
        ));
    }

    #[test]
    fn replay_rejects_prefix_reordering_or_quantifier_change() {
        let source = true_qbf_source();
        let built = build_portable_synqbf_artifact(source, budget()).unwrap();
        let different_prefix = "c forall x: x\np cnf 1 1\na 1 0\n1 0\n";
        assert!(matches!(
            replay_portable_synqbf_artifact(different_prefix, &built.bytes, budget()),
            Err(SynQbfPipelineError::Artifact(
                SynQbfArtifactError::SemanticVerification(
                    SynQbfError::PrefixBindingMismatch
                        | SynQbfError::ClaimedTruthValueMismatch { .. }
                )
            ))
        ));
    }

    #[test]
    fn parser_and_artifact_budgets_fail_closed() {
        let mut too_small_parser = budget();
        too_small_parser.qdimacs.max_input_bytes = 4;
        assert!(matches!(
            build_portable_synqbf_artifact(true_qbf_source(), too_small_parser),
            Err(SynQbfPipelineError::Qdimacs(
                QdimacsError::InputByteBudgetExceeded { .. }
            ))
        ));

        let mut too_small_artifact = budget();
        too_small_artifact.artifact.max_artifact_bytes = 8;
        assert!(matches!(
            build_portable_synqbf_artifact(true_qbf_source(), too_small_artifact),
            Err(SynQbfPipelineError::Artifact(
                SynQbfArtifactError::EncodingByteBudgetExceeded { .. }
            ))
        ));
    }

    #[test]
    fn portable_artifact_replay_rejects_trailing_bytes() {
        let built = build_portable_synqbf_artifact(true_qbf_source(), budget()).unwrap();
        let mut tampered = built.bytes;
        tampered.push(0);
        assert!(matches!(
            replay_portable_synqbf_artifact(true_qbf_source(), &tampered, budget()),
            Err(SynQbfPipelineError::Artifact(
                SynQbfArtifactError::TrailingBytes { remaining: 1 }
            ))
        ));
    }
}
