// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authority-free adapter from a content-addressed Forge survivor into the generic algorithm
//! discovery protocol.
//!
//! Forge does not invent the semantic problem or algorithm family. Those are supplied by the
//! caller as already-valid registry records. The adapter only proves that the Forge survivor is
//! tied to the same frozen baseline and that its exact artifact/ordered mutation lineage can be
//! represented as a [`CandidateProposal`].

use crate::certificate::{CertificateError, ForgeCandidate};
use symthaea_algorithms::discovery::{
    CandidateArtifact, CandidateArtifactKind, CandidateProposal, DiscoveryError, DiscoveryRun,
};
use symthaea_algorithms::{
    AlgorithmLineage, AlgorithmRecord, ImplementationRecord, RegistryError,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeProposalError {
    #[error(transparent)]
    Certificate(#[from] CertificateError),
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    #[error("Forge certificate does not contain an exact Git baseline revision")]
    MissingBaselineRevision,
    #[error("Forge certificate Git revision does not match the discovery run baseline")]
    BaselineRevisionMismatch,
    #[error("algorithm record does not describe the discovery run problem")]
    AlgorithmProblemMismatch,
    #[error("baseline implementation artifact does not match Forge's pristine source artifact")]
    BaselineArtifactMismatch,
    #[error("Forge generation cannot be represented by the discovery protocol")]
    GenerationOverflow,
}

/// Convert one exact Forge survivor into a generic candidate proposal.
///
/// The baseline implementation is retained as the lineage parent. Forge's accepted mutation
/// history becomes the ordered transformation trace. Intermediate generation winners are already
/// cryptographically linked inside the Forge certificate by parent->child artifact identities, so
/// they do not need to be fabricated as standalone registry implementations here.
pub fn proposal_from_forge(
    run: &DiscoveryRun,
    algorithm: &AlgorithmRecord,
    baseline_implementation: &ImplementationRecord,
    candidate: &ForgeCandidate,
) -> Result<CandidateProposal, ForgeProposalError> {
    run.validate()?;
    algorithm.validate()?;
    candidate.validate()?;
    let certificate = candidate.certificate();

    if algorithm.problem_id != run.problem_id {
        return Err(ForgeProposalError::AlgorithmProblemMismatch);
    }
    baseline_implementation.validate_for(algorithm)?;
    if baseline_implementation.artifact_id != certificate.baseline_artifact_id {
        return Err(ForgeProposalError::BaselineArtifactMismatch);
    }

    let certificate_revision = certificate
        .git_sha
        .as_deref()
        .ok_or(ForgeProposalError::MissingBaselineRevision)?;
    if certificate_revision != run.baseline_revision {
        return Err(ForgeProposalError::BaselineRevisionMismatch);
    }

    let generation = u64::try_from(certificate.generation)
        .map_err(|_| ForgeProposalError::GenerationOverflow)?;
    let artifact_id = candidate.artifact_id().clone();
    let source_ref = format!("forge://full-source/{}", artifact_id.as_str());
    let implementation = ImplementationRecord::new(
        run.problem_id.clone(),
        algorithm.id.clone(),
        source_ref.clone(),
        artifact_id.clone(),
        baseline_implementation.target_profile.clone(),
    )?;
    let lineage = AlgorithmLineage::new(
        implementation.id.clone(),
        vec![baseline_implementation.id.clone()],
        certificate.transformation_ids(),
    )?;
    let artifact = CandidateArtifact::new(
        CandidateArtifactKind::Other,
        artifact_id,
        source_ref,
    )?;

    Ok(CandidateProposal::new_at_generation(
        run,
        generation,
        implementation,
        lineage,
        artifact,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::{
        ForgeCertificate, GateEvidence, MutationRecord, full_source_artifact_id,
    };
    use symthaea_algorithms::discovery::{DiscoveryPolicy, SearchBudget};
    use symthaea_algorithms::{
        AlgorithmProvenance, ContentId, DeterminismRequirement, DiscoveryRisk, ProblemSpec,
        SemanticGuarantee,
    };
    use std::path::PathBuf;

    fn problem() -> ProblemSpec {
        ProblemSpec::new(
            "forge-test-problem",
            "Return the exact reference value.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["output equals oracle".into()],
            DiscoveryRisk::Ordinary,
        )
        .unwrap()
    }

    fn fixture() -> (
        DiscoveryRun,
        AlgorithmRecord,
        ImplementationRecord,
        ForgeCandidate,
    ) {
        let problem = problem();
        let algorithm = AlgorithmRecord::new(
            problem.id.clone(),
            "forge-local-search-family",
            "AST-local structural candidates generated from one valid baseline.",
            AlgorithmProvenance::Evolved,
        )
        .unwrap();
        let baseline_source = "fn target() -> i32 { 1 }\n";
        let candidate_source = "fn target() -> i32 { 2 }\n".to_string();
        let baseline_artifact_id = full_source_artifact_id(baseline_source);
        let candidate_artifact_id = full_source_artifact_id(&candidate_source);
        let baseline = ImplementationRecord::new(
            problem.id.clone(),
            algorithm.id.clone(),
            "repo://src/target.rs",
            baseline_artifact_id.clone(),
            None,
        )
        .unwrap();
        let run = DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            ContentId::derive("generator", [b"forge-v1".as_slice()]),
            "abc123",
            SearchBudget::new(10, 5, 10).unwrap(),
            7,
        )
        .unwrap();
        let mutation = MutationRecord::new(
            1,
            "NumericLiteralPerturb",
            "1 -> 2",
            baseline_artifact_id.clone(),
            candidate_artifact_id.clone(),
        );
        let certificate = ForgeCertificate {
            generated_at_unix_ms: 0,
            target_file: PathBuf::from("src/target.rs"),
            target_function: "target".into(),
            package: "test-package".into(),
            git_sha: Some("abc123".into()),
            generation: 1,
            baseline_artifact_id,
            candidate_artifact_id,
            mutation_operator: mutation.operator.clone(),
            mutation_detail: mutation.detail.clone(),
            mutation_history: vec![mutation],
            gates: vec![GateEvidence {
                gate: "test".into(),
                passed: true,
                duration_ms: 1,
                output_tail: String::new(),
            }],
            benchmark: None,
            before_source: "fn target() -> i32 { 1 }".into(),
            after_source: "fn target() -> i32 { 2 }".into(),
        };
        let candidate = ForgeCandidate::new(certificate, candidate_source).unwrap();
        (run, algorithm, baseline, candidate)
    }

    #[test]
    fn proposal_binds_baseline_candidate_and_ordered_transformations() {
        let (run, algorithm, baseline, candidate) = fixture();
        let proposal = proposal_from_forge(&run, &algorithm, &baseline, &candidate).unwrap();
        assert_eq!(proposal.run_id, run.id);
        assert_eq!(proposal.generation, 1);
        assert_eq!(proposal.implementation.artifact_id, *candidate.artifact_id());
        assert_eq!(proposal.lineage.parent_ids, vec![baseline.id]);
        assert_eq!(
            proposal.lineage.transformation_ids,
            candidate.certificate().transformation_ids()
        );
        assert!(proposal.validate_for(&run).is_ok());
    }

    #[test]
    fn wrong_run_baseline_is_rejected() {
        let (run, algorithm, baseline, candidate) = fixture();
        let problem = problem();
        let other_run = DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            run.generator_id.clone(),
            "different-revision",
            run.budget,
            run.seed,
        )
        .unwrap();
        assert!(matches!(
            proposal_from_forge(&other_run, &algorithm, &baseline, &candidate),
            Err(ForgeProposalError::BaselineRevisionMismatch)
        ));
    }

    #[test]
    fn substituted_baseline_artifact_is_rejected() {
        let (run, algorithm, baseline, candidate) = fixture();
        let wrong = ImplementationRecord::new(
            baseline.problem_id.clone(),
            baseline.algorithm_id.clone(),
            baseline.source_ref.clone(),
            full_source_artifact_id("different baseline\n"),
            baseline.target_profile.clone(),
        )
        .unwrap();
        assert!(matches!(
            proposal_from_forge(&run, &algorithm, &wrong, &candidate),
            Err(ForgeProposalError::BaselineArtifactMismatch)
        ));
    }
}