// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authority-free adapters from Forge output into the generic algorithm-discovery protocol.

use crate::certificate::{CertificateError, ForgeCandidate};
use crate::trace::{validate_forge_trace_observations, ForgeTraceError, ForgeTraceEvent};
use symthaea_algorithms::discovery::{
    CandidateArtifact, CandidateArtifactKind, CandidateProposal, DiscoveryError, DiscoveryRun,
};
use symthaea_algorithms::ledger::{DiscoveryLedger, LedgerError};
use symthaea_algorithms::observation::{ObservationError, ObservationStore};
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
    #[error(transparent)]
    Ledger(#[from] LedgerError),
    #[error(transparent)]
    Trace(#[from] ForgeTraceError),
    #[error(transparent)]
    Observation(#[from] ObservationError),
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
    let artifact = CandidateArtifact::new(CandidateArtifactKind::Other, artifact_id, source_ref)?;

    Ok(CandidateProposal::new_at_generation(
        run,
        generation,
        implementation,
        lineage,
        artifact,
    )?)
}

pub fn ledger_from_forge_trace(
    run: &DiscoveryRun,
    trace: &[ForgeTraceEvent],
    observations: &ObservationStore,
) -> Result<DiscoveryLedger, ForgeProposalError> {
    validate_forge_trace_observations(trace, observations)?;
    let mut ledger = DiscoveryLedger::new(run)?;
    for event in trace {
        ledger.append(
            run,
            event.generation,
            event.kind,
            event.candidate_artifact_id.clone(),
            event.observation_id.clone(),
        )?;
    }
    observations.validate_complete_for_ledger(run, &ledger)?;
    Ok(ledger)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certificate::{
        full_source_artifact_id, ForgeCertificate, GateEvidence, MutationRecord,
    };
    use crate::trace::ForgeAttemptId;
    use symthaea_algorithms::discovery::{DiscoveryPolicy, SearchBudget};
    use symthaea_algorithms::ledger::DiscoveryEventKind;
    use symthaea_algorithms::observation::{ObservationEncoding, ObservationObject};
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

    fn passing_gates() -> Vec<GateEvidence> {
        vec![
            GateEvidence {
                gate: "compile".into(),
                passed: true,
                duration_ms: 1,
                output_tail: String::new(),
            },
            GateEvidence {
                gate: "test".into(),
                passed: true,
                duration_ms: 1,
                output_tail: String::new(),
            },
        ]
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
            gates: passing_gates(),
            benchmark: None,
            before_source: "fn target() -> i32 { 1 }".into(),
            after_source: "fn target() -> i32 { 2 }".into(),
        };
        let candidate = ForgeCandidate::new(certificate, candidate_source).unwrap();
        (run, algorithm, baseline, candidate)
    }

    fn attempt(candidate: &ForgeCandidate) -> ForgeAttemptId {
        ForgeAttemptId::derive(
            &candidate.certificate().baseline_artifact_id,
            7,
            0,
            1,
        )
    }

    fn attempt_observation(attempt: &ForgeAttemptId, label: &str) -> ObservationObject {
        ObservationObject::new(
            "forge.test.attempt.v1",
            ObservationEncoding::Json,
            serde_json::to_vec(&serde_json::json!({
                "attempt_id": attempt.as_content_id().as_str(),
                "label": label
            }))
            .unwrap(),
        )
        .unwrap()
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

    #[test]
    fn forge_trace_replays_only_with_complete_observation_store() {
        let (run, _, _, candidate) = fixture();
        let artifact = candidate.artifact_id().clone();
        let attempt_id = attempt(&candidate);
        let generated = attempt_observation(&attempt_id, "generated");
        let rejected = attempt_observation(&attempt_id, "counterexample");
        let completed = ObservationObject::utf8("forge.completed.v1", "complete").unwrap();
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt_id.clone(),
                1,
                DiscoveryEventKind::CandidateGenerated,
                artifact.clone(),
                generated.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt_id,
                1,
                DiscoveryEventKind::RejectedCorrectness,
                artifact,
                rejected.id().clone(),
            ),
            ForgeTraceEvent::completed(completed.id().clone()),
        ];
        let observations =
            ObservationStore::from_objects(vec![generated, rejected, completed]).unwrap();
        let ledger = ledger_from_forge_trace(&run, &trace, &observations).unwrap();
        assert!(ledger.is_sealed());
        assert_eq!(ledger.len(), trace.len());
        assert!(ledger.validate_for(&run).is_ok());
    }

    #[test]
    fn missing_observation_blocks_semantic_replay() {
        let (run, _, _, candidate) = fixture();
        let attempt_id = attempt(&candidate);
        let generated = attempt_observation(&attempt_id, "generated");
        let completed = ObservationObject::utf8("forge.completed.v1", "complete").unwrap();
        let trace = vec![
            ForgeTraceEvent::candidate(
                attempt_id.clone(),
                1,
                DiscoveryEventKind::CandidateGenerated,
                candidate.artifact_id().clone(),
                generated.id().clone(),
            ),
            ForgeTraceEvent::candidate(
                attempt_id,
                1,
                DiscoveryEventKind::RejectedCorrectness,
                candidate.artifact_id().clone(),
                ContentId::derive("missing", [b"not-stored".as_slice()]),
            ),
            ForgeTraceEvent::completed(completed.id().clone()),
        ];
        let observations = ObservationStore::from_objects(vec![generated, completed]).unwrap();
        assert!(matches!(
            ledger_from_forge_trace(&run, &trace, &observations),
            Err(ForgeProposalError::Trace(ForgeTraceError::MissingObservation(_)))
        ));
    }

    #[test]
    fn unterminated_forge_trace_is_rejected_before_semantic_replay() {
        let (run, _, _, candidate) = fixture();
        let attempt_id = attempt(&candidate);
        let generated = attempt_observation(&attempt_id, "generated");
        let trace = vec![ForgeTraceEvent::candidate(
            attempt_id,
            1,
            DiscoveryEventKind::CandidateGenerated,
            candidate.artifact_id().clone(),
            generated.id().clone(),
        )];
        let observations = ObservationStore::from_objects(vec![generated]).unwrap();
        assert!(matches!(
            ledger_from_forge_trace(&run, &trace, &observations),
            Err(ForgeProposalError::Trace(ForgeTraceError::MissingCompletion))
        ));
    }
}
