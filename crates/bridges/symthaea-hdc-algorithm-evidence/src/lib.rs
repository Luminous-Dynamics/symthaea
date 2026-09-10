// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capsule-bound evidence for the first HDC algorithm-discovery laboratory.
//!
//! This bridge enforces the distinction:
//!
//! ```text
//! discovery run
//! != candidate artifact
//! != input corpus
//! != correctness evidence
//! != evaluation context
//! != performance receipt
//! ```
//!
//! Two candidates evaluated over the same exact corpus, evaluator command, machine, toolchain,
//! dependency locks, and frozen repository baseline must share one [`EvaluationContext`] so a
//! Pareto comparison is meaningful. Their implementation and correctness identities must remain
//! distinct.
//!
//! This crate creates evidence records only. It cannot execute benchmarks, edit source, use Git,
//! promote a candidate, or grant runtime authority.

use symthaea_algorithm_evidence_collector::{CollectorError, CommandSpec, ExperimentCapsule};
use symthaea_algorithm_lab::{
    HammingCandidate, HammingCorrectnessEvidence, HdcLabError, hamming_algorithm, hamming_problem,
    implementation_record,
};
use symthaea_algorithms::discovery::{
    CandidateArtifact, CandidateArtifactKind, CandidateProposal, DiscoveryError, DiscoveryRun,
};
use symthaea_algorithms::evaluation::{
    CorrectnessVerdict, EvaluationContext, EvaluationError, EvaluationReceipt, ObjectiveDirection,
    ObjectiveMeasurement,
};
use symthaea_algorithms::{
    AlgorithmLineage, ContentId, ImplementationId, ImplementationRecord, RegistryError,
};
use thiserror::Error;

pub const HDC_HAMMING_CORPUS_SCHEMA: &str = "symthaea-hdc-hamming-corpus-v1";
pub const HDC_HAMMING_ORACLE_SCHEMA: &str = "symthaea-hdc-bitwise-oracle-v1";
pub const HDC_HAMMING_EVALUATOR_SCHEMA: &str = "symthaea-hdc-criterion-evaluator-v1";

#[derive(Debug, Error)]
pub enum HdcEvidenceBridgeError {
    #[error(transparent)]
    Registry(#[from] RegistryError),
    #[error(transparent)]
    Lab(#[from] HdcLabError),
    #[error(transparent)]
    Collector(#[from] CollectorError),
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    #[error(transparent)]
    Evaluation(#[from] EvaluationError),
    #[error("discovery run is not for the exact HDC Hamming problem")]
    RunProblemMismatch,
    #[error("capsule repository revision differs from the discovery run baseline revision")]
    BaselineRevisionMismatch,
    #[error("candidate artifact group `{0}` is missing from the experiment capsule")]
    MissingCandidateArtifactGroup(String),
    #[error("candidate artifact identity does not match the collected artifact group")]
    CandidateArtifactMismatch,
    #[error("candidate implementation does not match the correctness candidate")]
    CandidateImplementationMismatch,
    #[error("correctness evidence is not a passing exact result")]
    CorrectnessNotPassed,
    #[error("bound correctness evidence does not match the supplied run/proposal")]
    CorrectnessBindingMismatch,
    #[error("bound correctness evidence input corpus does not match the evaluation context")]
    InputProfileMismatch,
    #[error("selected capsule command is not the canonical HDC Hamming benchmark command")]
    BenchmarkCommandMismatch,
}

/// Candidate-independent identity of the exact correctness workload.
///
/// This commits the semantic problem, fixed edge cases, deterministic pair-generation rule,
/// operand-order coverage, and canonical seed set. Candidate identity is deliberately absent.
pub fn hamming_input_profile_id(seeds: &[u64]) -> Result<ContentId, HdcEvidenceBridgeError> {
    let problem = hamming_problem()?;
    let mut seeds = seeds.to_vec();
    seeds.sort_unstable();
    seeds.dedup();

    let mut owned = vec![
        problem.id.as_content_id().as_str().as_bytes().to_vec(),
        HDC_HAMMING_CORPUS_SCHEMA.as_bytes().to_vec(),
        b"edge:zero-zero".to_vec(),
        b"edge:ones-ones".to_vec(),
        b"edge:zero-ones".to_vec(),
        b"edge:ones-zero".to_vec(),
        b"seed-pair:right=seed-xor-9e3779b97f4a7c15".to_vec(),
        b"orders:forward+reverse".to_vec(),
    ];
    owned.extend(seeds.iter().map(|seed| seed.to_be_bytes().to_vec()));
    Ok(ContentId::derive(
        "symthaea.hdc-hamming-input-profile.v1",
        owned.iter().map(Vec::as_slice),
    ))
}

pub fn hamming_oracle_id() -> ContentId {
    ContentId::derive(
        "symthaea.hdc-hamming-oracle.v1",
        [HDC_HAMMING_ORACLE_SCHEMA.as_bytes()],
    )
}

pub fn hamming_evaluator_id() -> ContentId {
    ContentId::derive(
        "symthaea.hdc-hamming-evaluator.v1",
        [HDC_HAMMING_EVALUATOR_SCHEMA.as_bytes()],
    )
}

/// Exact benchmark invocation admitted by this bridge.
///
/// Recording it is provenance only. This crate never executes the command.
pub fn canonical_hamming_benchmark_command() -> Result<CommandSpec, CollectorError> {
    CommandSpec::new(
        "cargo",
        vec![
            "bench".into(),
            "-p".into(),
            "symthaea-algorithm-lab".into(),
            "--bench".into(),
            "hdc_hamming_candidates".into(),
        ],
    )
}

fn validate_run_capsule(
    run: &DiscoveryRun,
    capsule: &ExperimentCapsule,
) -> Result<(), HdcEvidenceBridgeError> {
    run.validate()?;
    capsule.validate()?;
    let problem = hamming_problem()?;
    if run.problem_id != problem.id {
        return Err(HdcEvidenceBridgeError::RunProblemMismatch);
    }
    if capsule.repository.revision != run.baseline_revision {
        return Err(HdcEvidenceBridgeError::BaselineRevisionMismatch);
    }
    Ok(())
}

fn artifact_id(
    capsule: &ExperimentCapsule,
    candidate_artifact_group: &str,
) -> Result<ContentId, HdcEvidenceBridgeError> {
    capsule
        .artifact_group_id(candidate_artifact_group)
        .cloned()
        .ok_or_else(|| {
            HdcEvidenceBridgeError::MissingCandidateArtifactGroup(
                candidate_artifact_group.to_string(),
            )
        })
}

/// Build a generation-bound proposal directly from collector-produced candidate bytes.
pub fn proposal_from_capsule(
    run: &DiscoveryRun,
    capsule: &ExperimentCapsule,
    candidate: HammingCandidate,
    candidate_artifact_group: &str,
    generation: u64,
) -> Result<CandidateProposal, HdcEvidenceBridgeError> {
    validate_run_capsule(run, capsule)?;
    let artifact_id = artifact_id(capsule, candidate_artifact_group)?;
    let implementation = implementation_record(candidate, artifact_id.clone())?;
    let lineage = AlgorithmLineage::new(implementation.id.clone(), vec![], vec![])?;
    let artifact = CandidateArtifact::new(
        CandidateArtifactKind::SourceTree,
        artifact_id,
        format!("capsule://artifact-group/{candidate_artifact_group}"),
    )?;
    Ok(CandidateProposal::new_at_generation(
        run,
        generation,
        implementation,
        lineage,
        artifact,
    )?)
}

/// Opaque correctness binding connecting one exact discovery proposal to the candidate-independent
/// input corpus on which the HDC laboratory checked it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapsuleBoundHammingCorrectness {
    id: ContentId,
    run_id: ContentId,
    implementation_id: ImplementationId,
    candidate: HammingCandidate,
    candidate_artifact_id: ContentId,
    raw_correctness_id: ContentId,
    input_profile_id: ContentId,
    seeds: Vec<u64>,
}

impl CapsuleBoundHammingCorrectness {
    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn run_id(&self) -> &ContentId {
        &self.run_id
    }

    pub fn implementation_id(&self) -> &ImplementationId {
        &self.implementation_id
    }

    pub fn candidate(&self) -> HammingCandidate {
        self.candidate
    }

    pub fn candidate_artifact_id(&self) -> &ContentId {
        &self.candidate_artifact_id
    }

    pub fn input_profile_id(&self) -> &ContentId {
        &self.input_profile_id
    }

    pub fn seeds(&self) -> &[u64] {
        &self.seeds
    }

    pub fn validate(&self) -> Result<(), HdcEvidenceBridgeError> {
        let mut canonical_seeds = self.seeds.clone();
        canonical_seeds.sort_unstable();
        canonical_seeds.dedup();
        if canonical_seeds != self.seeds {
            return Err(HdcEvidenceBridgeError::InputProfileMismatch);
        }
        if hamming_input_profile_id(&self.seeds)? != self.input_profile_id {
            return Err(HdcEvidenceBridgeError::InputProfileMismatch);
        }
        let expected = derive_bound_correctness_id(
            &self.run_id,
            &self.implementation_id,
            self.candidate,
            &self.candidate_artifact_id,
            &self.raw_correctness_id,
            &self.input_profile_id,
        );
        if expected != self.id {
            return Err(HdcEvidenceBridgeError::CorrectnessBindingMismatch);
        }
        Ok(())
    }
}

fn derive_bound_correctness_id(
    run_id: &ContentId,
    implementation_id: &ImplementationId,
    candidate: HammingCandidate,
    candidate_artifact_id: &ContentId,
    raw_correctness_id: &ContentId,
    input_profile_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.hdc-hamming-capsule-correctness.v2",
        [
            run_id.as_str().as_bytes(),
            implementation_id.as_content_id().as_str().as_bytes(),
            candidate.name().as_bytes(),
            candidate_artifact_id.as_str().as_bytes(),
            raw_correctness_id.as_str().as_bytes(),
            input_profile_id.as_str().as_bytes(),
        ],
    )
}

/// Bind raw laboratory correctness to the exact run, proposal, and collected candidate artifact.
pub fn bind_correctness(
    run: &DiscoveryRun,
    capsule: &ExperimentCapsule,
    candidate_artifact_group: &str,
    proposal: &CandidateProposal,
    correctness: &HammingCorrectnessEvidence,
) -> Result<CapsuleBoundHammingCorrectness, HdcEvidenceBridgeError> {
    validate_run_capsule(run, capsule)?;
    proposal.validate_for(run)?;
    correctness.validate()?;
    if !correctness.passed() || correctness.verdict() != CorrectnessVerdict::Passed {
        return Err(HdcEvidenceBridgeError::CorrectnessNotPassed);
    }

    let problem = hamming_problem()?;
    let algorithm = hamming_algorithm(&problem)?;
    proposal.implementation.validate_for(&algorithm)?;

    let expected_source_ref = format!(
        "symthaea-algorithm-lab::{}",
        correctness.candidate().name()
    );
    if proposal.implementation.source_ref != expected_source_ref {
        return Err(HdcEvidenceBridgeError::CandidateImplementationMismatch);
    }

    let capsule_artifact_id = artifact_id(capsule, candidate_artifact_group)?;
    if proposal.implementation.artifact_id != capsule_artifact_id
        || proposal.artifact.content_id != capsule_artifact_id
    {
        return Err(HdcEvidenceBridgeError::CandidateArtifactMismatch);
    }

    let input_profile_id = hamming_input_profile_id(correctness.seeds())?;
    let candidate = correctness.candidate();
    let id = derive_bound_correctness_id(
        &run.id,
        &proposal.implementation.id,
        candidate,
        &capsule_artifact_id,
        correctness.id(),
        &input_profile_id,
    );
    let bound = CapsuleBoundHammingCorrectness {
        id,
        run_id: run.id.clone(),
        implementation_id: proposal.implementation.id.clone(),
        candidate,
        candidate_artifact_id: capsule_artifact_id,
        raw_correctness_id: correctness.id().clone(),
        input_profile_id,
        seeds: correctness.seeds().to_vec(),
    };
    bound.validate()?;
    Ok(bound)
}

/// Create a rankable comparison context without candidate bytes or candidate-specific correctness
/// evidence contaminating the context identity.
pub fn hamming_evaluation_context(
    run: &DiscoveryRun,
    capsule: &ExperimentCapsule,
    bound: &CapsuleBoundHammingCorrectness,
    command_index: usize,
) -> Result<EvaluationContext, HdcEvidenceBridgeError> {
    validate_run_capsule(run, capsule)?;
    bound.validate()?;
    if bound.run_id != run.id {
        return Err(HdcEvidenceBridgeError::CorrectnessBindingMismatch);
    }

    let command = capsule
        .commands
        .get(command_index)
        .ok_or(CollectorError::CommandIndexOutOfRange(command_index))?;
    if command != &canonical_hamming_benchmark_command()? {
        return Err(HdcEvidenceBridgeError::BenchmarkCommandMismatch);
    }

    Ok(capsule.evaluation_context(
        hamming_evaluator_id(),
        hamming_oracle_id(),
        bound.input_profile_id.clone(),
        command_index,
        bound.seeds.clone(),
    )?)
}

/// Mint one historical latency receipt from an externally observed Criterion measurement.
///
/// This function never runs Criterion. The measurement-run ID identifies the external execution
/// so append-only repeatability evidence can distinguish reruns.
pub fn latency_receipt_from_capsule(
    run: &DiscoveryRun,
    capsule: &ExperimentCapsule,
    proposal: &CandidateProposal,
    bound: &CapsuleBoundHammingCorrectness,
    command_index: usize,
    latency_ns_per_op: f64,
    measurement_run_id: impl Into<String>,
) -> Result<EvaluationReceipt, HdcEvidenceBridgeError> {
    validate_run_capsule(run, capsule)?;
    proposal.validate_for(run)?;
    bound.validate()?;
    if bound.run_id != run.id
        || proposal.implementation.id != bound.implementation_id
        || proposal.implementation.artifact_id != bound.candidate_artifact_id
    {
        return Err(HdcEvidenceBridgeError::CorrectnessBindingMismatch);
    }

    let context = hamming_evaluation_context(run, capsule, bound, command_index)?;
    if context.input_profile_id != bound.input_profile_id {
        return Err(HdcEvidenceBridgeError::InputProfileMismatch);
    }

    Ok(EvaluationReceipt::new(
        run.problem_id.clone(),
        proposal.implementation.id.clone(),
        context,
        CorrectnessVerdict::Passed,
        bound.id.clone(),
        vec![ObjectiveMeasurement::new(
            "latency",
            ObjectiveDirection::Minimize,
            latency_ns_per_op,
            "ns/op",
        )?],
        Some(measurement_run_id.into()),
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use symthaea_algorithm_evidence_collector::{
        ArtifactGroupSpec, CapturedEnvironment, MachineProfile, RepositoryState, ToolchainProfile,
        collect_artifact_group,
    };
    use symthaea_algorithm_lab::{pilot_run, verify_candidate};
    use symthaea_algorithms::pareto::ParetoCohort;

    const REVISION: &str = "0123456789abcdef";

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn temp_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "symthaea-hdc-evidence-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn capsule_with_candidate_bytes(label: &str, bytes: &[u8]) -> ExperimentCapsule {
        let root = temp_root(label);
        fs::write(root.join("candidate.rs"), bytes).unwrap();
        let group = collect_artifact_group(
            &root,
            &ArtifactGroupSpec::new("candidate-source", vec!["candidate.rs".into()]).unwrap(),
        )
        .unwrap();
        let capsule = ExperimentCapsule::new(
            RepositoryState::new(
                REVISION,
                true,
                cid("status", "clean"),
                Some(cid("cargo-lock", "same-lock")),
                Some(cid("flake-lock", "same-flake")),
                Some(cid("toolchain-file", "same-toolchain-file")),
            )
            .unwrap(),
            MachineProfile::new(
                "linux",
                "x86_64",
                "Linux test-kernel",
                Some("test-cpu".into()),
                vec!["avx2".into(), "popcnt".into()],
                "x86_64-unknown-linux-gnu",
                vec!["target_feature=\"sse2\"".into()],
            )
            .unwrap(),
            ToolchainProfile::new(
                "rustc 1.96.0\nhost: x86_64-unknown-linux-gnu",
                "cargo 1.96.0",
                Some("nix (Nix) test".into()),
            )
            .unwrap(),
            CapturedEnvironment::new(BTreeMap::new()),
            vec![group],
            vec![canonical_hamming_benchmark_command().unwrap()],
        )
        .unwrap();
        fs::remove_dir_all(root).unwrap();
        capsule
    }

    #[test]
    fn input_corpus_identity_is_canonical_and_candidate_independent() {
        assert_eq!(
            hamming_input_profile_id(&[7, 3, 7]).unwrap(),
            hamming_input_profile_id(&[3, 7]).unwrap()
        );
    }

    #[test]
    fn different_artifacts_share_context_but_not_correctness_or_implementation_identity() {
        let run = pilot_run(REVISION, 42).unwrap();
        let capsule_a = capsule_with_candidate_bytes("a", b"byte popcount candidate");
        let capsule_b = capsule_with_candidate_bytes("b", b"u64 popcount candidate");
        assert_ne!(capsule_a.id, capsule_b.id);
        assert_eq!(
            capsule_a.comparison_environment_id(),
            capsule_b.comparison_environment_id()
        );

        let proposal_a = proposal_from_capsule(
            &run,
            &capsule_a,
            HammingCandidate::BytePopcount,
            "candidate-source",
            0,
        )
        .unwrap();
        let proposal_b = proposal_from_capsule(
            &run,
            &capsule_b,
            HammingCandidate::U64Popcount,
            "candidate-source",
            0,
        )
        .unwrap();
        assert_ne!(proposal_a.implementation.id, proposal_b.implementation.id);
        assert_ne!(
            proposal_a.implementation.artifact_id,
            proposal_b.implementation.artifact_id
        );

        let seeds = [1, 2, 3, 5, 8];
        let raw_a = verify_candidate(HammingCandidate::BytePopcount, &seeds);
        let raw_b = verify_candidate(HammingCandidate::U64Popcount, &seeds);
        let bound_a = bind_correctness(
            &run,
            &capsule_a,
            "candidate-source",
            &proposal_a,
            &raw_a,
        )
        .unwrap();
        let bound_b = bind_correctness(
            &run,
            &capsule_b,
            "candidate-source",
            &proposal_b,
            &raw_b,
        )
        .unwrap();
        assert_ne!(bound_a.id(), bound_b.id());
        assert_eq!(bound_a.input_profile_id(), bound_b.input_profile_id());

        let context_a = hamming_evaluation_context(&run, &capsule_a, &bound_a, 0).unwrap();
        let context_b = hamming_evaluation_context(&run, &capsule_b, &bound_b, 0).unwrap();
        assert_eq!(context_a, context_b);

        let receipt_a = latency_receipt_from_capsule(
            &run,
            &capsule_a,
            &proposal_a,
            &bound_a,
            0,
            11.0,
            "measurement-a",
        )
        .unwrap();
        let receipt_b = latency_receipt_from_capsule(
            &run,
            &capsule_b,
            &proposal_b,
            &bound_b,
            0,
            9.0,
            "measurement-b",
        )
        .unwrap();
        let receipts = vec![receipt_a, receipt_b];
        let cohort = ParetoCohort::new(&receipts).unwrap();
        let frontier = cohort.frontier();
        assert_eq!(frontier.len(), 1);
        assert_eq!(&frontier[0].implementation_id, &proposal_b.implementation.id);
    }

    #[test]
    fn candidate_cannot_borrow_another_candidates_correctness() {
        let run = pilot_run(REVISION, 42).unwrap();
        let capsule = capsule_with_candidate_bytes("borrow", b"shared source artifact");
        let proposal = proposal_from_capsule(
            &run,
            &capsule,
            HammingCandidate::U64Popcount,
            "candidate-source",
            0,
        )
        .unwrap();
        let raw = verify_candidate(HammingCandidate::BytePopcount, &[1, 2, 3]);
        assert!(matches!(
            bind_correctness(&run, &capsule, "candidate-source", &proposal, &raw),
            Err(HdcEvidenceBridgeError::CandidateImplementationMismatch)
        ));
    }

    #[test]
    fn changed_artifact_invalidates_old_proposal_binding() {
        let run = pilot_run(REVISION, 42).unwrap();
        let old_capsule = capsule_with_candidate_bytes("old", b"candidate v1");
        let new_capsule = capsule_with_candidate_bytes("new", b"candidate v2");
        let old_proposal = proposal_from_capsule(
            &run,
            &old_capsule,
            HammingCandidate::BytePopcount,
            "candidate-source",
            0,
        )
        .unwrap();
        let raw = verify_candidate(HammingCandidate::BytePopcount, &[1, 2, 3]);
        assert!(matches!(
            bind_correctness(
                &run,
                &new_capsule,
                "candidate-source",
                &old_proposal,
                &raw
            ),
            Err(HdcEvidenceBridgeError::CandidateArtifactMismatch)
        ));
    }

    #[test]
    fn capsule_revision_must_equal_frozen_discovery_baseline() {
        let run = pilot_run("different-revision", 42).unwrap();
        let capsule = capsule_with_candidate_bytes("revision", b"candidate");
        assert!(matches!(
            proposal_from_capsule(
                &run,
                &capsule,
                HammingCandidate::BytePopcount,
                "candidate-source",
                0
            ),
            Err(HdcEvidenceBridgeError::BaselineRevisionMismatch)
        ));
    }

    #[test]
    fn wrong_benchmark_command_fails_closed() {
        let run = pilot_run(REVISION, 42).unwrap();
        let original = capsule_with_candidate_bytes("command", b"candidate");
        let capsule = ExperimentCapsule::new(
            original.repository,
            original.machine,
            original.toolchain,
            original.environment,
            original.artifact_groups,
            vec![CommandSpec::new("cargo", vec!["test".into()]).unwrap()],
        )
        .unwrap();
        let proposal = proposal_from_capsule(
            &run,
            &capsule,
            HammingCandidate::BytePopcount,
            "candidate-source",
            0,
        )
        .unwrap();
        let raw = verify_candidate(HammingCandidate::BytePopcount, &[1]);
        let bound = bind_correctness(
            &run,
            &capsule,
            "candidate-source",
            &proposal,
            &raw,
        )
        .unwrap();
        assert!(matches!(
            hamming_evaluation_context(&run, &capsule, &bound, 0),
            Err(HdcEvidenceBridgeError::BenchmarkCommandMismatch)
        ));
    }
}
