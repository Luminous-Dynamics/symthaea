// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capsule-bound evidence for the first HDC algorithm-discovery laboratory.
//!
//! This bridge fixes a subtle comparability boundary:
//!
//! ```text
//! input-corpus identity != candidate correctness identity != candidate artifact identity
//! ```
//!
//! Two candidates evaluated over the same exact corpus, evaluator command, machine, toolchain,
//! and dependency locks must share one [`EvaluationContext`] so Pareto comparison is meaningful.
//! Their implementation and correctness identities must still remain distinct.
//!
//! The bridge creates evidence records only. It cannot execute benchmarks, edit source, use Git,
//! promote a candidate, or grant runtime authority.

use symthaea_algorithm_evidence_collector::{
    CollectorError, CommandSpec, ExperimentCapsule,
};
use symthaea_algorithm_lab::{
    HammingCandidate, HammingCorrectnessEvidence, HdcLabError, hamming_algorithm, hamming_problem,
    implementation_record,
};
use symthaea_algorithms::evaluation::{
    CorrectnessVerdict, EvaluationContext, EvaluationError, EvaluationReceipt, ObjectiveDirection,
    ObjectiveMeasurement,
};
use symthaea_algorithms::{ContentId, ImplementationRecord, RegistryError};
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
    Evaluation(#[from] EvaluationError),
    #[error("candidate artifact group `{0}` is missing from the experiment capsule")]
    MissingCandidateArtifactGroup(String),
    #[error("implementation artifact identity does not match the collected candidate artifact")]
    CandidateArtifactMismatch,
    #[error("implementation source reference does not match the correctness candidate")]
    CandidateImplementationMismatch,
    #[error("correctness evidence is not a passing exact result")]
    CorrectnessNotPassed,
    #[error("selected capsule command is not the canonical HDC Hamming benchmark command")]
    BenchmarkCommandMismatch,
    #[error("bound correctness evidence does not match the supplied implementation")]
    CorrectnessImplementationMismatch,
    #[error("bound correctness evidence input corpus does not match the evaluation context")]
    InputProfileMismatch,
    #[error("capsule repository revision differs from the discovery run baseline revision")]
    BaselineRevisionMismatch,
}

/// Canonical corpus identity for the exact HDC correctness workload.
///
/// Candidate identity is deliberately absent. The corpus is the semantic problem plus the fixed
/// edge-case protocol and canonical seed set, so multiple candidates can be evaluated in one
/// comparable context.
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

/// Exact benchmark invocation this bridge is willing to bind into a rankable HDC context.
/// Recording this command is provenance only; this crate does not execute it.
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

/// Build an implementation identity directly from a collector-produced artifact group.
pub fn implementation_from_capsule(
    capsule: &ExperimentCapsule,
    candidate: HammingCandidate,
    candidate_artifact_group: &str,
) -> Result<ImplementationRecord, HdcEvidenceBridgeError> {
    capsule.validate()?;
    let artifact_id = capsule
        .artifact_group_id(candidate_artifact_group)
        .cloned()
        .ok_or_else(|| {
            HdcEvidenceBridgeError::MissingCandidateArtifactGroup(
                candidate_artifact_group.to_string(),
            )
        })?;
    Ok(implementation_record(candidate, artifact_id)?)
}

/// Opaque correctness binding connecting an exact candidate implementation/artifact to the
/// candidate-independent input corpus on which the laboratory checked it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CapsuleBoundHammingCorrectness {
    id: ContentId,
    implementation_id: symthaea_algorithms::ImplementationId,
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

    pub fn implementation_id(&self) -> &symthaea_algorithms::ImplementationId {
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
        let input_profile_id = hamming_input_profile_id(&self.seeds)?;
        if input_profile_id != self.input_profile_id {
            return Err(HdcEvidenceBridgeError::InputProfileMismatch);
        }
        let expected = derive_bound_correctness_id(
            &self.implementation_id,
            self.candidate,
            &self.candidate_artifact_id,
            &self.raw_correctness_id,
            &self.input_profile_id,
        );
        if expected != self.id {
            return Err(HdcEvidenceBridgeError::CorrectnessImplementationMismatch);
        }
        Ok(())
    }
}

fn derive_bound_correctness_id(
    implementation_id: &symthaea_algorithms::ImplementationId,
    candidate: HammingCandidate,
    candidate_artifact_id: &ContentId,
    raw_correctness_id: &ContentId,
    input_profile_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.hdc-hamming-capsule-correctness.v1",
        [
            implementation_id.as_content_id().as_str().as_bytes(),
            candidate.name().as_bytes(),
            candidate_artifact_id.as_str().as_bytes(),
            raw_correctness_id.as_str().as_bytes(),
            input_profile_id.as_str().as_bytes(),
        ],
    )
}

/// Bind laboratory correctness to the exact implementation bytes collected in the capsule.
pub fn bind_correctness(
    capsule: &ExperimentCapsule,
    candidate_artifact_group: &str,
    implementation: &ImplementationRecord,
    correctness: &HammingCorrectnessEvidence,
) -> Result<CapsuleBoundHammingCorrectness, HdcEvidenceBridgeError> {
    capsule.validate()?;
    correctness.validate()?;
    if !correctness.passed() || correctness.verdict() != CorrectnessVerdict::Passed {
        return Err(HdcEvidenceBridgeError::CorrectnessNotPassed);
    }

    let problem = hamming_problem()?;
    let algorithm = hamming_algorithm(&problem)?;
    implementation.validate_for(&algorithm)?;

    let expected_source_ref = format!(
        "symthaea-algorithm-lab::{}",
        correctness.candidate().name()
    );
    if implementation.source_ref != expected_source_ref {
        return Err(HdcEvidenceBridgeError::CandidateImplementationMismatch);
    }

    let capsule_artifact = capsule
        .artifact_group_id(candidate_artifact_group)
        .ok_or_else(|| {
            HdcEvidenceBridgeError::MissingCandidateArtifactGroup(
                candidate_artifact_group.to_string(),
            )
        })?;
    if &implementation.artifact_id != capsule_artifact {
        return Err(HdcEvidenceBridgeError::CandidateArtifactMismatch);
    }

    let input_profile_id = hamming_input_profile_id(correctness.seeds())?;
    let candidate = correctness.candidate();
    let id = derive_bound_correctness_id(
        &implementation.id,
        candidate,
        &implementation.artifact_id,
        correctness.id(),
        &input_profile_id,
    );
    let bound = CapsuleBoundHammingCorrectness {
        id,
        implementation_id: implementation.id.clone(),
        candidate,
        candidate_artifact_id: implementation.artifact_id.clone(),
        raw_correctness_id: correctness.id().clone(),
        input_profile_id,
        seeds: correctness.seeds().to_vec(),
    };
    bound.validate()?;
    Ok(bound)
}

/// Create the exact comparison context without folding candidate bytes or candidate-specific
/// correctness evidence into the context identity.
pub fn hamming_evaluation_context(
    capsule: &ExperimentCapsule,
    bound_correctness: &CapsuleBoundHammingCorrectness,
    command_index: usize,
) -> Result<EvaluationContext, HdcEvidenceBridgeError> {
    bound_correctness.validate()?;
    let command = capsule
        .commands
        .get(command_index)
        .ok_or(CollectorError::CommandIndexOutOfRange(command_index))?;
    let canonical = canonical_hamming_benchmark_command()?;
    if command != &canonical {
        return Err(HdcEvidenceBridgeError::BenchmarkCommandMismatch);
    }

    Ok(capsule.evaluation_context(
        hamming_evaluator_id(),
        hamming_oracle_id(),
        bound_correctness.input_profile_id.clone(),
        command_index,
        bound_correctness.seeds.clone(),
    )?)
}

/// Mint one historical latency receipt from an externally observed Criterion measurement.
///
/// This function does not run Criterion. `measurement_run_id` must identify the actual external
/// measurement run so append-only repeatability evidence can distinguish reruns.
pub fn latency_receipt_from_capsule(
    capsule: &ExperimentCapsule,
    implementation: &ImplementationRecord,
    bound_correctness: &CapsuleBoundHammingCorrectness,
    command_index: usize,
    latency_ns_per_op: f64,
    measurement_run_id: impl Into<String>,
) -> Result<EvaluationReceipt, HdcEvidenceBridgeError> {
    implementation.validate()?;
    bound_correctness.validate()?;
    if implementation.id != bound_correctness.implementation_id
        || implementation.artifact_id != bound_correctness.candidate_artifact_id
    {
        return Err(HdcEvidenceBridgeError::CorrectnessImplementationMismatch);
    }

    let problem = hamming_problem()?;
    if implementation.problem_id != problem.id {
        return Err(HdcEvidenceBridgeError::CorrectnessImplementationMismatch);
    }

    let context = hamming_evaluation_context(capsule, bound_correctness, command_index)?;
    if context.input_profile_id != bound_correctness.input_profile_id {
        return Err(HdcEvidenceBridgeError::InputProfileMismatch);
    }

    Ok(EvaluationReceipt::new(
        problem.id,
        implementation.id.clone(),
        context,
        CorrectnessVerdict::Passed,
        bound_correctness.id.clone(),
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
    use symthaea_algorithm_lab::verify_candidate;
    use symthaea_algorithms::pareto::ParetoCohort;

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
                "0123456789abcdef",
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
    fn same_corpus_is_candidate_independent() {
        let a = hamming_input_profile_id(&[7, 3, 7]).unwrap();
        let b = hamming_input_profile_id(&[3, 7]).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_candidate_artifacts_share_comparison_context() {
        let capsule_a = capsule_with_candidate_bytes("a", b"byte popcount candidate");
        let capsule_b = capsule_with_candidate_bytes("b", b"u64 popcount candidate");
        assert_ne!(
            capsule_a.artifact_group_id("candidate-source"),
            capsule_b.artifact_group_id("candidate-source")
        );
        assert_ne!(capsule_a.id, capsule_b.id);
        assert_eq!(
            capsule_a.comparison_environment_id(),
            capsule_b.comparison_environment_id()
        );

        let impl_a = implementation_from_capsule(
            &capsule_a,
            HammingCandidate::BytePopcount,
            "candidate-source",
        )
        .unwrap();
        let impl_b = implementation_from_capsule(
            &capsule_b,
            HammingCandidate::U64Popcount,
            "candidate-source",
        )
        .unwrap();
        assert_ne!(impl_a.id, impl_b.id);
        assert_ne!(impl_a.artifact_id, impl_b.artifact_id);

        let raw_a = verify_candidate(HammingCandidate::BytePopcount, &[1, 2, 3, 5, 8]);
        let raw_b = verify_candidate(HammingCandidate::U64Popcount, &[1, 2, 3, 5, 8]);
        let bound_a = bind_correctness(&capsule_a, "candidate-source", &impl_a, &raw_a).unwrap();
        let bound_b = bind_correctness(&capsule_b, "candidate-source", &impl_b, &raw_b).unwrap();
        assert_ne!(bound_a.id(), bound_b.id());
        assert_eq!(bound_a.input_profile_id(), bound_b.input_profile_id());

        let context_a = hamming_evaluation_context(&capsule_a, &bound_a, 0).unwrap();
        let context_b = hamming_evaluation_context(&capsule_b, &bound_b, 0).unwrap();
        assert_eq!(context_a, context_b);

        let receipt_a = latency_receipt_from_capsule(
            &capsule_a,
            &impl_a,
            &bound_a,
            0,
            11.0,
            "measurement-a",
        )
        .unwrap();
        let receipt_b = latency_receipt_from_capsule(
            &capsule_b,
            &impl_b,
            &bound_b,
            0,
            9.0,
            "measurement-b",
        )
        .unwrap();

        let cohort = ParetoCohort::new(vec![receipt_a, receipt_b]).unwrap();
        assert_eq!(cohort.frontier().len(), 1);
        assert_eq!(cohort.frontier()[0].implementation_id, impl_b.id);
    }

    #[test]
    fn candidate_cannot_borrow_another_candidates_correctness() {
        let capsule = capsule_with_candidate_bytes("borrow", b"shared source artifact");
        let implementation = implementation_from_capsule(
            &capsule,
            HammingCandidate::U64Popcount,
            "candidate-source",
        )
        .unwrap();
        let raw = verify_candidate(HammingCandidate::BytePopcount, &[1, 2, 3]);
        assert!(matches!(
            bind_correctness(&capsule, "candidate-source", &implementation, &raw),
            Err(HdcEvidenceBridgeError::CandidateImplementationMismatch)
        ));
    }

    #[test]
    fn changed_artifact_requires_new_implementation_and_correctness_binding() {
        let old_capsule = capsule_with_candidate_bytes("old", b"candidate v1");
        let new_capsule = capsule_with_candidate_bytes("new", b"candidate v2");
        let old_impl = implementation_from_capsule(
            &old_capsule,
            HammingCandidate::BytePopcount,
            "candidate-source",
        )
        .unwrap();
        let raw = verify_candidate(HammingCandidate::BytePopcount, &[1, 2, 3]);
        assert!(matches!(
            bind_correctness(&new_capsule, "candidate-source", &old_impl, &raw),
            Err(HdcEvidenceBridgeError::CandidateArtifactMismatch)
        ));
    }

    #[test]
    fn wrong_benchmark_command_fails_closed() {
        let mut capsule = capsule_with_candidate_bytes("command", b"candidate");
        capsule.commands = vec![CommandSpec::new("cargo", vec!["test".into()]).unwrap()];
        // Re-seal to make command substitution structurally canonical; semantic admission still
        // belongs to this HDC-specific bridge.
        capsule = ExperimentCapsule::new(
            capsule.repository,
            capsule.machine,
            capsule.toolchain,
            capsule.environment,
            capsule.artifact_groups,
            capsule.commands,
        )
        .unwrap();
        let implementation = implementation_from_capsule(
            &capsule,
            HammingCandidate::BytePopcount,
            "candidate-source",
        )
        .unwrap();
        let raw = verify_candidate(HammingCandidate::BytePopcount, &[1]);
        let bound = bind_correctness(&capsule, "candidate-source", &implementation, &raw).unwrap();
        assert!(matches!(
            hamming_evaluation_context(&capsule, &bound, 0),
            Err(HdcEvidenceBridgeError::BenchmarkCommandMismatch)
        ));
    }
}
