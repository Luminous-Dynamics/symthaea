// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-domain evidence binding between a Forge candidate and the experiment collector.
//!
//! Forge and the collector intentionally use different identities:
//!
//! ```text
//! Forge artifact ID      = domain("forge-full-source") + exact bytes
//! Collector source ID    = domain("algorithm-source-file") + relative path + exact bytes
//! Collector group ID     = label + ordered file evidence
//! ```
//!
//! Equality of those IDs would be a category error. This bridge proves instead that both identity
//! systems were derived from the **same supplied byte sequence** for one exact candidate file.
//! The resulting binding is descriptive evidence only; in particular it does not make a dirty
//! experiment capsule rankable. A later declared-overlay provenance layer must establish that the
//! candidate was the only admitted workspace difference during evaluation.

use symthaea_algorithm_evidence_collector::{ArtifactGroupEvidence, ExperimentCapsule};
use symthaea_algorithms::discovery::{CandidateProposal, DiscoveryError, DiscoveryRun};
use symthaea_algorithms::{ContentId, ImplementationId};
use symthaea_forge::{CertificateError, ForgeCandidate};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ForgeCollectorBindingError {
    #[error(transparent)]
    Certificate(#[from] CertificateError),
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    #[error("experiment capsule is invalid: {0}")]
    InvalidCapsule(String),
    #[error("experiment capsule revision does not match the discovery-run baseline")]
    BaselineRevisionMismatch,
    #[error("candidate proposal does not identify the exact Forge survivor artifact")]
    ProposalArtifactMismatch,
    #[error("collector artifact group `{0}` is missing")]
    MissingArtifactGroup(String),
    #[error("Forge binding requires an artifact group containing exactly one candidate file")]
    CandidateGroupNotSingleton,
    #[error("collector candidate path differs from the expected path")]
    CandidatePathMismatch,
    #[error("collector byte length differs from the exact Forge survivor")]
    CandidateLengthMismatch,
    #[error("collector path+bytes identity does not match the exact Forge survivor bytes")]
    CollectorFileIdentityMismatch,
}

/// Opaque proof that one Forge candidate/proposal and one collector artifact group refer to the
/// same exact candidate bytes under their respective identity domains.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ForgeCollectorBinding {
    id: ContentId,
    run_id: ContentId,
    implementation_id: ImplementationId,
    capsule_id: ContentId,
    forge_artifact_id: ContentId,
    collector_group_id: ContentId,
    collector_file_id: ContentId,
    relative_path: String,
    byte_len: u64,
    clean_capsule: bool,
}

impl ForgeCollectorBinding {
    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn implementation_id(&self) -> &ImplementationId {
        &self.implementation_id
    }

    pub fn forge_artifact_id(&self) -> &ContentId {
        &self.forge_artifact_id
    }

    pub fn collector_group_id(&self) -> &ContentId {
        &self.collector_group_id
    }

    pub fn collector_file_id(&self) -> &ContentId {
        &self.collector_file_id
    }

    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    pub fn byte_len(&self) -> u64 {
        self.byte_len
    }

    /// Descriptive only. `false` means the base capsule remains audit-only under the collector's
    /// existing clean-worktree rule; this binding does not override that rule.
    pub fn clean_capsule(&self) -> bool {
        self.clean_capsule
    }
}

/// Derive the collector's v1 path+bytes source-file identity for cross-domain verification.
///
/// This intentionally mirrors the collector's versioned v1 identity theorem. If that theorem
/// changes, this bridge's tests against real collector output must fail rather than silently
/// treating the new identity as equivalent.
fn collector_source_file_id(relative_path: &str, bytes: &[u8]) -> ContentId {
    ContentId::derive(
        "symthaea.algorithm-source-file.v1",
        [relative_path.as_bytes(), bytes],
    )
}

fn singleton_group<'a>(
    capsule: &'a ExperimentCapsule,
    label: &str,
) -> Result<&'a ArtifactGroupEvidence, ForgeCollectorBindingError> {
    let group = capsule
        .artifact_groups
        .iter()
        .find(|group| group.label == label)
        .ok_or_else(|| ForgeCollectorBindingError::MissingArtifactGroup(label.to_string()))?;
    if group.files.len() != 1 {
        return Err(ForgeCollectorBindingError::CandidateGroupNotSingleton);
    }
    Ok(group)
}

/// Prove cross-domain byte equivalence for one Forge-generated single-file candidate.
///
/// The capsule may be clean or dirty. A dirty capsule remains audit-only here. This function does
/// not call `ExperimentCapsule::evaluation_context()` and creates no performance receipt.
pub fn bind_forge_candidate_to_capsule(
    run: &DiscoveryRun,
    proposal: &CandidateProposal,
    candidate: &ForgeCandidate,
    capsule: &ExperimentCapsule,
    candidate_group_label: &str,
    expected_relative_path: &str,
) -> Result<ForgeCollectorBinding, ForgeCollectorBindingError> {
    run.validate()?;
    proposal.validate_for(run)?;
    candidate.validate()?;
    capsule
        .validate()
        .map_err(|error| ForgeCollectorBindingError::InvalidCapsule(error.to_string()))?;

    if capsule.repository.revision != run.baseline_revision {
        return Err(ForgeCollectorBindingError::BaselineRevisionMismatch);
    }
    if proposal.implementation.artifact_id != *candidate.artifact_id()
        || proposal.artifact.content_id != *candidate.artifact_id()
    {
        return Err(ForgeCollectorBindingError::ProposalArtifactMismatch);
    }

    let group = singleton_group(capsule, candidate_group_label)?;
    let file = &group.files[0];
    if file.relative_path != expected_relative_path {
        return Err(ForgeCollectorBindingError::CandidatePathMismatch);
    }
    let bytes = candidate.full_source().as_bytes();
    let byte_len = u64::try_from(bytes.len())
        .map_err(|_| ForgeCollectorBindingError::CandidateLengthMismatch)?;
    if file.byte_len != byte_len {
        return Err(ForgeCollectorBindingError::CandidateLengthMismatch);
    }
    let expected_collector_file_id = collector_source_file_id(expected_relative_path, bytes);
    if file.content_id != expected_collector_file_id {
        return Err(ForgeCollectorBindingError::CollectorFileIdentityMismatch);
    }

    let id = ContentId::derive(
        "symthaea.forge-collector-binding.v1",
        [
            run.id.as_str().as_bytes(),
            proposal.implementation.id.as_content_id().as_str().as_bytes(),
            capsule.id.as_str().as_bytes(),
            candidate.artifact_id().as_str().as_bytes(),
            group.id.as_str().as_bytes(),
            file.content_id.as_str().as_bytes(),
            expected_relative_path.as_bytes(),
            byte_len.to_be_bytes().as_slice(),
        ],
    );

    Ok(ForgeCollectorBinding {
        id,
        run_id: run.id.clone(),
        implementation_id: proposal.implementation.id.clone(),
        capsule_id: capsule.id.clone(),
        forge_artifact_id: candidate.artifact_id().clone(),
        collector_group_id: group.id.clone(),
        collector_file_id: file.content_id.clone(),
        relative_path: expected_relative_path.to_string(),
        byte_len,
        clean_capsule: capsule.repository.worktree_clean,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use symthaea_algorithm_evidence_collector::{
        ArtifactGroupSpec, CapturedEnvironment, CommandSpec, MachineProfile, RepositoryState,
        ToolchainProfile, collect_artifact_group,
    };
    use symthaea_algorithms::discovery::{DiscoveryPolicy, SearchBudget};
    use symthaea_algorithms::{
        AlgorithmProvenance, AlgorithmRecord, ContentId, DeterminismRequirement, DiscoveryRisk,
        ImplementationRecord, ProblemSpec, SemanticGuarantee,
    };
    use symthaea_forge::certificate::{
        ForgeCertificate, GateEvidence, MutationRecord, full_source_artifact_id,
    };
    use symthaea_forge::{ForgeCandidate, proposal_from_forge};

    const REVISION: &str = "abc123";

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn temp_root(label: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "symthaea-forge-collector-binding-{label}-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        root
    }

    fn candidate_fixture() -> (
        DiscoveryRun,
        CandidateProposal,
        ForgeCandidate,
    ) {
        let problem = ProblemSpec::new(
            "forge-binding-test",
            "Return the exact reference value.",
            SemanticGuarantee::Exact,
            DeterminismRequirement::Required,
            vec!["matches oracle".into()],
            DiscoveryRisk::Ordinary,
        )
        .unwrap();
        let algorithm = AlgorithmRecord::new(
            problem.id.clone(),
            "forge-local-family",
            "Local structural mutation search.",
            AlgorithmProvenance::Evolved,
        )
        .unwrap();
        let baseline_source = "fn target() -> i32 { 1 }\n";
        let candidate_source = "fn target() -> i32 { 2 }\n".to_string();
        let baseline_id = full_source_artifact_id(baseline_source);
        let candidate_id = full_source_artifact_id(&candidate_source);
        let baseline = ImplementationRecord::new(
            problem.id.clone(),
            algorithm.id.clone(),
            "repo://src/target.rs",
            baseline_id.clone(),
            None,
        )
        .unwrap();
        let run = DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            cid("generator", "forge"),
            REVISION,
            SearchBudget::new(10, 5, 10).unwrap(),
            7,
        )
        .unwrap();
        let mutation = MutationRecord::new(
            1,
            "NumericLiteralPerturb",
            "1 -> 2",
            baseline_id.clone(),
            candidate_id.clone(),
        );
        let certificate = ForgeCertificate {
            generated_at_unix_ms: 0,
            target_file: PathBuf::from("src/target.rs"),
            target_function: "target".into(),
            package: "test-package".into(),
            git_sha: Some(REVISION.into()),
            generation: 1,
            baseline_artifact_id: baseline_id,
            candidate_artifact_id: candidate_id,
            mutation_operator: mutation.operator.clone(),
            mutation_detail: mutation.detail.clone(),
            mutation_history: vec![mutation],
            gates: vec![
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
            ],
            benchmark: None,
            before_source: "fn target() -> i32 { 1 }".into(),
            after_source: "fn target() -> i32 { 2 }".into(),
        };
        let candidate = ForgeCandidate::new(certificate, candidate_source).unwrap();
        let proposal = proposal_from_forge(&run, &algorithm, &baseline, &candidate).unwrap();
        (run, proposal, candidate)
    }

    fn capsule_for_source(
        source: &str,
        extra_file: bool,
        worktree_clean: bool,
    ) -> ExperimentCapsule {
        let root = temp_root("capsule");
        fs::create_dir_all(root.join("generated")).unwrap();
        fs::write(root.join("generated/candidate.rs"), source).unwrap();
        let mut paths = vec!["generated/candidate.rs".to_string()];
        if extra_file {
            fs::write(root.join("generated/other.txt"), "other").unwrap();
            paths.push("generated/other.txt".into());
        }
        let group = collect_artifact_group(
            &root,
            &ArtifactGroupSpec::new("forge-candidate", paths).unwrap(),
        )
        .unwrap();
        let capsule = ExperimentCapsule::new(
            RepositoryState::new(
                REVISION,
                worktree_clean,
                cid("status", if worktree_clean { "clean" } else { "declared-dirty" }),
                Some(cid("cargo", "lock")),
                Some(cid("flake", "lock")),
                Some(cid("rust-toolchain", "file")),
            )
            .unwrap(),
            MachineProfile::new(
                "linux",
                "x86_64",
                "test-kernel",
                Some("test-cpu".into()),
                vec!["popcnt".into()],
                "x86_64-unknown-linux-gnu",
                vec!["target_feature=\"sse2\"".into()],
            )
            .unwrap(),
            ToolchainProfile::new(
                "rustc 1.96.0\nhost: x86_64-unknown-linux-gnu",
                "cargo 1.96.0",
                None,
            )
            .unwrap(),
            CapturedEnvironment::new(BTreeMap::new()),
            vec![group],
            vec![CommandSpec::new("cargo", vec!["test".into()]).unwrap()],
        )
        .unwrap();
        fs::remove_dir_all(root).unwrap();
        capsule
    }

    #[test]
    fn binds_different_identity_domains_to_same_exact_bytes() {
        let (run, proposal, candidate) = candidate_fixture();
        let capsule = capsule_for_source(candidate.full_source(), false, false);
        let binding = bind_forge_candidate_to_capsule(
            &run,
            &proposal,
            &candidate,
            &capsule,
            "forge-candidate",
            "generated/candidate.rs",
        )
        .unwrap();
        assert_eq!(binding.forge_artifact_id(), candidate.artifact_id());
        assert_ne!(binding.forge_artifact_id(), binding.collector_file_id());
        assert!(!binding.clean_capsule());
    }

    #[test]
    fn different_bytes_fail_cross_domain_binding() {
        let (run, proposal, candidate) = candidate_fixture();
        let capsule = capsule_for_source("fn target() -> i32 { 999 }\n", false, false);
        assert!(matches!(
            bind_forge_candidate_to_capsule(
                &run,
                &proposal,
                &candidate,
                &capsule,
                "forge-candidate",
                "generated/candidate.rs",
            ),
            Err(ForgeCollectorBindingError::CandidateLengthMismatch)
                | Err(ForgeCollectorBindingError::CollectorFileIdentityMismatch)
        ));
    }

    #[test]
    fn candidate_group_must_be_singleton() {
        let (run, proposal, candidate) = candidate_fixture();
        let capsule = capsule_for_source(candidate.full_source(), true, false);
        assert!(matches!(
            bind_forge_candidate_to_capsule(
                &run,
                &proposal,
                &candidate,
                &capsule,
                "forge-candidate",
                "generated/candidate.rs",
            ),
            Err(ForgeCollectorBindingError::CandidateGroupNotSingleton)
        ));
    }

    #[test]
    fn clean_capsule_status_is_descriptive_not_inferred() {
        let (run, proposal, candidate) = candidate_fixture();
        let capsule = capsule_for_source(candidate.full_source(), false, true);
        let binding = bind_forge_candidate_to_capsule(
            &run,
            &proposal,
            &candidate,
            &capsule,
            "forge-candidate",
            "generated/candidate.rs",
        )
        .unwrap();
        assert!(binding.clean_capsule());
    }
}
