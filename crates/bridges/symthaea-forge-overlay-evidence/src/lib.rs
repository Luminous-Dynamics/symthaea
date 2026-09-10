// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Declared candidate-overlay provenance for Forge evaluation.
//!
//! The ordinary experiment collector remains strict: a dirty worktree cannot call its normal
//! `evaluation_context()` method. This crate introduces one narrower theorem for generated Forge
//! candidates:
//!
//! ```text
//! dirty worktree
//! + exact frozen HEAD
//! + exactly one Git-visible unstaged modification
//! + modification path == collected candidate path
//! + current file bytes == exact Forge survivor bytes
//! + Forge <-> collector byte binding passes
//! => declared candidate overlay
//! ```
//!
//! Only then may this bridge construct an evaluation context equivalent to the collector's clean
//! context formula. Candidate/overlay identities are deliberately excluded from that context so
//! different implementations evaluated under the same baseline, command, machine, toolchain and
//! input profile remain comparable. Candidate identity stays in `ImplementationRecord`.
//!
//! This is provenance admission, not execution, correctness, performance, replication, promotion,
//! or runtime authority.

use std::path::{Component, Path};
use std::process::Command;
use symthaea_algorithm_evidence_collector::ExperimentCapsule;
use symthaea_algorithms::evaluation::{EvaluationContext, EvaluationError};
use symthaea_algorithms::discovery::{CandidateProposal, DiscoveryError, DiscoveryRun};
use symthaea_algorithms::ContentId;
use symthaea_forge::{CertificateError, ForgeCandidate};
use symthaea_forge_algorithm_evidence::{
    ForgeCollectorBinding, ForgeCollectorBindingError, bind_forge_candidate_to_capsule,
};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OverlayError {
    #[error(transparent)]
    Certificate(#[from] CertificateError),
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    #[error(transparent)]
    Binding(#[from] ForgeCollectorBindingError),
    #[error(transparent)]
    Evaluation(#[from] EvaluationError),
    #[error("experiment capsule is invalid: {0}")]
    InvalidCapsule(String),
    #[error("repository root is invalid or unavailable: {0}")]
    InvalidRepositoryRoot(String),
    #[error("provided repository root is not the Git top-level directory")]
    NotRepositoryTopLevel,
    #[error("Git command failed: {0}")]
    GitCommandFailed(String),
    #[error("Git command returned non-UTF-8 output")]
    NonUtf8GitOutput,
    #[error("live Git HEAD does not match the capsule/discovery baseline")]
    HeadRevisionMismatch,
    #[error("declared Forge overlay requires a dirty base capsule")]
    CapsuleUnexpectedlyClean,
    #[error("live worktree status does not match the status committed by the capsule")]
    WorktreeStatusMismatch,
    #[error("declared overlay requires exactly one unstaged modified tracked file")]
    UnexpectedWorktreeStatus,
    #[error("candidate path is not a canonical repository-relative path")]
    InvalidCandidatePath,
    #[error("candidate file is a symbolic link")]
    CandidateSymlink,
    #[error("candidate file resolves outside the repository root")]
    CandidateEscapesRepository,
    #[error("candidate file bytes differ from the exact Forge survivor")]
    CandidateBytesMismatch,
    #[error("overlay evidence does not belong to the supplied capsule")]
    CapsuleBindingMismatch,
    #[error("command index {0} is out of range")]
    CommandIndexOutOfRange(usize),
}

/// Opaque evidence that one dirty capsule's entire Git-visible delta is the exact Forge candidate
/// file admitted by the cross-domain byte binding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredForgeOverlay {
    id: ContentId,
    binding_id: ContentId,
    capsule_id: ContentId,
    worktree_status_id: ContentId,
    relative_path: String,
    baseline_revision: String,
}

impl DeclaredForgeOverlay {
    pub fn id(&self) -> &ContentId {
        &self.id
    }

    pub fn binding_id(&self) -> &ContentId {
        &self.binding_id
    }

    pub fn capsule_id(&self) -> &ContentId {
        &self.capsule_id
    }

    pub fn worktree_status_id(&self) -> &ContentId {
        &self.worktree_status_id
    }

    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    /// Construct a comparison context for this declared overlay.
    ///
    /// This intentionally mirrors the collector's v1 context formula except for its clean-worktree
    /// rejection. The overlay identity itself is not folded into the context; otherwise each
    /// candidate would become incomparable merely because its source bytes differ.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluation_context(
        &self,
        capsule: &ExperimentCapsule,
        base_evaluator_id: ContentId,
        oracle_id: ContentId,
        input_profile_id: ContentId,
        command_index: usize,
        seeds: Vec<u64>,
    ) -> Result<EvaluationContext, OverlayError> {
        capsule
            .validate()
            .map_err(|error| OverlayError::InvalidCapsule(error.to_string()))?;
        if capsule.id != self.capsule_id
            || capsule.repository.worktree_status_id != self.worktree_status_id
            || capsule.repository.revision != self.baseline_revision
            || capsule.repository.worktree_clean
        {
            return Err(OverlayError::CapsuleBindingMismatch);
        }
        let command = capsule
            .commands
            .get(command_index)
            .ok_or(OverlayError::CommandIndexOutOfRange(command_index))?;
        let evaluator_id = ContentId::derive(
            "symthaea.capsule-bound-evaluator.v1",
            [
                base_evaluator_id.as_str().as_bytes(),
                command.id.as_str().as_bytes(),
            ],
        );
        Ok(EvaluationContext::new(
            evaluator_id,
            oracle_id,
            input_profile_id,
            capsule.comparison_environment_id(),
            capsule.repository.revision.clone(),
            capsule.toolchain.id.to_string(),
            capsule.machine.host_triple.clone(),
            seeds,
        )?)
    }
}

fn validate_relative_path(path: &str) -> Result<(), OverlayError> {
    if path.trim().is_empty()
        || path.trim() != path
        || path.chars().any(char::is_control)
        || Path::new(path).is_absolute()
        || Path::new(path).components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(OverlayError::InvalidCandidatePath);
    }
    Ok(())
}

fn run_git(root: &Path, args: &[&str]) -> Result<String, OverlayError> {
    let output = Command::new("git")
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|error| OverlayError::GitCommandFailed(error.to_string()))?;
    if !output.status.success() {
        return Err(OverlayError::GitCommandFailed(
            String::from_utf8_lossy(&output.stderr).trim().to_string(),
        ));
    }
    let stdout = String::from_utf8(output.stdout).map_err(|_| OverlayError::NonUtf8GitOutput)?;
    // Match the collector v1 command normalization exactly. The v1 overlay theorem below accepts
    // one status line only, so trimming the outer whitespace cannot hide an additional entry.
    Ok(stdout.trim().to_string())
}

/// Observe and admit exactly one Forge candidate as the complete Git-visible worktree delta.
pub fn observe_declared_forge_overlay(
    repository_root: &Path,
    run: &DiscoveryRun,
    proposal: &CandidateProposal,
    candidate: &ForgeCandidate,
    capsule: &ExperimentCapsule,
    candidate_group_label: &str,
    expected_relative_path: &str,
) -> Result<(ForgeCollectorBinding, DeclaredForgeOverlay), OverlayError> {
    validate_relative_path(expected_relative_path)?;
    run.validate()?;
    proposal.validate_for(run)?;
    candidate.validate()?;
    capsule
        .validate()
        .map_err(|error| OverlayError::InvalidCapsule(error.to_string()))?;
    if capsule.repository.worktree_clean {
        return Err(OverlayError::CapsuleUnexpectedlyClean);
    }

    let root = repository_root
        .canonicalize()
        .map_err(|error| OverlayError::InvalidRepositoryRoot(error.to_string()))?;
    if !root.is_dir() {
        return Err(OverlayError::InvalidRepositoryRoot(
            root.display().to_string(),
        ));
    }
    let git_root = Path::new(&run_git(&root, &["rev-parse", "--show-toplevel"])? )
        .canonicalize()
        .map_err(|error| OverlayError::InvalidRepositoryRoot(error.to_string()))?;
    if git_root != root {
        return Err(OverlayError::NotRepositoryTopLevel);
    }

    let live_head = run_git(&root, &["rev-parse", "--verify", "HEAD"])?;
    if live_head != run.baseline_revision || live_head != capsule.repository.revision {
        return Err(OverlayError::HeadRevisionMismatch);
    }

    let status = run_git(
        &root,
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    let status_id = ContentId::derive("symthaea.git-worktree-status.v1", [status.as_bytes()]);
    if status_id != capsule.repository.worktree_status_id {
        return Err(OverlayError::WorktreeStatusMismatch);
    }
    if status.lines().count() != 1 || status != format!("M {expected_relative_path}") {
        // Collector v1 trims the leading space from a single raw ` M path` porcelain line. A
        // staged `M  path`, untracked `?? path`, rename, deletion, or any multi-file state fails.
        return Err(OverlayError::UnexpectedWorktreeStatus);
    }

    let requested = root.join(expected_relative_path);
    let metadata = std::fs::symlink_metadata(&requested)
        .map_err(|error| OverlayError::InvalidRepositoryRoot(error.to_string()))?;
    if metadata.file_type().is_symlink() {
        return Err(OverlayError::CandidateSymlink);
    }
    let canonical_candidate = requested
        .canonicalize()
        .map_err(|error| OverlayError::InvalidRepositoryRoot(error.to_string()))?;
    if !canonical_candidate.starts_with(&root) {
        return Err(OverlayError::CandidateEscapesRepository);
    }
    let live_source = std::fs::read_to_string(&canonical_candidate)
        .map_err(|error| OverlayError::InvalidRepositoryRoot(error.to_string()))?;
    if live_source != candidate.full_source() {
        return Err(OverlayError::CandidateBytesMismatch);
    }

    let binding = bind_forge_candidate_to_capsule(
        run,
        proposal,
        candidate,
        capsule,
        candidate_group_label,
        expected_relative_path,
    )?;
    let id = ContentId::derive(
        "symthaea.declared-forge-overlay.v1",
        [
            binding.id().as_str().as_bytes(),
            capsule.id.as_str().as_bytes(),
            status_id.as_str().as_bytes(),
            expected_relative_path.as_bytes(),
            live_head.as_bytes(),
        ],
    );
    let overlay = DeclaredForgeOverlay {
        id,
        binding_id: binding.id().clone(),
        capsule_id: capsule.id.clone(),
        worktree_status_id: status_id,
        relative_path: expected_relative_path.to_string(),
        baseline_revision: live_head,
    };
    Ok((binding, overlay))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};
    use symthaea_algorithm_evidence_collector::{
        ArtifactGroupSpec, CollectorPolicy, CommandSpec, collect_experiment_capsule,
    };
    use symthaea_algorithms::discovery::{DiscoveryPolicy, SearchBudget};
    use symthaea_algorithms::{
        AlgorithmProvenance, AlgorithmRecord, DeterminismRequirement, DiscoveryRisk,
        ImplementationRecord, ProblemSpec, SemanticGuarantee,
    };
    use symthaea_forge::certificate::{
        ForgeCertificate, GateEvidence, MutationRecord, full_source_artifact_id,
    };
    use symthaea_forge::{ForgeCandidate, proposal_from_forge};

    fn temp_repo() -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "symthaea-forge-overlay-{}-{nonce}",
            std::process::id()
        ));
        fs::create_dir_all(root.join("src")).unwrap();
        root
    }

    fn git(root: &Path, args: &[&str]) -> String {
        let output = Command::new("git")
            .args(args)
            .current_dir(root)
            .output()
            .unwrap();
        assert!(output.status.success(), "git {:?}: {}", args, String::from_utf8_lossy(&output.stderr));
        String::from_utf8(output.stdout).unwrap().trim().to_string()
    }

    fn init_repo(root: &Path, baseline_source: &str) -> String {
        fs::write(root.join("src/target.rs"), baseline_source).unwrap();
        git(root, &["init", "-q"]);
        git(root, &["config", "user.email", "forge-overlay-test@example.invalid"]);
        git(root, &["config", "user.name", "Forge Overlay Test"]);
        git(root, &["add", "src/target.rs"]);
        git(root, &["commit", "-q", "-m", "baseline"]);
        git(root, &["rev-parse", "HEAD"])
    }

    fn semantic_context(
        revision: &str,
        baseline_source: &str,
    ) -> (DiscoveryRun, AlgorithmRecord, ImplementationRecord) {
        let problem = ProblemSpec::new(
            "forge-overlay-test",
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
        let baseline = ImplementationRecord::new(
            problem.id.clone(),
            algorithm.id.clone(),
            "repo://src/target.rs",
            full_source_artifact_id(baseline_source),
            None,
        )
        .unwrap();
        let run = DiscoveryRun::new(
            &problem,
            DiscoveryPolicy::default(),
            ContentId::derive("generator", [b"forge-overlay-test".as_slice()]),
            revision,
            SearchBudget::new(10, 2, 10).unwrap(),
            7,
        )
        .unwrap();
        (run, algorithm, baseline)
    }

    fn forge_candidate(
        revision: &str,
        generation: usize,
        baseline_source: &str,
        candidate_source: &str,
    ) -> ForgeCandidate {
        let baseline_id = full_source_artifact_id(baseline_source);
        let candidate_id = full_source_artifact_id(candidate_source);
        let mutation = MutationRecord::new(
            generation,
            "NumericLiteralPerturb",
            format!("baseline -> candidate-{generation}"),
            baseline_id.clone(),
            candidate_id.clone(),
        );
        ForgeCandidate::new(
            ForgeCertificate {
                generated_at_unix_ms: 0,
                target_file: PathBuf::from("src/target.rs"),
                target_function: "target".into(),
                package: "test-package".into(),
                git_sha: Some(revision.into()),
                generation,
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
                before_source: baseline_source.trim().into(),
                after_source: candidate_source.trim().into(),
            },
            candidate_source.into(),
        )
        .unwrap()
    }

    fn collect_dirty_capsule(root: &Path) -> ExperimentCapsule {
        collect_experiment_capsule(
            root,
            &[ArtifactGroupSpec::new(
                "forge-candidate",
                vec!["src/target.rs".into()],
            )
            .unwrap()],
            vec![CommandSpec::new("cargo", vec!["test".into()]).unwrap()],
            CollectorPolicy {
                require_clean_worktree: false,
            },
        )
        .unwrap()
    }

    #[test]
    fn two_exact_overlays_share_context_but_keep_distinct_candidate_identity() {
        let root = temp_repo();
        let baseline_source = "fn target() -> i32 { 1 }\n";
        let revision = init_repo(&root, baseline_source);
        let (run, algorithm, baseline) = semantic_context(&revision, baseline_source);

        let candidate_a = forge_candidate(
            &revision,
            0,
            baseline_source,
            "fn target() -> i32 { 2 }\n",
        );
        fs::write(root.join("src/target.rs"), candidate_a.full_source()).unwrap();
        let capsule_a = collect_dirty_capsule(&root);
        let proposal_a = proposal_from_forge(&run, &algorithm, &baseline, &candidate_a).unwrap();
        let (binding_a, overlay_a) = observe_declared_forge_overlay(
            &root,
            &run,
            &proposal_a,
            &candidate_a,
            &capsule_a,
            "forge-candidate",
            "src/target.rs",
        )
        .unwrap();
        let context_a = overlay_a
            .evaluation_context(
                &capsule_a,
                ContentId::derive("evaluator", [b"criterion".as_slice()]),
                ContentId::derive("oracle", [b"reference".as_slice()]),
                ContentId::derive("inputs", [b"fixed".as_slice()]),
                0,
                vec![1, 2, 3],
            )
            .unwrap();

        fs::write(root.join("src/target.rs"), baseline_source).unwrap();
        assert!(git(&root, &["status", "--porcelain=v1"]).is_empty());

        let candidate_b = forge_candidate(
            &revision,
            0,
            baseline_source,
            "fn target() -> i32 { 3 }\n",
        );
        fs::write(root.join("src/target.rs"), candidate_b.full_source()).unwrap();
        let capsule_b = collect_dirty_capsule(&root);
        let proposal_b = proposal_from_forge(&run, &algorithm, &baseline, &candidate_b).unwrap();
        let (binding_b, overlay_b) = observe_declared_forge_overlay(
            &root,
            &run,
            &proposal_b,
            &candidate_b,
            &capsule_b,
            "forge-candidate",
            "src/target.rs",
        )
        .unwrap();
        let context_b = overlay_b
            .evaluation_context(
                &capsule_b,
                ContentId::derive("evaluator", [b"criterion".as_slice()]),
                ContentId::derive("oracle", [b"reference".as_slice()]),
                ContentId::derive("inputs", [b"fixed".as_slice()]),
                0,
                vec![1, 2, 3],
            )
            .unwrap();

        assert_ne!(candidate_a.artifact_id(), candidate_b.artifact_id());
        assert_ne!(binding_a.id(), binding_b.id());
        assert_ne!(overlay_a.id(), overlay_b.id());
        assert_eq!(context_a, context_b);

        fs::write(root.join("src/target.rs"), baseline_source).unwrap();
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn unrelated_dirty_file_blocks_overlay_admission() {
        let root = temp_repo();
        let baseline_source = "fn target() -> i32 { 1 }\n";
        let revision = init_repo(&root, baseline_source);
        fs::write(root.join("other.txt"), "baseline\n").unwrap();
        git(&root, &["add", "other.txt"]);
        git(&root, &["commit", "-q", "-m", "add other"]);
        let revision = git(&root, &["rev-parse", "HEAD"]);
        let (run, algorithm, baseline) = semantic_context(&revision, baseline_source);
        let candidate = forge_candidate(
            &revision,
            0,
            baseline_source,
            "fn target() -> i32 { 2 }\n",
        );
        fs::write(root.join("src/target.rs"), candidate.full_source()).unwrap();
        fs::write(root.join("other.txt"), "unrelated change\n").unwrap();
        let capsule = collect_dirty_capsule(&root);
        let proposal = proposal_from_forge(&run, &algorithm, &baseline, &candidate).unwrap();
        assert!(matches!(
            observe_declared_forge_overlay(
                &root,
                &run,
                &proposal,
                &candidate,
                &capsule,
                "forge-candidate",
                "src/target.rs",
            ),
            Err(OverlayError::UnexpectedWorktreeStatus)
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn staged_candidate_is_not_an_admitted_overlay() {
        let root = temp_repo();
        let baseline_source = "fn target() -> i32 { 1 }\n";
        let revision = init_repo(&root, baseline_source);
        let (run, algorithm, baseline) = semantic_context(&revision, baseline_source);
        let candidate = forge_candidate(
            &revision,
            0,
            baseline_source,
            "fn target() -> i32 { 2 }\n",
        );
        fs::write(root.join("src/target.rs"), candidate.full_source()).unwrap();
        git(&root, &["add", "src/target.rs"]);
        let capsule = collect_dirty_capsule(&root);
        let proposal = proposal_from_forge(&run, &algorithm, &baseline, &candidate).unwrap();
        assert!(matches!(
            observe_declared_forge_overlay(
                &root,
                &run,
                &proposal,
                &candidate,
                &capsule,
                "forge-candidate",
                "src/target.rs",
            ),
            Err(OverlayError::UnexpectedWorktreeStatus)
        ));
        let _ = fs::remove_dir_all(root);
    }
}