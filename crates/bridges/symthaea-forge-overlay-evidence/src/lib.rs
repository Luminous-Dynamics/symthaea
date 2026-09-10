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
//! Candidate/overlay identities are deliberately excluded from the resulting comparison context,
//! so two implementations evaluated under the same baseline, command, machine, toolchain and input
//! profile remain comparable. Candidate identity stays in `ImplementationRecord`.
//!
//! Context construction re-observes the live Git/worktree/file state. A proof that was valid
//! earlier cannot be reused after the candidate file or repository state changes.
//!
//! This is provenance admission, not execution, correctness, performance, replication, promotion,
//! or runtime authority.

use std::path::{Component, Path, PathBuf};
use std::process::Command;
use symthaea_algorithm_evidence_collector::ExperimentCapsule;
use symthaea_algorithms::discovery::{CandidateProposal, DiscoveryError, DiscoveryRun};
use symthaea_algorithms::evaluation::{EvaluationContext, EvaluationError};
use symthaea_algorithms::ContentId;
use symthaea_forge::{CertificateError, ForgeCandidate};
use symthaea_forge_algorithm_evidence::{
    bind_forge_candidate_to_capsule, ForgeCollectorBinding, ForgeCollectorBindingError,
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
    #[error("candidate file bytes differ from the collector-bound candidate bytes")]
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
    collector_file_id: ContentId,
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

    pub fn collector_file_id(&self) -> &ContentId {
        &self.collector_file_id
    }

    pub fn relative_path(&self) -> &str {
        &self.relative_path
    }

    /// Construct a comparison context only after re-validating the exact live overlay.
    ///
    /// This mirrors the collector's v1 context formula except for its clean-worktree rejection.
    /// The overlay identity itself is not folded into the context; otherwise each candidate would
    /// become incomparable merely because its source bytes differ.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluation_context_from_live_overlay(
        &self,
        repository_root: &Path,
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

        validate_live_overlay_state(
            repository_root,
            &self.baseline_revision,
            capsule,
            &self.relative_path,
            &self.collector_file_id,
        )?;

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
    // Match collector v1 command normalization exactly. The overlay theorem accepts one status
    // line only, so trimming outer whitespace cannot hide a second entry.
    Ok(stdout.trim().to_string())
}

fn canonical_repository_root(repository_root: &Path) -> Result<PathBuf, OverlayError> {
    let root = repository_root
        .canonicalize()
        .map_err(|error| OverlayError::InvalidRepositoryRoot(error.to_string()))?;
    if !root.is_dir() {
        return Err(OverlayError::InvalidRepositoryRoot(
            root.display().to_string(),
        ));
    }
    let reported_root = run_git(&root, &["rev-parse", "--show-toplevel"])?;
    let git_root = PathBuf::from(reported_root)
        .canonicalize()
        .map_err(|error| OverlayError::InvalidRepositoryRoot(error.to_string()))?;
    if git_root != root {
        return Err(OverlayError::NotRepositoryTopLevel);
    }
    Ok(root)
}

fn collector_source_file_id(relative_path: &str, bytes: &[u8]) -> ContentId {
    ContentId::derive(
        "symthaea.algorithm-source-file.v1",
        [relative_path.as_bytes(), bytes],
    )
}

fn validate_live_overlay_state(
    repository_root: &Path,
    baseline_revision: &str,
    capsule: &ExperimentCapsule,
    relative_path: &str,
    collector_file_id: &ContentId,
) -> Result<(), OverlayError> {
    validate_relative_path(relative_path)?;
    let root = canonical_repository_root(repository_root)?;

    let live_head = run_git(&root, &["rev-parse", "--verify", "HEAD"])?;
    if live_head != baseline_revision || live_head != capsule.repository.revision {
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
    if status.lines().count() != 1 || status != format!("M {relative_path}") {
        // Raw porcelain for one unstaged modification begins with ` M`; collector v1 trims the
        // outer leading space. Staged, untracked, renamed, deleted, or multi-file states differ.
        return Err(OverlayError::UnexpectedWorktreeStatus);
    }

    let requested = root.join(relative_path);
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
    let live_bytes = std::fs::read(&canonical_candidate)
        .map_err(|error| OverlayError::InvalidRepositoryRoot(error.to_string()))?;
    if collector_source_file_id(relative_path, &live_bytes) != *collector_file_id {
        return Err(OverlayError::CandidateBytesMismatch);
    }
    Ok(())
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

    let binding = bind_forge_candidate_to_capsule(
        run,
        proposal,
        candidate,
        capsule,
        candidate_group_label,
        expected_relative_path,
    )?;

    validate_live_overlay_state(
        repository_root,
        &run.baseline_revision,
        capsule,
        expected_relative_path,
        binding.collector_file_id(),
    )?;

    let id = ContentId::derive(
        "symthaea.declared-forge-overlay.v2",
        [
            binding.id().as_str().as_bytes(),
            capsule.id.as_str().as_bytes(),
            capsule.repository.worktree_status_id.as_str().as_bytes(),
            binding.collector_file_id().as_str().as_bytes(),
            expected_relative_path.as_bytes(),
            run.baseline_revision.as_bytes(),
        ],
    );
    let overlay = DeclaredForgeOverlay {
        id,
        binding_id: binding.id().clone(),
        capsule_id: capsule.id.clone(),
        worktree_status_id: capsule.repository.worktree_status_id.clone(),
        collector_file_id: binding.collector_file_id().clone(),
        relative_path: expected_relative_path.to_string(),
        baseline_revision: run.baseline_revision.clone(),
    };
    Ok((binding, overlay))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};
    use symthaea_algorithm_evidence_collector::{
        collect_experiment_capsule, ArtifactGroupSpec, CollectorPolicy, CommandSpec,
    };
    use symthaea_algorithms::discovery::{DiscoveryPolicy, SearchBudget};
    use symthaea_algorithms::{
        AlgorithmProvenance, AlgorithmRecord, DeterminismRequirement, DiscoveryRisk,
        ImplementationRecord, ProblemSpec, SemanticGuarantee,
    };
    use symthaea_forge::certificate::{
        full_source_artifact_id, ForgeCertificate, GateEvidence, MutationRecord,
    };
    use symthaea_forge::{proposal_from_forge, ForgeCandidate};

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
        assert!(
            output.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout)
            .unwrap()
            .trim()
            .to_string()
    }

    fn init_repo(root: &Path, baseline_source: &str) -> String {
        fs::write(root.join("src/target.rs"), baseline_source).unwrap();
        git(root, &["init", "-q"]);
        git(
            root,
            &["config", "user.email", "forge-overlay-test@example.invalid"],
        );
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

    fn context_inputs() -> (ContentId, ContentId, ContentId) {
        (
            ContentId::derive("evaluator", [b"criterion".as_slice()]),
            ContentId::derive("oracle", [b"reference".as_slice()]),
            ContentId::derive("inputs", [b"fixed".as_slice()]),
        )
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
        let (evaluator, oracle, inputs) = context_inputs();
        let context_a = overlay_a
            .evaluation_context_from_live_overlay(
                &root,
                &capsule_a,
                evaluator.clone(),
                oracle.clone(),
                inputs.clone(),
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
            .evaluation_context_from_live_overlay(
                &root,
                &capsule_b,
                evaluator,
                oracle,
                inputs,
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
        init_repo(&root, baseline_source);
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
    fn context_revalidation_rejects_post_observation_mutation() {
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
        let capsule = collect_dirty_capsule(&root);
        let proposal = proposal_from_forge(&run, &algorithm, &baseline, &candidate).unwrap();
        let (_, overlay) = observe_declared_forge_overlay(
            &root,
            &run,
            &proposal,
            &candidate,
            &capsule,
            "forge-candidate",
            "src/target.rs",
        )
        .unwrap();

        fs::write(root.join("src/target.rs"), "fn target() -> i32 { 99 }\n").unwrap();
        let (evaluator, oracle, inputs) = context_inputs();
        assert!(matches!(
            overlay.evaluation_context_from_live_overlay(
                &root,
                &capsule,
                evaluator,
                oracle,
                inputs,
                0,
                vec![1],
            ),
            Err(OverlayError::CandidateBytesMismatch)
        ));

        fs::write(root.join("src/target.rs"), baseline_source).unwrap();
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
