// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Read-only provenance collection for algorithm-discovery experiments.
//!
//! The collector is deliberately narrower than an executor: it may read explicitly named files
//! and invoke a fixed set of read-only metadata commands (`git`, `rustc`, `cargo`, `uname`, and
//! optionally `nix`). It has no API for running candidate code, mutating source, committing,
//! pushing, merging, or promoting an implementation.
//!
//! A capsule separates candidate/source artifact identity from comparison-environment identity.
//! This is important: two different candidate implementations must remain comparable when they
//! are evaluated under the same machine/toolchain/lock/command context. Candidate bytes belong in
//! `ImplementationRecord`; they must not accidentally make every Pareto evaluation context unique.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::process::Command;
use symthaea_algorithms::evaluation::{EvaluationContext, EvaluationError};
use symthaea_algorithms::ContentId;
use thiserror::Error;

const ENV_ALLOWLIST: &[&str] = &[
    "RUSTFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_BUILD_TARGET",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
];

#[derive(Debug, Error)]
pub enum CollectorError {
    #[error("repository root does not exist or is not a directory: {0}")]
    InvalidRepositoryRoot(String),
    #[error("path must be a canonical repository-relative file path: {0}")]
    InvalidRelativePath(String),
    #[error("artifact path escapes the canonical repository root: {0}")]
    ArtifactEscapesRepository(String),
    #[error("artifact group `{0}` must contain at least one file")]
    EmptyArtifactGroup(String),
    #[error("duplicate artifact group label: {0}")]
    DuplicateArtifactGroup(String),
    #[error("duplicate artifact file path in group `{group}`: {path}")]
    DuplicateArtifactPath { group: String, path: String },
    #[error("artifact path is not a regular file: {0}")]
    NotAFile(String),
    #[error("I/O error while {operation} `{path}`: {detail}")]
    Io {
        operation: &'static str,
        path: String,
        detail: String,
    },
    #[error("metadata command `{program}` failed: {detail}")]
    CommandFailed { program: &'static str, detail: String },
    #[error("metadata command `{program}` returned non-UTF-8 output")]
    NonUtf8CommandOutput { program: &'static str },
    #[error("provided root is not the repository top-level directory")]
    NotRepositoryTopLevel,
    #[error("worktree is not clean; default experiment collection fails closed")]
    DirtyWorktree,
    #[error("dirty-worktree capsules are audit evidence only and cannot mint evaluation contexts")]
    DirtyWorktreeIneligibleForEvaluation,
    #[error("experiment capsule must precommit at least one evaluation command")]
    NoCommands,
    #[error("command specification requires a non-empty program")]
    EmptyCommandProgram,
    #[error("command specification contains control characters")]
    InvalidCommandText,
    #[error("{field} must not be empty")]
    EmptyText { field: &'static str },
    #[error("{field} contains a NUL character")]
    NulText { field: &'static str },
    #[error("command index {0} is out of range")]
    CommandIndexOutOfRange(usize),
    #[error("capsule contains noncanonical fields")]
    NonCanonicalCapsule,
    #[error(transparent)]
    Evaluation(#[from] EvaluationError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectorPolicy {
    pub require_clean_worktree: bool,
}

impl Default for CollectorPolicy {
    fn default() -> Self {
        Self {
            require_clean_worktree: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactGroupSpec {
    pub label: String,
    pub relative_files: Vec<String>,
}

impl ArtifactGroupSpec {
    pub fn new(
        label: impl Into<String>,
        relative_files: Vec<String>,
    ) -> Result<Self, CollectorError> {
        let label = label.into();
        validate_single_line("artifact group label", &label)?;
        if relative_files.is_empty() {
            return Err(CollectorError::EmptyArtifactGroup(label));
        }
        for path in &relative_files {
            validate_relative_path(path)?;
        }
        Ok(Self {
            label,
            relative_files,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactFileEvidence {
    pub relative_path: String,
    pub byte_len: u64,
    pub content_id: ContentId,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArtifactGroupEvidence {
    pub id: ContentId,
    pub label: String,
    pub files: Vec<ArtifactFileEvidence>,
}

impl ArtifactGroupEvidence {
    pub fn validate(&self) -> Result<(), CollectorError> {
        validate_single_line("artifact group label", &self.label)?;
        if self.files.is_empty() {
            return Err(CollectorError::EmptyArtifactGroup(self.label.clone()));
        }
        let mut previous: Option<&str> = None;
        for file in &self.files {
            validate_relative_path(&file.relative_path)?;
            if previous.is_some_and(|prev| prev >= file.relative_path.as_str()) {
                return Err(CollectorError::NonCanonicalCapsule);
            }
            previous = Some(&file.relative_path);
        }
        let expected = derive_artifact_group_id(&self.label, &self.files);
        if expected == self.id {
            Ok(())
        } else {
            Err(CollectorError::NonCanonicalCapsule)
        }
    }
}

/// Exact command specification retained as provenance. The collector records this value but does
/// not execute it; execution remains the responsibility of the qualification/benchmark harness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandSpec {
    pub id: ContentId,
    pub program: String,
    pub args: Vec<String>,
}

impl CommandSpec {
    pub fn new(program: impl Into<String>, args: Vec<String>) -> Result<Self, CollectorError> {
        let program = program.into();
        if program.trim().is_empty() {
            return Err(CollectorError::EmptyCommandProgram);
        }
        if program.chars().any(char::is_control)
            || args.iter().any(|arg| arg.chars().any(char::is_control))
        {
            return Err(CollectorError::InvalidCommandText);
        }
        let mut parts = vec![program.as_bytes().to_vec()];
        parts.extend(args.iter().map(|arg| arg.as_bytes().to_vec()));
        let id = ContentId::derive(
            "symthaea.algorithm-command.v1",
            parts.iter().map(Vec::as_slice),
        );
        Ok(Self { id, program, args })
    }

    pub fn validate(&self) -> Result<(), CollectorError> {
        let rebuilt = Self::new(self.program.clone(), self.args.clone())?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(CollectorError::NonCanonicalCapsule)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RepositoryState {
    pub id: ContentId,
    pub revision: String,
    pub worktree_clean: bool,
    pub worktree_status_id: ContentId,
    pub cargo_lock_id: Option<ContentId>,
    pub flake_lock_id: Option<ContentId>,
    pub rust_toolchain_id: Option<ContentId>,
}

impl RepositoryState {
    pub fn new(
        revision: impl Into<String>,
        worktree_clean: bool,
        worktree_status_id: ContentId,
        cargo_lock_id: Option<ContentId>,
        flake_lock_id: Option<ContentId>,
        rust_toolchain_id: Option<ContentId>,
    ) -> Result<Self, CollectorError> {
        let revision = revision.into();
        validate_single_line("repository revision", &revision)?;
        let id = derive_repository_state_id(
            &revision,
            worktree_clean,
            &worktree_status_id,
            cargo_lock_id.as_ref(),
            flake_lock_id.as_ref(),
            rust_toolchain_id.as_ref(),
        );
        Ok(Self {
            id,
            revision,
            worktree_clean,
            worktree_status_id,
            cargo_lock_id,
            flake_lock_id,
            rust_toolchain_id,
        })
    }

    pub fn validate(&self) -> Result<(), CollectorError> {
        let rebuilt = Self::new(
            self.revision.clone(),
            self.worktree_clean,
            self.worktree_status_id.clone(),
            self.cargo_lock_id.clone(),
            self.flake_lock_id.clone(),
            self.rust_toolchain_id.clone(),
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(CollectorError::NonCanonicalCapsule)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineProfile {
    pub id: ContentId,
    pub operating_system: String,
    pub architecture: String,
    pub kernel: String,
    pub cpu_model: Option<String>,
    pub cpu_features: Vec<String>,
    pub host_triple: String,
    pub target_features: Vec<String>,
}

impl MachineProfile {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        operating_system: impl Into<String>,
        architecture: impl Into<String>,
        kernel: impl Into<String>,
        cpu_model: Option<String>,
        mut cpu_features: Vec<String>,
        host_triple: impl Into<String>,
        mut target_features: Vec<String>,
    ) -> Result<Self, CollectorError> {
        let operating_system = operating_system.into();
        let architecture = architecture.into();
        let kernel = kernel.into();
        let host_triple = host_triple.into();
        validate_single_line("operating system", &operating_system)?;
        validate_single_line("architecture", &architecture)?;
        validate_single_line("kernel", &kernel)?;
        validate_single_line("host triple", &host_triple)?;
        if let Some(cpu) = &cpu_model {
            validate_single_line("CPU model", cpu)?;
        }
        for feature in &cpu_features {
            validate_single_line("CPU feature", feature)?;
        }
        for feature in &target_features {
            validate_single_line("target feature", feature)?;
        }
        cpu_features.sort();
        cpu_features.dedup();
        target_features.sort();
        target_features.dedup();
        let id = derive_machine_id(
            &operating_system,
            &architecture,
            &kernel,
            cpu_model.as_deref(),
            &cpu_features,
            &host_triple,
            &target_features,
        );
        Ok(Self {
            id,
            operating_system,
            architecture,
            kernel,
            cpu_model,
            cpu_features,
            host_triple,
            target_features,
        })
    }

    pub fn validate(&self) -> Result<(), CollectorError> {
        let rebuilt = Self::new(
            self.operating_system.clone(),
            self.architecture.clone(),
            self.kernel.clone(),
            self.cpu_model.clone(),
            self.cpu_features.clone(),
            self.host_triple.clone(),
            self.target_features.clone(),
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(CollectorError::NonCanonicalCapsule)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ToolchainProfile {
    pub id: ContentId,
    pub rustc_verbose_version: String,
    pub cargo_version: String,
    pub nix_version: Option<String>,
}

impl ToolchainProfile {
    pub fn new(
        rustc_verbose_version: impl Into<String>,
        cargo_version: impl Into<String>,
        nix_version: Option<String>,
    ) -> Result<Self, CollectorError> {
        let rustc_verbose_version = rustc_verbose_version.into();
        let cargo_version = cargo_version.into();
        // `rustc -vV` is intentionally multiline. Preserve the complete output instead of
        // flattening away provenance; only NUL is forbidden.
        validate_multiline("rustc verbose version", &rustc_verbose_version)?;
        validate_single_line("cargo version", &cargo_version)?;
        if let Some(nix) = &nix_version {
            validate_single_line("nix version", nix)?;
        }
        let id = ContentId::derive(
            "symthaea.algorithm-toolchain.v1",
            [
                rustc_verbose_version.as_bytes(),
                cargo_version.as_bytes(),
                nix_version.as_deref().unwrap_or("").as_bytes(),
            ],
        );
        Ok(Self {
            id,
            rustc_verbose_version,
            cargo_version,
            nix_version,
        })
    }

    pub fn validate(&self) -> Result<(), CollectorError> {
        let rebuilt = Self::new(
            self.rustc_verbose_version.clone(),
            self.cargo_version.clone(),
            self.nix_version.clone(),
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(CollectorError::NonCanonicalCapsule)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CapturedEnvironment {
    pub id: ContentId,
    pub variables: BTreeMap<String, String>,
}

impl CapturedEnvironment {
    pub fn new(variables: BTreeMap<String, String>) -> Self {
        let mut parts = Vec::with_capacity(variables.len() * 2);
        for (key, value) in &variables {
            parts.push(key.as_bytes().to_vec());
            parts.push(value.as_bytes().to_vec());
        }
        let id = ContentId::derive(
            "symthaea.algorithm-environment-vars.v1",
            parts.iter().map(Vec::as_slice),
        );
        Self { id, variables }
    }

    pub fn validate(&self) -> Result<(), CollectorError> {
        if self
            .variables
            .keys()
            .any(|key| !ENV_ALLOWLIST.contains(&key.as_str()))
        {
            return Err(CollectorError::NonCanonicalCapsule);
        }
        if Self::new(self.variables.clone()) == *self {
            Ok(())
        } else {
            Err(CollectorError::NonCanonicalCapsule)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperimentCapsule {
    pub id: ContentId,
    pub repository: RepositoryState,
    pub machine: MachineProfile,
    pub toolchain: ToolchainProfile,
    pub environment: CapturedEnvironment,
    pub artifact_groups: Vec<ArtifactGroupEvidence>,
    pub commands: Vec<CommandSpec>,
}

impl ExperimentCapsule {
    pub fn new(
        repository: RepositoryState,
        machine: MachineProfile,
        toolchain: ToolchainProfile,
        environment: CapturedEnvironment,
        mut artifact_groups: Vec<ArtifactGroupEvidence>,
        commands: Vec<CommandSpec>,
    ) -> Result<Self, CollectorError> {
        repository.validate()?;
        machine.validate()?;
        toolchain.validate()?;
        environment.validate()?;
        for group in &artifact_groups {
            group.validate()?;
        }
        if commands.is_empty() {
            return Err(CollectorError::NoCommands);
        }
        for command in &commands {
            command.validate()?;
        }
        artifact_groups.sort_by(|a, b| a.label.cmp(&b.label));
        if let Some(pair) = artifact_groups
            .windows(2)
            .find(|pair| pair[0].label == pair[1].label)
        {
            return Err(CollectorError::DuplicateArtifactGroup(
                pair[0].label.clone(),
            ));
        }
        let id = derive_capsule_id(
            &repository,
            &machine,
            &toolchain,
            &environment,
            &artifact_groups,
            &commands,
        );
        Ok(Self {
            id,
            repository,
            machine,
            toolchain,
            environment,
            artifact_groups,
            commands,
        })
    }

    pub fn validate(&self) -> Result<(), CollectorError> {
        let rebuilt = Self::new(
            self.repository.clone(),
            self.machine.clone(),
            self.toolchain.clone(),
            self.environment.clone(),
            self.artifact_groups.clone(),
            self.commands.clone(),
        )?;
        if rebuilt == *self {
            Ok(())
        } else {
            Err(CollectorError::NonCanonicalCapsule)
        }
    }

    pub fn artifact_group_id(&self, label: &str) -> Option<&ContentId> {
        self.artifact_groups
            .iter()
            .find(|group| group.label == label)
            .map(|group| &group.id)
    }

    /// Identity of the comparison environment only. Candidate artifact groups and worktree status
    /// are deliberately excluded: candidate bytes are bound by `ImplementationRecord` and should
    /// not make otherwise comparable evaluations appear to use different hardware/toolchains.
    pub fn comparison_environment_id(&self) -> ContentId {
        ContentId::derive(
            "symthaea.algorithm-comparison-environment.v1",
            [
                self.machine.id.as_str().as_bytes(),
                self.toolchain.id.as_str().as_bytes(),
                self.environment.id.as_str().as_bytes(),
                self.repository
                    .cargo_lock_id
                    .as_ref()
                    .map(ContentId::as_str)
                    .unwrap_or("")
                    .as_bytes(),
                self.repository
                    .flake_lock_id
                    .as_ref()
                    .map(ContentId::as_str)
                    .unwrap_or("")
                    .as_bytes(),
                self.repository
                    .rust_toolchain_id
                    .as_ref()
                    .map(ContentId::as_str)
                    .unwrap_or("")
                    .as_bytes(),
            ],
        )
    }

    /// Bind an evaluator implementation to one exact recorded invocation without folding the
    /// candidate artifact into the comparison environment.
    pub fn evaluation_context(
        &self,
        base_evaluator_id: ContentId,
        oracle_id: ContentId,
        input_profile_id: ContentId,
        command_index: usize,
        seeds: Vec<u64>,
    ) -> Result<EvaluationContext, CollectorError> {
        self.validate()?;
        if !self.repository.worktree_clean {
            return Err(CollectorError::DirtyWorktreeIneligibleForEvaluation);
        }
        let command = self
            .commands
            .get(command_index)
            .ok_or(CollectorError::CommandIndexOutOfRange(command_index))?;
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
            self.comparison_environment_id(),
            self.repository.revision.clone(),
            self.toolchain.id.to_string(),
            self.machine.host_triple.clone(),
            seeds,
        )?)
    }
}

/// Collect one capsule from the current repository/environment without executing candidate code.
pub fn collect_experiment_capsule(
    repository_root: &Path,
    artifact_specs: &[ArtifactGroupSpec],
    commands: Vec<CommandSpec>,
    policy: CollectorPolicy,
) -> Result<ExperimentCapsule, CollectorError> {
    let root = repository_root
        .canonicalize()
        .map_err(|error| CollectorError::InvalidRepositoryRoot(error.to_string()))?;
    if !root.is_dir() {
        return Err(CollectorError::InvalidRepositoryRoot(
            root.display().to_string(),
        ));
    }

    let reported_root = PathBuf::from(run_required(&root, "git", &["rev-parse", "--show-toplevel"])?);
    let reported_root = reported_root
        .canonicalize()
        .map_err(|error| CollectorError::InvalidRepositoryRoot(error.to_string()))?;
    if reported_root != root {
        return Err(CollectorError::NotRepositoryTopLevel);
    }

    let revision = run_required(&root, "git", &["rev-parse", "--verify", "HEAD"])?;
    let status = run_required(
        &root,
        "git",
        &["status", "--porcelain=v1", "--untracked-files=all"],
    )?;
    let worktree_clean = status.is_empty();
    if policy.require_clean_worktree && !worktree_clean {
        return Err(CollectorError::DirtyWorktree);
    }
    let repository = RepositoryState::new(
        revision,
        worktree_clean,
        ContentId::derive("symthaea.git-worktree-status.v1", [status.as_bytes()]),
        content_id_if_exists(&root, "Cargo.lock", "symthaea.cargo-lock.v1")?,
        content_id_if_exists(&root, "flake.lock", "symthaea.flake-lock.v1")?,
        content_id_if_exists(
            &root,
            "rust-toolchain.toml",
            "symthaea.rust-toolchain-file.v1",
        )?,
    )?;

    let rustc_verbose = run_required(&root, "rustc", &["-vV"])?;
    let host_triple = rustc_verbose
        .lines()
        .find_map(|line| line.strip_prefix("host: "))
        .unwrap_or(std::env::consts::ARCH)
        .to_string();
    let cfg = run_required(&root, "rustc", &["--print", "cfg"])?;
    let target_features = cfg
        .lines()
        .filter(|line| line.starts_with("target_feature="))
        .map(str::to_string)
        .collect();
    let kernel = run_optional(&root, "uname", &["-srvm"])
        .unwrap_or_else(|| "unavailable".to_string());
    let (cpu_model, cpu_features) = detect_cpu_profile();
    let machine = MachineProfile::new(
        std::env::consts::OS,
        std::env::consts::ARCH,
        kernel,
        cpu_model,
        cpu_features,
        host_triple,
        target_features,
    )?;
    let toolchain = ToolchainProfile::new(
        rustc_verbose,
        run_required(&root, "cargo", &["--version"])?,
        run_optional(&root, "nix", &["--version"]),
    )?;

    let variables = ENV_ALLOWLIST
        .iter()
        .filter_map(|name| std::env::var(name).ok().map(|value| ((*name).to_string(), value)))
        .collect();
    let environment = CapturedEnvironment::new(variables);

    let mut labels = BTreeSet::new();
    let mut artifact_groups = Vec::with_capacity(artifact_specs.len());
    for spec in artifact_specs {
        if !labels.insert(spec.label.clone()) {
            return Err(CollectorError::DuplicateArtifactGroup(spec.label.clone()));
        }
        artifact_groups.push(collect_artifact_group(&root, spec)?);
    }

    ExperimentCapsule::new(
        repository,
        machine,
        toolchain,
        environment,
        artifact_groups,
        commands,
    )
}

pub fn collect_artifact_group(
    repository_root: &Path,
    spec: &ArtifactGroupSpec,
) -> Result<ArtifactGroupEvidence, CollectorError> {
    validate_single_line("artifact group label", &spec.label)?;
    if spec.relative_files.is_empty() {
        return Err(CollectorError::EmptyArtifactGroup(spec.label.clone()));
    }
    let canonical_root = repository_root
        .canonicalize()
        .map_err(|error| CollectorError::InvalidRepositoryRoot(error.to_string()))?;
    let mut seen = BTreeSet::new();
    let mut files = Vec::with_capacity(spec.relative_files.len());
    for relative in &spec.relative_files {
        validate_relative_path(relative)?;
        if !seen.insert(relative.clone()) {
            return Err(CollectorError::DuplicateArtifactPath {
                group: spec.label.clone(),
                path: relative.clone(),
            });
        }
        let requested = canonical_root.join(relative);
        let full = requested.canonicalize().map_err(|error| CollectorError::Io {
            operation: "canonicalizing artifact file",
            path: relative.clone(),
            detail: error.to_string(),
        })?;
        if !full.starts_with(&canonical_root) {
            return Err(CollectorError::ArtifactEscapesRepository(relative.clone()));
        }
        if !full.is_file() {
            return Err(CollectorError::NotAFile(relative.clone()));
        }
        let bytes = fs::read(&full).map_err(|error| CollectorError::Io {
            operation: "reading artifact file",
            path: relative.clone(),
            detail: error.to_string(),
        })?;
        files.push(ArtifactFileEvidence {
            relative_path: relative.clone(),
            byte_len: bytes.len() as u64,
            content_id: ContentId::derive(
                "symthaea.algorithm-source-file.v1",
                [relative.as_bytes(), bytes.as_slice()],
            ),
        });
    }
    files.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    let id = derive_artifact_group_id(&spec.label, &files);
    let evidence = ArtifactGroupEvidence {
        id,
        label: spec.label.clone(),
        files,
    };
    evidence.validate()?;
    Ok(evidence)
}

fn derive_artifact_group_id(label: &str, files: &[ArtifactFileEvidence]) -> ContentId {
    let mut parts = vec![label.as_bytes().to_vec(), (files.len() as u64).to_be_bytes().to_vec()];
    for file in files {
        parts.push(file.relative_path.as_bytes().to_vec());
        parts.push(file.byte_len.to_be_bytes().to_vec());
        parts.push(file.content_id.as_str().as_bytes().to_vec());
    }
    ContentId::derive(
        "symthaea.algorithm-artifact-group.v1",
        parts.iter().map(Vec::as_slice),
    )
}

fn derive_repository_state_id(
    revision: &str,
    worktree_clean: bool,
    worktree_status_id: &ContentId,
    cargo_lock_id: Option<&ContentId>,
    flake_lock_id: Option<&ContentId>,
    rust_toolchain_id: Option<&ContentId>,
) -> ContentId {
    ContentId::derive(
        "symthaea.algorithm-repository-state.v1",
        [
            revision.as_bytes(),
            if worktree_clean { b"clean" } else { b"dirty" },
            worktree_status_id.as_str().as_bytes(),
            cargo_lock_id.map(ContentId::as_str).unwrap_or("").as_bytes(),
            flake_lock_id.map(ContentId::as_str).unwrap_or("").as_bytes(),
            rust_toolchain_id
                .map(ContentId::as_str)
                .unwrap_or("")
                .as_bytes(),
        ],
    )
}

fn derive_machine_id(
    operating_system: &str,
    architecture: &str,
    kernel: &str,
    cpu_model: Option<&str>,
    cpu_features: &[String],
    host_triple: &str,
    target_features: &[String],
) -> ContentId {
    let mut parts = vec![
        operating_system.as_bytes().to_vec(),
        architecture.as_bytes().to_vec(),
        kernel.as_bytes().to_vec(),
        cpu_model.unwrap_or("").as_bytes().to_vec(),
        (cpu_features.len() as u64).to_be_bytes().to_vec(),
    ];
    parts.extend(cpu_features.iter().map(|feature| feature.as_bytes().to_vec()));
    parts.push(host_triple.as_bytes().to_vec());
    parts.push((target_features.len() as u64).to_be_bytes().to_vec());
    parts.extend(target_features.iter().map(|feature| feature.as_bytes().to_vec()));
    ContentId::derive(
        "symthaea.algorithm-machine-profile.v2",
        parts.iter().map(Vec::as_slice),
    )
}

fn derive_capsule_id(
    repository: &RepositoryState,
    machine: &MachineProfile,
    toolchain: &ToolchainProfile,
    environment: &CapturedEnvironment,
    artifact_groups: &[ArtifactGroupEvidence],
    commands: &[CommandSpec],
) -> ContentId {
    let mut parts = vec![
        repository.id.as_str().as_bytes().to_vec(),
        machine.id.as_str().as_bytes().to_vec(),
        toolchain.id.as_str().as_bytes().to_vec(),
        environment.id.as_str().as_bytes().to_vec(),
        (artifact_groups.len() as u64).to_be_bytes().to_vec(),
    ];
    for group in artifact_groups {
        parts.push(group.label.as_bytes().to_vec());
        parts.push(group.id.as_str().as_bytes().to_vec());
    }
    parts.push((commands.len() as u64).to_be_bytes().to_vec());
    for command in commands {
        parts.push(command.id.as_str().as_bytes().to_vec());
    }
    ContentId::derive(
        "symthaea.algorithm-experiment-capsule.v1",
        parts.iter().map(Vec::as_slice),
    )
}

fn content_id_if_exists(
    root: &Path,
    relative: &str,
    domain: &str,
) -> Result<Option<ContentId>, CollectorError> {
    let path = root.join(relative);
    if !path.exists() {
        return Ok(None);
    }
    let full = path.canonicalize().map_err(|error| CollectorError::Io {
        operation: "canonicalizing provenance file",
        path: relative.to_string(),
        detail: error.to_string(),
    })?;
    if !full.starts_with(root) {
        return Err(CollectorError::ArtifactEscapesRepository(relative.to_string()));
    }
    let bytes = fs::read(&full).map_err(|error| CollectorError::Io {
        operation: "reading provenance file",
        path: relative.to_string(),
        detail: error.to_string(),
    })?;
    Ok(Some(ContentId::derive(domain, [bytes.as_slice()])))
}

fn validate_single_line(field: &'static str, value: &str) -> Result<(), CollectorError> {
    if value.trim().is_empty() {
        return Err(CollectorError::EmptyText { field });
    }
    if value.chars().any(char::is_control) {
        return Err(CollectorError::NulText { field });
    }
    Ok(())
}

fn validate_multiline(field: &'static str, value: &str) -> Result<(), CollectorError> {
    if value.trim().is_empty() {
        return Err(CollectorError::EmptyText { field });
    }
    if value.contains('\0') {
        return Err(CollectorError::NulText { field });
    }
    Ok(())
}

fn validate_relative_path(value: &str) -> Result<(), CollectorError> {
    if value.trim().is_empty() || value.chars().any(char::is_control) {
        return Err(CollectorError::InvalidRelativePath(value.to_string()));
    }
    let path = Path::new(value);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(CollectorError::InvalidRelativePath(value.to_string()));
    }
    Ok(())
}

fn run_required(
    root: &Path,
    program: &'static str,
    args: &[&str],
) -> Result<String, CollectorError> {
    let output = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()
        .map_err(|error| CollectorError::CommandFailed {
            program,
            detail: error.to_string(),
        })?;
    if !output.status.success() {
        return Err(CollectorError::CommandFailed {
            program,
            detail: String::from_utf8_lossy(&output.stderr).trim().to_string(),
        });
    }
    let stdout = String::from_utf8(output.stdout)
        .map_err(|_| CollectorError::NonUtf8CommandOutput { program })?;
    Ok(stdout.trim().to_string())
}

fn run_optional(root: &Path, program: &'static str, args: &[&str]) -> Option<String> {
    run_required(root, program, args).ok()
}

fn detect_cpu_profile() -> (Option<String>, Vec<String>) {
    if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
        let model = cpuinfo.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            matches!(key.trim(), "model name" | "Processor")
                .then(|| value.trim().to_string())
                .filter(|value| !value.is_empty())
        });
        let mut features: Vec<String> = cpuinfo
            .lines()
            .find_map(|line| {
                let (key, value) = line.split_once(':')?;
                matches!(key.trim(), "flags" | "Features").then_some(value)
            })
            .map(|value| value.split_whitespace().map(str::to_string).collect())
            .unwrap_or_default();
        features.sort();
        features.dedup();
        return (model, features);
    }

    let model = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let mut features = Vec::new();
    for key in ["machdep.cpu.features", "machdep.cpu.leaf7_features"] {
        if let Ok(output) = Command::new("sysctl").args(["-n", key]).output() {
            if output.status.success() {
                if let Ok(value) = String::from_utf8(output.stdout) {
                    features.extend(value.split_whitespace().map(|feature| feature.to_lowercase()));
                }
            }
        }
    }
    features.sort();
    features.dedup();
    (model, features)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_TEMP: AtomicU64 = AtomicU64::new(1);

    fn temp_dir() -> PathBuf {
        let suffix = NEXT_TEMP.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!(
            "symthaea-algorithm-collector-test-{}-{suffix}",
            std::process::id()
        ));
        fs::create_dir_all(&path).unwrap();
        path
    }

    fn cid(domain: &str, value: &str) -> ContentId {
        ContentId::derive(domain, [value.as_bytes()])
    }

    fn repository(clean: bool) -> RepositoryState {
        RepositoryState::new(
            "deadbeef",
            clean,
            cid("status", if clean { "clean" } else { "dirty" }),
            Some(cid("lock", "cargo")),
            Some(cid("lock", "flake")),
            Some(cid("toolchain-file", "rust")),
        )
        .unwrap()
    }

    fn machine() -> MachineProfile {
        MachineProfile::new(
            "linux",
            "x86_64",
            "test-kernel",
            Some("test-cpu".into()),
            vec!["avx2".into(), "popcnt".into()],
            "x86_64-unknown-linux-gnu",
            vec!["target_feature=\"avx2\"".into()],
        )
        .unwrap()
    }

    fn toolchain() -> ToolchainProfile {
        ToolchainProfile::new(
            "rustc 1.96 test\nbinary: rustc\nhost: x86_64-unknown-linux-gnu",
            "cargo 1.96 test",
            Some("nix test".into()),
        )
        .unwrap()
    }

    fn command() -> CommandSpec {
        CommandSpec::new(
            "cargo",
            vec!["bench".into(), "-p".into(), "symthaea-algorithm-lab".into()],
        )
        .unwrap()
    }

    fn capsule(clean: bool, artifact_groups: Vec<ArtifactGroupEvidence>) -> ExperimentCapsule {
        ExperimentCapsule::new(
            repository(clean),
            machine(),
            toolchain(),
            CapturedEnvironment::new(BTreeMap::new()),
            artifact_groups,
            vec![command()],
        )
        .unwrap()
    }

    #[test]
    fn multiline_rustc_verbose_output_is_valid_provenance() {
        assert!(toolchain().validate().is_ok());
    }

    #[test]
    fn artifact_group_identity_changes_with_exact_file_bytes() {
        let root = temp_dir();
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/candidate.rs"), b"first").unwrap();
        let spec = ArtifactGroupSpec::new("candidate", vec!["src/candidate.rs".into()]).unwrap();
        let first = collect_artifact_group(&root, &spec).unwrap();
        fs::write(root.join("src/candidate.rs"), b"second").unwrap();
        let second = collect_artifact_group(&root, &spec).unwrap();
        assert_ne!(first.id, second.id);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn artifact_file_order_is_canonical() {
        let root = temp_dir();
        fs::write(root.join("a.rs"), b"a").unwrap();
        fs::write(root.join("b.rs"), b"b").unwrap();
        let a = collect_artifact_group(
            &root,
            &ArtifactGroupSpec::new("candidate", vec!["b.rs".into(), "a.rs".into()]).unwrap(),
        )
        .unwrap();
        let b = collect_artifact_group(
            &root,
            &ArtifactGroupSpec::new("candidate", vec!["a.rs".into(), "b.rs".into()]).unwrap(),
        )
        .unwrap();
        assert_eq!(a.id, b.id);
        assert_eq!(a.files, b.files);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn command_identity_binds_argument_order() {
        let a = CommandSpec::new("cargo", vec!["bench".into(), "--no-run".into()]).unwrap();
        let b = CommandSpec::new("cargo", vec!["--no-run".into(), "bench".into()]).unwrap();
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn no_command_capsule_fails_closed() {
        assert!(matches!(
            ExperimentCapsule::new(
                repository(true),
                machine(),
                toolchain(),
                CapturedEnvironment::new(BTreeMap::new()),
                vec![],
                vec![],
            )
            .unwrap_err(),
            CollectorError::NoCommands
        ));
    }

    #[test]
    fn dirty_capsule_is_audit_only_not_rankable() {
        let dirty = capsule(false, vec![]);
        assert_eq!(
            dirty
                .evaluation_context(
                    cid("evaluator", "criterion"),
                    cid("oracle", "reference"),
                    cid("inputs", "seeded"),
                    0,
                    vec![1, 2, 3],
                )
                .unwrap_err()
                .to_string(),
            CollectorError::DirtyWorktreeIneligibleForEvaluation.to_string()
        );
    }

    #[test]
    fn candidate_artifact_changes_capsule_but_not_comparison_environment() {
        let root = temp_dir();
        fs::write(root.join("candidate.rs"), b"a").unwrap();
        let spec = ArtifactGroupSpec::new("candidate", vec!["candidate.rs".into()]).unwrap();
        let valid_a = collect_artifact_group(&root, &spec).unwrap();
        fs::write(root.join("candidate.rs"), b"b").unwrap();
        let valid_b = collect_artifact_group(&root, &spec).unwrap();

        let capsule_a = capsule(true, vec![valid_a]);
        let capsule_b = capsule(true, vec![valid_b]);
        assert_ne!(capsule_a.id, capsule_b.id);
        assert_eq!(
            capsule_a.comparison_environment_id(),
            capsule_b.comparison_environment_id()
        );

        let context_a = capsule_a
            .evaluation_context(
                cid("evaluator", "criterion"),
                cid("oracle", "reference"),
                cid("inputs", "seeded"),
                0,
                vec![1, 2, 3],
            )
            .unwrap();
        let context_b = capsule_b
            .evaluation_context(
                cid("evaluator", "criterion"),
                cid("oracle", "reference"),
                cid("inputs", "seeded"),
                0,
                vec![1, 2, 3],
            )
            .unwrap();
        assert_eq!(context_a.content_id(), context_b.content_id());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn cpu_features_are_canonicalized_into_machine_identity() {
        let a = MachineProfile::new(
            "linux",
            "x86_64",
            "kernel",
            Some("cpu".into()),
            vec!["popcnt".into(), "avx2".into(), "popcnt".into()],
            "x86_64-unknown-linux-gnu",
            vec![],
        )
        .unwrap();
        let b = MachineProfile::new(
            "linux",
            "x86_64",
            "kernel",
            Some("cpu".into()),
            vec!["avx2".into(), "popcnt".into()],
            "x86_64-unknown-linux-gnu",
            vec![],
        )
        .unwrap();
        assert_eq!(a.id, b.id);
        assert_eq!(a.cpu_features, vec!["avx2", "popcnt"]);
    }

    #[test]
    fn parent_directory_escape_is_rejected() {
        assert!(matches!(
            ArtifactGroupSpec::new("candidate", vec!["../secret".into()]).unwrap_err(),
            CollectorError::InvalidRelativePath(_)
        ));
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_is_rejected() {
        use std::os::unix::fs::symlink;

        let root = temp_dir();
        let outside = temp_dir();
        fs::write(outside.join("secret"), b"outside").unwrap();
        symlink(outside.join("secret"), root.join("linked-secret")).unwrap();
        let spec = ArtifactGroupSpec::new("candidate", vec!["linked-secret".into()]).unwrap();
        assert!(matches!(
            collect_artifact_group(&root, &spec).unwrap_err(),
            CollectorError::ArtifactEscapesRepository(_)
        ));
        let _ = fs::remove_dir_all(root);
        let _ = fs::remove_dir_all(outside);
    }
}
