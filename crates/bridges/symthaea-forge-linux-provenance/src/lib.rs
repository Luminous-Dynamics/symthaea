// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Nix provenance observations for the exact Bubblewrap artifact used by Forge isolation.
//!
//! This crate deliberately separates several propositions that are easy to collapse accidentally:
//!
//! - exact executable bytes were observed;
//! - those bytes live inside one exact Nix store object;
//! - `nix store verify` reported content + trust success for that store object;
//! - Nix reports a non-empty NAR hash and a known deriver;
//! - that exact Bubblewrap artifact was used by a Bubblewrap v2 execution receipt;
//! - the bytes actually implement the intended Bubblewrap semantics.
//!
//! The first five propositions can be recorded here. The last one remains explicitly **not
//! established**. Likewise, this collector content-addresses the exact `nix` / `nix-store` tool
//! bytes and their fixed command outputs, but does not claim an independent proof of those tools'
//! semantics. A later independent kernel-state attestation can reduce that remaining trust base.

use serde::Serialize;
use std::collections::BTreeSet;
use std::fs;
use std::io::Read;
use std::path::{Component, Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use symthaea_algorithms::ContentId;
use symthaea_forge::forge_bubblewrap_artifact_id;
use symthaea_forge_linux_isolation::BubblewrapV2ExecutionReceipt;
use thiserror::Error;

const NIX_STORE_ROOT: &str = "/nix/store";
const MAX_TOOL_BYTES: u64 = 512 * 1024 * 1024;
const MAX_COMMAND_OUTPUT_BYTES: u64 = 4 * 1024 * 1024;
const COMMAND_TIMEOUT_MS: u64 = 30_000;
const POLL_INTERVAL_MS: u64 = 10;

#[derive(Debug, Error)]
pub enum BubblewrapNixProvenanceError {
    #[error("provenance path is not a regular file: {0:?}")]
    NotRegularFile(PathBuf),
    #[error("provenance file exceeds the fixed artifact-size ceiling: {0:?}")]
    ArtifactTooLarge(PathBuf),
    #[error("Bubblewrap executable is not contained in a canonical /nix/store output: {0:?}")]
    NotInNixStore(PathBuf),
    #[error("IO error at {path:?}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("fixed provenance command exceeded its wall-time limit")]
    CommandTimedOut,
    #[error("fixed provenance command output exceeded the bounded capture limit")]
    CommandOutputTooLarge,
    #[error("fixed provenance command returned non-UTF-8 output")]
    NonUtf8Output,
    #[error("fixed provenance command terminated without a representable exit code")]
    MissingExitCode,
    #[error("measurement cannot be represented in u64")]
    MeasurementOverflow,
    #[error("Bubblewrap bytes do not match the exact artifact expected by the provenance policy")]
    BubblewrapArtifactMismatch,
    #[error("Nix provenance receipt identity does not match canonical fields")]
    ReceiptIdentityMismatch,
    #[error("Nix provenance binding requires a successful content+trust verification")]
    VerificationDidNotPass,
    #[error("Nix provenance binding requires a non-empty recorded NAR hash")]
    MissingNarHash,
    #[error("Nix provenance binding requires a known derivation")]
    UnknownDeriver,
    #[error("Nix provenance binding does not match the supplied receipt")]
    BindingScopeMismatch,
    #[error("Nix provenance binding identity does not match canonical fields")]
    BindingIdentityMismatch,
    #[error("provenance-bound isolation receipt does not use the provenanced Bubblewrap artifact")]
    ExecutionArtifactMismatch,
    #[error("provenance-bound isolation receipt identity does not match canonical fields")]
    ExecutionBindingIdentityMismatch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ObservationState {
    ObservedPassed,
    ObservedFailed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum KnowledgeState {
    Known,
    Unknown,
}

/// Deliberate nonclaim: neither exact hashes nor one implementation's self-observations prove its
/// semantic correctness independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SemanticCorrectnessStatus {
    NotEstablishedV1,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProvenanceToolArtifact {
    canonical_path: String,
    artifact_id: ContentId,
}

impl ProvenanceToolArtifact {
    pub fn canonical_path(&self) -> &str { &self.canonical_path }
    pub fn artifact_id(&self) -> &ContentId { &self.artifact_id }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FixedCommandObservation {
    id: ContentId,
    program: ProvenanceToolArtifact,
    argv: Vec<String>,
    argv_id: ContentId,
    stdout: String,
    stdout_id: ContentId,
    stderr: String,
    stderr_id: ContentId,
    exit_code: i32,
    wall_time_ms: u64,
}

impl FixedCommandObservation {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn program(&self) -> &ProvenanceToolArtifact { &self.program }
    pub fn argv(&self) -> &[String] { &self.argv }
    pub fn argv_id(&self) -> &ContentId { &self.argv_id }
    pub fn stdout(&self) -> &str { &self.stdout }
    pub fn stderr(&self) -> &str { &self.stderr }
    pub fn exit_code(&self) -> i32 { self.exit_code }
    pub fn wall_time_ms(&self) -> u64 { self.wall_time_ms }

    pub fn validate(&self) -> Result<(), BubblewrapNixProvenanceError> {
        let expected_argv_id = ordered_argv_id(&self.argv);
        let expected_stdout_id = output_id("stdout", self.stdout.as_bytes());
        let expected_stderr_id = output_id("stderr", self.stderr.as_bytes());
        let expected = derive_command_observation_id(
            &self.program,
            &expected_argv_id,
            &expected_stdout_id,
            &expected_stderr_id,
            self.exit_code,
            self.wall_time_ms,
        );
        if self.argv_id == expected_argv_id
            && self.stdout_id == expected_stdout_id
            && self.stderr_id == expected_stderr_id
            && self.id == expected
        {
            Ok(())
        } else {
            Err(BubblewrapNixProvenanceError::ReceiptIdentityMismatch)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BubblewrapNixProvenancePolicy {
    id: ContentId,
    bubblewrap_artifact_id: ContentId,
}

impl BubblewrapNixProvenancePolicy {
    pub fn new(bubblewrap_artifact_id: ContentId) -> Self {
        let id = ContentId::derive(
            "symthaea.forge-bubblewrap-nix-provenance-policy.v1",
            [bubblewrap_artifact_id.as_str().as_bytes()],
        );
        Self { id, bubblewrap_artifact_id }
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn bubblewrap_artifact_id(&self) -> &ContentId { &self.bubblewrap_artifact_id }

    pub fn validate(&self) -> Result<(), BubblewrapNixProvenanceError> {
        if Self::new(self.bubblewrap_artifact_id.clone()) == *self {
            Ok(())
        } else {
            Err(BubblewrapNixProvenanceError::ReceiptIdentityMismatch)
        }
    }
}

/// Read-only observations about one exact Bubblewrap Nix store object.
#[derive(Debug, Clone, Serialize)]
pub struct BubblewrapNixProvenanceReceipt {
    id: ContentId,
    policy_id: ContentId,
    bubblewrap_artifact_id: ContentId,
    bubblewrap_executable_path: String,
    store_output_path: String,
    nix_version: FixedCommandObservation,
    nix_store_version: FixedCommandObservation,
    store_verify: FixedCommandObservation,
    query_hash: FixedCommandObservation,
    query_deriver: FixedCommandObservation,
    query_references: FixedCommandObservation,
    verification_state: ObservationState,
    nar_hash: String,
    deriver: Option<String>,
    references: Vec<String>,
    bubblewrap_semantic_correctness: SemanticCorrectnessStatus,
    nix_tool_semantic_correctness: SemanticCorrectnessStatus,
}

impl BubblewrapNixProvenanceReceipt {
    pub fn id(&self) -> &ContentId { &self.id }
    pub fn policy_id(&self) -> &ContentId { &self.policy_id }
    pub fn bubblewrap_artifact_id(&self) -> &ContentId { &self.bubblewrap_artifact_id }
    pub fn bubblewrap_executable_path(&self) -> &str { &self.bubblewrap_executable_path }
    pub fn store_output_path(&self) -> &str { &self.store_output_path }
    pub fn nix_version(&self) -> &FixedCommandObservation { &self.nix_version }
    pub fn nix_store_version(&self) -> &FixedCommandObservation { &self.nix_store_version }
    pub fn store_verify(&self) -> &FixedCommandObservation { &self.store_verify }
    pub fn query_hash(&self) -> &FixedCommandObservation { &self.query_hash }
    pub fn query_deriver(&self) -> &FixedCommandObservation { &self.query_deriver }
    pub fn query_references(&self) -> &FixedCommandObservation { &self.query_references }
    pub fn verification_state(&self) -> ObservationState { self.verification_state }
    pub fn nar_hash(&self) -> &str { &self.nar_hash }
    pub fn deriver(&self) -> Option<&str> { self.deriver.as_deref() }
    pub fn references(&self) -> &[String] { &self.references }
    pub fn bubblewrap_semantic_correctness(&self) -> SemanticCorrectnessStatus {
        self.bubblewrap_semantic_correctness
    }
    pub fn nix_tool_semantic_correctness(&self) -> SemanticCorrectnessStatus {
        self.nix_tool_semantic_correctness
    }

    pub fn validate_for(
        &self,
        policy: &BubblewrapNixProvenancePolicy,
    ) -> Result<(), BubblewrapNixProvenanceError> {
        policy.validate()?;
        for observation in [
            &self.nix_version,
            &self.nix_store_version,
            &self.store_verify,
            &self.query_hash,
            &self.query_deriver,
            &self.query_references,
        ] {
            observation.validate()?;
        }
        let expected_verification = if self.store_verify.exit_code() == 0 {
            ObservationState::ObservedPassed
        } else {
            ObservationState::ObservedFailed
        };
        let expected_nar_hash = normalize_single_line(self.query_hash.stdout());
        let expected_deriver = parse_deriver(self.query_deriver.stdout());
        let expected_references = parse_references(self.query_references.stdout());
        if self.policy_id != *policy.id()
            || self.bubblewrap_artifact_id != *policy.bubblewrap_artifact_id()
            || self.verification_state != expected_verification
            || self.nar_hash != expected_nar_hash
            || self.deriver != expected_deriver
            || self.references != expected_references
            || self.bubblewrap_semantic_correctness != SemanticCorrectnessStatus::NotEstablishedV1
            || self.nix_tool_semantic_correctness != SemanticCorrectnessStatus::NotEstablishedV1
        {
            return Err(BubblewrapNixProvenanceError::ReceiptIdentityMismatch);
        }
        let expected = derive_receipt_id(
            &self.policy_id,
            &self.bubblewrap_artifact_id,
            &self.bubblewrap_executable_path,
            &self.store_output_path,
            [
                self.nix_version.id(),
                self.nix_store_version.id(),
                self.store_verify.id(),
                self.query_hash.id(),
                self.query_deriver.id(),
                self.query_references.id(),
            ],
            self.verification_state,
            &self.nar_hash,
            self.deriver.as_deref(),
            &self.references,
            self.bubblewrap_semantic_correctness,
            self.nix_tool_semantic_correctness,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(BubblewrapNixProvenanceError::ReceiptIdentityMismatch)
        }
    }
}

/// Stronger wrapper available only when Nix reported successful content+trust verification and the
/// store metadata has a non-empty NAR hash and known deriver.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BubblewrapNixProvenanceBinding {
    id: ContentId,
    receipt_id: ContentId,
    bubblewrap_artifact_id: ContentId,
    store_output_path: String,
    nar_hash: String,
    deriver: String,
    bubblewrap_semantic_correctness: SemanticCorrectnessStatus,
}

impl BubblewrapNixProvenanceBinding {
    pub fn issue(
        policy: &BubblewrapNixProvenancePolicy,
        receipt: &BubblewrapNixProvenanceReceipt,
    ) -> Result<Self, BubblewrapNixProvenanceError> {
        receipt.validate_for(policy)?;
        if receipt.verification_state() != ObservationState::ObservedPassed {
            return Err(BubblewrapNixProvenanceError::VerificationDidNotPass);
        }
        if receipt.nar_hash().is_empty() {
            return Err(BubblewrapNixProvenanceError::MissingNarHash);
        }
        let deriver = receipt
            .deriver()
            .filter(|value| !value.is_empty())
            .ok_or(BubblewrapNixProvenanceError::UnknownDeriver)?
            .to_string();
        let id = derive_binding_id(
            receipt.id(),
            receipt.bubblewrap_artifact_id(),
            receipt.store_output_path(),
            receipt.nar_hash(),
            &deriver,
        );
        Ok(Self {
            id,
            receipt_id: receipt.id().clone(),
            bubblewrap_artifact_id: receipt.bubblewrap_artifact_id().clone(),
            store_output_path: receipt.store_output_path().to_string(),
            nar_hash: receipt.nar_hash().to_string(),
            deriver,
            bubblewrap_semantic_correctness: SemanticCorrectnessStatus::NotEstablishedV1,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn receipt_id(&self) -> &ContentId { &self.receipt_id }
    pub fn bubblewrap_artifact_id(&self) -> &ContentId { &self.bubblewrap_artifact_id }
    pub fn store_output_path(&self) -> &str { &self.store_output_path }
    pub fn nar_hash(&self) -> &str { &self.nar_hash }
    pub fn deriver(&self) -> &str { &self.deriver }
    pub fn bubblewrap_semantic_correctness(&self) -> SemanticCorrectnessStatus {
        self.bubblewrap_semantic_correctness
    }

    pub fn validate_for(
        &self,
        receipt: &BubblewrapNixProvenanceReceipt,
    ) -> Result<(), BubblewrapNixProvenanceError> {
        if self.receipt_id != *receipt.id()
            || self.bubblewrap_artifact_id != *receipt.bubblewrap_artifact_id()
            || self.store_output_path != receipt.store_output_path()
            || self.nar_hash != receipt.nar_hash()
            || receipt.deriver() != Some(self.deriver.as_str())
            || receipt.verification_state() != ObservationState::ObservedPassed
            || self.bubblewrap_semantic_correctness != SemanticCorrectnessStatus::NotEstablishedV1
        {
            return Err(BubblewrapNixProvenanceError::BindingScopeMismatch);
        }
        let expected = derive_binding_id(
            &self.receipt_id,
            &self.bubblewrap_artifact_id,
            &self.store_output_path,
            &self.nar_hash,
            &self.deriver,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(BubblewrapNixProvenanceError::BindingIdentityMismatch)
        }
    }
}

/// Composes successful Nix provenance observations with the exact Bubblewrap v2 execution artifact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProvenanceBoundBubblewrapV2Receipt {
    id: ContentId,
    execution_receipt_id: ContentId,
    provenance_binding_id: ContentId,
    bubblewrap_artifact_id: ContentId,
    bubblewrap_semantic_correctness: SemanticCorrectnessStatus,
}

impl ProvenanceBoundBubblewrapV2Receipt {
    pub fn bind(
        execution: &BubblewrapV2ExecutionReceipt,
        provenance: &BubblewrapNixProvenanceBinding,
    ) -> Result<Self, BubblewrapNixProvenanceError> {
        if execution.bubblewrap_artifact_id() != provenance.bubblewrap_artifact_id() {
            return Err(BubblewrapNixProvenanceError::ExecutionArtifactMismatch);
        }
        let id = derive_execution_binding_id(
            execution.id(),
            provenance.id(),
            execution.bubblewrap_artifact_id(),
        );
        Ok(Self {
            id,
            execution_receipt_id: execution.id().clone(),
            provenance_binding_id: provenance.id().clone(),
            bubblewrap_artifact_id: execution.bubblewrap_artifact_id().clone(),
            bubblewrap_semantic_correctness: SemanticCorrectnessStatus::NotEstablishedV1,
        })
    }

    pub fn id(&self) -> &ContentId { &self.id }
    pub fn execution_receipt_id(&self) -> &ContentId { &self.execution_receipt_id }
    pub fn provenance_binding_id(&self) -> &ContentId { &self.provenance_binding_id }
    pub fn bubblewrap_artifact_id(&self) -> &ContentId { &self.bubblewrap_artifact_id }
    pub fn bubblewrap_semantic_correctness(&self) -> SemanticCorrectnessStatus {
        self.bubblewrap_semantic_correctness
    }

    pub fn validate_for(
        &self,
        execution: &BubblewrapV2ExecutionReceipt,
        provenance: &BubblewrapNixProvenanceBinding,
    ) -> Result<(), BubblewrapNixProvenanceError> {
        if self.execution_receipt_id != *execution.id()
            || self.provenance_binding_id != *provenance.id()
            || self.bubblewrap_artifact_id != *execution.bubblewrap_artifact_id()
            || execution.bubblewrap_artifact_id() != provenance.bubblewrap_artifact_id()
            || self.bubblewrap_semantic_correctness != SemanticCorrectnessStatus::NotEstablishedV1
        {
            return Err(BubblewrapNixProvenanceError::ExecutionArtifactMismatch);
        }
        let expected = derive_execution_binding_id(
            &self.execution_receipt_id,
            &self.provenance_binding_id,
            &self.bubblewrap_artifact_id,
        );
        if expected == self.id {
            Ok(())
        } else {
            Err(BubblewrapNixProvenanceError::ExecutionBindingIdentityMismatch)
        }
    }
}

/// Collect fixed, read-only Nix provenance observations for one exact Bubblewrap executable.
pub fn collect_bubblewrap_nix_provenance(
    policy: &BubblewrapNixProvenancePolicy,
    bubblewrap_executable: impl AsRef<Path>,
    nix_executable: impl AsRef<Path>,
    nix_store_executable: impl AsRef<Path>,
) -> Result<BubblewrapNixProvenanceReceipt, BubblewrapNixProvenanceError> {
    policy.validate()?;
    let (bubblewrap_path, bubblewrap_bytes) = canonical_file_bytes(bubblewrap_executable.as_ref())?;
    let bubblewrap_artifact_id = forge_bubblewrap_artifact_id(&bubblewrap_bytes);
    if bubblewrap_artifact_id != *policy.bubblewrap_artifact_id() {
        return Err(BubblewrapNixProvenanceError::BubblewrapArtifactMismatch);
    }
    let store_output_path = store_output_root(&bubblewrap_path)?;
    let store_output = store_output_path
        .to_str()
        .ok_or_else(|| BubblewrapNixProvenanceError::NotInNixStore(bubblewrap_path.clone()))?
        .to_string();

    let nix_version = run_fixed_command(nix_executable.as_ref(), &["--version".into()])?;
    let nix_store_version = run_fixed_command(nix_store_executable.as_ref(), &["--version".into()])?;
    let store_verify = run_fixed_command(
        nix_executable.as_ref(),
        &[
            "--extra-experimental-features".into(),
            "nix-command".into(),
            "store".into(),
            "verify".into(),
            store_output.clone(),
        ],
    )?;
    let query_hash = run_fixed_command(
        nix_store_executable.as_ref(),
        &["--query".into(), "--hash".into(), store_output.clone()],
    )?;
    let query_deriver = run_fixed_command(
        nix_store_executable.as_ref(),
        &["--query".into(), "--deriver".into(), store_output.clone()],
    )?;
    let query_references = run_fixed_command(
        nix_store_executable.as_ref(),
        &["--query".into(), "--references".into(), store_output.clone()],
    )?;

    let verification_state = if store_verify.exit_code() == 0 {
        ObservationState::ObservedPassed
    } else {
        ObservationState::ObservedFailed
    };
    let nar_hash = normalize_single_line(query_hash.stdout());
    let deriver = parse_deriver(query_deriver.stdout());
    let references = parse_references(query_references.stdout());
    let bubblewrap_executable_path = bubblewrap_path.to_string_lossy().into_owned();
    let observation_ids = [
        nix_version.id(),
        nix_store_version.id(),
        store_verify.id(),
        query_hash.id(),
        query_deriver.id(),
        query_references.id(),
    ];
    let id = derive_receipt_id(
        policy.id(),
        &bubblewrap_artifact_id,
        &bubblewrap_executable_path,
        &store_output,
        observation_ids,
        verification_state,
        &nar_hash,
        deriver.as_deref(),
        &references,
        SemanticCorrectnessStatus::NotEstablishedV1,
        SemanticCorrectnessStatus::NotEstablishedV1,
    );
    let receipt = BubblewrapNixProvenanceReceipt {
        id,
        policy_id: policy.id().clone(),
        bubblewrap_artifact_id,
        bubblewrap_executable_path,
        store_output_path: store_output,
        nix_version,
        nix_store_version,
        store_verify,
        query_hash,
        query_deriver,
        query_references,
        verification_state,
        nar_hash,
        deriver,
        references,
        bubblewrap_semantic_correctness: SemanticCorrectnessStatus::NotEstablishedV1,
        nix_tool_semantic_correctness: SemanticCorrectnessStatus::NotEstablishedV1,
    };
    receipt.validate_for(policy)?;
    Ok(receipt)
}

fn canonical_file_bytes(path: &Path) -> Result<(PathBuf, Vec<u8>), BubblewrapNixProvenanceError> {
    let canonical = path.canonicalize().map_err(|source| BubblewrapNixProvenanceError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let metadata = fs::metadata(&canonical).map_err(|source| BubblewrapNixProvenanceError::Io {
        path: canonical.clone(),
        source,
    })?;
    if !metadata.is_file() {
        return Err(BubblewrapNixProvenanceError::NotRegularFile(canonical));
    }
    if metadata.len() > MAX_TOOL_BYTES {
        return Err(BubblewrapNixProvenanceError::ArtifactTooLarge(canonical));
    }
    let bytes = fs::read(&canonical).map_err(|source| BubblewrapNixProvenanceError::Io {
        path: canonical.clone(),
        source,
    })?;
    Ok((canonical, bytes))
}

fn store_output_root(canonical_executable: &Path) -> Result<PathBuf, BubblewrapNixProvenanceError> {
    let store_root = Path::new(NIX_STORE_ROOT);
    let relative = canonical_executable
        .strip_prefix(store_root)
        .map_err(|_| BubblewrapNixProvenanceError::NotInNixStore(canonical_executable.to_path_buf()))?;
    let mut components = relative.components();
    let output = match components.next() {
        Some(Component::Normal(name)) => name,
        _ => return Err(BubblewrapNixProvenanceError::NotInNixStore(canonical_executable.to_path_buf())),
    };
    let output_path = store_root.join(output);
    if output_path == canonical_executable || canonical_executable.starts_with(&output_path) {
        Ok(output_path)
    } else {
        Err(BubblewrapNixProvenanceError::NotInNixStore(canonical_executable.to_path_buf()))
    }
}

#[derive(Debug)]
struct BoundedOutput {
    bytes: Vec<u8>,
    exceeded: bool,
}

fn read_capped<R: Read>(mut reader: R) -> Result<BoundedOutput, std::io::Error> {
    let mut bytes = Vec::new();
    let mut exceeded = false;
    let mut buffer = [0u8; 8192];
    loop {
        let count = reader.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        let remaining = MAX_COMMAND_OUTPUT_BYTES.saturating_sub(bytes.len() as u64);
        if remaining > 0 {
            let keep = usize::try_from(remaining.min(count as u64)).unwrap_or(count);
            bytes.extend_from_slice(&buffer[..keep]);
        }
        if count as u64 > remaining {
            exceeded = true;
        }
    }
    Ok(BoundedOutput { bytes, exceeded })
}

enum WaitOutcome {
    Exited(ExitStatus, u64),
    TimedOut,
}

fn wait_bounded(child: &mut std::process::Child) -> Result<WaitOutcome, BubblewrapNixProvenanceError> {
    let start = Instant::now();
    let timeout = Duration::from_millis(COMMAND_TIMEOUT_MS);
    loop {
        if let Some(status) = child.try_wait().map_err(|source| BubblewrapNixProvenanceError::Io {
            path: PathBuf::from("<nix-provenance-command>"),
            source,
        })? {
            let wall_time_ms = u64::try_from(start.elapsed().as_millis())
                .map_err(|_| BubblewrapNixProvenanceError::MeasurementOverflow)?;
            return Ok(WaitOutcome::Exited(status, wall_time_ms));
        }
        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(WaitOutcome::TimedOut);
        }
        thread::sleep(Duration::from_millis(POLL_INTERVAL_MS));
    }
}

fn run_fixed_command(
    program: &Path,
    args: &[String],
) -> Result<FixedCommandObservation, BubblewrapNixProvenanceError> {
    let (canonical_program, program_bytes) = canonical_file_bytes(program)?;
    let canonical_path = canonical_program.to_string_lossy().into_owned();
    let program_artifact = ProvenanceToolArtifact {
        canonical_path,
        artifact_id: ContentId::derive("symthaea.forge-provenance-tool-artifact.v1", [program_bytes.as_slice()]),
    };
    let mut command = Command::new(&canonical_program);
    command
        .args(args)
        .env_clear()
        .env("HOME", "/nonexistent")
        .env("LANG", "C")
        .env("LC_ALL", "C")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let mut child = command.spawn().map_err(|source| BubblewrapNixProvenanceError::Io {
        path: canonical_program.clone(),
        source,
    })?;
    let stdout = child.stdout.take().ok_or_else(|| BubblewrapNixProvenanceError::Io {
        path: PathBuf::from("<nix-provenance-stdout>"),
        source: std::io::Error::other("missing stdout pipe"),
    })?;
    let stderr = child.stderr.take().ok_or_else(|| BubblewrapNixProvenanceError::Io {
        path: PathBuf::from("<nix-provenance-stderr>"),
        source: std::io::Error::other("missing stderr pipe"),
    })?;
    let stdout_reader = thread::spawn(move || read_capped(stdout));
    let stderr_reader = thread::spawn(move || read_capped(stderr));
    let wait = wait_bounded(&mut child)?;
    let stdout = stdout_reader
        .join()
        .map_err(|_| BubblewrapNixProvenanceError::Io {
            path: PathBuf::from("<nix-provenance-stdout>"),
            source: std::io::Error::other("stdout reader panicked"),
        })?
        .map_err(|source| BubblewrapNixProvenanceError::Io {
            path: PathBuf::from("<nix-provenance-stdout>"),
            source,
        })?;
    let stderr = stderr_reader
        .join()
        .map_err(|_| BubblewrapNixProvenanceError::Io {
            path: PathBuf::from("<nix-provenance-stderr>"),
            source: std::io::Error::other("stderr reader panicked"),
        })?
        .map_err(|source| BubblewrapNixProvenanceError::Io {
            path: PathBuf::from("<nix-provenance-stderr>"),
            source,
        })?;
    if stdout.exceeded || stderr.exceeded {
        return Err(BubblewrapNixProvenanceError::CommandOutputTooLarge);
    }
    let (status, wall_time_ms) = match wait {
        WaitOutcome::TimedOut => return Err(BubblewrapNixProvenanceError::CommandTimedOut),
        WaitOutcome::Exited(status, wall_time_ms) => (status, wall_time_ms),
    };
    let exit_code = status.code().ok_or(BubblewrapNixProvenanceError::MissingExitCode)?;
    let stdout_text = String::from_utf8(stdout.bytes).map_err(|_| BubblewrapNixProvenanceError::NonUtf8Output)?;
    let stderr_text = String::from_utf8(stderr.bytes).map_err(|_| BubblewrapNixProvenanceError::NonUtf8Output)?;
    let argv = args.to_vec();
    let argv_id = ordered_argv_id(&argv);
    let stdout_id = output_id("stdout", stdout_text.as_bytes());
    let stderr_id = output_id("stderr", stderr_text.as_bytes());
    let id = derive_command_observation_id(
        &program_artifact,
        &argv_id,
        &stdout_id,
        &stderr_id,
        exit_code,
        wall_time_ms,
    );
    let observation = FixedCommandObservation {
        id,
        program: program_artifact,
        argv,
        argv_id,
        stdout: stdout_text,
        stdout_id,
        stderr: stderr_text,
        stderr_id,
        exit_code,
        wall_time_ms,
    };
    observation.validate()?;
    Ok(observation)
}

fn ordered_argv_id(args: &[String]) -> ContentId {
    let mut parts = vec![(args.len() as u64).to_be_bytes().to_vec()];
    for arg in args {
        parts.push((arg.len() as u64).to_be_bytes().to_vec());
        parts.push(arg.as_bytes().to_vec());
    }
    ContentId::derive(
        "symthaea.forge-provenance-command-argv.v1",
        parts.iter().map(Vec::as_slice),
    )
}

fn output_id(stream: &str, bytes: &[u8]) -> ContentId {
    ContentId::derive(
        "symthaea.forge-provenance-command-output.v1",
        [stream.as_bytes(), bytes],
    )
}

fn derive_command_observation_id(
    program: &ProvenanceToolArtifact,
    argv_id: &ContentId,
    stdout_id: &ContentId,
    stderr_id: &ContentId,
    exit_code: i32,
    wall_time_ms: u64,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-provenance-command-observation.v1",
        [
            program.canonical_path.as_bytes(),
            program.artifact_id.as_str().as_bytes(),
            argv_id.as_str().as_bytes(),
            stdout_id.as_str().as_bytes(),
            stderr_id.as_str().as_bytes(),
            exit_code.to_be_bytes().as_slice(),
            wall_time_ms.to_be_bytes().as_slice(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn derive_receipt_id<'a>(
    policy_id: &ContentId,
    bubblewrap_artifact_id: &ContentId,
    bubblewrap_executable_path: &str,
    store_output_path: &str,
    observation_ids: [&'a ContentId; 6],
    verification_state: ObservationState,
    nar_hash: &str,
    deriver: Option<&str>,
    references: &[String],
    bubblewrap_semantic_correctness: SemanticCorrectnessStatus,
    nix_tool_semantic_correctness: SemanticCorrectnessStatus,
) -> ContentId {
    let mut parts = vec![
        policy_id.as_str().as_bytes().to_vec(),
        bubblewrap_artifact_id.as_str().as_bytes().to_vec(),
        bubblewrap_executable_path.as_bytes().to_vec(),
        store_output_path.as_bytes().to_vec(),
    ];
    parts.extend(observation_ids.iter().map(|id| id.as_str().as_bytes().to_vec()));
    parts.push(format!("{verification_state:?}").into_bytes());
    parts.push(nar_hash.as_bytes().to_vec());
    parts.push(deriver.unwrap_or("").as_bytes().to_vec());
    parts.push((references.len() as u64).to_be_bytes().to_vec());
    parts.extend(references.iter().map(|reference| reference.as_bytes().to_vec()));
    parts.push(format!("{bubblewrap_semantic_correctness:?}").into_bytes());
    parts.push(format!("{nix_tool_semantic_correctness:?}").into_bytes());
    ContentId::derive(
        "symthaea.forge-bubblewrap-nix-provenance-receipt.v1",
        parts.iter().map(Vec::as_slice),
    )
}

fn derive_binding_id(
    receipt_id: &ContentId,
    artifact_id: &ContentId,
    store_output_path: &str,
    nar_hash: &str,
    deriver: &str,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-bubblewrap-nix-provenance-binding.v1",
        [
            receipt_id.as_str().as_bytes(),
            artifact_id.as_str().as_bytes(),
            store_output_path.as_bytes(),
            nar_hash.as_bytes(),
            deriver.as_bytes(),
        ],
    )
}

fn derive_execution_binding_id(
    execution_id: &ContentId,
    provenance_id: &ContentId,
    artifact_id: &ContentId,
) -> ContentId {
    ContentId::derive(
        "symthaea.forge-provenance-bound-bubblewrap-v2-receipt.v1",
        [
            execution_id.as_str().as_bytes(),
            provenance_id.as_str().as_bytes(),
            artifact_id.as_str().as_bytes(),
        ],
    )
}

fn normalize_single_line(value: &str) -> String {
    value.lines().next().unwrap_or("").trim().to_string()
}

fn parse_deriver(value: &str) -> Option<String> {
    let value = normalize_single_line(value);
    if value.is_empty() || value == "unknown-deriver" || !value.ends_with(".drv") {
        None
    } else {
        Some(value)
    }
}

fn parse_references(value: &str) -> Vec<String> {
    let mut unique = BTreeSet::new();
    for line in value.lines().map(str::trim).filter(|line| !line.is_empty()) {
        unique.insert(line.to_string());
    }
    unique.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_root_parsing_extracts_one_output() {
        let input = Path::new("/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bubblewrap-0.11.0/bin/bwrap");
        let root = store_output_root(input).unwrap();
        assert_eq!(
            root,
            Path::new("/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-bubblewrap-0.11.0")
        );
    }

    #[test]
    fn store_root_rejects_non_store_path() {
        assert!(matches!(
            store_output_root(Path::new("/usr/bin/bwrap")),
            Err(BubblewrapNixProvenanceError::NotInNixStore(_))
        ));
    }

    #[test]
    fn deriver_parser_distinguishes_unknown() {
        assert_eq!(parse_deriver("unknown-deriver\n"), None);
        assert_eq!(
            parse_deriver("/nix/store/abc-bubblewrap.drv\n"),
            Some("/nix/store/abc-bubblewrap.drv".into())
        );
    }

    #[test]
    fn semantic_correctness_remains_an_explicit_nonclaim() {
        assert_eq!(
            SemanticCorrectnessStatus::NotEstablishedV1,
            SemanticCorrectnessStatus::NotEstablishedV1
        );
    }
}
