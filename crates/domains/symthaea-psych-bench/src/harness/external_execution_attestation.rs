// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! Controller-owned process execution for external benchmark qualification.
//!
//! [`crate::external_qualification`] deliberately stops at serialized consistency:
//! a caller can report `Completed`, but that record alone does not prove an evaluator
//! process ran. This module adds the next evidence layer for one requested benchmark:
//!
//! 1. validate the qualification manifest and exact benchmark requirement;
//! 2. hash the materialized dataset, evaluator, and adapter before launch;
//! 3. expand a canonical invocation template and spawn the verified evaluator;
//! 4. require terminal success and a newly-created result artifact;
//! 5. recapture all three input assets after the process exits;
//! 6. bind result/stdout/stderr bytes and invocation identity into a receipt;
//! 7. return an opaque capability only from this controller path.
//!
//! Materialized paths must be absolute canonical paths. The evaluator runs in an
//! initially-empty working directory and may leave only the declared result file.
//! These are boundary-time controls, not a continuous OS sandbox: this module does
//! not bind the dynamic linker/interpreter closure, prevent arbitrary reads outside
//! the working directory, or continuously mediate filesystem activity. A serialized
//! receipt can be independently checked for byte consistency, but deserializing it
//! does not recreate the opaque execution capability.

use crate::external_qualification::{
    ExternalBenchmarkExecutionClaim, ExternalBenchmarkRequirement, ExternalQualificationError,
    ExternalQualificationManifest, ExternalReportedDisposition,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const EXTERNAL_EXECUTION_ATTESTATION_SCHEMA_VERSION: u32 = 1;
const INVOCATION_DOMAIN: &[u8] = b"symthaea-external-invocation-v1\0";
const ATTESTATION_DOMAIN: &[u8] = b"symthaea-external-execution-attestation-v1\0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum ExternalInvocationArg {
    Literal(String),
    DatasetPath,
    AdapterPath,
    ResultPath,
}

/// Portable invocation identity. Materialized absolute paths are deliberately not
/// identity-bearing; placeholders are expanded by the controller after their bytes
/// have been verified against the manifest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalInvocationSpec {
    pub benchmark_id: String,
    pub args: Vec<ExternalInvocationArg>,
    pub environment: BTreeMap<String, String>,
}

/// Runtime-only filesystem locations. These paths are not serializable authority.
#[derive(Debug, Clone)]
pub struct ExternalExecutionPaths {
    pub evaluator: PathBuf,
    pub dataset: PathBuf,
    pub adapter: PathBuf,
    pub result: PathBuf,
    pub working_dir: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExternalProcessExecutionReceipt {
    pub schema_version: u32,
    pub manifest_digest: String,
    pub code_subject: String,
    pub benchmark_id: String,
    pub result_schema_id: String,
    pub invocation_digest: String,
    pub dataset_digest: String,
    pub evaluator_digest: String,
    pub adapter_digest: String,
    pub exit_code: i32,
    pub result_digest: String,
    pub stdout_digest: String,
    pub stderr_digest: String,
    pub attestation_digest: String,
}

/// Non-serializable authority returned only after this module launched and checked
/// the evaluator process. The contained receipt remains a consistency artifact.
#[derive(Debug)]
pub struct VerifiedExternalProcessExecution {
    receipt: ExternalProcessExecutionReceipt,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

impl VerifiedExternalProcessExecution {
    pub fn receipt(&self) -> &ExternalProcessExecutionReceipt {
        &self.receipt
    }

    pub fn stdout(&self) -> &[u8] {
        &self.stdout
    }

    pub fn stderr(&self) -> &[u8] {
        &self.stderr
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternalExecutionAttestationError {
    Qualification(ExternalQualificationError),
    BenchmarkNotRequested(String),
    NonCanonicalIdentifier {
        field: &'static str,
        value: String,
    },
    InvocationBenchmarkMismatch,
    EmptyInvocation,
    NonAbsolutePath {
        field: &'static str,
        path: String,
    },
    NonCanonicalPath {
        field: &'static str,
        path: String,
    },
    InvalidMaterializedAsset {
        field: &'static str,
        path: String,
    },
    MaterializedAssetDigestMismatch {
        field: &'static str,
    },
    WorkingDirectoryInvalid(String),
    WorkingDirectoryNotEmpty(String),
    ResultOutsideWorkingDirectory,
    ResultAlreadyExists(String),
    ResultMissing(String),
    UnexpectedWorkingDirectoryEntry(String),
    SpawnFailed(String),
    ProcessFailed {
        exit_code: Option<i32>,
    },
    InvocationDigestMismatch,
    ManifestDigestMismatch,
    CodeSubjectMismatch,
    ResultSchemaMismatch,
    ReceiptFieldMismatch(&'static str),
    AttestationDigestMismatch,
    Io {
        operation: &'static str,
        path: String,
        error: String,
    },
}

impl From<ExternalQualificationError> for ExternalExecutionAttestationError {
    fn from(value: ExternalQualificationError) -> Self {
        Self::Qualification(value)
    }
}

impl fmt::Display for ExternalExecutionAttestationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Qualification(error) => write!(f, "external qualification contract failed: {error}"),
            Self::BenchmarkNotRequested(id) => write!(f, "benchmark is not requested by manifest: {id}"),
            Self::NonCanonicalIdentifier { field, value } => {
                write!(f, "non-canonical {field}: {value:?}")
            }
            Self::InvocationBenchmarkMismatch => {
                write!(f, "invocation benchmark ID does not match requested benchmark")
            }
            Self::EmptyInvocation => write!(f, "external evaluator invocation has no arguments"),
            Self::NonAbsolutePath { field, path } => {
                write!(f, "{field} path must be absolute: {path}")
            }
            Self::NonCanonicalPath { field, path } => {
                write!(f, "{field} path is not canonical: {path}")
            }
            Self::InvalidMaterializedAsset { field, path } => {
                write!(f, "{field} is not a regular non-symlink file: {path}")
            }
            Self::MaterializedAssetDigestMismatch { field } => {
                write!(f, "materialized {field} bytes do not match manifest digest")
            }
            Self::WorkingDirectoryInvalid(path) => {
                write!(f, "working directory is not a regular directory: {path}")
            }
            Self::WorkingDirectoryNotEmpty(path) => {
                write!(f, "working directory must start empty: {path}")
            }
            Self::ResultOutsideWorkingDirectory => {
                write!(f, "result must be a direct child of the controller working directory")
            }
            Self::ResultAlreadyExists(path) => {
                write!(f, "result path exists before evaluator launch: {path}")
            }
            Self::ResultMissing(path) => {
                write!(f, "successful evaluator did not create result artifact: {path}")
            }
            Self::UnexpectedWorkingDirectoryEntry(path) => {
                write!(f, "evaluator left undeclared working-directory entry: {path}")
            }
            Self::SpawnFailed(error) => write!(f, "failed to launch verified evaluator: {error}"),
            Self::ProcessFailed { exit_code } => {
                write!(f, "external evaluator did not terminate successfully: {exit_code:?}")
            }
            Self::InvocationDigestMismatch => write!(f, "invocation digest does not revalidate"),
            Self::ManifestDigestMismatch => write!(f, "attestation manifest digest mismatch"),
            Self::CodeSubjectMismatch => write!(f, "attestation code subject mismatch"),
            Self::ResultSchemaMismatch => write!(f, "attestation result schema mismatch"),
            Self::ReceiptFieldMismatch(field) => write!(f, "attestation field mismatch: {field}"),
            Self::AttestationDigestMismatch => write!(f, "attestation digest does not revalidate"),
            Self::Io {
                operation,
                path,
                error,
            } => write!(f, "{operation} failed for {path}: {error}"),
        }
    }
}

impl std::error::Error for ExternalExecutionAttestationError {}

impl ExternalInvocationSpec {
    pub fn validate(&self) -> Result<(), ExternalExecutionAttestationError> {
        validate_identifier("invocation benchmark_id", &self.benchmark_id)?;
        if self.args.is_empty() {
            return Err(ExternalExecutionAttestationError::EmptyInvocation);
        }
        for arg in &self.args {
            if let ExternalInvocationArg::Literal(value) = arg {
                validate_component("literal argument", value)?;
            }
        }
        for (key, value) in &self.environment {
            validate_identifier("environment key", key)?;
            if key.contains('=') {
                return Err(ExternalExecutionAttestationError::NonCanonicalIdentifier {
                    field: "environment key",
                    value: key.clone(),
                });
            }
            validate_component("environment value", value)?;
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<String, ExternalExecutionAttestationError> {
        self.validate()?;
        let mut hasher = blake3::Hasher::new();
        hasher.update(INVOCATION_DOMAIN);
        push_str(&mut hasher, &self.benchmark_id);
        hasher.update(&(self.args.len() as u64).to_le_bytes());
        for arg in &self.args {
            match arg {
                ExternalInvocationArg::Literal(value) => {
                    hasher.update(&[0]);
                    push_str(&mut hasher, value);
                }
                ExternalInvocationArg::DatasetPath => hasher.update(&[1]),
                ExternalInvocationArg::AdapterPath => hasher.update(&[2]),
                ExternalInvocationArg::ResultPath => hasher.update(&[3]),
            };
        }
        hasher.update(&(self.environment.len() as u64).to_le_bytes());
        for (key, value) in &self.environment {
            push_str(&mut hasher, key);
            push_str(&mut hasher, value);
        }
        Ok(hasher.finalize().to_hex().to_string())
    }
}

impl ExternalProcessExecutionReceipt {
    pub fn to_consistency_claim(&self) -> ExternalBenchmarkExecutionClaim {
        ExternalBenchmarkExecutionClaim {
            benchmark_id: self.benchmark_id.clone(),
            reported_disposition: ExternalReportedDisposition::Completed,
            result_schema_id: self.result_schema_id.clone(),
            result_digest: self.result_digest.clone(),
        }
    }

    /// Independently re-check serialized receipt fields against the manifest,
    /// invocation, and materialized bytes. This checks consistency; it cannot
    /// recreate the controller-owned process-execution capability.
    pub fn verify_against_materialized(
        &self,
        manifest: &ExternalQualificationManifest,
        code_subject: &str,
        invocation: &ExternalInvocationSpec,
        paths: &ExternalExecutionPaths,
        stdout: &[u8],
        stderr: &[u8],
    ) -> Result<(), ExternalExecutionAttestationError> {
        if self.schema_version != EXTERNAL_EXECUTION_ATTESTATION_SCHEMA_VERSION {
            return Err(ExternalExecutionAttestationError::ReceiptFieldMismatch(
                "schema_version",
            ));
        }
        manifest.validate()?;
        validate_identifier("code_subject", code_subject)?;
        invocation.validate()?;
        validate_path_contract(paths, false)?;
        if self.benchmark_id != invocation.benchmark_id {
            return Err(ExternalExecutionAttestationError::InvocationBenchmarkMismatch);
        }
        let requirement = requirement_for(manifest, &self.benchmark_id)?;
        if self.manifest_digest != manifest.digest()? {
            return Err(ExternalExecutionAttestationError::ManifestDigestMismatch);
        }
        if self.code_subject != code_subject {
            return Err(ExternalExecutionAttestationError::CodeSubjectMismatch);
        }
        if self.result_schema_id != requirement.result_schema_id {
            return Err(ExternalExecutionAttestationError::ResultSchemaMismatch);
        }
        if self.invocation_digest != invocation.digest()? {
            return Err(ExternalExecutionAttestationError::InvocationDigestMismatch);
        }
        if self.exit_code != 0 {
            return Err(ExternalExecutionAttestationError::ReceiptFieldMismatch("exit_code"));
        }

        let dataset = read_regular_file(&paths.dataset, "dataset")?;
        let evaluator = read_regular_file(&paths.evaluator, "evaluator")?;
        let adapter = read_regular_file(&paths.adapter, "adapter")?;
        let result = read_regular_file(&paths.result, "result")?;
        verify_asset_digest("dataset", &requirement.assets.dataset_digest, &dataset)?;
        verify_asset_digest("evaluator", &requirement.assets.evaluator_digest, &evaluator)?;
        verify_asset_digest("adapter", &requirement.assets.adapter_digest, &adapter)?;
        validate_working_dir_post(paths)?;

        check_digest_field("dataset_digest", &self.dataset_digest, &dataset)?;
        check_digest_field("evaluator_digest", &self.evaluator_digest, &evaluator)?;
        check_digest_field("adapter_digest", &self.adapter_digest, &adapter)?;
        check_digest_field("result_digest", &self.result_digest, &result)?;
        check_digest_field("stdout_digest", &self.stdout_digest, stdout)?;
        check_digest_field("stderr_digest", &self.stderr_digest, stderr)?;

        let expected = compute_attestation_digest(self);
        if self.attestation_digest != expected {
            return Err(ExternalExecutionAttestationError::AttestationDigestMismatch);
        }
        Ok(())
    }
}

/// Verify inputs, launch the evaluator, recapture inputs, and return opaque authority.
pub fn execute_external_benchmark(
    manifest: &ExternalQualificationManifest,
    code_subject: &str,
    invocation: &ExternalInvocationSpec,
    paths: &ExternalExecutionPaths,
) -> Result<VerifiedExternalProcessExecution, ExternalExecutionAttestationError> {
    manifest.validate()?;
    validate_identifier("code_subject", code_subject)?;
    invocation.validate()?;
    validate_path_contract(paths, true)?;
    let requirement = requirement_for(manifest, &invocation.benchmark_id)?;

    let dataset_before = read_regular_file(&paths.dataset, "dataset")?;
    let evaluator_before = read_regular_file(&paths.evaluator, "evaluator")?;
    let adapter_before = read_regular_file(&paths.adapter, "adapter")?;
    verify_asset_digest("dataset", &requirement.assets.dataset_digest, &dataset_before)?;
    verify_asset_digest(
        "evaluator",
        &requirement.assets.evaluator_digest,
        &evaluator_before,
    )?;
    verify_asset_digest("adapter", &requirement.assets.adapter_digest, &adapter_before)?;
    ensure_result_absent(&paths.result)?;

    let mut command = Command::new(&paths.evaluator);
    command.current_dir(&paths.working_dir);
    command.env_clear();
    for (key, value) in &invocation.environment {
        command.env(key, value);
    }
    for arg in &invocation.args {
        match arg {
            ExternalInvocationArg::Literal(value) => command.arg(value),
            ExternalInvocationArg::DatasetPath => command.arg(&paths.dataset),
            ExternalInvocationArg::AdapterPath => command.arg(&paths.adapter),
            ExternalInvocationArg::ResultPath => command.arg(&paths.result),
        };
    }

    let output = command
        .output()
        .map_err(|error| ExternalExecutionAttestationError::SpawnFailed(error.to_string()))?;
    if !output.status.success() {
        return Err(ExternalExecutionAttestationError::ProcessFailed {
            exit_code: output.status.code(),
        });
    }
    let exit_code = output
        .status
        .code()
        .ok_or(ExternalExecutionAttestationError::ProcessFailed { exit_code: None })?;
    if exit_code != 0 {
        return Err(ExternalExecutionAttestationError::ProcessFailed {
            exit_code: Some(exit_code),
        });
    }

    let result = read_regular_file(&paths.result, "result").map_err(|error| match error {
        ExternalExecutionAttestationError::InvalidMaterializedAsset { .. }
        | ExternalExecutionAttestationError::Io { .. } => {
            ExternalExecutionAttestationError::ResultMissing(display_path(&paths.result))
        }
        other => other,
    })?;
    validate_working_dir_post(paths)?;

    // Boundary-time recapture prevents a successful receipt if any committed input
    // differs after execution. This is not continuous filesystem mediation.
    let dataset_after = read_regular_file(&paths.dataset, "dataset")?;
    let evaluator_after = read_regular_file(&paths.evaluator, "evaluator")?;
    let adapter_after = read_regular_file(&paths.adapter, "adapter")?;
    verify_asset_digest("dataset", &requirement.assets.dataset_digest, &dataset_after)?;
    verify_asset_digest(
        "evaluator",
        &requirement.assets.evaluator_digest,
        &evaluator_after,
    )?;
    verify_asset_digest("adapter", &requirement.assets.adapter_digest, &adapter_after)?;
    if dataset_before != dataset_after {
        return Err(ExternalExecutionAttestationError::ReceiptFieldMismatch(
            "dataset_changed_during_execution",
        ));
    }
    if evaluator_before != evaluator_after {
        return Err(ExternalExecutionAttestationError::ReceiptFieldMismatch(
            "evaluator_changed_during_execution",
        ));
    }
    if adapter_before != adapter_after {
        return Err(ExternalExecutionAttestationError::ReceiptFieldMismatch(
            "adapter_changed_during_execution",
        ));
    }

    let mut receipt = ExternalProcessExecutionReceipt {
        schema_version: EXTERNAL_EXECUTION_ATTESTATION_SCHEMA_VERSION,
        manifest_digest: manifest.digest()?,
        code_subject: code_subject.to_owned(),
        benchmark_id: invocation.benchmark_id.clone(),
        result_schema_id: requirement.result_schema_id.clone(),
        invocation_digest: invocation.digest()?,
        dataset_digest: digest_bytes(&dataset_after),
        evaluator_digest: digest_bytes(&evaluator_after),
        adapter_digest: digest_bytes(&adapter_after),
        exit_code,
        result_digest: digest_bytes(&result),
        stdout_digest: digest_bytes(&output.stdout),
        stderr_digest: digest_bytes(&output.stderr),
        attestation_digest: String::new(),
    };
    receipt.attestation_digest = compute_attestation_digest(&receipt);
    receipt.verify_against_materialized(
        manifest,
        code_subject,
        invocation,
        paths,
        &output.stdout,
        &output.stderr,
    )?;

    Ok(VerifiedExternalProcessExecution {
        receipt,
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

fn requirement_for<'a>(
    manifest: &'a ExternalQualificationManifest,
    benchmark_id: &str,
) -> Result<&'a ExternalBenchmarkRequirement, ExternalExecutionAttestationError> {
    manifest
        .benchmarks
        .iter()
        .find(|entry| entry.benchmark_id == benchmark_id)
        .ok_or_else(|| ExternalExecutionAttestationError::BenchmarkNotRequested(benchmark_id.into()))
}

fn validate_identifier(
    field: &'static str,
    value: &str,
) -> Result<(), ExternalExecutionAttestationError> {
    if value.is_empty() || value.trim() != value || value.chars().any(char::is_control) {
        return Err(ExternalExecutionAttestationError::NonCanonicalIdentifier {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_component(
    field: &'static str,
    value: &str,
) -> Result<(), ExternalExecutionAttestationError> {
    if value.is_empty() || value.chars().any(char::is_control) {
        return Err(ExternalExecutionAttestationError::NonCanonicalIdentifier {
            field,
            value: value.to_owned(),
        });
    }
    Ok(())
}

fn validate_path_contract(
    paths: &ExternalExecutionPaths,
    require_empty_working_dir: bool,
) -> Result<(), ExternalExecutionAttestationError> {
    for (field, path) in [
        ("evaluator", &paths.evaluator),
        ("dataset", &paths.dataset),
        ("adapter", &paths.adapter),
        ("result", &paths.result),
        ("working_dir", &paths.working_dir),
    ] {
        if !path.is_absolute() {
            return Err(ExternalExecutionAttestationError::NonAbsolutePath {
                field,
                path: display_path(path),
            });
        }
    }

    ensure_canonical_existing_path(&paths.evaluator, "evaluator")?;
    ensure_canonical_existing_path(&paths.dataset, "dataset")?;
    ensure_canonical_existing_path(&paths.adapter, "adapter")?;
    ensure_canonical_existing_path(&paths.working_dir, "working_dir")?;
    validate_working_dir(&paths.working_dir)?;

    if paths.result.parent() != Some(paths.working_dir.as_path()) {
        return Err(ExternalExecutionAttestationError::ResultOutsideWorkingDirectory);
    }
    if require_empty_working_dir {
        let mut entries = fs::read_dir(&paths.working_dir).map_err(|error| {
            ExternalExecutionAttestationError::Io {
                operation: "read working directory",
                path: display_path(&paths.working_dir),
                error: error.to_string(),
            }
        })?;
        if entries.next().transpose().map_err(|error| {
            ExternalExecutionAttestationError::Io {
                operation: "read working directory entry",
                path: display_path(&paths.working_dir),
                error: error.to_string(),
            }
        })?.is_some()
        {
            return Err(ExternalExecutionAttestationError::WorkingDirectoryNotEmpty(
                display_path(&paths.working_dir),
            ));
        }
    }
    Ok(())
}

fn validate_working_dir_post(
    paths: &ExternalExecutionPaths,
) -> Result<(), ExternalExecutionAttestationError> {
    let expected_name = paths
        .result
        .file_name()
        .ok_or(ExternalExecutionAttestationError::ResultOutsideWorkingDirectory)?;
    for entry in fs::read_dir(&paths.working_dir).map_err(|error| {
        ExternalExecutionAttestationError::Io {
            operation: "read working directory",
            path: display_path(&paths.working_dir),
            error: error.to_string(),
        }
    })? {
        let entry = entry.map_err(|error| ExternalExecutionAttestationError::Io {
            operation: "read working directory entry",
            path: display_path(&paths.working_dir),
            error: error.to_string(),
        })?;
        if entry.file_name() != expected_name {
            return Err(ExternalExecutionAttestationError::UnexpectedWorkingDirectoryEntry(
                display_path(&entry.path()),
            ));
        }
    }
    Ok(())
}

fn ensure_canonical_existing_path(
    path: &Path,
    field: &'static str,
) -> Result<(), ExternalExecutionAttestationError> {
    let canonical = fs::canonicalize(path).map_err(|error| ExternalExecutionAttestationError::Io {
        operation: "canonicalize materialized path",
        path: display_path(path),
        error: error.to_string(),
    })?;
    if canonical != path {
        return Err(ExternalExecutionAttestationError::NonCanonicalPath {
            field,
            path: display_path(path),
        });
    }
    Ok(())
}

fn validate_working_dir(path: &Path) -> Result<(), ExternalExecutionAttestationError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| ExternalExecutionAttestationError::Io {
        operation: "inspect working directory",
        path: display_path(path),
        error: error.to_string(),
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(ExternalExecutionAttestationError::WorkingDirectoryInvalid(
            display_path(path),
        ));
    }
    Ok(())
}

fn ensure_result_absent(path: &Path) -> Result<(), ExternalExecutionAttestationError> {
    match fs::symlink_metadata(path) {
        Ok(_) => Err(ExternalExecutionAttestationError::ResultAlreadyExists(
            display_path(path),
        )),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(ExternalExecutionAttestationError::Io {
            operation: "inspect result path",
            path: display_path(path),
            error: error.to_string(),
        }),
    }
}

fn read_regular_file(
    path: &Path,
    field: &'static str,
) -> Result<Vec<u8>, ExternalExecutionAttestationError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| ExternalExecutionAttestationError::Io {
        operation: "inspect materialized asset",
        path: display_path(path),
        error: error.to_string(),
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(ExternalExecutionAttestationError::InvalidMaterializedAsset {
            field,
            path: display_path(path),
        });
    }
    fs::read(path).map_err(|error| ExternalExecutionAttestationError::Io {
        operation: "read materialized asset",
        path: display_path(path),
        error: error.to_string(),
    })
}

fn verify_asset_digest(
    field: &'static str,
    expected: &str,
    bytes: &[u8],
) -> Result<(), ExternalExecutionAttestationError> {
    if digest_bytes(bytes) != expected {
        return Err(ExternalExecutionAttestationError::MaterializedAssetDigestMismatch { field });
    }
    Ok(())
}

fn check_digest_field(
    field: &'static str,
    recorded: &str,
    bytes: &[u8],
) -> Result<(), ExternalExecutionAttestationError> {
    if recorded != digest_bytes(bytes) {
        return Err(ExternalExecutionAttestationError::ReceiptFieldMismatch(field));
    }
    Ok(())
}

fn digest_bytes(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

fn compute_attestation_digest(receipt: &ExternalProcessExecutionReceipt) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(ATTESTATION_DOMAIN);
    hasher.update(&receipt.schema_version.to_le_bytes());
    for value in [
        &receipt.manifest_digest,
        &receipt.code_subject,
        &receipt.benchmark_id,
        &receipt.result_schema_id,
        &receipt.invocation_digest,
        &receipt.dataset_digest,
        &receipt.evaluator_digest,
        &receipt.adapter_digest,
    ] {
        push_str(&mut hasher, value);
    }
    hasher.update(&receipt.exit_code.to_le_bytes());
    for value in [
        &receipt.result_digest,
        &receipt.stdout_digest,
        &receipt.stderr_digest,
    ] {
        push_str(&mut hasher, value);
    }
    hasher.finalize().to_hex().to_string()
}

fn push_str(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn display_path(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[cfg(all(test, unix))]
mod tests {
    use super::*;
    use crate::external_qualification::{
        ExternalAssetIdentity, ExternalQualificationConsistencyReceipt,
        EXTERNAL_QUALIFICATION_SCHEMA_VERSION,
    };
    use std::os::unix::fs::PermissionsExt;
    use std::sync::atomic::{AtomicU64, Ordering};

    static NEXT_FIXTURE: AtomicU64 = AtomicU64::new(1);

    struct Fixture {
        root: PathBuf,
        manifest: ExternalQualificationManifest,
        invocation: ExternalInvocationSpec,
        paths: ExternalExecutionPaths,
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn fixture(script_body: &str) -> Fixture {
        let id = NEXT_FIXTURE.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!(
            "symthaea-external-attestation-{}-{id}",
            std::process::id()
        ));
        let input = root.join("input");
        let working_dir = root.join("work");
        fs::create_dir_all(&input).unwrap();
        fs::create_dir_all(&working_dir).unwrap();
        let dataset = input.join("dataset.bin");
        let adapter = input.join("adapter.bin");
        let evaluator = input.join("evaluator.sh");
        let result = working_dir.join("result.json");
        fs::write(&dataset, b"dataset-v1").unwrap();
        fs::write(&adapter, b"adapter-v1").unwrap();
        fs::write(&evaluator, script_body.as_bytes()).unwrap();
        let mut permissions = fs::metadata(&evaluator).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&evaluator, permissions).unwrap();

        let manifest = ExternalQualificationManifest {
            schema_version: EXTERNAL_QUALIFICATION_SCHEMA_VERSION,
            qualification_id: "synthetic-external-campaign-v1".into(),
            benchmarks: vec![ExternalBenchmarkRequirement {
                benchmark_id: "synthetic-benchmark".into(),
                assets: ExternalAssetIdentity {
                    dataset_digest: digest_bytes(b"dataset-v1"),
                    evaluator_digest: digest_bytes(script_body.as_bytes()),
                    adapter_digest: digest_bytes(b"adapter-v1"),
                },
                result_schema_id: "synthetic-result-v1".into(),
            }],
        };
        let invocation = ExternalInvocationSpec {
            benchmark_id: "synthetic-benchmark".into(),
            args: vec![
                ExternalInvocationArg::DatasetPath,
                ExternalInvocationArg::AdapterPath,
                ExternalInvocationArg::ResultPath,
            ],
            environment: BTreeMap::new(),
        };
        let paths = ExternalExecutionPaths {
            evaluator,
            dataset,
            adapter,
            result,
            working_dir,
        };
        Fixture {
            root,
            manifest,
            invocation,
            paths,
        }
    }

    fn success_script() -> &'static str {
        "#!/bin/sh\nset -eu\ndataset=\"$1\"\nadapter=\"$2\"\nresult=\"$3\"\nprintf 'result:' > \"$result\"\n/bin/cat \"$dataset\" >> \"$result\"\nprintf ':' >> \"$result\"\n/bin/cat \"$adapter\" >> \"$result\"\nprintf 'stdout-ok'\nprintf 'stderr-ok' >&2\n"
    }

    #[test]
    fn controller_launches_verified_evaluator_and_claim_composes_with_consistency_contract() {
        let fixture = fixture(success_script());
        let verified = execute_external_benchmark(
            &fixture.manifest,
            "git:0123456789abcdef",
            &fixture.invocation,
            &fixture.paths,
        )
        .unwrap();
        let receipt = verified.receipt();
        assert_eq!(receipt.exit_code, 0);
        assert_eq!(verified.stdout(), b"stdout-ok");
        assert_eq!(verified.stderr(), b"stderr-ok");
        receipt
            .verify_against_materialized(
                &fixture.manifest,
                "git:0123456789abcdef",
                &fixture.invocation,
                &fixture.paths,
                verified.stdout(),
                verified.stderr(),
            )
            .unwrap();

        let claim = receipt.to_consistency_claim();
        assert_eq!(claim.reported_disposition, ExternalReportedDisposition::Completed);
        let consistency = ExternalQualificationConsistencyReceipt {
            schema_version: EXTERNAL_QUALIFICATION_SCHEMA_VERSION,
            manifest_digest: fixture.manifest.digest().unwrap(),
            code_subject: "git:0123456789abcdef".into(),
            execution_claims: vec![claim],
        };
        consistency.validate_against(&fixture.manifest).unwrap();
    }

    #[test]
    fn tampered_materialized_input_fails_before_launch() {
        let fixture = fixture(success_script());
        fs::write(&fixture.paths.dataset, b"tampered").unwrap();
        assert_eq!(
            execute_external_benchmark(
                &fixture.manifest,
                "git:0123456789abcdef",
                &fixture.invocation,
                &fixture.paths,
            )
            .unwrap_err(),
            ExternalExecutionAttestationError::MaterializedAssetDigestMismatch { field: "dataset" }
        );
        assert!(!fixture.paths.result.exists());
    }

    #[test]
    fn preexisting_result_is_rejected() {
        let fixture = fixture(success_script());
        fs::write(&fixture.paths.result, b"stale-result").unwrap();
        assert!(matches!(
            execute_external_benchmark(
                &fixture.manifest,
                "git:0123456789abcdef",
                &fixture.invocation,
                &fixture.paths,
            ),
            Err(ExternalExecutionAttestationError::WorkingDirectoryNotEmpty(_))
                | Err(ExternalExecutionAttestationError::ResultAlreadyExists(_))
        ));
    }

    #[test]
    fn nonzero_process_cannot_mint_verified_execution() {
        let fixture = fixture("#!/bin/sh\nexit 7\n");
        assert_eq!(
            execute_external_benchmark(
                &fixture.manifest,
                "git:0123456789abcdef",
                &fixture.invocation,
                &fixture.paths,
            )
            .unwrap_err(),
            ExternalExecutionAttestationError::ProcessFailed { exit_code: Some(7) }
        );
    }

    #[test]
    fn successful_process_without_result_is_rejected() {
        let fixture = fixture("#!/bin/sh\nprintf 'no-result'\nexit 0\n");
        assert!(matches!(
            execute_external_benchmark(
                &fixture.manifest,
                "git:0123456789abcdef",
                &fixture.invocation,
                &fixture.paths,
            ),
            Err(ExternalExecutionAttestationError::ResultMissing(_))
        ));
    }

    #[test]
    fn undeclared_working_directory_output_is_rejected() {
        let fixture = fixture(
            "#!/bin/sh\nset -eu\nresult=\"$3\"\nprintf ok > \"$result\"\nprintf stray > stray.tmp\n",
        );
        assert!(matches!(
            execute_external_benchmark(
                &fixture.manifest,
                "git:0123456789abcdef",
                &fixture.invocation,
                &fixture.paths,
            ),
            Err(ExternalExecutionAttestationError::UnexpectedWorkingDirectoryEntry(_))
        ));
    }

    #[test]
    fn serialized_receipt_tampering_or_invocation_drift_fails_replay() {
        let fixture = fixture(success_script());
        let verified = execute_external_benchmark(
            &fixture.manifest,
            "git:0123456789abcdef",
            &fixture.invocation,
            &fixture.paths,
        )
        .unwrap();
        let bytes = serde_json::to_vec(verified.receipt()).unwrap();
        let mut decoded: ExternalProcessExecutionReceipt = serde_json::from_slice(&bytes).unwrap();
        decoded.result_digest = "0".repeat(64);
        assert!(decoded
            .verify_against_materialized(
                &fixture.manifest,
                "git:0123456789abcdef",
                &fixture.invocation,
                &fixture.paths,
                verified.stdout(),
                verified.stderr(),
            )
            .is_err());

        let mut drifted = fixture.invocation.clone();
        drifted
            .args
            .insert(0, ExternalInvocationArg::Literal("--drift".into()));
        assert_eq!(
            verified
                .receipt()
                .verify_against_materialized(
                    &fixture.manifest,
                    "git:0123456789abcdef",
                    &drifted,
                    &fixture.paths,
                    verified.stdout(),
                    verified.stderr(),
                )
                .unwrap_err(),
            ExternalExecutionAttestationError::InvocationDigestMismatch
        );
    }
}
