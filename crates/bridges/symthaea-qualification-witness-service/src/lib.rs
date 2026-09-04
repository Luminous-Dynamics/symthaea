// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Verify-then-sign service boundary for Agency qualification witnesses.
//!
//! The lower-level `symthaea-qualification-witness` crate intentionally exposes
//! deterministic signing/verification primitives. A production witness should
//! not expose those primitives as a remote "sign arbitrary acceptance JSON"
//! endpoint. This crate provides the operational boundary: it runs the exact
//! reviewed evidence verifier itself, with independently supplied release
//! commitments, parses only that verifier's release-bound acceptance, and only
//! then asks the enrolled witness key to sign.
//!
//! The verifier runtime identity commits both the Python interpreter and the
//! verifier script, their canonical paths, isolated invocation profile, and
//! resource limits. Production policy should require both paths in `/nix/store`.
//! A fixture policy may relax that bit for unit tests, but doing so changes the
//! implementation digest and therefore cannot satisfy a production witness
//! policy accidentally.

#![deny(unsafe_code)]

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};
use symthaea_authority::Digest32;
use symthaea_qualification_witness::{
    parse_release_acceptance_v1, sign_qualification_acceptance_v1,
    QualificationWitnessAttestationV1, QualificationWitnessError, QualificationWitnessPolicyV1,
    ACCEPTANCE_SCHEMA,
};
use thiserror::Error;

pub const VERIFIER_RUNTIME_SCHEMA_VERSION: u16 = 1;
pub const MAX_VERIFIER_FILE_BYTES: u64 = 128 * 1024 * 1024;
pub const MAX_VERIFIER_RUNTIME_MS: u64 = 120_000;
pub const MAX_VERIFIER_OUTPUT_BYTES: u64 = 1024 * 1024;

const VERIFIER_RUNTIME_DOMAIN: &[u8] = b"symthaea.qualification-evidence-verifier-runtime.v1\0";
const FILE_DIGEST_DOMAIN: &[u8] = b"symthaea.qualification-verifier-file.v1\0";
const INVOCATION_PROFILE: &[u8] = b"python-isolated-v1:-I,-B,env-clear,cwd-root,release-only";

/// Exact runtime policy for the evidence verifier executed by this witness.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationVerifierRuntimePolicyV1 {
    pub schema_version: u16,
    pub runtime_policy_id: [u8; 16],
    /// Canonical absolute Python executable path.
    pub python_executable_path: String,
    /// Domain-separated BLAKE3 commitment to the exact executable bytes.
    pub python_executable_digest: Digest32,
    /// Canonical absolute #431 verifier script path.
    pub verifier_script_path: String,
    /// Domain-separated BLAKE3 commitment to the exact script bytes.
    pub verifier_script_digest: Digest32,
    /// Production profiles should set this true for both runtime paths.
    pub require_nix_store_paths: bool,
    pub maximum_runtime_ms: u64,
    pub maximum_stdout_bytes: u64,
    pub maximum_stderr_bytes: u64,
}

impl QualificationVerifierRuntimePolicyV1 {
    pub fn validate(&self) -> Result<(), QualificationWitnessServiceError> {
        if self.schema_version != VERIFIER_RUNTIME_SCHEMA_VERSION
            || self.runtime_policy_id == [0; 16]
            || self.python_executable_digest.0 == [0; 32]
            || self.verifier_script_digest.0 == [0; 32]
            || !valid_absolute_path_string(&self.python_executable_path)
            || !valid_absolute_path_string(&self.verifier_script_path)
            || self.maximum_runtime_ms == 0
            || self.maximum_runtime_ms > MAX_VERIFIER_RUNTIME_MS
            || self.maximum_stdout_bytes == 0
            || self.maximum_stdout_bytes > MAX_VERIFIER_OUTPUT_BYTES
            || self.maximum_stderr_bytes == 0
            || self.maximum_stderr_bytes > MAX_VERIFIER_OUTPUT_BYTES
        {
            return Err(QualificationWitnessServiceError::InvalidRuntimePolicy);
        }
        Ok(())
    }

    /// Stable commitment used as #439's `verifier_digest`.
    pub fn implementation_digest(&self) -> Result<Digest32, QualificationWitnessServiceError> {
        self.validate()?;
        let mut transcript = Transcript::new(VERIFIER_RUNTIME_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.runtime_policy_id);
        transcript.bytes(self.python_executable_path.as_bytes())?;
        transcript.fixed(&self.python_executable_digest.0);
        transcript.bytes(self.verifier_script_path.as_bytes())?;
        transcript.fixed(&self.verifier_script_digest.0);
        transcript.u8(u8::from(self.require_nix_store_paths));
        transcript.u64(self.maximum_runtime_ms);
        transcript.u64(self.maximum_stdout_bytes);
        transcript.u64(self.maximum_stderr_bytes);
        transcript.bytes(INVOCATION_PROFILE)?;
        transcript.bytes(ACCEPTANCE_SCHEMA.as_bytes())?;
        Ok(Digest32(transcript.finish()))
    }
}

/// Independently obtained release bindings supplied to the witness service.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReleaseEvidenceBindingsV1 {
    pub archive_sha256: Digest32,
    pub git_head: [u8; 20],
    pub git_tree: [u8; 20],
}

impl ReleaseEvidenceBindingsV1 {
    fn validate(self) -> Result<(), QualificationWitnessServiceError> {
        if self.archive_sha256.0 == [0; 32]
            || self.git_head == [0; 20]
            || self.git_tree == [0; 20]
        {
            return Err(QualificationWitnessServiceError::InvalidReleaseBindings);
        }
        Ok(())
    }
}

/// Privacy-minimized result of one verify-then-sign operation.
#[derive(Debug)]
pub struct VerifiedThenSignedQualificationV1 {
    pub attestation: QualificationWitnessAttestationV1,
    acceptance_digest: Digest32,
    verifier_digest: Digest32,
    archive_sha256: Digest32,
    git_head: [u8; 20],
    git_tree: [u8; 20],
}

impl VerifiedThenSignedQualificationV1 {
    pub fn acceptance_digest(&self) -> Digest32 {
        self.acceptance_digest
    }

    pub fn verifier_digest(&self) -> Digest32 {
        self.verifier_digest
    }

    pub fn archive_sha256(&self) -> Digest32 {
        self.archive_sha256
    }

    pub fn git_head(&self) -> [u8; 20] {
        self.git_head
    }

    pub fn git_tree(&self) -> [u8; 20] {
        self.git_tree
    }
}

/// Production entry point: run the reviewed verifier and sign only its own
/// release-bound acceptance.
pub fn verify_archive_then_sign_v1(
    runtime_policy: &QualificationVerifierRuntimePolicyV1,
    witness_policy: &QualificationWitnessPolicyV1,
    witness_id: [u8; 16],
    witness_sequence: u64,
    signing_key: &SigningKey,
    archive_path: &Path,
    release_bindings: ReleaseEvidenceBindingsV1,
) -> Result<VerifiedThenSignedQualificationV1, QualificationWitnessServiceError> {
    verify_archive_then_sign_with_runner(
        runtime_policy,
        witness_policy,
        witness_id,
        witness_sequence,
        signing_key,
        archive_path,
        release_bindings,
        &SystemEvidenceVerifierRunner,
    )
}

fn verify_archive_then_sign_with_runner<R: EvidenceVerifierRunner>(
    runtime_policy: &QualificationVerifierRuntimePolicyV1,
    witness_policy: &QualificationWitnessPolicyV1,
    witness_id: [u8; 16],
    witness_sequence: u64,
    signing_key: &SigningKey,
    archive_path: &Path,
    release_bindings: ReleaseEvidenceBindingsV1,
    runner: &R,
) -> Result<VerifiedThenSignedQualificationV1, QualificationWitnessServiceError> {
    runtime_policy.validate()?;
    release_bindings.validate()?;
    if witness_sequence == 0 {
        return Err(QualificationWitnessServiceError::InvalidWitnessSequence);
    }

    let archive_meta = fs::symlink_metadata(archive_path)?;
    if !archive_meta.file_type().is_file() {
        return Err(QualificationWitnessServiceError::ArchiveNotRegularFile);
    }

    let before = measure_runtime(runtime_policy)?;
    let verifier_digest = runtime_policy.implementation_digest()?;

    let invocation = VerifierInvocation {
        python_executable: before.python_path.clone(),
        verifier_script: before.script_path.clone(),
        archive_path: archive_path.to_path_buf(),
        release_bindings,
        maximum_runtime_ms: runtime_policy.maximum_runtime_ms,
        maximum_stdout_bytes: runtime_policy.maximum_stdout_bytes,
        maximum_stderr_bytes: runtime_policy.maximum_stderr_bytes,
    };
    let output = runner.run(&invocation)?;
    if !output.stderr.is_empty() {
        return Err(QualificationWitnessServiceError::VerifierUnexpectedStderr(
            digest_bytes(&output.stderr),
        ));
    }
    if output.stdout.is_empty() {
        return Err(QualificationWitnessServiceError::VerifierEmptyAcceptance);
    }

    // Runtime identity must still be exact after execution. For Nix-store
    // production paths this also gives a strong immutable-path expectation;
    // the double measurement catches ordinary mutation/race conditions.
    let after = measure_runtime(runtime_policy)?;
    if before != after {
        return Err(QualificationWitnessServiceError::VerifierRuntimeChanged);
    }

    let acceptance = parse_release_acceptance_v1(&output.stdout)?;
    if acceptance.archive_sha256() != release_bindings.archive_sha256
        || acceptance.head() != release_bindings.git_head
        || acceptance.tree() != release_bindings.git_tree
    {
        return Err(QualificationWitnessServiceError::AcceptanceBindingMismatch);
    }

    let acceptance_digest = acceptance.digest()?;
    let attestation = sign_qualification_acceptance_v1(
        &acceptance,
        verifier_digest,
        witness_policy,
        witness_id,
        witness_sequence,
        signing_key,
    )?;

    Ok(VerifiedThenSignedQualificationV1 {
        attestation,
        acceptance_digest,
        verifier_digest,
        archive_sha256: release_bindings.archive_sha256,
        git_head: release_bindings.git_head,
        git_tree: release_bindings.git_tree,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RuntimeMeasurement {
    python_path: PathBuf,
    python_digest: Digest32,
    script_path: PathBuf,
    script_digest: Digest32,
}

fn measure_runtime(
    policy: &QualificationVerifierRuntimePolicyV1,
) -> Result<RuntimeMeasurement, QualificationWitnessServiceError> {
    let python_path = canonical_exact_path(&policy.python_executable_path)?;
    let script_path = canonical_exact_path(&policy.verifier_script_path)?;
    if policy.require_nix_store_paths
        && (!path_in_nix_store(&python_path) || !path_in_nix_store(&script_path))
    {
        return Err(QualificationWitnessServiceError::VerifierOutsideNixStore);
    }

    let python_digest = digest_file(&python_path, MAX_VERIFIER_FILE_BYTES)?;
    let script_digest = digest_file(&script_path, MAX_VERIFIER_FILE_BYTES)?;
    if python_digest != policy.python_executable_digest
        || script_digest != policy.verifier_script_digest
    {
        return Err(QualificationWitnessServiceError::VerifierRuntimeDigestMismatch);
    }
    Ok(RuntimeMeasurement {
        python_path,
        python_digest,
        script_path,
        script_digest,
    })
}

fn canonical_exact_path(configured: &str) -> Result<PathBuf, QualificationWitnessServiceError> {
    if !valid_absolute_path_string(configured) {
        return Err(QualificationWitnessServiceError::InvalidRuntimePolicy);
    }
    let configured_path = PathBuf::from(configured);
    let canonical = fs::canonicalize(&configured_path)?;
    if canonical != configured_path {
        return Err(QualificationWitnessServiceError::VerifierPathNotCanonical);
    }
    let metadata = fs::metadata(&canonical)?;
    if !metadata.is_file() {
        return Err(QualificationWitnessServiceError::VerifierRuntimeNotRegularFile);
    }
    Ok(canonical)
}

fn valid_absolute_path_string(value: &str) -> bool {
    !value.is_empty()
        && Path::new(value).is_absolute()
        && !value
            .bytes()
            .any(|byte| byte == 0 || byte.is_ascii_control())
}

fn path_in_nix_store(path: &Path) -> bool {
    let text = path.to_string_lossy();
    text.starts_with("/nix/store/") && !text.contains("/../") && !text.contains("/./")
}

fn digest_file(path: &Path, maximum_bytes: u64) -> Result<Digest32, QualificationWitnessServiceError> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > maximum_bytes {
        return Err(QualificationWitnessServiceError::VerifierRuntimeNotRegularFile);
    }
    let bytes = fs::read(path)?;
    if bytes.is_empty()
        || u64::try_from(bytes.len()).map_err(|_| QualificationWitnessServiceError::Encoding)?
            > maximum_bytes
    {
        return Err(QualificationWitnessServiceError::VerifierRuntimeNotRegularFile);
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(FILE_DIGEST_DOMAIN);
    hasher.update(&bytes);
    Ok(Digest32(*hasher.finalize().as_bytes()))
}

fn digest_bytes(bytes: &[u8]) -> Digest32 {
    Digest32(*blake3::hash(bytes).as_bytes())
}

#[derive(Debug)]
struct VerifierInvocation {
    python_executable: PathBuf,
    verifier_script: PathBuf,
    archive_path: PathBuf,
    release_bindings: ReleaseEvidenceBindingsV1,
    maximum_runtime_ms: u64,
    maximum_stdout_bytes: u64,
    maximum_stderr_bytes: u64,
}

#[derive(Debug)]
struct VerifierOutput {
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

trait EvidenceVerifierRunner {
    fn run(
        &self,
        invocation: &VerifierInvocation,
    ) -> Result<VerifierOutput, QualificationWitnessServiceError>;
}

struct SystemEvidenceVerifierRunner;

impl EvidenceVerifierRunner for SystemEvidenceVerifierRunner {
    fn run(
        &self,
        invocation: &VerifierInvocation,
    ) -> Result<VerifierOutput, QualificationWitnessServiceError> {
        let stdout_limit = usize::try_from(invocation.maximum_stdout_bytes)
            .map_err(|_| QualificationWitnessServiceError::Encoding)?;
        let stderr_limit = usize::try_from(invocation.maximum_stderr_bytes)
            .map_err(|_| QualificationWitnessServiceError::Encoding)?;

        let mut command = Command::new(&invocation.python_executable);
        command
            .env_clear()
            .current_dir("/")
            .arg("-I")
            .arg("-B")
            .arg(&invocation.verifier_script)
            .arg(&invocation.archive_path)
            .arg("--release")
            .arg("--expected-archive-sha256")
            .arg(hex_lower(&invocation.release_bindings.archive_sha256.0))
            .arg("--expected-head")
            .arg(hex_lower(&invocation.release_bindings.git_head))
            .arg("--expected-tree")
            .arg(hex_lower(&invocation.release_bindings.git_tree))
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let mut child = command.spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or(QualificationWitnessServiceError::VerifierPipeUnavailable)?;
        let stderr = child
            .stderr
            .take()
            .ok_or(QualificationWitnessServiceError::VerifierPipeUnavailable)?;

        let overflow = Arc::new(AtomicBool::new(false));
        let stdout_reader = spawn_bounded_reader(stdout, stdout_limit, Arc::clone(&overflow));
        let stderr_reader = spawn_bounded_reader(stderr, stderr_limit, Arc::clone(&overflow));

        let deadline = Instant::now()
            .checked_add(Duration::from_millis(invocation.maximum_runtime_ms))
            .ok_or(QualificationWitnessServiceError::Encoding)?;
        let mut timed_out = false;
        let status = loop {
            if overflow.load(Ordering::SeqCst) {
                let _ = child.kill();
                break child.wait()?;
            }
            if Instant::now() >= deadline {
                timed_out = true;
                let _ = child.kill();
                break child.wait()?;
            }
            if let Some(status) = child.try_wait()? {
                break status;
            }
            thread::sleep(Duration::from_millis(5));
        };

        let stdout = stdout_reader
            .join()
            .map_err(|_| QualificationWitnessServiceError::VerifierReaderPanicked)??;
        let stderr = stderr_reader
            .join()
            .map_err(|_| QualificationWitnessServiceError::VerifierReaderPanicked)??;

        if overflow.load(Ordering::SeqCst) {
            return Err(QualificationWitnessServiceError::VerifierOutputLimitExceeded);
        }
        if timed_out {
            return Err(QualificationWitnessServiceError::VerifierTimedOut);
        }
        if !status.success() {
            return Err(QualificationWitnessServiceError::EvidenceVerifierRejected {
                stdout_digest: digest_bytes(&stdout),
                stderr_digest: digest_bytes(&stderr),
            });
        }

        Ok(VerifierOutput { stdout, stderr })
    }
}

fn spawn_bounded_reader<R>(
    mut reader: R,
    maximum_bytes: usize,
    overflow: Arc<AtomicBool>,
) -> thread::JoinHandle<Result<Vec<u8>, QualificationWitnessServiceError>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut retained = Vec::with_capacity(maximum_bytes.min(64 * 1024));
        let mut buffer = [0u8; 8192];
        loop {
            let count = reader.read(&mut buffer)?;
            if count == 0 {
                break;
            }
            let remaining = maximum_bytes.saturating_sub(retained.len());
            if count <= remaining {
                retained.extend_from_slice(&buffer[..count]);
            } else {
                retained.extend_from_slice(&buffer[..remaining]);
                overflow.store(true, Ordering::SeqCst);
            }
        }
        Ok(retained)
    })
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

#[derive(Debug, Error)]
pub enum QualificationWitnessServiceError {
    #[error("invalid evidence-verifier runtime policy")]
    InvalidRuntimePolicy,
    #[error("invalid independently supplied release bindings")]
    InvalidReleaseBindings,
    #[error("witness sequence must be nonzero")]
    InvalidWitnessSequence,
    #[error("qualification archive path is not a regular file")]
    ArchiveNotRegularFile,
    #[error("evidence-verifier path is not already canonical")]
    VerifierPathNotCanonical,
    #[error("evidence-verifier runtime file is invalid or oversized")]
    VerifierRuntimeNotRegularFile,
    #[error("production evidence-verifier runtime resolved outside /nix/store")]
    VerifierOutsideNixStore,
    #[error("evidence-verifier runtime bytes do not match reviewed policy")]
    VerifierRuntimeDigestMismatch,
    #[error("evidence-verifier runtime changed during verification")]
    VerifierRuntimeChanged,
    #[error("evidence verifier exceeded its output bound")]
    VerifierOutputLimitExceeded,
    #[error("evidence verifier exceeded its runtime bound")]
    VerifierTimedOut,
    #[error("evidence verifier stdout/stderr pipe unavailable")]
    VerifierPipeUnavailable,
    #[error("evidence verifier output reader panicked")]
    VerifierReaderPanicked,
    #[error("evidence verifier returned success with stderr commitment {0:?}")]
    VerifierUnexpectedStderr(Digest32),
    #[error("evidence verifier returned success without acceptance JSON")]
    VerifierEmptyAcceptance,
    #[error("evidence verifier rejected archive; stdout {stdout_digest:?}, stderr {stderr_digest:?}")]
    EvidenceVerifierRejected {
        stdout_digest: Digest32,
        stderr_digest: Digest32,
    },
    #[error("release acceptance disagrees with independently supplied bindings")]
    AcceptanceBindingMismatch,
    #[error("canonical encoding failed")]
    Encoding,
    #[error("witness protocol rejected operation: {0}")]
    Witness(#[from] QualificationWitnessError),
    #[error("I/O failure: {0}")]
    Io(#[from] std::io::Error),
}

struct Transcript {
    bytes: Vec<u8>,
}

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(domain.len() + 512);
        bytes.extend_from_slice(domain);
        Self { bytes }
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn fixed(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn bytes(&mut self, value: &[u8]) -> Result<(), QualificationWitnessServiceError> {
        let len = u32::try_from(value.len()).map_err(|_| QualificationWitnessServiceError::Encoding)?;
        self.bytes.extend_from_slice(&len.to_be_bytes());
        self.bytes.extend_from_slice(value);
        Ok(())
    }

    fn finish(self) -> [u8; 32] {
        *blake3::hash(&self.bytes).as_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Mutex;

    struct Fixture {
        root: PathBuf,
        python: PathBuf,
        script: PathBuf,
        archive: PathBuf,
    }

    impl Fixture {
        fn new() -> Self {
            let suffix = format!("{}-{}", std::process::id(), Instant::now().elapsed().as_nanos());
            let root = std::env::temp_dir().join(format!("symthaea-witness-service-{suffix}"));
            fs::create_dir(&root).unwrap();
            let python = root.join("python3");
            let script = root.join("verify.py");
            let archive = root.join("evidence.tar.gz");
            fs::write(&python, b"fixture python runtime").unwrap();
            fs::write(&script, b"fixture verifier script").unwrap();
            fs::write(&archive, b"fixture archive").unwrap();
            Self { root, python, script, archive }
        }
    }

    impl Drop for Fixture {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.root);
        }
    }

    fn runtime_policy(fixture: &Fixture) -> QualificationVerifierRuntimePolicyV1 {
        QualificationVerifierRuntimePolicyV1 {
            schema_version: VERIFIER_RUNTIME_SCHEMA_VERSION,
            runtime_policy_id: [0x31; 16],
            python_executable_path: fs::canonicalize(&fixture.python)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            python_executable_digest: digest_file(&fixture.python, MAX_VERIFIER_FILE_BYTES).unwrap(),
            verifier_script_path: fs::canonicalize(&fixture.script)
                .unwrap()
                .to_string_lossy()
                .into_owned(),
            verifier_script_digest: digest_file(&fixture.script, MAX_VERIFIER_FILE_BYTES).unwrap(),
            require_nix_store_paths: false,
            maximum_runtime_ms: 5_000,
            maximum_stdout_bytes: 64 * 1024,
            maximum_stderr_bytes: 64 * 1024,
        }
    }

    fn bindings() -> ReleaseEvidenceBindingsV1 {
        ReleaseEvidenceBindingsV1 {
            archive_sha256: Digest32([0x11; 32]),
            git_head: [0x22; 20],
            git_tree: [0x33; 20],
        }
    }

    fn acceptance_json(bindings: ReleaseEvidenceBindingsV1) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema": ACCEPTANCE_SCHEMA,
            "accepted": true,
            "qualification_result": "PASS",
            "archive_sha256": hex_lower(&bindings.archive_sha256.0),
            "archive_hash_source": "caller",
            "manifest_sha256": "12".repeat(32),
            "head": hex_lower(&bindings.git_head),
            "tree": hex_lower(&bindings.git_tree),
            "external_head_bound": true,
            "external_tree_bound": true,
            "release_bound": true,
            "nixpkgs_locked": {
                "type": "github",
                "owner": "NixOS",
                "repo": "nixpkgs",
                "rev": "abc123",
                "narHash": "sha256-example"
            },
            "flake_lock_sha256": "15".repeat(32),
            "rust_toolchain_sha256": "16".repeat(32),
            "approved_pcr_profile": "17".repeat(32),
            "policy_digest": "18".repeat(32),
            "ak_public_digest": "19".repeat(32),
            "challenge_digest": "1a".repeat(32),
            "probe_sha256": "1b".repeat(32),
            "quote_wrapper_sha256": "1c".repeat(32),
            "checkquote_wrapper_sha256": "1d".repeat(32),
            "verifier_store": "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-symthaea-tpm2-verifier-v1"
        }))
        .unwrap()
    }

    fn witness_key() -> SigningKey {
        SigningKey::from_bytes(&[7; 32])
    }

    fn witness_policy(
        runtime: &QualificationVerifierRuntimePolicyV1,
        key: &SigningKey,
    ) -> QualificationWitnessPolicyV1 {
        symthaea_qualification_witness::QualificationWitnessPolicyV1 {
            schema_version: symthaea_qualification_witness::WITNESS_SCHEMA_VERSION,
            policy_id: [0x41; 16],
            witness_epoch: 1,
            threshold: 1,
            minimum_organizations: 1,
            minimum_services: 1,
            allowed_verifier_digests: vec![runtime.implementation_digest().unwrap()],
            witnesses: vec![symthaea_qualification_witness::QualificationWitnessIdentityV1 {
                witness_id: [1; 16],
                organization_id: [2; 16],
                service_id: [3; 16],
                public_key: key.verifying_key().to_bytes(),
            }],
        }
    }

    struct FakeRunner {
        stdout: Vec<u8>,
        stderr: Vec<u8>,
        mutate_script: Option<PathBuf>,
        calls: Mutex<u32>,
    }

    impl EvidenceVerifierRunner for FakeRunner {
        fn run(
            &self,
            _invocation: &VerifierInvocation,
        ) -> Result<VerifierOutput, QualificationWitnessServiceError> {
            *self.calls.lock().unwrap() += 1;
            if let Some(path) = &self.mutate_script {
                fs::write(path, b"changed verifier script").unwrap();
            }
            Ok(VerifierOutput {
                stdout: self.stdout.clone(),
                stderr: self.stderr.clone(),
            })
        }
    }

    #[test]
    fn verify_then_sign_accepts_only_verifier_owned_release_acceptance() {
        let fixture = Fixture::new();
        let runtime = runtime_policy(&fixture);
        let bindings = bindings();
        let key = witness_key();
        let witnesses = witness_policy(&runtime, &key);
        let runner = FakeRunner {
            stdout: acceptance_json(bindings),
            stderr: Vec::new(),
            mutate_script: None,
            calls: Mutex::new(0),
        };
        let result = verify_archive_then_sign_with_runner(
            &runtime,
            &witnesses,
            [1; 16],
            9,
            &key,
            &fixture.archive,
            bindings,
            &runner,
        )
        .unwrap();
        assert_eq!(*runner.calls.lock().unwrap(), 1);
        assert_eq!(result.archive_sha256(), bindings.archive_sha256);
        assert_eq!(result.verifier_digest(), runtime.implementation_digest().unwrap());
    }

    #[test]
    fn caller_cannot_substitute_a_different_release_acceptance() {
        let fixture = Fixture::new();
        let runtime = runtime_policy(&fixture);
        let bindings = bindings();
        let key = witness_key();
        let witnesses = witness_policy(&runtime, &key);
        let mut wrong = bindings;
        wrong.git_tree = [0x77; 20];
        let runner = FakeRunner {
            stdout: acceptance_json(wrong),
            stderr: Vec::new(),
            mutate_script: None,
            calls: Mutex::new(0),
        };
        assert!(matches!(
            verify_archive_then_sign_with_runner(
                &runtime,
                &witnesses,
                [1; 16],
                9,
                &key,
                &fixture.archive,
                bindings,
                &runner,
            ),
            Err(QualificationWitnessServiceError::AcceptanceBindingMismatch)
        ));
    }

    #[test]
    fn verifier_runtime_change_after_execution_prevents_signature() {
        let fixture = Fixture::new();
        let runtime = runtime_policy(&fixture);
        let bindings = bindings();
        let key = witness_key();
        let witnesses = witness_policy(&runtime, &key);
        let runner = FakeRunner {
            stdout: acceptance_json(bindings),
            stderr: Vec::new(),
            mutate_script: Some(fixture.script.clone()),
            calls: Mutex::new(0),
        };
        assert!(matches!(
            verify_archive_then_sign_with_runner(
                &runtime,
                &witnesses,
                [1; 16],
                9,
                &key,
                &fixture.archive,
                bindings,
                &runner,
            ),
            Err(QualificationWitnessServiceError::VerifierRuntimeDigestMismatch)
                | Err(QualificationWitnessServiceError::VerifierRuntimeChanged)
        ));
    }

    #[test]
    fn success_with_verifier_stderr_is_not_signable() {
        let fixture = Fixture::new();
        let runtime = runtime_policy(&fixture);
        let bindings = bindings();
        let key = witness_key();
        let witnesses = witness_policy(&runtime, &key);
        let runner = FakeRunner {
            stdout: acceptance_json(bindings),
            stderr: b"unexpected warning".to_vec(),
            mutate_script: None,
            calls: Mutex::new(0),
        };
        assert!(matches!(
            verify_archive_then_sign_with_runner(
                &runtime,
                &witnesses,
                [1; 16],
                9,
                &key,
                &fixture.archive,
                bindings,
                &runner,
            ),
            Err(QualificationWitnessServiceError::VerifierUnexpectedStderr(_))
        ));
    }

    #[test]
    fn production_nix_store_policy_rejects_fixture_runtime_before_runner() {
        let fixture = Fixture::new();
        let mut runtime = runtime_policy(&fixture);
        runtime.require_nix_store_paths = true;
        let bindings = bindings();
        let key = witness_key();
        // This witness policy deliberately permits the changed runtime digest;
        // the local filesystem trust gate must still fail first.
        let witnesses = witness_policy(&runtime, &key);
        let runner = FakeRunner {
            stdout: acceptance_json(bindings),
            stderr: Vec::new(),
            mutate_script: None,
            calls: Mutex::new(0),
        };
        assert!(matches!(
            verify_archive_then_sign_with_runner(
                &runtime,
                &witnesses,
                [1; 16],
                9,
                &key,
                &fixture.archive,
                bindings,
                &runner,
            ),
            Err(QualificationWitnessServiceError::VerifierOutsideNixStore)
        ));
        assert_eq!(*runner.calls.lock().unwrap(), 0);
    }
}
