// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! File-descriptor-pinned execution identity for the TPM2 checkquote adapter.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_tpm2_attestation_possession::{AttestationChallenge, AttestationKeyBinding};
use symthaea_assurance_tpm2_checkquote_adapter::{
    verify_tpm2_quote, CheckquoteExecutionRequest, CheckquoteExecutor, RawTpm2QuoteBundle,
    ToolExecution, Tpm2CheckquoteDisposition, Tpm2CheckquotePolicy, Tpm2QuoteQualification,
};

pub const FD_PINNED_CHECKQUOTE_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.tpm2-fd-pinned-checkquote-policy.v1";
pub const FD_PINNED_CHECKQUOTE_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.tpm2-fd-pinned-checkquote-report.v1";
pub const FD_PINNED_EXECUTION_MODE_V1: &str =
    "same-open-elf-hash-and-exec-via-proc-self-fd-0-v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm2-fd-pinned-checkquote-policy.digest.v1\0";
const EXECUTION_RECORD_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm2-fd-pinned-checkquote-execution.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm2-fd-pinned-checkquote-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm2-fd-pinned-checkquote-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 1024;
const MAX_EVIDENCE_REFS: usize = 128;
const NIX_BASE32: &[u8] = b"0123456789abcdfghijklmnpqrsvwxyz";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FdPinnedCheckquotePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_inner_policy_digest: String,
    pub require_direct_nix_store_path: bool,
    pub require_root_owned_store_object: bool,
    pub require_non_writable_store_object: bool,
    pub require_elf: bool,
    pub evidence_refs: Vec<String>,
}

impl FdPinnedCheckquotePolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == FD_PINNED_CHECKQUOTE_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3_digest(&self.expected_inner_policy_digest)
            && self.require_direct_nix_store_path
            && self.require_root_owned_store_object
            && self.require_non_writable_store_object
            && self.require_elf
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.policy_id);
        push_field(&mut hasher, &self.expected_inner_policy_digest);
        for value in [
            self.require_direct_nix_store_path,
            self.require_root_owned_store_object,
            self.require_non_writable_store_object,
            self.require_elf,
        ] {
            hasher.update(&[u8::from(value)]);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FdPinnedExecutionRecord {
    pub execution_mode: String,
    pub canonical_path: String,
    pub nix_store_root: String,
    pub executable_blake3: String,
    pub device: u64,
    pub inode: u64,
    pub file_size: u64,
    pub root_owned: bool,
    pub non_writable: bool,
    pub elf_magic_valid: bool,
}

impl FdPinnedExecutionRecord {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(EXECUTION_RECORD_DIGEST_DOMAIN);
        for value in [
            self.execution_mode.as_str(),
            self.canonical_path.as_str(),
            self.nix_store_root.as_str(),
            self.executable_blake3.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        hasher.update(&self.device.to_le_bytes());
        hasher.update(&self.inode.to_le_bytes());
        hasher.update(&self.file_size.to_le_bytes());
        for value in [self.root_owned, self.non_writable, self.elf_magic_valid] {
            hasher.update(&[u8::from(value)]);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FdPinnedCheckquoteDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FdPinnedCheckquoteIssue {
    InvalidPolicy,
    InvalidInnerPolicy,
    InnerPolicyMismatch,
    UnsupportedPlatform,
    InnerQuoteRejected {
        inner_report_digest: String,
        inner_disposition: String,
    },
    MissingExecutionRecord,
    ExecutionPathMismatch,
    ExecutionDigestMismatch,
    ExecutionReceiptMismatch,
}

impl FdPinnedCheckquoteIssue {
    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::InvalidInnerPolicy => "invalid-inner-policy".into(),
            Self::InnerPolicyMismatch => "inner-policy-mismatch".into(),
            Self::UnsupportedPlatform => "unsupported-platform".into(),
            Self::InnerQuoteRejected {
                inner_report_digest,
                inner_disposition,
            } => format!("inner-quote-rejected:{inner_disposition}:{inner_report_digest}"),
            Self::MissingExecutionRecord => "missing-execution-record".into(),
            Self::ExecutionPathMismatch => "execution-path-mismatch".into(),
            Self::ExecutionDigestMismatch => "execution-digest-mismatch".into(),
            Self::ExecutionReceiptMismatch => "execution-receipt-mismatch".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FdPinnedCheckquoteReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub inner_policy_digest: Option<String>,
    pub inner_quote_report_digest: Option<String>,
    pub inner_quote_qualification_digest: Option<String>,
    pub execution_record_digest: Option<String>,
    pub executable_digest: Option<String>,
    pub canonical_path: Option<String>,
    pub nix_store_root: Option<String>,
    pub disposition: FdPinnedCheckquoteDisposition,
    pub issues: Vec<FdPinnedCheckquoteIssue>,
}

impl FdPinnedCheckquoteReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.inner_policy_digest.as_deref().unwrap_or("-"),
            self.inner_quote_report_digest.as_deref().unwrap_or("-"),
            self.inner_quote_qualification_digest.as_deref().unwrap_or("-"),
            self.execution_record_digest.as_deref().unwrap_or("-"),
            self.executable_digest.as_deref().unwrap_or("-"),
            self.canonical_path.as_deref().unwrap_or("-"),
            self.nix_store_root.as_deref().unwrap_or("-"),
        ] {
            push_field(&mut hasher, value);
        }
        push_field(
            &mut hasher,
            match self.disposition {
                FdPinnedCheckquoteDisposition::Invalid => "invalid",
                FdPinnedCheckquoteDisposition::Blocked => "blocked",
                FdPinnedCheckquoteDisposition::Qualified => "qualified",
            },
        );
        for issue in &self.issues {
            push_field(&mut hasher, &issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedFdPinnedTpm2Quote {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    inner_policy_digest: String,
    inner_quote_qualification_digest: String,
    inner_quote_policy_digest: String,
    challenge_digest: String,
    ak_binding_digest: String,
    quote_artifact_digest: String,
    verification_receipt_digest: String,
    pcr_selection_digest: String,
    execution_record_digest: String,
    executable_digest: String,
    canonical_path: String,
    nix_store_root: String,
    device: u64,
    inode: u64,
    file_size: u64,
}

impl VerifiedFdPinnedTpm2Quote {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn inner_policy_digest(&self) -> &str { &self.inner_policy_digest }
    pub fn inner_quote_qualification_digest(&self) -> &str { &self.inner_quote_qualification_digest }
    pub fn inner_quote_policy_digest(&self) -> &str { &self.inner_quote_policy_digest }
    pub fn challenge_digest(&self) -> &str { &self.challenge_digest }
    pub fn ak_binding_digest(&self) -> &str { &self.ak_binding_digest }
    pub fn quote_artifact_digest(&self) -> &str { &self.quote_artifact_digest }
    pub fn verification_receipt_digest(&self) -> &str { &self.verification_receipt_digest }
    pub fn pcr_selection_digest(&self) -> &str { &self.pcr_selection_digest }
    pub fn execution_record_digest(&self) -> &str { &self.execution_record_digest }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }
    pub fn canonical_path(&self) -> &str { &self.canonical_path }
    pub fn nix_store_root(&self) -> &str { &self.nix_store_root }
    pub const fn device(&self) -> u64 { self.device }
    pub const fn inode(&self) -> u64 { self.inode }
    pub const fn file_size(&self) -> u64 { self.file_size }
    pub const fn execution_mode(&self) -> &'static str { FD_PINNED_EXECUTION_MODE_V1 }
    pub const fn executable_path_toctou_closed(&self) -> bool { true }
    pub const fn dependency_closure_atomically_pinned(&self) -> bool { false }
    pub const fn root_resistant_immutability_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FdPinnedTpm2QuoteQualification {
    pub report: FdPinnedCheckquoteReport,
    pub quote: Tpm2QuoteQualification,
    pub execution_record: FdPinnedExecutionRecord,
    verified: VerifiedFdPinnedTpm2Quote,
}

impl FdPinnedTpm2QuoteQualification {
    pub fn verified(&self) -> &VerifiedFdPinnedTpm2Quote { &self.verified }
    pub fn into_verified(self) -> VerifiedFdPinnedTpm2Quote { self.verified }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::ffi::OsString;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom, Write};
    use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt};
    use std::path::{Path, PathBuf};
    use std::process::{Command, Output, Stdio};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    struct PinnedExecutable {
        file: File,
        record: FdPinnedExecutionRecord,
    }

    #[derive(Default)]
    struct ExecutorState {
        pending: Option<PinnedExecutable>,
        last: Option<FdPinnedExecutionRecord>,
    }

    pub struct FdPinnedCheckquoteExecutor {
        expected_path: String,
        state: Mutex<ExecutorState>,
    }

    impl FdPinnedCheckquoteExecutor {
        pub fn new(expected_path: String) -> Self {
            Self {
                expected_path,
                state: Mutex::new(ExecutorState::default()),
            }
        }

        pub fn last_execution_record(&self) -> Result<Option<FdPinnedExecutionRecord>, String> {
            self.state
                .lock()
                .map(|state| state.last.clone())
                .map_err(|_| "fd-pinned executor state poisoned".into())
        }
    }

    impl CheckquoteExecutor for FdPinnedCheckquoteExecutor {
        fn executable_blake3(&self, executable: &str) -> Result<String, String> {
            if executable != self.expected_path {
                return Err("checkquote path differs from fd-pinned executor policy".into());
            }
            let pinned = open_production_pinned_executable(executable)?;
            let digest = pinned.record.executable_blake3.clone();
            let mut state = self
                .state
                .lock()
                .map_err(|_| "fd-pinned executor state poisoned".to_string())?;
            if state.pending.is_some() {
                return Err("fd-pinned executor already has a pending executable".into());
            }
            state.pending = Some(pinned);
            Ok(digest)
        }

        fn execute_checkquote(
            &self,
            executable: &str,
            request: &CheckquoteExecutionRequest,
        ) -> Result<ToolExecution, String> {
            if executable != self.expected_path {
                return Err("checkquote execution path differs from fd-pinned policy".into());
            }
            let pinned = {
                let mut state = self
                    .state
                    .lock()
                    .map_err(|_| "fd-pinned executor state poisoned".to_string())?;
                state
                    .pending
                    .take()
                    .ok_or_else(|| "no previously hashed open executable is pending".to_string())?
            };

            let workspace = SecureTempWorkspace::create()?;
            let public_path = workspace.write("ak-public.bin", &request.ak_public)?;
            let message_path = workspace.write("quote-message.bin", &request.quote_message)?;
            let signature_path = workspace.write("quote-signature.bin", &request.signature)?;
            let pcr_path = workspace.write("pcr-values.bin", &request.pcr_values_raw)?;

            let args = vec![
                OsString::from("-u"),
                public_path.into_os_string(),
                OsString::from("-m"),
                message_path.into_os_string(),
                OsString::from("-s"),
                signature_path.into_os_string(),
                OsString::from("-f"),
                pcr_path.into_os_string(),
                OsString::from("-l"),
                OsString::from(&request.pcr_list),
                OsString::from("-g"),
                OsString::from(&request.hash_algorithm),
                OsString::from("-q"),
                OsString::from(&request.qualification_hex),
            ];

            let output = execute_open_file(pinned.file, &args, &[])?;
            let record = pinned.record;
            let mut state = self
                .state
                .lock()
                .map_err(|_| "fd-pinned executor state poisoned".to_string())?;
            state.last = Some(record);

            Ok(ToolExecution {
                exit_code: output.status.code(),
                stdout: output.stdout,
                stderr: output.stderr,
            })
        }
    }

    fn execute_open_file(
        file: File,
        args: &[OsString],
        env: &[(OsString, OsString)],
    ) -> Result<Output, String> {
        let mut command = Command::new("/proc/self/fd/0");
        command.args(args).env_clear();
        for (key, value) in env {
            command.env(key, value);
        }
        command
            .stdin(Stdio::from(file))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .map_err(|error| format!("fd-pinned executable launch failed: {error}"))
    }

    fn open_production_pinned_executable(path: &str) -> Result<PinnedExecutable, String> {
        let requested = PathBuf::from(path);
        if !requested.is_absolute() {
            return Err("checkquote path must be absolute".into());
        }
        let canonical = std::fs::canonicalize(&requested)
            .map_err(|error| format!("canonicalize checkquote path: {error}"))?;
        if canonical != requested {
            return Err("checkquote path must already be canonical and symlink-free".into());
        }
        let store_root = direct_nix_store_root(&canonical)
            .ok_or_else(|| "checkquote must be a direct /nix/store object path".to_string())?;

        let store_metadata = std::fs::metadata(&store_root)
            .map_err(|error| format!("stat Nix store object: {error}"))?;
        if !store_metadata.is_dir() || store_metadata.uid() != 0 {
            return Err("Nix store object must be a root-owned directory".into());
        }
        if store_metadata.mode() & 0o222 != 0 {
            return Err("Nix store object must not have writable permission bits".into());
        }

        open_pinned_file(&canonical, &store_root, true, true, true)
    }

    fn open_pinned_file(
        path: &Path,
        store_root: &Path,
        require_root_owned: bool,
        require_non_writable: bool,
        require_elf: bool,
    ) -> Result<PinnedExecutable, String> {
        let metadata = std::fs::metadata(path)
            .map_err(|error| format!("stat executable: {error}"))?;
        if !metadata.is_file() {
            return Err("checkquote executable is not a regular file".into());
        }
        let root_owned = metadata.uid() == 0;
        let non_writable = metadata.mode() & 0o222 == 0;
        if require_root_owned && !root_owned {
            return Err("checkquote executable is not root-owned".into());
        }
        if require_non_writable && !non_writable {
            return Err("checkquote executable has writable permission bits".into());
        }
        if metadata.mode() & 0o111 == 0 {
            return Err("checkquote executable has no execute permission bit".into());
        }

        let mut file = File::open(path).map_err(|error| format!("open executable: {error}"))?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|error| format!("read ELF magic: {error}"))?;
        let elf_magic_valid = magic == [0x7f, b'E', b'L', b'F'];
        if require_elf && !elf_magic_valid {
            return Err("checkquote executable is not an ELF binary".into());
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|error| format!("rewind executable before hashing: {error}"))?;
        let mut hasher = blake3::Hasher::new();
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|error| format!("read executable while hashing: {error}"))?;
            if read == 0 {
                break;
            }
            hasher.update(&buffer[..read]);
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|error| format!("rewind executable after hashing: {error}"))?;

        let canonical_path = path
            .to_str()
            .ok_or_else(|| "checkquote path is not valid UTF-8".to_string())?
            .to_string();
        let nix_store_root = store_root
            .to_str()
            .ok_or_else(|| "Nix store root is not valid UTF-8".to_string())?
            .to_string();
        let record = FdPinnedExecutionRecord {
            execution_mode: FD_PINNED_EXECUTION_MODE_V1.into(),
            canonical_path,
            nix_store_root,
            executable_blake3: format!("blake3:{}", hasher.finalize().to_hex()),
            device: metadata.dev(),
            inode: metadata.ino(),
            file_size: metadata.len(),
            root_owned,
            non_writable,
            elf_magic_valid,
        };
        Ok(PinnedExecutable { file, record })
    }

    fn direct_nix_store_root(path: &Path) -> Option<PathBuf> {
        let text = path.to_str()?;
        let suffix = text.strip_prefix("/nix/store/")?;
        let component = suffix.split('/').next()?;
        if !valid_nix_store_component(component) || suffix == component {
            return None;
        }
        Some(PathBuf::from(format!("/nix/store/{component}")))
    }

    fn valid_nix_store_component(component: &str) -> bool {
        let bytes = component.as_bytes();
        bytes.len() > 33
            && bytes[32] == b'-'
            && bytes[..32]
                .iter()
                .all(|byte| NIX_BASE32.contains(byte))
            && component[33..].bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.' | b'_' | b'?' | b'=')
            })
    }

    struct SecureTempWorkspace {
        path: PathBuf,
    }

    impl SecureTempWorkspace {
        fn create() -> Result<Self, String> {
            static COUNTER: AtomicU64 = AtomicU64::new(1);
            let base = std::env::temp_dir();
            for _ in 0..32 {
                let nonce = COUNTER.fetch_add(1, Ordering::Relaxed);
                let now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos();
                let path = base.join(format!(
                    ".symthaea-fd-pinned-checkquote-{}-{now}-{nonce}",
                    std::process::id()
                ));
                let mut builder = std::fs::DirBuilder::new();
                builder.mode(0o700);
                match builder.create(&path) {
                    Ok(()) => return Ok(Self { path }),
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                    Err(error) => return Err(format!("create private temp dir: {error}")),
                }
            }
            Err("could not create unique private temp dir".into())
        }

        fn write(&self, name: &str, bytes: &[u8]) -> Result<PathBuf, String> {
            let path = self.path.join(name);
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .open(&path)
                .map_err(|error| format!("create private temp file: {error}"))?;
            file.write_all(bytes)
                .map_err(|error| format!("write private temp file: {error}"))?;
            file.sync_all()
                .map_err(|error| format!("sync private temp file: {error}"))?;
            Ok(path)
        }
    }

    impl Drop for SecureTempWorkspace {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    pub fn verify_fd_pinned_tpm2_quote(
        fd_policy: &FdPinnedCheckquotePolicy,
        inner_policy: &Tpm2CheckquotePolicy,
        challenge: &AttestationChallenge,
        ak: &AttestationKeyBinding,
        bundle: &RawTpm2QuoteBundle,
        verified_at_ms: u64,
    ) -> Result<FdPinnedTpm2QuoteQualification, FdPinnedCheckquoteReport> {
        let policy_digest = fd_policy.canonical_digest();
        let inner_policy_digest = inner_policy.canonical_digest();
        let mut report = FdPinnedCheckquoteReport {
            schema_version: FD_PINNED_CHECKQUOTE_REPORT_SCHEMA_V1.into(),
            policy_id: fd_policy.policy_id.clone(),
            policy_digest: policy_digest.clone(),
            inner_policy_digest: inner_policy_digest.clone(),
            inner_quote_report_digest: None,
            inner_quote_qualification_digest: None,
            execution_record_digest: None,
            executable_digest: None,
            canonical_path: None,
            nix_store_root: None,
            disposition: FdPinnedCheckquoteDisposition::Invalid,
            issues: Vec::new(),
        };

        if !fd_policy.validate() {
            report.issues.push(FdPinnedCheckquoteIssue::InvalidPolicy);
            return Err(report);
        }
        let Some(inner_policy_digest) = inner_policy_digest else {
            report.issues.push(FdPinnedCheckquoteIssue::InvalidInnerPolicy);
            return Err(report);
        };
        if inner_policy_digest != fd_policy.expected_inner_policy_digest {
            report.issues.push(FdPinnedCheckquoteIssue::InnerPolicyMismatch);
            return Err(report);
        }

        let executor = FdPinnedCheckquoteExecutor::new(inner_policy.checkquote_path.clone());
        let quote = match verify_tpm2_quote(
            inner_policy,
            challenge,
            ak,
            bundle,
            verified_at_ms,
            &executor,
        ) {
            Ok(value) => value,
            Err(inner_report) => {
                let inner_report_digest = inner_report.canonical_digest();
                let inner_disposition = format!("{:?}", inner_report.disposition);
                report.inner_quote_report_digest = Some(inner_report_digest.clone());
                report.disposition = if inner_report.disposition == Tpm2CheckquoteDisposition::Invalid {
                    FdPinnedCheckquoteDisposition::Invalid
                } else {
                    FdPinnedCheckquoteDisposition::Blocked
                };
                report.issues.push(FdPinnedCheckquoteIssue::InnerQuoteRejected {
                    inner_report_digest,
                    inner_disposition,
                });
                return Err(report);
            }
        };

        let execution_record = match executor.last_execution_record() {
            Ok(Some(record)) => record,
            _ => {
                report.issues.push(FdPinnedCheckquoteIssue::MissingExecutionRecord);
                return Err(report);
            }
        };
        let execution_record_digest = execution_record.canonical_digest();
        report.inner_quote_report_digest = Some(quote.report.canonical_digest());
        report.inner_quote_qualification_digest =
            Some(quote.verified().qualification_digest().to_string());
        report.execution_record_digest = Some(execution_record_digest.clone());
        report.executable_digest = Some(execution_record.executable_blake3.clone());
        report.canonical_path = Some(execution_record.canonical_path.clone());
        report.nix_store_root = Some(execution_record.nix_store_root.clone());

        if execution_record.canonical_path != inner_policy.checkquote_path
            || quote.verification_receipt.verification_tool_ref != execution_record.canonical_path
        {
            report.issues.push(FdPinnedCheckquoteIssue::ExecutionPathMismatch);
            return Err(report);
        }
        if execution_record.executable_blake3 != inner_policy.expected_checkquote_blake3 {
            report.issues.push(FdPinnedCheckquoteIssue::ExecutionDigestMismatch);
            return Err(report);
        }
        if quote.verification_receipt.verification_tool_digest != execution_record.executable_blake3 {
            report.issues.push(FdPinnedCheckquoteIssue::ExecutionReceiptMismatch);
            return Err(report);
        }

        report.disposition = FdPinnedCheckquoteDisposition::Qualified;
        let report_digest = report.canonical_digest();
        let policy_digest = policy_digest.expect("validated fd-pinned policy has digest");
        let qualification_digest = fd_pinned_qualification_digest(
            &report_digest,
            &policy_digest,
            &inner_policy_digest,
            quote.verified().qualification_digest(),
            &execution_record_digest,
        );
        let verified = VerifiedFdPinnedTpm2Quote {
            qualification_digest,
            report_digest,
            policy_digest,
            inner_policy_digest,
            inner_quote_qualification_digest: quote.verified().qualification_digest().to_string(),
            inner_quote_policy_digest: quote.verified().policy_digest().to_string(),
            challenge_digest: quote.verified().challenge_digest().to_string(),
            ak_binding_digest: quote.verified().ak_binding_digest().to_string(),
            quote_artifact_digest: quote.verified().quote_artifact_digest().to_string(),
            verification_receipt_digest: quote.verified().verification_receipt_digest().to_string(),
            pcr_selection_digest: quote.verified().pcr_selection_digest().to_string(),
            execution_record_digest,
            executable_digest: execution_record.executable_blake3.clone(),
            canonical_path: execution_record.canonical_path.clone(),
            nix_store_root: execution_record.nix_store_root.clone(),
            device: execution_record.device,
            inode: execution_record.inode,
            file_size: execution_record.file_size,
        };
        Ok(FdPinnedTpm2QuoteQualification {
            report,
            quote,
            execution_record,
            verified,
        })
    }

    fn fd_pinned_qualification_digest(
        report_digest: &str,
        policy_digest: &str,
        inner_policy_digest: &str,
        inner_quote_qualification_digest: &str,
        execution_record_digest: &str,
    ) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(QUALIFICATION_DIGEST_DOMAIN);
        for value in [
            report_digest,
            policy_digest,
            inner_policy_digest,
            inner_quote_qualification_digest,
            execution_record_digest,
        ] {
            push_field(&mut hasher, value);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::os::unix::fs::PermissionsExt;

        #[test]
        fn nix_store_component_is_strict() {
            assert!(valid_nix_store_component(
                "00000000000000000000000000000000-tpm2-tools-5.7"
            ));
            assert!(!valid_nix_store_component("short-tpm2-tools"));
            assert!(!valid_nix_store_component(
                "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee-tpm2-tools"
            ));
        }

        #[test]
        fn fd_launch_survives_original_path_replacement() {
            if std::env::var_os("SYMTHAEA_FD_PINNED_CHILD").is_some() {
                return;
            }
            let current = std::env::current_exe().expect("current test executable");
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "symthaea-fd-pinned-test-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir(&dir).expect("create test dir");
            let candidate = dir.join("candidate");
            std::fs::copy(&current, &candidate).expect("copy test executable");
            std::fs::set_permissions(&candidate, std::fs::Permissions::from_mode(0o755))
                .expect("set executable permissions");

            let pinned = open_pinned_file(&candidate, &dir, false, false, true)
                .expect("open test executable once");
            std::fs::remove_file(&candidate).expect("remove original pathname");
            std::fs::write(&candidate, b"replacement-not-an-elf")
                .expect("replace original pathname");
            std::fs::set_permissions(&candidate, std::fs::Permissions::from_mode(0o755))
                .expect("set replacement permissions");

            let args = vec![
                OsString::from("--exact"),
                OsString::from("linux::tests::fd_pinned_child_probe"),
                OsString::from("--nocapture"),
            ];
            let env = vec![(
                OsString::from("SYMTHAEA_FD_PINNED_CHILD"),
                OsString::from("1"),
            )];
            let output = execute_open_file(pinned.file, &args, &env)
                .expect("execute already-open original inode");
            assert!(output.status.success(), "child stderr: {}", String::from_utf8_lossy(&output.stderr));
            assert!(String::from_utf8_lossy(&output.stdout).contains("fd-pinned-original"));
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn fd_pinned_child_probe() {
            if std::env::var_os("SYMTHAEA_FD_PINNED_CHILD").is_some() {
                println!("fd-pinned-original");
            }
        }
    }
}

#[cfg(target_os = "linux")]
pub use linux::verify_fd_pinned_tpm2_quote;

#[cfg(not(target_os = "linux"))]
pub fn verify_fd_pinned_tpm2_quote(
    fd_policy: &FdPinnedCheckquotePolicy,
    _inner_policy: &Tpm2CheckquotePolicy,
    _challenge: &AttestationChallenge,
    _ak: &AttestationKeyBinding,
    _bundle: &RawTpm2QuoteBundle,
    _verified_at_ms: u64,
) -> Result<FdPinnedTpm2QuoteQualification, FdPinnedCheckquoteReport> {
    Err(FdPinnedCheckquoteReport {
        schema_version: FD_PINNED_CHECKQUOTE_REPORT_SCHEMA_V1.into(),
        policy_id: fd_policy.policy_id.clone(),
        policy_digest: fd_policy.canonical_digest(),
        inner_policy_digest: None,
        inner_quote_report_digest: None,
        inner_quote_qualification_digest: None,
        execution_record_digest: None,
        executable_digest: None,
        canonical_path: None,
        nix_store_root: None,
        disposition: FdPinnedCheckquoteDisposition::Blocked,
        issues: vec![FdPinnedCheckquoteIssue::UnsupportedPlatform],
    })
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_refs(values: &[String]) -> bool {
    !values.is_empty()
        && values.len() <= MAX_EVIDENCE_REFS
        && values.iter().all(|value| canonical_text(value))
        && values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

fn valid_blake3_digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64
            && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            && !digest.bytes().any(|byte| byte.is_ascii_uppercase())
    })
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    hasher.update(&(refs.len() as u64).to_le_bytes());
    for reference in refs {
        push_field(hasher, &reference);
    }
}
