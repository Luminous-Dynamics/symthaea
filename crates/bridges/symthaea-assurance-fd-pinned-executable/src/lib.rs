// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Reusable file-descriptor-pinned executable identity for Linux assurance tools.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Component, Path};

pub const FD_PINNED_EXECUTABLE_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.fd-pinned-executable-policy.v1";
pub const FD_PINNED_EXECUTION_IDENTITY_SCHEMA_V1: &str =
    "symthaea.assurance.fd-pinned-execution-identity.v1";
pub const FD_PINNED_INVOCATION_RECEIPT_SCHEMA_V1: &str =
    "symthaea.assurance.fd-pinned-invocation-receipt.v1";
pub const FD_PINNED_EXECUTION_MODE_V1: &str =
    "same-open-elf-hash-exec-postflight-proc-self-fd0-v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.fd-pinned-executable-policy.digest.v1\0";
const IDENTITY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.fd-pinned-execution-identity.digest.v1\0";
const ARGS_DIGEST_DOMAIN: &[u8] = b"symthaea.assurance.fd-pinned-args.digest.v1\0";
const ENV_DIGEST_DOMAIN: &[u8] = b"symthaea.assurance.fd-pinned-env.digest.v1\0";
const INVOCATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.fd-pinned-invocation.digest.v1\0";
const MAX_TEXT_BYTES: usize = 1024;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_EXECUTABLE_BYTES: u64 = 512 * 1024 * 1024;
const MAX_OUTPUT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_INVOCATIONS: u64 = 128;
const NIX_STORE_PREFIX: &str = "/nix/store/";
const NIX_BASE32: &[u8] = b"0123456789abcdfghijklmnpqrsvwxyz";
const FD_EXEC_PATH: &str = "/proc/self/fd/0";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FdPinnedExecutablePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub executable_path: String,
    pub expected_executable_blake3: String,
    pub max_executable_bytes: u64,
    pub max_output_bytes: u64,
    pub max_invocations: u64,
    pub evidence_refs: Vec<String>,
}

impl FdPinnedExecutablePolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == FD_PINNED_EXECUTABLE_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && direct_nix_store_root(&self.executable_path).is_some()
            && valid_blake3_digest(&self.expected_executable_blake3)
            && (1..=MAX_EXECUTABLE_BYTES).contains(&self.max_executable_bytes)
            && (1..=MAX_OUTPUT_BYTES).contains(&self.max_output_bytes)
            && (1..=MAX_INVOCATIONS).contains(&self.max_invocations)
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.executable_path.as_str(),
            self.expected_executable_blake3.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        for value in [
            self.max_executable_bytes,
            self.max_output_bytes,
            self.max_invocations,
        ] {
            hasher.update(&value.to_le_bytes());
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FdPinnedExecutionIdentity {
    pub schema_version: String,
    pub execution_mode: String,
    pub canonical_path: String,
    pub nix_store_root: String,
    pub executable_blake3: String,
    pub device: u64,
    pub inode: u64,
    pub file_size: u64,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub mtime_seconds: i64,
    pub mtime_nanoseconds: i64,
}

impl FdPinnedExecutionIdentity {
    pub fn canonical_digest(&self) -> Option<String> {
        if self.schema_version != FD_PINNED_EXECUTION_IDENTITY_SCHEMA_V1
            || self.execution_mode != FD_PINNED_EXECUTION_MODE_V1
            || !canonical_text(&self.canonical_path)
            || direct_nix_store_root(&self.canonical_path).as_deref()
                != Some(self.nix_store_root.as_str())
            || !valid_blake3_digest(&self.executable_blake3)
            || self.file_size == 0
            || self.uid != 0
            || self.mode & 0o222 != 0
            || self.mode & 0o111 == 0
        {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(IDENTITY_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.execution_mode.as_str(),
            self.canonical_path.as_str(),
            self.nix_store_root.as_str(),
            self.executable_blake3.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        hasher.update(&self.device.to_le_bytes());
        hasher.update(&self.inode.to_le_bytes());
        hasher.update(&self.file_size.to_le_bytes());
        hasher.update(&self.mode.to_le_bytes());
        hasher.update(&self.uid.to_le_bytes());
        hasher.update(&self.gid.to_le_bytes());
        hasher.update(&self.mtime_seconds.to_le_bytes());
        hasher.update(&self.mtime_nanoseconds.to_le_bytes());
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FdPinnedInvocationReceipt {
    pub schema_version: String,
    pub policy_digest: String,
    pub execution_identity_digest: String,
    pub invocation_sequence: u64,
    pub args_digest: String,
    pub environment_digest: String,
    pub exit_code: Option<i32>,
    pub stdout_digest: String,
    pub stdout_bytes: u64,
    pub stderr_digest: String,
    pub stderr_bytes: u64,
    pub postflight_executable_blake3: String,
}

impl FdPinnedInvocationReceipt {
    pub fn canonical_digest(&self) -> Option<String> {
        if self.schema_version != FD_PINNED_INVOCATION_RECEIPT_SCHEMA_V1
            || self.invocation_sequence == 0
            || [
                self.policy_digest.as_str(),
                self.execution_identity_digest.as_str(),
                self.args_digest.as_str(),
                self.environment_digest.as_str(),
                self.stdout_digest.as_str(),
                self.stderr_digest.as_str(),
                self.postflight_executable_blake3.as_str(),
            ]
            .iter()
            .any(|digest| !valid_blake3_digest(digest))
        {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(INVOCATION_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_digest.as_str(),
            self.execution_identity_digest.as_str(),
            self.args_digest.as_str(),
            self.environment_digest.as_str(),
            self.stdout_digest.as_str(),
            self.stderr_digest.as_str(),
            self.postflight_executable_blake3.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        hasher.update(&self.invocation_sequence.to_le_bytes());
        match self.exit_code {
            Some(code) => {
                hasher.update(&[1]);
                hasher.update(&code.to_le_bytes());
            }
            None => hasher.update(&[0]),
        }
        hasher.update(&self.stdout_bytes.to_le_bytes());
        hasher.update(&self.stderr_bytes.to_le_bytes());
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FdPinnedCommandOutput {
    pub exit_code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub receipt: FdPinnedInvocationReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FdPinnedExecutableError {
    UnsupportedPlatform,
    InvalidPolicy,
    PathNotCanonical,
    InvalidNixStorePath,
    StoreObjectNotRootOwned,
    StoreObjectWritable,
    ExecutableNotRegular,
    ExecutableNotRootOwned,
    ExecutableWritable,
    ExecutableNotRunnable,
    ExecutableNotElf,
    ExecutableSizeOutOfBounds,
    ExecutableDigestMismatch,
    ExecutableChanged,
    InvocationLimitExceeded,
    DuplicateEnvironmentKey,
    LaunchFailed(String),
    OutputTooLarge {
        stream: String,
        observed: u64,
        maximum: u64,
    },
    Io(String),
}

#[cfg(target_os = "linux")]
mod linux {
    use super::*;
    use std::ffi::OsString;
    use std::fs::File;
    use std::io::{Read, Seek, SeekFrom};
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::MetadataExt;
    use std::path::{Path, PathBuf};
    use std::process::{Command, Stdio};

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct FileSnapshot {
        device: u64,
        inode: u64,
        file_size: u64,
        mode: u32,
        uid: u32,
        gid: u32,
        mtime_seconds: i64,
        mtime_nanoseconds: i64,
    }

    impl FileSnapshot {
        fn from_metadata(metadata: &std::fs::Metadata) -> Self {
            Self {
                device: metadata.dev(),
                inode: metadata.ino(),
                file_size: metadata.len(),
                mode: metadata.mode(),
                uid: metadata.uid(),
                gid: metadata.gid(),
                mtime_seconds: metadata.mtime(),
                mtime_nanoseconds: metadata.mtime_nsec(),
            }
        }
    }

    struct OpenedExecutable {
        file: File,
        snapshot: FileSnapshot,
        digest: String,
        canonical_path: String,
        store_root: String,
    }

    pub struct FdPinnedExecutable {
        policy: FdPinnedExecutablePolicy,
        policy_digest: String,
        identity: FdPinnedExecutionIdentity,
        identity_digest: String,
        opened: OpenedExecutable,
        invocation_sequence: u64,
    }

    impl FdPinnedExecutable {
        pub fn open(policy: &FdPinnedExecutablePolicy) -> Result<Self, FdPinnedExecutableError> {
            let policy_digest = policy
                .canonical_digest()
                .ok_or(FdPinnedExecutableError::InvalidPolicy)?;
            let opened = open_production_executable(policy)?;
            if opened.digest != policy.expected_executable_blake3 {
                return Err(FdPinnedExecutableError::ExecutableDigestMismatch);
            }
            let identity = identity_from_opened(&opened);
            let identity_digest = identity
                .canonical_digest()
                .ok_or(FdPinnedExecutableError::InvalidPolicy)?;
            Ok(Self {
                policy: policy.clone(),
                policy_digest,
                identity,
                identity_digest,
                opened,
                invocation_sequence: 0,
            })
        }

        pub fn policy_digest(&self) -> &str {
            &self.policy_digest
        }

        pub fn identity(&self) -> &FdPinnedExecutionIdentity {
            &self.identity
        }

        pub fn identity_digest(&self) -> &str {
            &self.identity_digest
        }

        pub const fn executable_path_toctou_closed(&self) -> bool {
            true
        }

        pub const fn dependency_closure_atomically_pinned(&self) -> bool {
            false
        }

        pub const fn privileged_in_place_mutation_excluded(&self) -> bool {
            false
        }

        pub const fn trusted_time_established(&self) -> bool {
            false
        }

        pub const fn grants_physical_authority(&self) -> bool {
            false
        }

        pub fn execute(
            &mut self,
            args: &[OsString],
            environment: &[(OsString, OsString)],
        ) -> Result<FdPinnedCommandOutput, FdPinnedExecutableError> {
            let next = self
                .invocation_sequence
                .checked_add(1)
                .ok_or(FdPinnedExecutableError::InvocationLimitExceeded)?;
            if next > self.policy.max_invocations {
                return Err(FdPinnedExecutableError::InvocationLimitExceeded);
            }
            validate_environment(environment)?;
            verify_opened_unchanged(&mut self.opened, self.policy.max_executable_bytes)?;

            let child_file = self
                .opened
                .file
                .try_clone()
                .map_err(|error| FdPinnedExecutableError::Io(format!("clone executable fd: {error}")))?;
            let mut command = Command::new(FD_EXEC_PATH);
            command
                .args(args)
                .env_clear()
                .stdin(Stdio::from(child_file))
                .stdout(Stdio::piped())
                .stderr(Stdio::piped());
            for (key, value) in environment {
                command.env(key, value);
            }
            let output = command
                .output()
                .map_err(|error| FdPinnedExecutableError::LaunchFailed(error.to_string()))?;

            verify_opened_unchanged(&mut self.opened, self.policy.max_executable_bytes)?;
            for (stream, bytes) in [("stdout", &output.stdout), ("stderr", &output.stderr)] {
                if bytes.len() as u64 > self.policy.max_output_bytes {
                    return Err(FdPinnedExecutableError::OutputTooLarge {
                        stream: stream.into(),
                        observed: bytes.len() as u64,
                        maximum: self.policy.max_output_bytes,
                    });
                }
            }

            let receipt = FdPinnedInvocationReceipt {
                schema_version: FD_PINNED_INVOCATION_RECEIPT_SCHEMA_V1.into(),
                policy_digest: self.policy_digest.clone(),
                execution_identity_digest: self.identity_digest.clone(),
                invocation_sequence: next,
                args_digest: args_digest(args),
                environment_digest: environment_digest(environment),
                exit_code: output.status.code(),
                stdout_digest: digest_bytes(&output.stdout),
                stdout_bytes: output.stdout.len() as u64,
                stderr_digest: digest_bytes(&output.stderr),
                stderr_bytes: output.stderr.len() as u64,
                postflight_executable_blake3: self.opened.digest.clone(),
            };
            if receipt.canonical_digest().is_none() {
                return Err(FdPinnedExecutableError::Io(
                    "invocation receipt failed canonical validation".into(),
                ));
            }
            self.invocation_sequence = next;
            Ok(FdPinnedCommandOutput {
                exit_code: output.status.code(),
                stdout: output.stdout,
                stderr: output.stderr,
                receipt,
            })
        }
    }

    fn open_production_executable(
        policy: &FdPinnedExecutablePolicy,
    ) -> Result<OpenedExecutable, FdPinnedExecutableError> {
        let requested = PathBuf::from(&policy.executable_path);
        let canonical = std::fs::canonicalize(&requested)
            .map_err(|error| FdPinnedExecutableError::Io(format!("canonicalize executable: {error}")))?;
        if canonical != requested {
            return Err(FdPinnedExecutableError::PathNotCanonical);
        }
        let store_root = direct_nix_store_root(&policy.executable_path)
            .ok_or(FdPinnedExecutableError::InvalidNixStorePath)?;
        let store_metadata = std::fs::metadata(&store_root)
            .map_err(|error| FdPinnedExecutableError::Io(format!("stat store object: {error}")))?;
        if !store_metadata.is_dir() || store_metadata.uid() != 0 {
            return Err(FdPinnedExecutableError::StoreObjectNotRootOwned);
        }
        if store_metadata.mode() & 0o222 != 0 {
            return Err(FdPinnedExecutableError::StoreObjectWritable);
        }
        open_file(
            &canonical,
            &store_root,
            policy.max_executable_bytes,
            true,
            true,
        )
    }

    fn open_file(
        path: &Path,
        store_root: &str,
        maximum: u64,
        require_root: bool,
        require_non_writable: bool,
    ) -> Result<OpenedExecutable, FdPinnedExecutableError> {
        let mut file = File::open(path)
            .map_err(|error| FdPinnedExecutableError::Io(format!("open executable: {error}")))?;
        let metadata = file
            .metadata()
            .map_err(|error| FdPinnedExecutableError::Io(format!("stat open executable: {error}")))?;
        if !metadata.is_file() {
            return Err(FdPinnedExecutableError::ExecutableNotRegular);
        }
        if require_root && metadata.uid() != 0 {
            return Err(FdPinnedExecutableError::ExecutableNotRootOwned);
        }
        if require_non_writable && metadata.mode() & 0o222 != 0 {
            return Err(FdPinnedExecutableError::ExecutableWritable);
        }
        if metadata.mode() & 0o111 == 0 {
            return Err(FdPinnedExecutableError::ExecutableNotRunnable);
        }
        if metadata.len() == 0 || metadata.len() > maximum {
            return Err(FdPinnedExecutableError::ExecutableSizeOutOfBounds);
        }
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)
            .map_err(|error| FdPinnedExecutableError::Io(format!("read ELF magic: {error}")))?;
        if magic != [0x7f, b'E', b'L', b'F'] {
            return Err(FdPinnedExecutableError::ExecutableNotElf);
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|error| FdPinnedExecutableError::Io(format!("rewind executable: {error}")))?;
        let snapshot = FileSnapshot::from_metadata(&metadata);
        let digest = hash_open_file(&mut file, maximum)?;
        if FileSnapshot::from_metadata(
            &file
                .metadata()
                .map_err(|error| FdPinnedExecutableError::Io(format!("restat executable: {error}")))?,
        ) != snapshot
        {
            return Err(FdPinnedExecutableError::ExecutableChanged);
        }
        let canonical_path = path
            .to_str()
            .ok_or_else(|| FdPinnedExecutableError::Io("executable path is not UTF-8".into()))?
            .to_string();
        Ok(OpenedExecutable {
            file,
            snapshot,
            digest,
            canonical_path,
            store_root: store_root.to_string(),
        })
    }

    fn identity_from_opened(opened: &OpenedExecutable) -> FdPinnedExecutionIdentity {
        FdPinnedExecutionIdentity {
            schema_version: FD_PINNED_EXECUTION_IDENTITY_SCHEMA_V1.into(),
            execution_mode: FD_PINNED_EXECUTION_MODE_V1.into(),
            canonical_path: opened.canonical_path.clone(),
            nix_store_root: opened.store_root.clone(),
            executable_blake3: opened.digest.clone(),
            device: opened.snapshot.device,
            inode: opened.snapshot.inode,
            file_size: opened.snapshot.file_size,
            mode: opened.snapshot.mode,
            uid: opened.snapshot.uid,
            gid: opened.snapshot.gid,
            mtime_seconds: opened.snapshot.mtime_seconds,
            mtime_nanoseconds: opened.snapshot.mtime_nanoseconds,
        }
    }

    fn verify_opened_unchanged(
        opened: &mut OpenedExecutable,
        maximum: u64,
    ) -> Result<(), FdPinnedExecutableError> {
        let metadata = opened
            .file
            .metadata()
            .map_err(|error| FdPinnedExecutableError::Io(format!("stat retained executable: {error}")))?;
        if FileSnapshot::from_metadata(&metadata) != opened.snapshot {
            return Err(FdPinnedExecutableError::ExecutableChanged);
        }
        if hash_open_file(&mut opened.file, maximum)? != opened.digest {
            return Err(FdPinnedExecutableError::ExecutableChanged);
        }
        Ok(())
    }

    fn hash_open_file(file: &mut File, maximum: u64) -> Result<String, FdPinnedExecutableError> {
        file.seek(SeekFrom::Start(0))
            .map_err(|error| FdPinnedExecutableError::Io(format!("rewind for hash: {error}")))?;
        let mut hasher = blake3::Hasher::new();
        let mut buffer = [0u8; 64 * 1024];
        let mut total = 0u64;
        loop {
            let read = file
                .read(&mut buffer)
                .map_err(|error| FdPinnedExecutableError::Io(format!("read for hash: {error}")))?;
            if read == 0 {
                break;
            }
            total = total
                .checked_add(read as u64)
                .ok_or(FdPinnedExecutableError::ExecutableSizeOutOfBounds)?;
            if total > maximum {
                return Err(FdPinnedExecutableError::ExecutableSizeOutOfBounds);
            }
            hasher.update(&buffer[..read]);
        }
        file.seek(SeekFrom::Start(0))
            .map_err(|error| FdPinnedExecutableError::Io(format!("rewind after hash: {error}")))?;
        Ok(format!("blake3:{}", hasher.finalize().to_hex()))
    }

    fn validate_environment(
        environment: &[(OsString, OsString)],
    ) -> Result<(), FdPinnedExecutableError> {
        let mut keys = BTreeSet::new();
        for (key, _) in environment {
            if !keys.insert(key.as_os_str().as_bytes().to_vec()) {
                return Err(FdPinnedExecutableError::DuplicateEnvironmentKey);
            }
        }
        Ok(())
    }

    fn args_digest(args: &[OsString]) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(ARGS_DIGEST_DOMAIN);
        hasher.update(&(args.len() as u64).to_le_bytes());
        for arg in args {
            push_bytes(&mut hasher, arg.as_os_str().as_bytes());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    fn environment_digest(environment: &[(OsString, OsString)]) -> String {
        let mut fields = environment
            .iter()
            .map(|(key, value)| {
                (
                    key.as_os_str().as_bytes().to_vec(),
                    value.as_os_str().as_bytes().to_vec(),
                )
            })
            .collect::<Vec<_>>();
        fields.sort();
        let mut hasher = blake3::Hasher::new();
        hasher.update(ENV_DIGEST_DOMAIN);
        hasher.update(&(fields.len() as u64).to_le_bytes());
        for (key, value) in fields {
            push_bytes(&mut hasher, &key);
            push_bytes(&mut hasher, &value);
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::os::unix::fs::PermissionsExt;
        use std::time::{SystemTime, UNIX_EPOCH};

        #[test]
        fn retained_open_file_executes_after_path_replacement() {
            if std::env::var_os("SYMTHAEA_GENERIC_FD_PIN_CHILD").is_some() {
                return;
            }
            let current = std::env::current_exe().expect("current test executable");
            let unique = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let dir = std::env::temp_dir().join(format!(
                "symthaea-generic-fd-pin-{}-{unique}",
                std::process::id()
            ));
            std::fs::create_dir(&dir).expect("create test dir");
            let candidate = dir.join("candidate");
            std::fs::copy(&current, &candidate).expect("copy test executable");
            std::fs::set_permissions(&candidate, std::fs::Permissions::from_mode(0o755))
                .expect("chmod test executable");
            let mut opened = open_file(
                &candidate,
                "test:store-root",
                MAX_EXECUTABLE_BYTES,
                false,
                false,
            )
            .expect("open candidate once");
            let original_digest = opened.digest.clone();

            let old = dir.join("old");
            std::fs::rename(&candidate, &old).expect("move original pathname");
            std::fs::write(&candidate, b"replacement-not-elf").expect("replace pathname");
            std::fs::set_permissions(&candidate, std::fs::Permissions::from_mode(0o755))
                .expect("chmod replacement");

            let child_file = opened.file.try_clone().expect("clone retained fd");
            let output = Command::new(FD_EXEC_PATH)
                .arg("--exact")
                .arg("linux::tests::fd_pinned_child_probe")
                .arg("--nocapture")
                .env_clear()
                .env("SYMTHAEA_GENERIC_FD_PIN_CHILD", "1")
                .stdin(Stdio::from(child_file))
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .output()
                .expect("execute retained original");
            assert!(
                output.status.success(),
                "child stderr: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("generic-fd-pinned-original"));
            verify_opened_unchanged(&mut opened, MAX_EXECUTABLE_BYTES)
                .expect("retained inode/content unchanged");
            assert_eq!(opened.digest, original_digest);
            let _ = std::fs::remove_dir_all(dir);
        }

        #[test]
        fn fd_pinned_child_probe() {
            if std::env::var_os("SYMTHAEA_GENERIC_FD_PIN_CHILD").is_some() {
                println!("generic-fd-pinned-original");
            }
        }
    }

    pub use FdPinnedExecutable as PublicFdPinnedExecutable;
}

#[cfg(target_os = "linux")]
pub use linux::PublicFdPinnedExecutable as FdPinnedExecutable;

#[cfg(not(target_os = "linux"))]
pub struct FdPinnedExecutable;

#[cfg(not(target_os = "linux"))]
impl FdPinnedExecutable {
    pub fn open(_policy: &FdPinnedExecutablePolicy) -> Result<Self, FdPinnedExecutableError> {
        Err(FdPinnedExecutableError::UnsupportedPlatform)
    }
}

pub fn direct_nix_store_root(path: &str) -> Option<String> {
    if !path.starts_with(NIX_STORE_PREFIX) || path.len() > MAX_TEXT_BYTES {
        return None;
    }
    if Path::new(path)
        .components()
        .any(|component| !matches!(component, Component::RootDir | Component::Normal(_)))
    {
        return None;
    }
    let suffix = path.strip_prefix(NIX_STORE_PREFIX)?;
    let component = suffix.split('/').next()?;
    if suffix == component || !valid_nix_store_component(component) {
        return None;
    }
    Some(format!("{NIX_STORE_PREFIX}{component}"))
}

fn valid_nix_store_component(component: &str) -> bool {
    let bytes = component.as_bytes();
    bytes.len() > 33
        && bytes[32] == b'-'
        && bytes[..32].iter().all(|byte| NIX_BASE32.contains(byte))
        && component[33..].bytes().all(|byte| {
            byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'-' | b'.' | b'_' | b'?' | b'=')
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
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    push_bytes(hasher, value.as_bytes());
}

fn push_bytes(hasher: &mut blake3::Hasher, value: &[u8]) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value);
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    hasher.update(&(refs.len() as u64).to_le_bytes());
    for reference in refs {
        push_field(hasher, &reference);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn policy() -> FdPinnedExecutablePolicy {
        FdPinnedExecutablePolicy {
            schema_version: FD_PINNED_EXECUTABLE_POLICY_SCHEMA_V1.into(),
            policy_id: "fd-executable:nix:1".into(),
            executable_path:
                "/nix/store/00000000000000000000000000000000-nix-2.35/bin/nix".into(),
            expected_executable_blake3: d("nix"),
            max_executable_bytes: 128 * 1024 * 1024,
            max_output_bytes: 8 * 1024 * 1024,
            max_invocations: 4,
            evidence_refs: vec!["review:fd-executable".into()],
        }
    }

    #[test]
    fn policy_digest_binds_execution_limits_and_identity() {
        let first = policy().canonical_digest().unwrap();
        let mut changed = policy();
        changed.max_invocations += 1;
        assert_ne!(first, changed.canonical_digest().unwrap());
        let mut changed = policy();
        changed.expected_executable_blake3 = d("other");
        assert_ne!(first, changed.canonical_digest().unwrap());
    }

    #[test]
    fn evidence_ref_order_is_nonsemantic() {
        let mut left = policy();
        left.evidence_refs = vec!["review:a".into(), "review:b".into()];
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn direct_nix_store_path_is_strict() {
        assert_eq!(
            direct_nix_store_root(
                "/nix/store/00000000000000000000000000000000-nix-2.35/bin/nix"
            ),
            Some("/nix/store/00000000000000000000000000000000-nix-2.35".into())
        );
        assert!(direct_nix_store_root("/usr/bin/nix").is_none());
        assert!(direct_nix_store_root("/nix/store/short-nix/bin/nix").is_none());
        assert!(direct_nix_store_root(
            "/nix/store/00000000000000000000000000000000-nix/../other"
        )
        .is_none());
    }
}
