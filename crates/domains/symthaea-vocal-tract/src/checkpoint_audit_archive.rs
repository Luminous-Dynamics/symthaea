// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent retention receipts for checkpoint key-audit exports.
//!
//! The archive authority is intentionally separate from the live audit-log key.
//! A receipt proves that one exact authenticated export artifact was accepted by
//! a named external repository. It does not itself delete or compact live data.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use zeroize::Zeroize;

/// Whether an audit-log export's artifact has actually reached durable storage at its
/// destination repository yet, or is still in flight. `Synced` is the only value a receipt
/// may legitimately be sealed against (see `CheckpointAuditArchiveAuthority::seal_receipt`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointAuditExportDurability {
    Pending,
    Synced,
}

/// A local, unsigned claim from the key-audit-export process that one export of the audit log
/// completed. This is the INPUT to [`CheckpointAuditArchiveAuthority::seal_receipt`], which
/// turns it into an authenticated, retained archive receipt -- it carries no signature of its
/// own, which is why `seal_receipt` independently re-derives everything it needs to trust
/// rather than accepting these fields as already-authoritative.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointKeyAuditExportReceipt {
    pub export_id: [u8; 16],
    pub record_count: u64,
    pub head_record_digest: [u8; 32],
    pub artifact_digest: [u8; 32],
    pub artifact_bytes: u64,
    pub durability: CheckpointAuditExportDurability,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointAuditError {
    InvalidKey,
    EntropyUnavailable,
    InvalidChain,
    Encoding,
    TooLarge,
    AuthenticationFailed,
    UnsafeFilesystemObject,
    Unavailable(&'static str),
    Io(std::io::ErrorKind),
}

impl From<std::io::Error> for CheckpointAuditError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error.kind())
    }
}

impl std::fmt::Display for CheckpointAuditError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKey => f.write_str("invalid checkpoint audit archive key"),
            Self::EntropyUnavailable => f.write_str("checkpoint audit archive entropy unavailable"),
            Self::InvalidChain => f.write_str("invalid checkpoint audit archive receipt chain"),
            Self::Encoding => f.write_str("checkpoint audit archive encoding failed"),
            Self::TooLarge => f.write_str("checkpoint audit archive artifact exceeds its bound"),
            Self::AuthenticationFailed => {
                f.write_str("checkpoint audit archive authentication failed")
            }
            Self::UnsafeFilesystemObject => {
                f.write_str("unsafe checkpoint audit archive filesystem object")
            }
            Self::Unavailable(reason) => {
                write!(f, "checkpoint audit archive unavailable: {reason}")
            }
            Self::Io(kind) => write!(f, "checkpoint audit archive I/O failed: {kind}"),
        }
    }
}

impl std::error::Error for CheckpointAuditError {}

pub const CHECKPOINT_AUDIT_ARCHIVE_RECEIPT_SCHEMA: &str =
    "symthaea.checkpoint-audit-archive-receipt.v1";
pub const CHECKPOINT_AUDIT_COMPACTION_PROOF_SCHEMA: &str =
    "symthaea.checkpoint-audit-compaction-proof.v1";
pub const CHECKPOINT_AUDIT_RETENTION_COMMITMENT_SCHEMA: &str =
    "symthaea.checkpoint-audit-retention-commitment.v1";
const ARCHIVE_RECEIPT_DOMAIN: &[u8] = b"symthaea-checkpoint-audit-archive-receipt-v1\0";
const RETENTION_COMMITMENT_DOMAIN: &[u8] = b"symthaea-checkpoint-audit-retention-commitment-v1\0";
const MAX_ARCHIVE_RECEIPT_BYTES: u64 = 64 * 1024;

pub struct CheckpointAuditArchiveKey([u8; 32]);

impl CheckpointAuditArchiveKey {
    pub fn new(bytes: [u8; 32]) -> Result<Self, CheckpointAuditError> {
        if bytes.iter().all(|byte| *byte == 0) {
            return Err(CheckpointAuditError::InvalidKey);
        }
        Ok(Self(bytes))
    }

    pub fn generate() -> Result<Self, CheckpointAuditError> {
        let mut bytes = [0u8; 32];
        getrandom::fill(&mut bytes).map_err(|_| CheckpointAuditError::EntropyUnavailable)?;
        let result = Self::new(bytes);
        bytes.zeroize();
        result
    }
}

impl Drop for CheckpointAuditArchiveKey {
    fn drop(&mut self) {
        self.0.zeroize();
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointAuditArchiveReceipt {
    pub schema: String,
    pub archive_id: [u8; 16],
    pub repository_binding: [u8; 32],
    pub retained_at_unix_seconds: u64,
    pub export_id: [u8; 16],
    pub export_artifact_digest: [u8; 32],
    pub export_head_record_digest: [u8; 32],
    pub export_record_count: u64,
    pub export_artifact_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointAuditArchiveReceiptWire {
    body: Vec<u8>,
    authentication_tag: [u8; 32],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CheckpointAuditRetentionRequirement {
    pub minimum_retained_until_unix_seconds: u64,
    pub minimum_replicas: u16,
    pub expected_storage_class_binding: [u8; 32],
}

impl CheckpointAuditRetentionRequirement {
    pub fn validate(&self) -> Result<(), CheckpointAuditError> {
        if self.minimum_retained_until_unix_seconds == 0
            || self.minimum_replicas == 0
            || self.minimum_replicas > 64
            || self.expected_storage_class_binding == [0u8; 32]
        {
            return Err(CheckpointAuditError::InvalidChain);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointAuditRetentionCommitment {
    pub schema: String,
    pub commitment_id: [u8; 16],
    pub archive_id: [u8; 16],
    pub repository_binding: [u8; 32],
    pub archive_receipt_digest: [u8; 32],
    pub export_artifact_digest: [u8; 32],
    pub committed_until_unix_seconds: u64,
    pub minimum_replicas: u16,
    pub storage_class_binding: [u8; 32],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CheckpointAuditRetentionCommitmentWire {
    body: Vec<u8>,
    authentication_tag: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointAuditCompactionProof {
    pub schema: String,
    pub archive_id: [u8; 16],
    pub repository_binding: [u8; 32],
    pub export_id: [u8; 16],
    pub export_artifact_digest: [u8; 32],
    pub live_record_count: u64,
    pub live_log_bytes: u64,
    pub live_head_record_digest: [u8; 32],
    pub independently_retained: bool,
    pub destructive_compaction_permitted: bool,
}

pub struct CheckpointAuditArchiveAuthority {
    key: CheckpointAuditArchiveKey,
}

impl CheckpointAuditArchiveAuthority {
    pub fn new(key: CheckpointAuditArchiveKey) -> Self {
        Self { key }
    }

    pub fn seal_receipt(
        &self,
        export: &CheckpointKeyAuditExportReceipt,
        archive_id: [u8; 16],
        repository_binding: [u8; 32],
        retained_at_unix_seconds: u64,
    ) -> Result<Vec<u8>, CheckpointAuditError> {
        if archive_id == [0u8; 16]
            || repository_binding == [0u8; 32]
            || retained_at_unix_seconds == 0
            || export.record_count == 0
            || export.artifact_digest == [0u8; 32]
            || export.head_record_digest == [0u8; 32]
            || export.artifact_bytes == 0
            || export.durability != CheckpointAuditExportDurability::Synced
        {
            return Err(CheckpointAuditError::InvalidChain);
        }
        let receipt = CheckpointAuditArchiveReceipt {
            schema: CHECKPOINT_AUDIT_ARCHIVE_RECEIPT_SCHEMA.to_owned(),
            archive_id,
            repository_binding,
            retained_at_unix_seconds,
            export_id: export.export_id,
            export_artifact_digest: export.artifact_digest,
            export_head_record_digest: export.head_record_digest,
            export_record_count: export.record_count,
            export_artifact_bytes: export.artifact_bytes,
        };
        let body = postcard::to_stdvec(&receipt).map_err(|_| CheckpointAuditError::Encoding)?;
        let wire = CheckpointAuditArchiveReceiptWire {
            authentication_tag: authenticate_receipt(&body, &self.key),
            body,
        };
        let encoded = postcard::to_stdvec(&wire).map_err(|_| CheckpointAuditError::Encoding)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_ARCHIVE_RECEIPT_BYTES {
            return Err(CheckpointAuditError::TooLarge);
        }
        Ok(encoded)
    }

    pub fn open_receipt(
        &self,
        encoded: &[u8],
        expected_repository_binding: [u8; 32],
    ) -> Result<CheckpointAuditArchiveReceipt, CheckpointAuditError> {
        if encoded.is_empty() || encoded.len() as u64 > MAX_ARCHIVE_RECEIPT_BYTES {
            return Err(CheckpointAuditError::TooLarge);
        }
        let wire: CheckpointAuditArchiveReceiptWire =
            postcard::from_bytes(encoded).map_err(|_| CheckpointAuditError::Encoding)?;
        if !constant_time_equal(
            &wire.authentication_tag,
            &authenticate_receipt(&wire.body, &self.key),
        ) {
            return Err(CheckpointAuditError::AuthenticationFailed);
        }
        let receipt: CheckpointAuditArchiveReceipt =
            postcard::from_bytes(&wire.body).map_err(|_| CheckpointAuditError::Encoding)?;
        validate_receipt(&receipt, expected_repository_binding)?;
        Ok(receipt)
    }

    pub fn write_receipt_no_overwrite(
        &self,
        target: impl AsRef<Path>,
        encoded: &[u8],
    ) -> Result<(), CheckpointAuditError> {
        self.open_receipt(encoded, receipt_repository_binding(encoded, &self.key)?)?;
        write_no_overwrite_atomic(target.as_ref(), encoded)
    }

    pub fn verify_receipt_file(
        &self,
        source: impl AsRef<Path>,
        expected_repository_binding: [u8; 32],
    ) -> Result<CheckpointAuditArchiveReceipt, CheckpointAuditError> {
        let encoded = read_bounded_regular_file(source.as_ref(), MAX_ARCHIVE_RECEIPT_BYTES)?;
        self.open_receipt(&encoded, expected_repository_binding)
    }

    pub fn seal_retention_commitment(
        &self,
        encoded_archive_receipt: &[u8],
        expected_repository_binding: [u8; 32],
        commitment_id: [u8; 16],
        committed_until_unix_seconds: u64,
        minimum_replicas: u16,
        storage_class_binding: [u8; 32],
    ) -> Result<Vec<u8>, CheckpointAuditError> {
        let receipt = self.open_receipt(encoded_archive_receipt, expected_repository_binding)?;
        if commitment_id == [0u8; 16]
            || committed_until_unix_seconds <= receipt.retained_at_unix_seconds
            || minimum_replicas == 0
            || minimum_replicas > 64
            || storage_class_binding == [0u8; 32]
        {
            return Err(CheckpointAuditError::InvalidChain);
        }
        let commitment = CheckpointAuditRetentionCommitment {
            schema: CHECKPOINT_AUDIT_RETENTION_COMMITMENT_SCHEMA.to_owned(),
            commitment_id,
            archive_id: receipt.archive_id,
            repository_binding: receipt.repository_binding,
            archive_receipt_digest: *blake3::hash(encoded_archive_receipt).as_bytes(),
            export_artifact_digest: receipt.export_artifact_digest,
            committed_until_unix_seconds,
            minimum_replicas,
            storage_class_binding,
        };
        let body = postcard::to_stdvec(&commitment).map_err(|_| CheckpointAuditError::Encoding)?;
        let wire = CheckpointAuditRetentionCommitmentWire {
            authentication_tag: authenticate_retention_commitment(&body, &self.key),
            body,
        };
        let encoded = postcard::to_stdvec(&wire).map_err(|_| CheckpointAuditError::Encoding)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_ARCHIVE_RECEIPT_BYTES {
            return Err(CheckpointAuditError::TooLarge);
        }
        Ok(encoded)
    }

    pub fn open_retention_commitment(
        &self,
        encoded: &[u8],
        encoded_archive_receipt: &[u8],
        expected_repository_binding: [u8; 32],
        requirement: CheckpointAuditRetentionRequirement,
    ) -> Result<CheckpointAuditRetentionCommitment, CheckpointAuditError> {
        requirement.validate()?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_ARCHIVE_RECEIPT_BYTES {
            return Err(CheckpointAuditError::TooLarge);
        }
        let receipt = self.open_receipt(encoded_archive_receipt, expected_repository_binding)?;
        let wire: CheckpointAuditRetentionCommitmentWire =
            postcard::from_bytes(encoded).map_err(|_| CheckpointAuditError::Encoding)?;
        if !constant_time_equal(
            &wire.authentication_tag,
            &authenticate_retention_commitment(&wire.body, &self.key),
        ) {
            return Err(CheckpointAuditError::AuthenticationFailed);
        }
        let commitment: CheckpointAuditRetentionCommitment =
            postcard::from_bytes(&wire.body).map_err(|_| CheckpointAuditError::Encoding)?;
        if commitment.schema != CHECKPOINT_AUDIT_RETENTION_COMMITMENT_SCHEMA
            || commitment.commitment_id == [0u8; 16]
            || commitment.archive_id != receipt.archive_id
            || commitment.repository_binding != expected_repository_binding
            || commitment.archive_receipt_digest
                != *blake3::hash(encoded_archive_receipt).as_bytes()
            || commitment.export_artifact_digest != receipt.export_artifact_digest
            || commitment.committed_until_unix_seconds
                < requirement.minimum_retained_until_unix_seconds
            || commitment.minimum_replicas < requirement.minimum_replicas
            || commitment.storage_class_binding != requirement.expected_storage_class_binding
        {
            return Err(CheckpointAuditError::InvalidChain);
        }
        Ok(commitment)
    }

    pub fn write_retention_commitment_no_overwrite(
        &self,
        target: impl AsRef<Path>,
        encoded: &[u8],
        encoded_archive_receipt: &[u8],
        expected_repository_binding: [u8; 32],
        requirement: CheckpointAuditRetentionRequirement,
    ) -> Result<(), CheckpointAuditError> {
        self.open_retention_commitment(
            encoded,
            encoded_archive_receipt,
            expected_repository_binding,
            requirement,
        )?;
        write_no_overwrite_atomic(target.as_ref(), encoded)
    }

    pub fn verify_retention_commitment_file(
        &self,
        commitment_path: impl AsRef<Path>,
        archive_receipt_path: impl AsRef<Path>,
        expected_repository_binding: [u8; 32],
        requirement: CheckpointAuditRetentionRequirement,
    ) -> Result<CheckpointAuditRetentionCommitment, CheckpointAuditError> {
        let commitment =
            read_bounded_regular_file(commitment_path.as_ref(), MAX_ARCHIVE_RECEIPT_BYTES)?;
        let receipt =
            read_bounded_regular_file(archive_receipt_path.as_ref(), MAX_ARCHIVE_RECEIPT_BYTES)?;
        self.open_retention_commitment(
            &commitment,
            &receipt,
            expected_repository_binding,
            requirement,
        )
    }
}

fn validate_receipt(
    receipt: &CheckpointAuditArchiveReceipt,
    expected_repository_binding: [u8; 32],
) -> Result<(), CheckpointAuditError> {
    if receipt.schema != CHECKPOINT_AUDIT_ARCHIVE_RECEIPT_SCHEMA
        || receipt.archive_id == [0u8; 16]
        || receipt.repository_binding == [0u8; 32]
        || receipt.repository_binding != expected_repository_binding
        || receipt.retained_at_unix_seconds == 0
        || receipt.export_id == [0u8; 16]
        || receipt.export_artifact_digest == [0u8; 32]
        || receipt.export_head_record_digest == [0u8; 32]
        || receipt.export_record_count == 0
        || receipt.export_artifact_bytes == 0
    {
        return Err(CheckpointAuditError::InvalidChain);
    }
    Ok(())
}

fn receipt_repository_binding(
    encoded: &[u8],
    key: &CheckpointAuditArchiveKey,
) -> Result<[u8; 32], CheckpointAuditError> {
    let wire: CheckpointAuditArchiveReceiptWire =
        postcard::from_bytes(encoded).map_err(|_| CheckpointAuditError::Encoding)?;
    if !constant_time_equal(
        &wire.authentication_tag,
        &authenticate_receipt(&wire.body, key),
    ) {
        return Err(CheckpointAuditError::AuthenticationFailed);
    }
    let receipt: CheckpointAuditArchiveReceipt =
        postcard::from_bytes(&wire.body).map_err(|_| CheckpointAuditError::Encoding)?;
    Ok(receipt.repository_binding)
}

fn authenticate_receipt(body: &[u8], key: &CheckpointAuditArchiveKey) -> [u8; 32] {
    let mut input = Vec::with_capacity(ARCHIVE_RECEIPT_DOMAIN.len() + body.len());
    input.extend_from_slice(ARCHIVE_RECEIPT_DOMAIN);
    input.extend_from_slice(body);
    *blake3::keyed_hash(&key.0, &input).as_bytes()
}

fn authenticate_retention_commitment(body: &[u8], key: &CheckpointAuditArchiveKey) -> [u8; 32] {
    let mut input = Vec::with_capacity(RETENTION_COMMITMENT_DOMAIN.len() + body.len());
    input.extend_from_slice(RETENTION_COMMITMENT_DOMAIN);
    input.extend_from_slice(body);
    *blake3::keyed_hash(&key.0, &input).as_bytes()
}

fn constant_time_equal(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0u8, |difference, (left, right)| difference | (left ^ right))
        == 0
}

fn read_bounded_regular_file(path: &Path, maximum: u64) -> Result<Vec<u8>, CheckpointAuditError> {
    use std::os::unix::fs::OpenOptionsExt;
    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > maximum {
        return Err(CheckpointAuditError::UnsafeFilesystemObject);
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if bytes.is_empty() || bytes.len() as u64 > maximum {
        return Err(CheckpointAuditError::TooLarge);
    }
    Ok(bytes)
}

fn write_no_overwrite_atomic(path: &Path, bytes: &[u8]) -> Result<(), CheckpointAuditError> {
    use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
    let parent = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .ok_or(CheckpointAuditError::UnsafeFilesystemObject)?;
    fs::create_dir_all(parent)?;
    fs::set_permissions(parent, fs::Permissions::from_mode(0o700))?;
    let file_name = path
        .file_name()
        .filter(|name| !name.is_empty())
        .ok_or(CheckpointAuditError::UnsafeFilesystemObject)?;
    let mut directory_options = OpenOptions::new();
    directory_options
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let directory = directory_options.open(parent)?;
    let operation_parent = operation_parent(parent, &directory)?;
    let target = operation_parent.join(file_name);
    let mut nonce = [0u8; 16];
    getrandom::fill(&mut nonce).map_err(|_| CheckpointAuditError::EntropyUnavailable)?;
    let suffix = nonce
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let temp = operation_parent.join(format!(
        ".checkpoint-audit-receipt-{}-{suffix}.tmp",
        std::process::id(),
    ));
    let result = (|| {
        let mut options = OpenOptions::new();
        options
            .write(true)
            .create_new(true)
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let mut file = options.open(&temp)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::hard_link(&temp, &target)?;
        directory.sync_all()?;
        Ok::<(), CheckpointAuditError>(())
    })();
    let _ = fs::remove_file(&temp);
    result
}

fn operation_parent(parent: &Path, directory: &File) -> Result<PathBuf, CheckpointAuditError> {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        let _ = parent;
        let path = PathBuf::from(format!("/proc/self/fd/{}", directory.as_raw_fd()));
        if !path.is_dir() {
            return Err(CheckpointAuditError::Unavailable(
                "descriptor-relative archive receipt directory unavailable",
            ));
        }
        Ok(path)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = directory;
        Ok(parent.to_owned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn export_receipt() -> CheckpointKeyAuditExportReceipt {
        CheckpointKeyAuditExportReceipt {
            export_id: [0x11; 16],
            record_count: 4,
            head_record_digest: [0x22; 32],
            artifact_digest: [0x33; 32],
            artifact_bytes: 4096,
            durability: CheckpointAuditExportDurability::Synced,
        }
    }

    #[test]
    fn independent_archive_receipt_round_trips() {
        let authority = CheckpointAuditArchiveAuthority::new(
            CheckpointAuditArchiveKey::new([0x44; 32]).unwrap(),
        );
        let repository = [0x55; 32];
        let encoded = authority
            .seal_receipt(&export_receipt(), [0x66; 16], repository, 1_800_000_000)
            .unwrap();
        let receipt = authority.open_receipt(&encoded, repository).unwrap();
        assert_eq!(receipt.export_artifact_digest, [0x33; 32]);
        assert_eq!(receipt.export_record_count, 4);
    }

    #[test]
    fn wrong_repository_or_key_fails_closed() {
        let authority = CheckpointAuditArchiveAuthority::new(
            CheckpointAuditArchiveKey::new([0x47; 32]).unwrap(),
        );
        let encoded = authority
            .seal_receipt(&export_receipt(), [0x48; 16], [0x49; 32], 1_800_000_001)
            .unwrap();
        assert!(authority.open_receipt(&encoded, [0x50; 32]).is_err());
        let wrong = CheckpointAuditArchiveAuthority::new(
            CheckpointAuditArchiveKey::new([0x51; 32]).unwrap(),
        );
        assert!(wrong.open_receipt(&encoded, [0x49; 32]).is_err());
    }
}
