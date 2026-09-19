// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! MEL-003P1EILR-H: qualification-only durable idempotency fixture for the
//! provider-neutral enrollment-witness protocol.
//!
//! This is deliberately **not** production external-witness evidence. It gives
//! the executable qualification corpus two separately persisted service roles
//! whose accepted responses survive process loss and replay exactly under the
//! same P1EILR-C `request_sha256`.

use crate::evidence_digest::{
    canonical_json_bytes, canonical_json_sha256, decode_hex_32,
    perceptual_enrollment_orchestrator::{
        EnrollmentWitnessAdapterFailureKindV1, EnrollmentWitnessAdapterFailureV1,
        EnrollmentWitnessProviderAdapterV1,
    },
    perceptual_enrollment_witness::FrozenPerceptualEnrollmentWitnessPolicyV1,
    perceptual_enrollment_witness_provider::{
        collection_signing_request_commitment, enrollment_witness_signature_transcript_v1,
        witness_service_request_commitment, ExternalEnrollmentCollectionSignatureV1,
        ExternalEnrollmentCollectionSigningRequestV1,
        ExternalEnrollmentWitnessServiceRequestV1, ExternalEnrollmentWitnessServiceResponseV1,
        ENROLLMENT_WITNESS_EXTERNAL_SIGNATURE_DOMAIN_V1,
        ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION,
    },
};
use ed25519_dalek::{Signer, SigningKey};
use rand::{rngs::OsRng, RngCore};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

pub const QUALIFICATION_PROVIDER_JOURNAL_VERSION: &str =
    "mel003-enrollment-witness-qualification-provider-journal-v1";
const COLLECTION_ROLE: &str = "collection-signer";
const WITNESS_ROLE: &str = "independent-witness";
const LOCK_FILE_NAME: &str = ".mel003-provider-fixture.lock";
const MAX_ENTRY_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationProviderFaultPointV1 {
    /// Fail before any durable acceptance exists.
    BeforePersistUnavailable,
    /// Persist and fsync the accepted response, then simulate response loss.
    AfterPersistUncertain,
}

#[derive(Debug)]
pub enum QualificationProviderFixtureErrorV1 {
    UnsupportedPlatform,
    RootUnavailable,
    InvalidSigningSeed,
    SignerIdentityMismatch { role: &'static str },
    InvalidRequestIdentity,
    RequestCommitmentMismatch,
    IdempotencyConflict,
    JournalMalformed,
    JournalDigestMismatch,
    JournalTooLarge,
    ReadBackMismatch,
    Serialization,
    EntropyUnavailable,
    LocalLockPoisoned,
    KernelLockUnavailable,
    Io(std::io::Error),
}

impl std::fmt::Display for QualificationProviderFixtureErrorV1 {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedPlatform => write!(formatter, "provider fixture requires Linux"),
            Self::RootUnavailable => write!(formatter, "provider fixture root unavailable"),
            Self::InvalidSigningSeed => write!(formatter, "provider fixture signing seed invalid"),
            Self::SignerIdentityMismatch { role } => {
                write!(formatter, "provider fixture {role} identity does not match frozen policy")
            }
            Self::InvalidRequestIdentity => write!(formatter, "provider request identity invalid"),
            Self::RequestCommitmentMismatch => write!(formatter, "provider request commitment mismatch"),
            Self::IdempotencyConflict => write!(formatter, "same request id was replayed with different payload"),
            Self::JournalMalformed => write!(formatter, "provider fixture journal malformed"),
            Self::JournalDigestMismatch => write!(formatter, "provider fixture journal digest mismatch"),
            Self::JournalTooLarge => write!(formatter, "provider fixture journal entry exceeds bound"),
            Self::ReadBackMismatch => write!(formatter, "provider fixture durable read-back mismatch"),
            Self::Serialization => write!(formatter, "provider fixture serialization failed"),
            Self::EntropyUnavailable => write!(formatter, "provider fixture temporary-file entropy unavailable"),
            Self::LocalLockPoisoned => write!(formatter, "provider fixture local lock poisoned"),
            Self::KernelLockUnavailable => write!(formatter, "provider fixture kernel lock unavailable"),
            Self::Io(error) => write!(formatter, "provider fixture I/O failed: {error}"),
        }
    }
}

impl std::error::Error for QualificationProviderFixtureErrorV1 {}

impl From<std::io::Error> for QualificationProviderFixtureErrorV1 {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DurableProviderAcceptanceV1<Request, Response> {
    journal_version: String,
    service_role: String,
    request_sha256: String,
    request: Request,
    response: Response,
    entry_sha256: String,
}

fn acceptance_commitment<Request, Response>(
    entry: &DurableProviderAcceptanceV1<Request, Response>,
) -> Result<String, QualificationProviderFixtureErrorV1>
where
    Request: Clone + Serialize,
    Response: Clone + Serialize,
{
    let mut unsigned = entry.clone();
    unsigned.entry_sha256.clear();
    canonical_json_sha256(&unsigned).map_err(|_| QualificationProviderFixtureErrorV1::Serialization)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct QualificationWitnessStatementV1<'a> {
    witness_log_id: &'a str,
    sequence: u32,
    witness_head_sha256: &'a str,
    witness_anchor_reference: &'a str,
}

struct DurableProviderRoleJournalV1 {
    root: PathBuf,
    role: &'static str,
    local_lock: Mutex<()>,
    pinned_root: Mutex<Option<Arc<File>>>,
}

impl DurableProviderRoleJournalV1 {
    fn new(root: impl Into<PathBuf>, role: &'static str) -> Self {
        Self {
            root: root.into(),
            role,
            local_lock: Mutex::new(()),
            pinned_root: Mutex::new(None),
        }
    }

    fn accept<Request, Response, F>(
        &self,
        request_sha256: &str,
        request: &Request,
        create_response: F,
    ) -> Result<(Response, bool), QualificationProviderFixtureErrorV1>
    where
        Request: Clone + PartialEq + Serialize + DeserializeOwned,
        Response: Clone + PartialEq + Serialize + DeserializeOwned,
        F: FnOnce() -> Result<Response, QualificationProviderFixtureErrorV1>,
    {
        if decode_hex_32(request_sha256).is_none() {
            return Err(QualificationProviderFixtureErrorV1::InvalidRequestIdentity);
        }
        let _local = self
            .local_lock
            .lock()
            .map_err(|_| QualificationProviderFixtureErrorV1::LocalLockPoisoned)?;
        let lock_file = self.open_lock_file()?;
        let _kernel = KernelProviderLock::exclusive(&lock_file)?;
        let path = self.entry_path(request_sha256)?;
        if path.exists() {
            let entry: DurableProviderAcceptanceV1<Request, Response> =
                self.read_entry_locked(&path)?;
            self.validate_entry(&entry, request_sha256)?;
            if &entry.request != request {
                return Err(QualificationProviderFixtureErrorV1::IdempotencyConflict);
            }
            return Ok((entry.response, true));
        }

        let response = create_response()?;
        let mut entry = DurableProviderAcceptanceV1 {
            journal_version: QUALIFICATION_PROVIDER_JOURNAL_VERSION.into(),
            service_role: self.role.into(),
            request_sha256: request_sha256.into(),
            request: request.clone(),
            response: response.clone(),
            entry_sha256: String::new(),
        };
        entry.entry_sha256 = acceptance_commitment(&entry)?;
        self.write_entry_locked(&path, &entry)?;
        let read_back: DurableProviderAcceptanceV1<Request, Response> =
            self.read_entry_locked(&path)?;
        self.validate_entry(&read_back, request_sha256)?;
        if read_back != entry {
            return Err(QualificationProviderFixtureErrorV1::ReadBackMismatch);
        }
        Ok((response, false))
    }

    fn validate_entry<Request, Response>(
        &self,
        entry: &DurableProviderAcceptanceV1<Request, Response>,
        request_sha256: &str,
    ) -> Result<(), QualificationProviderFixtureErrorV1>
    where
        Request: Clone + Serialize,
        Response: Clone + Serialize,
    {
        if entry.journal_version != QUALIFICATION_PROVIDER_JOURNAL_VERSION
            || entry.service_role != self.role
            || entry.request_sha256 != request_sha256
        {
            return Err(QualificationProviderFixtureErrorV1::JournalMalformed);
        }
        if acceptance_commitment(entry)? != entry.entry_sha256 {
            return Err(QualificationProviderFixtureErrorV1::JournalDigestMismatch);
        }
        Ok(())
    }

    fn read_entry_locked<T: DeserializeOwned>(
        &self,
        path: &Path,
    ) -> Result<T, QualificationProviderFixtureErrorV1> {
        let file = open_private_regular_file(path, false, false)?;
        let metadata = file.metadata()?;
        if metadata.len() == 0 || metadata.len() > MAX_ENTRY_BYTES {
            return Err(QualificationProviderFixtureErrorV1::JournalTooLarge);
        }
        let mut encoded = Vec::with_capacity(metadata.len() as usize);
        file.take(MAX_ENTRY_BYTES.saturating_add(1))
            .read_to_end(&mut encoded)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_ENTRY_BYTES {
            return Err(QualificationProviderFixtureErrorV1::JournalTooLarge);
        }
        serde_json::from_slice(&encoded)
            .map_err(|_| QualificationProviderFixtureErrorV1::JournalMalformed)
    }

    fn write_entry_locked<T: Serialize>(
        &self,
        target: &Path,
        value: &T,
    ) -> Result<(), QualificationProviderFixtureErrorV1> {
        let encoded = canonical_json_bytes(value)
            .map_err(|_| QualificationProviderFixtureErrorV1::Serialization)?;
        if encoded.is_empty() || encoded.len() as u64 > MAX_ENTRY_BYTES {
            return Err(QualificationProviderFixtureErrorV1::JournalTooLarge);
        }
        let root = self.ensure_root()?;
        let operation_root = self.operation_root_path()?;
        let mut nonce = [0u8; 16];
        OsRng
            .try_fill_bytes(&mut nonce)
            .map_err(|_| QualificationProviderFixtureErrorV1::EntropyUnavailable)?;
        let suffix = nonce
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let temp = operation_root.join(format!(
            ".mel003-provider-{}-{}-{suffix}.tmp",
            self.role,
            std::process::id()
        ));
        let result = (|| {
            let mut file = open_private_regular_file(&temp, true, true)?;
            file.write_all(&encoded)?;
            file.sync_all()?;
            // First-write-wins: never replace an existing accepted operation.
            match fs::hard_link(&temp, target) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    return Err(QualificationProviderFixtureErrorV1::IdempotencyConflict);
                }
                Err(error) => return Err(error.into()),
            }
            root.sync_all()?;
            Ok::<(), QualificationProviderFixtureErrorV1>(())
        })();
        let _ = fs::remove_file(&temp);
        result
    }

    fn open_lock_file(&self) -> Result<File, QualificationProviderFixtureErrorV1> {
        let path = self.operation_root_path()?.join(LOCK_FILE_NAME);
        open_private_regular_file(&path, true, false).map_err(Into::into)
    }

    fn entry_path(&self, request_sha256: &str) -> Result<PathBuf, QualificationProviderFixtureErrorV1> {
        Ok(self
            .operation_root_path()?
            .join(format!("{request_sha256}.accepted.json")))
    }

    fn ensure_root(&self) -> Result<Arc<File>, QualificationProviderFixtureErrorV1> {
        #[cfg(not(target_os = "linux"))]
        {
            return Err(QualificationProviderFixtureErrorV1::UnsupportedPlatform);
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
            let mut pinned = self
                .pinned_root
                .lock()
                .map_err(|_| QualificationProviderFixtureErrorV1::LocalLockPoisoned)?;
            if let Some(root) = pinned.as_ref() {
                return Ok(Arc::clone(root));
            }
            fs::create_dir_all(&self.root)?;
            let metadata = fs::symlink_metadata(&self.root)?;
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(QualificationProviderFixtureErrorV1::RootUnavailable);
            }
            fs::set_permissions(&self.root, fs::Permissions::from_mode(0o700))?;
            let mut options = OpenOptions::new();
            options
                .read(true)
                .custom_flags(libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW);
            let root = Arc::new(options.open(&self.root)?);
            if !root.metadata()?.is_dir() {
                return Err(QualificationProviderFixtureErrorV1::RootUnavailable);
            }
            *pinned = Some(Arc::clone(&root));
            Ok(root)
        }
    }

    fn operation_root_path(&self) -> Result<PathBuf, QualificationProviderFixtureErrorV1> {
        #[cfg(not(target_os = "linux"))]
        {
            Err(QualificationProviderFixtureErrorV1::UnsupportedPlatform)
        }
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            let root = self.ensure_root()?;
            let path = PathBuf::from(format!("/proc/self/fd/{}", root.as_raw_fd()));
            if !path.is_dir() {
                return Err(QualificationProviderFixtureErrorV1::RootUnavailable);
            }
            Ok(path)
        }
    }
}

/// Local qualification provider with separately persisted signer/witness roles.
/// Private signing keys stay runtime-only; journals contain only frozen protocol
/// DTOs and responses.
pub struct DurableQualificationEnrollmentWitnessProviderV1 {
    collection_journal: DurableProviderRoleJournalV1,
    witness_journal: DurableProviderRoleJournalV1,
    collection_signing_key: SigningKey,
    witness_signing_key: SigningKey,
    collection_signer_id: String,
    collection_key_epoch: u64,
    witness_signer_id: String,
    witness_key_epoch: u64,
    collection_fault: Option<QualificationProviderFaultPointV1>,
    witness_fault: Option<QualificationProviderFaultPointV1>,
}

impl DurableQualificationEnrollmentWitnessProviderV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        collection_root: impl Into<PathBuf>,
        witness_root: impl Into<PathBuf>,
        policy: &FrozenPerceptualEnrollmentWitnessPolicyV1,
        collection_seed: [u8; 32],
        witness_seed: [u8; 32],
    ) -> Result<Self, QualificationProviderFixtureErrorV1> {
        if collection_seed == [0u8; 32] || witness_seed == [0u8; 32] {
            return Err(QualificationProviderFixtureErrorV1::InvalidSigningSeed);
        }
        let collection_signing_key = SigningKey::from_bytes(&collection_seed);
        let witness_signing_key = SigningKey::from_bytes(&witness_seed);
        if collection_signing_key.verifying_key().to_bytes().as_slice()
            != policy.collection_signer.verifying_key_bytes.as_slice()
        {
            return Err(QualificationProviderFixtureErrorV1::SignerIdentityMismatch {
                role: COLLECTION_ROLE,
            });
        }
        if witness_signing_key.verifying_key().to_bytes().as_slice()
            != policy.witness_signer.verifying_key_bytes.as_slice()
        {
            return Err(QualificationProviderFixtureErrorV1::SignerIdentityMismatch {
                role: WITNESS_ROLE,
            });
        }
        if policy.collection_signer.signer_id == policy.witness_signer.signer_id
            || policy.collection_signer.verifying_key_bytes == policy.witness_signer.verifying_key_bytes
        {
            return Err(QualificationProviderFixtureErrorV1::SignerIdentityMismatch {
                role: WITNESS_ROLE,
            });
        }
        Ok(Self {
            collection_journal: DurableProviderRoleJournalV1::new(collection_root, COLLECTION_ROLE),
            witness_journal: DurableProviderRoleJournalV1::new(witness_root, WITNESS_ROLE),
            collection_signing_key,
            witness_signing_key,
            collection_signer_id: policy.collection_signer.signer_id.clone(),
            collection_key_epoch: policy.collection_signer.key_epoch,
            witness_signer_id: policy.witness_signer.signer_id.clone(),
            witness_key_epoch: policy.witness_signer.key_epoch,
            collection_fault: None,
            witness_fault: None,
        })
    }

    pub fn inject_collection_fault(&mut self, fault: QualificationProviderFaultPointV1) {
        self.collection_fault = Some(fault);
    }

    pub fn inject_witness_fault(&mut self, fault: QualificationProviderFaultPointV1) {
        self.witness_fault = Some(fault);
    }

    fn map_fixture_error(
        error: QualificationProviderFixtureErrorV1,
    ) -> EnrollmentWitnessAdapterFailureV1 {
        let kind = match error {
            QualificationProviderFixtureErrorV1::IdempotencyConflict
            | QualificationProviderFixtureErrorV1::InvalidRequestIdentity
            | QualificationProviderFixtureErrorV1::RequestCommitmentMismatch
            | QualificationProviderFixtureErrorV1::JournalMalformed
            | QualificationProviderFixtureErrorV1::JournalDigestMismatch
            | QualificationProviderFixtureErrorV1::SignerIdentityMismatch { .. } => {
                EnrollmentWitnessAdapterFailureKindV1::Rejected
            }
            _ => EnrollmentWitnessAdapterFailureKindV1::Unavailable,
        };
        EnrollmentWitnessAdapterFailureV1::with_diagnostic_code(kind, error.to_string())
    }

    fn deterministic_anchor(request_sha256: &str) -> String {
        format!("mel003-local-qualification-anchor-v1:{request_sha256}")
    }
}

impl EnrollmentWitnessProviderAdapterV1 for DurableQualificationEnrollmentWitnessProviderV1 {
    fn sign_collection(
        &mut self,
        request: &ExternalEnrollmentCollectionSigningRequestV1,
    ) -> Result<ExternalEnrollmentCollectionSignatureV1, EnrollmentWitnessAdapterFailureV1> {
        let fault = self.collection_fault.take();
        if fault == Some(QualificationProviderFaultPointV1::BeforePersistUnavailable) {
            return Err(EnrollmentWitnessAdapterFailureV1::new(
                EnrollmentWitnessAdapterFailureKindV1::Unavailable,
            ));
        }
        if request.protocol_version != ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION
            || request.signer_id != self.collection_signer_id
            || request.key_epoch != self.collection_key_epoch
            || decode_hex_32(&request.request_sha256).is_none()
            || collection_signing_request_commitment(request)
                .map(|value| value != request.collection_request_sha256)
                .unwrap_or(true)
        {
            return Err(EnrollmentWitnessAdapterFailureV1::new(
                EnrollmentWitnessAdapterFailureKindV1::Rejected,
            ));
        }

        let key = self.collection_signing_key.clone();
        let signer_id = self.collection_signer_id.clone();
        let key_epoch = self.collection_key_epoch;
        let (response, replayed) = self
            .collection_journal
            .accept(&request.request_sha256, request, || {
                Ok(ExternalEnrollmentCollectionSignatureV1 {
                    protocol_version: ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION.into(),
                    request_sha256: request.request_sha256.clone(),
                    witness_head_sha256: request.witness_head_sha256.clone(),
                    collection_request_sha256: request.collection_request_sha256.clone(),
                    signer_id,
                    key_epoch,
                    signature: key.sign(&request.signing_transcript).to_bytes().to_vec(),
                })
            })
            .map_err(Self::map_fixture_error)?;
        if !replayed && fault == Some(QualificationProviderFaultPointV1::AfterPersistUncertain) {
            return Err(EnrollmentWitnessAdapterFailureV1::new(
                EnrollmentWitnessAdapterFailureKindV1::Uncertain,
            ));
        }
        Ok(response)
    }

    fn witness_enrollment(
        &mut self,
        request: &ExternalEnrollmentWitnessServiceRequestV1,
    ) -> Result<ExternalEnrollmentWitnessServiceResponseV1, EnrollmentWitnessAdapterFailureV1> {
        let fault = self.witness_fault.take();
        if fault == Some(QualificationProviderFaultPointV1::BeforePersistUnavailable) {
            return Err(EnrollmentWitnessAdapterFailureV1::new(
                EnrollmentWitnessAdapterFailureKindV1::Unavailable,
            ));
        }
        if request.protocol_version != ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION
            || decode_hex_32(&request.request_sha256).is_none()
            || witness_service_request_commitment(request)
                .map(|value| value != request.witness_service_request_sha256)
                .unwrap_or(true)
        {
            return Err(EnrollmentWitnessAdapterFailureV1::new(
                EnrollmentWitnessAdapterFailureKindV1::Rejected,
            ));
        }

        let key = self.witness_signing_key.clone();
        let signer_id = self.witness_signer_id.clone();
        let key_epoch = self.witness_key_epoch;
        let anchor = Self::deterministic_anchor(&request.request_sha256);
        let statement = QualificationWitnessStatementV1 {
            witness_log_id: &request.witness_log_id,
            sequence: request.sequence,
            witness_head_sha256: &request.witness_head_sha256,
            witness_anchor_reference: &anchor,
        };
        let message = canonical_json_bytes(&statement).map_err(|_| {
            EnrollmentWitnessAdapterFailureV1::new(
                EnrollmentWitnessAdapterFailureKindV1::Rejected,
            )
        })?;
        let transcript = enrollment_witness_signature_transcript_v1(
            ENROLLMENT_WITNESS_EXTERNAL_SIGNATURE_DOMAIN_V1,
            &message,
        )
        .map_err(|_| {
            EnrollmentWitnessAdapterFailureV1::new(
                EnrollmentWitnessAdapterFailureKindV1::Rejected,
            )
        })?;
        let (response, replayed) = self
            .witness_journal
            .accept(&request.request_sha256, request, || {
                Ok(ExternalEnrollmentWitnessServiceResponseV1 {
                    protocol_version: ENROLLMENT_WITNESS_PROVIDER_PROTOCOL_VERSION.into(),
                    request_sha256: request.request_sha256.clone(),
                    witness_head_sha256: request.witness_head_sha256.clone(),
                    witness_service_request_sha256: request
                        .witness_service_request_sha256
                        .clone(),
                    witness_anchor_reference: anchor,
                    witness_signer_id: signer_id,
                    witness_key_epoch: key_epoch,
                    witness_signature: key.sign(&transcript).to_bytes().to_vec(),
                })
            })
            .map_err(Self::map_fixture_error)?;
        if !replayed && fault == Some(QualificationProviderFaultPointV1::AfterPersistUncertain) {
            return Err(EnrollmentWitnessAdapterFailureV1::new(
                EnrollmentWitnessAdapterFailureKindV1::Uncertain,
            ));
        }
        Ok(response)
    }
}

fn open_private_regular_file(path: &Path, create: bool, create_new: bool) -> std::io::Result<File> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
        let mut options = OpenOptions::new();
        options.read(true).write(create || create_new);
        if create_new {
            options.create_new(true);
        } else if create {
            options.create(true);
        }
        options
            .mode(0o600)
            .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
        let file = options.open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.permissions().mode() & 0o077 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "provider fixture state/lock must be a private regular file",
            ));
        }
        Ok(file)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = (path, create, create_new);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "provider fixture requires Linux",
        ))
    }
}

struct KernelProviderLock<'a> {
    file: &'a File,
}

impl<'a> KernelProviderLock<'a> {
    fn exclusive(file: &'a File) -> Result<Self, QualificationProviderFixtureErrorV1> {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: `file` owns a live descriptor for this guard's lifetime.
            let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
            if result != 0 {
                return Err(QualificationProviderFixtureErrorV1::KernelLockUnavailable);
            }
            Ok(Self { file })
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = file;
            Err(QualificationProviderFixtureErrorV1::UnsupportedPlatform)
        }
    }
}

impl Drop for KernelProviderLock<'_> {
    fn drop(&mut self) {
        #[cfg(target_os = "linux")]
        {
            use std::os::fd::AsRawFd;
            // SAFETY: unlock the same live descriptor retained by this guard.
            let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stable_anchor_depends_only_on_request_identity() {
        let request = "ab".repeat(32);
        assert_eq!(
            DurableQualificationEnrollmentWitnessProviderV1::deterministic_anchor(&request),
            DurableQualificationEnrollmentWitnessProviderV1::deterministic_anchor(&request)
        );
    }

    #[test]
    fn post_persist_fault_is_semantically_uncertain_not_rejected() {
        assert_ne!(
            EnrollmentWitnessAdapterFailureKindV1::Uncertain,
            EnrollmentWitnessAdapterFailureKindV1::Rejected
        );
    }
}
